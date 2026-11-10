"""Musical interpretation between the quantized grid and MusicXML.

This stage never changes ``notes.json`` or ``raw.mid``. It assigns voices,
writes separate notated durations, and records a trace so every source note
stays accountable in the score.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from baroque_transcribe.defaults import DEFAULT_STAFF_SPLIT, NOTATION_MAP_JSON
from baroque_transcribe.notation_settings import NotationSettings
from baroque_transcribe.quantize import GridNote, Quantization
from baroque_transcribe.schema import PedalEvent


class EngraveError(ValueError):
    pass


@dataclass
class WrittenNote:
    source_index: int
    pitch: int
    onset_step: int
    sounding_offset_step: int
    written_offset_step: int
    velocity: int
    staff: int
    voice: int
    chord_with: list[int] = field(default_factory=list)
    gap_filled_steps: int = 0
    overlap_trimmed_steps: int = 0
    clustered_ms: float = 0.0
    adjustments: list[str] = field(default_factory=list)

    @property
    def written_duration_steps(self) -> int:
        return max(1, self.written_offset_step - self.onset_step)


@dataclass
class SuppressedRest:
    staff: int
    voice: int
    start_step: int
    end_step: int
    reason: str


@dataclass
class VoiceEvent:
    """One notated event in a voice: a note or a chord."""

    onset_step: int
    offset_step: int
    pitches: list[int]
    velocity: int
    source_indices: list[int]
    staff: int
    voice: int


@dataclass
class Interpretation:
    notes: list[WrittenNote]
    events_by_staff: dict[int, list[list[VoiceEvent]]]
    suppressed_rests: list[SuppressedRest]
    settings: NotationSettings
    staff_split: int
    warnings: list[str] = field(default_factory=list)

    def events_for_staff(self, staff: int) -> list[list[VoiceEvent]]:
        return self.events_by_staff.get(staff, [[]])


@dataclass
class _VoiceState:
    staff: int
    end_step: int
    last_pitch: int
    last_onset: int
    last_offset: int
    members: list[int] = field(default_factory=list)


def _predicted_staff(pitch: int, staff_split: int, sticky_staff: int | None) -> int:
    if sticky_staff is not None and abs(pitch - staff_split) <= 5:
        return sticky_staff
    return 1 if pitch >= staff_split else 2


def _active_on_staff(voices: list[_VoiceState], staff: int, at_step: int) -> int:
    return sum(1 for voice in voices if voice.staff == staff and voice.end_step > at_step)


def assign_voices(
    grid: list[GridNote],
    *,
    settings: NotationSettings,
    staff_split: int,
) -> list[WrittenNote]:
    """Assign each grid note to a staff and voice with a soft two-voice preference."""

    ordered = sorted(grid, key=lambda item: (item.onset_step, item.pitch, item.source_index))
    voices: list[_VoiceState] = []
    assignments: dict[int, tuple[int, int]] = {}

    for item in ordered:
        best: tuple[float, int, str] | None = None
        for index, voice in enumerate(voices):
            staff = _predicted_staff(item.pitch, staff_split, voice.staff)
            pitch_cost = abs(item.pitch - voice.last_pitch) / 12.0
            staff_cost = 0.0 if staff == voice.staff else 0.75

            if item.onset_step > voice.end_step:
                gap = item.onset_step - voice.end_step
                # Large gaps weaken continuity; a new voice may be better.
                gap_cost = min(gap, 16) * 0.04
                kind = "continue"
                score = pitch_cost + staff_cost + gap_cost
            elif item.onset_step == voice.last_onset:
                duration_delta = abs(item.offset_step - voice.last_offset)
                if duration_delta <= settings.chord_release_tolerance_steps:
                    kind = "chord"
                    score = 0.15 * pitch_cost + staff_cost
                else:
                    # Same attack, independent duration: sustain / suspension.
                    continue
            elif item.onset_step < voice.end_step:
                # Starts while this voice is still holding: cannot join it.
                continue
            else:
                # onset == end: contiguous
                kind = "continue"
                score = pitch_cost + staff_cost

            if best is None or score < best[0]:
                best = (score, index, kind)

        staff = _predicted_staff(
            item.pitch,
            staff_split,
            voices[best[1]].staff if best is not None else None,
        )
        active = _active_on_staff(voices, staff, item.onset_step)
        new_cost = 1.15 + settings.extra_voice_cost * max(0, active - 1)
        # Prefer opening a new voice over a weak continuation.
        if best is None or best[0] > new_cost:
            voice_id = len(voices) + 1
            voices.append(
                _VoiceState(
                    staff=staff,
                    end_step=item.offset_step,
                    last_pitch=item.pitch,
                    last_onset=item.onset_step,
                    last_offset=item.offset_step,
                    members=[item.source_index],
                )
            )
            assignments[item.source_index] = (staff, voice_id)
            continue

        score, index, kind = best
        voice = voices[index]
        staff = _predicted_staff(item.pitch, staff_split, voice.staff)
        if kind == "chord":
            voice.end_step = max(voice.end_step, item.offset_step)
            voice.last_offset = max(voice.last_offset, item.offset_step)
            voice.members.append(item.source_index)
        else:
            voice.end_step = item.offset_step
            voice.last_onset = item.onset_step
            voice.last_offset = item.offset_step
            voice.last_pitch = item.pitch
            voice.staff = staff
            voice.members.append(item.source_index)
        # Voice ids are 1-based in creation order; renumber per staff later.
        assignments[item.source_index] = (staff, index + 1)

    # Renumber voices per staff by first appearance.
    staff_voice_map: dict[tuple[int, int], int] = {}
    next_id = {1: 1, 2: 1}
    written: list[WrittenNote] = []
    by_source = {item.source_index: item for item in grid}
    for item in ordered:
        staff, raw_voice = assignments[item.source_index]
        key = (staff, raw_voice)
        if key not in staff_voice_map:
            staff_voice_map[key] = next_id[staff]
            next_id[staff] += 1
        written.append(
            WrittenNote(
                source_index=item.source_index,
                pitch=item.pitch,
                onset_step=item.onset_step,
                sounding_offset_step=item.offset_step,
                written_offset_step=item.offset_step,
                velocity=item.velocity,
                staff=staff,
                voice=staff_voice_map[key],
            )
        )
    return written


def _unify_chords(notes: list[WrittenNote], settings: NotationSettings) -> list[WrittenNote]:
    """Merge same-voice same-attack notes into chords when durations agree."""

    groups: dict[tuple[int, int, int], list[WrittenNote]] = {}
    for note in notes:
        groups.setdefault((note.staff, note.voice, note.onset_step), []).append(note)
    result: list[WrittenNote] = []
    for members in groups.values():
        members = sorted(members, key=lambda item: item.pitch)
        if len(members) == 1:
            result.append(members[0])
            continue
        offsets = [item.sounding_offset_step for item in members]
        if max(offsets) - min(offsets) > settings.chord_release_tolerance_steps:
            # Independent sustains: keep separate by promoting later pitches
            # that disagree in duration onto new voices within the staff.
            primary = members[0]
            result.append(primary)
            used_voices = {primary.voice}
            for item in members[1:]:
                if abs(item.sounding_offset_step - primary.sounding_offset_step) <= settings.chord_release_tolerance_steps:
                    item.written_offset_step = max(primary.written_offset_step, item.sounding_offset_step)
                    item.chord_with = [primary.source_index]
                    primary.chord_with.append(item.source_index)
                    item.voice = primary.voice
                    item.adjustments.append("chord_unified")
                    result.append(item)
                else:
                    voice = max(used_voices) + 1
                    while voice in used_voices:
                        voice += 1
                    used_voices.add(voice)
                    item.voice = voice
                    item.adjustments.append("independent_sustain")
                    result.append(item)
            continue
        written_offset = max(offsets)
        ids = [item.source_index for item in members]
        for item in members:
            item.written_offset_step = written_offset
            item.chord_with = [index for index in ids if index != item.source_index]
            item.adjustments.append("chord_unified")
            result.append(item)
    return result


def _apply_overlap_trim(notes: list[WrittenNote], settings: NotationSettings) -> None:
    by_voice: dict[tuple[int, int], list[WrittenNote]] = {}
    for note in notes:
        by_voice.setdefault((note.staff, note.voice), []).append(note)
    for members in by_voice.values():
        members.sort(key=lambda item: (item.onset_step, item.pitch))
        # One representative per onset (chord).
        unique: list[WrittenNote] = []
        seen = set()
        for note in members:
            if note.onset_step in seen:
                continue
            seen.add(note.onset_step)
            unique.append(note)
        for previous, current in zip(unique, unique[1:]):
            overlap = previous.written_offset_step - current.onset_step
            if 0 < overlap <= settings.overlap_trim_steps:
                for note in members:
                    if note.onset_step == previous.onset_step:
                        note.written_offset_step = current.onset_step
                        note.overlap_trimmed_steps += overlap
                        note.adjustments.append(f"overlap_trim:{overlap}")


def _apply_gap_fill(
    notes: list[WrittenNote],
    settings: NotationSettings,
    *,
    pedals: list[PedalEvent] | None,
    quantization: Quantization,
) -> None:
    by_voice: dict[tuple[int, int], list[WrittenNote]] = {}
    for note in notes:
        by_voice.setdefault((note.staff, note.voice), []).append(note)

    def under_pedal(step: int) -> bool:
        if not pedals:
            return False
        time = quantization.step_to_audio(step)
        return any(pedal.onset <= time < pedal.offset for pedal in pedals)

    for members in by_voice.values():
        members.sort(key=lambda item: (item.onset_step, item.pitch))
        unique: list[WrittenNote] = []
        seen = set()
        for note in members:
            if note.onset_step in seen:
                continue
            seen.add(note.onset_step)
            unique.append(note)
        for previous, current in zip(unique, unique[1:]):
            gap = current.onset_step - previous.written_offset_step
            if gap <= 0:
                continue
            if gap > settings.fill_gap_steps:
                continue
            if gap >= settings.keep_rest_steps and not under_pedal(previous.written_offset_step):
                continue
            span = current.onset_step - previous.onset_step
            if gap > max(1, span // 2):
                continue
            for note in members:
                if note.onset_step == previous.onset_step:
                    note.written_offset_step = current.onset_step
                    note.gap_filled_steps += gap
                    note.adjustments.append(f"gap_fill:{gap}")


def _build_events(notes: list[WrittenNote]) -> dict[int, list[list[VoiceEvent]]]:
    by_staff: dict[int, dict[int, list[WrittenNote]]] = {1: {}, 2: {}}
    for note in notes:
        by_staff.setdefault(note.staff, {}).setdefault(note.voice, []).append(note)
    result: dict[int, list[list[VoiceEvent]]] = {}
    for staff, voices in by_staff.items():
        ordered_voices = []
        for voice_id in sorted(voices):
            members = sorted(voices[voice_id], key=lambda item: (item.onset_step, item.pitch))
            events: list[VoiceEvent] = []
            grouped: dict[int, list[WrittenNote]] = {}
            for note in members:
                grouped.setdefault(note.onset_step, []).append(note)
            for onset, chord_notes in sorted(grouped.items()):
                events.append(
                    VoiceEvent(
                        onset_step=onset,
                        offset_step=max(item.written_offset_step for item in chord_notes),
                        pitches=[item.pitch for item in chord_notes],
                        velocity=int(round(sum(item.velocity for item in chord_notes) / len(chord_notes))),
                        source_indices=[item.source_index for item in chord_notes],
                        staff=staff,
                        voice=voice_id,
                    )
                )
            ordered_voices.append(events)
        result[staff] = ordered_voices or [[]]
    return result


def _record_suppressed_rests(
    events_by_staff: dict[int, list[list[VoiceEvent]]],
    quantization: Quantization,
) -> list[SuppressedRest]:
    """Whole-measure rests for voices that never sound in a measure are omitted.

    Those omissions are recorded here so the trace stays honest.
    """

    suppressed: list[SuppressedRest] = []
    per = quantization.steps_per_measure
    first = -quantization.lead_steps
    last = first
    for voices in events_by_staff.values():
        for voice in voices:
            for event in voice:
                last = max(last, event.offset_step)
    measure = first
    while measure < last:
        end = measure + per
        for staff, voices in events_by_staff.items():
            for voice_id, events in enumerate(voices, start=1):
                sounding = any(event.onset_step < end and event.offset_step > measure for event in events)
                if not sounding:
                    suppressed.append(
                        SuppressedRest(
                            staff=staff,
                            voice=voice_id,
                            start_step=measure,
                            end_step=end,
                            reason="absent_voice_measure",
                        )
                    )
        measure = end
    return suppressed


def interpret_quantization(
    quantization: Quantization,
    *,
    staff_split: int = DEFAULT_STAFF_SPLIT,
    settings: NotationSettings | None = None,
    pedals: list[PedalEvent] | None = None,
) -> Interpretation:
    """Build a readable interpretation of the quantized grid."""

    resolved = settings or quantization.settings
    notes = assign_voices(quantization.grid, settings=resolved, staff_split=staff_split)
    # Cluster membership timing for the trace.
    cluster_by_source: dict[int, float] = {}
    for cluster in quantization.clusters:
        seed = cluster.seed_onset
        for source_index in cluster.members:
            note = next(item for item in quantization.grid if item.source_index == source_index)
            cluster_by_source[source_index] = abs(note.source_onset - seed) * 1000.0
    for note in notes:
        clustered = cluster_by_source.get(note.source_index, 0.0)
        if clustered > 0.05:
            note.clustered_ms = round(clustered, 3)
            note.adjustments.append(f"clustered_ms:{note.clustered_ms}")

    notes = _unify_chords(notes, resolved)
    _apply_overlap_trim(notes, resolved)
    _apply_gap_fill(notes, resolved, pedals=pedals, quantization=quantization)
    events_by_staff = _build_events(notes)
    suppressed = _record_suppressed_rests(events_by_staff, quantization)
    warnings: list[str] = []
    for staff, voices in events_by_staff.items():
        if len(voices) > 2:
            warnings.append(
                f"Staff {staff} uses {len(voices)} voices to keep independent lines; "
                "the two-voice preference was overridden."
            )
    return Interpretation(
        notes=notes,
        events_by_staff=events_by_staff,
        suppressed_rests=suppressed,
        settings=resolved,
        staff_split=staff_split,
        warnings=warnings,
    )


def write_notation_map(path: Path, interpretation: Interpretation, quantization: Quantization) -> Path:
    """Write ``notation_map.json`` tracing each source note into the score."""

    per = quantization.steps_per_measure
    first = -quantization.lead_steps
    by_source = {item.source_index: item for item in quantization.grid}
    entries = []
    for note in sorted(interpretation.notes, key=lambda item: item.source_index):
        source = by_source[note.source_index]
        measure = ((note.onset_step - first) // per) + 1
        beat_steps = max(1, round(quantization.subdivision / quantization.denominator))
        beat = (note.onset_step - first) % per / beat_steps
        entries.append(
            {
                "source_index": note.source_index,
                "pitch": note.pitch,
                "source_onset": source.source_onset,
                "source_offset": source.source_offset,
                "onset_step": note.onset_step,
                "sounding_offset_step": note.sounding_offset_step,
                "written_offset_step": note.written_offset_step,
                "written_duration_steps": note.written_duration_steps,
                "measure": measure,
                "beat": round(beat, 3),
                "staff": note.staff,
                "voice": note.voice,
                "chord_with": note.chord_with,
                "gap_filled_steps": note.gap_filled_steps,
                "overlap_trimmed_steps": note.overlap_trimmed_steps,
                "clustered_ms": note.clustered_ms,
                "adjustments": note.adjustments,
            }
        )
    payload = {
        "version": 1,
        "grid_mode": quantization.grid_mode,
        "style": "readable",
        "settings": interpretation.settings.to_dict(),
        "staff_split": interpretation.staff_split,
        "notes": entries,
        "suppressed_rests": [asdict(item) for item in interpretation.suppressed_rests],
        "warnings": interpretation.warnings,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def notation_map_path(project: Path) -> Path:
    return project / NOTATION_MAP_JSON


@dataclass
class UnmatchedNote:
    source_index: int
    pitch: int
    onset_step: int
    source_onset: float
    reason: str


@dataclass
class PreservationReport:
    total_source: int
    matched: int
    unmatched: list[UnmatchedNote] = field(default_factory=list)
    duplicate_source_indices: list[int] = field(default_factory=list)
    pitch_mismatches: list[dict] = field(default_factory=list)
    onset_mismatches: list[dict] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            not self.unmatched
            and not self.duplicate_source_indices
            and not self.pitch_mismatches
            and not self.onset_mismatches
            and self.matched == self.total_source
        )

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "total_source": self.total_source,
            "matched": self.matched,
            "unmatched": [asdict(item) for item in self.unmatched],
            "duplicate_source_indices": self.duplicate_source_indices,
            "pitch_mismatches": self.pitch_mismatches,
            "onset_mismatches": self.onset_mismatches,
        }


def validate_note_preservation(
    quantization: Quantization,
    interpretation: Interpretation,
) -> PreservationReport:
    """Verify every grid note appears once in the interpretation.

    Ties and chord grouping are allowed: a source note may share an event with
    others, but its ``source_index``, pitch and onset step must still be present.
    """

    by_source = {item.source_index: item for item in quantization.grid}
    seen: dict[int, WrittenNote] = {}
    duplicates: list[int] = []
    for note in interpretation.notes:
        if note.source_index in seen:
            duplicates.append(note.source_index)
        seen[note.source_index] = note

    unmatched: list[UnmatchedNote] = []
    pitch_mismatches: list[dict] = []
    onset_mismatches: list[dict] = []
    for source_index, grid_note in sorted(by_source.items()):
        written = seen.get(source_index)
        if written is None:
            unmatched.append(
                UnmatchedNote(
                    source_index=source_index,
                    pitch=grid_note.pitch,
                    onset_step=grid_note.onset_step,
                    source_onset=grid_note.source_onset,
                    reason="missing_from_interpretation",
                )
            )
            continue
        if written.pitch != grid_note.pitch:
            pitch_mismatches.append(
                {
                    "source_index": source_index,
                    "grid_pitch": grid_note.pitch,
                    "written_pitch": written.pitch,
                }
            )
        if written.onset_step != grid_note.onset_step:
            onset_mismatches.append(
                {
                    "source_index": source_index,
                    "grid_onset_step": grid_note.onset_step,
                    "written_onset_step": written.onset_step,
                }
            )

    for source_index, written in seen.items():
        if source_index not in by_source:
            unmatched.append(
                UnmatchedNote(
                    source_index=source_index,
                    pitch=written.pitch,
                    onset_step=written.onset_step,
                    source_onset=0.0,
                    reason="extra_in_interpretation",
                )
            )

    return PreservationReport(
        total_source=len(by_source),
        matched=sum(1 for index in by_source if index in seen),
        unmatched=unmatched,
        duplicate_source_indices=duplicates,
        pitch_mismatches=pitch_mismatches,
        onset_mismatches=onset_mismatches,
    )


def write_preservation_report(path: Path, report: PreservationReport) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def validate_against_source_notes(
    notes: list,
    quantization: Quantization,
    interpretation: Interpretation,
) -> PreservationReport:
    """Compare interpretation to the original ``notes.json`` events by identity.

    Checks source_index, pitch, and that the grid onset still corresponds to the
    original onset (via ``GridNote.source_onset``), allowing ties and chords.
    """

    base = validate_note_preservation(quantization, interpretation)
    by_grid = {item.source_index: item for item in quantization.grid}
    by_written = {item.source_index: item for item in interpretation.notes}
    for index, note in enumerate(notes):
        grid = by_grid.get(index)
        written = by_written.get(index)
        if grid is None:
            base.unmatched.append(
                UnmatchedNote(
                    source_index=index,
                    pitch=int(note.pitch),
                    onset_step=-1,
                    source_onset=float(note.onset),
                    reason="missing_from_grid",
                )
            )
            continue
        if abs(grid.source_onset - float(note.onset)) > 1e-6 or grid.pitch != int(note.pitch):
            base.onset_mismatches.append(
                {
                    "source_index": index,
                    "source_onset": float(note.onset),
                    "grid_source_onset": grid.source_onset,
                    "source_pitch": int(note.pitch),
                    "grid_pitch": grid.pitch,
                }
            )
        if written is None:
            continue
        if written.pitch != int(note.pitch):
            base.pitch_mismatches.append(
                {
                    "source_index": index,
                    "source_pitch": int(note.pitch),
                    "written_pitch": written.pitch,
                }
            )
    base.matched = sum(1 for index in range(len(notes)) if index in by_written)
    base.total_source = len(notes)
    return base
