"""Extract a passage of a transcription as a self-contained fragment.

A fragment is a folder under ``data/fragments/<id>/``. It holds its own
notes, grid, MIDI, MusicXML, and harmony, so it can be opened on a machine
that does not have the parent transcription. ``fragment.json`` is the
authoritative manifest. The parent project is only ever read.

Three things are kept apart here:

* **Membership** is decided on the parent's quantized grid and nowhere else.
  A note belongs to the fragment when its grid span touches the selected
  measures. Performance times play no part, so a note printed in the
  selection is never dropped for being played early or late.
* **The playback window** is derived from the members, in working-audio
  seconds. It starts at the first measure's anchor, or earlier when a member
  was struck before that estimate, and ends where the last member ends.
* **Clipping** applies only to notes that cross a boundary barline.

The quantized artifacts are a slice of the parent grid. The notes are never
quantized a second time, so the selected measures are reproduced by integer
arithmetic rather than by rounding rebased times.

Four clocks are in use. ``V`` is source-video seconds, ``W`` is working-audio
seconds (the parent's ``notes.json``), ``F`` is fragment seconds (zero at the
window start), and ``G`` is the grid (steps and measures from the fragment's
first barline). With ``E`` the parent's excerpt start and ``t0`` the window
start in ``W``::

    F = W - t0
    V = F + (E + t0) + sync_offset

The source interval is written once. The sync offset is a separate field.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import threading
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

from baroque_transcribe.analysis import (
    HarmonicAnalysis,
    analysis_from_dict,
    analysis_is_stale,
    estimate_key,
    load_analysis,
    notes_fingerprint,
)
from baroque_transcribe.defaults import (
    ANNOTATIONS_JSON,
    DEFAULT_STAFF_SPLIT,
    FRAGMENT_AUDIO_DIR,
    FRAGMENT_GRID_JSON,
    FRAGMENT_HARMONY_JSON,
    FRAGMENT_JSON,
    FRAGMENT_PAGES_DIR,
    FRAGMENT_PERFORMANCE_MID,
    FRAGMENTS_DIR,
    LOCAL_DIR,
    MANIFEST,
    NOTES_JSON,
    QUANTIZED_MID,
    SCORE_XML,
)
from baroque_transcribe.notation_settings import NotationSettings, settings_from_dict, settings_from_manifest
from baroque_transcribe.project_manifest import load_project_manifest
from baroque_transcribe.quantize import GridNote, Quantization, quantization_from_manifest
from baroque_transcribe.schema import NoteEvent, PedalEvent, Transcription, read_transcription
from baroque_transcribe.sync_map import measure_anchors, printed_measure_span

FRAGMENT_SCHEMA_VERSION = 2
LEGACY_FRAGMENT_SCHEMA_VERSION = 1
GRID_SCHEMA_VERSION = 1
SYNC_OFFSET_LIMIT = 60.0
# The shortest note this program writes (``schema.clean_notes`` uses the same floor).
MIN_NOTE_SECONDS = 0.01
_EPSILON = 1e-9

_EDITABLE_TEXT = ("title", "description", "practice_notes", "composer", "performer")
_ARTIFACT_KEYS = ("notes", "grid", "performance_midi", "quantized_midi", "musicxml", "harmony")
# An edit reads the manifest, changes a field, and writes it back. The details
# form and the player's sync buttons run in one process, on different threads.
_MANIFEST_LOCK = threading.Lock()


class FragmentError(ValueError):
    pass


# --------------------------------------------------------------------------
# Locations
# --------------------------------------------------------------------------


def fragments_root(root: Path | None = None) -> Path:
    """``data/fragments`` in this repository, or an explicit root.

    A test that calls the extraction code must pass its own root. The same
    rule keeps the parse history out of the repository under pytest.
    """

    if root is not None:
        return root
    if os.environ.get("PYTEST_CURRENT_TEST"):
        raise FragmentError("A test must pass an explicit fragments root.")
    from baroque_transcribe.parse_store import repo_root

    return repo_root() / "data" / FRAGMENTS_DIR


def local_root(root: Path | None = None) -> Path:
    """``.local`` in this repository: machine-local files that are not committed."""

    if root is not None:
        return root
    if os.environ.get("PYTEST_CURRENT_TEST"):
        raise FragmentError("A test must pass an explicit local root.")
    from baroque_transcribe.parse_store import repo_root

    return repo_root() / LOCAL_DIR


def fragment_pages_root(local: Path | None = None) -> Path:
    """Where the generated browser pages of every fragment are written.

    One folder for all of them, each page named after its fragment, so the
    notation drawing library (over a megabyte) is copied there once and not
    once per fragment. The pages are rebuilt from the fragment whenever they
    are opened; the folder can be deleted at any time.
    """

    return local_root(local) / FRAGMENT_PAGES_DIR


def fragment_audio_root(local: Path | None = None) -> Path:
    """Where derived preview WAVs of fragments are cached.

    These are synthesized from the fragment's notes for the library list.
    They are not committed and can be deleted at any time.
    """

    return local_root(local) / FRAGMENT_AUDIO_DIR


def is_fragment_dir(folder: Path) -> bool:
    return (folder / FRAGMENT_JSON).is_file()


def fragment_folders(root: Path | None = None) -> list[Path]:
    """Every folder under the fragments root that holds a manifest, by name."""

    base = fragments_root(root)
    if not base.is_dir():
        return []
    return sorted(
        folder
        for folder in base.iterdir()
        if folder.is_dir() and not folder.name.startswith(".") and is_fragment_dir(folder)
    )


def find_fragment(fragment_id: str, root: Path | None = None) -> Path | None:
    """The folder of a fragment, found by id rather than by folder name."""

    wanted = str(fragment_id or "").strip()
    if not wanted:
        return None
    base = fragments_root(root)
    direct = base / wanted
    if Path(wanted).name == wanted and is_fragment_dir(direct):
        try:
            if str(read_manifest(direct).get("id")) == wanted:
                return direct
        except FragmentError:
            pass
    for folder in fragment_folders(base):
        try:
            if str(read_manifest(folder).get("id")) == wanted:
                return folder
        except FragmentError:
            continue
    return None


# --------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------


@dataclass
class FragmentNote:
    """One member note, in fragment coordinates.

    ``index`` is its position in the fragment's ``notes.json``. Steps count
    from the fragment's first barline. Seconds count from the window start.
    ``staff``, ``voice`` and ``written_offset_step`` are set for a readable
    score, where the parent's interpretation is carried over. ``printed`` is
    False for a note the parent's readable score does not print in these
    measures; it is still in the notes, the grid, and both MIDI files.
    """

    index: int
    pitch: int
    velocity: int
    onset_step: int
    offset_step: int
    onset_seconds: float
    offset_seconds: float
    carried_in: bool
    cut_off: bool
    parent_index: int
    source_staff: int | None = None
    staff: int | None = None
    voice: int | None = None
    written_offset_step: int | None = None
    printed: bool = True

    def to_dict(self) -> dict:
        payload = {
            "index": self.index,
            "pitch": self.pitch,
            "velocity": self.velocity,
            "onset_step": self.onset_step,
            "offset_step": self.offset_step,
            "onset_seconds": self.onset_seconds,
            "offset_seconds": self.offset_seconds,
            "carried_in": self.carried_in,
            "cut_off": self.cut_off,
        }
        if self.staff is not None:
            payload["staff"] = self.staff
            payload["voice"] = self.voice
            payload["written_offset_step"] = self.written_offset_step
            payload["printed"] = self.printed
        return payload


@dataclass
class FragmentPlan:
    """Everything an extraction will write, worked out without writing it."""

    project: Path
    parent_manifest: dict
    measure_start: int
    measure_end: int
    parent_measures: int
    step_start: int
    step_end: int
    working_start: float
    duration: float
    barline_seconds: float
    excerpt_start: float
    parent_sync_offset: float
    notes: list[FragmentNote]
    pedals: list[PedalEvent]
    pedal_down_at_start: bool
    anchors: list[dict]
    quantization: Quantization
    style: str
    settings: NotationSettings
    staff_split: int
    key_tonic: str
    key_mode: str
    key_source: str
    key_confidence: float | None
    harmony: HarmonicAnalysis | None
    parent_fingerprint: str
    parent_analysis_sha256: str
    parent_analysis_stale: bool
    model: str
    warnings: list[str] = field(default_factory=list)

    @property
    def measures(self) -> int:
        return self.measure_end - self.measure_start + 1

    @property
    def working_end(self) -> float:
        return round(self.working_start + self.duration, 5)

    @property
    def video_start(self) -> float:
        """Source-video seconds of the window start, before any sync offset."""

        return round(self.excerpt_start + self.working_start, 5)

    @property
    def video_end(self) -> float:
        return round(self.excerpt_start + self.working_start + self.duration, 5)

    @property
    def owned(self) -> int:
        return sum(1 for note in self.notes if not note.carried_in)

    @property
    def carried_in(self) -> int:
        return sum(1 for note in self.notes if note.carried_in)

    @property
    def cut_off(self) -> int:
        return sum(1 for note in self.notes if note.cut_off)

    @property
    def unprinted(self) -> int:
        return sum(1 for note in self.notes if not note.printed)

    @property
    def roman(self) -> list[str]:
        return [chord.roman for chord in self.harmony.chords] if self.harmony else []

    @property
    def symbols(self) -> list[str]:
        return [chord.symbol for chord in self.harmony.chords] if self.harmony else []


def _read_json_object(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise FragmentError(f"Could not read {path.name}: {exc}") from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise FragmentError(f"{path.name} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FragmentError(f"{path.name} is not a JSON object.")
    return payload


def _number(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def plan_fragment(project: Path, measure_start: int, measure_end: int) -> FragmentPlan:
    """Work out a fragment of whole printed measures ``measure_start..measure_end``.

    Nothing is written. The parent project is read: its notes, its manifest,
    and its harmonic analysis when there is one.
    """

    project = Path(project)
    notes_path = project / NOTES_JSON
    if not notes_path.is_file():
        raise FragmentError(f"No {NOTES_JSON} in {project}. Transcribe the recording first.")
    manifest = load_project_manifest(project)
    tempo = manifest.get("tempo") if isinstance(manifest.get("tempo"), dict) else {}
    if not tempo.get("bpm"):
        raise FragmentError("This project has no tempo, so it has no measures to select. Rebuild its score first.")
    try:
        transcription = read_transcription(notes_path)
        settings = settings_from_manifest(manifest)
        parent = quantization_from_manifest(transcription.notes, tempo, settings=settings, manifest=manifest)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise FragmentError(f"Could not rebuild the notation grid of {project.name}: {exc}") from exc

    per_measure = parent.steps_per_measure
    first_step, measure_count = printed_measure_span(parent)
    try:
        measure_start = int(measure_start)
        measure_end = int(measure_end)
    except (TypeError, ValueError) as exc:
        raise FragmentError("Measure numbers must be whole numbers.") from exc
    if measure_start < 1 or measure_end < measure_start or measure_end > measure_count:
        raise FragmentError(
            f"Choose measures between 1 and {measure_count}, with the first not after the last. "
            f"Got {measure_start} to {measure_end}."
        )
    s0 = first_step + (measure_start - 1) * per_measure
    s1 = first_step + measure_end * per_measure

    # Musical membership: the grid span touches the selection. Seconds are not consulted.
    members = [item for item in parent.grid if item.onset_step < s1 and item.offset_step > s0]
    if not members:
        raise FragmentError("The selected measures contain no notes.")

    # Playback window, in working-audio seconds.
    parent_anchors = measure_anchors(parent)
    anchor_start = parent_anchors[measure_start - 1].time
    if measure_end < measure_count:
        end_anchor = parent_anchors[measure_end].time
    else:
        end_anchor = parent.step_to_audio(s1)
    end_anchor = max(end_anchor, anchor_start + MIN_NOTE_SECONDS)
    owned_onsets = [item.source_onset for item in members if item.onset_step >= s0]
    # The anchor is normally the earliest owned attack already. The minimum
    # covers the anchor's collision fallback, which substitutes a later time.
    t0 = round(min([anchor_start, *owned_onsets]), 5)

    def written_seconds(item: GridNote) -> float:
        """How long the clipped note lasts at the written tempo."""

        clip_start = max(item.onset_step, s0)
        clip_end = min(item.offset_step, s1)
        return max(parent.step_to_audio(clip_end) - parent.step_to_audio(clip_start), MIN_NOTE_SECONDS)

    timed: list[tuple[GridNote, float, float, bool, bool]] = []
    for item in members:
        carried = item.onset_step < s0
        cut = item.offset_step > s1
        start = t0 if carried else item.source_onset
        release = item.source_offset
        if release - start < MIN_NOTE_SECONDS - _EPSILON:
            # The recorded release is not after the start. For a carried-in
            # note that means it was released before the estimated barline.
            # The written duration is the only honest length left.
            release = start + written_seconds(item)
        if cut:
            # End at the end barline, but never before the note's own start:
            # an attack later than the estimated end anchor keeps its written length.
            release = min(release, max(end_anchor, start + written_seconds(item)))
        timed.append((item, start, release, carried, cut))
    t1 = max([end_anchor, *(release for _item, _start, release, _carried, _cut in timed)])
    duration = round(t1 - t0, 5)

    interpretation_by_source: dict[int, object] = {}
    style = settings.style
    staff_split = int(manifest.get("staff_split") or DEFAULT_STAFF_SPLIT)
    if style == "readable":
        from baroque_transcribe.engrave import interpret_quantization

        parent_interpretation = interpret_quantization(
            parent,
            staff_split=staff_split,
            settings=parent.settings,
            pedals=transcription.pedals,
        )
        interpretation_by_source = {note.source_index: note for note in parent_interpretation.notes}

    drafts: list[FragmentNote] = []
    for item, start, release, carried, cut in timed:
        onset_seconds = max(0.0, round(start - t0, 5))
        offset_seconds = round(release - t0, 5)
        if offset_seconds - onset_seconds < MIN_NOTE_SECONDS - _EPSILON:
            offset_seconds = round(onset_seconds + MIN_NOTE_SECONDS, 5)
        duration = max(duration, offset_seconds)
        onset_step = max(item.onset_step, s0) - s0
        offset_step = min(item.offset_step, s1) - s0
        source_staff = None
        if 0 <= item.source_index < len(transcription.notes):
            candidate_staff = transcription.notes[item.source_index].staff
            if candidate_staff in (1, 2):
                source_staff = int(candidate_staff)
        note = FragmentNote(
            index=-1,
            pitch=int(item.pitch),
            velocity=int(item.velocity),
            onset_step=onset_step,
            offset_step=offset_step,
            onset_seconds=onset_seconds,
            offset_seconds=offset_seconds,
            carried_in=carried,
            cut_off=cut,
            parent_index=item.source_index,
            source_staff=source_staff,
            staff=source_staff,
        )
        written = interpretation_by_source.get(item.source_index)
        if written is not None:
            if note.staff not in (1, 2):
                note.staff = int(written.staff)  # type: ignore[attr-defined]
            note.voice = int(written.voice)  # type: ignore[attr-defined]
            written_offset = min(int(written.written_offset_step), s1) - s0  # type: ignore[attr-defined]
            if written_offset <= onset_step:
                # The parent wrote this note as ending at or before the first
                # barline (a legato overlap it trimmed), yet on the grid it
                # still sounds into the fragment. The parent prints nothing
                # for it in these measures, so neither does the fragment.
                written_offset = offset_step
                note.printed = False
            note.written_offset_step = written_offset
        drafts.append(note)

    # The order ``notes.json`` is read back in.
    drafts.sort(key=lambda note: (note.onset_seconds, note.pitch, note.offset_seconds))
    for index, note in enumerate(drafts):
        note.index = index
        if not (0.0 <= note.onset_seconds < note.offset_seconds <= duration + _EPSILON):
            raise FragmentError(
                f"Internal error: note {note.pitch} would run from {note.onset_seconds}s to "
                f"{note.offset_seconds}s in a {duration}s fragment."
            )
        if not (0 <= note.onset_step < note.offset_step <= s1 - s0):
            raise FragmentError(
                f"Internal error: note {note.pitch} would run from step {note.onset_step} to {note.offset_step}."
            )

    pedals: list[PedalEvent] = []
    for pedal in transcription.pedals:
        start = max(pedal.onset, t0)
        end = min(pedal.offset, t0 + duration)
        if end - start >= MIN_NOTE_SECONDS:
            pedals.append(PedalEvent(onset=max(0.0, round(start - t0, 5)), offset=round(end - t0, 5)))
    pedal_down_at_start = any(pedal.onset < t0 < pedal.offset for pedal in transcription.pedals)

    anchors = []
    for position in range(measure_start - 1, measure_end):
        anchor = parent_anchors[position]
        anchors.append(
            {
                "measure": position - measure_start + 2,
                "parent_measure": anchor.measure,
                "seconds": max(0.0, round(anchor.time - t0, 5)),
                "opens_before_attack": anchor.opens_before_attack,
                "collided": anchor.collided,
            }
        )

    barline_seconds = round(parent.step_to_audio(s0) - t0, 5)
    quantization = _fragment_quantization(
        drafts,
        bpm=parent.bpm,
        numerator=parent.numerator,
        denominator=parent.denominator,
        subdivision=parent.subdivision,
        barline_seconds=barline_seconds,
        grid_mode=parent.grid_mode,
        settings=parent.settings,
    )

    warnings: list[str] = []
    stored = load_analysis(project / ANNOTATIONS_JSON)
    analysis_sha = ""
    stale = False
    harmony: HarmonicAnalysis | None = None
    if stored is None:
        tonic, mode, confidence = estimate_key(parent)
        key_source = "estimated-at-extraction"
        key_confidence: float | None = confidence
        warnings.append(
            "The parent project has no harmonic analysis, so the fragment has no chord labels. "
            "The key was estimated from the parent's notes."
        )
    else:
        tonic, mode = stored.key_tonic, stored.key_mode
        key_source = stored.key_source
        key_confidence = stored.key_confidence
        analysis_sha = _analysis_digest(project / ANNOTATIONS_JSON)
        stale = analysis_is_stale(stored, parent)
        if stale:
            warnings.append(
                "The parent's notes or tempo changed after its harmonic analysis was made, so the chord "
                "labels no longer line up with the measures. The fragment was saved without them. "
                "Rebuild the analysis on the parent and extract again to include them."
            )
        else:
            harmony = _slice_harmony(
                stored,
                s0=s0,
                s1=s1,
                measure_start=measure_start,
                measure_end=measure_end,
                quantization=quantization,
                barline_seconds=barline_seconds,
            )

    return FragmentPlan(
        project=project,
        parent_manifest=manifest,
        measure_start=measure_start,
        measure_end=measure_end,
        parent_measures=measure_count,
        step_start=s0,
        step_end=s1,
        working_start=t0,
        duration=round(duration, 5),
        barline_seconds=barline_seconds,
        excerpt_start=_number((manifest.get("audio") or {}).get("excerpt_start_seconds")),
        parent_sync_offset=_number(manifest.get("sync_offset_seconds")),
        notes=drafts,
        pedals=pedals,
        pedal_down_at_start=pedal_down_at_start,
        anchors=anchors,
        quantization=quantization,
        style=style,
        settings=parent.settings,
        staff_split=staff_split,
        key_tonic=tonic,
        key_mode=mode,
        key_source=key_source,
        key_confidence=key_confidence,
        harmony=harmony,
        parent_fingerprint=notes_fingerprint(parent),
        parent_analysis_sha256=analysis_sha,
        parent_analysis_stale=stale,
        model=transcription.model,
        warnings=warnings,
    )


def _fragment_quantization(
    notes: list[FragmentNote],
    *,
    bpm: float,
    numerator: int,
    denominator: int,
    subdivision: int,
    barline_seconds: float,
    grid_mode: str,
    settings: NotationSettings,
) -> Quantization:
    """A grid built from sliced steps. ``quantize_notes`` is not involved."""

    quantum = (60.0 / bpm) * 4.0 / subdivision
    grid = [
        GridNote(
            pitch=note.pitch,
            onset_step=note.onset_step,
            offset_step=note.offset_step,
            velocity=note.velocity,
            source_index=note.index,
            source_onset=note.onset_seconds,
            source_offset=note.offset_seconds,
        )
        for note in notes
    ]
    grid.sort(key=lambda item: (item.onset_step, item.pitch, item.offset_step))
    snapped = [
        NoteEvent(
            pitch=item.pitch,
            onset=barline_seconds + item.onset_step * quantum,
            offset=barline_seconds + item.offset_step * quantum,
            velocity=item.velocity,
        )
        for item in grid
    ]
    return Quantization(
        bpm=float(bpm),
        numerator=int(numerator),
        denominator=int(denominator),
        subdivision=int(subdivision),
        downbeat=float(barline_seconds),
        quantum=quantum,
        steps_per_measure=numerator * subdivision // denominator,
        lead_steps=0,
        notes=snapped,
        grid=grid,
        # The parent's mode is kept as a label. No step-time map is carried,
        # so ``step_to_audio`` is metronomic from the first barline. The
        # parent's tracked timing survives in the stored measure anchors and
        # in each note's own seconds.
        grid_mode=grid_mode,
        settings=settings,
    )


def _analysis_digest(path: Path) -> str:
    """Digest of the analysis content, the same on every machine's line endings."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return ""
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _slice_harmony(
    stored: HarmonicAnalysis,
    *,
    s0: int,
    s1: int,
    measure_start: int,
    measure_end: int,
    quantization: Quantization,
    barline_seconds: float,
) -> HarmonicAnalysis:
    """Copy the parent's labels for the selection. No analysis is run."""

    per_measure = quantization.steps_per_measure
    beat_steps = max(1, round(quantization.subdivision / quantization.denominator))
    chords = []
    for chord in stored.chords:
        if chord.start_step >= s1 or chord.end_step <= s0:
            continue
        start = max(chord.start_step, s0) - s0
        end = min(chord.end_step, s1) - s0
        within = start % per_measure
        chords.append(
            replace(
                chord,
                measure=start // per_measure + 1,
                beat=round(within / beat_steps, 3),
                measure_position=round(within / per_measure, 4),
                start_step=start,
                end_step=end,
                edited_fields=list(chord.edited_fields),
            )
        )
    chords.sort(key=lambda item: item.start_step)
    sequences = []
    for sequence in stored.sequences:
        if sequence.start_measure > measure_end or sequence.end_measure < measure_start:
            continue
        start = max(sequence.start_measure, measure_start) - measure_start + 1
        end = min(sequence.end_measure, measure_end) - measure_start + 1
        if end <= start:
            continue
        clipped = start != sequence.start_measure - measure_start + 1 or end != sequence.end_measure - measure_start + 1
        sequences.append(
            replace(
                sequence,
                start_measure=start,
                end_measure=end,
                repetitions=end - start if clipped else sequence.repetitions,
            )
        )
    return HarmonicAnalysis(
        key_tonic=stored.key_tonic,
        key_mode=stored.key_mode,
        key_source=stored.key_source,
        key_confidence=stored.key_confidence,
        chords=chords,
        sequences=sequences,
        grid={
            "bpm": quantization.bpm,
            "numerator": quantization.numerator,
            "denominator": quantization.denominator,
            "subdivision": quantization.subdivision,
            "downbeat_seconds": barline_seconds,
            "steps_per_measure": per_measure,
            "first_step": 0,
            "grid_mode": quantization.grid_mode,
        },
        notes_fingerprint=notes_fingerprint(quantization),
    )


def _interpretation(notes: list[FragmentNote], settings: NotationSettings, staff_split: int):
    """The parent's voices and written durations, restricted to the fragment.

    Voice numbers are the parent's. A voice that does not sound in the
    fragment stays an empty slot, which the readable writer does not print.
    """

    from baroque_transcribe.engrave import Interpretation, VoiceEvent, WrittenNote

    written = [
        WrittenNote(
            source_index=note.index,
            pitch=note.pitch,
            onset_step=note.onset_step,
            sounding_offset_step=note.offset_step,
            written_offset_step=int(note.written_offset_step or note.offset_step),
            velocity=note.velocity,
            staff=int(note.staff or 1),
            voice=int(note.voice or 1),
        )
        for note in notes
    ]
    events_by_staff: dict[int, list[list[VoiceEvent]]] = {}
    for staff in (1, 2):
        grouped: dict[int, dict[int, list[WrittenNote]]] = {}
        for item, note in zip(written, notes):
            if item.staff == staff and note.printed:
                grouped.setdefault(item.voice, {}).setdefault(item.onset_step, []).append(item)
        voices: list[list[VoiceEvent]] = []
        for voice_id in range(1, max(grouped, default=0) + 1):
            events = []
            for onset, members in sorted(grouped.get(voice_id, {}).items()):
                members.sort(key=lambda item: item.pitch)
                events.append(
                    VoiceEvent(
                        onset_step=onset,
                        offset_step=max(item.written_offset_step for item in members),
                        pitches=[item.pitch for item in members],
                        velocity=int(round(sum(item.velocity for item in members) / len(members))),
                        source_indices=[item.source_index for item in members],
                        staff=staff,
                        voice=voice_id,
                    )
                )
            voices.append(events)
        events_by_staff[staff] = voices or [[]]
    return Interpretation(
        notes=written,
        events_by_staff=events_by_staff,
        suppressed_rests=[],
        settings=settings,
        staff_split=staff_split,
    )


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def parse_measures(text: str) -> tuple[int, int]:
    """``12-15`` or ``12`` as a pair of printed measure numbers."""

    cleaned = str(text or "").replace("\u2013", "-").replace(" ", "")
    first, _dash, last = cleaned.partition("-")
    try:
        start = int(first)
        end = int(last) if last else start
    except ValueError as exc:
        raise FragmentError(f"Measures must look like 12-15 or 12, not {text!r}.") from exc
    if start < 1 or end < 1:
        raise FragmentError("Measure numbers start at 1.")
    if end < start:
        start, end = end, start
    return start, end


def normalize_tags(tags) -> list[str]:
    """Trimmed, single-spaced tags with case-insensitive duplicates removed."""

    if isinstance(tags, str):
        tags = tags.split(",")
    cleaned: list[str] = []
    seen: set[str] = set()
    for tag in tags or []:
        text = " ".join(str(tag).split())
        if not text or text.casefold() in seen:
            continue
        seen.add(text.casefold())
        cleaned.append(text)
    return cleaned


def _timestamp(moment: datetime | None = None) -> str:
    when = moment or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    return when.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _is_plain_name(value: object) -> bool:
    text = str(value or "")
    return bool(text) and Path(text).name == text and text not in {".", ".."}


def validate_manifest(payload: object) -> dict:
    """Return the manifest when it can be used, or raise ``FragmentError`` saying why not."""

    if not isinstance(payload, dict):
        raise FragmentError("The manifest is not a JSON object.")
    try:
        version = int(payload.get("schema_version"))  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise FragmentError("The manifest has no schema version.") from exc
    if version not in {LEGACY_FRAGMENT_SCHEMA_VERSION, FRAGMENT_SCHEMA_VERSION}:
        raise FragmentError(
            f"The manifest uses schema version {version}; this program reads versions "
            f"{LEGACY_FRAGMENT_SCHEMA_VERSION} and {FRAGMENT_SCHEMA_VERSION}."
        )
    ident = payload.get("id")
    if not isinstance(ident, str) or not ident.strip():
        raise FragmentError("The manifest has no fragment id.")
    title = payload.get("title")
    if not isinstance(title, str) or not title.strip():
        raise FragmentError("The manifest has no title.")
    tags = payload.get("tags", [])
    if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
        raise FragmentError("The manifest's tags are not a list of text.")
    time = payload.get("time")
    if not isinstance(time, dict):
        raise FragmentError("The manifest has no time block.")
    for clock in ("working_audio", "source_video"):
        block = time.get(clock)
        if not isinstance(block, dict):
            raise FragmentError(f"The manifest has no {clock} interval.")
        for key in ("start_seconds", "end_seconds"):
            if not isinstance(block.get(key), (int, float)) or isinstance(block.get(key), bool):
                raise FragmentError(f"The manifest's {clock}.{key} is not a number.")
    if not isinstance(time.get("duration_seconds"), (int, float)) or time["duration_seconds"] <= 0:
        raise FragmentError("The manifest's duration is not a positive number.")
    sync = payload.get("sync", {})
    if not isinstance(sync, dict) or not isinstance(sync.get("offset_seconds", 0.0), (int, float)):
        raise FragmentError("The manifest's sync offset is not a number.")
    for block in ("source", "parent", "music"):
        if not isinstance(payload.get(block), dict):
            raise FragmentError(f"The manifest has no {block} block.")
    if version == FRAGMENT_SCHEMA_VERSION:
        parent = payload["parent"]
        for key in (
            "lineage_id",
            "base_event_id",
            "base_snapshot_sha256",
            "snapshot_sha256",
            "revision_state",
        ):
            if not isinstance(parent.get(key), str):
                raise FragmentError(f"The manifest parent.{key} field is not text.")
        if not parent.get("snapshot_sha256"):
            raise FragmentError("The manifest has no parent snapshot fingerprint.")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict):
        raise FragmentError("The manifest has no artifacts block.")
    for key, value in artifacts.items():
        if value is None:
            continue
        if not _is_plain_name(value):
            raise FragmentError(f"The manifest's artifact {key!r} is not a plain file name.")
    return payload


def read_manifest(folder: Path) -> dict:
    """The validated manifest of one fragment folder."""

    return validate_manifest(_read_json_object(Path(folder) / FRAGMENT_JSON))


def fragment_parent_status(
    fragment: Path | dict,
    *,
    projects_root: Path | None = None,
    store: Path | None = None,
) -> str:
    """Return current, parent_changed, published_parent_changed, or parent_unavailable."""

    from baroque_transcribe.parse_store import load_events, repo_root, resolve_store
    from baroque_transcribe.revision import event_heads, event_lineage_id, snapshot_for_project

    manifest = read_manifest(fragment) if isinstance(fragment, Path) else validate_manifest(fragment)
    parent = manifest.get("parent") if isinstance(manifest.get("parent"), dict) else {}
    project_name = str(parent.get("project") or "")
    root = Path(projects_root) if projects_root is not None else repo_root() / "transcriptions"
    project = root / Path(project_name).name
    if not project_name or not (project / NOTES_JSON).is_file():
        return "parent_unavailable"
    recorded_snapshot = str(parent.get("snapshot_sha256") or "")
    if recorded_snapshot:
        try:
            if snapshot_for_project(project).sha256 != recorded_snapshot:
                return "parent_changed"
        except OSError:
            return "parent_unavailable"
    else:
        recorded_notes = str(parent.get("notes_fingerprint") or "")
        try:
            transcription = read_transcription(project / NOTES_JSON)
            project_manifest = _read_json_object(project / MANIFEST)
            tempo = project_manifest.get("tempo") or {}
            quantization = quantization_from_manifest(
                transcription.notes,
                tempo,
                settings=settings_from_manifest(project_manifest),
                manifest=project_manifest,
            )
            if recorded_notes and notes_fingerprint(quantization) != recorded_notes:
                return "parent_changed"
        except (OSError, KeyError, TypeError, ValueError):
            return "parent_unavailable"
    lineage_id = str(parent.get("lineage_id") or "")
    base_event_id = str(parent.get("base_event_id") or parent.get("event_id") or "")
    target = resolve_store(store)
    if target is not None and lineage_id and base_event_id:
        events = load_events(target)
        heads = event_heads(events, lineage_id)
        if any(str(head.get("id") or "") != base_event_id for head in heads):
            return "published_parent_changed"
        if not any(str(event.get("id") or "") == base_event_id for event in events):
            return "published_parent_changed"
    return "current"


def _write_json(path: Path, payload: dict) -> None:
    """Write through a temporary file, so a reader never sees half a document.

    On Windows, ``os.replace`` onto an existing file can raise Access Denied
    when another process briefly holds it. Retry a few times, then fail
    without changing the prior document.
    """

    import time

    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    last_error: OSError | None = None
    try:
        for attempt in range(8):
            try:
                os.replace(temporary, path)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.05 * (attempt + 1))
        if last_error is not None:
            raise last_error
        raise OSError(f"Could not atomically replace {path}.")
    finally:
        temporary.unlink(missing_ok=True)
        stale = path.with_name(path.name + ".tmp")
        if stale.is_file():
            stale.unlink(missing_ok=True)


def _parent_provenance(project: Path, store: Path | None) -> dict:
    """Causal base and local snapshot at extraction, without mutating the parent."""

    from baroque_transcribe.parse_store import load_events, resolve_store
    from baroque_transcribe.project_manifest import load_project_manifest, revision_block
    from baroque_transcribe.revision import (
        event_lineage_id,
        event_snapshot_sha256,
        revision_status,
        snapshot_for_project,
    )

    manifest = load_project_manifest(project)
    revision = revision_block(manifest)
    lineage_id = str(revision.get("lineage_id") or "")
    base_event_id = str(revision.get("base_event_id") or "")
    base_snapshot = str(revision.get("base_snapshot_sha256") or "")
    local_snapshot = snapshot_for_project(project).sha256
    state = "independent"
    target = resolve_store(store)
    if target is not None:
        events = load_events(target)
        status = revision_status(project.name, project, events, target)
        state = status.state
        if not base_event_id and status.adoptable_event_id:
            base_event_id = status.adoptable_event_id
            matched = next(
                (
                    event
                    for event in events
                    if str(event.get("id") or "") == base_event_id
                ),
                {},
            )
            lineage_id = event_lineage_id(matched) or base_event_id
            base_snapshot = event_snapshot_sha256(matched) or local_snapshot
    return {
        "event_id": base_event_id,
        "lineage_id": lineage_id,
        "base_event_id": base_event_id,
        "base_snapshot_sha256": base_snapshot,
        "snapshot_sha256": local_snapshot,
        "revision_state": state,
    }


def _describe_parent_source(manifest: dict) -> dict:
    from baroque_transcribe.parse_store import describe_source

    return describe_source(
        str(manifest.get("origin") or ""),
        title=str(manifest.get("title") or ""),
        youtube_url=str(manifest.get("youtube_url") or "") or None,
    )


def catalog_from_parent(manifest: dict) -> dict:
    """Optional catalog provenance copied from a parent project manifest."""

    catalog = manifest.get("catalog") if isinstance(manifest.get("catalog"), dict) else {}
    if not catalog:
        return {}
    cleaned = {
        key: value
        for key, value in {
            "entry_id": str(catalog.get("entry_id") or "").strip(),
            "bwv": str(catalog.get("bwv") or "").strip(),
            "collection_id": str(catalog.get("collection_id") or "").strip(),
            "collection": str(catalog.get("collection") or "").strip(),
            "source_page": str(catalog.get("source_page") or "").strip(),
            "attribution": str(catalog.get("attribution") or "").strip(),
        }.items()
        if value
    }
    return cleaned


def catalog_extract_defaults(project: Path, plan: FragmentPlan | None = None) -> dict:
    """Prefill extract-form fields from a catalog-backed parent project."""

    from baroque_transcribe.project_manifest import load_project_manifest

    manifest = load_project_manifest(project)
    catalog = catalog_from_parent(manifest)
    if not catalog:
        return {}
    bwv = catalog.get("bwv") or ""
    measures = ""
    if plan is not None:
        measures = (
            f"mm. {plan.measure_start}–{plan.measure_end}"
            if plan.measure_start != plan.measure_end
            else f"m. {plan.measure_start}"
        )
    title_bits = [f"BWV {bwv}" if bwv else "", str(manifest.get("title") or "").strip(), measures]
    title = " — ".join(bit for bit in title_bits if bit)
    tags = []
    if bwv:
        tags.append(f"bwv-{bwv.lower()}")
    if catalog.get("collection_id"):
        tags.append(str(catalog["collection_id"]))
    return {
        "title": title,
        "composer": "Johann Sebastian Bach",
        "tags": tags,
        "catalog": catalog,
    }


def build_manifest(
    plan: FragmentPlan,
    *,
    fragment_id: str,
    title: str,
    description: str = "",
    practice_notes: str = "",
    tags=(),
    composer: str = "",
    performer: str = "",
    event_id: str = "",
    parent_provenance: dict | None = None,
    now: datetime | None = None,
) -> dict:
    stamp = _timestamp(now)
    harmony_file = FRAGMENT_HARMONY_JSON if plan.harmony is not None else None
    provenance = dict(parent_provenance or {})
    if event_id and not provenance.get("event_id"):
        provenance["event_id"] = event_id
    catalog = catalog_from_parent(plan.parent_manifest)
    payload = {
        "schema_version": FRAGMENT_SCHEMA_VERSION,
        "id": fragment_id,
        "title": " ".join(str(title).split()),
        "description": str(description or "").strip(),
        "practice_notes": str(practice_notes or "").strip(),
        "tags": normalize_tags(tags),
        "composer": str(composer or "").strip(),
        "performer": str(performer or "").strip(),
        "source": _describe_parent_source(plan.parent_manifest),
        "parent": {
            "project": plan.project.name,
            "event_id": str(provenance.get("event_id") or ""),
            "lineage_id": str(provenance.get("lineage_id") or ""),
            "base_event_id": str(provenance.get("base_event_id") or ""),
            "base_snapshot_sha256": str(
                provenance.get("base_snapshot_sha256") or ""
            ),
            "snapshot_sha256": str(provenance.get("snapshot_sha256") or ""),
            "revision_state": str(provenance.get("revision_state") or ""),
            "notes_fingerprint": plan.parent_fingerprint,
            "analysis_sha256": plan.parent_analysis_sha256,
            "measure_start": plan.measure_start,
            "measure_end": plan.measure_end,
            "step_start": plan.step_start,
            "step_end": plan.step_end,
        },
        "time": {
            "working_audio": {"start_seconds": plan.working_start, "end_seconds": plan.working_end},
            "source_video": {
                "start_seconds": plan.video_start,
                "end_seconds": plan.video_end,
                "excerpt_start_seconds": plan.excerpt_start,
            },
            "duration_seconds": plan.duration,
        },
        "sync": {
            "offset_seconds": plan.parent_sync_offset,
            "parent_offset_at_extraction": plan.parent_sync_offset,
        },
        "music": {
            "analyzed_key": {
                "tonic": plan.key_tonic,
                "mode": plan.key_mode,
                "source": plan.key_source,
                "confidence": plan.key_confidence,
            },
            "written_key_signature": {"tonic": plan.key_tonic, "mode": plan.key_mode},
            "time_signature": f"{plan.quantization.numerator}/{plan.quantization.denominator}",
            "bpm": plan.quantization.bpm,
            "subdivision": plan.quantization.subdivision,
            "staff_split": plan.staff_split,
            "measures": plan.measures,
            "parent_grid_mode": str(plan.settings.grid),
            "notation_style": plan.style,
        },
        "harmony": {"roman": plan.roman, "symbols": plan.symbols},
        "boundary": {
            "notes": len(plan.notes),
            "owned": plan.owned,
            "carried_in": plan.carried_in,
            "cut_off": plan.cut_off,
            "unprinted_overlaps": plan.unprinted,
            "pedal_down_at_start": plan.pedal_down_at_start,
        },
        "artifacts": {
            "notes": NOTES_JSON,
            "grid": FRAGMENT_GRID_JSON,
            "performance_midi": FRAGMENT_PERFORMANCE_MID,
            "quantized_midi": QUANTIZED_MID,
            "musicxml": SCORE_XML,
            "harmony": harmony_file,
        },
        "warnings": list(plan.warnings),
        "created_at": stamp,
        "modified_at": stamp,
    }
    if catalog:
        payload["catalog"] = catalog
    return payload


def grid_payload(plan: FragmentPlan) -> dict:
    quantization = plan.quantization
    return {
        "version": GRID_SCHEMA_VERSION,
        "bpm": quantization.bpm,
        "numerator": quantization.numerator,
        "denominator": quantization.denominator,
        "subdivision": quantization.subdivision,
        "steps_per_measure": quantization.steps_per_measure,
        "measures": plan.measures,
        "barline_seconds": plan.barline_seconds,
        "grid_mode": quantization.grid_mode,
        "staff_split": plan.staff_split,
        "notation_style": plan.style,
        "settings": plan.settings.to_dict(),
        "notes": [note.to_dict() for note in plan.notes],
        "anchors": plan.anchors,
    }


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------


def write_fragment_score(
    path: Path,
    quantization: Quantization,
    notes: list[FragmentNote],
    *,
    measures: int,
    staff_split: int,
    style: str,
    settings: NotationSettings,
    key: tuple[str, str],
    title: str,
) -> Path:
    """Write the fragment's MusicXML with the existing writer.

    A literal score packs voices from the sliced grid. A readable score
    carries over the parent's voices and written durations, because in that
    style the durations and the staff depend on the voice assignment.
    """

    from baroque_transcribe.notation import NotationError, write_musicxml

    interpretation = _interpretation(notes, settings, staff_split) if style == "readable" else None
    try:
        return write_musicxml(
            path,
            quantization,
            staff_split=staff_split,
            title=title,
            style=style,
            key=key,
            interpretation=interpretation,
            end_step=measures * quantization.steps_per_measure,
        )
    except NotationError as exc:
        raise FragmentError(f"Could not write the fragment's score: {exc}") from exc


def extract_fragment(
    project: Path,
    measure_start: int,
    measure_end: int,
    *,
    title: str,
    description: str = "",
    practice_notes: str = "",
    tags=(),
    composer: str = "",
    performer: str = "",
    root: Path | None = None,
    store: Path | None = None,
    now: datetime | None = None,
) -> Path:
    """Save measures ``measure_start..measure_end`` of ``project`` as a new fragment.

    Returns the fragment folder. Every file is written into a staging folder
    first and the folder is renamed into place last, so a failed extraction
    leaves nothing behind. Nothing in ``project`` is written.
    """

    if not " ".join(str(title or "").split()):
        raise FragmentError("Give the fragment a title.")
    plan = plan_fragment(project, measure_start, measure_end)
    base = fragments_root(root)
    fragment_id = str(uuid.uuid4())
    final = base / fragment_id
    staging = base / f".tmp-{fragment_id}"
    manifest = build_manifest(
        plan,
        fragment_id=fragment_id,
        title=title,
        description=description,
        practice_notes=practice_notes,
        tags=tags,
        composer=composer,
        performer=performer,
        parent_provenance=_parent_provenance(plan.project, store),
        now=now,
    )
    source = manifest["source"]
    staging.mkdir(parents=True)
    try:
        from baroque_transcribe.midi_export import write_quantized_midi, write_raw_midi

        performance = [
            NoteEvent(
                pitch=note.pitch,
                onset=note.onset_seconds,
                offset=note.offset_seconds,
                velocity=note.velocity,
                staff=note.source_staff,
            )
            for note in plan.notes
        ]
        Transcription(
            notes=performance,
            pedals=plan.pedals,
            model=plan.model,
            source=str(source.get("url") or source.get("name") or ""),
        ).write(staging / NOTES_JSON)
        _write_json(staging / FRAGMENT_GRID_JSON, grid_payload(plan))
        write_raw_midi(staging / FRAGMENT_PERFORMANCE_MID, performance, plan.pedals)
        staff_by_source = {
            note.parent_index: int(note.source_staff)
            for note in plan.notes
            if note.source_staff in (1, 2)
        }
        write_quantized_midi(
            staging / QUANTIZED_MID,
            plan.quantization,
            plan.staff_split,
            staff_by_source=staff_by_source or None,
        )
        write_fragment_score(
            staging / SCORE_XML,
            plan.quantization,
            plan.notes,
            measures=plan.measures,
            staff_split=plan.staff_split,
            style=plan.style,
            settings=plan.settings,
            key=(plan.key_tonic, plan.key_mode),
            title=manifest["title"],
        )
        if plan.harmony is not None:
            harmony = plan.harmony.to_dict()
            harmony["provenance"] = {
                "parent_project": plan.project.name,
                "parent_event_id": manifest["parent"]["event_id"],
                "parent_notes_fingerprint": plan.parent_fingerprint,
                "parent_analysis_sha256": plan.parent_analysis_sha256,
                "parent_measure_start": plan.measure_start,
                "parent_measure_end": plan.measure_end,
                "parent_step_start": plan.step_start,
            }
            _write_json(staging / FRAGMENT_HARMONY_JSON, harmony)
        # The manifest is last: a folder with a manifest is a complete fragment.
        _write_json(staging / FRAGMENT_JSON, manifest)
        _validate_staged_fragment(staging, expected_id=fragment_id)
        os.replace(staging, final)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return final


def _validate_staged_fragment(folder: Path, *, expected_id: str) -> None:
    """Read back every referenced artifact before publishing the staged folder."""

    manifest = read_manifest(folder)
    if str(manifest.get("id") or "") != expected_id:
        raise FragmentError("The staged fragment id does not match its destination.")
    artifacts = manifest.get("artifacts") or {}
    referenced = [str(name) for name in artifacts.values() if name is not None]
    missing = [name for name in referenced if not (folder / name).is_file()]
    if missing:
        raise FragmentError(
            f"The staged fragment is missing {', '.join(sorted(missing))}."
        )
    transcription = read_transcription(folder / NOTES_JSON)
    grid = read_grid(folder)
    expected_notes = int((manifest.get("boundary") or {}).get("notes") or 0)
    if len(transcription.notes) != expected_notes or len(grid.get("notes") or []) != expected_notes:
        raise FragmentError("The staged fragment note counts do not match its manifest.")

    import pretty_midi
    import xml.etree.ElementTree as element_tree

    for name in (FRAGMENT_PERFORMANCE_MID, QUANTIZED_MID):
        try:
            pretty_midi.PrettyMIDI(str(folder / name))
        except Exception as exc:
            raise FragmentError(f"The staged fragment has invalid {name}: {exc}") from exc
    try:
        element_tree.parse(folder / SCORE_XML)
    except (OSError, element_tree.ParseError) as exc:
        raise FragmentError(f"The staged fragment has invalid MusicXML: {exc}") from exc
    harmony_name = artifacts.get("harmony")
    if harmony_name:
        _read_json_object(folder / str(harmony_name))


# --------------------------------------------------------------------------
# Reading a saved fragment
# --------------------------------------------------------------------------


def read_grid(folder: Path) -> dict:
    payload = _read_json_object(Path(folder) / FRAGMENT_GRID_JSON)
    if int(payload.get("version") or 0) != GRID_SCHEMA_VERSION:
        raise FragmentError(f"{FRAGMENT_GRID_JSON} uses a schema version this program does not read.")
    if not isinstance(payload.get("notes"), list) or not isinstance(payload.get("anchors"), list):
        raise FragmentError(f"{FRAGMENT_GRID_JSON} has no notes or no anchors.")
    return payload


def grid_notes(grid: dict) -> list[FragmentNote]:
    notes = []
    try:
        for position, item in enumerate(grid.get("notes") or []):
            notes.append(
                FragmentNote(
                    index=int(item.get("index", position)),
                    pitch=int(item["pitch"]),
                    velocity=int(item["velocity"]),
                    onset_step=int(item["onset_step"]),
                    offset_step=int(item["offset_step"]),
                    onset_seconds=float(item["onset_seconds"]),
                    offset_seconds=float(item["offset_seconds"]),
                    carried_in=bool(item.get("carried_in")),
                    cut_off=bool(item.get("cut_off")),
                    parent_index=-1,
                    staff=int(item["staff"]) if item.get("staff") is not None else None,
                    voice=int(item["voice"]) if item.get("voice") is not None else None,
                    written_offset_step=(
                        int(item["written_offset_step"]) if item.get("written_offset_step") is not None else None
                    ),
                    printed=bool(item.get("printed", True)),
                )
            )
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise FragmentError(f"{FRAGMENT_GRID_JSON} has a note this program cannot read: {exc}") from exc
    return notes


def grid_quantization(grid: dict) -> Quantization:
    """The fragment's stored grid as a ``Quantization``, for the existing writers and pages."""

    try:
        return _fragment_quantization(
            grid_notes(grid),
            bpm=float(grid["bpm"]),
            numerator=int(grid["numerator"]),
            denominator=int(grid["denominator"]),
            subdivision=int(grid["subdivision"]),
            barline_seconds=float(grid.get("barline_seconds") or 0.0),
            grid_mode=str(grid.get("grid_mode") or "fixed"),
            settings=settings_from_dict(grid.get("settings") if isinstance(grid.get("settings"), dict) else None),
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        raise FragmentError(f"{FRAGMENT_GRID_JSON} does not describe a grid: {exc}") from exc


def read_harmony(folder: Path, manifest: dict | None = None) -> HarmonicAnalysis | None:
    """The fragment's harmony, or None when it was saved without any."""

    manifest = manifest if manifest is not None else read_manifest(folder)
    name = (manifest.get("artifacts") or {}).get("harmony")
    if not name:
        return None
    path = Path(folder) / str(name)
    if not path.is_file():
        return None
    try:
        return analysis_from_dict(_read_json_object(path))
    except (FragmentError, KeyError, TypeError, ValueError):
        return None


# --------------------------------------------------------------------------
# Preview audio (derived; never stored in the fragment folder)
# --------------------------------------------------------------------------


PREVIEW_MODES = ("performance", "quantized")
PREVIEW_SAMPLE_RATE = 22_050


def preview_notes(folder: Path, mode: str = "performance") -> tuple[list[NoteEvent], float]:
    """Notes and duration for a library preview of ``folder``.

    ``performance`` uses the as-played times in ``notes.json``. ``quantized``
    uses the metronomic grid times. Pedal is omitted: the preview synth has
    no sustain-pedal model.
    """

    folder = Path(folder)
    chosen = str(mode or "performance").strip().lower()
    if chosen not in PREVIEW_MODES:
        raise FragmentError(f"Preview mode must be one of {', '.join(PREVIEW_MODES)}.")
    manifest = read_manifest(folder)
    duration = duration_seconds(manifest)
    if chosen == "performance":
        notes_path = folder / str((manifest.get("artifacts") or {}).get("notes") or NOTES_JSON)
        if not notes_path.is_file():
            raise FragmentError(f"This fragment has no {notes_path.name} to preview.")
        notes = read_transcription(notes_path).notes
    else:
        grid = read_grid(folder)
        quantization = grid_quantization(grid)
        notes = [
            NoteEvent(
                pitch=item.pitch,
                onset=quantization.step_to_score(item.onset_step),
                offset=max(
                    quantization.step_to_score(item.offset_step),
                    quantization.step_to_score(item.onset_step) + MIN_NOTE_SECONDS,
                ),
                velocity=item.velocity,
            )
            for item in grid_notes(grid)
        ]
    if notes:
        duration = max(duration, max(note.offset for note in notes))
    return notes, float(duration)


def ensure_fragment_preview_wav(
    folder: Path,
    *,
    mode: str = "performance",
    sound: str | None = None,
    local: Path | None = None,
    sample_rate: int = PREVIEW_SAMPLE_RATE,
) -> Path:
    """Return a cached tone rendering of the fragment's notes.

    The WAV is written under ``.local/fragment_audio/`` and named with the
    sound id and a hash of the source notes or grid, so a changed fragment or
    a different timbre rebuilds itself. The fragment folder is never written.
    Deleting the cache only forces a rewrite.
    """

    import soundfile as sf

    from baroque_transcribe.preview_synth import DEFAULT_PREVIEW_SOUND, resolve_preview_sound, synthesize_preview

    folder = Path(folder)
    chosen = str(mode or "performance").strip().lower()
    if chosen not in PREVIEW_MODES:
        raise FragmentError(f"Preview mode must be one of {', '.join(PREVIEW_MODES)}.")
    if sample_rate < 8_000:
        raise FragmentError("The preview sample rate is too low.")
    try:
        preset = resolve_preview_sound(sound or DEFAULT_PREVIEW_SOUND)
    except ValueError as exc:
        raise FragmentError(str(exc)) from exc
    manifest = read_manifest(folder)
    ident = str(manifest["id"])
    source = folder / (
        str((manifest.get("artifacts") or {}).get("notes") or NOTES_JSON)
        if chosen == "performance"
        else FRAGMENT_GRID_JSON
    )
    if not source.is_file():
        raise FragmentError(f"This fragment has no {source.name} to preview.")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    cache = fragment_audio_root(local)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{ident}-{chosen}-{preset.id}-{digest}.wav"
    if target.is_file() and target.stat().st_size > 0:
        return target
    notes, duration = preview_notes(folder, chosen)
    if not notes:
        raise FragmentError("This fragment has no notes to preview.")
    audio = synthesize_preview(notes, int(sample_rate), duration, sound=preset)
    for old in cache.glob(f"{ident}-{chosen}-{preset.id}-*.wav"):
        if old != target:
            old.unlink(missing_ok=True)
    temporary = target.with_name(target.stem + ".tmp.wav")
    try:
        sf.write(temporary, audio, int(sample_rate))
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


# --------------------------------------------------------------------------
# Clocks
# --------------------------------------------------------------------------


def duration_seconds(manifest: dict) -> float:
    return float(manifest["time"]["duration_seconds"])


def working_start(manifest: dict) -> float:
    """``t0``: where fragment time zero falls in the parent's working audio."""

    return float(manifest["time"]["working_audio"]["start_seconds"])


def video_origin(manifest: dict) -> float:
    """``E + t0``: where fragment time zero falls in the source video, before any sync offset."""

    return float(manifest["time"]["source_video"]["start_seconds"])


def sync_offset(manifest: dict) -> float:
    sync = manifest.get("sync") if isinstance(manifest.get("sync"), dict) else {}
    return _number(sync.get("offset_seconds"))


def fragment_to_working(seconds: float, manifest: dict) -> float:
    return float(seconds) + working_start(manifest)


def working_to_fragment(seconds: float, manifest: dict) -> float:
    return float(seconds) - working_start(manifest)


def fragment_to_video(seconds: float, manifest: dict) -> float:
    """Source-video seconds for a fragment time, with the fragment's sync offset applied."""

    return float(seconds) + video_origin(manifest) + sync_offset(manifest)


def video_to_fragment(seconds: float, manifest: dict) -> float:
    """Fragment seconds for a video playhead. Inverse of ``fragment_to_video``."""

    return float(seconds) - video_origin(manifest) - sync_offset(manifest)


def video_window(manifest: dict) -> tuple[float, float]:
    """Where the player should loop: the fragment's ends on the video clock.

    Both ends carry the same sync offset, so the length is always the
    fragment's duration.
    """

    return fragment_to_video(0.0, manifest), fragment_to_video(duration_seconds(manifest), manifest)


# --------------------------------------------------------------------------
# Editing
# --------------------------------------------------------------------------


def update_fragment_metadata(
    folder: Path,
    *,
    title: str | None = None,
    description: str | None = None,
    practice_notes: str | None = None,
    tags=None,
    composer: str | None = None,
    performer: str | None = None,
    now: datetime | None = None,
) -> dict:
    """Change the descriptive fields of a fragment and leave everything else.

    A field left as None is not touched. The manifest is the only file
    written.
    """

    folder = Path(folder)
    updates = {
        "title": title,
        "description": description,
        "practice_notes": practice_notes,
        "composer": composer,
        "performer": performer,
    }
    with _MANIFEST_LOCK:
        manifest = read_manifest(folder)
        for name in _EDITABLE_TEXT:
            value = updates[name]
            if value is None:
                continue
            text = " ".join(str(value).split()) if name == "title" else str(value).strip()
            if name == "title" and not text:
                raise FragmentError("A fragment needs a title.")
            manifest[name] = text
        if tags is not None:
            manifest["tags"] = normalize_tags(tags)
        manifest["modified_at"] = _timestamp(now)
        _write_json(folder / FRAGMENT_JSON, manifest)
    return manifest


def set_fragment_sync(folder: Path, offset: float, *, now: datetime | None = None) -> float:
    """Store the fragment's sync offset and leave every other field.

    The offset says where the player should look for the passage. It does
    not change what the passage is: the duration and both stored intervals
    stay as they were written.
    """

    folder = Path(folder)
    value = float(offset)
    if not math.isfinite(value):
        # NaN would be written as a token that is not JSON, and the manifest
        # would stop being readable.
        raise FragmentError("The sync offset is not a number.")
    stored = max(-SYNC_OFFSET_LIMIT, min(SYNC_OFFSET_LIMIT, round(value, 4)))
    with _MANIFEST_LOCK:
        manifest = read_manifest(folder)
        sync = manifest.get("sync") if isinstance(manifest.get("sync"), dict) else {}
        sync["offset_seconds"] = stored
        manifest["sync"] = sync
        manifest["modified_at"] = _timestamp(now)
        _write_json(folder / FRAGMENT_JSON, manifest)
    return stored


def describe_plan(plan: FragmentPlan) -> list[str]:
    """Plain sentences about a planned fragment, for the command line and the form."""

    lines = [
        f"Measures {plan.measure_start} to {plan.measure_end} of {plan.parent_measures} "
        f"({plan.measures} measure{'' if plan.measures == 1 else 's'}), {plan.duration:.2f} seconds.",
        f"{len(plan.notes)} notes: {plan.owned} begin in the selection, {plan.carried_in} already sounding "
        f"at the first barline, {plan.cut_off} cut at the last barline.",
    ]
    if plan.carried_in:
        lines.append(
            "A note already sounding at the first barline is struck again at the start of the fragment, "
            "at its original loudness. That is an approximation: the real attack came earlier."
        )
    if plan.pedal_down_at_start:
        lines.append("The sustain pedal is already down at the start and is pressed at time zero.")
    lines.append(f"Key: {plan.key_tonic} {plan.key_mode} ({plan.key_source}).")
    if plan.roman:
        lines.append("Harmony: " + " ".join(plan.roman))
    lines.extend(plan.warnings)
    return lines
