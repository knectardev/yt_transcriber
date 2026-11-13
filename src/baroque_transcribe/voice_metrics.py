"""Continuity metrics for interpreted contrapuntal voices."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from baroque_transcribe.engrave import Interpretation, WrittenNote
from baroque_transcribe.quantize import Quantization
from baroque_transcribe.storage_io import write_json_atomic


@dataclass
class VoiceContinuity:
    staff: int
    voice: int
    note_count: int
    pitch_min: int
    pitch_max: int
    avg_melodic_interval: float
    max_melodic_interval: int
    substantial_interruptions: int
    max_gap_steps: int
    flags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _stats_for_voice(members: list[WrittenNote], *, interruption_steps: int) -> VoiceContinuity:
    ordered = sorted(members, key=lambda item: (item.onset_step, item.pitch))
    pitches = [item.pitch for item in ordered]
    intervals = [abs(b - a) for a, b in zip(pitches, pitches[1:])]
    gaps = [
        following.onset_step - previous.written_offset_step
        for previous, following in zip(ordered, ordered[1:])
    ]
    interruptions = sum(1 for gap in gaps if gap >= interruption_steps)
    flags: list[str] = []
    if len(ordered) <= 2:
        flags.append("sparse")
    if intervals and max(intervals) >= 12:
        flags.append("large_leap")
    if interruptions >= 3:
        flags.append("discontinuous")
    if pitches and (max(pitches) - min(pitches)) >= 24 and len(ordered) <= 5:
        flags.append("register_odd")
    return VoiceContinuity(
        staff=ordered[0].staff,
        voice=ordered[0].voice,
        note_count=len(ordered),
        pitch_min=min(pitches),
        pitch_max=max(pitches),
        avg_melodic_interval=round(sum(intervals) / len(intervals), 2) if intervals else 0.0,
        max_melodic_interval=max(intervals) if intervals else 0,
        substantial_interruptions=interruptions,
        max_gap_steps=max(gaps) if gaps else 0,
        flags=flags,
    )


def voice_continuity_report(
    interpretation: Interpretation,
    quantization: Quantization,
    *,
    start_measure: int | None = None,
    end_measure: int | None = None,
) -> list[VoiceContinuity]:
    """Summarise each interpreted voice, optionally clipped to a measure span."""

    per = quantization.steps_per_measure
    first = -quantization.lead_steps
    interruption = max(1, round(quantization.subdivision / 8))  # eighth note
    notes = interpretation.notes
    if start_measure is not None or end_measure is not None:
        lo = first + ((start_measure or 1) - 1) * per
        hi = first + (end_measure or 10_000) * per
        notes = [note for note in notes if lo <= note.onset_step < hi]
    groups: dict[tuple[int, int], list[WrittenNote]] = {}
    for note in notes:
        groups.setdefault((note.staff, note.voice), []).append(note)
    return [
        _stats_for_voice(members, interruption_steps=interruption)
        for key, members in sorted(groups.items())
    ]


def write_voice_metrics(path: Path, rows: list[VoiceContinuity]) -> Path:
    write_json_atomic(path, {"voices": [row.to_dict() for row in rows]})
    return path


def staff_split_assessment(interpretation: Interpretation, staff_split: int = 60) -> dict:
    """Describe how the pitch threshold interacts with voice assignment."""

    near = [note for note in interpretation.notes if abs(note.pitch - staff_split) <= 5]
    spanning = []
    groups: dict[tuple[int, int], list[WrittenNote]] = {}
    for note in interpretation.notes:
        groups.setdefault((note.staff, note.voice), []).append(note)
    for (staff, voice), members in groups.items():
        pitches = [item.pitch for item in members]
        if min(pitches) < staff_split - 5 and max(pitches) > staff_split + 5:
            spanning.append(
                {
                    "staff": staff,
                    "voice": voice,
                    "pitch_min": min(pitches),
                    "pitch_max": max(pitches),
                    "note_count": len(members),
                }
            )
    return {
        "staff_split": staff_split,
        "timing": (
            "Staff is chosen during voice assignment (not as a pre-pass). "
            "Each note predicts a staff from pitch, with sticky continuation "
            "within five semitones of the split so a line can keep its staff."
        ),
        "notes_near_split": len(near),
        "near_split_treble": sum(1 for note in near if note.staff == 1),
        "near_split_bass": sum(1 for note in near if note.staff == 2),
        "voices_spanning_both_registers": spanning,
        "recommendation": (
            "Voice ids are global and keep Colour voices stable across the staff "
            "split; the pitch threshold still chooses which staff prints each note. "
            "Near-split sticky continuation can leave a mid-range line on the treble "
            "a little longer than a hand change would."
        ),
    }
