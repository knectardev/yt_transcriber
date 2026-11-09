"""Snap note times onto a metrical grid without changing pitches.

The grid is defined by a tempo, a time signature, a subdivision, and the
audio time of the first downbeat. Pitches and velocities are copied. This is
a rhythmic reading of the performance, not a new arrangement.
"""

from __future__ import annotations

from dataclasses import dataclass

from baroque_transcribe.schema import NoteEvent


class QuantizeError(ValueError):
    pass


@dataclass
class GridNote:
    pitch: int
    onset_step: int
    offset_step: int
    velocity: int
    source_index: int
    source_onset: float
    source_offset: float


@dataclass
class Quantization:
    bpm: float
    numerator: int
    denominator: int
    subdivision: int
    downbeat: float
    quantum: float
    steps_per_measure: int
    lead_steps: int
    notes: list[NoteEvent]
    grid: list[GridNote]

    @property
    def quarter_seconds(self) -> float:
        return 60.0 / self.bpm

    @property
    def measure_seconds(self) -> float:
        return self.steps_per_measure * self.quantum

    def step_to_audio(self, step: int) -> float:
        return self.downbeat + step * self.quantum

    def step_to_score(self, step: int) -> float:
        """Seconds from the start of the notated score, with the downbeat on a barline."""

        return (step + self.lead_steps) * self.quantum


def parse_time_signature(text: str) -> tuple[int, int]:
    parts = text.replace(" ", "").split("/")
    if len(parts) != 2:
        raise QuantizeError(f"Time signature must look like 4/4 or 6/8, not {text!r}.")
    try:
        numerator, denominator = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise QuantizeError(f"Time signature must look like 4/4 or 6/8, not {text!r}.") from exc
    if numerator < 1 or denominator not in {1, 2, 4, 8, 16, 32}:
        raise QuantizeError(f"Unsupported time signature {numerator}/{denominator}.")
    return numerator, denominator


def steps_per_measure(numerator: int, denominator: int, subdivision: int) -> int:
    if subdivision not in {8, 16, 32}:
        raise QuantizeError("Subdivision must be 8, 16, or 32.")
    if (numerator * subdivision) % denominator != 0:
        raise QuantizeError(
            f"{numerator}/{denominator} does not land on a {subdivision}-note grid. "
            "Use a finer subdivision or a different time signature."
        )
    return numerator * subdivision // denominator


def quantize_notes(
    notes: list[NoteEvent],
    *,
    bpm: float,
    numerator: int,
    denominator: int,
    subdivision: int,
    downbeat: float = 0.0,
) -> Quantization:
    if bpm <= 0:
        raise QuantizeError("Tempo must be a positive BPM.")
    per_measure = steps_per_measure(numerator, denominator, subdivision)
    quarter = 60.0 / bpm
    quantum = quarter * 4.0 / subdivision
    grid: list[GridNote] = []
    for index, note in enumerate(notes):
        onset_step = int(round((note.onset - downbeat) / quantum))
        offset_step = int(round((note.offset - downbeat) / quantum))
        if offset_step <= onset_step:
            offset_step = onset_step + 1
        grid.append(
            GridNote(
                pitch=int(note.pitch),
                onset_step=onset_step,
                offset_step=offset_step,
                velocity=int(note.velocity),
                source_index=index,
                source_onset=float(note.onset),
                source_offset=float(note.offset),
            )
        )
    grid.sort(key=lambda item: (item.onset_step, item.pitch, item.offset_step))
    snapped = [
        NoteEvent(
            pitch=item.pitch,
            onset=downbeat + item.onset_step * quantum,
            offset=downbeat + item.offset_step * quantum,
            velocity=item.velocity,
        )
        for item in grid
    ]
    if grid:
        earliest = min(item.onset_step for item in grid)
    else:
        earliest = 0
    first_bar = (min(earliest, 0) // per_measure) * per_measure
    lead_steps = -first_bar
    return Quantization(
        bpm=float(bpm),
        numerator=numerator,
        denominator=denominator,
        subdivision=subdivision,
        downbeat=float(downbeat),
        quantum=quantum,
        steps_per_measure=per_measure,
        lead_steps=lead_steps,
        notes=snapped,
        grid=grid,
    )
