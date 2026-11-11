"""Write standard MIDI files.

``raw.mid`` uses a 120 BPM tempo only as a clock. Note times are the detected
times in seconds, so playback stays lined up with ``working.wav`` as long as
that tempo is left alone. ``quantized.mid`` uses the musical tempo and the
snapped grid, split into treble and bass tracks for MuseScore.
"""

from __future__ import annotations

import os
from pathlib import Path

import pretty_midi

from baroque_transcribe.defaults import RAW_MIDI_RESOLUTION, RAW_MIDI_TEMPO
from baroque_transcribe.quantize import Quantization
from baroque_transcribe.schema import NoteEvent, PedalEvent
from baroque_transcribe.storage_io import temporary_path


def _instrument(name: str) -> pretty_midi.Instrument:
    return pretty_midi.Instrument(program=0, is_drum=False, name=name)


def _add_notes(instrument: pretty_midi.Instrument, notes: list[tuple[int, float, float, int]]) -> None:
    for pitch, start, end, velocity in notes:
        if end <= start:
            end = start + 0.01
        instrument.notes.append(
            pretty_midi.Note(
                velocity=int(min(127, max(1, velocity))),
                pitch=int(pitch),
                start=max(0.0, float(start)),
                end=max(0.0, float(end)),
            )
        )
    instrument.notes.sort(key=lambda item: (item.start, item.pitch))


def write_raw_midi(
    path: Path,
    notes: list[NoteEvent],
    pedals: list[PedalEvent],
) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=RAW_MIDI_TEMPO, resolution=RAW_MIDI_RESOLUTION)
    piano = _instrument("Piano")
    _add_notes(piano, [(note.pitch, note.onset, note.offset, note.velocity) for note in notes])
    for pedal in pedals:
        piano.control_changes.append(
            pretty_midi.ControlChange(number=64, value=127, time=max(0.0, pedal.onset))
        )
        piano.control_changes.append(
            pretty_midi.ControlChange(number=64, value=0, time=max(0.0, pedal.offset))
        )
    piano.control_changes.sort(key=lambda change: (change.time, -change.value))
    midi.instruments.append(piano)
    _write_midi_atomic(midi, path)


def write_quantized_midi(
    path: Path,
    quantization: Quantization,
    staff_split: int,
    *,
    staff_by_source: dict[int, int] | None = None,
) -> None:
    midi = pretty_midi.PrettyMIDI(initial_tempo=quantization.bpm, resolution=RAW_MIDI_RESOLUTION)
    midi.time_signature_changes.append(
        pretty_midi.TimeSignature(quantization.numerator, quantization.denominator, 0.0)
    )
    treble = _instrument("Piano right hand")
    bass = _instrument("Piano left hand")
    treble_notes: list[tuple[int, float, float, int]] = []
    bass_notes: list[tuple[int, float, float, int]] = []
    for note in quantization.grid:
        start = quantization.step_to_score(note.onset_step)
        end = quantization.step_to_score(note.offset_step)
        staff = None if staff_by_source is None else staff_by_source.get(note.source_index)
        if staff == 1:
            treble_notes.append((note.pitch, start, end, note.velocity))
        elif staff == 2:
            bass_notes.append((note.pitch, start, end, note.velocity))
        elif note.pitch >= staff_split:
            treble_notes.append((note.pitch, start, end, note.velocity))
        else:
            bass_notes.append((note.pitch, start, end, note.velocity))
    _add_notes(treble, treble_notes)
    _add_notes(bass, bass_notes)
    midi.instruments.append(treble)
    midi.instruments.append(bass)
    _write_midi_atomic(midi, path)


def _write_midi_atomic(midi: pretty_midi.PrettyMIDI, path: Path) -> None:
    """Publish a MIDI only after pretty_midi can read the completed temporary file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    try:
        midi.write(str(temporary))
        pretty_midi.PrettyMIDI(str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
