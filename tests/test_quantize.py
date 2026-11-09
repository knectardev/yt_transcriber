import pytest

from baroque_transcribe.quantize import QuantizeError, parse_time_signature, quantize_notes
from baroque_transcribe.schema import NoteEvent


def test_sixteenth_grid_at_120_bpm():
    notes = [NoteEvent(pitch=60, onset=0.13, offset=0.50, velocity=80)]
    result = quantize_notes(
        notes,
        bpm=120,
        numerator=4,
        denominator=4,
        subdivision=16,
        downbeat=0.0,
    )
    assert result.quantum == pytest.approx(0.125)
    assert result.notes[0].onset == pytest.approx(0.125)
    assert result.notes[0].offset == pytest.approx(0.5)
    assert result.notes[0].pitch == 60
    assert result.lead_steps == 0


def test_downbeat_shifts_the_grid_and_keeps_a_leading_bar():
    notes = [NoteEvent(pitch=72, onset=0.4, offset=0.7, velocity=60)]
    result = quantize_notes(
        notes,
        bpm=120,
        numerator=4,
        denominator=4,
        subdivision=16,
        downbeat=0.5,
    )
    assert result.grid[0].onset_step == -1
    assert result.lead_steps == 16
    assert result.step_to_score(0) == pytest.approx(2.0)


def test_short_notes_keep_one_grid_step():
    notes = [NoteEvent(pitch=76, onset=0.0, offset=0.02, velocity=90)]
    result = quantize_notes(notes, bpm=60, numerator=3, denominator=4, subdivision=32)
    assert result.grid[0].offset_step - result.grid[0].onset_step == 1


def test_incompatible_meter_is_rejected():
    with pytest.raises(QuantizeError):
        parse_time_signature("4")
    with pytest.raises(QuantizeError):
        quantize_notes([], bpm=120, numerator=7, denominator=16, subdivision=8)
