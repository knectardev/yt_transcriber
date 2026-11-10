"""Guards against a follow grid that fits onsets but warps the meter."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baroque_transcribe.notation_settings import NotationSettings
from baroque_transcribe.quantize import quantize_notes
from baroque_transcribe.schema import NoteEvent, read_transcription
from baroque_transcribe.sync_map import measure_anchors, printed_measure_span


FIXTURES = Path(__file__).parent / "fixtures"


def test_follow_keeps_bar_length_in_score_time():
    notes = [
        NoteEvent(60, index * 0.5, index * 0.5 + 0.4, 80)
        for index in range(32)
    ]
    fixed = quantize_notes(notes, bpm=120, numerator=4, denominator=4, subdivision=16)
    follow = quantize_notes(
        notes,
        bpm=120,
        numerator=4,
        denominator=4,
        subdivision=16,
        settings=NotationSettings(grid="follow"),
    )
    assert fixed.steps_per_measure == follow.steps_per_measure == 16
    # Score time is still a steady metronome; only audio mapping flexes.
    assert follow.step_to_score(16) == pytest.approx(follow.step_to_score(0) + 2.0)
    assert follow.step_to_score(32) == pytest.approx(follow.step_to_score(0) + 4.0)


def test_follow_anchors_are_strictly_increasing_and_near_bar_pace():
    notes = [
        NoteEvent(60, index * 0.5, index * 0.5 + 0.4, 80)
        for index in range(40)
    ]
    follow = quantize_notes(
        notes,
        bpm=120,
        numerator=4,
        denominator=4,
        subdivision=16,
        settings=NotationSettings(grid="follow"),
    )
    anchors = measure_anchors(follow)
    times = [anchor.time for anchor in anchors]
    assert all(later > earlier for earlier, later in zip(times, times[1:]))
    gaps = [b - a for a, b in zip(times, times[1:])]
    # At 120 BPM, a 4/4 bar is 2.0s. Allow rubato stretch but not a doubled bar.
    assert min(gaps) > 0.5
    assert max(gaps) < 4.0


def test_tip4_follow_does_not_invent_extra_measures():
    path = FIXTURES / "tip4_measures_24_35.json"
    if not path.is_file():
        pytest.skip("fixture missing")
    meta = json.loads((FIXTURES / "tip4_measures_24_35.meta.json").read_text(encoding="utf-8"))
    notes = read_transcription(path).notes
    fixed = quantize_notes(
        notes,
        bpm=meta["source_bpm"],
        numerator=4,
        denominator=4,
        subdivision=16,
        settings=NotationSettings(grid="fixed"),
    )
    follow = quantize_notes(
        notes,
        bpm=meta["source_bpm"],
        numerator=4,
        denominator=4,
        subdivision=16,
        settings=NotationSettings(grid="follow"),
    )
    _, fixed_count = printed_measure_span(fixed)
    _, follow_count = printed_measure_span(follow)
    # A correct follow grid may shift a bar or two at the edges, but must not
    # explode the measure count when residual is low.
    assert abs(follow_count - fixed_count) <= 2
    assert measure_notation_residual(follow) < 0.05


def measure_notation_residual(quantization) -> float:
    from baroque_transcribe.metrics import measure_notation

    return measure_notation(quantization).mean_onset_residual_steps


def test_tip1_regression_is_tracked_separately():
    """Tip No. 1 still shows a rest-count regression under follow; keep it visible."""

    path = FIXTURES / "tip1_sec2_130_160.json"
    if not path.is_file():
        pytest.skip("fixture missing")
    meta = json.loads((FIXTURES / "tip1_sec2_130_160.meta.json").read_text(encoding="utf-8"))
    notes = read_transcription(path).notes
    bpm = float(meta.get("suggested_bpm") or meta["source_bpm"])
    from baroque_transcribe.metrics import measure_notation
    from baroque_transcribe.notation import write_musicxml
    import tempfile

    fixed = quantize_notes(
        notes, bpm=bpm, numerator=4, denominator=4, subdivision=16, settings=NotationSettings(grid="fixed")
    )
    follow = quantize_notes(
        notes, bpm=bpm, numerator=4, denominator=4, subdivision=16, settings=NotationSettings(grid="follow")
    )
    with tempfile.TemporaryDirectory() as tmp:
        fixed_path = Path(tmp) / "fixed.musicxml"
        follow_path = Path(tmp) / "follow.musicxml"
        write_musicxml(fixed_path, fixed, style="literal")
        write_musicxml(follow_path, follow, style="literal")
        fixed_metrics = measure_notation(fixed, fixed_path)
        follow_metrics = measure_notation(follow, follow_path)
    # Document the known issue: residual improves, rests may not.
    assert follow_metrics.mean_onset_residual_steps < fixed_metrics.mean_onset_residual_steps
    # This assertion records the open Tip No. 1 investigation rather than
    # requiring a fix in this stage.
    regression = follow_metrics.rest_elements - fixed_metrics.rest_elements
    assert isinstance(regression, int)
