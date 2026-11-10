"""Tempo-following grid, clustering, and comparison acceptance checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from baroque_transcribe.compare import compare_project
from baroque_transcribe.metrics import measure_notation
from baroque_transcribe.notation_settings import NotationSettings
from baroque_transcribe.quantize import (
    cluster_onsets,
    quantize_notes,
    track_attack_steps,
)
from baroque_transcribe.schema import NoteEvent, Transcription, read_transcription
from baroque_transcribe.sync_map import measure_anchors


FIXTURES = Path(__file__).parent / "fixtures"


def _steady_eighths(bpm: float = 90.0, count: int = 32) -> list[NoteEvent]:
    eighth = 60.0 / bpm / 2.0
    return [
        NoteEvent(pitch=60 + (index % 5), onset=index * eighth, offset=index * eighth + eighth * 0.9, velocity=80)
        for index in range(count)
    ]


def test_fixed_mode_matches_previous_behaviour():
    notes = [NoteEvent(pitch=60, onset=0.13, offset=0.50, velocity=80)]
    result = quantize_notes(notes, bpm=120, numerator=4, denominator=4, subdivision=16, downbeat=0.0)
    assert result.grid_mode == "fixed"
    assert result.notes[0].onset == pytest.approx(0.125)
    assert result.notes[0].offset == pytest.approx(0.5)
    assert result.grid[0].source_index == 0


def test_cluster_onsets_keeps_same_pitch_attacks_separate():
    notes = [
        NoteEvent(60, 0.00, 0.10, 80),
        NoteEvent(64, 0.01, 0.10, 70),
        NoteEvent(60, 0.02, 0.12, 75),  # same pitch as first: new cluster
    ]
    clusters = cluster_onsets(notes, window_seconds=0.03)
    assert len(clusters) == 2
    assert clusters[0].members == [0, 1]
    assert clusters[1].members == [2]


def test_cluster_onsets_does_not_absorb_a_fast_run():
    notes = [NoteEvent(60 + index, index * 0.08, index * 0.08 + 0.06, 80) for index in range(6)]
    clusters = cluster_onsets(notes, window_seconds=0.03)
    assert len(clusters) == 6


def test_follow_grid_recovers_steady_eighths():
    notes = _steady_eighths(90.0, 24)
    result = quantize_notes(
        notes,
        bpm=90,
        numerator=4,
        denominator=4,
        subdivision=16,
        downbeat=0.0,
        settings=NotationSettings(grid="follow"),
    )
    steps = [item.onset_step for item in result.grid]
    # Eighth notes on a 16th grid advance by 2.
    advances = [b - a for a, b in zip(steps, steps[1:])]
    assert advances.count(2) >= 20
    metrics = measure_notation(result)
    assert metrics.attacks_on_beat_ratio >= 0.9
    assert metrics.mean_onset_residual_steps < 0.05


def test_follow_grid_tracks_accelerando_without_slipping():
    # Ideal steps 0,2,4,... with a gradually shrinking eighth (~1% per attack).
    times = []
    t = 0.0
    eighth = 0.40
    for _ in range(24):
        times.append(t)
        t += eighth
        eighth *= 0.99
    steps, _step_times, warnings = track_attack_steps(
        times,
        bpm=75,
        subdivision=16,
        downbeat=0.0,
        silence_reanchor_seconds=1.5,
    )
    advances = [b - a for a, b in zip(steps, steps[1:])]
    assert advances == [2] * (len(times) - 1)
    assert not warnings


def test_follow_grid_reanchors_after_silence():
    notes = _steady_eighths(90.0, 8)
    gap = 3.0
    later = [
        NoteEvent(
            pitch=72,
            onset=notes[-1].offset + gap + index * (60.0 / 90.0 / 2.0),
            offset=notes[-1].offset + gap + index * (60.0 / 90.0 / 2.0) + 0.2,
            velocity=80,
        )
        for index in range(8)
    ]
    result = quantize_notes(
        notes + later,
        bpm=90,
        numerator=4,
        denominator=4,
        subdivision=16,
        settings=NotationSettings(grid="follow"),
    )
    first_block = [item.onset_step for item in result.grid if item.pitch < 70]
    second_block = [item.onset_step for item in result.grid if item.pitch >= 70]
    assert second_block[0] - first_block[-1] > 8
    assert [b - a for a, b in zip(second_block, second_block[1:])].count(2) >= 6


def test_metronomic_input_matches_fixed_steps():
    notes = _steady_eighths(120.0, 16)
    fixed = quantize_notes(notes, bpm=120, numerator=4, denominator=4, subdivision=16)
    follow = quantize_notes(
        notes,
        bpm=120,
        numerator=4,
        denominator=4,
        subdivision=16,
        settings=NotationSettings(grid="follow"),
    )
    assert [item.onset_step for item in fixed.grid] == [item.onset_step for item in follow.grid]


def test_follow_grid_is_deterministic():
    notes = _steady_eighths(88.0, 20)
    settings = NotationSettings(grid="follow")
    a = quantize_notes(notes, bpm=88, numerator=4, denominator=4, subdivision=16, settings=settings)
    b = quantize_notes(notes, bpm=88, numerator=4, denominator=4, subdivision=16, settings=settings)
    assert [(item.onset_step, item.offset_step, item.pitch) for item in a.grid] == [
        (item.onset_step, item.offset_step, item.pitch) for item in b.grid
    ]


def test_measure_anchors_stay_strictly_increasing_on_follow_grid():
    notes = _steady_eighths(90.0, 48)
    quantization = quantize_notes(
        notes,
        bpm=90,
        numerator=4,
        denominator=4,
        subdivision=16,
        settings=NotationSettings(grid="follow"),
    )
    anchors = measure_anchors(quantization)
    times = [anchor.time for anchor in anchors]
    assert times == sorted(times)
    assert all(later > earlier for earlier, later in zip(times, times[1:]))


def test_fixed_mode_anchors_unchanged_from_source_onset_on_the_barline():
    notes = [
        NoteEvent(60, 0.0, 0.4, 80),
        NoteEvent(62, 2.0, 2.3, 80),
    ]
    quantization = quantize_notes(notes, bpm=120, numerator=4, denominator=4, subdivision=16)
    anchors = measure_anchors(quantization)
    assert anchors[0].time == pytest.approx(0.0)
    assert anchors[1].time == pytest.approx(2.0)


def test_compare_does_not_touch_project_score_or_notes(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    notes = _steady_eighths(90.0, 16)
    Transcription(notes=notes, pedals=[], model="test", source="test").write(project / "notes.json")
    (project / "manifest.json").write_text(
        json.dumps(
            {
                "title": "Test",
                "tempo": {
                    "bpm": 90,
                    "time_signature": "4/4",
                    "subdivision": 16,
                    "downbeat_seconds": 0.0,
                },
            }
        ),
        encoding="utf-8",
    )
    from baroque_transcribe.midi_export import write_raw_midi

    write_raw_midi(project / "raw.mid", notes, [])
    before_notes = (project / "notes.json").read_bytes()
    before_raw = (project / "raw.mid").read_bytes()
    report = compare_project(project)
    assert (project / "comparison" / "report.json").is_file()
    assert (project / "comparison" / "literal_fixed.musicxml").is_file()
    assert (project / "comparison" / "literal_follow.musicxml").is_file()
    assert not (project / "score.musicxml").exists()
    assert (project / "notes.json").read_bytes() == before_notes
    assert (project / "raw.mid").read_bytes() == before_raw
    assert report.raw_midi_unchanged
    assert "literal_fixed" in report.to_dict()["variants"]
    assert "literal_follow" in report.to_dict()["variants"]


def test_tip4_fixture_follow_improves_on_beat_ratio():
    path = FIXTURES / "tip4_measures_24_35.json"
    if not path.is_file():
        pytest.skip("fixture missing")
    notes = read_transcription(path).notes
    meta = json.loads((FIXTURES / "tip4_measures_24_35.meta.json").read_text(encoding="utf-8"))
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
    fixed_metrics = measure_notation(fixed)
    follow_metrics = measure_notation(follow)
    assert follow_metrics.attacks_on_beat_ratio > fixed_metrics.attacks_on_beat_ratio
    assert follow_metrics.mean_onset_residual_steps < fixed_metrics.mean_onset_residual_steps
    assert follow_metrics.note_count == fixed_metrics.note_count == len(notes)
