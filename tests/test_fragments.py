"""Fragment extraction: membership, the playback window, clipping, clocks, and files."""

from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pytest

from baroque_transcribe import fragments as F
from baroque_transcribe.analysis import analysis_is_stale, load_analysis, notes_fingerprint
from baroque_transcribe.cli import main
from baroque_transcribe.defaults import (
    ANNOTATIONS_JSON,
    FRAGMENT_GRID_JSON,
    FRAGMENT_HARMONY_JSON,
    FRAGMENT_JSON,
    FRAGMENT_PERFORMANCE_MID,
    MANIFEST,
    NOTES_JSON,
    QUANTIZED_MID,
    SCORE_XML,
)
from baroque_transcribe.notation import write_musicxml
from baroque_transcribe.notation_settings import NotationSettings
from baroque_transcribe.quantize import GridNote, Quantization, quantize_notes
from baroque_transcribe.schema import PedalEvent, read_transcription
from baroque_transcribe.sync_map import measure_anchors, printed_measure_span, video_time

from fractions import Fraction

from fragment_helpers import (
    cut_at_barlines,
    fixture_project,
    make_project,
    measure_count,
    midi_notes,
    displaced_measures,
    n,
    parent_interpretation,
    parent_quantization,
    score_measures,
    score_notes,
    score_ties,
    total_measures,
    tree_digest,
)

STEP = 0.125  # one sixteenth at 120 BPM
BAR = 2.0  # one 4/4 measure at 120 BPM


def by_pitch(plan: F.FragmentPlan) -> dict[int, F.FragmentNote]:
    return {note.pitch: note for note in plan.notes}


def hand_grid(monkeypatch, tmp_path: Path, rows, *, pedals=None):
    """A parent whose grid is given outright, for timings a real quantizer would not produce.

    Each row is ``(pitch, onset_step, offset_step, source_onset, source_offset)``
    on a fixed 120 BPM grid of sixteenths.
    """

    project = make_project(
        tmp_path,
        [n(pitch, min(on, off), max(on, off) + 0.02) for pitch, _a, _b, on, off in rows],
        build=False,
        pedals=pedals,
    )
    grid = [
        GridNote(
            pitch=pitch,
            onset_step=onset_step,
            offset_step=offset_step,
            velocity=80,
            source_index=index,
            source_onset=on,
            source_offset=off,
        )
        for index, (pitch, onset_step, offset_step, on, off) in enumerate(rows)
    ]
    quantization = Quantization(
        bpm=120.0,
        numerator=4,
        denominator=4,
        subdivision=16,
        downbeat=0.0,
        quantum=STEP,
        steps_per_measure=16,
        lead_steps=0,
        notes=[],
        grid=grid,
        settings=NotationSettings(grid="fixed", style="literal"),
    )
    monkeypatch.setattr(F, "quantization_from_manifest", lambda *args, **kwargs: quantization)
    return project, quantization


# --------------------------------------------------------------------------
# Membership is decided on the grid
# --------------------------------------------------------------------------


def barline_piece():
    return [
        # Measure 1.
        n(60, 0.0, 0.5),
        n(62, 0.5, 1.0),
        n(64, 1.0, 1.5),
        n(65, 1.5, 1.9),
        # Measure 2 opens with an early attack: 50 ms before the metronome barline at 2.0 s.
        n(72, 1.95, 2.4),
        n(67, 2.5, 3.0),
        n(69, 3.0, 3.9),
        # Measure 3 opens with a late attack: 50 ms after the metronome barline at 4.0 s.
        n(74, 4.05, 4.5),
        n(76, 4.5, 5.0),
    ]


def test_early_attack_belongs_to_the_measure_it_is_printed_in(tmp_path: Path):
    project = make_project(tmp_path, barline_piece())
    grid = {item.pitch: item for item in parent_quantization(project).grid}
    assert grid[72].onset_step == 16  # printed on the first step of measure 2
    assert grid[72].source_onset < BAR  # but played before the metronome barline

    second = F.plan_fragment(project, 2, 2)
    assert sorted(by_pitch(second)) == [67, 69, 72]
    early = by_pitch(second)[72]
    assert (early.onset_step, early.carried_in, early.cut_off) == (0, False, False)
    assert early.onset_seconds == 0.0
    assert second.working_start == pytest.approx(1.95)
    # The written barline falls 50 ms into the fragment; the note is not moved onto it.
    assert second.barline_seconds == pytest.approx(0.05)

    # The same note sounds inside measure 1's metronome window, and is not in measure 1.
    first = F.plan_fragment(project, 1, 1)
    assert sorted(by_pitch(first)) == [60, 62, 64, 65]
    assert first.duration == pytest.approx(1.95)


def test_late_attack_belongs_to_the_measure_it_is_printed_in(tmp_path: Path):
    project = make_project(tmp_path, barline_piece())
    grid = {item.pitch: item for item in parent_quantization(project).grid}
    assert grid[74].onset_step == 32
    assert grid[74].source_onset > 2 * BAR

    third = F.plan_fragment(project, 3, 3)
    late = by_pitch(third)[74]
    assert late.onset_step == 0
    assert late.onset_seconds == 0.0
    assert third.working_start == pytest.approx(4.05)

    # It is not pulled into measure 2, whose window ends where measure 3 begins.
    second = F.plan_fragment(project, 2, 2)
    assert 74 not in by_pitch(second)
    assert second.working_start + second.duration == pytest.approx(4.05)


def test_note_struck_before_a_collided_anchor_is_not_dropped(monkeypatch, tmp_path: Path):
    # Measure 1's only attack is at 1.85 s. Measure 2's first attack is at 1.80 s,
    # which is not later, so the anchor falls back to the metronome barline at 2.0 s.
    project, quantization = hand_grid(
        monkeypatch,
        tmp_path,
        [
            (60, 0, 4, 1.85, 2.2),
            (72, 16, 20, 1.80, 2.3),
            (74, 24, 28, 3.0, 3.5),
        ],
    )
    anchors = measure_anchors(quantization)
    assert anchors[1].collided and anchors[1].time == pytest.approx(2.0)

    plan = F.plan_fragment(project, 2, 2)
    notes = by_pitch(plan)
    assert sorted(notes) == [72, 74]
    # The window opens at the attack, not at the later anchor.
    assert plan.working_start == pytest.approx(1.80)
    assert notes[72].onset_seconds == 0.0
    assert notes[74].onset_seconds == pytest.approx(1.2)
    assert plan.anchors[0]["seconds"] == pytest.approx(0.2)
    assert plan.anchors[0]["collided"] is True


def test_membership_matches_the_grid_rule_for_every_measure_range(tmp_path: Path):
    project = fixture_project(tmp_path, grid="follow", style="literal", build=False)
    parent = parent_quantization(project)
    first, count = printed_measure_span(parent)
    per = parent.steps_per_measure
    for start in range(1, count + 1):
        for end in range(start, min(count, start + 2) + 1):
            s0 = first + (start - 1) * per
            s1 = first + end * per
            expected = [item for item in parent.grid if item.onset_step < s1 and item.offset_step > s0]
            if not expected:
                continue
            plan = F.plan_fragment(project, start, end)
            assert sorted(note.parent_index for note in plan.notes) == sorted(item.source_index for item in expected)
            assert plan.owned == sum(1 for item in expected if item.onset_step >= s0)
            assert plan.carried_in == sum(1 for item in expected if item.onset_step < s0)
            assert plan.cut_off == sum(1 for item in expected if item.offset_step > s1)


# --------------------------------------------------------------------------
# Clipping
# --------------------------------------------------------------------------


def held_bass_piece(bass_end: float):
    return [
        n(60, 0.0, 0.5),
        n(48, 1.0, bass_end, velocity=53),
        n(72, 2.0, 2.5),
        n(74, 2.5, 3.0),
        n(76, 3.0, 3.9),
        n(77, 4.0, 4.5),
        n(79, 6.0, 7.5),
    ]


def test_carried_in_note_is_struck_again_at_the_start(tmp_path: Path):
    project = make_project(tmp_path, held_bass_piece(3.0))
    plan = F.plan_fragment(project, 2, 2)
    bass = by_pitch(plan)[48]
    assert (bass.carried_in, bass.cut_off) == (True, False)
    assert (bass.onset_step, bass.offset_step) == (0, 8)
    assert bass.onset_seconds == 0.0
    assert bass.offset_seconds == pytest.approx(1.0)  # its real release, 3.0 s, on the fragment clock
    assert bass.velocity == 53  # the original loudness, by design
    assert (plan.owned, plan.carried_in, plan.cut_off) == (3, 1, 0)
    assert any("approximation" in line for line in F.describe_plan(plan))

    folder = F.extract_fragment(project, 2, 2, title="Held bass", root=tmp_path / "fragments")
    stored = [note for note in F.read_grid(folder)["notes"] if note["carried_in"]]
    assert [note["pitch"] for note in stored] == [48]
    assert F.read_manifest(folder)["boundary"]["carried_in"] == 1
    performance = midi_notes(folder / FRAGMENT_PERFORMANCE_MID)[0]
    assert (48, 0.0) in [(pitch, round(start, 3)) for pitch, start, _end, _velocity in performance]


def test_note_crossing_both_barlines_fills_the_fragment(tmp_path: Path):
    project = make_project(tmp_path, held_bass_piece(7.0))
    plan = F.plan_fragment(project, 2, 2)
    bass = by_pitch(plan)[48]
    assert (bass.carried_in, bass.cut_off) == (True, True)
    assert (bass.onset_step, bass.offset_step) == (0, 16)
    assert bass.onset_seconds == 0.0
    assert bass.offset_seconds == pytest.approx(2.0)
    assert plan.duration == pytest.approx(2.0)


def test_cut_off_note_ends_at_the_end_barline(tmp_path: Path):
    project = make_project(tmp_path, held_bass_piece(3.0))
    plan = F.plan_fragment(project, 1, 1)
    bass = by_pitch(plan)[48]
    assert (bass.carried_in, bass.cut_off) == (False, True)
    assert (bass.onset_step, bass.offset_step) == (8, 16)
    assert bass.onset_seconds == pytest.approx(1.0)
    assert bass.offset_seconds == pytest.approx(2.0)
    assert plan.duration == pytest.approx(2.0)


@pytest.mark.parametrize("recorded_release", [2.60, 2.105], ids=["released-later", "released-at-once"])
def test_cut_off_note_struck_after_the_end_anchor_keeps_a_real_duration(
    monkeypatch, tmp_path: Path, recorded_release: float
):
    # Pitch 64 is printed in measure 1 (step 14) and tied over the barline, but
    # it was struck at 2.10 s. Measure 2 is estimated to begin at 1.95 s, the
    # attack of pitch 72. Ending the cut note at that estimate would put its
    # note-off 150 ms before its note-on.
    project, quantization = hand_grid(
        monkeypatch,
        tmp_path,
        [
            (60, 0, 4, 0.0, 0.5),
            (64, 14, 20, 2.10, recorded_release),
            (72, 16, 20, 1.95, 2.4),
        ],
    )
    assert measure_anchors(quantization)[1].time == pytest.approx(1.95)

    plan = F.plan_fragment(project, 1, 1)
    late = by_pitch(plan)[64]
    assert late.cut_off
    assert late.onset_seconds == pytest.approx(2.10)  # the real attack, not moved
    # Two written steps to the barline: a quarter of a second, not the 10 ms floor.
    assert late.offset_seconds - late.onset_seconds == pytest.approx(2 * STEP)
    assert plan.duration >= late.offset_seconds

    folder = F.extract_fragment(project, 1, 1, title="Late tie", root=tmp_path / "fragments")
    performance = midi_notes(folder / FRAGMENT_PERFORMANCE_MID)[0]
    assert all(end > start for _pitch, start, end, _velocity in performance)
    written = {pitch: (start, end) for pitch, start, end, _velocity in performance}
    assert written[64][0] == pytest.approx(2.10, abs=2e-3)
    assert written[64][1] == pytest.approx(2.35, abs=2e-3)
    stored = read_transcription(folder / NOTES_JSON).notes
    assert all(note.offset > note.onset for note in stored)


def test_carried_in_note_released_before_the_estimated_barline_gets_its_written_length(monkeypatch, tmp_path: Path):
    # Measure 2 opens with a rest; its anchor backs up from the first attack to
    # 2.06 s. The carried-in note was released at 2.063 s, 3 ms later.
    project, quantization = hand_grid(
        monkeypatch,
        tmp_path,
        [
            (50, 8, 17, 1.0, 2.063),
            (72, 18, 22, 2.31, 2.8),
        ],
    )
    assert measure_anchors(quantization)[1].time == pytest.approx(2.06)
    plan = F.plan_fragment(project, 2, 2)
    carried = by_pitch(plan)[50]
    assert carried.carried_in
    assert carried.onset_seconds == 0.0
    assert carried.offset_seconds == pytest.approx(STEP)  # one written step into the measure


def test_every_fragment_note_is_a_valid_interval_under_wild_timing(monkeypatch, tmp_path: Path):
    """Timings no quantizer would produce: anchors collide and attacks land far from their steps."""

    rng = random.Random(20261110)
    for case in range(60):
        rows = []
        for _ in range(rng.randint(3, 14)):
            onset_step = rng.randint(0, 47)
            offset_step = onset_step + rng.randint(1, 24)
            on = max(0.0, onset_step * STEP + rng.uniform(-0.35, 0.35))
            off = on + rng.choice([0.01, 0.02, rng.uniform(0.05, 3.0)])
            rows.append((rng.randint(36, 84), onset_step, offset_step, round(on, 5), round(off, 5)))
        project, quantization = hand_grid(monkeypatch, tmp_path / f"case{case}", rows)
        first, count = printed_measure_span(quantization)
        for start in range(1, count + 1):
            for end in range(start, count + 1):
                s0 = first + (start - 1) * 16
                s1 = first + end * 16
                if not any(item.onset_step < s1 and item.offset_step > s0 for item in quantization.grid):
                    with pytest.raises(F.FragmentError):
                        F.plan_fragment(project, start, end)
                    continue
                plan = F.plan_fragment(project, start, end)
                for note in plan.notes:
                    source = quantization.grid[note.parent_index]
                    assert 0.0 <= note.onset_seconds < note.offset_seconds <= plan.duration + 1e-9
                    assert note.offset_seconds - note.onset_seconds >= F.MIN_NOTE_SECONDS - 1e-9
                    assert 0 <= note.onset_step < note.offset_step <= (end - start + 1) * 16
                    if not note.carried_in:
                        # An attack inside the selection is never moved.
                        assert note.onset_seconds == pytest.approx(source.source_onset - plan.working_start, abs=2e-5)
                        if not note.cut_off:
                            assert note.offset_seconds == pytest.approx(
                                source.source_offset - plan.working_start, abs=2e-5
                            )


def test_pedal_is_cut_to_the_window(tmp_path: Path):
    project = make_project(
        tmp_path,
        held_bass_piece(3.0),
        pedals=[PedalEvent(0.1, 0.4), PedalEvent(1.0, 3.0), PedalEvent(3.5, 9.0)],
    )
    plan = F.plan_fragment(project, 2, 2)
    assert plan.working_start == pytest.approx(2.0)
    assert [(pedal.onset, pedal.offset) for pedal in plan.pedals] == [
        pytest.approx((0.0, 1.0)),
        pytest.approx((1.5, 2.0)),
    ]
    assert plan.pedal_down_at_start is True

    import pretty_midi

    folder = F.extract_fragment(project, 2, 2, title="Pedalled", root=tmp_path / "fragments")
    changes = pretty_midi.PrettyMIDI(str(folder / FRAGMENT_PERFORMANCE_MID)).instruments[0].control_changes
    assert [(change.number, change.value, round(change.time, 3)) for change in changes] == [
        (64, 127, 0.0),
        (64, 0, 1.0),
        (64, 127, 1.5),
        (64, 0, 2.0),
    ]
    assert F.read_manifest(folder)["boundary"]["pedal_down_at_start"] is True
    assert [(p.onset, p.offset) for p in read_transcription(folder / NOTES_JSON).pedals] == [(0.0, 1.0), (1.5, 2.0)]


# --------------------------------------------------------------------------
# The quantized artifacts reproduce the parent's measures
# --------------------------------------------------------------------------


def expected_quantized_slice(project: Path, start: int, end: int):
    """The parent's quantized.mid, restricted to the selected measures and moved to zero."""

    parent = parent_quantization(project)
    first, _count = printed_measure_span(parent)
    per = parent.steps_per_measure
    begin = parent.step_to_score(first + (start - 1) * per)
    finish = parent.step_to_score(first + end * per)
    tracks = []
    for track in midi_notes(project / QUANTIZED_MID):
        kept = []
        for pitch, note_start, note_end, velocity in track:
            if note_start < finish - 1e-3 and note_end > begin + 1e-3:
                kept.append((pitch, max(note_start, begin) - begin, min(note_end, finish) - begin, velocity))
        tracks.append(sorted(kept))
    return tracks


@pytest.mark.parametrize(
    ("grid", "style"),
    [("fixed", "literal"), ("follow", "literal"), ("follow", "readable")],
)
def test_fragment_reproduces_the_parent_measures(tmp_path: Path, grid: str, style: str):
    project = fixture_project(tmp_path, grid=grid, style=style)
    count = F.plan_fragment(project, 1, 1).parent_measures
    parent_score = score_notes(project / SCORE_XML)
    # Measures where a readable parent's own file misplaces notes (see
    # ``displaced_measures``). They are checked against the parent's
    # interpretation in the next test. A literal parent has none.
    unusable = displaced_measures(project)
    if style == "literal":
        assert not unusable
    before = tree_digest(project)
    compared = 0

    for start, end in [(1, 2), (4, 6), (7, 7), (9, 11), (count - 1, count)]:
        folder = F.extract_fragment(project, start, end, title=f"m{start}-{end}", root=tmp_path / "fragments")
        measures = end - start + 1

        # Quantized MIDI: the same notes, to the tick.
        expected = expected_quantized_slice(project, start, end)
        actual = midi_notes(folder / QUANTIZED_MID)
        assert len(actual) == len(expected) == 2
        for wanted, got in zip(expected, actual):
            assert len(got) == len(wanted)
            for want, have in zip(wanted, got):
                assert have[0] == want[0] and have[3] == want[3]
                assert have[1] == pytest.approx(want[1], abs=2e-3)
                assert have[2] == pytest.approx(want[2], abs=2e-3)

        # MusicXML: every measure has the parent's pitches, positions, written
        # durations and ties. The only change is at the two outer barlines,
        # where a tie into or out of the selection is dropped.
        assert total_measures(folder / SCORE_XML) == measures
        assert measure_count(folder / SCORE_XML) == [measures, measures]
        fragment_score = score_notes(folder / SCORE_XML)
        for staff in (1, 2):
            for number in range(1, measures + 1):
                if (staff, start + number - 1) in unusable:
                    continue
                compared += 1
                parent_measure = parent_score.get((staff, start + number - 1), [])
                expected_measure = cut_at_barlines(
                    parent_measure,
                    first=number == 1,
                    last=number == measures,
                    bar_length=Fraction(4),
                )
                fragment_measure = fragment_score[(staff, number)]
                if style == "readable" and number == 1:
                    # Second documented difference, readable scores only: the
                    # parent may print the short tied tail of a note that ended,
                    # on the grid, before this barline. That note is not a
                    # member, so the fragment leaves the tail out. Nothing else
                    # may differ, and the fragment never adds a note.
                    missing = Counter(expected_measure) - Counter(fragment_measure)
                    extra = Counter(fragment_measure) - Counter(expected_measure)
                    assert not extra
                    tails = {
                        (offset, pitch)
                        for offset, pitch, _length, tie in parent_measure
                        if offset == 0 and tie in {"stop", "continue"}
                    }
                    assert all((offset, pitch) in tails for offset, pitch, _length, _tie in missing)
                else:
                    assert fragment_measure == expected_measure

    # Every selected measure of a literal score was compared with the parent's
    # file, and a useful share of a readable one.
    assert compared == 22 if style == "literal" else compared >= 10

    # Extraction only reads the parent.
    assert tree_digest(project) == before


def test_readable_fragment_keeps_the_parent_voices_and_written_durations(tmp_path: Path):
    """Every measure, including those the parent's file prints too long."""

    project = fixture_project(tmp_path, grid="follow", style="readable", build=False)
    parent = parent_quantization(project)
    written = {note.source_index: note for note in parent_interpretation(project).notes}
    first, count = printed_measure_span(parent)
    per = parent.steps_per_measure
    for start in range(1, count + 1):
        end = min(count, start + 2)
        s0, s1 = first + (start - 1) * per, first + end * per
        plan = F.plan_fragment(project, start, end)
        for note in plan.notes:
            source = written[note.parent_index]
            assert (note.staff, note.voice) == (source.staff, source.voice)
            assert note.onset_step == max(source.onset_step, s0) - s0
            assert note.offset_step == min(source.sounding_offset_step, s1) - s0
            if source.written_offset_step <= s0:
                # The parent ended this note on paper before the barline. It is
                # in the fragment's notes and MIDI and is not printed.
                assert note.carried_in and not note.printed
            else:
                assert note.printed
                assert note.written_offset_step == min(source.written_offset_step, s1) - s0
        # The events handed to the writer are those notes and nothing else.
        events = F._interpretation(plan.notes, plan.settings, plan.staff_split).events_by_staff
        printed = sorted(
            (staff, voice_id, event.onset_step, pitch)
            for staff, voices in events.items()
            for voice_id, voice in enumerate(voices, start=1)
            for event in voice
            for pitch in event.pitches
        )
        assert printed == sorted((n_.staff, n_.voice, n_.onset_step, n_.pitch) for n_ in plan.notes if n_.printed)
        assert plan.unprinted == sum(1 for n_ in plan.notes if not n_.printed)


def test_overlap_the_parent_trimmed_at_the_barline_is_kept_but_not_printed(monkeypatch, tmp_path: Path):
    # Pitch 72 is held one step past the barline (a legato overlap). On the
    # grid it sounds in measure 2, so it is a member. Here the parent's readable
    # score ends it on paper at the barline, as its overlap trim does.
    notes = [
        n(67, 0.0, 0.5),
        n(69, 0.5, 1.0),
        n(71, 1.0, 1.5),
        n(72, 1.5, 2.125),
        n(74, 2.0, 2.5),
        n(76, 2.5, 3.0),
        n(77, 3.0, 4.0),
        n(48, 0.0, 2.0),
        n(43, 2.0, 4.0),
    ]
    project = make_project(tmp_path, notes, style="readable", build=False)
    interpretation = parent_interpretation(project)
    held = next(note for note in interpretation.notes if note.pitch == 72)
    assert (held.onset_step, held.sounding_offset_step) == (12, 17)
    held.written_offset_step = 16
    monkeypatch.setattr("baroque_transcribe.engrave.interpret_quantization", lambda *args, **kwargs: interpretation)

    plan = F.plan_fragment(project, 2, 2)
    carried = by_pitch(plan)[72]
    assert carried.carried_in and not carried.printed
    assert (carried.onset_step, carried.offset_step) == (0, 1)
    assert plan.unprinted == 1

    folder = F.extract_fragment(project, 2, 2, title="Trimmed overlap", root=tmp_path / "fragments")
    # Not on the page, as it is not on the parent's page in this measure.
    assert 72 not in [pitch for _offset, pitch, _length in score_measures(folder / SCORE_XML)[(1, 1)]]
    assert [pitch for _offset, pitch, _length in score_measures(folder / SCORE_XML)[(1, 1)]] == [74, 76, 77]
    # Still in the notes and in both MIDI files, where the parent's grid has it.
    assert 72 in [note.pitch for note in read_transcription(folder / NOTES_JSON).notes]
    treble = midi_notes(folder / QUANTIZED_MID)[0]
    assert (72, 0.0, STEP) in [(pitch, round(start, 3), round(end, 3)) for pitch, start, end, _velocity in treble]
    assert 72 in [pitch for pitch, *_rest in midi_notes(folder / FRAGMENT_PERFORMANCE_MID)[0]]
    stored = {note["pitch"]: note for note in F.read_grid(folder)["notes"]}
    assert stored[72]["printed"] is False and stored[74]["printed"] is True
    assert F.read_manifest(folder)["boundary"]["unprinted_overlaps"] == 1


def test_boundary_ties_are_removed_from_the_fragment_score(tmp_path: Path):
    project = make_project(tmp_path, held_bass_piece(7.0))
    folder = F.extract_fragment(project, 2, 2, title="Tied both ways", root=tmp_path / "fragments")
    # The parent ties pitch 48 into and out of measure 2. The fragment begins
    # and ends it at its own barlines.
    assert (0, 48, "continue") in [(int(o), p, k) for o, p, k in score_ties(project / SCORE_XML)[(2, 2)]]
    assert score_ties(folder / SCORE_XML)[(2, 1)] == []
    assert (0, 48, 4) in [(int(o), p, int(length)) for o, p, length in score_measures(folder / SCORE_XML)[(2, 1)]]


def test_both_staves_print_every_selected_measure(tmp_path: Path):
    notes = [n(48, 0.0, 1.0), n(72, 0.0, 0.5), n(74, 2.0, 2.5), n(76, 4.0, 4.5), n(77, 5.0, 5.5)]
    project = make_project(tmp_path, notes)
    # A whole transcription still ends each staff at its own last note.
    assert measure_count(project / SCORE_XML) == [3, 1]
    folder = F.extract_fragment(project, 2, 3, title="Silent bass", root=tmp_path / "fragments")
    assert measure_count(folder / SCORE_XML) == [2, 2]
    assert total_measures(folder / SCORE_XML) == 2


def test_readable_fragment_never_runs_past_its_last_barline(tmp_path: Path):
    # In this excerpt the readable interpretation overlaps two events in one
    # voice of measure 6, so that bar is written longer than four quarters.
    # In the parent the surplus is tied into measure 7.
    project = fixture_project(tmp_path, grid="follow", style="readable")
    for start, end in [(4, 6), (6, 6), (5, 8)]:
        folder = F.extract_fragment(project, start, end, title=f"m{start}-{end}", root=tmp_path / "fragments")
        assert total_measures(folder / SCORE_XML) == end - start + 1


def test_end_step_only_extends_a_score(tmp_path: Path):
    notes = [n(48, 0.0, 1.0), n(72, 0.0, 0.5), n(74, 2.0, 2.5), n(76, 4.0, 4.5)]
    quantization = quantize_notes(notes, bpm=120.0, numerator=4, denominator=4, subdivision=16)
    default = write_musicxml(tmp_path / "default.musicxml", quantization, key=("C", "major"))
    assert measure_count(default) == [3, 1]
    short = write_musicxml(tmp_path / "short.musicxml", quantization, key=("C", "major"), end_step=16)
    assert measure_count(short) == [3, 1]  # never shortens
    assert score_notes(short) == score_notes(default)
    padded = write_musicxml(tmp_path / "padded.musicxml", quantization, key=("C", "major"), end_step=64)
    assert measure_count(padded) == [4, 4]
    assert total_measures(padded) == 4
    for staff, number in score_notes(default):
        assert score_notes(padded)[(staff, number)] == score_notes(default)[(staff, number)]


@pytest.mark.parametrize(("grid", "style"), [("fixed", "literal"), ("follow", "readable")])
def test_whole_transcription_score_is_unchanged_by_the_end_step_option(tmp_path: Path, grid: str, style: str):
    """The option is off for every existing caller. Off must mean exactly the old score."""

    project = fixture_project(tmp_path, grid=grid, style=style, build=False)
    quantization = parent_quantization(project)
    pedals = read_transcription(project / NOTES_JSON).pedals
    written = write_musicxml(tmp_path / "now.musicxml", quantization, style=style, key=("D", "major"), pedals=pedals)
    again = write_musicxml(
        tmp_path / "explicit.musicxml", quantization, style=style, key=("D", "major"), pedals=pedals, end_step=None
    )
    # music21 gives each export a random part id; everything else is byte-identical.
    strip = lambda path: re.sub(r'id="P[0-9a-f]{32}"', 'id="P"', path.read_text(encoding="utf-8"))  # noqa: E731
    assert strip(written) == strip(again)


def test_fragment_does_not_depend_on_where_the_passage_sits(tmp_path: Path):
    passage = [
        n(48, 0.0, 2.0),
        n(60, 0.0, 0.5),
        n(64, 0.5, 1.0),
        n(67, 1.0, 1.5),
        n(72, 1.5, 2.5),
        n(55, 2.0, 4.0),
        n(71, 2.5, 3.0),
        n(67, 3.0, 4.0),
    ]
    closing = [n(60, 4.0, 5.0), n(48, 4.0, 5.0)]
    shift = 3 * BAR
    opening = [n(60, 0.0, 1.0), n(67, 2.0, 3.0), n(64, 4.0, 5.0), n(48, 0.0, 1.0)]
    near = make_project(tmp_path, passage + closing, name="near")
    far = make_project(
        tmp_path,
        opening + [n(x.pitch, x.onset + shift, x.offset + shift, x.velocity) for x in passage + closing],
        name="far",
        excerpt=412.5,
        sync=-0.75,
    )
    one = F.extract_fragment(near, 1, 2, title="Passage", root=tmp_path / "fragments")
    two = F.extract_fragment(far, 4, 5, title="Passage", root=tmp_path / "fragments")

    assert (one / QUANTIZED_MID).read_bytes() == (two / QUANTIZED_MID).read_bytes()
    assert (one / FRAGMENT_PERFORMANCE_MID).read_bytes() == (two / FRAGMENT_PERFORMANCE_MID).read_bytes()
    assert score_measures(one / SCORE_XML) == score_measures(two / SCORE_XML)
    assert F.read_grid(one)["notes"] == F.read_grid(two)["notes"]
    first, second = F.read_manifest(one), F.read_manifest(two)
    assert first["time"]["duration_seconds"] == second["time"]["duration_seconds"]
    assert second["time"]["working_audio"]["start_seconds"] == pytest.approx(shift)
    assert second["time"]["source_video"]["start_seconds"] == pytest.approx(412.5 + shift)


# --------------------------------------------------------------------------
# Clocks
# --------------------------------------------------------------------------


def test_clocks_with_excerpt_fragment_start_and_sync_offset_all_nonzero(tmp_path: Path):
    excerpt, sigma = 30.0, 0.4
    project = make_project(tmp_path, barline_piece(), excerpt=excerpt, sync=sigma, analyze=True)
    parent = parent_quantization(project)
    parent_anchors = measure_anchors(parent)
    folder = F.extract_fragment(project, 2, 3, title="Clocks", root=tmp_path / "fragments")
    manifest = F.read_manifest(folder)
    t0 = 1.95

    # Measure 3 is the parent's last, so the window runs to its written end barline at 6.0 s.
    end = 6.0
    assert manifest["time"]["working_audio"] == {"start_seconds": pytest.approx(t0), "end_seconds": pytest.approx(end)}
    # The stored video interval is the passage itself. It does not contain the sync offset.
    assert manifest["time"]["source_video"]["start_seconds"] == pytest.approx(excerpt + t0)
    assert manifest["time"]["source_video"]["end_seconds"] == pytest.approx(excerpt + end)
    assert manifest["time"]["source_video"]["excerpt_start_seconds"] == excerpt
    assert manifest["time"]["duration_seconds"] == pytest.approx(end - t0)
    assert manifest["sync"] == {"offset_seconds": sigma, "parent_offset_at_extraction": sigma}

    # F -> W -> V agrees with the parent's own mapping for the same instant.
    for seconds in (0.0, 0.7, manifest["time"]["duration_seconds"]):
        working = F.fragment_to_working(seconds, manifest)
        assert working == pytest.approx(seconds + t0)
        assert F.fragment_to_video(seconds, manifest) == pytest.approx(
            video_time(working, excerpt_start=excerpt, sync_offset=sigma)
        )
        assert F.video_to_fragment(F.fragment_to_video(seconds, manifest), manifest) == pytest.approx(seconds)
        assert F.working_to_fragment(working, manifest) == pytest.approx(seconds)

    # Every note sounds at the video time the parent would play it.
    parent_onsets = {(note.pitch, round(note.onset, 5)) for note in read_transcription(project / NOTES_JSON).notes}
    grid = F.read_grid(folder)
    for note in grid["notes"]:
        assert (note["pitch"], round(F.fragment_to_working(note["onset_seconds"], manifest), 5)) in parent_onsets

    # Measure highlighting: fragment measure k begins where parent measure k+1 begins.
    assert [anchor["measure"] for anchor in grid["anchors"]] == [1, 2]
    assert [anchor["parent_measure"] for anchor in grid["anchors"]] == [2, 3]
    for anchor in grid["anchors"]:
        parent_anchor = parent_anchors[anchor["parent_measure"] - 1]
        assert F.fragment_to_working(anchor["seconds"], manifest) == pytest.approx(parent_anchor.time)
        assert anchor["seconds"] >= 0.0

    # Harmony is positioned on the grid only.
    harmony = json.loads((folder / FRAGMENT_HARMONY_JSON).read_text(encoding="utf-8"))
    assert harmony["chords"]
    for chord in harmony["chords"]:
        assert not [key for key in chord if "second" in key or "time" in key]
        assert 0 <= chord["start_step"] < chord["end_step"] <= 32
        assert chord["measure"] == chord["start_step"] // 16 + 1


def test_sync_adjustment_moves_the_player_and_nothing_else(tmp_path: Path):
    excerpt, sigma = 30.0, 0.4
    project = make_project(tmp_path, barline_piece(), excerpt=excerpt, sync=sigma, analyze=True)
    parent_before = tree_digest(project)
    folder = F.extract_fragment(
        project,
        2,
        3,
        title="Sync",
        root=tmp_path / "fragments",
        now=datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc),
    )
    before = F.read_manifest(folder)
    files_before = tree_digest(folder)
    window_before = F.video_window(before)
    assert window_before[0] == pytest.approx(excerpt + 1.95 + sigma)

    stored = F.set_fragment_sync(folder, 1.25, now=datetime(2026, 11, 10, 9, 5, tzinfo=timezone.utc))
    assert stored == 1.25
    after = F.read_manifest(folder)

    assert after["sync"] == {"offset_seconds": 1.25, "parent_offset_at_extraction": sigma}
    assert after["time"] == before["time"]  # the source interval and the duration are as written
    assert F.duration_seconds(after) == F.duration_seconds(before)
    window_after = F.video_window(after)
    assert window_after[0] - window_before[0] == pytest.approx(1.25 - sigma)
    assert window_after[1] - window_after[0] == pytest.approx(window_before[1] - window_before[0])
    assert window_after[1] - window_after[0] == pytest.approx(F.duration_seconds(after))
    assert after["modified_at"] == "2026-11-10T09:05:00Z" and after["created_at"] == "2026-11-10T09:00:00Z"

    # Only the manifest changed, and in it only the offset and the modification time.
    files_after = tree_digest(folder)
    assert {name for name in files_after if files_after[name] != files_before[name]} == {FRAGMENT_JSON}
    changed = {key for key in after if after[key] != before[key]}
    assert changed == {"sync", "modified_at"}
    # The parent's own offset is a separate setting.
    assert tree_digest(project) == parent_before

    assert F.set_fragment_sync(folder, 500.0) == F.SYNC_OFFSET_LIMIT
    assert F.set_fragment_sync(folder, -500.0) == -F.SYNC_OFFSET_LIMIT
    assert F.read_manifest(folder)["time"] == before["time"]


# --------------------------------------------------------------------------
# Key and harmony
# --------------------------------------------------------------------------


def key_signature_sharps(path: Path) -> int:
    from music21 import converter

    return int(converter.parse(str(path)).recurse().getElementsByClass("KeySignature").first().sharps)


def test_harmony_is_a_slice_of_the_parent_analysis(tmp_path: Path):
    project = fixture_project(tmp_path, grid="fixed", style="literal", analyze=True)
    analysis = load_analysis(project / ANNOTATIONS_JSON)
    parent = parent_quantization(project)
    first, _count = printed_measure_span(parent)
    per = parent.steps_per_measure
    s0, s1 = first + 2 * per, first + 5 * per
    expected = [chord for chord in analysis.chords if chord.start_step < s1 and chord.end_step > s0]
    assert expected

    folder = F.extract_fragment(project, 3, 5, title="Harmony", root=tmp_path / "fragments")
    manifest = F.read_manifest(folder)
    harmony = F.read_harmony(folder)
    assert harmony is not None
    assert [chord.roman for chord in harmony.chords] == [chord.roman for chord in expected]
    assert [chord.symbol for chord in harmony.chords] == [chord.symbol for chord in expected]
    assert [chord.start_step for chord in harmony.chords] == [max(chord.start_step, s0) - s0 for chord in expected]
    assert [chord.end_step for chord in harmony.chords] == [min(chord.end_step, s1) - s0 for chord in expected]
    assert all(1 <= chord.measure <= 3 for chord in harmony.chords)
    assert manifest["harmony"] == {
        "roman": [chord.roman for chord in expected],
        "symbols": [chord.symbol for chord in expected],
    }
    assert manifest["artifacts"]["harmony"] == FRAGMENT_HARMONY_JSON

    # The slice lines up with the fragment's own grid, so a page can draw it.
    stored_grid = F.grid_quantization(F.read_grid(folder))
    assert analysis_is_stale(harmony, stored_grid) is False
    provenance = json.loads((folder / FRAGMENT_HARMONY_JSON).read_text(encoding="utf-8"))["provenance"]
    assert provenance["parent_measure_start"] == 3 and provenance["parent_step_start"] == s0
    assert provenance["parent_notes_fingerprint"] == notes_fingerprint(parent)
    assert manifest["parent"]["notes_fingerprint"] == notes_fingerprint(parent)
    assert len(manifest["parent"]["analysis_sha256"]) == 64


def test_analyzed_key_and_written_key_signature_are_recorded_separately(tmp_path: Path):
    project = fixture_project(tmp_path, grid="fixed", style="literal", analyze=True)
    analysis = load_analysis(project / ANNOTATIONS_JSON)
    folder = F.extract_fragment(project, 3, 5, title="Key", root=tmp_path / "fragments")
    music = F.read_manifest(folder)["music"]
    assert music["analyzed_key"] == {
        "tonic": analysis.key_tonic,
        "mode": analysis.key_mode,
        "source": analysis.key_source,
        "confidence": analysis.key_confidence,
    }
    # The signature printed in the fragment is the parent's, whatever the passage tonicizes.
    assert music["written_key_signature"] == {"tonic": analysis.key_tonic, "mode": analysis.key_mode}
    assert key_signature_sharps(folder / SCORE_XML) == key_signature_sharps(project / SCORE_XML)


def test_fragment_without_parent_analysis_has_no_harmony(tmp_path: Path):
    project = fixture_project(tmp_path, grid="fixed", style="literal")
    plan = F.plan_fragment(project, 3, 5)
    assert plan.harmony is None and plan.key_source == "estimated-at-extraction"
    assert any("no harmonic analysis" in warning for warning in plan.warnings)
    folder = F.extract_fragment(project, 3, 5, title="No harmony", root=tmp_path / "fragments")
    manifest = F.read_manifest(folder)
    assert manifest["artifacts"]["harmony"] is None
    assert manifest["harmony"] == {"roman": [], "symbols": []}
    assert not (folder / FRAGMENT_HARMONY_JSON).exists()
    assert F.read_harmony(folder) is None
    assert key_signature_sharps(folder / SCORE_XML) == key_signature_sharps(project / SCORE_XML)


def test_stale_parent_analysis_is_not_copied(tmp_path: Path):
    project = fixture_project(tmp_path, grid="fixed", style="literal", analyze=True)
    notes_path = project / NOTES_JSON
    payload = json.loads(notes_path.read_text(encoding="utf-8"))
    payload["notes"][0]["velocity"] = 1 + payload["notes"][0]["velocity"] % 100
    notes_path.write_text(json.dumps(payload), encoding="utf-8")

    plan = F.plan_fragment(project, 3, 5)
    assert plan.parent_analysis_stale and plan.harmony is None
    assert any("no longer line up" in warning for warning in plan.warnings)
    folder = F.extract_fragment(project, 3, 5, title="Stale", root=tmp_path / "fragments")
    manifest = F.read_manifest(folder)
    assert manifest["artifacts"]["harmony"] is None
    assert not (folder / FRAGMENT_HARMONY_JSON).exists()
    assert manifest["warnings"]
    # The key the parent chose is still recorded.
    assert manifest["music"]["analyzed_key"]["tonic"] == load_analysis(project / ANNOTATIONS_JSON).key_tonic


# --------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------


def test_fragment_folder_is_complete_and_self_contained(tmp_path: Path):
    project = fixture_project(tmp_path, grid="follow", style="readable", analyze=True, youtube=None)
    root = tmp_path / "library" / "fragments"
    folder = F.extract_fragment(
        project,
        3,
        5,
        title="  Descending   fifths ",
        description="Sequence in the left hand.",
        practice_notes="Slow, hands separate.",
        tags=["Sequence", "sequence ", " circle of  fifths", ""],
        composer="Improvised",
        performer="C.",
        root=root,
    )
    manifest = F.read_manifest(folder)
    assert folder.parent == root and folder.name == manifest["id"]
    assert sorted(path.name for path in folder.iterdir()) == sorted(
        [
            FRAGMENT_JSON,
            NOTES_JSON,
            FRAGMENT_GRID_JSON,
            FRAGMENT_PERFORMANCE_MID,
            QUANTIZED_MID,
            SCORE_XML,
            FRAGMENT_HARMONY_JSON,
        ]
    )
    assert manifest["schema_version"] == F.FRAGMENT_SCHEMA_VERSION
    assert manifest["title"] == "Descending fifths"
    assert manifest["tags"] == ["Sequence", "circle of fifths"]
    assert manifest["created_at"] == manifest["modified_at"]
    assert manifest["parent"]["project"] == project.name
    assert (manifest["parent"]["measure_start"], manifest["parent"]["measure_end"]) == (3, 5)
    assert manifest["music"]["measures"] == 3
    assert manifest["music"]["notation_style"] == "readable"
    for name in manifest["artifacts"].values():
        assert (folder / name).is_file()

    # A local recording is known by its file name only. No folder of this machine is stored.
    assert manifest["source"]["kind"] == "file"
    assert manifest["source"]["name"] == "private take.wav"
    for path in folder.iterdir():
        if path.suffix in {".json", ".musicxml"}:
            text = path.read_text(encoding="utf-8")
            assert "someone" not in text
            assert str(tmp_path) not in text and tmp_path.as_posix() not in text
            assert json.dumps(str(tmp_path))[1:-1] not in text

    # The folder can be read with the parent gone, from anywhere.
    moved = tmp_path / "elsewhere" / "renamed-folder"
    moved.parent.mkdir()
    shutil.copytree(folder, moved)
    shutil.rmtree(project)
    assert F.read_manifest(moved) == manifest
    assert len(read_transcription(moved / NOTES_JSON).notes) == manifest["boundary"]["notes"]
    assert F.find_fragment(manifest["id"], moved.parent) == moved
    assert F.find_fragment(manifest["id"], root) == folder
    assert F.find_fragment("no-such-id", root) is None
    assert F.fragment_folders(root) == [folder]


def test_stored_grid_reads_back_as_the_planned_grid(tmp_path: Path):
    project = fixture_project(tmp_path, grid="follow", style="readable")
    plan = F.plan_fragment(project, 4, 6)
    folder = F.extract_fragment(project, 4, 6, title="Grid", root=tmp_path / "fragments")
    stored = F.grid_quantization(F.read_grid(folder))
    assert stored.grid == plan.quantization.grid
    assert stored.lead_steps == 0 and stored.steps_per_measure == plan.quantization.steps_per_measure
    assert notes_fingerprint(stored) == notes_fingerprint(plan.quantization)
    # ``index`` is the note's position in the fragment's own notes.json.
    notes = read_transcription(folder / NOTES_JSON).notes
    for item in F.grid_notes(F.read_grid(folder)):
        assert notes[item.index].pitch == item.pitch
        assert notes[item.index].onset == pytest.approx(item.onset_seconds)
        assert notes[item.index].offset == pytest.approx(item.offset_seconds)


def test_failed_extraction_leaves_nothing_behind(monkeypatch, tmp_path: Path):
    project = make_project(tmp_path, barline_piece())
    root = tmp_path / "fragments"

    def broken(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("baroque_transcribe.midi_export.write_quantized_midi", broken)
    with pytest.raises(OSError):
        F.extract_fragment(project, 1, 2, title="Never", root=root)
    assert list(root.iterdir()) == []
    assert F.fragment_folders(root) == []


def test_staged_fragment_is_read_back_before_publication(monkeypatch, tmp_path: Path):
    project = make_project(tmp_path, barline_piece())
    root = tmp_path / "fragments"

    def invalid_score(path, *args, **kwargs):
        path.write_text("<not-closed>", encoding="utf-8")
        return path

    monkeypatch.setattr(F, "write_fragment_score", invalid_score)
    with pytest.raises(F.FragmentError, match="invalid MusicXML"):
        F.extract_fragment(project, 1, 2, title="Never", root=root)
    assert list(root.iterdir()) == []


def test_selection_is_validated(tmp_path: Path):
    project = make_project(tmp_path, barline_piece() + [n(60, 8.0, 8.5)])
    root = tmp_path / "fragments"
    count = F.plan_fragment(project, 1, 1).parent_measures
    assert count == 5
    for start, end in [(0, 1), (2, 1), (1, count + 1), (count + 1, count + 1)]:
        with pytest.raises(F.FragmentError, match="between 1 and 5"):
            F.plan_fragment(project, start, end)
    with pytest.raises(F.FragmentError, match="no notes"):
        F.plan_fragment(project, 4, 4)
    with pytest.raises(F.FragmentError, match="title"):
        F.extract_fragment(project, 1, 1, title="   ", root=root)
    with pytest.raises(F.FragmentError, match="Transcribe"):
        F.plan_fragment(tmp_path / "missing", 1, 1)
    assert not root.exists()


def test_tests_cannot_write_into_the_repository():
    with pytest.raises(F.FragmentError):
        F.fragments_root()
    with pytest.raises(F.FragmentError):
        F.local_root()


def test_manifest_validation_names_the_problem(tmp_path: Path):
    project = make_project(tmp_path, barline_piece())
    folder = F.extract_fragment(project, 1, 2, title="Valid", root=tmp_path / "fragments")
    good = F.read_manifest(folder)

    def changed(**updates):
        payload = json.loads(json.dumps(good))
        payload.update(updates)
        return payload

    with pytest.raises(F.FragmentError, match="versions 1 and 2"):
        F.validate_manifest(changed(schema_version=3))
    legacy = changed(schema_version=1)
    for key in (
        "lineage_id",
        "base_event_id",
        "base_snapshot_sha256",
        "snapshot_sha256",
        "revision_state",
    ):
        legacy["parent"].pop(key, None)
    assert F.validate_manifest(legacy)["schema_version"] == 1
    with pytest.raises(F.FragmentError, match="no fragment id"):
        F.validate_manifest(changed(id=""))
    with pytest.raises(F.FragmentError, match="no title"):
        F.validate_manifest(changed(title=None))
    with pytest.raises(F.FragmentError, match="tags"):
        F.validate_manifest(changed(tags="one, two"))
    with pytest.raises(F.FragmentError, match="plain file name"):
        F.validate_manifest(changed(artifacts={**good["artifacts"], "musicxml": "../score.musicxml"}))
    with pytest.raises(F.FragmentError, match="duration"):
        F.validate_manifest(changed(time={**good["time"], "duration_seconds": 0}))
    with pytest.raises(F.FragmentError, match="not a JSON object"):
        F.validate_manifest([good])

    (folder / FRAGMENT_JSON).write_text("{ not json", encoding="utf-8")
    with pytest.raises(F.FragmentError, match="not valid JSON"):
        F.read_manifest(folder)


def test_fragment_v2_records_parent_snapshot_and_reports_local_staleness(tmp_path: Path):
    project = make_project(tmp_path, barline_piece())
    folder = F.extract_fragment(
        project,
        1,
        2,
        title="Provenance",
        root=tmp_path / "fragments",
    )
    manifest = F.read_manifest(folder)
    assert manifest["schema_version"] == 2
    assert manifest["parent"]["snapshot_sha256"]
    assert (
        F.fragment_parent_status(manifest, projects_root=project.parent)
        == "current"
    )

    notes_path = project / NOTES_JSON
    notes = json.loads(notes_path.read_text(encoding="utf-8"))
    notes["notes"][0]["pitch"] += 1
    notes_path.write_text(json.dumps(notes), encoding="utf-8")
    assert (
        F.fragment_parent_status(manifest, projects_root=project.parent)
        == "parent_changed"
    )


def test_fragment_reports_published_parent_change_and_unavailable_parent(
    tmp_path: Path,
):
    project = make_project(tmp_path, barline_piece())
    folder = F.extract_fragment(
        project,
        1,
        2,
        title="Published provenance",
        root=tmp_path / "fragments",
    )
    manifest = F.read_manifest(folder)
    manifest["parent"].update(
        {
            "lineage_id": "lineage",
            "base_event_id": "base",
            "event_id": "base",
        }
    )
    store = tmp_path / "store"
    for ident, parent in (("base", None), ("new-head", "base")):
        event_folder = store / "events" / ident
        event_folder.mkdir(parents=True)
        (event_folder / "event.json").write_text(
            json.dumps(
                {
                    "version": 2,
                    "id": ident,
                    "recorded_at": f"2026-01-0{1 if ident == 'base' else 2}T00:00:00Z",
                    "action": "prepare",
                    "status": "ok",
                    "project": project.name,
                    "source": {"kind": "file", "name": "source.wav"},
                    "revision": {
                        "lineage_id": "lineage",
                        "parent_event_id": parent,
                        "complete_snapshot": False,
                        "snapshot": {"sha256": ident, "artifacts": {}},
                    },
                    "project_state": {},
                }
            ),
            encoding="utf-8",
        )
    assert (
        F.fragment_parent_status(
            manifest,
            projects_root=project.parent,
            store=store,
        )
        == "published_parent_changed"
    )
    assert (
        F.fragment_parent_status(
            manifest,
            projects_root=tmp_path / "missing-projects",
            store=store,
        )
        == "parent_unavailable"
    )


def test_fragment_metadata_atomic_failure_preserves_prior_manifest(
    tmp_path: Path, monkeypatch
):
    project = make_project(tmp_path, barline_piece())
    folder = F.extract_fragment(
        project,
        1,
        2,
        title="Before",
        root=tmp_path / "fragments",
    )
    manifest_path = folder / FRAGMENT_JSON
    before = manifest_path.read_bytes()

    def locked(source, destination):
        raise PermissionError("locked")

    monkeypatch.setattr(F.os, "replace", locked)
    monkeypatch.setattr("time.sleep", lambda _seconds: None)
    with pytest.raises(PermissionError, match="locked"):
        F.update_fragment_metadata(folder, title="After")
    assert manifest_path.read_bytes() == before
    assert not list(folder.glob("fragment.json.*.tmp"))


def test_metadata_edit_changes_only_the_named_fields(tmp_path: Path):
    project = make_project(tmp_path, barline_piece(), analyze=True)
    folder = F.extract_fragment(
        project,
        1,
        2,
        title="Before",
        tags=["one"],
        root=tmp_path / "fragments",
        now=datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc),
    )
    before = F.read_manifest(folder)
    files_before = tree_digest(folder)
    after = F.update_fragment_metadata(
        folder,
        title="After",
        practice_notes="Left hand alone first.",
        tags="cadence, Cadence, two",
        now=datetime(2026, 11, 11, 9, 0, tzinfo=timezone.utc),
    )
    assert after == F.read_manifest(folder)
    assert (after["title"], after["practice_notes"], after["tags"]) == (
        "After",
        "Left hand alone first.",
        ["cadence", "two"],
    )
    assert {key for key in after if after[key] != before[key]} == {"title", "practice_notes", "tags", "modified_at"}
    assert after["created_at"] == "2026-11-10T09:00:00Z" and after["modified_at"] == "2026-11-11T09:00:00Z"
    files_after = tree_digest(folder)
    assert {name for name in files_after if files_after[name] != files_before[name]} == {FRAGMENT_JSON}
    with pytest.raises(F.FragmentError, match="title"):
        F.update_fragment_metadata(folder, title="  ")
    assert F.read_manifest(folder)["title"] == "After"


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def test_cli_extract(tmp_path: Path, capsys):
    project = make_project(tmp_path, held_bass_piece(3.0))
    before = tree_digest(project)
    library = tmp_path / "fragments"

    assert main(["fragments", "extract", str(project), "--measures", "2", "--dry-run", "--library", str(library)]) == 0
    out = capsys.readouterr().out
    assert "Measures 2 to 2 of 4" in out and "Nothing was saved." in out
    assert not library.exists()

    code = main(
        [
            "fragments",
            "extract",
            str(project),
            "--measures",
            "2-3",
            "--title",
            "From the command line",
            "--tag",
            "cli",
            "--tag",
            "bass",
            "--library",
            str(library),
        ]
    )
    assert code == 0
    folders = F.fragment_folders(library)
    assert len(folders) == 1
    manifest = F.read_manifest(folders[0])
    assert manifest["title"] == "From the command line" and manifest["tags"] == ["cli", "bass"]
    assert (manifest["parent"]["measure_start"], manifest["parent"]["measure_end"]) == (2, 3)
    assert f"Saved fragment {manifest['id']}" in capsys.readouterr().out

    assert main(["fragments", "extract", str(project), "--measures", "9-12", "--title", "x", "--library", str(library)]) == 1
    assert "between 1 and 4" in capsys.readouterr().err
    assert main(["fragments", "extract", str(project), "--measures", "two", "--title", "x", "--library", str(library)]) == 1
    assert main(["fragments", "extract", str(project), "--measures", "2", "--library", str(library)]) == 1
    assert "--title" in capsys.readouterr().err
    assert len(F.fragment_folders(library)) == 1
    assert tree_digest(project) == before


def test_measure_ranges_are_read_the_same_way_from_the_cli_and_the_form():
    assert F.parse_measures("12-15") == (12, 15)
    assert F.parse_measures("12") == (12, 12)
    assert F.parse_measures("15–12") == (12, 15)
    with pytest.raises(F.FragmentError, match="look like"):
        F.parse_measures("two")
    with pytest.raises(F.FragmentError, match="start at 1"):
        F.parse_measures("0-2")


# --------------------------------------------------------------------------
# Git
# --------------------------------------------------------------------------


def _git_ignored(repo: Path, path: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "--no-index", path],
        cwd=repo,
        capture_output=True,
        timeout=30,
    )
    if result.returncode not in (0, 1):
        pytest.skip(f"git check-ignore failed: {result.stderr.decode(errors='replace')}")
    return result.returncode == 0


def test_gitignore_tracks_fragments_and_ignores_the_local_index():
    repo = Path(__file__).resolve().parents[1]
    if shutil.which("git") is None or not (repo / ".git").exists() or not (repo / ".gitignore").is_file():
        pytest.skip("not a git checkout")
    fragment = "data/fragments/0b1c2d3e-0000-4000-8000-000000000000"
    for name in (
        FRAGMENT_JSON,
        NOTES_JSON,
        FRAGMENT_GRID_JSON,
        FRAGMENT_HARMONY_JSON,
        FRAGMENT_PERFORMANCE_MID,
        QUANTIZED_MID,
        SCORE_XML,
    ):
        assert not _git_ignored(repo, f"{fragment}/{name}"), f"{name} would not be committed"
    assert _git_ignored(repo, ".local/library.sqlite")
    assert _git_ignored(repo, ".local/fragment_pages/abc/playback.html")
    assert _git_ignored(repo, ".local/fragment_audio/abc-performance-deadbeef.wav")
    assert _git_ignored(repo, "data/fragments/.tmp-0b1c2d3e/fragment.json")
    assert _git_ignored(repo, "data/fragments/.tmp-0b1c2d3e/score.musicxml")
    # The rules that keep a whole transcription out of Git are unchanged.
    assert _git_ignored(repo, "transcriptions/some-run/raw.mid")
    assert _git_ignored(repo, "transcriptions/some-run/score.musicxml")
    assert _git_ignored(repo, "stray.mid")


# --------------------------------------------------------------------------
# Library preview audio
# --------------------------------------------------------------------------


def test_preview_wav_is_cached_outside_the_fragment(tmp_path: Path):
    project = make_project(tmp_path, held_bass_piece(3.0))
    before_project = tree_digest(project)
    folder = F.extract_fragment(project, 2, 2, title="Preview", root=tmp_path / "fragments")
    before = tree_digest(folder)
    local = tmp_path / ".local"

    first = F.ensure_fragment_preview_wav(folder, mode="performance", sound="rhodes", local=local)
    second = F.ensure_fragment_preview_wav(folder, mode="performance", sound="rhodes", local=local)
    quantized = F.ensure_fragment_preview_wav(folder, mode="quantized", sound="rhodes", local=local)
    other_sound = F.ensure_fragment_preview_wav(folder, mode="performance", sound="wurli", local=local)

    assert first == second and first.is_file()
    assert first.parent == local / "fragment_audio"
    assert first.name.startswith(f"{F.read_manifest(folder)['id']}-performance-rhodes-")
    assert quantized != first and quantized.is_file()
    assert other_sound != first and other_sound.is_file()
    assert tree_digest(folder) == before
    assert tree_digest(project) == before_project

    import soundfile as sf

    audio, rate = sf.read(first)
    assert rate == F.PREVIEW_SAMPLE_RATE
    assert len(audio) >= int(0.5 * F.PREVIEW_SAMPLE_RATE)
    # A later note edit invalidates the cache.
    notes = json.loads((folder / NOTES_JSON).read_text(encoding="utf-8"))
    notes["notes"][0]["velocity"] = 1 + notes["notes"][0]["velocity"] % 100
    (folder / NOTES_JSON).write_text(json.dumps(notes), encoding="utf-8")
    rebuilt = F.ensure_fragment_preview_wav(folder, mode="performance", sound="rhodes", local=local)
    assert rebuilt != first
    assert not first.exists()
    assert list(local.joinpath("fragment_audio").glob("*-performance-rhodes-*.wav")) == [rebuilt]


def test_preview_notes_distinguish_performance_from_quantized(tmp_path: Path):
    project = make_project(tmp_path, barline_piece())
    folder = F.extract_fragment(project, 2, 2, title="Timing", root=tmp_path / "fragments")
    played, played_duration = F.preview_notes(folder, "performance")
    grid, grid_duration = F.preview_notes(folder, "quantized")
    assert played and grid
    assert played_duration > 0 and grid_duration > 0
    # Pitch 67 is played 50 ms after its grid step in this fixture.
    as_played = next(note for note in played if note.pitch == 67)
    on_grid = next(note for note in grid if note.pitch == 67)
    assert as_played.onset == pytest.approx(0.55)
    assert on_grid.onset == pytest.approx(0.5)
    with pytest.raises(F.FragmentError, match="Preview mode"):
        F.preview_notes(folder, "piano")
