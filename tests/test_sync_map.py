"""Timing map from original note onsets to printed measures."""

import pytest

from baroque_transcribe.quantize import GridNote, Quantization, quantize_notes
from baroque_transcribe.schema import NoteEvent
from baroque_transcribe.sync_map import (
    measure_anchors,
    measure_at_working_time,
    measure_seek_video,
    note_anchors,
    video_time,
    working_time,
)


def _at_120(*notes: NoteEvent) -> Quantization:
    return quantize_notes(list(notes), bpm=120, numerator=4, denominator=4, subdivision=16)


def _hand_built(grid: list[GridNote]) -> Quantization:
    quantum = 0.125
    per_measure = 16
    earliest = min((item.onset_step for item in grid), default=0)
    first_bar = (min(earliest, 0) // per_measure) * per_measure
    return Quantization(
        bpm=120,
        numerator=4,
        denominator=4,
        subdivision=16,
        downbeat=0.0,
        quantum=quantum,
        steps_per_measure=per_measure,
        lead_steps=-first_bar,
        notes=[],
        grid=grid,
    )


def test_note_anchors_carry_pitch_onset_and_measure():
    quantization = _at_120(
        NoteEvent(60, 0.0, 0.5, 80),
        NoteEvent(64, 0.5, 1.0, 70),
        NoteEvent(48, 0.0, 1.0, 60),
    )
    anchors = note_anchors(quantization)
    assert [(item.pitch, round(item.onset, 3), item.measure) for item in anchors] == [
        (48, 0.0, 1),
        (60, 0.0, 1),
        (64, 0.5, 1),
    ]
    assert all(item.offset > item.onset for item in anchors)


def test_source_index_survives_the_grid_sort():
    notes = [
        NoteEvent(pitch=60, onset=1.0, offset=1.2, velocity=70),
        NoteEvent(pitch=64, onset=0.0, offset=0.2, velocity=80),
    ]
    result = _at_120(*notes)
    assert result.grid[0].source_index == 1
    assert result.grid[0].source_onset == pytest.approx(0.0)
    assert result.grid[0].source_offset == pytest.approx(0.2)
    assert result.grid[0].onset_step == 0
    assert result.grid[1].source_index == 0
    assert result.grid[1].source_onset == pytest.approx(1.0)
    assert result.notes[0].onset == pytest.approx(0.0)


def test_excerpt_and_offset_round_trip_into_the_click_time():
    notes = [NoteEvent(pitch=60, onset=index * 2.0, offset=index * 2.0 + 0.2, velocity=80) for index in range(5)]
    anchors = measure_anchors(_at_120(*notes))
    excerpt = 12.5
    offset = -0.25
    working = anchors[3].time
    video = video_time(working, excerpt_start=excerpt, sync_offset=offset)
    assert working_time(video, excerpt_start=excerpt, sync_offset=offset) == pytest.approx(working)
    assert measure_seek_video(anchors[3], excerpt_start=excerpt, sync_offset=offset) == pytest.approx(max(0.0, video))


def test_lookup_moves_backward():
    notes = [NoteEvent(pitch=60, onset=index * 2.0, offset=index * 2.0 + 0.2, velocity=80) for index in range(5)]
    anchors = measure_anchors(_at_120(*notes))
    assert measure_at_working_time(anchors, 8.1) == 5
    assert measure_at_working_time(anchors, 2.5) == 2


def test_simultaneous_notes_share_the_earlier_onset():
    notes = [
        NoteEvent(pitch=60, onset=0.0, offset=0.4, velocity=80),
        NoteEvent(pitch=64, onset=2.0, offset=2.3, velocity=70),
        NoteEvent(pitch=67, onset=2.04, offset=2.3, velocity=70),
    ]
    anchors = measure_anchors(_at_120(*notes))
    assert len([item for item in anchors if item.measure == 2]) == 1
    assert anchors[1].time == pytest.approx(2.0)
    assert anchors[1].opens_before_attack is False


def test_a_tie_anchors_only_the_measure_where_it_begins():
    notes = [NoteEvent(pitch=60, onset=0.0, offset=3.0, velocity=80)]
    anchors = measure_anchors(_at_120(*notes))
    assert anchors[0].time == pytest.approx(0.0)
    assert anchors[1].measure == 2
    assert anchors[1].opens_before_attack is True
    assert anchors[1].time == pytest.approx(2.0)


def test_opening_rest_is_an_estimate_before_the_first_attack():
    """The anchor is not the attack and not a detected barline.

    At 120 BPM a sixteenth is 0.125 s. The note at 2.5 s snaps to step 20,
    four sixteenths after measure 2 begins, so the estimate backs up by
    4 * 0.125 s. Audio during that written opening belongs to measure 2.
    """

    notes = [
        NoteEvent(pitch=60, onset=0.0, offset=0.4, velocity=80),
        NoteEvent(pitch=64, onset=2.5, offset=2.8, velocity=80),
    ]
    anchors = measure_anchors(_at_120(*notes))
    opening = anchors[1]
    assert opening.opens_before_attack is True
    assert opening.time == pytest.approx(2.0)
    assert opening.time < 2.5
    assert measure_at_working_time(anchors, 2.2) == 2


def test_a_held_note_before_a_later_attack_uses_the_written_opening():
    notes = [
        NoteEvent(pitch=60, onset=0.0, offset=3.0, velocity=70),
        NoteEvent(pitch=67, onset=2.5, offset=2.9, velocity=80),
    ]
    anchors = measure_anchors(_at_120(*notes))
    assert anchors[1].opens_before_attack is True
    assert anchors[1].time == pytest.approx(2.0)
    assert measure_at_working_time(anchors, 2.2) == 2


def test_collision_uses_the_metronome_barline_when_it_is_later():
    grid = [
        GridNote(60, 0, 4, 80, 0, 0.0, 0.2),
        GridNote(62, 16, 20, 80, 1, -1.0, -0.8),
    ]
    anchors = measure_anchors(_hand_built(grid))
    assert anchors[1].collided is True
    assert anchors[1].time == pytest.approx(2.0)
    assert anchors[1].time > anchors[0].time


def test_collision_fallback_keeps_positive_distinct_durations():
    quantum = 0.125
    grid = [
        GridNote(60, 0, 4, 80, 0, 5.0, 5.2),
        GridNote(62, 16, 20, 80, 1, 4.0, 4.2),
        GridNote(64, 32, 36, 80, 2, 4.5, 4.7),
    ]
    anchors = measure_anchors(_hand_built(grid))
    assert anchors[0].collided is False
    assert anchors[0].time == pytest.approx(5.0)
    assert anchors[1].collided is True
    assert anchors[1].time == pytest.approx(5.0 + quantum)
    assert anchors[2].collided is True
    assert anchors[2].time == pytest.approx(5.0 + 2 * quantum)
    times = [item.time for item in anchors]
    assert len({round(item, 6) for item in times}) == len(times)
    assert all(later - earlier > 0 for earlier, later in zip(times, times[1:]))
    assert measure_at_working_time(anchors, 5.0) == 1
    assert measure_at_working_time(anchors, 5.0 + quantum) == 2
    assert measure_at_working_time(anchors, 5.0 + 2 * quantum) == 3
    seek_times = [measure_seek_video(item, excerpt_start=10.0, sync_offset=0.0) for item in anchors]
    assert len(set(seek_times)) == len(seek_times)


def test_downbeat_attacks_keep_their_own_timing_gap():
    notes = [
        NoteEvent(pitch=60, onset=0.0, offset=0.2, velocity=80),
        NoteEvent(pitch=62, onset=2.06, offset=2.2, velocity=80),
        NoteEvent(pitch=64, onset=4.0, offset=4.2, velocity=80),
    ]
    anchors = measure_anchors(_at_120(*notes))
    assert anchors[1].time == pytest.approx(2.06)
    assert anchors[2].time == pytest.approx(4.0)
    assert anchors[1].time - anchors[0].time == pytest.approx(2.06)
    assert anchors[2].time - anchors[1].time == pytest.approx(1.94)
    assert measure_at_working_time(anchors, 2.03) == 1


def test_click_time_is_clamped_at_zero():
    grid = [GridNote(60, 0, 4, 80, 0, 0.2, 0.4)]
    anchors = measure_anchors(_hand_built(grid))
    assert measure_seek_video(anchors[0], excerpt_start=0.0, sync_offset=-1.0) == 0.0
