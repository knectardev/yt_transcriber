"""Estimate when each printed measure starts in the working audio.

The YouTube player is the master clock. These functions turn a playhead into
a measure number, and a measure number back into a video time. They do not
render notation and they do not move a note cursor.

Anchors are estimates, not detected barlines. A measure often opens with a
rest or a note held over from the previous bar before its first new attack.
That opening is placed with the written tempo. Rubato during the opening
shows up only as how far the first new attack sits from its grid point.
"""

from __future__ import annotations

from dataclasses import dataclass

from baroque_transcribe.quantize import Quantization

_TIME_EPSILON = 1e-9


@dataclass(frozen=True)
class MeasureAnchor:
    """Estimated start of one printed measure, in working-audio seconds.

    ``time`` is the estimate used both to highlight the measure and to seek
    the video. ``opens_before_attack`` means the printed measure contains
    grid time before its first new note, or no new note at all. ``collided``
    means the raw estimate was not later than the previous measure, so a
    deterministic fallback supplied ``time``.
    """

    measure: int
    time: float
    collided: bool
    opens_before_attack: bool


@dataclass(frozen=True)
class NoteAnchor:
    """One quantized note for playback note-cursor highlighting and seek."""

    source_index: int
    pitch: int
    onset: float
    offset: float
    measure: int


def note_anchors(quantization: Quantization) -> list[NoteAnchor]:
    """Build working-clock note anchors from the quantized grid."""

    first, _count = printed_measure_span(quantization)
    per = quantization.steps_per_measure
    anchors: list[NoteAnchor] = []
    for note in sorted(quantization.grid, key=lambda item: (item.onset_step, item.pitch, item.source_index)):
        measure = ((note.onset_step - first) // per) + 1
        onset = float(note.source_onset)
        offset = float(note.source_offset)
        if offset <= onset:
            offset = onset + 0.05
        anchors.append(
            NoteAnchor(
                source_index=note.source_index,
                pitch=int(note.pitch),
                onset=onset,
                offset=offset,
                measure=int(measure),
            )
        )
    return anchors


def printed_measure_span(quantization: Quantization) -> tuple[int, int]:
    """Return ``(first_step, measure_count)`` for the printed score.

    The loop matches ``_fill_part``: measure 1 starts at ``-lead_steps``, and
    the last measure is the bar that contains the latest note offset. Offsets
    are unchanged by the staff split, so this count is the longer staff.
    """

    per_measure = quantization.steps_per_measure
    first = -quantization.lead_steps
    if quantization.grid:
        last_note = max(item.offset_step for item in quantization.grid)
    else:
        last_note = per_measure
    last_bar = (max(last_note - 1, first) // per_measure) * per_measure
    if last_bar < first:
        last_bar = first
    count = ((last_bar - first) // per_measure) + 1
    return first, count


def measure_anchors(quantization: Quantization) -> list[MeasureAnchor]:
    """Estimate a start time for every printed measure.

    A note begins in a measure when its onset step falls in that measure.
    A tie continued from the previous measure does not begin there.

    When notes do begin there, the earliest attack is the one with the
    smallest onset step, breaking ties by original onset. If that attack is
    on the measure's first step, the candidate is the attack's original
    onset. If the attack is later, the measure opens with a rest or a
    sustained note, and the candidate backs up by the written duration of
    those opening steps. That backup is an estimate at the written tempo.

    A measure with no new attack uses the metronome time of its first step.
    The sustained note's original onset stays with the measure where it began.

    Candidates are then made strictly increasing. A candidate that is not
    later than the previous anchor is a collision. The fallback is the
    metronome barline when that barline is still later, and otherwise one
    subdivision after the previous anchor. Collided measures keep a positive
    duration and a seek time of their own.
    """

    per_measure = quantization.steps_per_measure
    first, count = printed_measure_span(quantization)
    notes = quantization.grid
    anchors: list[MeasureAnchor] = []
    previous: float | None = None
    for index in range(count):
        start = first + index * per_measure
        end = start + per_measure
        beginning = [note for note in notes if start <= note.onset_step < end]
        grid_start = quantization.step_to_audio(start)
        if not beginning:
            candidate = grid_start
            opens_before_attack = True
        else:
            first_note = min(beginning, key=lambda note: (note.onset_step, note.source_onset, note.source_index))
            # Back up from the first attack using the step-time map so a
            # follow-grid score stays aligned with the performance.
            if first_note.onset_step == start:
                candidate = first_note.source_onset
            else:
                candidate = first_note.source_onset - (
                    quantization.step_to_audio(first_note.onset_step) - quantization.step_to_audio(start)
                )
            opens_before_attack = first_note.onset_step > start
        collided = False
        if previous is not None and candidate <= previous + _TIME_EPSILON:
            collided = True
            fallback_gap = quantization.step_to_audio(start + 1) - quantization.step_to_audio(start)
            if fallback_gap <= _TIME_EPSILON:
                fallback_gap = quantization.quantum
            if grid_start > previous + _TIME_EPSILON:
                chosen = grid_start
            else:
                chosen = previous + fallback_gap
        else:
            chosen = candidate
        anchors.append(
            MeasureAnchor(
                measure=index + 1,
                time=chosen,
                collided=collided,
                opens_before_attack=opens_before_attack,
            )
        )
        previous = chosen
    return anchors


def measure_at_working_time(anchors: list[MeasureAnchor], working_seconds: float) -> int | None:
    """Return the measure whose estimated window contains ``working_seconds``.

    The lookup is stateless, so a later call with an earlier time seeks
    backward. Times before the first anchor stay on the first measure, and
    times after the last anchor stay on the last measure.
    """

    if not anchors:
        return None
    chosen = anchors[0].measure
    for anchor in anchors:
        if working_seconds + _TIME_EPSILON >= anchor.time:
            chosen = anchor.measure
        else:
            break
    return chosen


def working_time(video_seconds: float, *, excerpt_start: float, sync_offset: float) -> float:
    """Working-audio seconds for a video playhead.

    ``sync_offset`` moves the score relative to the video. A larger offset
    selects an earlier measure for the same video time.
    """

    return video_seconds - excerpt_start - sync_offset


def video_time(working_seconds: float, *, excerpt_start: float, sync_offset: float) -> float:
    """Video seconds for a working-audio time. Inverse of ``working_time``."""

    return working_seconds + excerpt_start + sync_offset


def measure_seek_video(anchor: MeasureAnchor, *, excerpt_start: float, sync_offset: float) -> float:
    """Video time to seek when the measure is clicked. Never negative."""

    return max(0.0, video_time(anchor.time, excerpt_start=excerpt_start, sync_offset=sync_offset))
