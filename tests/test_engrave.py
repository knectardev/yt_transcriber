"""Interpretation: voices, chords, sustains, rests, and note preservation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from music21 import converter

from baroque_transcribe.engrave import assign_voices, interpret_quantization, write_notation_map
from baroque_transcribe.notation import write_musicxml
from baroque_transcribe.notation_settings import NotationSettings
from baroque_transcribe.quantize import quantize_notes
from baroque_transcribe.schema import NoteEvent
from baroque_transcribe.sync_map import printed_measure_span


def _q(notes: list[NoteEvent], *, bpm: float = 120.0, grid: str = "fixed"):
    return quantize_notes(
        notes,
        bpm=bpm,
        numerator=4,
        denominator=4,
        subdivision=16,
        settings=NotationSettings(grid=grid, style="readable"),
    )


def test_three_independent_voices_stay_three():
    # Three stacked lines with independent rhythms on the treble staff.
    notes = [
        NoteEvent(72, 0.0, 2.0, 80),   # sustained upper
        NoteEvent(67, 0.0, 0.5, 70),
        NoteEvent(67, 0.5, 1.0, 70),
        NoteEvent(67, 1.0, 1.5, 70),
        NoteEvent(67, 1.5, 2.0, 70),
        NoteEvent(60, 0.0, 1.0, 65),
        NoteEvent(60, 1.0, 2.0, 65),
    ]
    interpretation = interpret_quantization(_q(notes), staff_split=60)
    treble_voices = interpretation.events_for_staff(1)
    assert len(treble_voices) >= 3


def test_gap_continues_same_voice_instead_of_new_id():
    # A line pauses, then resumes nearby in pitch: keep one voice ID.
    notes = [
        NoteEvent(72, 0.0, 0.5, 80),
        NoteEvent(71, 0.5, 1.0, 80),
        # Gap of about a half note, then continue the same register.
        NoteEvent(74, 2.0, 2.5, 80),
        NoteEvent(72, 2.5, 3.0, 80),
        NoteEvent(48, 0.0, 3.0, 60),
    ]
    quantization = _q(notes)
    written = assign_voices(quantization.grid, settings=quantization.settings, staff_split=60)
    treble = [note for note in written if note.staff == 1]
    assert len({note.voice for note in treble}) == 1


def test_third_simultaneous_voice_is_discouraged_when_two_suffice():
    # Two alternating mid-range lines should not mint a third ID for a later reprise.
    notes = [
        NoteEvent(67, 0.0, 0.5, 70),
        NoteEvent(60, 0.0, 0.5, 65),
        NoteEvent(67, 0.5, 1.0, 70),
        NoteEvent(60, 0.5, 1.0, 65),
        NoteEvent(69, 1.5, 2.0, 70),  # resumes upper after a gap
        NoteEvent(62, 1.5, 2.0, 65),  # resumes lower after a gap
        NoteEvent(48, 0.0, 2.0, 60),
    ]
    quantization = _q(notes)
    written = assign_voices(quantization.grid, settings=quantization.settings, staff_split=60)
    treble = [note for note in written if note.staff == 1]
    assert len({note.voice for note in treble}) <= 2


def test_sparse_third_voice_compacts_into_two():
    # A brief third line that never collides with the two busy voices should fold in.
    notes = [
        NoteEvent(72, 0.0, 0.5, 80),
        NoteEvent(60, 0.0, 0.5, 70),
        NoteEvent(72, 0.5, 1.0, 80),
        NoteEvent(60, 0.5, 1.0, 70),
        NoteEvent(67, 1.25, 1.5, 75),  # mid-register fill between the pair
        NoteEvent(72, 1.5, 2.0, 80),
        NoteEvent(60, 1.5, 2.0, 70),
        NoteEvent(48, 0.0, 2.0, 60),
    ]
    quantization = _q(notes)
    written = assign_voices(quantization.grid, settings=quantization.settings, staff_split=60)
    treble = [note for note in written if note.staff == 1]
    assert len({note.voice for note in treble}) <= 2
    assert len(treble) == 7


def test_suspension_is_not_chorded_with_moving_line():
    notes = [
        NoteEvent(64, 0.0, 1.5, 80),  # holds across the moving attack
        NoteEvent(60, 0.0, 0.5, 70),
        NoteEvent(62, 0.5, 1.0, 70),
        NoteEvent(64, 1.0, 1.5, 70),  # resolution pitch arrives later as motion
    ]
    interpretation = interpret_quantization(_q(notes), staff_split=55)
    # The long hold and the moving line must not share one chord event.
    sustained = [note for note in interpretation.notes if note.source_index == 0][0]
    movers = [note for note in interpretation.notes if note.source_index in {1, 2}]
    assert all(note.voice != sustained.voice or note.onset_step >= sustained.written_offset_step for note in movers)
    assert sustained.written_duration_steps >= 4


def test_block_chord_with_jittered_releases_unifies():
    notes = [
        NoteEvent(60, 0.00, 0.50, 80),
        NoteEvent(64, 0.01, 0.48, 70),
        NoteEvent(67, 0.02, 0.52, 75),
    ]
    quantization = _q(notes, grid="follow")
    interpretation = interpret_quantization(quantization, staff_split=60)
    # Follow clustering puts them on one onset; readable should chord them.
    onsets = {note.onset_step for note in interpretation.notes}
    assert len(onsets) == 1
    voices = {note.voice for note in interpretation.notes}
    assert len(voices) == 1
    assert all(note.chord_with for note in interpretation.notes)


def test_fast_run_is_not_gap_filled_away():
    notes = [NoteEvent(60 + index, index * 0.125, index * 0.125 + 0.1, 80) for index in range(8)]
    interpretation = interpret_quantization(_q(notes), staff_split=60)
    assert len(interpretation.notes) == 8
    assert all(note.gap_filled_steps == 0 for note in interpretation.notes)


def test_line_crossing_middle_c_stays_one_voice():
    notes = [
        NoteEvent(64, 0.0, 0.5, 80),
        NoteEvent(62, 0.5, 1.0, 80),
        NoteEvent(60, 1.0, 1.5, 80),
        NoteEvent(58, 1.5, 2.0, 80),
        NoteEvent(56, 2.0, 2.5, 80),
    ]
    interpretation = interpret_quantization(_q(notes), staff_split=60)
    # Sticky staff near the split should keep these as one continuing voice.
    assert len({(note.staff, note.voice) for note in interpretation.notes}) == 1


def test_notation_map_preserves_every_source_note(tmp_path: Path):
    notes = [
        NoteEvent(60, 0.0, 0.5, 80),
        NoteEvent(64, 0.0, 0.5, 70),
        NoteEvent(48, 0.0, 1.0, 60),
        NoteEvent(67, 0.5, 1.0, 75),
    ]
    quantization = _q(notes)
    interpretation = interpret_quantization(quantization, staff_split=60)
    path = write_notation_map(tmp_path / "notation_map.json", interpretation, quantization)
    payload = json.loads(path.read_text(encoding="utf-8"))
    indices = sorted(entry["source_index"] for entry in payload["notes"])
    assert indices == [0, 1, 2, 3]
    pitches = {entry["source_index"]: entry["pitch"] for entry in payload["notes"]}
    assert pitches == {0: 60, 1: 64, 2: 48, 3: 67}


def test_readable_musicxml_preserves_pitches_and_measure_math(tmp_path: Path):
    notes = [
        NoteEvent(72, 0.0, 1.0, 80),
        NoteEvent(67, 0.0, 0.5, 70),
        NoteEvent(67, 0.5, 1.0, 70),
        NoteEvent(48, 0.0, 2.0, 60),
    ]
    quantization = _q(notes)
    score_path = tmp_path / "score.musicxml"
    write_musicxml(
        score_path,
        quantization,
        staff_split=60,
        style="readable",
        notation_map=tmp_path / "notation_map.json",
    )
    parsed = converter.parse(str(score_path))
    found = []
    for element in parsed.recurse().notes:
        if element.isChord:
            found.extend(inner.pitch.midi for inner in element.notes)
        else:
            found.append(element.pitch.midi)
    assert sorted(found) == [48, 67, 67, 72]
    first, count = printed_measure_span(quantization)
    assert count == len(list(parsed.parts[0].getElementsByClass("Measure")))
    # Readable export hides gap rests as <forward>. Re-parsed music21 voices may
    # drop those gaps, so validate timing from the MusicXML text instead.
    import re

    xml = score_path.read_text(encoding="utf-8")
    divisions_match = re.search(r"<divisions>(\d+)</divisions>", xml)
    assert divisions_match is not None
    divisions = int(divisions_match.group(1))
    bar_duration = divisions * 4  # 4/4 test fixture
    for measure_xml in re.findall(r"<measure\b[^>]*>.*?</measure>", xml, re.S):
        # Split on backup boundaries so each voice stream is checked separately.
        streams = re.split(r"<backup>.*?</backup>", measure_xml, flags=re.S)
        for stream_xml in streams:
            total = 0
            for note_xml in re.findall(r"<note\b[^>]*>.*?</note>", stream_xml, re.S):
                if "<chord" in note_xml:
                    continue
                duration = re.search(r"<duration>(\d+)</duration>", note_xml)
                if duration:
                    total += int(duration.group(1))
            for forward_xml in re.findall(r"<forward>.*?</forward>", stream_xml, re.S):
                duration = re.search(r"<duration>(\d+)</duration>", forward_xml)
                if duration:
                    total += int(duration.group(1))
            if total == 0:
                continue
            assert total == bar_duration, (total, bar_duration)


def test_absent_voice_measures_are_suppressed():
    notes = [
        NoteEvent(72, 0.0, 0.5, 80),
        NoteEvent(60, 2.0, 2.5, 70),  # later measure only
    ]
    interpretation = interpret_quantization(_q(notes), staff_split=60)
    assert any(rest.reason == "absent_voice_measure" for rest in interpretation.suppressed_rests)
