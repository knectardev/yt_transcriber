"""Engraving: grand staff, key spelling, preservation, and voice colours."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from music21 import converter

from baroque_transcribe.analysis import spell_pitch_name
from baroque_transcribe.engrave import interpret_quantization, validate_note_preservation
from baroque_transcribe.notation import write_musicxml
from baroque_transcribe.notation_settings import NotationSettings
from baroque_transcribe.quantize import quantize_notes
from baroque_transcribe.schema import NoteEvent


def _q(notes, *, grid="follow", style="readable"):
    return quantize_notes(
        notes,
        bpm=120,
        numerator=4,
        denominator=4,
        subdivision=16,
        settings=NotationSettings(grid=grid, style=style),
    )


def test_spell_pitch_name_prefers_key_and_raised_leading_tone():
    assert spell_pitch_name(66, "G", "minor") == "F#"  # raised 7 in G minor
    assert spell_pitch_name(58, "G", "minor") == "Bb"
    assert spell_pitch_name(61, "C", "major", rising=True) == "C#"
    assert spell_pitch_name(61, "C", "major", rising=False) == "Db"


def test_readable_score_is_one_piano_part_with_key_and_color(tmp_path: Path):
    notes = [
        NoteEvent(67, 0.0, 1.0, 80),
        NoteEvent(62, 0.0, 0.5, 70),
        NoteEvent(62, 0.5, 1.0, 70),
        NoteEvent(43, 0.0, 2.0, 60),
    ]
    quantization = _q(notes)
    path = tmp_path / "score.musicxml"
    write_musicxml(
        path,
        quantization,
        style="readable",
        key=("G", "minor"),
        colorize_voices=True,
        notation_map=tmp_path / "map.json",
        preservation_report=tmp_path / "preservation.json",
    )
    text = path.read_text(encoding="utf-8")
    assert text.count("<score-part ") == 1
    assert "<staff>" in text
    assert "color=" in text
    assert "<fifths>" in text
    report = json.loads((tmp_path / "preservation.json").read_text(encoding="utf-8"))
    assert report["ok"] is True
    assert report["matched"] == 4


def test_secondary_voice_rests_are_hidden_on_print(tmp_path: Path):
    # Sustained upper voice plus a mid-bar moving voice: gap rests belong to voice 2.
    notes = [
        NoteEvent(72, 0.0, 2.0, 80),
        NoteEvent(64, 0.5, 1.0, 70),
        NoteEvent(64, 1.0, 1.5, 70),
        NoteEvent(48, 0.0, 2.0, 60),
    ]
    path = tmp_path / "score.musicxml"
    write_musicxml(path, _q(notes, grid="fixed"), style="readable", key=("C", "major"))
    text = path.read_text(encoding="utf-8")
    # Hidden rests become <forward> so OSMD cannot paint floating glyphs.
    assert text.count("<forward>") >= 2
    assert 'print-object="no"' not in text or "<forward>" in text


def test_preservation_catches_a_dropped_note():
    notes = [NoteEvent(60, 0.0, 0.5, 80), NoteEvent(64, 0.5, 1.0, 70)]
    quantization = _q(notes)
    interpretation = interpret_quantization(quantization)
    interpretation.notes = interpretation.notes[:1]
    report = validate_note_preservation(quantization, interpretation)
    assert report.ok is False
    assert report.unmatched[0].source_index == 1


def test_existing_pitch_round_trip_still_holds(tmp_path: Path):
    notes = [
        NoteEvent(60, 0.0, 0.5, 80),
        NoteEvent(64, 0.0, 0.5, 70),
        NoteEvent(67, 0.0, 1.0, 75),
        NoteEvent(62, 0.5, 1.0, 68),
        NoteEvent(48, 0.0, 2.0, 60),
    ]
    quantization = _q(notes, grid="fixed", style="literal")
    path = tmp_path / "score.musicxml"
    write_musicxml(path, quantization, style="literal", key=("C", "major"))
    score = converter.parse(str(path))
    found = []
    for element in score.recurse().notes:
        if element.isChord:
            found.extend(inner.pitch.midi for inner in element.notes)
        else:
            found.append(element.pitch.midi)
    assert sorted(set(found)) == [48, 60, 62, 64, 67]
