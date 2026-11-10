"""Voice-colour toggle and continuity metrics."""

from __future__ import annotations

from pathlib import Path

from baroque_transcribe.engrave import interpret_quantization, validate_against_source_notes
from baroque_transcribe.notation import write_musicxml, write_notation_page
from baroque_transcribe.notation_settings import NotationSettings
from baroque_transcribe.quantize import quantize_notes
from baroque_transcribe.schema import NoteEvent
from baroque_transcribe.voice_metrics import staff_split_assessment, voice_continuity_report


def test_notation_page_embeds_voice_toggle_when_coloured_score_exists(tmp_path: Path):
    notes = [
        NoteEvent(67, 0.0, 1.0, 80),
        NoteEvent(62, 0.0, 0.5, 70),
        NoteEvent(62, 0.5, 1.0, 70),
        NoteEvent(48, 0.0, 2.0, 60),
    ]
    quantization = quantize_notes(
        notes,
        bpm=120,
        numerator=4,
        denominator=4,
        subdivision=16,
        settings=NotationSettings(grid="follow", style="readable"),
    )
    plain = tmp_path / "score.musicxml"
    voices = tmp_path / "score.voices.musicxml"
    write_musicxml(
        plain,
        quantization,
        style="readable",
        voices_path=voices,
        notation_map=tmp_path / "map.json",
        preservation_report=tmp_path / "preservation.json",
    )
    assert voices.is_file()
    assert "color=" not in plain.read_text(encoding="utf-8")
    assert "color=" in voices.read_text(encoding="utf-8")
    text = write_notation_page(plain, tmp_path / "notation.html", voices_score_path=voices).read_text(
        encoding="utf-8"
    )
    assert "Color voices" in text
    assert "xmlVoices" in text
    assert "baroque-transcribe.color-voices" in text
    assert "color-toggle" in text


def test_source_note_preservation_checks_pitch_and_onset():
    notes = [
        NoteEvent(60, 0.0, 0.5, 80),
        NoteEvent(64, 0.5, 1.0, 70),
        NoteEvent(48, 0.0, 1.0, 60),
    ]
    quantization = quantize_notes(
        notes,
        bpm=120,
        numerator=4,
        denominator=4,
        subdivision=16,
        settings=NotationSettings(grid="fixed", style="readable"),
    )
    interpretation = interpret_quantization(quantization)
    report = validate_against_source_notes(notes, quantization, interpretation)
    assert report.ok
    assert report.matched == 3


def test_voice_continuity_and_staff_assessment():
    notes = [
        NoteEvent(72, 0.0, 2.0, 80),
        NoteEvent(67, 0.0, 0.5, 70),
        NoteEvent(67, 0.5, 1.0, 70),
        NoteEvent(67, 1.0, 1.5, 70),
        NoteEvent(48, 0.0, 2.0, 60),
    ]
    quantization = quantize_notes(
        notes,
        bpm=120,
        numerator=4,
        denominator=4,
        subdivision=16,
        settings=NotationSettings(grid="fixed", style="readable"),
    )
    interpretation = interpret_quantization(quantization)
    rows = voice_continuity_report(interpretation, quantization)
    assert rows
    assert all(row.note_count >= 1 for row in rows)
    assessment = staff_split_assessment(interpretation)
    assert assessment["staff_split"] == 60
    assert "during voice assignment" in assessment["timing"]


def test_no_duplicate_tempo_on_bass_staff(tmp_path: Path):
    notes = [NoteEvent(60, 0.0, 1.0, 80), NoteEvent(48, 0.0, 1.0, 60)]
    quantization = quantize_notes(
        notes,
        bpm=96,
        numerator=4,
        denominator=4,
        subdivision=16,
        settings=NotationSettings(grid="fixed", style="readable"),
    )
    path = tmp_path / "score.musicxml"
    write_musicxml(path, quantization, style="readable", key=("C", "major"))
    text = path.read_text(encoding="utf-8")
    # One metronome/sound tempo is enough for a PartStaff score.
    assert text.count('tempo="96') + text.count("<per-minute>96") <= 2
