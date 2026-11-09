"""Harmonic draft: key/chord detection, Romans, symbols, sequences, edits."""

import json

import pytest

from baroque_transcribe.analysis import (
    AnalysisError,
    analysis_is_stale,
    analyze_harmony,
    load_analysis,
    preserve_edited,
    read_analysis,
    write_analysis,
)
from baroque_transcribe.defaults import ANNOTATIONS_JSON, MANIFEST, NOTES_JSON
from baroque_transcribe.pipeline import _try_generate_analysis, analyze_project
from baroque_transcribe.quantize import quantize_notes
from baroque_transcribe.schema import NoteEvent, Transcription

C = [48, 60, 64, 67]
F = [41, 53, 57, 60]
G = [43, 55, 59, 62]
AM = [45, 57, 60, 64]
DM = [38, 50, 53, 57]
EM = [40, 52, 55, 59]
BB = [46, 58, 62, 65]
EB = [39, 51, 55, 58]


def _quantization(measures, *, bpm=120.0):
    """One sustained chord per 4/4 measure at 120 BPM (2 seconds each)."""

    notes = []
    for index, pitches in enumerate(measures):
        start = index * (240.0 / bpm)
        for pitch in pitches:
            notes.append(NoteEvent(pitch, start, start + 240.0 / bpm - 0.1, 90))
    return quantize_notes(
        notes, bpm=bpm, numerator=4, denominator=4, subdivision=16, downbeat=0.0
    )


def _romans(analysis):
    return [chord.roman for chord in analysis.chords]


def _symbols(analysis):
    return [chord.symbol for chord in analysis.chords]


def test_major_progression_roman_and_symbols():
    analysis = analyze_harmony(_quantization([C, F, G, C]), key_tonic="C", key_mode="major")
    assert _romans(analysis) == ["I", "IV", "V", "I"]
    assert _symbols(analysis) == ["C", "F", "G", "C"]
    assert [chord.measure for chord in analysis.chords] == [1, 2, 3, 4]
    assert all(chord.confidence > 0.5 for chord in analysis.chords)


def test_automatic_key_estimate_finds_c_major():
    analysis = analyze_harmony(_quantization([C, F, G, C]))
    assert (analysis.key_tonic, analysis.key_mode) == ("C", "major")
    assert analysis.key_source == "automatic"
    assert _romans(analysis) == ["I", "IV", "V", "I"]


def test_minor_progression_uses_minor_numerals():
    analysis = analyze_harmony(_quantization([AM, DM, G, C]), key_tonic="A", key_mode="minor")
    assert _romans(analysis) == ["i", "iv", "VII", "III"]
    assert _symbols(analysis) == ["Am", "Dm", "G", "C"]


def test_flat_key_override_spells_symbols_with_flats():
    analysis = analyze_harmony(_quantization([BB, EB, BB]), key_tonic="Bb", key_mode="major")
    assert _symbols(analysis) == ["Bb", "Eb", "Bb"]
    assert _romans(analysis) == ["I", "IV", "I"]


def test_inversion_becomes_slash_bass_and_figured_suffix():
    analysis = analyze_harmony(_quantization([[40, 60, 64, 67]]), key_tonic="C", key_mode="major")
    assert len(analysis.chords) == 1
    assert analysis.chords[0].symbol == "C/E"
    assert analysis.chords[0].roman == "I6"
    assert analysis.chords[0].inversion == 1


def test_six_chord_is_not_labelled_as_a_seventh():
    analysis = analyze_harmony(_quantization([[40, 56, 59, 61]]), key_tonic="E", key_mode="major")
    assert analysis.chords[0].symbol == "E6"
    assert analysis.chords[0].roman == "I"


def test_short_passing_tone_does_not_overturn_the_chord():
    notes = [NoteEvent(pitch, 0.0, 1.9, 90) for pitch in C]
    notes.append(NoteEvent(62, 0.5, 0.58, 25))  # brief quiet D between chord tones
    quantization = quantize_notes(
        notes, bpm=120, numerator=4, denominator=4, subdivision=16, downbeat=0.0
    )
    analysis = analyze_harmony(quantization, key_tonic="C", key_mode="major")
    assert _symbols(analysis) == ["C"]


def test_mid_measure_chord_change_splits_at_the_beat():
    notes = [NoteEvent(pitch, 0.0, 0.95, 90) for pitch in C]
    notes += [NoteEvent(pitch, 1.0, 1.95, 90) for pitch in G]
    quantization = quantize_notes(
        notes, bpm=120, numerator=4, denominator=4, subdivision=16, downbeat=0.0
    )
    analysis = analyze_harmony(quantization, key_tonic="C", key_mode="major")
    assert [chord.measure for chord in analysis.chords] == [1, 1]
    assert _romans(analysis) == ["I", "V"]
    assert analysis.chords[1].beat == pytest.approx(2.0)


def test_empty_measure_gets_no_label():
    analysis = analyze_harmony(_quantization([C, [], G]), key_tonic="C", key_mode="major")
    assert [chord.measure for chord in analysis.chords] == [1, 3]


def test_descending_fifth_sequence_span():
    analysis = analyze_harmony(_quantization([AM, DM, G, C]), key_tonic="A", key_mode="minor")
    kinds = {(item.kind, item.start_measure, item.end_measure) for item in analysis.sequences}
    assert ("descending-fifth", 1, 4) in kinds


def test_diatonic_step_sequence_despite_mixed_tones():
    analysis = analyze_harmony(_quantization([C, DM, EM, F]), key_tonic="C", key_mode="major")
    assert any(
        item.kind == "ascending-step" and item.start_measure == 1 and item.end_measure == 4
        for item in analysis.sequences
    )


def test_edited_labels_survive_preserve_but_not_a_fresh_draft():
    quantization = _quantization([C, F])
    old = analyze_harmony(quantization, key_tonic="C", key_mode="major")
    old.chords[0].roman = "I (home)"
    old.chords[0].edited = True
    old.chords[0].edited_fields = ["roman"]
    fresh = analyze_harmony(quantization, key_tonic="C", key_mode="major")
    merged = preserve_edited(fresh, old)
    assert merged.chords[0].roman == "I (home)"
    assert merged.chords[0].edited is True
    assert merged.chords[1].edited is False


def test_stale_when_a_note_changes():
    analysis = analyze_harmony(_quantization([C]), key_tonic="C", key_mode="major")
    assert not analysis_is_stale(analysis, _quantization([C]))
    changed = _quantization([[48, 60, 65, 67]])
    assert analysis_is_stale(analysis, changed)


def test_analysis_round_trip_and_bad_file(tmp_path):
    analysis = analyze_harmony(_quantization([C, G]), key_tonic="C", key_mode="major")
    path = tmp_path / ANNOTATIONS_JSON
    write_analysis(path, analysis)
    loaded = read_analysis(path)
    assert loaded.key_tonic == "C"
    assert _romans(loaded) == ["I", "V"]
    path.write_text("{ not json", encoding="utf-8")
    assert load_analysis(path) is None
    with pytest.raises(AnalysisError):
        read_analysis(path)


def _write_project(folder, measures):
    folder.mkdir(parents=True, exist_ok=True)
    notes = []
    for index, pitches in enumerate(measures):
        start = index * 2.0
        for pitch in pitches:
            notes.append(NoteEvent(pitch, start, start + 1.9, 90))
    Transcription(notes=notes, model="test", source="test").write(folder / NOTES_JSON)
    manifest = {
        "title": "Piece",
        "tempo": {"bpm": 120, "downbeat_seconds": 0, "subdivision": 16, "time_signature": "4/4"},
    }
    (folder / MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return folder


def test_analyze_project_writes_file_and_manifest_summary(tmp_path):
    project = _write_project(tmp_path / "piece", [C, F, G, C])
    analysis = analyze_project(project, key="C", mode="major")
    assert _romans(analysis) == ["I", "IV", "V", "I"]
    stored = read_analysis(project / ANNOTATIONS_JSON)
    assert _symbols(stored) == ["C", "F", "G", "C"]
    manifest = json.loads((project / MANIFEST).read_text(encoding="utf-8"))
    assert manifest["analysis"]["chords"] == 4
    assert manifest["analysis"]["key"] == "C major"


def test_analyze_project_preserves_edits_unless_forced(tmp_path):
    project = _write_project(tmp_path / "piece", [C, F])
    analyze_project(project, key="C", mode="major")
    stored = read_analysis(project / ANNOTATIONS_JSON)
    stored.chords[0].roman = "I (home)"
    stored.chords[0].edited = True
    stored.chords[0].edited_fields = ["roman"]
    write_analysis(project / ANNOTATIONS_JSON, stored)

    rebuilt = analyze_project(project)
    assert rebuilt.chords[0].roman == "I (home)"
    forced = analyze_project(project, force=True)
    assert forced.chords[0].roman == "I"


def test_analyze_project_without_tempo_is_a_clear_error(tmp_path):
    project = tmp_path / "piece"
    project.mkdir()
    Transcription(notes=[NoteEvent(60, 0.0, 0.5, 80)], model="test").write(project / NOTES_JSON)
    (project / MANIFEST).write_text("{}", encoding="utf-8")
    with pytest.raises(AnalysisError, match="tempo"):
        analyze_project(project)


def test_failed_analysis_never_breaks_the_transcription(tmp_path):
    project = tmp_path / "piece"
    project.mkdir()
    Transcription(notes=[NoteEvent(60, 0.0, 0.5, 80)], model="test").write(project / NOTES_JSON)
    (project / MANIFEST).write_text("{}", encoding="utf-8")
    before = (project / NOTES_JSON).read_bytes()
    warning = _try_generate_analysis(project, None)
    assert warning is not None and "not generated" in warning
    assert (project / NOTES_JSON).read_bytes() == before
    assert not (project / ANNOTATIONS_JSON).exists()
