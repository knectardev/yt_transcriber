from pathlib import Path

import pretty_midi
from music21 import converter

from baroque_transcribe.midi_export import write_quantized_midi, write_raw_midi
from baroque_transcribe.notation import (
    format_measure_time,
    measure_time_labels,
    musicxml_for_midi,
    write_musicxml,
    write_notation_page,
)
from baroque_transcribe.quantize import quantize_notes
from baroque_transcribe.schema import NoteEvent, PedalEvent


def _performance() -> list[NoteEvent]:
    return [
        NoteEvent(60, 0.0, 0.5, 80),
        NoteEvent(64, 0.0, 0.5, 70),
        NoteEvent(67, 0.0, 1.0, 75),
        NoteEvent(62, 0.5, 1.0, 68),
        NoteEvent(48, 0.0, 2.0, 60),
    ]


def test_raw_midi_preserves_timing_and_pedal(tmp_path: Path):
    notes = [NoteEvent(61, 0.137, 0.58, 90)]
    pedals = [PedalEvent(0.05, 0.9)]
    path = tmp_path / "raw.mid"
    write_raw_midi(path, notes, pedals)
    midi = pretty_midi.PrettyMIDI(str(path))
    assert len(midi.instruments) == 1
    note = midi.instruments[0].notes[0]
    assert note.pitch == 61
    assert note.start == pytest_approx(0.137)
    assert note.end == pytest_approx(0.58)
    assert note.velocity == 90
    controls = [change for change in midi.instruments[0].control_changes if change.number == 64]
    assert [change.value for change in controls] == [127, 0]
    assert midi.get_tempo_changes()[1][0] == pytest_approx(120)


def test_quantized_midi_and_musicxml_keep_the_same_pitches(tmp_path: Path):
    quantization = quantize_notes(
        _performance(),
        bpm=120,
        numerator=4,
        denominator=4,
        subdivision=16,
        downbeat=0.0,
    )
    midi_path = tmp_path / "quantized.mid"
    score_path = tmp_path / "score.musicxml"
    write_quantized_midi(midi_path, quantization, staff_split=60)
    write_musicxml(score_path, quantization, staff_split=60, title="Test")

    midi = pretty_midi.PrettyMIDI(str(midi_path))
    assert midi.time_signature_changes[0].numerator == 4
    midi_pitches = sorted(note.pitch for instrument in midi.instruments for note in instrument.notes)
    assert midi_pitches == [48, 60, 62, 64, 67]
    right = midi.instruments[0].notes
    left = midi.instruments[1].notes
    assert {note.pitch for note in left} == {48}
    assert 60 in {note.pitch for note in right}

    score = converter.parse(str(score_path))
    found = []
    for element in score.recurse().notes:
        if element.isChord:
            found.extend(inner.pitch.midi for inner in element.notes)
        else:
            found.append(element.pitch.midi)
    assert sorted(set(found)) == [48, 60, 62, 64, 67]
    parts = list(score.parts)
    assert len(parts) == 2
    assert len(parts[0].getElementsByClass("Measure")) == 1


def test_measure_time_is_rounded_to_the_nearest_second():
    assert format_measure_time(0) == "(0:00)"
    assert format_measure_time(4.35) == "(0:04)"
    assert format_measure_time(4.5) == "(0:05)"
    assert format_measure_time(59.6) == "(1:00)"
    assert format_measure_time(65.4) == "(1:05)"


def test_each_measure_is_labeled_with_its_start_time(tmp_path: Path):
    quantization = quantize_notes(
        [NoteEvent(60, 0.0, 0.5, 80), NoteEvent(64, 3.0, 3.5, 80)],
        bpm=120,
        numerator=4,
        denominator=4,
        subdivision=16,
        downbeat=0.0,
    )
    score_path = tmp_path / "score.musicxml"
    write_musicxml(score_path, quantization, title="Timed")
    assert measure_time_labels(score_path) == {"1": "(0:00)", "2": "(0:02)"}
    page = write_notation_page(score_path, tmp_path / "notation.html")
    text = page.read_text(encoding="utf-8")
    assert '"1": "(0:00)"' in text
    assert '"2": "(0:02)"' in text
    assert "stampMeasureTimes" in text
    assert "measureNumberInterval: 1" in text


def test_notation_page_embeds_the_score_without_closing_the_script(tmp_path):
    score_path = tmp_path / "score.musicxml"
    score_path.write_text(
        "<score-partwise><part id='right'/></script><part id='left'/></score-partwise>",
        encoding="utf-8",
    )
    page = write_notation_page(score_path, tmp_path / "notation.html", title="Two staves")
    text = page.read_text(encoding="utf-8")
    assert "opensheetmusicdisplay.min.js" in text
    assert "cdn.jsdelivr.net" not in text
    assert (tmp_path / "opensheetmusicdisplay.min.js").is_file()
    assert "Upper staff: right-hand track" in text
    assert "Lower staff: left-hand track" in text
    assert "</script><part" not in text
    assert "\\u003c/script>" in text
    assert "Library" not in text
    linked = write_notation_page(
        score_path, tmp_path / "notation.html", title="Two staves", home_url="http://127.0.0.1:8765/?a=1&b=2"
    ).read_text(encoding="utf-8")
    assert 'href="http://127.0.0.1:8765/?a=1&amp;b=2"' in linked
    assert "&larr; Library" in linked


def test_notation_page_heading_for_a_project(tmp_path):
    score_path = tmp_path / "score.musicxml"
    score_path.write_text("<score-partwise><part id='right'/></score-partwise>", encoding="utf-8")
    # A score that belongs to no project keeps the plain heading.
    plain = write_notation_page(score_path, tmp_path / "notation.html", title="Loose").read_text(encoding="utf-8")
    assert '<div class="facts">' not in plain
    assert '<div class="path">' not in plain
    assert "Play with score" not in plain
    assert "Edit &amp; files" not in plain

    text = write_notation_page(
        score_path,
        tmp_path / "notation.html",
        title="Take",
        home_url="http://127.0.0.1:8765/",
        facts=[
            {"before": "from ", "value": "0:12", "after": " in the video"},
            {"before": "", "value": "", "after": "model <x>"},
        ],
        folder=r"C:\music\take & one",
        tools_url="http://127.0.0.1:8765/project?project=take",
        playback_url="http://127.0.0.1:5000/p/abc/",
    ).read_text(encoding="utf-8")
    assert '<span class="fact">from <b>0:12</b> in the video</span>' in text
    assert '<span class="fact">model &lt;x&gt;</span>' in text
    assert '<div class="path">C:\\music\\take &amp; one</div>' in text
    assert 'href="http://127.0.0.1:8765/">&larr; Library</a>' in text
    assert 'href="http://127.0.0.1:5000/p/abc/">Play with score</a>' in text
    assert 'href="http://127.0.0.1:8765/project?project=take">Edit &amp; files</a>' in text
    # Every link stays in this tab.
    assert "target=" not in text


def test_notation_page_draws_the_harmony_layer_with_a_toggle(tmp_path):
    from baroque_transcribe.analysis import ChordEvent, HarmonicAnalysis, SequenceEvent, write_analysis

    score_path = tmp_path / "score.musicxml"
    score_path.write_text("<score-partwise><part id='right'/></score-partwise>", encoding="utf-8")
    write_analysis(
        tmp_path / "annotations.json",
        HarmonicAnalysis(
            key_tonic="C",
            key_mode="major",
            key_source="manual",
            key_confidence=1.0,
            chords=[
                ChordEvent(
                    measure=1, beat=0.0, measure_position=0.0, start_step=0, end_step=16,
                    root_pc=0, root_name="C", quality="major", bass_pc=0, bass_name="C",
                    roman="I", symbol="C", inversion=0, confidence=0.9,
                )
            ],
            sequences=[
                SequenceEvent(
                    start_measure=1, end_measure=2, kind="descending-fifth",
                    description="Descending fifth sequence", repetitions=1, confidence=0.8,
                )
            ],
        ),
    )
    text = write_notation_page(score_path, tmp_path / "notation.html").read_text(encoding="utf-8")
    assert "Lead-sheet chord symbols" in text
    assert '"roman": "I"' in text
    assert '"symbol": "C"' in text
    assert "drawHarmony" in text
    assert "Descending fifth sequence" in text


def test_notation_page_survives_malformed_annotations(tmp_path):
    score_path = tmp_path / "score.musicxml"
    score_path.write_text("<score-partwise><part id='right'/></score-partwise>", encoding="utf-8")
    (tmp_path / "annotations.json").write_text("{ not json", encoding="utf-8")
    text = write_notation_page(score_path, tmp_path / "notation.html").read_text(encoding="utf-8")
    assert "const harmony = null;" in text
    assert "opensheetmusicdisplay.min.js" in text


def test_project_midi_reuses_the_saved_score(tmp_path):
    score = tmp_path / "score.musicxml"
    score.write_text("<score-partwise><!-- saved --></score-partwise>", encoding="utf-8")
    (tmp_path / "notes.json").write_text("{}\n", encoding="utf-8")
    midi = tmp_path / "quantized.mid"
    midi.write_bytes(b"MThd")
    before = score.read_bytes()
    assert musicxml_for_midi(midi) == score.resolve()
    assert score.read_bytes() == before


def test_other_midi_is_converted_without_a_project_score(tmp_path):
    import pretty_midi

    midi_path = tmp_path / "hand.mid"
    midi = pretty_midi.PrettyMIDI(initial_tempo=90)
    piano = pretty_midi.Instrument(program=0, name="Piano")
    piano.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=1.0))
    midi.instruments.append(piano)
    midi.write(str(midi_path))
    written = musicxml_for_midi(midi_path)
    assert written.is_file()
    assert written != midi_path
    text = written.read_text(encoding="utf-8")
    assert "score-partwise" in text
    assert musicxml_for_midi(midi_path) == written


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, abs=0.01)
