"""MIDI import becomes a first-class Library project (BWV 870 fixture)."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.request import urlopen

import pretty_midi
import pytest

from baroque_transcribe.fragments import extract_fragment, plan_fragment
from baroque_transcribe.library import library_cards, read_run
from baroque_transcribe.midi_import import MidiImportError, import_midi_project, parse_midi
from baroque_transcribe.playback import PlaybackServer, load_playback, write_playback_page
from baroque_transcribe.schema import read_transcription

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "wtk2prelude1.mid"


@pytest.fixture(scope="module")
def bwv_midi() -> Path:
    if not FIXTURE.is_file():
        pytest.skip(f"Missing fixture {FIXTURE}")
    return FIXTURE


def test_parse_midi_keeps_absolute_times_and_track_staffs(bwv_midi: Path):
    transcription, timing = parse_midi(bwv_midi)
    assert timing.track_staffs
    assert timing.tempo_map
    assert timing.time_signature_map
    assert timing.bpm == pytest.approx(70.0, rel=1e-3)
    assert timing.time_signature == (4, 4)
    assert transcription.notes
    assert all(note.staff in (1, 2) for note in transcription.notes)
    assert {note.staff for note in transcription.notes} == {1, 2}
    # Absolute seconds match pretty_midi (tempo map already applied).
    midi = pretty_midi.PrettyMIDI(str(bwv_midi))
    assert max(note.offset for note in transcription.notes) == pytest.approx(midi.get_end_time(), rel=1e-3)


def test_import_midi_project_creates_library_run(bwv_midi: Path, tmp_path: Path):
    project = import_midi_project(bwv_midi, tmp_path / "wtk2prelude1", title="WTC II Prelude 1")
    run = read_run(project)
    assert run is not None
    assert run.source_kind == "midi"
    assert run.model == "midi-import"
    assert run.notes and run.notes > 700
    assert (project / "notes.json").is_file()
    assert (project / "score.musicxml").is_file()
    assert (project / "source.mid").is_file()
    assert (project / "quantized.mid").is_file()
    manifest = json.loads((project / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["source_kind"] == "midi"
    assert manifest["staff_assignment"] == "tracks"
    assert manifest["tempo"]["tempo_map"]
    assert manifest["tempo"]["time_signature_map"]
    assert manifest["notation"].get("score_from") == "midi_tracks"

    cards = library_cards(root=tmp_path, store=tmp_path / "empty-store")
    # library_cards looks under transcriptions/; point root at parent holding project
    cards = library_cards(root=project.parent, store=tmp_path / "empty-store")
    midi_cards = [card for card in cards if card.primary and card.primary.name == project.name]
    assert midi_cards
    assert midi_cards[0].video_id is None


def test_imported_score_keeps_two_staves(bwv_midi: Path, tmp_path: Path):
    project = import_midi_project(bwv_midi, tmp_path / "score-check")
    xml = (project / "score.musicxml").read_text(encoding="utf-8")
    # music21 piano export uses PartStaff / staff elements for both hands.
    assert xml.lower().count("staff") >= 2 or "partstaff" in xml.lower() or xml.count("<clef>") >= 2


def test_import_playback_timing_highlight_and_passage(bwv_midi: Path, tmp_path: Path):
    project = import_midi_project(bwv_midi, tmp_path / "play-check")
    context = load_playback(project)
    assert context.media_kind in {"audio", "silent"}
    assert context.note_anchors
    assert context.anchors
    assert context.end_time and context.end_time > 100
    # Note anchors use absolute performance times from notes.json.
    notes = read_transcription(project / "notes.json").notes
    first = min(notes, key=lambda item: item.onset)
    matched = [anchor for anchor in context.note_anchors if anchor.pitch == first.pitch]
    assert matched
    assert matched[0].onset == pytest.approx(first.onset, abs=0.05)

    page = write_playback_page(context, project / "playback.html", extract_url="/extract").read_text(
        encoding="utf-8"
    )
    assert f'"mediaKind": "{context.media_kind}"' in page
    assert "Select passage" in page
    assert "extractUrl" in page
    assert "noteAnchors" in page

    planned = plan_fragment(project, 1, 2)
    assert planned.measure_start == 1
    assert planned.measure_end == 2
    assert planned.notes
    fragment_dir = extract_fragment(
        project,
        1,
        2,
        title="BWV 870 mm.1-2",
        root=tmp_path / "fragments",
    )
    assert (fragment_dir / "fragment.json").is_file()
    assert (fragment_dir / "score.musicxml").is_file()

    server = PlaybackServer(home_url="http://127.0.0.1:8765/")
    server.start()
    try:
        token = server.register(project, "BWV 870")
        with urlopen(server.page_url(token)) as response:
            body = response.read().decode("utf-8")
        assert "BWV 870" in body or "mediaKind" in body
        assert "Select passage" in body
        if context.media_kind == "audio":
            status = urlopen(server.page_url(token) + "preview.wav").status
            assert status == 200
    finally:
        server.close()


def test_single_track_midi_uses_pitch_split(tmp_path: Path):
    midi = pretty_midi.PrettyMIDI(initial_tempo=100)
    inst = pretty_midi.Instrument(program=0, name="Piano")
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=72, start=0.0, end=0.5))
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=48, start=0.0, end=0.5))
    midi.instruments.append(inst)
    path = tmp_path / "single.mid"
    midi.write(str(path))
    transcription, timing = parse_midi(path)
    assert not timing.track_staffs
    assert all(note.staff is None for note in transcription.notes)
    project = import_midi_project(path, tmp_path / "single-out")
    manifest = json.loads((project / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["staff_assignment"] == "pitch_split"


def test_missing_midi_raises(tmp_path: Path):
    with pytest.raises(MidiImportError):
        parse_midi(tmp_path / "missing.mid")
