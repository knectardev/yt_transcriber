import json
from pathlib import Path

import pytest

from baroque_transcribe.audio import AudioError, PreparedAudio
from baroque_transcribe.cli import main
from baroque_transcribe.parse_store import (
    default_store,
    describe_source,
    import_projects,
    load_events,
    record_parse,
    thumbnail_url,
    transcribed_videos,
)
from baroque_transcribe.pipeline import prepare_only


def _prepared(source: str, output_dir: Path) -> PreparedAudio:
    output_dir.mkdir(parents=True, exist_ok=True)
    working = output_dir / "working.wav"
    working.write_bytes(b"wav")
    (output_dir / "source.wav").write_bytes(b"wav")
    (output_dir / "notes.json").write_text("{}\n", encoding="utf-8")
    (output_dir / "score.musicxml").write_text("<score/>\n", encoding="utf-8")
    return PreparedAudio(
        source_wav=output_dir / "source.wav",
        working_wav=working,
        sample_rate=16000,
        duration=0.5,
        full_duration=1.0,
        start=0.1,
        end=0.6,
        original_peak=0.2,
        title="phrase",
        origin=str(source),
    )


def _score_project(path: Path, *, origin: str, youtube_url: str | None = None) -> None:
    path.mkdir(parents=True)
    manifest = {
        "origin": origin,
        "title": "Piece",
        "audio": {
            "sample_rate": 16000,
            "full_duration_seconds": 30.0,
            "excerpt_start_seconds": 10.0,
            "excerpt_end_seconds": 20.0,
            "working_duration_seconds": 10.0,
        },
    }
    if youtube_url:
        manifest["youtube_url"] = youtube_url
    (path / "manifest.json").write_text(json.dumps(manifest) + "\n", encoding="utf-8")
    (path / "notes.json").write_text(
        json.dumps({"source": origin, "notes": [{"pitch": 60}]}) + "\n",
        encoding="utf-8",
    )
    (path / "pedal.json").write_text('{"pedals": []}\n', encoding="utf-8")
    (path / "raw.mid").write_bytes(b"MThd")
    (path / "quantized.mid").write_bytes(b"MThd")
    (path / "score.musicxml").write_text("<score/>\n", encoding="utf-8")
    (path / "source.wav").write_bytes(b"RIFFaudio")
    (path / "working.wav").write_bytes(b"RIFFaudio")
    diagnostics = path / "diagnostics"
    diagnostics.mkdir()
    (diagnostics / "overview.png").write_bytes(b"png")
    (diagnostics / "comparison.wav").write_bytes(b"RIFFmix")


def test_record_copies_score_formats_and_leaves_audio(tmp_path):
    project = tmp_path / "recital"
    _score_project(project, origin="https://youtu.be/abcdefghijk")
    store = tmp_path / "store"
    first = record_parse(
        action="transcribe",
        status="ok",
        source="https://youtu.be/abcdefghijk",
        title="Piece",
        project="recital",
        project_dir=project,
        excerpt_start=10,
        excerpt_end=20,
        duration=10,
        full_duration=30,
        sample_rate=16000,
        store=store,
    )
    second = record_parse(
        action="quantize",
        status="ok",
        source="https://youtu.be/abcdefghijk",
        title="Piece",
        project="recital",
        project_dir=project,
        store=store,
    )
    assert first is not None and second is not None
    assert first != second
    event = json.loads((first / "event.json").read_text(encoding="utf-8"))
    assert event["version"] == 1
    assert event["source"]["video_id"] == "abcdefghijk"
    assert event["formats"]["notes_json"]["stored"] is True
    assert event["formats"]["musicxml"]["stored"] is True
    assert event["formats"]["score_pdf"] == {
        "file": "score.pdf",
        "present": False,
        "status": "planned",
        "stored": False,
    }
    assert event["audio"]["source_wav"]["stored"] is False
    assert event["audio"]["diagnostics"] == ["comparison.wav", "overview.png"]
    for name in ("notes.json", "pedal.json", "raw.mid", "quantized.mid", "score.musicxml"):
        assert (first / name).is_file()
    assert not (first / "source.wav").exists()
    assert not (first / "working.wav").exists()
    assert not (first / "comparison.wav").exists()
    assert not (first / "overview.png").exists()
    assert not (first / "score.pdf").exists()
    assert len(list((store / "events").iterdir())) == 2
    loaded = load_events(store)
    assert {item["id"] for item in loaded} == {first.name, second.name}
    assert loaded == sorted(loaded, key=lambda item: (item["recorded_at"], item["id"]))


def test_local_paths_and_home_directories_stay_out_of_the_event(tmp_path):
    origin = tmp_path / "private" / "recital.flac"
    described = describe_source(str(origin), title="Recital", youtube_url="https://youtu.be/abcdefghijk?t=12")
    assert described["kind"] == "youtube"
    assert described["video_id"] == "abcdefghijk"
    assert described["local_name"] == "recital.flac"
    assert "private" not in json.dumps(described)

    store = tmp_path / "store"
    message = f"Could not read {Path.home() / 'Cookies'}"
    folder = record_parse(
        action="prepare",
        status="error",
        source=str(origin),
        project="recital",
        error=message,
        store=store,
    )
    assert folder is not None
    text = (folder / "event.json").read_text(encoding="utf-8")
    assert str(Path.home()) not in text
    assert "~" in text
    assert "recital.flac" in text
    assert "private" not in text
    assert not (folder / "notes.json").exists()


def test_import_is_idempotent_and_uses_an_attached_youtube_url(tmp_path):
    projects = tmp_path / "transcriptions"
    local_origin = tmp_path / "Users" / "someone" / "source.wav"
    _score_project(
        projects / "from-file",
        origin=str(local_origin),
        youtube_url="https://www.youtube.com/watch?v=Anx1SFCbGow",
    )
    _score_project(projects / "from-url", origin="https://youtu.be/bSfPnEfsEMY")
    store = tmp_path / "store"
    added, skipped = import_projects(projects, store)
    assert (added, skipped) == (2, 0)
    added, skipped = import_projects(projects, store)
    assert (added, skipped) == (0, 2)
    events = load_events(store)
    assert len(events) == 2
    attached = next(event for event in events if event["project"] == "from-file")
    assert attached["via"] == "import"
    assert attached["source"]["video_id"] == "Anx1SFCbGow"
    assert attached["source"]["url"].startswith("https://www.youtube.com/watch?v=")
    serialized = json.dumps(attached)
    assert "someone" not in serialized
    assert str(local_origin) not in serialized
    event_dir = store / "events" / attached["id"]
    copied_notes = (event_dir / "notes.json").read_text(encoding="utf-8")
    assert (event_dir / "score.musicxml").is_file()
    assert not (event_dir / "source.wav").exists()
    assert str(local_origin) not in copied_notes
    assert '"source": "source.wav"' in copied_notes
    assert '"pitch": 60' in copied_notes


def test_prepare_records_success_and_failure(tmp_path, monkeypatch):
    store = tmp_path / "store"
    output = tmp_path / "project"

    monkeypatch.setattr(
        "baroque_transcribe.pipeline.prepare_audio",
        lambda source, output_dir, **kwargs: _prepared(source, output_dir),
    )
    prepared = prepare_only(str(tmp_path / "phrase.wav"), output, "0.1", "0.6", history_store=store)
    assert prepared.working_wav.is_file()
    event = next(item for item in load_events(store) if item["status"] == "ok")
    assert event["action"] == "prepare"
    assert event["source"] == {
        "extension": ".wav",
        "kind": "file",
        "name": "phrase.wav",
        "title": "phrase",
    }
    assert event["formats"]["score_pdf"]["present"] is False
    event_dir = store / "events" / event["id"]
    assert (event_dir / "notes.json").is_file()
    assert (event_dir / "score.musicxml").is_file()
    assert not (event_dir / "source.wav").exists()
    assert str(tmp_path) not in json.dumps(event)

    def fail(source, output_dir, **kwargs):
        raise AudioError(f"Audio file not found: {Path.home() / 'secret' / 'missing.wav'}")

    monkeypatch.setattr("baroque_transcribe.pipeline.prepare_audio", fail)
    with pytest.raises(AudioError):
        prepare_only(str(tmp_path / "missing.wav"), tmp_path / "other", None, None, history_store=store)
    failure = next(item for item in load_events(store) if item["status"] == "error")
    assert failure["action"] == "prepare"
    assert str(Path.home()) not in json.dumps(failure)


def test_store_write_error_does_not_hide_the_prepared_audio(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "baroque_transcribe.pipeline.prepare_audio",
        lambda source, output_dir, **kwargs: _prepared(source, output_dir),
    )
    blocker = tmp_path / "not-a-directory"
    blocker.write_text("blocked", encoding="utf-8")
    prepared = prepare_only(str(tmp_path / "phrase.wav"), tmp_path / "out", "0.1", "0.4", history_store=blocker)
    assert prepared.working_wav.is_file()
    assert (tmp_path / "out" / "manifest.json").is_file()
    captured = capsys.readouterr()
    assert "could not write parse history" in captured.err


def test_default_store_is_not_written_during_pytest(tmp_path):
    before = set(default_store().glob("events/*/event.json")) if default_store().exists() else set()
    written = record_parse(
        action="prepare",
        status="ok",
        source="https://youtu.be/abcdefghijk",
        project="recital",
        store=None,
    )
    after = set(default_store().glob("events/*/event.json")) if default_store().exists() else set()
    assert written is None
    assert after == before


def test_history_command_rejects_a_missing_import_directory(tmp_path):
    assert main(["history", "import", str(tmp_path / "missing")]) == 1
    assert main(["history"]) == 0


_SCORE_XML = """<?xml version="1.0"?>
<score-partwise><part><measure>
<attributes><time><beats>3</beats><beat-type>4</beat-type></time></attributes>
<direction><sound tempo="84"/></direction>
</measure></part></score-partwise>
"""


def _scored_project(path: Path, *, note_count: int = 2) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    notes = [
        {"pitch": 60 + index, "onset": float(index), "offset": float(index) + 0.5, "velocity": 80}
        for index in range(note_count)
    ]
    (path / "notes.json").write_text(
        json.dumps({"version": 1, "model": "bytedance-note-pedal", "source": "https://youtu.be/abcdefghijk", "notes": notes, "pedals": [{"onset": 0.0, "offset": 1.0}]}),
        encoding="utf-8",
    )
    (path / "pedal.json").write_text("{}\n", encoding="utf-8")
    (path / "raw.mid").write_bytes(b"mid")
    (path / "quantized.mid").write_bytes(b"mid")
    (path / "score.musicxml").write_text(_SCORE_XML, encoding="utf-8")
    return path


def test_restore_rebuilds_a_project_without_downloading(tmp_path):
    from baroque_transcribe.parse_store import restore_projects

    store = tmp_path / "store"
    record_parse(
        action="transcribe",
        status="ok",
        source="https://youtu.be/abcdefghijk",
        title="Toccata",
        project="take",
        project_dir=_scored_project(tmp_path / "original"),
        store=store,
        excerpt_start=12.0,
        excerpt_end=42.0,
        duration=30.0,
        full_duration=212.0,
        recorded_at=_when("2026-11-08T04:21:03Z"),
    )
    # A later run of the same project is the one restored.
    record_parse(
        action="quantize",
        status="ok",
        source="https://youtu.be/abcdefghijk",
        title="Toccata",
        project="take",
        project_dir=_scored_project(tmp_path / "edited", note_count=3),
        store=store,
        excerpt_start=12.0,
        excerpt_end=42.0,
        duration=30.0,
        full_duration=212.0,
        recorded_at=_when("2026-11-08T05:21:03Z"),
    )
    root = tmp_path / "transcriptions"
    restored, skipped = restore_projects(store=store, projects_root=root)
    assert skipped == []
    assert restored == [root / "take"]
    dest = root / "take"
    for name in ("notes.json", "pedal.json", "raw.mid", "quantized.mid", "score.musicxml", "manifest.json"):
        assert (dest / name).is_file(), name
    assert not (dest / "working.wav").exists()
    assert not (dest / "source.wav").exists()
    manifest = json.loads((dest / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["counts"] == {"notes": 3, "pedals": 1}
    assert manifest["model"] == "bytedance-note-pedal"
    assert manifest["youtube_url"] == "https://youtu.be/abcdefghijk"
    assert manifest["tempo"]["bpm"] == 84.0
    assert manifest["tempo"]["time_signature"] == "3/4"
    assert manifest["audio"]["excerpt_start_seconds"] == 12.0
    assert manifest["restored_from"]["machine"] == manifest["restored_from"]["machine"]

    # Restoring again leaves the local project alone.
    (dest / "manifest.json").write_text('{"title": "Local edit"}\n', encoding="utf-8")
    restored, skipped = restore_projects(store=store, projects_root=root)
    assert restored == []
    assert skipped == ["take"]
    assert json.loads((dest / "manifest.json").read_text(encoding="utf-8"))["title"] == "Local edit"

    restored, skipped = restore_projects(store=store, projects_root=root, projects=["missing"])
    assert restored == []
    assert skipped == ["missing"]


def test_restore_with_audio_fetches_only_the_excerpt(tmp_path, monkeypatch):
    from baroque_transcribe import parse_store

    store = tmp_path / "store"
    record_parse(
        action="transcribe",
        status="ok",
        source="https://youtu.be/abcdefghijk",
        title="Toccata",
        project="take",
        project_dir=_scored_project(tmp_path / "original"),
        store=store,
        excerpt_start=12.0,
        excerpt_end=42.0,
        duration=30.0,
        full_duration=212.0,
    )
    calls = {}

    def fake_prepare(source, output_dir, **kwargs):
        calls.update(source=source, **kwargs)
        (output_dir / "working.wav").write_bytes(b"wav")

    monkeypatch.setattr("baroque_transcribe.audio.prepare_audio", fake_prepare)
    restored, _skipped = parse_store.restore_projects(
        store=store, projects_root=tmp_path / "transcriptions", with_audio=True
    )
    assert calls == {"source": "https://youtu.be/abcdefghijk", "start": 12.0, "end": 42.0}
    assert (restored[0] / "working.wav").is_file()
    manifest = json.loads((restored[0] / "manifest.json").read_text(encoding="utf-8"))
    assert any("no model was run" in warning for warning in manifest["warnings"])


def test_transcribed_videos_are_one_card_per_youtube_video(tmp_path):
    store = tmp_path / "store"
    record_parse(
        action="prepare",
        status="ok",
        source="https://youtu.be/abcdefghijk",
        title="Not transcribed yet",
        project="download-only",
        store=store,
    )
    record_parse(
        action="transcribe",
        status="error",
        source="https://youtu.be/zzzzzzzzzzz",
        title="Failed",
        project="failed",
        error="blocked",
        store=store,
    )
    record_parse(
        action="transcribe",
        status="ok",
        source="https://youtu.be/abcdefghijk",
        title="Baroque Improvisation Tip No. 1",
        project="excerpt",
        project_dir=_notes_dir(tmp_path / "excerpt"),
        store=store,
        recorded_at=_when("2026-11-07T20:35:15Z"),
    )
    record_parse(
        action="transcribe",
        status="ok",
        source="https://www.youtube.com/watch?v=abcdefghijk",
        title="source",
        project="full-take",
        project_dir=_notes_dir(tmp_path / "full-take"),
        store=store,
        recorded_at=_when("2026-11-08T04:21:03Z"),
    )
    record_parse(
        action="transcribe",
        status="ok",
        source="https://youtu.be/bSfPnEfsEMY",
        title="Improvised baroque toccata",
        project="toccata",
        project_dir=_notes_dir(tmp_path / "toccata"),
        store=store,
        recorded_at=_when("2026-11-08T04:49:18Z"),
    )
    cards = transcribed_videos(store)
    assert [card["video_id"] for card in cards] == ["bSfPnEfsEMY", "abcdefghijk"]
    shared = cards[1]
    assert shared["title"] == "Baroque Improvisation Tip No. 1"
    assert shared["projects"] == ["excerpt", "full-take"]
    assert shared["thumbnail"] == thumbnail_url("abcdefghijk")
    assert "hqdefault.jpg" in shared["thumbnail"]


def test_published_analysis_restores_and_fills_a_missing_file(tmp_path):
    from baroque_transcribe.parse_store import publish_analysis, restore_projects

    store = tmp_path / "store"
    project = _scored_project(tmp_path / "take")
    (project / "manifest.json").write_text(
        json.dumps(
            {
                "origin": "https://youtu.be/abcdefghijk",
                "title": "Toccata",
                "youtube_url": "https://youtu.be/abcdefghijk",
                "audio": {
                    "excerpt_start_seconds": 12.0,
                    "excerpt_end_seconds": 42.0,
                    "working_duration_seconds": 30.0,
                    "full_duration_seconds": 212.0,
                    "sample_rate": 16000,
                },
            }
        ),
        encoding="utf-8",
    )
    record_parse(
        action="transcribe",
        status="ok",
        source="https://youtu.be/abcdefghijk",
        title="Toccata",
        project="take",
        project_dir=project,
        store=store,
        excerpt_start=12.0,
        excerpt_end=42.0,
        duration=30.0,
        full_duration=212.0,
        recorded_at=_when("2026-11-08T04:21:03Z"),
    )
    (project / "annotations.json").write_text('{"version": 1, "key": {"tonic": "G", "mode": "minor"}}\n', encoding="utf-8")
    published = publish_analysis(project, store=store)
    assert published is not None
    event = json.loads((published / "event.json").read_text(encoding="utf-8"))
    assert event["action"] == "analyze"
    assert event["formats"]["annotations_json"]["stored"] is True
    assert event["formats"]["notes_json"]["stored"] is False
    assert (published / "annotations.json").is_file()
    assert not (published / "notes.json").exists()

    root = tmp_path / "transcriptions"
    restored, skipped = restore_projects(store=store, projects_root=root)
    assert skipped == []
    assert restored == [root / "take"]
    assert "G" in (root / "take" / "annotations.json").read_text(encoding="utf-8")
    assert (root / "take" / "notes.json").is_file()

    (root / "take" / "annotations.json").unlink()
    (root / "take" / "notes.json").write_text('{"notes": [{"pitch": 60}]}\n', encoding="utf-8")
    restored, skipped = restore_projects(store=store, projects_root=root)
    assert restored == [root / "take"]
    assert skipped == []
    assert "G" in (root / "take" / "annotations.json").read_text(encoding="utf-8")
    assert json.loads((root / "take" / "notes.json").read_text(encoding="utf-8"))["notes"][0]["pitch"] == 60

    (root / "take" / "annotations.json").write_text('{"version": 1, "kept": true}\n', encoding="utf-8")
    restored, skipped = restore_projects(store=store, projects_root=root)
    assert restored == []
    assert skipped == ["take"]
    assert json.loads((root / "take" / "annotations.json").read_text(encoding="utf-8"))["kept"] is True


def test_an_older_analysis_is_not_applied_to_a_newer_score(tmp_path):
    from baroque_transcribe.parse_store import restore_projects

    store = tmp_path / "store"
    early = _scored_project(tmp_path / "early")
    (early / "annotations.json").write_text('{"label": "old"}\n', encoding="utf-8")
    record_parse(
        action="transcribe",
        status="ok",
        source="https://youtu.be/abcdefghijk",
        title="Toccata",
        project="take",
        project_dir=early,
        store=store,
        recorded_at=_when("2026-11-08T04:21:03Z"),
    )
    later = _scored_project(tmp_path / "later", note_count=4)
    record_parse(
        action="quantize",
        status="ok",
        source="https://youtu.be/abcdefghijk",
        title="Toccata",
        project="take",
        project_dir=later,
        store=store,
        recorded_at=_when("2026-11-08T05:21:03Z"),
    )
    restored, skipped = restore_projects(store=store, projects_root=tmp_path / "transcriptions")
    assert skipped == []
    dest = restored[0]
    notes = json.loads((dest / "notes.json").read_text(encoding="utf-8"))
    assert len(notes["notes"]) == 4
    assert not (dest / "annotations.json").exists()


def _notes_dir(path: Path) -> Path:
    path.mkdir(parents=True)
    (path / "notes.json").write_text("{}\n", encoding="utf-8")
    return path


def _when(text: str):
    from datetime import datetime

    return datetime.fromisoformat(text.replace("Z", "+00:00"))
