"""YouTube id parsing, the playback page, and the localhost sync offset."""

import json
from urllib.request import Request, urlopen

import pytest

from baroque_transcribe.playback import (
    PlaybackServer,
    SyncError,
    load_playback,
    project_youtube_id,
    write_playback_page,
    write_sync_offset,
    write_youtube_url,
    youtube_embed_html,
    youtube_project_for_midi,
    youtube_video_id,
)
from baroque_transcribe.schema import SCHEMA_VERSION


def test_youtube_ids_from_watch_share_and_embed_urls():
    assert youtube_video_id("https://youtu.be/abcdefghijk") == "abcdefghijk"
    assert youtube_video_id("https://www.youtube.com/watch?v=abcdefghijk&t=12s") == "abcdefghijk"
    assert youtube_video_id("https://youtube.com/shorts/abcdefghijk") == "abcdefghijk"
    assert youtube_video_id("https://www.youtube.com/embed/abcdefghijk") == "abcdefghijk"
    assert youtube_video_id("https://www.youtube.com/live/abcdefghijk") == "abcdefghijk"
    assert youtube_video_id("not a url") is None
    assert youtube_video_id("https://youtu.be/short") is None


def test_embed_html_cues_the_excerpt_start():
    html = youtube_embed_html("abcdefghijk", 12.5)
    assert "abcdefghijk" in html
    assert "12.5" in html
    assert "cueVideoById" in html


def test_sync_offset_preserves_the_rest_of_the_manifest(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"title": "Piece", "origin": "https://youtu.be/abcdefghijk"}) + "\n", encoding="utf-8")
    assert write_sync_offset(tmp_path, 0.25) == pytest.approx(0.25)
    saved = json.loads(manifest.read_text(encoding="utf-8"))
    assert saved["title"] == "Piece"
    assert saved["sync_offset_seconds"] == pytest.approx(0.25)
    assert write_sync_offset(tmp_path, 80) == 60.0
    assert write_sync_offset(tmp_path, -80) == -60.0


def test_attached_youtube_url_overrides_a_local_origin(tmp_path):
    project = _write_min_project(tmp_path, origin=r"C:\recordings\piece.wav")
    assert project_youtube_id(json.loads((project / "manifest.json").read_text(encoding="utf-8"))) is None
    write_youtube_url(project, "https://youtu.be/abcdefghijk?t=12")
    saved = json.loads((project / "manifest.json").read_text(encoding="utf-8"))
    assert saved["origin"] == r"C:\recordings\piece.wav"
    assert saved["youtube_url"] == "https://youtu.be/abcdefghijk?t=12"
    assert project_youtube_id(saved) == "abcdefghijk"
    context = load_playback(project)
    assert context.video_id == "abcdefghijk"
    with pytest.raises(SyncError, match="not a YouTube URL"):
        write_youtube_url(project, "https://example.com/watch")


def test_youtube_address_is_saved_on_the_project_that_owns_the_midi(tmp_path):
    project = _write_min_project(tmp_path / "piece", origin=r"C:\recordings\piece.wav")
    midi = project / "quantized.mid"
    midi.write_bytes(b"MThd")
    assert youtube_project_for_midi(midi, tmp_path / "other") == project
    loose = tmp_path / "loose" / "take.mid"
    loose.parent.mkdir()
    loose.write_bytes(b"MThd")
    assert youtube_project_for_midi(loose, project) == project
    assert youtube_project_for_midi(loose, tmp_path / "empty") is None


def test_load_playback_rejects_a_project_without_a_video(tmp_path):
    _write_min_project(tmp_path, origin="C:/recordings/piece.wav")
    with pytest.raises(SyncError, match="no YouTube video"):
        load_playback(tmp_path)


def test_playback_page_records_the_estimate_limitation(tmp_path):
    context_dir = _write_min_project(tmp_path, origin="https://youtu.be/abcdefghijk")
    context = load_playback(context_dir)
    dest = write_playback_page(context, context_dir / "playback.html")
    text = dest.read_text(encoding="utf-8")
    assert "abcdefghijk" in text
    assert "place on the video clock" in text
    assert "Follow score" in text
    assert "Notation earlier" in text
    assert "seekTo" in text
    assert '"place": "(0:02)"' in text
    assert "stampMeasurePlaces" in text
    assert "getPlaybackRate" in text
    assert "onPlaybackRateChange" in text
    assert (context_dir / "opensheetmusicdisplay.min.js").is_file()
    assert context.anchors[0].measure == 1


def test_playback_page_carries_the_harmony_layer_and_toggle(tmp_path):
    project = _write_min_project(tmp_path, origin="https://youtu.be/abcdefghijk")
    annotations = {
        "version": 1,
        "key": {"tonic": "C", "mode": "major", "source": "manual", "confidence": 1.0},
        "grid": {},
        "notes_fingerprint": "x",
        "chords": [
            {
                "measure": 1, "beat": 0.0, "measure_position": 0.0, "start_step": 0, "end_step": 16,
                "root_pc": 0, "root_name": "C", "quality": "major", "bass_pc": 0, "bass_name": "C",
                "roman": "I", "symbol": "C", "inversion": 0, "confidence": 0.9,
                "edited": False, "edited_fields": [],
            }
        ],
        "sequences": [
            {
                "start_measure": 1, "end_measure": 2, "kind": "descending-fifth",
                "description": "Descending fifth sequence", "repetitions": 1, "confidence": 0.8,
                "edited": False,
            }
        ],
    }
    (project / "annotations.json").write_text(json.dumps(annotations), encoding="utf-8")
    context = load_playback(project)
    assert context.harmony is not None
    assert context.harmony["chords"][0]["roman"] == "I"
    assert context.harmony["chords"][0]["symbol"] == "C"
    text = write_playback_page(context, project / "playback.html").read_text(encoding="utf-8")
    assert 'id="show-harmony"' in text
    assert "Lead-sheet chord symbols" in text
    assert "drawHarmony" in text
    assert '"roman": "I"' in text
    assert "Descending fifth sequence" in text


def test_malformed_annotations_never_break_playback(tmp_path):
    project = _write_min_project(tmp_path, origin="https://youtu.be/abcdefghijk")
    (project / "annotations.json").write_text("{ not json", encoding="utf-8")
    context = load_playback(project)
    assert context.harmony is None
    text = write_playback_page(context, project / "playback.html").read_text(encoding="utf-8")
    assert "abcdefghijk" in text


def test_server_round_trips_the_offset_without_dropping_manifest_fields(tmp_path):
    project = _write_min_project(tmp_path, origin="https://youtu.be/abcdefghijk")
    context = load_playback(project)
    write_playback_page(context, project / "playback.html")
    server = PlaybackServer()
    server.start()
    try:
        token = server.register(project)
        page = server.page_url(token)
        with urlopen(page) as response:
            body = response.read()
        assert b"place on the video clock" in body
        request = Request(
            page + "sync-offset",
            data=json.dumps({"offset": -0.4}).encode("utf-8"),
            method="POST",
        )
        request.add_header("Content-Type", "application/json")
        with urlopen(request) as response:
            stored = json.loads(response.read().decode("utf-8"))
        assert stored["offset"] == pytest.approx(-0.4)
        saved = json.loads((project / "manifest.json").read_text(encoding="utf-8"))
        assert saved["title"] == "Piece"
        assert saved["sync_offset_seconds"] == pytest.approx(-0.4)
        with urlopen(page + "sync-offset") as response:
            assert json.loads(response.read().decode("utf-8"))["offset"] == pytest.approx(-0.4)
        with urlopen(page + "opensheetmusicdisplay.min.js") as response:
            assert response.status == 200
            assert len(response.read()) > 1000
        assert server.register(project) == token
    finally:
        server.close()


def test_playback_page_links_home_and_uses_the_given_title(tmp_path):
    project = _write_min_project(tmp_path, origin="https://youtu.be/abcdefghijk")
    context = load_playback(project, title="Toccata in the style of Bach")
    assert context.title == "Toccata in the style of Bach"
    text = write_playback_page(context, project / "playback.html", home_url="http://127.0.0.1:8765/").read_text(
        encoding="utf-8"
    )
    assert "Toccata in the style of Bach" in text
    assert "http://127.0.0.1:8765/" in text
    assert "embed" in text


def test_server_serves_a_thumbnail_without_a_project(monkeypatch):
    from urllib.error import HTTPError

    monkeypatch.setattr(
        "baroque_transcribe.library.load_thumbnail",
        lambda video_id: b"jpeg" if video_id == "abcdefghijk" else None,
    )
    server = PlaybackServer()
    server.start()
    try:
        with urlopen(f"http://127.0.0.1:{server.port}/thumb/abcdefghijk.jpg") as response:
            assert response.status == 200
            assert response.headers["Cache-Control"].startswith("public")
            assert response.read() == b"jpeg"
        with pytest.raises(HTTPError):
            urlopen(f"http://127.0.0.1:{server.port}/thumb/nope.jpg")
    finally:
        server.close()


def test_server_serves_the_static_notation_page(tmp_path, monkeypatch):
    import baroque_transcribe.playback as playback

    project = _write_min_project(tmp_path, origin="https://youtu.be/abcdefghijk")
    monkeypatch.setattr(playback, "write_notation_page", _fake_notation_page)
    server = PlaybackServer(home_url="http://127.0.0.1:8765/")
    server.start()
    monkeypatch.setattr(playback, "_SERVER", server)
    try:
        url = playback.notation_page_url(project, title="Piece")
        assert url.endswith("/notation.html")
        with urlopen(url) as response:
            body = response.read().decode("utf-8")
        assert "Piece" in body
        assert "http://127.0.0.1:8765/" in body
        with pytest.raises(SyncError, match="no score"):
            playback.notation_page_url(tmp_path / "empty")
    finally:
        server.close()
        monkeypatch.setattr(playback, "_SERVER", None)


def _fake_notation_page(score_path, dest, *, title=None, note=None, home_url=None):
    dest.write_text(f"<html><a href='{home_url}'>home</a><h1>{title}</h1></html>", encoding="utf-8")
    return dest


def _write_min_project(folder, *, origin: str):
    folder.mkdir(parents=True, exist_ok=True)
    notes = {
        "version": SCHEMA_VERSION,
        "model": "test",
        "source": origin,
        "notes": [{"pitch": 60, "onset": 0.0, "offset": 0.4, "velocity": 80}],
        "pedals": [],
    }
    (folder / "notes.json").write_text(json.dumps(notes), encoding="utf-8")
    (folder / "score.musicxml").write_text("<score/>", encoding="utf-8")
    manifest = {
        "title": "Piece",
        "origin": origin,
        "tempo": {
            "bpm": 120,
            "downbeat_seconds": 0,
            "subdivision": 16,
            "time_signature": "4/4",
        },
        "audio": {"excerpt_start_seconds": 1.5},
    }
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder
