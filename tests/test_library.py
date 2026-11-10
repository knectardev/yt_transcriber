"""Home-page cards built from the parse history and the folders on disk."""

import json
from datetime import datetime, timezone
from pathlib import Path

from baroque_transcribe.library import (
    Run,
    card_for,
    find_project,
    format_clock,
    library_cards,
    project_param,
    read_run,
    run_facts,
)
from baroque_transcribe.parse_store import record_parse


def _project(folder: Path, *, origin: str, title: str = "Piece", notes: int = 3, youtube_url: str | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    manifest = {
        "title": title,
        "origin": origin,
        "model": "bytedance-note-pedal",
        "counts": {"notes": notes, "pedals": 1},
        "tempo": {"bpm": 72, "time_signature": "3/4", "subdivision": 16},
        "audio": {"excerpt_start_seconds": 12.0, "excerpt_end_seconds": 42.0, "working_duration_seconds": 30.0},
    }
    if youtube_url:
        manifest["youtube_url"] = youtube_url
    (folder / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (folder / "notes.json").write_text('{"notes": []}', encoding="utf-8")
    (folder / "score.musicxml").write_text("<score/>", encoding="utf-8")
    return folder


def test_read_run_describes_the_folder(tmp_path):
    folder = _project(tmp_path / "take", origin="https://youtu.be/abcdefghijk", title="source")
    run = read_run(folder)
    assert run is not None
    assert run.available is True
    assert run.title == ""  # "source" is only the audio file name
    assert run.video_id == "abcdefghijk"
    assert run.notes == 3
    assert run.bpm == 72
    assert run.time_signature == "3/4"
    assert run.duration == 30.0
    assert run.has_score is True
    assert run.has_audio is False
    assert isinstance(run.modified, datetime)
    assert read_run(tmp_path / "missing") is None


def test_run_facts_are_the_header_chips_in_reading_order(tmp_path):
    folder = _project(tmp_path / "take", origin="https://youtu.be/abcdefghijk", notes=1234)
    run = read_run(folder)
    assert run is not None
    facts = run_facts(run)
    assert ["".join((fact["before"], fact["value"], fact["after"])) for fact in facts] == [
        "1,234 notes",
        "1 pedal events",
        "0:30 excerpt",
        "from 0:12 in the video",
        "72 BPM",
        "3/4",
        "bytedance-note-pedal",
        run.modified.strftime("%b %d, %Y %H:%M"),
    ]
    # The number is the emphasised part. The model and the date have none.
    assert [fact["value"] for fact in facts] == ["1,234", "1", "0:30", "0:12", "72", "3/4", "", ""]


def test_run_facts_leave_out_what_the_manifest_does_not_say(tmp_path):
    assert run_facts(Run(name="bare", folder=tmp_path, available=True)) == []
    # An excerpt that starts at the top of the video says nothing about where it starts.
    facts = run_facts(Run(name="top", folder=tmp_path, available=True, notes=0, excerpt_start=0.0, bpm=89.29))
    assert facts == [
        {"before": "", "value": "0", "after": " notes"},
        {"before": "", "value": "89.29", "after": " BPM"},
    ]


def test_clock_text_grows_an_hour_field():
    assert format_clock(0) == "0:00"
    assert format_clock(130.4) == "2:10"
    assert format_clock(3725) == "1:02:05"


def test_cards_merge_history_with_folders_on_disk(tmp_path):
    root = tmp_path / "transcriptions"
    store = tmp_path / "store"
    _project(root / "excerpt", origin="https://youtu.be/abcdefghijk", title="source")
    _project(root / "full", origin=r"C:\audio\source.wav", title="source", youtube_url="https://youtu.be/abcdefghijk")
    _project(root / "recital", origin=r"C:\audio\recital.flac", title="Recital")
    _project(root / "unlisted", origin="https://youtu.be/zzzzzzzzzzz", title="Not in the history yet")
    # The generic title comes first in the history; the real title must still win.
    record_parse(
        action="transcribe",
        status="ok",
        source="https://youtu.be/abcdefghijk",
        title="source",
        project="elsewhere",
        project_dir=root / "excerpt",
        store=store,
        recorded_at=datetime(2026, 11, 7, 20, 0, 0, tzinfo=timezone.utc),
    )
    record_parse(
        action="transcribe",
        status="ok",
        source="https://youtu.be/abcdefghijk",
        title="Baroque Improvisation Tip No. 1",
        project="excerpt",
        project_dir=root / "excerpt",
        store=store,
        recorded_at=datetime(2026, 11, 8, 4, 0, 0, tzinfo=timezone.utc),
    )
    cards = {card.key: card for card in library_cards(root, store)}
    assert set(cards) == {"abcdefghijk", "zzzzzzzzzzz", "file:recital"}

    shared = cards["abcdefghijk"]
    assert shared.title == "Baroque Improvisation Tip No. 1"
    assert shared.thumbnail and "abcdefghijk" in shared.thumbnail
    assert {run.name for run in shared.runs} == {"excerpt", "elsewhere", "full"}
    assert {run.name for run in shared.runs if run.available} == {"excerpt", "full"}
    assert shared.primary is not None and shared.primary.available

    assert cards["zzzzzzzzzzz"].title == "Not in the history yet"
    local = cards["file:recital"]
    assert local.thumbnail is None
    assert local.video_id is None
    assert [run.name for run in local.runs] == ["recital"]

    found = card_for(root / "full", root, store)
    assert found is not None and found.key == "abcdefghijk"
    assert card_for(tmp_path / "nowhere", root, store) is None


def test_grid_listing_does_not_open_the_transcription_files(tmp_path):
    root = tmp_path / "transcriptions"
    folder = _project(root / "take", origin="https://youtu.be/abcdefghijk", title="Toccata", notes=9)
    (folder / "working.wav").write_bytes(b"not-read")
    cards = library_cards(root, tmp_path / "store", for_grid=True)
    assert len(cards) == 1
    run = cards[0].primary
    assert run is not None
    assert run.notes == 9
    assert run.duration == 30.0
    assert run.has_audio is False
    assert run.has_score is False
    assert run.warnings == []


def test_thumbnail_is_downloaded_once(tmp_path, monkeypatch):
    from baroque_transcribe.library import load_thumbnail

    calls = {"count": 0}

    class _Response:
        def read(self, _limit):
            return b"jpeg-bytes"

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fetch(_request, timeout=0):
        calls["count"] += 1
        assert timeout == 4
        return _Response()

    monkeypatch.setattr("urllib.request.urlopen", fetch)
    first = load_thumbnail("abcdefghijk", cache_dir=tmp_path)
    second = load_thumbnail("abcdefghijk", cache_dir=tmp_path)
    assert first == second == b"jpeg-bytes"
    assert calls["count"] == 1
    assert load_thumbnail("short", cache_dir=tmp_path) is None


def test_find_project_accepts_a_name_or_a_path(tmp_path):
    root = tmp_path / "transcriptions"
    folder = _project(root / "piece", origin="https://youtu.be/abcdefghijk")
    assert find_project("piece", root) == folder.resolve()
    assert find_project(str(folder), root) == folder.resolve()
    assert find_project("missing", root) is None
    assert find_project("", root) is None
    assert find_project(None, root) is None
    assert project_param(folder, root) == "piece"
    elsewhere = _project(tmp_path / "elsewhere" / "take", origin="https://youtu.be/abcdefghijk")
    assert project_param(elsewhere, root) == str(elsewhere)
    assert find_project(project_param(elsewhere, root), root) == elsewhere.resolve()
