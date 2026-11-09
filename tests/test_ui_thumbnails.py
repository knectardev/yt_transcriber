"""Home-page card thumbnails: fetched once, then read from the cache."""

import base64
from pathlib import Path

from baroque_transcribe.library import Card
from baroque_transcribe.ui import _with_local_thumbnails


def _card(video_id: str | None) -> Card:
    return Card(key=video_id or "file:x", title="Piece", video_id=video_id, url=None, thumbnail="https://example.invalid/remote.jpg")


def test_thumbnail_comes_from_the_cache_loader(monkeypatch):
    import baroque_transcribe.ui as ui

    monkeypatch.setattr(ui, "load_thumbnail", lambda video_id: b"jpeg-bytes")
    cards = _with_local_thumbnails([_card("abcdefghijk")])
    expected = "data:image/jpeg;base64," + base64.b64encode(b"jpeg-bytes").decode("ascii")
    assert cards[0].thumbnail == expected


def test_card_without_a_video_or_a_fetchable_thumbnail_has_no_image(monkeypatch):
    import baroque_transcribe.ui as ui

    monkeypatch.setattr(ui, "load_thumbnail", lambda video_id: None)
    # The committed static fallback has no file for this id, so the card
    # keeps no thumbnail rather than pointing at a remote URL.
    cards = _with_local_thumbnails([_card("abcdefghijk"), _card(None)])
    assert cards[0].thumbnail is None
    assert cards[1].thumbnail is None
