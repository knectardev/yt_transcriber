"""The transcribe-page YouTube player markup."""

import pytest

from baroque_transcribe.ui import _youtube_excerpt_html


def test_excerpt_player_mentions_the_video_and_the_mark_buttons():
    html = _youtube_excerpt_html("abcdefghijk", 12.5)
    assert "abcdefghijk" in html
    assert "12.5" in html
    assert "Set start" in html
    assert "Set end" in html
    assert "Preview excerpt" in html
    assert ".st-key-" in html


def test_excerpt_player_rejects_a_video_id_that_could_break_the_page():
    with pytest.raises(ValueError):
        _youtube_excerpt_html("</script><script>", 0)


def test_excerpt_markup_ignores_the_start_field_so_edits_do_not_rebuild_the_player():
    """A later change to Start must not alter this string, or the video reloads."""

    html = _youtube_excerpt_html("abcdefghijk", 0)
    assert "st.session_state" not in html
    assert html == _youtube_excerpt_html("abcdefghijk", 0)
