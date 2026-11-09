import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from baroque_transcribe.audio import parse_timestamp, prepare_audio, slugify


def test_parse_timestamp_accepts_clock_times():
    assert parse_timestamp("90") == 90
    assert parse_timestamp("1:30") == 90
    assert parse_timestamp("1:02:03") == 3723
    assert parse_timestamp(None) is None
    assert parse_timestamp("") is None


def test_youtube_sign_in_message_is_recognized():
    from baroque_transcribe.audio import (
        _browser_cookie_attempts,
        _cookie_database_unreadable,
        _js_runtime_args,
        _missing_browser_cookies,
        _youtube_sign_in_required,
    )

    assert _youtube_sign_in_required("Sign in to confirm you're not a bot")
    assert not _youtube_sign_in_required("Video unavailable")
    assert _missing_browser_cookies("could not find chrome cookies database")
    assert _cookie_database_unreadable("Could not copy Chrome cookie database")
    assert _cookie_database_unreadable("could not find chrome cookies database")
    assert not _cookie_database_unreadable("Video unavailable")
    attempts = _browser_cookie_attempts()
    assert attempts[0] is None
    assert "chrome" in attempts
    assert all("Guest Profile" not in (item or "") for item in attempts)
    runtimes = _js_runtime_args()
    if runtimes:
        assert runtimes[0] == "--js-runtimes"
        _name, _, binary = runtimes[1].partition(":")
        assert Path(binary).is_file()


def test_slugify_collapses_punctuation():
    assert slugify("Baroque Improvisation!") == "baroque_improvisation"


def test_tool_path_falls_back_when_path_is_stale(tmp_path, monkeypatch):
    # A server started before FFmpeg was installed keeps the old PATH.
    # The tool must still be found, and its folder added to PATH so
    # child processes (yt-dlp) can find it too.
    from baroque_transcribe import audio

    fake = tmp_path / "ffmpeg.exe"
    fake.write_bytes(b"")
    monkeypatch.setattr(audio, "_TOOL_PATHS", {})
    monkeypatch.setattr(audio.shutil, "which", lambda _name: None)
    monkeypatch.setattr(audio, "_tool_path_windows", lambda _name: str(fake))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert audio.tool_path("ffmpeg") == str(fake)
    assert str(tmp_path) in audio.os.environ["PATH"].split(audio.os.pathsep)


def test_require_ffmpeg_names_the_missing_tools(monkeypatch):
    from baroque_transcribe import audio

    monkeypatch.setattr(audio, "tool_path", lambda _name: None)
    try:
        audio.require_ffmpeg()
    except audio.AudioError as exc:
        assert "ffmpeg and ffprobe were not found" in str(exc)
    else:
        raise AssertionError("require_ffmpeg should have raised")


def _write_stereo(path: Path) -> None:
    sample_rate = 44100
    times = np.linspace(0, 2.0, sample_rate * 2, endpoint=False)
    tone = (0.2 * np.sin(2 * np.pi * 440 * times)).astype(np.float32)
    sf.write(path, np.stack([tone, tone * 0.5], axis=1), sample_rate)


def test_prepare_trims_without_touching_the_original(tmp_path):
    original = tmp_path / "recital.wav"
    _write_stereo(original)
    digest = hashlib.sha256(original.read_bytes()).hexdigest()
    prepared = prepare_audio(str(original), tmp_path / "out", start=0.5, end=1.25)
    assert hashlib.sha256(original.read_bytes()).hexdigest() == digest
    audio, sample_rate = sf.read(prepared.working_wav)
    assert sample_rate == 16000
    assert audio.ndim == 1
    assert 0.7 < len(audio) / sample_rate < 0.8
    assert abs(float(np.max(np.abs(audio))) - 0.95) < 0.02
    source, source_rate = sf.read(prepared.source_wav)
    assert source_rate == 44100
    assert source.ndim == 2
