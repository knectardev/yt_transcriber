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


def test_download_section_only_for_a_real_excerpt():
    from baroque_transcribe.audio import AudioError, _download_section

    assert _download_section(None, None) is None
    assert _download_section(0, None) is None
    assert _download_section(0, 20) == "*0.000000-20.000000"
    assert _download_section(65, None) == "*65.000000-inf"
    assert _download_section(65, 125.5) == "*65.000000-125.500000"
    for start, end in ((20, 20), (20, 10), (-1, 10)):
        try:
            _download_section(start, end)
        except AudioError:
            pass
        else:
            raise AssertionError(f"_download_section({start}, {end}) should have raised")


def test_youtube_info_parses_title_and_duration(monkeypatch):
    from baroque_transcribe import audio

    monkeypatch.setattr(audio, "_yt_dlp", lambda *args: "Some Title\n212.5")
    assert audio._youtube_info("https://youtu.be/x") == ("Some Title", 212.5)
    monkeypatch.setattr(audio, "_yt_dlp", lambda *args: "Some Title\nNA")
    assert audio._youtube_info("https://youtu.be/x") == ("Some Title", None)


def _fake_download(calls: list, dest: Path, *, fail_on_section: bool = False):
    from baroque_transcribe.audio import AudioError

    def fake_yt_dlp(*args):
        calls.append(args)
        if fail_on_section and "--download-sections" in args:
            raise AudioError("no ranges")
        dest.write_bytes(b"wav")
        return ""

    return fake_yt_dlp


def test_download_youtube_once_requests_only_the_section(tmp_path, monkeypatch):
    from baroque_transcribe import audio

    dest = tmp_path / "source.wav"
    calls: list = []
    monkeypatch.setattr(audio, "_youtube_info", lambda *a, **k: ("Title", 600.0))
    monkeypatch.setattr(audio, "_yt_dlp", _fake_download(calls, dest))
    result = audio._download_youtube_once(
        "https://youtu.be/x", dest, cookies=None, cookies_from_browser=None, section="*0.000000-20.000000"
    )
    assert result.title == "Title"
    assert result.full_duration == 600.0
    assert result.excerpt_only is True
    assert len(calls) == 1
    args = calls[0]
    assert args[args.index("--download-sections") + 1] == "*0.000000-20.000000"
    assert "--force-keyframes-at-cuts" in args


def test_download_youtube_once_falls_back_to_the_full_download(tmp_path, monkeypatch):
    from baroque_transcribe import audio

    dest = tmp_path / "source.wav"
    calls: list = []
    monkeypatch.setattr(audio, "_youtube_info", lambda *a, **k: ("Title", 600.0))
    monkeypatch.setattr(audio, "_yt_dlp", _fake_download(calls, dest, fail_on_section=True))
    result = audio._download_youtube_once(
        "https://youtu.be/x", dest, cookies=None, cookies_from_browser=None, section="*0.000000-20.000000"
    )
    assert result.excerpt_only is False
    assert len(calls) == 2
    assert "--download-sections" in calls[0]
    assert "--download-sections" not in calls[1]


def test_prepare_youtube_excerpt_is_not_trimmed_twice(tmp_path, monkeypatch):
    from baroque_transcribe import audio

    def fake_download(source, dest, **kwargs):
        assert kwargs["start"] == 0.5
        assert kwargs["end"] == 1.25
        # The section download already holds only 0.5s-1.25s of the video.
        section = tmp_path / "section.wav"
        _write_stereo(section)
        audio_data, rate = sf.read(section)
        sf.write(dest, audio_data[: int(rate * 0.75)], rate)
        return audio.DownloadedAudio(title="Video", full_duration=2.0, excerpt_only=True)

    monkeypatch.setattr(audio, "download_youtube", fake_download)
    prepared = prepare_audio("https://youtu.be/x", tmp_path / "out", start=0.5, end=1.25)
    assert prepared.full_duration == 2.0
    assert prepared.start == 0.5
    assert abs(prepared.duration - 0.75) < 0.05
    assert abs(prepared.end - 1.25) < 0.05


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
