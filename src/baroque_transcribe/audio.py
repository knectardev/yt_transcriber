"""Acquire a recording and make a working copy for the transcription model.

The file the user points at is never opened for writing. A YouTube download
is stored once -- only the requested section, when an excerpt is given --
then a trimmed mono 16 kHz copy is peak-normalized for the model. Times in
the MIDI files are relative to that working copy.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import soundfile as sf

from baroque_transcribe.defaults import MODEL_SAMPLE_RATE, WORKING_PEAK
from baroque_transcribe.progress import ProgressCallback, ProgressEvent, emit, fraction_for

LOCAL_EXTENSIONS = {".wav", ".mp3", ".flac"}


class AudioError(RuntimeError):
    pass


@dataclass
class PreparedAudio:
    source_wav: Path
    working_wav: Path
    sample_rate: int
    duration: float
    full_duration: float
    start: float
    end: float
    original_peak: float
    title: str
    origin: str


@dataclass
class DownloadedAudio:
    """Result of a YouTube download.

    ``excerpt_only`` is True when only the requested section was downloaded
    (``source.wav`` then holds that section, not the whole recording), in
    which case ``full_duration`` is the duration of the complete video as
    reported by yt-dlp, so callers can still report excerpt positions
    against the full recording.
    """

    title: str
    full_duration: float | None
    excerpt_only: bool


def parse_timestamp(value: str | float | int | None) -> float | None:
    """Accept seconds, ``M:SS``, or ``H:MM:SS``. Empty means "not set"."""

    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        seconds = float(value)
    else:
        text = str(value).strip()
        if text == "" or text.lower() in {"none", "start", "end"}:
            return None
        parts = text.split(":")
        try:
            if len(parts) == 1:
                seconds = float(parts[0])
            elif len(parts) == 2:
                seconds = int(parts[0]) * 60 + float(parts[1])
            elif len(parts) == 3:
                seconds = int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            else:
                raise ValueError
        except ValueError as exc:
            raise AudioError(
                f"Could not read time {value!r}. Use seconds, M:SS, or H:MM:SS."
            ) from exc
    if seconds < 0:
        raise AudioError("Times cannot be negative.")
    return seconds


def is_url(source: str) -> bool:
    parsed = urlparse(source.strip())
    return parsed.scheme in {"http", "https"}


def slugify(text: str, fallback: str = "transcription") -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()
    return slug[:80] or fallback


_TOOL_PATHS: dict[str, str] = {}


def tool_path(name: str) -> str | None:
    """Full path to an external tool, or None when it cannot be found.

    PATH is tried first. On Windows a long-running process (the review UI)
    keeps the PATH it was started with, so a tool installed afterwards -- or
    a terminal opened before the install -- looks missing even though it is
    on the machine. The registry PATH and the WinGet package folder are
    therefore checked too. A tool found off PATH has its folder added to
    PATH, so child processes that look the tool up themselves (yt-dlp
    calling ffmpeg) find it as well.
    """

    if name in _TOOL_PATHS:
        return _TOOL_PATHS[name]
    found = shutil.which(name)
    if found is None and sys.platform.startswith("win"):
        found = _tool_path_windows(name)
    if found is not None:
        _TOOL_PATHS[name] = found
        folder = str(Path(found).parent)
        if folder not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = folder + os.pathsep + os.environ.get("PATH", "")
    return found


def _tool_path_windows(name: str) -> str | None:
    for directory in _registry_path_dirs():
        candidate = Path(directory) / f"{name}.exe"
        if candidate.is_file():
            return str(candidate)
    local_app_data = os.environ.get("LOCALAPPDATA", "")
    if local_app_data:
        packages = Path(local_app_data) / "Microsoft" / "WinGet" / "Packages"
        try:
            candidates = sorted(packages.glob(f"Gyan.FFmpeg*/**/bin/{name}.exe"))
        except OSError:
            candidates = []
        if candidates:
            return str(candidates[-1])
    return None


def _registry_path_dirs() -> list[str]:
    """PATH folders from the Windows registry, which is current even when
    this process inherited a stale PATH from its parent terminal."""

    try:
        import winreg
    except ImportError:
        return []
    locations = (
        (winreg.HKEY_CURRENT_USER, "Environment"),
        (
            winreg.HKEY_LOCAL_MACHINE,
            r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment",
        ),
    )
    folders: list[str] = []
    for hive, key in locations:
        try:
            with winreg.OpenKey(hive, key) as handle:
                value, _kind = winreg.QueryValueEx(handle, "Path")
        except OSError:
            continue
        for folder in str(value).split(";"):
            folder = os.path.expandvars(folder.strip())
            if folder:
                folders.append(folder)
    return folders


def require_ffmpeg() -> None:
    if tool_path("ffmpeg") is None or tool_path("ffprobe") is None:
        raise AudioError(
            "ffmpeg and ffprobe were not found on this computer. "
            "On Windows, install FFmpeg and reopen the terminal. "
            "One option is: winget install Gyan.FFmpeg"
        )


def _run(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise AudioError(f"Command failed: {' '.join(command)}\n{detail}") from exc
    except FileNotFoundError as exc:
        raise AudioError(f"Command not found: {command[0]}") from exc
    return completed.stdout.strip()


def probe_duration(path: Path) -> float:
    raw = _run(
        [
            tool_path("ffprobe") or "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ]
    )
    try:
        return float(raw)
    except ValueError as exc:
        raise AudioError(f"ffprobe did not return a duration for {path}.") from exc


def _ffmpeg_wav(
    source: Path,
    dest: Path,
    *,
    start: float | None = None,
    end: float | None = None,
    sample_rate: int | None = None,
    mono: bool = False,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    command = [tool_path("ffmpeg") or "ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    if start is not None and start > 0:
        command += ["-ss", f"{start:.6f}"]
    command += ["-i", str(source)]
    if end is not None:
        duration = end - (start or 0.0)
        if duration <= 0:
            raise AudioError("The end time must be later than the start time.")
        command += ["-t", f"{duration:.6f}"]
    command.append("-vn")
    if mono:
        command += ["-ac", "1"]
    if sample_rate is not None:
        command += ["-ar", str(sample_rate)]
    command += ["-c:a", "pcm_s16le", str(dest)]
    _run(command)


# yt-dlp only enables Deno unless told otherwise. Node is the runtime most
# Windows machines already have, and YouTube extraction now expects one.
JS_RUNTIMES = ("deno", "node", "bun")
# Tried, in order, when YouTube asks for a sign-in and the user did not name a browser.
BROWSER_COOKIE_SOURCES = ("chrome", "edge", "firefox", "brave", "chromium")


def _yt_dlp(*arguments: str) -> str:
    return _run([sys.executable, "-m", "yt_dlp", *_js_runtime_args(), *arguments])


def _js_runtime_args() -> list[str]:
    for runtime in JS_RUNTIMES:
        found = shutil.which(runtime)
        if found:
            return ["--js-runtimes", f"{runtime}:{found}"]
    return []


def _youtube_auth_args(cookies: Path | None, cookies_from_browser: str | None) -> list[str]:
    arguments: list[str] = []
    if cookies is not None:
        arguments += ["--cookies", str(cookies)]
    if cookies_from_browser:
        arguments += ["--cookies-from-browser", cookies_from_browser]
    return arguments


def _youtube_sign_in_required(message: str) -> bool:
    text = message.lower()
    return "sign in" in text or "not a bot" in text or "confirm you" in text


def _missing_browser_cookies(message: str) -> bool:
    text = message.lower()
    return "could not find" in text and "cookie" in text


def _cookie_database_unreadable(message: str) -> bool:
    """True when a browser is installed but its cookie file cannot be read.

    yt-dlp reports a locked Chrome database as "Could not copy Chrome cookie
    database" (https://github.com/yt-dlp/yt-dlp/issues/7271). That is not the
    same text as a missing browser, and it is not a YouTube sign-in error.
    """

    text = message.lower()
    if _missing_browser_cookies(message):
        return True
    return "cookie" in text and ("could not copy" in text or "locked" in text)


def _chromium_user_data(browser: str) -> Path | None:
    local = Path.home() / "AppData" / "Local"
    roots = {
        "chrome": local / "Google" / "Chrome" / "User Data",
        "edge": local / "Microsoft" / "Edge" / "User Data",
        "brave": local / "BraveSoftware" / "Brave-Browser" / "User Data",
        "chromium": local / "Chromium" / "User Data",
    }
    root = roots.get(browser)
    if root is not None and root.is_dir():
        return root
    return None


def _extra_browser_profiles(browser: str) -> list[str]:
    """Other Chromium profiles, as ``browser:Profile 1`` arguments.

    ``--cookies-from-browser chrome`` reads only the Default profile. On a
    machine with several Chrome profiles, YouTube may be signed in on one
    whose cookie file is not locked.
    """

    root = _chromium_user_data(browser)
    if root is None:
        return []
    skip = {"default", "guest profile", "system profile"}
    profiles: list[str] = []
    for entry in sorted(root.iterdir(), key=lambda path: path.name.lower()):
        if not entry.is_dir() or entry.name.lower() in skip:
            continue
        if (entry / "Network" / "Cookies").is_file() or (entry / "Cookies").is_file():
            profiles.append(f"{browser}:{entry.name}")
    return profiles


def _browser_cookie_attempts() -> list[str | None]:
    attempts: list[str | None] = [None]
    for browser in BROWSER_COOKIE_SOURCES:
        attempts.append(browser)
        attempts.extend(_extra_browser_profiles(browser))
    return attempts


def _youtube_title(
    url: str,
    *,
    cookies: Path | None = None,
    cookies_from_browser: str | None = None,
) -> str:
    title, _duration = _youtube_info(url, cookies=cookies, cookies_from_browser=cookies_from_browser)
    return title


def _youtube_info(
    url: str,
    *,
    cookies: Path | None = None,
    cookies_from_browser: str | None = None,
) -> tuple[str, float | None]:
    """Video title and full duration in seconds, without downloading.

    The duration lets an excerpt-only download still report positions
    against the complete video. Either value is missing when the metadata
    request fails (for example before a sign-in retry succeeds).
    """

    try:
        raw = _yt_dlp(
            "--no-download",
            "--no-playlist",
            *_youtube_auth_args(cookies, cookies_from_browser),
            "--print",
            "title",
            "--print",
            "duration",
            url,
        )
    except AudioError:
        return "", None
    lines = raw.splitlines()
    title = lines[0].strip() if lines else ""
    duration: float | None = None
    if len(lines) > 1:
        try:
            duration = float(lines[1].strip())
        except ValueError:
            duration = None
    return title, duration


def _download_section(start: float | None, end: float | None) -> str | None:
    """yt-dlp ``--download-sections`` value for an excerpt, or None for the whole video.

    The value is a ``*start-end`` time range in seconds; ``inf`` leaves the
    end open. A range is only worth requesting when it actually excludes
    part of the video, which cannot be known for an open-ended range that
    starts at zero.
    """

    if start is None and end is None:
        return None
    begin = 0.0 if start is None else float(start)
    if begin < 0:
        raise AudioError("Times cannot be negative.")
    if end is not None and float(end) <= begin:
        raise AudioError("The end time must be later than the start time.")
    if begin == 0 and end is None:
        return None
    end_text = "inf" if end is None else f"{float(end):.6f}"
    return f"*{begin:.6f}-{end_text}"


def download_youtube(
    url: str,
    dest: Path,
    *,
    cookies: Path | None = None,
    cookies_from_browser: str | None = None,
    start: float | None = None,
    end: float | None = None,
) -> DownloadedAudio:
    """Download the best available audio and decode it to PCM WAV.

    When ``start``/``end`` select an excerpt, only that section is
    downloaded (yt-dlp ``--download-sections``); ``dest`` then holds just
    the excerpt. If a section download fails for a reason other than the
    sign-in retry below, the whole audio is downloaded instead, as before.
    Without an excerpt the whole audio is always downloaded.

    A signed-in YouTube page in your browser is not visible to a different
    computer. When YouTube asks for a sign-in, this retries the browsers on
    *this* machine. A login on another PC has to be used by running the
    command there.
    """

    require_ffmpeg()
    section = _download_section(start, end)
    dest.parent.mkdir(parents=True, exist_ok=True)
    attempts: list[str | None] = [cookies_from_browser]
    if cookies is None and cookies_from_browser is None:
        attempts = _browser_cookie_attempts()

    sign_in_error: AudioError | None = None
    for browser in attempts:
        try:
            return _download_youtube_once(
                url,
                dest,
                cookies=cookies,
                cookies_from_browser=browser,
                section=section,
            )
        except AudioError as exc:
            message = str(exc)
            unreadable = _cookie_database_unreadable(message)
            if (
                unreadable
                and cookies_from_browser is not None
                and not _missing_browser_cookies(message)
            ):
                raise AudioError(
                    message
                    + "\nThe browser cookie database is locked. Close that browser and run the command again, "
                    'or name another profile, for example --cookies-from-browser "chrome:Profile 1".'
                ) from exc
            if browser is not None and cookies_from_browser is None and unreadable:
                sign_in_error = exc
                continue
            if browser is None and _youtube_sign_in_required(message):
                sign_in_error = exc
                continue
            if browser is not None and _missing_browser_cookies(message):
                continue
            if browser is not None and cookies_from_browser is None and _youtube_sign_in_required(message):
                sign_in_error = exc
                continue
            raise
    if sign_in_error is None:
        raise AudioError("YouTube download failed before a browser cookie could be tried.")
    raise AudioError(
        "YouTube asked for a sign-in, and this computer has no readable Chrome, Edge, "
        "Firefox, Brave, or Chromium cookie database. A login in a browser on another "
        "computer is not available here. On the PC where you are signed in to YouTube, run:\n"
        f'  baroque-transcribe transcribe "{url}" -o transcriptions/baroque_improvisation '
        "--cookies-from-browser chrome"
    ) from sign_in_error


def _clear_download_files(dest: Path, template: Path) -> None:
    """Remove a finished or partial download so a retry starts clean."""

    for path in dest.parent.glob(template.name + ".*"):
        try:
            path.unlink()
        except OSError:
            pass
    if dest.exists():
        dest.unlink()


def _download_youtube_once(
    url: str,
    dest: Path,
    *,
    cookies: Path | None,
    cookies_from_browser: str | None,
    section: str | None = None,
) -> DownloadedAudio:
    if cookies_from_browser:
        print(f"Trying YouTube download with {cookies_from_browser} cookies.")
    title, full_duration = _youtube_info(url, cookies=cookies, cookies_from_browser=cookies_from_browser)
    template = dest.with_suffix("")
    # yt-dlp appends the extension from the template. A fixed wav name keeps
    # the decoded file where the rest of the pipeline expects it.
    output = str(template) + ".%(ext)s"

    def attempt(section_arg: str | None) -> None:
        _clear_download_files(dest, template)
        arguments = [
            "--no-playlist",
            *_youtube_auth_args(cookies, cookies_from_browser),
            "-f",
            "bestaudio/best",
            "-x",
            "--audio-format",
            "wav",
            "--audio-quality",
            "0",
            "--no-progress",
        ]
        if section_arg is not None:
            # Re-encoding at the cuts keeps the excerpt start sample-accurate
            # instead of landing on the previous audio frame boundary.
            arguments += ["--download-sections", section_arg, "--force-keyframes-at-cuts"]
        arguments += ["-o", output, url]
        _yt_dlp(*arguments)

    excerpt_only = section is not None
    try:
        attempt(section)
    except AudioError:
        if section is None:
            raise
        # A section download can fail on formats yt-dlp cannot range-request.
        # The full download is slower but is the behaviour this replaces, so
        # fall back to it within the same cookie attempt. A sign-in failure
        # simply fails again here and is handled by the caller's retry loop.
        print("Downloading only the requested section failed. Downloading the full audio instead.")
        excerpt_only = False
        attempt(None)
    if not dest.exists():
        produced = list(dest.parent.glob(template.name + ".*"))
        wavs = [path for path in produced if path.suffix.lower() == ".wav"]
        if len(wavs) == 1:
            wavs[0].replace(dest)
        else:
            raise AudioError(
                "yt-dlp finished without writing a WAV file. "
                "The video may be unavailable, age-restricted, or blocked."
            )
    return DownloadedAudio(
        title=title or dest.stem,
        full_duration=full_duration,
        excerpt_only=excerpt_only,
    )


def peak_normalize(path: Path, peak: float = WORKING_PEAK) -> float:
    """Scale a WAV so its peak sample is ``peak``. Returns the peak before scaling."""

    audio, sample_rate = sf.read(path, always_2d=False)
    array = np.asarray(audio, dtype=np.float64)
    original = float(np.max(np.abs(array))) if array.size else 0.0
    if original < 1e-8:
        raise AudioError(
            "The working audio is silent, so there is nothing to transcribe. "
            "Check the start and end times."
        )
    scaled = array * (peak / original)
    sf.write(path, scaled, sample_rate, subtype="PCM_16")
    return original


def load_working_audio(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=False)
    array = np.asarray(audio, dtype=np.float32)
    if array.ndim != 1:
        array = array.mean(axis=1).astype(np.float32)
    return array, int(sample_rate)


def prepare_audio(
    source: str,
    output_dir: Path,
    *,
    start: float | None = None,
    end: float | None = None,
    cookies: Path | None = None,
    cookies_from_browser: str | None = None,
    on_progress: ProgressCallback | None = None,
) -> PreparedAudio:
    """Copy or download ``source`` and write ``source.wav`` plus ``working.wav``.

    For a YouTube source with a start or end time, only that section is
    downloaded and ``source.wav`` holds just the section; the trim below
    then runs from the start of that file instead of trimming twice. A
    local file, or a YouTube download that fell back to the full audio,
    keeps the original behaviour: ``source.wav`` is the whole recording.
    """

    require_ffmpeg()
    output_dir.mkdir(parents=True, exist_ok=True)
    source_wav = output_dir / "source.wav"
    working_wav = output_dir / "working.wav"
    origin = source
    downloaded: DownloadedAudio | None = None

    if is_url(source):
        emit(
            on_progress,
            ProgressEvent(
                "prepare",
                "Downloading the requested excerpt…"
                if _download_section(start, end) is not None
                else "Downloading audio…",
                fraction_for("prepare", 0.25),
            ),
        )
        downloaded = download_youtube(
            source,
            source_wav,
            cookies=cookies,
            cookies_from_browser=cookies_from_browser,
            start=start,
            end=end,
        )
        title = downloaded.title
    else:
        emit(
            on_progress,
            ProgressEvent("prepare", "Reading the audio file…", fraction_for("prepare", 0.25)),
        )
        path = Path(source).expanduser().resolve()
        if not path.is_file():
            raise AudioError(f"Audio file not found: {path}")
        if path.suffix.lower() not in LOCAL_EXTENSIONS:
            allowed = ", ".join(sorted(LOCAL_EXTENSIONS))
            raise AudioError(f"Use a {allowed} file, or pass a YouTube URL. Got {path.suffix!r}.")
        title = path.stem
        _ffmpeg_wav(path, source_wav)
        origin = str(path)

    source_duration = probe_duration(source_wav)
    trim_start = 0.0 if start is None else start
    excerpt_only = downloaded is not None and downloaded.excerpt_only
    if excerpt_only:
        # source.wav already starts at trim_start. The full duration comes
        # from the video metadata; without it the best available figure is
        # the section placed at its requested offset.
        full_duration = (
            downloaded.full_duration
            if downloaded.full_duration and downloaded.full_duration > 0
            else trim_start + source_duration
        )
    else:
        full_duration = source_duration
    trim_end = full_duration if end is None else end
    if trim_start > full_duration:
        raise AudioError(
            f"Start time {trim_start:.2f}s is past the end of the audio ({full_duration:.2f}s)."
        )
    trim_end = min(trim_end, full_duration)
    if trim_end - trim_start < 0.2:
        raise AudioError("The selected region is shorter than 0.2 seconds.")

    if excerpt_only:
        # Trim in coordinates of the already-trimmed section file, capped
        # at what was actually downloaded.
        local_end = min(source_duration, trim_end - trim_start)
        _ffmpeg_wav(
            source_wav,
            working_wav,
            start=0.0,
            end=local_end,
            sample_rate=MODEL_SAMPLE_RATE,
            mono=True,
        )
    else:
        _ffmpeg_wav(
            source_wav,
            working_wav,
            start=trim_start,
            end=trim_end,
            sample_rate=MODEL_SAMPLE_RATE,
            mono=True,
        )
    original_peak = peak_normalize(working_wav)
    working, sample_rate = load_working_audio(working_wav)
    if sample_rate != MODEL_SAMPLE_RATE:
        raise AudioError(f"Working audio is {sample_rate} Hz; the model expects {MODEL_SAMPLE_RATE} Hz.")
    duration = float(len(working) / sample_rate)
    emit(
        on_progress,
        ProgressEvent(
            "prepare",
            f"Audio ready, {duration:.0f}s excerpt.",
            fraction_for("prepare", 1.0),
        ),
    )
    return PreparedAudio(
        source_wav=source_wav,
        working_wav=working_wav,
        sample_rate=sample_rate,
        duration=duration,
        full_duration=full_duration,
        start=trim_start,
        end=trim_start + duration,
        original_peak=original_peak,
        title=title,
        origin=origin,
    )
