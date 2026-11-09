"""Play a YouTube video with the printed measures following it.

The player is the master clock. This module serves a localhost page, because
the YouTube IFrame API does not complete its handshake from a ``file://`` tab.
The static notation page is left alone.
"""

from __future__ import annotations

import json
import re
import secrets
import shutil
import threading
from dataclasses import dataclass
from html import escape
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from baroque_transcribe.defaults import MANIFEST, NOTES_JSON, SCORE_XML
# ``notation`` (and its music21 import) is loaded inside the functions that
# write score pages, not here: this module is also imported for YouTube id
# parsing on the UI home page, which must start without that weight.
from baroque_transcribe.quantize import parse_time_signature, quantize_notes
from baroque_transcribe.schema import read_transcription
from baroque_transcribe.sync_map import MeasureAnchor, measure_anchors

# Same file name as ``notation._NOTATION_LIBRARY``; duplicated so this module
# (and the page template below) does not need the music21 import at startup.
_NOTATION_LIBRARY = "opensheetmusicdisplay.min.js"


def _notation_library() -> Path:
    """Locate the vendored notation drawing library (see ``notation``)."""

    from baroque_transcribe.notation import _notation_library as locate

    return locate()


def write_notation_page(*args: object, **kwargs: object) -> Path:
    """Write the static score page (delegates to ``notation`` on first use)."""

    from baroque_transcribe.notation import write_notation_page as write

    return write(*args, **kwargs)  # type: ignore[arg-type]


_VIDEO_ID = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YOUTUBE_HOSTS = {"youtube.com", "m.youtube.com", "music.youtube.com"}
_OFFSET_LIMIT = 60.0
_SERVER: "PlaybackServer | None" = None
_SERVER_LOCK = threading.Lock()


class SyncError(ValueError):
    pass


def youtube_video_id(url: str) -> str | None:
    """Return the 11-character id of a YouTube watch, share, or embed URL."""

    raw = (url or "").strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    host = parsed.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    ident = ""
    if host == "youtu.be":
        ident = parsed.path.strip("/").split("/")[0]
    elif host in _YOUTUBE_HOSTS:
        parts = [part for part in parsed.path.split("/") if part]
        if parsed.path == "/watch" or parsed.path.startswith("/watch"):
            ident = (parse_qs(parsed.query).get("v") or [""])[0]
        elif len(parts) >= 2 and parts[0] in {"embed", "shorts", "live", "v"}:
            ident = parts[1]
    ident = ident.split("?")[0].split("&")[0]
    if not _VIDEO_ID.match(ident):
        return None
    return ident


def project_youtube_id(manifest: dict) -> str | None:
    """Video id for a project: an attached URL, or the transcription origin.

    ``origin`` is the audio that was transcribed. A MIDI file does not carry
    that address, so a later upload can set ``youtube_url`` without replacing
    ``origin``.
    """

    for key in ("youtube_url", "origin"):
        ident = youtube_video_id(str(manifest.get(key) or ""))
        if ident:
            return ident
    return None


def youtube_project_for_midi(midi: Path, output_dir: Path) -> Path | None:
    """Folder whose manifest should remember a video attached to this MIDI.

    A project MIDI lives beside ``notes.json``. An upload is stored under the
    output directory, and the address belongs to that project when the
    project already has a transcription.
    """

    if (midi.parent / NOTES_JSON).is_file():
        return midi.parent
    if (output_dir / NOTES_JSON).is_file():
        return output_dir
    return None


def write_youtube_url(project: Path, url: str) -> str:
    """Store a YouTube address on the project manifest and leave every other field."""

    if youtube_video_id(url) is None:
        raise SyncError("That is not a YouTube URL.")
    stored = url.strip()
    path = project / MANIFEST
    payload: dict = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload["youtube_url"] = stored
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return stored


def read_sync_offset(project: Path) -> float:
    path = project / MANIFEST
    if not path.is_file():
        return 0.0
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return float(payload.get("sync_offset_seconds") or 0.0)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return 0.0


def write_sync_offset(project: Path, offset: float) -> float:
    """Store the score shift on the project manifest and leave every other field.

    ``notate_transcription`` rewrites the manifest with ``dict.update``, so this
    field survives a later requantize.
    """

    stored = max(-_OFFSET_LIMIT, min(_OFFSET_LIMIT, round(float(offset), 4)))
    path = project / MANIFEST
    payload: dict = {}
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
    payload["sync_offset_seconds"] = stored
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return stored


def youtube_embed_html(video_id: str, start_seconds: float) -> str:
    """A paused IFrame API player cued at ``start_seconds``."""

    config = json.dumps({"videoId": video_id, "start": max(0.0, float(start_seconds))}).replace("<", "\\u003c")
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <style>
    html, body {{ margin: 0; height: 100%; background: #111; }}
    #player {{ width: 100%; height: 100%; }}
  </style>
</head>
<body>
  <div id="player"></div>
  <script>
    const config = {config};
    function onYouTubeIframeAPIReady() {{
      new YT.Player("player", {{
        playerVars: {{ rel: 0, modestbranding: 1, origin: window.location.origin }},
        events: {{
          onReady: function (event) {{
            event.target.cueVideoById({{ videoId: config.videoId, startSeconds: config.start }});
          }}
        }}
      }});
    }}
  </script>
  <script src="https://www.youtube.com/iframe_api"></script>
</body>
</html>
"""


@dataclass(frozen=True)
class PlaybackContext:
    project: Path
    score_path: Path
    title: str
    video_id: str
    anchors: list[MeasureAnchor]
    excerpt_start: float
    sync_offset: float


def load_playback(project: Path, *, title: str | None = None) -> PlaybackContext:
    """Read the score timing from the project without rewriting the notation.

    ``title`` replaces the manifest title on the page, for a manifest whose
    title is only the audio file name.
    """

    notes_path = project / NOTES_JSON
    score_path = project / SCORE_XML
    if not notes_path.is_file() or not score_path.is_file():
        raise SyncError("Transcribe the recording before playing it with the score.")
    manifest_path = project / MANIFEST
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    except json.JSONDecodeError as exc:
        raise SyncError(f"Could not read {manifest_path.name}.") from exc
    video_id = project_youtube_id(manifest)
    if video_id is None:
        raise SyncError("This project has no YouTube video to play.")
    tempo = manifest.get("tempo") or {}
    if not tempo.get("bpm") or not tempo.get("time_signature"):
        raise SyncError("This project has no tempo, so the measures cannot be aligned.")
    try:
        signature = parse_time_signature(str(tempo["time_signature"]))
        transcription = read_transcription(notes_path)
        quantization = quantize_notes(
            transcription.notes,
            bpm=float(tempo["bpm"]),
            numerator=signature[0],
            denominator=signature[1],
            subdivision=int(tempo.get("subdivision") or 16),
            downbeat=float(tempo.get("downbeat_seconds") or 0.0),
        )
        excerpt = float((manifest.get("audio") or {}).get("excerpt_start_seconds") or 0.0)
        offset = float(manifest.get("sync_offset_seconds") or 0.0)
    except (OSError, TypeError, ValueError) as exc:
        raise SyncError(f"Could not align the measures: {exc}") from exc
    return PlaybackContext(
        project=project,
        score_path=score_path,
        title=title or str(manifest.get("title") or project.name),
        video_id=video_id,
        anchors=measure_anchors(quantization),
        excerpt_start=excerpt,
        sync_offset=offset,
    )


def _ensure_library(dest_dir: Path) -> None:
    from baroque_transcribe.notation import NotationError

    try:
        library = _notation_library()
    except NotationError as exc:
        raise SyncError(str(exc)) from exc
    local = dest_dir / _NOTATION_LIBRARY
    dest_dir.mkdir(parents=True, exist_ok=True)
    if library.resolve() != local.resolve():
        shutil.copyfile(library, local)


def write_playback_page(context: PlaybackContext, dest: Path, *, home_url: str = "") -> Path:
    """Write the synced playback page next to the score.

    ``home_url`` is the library page. When it is given, the page shows a link
    back to it. ``?embed=1`` on the page address hides that bar, for the
    project page that frames this one.
    """

    if not context.score_path.is_file():
        raise SyncError(f"No score to display at {context.score_path}.")
    _ensure_library(dest.parent)
    from baroque_transcribe.notation import format_measure_time

    xml = json.dumps(context.score_path.read_text(encoding="utf-8")).replace("<", "\\u003c")
    config = json.dumps(
        {
            "videoId": context.video_id,
            "homeUrl": home_url or "",
            "excerptStart": context.excerpt_start,
            "syncOffset": context.sync_offset,
            "anchors": [
                {
                    "measure": anchor.measure,
                    "time": anchor.time,
                    "place": format_measure_time(
                        anchor.time + context.excerpt_start + context.sync_offset
                    ),
                    "collided": anchor.collided,
                    "opensBeforeAttack": anchor.opens_before_attack,
                }
                for anchor in context.anchors
            ],
        }
    ).replace("<", "\\u003c")
    title = escape(context.title)
    page = _PLAYBACK_TEMPLATE.replace("__TITLE__", title).replace("__CONFIG__", config).replace("__XML__", xml)
    page = page.replace("__LIBRARY__", _NOTATION_LIBRARY)
    dest.write_text(page, encoding="utf-8")
    return dest


def open_synced_playback(project: Path, *, title: str | None = None) -> str:
    """Write the page, serve it on localhost, and return the tab URL."""

    server = shared_playback_server()
    context = load_playback(project, title=title)
    write_playback_page(context, project / "playback.html", home_url=server.home_url)
    token = server.register(project)
    return server.page_url(token)


def notation_page_url(project: Path, *, title: str | None = None) -> str:
    """Write the static score page, serve it on localhost, and return its URL.

    The page is the same ``notation.html`` that opens from a ``file://`` tab.
    Serving it lets the project page frame it when there is no video to sync.
    """

    from baroque_transcribe.notation import NotationError

    score = project / SCORE_XML
    if not score.is_file():
        raise SyncError("This project has no score to display.")
    server = shared_playback_server()
    try:
        write_notation_page(score, project / "notation.html", title=title, home_url=server.home_url)
    except NotationError as exc:
        raise SyncError(str(exc)) from exc
    token = server.register(project)
    return server.page_url(token) + "notation.html"


def shared_playback_server(home_url: str | None = None) -> "PlaybackServer":
    """The one server for this process. ``home_url`` is remembered when given."""

    global _SERVER
    with _SERVER_LOCK:
        if _SERVER is None:
            _SERVER = PlaybackServer()
            _SERVER.start()
        if home_url is not None:
            _SERVER.home_url = home_url
        return _SERVER


class PlaybackServer:
    """Serve one project's playback page and its sync-offset field.

    Bound to 127.0.0.1. Tokens are random, and the only files offered are the
    generated pages and the local notation library.
    """

    def __init__(self, home_url: str = "") -> None:
        self._projects: dict[str, Path] = {}
        self._lock = threading.Lock()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.port = 0
        self.home_url = home_url

    def start(self) -> None:
        if self._httpd is not None:
            return
        handler = _handler_class(self)
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = int(self._httpd.server_address[1])
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="playback", daemon=True)
        self._thread.start()

    def close(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
        self._httpd = None
        self._thread = None

    def register(self, project: Path) -> str:
        """Return the token for a project. One project keeps one token for the life of the server."""

        resolved = project.resolve()
        with self._lock:
            for existing, folder in self._projects.items():
                if folder == resolved:
                    return existing
            token = secrets.token_urlsafe(8)
            self._projects[token] = resolved
        return token

    def project_for(self, token: str) -> Path | None:
        with self._lock:
            return self._projects.get(token)

    def page_url(self, token: str) -> str:
        return f"http://127.0.0.1:{self.port}/p/{token}/"


def _thumbnail_id(path: str) -> str | None:
    """Video id for ``/thumb/<id>.jpg``, or None when the path is something else."""

    parts = [part for part in urlparse(path).path.split("/") if part]
    if len(parts) != 2 or parts[0] != "thumb" or not parts[1].endswith(".jpg"):
        return None
    video_id = parts[1][: -len(".jpg")]
    if not _VIDEO_ID.match(video_id):
        return None
    return video_id


def _route(path: str) -> tuple[str, str] | None:
    parts = [part for part in urlparse(path).path.split("/") if part]
    if len(parts) < 2 or parts[0] != "p" or len(parts) > 3:
        return None
    name = parts[2] if len(parts) == 3 else ""
    allowed = {"", "index.html", "playback.html", "notation.html", "opensheetmusicdisplay.min.js", "sync-offset"}
    if name not in allowed:
        return None
    return parts[1], name


def _handler_class(server: PlaybackServer) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            video_id = _thumbnail_id(self.path)
            if video_id:
                from baroque_transcribe.library import load_thumbnail

                body = load_thumbnail(video_id)
                if body is None:
                    self.send_error(404)
                    return
                self._send(200, "image/jpeg", body, cache=True)
                return
            route = _route(self.path)
            project = server.project_for(route[0]) if route else None
            if route is None or project is None:
                self.send_error(404)
                return
            _token, name = route
            if name == "sync-offset":
                body = json.dumps({"offset": read_sync_offset(project)}).encode("utf-8")
                self._send(200, "application/json", body)
                return
            if name == "opensheetmusicdisplay.min.js":
                target = project / name
                content_type = "text/javascript; charset=utf-8"
            elif name == "notation.html":
                target = project / name
                content_type = "text/html; charset=utf-8"
            else:
                target = project / "playback.html"
                content_type = "text/html; charset=utf-8"
            if not target.is_file():
                self.send_error(404)
                return
            self._send(200, content_type, target.read_bytes())

        def do_POST(self) -> None:  # noqa: N802
            route = _route(self.path)
            project = server.project_for(route[0]) if route else None
            if route is None or project is None or route[1] != "sync-offset":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0 or length > 4096:
                self.send_error(400)
                return
            try:
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                stored = write_sync_offset(project, float(payload["offset"]))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
                self.send_error(400)
                return
            self._send(200, "application/json", json.dumps({"offset": stored}).encode("utf-8"))

        def _send(self, status: int, content_type: str, body: bytes, cache: bool = False) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "public, max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

    return Handler


_PLAYBACK_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      --bg: #faf7f2; --panel: #fffdf9; --ink: #1c1917; --muted: #6b6560; --line: #e7e2da;
      --accent: #b45309; --accent-ink: #7c3a07; --accent-soft: rgba(251, 191, 36, 0.28);
      --serif: Georgia, "Iowan Old Style", "Times New Roman", serif;
      --sans: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    * { box-sizing: border-box; }
    html, body { height: 100%; }
    body {
      margin: 0; height: 100vh; overflow: hidden;
      display: flex; flex-direction: column;
      font-family: var(--sans); color: var(--ink); background: var(--bg);
    }
    .topbar {
      flex: 0 0 auto; display: flex; align-items: center; gap: 1rem;
      padding: 0.55rem 1.25rem; border-bottom: 1px solid var(--line); background: var(--panel);
      font-size: 0.9rem;
    }
    .topbar a { color: var(--accent-ink); text-decoration: none; font-weight: 500; }
    .topbar a:hover { text-decoration: underline; }
    .topbar .crumb { color: var(--muted); }
    .topbar h1 {
      flex: 1 1 auto; margin: 0; font-family: var(--serif); font-size: 1.05rem; font-weight: 600;
      white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }
    body.embed .topbar { display: none; }
    .stage {
      flex: 0 0 auto; display: grid; grid-template-columns: minmax(0, 560px) minmax(260px, 1fr);
      gap: 1rem 1.5rem; align-items: start; padding: 0.9rem 1.25rem; border-bottom: 1px solid var(--line);
      max-height: 58vh; overflow: auto;
    }
    @media (max-width: 860px) { .stage { grid-template-columns: 1fr; } }
    .player-wrap {
      width: 100%; aspect-ratio: 16 / 9; background: #111; border-radius: 10px; overflow: hidden;
      box-shadow: 0 1px 2px rgba(28, 25, 23, 0.08), 0 8px 24px -16px rgba(28, 25, 23, 0.35);
    }
    #player, #player iframe { width: 100%; height: 100%; display: block; }
    .controls { display: flex; flex-direction: column; gap: 0.8rem; min-width: 0; }
    .transport { display: flex; flex-wrap: wrap; align-items: center; gap: 0.6rem 0.9rem; }
    button {
      font-family: inherit; font-size: 0.92rem; padding: 0.42rem 0.9rem; border-radius: 999px;
      border: 1px solid var(--line); background: white; color: var(--ink); cursor: pointer;
    }
    button:hover { border-color: var(--accent); color: var(--accent-ink); }
    button#play {
      min-width: 6rem; background: var(--accent); border-color: var(--accent); color: white; font-weight: 600;
    }
    button#play:hover { background: var(--accent-ink); border-color: var(--accent-ink); color: white; }
    label, output, .readout { font-size: 0.92rem; }
    #clock, #measure-label, #offset-label { font-variant-numeric: tabular-nums; }
    #clock { font-size: 1rem; }
    #measure-label { color: var(--accent-ink); font-weight: 600; }
    label.follow { display: inline-flex; align-items: center; gap: 0.35rem; color: var(--muted); }
    .sync {
      display: flex; flex-wrap: wrap; align-items: center; gap: 0.5rem 0.7rem;
      padding: 0.6rem 0.8rem; border: 1px solid var(--line); border-radius: 10px; background: var(--panel);
    }
    .sync .readout { color: var(--muted); }
    .sync .readout output { color: var(--ink); font-weight: 600; }
    .limit { margin: 0; max-width: 46rem; color: var(--muted); font-size: 0.84rem; line-height: 1.45; }
    #player-error { color: #9a3412; }
    #score-pane { flex: 1 1 auto; min-height: 0; overflow: auto; background: white; }
    #score { padding: 0.5rem 1rem 2rem; }
    .measure-hit { position: absolute; z-index: 3; cursor: pointer; border-radius: 3px; }
    .measure-hit:hover { outline: 1px solid rgba(180, 83, 9, 0.45); }
    .measure-hit.active { background: var(--accent-soft); }
  </style>
</head>
<body>
  <nav class="topbar">
    <a id="home-link" href="#" hidden>&larr; Library</a>
    <span class="crumb" id="home-sep" hidden>/</span>
    <h1>__TITLE__</h1>
    <a id="youtube-link" href="#" target="_blank" rel="noopener">Open on YouTube &nearr;</a>
  </nav>
  <header class="stage">
    <div class="player-wrap"><div id="player"></div></div>
    <div class="controls">
      <div class="transport">
        <button id="play" type="button">Play</button>
        <span id="clock">0:00 / 0:00</span>
        <span id="measure-label">Measure —</span>
        <label class="follow"><input id="follow" type="checkbox" checked> Follow score</label>
      </div>
      <div class="sync">
        <span class="readout">Sync adjustment <output id="offset-label">0.00 s</output></span>
        <button id="earlier" type="button">Notation earlier</button>
        <button id="reset" type="button">Reset</button>
        <button id="later" type="button">Notation later</button>
      </div>
      <p class="limit">The time in parentheses is that measure's place on the video clock. A measure can open with a rest or a held note before its first new note, and that opening follows the written tempo. Notation earlier moves the highlight backward. Notation later moves it forward. Click a measure to jump the video there.</p>
      <p id="player-error" class="limit" hidden>This video cannot be embedded. The working audio is still on the transcription page.</p>
    </div>
  </header>
  <div id="score-pane">
    <div id="score"></div>
  </div>
  <script src="__LIBRARY__"></script>
  <script>
    const config = __CONFIG__;
    const xml = __XML__;
    (function () {
      const params = new URLSearchParams(window.location.search);
      if (params.get("embed") === "1") document.body.classList.add("embed");
      const home = document.getElementById("home-link");
      if (config.homeUrl) {
        home.href = config.homeUrl;
        home.hidden = false;
        document.getElementById("home-sep").hidden = false;
      }
      document.getElementById("youtube-link").href = "https://www.youtube.com/watch?v=" + encodeURIComponent(config.videoId);
    })();
    const anchors = config.anchors || [];
    const excerptStart = Number(config.excerptStart) || 0;
    let syncOffset = Number(config.syncOffset) || 0;
    const anchorByMeasure = {};
    anchors.forEach(function (anchor) { anchorByMeasure[anchor.measure] = anchor; });
    let player = null;
    let osmd = null;

    function formatClock(seconds) {
      if (!isFinite(seconds) || seconds < 0) seconds = 0;
      const whole = Math.floor(seconds);
      const minutes = Math.floor(whole / 60);
      const secs = whole % 60;
      return String(minutes).padStart(2, "0") + ":" + String(secs).padStart(2, "0");
    }

    function measureAt(working) {
      if (!anchors.length) return null;
      let chosen = anchors[0].measure;
      for (let index = 0; index < anchors.length; index += 1) {
        if (working + 1e-9 >= anchors[index].time) chosen = anchors[index].measure;
        else break;
      }
      return chosen;
    }

    function revealMeasure(element) {
      if (!document.getElementById("follow").checked) return;
      const pane = document.getElementById("score-pane");
      const paneRect = pane.getBoundingClientRect();
      const rect = element.getBoundingClientRect();
      const fullyVisible = rect.top >= paneRect.top + 8 && rect.bottom <= paneRect.bottom - 8;
      const hasLookahead = rect.bottom <= paneRect.top + paneRect.height * 0.72;
      const alreadyHigh = rect.top < paneRect.top + paneRect.height * 0.45;
      if (fullyVisible && (hasLookahead || alreadyHigh)) return;
      const desired = paneRect.top + paneRect.height * 0.18;
      const delta = rect.top - desired;
      if (Math.abs(delta) < 2) return;
      pane.scrollTop += delta;
    }

    let playbackRate = 1;
    let clockMedia = 0;
    let clockWall = 0;
    let clockPlaying = false;
    let lastReported = null;

    function readPlaybackRate() {
      try {
        const rate = player.getPlaybackRate();
        if (typeof rate === "number" && isFinite(rate) && rate > 0) playbackRate = rate;
      } catch (error) {}
      return playbackRate;
    }

    function rebaseClock(media) {
      clockMedia = media;
      clockWall = performance.now();
    }

    function mediaTime() {
      // The highlight follows the video picture. getCurrentTime() is that
      // position when the player is at 1x. At 0.5x the picture moves half as
      // fast; if the reported clock keeps ticking near full speed, advance
      // the highlight at getPlaybackRate() instead so it stays with the frame.
      if (!player || !player.getCurrentTime) return 0;
      let reported = 0;
      let playing = false;
      try {
        reported = player.getCurrentTime() || 0;
        playing = player.getPlayerState() === 1;
      } catch (error) {
        return clockMedia;
      }
      readPlaybackRate();
      const jumped = lastReported !== null && Math.abs(reported - lastReported) > 1.5;
      lastReported = reported;
      if (!playing || !clockPlaying || jumped) {
        clockPlaying = playing;
        rebaseClock(reported);
        return reported;
      }
      const elapsed = (performance.now() - clockWall) / 1000;
      const predicted = clockMedia + elapsed * playbackRate;
      const advanced = reported - clockMedia;
      if (playbackRate < 0.999 && advanced > elapsed * playbackRate + 0.35) return predicted;
      rebaseClock(reported);
      return reported;
    }

    function formatPlace(seconds) {
      if (!isFinite(seconds)) return "";
      const negative = seconds < 0;
      const whole = Math.floor(Math.abs(seconds) + 0.5);
      const sign = negative && whole !== 0 ? "-" : "";
      const secs = whole % 60;
      const minutes = Math.floor(whole / 60) % 60;
      const hours = Math.floor(whole / 3600);
      const clock = hours
        ? hours + ":" + String(minutes).padStart(2, "0") + ":" + String(secs).padStart(2, "0")
        : minutes + ":" + String(secs).padStart(2, "0");
      return "(" + sign + clock + ")";
    }

    function videoPlace(anchor) {
      if (!anchor) return "";
      return formatPlace(anchor.time + excerptStart + syncOffset);
    }

    function applyMeasure(measureNumber) {
      const hits = document.querySelectorAll(".measure-hit");
      hits.forEach(function (hit) {
        hit.classList.toggle("active", Number(hit.dataset.measure) === measureNumber);
      });
      const chosen = measureNumber ? anchorByMeasure[measureNumber] : null;
      const place = chosen ? " " + videoPlace(chosen) : "";
      document.getElementById("measure-label").textContent = measureNumber ? ("Measure " + measureNumber + place) : "Measure —";
      const active = document.querySelector(".measure-hit.active");
      if (active) revealMeasure(active);
    }

    function stampMeasurePlaces() {
      const places = {};
      anchors.forEach(function (anchor) { places[String(anchor.measure)] = videoPlace(anchor); });
      const pages = (osmd && osmd.GraphicSheet && osmd.GraphicSheet.MusicPages) || [];
      pages.forEach(function (page) {
        (page.MusicSystems || []).forEach(function (system) {
          (system.MeasureNumberLabels || []).forEach(function (label) {
            const node = label.SVGNode;
            if (!node) return;
            const text = node.tagName === "text" ? node : node.querySelector("text, tspan");
            if (!text) return;
            const raw = (text.textContent || "").trim().split(" ")[0];
            const stamp = places[raw];
            if (!stamp) return;
            text.textContent = raw + " " + stamp;
          });
        });
      });
    }

    function syncFromPlayer() {
      if (!player || !player.getCurrentTime) return;
      let current = 0;
      let duration = 0;
      try {
        current = mediaTime();
        duration = player.getDuration() || 0;
      } catch (error) {
        return;
      }
      document.getElementById("clock").textContent = formatClock(current) + " / " + formatClock(duration);
      applyMeasure(measureAt(current - excerptStart - syncOffset));
    }

    function showOffset() {
      document.getElementById("offset-label").textContent = syncOffset.toFixed(2) + " s";
    }

    function setOffset(next) {
      syncOffset = Math.max(-60, Math.min(60, next));
      showOffset();
      stampMeasurePlaces();
      syncFromPlayer();
      fetch("sync-offset", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ offset: syncOffset })
      }).catch(function () {});
    }

    function buildMeasureHits() {
      document.querySelectorAll(".measure-hit").forEach(function (node) { node.remove(); });
      const list = (osmd && osmd.GraphicSheet && osmd.GraphicSheet.MeasureList) || [];
      const scale = 10 * (osmd.Zoom || 1);
      for (let index = 0; index < list.length; index += 1) {
        const staffs = list[index] || [];
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        let pageNumber = null;
        let measureNumber = index + 1;
        for (let staff = 0; staff < staffs.length; staff += 1) {
          const measure = staffs[staff];
          if (!measure || !measure.PositionAndShape) continue;
          const source = measure.parentSourceMeasure;
          if (source && source.MeasureNumber) measureNumber = source.MeasureNumber;
          const box = measure.PositionAndShape;
          const x = box.AbsolutePosition.x + box.BorderLeft;
          const y = box.AbsolutePosition.y + box.BorderTop;
          const right = x + box.Size.width;
          const bottom = y + box.Size.height;
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (right > maxX) maxX = right;
          if (bottom > maxY) maxY = bottom;
          const system = measure.ParentStaffLine && measure.ParentStaffLine.ParentMusicSystem;
          const page = system && system.Parent;
          if (page && page.PageNumber) pageNumber = page.PageNumber;
        }
        if (!isFinite(minX) || maxX <= minX || maxY <= minY || !anchorByMeasure[measureNumber]) continue;
        const pageEl = document.getElementById("osmdCanvasPage" + pageNumber) || document.getElementById("score");
        pageEl.style.position = "relative";
        const hit = document.createElement("div");
        hit.className = "measure-hit";
        hit.dataset.measure = String(measureNumber);
        hit.title = "Estimated measure start";
        hit.style.left = (minX * scale) + "px";
        hit.style.top = (minY * scale) + "px";
        hit.style.width = ((maxX - minX) * scale) + "px";
        hit.style.height = ((maxY - minY) * scale) + "px";
        hit.addEventListener("click", function (event) {
          const chosen = anchorByMeasure[Number(event.currentTarget.dataset.measure)];
          if (!chosen || !player || !player.seekTo) return;
          player.seekTo(Math.max(0, chosen.time + excerptStart + syncOffset), true);
          syncFromPlayer();
        });
        pageEl.appendChild(hit);
      }
      syncFromPlayer();
    }

    function onYouTubeIframeAPIReady() {
      player = new YT.Player("player", {
        playerVars: { rel: 0, modestbranding: 1, origin: window.location.origin },
        events: {
          onReady: function (event) {
            event.target.cueVideoById({ videoId: config.videoId, startSeconds: Math.max(0, excerptStart) });
            syncFromPlayer();
          },
          onStateChange: function (event) {
            document.getElementById("play").textContent = event.data === 1 ? "Pause" : "Play";
            syncFromPlayer();
          },
          onPlaybackRateChange: function (event) {
            if (typeof event.data === "number" && isFinite(event.data) && event.data > 0) playbackRate = event.data;
            clockPlaying = false;
            syncFromPlayer();
          },
          onError: function (event) {
            const note = document.getElementById("player-error");
            note.hidden = false;
            if (event.data === 101 || event.data === 150) {
              note.textContent = "The owner of this video has disabled embedding. The working audio is still on the transcription page.";
            }
          }
        }
      });
    }

    document.getElementById("play").addEventListener("click", function () {
      if (!player || !player.getPlayerState) return;
      if (player.getPlayerState() === 1) player.pauseVideo();
      else player.playVideo();
    });
    document.getElementById("earlier").addEventListener("click", function () { setOffset(syncOffset + 0.1); });
    document.getElementById("later").addEventListener("click", function () { setOffset(syncOffset - 0.1); });
    document.getElementById("reset").addEventListener("click", function () { setOffset(0); });
    showOffset();
    setInterval(syncFromPlayer, 100);
    fetch("sync-offset").then(function (response) { return response.json(); }).then(function (payload) {
      if (payload && typeof payload.offset === "number") {
        syncOffset = payload.offset;
        showOffset();
        syncFromPlayer();
      }
    }).catch(function () {});

    const library = window.opensheetmusicdisplay;
    const score = document.getElementById("score");
    if (!library) {
      score.textContent = "The notation library did not load from __LIBRARY__.";
    } else {
      osmd = new library.OpenSheetMusicDisplay("score", {
        autoResize: true,
        backend: "svg",
        drawTitle: false,
        drawComposer: false,
        drawPartNames: true,
        drawMeasureNumbers: true,
        measureNumberInterval: 1
      });
      const drawScore = osmd.render.bind(osmd);
      osmd.render = function () {
        const drawn = drawScore();
        stampMeasurePlaces();
        buildMeasureHits();
        return drawn;
      };
      osmd.load(xml).then(function () { osmd.render(); }).catch(function (error) {
        score.textContent = "Could not draw the score. " + error;
      });
    }
  </script>
  <script src="https://www.youtube.com/iframe_api"></script>
</body>
</html>
"""
