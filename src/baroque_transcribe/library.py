"""What is on this computer, arranged for the home page.

A card is one YouTube video or one local recording. Each card lists its
transcription runs, which are the project folders under ``transcriptions/``.
The parse history contributes titles and remembers runs made on another
machine. A run can only be opened when its folder is on this computer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from baroque_transcribe.defaults import DIAGNOSTICS_DIR, MANIFEST, NOTES_JSON, SCORE_XML, WORKING_WAV
from baroque_transcribe.parse_store import repo_root, thumbnail_url, transcribed_videos
from baroque_transcribe.playback import project_youtube_id, youtube_video_id

TRANSCRIPTIONS_DIR = "transcriptions"
_GENERIC_TITLES = {"", "source", "working", "audio"}


@dataclass
class Run:
    """One project folder."""

    name: str
    folder: Path
    available: bool
    title: str = ""
    video_id: str | None = None
    url: str | None = None
    notes: int | None = None
    pedals: int | None = None
    model: str = ""
    source_kind: str = ""
    bpm: float | None = None
    time_signature: str = ""
    excerpt_start: float | None = None
    excerpt_end: float | None = None
    duration: float | None = None
    modified: datetime | None = None
    has_score: bool = False
    has_audio: bool = False
    has_overview: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass
class Card:
    """One video or recording, with every run made from it."""

    key: str
    title: str
    video_id: str | None
    url: str | None
    thumbnail: str | None
    runs: list[Run] = field(default_factory=list)

    @property
    def primary(self) -> Run | None:
        """The newest run that can be opened here."""

        available = [run for run in self.runs if run.available]
        if not available:
            return None
        return max(available, key=lambda run: run.modified or datetime.min)

    @property
    def newest(self) -> datetime | None:
        stamps = [run.modified for run in self.runs if run.modified is not None]
        return max(stamps) if stamps else None


def format_clock(seconds: float) -> str:
    """``M:SS``, or ``H:MM:SS`` from an hour up."""

    whole = int(round(seconds))
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def run_facts(run: Run) -> list[dict[str, str]]:
    """What the header of a project's page says about the run, in reading order.

    Each fact is the text of one chip in three parts: ``before``, the
    emphasised ``value``, and ``after``. A part that is not used is empty.
    Plain strings, so the pages can place them as text.
    """

    facts: list[dict[str, str]] = []

    def add(value: str, after: str = "", before: str = "") -> None:
        facts.append({"before": before, "value": value, "after": after})

    if run.notes is not None:
        add(f"{run.notes:,}", " notes")
    if run.pedals is not None:
        add(f"{run.pedals:,}", " pedal events")
    if run.duration:
        add(format_clock(run.duration), " excerpt")
    if run.excerpt_start:
        add(format_clock(run.excerpt_start), " in the video", "from ")
    if run.bpm:
        add(f"{run.bpm:g}", " BPM")
    if run.time_signature:
        add(run.time_signature)
    if run.source_kind == "midi" or run.model == "midi-import":
        add("", "MIDI file")
    elif run.model:
        add("", run.model)
    if run.modified:
        add("", run.modified.strftime("%b %d, %Y %H:%M"))
    return facts


def transcriptions_root(root: Path | None = None) -> Path:
    return root if root is not None else repo_root() / TRANSCRIPTIONS_DIR


def read_run(folder: Path, *, for_grid: bool = False) -> Run | None:
    """Describe a project folder, or return None when it holds no transcription.

    ``for_grid`` reads the manifest fields a library card shows and checks that
    ``notes.json`` exists. It does not open the notes, the audio, the MIDI, or
    the score. Those are read on the project page.
    """

    notes_path = folder / NOTES_JSON
    if not notes_path.is_file():
        return None
    manifest: dict = {}
    manifest_path = folder / MANIFEST
    if manifest_path.is_file():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = loaded if isinstance(loaded, dict) else {}
        except (OSError, json.JSONDecodeError):
            manifest = {}
    counts = manifest.get("counts") if isinstance(manifest.get("counts"), dict) else {}
    tempo = manifest.get("tempo") if isinstance(manifest.get("tempo"), dict) else {}
    audio = manifest.get("audio") if isinstance(manifest.get("audio"), dict) else {}
    video_id = project_youtube_id(manifest)
    url = None
    if video_id:
        for key in ("youtube_url", "origin"):
            candidate = str(manifest.get(key) or "")
            if youtube_video_id(candidate) == video_id:
                url = candidate
                break
    try:
        modified = datetime.fromtimestamp(notes_path.stat().st_mtime)
    except OSError:
        modified = None
    return Run(
        name=folder.name,
        folder=folder,
        available=True,
        title=_clean_title(str(manifest.get("title") or "")),
        video_id=video_id,
        url=url,
        notes=_int(counts.get("notes")),
        pedals=_int(counts.get("pedals")),
        model=str(manifest.get("model") or ""),
        source_kind=str(manifest.get("source_kind") or ""),
        bpm=_float(tempo.get("bpm")),
        time_signature=str(tempo.get("time_signature") or ""),
        excerpt_start=_float(audio.get("excerpt_start_seconds")),
        excerpt_end=_float(audio.get("excerpt_end_seconds")),
        duration=_float(audio.get("working_duration_seconds")),
        modified=modified,
        has_score=False if for_grid else (folder / SCORE_XML).is_file(),
        has_audio=False if for_grid else (folder / WORKING_WAV).is_file(),
        has_overview=False if for_grid else (folder / DIAGNOSTICS_DIR / "overview.png").is_file(),
        warnings=[] if for_grid else [str(item) for item in (manifest.get("warnings") or [])],
    )


def disk_runs(root: Path | None = None, *, for_grid: bool = False) -> list[Run]:
    """Every project folder on this computer that has a transcription."""

    base = transcriptions_root(root)
    if not base.is_dir():
        return []
    runs: list[Run] = []
    for folder in sorted(base.iterdir()):
        if not folder.is_dir() or folder.name.startswith(("_", ".")):
            continue
        run = read_run(folder, for_grid=for_grid)
        if run is not None:
            runs.append(run)
    return runs


def library_cards(root: Path | None = None, store: Path | None = None, *, for_grid: bool = False) -> list[Card]:
    """Cards for the home page, newest first.

    YouTube videos come from the parse history and from the manifests on
    disk, merged on the video id. A recording without a video gets its own
    card. A run named in the history but missing from this computer is kept,
    marked unavailable, so the record is visible without being clickable.
    """

    base = transcriptions_root(root)
    on_disk = {run.name: run for run in disk_runs(base, for_grid=for_grid)}
    cards: dict[str, Card] = {}
    for video in transcribed_videos(store):
        video_id = str(video["video_id"])
        card = Card(
            key=video_id,
            title=str(video["title"]),
            video_id=video_id,
            url=str(video["url"]),
            thumbnail=str(video["thumbnail"]),
        )
        for name in video["projects"]:
            run = on_disk.get(name)
            if run is None:
                run = Run(name=name, folder=base / name, available=False, video_id=video_id)
            card.runs.append(run)
        cards[video_id] = card
    for run in on_disk.values():
        if run.video_id:
            card = cards.get(run.video_id)
            if card is None:
                card = Card(
                    key=run.video_id,
                    title=run.title or run.video_id,
                    video_id=run.video_id,
                    url=run.url or f"https://www.youtube.com/watch?v={run.video_id}",
                    thumbnail=thumbnail_url(run.video_id),
                )
                cards[run.video_id] = card
            elif run.title and card.title in {card.video_id, ""}:
                card.title = run.title
            if all(existing.name != run.name for existing in card.runs):
                card.runs.append(run)
            continue
        key = f"file:{run.name}"
        cards[key] = Card(
            key=key,
            title=run.title or run.name,
            video_id=None,
            url=None,
            thumbnail=None,
            runs=[run],
        )
    for card in cards.values():
        card.runs.sort(key=lambda run: run.modified or datetime.min, reverse=True)
    return sorted(cards.values(), key=lambda card: card.newest or datetime.min, reverse=True)


def find_project(value: str | None, root: Path | None = None) -> Path | None:
    """Resolve a project named in the address bar to a folder with a transcription.

    A bare name is a folder under ``transcriptions/``. A path is used as given.
    """

    raw = (value or "").strip()
    if not raw:
        return None
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute() and len(candidate.parts) == 1:
        candidate = transcriptions_root(root) / candidate
    if not (candidate / NOTES_JSON).is_file():
        return None
    return candidate.resolve()


def project_param(folder: Path, root: Path | None = None) -> str:
    """The shortest value that ``find_project`` maps back to ``folder``."""

    base = transcriptions_root(root)
    try:
        relative = folder.resolve().relative_to(base.resolve())
    except (ValueError, OSError):
        return str(folder)
    if len(relative.parts) == 1:
        return relative.parts[0]
    return str(folder)


def card_for(folder: Path, root: Path | None = None, store: Path | None = None) -> Card | None:
    """The card that lists ``folder`` among its runs."""

    target = folder.resolve()
    for card in library_cards(root, store):
        for run in card.runs:
            try:
                if run.folder.resolve() == target:
                    return card
            except OSError:
                continue
    return None


_THUMBNAIL_LIMIT = 2_000_000


def thumbnail_cache_dir() -> Path:
    return repo_root() / "data" / "thumbnails"


def load_thumbnail(video_id: str, *, cache_dir: Path | None = None) -> bytes | None:
    """JPEG for one video, fetched from YouTube the first time and read from disk after that.

    Returns None when the id is not a YouTube id or the image cannot be fetched.
    Nothing else about the transcription is read.
    """

    if youtube_video_id(f"https://youtu.be/{video_id}") != video_id:
        return None
    folder = cache_dir if cache_dir is not None else thumbnail_cache_dir()
    cached = folder / f"{video_id}.jpg"
    try:
        if cached.is_file() and cached.stat().st_size > 0:
            return cached.read_bytes()
    except OSError:
        return None
    try:
        from urllib.request import Request, urlopen

        request = Request(thumbnail_url(video_id), headers={"User-Agent": "baroque-transcribe"})
        with urlopen(request, timeout=4) as response:
            data = response.read(_THUMBNAIL_LIMIT + 1)
    except Exception:
        return None
    if not data or len(data) > _THUMBNAIL_LIMIT:
        return None
    try:
        folder.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(data)
    except OSError:
        return data
    return data


def _clean_title(title: str) -> str:
    text = title.strip()
    return "" if text.lower() in _GENERIC_TITLES else text


def _int(value: object) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
