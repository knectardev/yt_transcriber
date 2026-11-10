"""The browser pages of a saved fragment.

A fragment is shown with the pages a transcription already has: the synced
page from ``playback`` (the video with the score following it) and the static
score page from ``notation``. This module reads a fragment folder and hands
those writers what they need. Nothing in the fragment folder is written; the
pages go to a machine-local folder and are rebuilt whenever they are opened.

The synced page needs no clock arithmetic of its own. It already works with
anchors on one clock, an origin that places that clock in the video, and a
sync offset. For a transcription those are working-audio seconds, the excerpt
start, and the project's offset. For a fragment they are::

    anchors and notes   fragment seconds (zero at the window start)
    origin              E + t0, written once at extraction
    sync offset         the fragment's own, in fragment.json

so ``video = fragment + origin + offset``, which is ``fragments.fragment_to_video``.
The page's loop runs over ``0 .. duration`` on the fragment clock. A sync
adjustment moves both ends by the same amount and cannot change the length.

Harmony is placed by measure and position in the bar, never by seconds.

A fragment taken from a local recording has no video, so it has no synced
page. Its page is the score with the harmony.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

from baroque_transcribe.defaults import MANIFEST, SCORE_XML
from baroque_transcribe.fragments import (
    FragmentError,
    duration_seconds,
    fragment_to_video,
    grid_notes,
    read_grid,
    read_harmony,
    read_manifest,
    sync_offset,
    video_origin,
)
from baroque_transcribe.playback import (
    PlaybackContext,
    SyncError,
    project_youtube_id,
    shared_playback_server,
    write_playback_page,
    youtube_video_id,
)
from baroque_transcribe.sync_map import MeasureAnchor, NoteAnchor

HOME_LABEL = "Fragments"
DETAILS_LABEL = "Details & files"
_TITLE_LIMIT = 60


# --------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------


def fragments_page_url(home_url: str) -> str:
    """Address of the fragment library on the library site, or empty without one."""

    if not home_url:
        return ""
    return f"{home_url.rstrip('/')}/fragments"


def fragment_details_url(home_url: str, fragment_id: str) -> str:
    """Address of one fragment's details page on the library site, or empty without one."""

    if not home_url or not fragment_id:
        return ""
    return f"{home_url.rstrip('/')}/fragment?id={quote(str(fragment_id), safe='')}"


def fragment_detail_url(folder: Path) -> str:
    """Address of the fragment's own page, without writing it.

    The server writes the page when the address is opened, as it does for a
    transcription. A fragment with a video opens on the synced page, and one
    without opens on its score.
    """

    server = shared_playback_server()
    return server.fragment_url(server.register_fragment(folder))


def youtube_place_url(manifest: dict) -> str:
    """The source video on YouTube at the start of the passage, in whole seconds."""

    video_id = fragment_video_id(manifest)
    if video_id is None:
        return ""
    address = f"https://www.youtube.com/watch?v={video_id}"
    start = int(max(0.0, fragment_to_video(0.0, manifest)))
    return f"{address}&t={start}s" if start >= 1 else address


# --------------------------------------------------------------------------
# Reading a fragment for its pages
# --------------------------------------------------------------------------


def fragment_id_of(folder: Path) -> str:
    """The fragment's id, or the folder name when the manifest cannot be read."""

    try:
        return str(read_manifest(folder)["id"])
    except (FragmentError, KeyError):
        return Path(folder).name


def fragment_video_id(manifest: dict) -> str | None:
    """The YouTube id of the fragment's source, or None for a local recording."""

    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    ident = str(source.get("video_id") or "")
    if ident and youtube_video_id(f"https://youtu.be/{ident}") == ident:
        return ident
    return youtube_video_id(str(source.get("url") or ""))


def _score_path(folder: Path, manifest: dict) -> Path:
    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    return Path(folder) / str(artifacts.get("musicxml") or SCORE_XML)


def fragment_has_score(folder: Path) -> bool:
    try:
        return _score_path(folder, read_manifest(folder)).is_file()
    except FragmentError:
        return False


def fragment_has_video(folder: Path) -> bool:
    try:
        return fragment_video_id(read_manifest(folder)) is not None
    except FragmentError:
        return False


def read_fragment_sync(folder: Path) -> float:
    """The fragment's sync offset, or zero when its manifest cannot be read."""

    try:
        return sync_offset(read_manifest(folder))
    except FragmentError:
        return 0.0


def _source_title(manifest: dict) -> str:
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    return str(source.get("title") or source.get("name") or "").strip()


def measures_label(manifest: dict) -> str:
    """``measures 12–15`` of the parent, or ``measure 12`` for a single bar."""

    parent = manifest.get("parent") if isinstance(manifest.get("parent"), dict) else {}
    first, last = parent.get("measure_start"), parent.get("measure_end")
    if first is None or last is None:
        return ""
    return f"measure {first}" if first == last else f"measures {first}\u2013{last}"


def key_label(manifest: dict) -> str:
    music = manifest.get("music") if isinstance(manifest.get("music"), dict) else {}
    key = music.get("analyzed_key") if isinstance(music.get("analyzed_key"), dict) else {}
    return f"{key.get('tonic') or ''} {key.get('mode') or ''}".strip()


def fragment_facts(manifest: dict) -> list[dict[str, str]]:
    """The header chips of a fragment's page, in the form ``library.run_facts`` uses."""

    from baroque_transcribe.library import format_clock

    facts: list[dict[str, str]] = []

    def add(value: str, after: str = "", before: str = "") -> None:
        facts.append({"before": before, "value": value, "after": after})

    source = _source_title(manifest)
    if source:
        shown = source if len(source) <= _TITLE_LIMIT else source[: _TITLE_LIMIT - 1].rstrip() + "\u2026"
        add(shown, before="from ")
    measures = measures_label(manifest)
    if measures:
        word, _, numbers = measures.partition(" ")
        add(numbers, before=f"{word} ")
    add(f"{duration_seconds(manifest):.1f}", " s")
    add(
        format_clock(max(0.0, video_origin(manifest))),
        " in the video" if fragment_video_id(manifest) else " in the recording",
        "at ",
    )
    music = manifest.get("music") if isinstance(manifest.get("music"), dict) else {}
    key = key_label(manifest)
    if key:
        add(key, before="analyzed key ")
    if music.get("bpm"):
        add(f"{float(music['bpm']):g}", " BPM")
    if music.get("time_signature"):
        add(str(music["time_signature"]))
    boundary = manifest.get("boundary") if isinstance(manifest.get("boundary"), dict) else {}
    if boundary.get("notes") is not None:
        add(f"{int(boundary['notes']):,}", " notes")
    return facts


def fragment_context(folder: Path, *, title: str | None = None) -> PlaybackContext:
    """What the synced page needs to show a fragment, read from its folder.

    Raises ``SyncError`` when the fragment has no video or no score, and
    ``FragmentError`` when its files cannot be read.
    """

    folder = Path(folder)
    manifest = read_manifest(folder)
    video_id = fragment_video_id(manifest)
    if video_id is None:
        raise SyncError("This fragment was taken from a local recording, so there is no video to play.")
    score_path = _score_path(folder, manifest)
    if not score_path.is_file():
        raise SyncError("This fragment has no score to display.")
    grid = read_grid(folder)
    try:
        per_measure = int(grid["steps_per_measure"])
        anchors = [
            MeasureAnchor(
                measure=int(item["measure"]),
                time=float(item["seconds"]),
                collided=bool(item.get("collided")),
                opens_before_attack=bool(item.get("opens_before_attack")),
            )
            for item in grid["anchors"]
        ]
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        raise FragmentError(f"The fragment's grid has no usable measure anchors: {exc}") from exc
    if per_measure <= 0 or not anchors:
        raise FragmentError("The fragment's grid has no usable measure anchors.")
    notes = []
    for note in sorted(grid_notes(grid), key=lambda item: (item.onset_step, item.pitch, item.index)):
        # A note already sounding at the first barline was struck before the
        # fragment. It is in the MIDI as a re-strike, but the video has no
        # attack there, so the attack dot does not mark it.
        if note.carried_in:
            continue
        notes.append(
            NoteAnchor(
                source_index=note.index,
                pitch=note.pitch,
                onset=note.onset_seconds,
                offset=note.offset_seconds if note.offset_seconds > note.onset_seconds else note.onset_seconds + 0.05,
                measure=note.onset_step // per_measure + 1,
            )
        )
    harmony = None
    analysis = read_harmony(folder, manifest)
    if analysis is not None:
        harmony = analysis.to_dict()
        # The slice was taken from an analysis that matched its notes, and
        # neither has changed since.
        harmony["stale"] = False
    duration = duration_seconds(manifest)
    return PlaybackContext(
        project=folder,
        score_path=score_path,
        title=title or str(manifest["title"]),
        video_id=video_id,
        anchors=anchors,
        excerpt_start=video_origin(manifest),
        sync_offset=sync_offset(manifest),
        harmony=harmony,
        facts=fragment_facts(manifest),
        note_anchors=notes,
        end_time=duration,
        passage=(0.0, duration),
    )


def parent_place(folder: Path, *, transcriptions: Path | None = None) -> tuple[Path, float] | None:
    """The fragment's parent project on this computer, and where the passage starts in it.

    The second value is working-audio seconds in the parent as it is now. It
    is worked out from the source-video time, so it stays right when the
    parent was transcribed again from a different excerpt start. None when
    the project is not here, or when the project of that name is a different
    recording.
    """

    from baroque_transcribe.library import find_project

    try:
        manifest = read_manifest(folder)
    except FragmentError:
        return None
    parent = manifest.get("parent") if isinstance(manifest.get("parent"), dict) else {}
    name = str(parent.get("project") or "")
    if not name or Path(name).name != name:
        return None
    project = find_project(name, transcriptions)
    if project is None:
        return None
    try:
        loaded = json.loads((project / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(loaded, dict):
        return None
    audio = loaded.get("audio") if isinstance(loaded.get("audio"), dict) else {}
    try:
        excerpt = float(audio.get("excerpt_start_seconds") or 0.0)
    except (TypeError, ValueError):
        return None
    video_id = fragment_video_id(manifest)
    if video_id is not None:
        if project_youtube_id(loaded) != video_id:
            return None
    else:
        # A local recording carries no address to compare. The excerpt start
        # is the one thing both sides recorded.
        stored = manifest["time"]["source_video"].get("excerpt_start_seconds")
        if not isinstance(stored, (int, float)) or abs(float(stored) - excerpt) > 0.01:
            return None
    working = video_origin(manifest) - excerpt
    if working < -1e-6:
        return None
    return project, max(0.0, working)


# --------------------------------------------------------------------------
# Writing the pages
# --------------------------------------------------------------------------


def write_fragment_playback(folder: Path, pages: Path, *, home_url: str = "", source_url: str = "") -> Path:
    """Write the fragment's synced page into ``pages`` and return it.

    ``source_url`` is the parent transcription's page, when there is one to
    link to.
    """

    folder = Path(folder)
    context = fragment_context(folder)
    return write_playback_page(
        context,
        Path(pages) / f"{folder.name}.playback.html",
        home_url=fragments_page_url(home_url),
        tools_url=fragment_details_url(home_url, fragment_id_of(folder)),
        source_url=source_url,
        home_label=HOME_LABEL,
        tools_label=DETAILS_LABEL,
    )


def write_fragment_notation(
    folder: Path,
    pages: Path,
    *,
    home_url: str = "",
    playback_url: str | None = None,
) -> Path:
    """Write the fragment's static score page into ``pages`` and return it.

    The time beside each measure number is where that bar falls in the
    source, counted at the written tempo from the fragment's first barline.
    """

    from baroque_transcribe.notation import NotationError, write_notation_page

    folder = Path(folder)
    manifest = read_manifest(folder)
    score_path = _score_path(folder, manifest)
    if not score_path.is_file():
        raise SyncError("This fragment has no score to display.")
    analysis = read_harmony(folder, manifest)
    first_barline = 0.0
    try:
        first_barline = float(read_grid(folder)["anchors"][0]["seconds"])
    except (FragmentError, KeyError, IndexError, TypeError, ValueError):
        first_barline = 0.0
    has_video = fragment_video_id(manifest) is not None
    source = _source_title(manifest)
    measures = measures_label(manifest)
    where = " ".join(part for part in (measures.capitalize(), f"of {source}" if source else "") if part)
    note = (
        (f"{where}. " if where else "")
        + "The time beside each measure number is an estimate of where the bar falls in the "
        + ("video" if has_video else "recording")
        + ", counted at the written tempo."
    )
    try:
        return write_notation_page(
            score_path,
            Path(pages) / f"{folder.name}.notation.html",
            title=str(manifest["title"]),
            note=note,
            home_url=fragments_page_url(home_url) or None,
            harmony=analysis.to_dict() if analysis is not None else None,
            facts=fragment_facts(manifest),
            folder=str(folder),
            tools_url=fragment_details_url(home_url, str(manifest["id"])) or None,
            playback_url=playback_url,
            time_offset_seconds=fragment_to_video(first_barline, manifest),
            home_label=HOME_LABEL,
            tools_label=DETAILS_LABEL,
        )
    except NotationError as exc:
        raise SyncError(str(exc)) from exc
