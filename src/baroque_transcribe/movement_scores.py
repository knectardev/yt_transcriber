"""Cached per-movement MusicXML slices for multi-movement catalog projects.

Each movement opens as a standalone score (measures renumbered from 1) under
the suite's BWV page. Absolute suite times are converted to movement-local
time (0 at the section start) for playback.

Slicing prefers a fast MusicXML edit (carry clef/key/time into the first
kept measure) so opening one section does not wait on a full music21 parse of
the suite. music21 remains the fallback if the XML path fails.
"""

from __future__ import annotations

import copy
import logging
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

from baroque_transcribe.defaults import LOCAL_DIR, SCORE_VOICES_XML, SCORE_XML
from baroque_transcribe.parse_store import repo_root
from baroque_transcribe.sync_map import MeasureAnchor, NoteAnchor

logger = logging.getLogger(__name__)

# Bump when the on-disk slice format changes (invalidates ``*.stamp`` caches).
_SLICE_VERSION = "xmlv2"


class MovementScoreError(ValueError):
    pass


def movement_scores_root(local: Path | None = None) -> Path:
    base = local if local is not None else repo_root() / LOCAL_DIR
    return Path(base) / "movement_scores"


def measures_for_time_span(
    anchors: list[MeasureAnchor],
    start_seconds: float,
    end_seconds: float,
) -> tuple[int, int]:
    """Printed measure numbers covering ``[start_seconds, end_seconds)``.

    Includes the measure that is already under way at ``start_seconds``.
    """

    if not anchors:
        raise MovementScoreError("This project has no measures to slice.")
    ordered = sorted(anchors, key=lambda row: (row.time, row.measure))
    first = ordered[0].measure
    for anchor in ordered:
        if anchor.time <= start_seconds + 1e-6:
            first = anchor.measure
        else:
            break
    last = first
    for anchor in ordered:
        if anchor.time + 1e-6 >= end_seconds:
            break
        if anchor.measure >= first:
            last = anchor.measure
    return first, last


def _local_tag(tag: str) -> str:
    return tag.split("}")[-1] if isinstance(tag, str) else ""


def _find_child(parent: ET.Element, name: str) -> ET.Element | None:
    for child in parent:
        if _local_tag(child.tag) == name:
            return child
    return None


def _clef_key(clef: ET.Element) -> str:
    return clef.get("number") or "1"


def _merge_attributes(state: ET.Element | None, incoming: ET.Element) -> ET.Element:
    """Accumulate MusicXML ``attributes`` so mid-score excerpts keep clef/key/time."""

    if state is None:
        return copy.deepcopy(incoming)
    for child in list(incoming):
        name = _local_tag(child.tag)
        if name == "clef":
            key = _clef_key(child)
            for existing in list(state):
                if _local_tag(existing.tag) == "clef" and _clef_key(existing) == key:
                    state.remove(existing)
            state.append(copy.deepcopy(child))
            continue
        if name in {"", "footnote", "level"}:
            continue
        for existing in list(state):
            if _local_tag(existing.tag) == name and name != "clef":
                state.remove(existing)
        state.append(copy.deepcopy(child))
    return state


def _set_movement_title(root: ET.Element, title: str) -> None:
    if not title:
        return
    work = _find_child(root, "work")
    if work is None:
        work = ET.Element("work")
        root.insert(0, work)
    movement = _find_child(work, "work-title")
    # Prefer movement-title when present; else work-title as a visible label.
    movement_title = _find_child(root, "movement-title")
    if movement_title is None:
        # Insert after identification/work/defaults-ish preamble.
        movement_title = ET.Element("movement-title")
        movement_title.text = title
        insert_at = 0
        for index, child in enumerate(list(root)):
            if _local_tag(child.tag) in {"work", "movement-number", "identification", "defaults"}:
                insert_at = index + 1
        root.insert(insert_at, movement_title)
    else:
        movement_title.text = title


def slice_musicxml_measures(
    source: Path,
    dest: Path,
    *,
    first: int,
    last: int,
    title: str = "",
) -> None:
    """Copy ``[first, last]`` measures into ``dest``, renumbered from 1.

    Carries the running ``attributes`` (divisions/key/time/staves/clefs) into the
    first kept measure so OSMD can open a mid-suite excerpt alone.
    """

    if last < first:
        raise MovementScoreError(f"Invalid measure span {first}-{last}.")
    try:
        tree = ET.parse(source)
    except ET.ParseError as exc:
        raise MovementScoreError(f"Could not parse MusicXML: {exc}") from exc
    root = tree.getroot()
    parts = [el for el in list(root) if _local_tag(el.tag) == "part"]
    if not parts:
        raise MovementScoreError("MusicXML has no parts to slice.")

    kept_any = False
    for part in parts:
        children = list(part)
        measures = [el for el in children if _local_tag(el.tag) == "measure"]
        non_measures = [el for el in children if _local_tag(el.tag) != "measure"]
        attr_state: ET.Element | None = None
        kept: list[ET.Element] = []
        for measure in measures:
            raw = measure.get("number") or ""
            try:
                number = int(str(raw).split("+")[0])
            except ValueError:
                continue
            incoming = _find_child(measure, "attributes")
            if incoming is not None:
                attr_state = _merge_attributes(attr_state, incoming)
            if number < first:
                continue
            if number > last:
                break
            clone = copy.deepcopy(measure)
            if not kept and attr_state is not None:
                existing = _find_child(clone, "attributes")
                if existing is not None:
                    clone.remove(existing)
                clone.insert(0, copy.deepcopy(attr_state))
            kept.append(clone)
        if not kept:
            continue
        kept_any = True
        for index, measure in enumerate(kept, start=1):
            measure.set("number", str(index))
        for child in list(part):
            part.remove(child)
        for child in non_measures:
            part.append(child)
        for measure in kept:
            part.append(measure)

    if not kept_any:
        raise MovementScoreError(f"No measures found in span {first}-{last}.")

    _set_movement_title(root, title)
    temporary = dest.with_suffix(dest.suffix + ".tmp")
    try:
        tree.write(temporary, encoding="utf-8", xml_declaration=True)
        temporary.replace(dest)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _slice_with_music21(
    source: Path,
    dest: Path,
    *,
    first: int,
    last: int,
    title: str = "",
) -> None:
    from music21 import converter

    score = converter.parse(str(source))
    try:
        excerpt = score.measures(first, last)
    except Exception as exc:
        raise MovementScoreError(f"Could not slice measures {first}-{last}: {exc}") from exc

    parts = list(excerpt.parts) if hasattr(excerpt, "parts") else []
    if parts:
        for part in parts:
            for index, measure in enumerate(part.getElementsByClass("Measure"), start=1):
                measure.number = index
    else:
        for index, measure in enumerate(excerpt.recurse().getElementsByClass("Measure"), start=1):
            measure.number = index

    if title:
        try:
            if excerpt.metadata is None:
                from music21 import metadata

                excerpt.metadata = metadata.Metadata()
            excerpt.metadata.movementName = title
        except Exception:
            pass

    temporary = dest.with_suffix(".tmp.musicxml")
    try:
        excerpt.write("musicxml", fp=str(temporary))
        temporary.replace(dest)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def ensure_movement_score(
    project: Path,
    *,
    movement_id: str,
    start_seconds: float,
    end_seconds: float,
    anchors: list[MeasureAnchor],
    title: str = "",
    local: Path | None = None,
    source: Path | None = None,
    cache_suffix: str = "",
) -> tuple[Path, int, int]:
    """Write (or reuse) a standalone MusicXML for one movement.

    Measures are renumbered from 1. Returns ``(path, source_first, source_last)``.
    Pass ``source`` / ``cache_suffix`` to slice ``score.voices.musicxml`` beside
    the plain score (``cache_suffix=".voices"``).
    """

    project = Path(project)
    source_path = Path(source) if source is not None else project / SCORE_XML
    if not source_path.is_file():
        raise MovementScoreError(f"No score to slice at {source_path}.")
    if end_seconds <= start_seconds:
        raise MovementScoreError("Movement end must be after start.")

    first, last = measures_for_time_span(anchors, start_seconds, end_seconds)
    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in movement_id).strip("-") or "section"
    suffix = str(cache_suffix or "")
    cache = movement_scores_root(local) / project.name
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"{safe_id}{suffix}.musicxml"
    stamp = cache / f"{safe_id}{suffix}.stamp"
    token = f"{source_path.stat().st_mtime_ns}:{first}:{last}:{_SLICE_VERSION}:{title}"
    if dest.is_file() and stamp.is_file() and stamp.read_text(encoding="utf-8") == token:
        return dest, first, last

    try:
        slice_musicxml_measures(source_path, dest, first=first, last=last, title=title)
    except MovementScoreError:
        _slice_with_music21(source_path, dest, first=first, last=last, title=title)
    stamp.write_text(token, encoding="utf-8")
    return dest, first, last


def ensure_movement_scores(
    project: Path,
    *,
    movement_id: str,
    start_seconds: float,
    end_seconds: float,
    anchors: list[MeasureAnchor],
    title: str = "",
    local: Path | None = None,
) -> tuple[Path, Path | None, int, int]:
    """Slice plain score and, when present, the coloured voices sibling."""

    score_path, first, last = ensure_movement_score(
        project,
        movement_id=movement_id,
        start_seconds=start_seconds,
        end_seconds=end_seconds,
        anchors=anchors,
        title=title,
        local=local,
    )
    voices_source = Path(project) / SCORE_VOICES_XML
    voices_path: Path | None = None
    if voices_source.is_file():
        voices_path, _, _ = ensure_movement_score(
            project,
            movement_id=movement_id,
            start_seconds=start_seconds,
            end_seconds=end_seconds,
            anchors=anchors,
            title=title,
            local=local,
            source=voices_source,
            cache_suffix=".voices",
        )
    return score_path, voices_path, first, last


_warm_lock = threading.Lock()
_warm_in_flight: set[str] = set()


def warm_project_movements(
    project: Path,
    *,
    movements: list[dict],
    anchors: list[MeasureAnchor],
    skip_id: str = "",
    local: Path | None = None,
) -> None:
    """Cache sibling movement MusicXML (and preview WAV) in the background."""

    project = Path(project)
    key = str(project.resolve())
    with _warm_lock:
        if key in _warm_in_flight:
            return
        _warm_in_flight.add(key)

    def worker() -> None:
        try:
            for row in movements:
                movement_id = str(row.get("id") or "").strip()
                if not movement_id or movement_id == skip_id:
                    continue
                title = str(row.get("title") or movement_id)
                start = float(row["start_seconds"])
                end = float(row["end_seconds"])
                try:
                    ensure_movement_scores(
                        project,
                        movement_id=movement_id,
                        start_seconds=start,
                        end_seconds=end,
                        anchors=anchors,
                        title=title,
                        local=local,
                    )
                except Exception as exc:  # noqa: BLE001 — keep warming other sections
                    logger.info("Movement score warm failed for %s/%s: %s", project.name, movement_id, exc)
                    continue
                try:
                    from baroque_transcribe.preview_audio import ensure_movement_preview_wav

                    ensure_movement_preview_wav(
                        project,
                        movement_id=movement_id,
                        start_seconds=start,
                        end_seconds=end,
                        local=local,
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.info("Movement audio warm failed for %s/%s: %s", project.name, movement_id, exc)
        finally:
            with _warm_lock:
                _warm_in_flight.discard(key)

    threading.Thread(target=worker, name=f"warm-movements-{project.name}", daemon=True).start()


def remap_movement_timeline(
    *,
    anchors: list[MeasureAnchor],
    note_anchors: list[NoteAnchor],
    start_seconds: float,
    end_seconds: float,
    source_first: int,
    source_last: int,
) -> tuple[list[MeasureAnchor], list[NoteAnchor], float]:
    """Shift suite times to movement-local time and renumber measures from 1."""

    duration = max(0.0, end_seconds - start_seconds)
    mapping = {
        measure: index
        for index, measure in enumerate(range(source_first, source_last + 1), start=1)
    }
    new_anchors: list[MeasureAnchor] = []
    for anchor in anchors:
        if anchor.measure not in mapping:
            continue
        if anchor.time + 1e-6 >= end_seconds:
            continue
        if anchor.time + 1e-6 < start_seconds and anchor.measure != source_first:
            continue
        new_anchors.append(
            MeasureAnchor(
                measure=mapping[anchor.measure],
                time=max(0.0, float(anchor.time) - start_seconds),
                collided=anchor.collided,
                opens_before_attack=anchor.opens_before_attack,
            )
        )
    if not new_anchors:
        new_anchors = [
            MeasureAnchor(measure=1, time=0.0, collided=False, opens_before_attack=False)
        ]
    new_notes: list[NoteAnchor] = []
    for note in note_anchors:
        if note.onset + 1e-6 >= end_seconds:
            continue
        if note.offset <= start_seconds + 1e-6:
            continue
        measure = mapping.get(note.measure)
        if measure is None:
            # Keep notes that sound inside the window even if measure id was edged.
            measure = 1
        new_notes.append(
            NoteAnchor(
                source_index=note.source_index,
                pitch=note.pitch,
                onset=max(0.0, float(note.onset) - start_seconds),
                offset=max(0.05, float(note.offset) - start_seconds),
                measure=measure,
            )
        )
    return new_anchors, new_notes, duration
