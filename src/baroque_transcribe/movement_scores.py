"""Cached per-movement MusicXML slices for multi-movement catalog projects.

Each movement opens as a standalone score (measures renumbered from 1) under
the suite's BWV page. Absolute suite times are converted to movement-local
time (0 at the section start) for playback.
"""

from __future__ import annotations

from pathlib import Path

from baroque_transcribe.defaults import LOCAL_DIR, SCORE_XML
from baroque_transcribe.parse_store import repo_root
from baroque_transcribe.sync_map import MeasureAnchor, NoteAnchor


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


def ensure_movement_score(
    project: Path,
    *,
    movement_id: str,
    start_seconds: float,
    end_seconds: float,
    anchors: list[MeasureAnchor],
    title: str = "",
    local: Path | None = None,
) -> tuple[Path, int, int]:
    """Write (or reuse) a standalone MusicXML for one movement.

    Measures are renumbered from 1. Returns ``(path, source_first, source_last)``.
    """

    project = Path(project)
    source = project / SCORE_XML
    if not source.is_file():
        raise MovementScoreError(f"No score to slice at {source}.")
    if end_seconds <= start_seconds:
        raise MovementScoreError("Movement end must be after start.")

    first, last = measures_for_time_span(anchors, start_seconds, end_seconds)
    safe_id = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in movement_id).strip("-") or "section"
    cache = movement_scores_root(local) / project.name
    cache.mkdir(parents=True, exist_ok=True)
    dest = cache / f"{safe_id}.musicxml"
    stamp = cache / f"{safe_id}.stamp"
    token = f"{source.stat().st_mtime_ns}:{first}:{last}:renum1-parts:{title}"
    if dest.is_file() and stamp.is_file() and stamp.read_text(encoding="utf-8") == token:
        return dest, first, last

    from music21 import converter

    score = converter.parse(str(source))
    try:
        excerpt = score.measures(first, last)
    except Exception as exc:
        raise MovementScoreError(f"Could not slice measures {first}-{last}: {exc}") from exc

    # Standalone section: each part restarts measure numbers at 1 so
    # treble/bass (and any other staves) stay aligned for OSMD highlighting.
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
        stamp.write_text(token, encoding="utf-8")
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return dest, first, last


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
