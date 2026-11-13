"""Curated / draft movement partitions for multi-movement catalog MIDIs.

Tobis ships one MIDI per suite without movement names. Boundaries are drafted
from tempo and meter change points, then labeled from standard Bach suite
orders. ``status: draft`` means a human must approve before treating timings
as final practice targets.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import pretty_midi

from baroque_transcribe.defaults import CATALOG_DATA_DIR
from baroque_transcribe.parse_store import repo_root

MOVEMENTS_DRAFT_FILE = "movements-draft.json"
_MOVEMENT_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")

# Best-effort titles when the tempo-map section count matches a known layout.
# Nested dances that share one tempo/meter block are named as a group.
_ENGLISH_SUITE_TITLES = {
    6: (
        "Prélude",
        "Allemande",
        "Courantes (I, II & Doubles)",
        "Sarabande",
        "Bourrées (I & II)",
        "Gigue",
    ),
    7: (
        "Prélude",
        "Allemande",
        "Courante",
        "Sarabande",
        "Bourrée I",
        "Bourrée II",
        "Gigue",
    ),
}
_FRENCH_SUITE_TITLES = {
    5: ("Allemande", "Courante", "Sarabande", "Menuet / Gavotte / Bourrée", "Gigue"),
    6: ("Allemande", "Courante", "Sarabande", "Menuet / Gavotte", "Bourrée / Loure", "Gigue"),
    7: ("Allemande", "Courante", "Sarabande", "Menuet I", "Menuet II", "Bourrée / Gavotte", "Gigue"),
}
_PARTITA_TITLES = {
    4: ("Praeludium / Sinfonia", "Allemande", "Courante / Sarabande", "Gigue / Capriccio"),
    5: ("Praeludium", "Allemande", "Courante", "Sarabande", "Gigue"),
    6: ("Praeludium", "Allemande", "Courante", "Sarabande", "Menuet / Burlesca / Scherzo", "Gigue"),
    7: ("Praeludium", "Allemande", "Courante", "Sarabande", "Menuet I", "Menuet II", "Gigue"),
}


@dataclass(frozen=True)
class Movement:
    id: str
    title: str
    start_seconds: float
    end_seconds: float
    source: str = "tempo_map"


@dataclass(frozen=True)
class EntryMovements:
    entry_id: str
    status: str
    movements: tuple[Movement, ...]
    notes: str = ""

    @property
    def approved(self) -> bool:
        return self.status == "approved"

    @property
    def is_draft(self) -> bool:
        return self.status == "draft"


def movements_draft_path(root: Path | None = None) -> Path:
    base = Path(root) if root is not None else repo_root()
    return base / CATALOG_DATA_DIR / MOVEMENTS_DRAFT_FILE


def load_movements_overlay(path: Path | None = None, root: Path | None = None) -> dict[str, EntryMovements]:
    target = Path(path) if path is not None else movements_draft_path(root)
    if not target.is_file():
        return {}
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Movements file must be a JSON object: {target}")
    default_status = str(payload.get("status") or "draft").strip() or "draft"
    entries_raw = payload.get("entries") or {}
    if not isinstance(entries_raw, dict):
        raise ValueError("Movements entries must be an object keyed by catalog id.")
    out: dict[str, EntryMovements] = {}
    for entry_id, body in entries_raw.items():
        if not isinstance(body, dict):
            continue
        status = str(body.get("status") or default_status).strip() or default_status
        notes = str(body.get("notes") or "").strip()
        movements = tuple(_parse_movement(item, index=index) for index, item in enumerate(body.get("movements") or []))
        if not movements:
            continue
        out[str(entry_id)] = EntryMovements(
            entry_id=str(entry_id),
            status=status,
            movements=movements,
            notes=notes,
        )
    return out


def movements_for_entry(
    entry_id: str,
    *,
    path: Path | None = None,
    root: Path | None = None,
    include_draft: bool = True,
) -> EntryMovements | None:
    overlay = load_movements_overlay(path=path, root=root)
    row = overlay.get(entry_id)
    if row is None:
        return None
    if row.is_draft and not include_draft:
        return None
    return row


def movements_as_dicts(row: EntryMovements) -> list[dict]:
    return [
        {
            "id": movement.id,
            "title": movement.title,
            "start_seconds": movement.start_seconds,
            "end_seconds": movement.end_seconds,
            "source": movement.source,
        }
        for movement in row.movements
    ]


def propose_movements_from_midi(
    midi_path: Path,
    *,
    entry_id: str,
    collection_id: str = "",
) -> EntryMovements:
    """Build a draft partition from tempo (and meter) change points."""

    pm = pretty_midi.PrettyMIDI(str(midi_path))
    end = float(pm.get_end_time())
    times, _tempos = pm.get_tempo_changes()
    starts = [float(t) for t in times]
    if not starts or starts[0] > 0.05:
        starts = [0.0] + starts
    # Deduplicate near-equal starts.
    cleaned: list[float] = []
    for start in starts:
        if not cleaned or abs(start - cleaned[-1]) > 0.05:
            cleaned.append(max(0.0, start))
    if cleaned[-1] >= end - 0.05:
        cleaned = cleaned[:-1]
    if not cleaned:
        cleaned = [0.0]
    ends = cleaned[1:] + [end]
    titles = _titles_for(collection_id, len(cleaned))
    movements: list[Movement] = []
    for index, (start, stop) in enumerate(zip(cleaned, ends)):
        title = titles[index] if index < len(titles) else f"Section {index + 1}"
        movements.append(
            Movement(
                id=_slug(title, index),
                title=title,
                start_seconds=round(start, 3),
                end_seconds=round(float(stop), 3),
                source="tempo_map",
            )
        )
    notes = (
        f"Draft from tempo/meter map ({len(movements)} sections). "
        "Nested dances that share one tempo block are grouped. Pending human review."
    )
    return EntryMovements(entry_id=entry_id, status="draft", movements=tuple(movements), notes=notes)


def build_draft_overlay(
    *,
    root: Path | None = None,
    collection_ids: tuple[str, ...] = (
        "bwv-806-811-english-suites",
        "bwv-812-817-french-suites",
        "bwv-825-831-partitas",
    ),
) -> dict:
    """Scan local MIDI for selected collections and return a draft JSON payload."""

    from baroque_transcribe.catalog import load_catalog, entry_midi_path

    catalog = load_catalog(root=root)
    entries: dict[str, dict] = {}
    for entry in catalog.entries:
        if entry.collection_id not in collection_ids:
            continue
        midi = entry_midi_path(entry.id, root=root)
        if not midi.is_file():
            continue
        row = propose_movements_from_midi(midi, entry_id=entry.id, collection_id=entry.collection_id)
        entries[entry.id] = {
            "status": "draft",
            "notes": row.notes,
            "movements": movements_as_dicts(row),
        }
    return {
        "schema_version": 1,
        "status": "draft",
        "review_required": True,
        "notes": (
            "Draft movement partitions for multi-movement Bach suites. "
            "Approve after listening before treating times as final."
        ),
        "entries": entries,
    }


def write_draft_overlay(payload: dict, *, root: Path | None = None) -> Path:
    target = movements_draft_path(root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target


def review_markdown(payload: dict) -> str:
    lines = [
        "# Bach suite movement draft — review",
        "",
        payload.get("notes") or "",
        "",
        "Status: **draft** (not approved).",
        "",
        "Please confirm:",
        "1. Tab labels / grouping look right (including Courantes & Bourrées).",
        "2. Start/end times are accurate enough to practice against.",
        "",
    ]
    entries = payload.get("entries") or {}
    for entry_id, body in entries.items():
        movements = body.get("movements") or []
        lines.append(f"## {entry_id}")
        if body.get("notes"):
            lines.append(body["notes"])
            lines.append("")
        lines.append("| # | Id | Title | Start | End |")
        lines.append("| --- | --- | --- | ---: | ---: |")
        for index, movement in enumerate(movements, start=1):
            lines.append(
                f"| {index} | `{movement.get('id')}` | {movement.get('title')} | "
                f"{float(movement.get('start_seconds') or 0):.1f}s | "
                f"{float(movement.get('end_seconds') or 0):.1f}s |"
            )
        lines.append("")
    lines.append("After approval, set each entry (or the file) `status` to `approved`.")
    lines.append("")
    return "\n".join(lines)


def _titles_for(collection_id: str, count: int) -> tuple[str, ...]:
    if "english-suites" in collection_id:
        return _ENGLISH_SUITE_TITLES.get(count, ())
    if "french-suites" in collection_id:
        return _FRENCH_SUITE_TITLES.get(count, ())
    if "partitas" in collection_id:
        return _PARTITA_TITLES.get(count, ())
    return ()


def _slug(title: str, index: int) -> str:
    import unicodedata

    normalized = unicodedata.normalize("NFKD", title)
    ascii_title = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    base = re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-")
    base = base or f"section-{index + 1}"
    if not _MOVEMENT_ID_RE.match(base):
        base = f"section-{index + 1}"
    return base[:48]


def _parse_movement(payload: object, *, index: int) -> Movement:
    if not isinstance(payload, dict):
        raise ValueError(f"Movement {index} must be an object.")
    movement_id = str(payload.get("id") or "").strip() or f"section-{index + 1}"
    if not _MOVEMENT_ID_RE.match(movement_id):
        raise ValueError(f"Movement id is invalid: {movement_id}")
    title = str(payload.get("title") or "").strip() or movement_id
    start = float(payload.get("start_seconds"))
    end = float(payload.get("end_seconds"))
    if end <= start:
        raise ValueError(f"Movement {movement_id} end must be after start.")
    return Movement(
        id=movement_id,
        title=title,
        start_seconds=start,
        end_seconds=end,
        source=str(payload.get("source") or "curated").strip() or "curated",
    )
