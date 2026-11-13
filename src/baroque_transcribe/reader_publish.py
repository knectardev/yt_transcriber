"""Incremental publish of read-only score packages into the ``reader/`` partition.

The desktop library remains authoritative. Vercel serves only ``reader/``.
These functions are the stable API; the CLI is a thin caller for v1.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from baroque_transcribe.catalog import (
    CatalogError,
    catalog_provenance,
    find_project_for_entry,
    get_entry,
    load_catalog,
    search_catalog,
)
from baroque_transcribe.defaults import SCORE_VOICES_XML, SCORE_XML
from baroque_transcribe.parse_store import repo_root

READER_DIR = "reader"
CATALOG_INDEX = "catalog/index.json"
CATALOG_LIBRARY = "catalog/library.json"
CATALOG_PACKAGES = "catalog/packages"
OSMD_NAME = "opensheetmusicdisplay.min.js"
INDEX_SCHEMA_VERSION = 1
LIBRARY_SCHEMA_VERSION = 1


class ReaderPublishError(RuntimeError):
    """Publish or remove failed for a reader package."""


@dataclass
class PublishResult:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def reader_root(root: Path | None = None) -> Path:
    base = Path(root) if root is not None else repo_root()
    return base / READER_DIR


def default_dist_root(root: Path | None = None) -> Path:
    """Deployable static tree. Same as ``reader/`` for the Git-tracked v1 layout."""

    return reader_root(root)


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _package_id(entry_id: str, movement_id: str | None = None) -> str:
    if movement_id:
        return f"catalog:{entry_id}:{movement_id}"
    return f"catalog:{entry_id}"


def _section_package_name(entry_id: str, movement_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in movement_id).strip("-") or "section"
    return f"{entry_id}__{safe}.json"


def _content_hash(musicxml: str | None, musicxml_voices: str | None) -> str:
    digest = hashlib.sha256()
    digest.update((musicxml or "").encode("utf-8"))
    if musicxml_voices:
        digest.update(b"\0")
        digest.update(musicxml_voices.encode("utf-8"))
    return digest.hexdigest()


def _sections_hash(section_rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in section_rows:
        digest.update(str(row.get("id") or "").encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.get("contentHash") or "").encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_index(dist_root: Path) -> dict:
    path = dist_root / CATALOG_INDEX
    if not path.is_file():
        return {"schemaVersion": INDEX_SCHEMA_VERSION, "updatedAt": _now(), "entries": []}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ReaderPublishError(f"Invalid catalog index: {path}")
    entries = data.get("entries")
    if entries is None:
        entries = []
    if not isinstance(entries, list):
        raise ReaderPublishError(f"catalog index entries must be a list: {path}")
    data["entries"] = entries
    data.setdefault("schemaVersion", INDEX_SCHEMA_VERSION)
    return data


def _write_index(dist_root: Path, index: dict) -> None:
    index = dict(index)
    index["schemaVersion"] = INDEX_SCHEMA_VERSION
    index["updatedAt"] = _now()
    entries = list(index.get("entries") or [])
    entries.sort(key=lambda row: (str(row.get("bwv") or ""), str(row.get("entryId") or "")))
    index["entries"] = entries
    path = dist_root / CATALOG_INDEX
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def list_published(dist_root: Path | None = None, *, root: Path | None = None) -> list[dict]:
    base = Path(dist_root) if dist_root is not None else default_dist_root(root)
    return list(_load_index(base).get("entries") or [])


def ensure_shell_assets(dist_root: Path | None = None, *, root: Path | None = None) -> Path:
    """Copy OSMD into ``reader/vendor`` from the desktop vendor tree."""

    base = Path(dist_root) if dist_root is not None else default_dist_root(root)
    repo = Path(root) if root is not None else repo_root()
    source = Path(__file__).resolve().parent / "vendor" / OSMD_NAME
    if not source.is_file():
        alt = repo / "src" / "baroque_transcribe" / "vendor" / OSMD_NAME
        source = alt if alt.is_file() else source
    if not source.is_file():
        raise ReaderPublishError(f"Missing OSMD vendor file: {source}")
    dest_dir = base / "vendor"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / OSMD_NAME
    shutil.copy2(source, dest)
    (base / CATALOG_PACKAGES).mkdir(parents=True, exist_ok=True)
    return dest


def write_catalog_library(dist_root: Path | None = None, *, root: Path | None = None) -> Path:
    """Write lightweight metadata for the full Bach catalog (no MusicXML)."""

    base = Path(dist_root) if dist_root is not None else default_dist_root(root)
    catalog = load_catalog(root=root)
    entries = [
        {
            "id": f"catalog:{entry.id}",
            "entryId": entry.id,
            "bwv": entry.bwv,
            "title": entry.title,
            "displayTitle": entry.display_title,
            "collection": entry.collection,
            "collectionId": entry.collection_id,
            "key": entry.key,
            "isVariant": bool(entry.is_variant),
            "isArrangement": bool(entry.is_arrangement),
        }
        for entry in catalog.entries
    ]
    payload = {
        "schemaVersion": LIBRARY_SCHEMA_VERSION,
        "updatedAt": _now(),
        "entryCount": len(entries),
        "attribution": "MIDI files from Tobis Notenarchiv (tobis-notenarchiv.de).",
        "entries": entries,
    }
    path = base / CATALOG_LIBRARY
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def resolve_publish_ids(
    *,
    entry_ids: list[str] | None = None,
    collection_id: str = "",
    root: Path | None = None,
) -> list[str]:
    """Resolve ``--id`` and/or ``--collection`` into catalog entry ids."""

    ids: list[str] = []
    seen: set[str] = set()
    for raw in entry_ids or []:
        entry_id = str(raw).strip()
        if entry_id and entry_id not in seen:
            ids.append(entry_id)
            seen.add(entry_id)
    collection = str(collection_id or "").strip()
    if collection:
        catalog = load_catalog(root=root)
        for row in search_catalog(catalog, collection_id=collection):
            entry_id = row.entry.id
            if entry_id not in seen:
                ids.append(entry_id)
                seen.add(entry_id)
    return ids


def _resolve_musicxml_texts(entry_id: str, *, root: Path | None = None) -> tuple[str, str | None, str | None]:
    """Return ``(musicxml, musicxml_voices_or_none, project_name_or_none)``."""

    project = find_project_for_entry(entry_id, root=root)
    if project is not None:
        voices = project / SCORE_VOICES_XML
        plain = project / SCORE_XML
        if voices.is_file():
            text = voices.read_text(encoding="utf-8")
            plain_text = plain.read_text(encoding="utf-8") if plain.is_file() else text
            return plain_text, text, project.name
        if plain.is_file():
            text = plain.read_text(encoding="utf-8")
            return text, None, project.name
        raise ReaderPublishError(
            f"Project {project.name} for {entry_id} has no {SCORE_XML} or {SCORE_VOICES_XML}."
        )

    from baroque_transcribe.catalog import entry_midi_path
    from baroque_transcribe.notation import NotationError, musicxml_for_midi

    midi = entry_midi_path(entry_id, root=root)
    if not midi.is_file():
        raise ReaderPublishError(
            f"No Library project or local MIDI for {entry_id}. "
            "Open or preprocess the catalog entry on the desktop first."
        )
    try:
        score_path = musicxml_for_midi(midi)
    except NotationError as exc:
        raise ReaderPublishError(str(exc)) from exc
    text = score_path.read_text(encoding="utf-8")
    return text, None, None


def _build_section_packages(entry_id: str, *, root: Path | None = None) -> list[dict]:
    """Slice multi-movement works into separate published packages."""

    from baroque_transcribe.catalog_movements import movements_for_entry
    from baroque_transcribe.movement_scores import MovementScoreError, ensure_movement_scores
    from baroque_transcribe.playback import load_playback

    movements = movements_for_entry(entry_id, root=root, include_draft=True)
    if movements is None or len(movements.movements) < 2:
        return []
    project = find_project_for_entry(entry_id, root=root)
    if project is None:
        return []
    entry = get_entry(entry_id, root=root)
    context = load_playback(project)
    provenance = catalog_provenance(entry)
    packages: list[dict] = []
    for movement in movements.movements:
        try:
            score_path, voices_path, _first, _last = ensure_movement_scores(
                project,
                movement_id=movement.id,
                start_seconds=float(movement.start_seconds),
                end_seconds=float(movement.end_seconds),
                anchors=context.anchors,
                title=str(movement.title or movement.id),
            )
        except (MovementScoreError, OSError, ValueError) as exc:
            raise ReaderPublishError(f"{entry_id}/{movement.id}: {exc}") from exc
        musicxml = score_path.read_text(encoding="utf-8")
        voices = voices_path.read_text(encoding="utf-8") if voices_path and voices_path.is_file() else None
        packages.append(
            {
                "id": _package_id(entry.id, movement.id),
                "entryId": entry.id,
                "movementId": movement.id,
                "parentId": _package_id(entry.id),
                "bwv": entry.bwv,
                "title": movement.title,
                "displayTitle": f"{entry.display_title} — {movement.title}",
                "collection": entry.collection,
                "collectionId": entry.collection_id,
                "attribution": provenance.get("attribution") or "",
                "musicxml": musicxml,
                "musicxmlVoices": voices,
                "contentHash": _content_hash(musicxml, voices),
                "publishedAt": _now(),
                "sectionsStatus": movements.status,
                "source": {
                    "kind": "catalog-section",
                    "entryId": entry.id,
                    "movementId": movement.id,
                    "project": project.name,
                    "startSeconds": movement.start_seconds,
                    "endSeconds": movement.end_seconds,
                },
            }
        )
    return packages


def _build_package(entry_id: str, *, root: Path | None = None) -> dict:
    entry = get_entry(entry_id, root=root)
    musicxml, voices, project_name = _resolve_musicxml_texts(entry_id, root=root)
    content_hash = _content_hash(musicxml, voices)
    provenance = catalog_provenance(entry)
    return {
        "id": _package_id(entry.id),
        "entryId": entry.id,
        "bwv": entry.bwv,
        "title": entry.title,
        "displayTitle": entry.display_title,
        "collection": entry.collection,
        "collectionId": entry.collection_id,
        "attribution": provenance.get("attribution") or "",
        "musicxml": musicxml,
        "musicxmlVoices": voices,
        "contentHash": content_hash,
        "publishedAt": _now(),
        "source": {
            "kind": "catalog",
            "entryId": entry.id,
            "project": project_name,
            "midiUrl": entry.midi_url,
            "sourcePage": entry.source_page,
        },
    }


def _write_package_file(base: Path, package: dict, filename: str) -> Path:
    package_path = Path(CATALOG_PACKAGES) / filename
    abs_package = base / package_path
    abs_package.parent.mkdir(parents=True, exist_ok=True)
    abs_package.write_text(json.dumps(package, ensure_ascii=False) + "\n", encoding="utf-8")
    return package_path


def _index_row_from_package(package: dict, package_path: Path, *, sections: list[dict] | None = None) -> dict:
    row = {
        "id": package["id"],
        "entryId": package["entryId"],
        "bwv": package["bwv"],
        "title": package["title"],
        "displayTitle": package["displayTitle"],
        "collection": package["collection"],
        "collectionId": package.get("collectionId") or "",
        "packagePath": package_path.as_posix(),
        "contentHash": package["contentHash"],
        "publishedAt": package["publishedAt"],
        "attribution": package["attribution"],
    }
    if sections is not None:
        row["sections"] = sections
        row["sectionsStatus"] = package.get("sectionsStatus") or ""
    return row


def publish_scores(
    entry_ids: list[str],
    *,
    dist_root: Path | None = None,
    root: Path | None = None,
    write_library: bool = True,
) -> PublishResult:
    """Add or refresh packages for ``entry_ids`` without removing other published scores."""

    if not entry_ids:
        raise ReaderPublishError("Provide at least one catalog entry id.")
    base = Path(dist_root) if dist_root is not None else default_dist_root(root)
    base.mkdir(parents=True, exist_ok=True)
    ensure_shell_assets(base, root=root)
    if write_library:
        write_catalog_library(base, root=root)
    index = _load_index(base)
    by_id = {str(row.get("entryId")): row for row in index.get("entries") or [] if row.get("entryId")}
    result = PublishResult()

    for raw_id in entry_ids:
        entry_id = str(raw_id).strip()
        if not entry_id:
            continue
        try:
            section_packages = _build_section_packages(entry_id, root=root)
            if section_packages:
                section_rows: list[dict] = []
                for section in section_packages:
                    filename = _section_package_name(entry_id, str(section["movementId"]))
                    package_path = _write_package_file(base, section, filename)
                    section_rows.append(
                        {
                            "id": section["movementId"],
                            "title": section["title"],
                            "packagePath": package_path.as_posix(),
                            "contentHash": section["contentHash"],
                            "displayTitle": section["displayTitle"],
                        }
                    )
                # Parent stub: avoid embedding multi‑MB full-suite MusicXML.
                entry = get_entry(entry_id, root=root)
                provenance = catalog_provenance(entry)
                parent_hash = _sections_hash(section_rows)
                parent = {
                    "id": _package_id(entry.id),
                    "entryId": entry.id,
                    "bwv": entry.bwv,
                    "title": entry.title,
                    "displayTitle": entry.display_title,
                    "collection": entry.collection,
                    "collectionId": entry.collection_id,
                    "attribution": provenance.get("attribution") or "",
                    "musicxml": None,
                    "musicxmlVoices": None,
                    "contentHash": parent_hash,
                    "publishedAt": _now(),
                    "sections": section_rows,
                    "sectionsStatus": section_packages[0].get("sectionsStatus") or "",
                    "source": {
                        "kind": "catalog",
                        "entryId": entry.id,
                        "project": (section_packages[0].get("source") or {}).get("project"),
                        "midiUrl": entry.midi_url,
                        "sourcePage": entry.source_page,
                    },
                }
                package_path = _write_package_file(base, parent, f"{entry_id}.json")
                previous = by_id.get(entry_id)
                if previous and previous.get("contentHash") == parent_hash:
                    result.unchanged.append(entry_id)
                elif previous:
                    result.updated.append(entry_id)
                else:
                    result.added.append(entry_id)
                by_id[entry_id] = _index_row_from_package(parent, package_path, sections=section_rows)
                continue

            package = _build_package(entry_id, root=root)
            package_path = Path(CATALOG_PACKAGES) / f"{entry_id}.json"
            abs_package = base / package_path
            previous = by_id.get(entry_id)
            if previous and previous.get("contentHash") == package["contentHash"] and abs_package.is_file():
                result.unchanged.append(entry_id)
                continue
            _write_package_file(base, package, f"{entry_id}.json")
            if previous:
                result.updated.append(entry_id)
            else:
                result.added.append(entry_id)
            by_id[entry_id] = _index_row_from_package(package, package_path)
        except (CatalogError, ReaderPublishError, OSError, json.JSONDecodeError, ValueError) as exc:
            result.errors.append((entry_id, str(exc)))

    index["entries"] = list(by_id.values())
    _write_index(base, index)
    return result


def remove_published(
    entry_ids: list[str],
    *,
    dist_root: Path | None = None,
    root: Path | None = None,
) -> PublishResult:
    """Remove selected published entries. Other entries are preserved."""

    if not entry_ids:
        raise ReaderPublishError("Provide at least one catalog entry id to remove.")
    base = Path(dist_root) if dist_root is not None else default_dist_root(root)
    index = _load_index(base)
    wanted = {str(item).strip() for item in entry_ids if str(item).strip()}
    result = PublishResult()
    kept: list[dict] = []
    for row in index.get("entries") or []:
        entry_id = str(row.get("entryId") or "")
        if entry_id not in wanted:
            kept.append(row)
            continue
        package_rel = str(row.get("packagePath") or f"{CATALOG_PACKAGES}/{entry_id}.json")
        package_path = base / package_rel
        if package_path.is_file():
            package_path.unlink()
        for section in row.get("sections") or []:
            section_rel = str(section.get("packagePath") or "")
            if section_rel:
                section_path = base / section_rel
                if section_path.is_file():
                    section_path.unlink()
        result.removed.append(entry_id)
    missing = sorted(wanted - set(result.removed))
    for entry_id in missing:
        result.errors.append((entry_id, "Not present in published catalog."))
    index["entries"] = kept
    _write_index(base, index)
    return result
