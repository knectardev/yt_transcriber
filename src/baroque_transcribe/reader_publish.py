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
)
from baroque_transcribe.defaults import SCORE_VOICES_XML, SCORE_XML
from baroque_transcribe.parse_store import repo_root

READER_DIR = "reader"
CATALOG_INDEX = "catalog/index.json"
CATALOG_PACKAGES = "catalog/packages"
OSMD_NAME = "opensheetmusicdisplay.min.js"
INDEX_SCHEMA_VERSION = 1


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


def _package_id(entry_id: str) -> str:
    return f"catalog:{entry_id}"


def _content_hash(musicxml: str, musicxml_voices: str | None) -> str:
    digest = hashlib.sha256()
    digest.update(musicxml.encode("utf-8"))
    if musicxml_voices:
        digest.update(b"\0")
        digest.update(musicxml_voices.encode("utf-8"))
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


def publish_scores(
    entry_ids: list[str],
    *,
    dist_root: Path | None = None,
    root: Path | None = None,
) -> PublishResult:
    """Add or refresh packages for ``entry_ids`` without removing other published scores."""

    if not entry_ids:
        raise ReaderPublishError("Provide at least one catalog entry id.")
    base = Path(dist_root) if dist_root is not None else default_dist_root(root)
    base.mkdir(parents=True, exist_ok=True)
    ensure_shell_assets(base, root=root)
    index = _load_index(base)
    by_id = {str(row.get("entryId")): row for row in index.get("entries") or [] if row.get("entryId")}
    result = PublishResult()

    for raw_id in entry_ids:
        entry_id = str(raw_id).strip()
        if not entry_id:
            continue
        try:
            package = _build_package(entry_id, root=root)
            package_name = f"{entry_id}.json"
            package_path = Path(CATALOG_PACKAGES) / package_name
            abs_package = base / package_path
            abs_package.parent.mkdir(parents=True, exist_ok=True)
            previous = by_id.get(entry_id)
            if previous and previous.get("contentHash") == package["contentHash"] and abs_package.is_file():
                result.unchanged.append(entry_id)
                continue
            abs_package.write_text(json.dumps(package, ensure_ascii=False) + "\n", encoding="utf-8")
            row = {
                "id": package["id"],
                "entryId": package["entryId"],
                "bwv": package["bwv"],
                "title": package["title"],
                "displayTitle": package["displayTitle"],
                "collection": package["collection"],
                "packagePath": package_path.as_posix(),
                "contentHash": package["contentHash"],
                "publishedAt": package["publishedAt"],
                "attribution": package["attribution"],
            }
            if previous:
                result.updated.append(entry_id)
            else:
                result.added.append(entry_id)
            by_id[entry_id] = row
        except (CatalogError, ReaderPublishError, OSError, json.JSONDecodeError) as exc:
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
        result.removed.append(entry_id)
    missing = sorted(wanted - set(result.removed))
    for entry_id in missing:
        result.errors.append((entry_id, "Not present in published catalog."))
    index["entries"] = kept
    _write_index(base, index)
    return result
