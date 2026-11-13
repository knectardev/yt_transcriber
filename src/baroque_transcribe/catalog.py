"""Bach keyboard MIDI catalog: committed metadata and local file status.

The checked-in JSON under ``data/catalog/`` is authoritative for titles, BWV
numbers, source URLs, and pinned hashes. Downloaded MIDI bytes live under
``catalog/midi/<entry-id>/source.mid`` and are never committed. Opening an
entry materializes a normal Library project through ``import_midi_project``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from baroque_transcribe.defaults import (
    CATALOG_DATA_DIR,
    CATALOG_MANIFEST,
    CATALOG_MIDI_DIR,
    MANIFEST,
    SOURCE_MID,
)
from baroque_transcribe.parse_store import repo_root
from baroque_transcribe.storage_io import sha256_file

CATALOG_SCHEMA_VERSION = 1
_ENTRY_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class CatalogError(ValueError):
    pass


@dataclass(frozen=True)
class CatalogSource:
    name: str
    home_url: str
    keyboard_index_url: str
    terms_en_url: str
    terms_de_url: str
    license_notes: str
    attribution: str
    retrieved_at: str
    allowed_hosts: tuple[str, ...]
    collections: tuple[dict, ...]


@dataclass(frozen=True)
class CatalogEntry:
    id: str
    bwv: str
    title: str
    collection_id: str
    collection: str
    order_in_collection: int
    key: str = ""
    is_variant: bool = False
    is_arrangement: bool = False
    source_page: str = ""
    midi_url: str = ""
    source_filename: str = ""
    sha256: str = ""
    size_bytes: int | None = None
    duration_seconds: float | None = None
    note_count: int | None = None
    track_count: int | None = None
    import_defaults: dict = field(default_factory=dict)

    @property
    def display_title(self) -> str:
        if self.bwv:
            return f"BWV {self.bwv} — {self.title}".strip(" —")
        return self.title

    @property
    def search_blob(self) -> str:
        parts = [
            self.id,
            self.bwv,
            f"bwv {self.bwv}" if self.bwv else "",
            self.title,
            self.collection,
            self.collection_id,
            self.key,
            self.source_filename,
            "variant" if self.is_variant else "",
            "arrangement" if self.is_arrangement else "",
        ]
        return " ".join(part for part in parts if part).lower()


@dataclass(frozen=True)
class CatalogManifest:
    schema_version: int
    id: str
    title: str
    composer: str
    source: CatalogSource
    entries: tuple[CatalogEntry, ...]

    @property
    def entry_count(self) -> int:
        return len(self.entries)


@dataclass(frozen=True)
class EntryStatus:
    entry: CatalogEntry
    local: bool
    local_path: Path | None
    local_sha256: str = ""
    hash_ok: bool | None = None
    imported_projects: tuple[str, ...] = ()
    fragment_count: int = 0

    @property
    def status_label(self) -> str:
        if not self.local:
            return "not downloaded"
        if self.hash_ok is False:
            return "hash mismatch"
        if self.imported_projects:
            return "imported"
        return "downloaded"


def catalog_manifest_path(root: Path | None = None) -> Path:
    base = Path(root) if root is not None else repo_root()
    return base / CATALOG_DATA_DIR / CATALOG_MANIFEST


def catalog_midi_root(root: Path | None = None) -> Path:
    base = Path(root) if root is not None else repo_root()
    return base / CATALOG_MIDI_DIR


def entry_midi_path(entry_id: str, root: Path | None = None) -> Path:
    return catalog_midi_root(root) / str(entry_id) / SOURCE_MID


def load_catalog(path: Path | None = None, root: Path | None = None) -> CatalogManifest:
    target = Path(path) if path is not None else catalog_manifest_path(root)
    if not target.is_file():
        raise CatalogError(f"Catalog manifest not found: {target}")
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CatalogError(f"Could not read catalog manifest: {exc}") from exc
    return validate_catalog(payload)


def validate_catalog(payload: object) -> CatalogManifest:
    if not isinstance(payload, dict):
        raise CatalogError("Catalog manifest must be a JSON object.")
    version = payload.get("schema_version")
    try:
        schema_version = int(version)
    except (TypeError, ValueError) as exc:
        raise CatalogError("Catalog schema_version must be an integer.") from exc
    if schema_version != CATALOG_SCHEMA_VERSION:
        raise CatalogError(
            f"Catalog uses schema version {schema_version}; "
            f"this program reads version {CATALOG_SCHEMA_VERSION}."
        )
    source_raw = payload.get("source")
    if not isinstance(source_raw, dict):
        raise CatalogError("Catalog source block is required.")
    hosts = source_raw.get("allowed_hosts") or []
    if not isinstance(hosts, list) or not all(isinstance(item, str) and item for item in hosts):
        raise CatalogError("Catalog source.allowed_hosts must be a list of host names.")
    collections = source_raw.get("collections") or []
    if not isinstance(collections, list):
        raise CatalogError("Catalog source.collections must be a list.")
    source = CatalogSource(
        name=str(source_raw.get("name") or "").strip(),
        home_url=str(source_raw.get("home_url") or "").strip(),
        keyboard_index_url=str(source_raw.get("keyboard_index_url") or "").strip(),
        terms_en_url=str(source_raw.get("terms_en_url") or "").strip(),
        terms_de_url=str(source_raw.get("terms_de_url") or "").strip(),
        license_notes=str(source_raw.get("license_notes") or "").strip(),
        attribution=str(source_raw.get("attribution") or "").strip(),
        retrieved_at=str(source_raw.get("retrieved_at") or "").strip(),
        allowed_hosts=tuple(str(item).strip().lower() for item in hosts),
        collections=tuple(item for item in collections if isinstance(item, dict)),
    )
    if not source.name or not source.attribution or not source.allowed_hosts:
        raise CatalogError("Catalog source name, attribution, and allowed_hosts are required.")

    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise CatalogError("Catalog entries must be a non-empty list.")
    entries: list[CatalogEntry] = []
    seen: set[str] = set()
    for index, item in enumerate(raw_entries):
        entry = _validate_entry(item, index=index)
        if entry.id in seen:
            raise CatalogError(f"Duplicate catalog entry id: {entry.id}")
        seen.add(entry.id)
        entries.append(entry)
    declared = payload.get("entry_count")
    if declared is not None and int(declared) != len(entries):
        raise CatalogError(
            f"Catalog entry_count is {declared} but there are {len(entries)} entries."
        )
    return CatalogManifest(
        schema_version=schema_version,
        id=str(payload.get("id") or "").strip() or "bach-keyboard",
        title=str(payload.get("title") or "").strip() or "Bach keyboard catalog",
        composer=str(payload.get("composer") or "").strip() or "Johann Sebastian Bach",
        source=source,
        entries=tuple(entries),
    )


def _validate_entry(payload: object, *, index: int) -> CatalogEntry:
    if not isinstance(payload, dict):
        raise CatalogError(f"Catalog entry {index} must be an object.")
    entry_id = str(payload.get("id") or "").strip()
    if not entry_id or not _ENTRY_ID_RE.match(entry_id):
        raise CatalogError(f"Catalog entry {index} has an invalid id.")
    midi_url = str(payload.get("midi_url") or "").strip()
    if not midi_url.startswith("https://"):
        raise CatalogError(f"Catalog entry {entry_id} midi_url must be https.")
    order = payload.get("order_in_collection", 0)
    try:
        order_in_collection = int(order)
    except (TypeError, ValueError) as exc:
        raise CatalogError(f"Catalog entry {entry_id} order_in_collection is invalid.") from exc
    size = payload.get("size_bytes")
    size_bytes = None if size in (None, "") else int(size)
    duration = payload.get("duration_seconds")
    duration_seconds = None if duration in (None, "") else float(duration)
    notes = payload.get("note_count")
    note_count = None if notes in (None, "") else int(notes)
    tracks = payload.get("track_count")
    track_count = None if tracks in (None, "") else int(tracks)
    defaults = payload.get("import_defaults") or {}
    if not isinstance(defaults, dict):
        raise CatalogError(f"Catalog entry {entry_id} import_defaults must be an object.")
    return CatalogEntry(
        id=entry_id,
        bwv=str(payload.get("bwv") or "").strip(),
        title=str(payload.get("title") or "").strip() or entry_id,
        collection_id=str(payload.get("collection_id") or "").strip(),
        collection=str(payload.get("collection") or "").strip(),
        order_in_collection=order_in_collection,
        key=str(payload.get("key") or "").strip(),
        is_variant=bool(payload.get("is_variant")),
        is_arrangement=bool(payload.get("is_arrangement")),
        source_page=str(payload.get("source_page") or "").strip(),
        midi_url=midi_url,
        source_filename=str(payload.get("source_filename") or "").strip(),
        sha256=str(payload.get("sha256") or "").strip().lower(),
        size_bytes=size_bytes,
        duration_seconds=duration_seconds,
        note_count=note_count,
        track_count=track_count,
        import_defaults=dict(defaults),
    )


def get_entry(entry_id: str, *, path: Path | None = None, root: Path | None = None) -> CatalogEntry:
    catalog = load_catalog(path=path, root=root)
    for entry in catalog.entries:
        if entry.id == entry_id:
            return entry
    raise CatalogError(f"Unknown catalog entry: {entry_id}")


def search_catalog(
    catalog: CatalogManifest | None = None,
    *,
    path: Path | None = None,
    root: Path | None = None,
    text: str = "",
    collection_id: str = "",
    bwv: str = "",
    key: str = "",
    variants: str = "include",
    status: str = "all",
    statuses: Iterable[EntryStatus] | None = None,
    sort: str = "bwv",
) -> list[EntryStatus]:
    """Filter catalog entries. ``variants`` is include/only/exclude."""

    loaded = catalog if catalog is not None else load_catalog(path=path, root=root)
    rows = list(statuses) if statuses is not None else entry_statuses(loaded, root=root)
    words = [word.lower() for word in text.split() if word.strip()]
    collection_needle = collection_id.strip().lower()
    bwv_needle = bwv.strip().lower().removeprefix("bwv").strip()
    key_needle = key.strip().lower()
    variant_mode = (variants or "include").strip().lower()
    status_needle = (status or "all").strip().lower()

    found: list[EntryStatus] = []
    for row in rows:
        entry = row.entry
        if words and not all(word in entry.search_blob for word in words):
            continue
        if collection_needle and entry.collection_id.lower() != collection_needle:
            continue
        if bwv_needle and entry.bwv.lower() != bwv_needle:
            continue
        if key_needle and key_needle not in entry.key.lower():
            continue
        if variant_mode == "only" and not entry.is_variant:
            continue
        if variant_mode == "exclude" and entry.is_variant:
            continue
        if status_needle not in {"", "all"}:
            label = row.status_label
            if status_needle == "downloaded" and label not in {"downloaded", "imported"}:
                continue
            if status_needle == "missing" and row.local:
                continue
            if status_needle == "imported" and not row.imported_projects:
                continue
            if status_needle == "hash mismatch" and row.hash_ok is not False:
                continue
            if status_needle not in {
                "downloaded",
                "missing",
                "imported",
                "hash mismatch",
                label,
            }:
                continue
        found.append(row)

    if sort == "title":
        found.sort(key=lambda row: (row.entry.title.lower(), row.entry.bwv))
    elif sort == "collection":
        found.sort(
            key=lambda row: (
                row.entry.collection.lower(),
                row.entry.order_in_collection,
                row.entry.bwv,
            )
        )
    else:
        found.sort(key=lambda row: (_bwv_sort_key(row.entry.bwv), row.entry.id))
    return found


def entry_statuses(
    catalog: CatalogManifest | None = None,
    *,
    path: Path | None = None,
    root: Path | None = None,
    projects_root: Path | None = None,
) -> list[EntryStatus]:
    loaded = catalog if catalog is not None else load_catalog(path=path, root=root)
    imported = projects_by_catalog_entry(root=root, projects_root=projects_root)
    rows: list[EntryStatus] = []
    for entry in loaded.entries:
        local_path = entry_midi_path(entry.id, root=root)
        local = local_path.is_file()
        local_hash = ""
        hash_ok: bool | None = None
        if local:
            try:
                local_hash = sha256_file(local_path)
            except OSError:
                local_hash = ""
            if entry.sha256:
                hash_ok = bool(local_hash) and local_hash.lower() == entry.sha256.lower()
            else:
                hash_ok = True if local_hash else None
        rows.append(
            EntryStatus(
                entry=entry,
                local=local,
                local_path=local_path if local else None,
                local_sha256=local_hash,
                hash_ok=hash_ok,
                imported_projects=tuple(imported.get(entry.id, ())),
            )
        )
    return rows


def projects_root_for(root: Path | None = None) -> Path:
    """``transcriptions/`` under a repo root, or the default library location.

    Library helpers historically take the transcriptions folder itself as
    ``root``. Catalog code treats ``root`` as the repository root.
    """

    from baroque_transcribe.library import transcriptions_root

    if root is None:
        return transcriptions_root()
    return Path(root) / "transcriptions"


def projects_by_catalog_entry(
    *,
    root: Path | None = None,
    projects_root: Path | None = None,
) -> dict[str, list[str]]:
    from baroque_transcribe.project_manifest import load_project_manifest

    base = Path(projects_root) if projects_root is not None else projects_root_for(root)
    mapping: dict[str, list[str]] = {}
    if not base.is_dir():
        return mapping
    for folder in sorted(base.iterdir()):
        if not folder.is_dir() or folder.name.startswith((".", "_")):
            continue
        manifest = load_project_manifest(folder)
        catalog = manifest.get("catalog") if isinstance(manifest.get("catalog"), dict) else {}
        entry_id = str(catalog.get("entry_id") or "").strip()
        if not entry_id:
            continue
        mapping.setdefault(entry_id, []).append(folder.name)
    return mapping


def catalog_provenance(entry: CatalogEntry, *, midi_sha256: str = "") -> dict:
    return {
        "entry_id": entry.id,
        "bwv": entry.bwv,
        "collection_id": entry.collection_id,
        "collection": entry.collection,
        "source_page": entry.source_page,
        "midi_url": entry.midi_url,
        "source_filename": entry.source_filename,
        "attribution": "MIDI files from Tobis Notenarchiv (tobis-notenarchiv.de).",
        "pinned_sha256": entry.sha256 or midi_sha256,
    }


def all_collections(catalog: CatalogManifest | None = None, *, path: Path | None = None) -> list[tuple[str, str, int]]:
    loaded = catalog if catalog is not None else load_catalog(path=path)
    counts: dict[str, list] = {}
    titles: dict[str, str] = {}
    for entry in loaded.entries:
        counts.setdefault(entry.collection_id, []).append(entry)
        titles[entry.collection_id] = entry.collection
    return [
        (collection_id, titles[collection_id], len(counts[collection_id]))
        for collection_id, _title, _page in [
            (item.get("id"), item.get("title"), item.get("page_url"))
            for item in loaded.source.collections
        ]
        if collection_id in counts
    ] or [
        (collection_id, title, len(items))
        for collection_id, items in counts.items()
        for title in [titles[collection_id]]
    ]


def all_keys(catalog: CatalogManifest | None = None, *, path: Path | None = None) -> list[tuple[str, int]]:
    loaded = catalog if catalog is not None else load_catalog(path=path)
    counts: dict[str, int] = {}
    for entry in loaded.entries:
        if not entry.key:
            continue
        counts[entry.key] = counts.get(entry.key, 0) + 1
    return sorted(counts.items(), key=lambda item: item[0].lower())


def _bwv_sort_key(bwv: str) -> tuple:
    match = re.match(r"^(\d+)([a-z]?)$", bwv.strip(), re.I)
    if not match:
        return (10**9, bwv.lower())
    return (int(match.group(1)), match.group(2).lower())


def find_project_for_entry(
    entry_id: str,
    *,
    root: Path | None = None,
    projects_root: Path | None = None,
) -> Path | None:
    names = projects_by_catalog_entry(root=root, projects_root=projects_root).get(entry_id) or []
    if not names:
        return None
    base = Path(projects_root) if projects_root is not None else projects_root_for(root)
    return base / names[-1]


def fragment_counts_by_catalog_entry(db: Path | None = None) -> dict[str, int]:
    """How many saved fragments point at each catalog entry id."""

    from baroque_transcribe.fragment_index import index_path
    import sqlite3

    path = index_path(db)
    if not path.is_file():
        return {}
    try:
        connection = sqlite3.connect(str(path), timeout=10.0)
    except sqlite3.Error:
        return {}
    try:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(fragments)").fetchall()
        }
        if "catalog_entry_id" not in columns:
            return {}
        rows = connection.execute(
            "SELECT catalog_entry_id, COUNT(*) FROM fragments "
            "WHERE catalog_entry_id != '' GROUP BY catalog_entry_id"
        ).fetchall()
        return {str(entry_id): int(count) for entry_id, count in rows}
    except sqlite3.Error:
        return {}
    finally:
        connection.close()


def entry_statuses_with_fragments(
    catalog: CatalogManifest | None = None,
    *,
    path: Path | None = None,
    root: Path | None = None,
    projects_root: Path | None = None,
    db: Path | None = None,
) -> list[EntryStatus]:
    counts = fragment_counts_by_catalog_entry(db)
    rows = entry_statuses(catalog, path=path, root=root, projects_root=projects_root)
    return [
        EntryStatus(
            entry=row.entry,
            local=row.local,
            local_path=row.local_path,
            local_sha256=row.local_sha256,
            hash_ok=row.hash_ok,
            imported_projects=row.imported_projects,
            fragment_count=counts.get(row.entry.id, 0),
        )
        for row in rows
    ]
