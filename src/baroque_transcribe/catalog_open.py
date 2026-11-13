"""Materialize a catalog entry as a normal Library MIDI project."""

from __future__ import annotations

from pathlib import Path

from baroque_transcribe.catalog import (
    CatalogError,
    catalog_provenance,
    find_project_for_entry,
    get_entry,
    load_catalog,
    projects_root_for,
)
from baroque_transcribe.catalog_sync import ensure_local_midi
from baroque_transcribe.defaults import NOTES_JSON, SOURCE_MID
from baroque_transcribe.library import find_project
from baroque_transcribe.midi_import import (
    DuplicateMidiImportError,
    import_midi_project,
)
from baroque_transcribe.parse_store import load_events, resolve_store
from baroque_transcribe.project_manifest import load_project_manifest
from baroque_transcribe.storage_io import sha256_file


def open_catalog_entry(
    entry_id: str,
    *,
    output_dir: Path | None = None,
    root: Path | None = None,
    manifest_path: Path | None = None,
    history_store: Path | None = None,
    allow_duplicate: bool = False,
    sync_if_missing: bool = True,
) -> Path:
    """Ensure local MIDI exists, then open or import a Library project.

    Preference order:
    1. An existing ``transcriptions/`` project that already records this catalog id
    2. A published MIDI import with the same source hash (unless ``allow_duplicate``)
    3. A fresh ``import_midi_project`` with catalog provenance
    """

    catalog = load_catalog(path=manifest_path, root=root)
    entry = get_entry(entry_id, path=manifest_path, root=root)
    existing = find_project_for_entry(entry.id, root=root)
    if existing is not None and (existing / NOTES_JSON).is_file():
        return existing.resolve()

    if sync_if_missing:
        midi_path = ensure_local_midi(
            entry,
            root=root,
            allowed_hosts=catalog.source.allowed_hosts,
            validate=True,
        )
    else:
        from baroque_transcribe.catalog import entry_midi_path

        midi_path = entry_midi_path(entry.id, root=root)
        if not midi_path.is_file():
            raise CatalogError(
                f"Catalog MIDI for {entry.id} is not downloaded. Sync it first."
            )

    source_hash = sha256_file(midi_path)
    if entry.sha256 and source_hash.lower() != entry.sha256.lower():
        raise CatalogError(
            f"Local MIDI hash for {entry.id} does not match the catalog pin."
        )

    # Prefer an already-imported local project for this catalog MIDI, even when
    # older projects predate catalog provenance fields.
    local_match = _find_local_by_hash(source_hash, root=root)
    if local_match is not None and not allow_duplicate:
        _ensure_catalog_provenance(local_match, entry, catalog.source.attribution, source_hash)
        return local_match

    if not allow_duplicate:
        duplicate = _find_published_project(source_hash, history_store=history_store, root=root)
        if duplicate is not None:
            _ensure_catalog_provenance(duplicate, entry, catalog.source.attribution, source_hash)
            return duplicate

    target = Path(output_dir) if output_dir is not None else projects_root_for(root) / "catalog"
    defaults = dict(entry.import_defaults or {})
    bpm = defaults.get("bpm")
    meter = defaults.get("time_signature")
    if isinstance(meter, list) and len(meter) == 2:
        time_signature = (int(meter[0]), int(meter[1]))
    else:
        time_signature = None
    subdivision = int(defaults.get("subdivision") or 16)
    staff_split = int(defaults.get("staff_split") or 60)
    provenance = catalog_provenance(entry, midi_sha256=source_hash)
    provenance["attribution"] = catalog.source.attribution

    try:
        return import_midi_project(
            midi_path,
            target,
            title=entry.display_title,
            bpm=float(bpm) if bpm is not None else None,
            time_signature=time_signature,
            subdivision=subdivision,
            staff_split=staff_split,
            history_store=history_store,
            allow_duplicate=allow_duplicate,
            catalog=provenance,
        )
    except DuplicateMidiImportError as exc:
        # Last-resort: locate any local project whose source.mid matches.
        local = _find_local_by_hash(source_hash, root=root)
        if local is not None:
            return local
        raise CatalogError(str(exc)) from exc


def _find_published_project(
    source_sha256: str,
    *,
    history_store: Path | None,
    root: Path | None,
) -> Path | None:
    store = resolve_store(history_store)
    if store is None:
        return None
    projects = projects_root_for(root)
    for event in reversed(load_events(store)):
        source = event.get("source") if isinstance(event.get("source"), dict) else {}
        if (
            event.get("status") == "ok"
            and source.get("kind") == "midi"
            and str(source.get("sha256") or "").lower() == source_sha256.lower()
        ):
            project_name = str(event.get("project") or "")
            found = find_project(project_name, projects)
            if found is not None:
                return found
    return None


def _ensure_catalog_provenance(
    project: Path,
    entry,
    attribution: str,
    source_hash: str,
) -> None:
    from baroque_transcribe.project_manifest import read_project_manifest, write_project_manifest

    try:
        manifest = read_project_manifest(project, missing_ok=False)
    except Exception:
        return
    current = manifest.get("catalog") if isinstance(manifest.get("catalog"), dict) else {}
    if str(current.get("entry_id") or "") == entry.id:
        return
    manifest["catalog"] = catalog_provenance(entry, midi_sha256=source_hash)
    manifest["catalog"]["attribution"] = attribution
    write_project_manifest(project, manifest)


def _find_local_by_hash(source_sha256: str, *, root: Path | None) -> Path | None:
    base = projects_root_for(root)
    if not base.is_dir():
        return None
    for folder in sorted(base.iterdir()):
        midi = folder / SOURCE_MID
        if not midi.is_file():
            continue
        try:
            if sha256_file(midi).lower() == source_sha256.lower():
                if (folder / NOTES_JSON).is_file():
                    return folder.resolve()
        except OSError:
            continue
        manifest = load_project_manifest(folder)
        if str(manifest.get("source_sha256") or "").lower() == source_sha256.lower():
            if (folder / NOTES_JSON).is_file():
                return folder.resolve()
    return None
