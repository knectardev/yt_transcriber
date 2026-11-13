"""Validated event materialization, recovery copies, and safe project updates."""

from __future__ import annotations

import json
import os
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from baroque_transcribe.defaults import (
    ANNOTATIONS_JSON,
    MANIFEST,
    NOTES_JSON,
    NOTATION_MAP_JSON,
    PEDAL_JSON,
    PRESERVATION_JSON,
    QUANTIZED_MID,
    RAW_MID,
    SCORE_XML,
    SCORE_VOICES_XML,
    SOURCE_MID,
    SOURCE_WAV,
    WORKING_WAV,
)
from baroque_transcribe.project_manifest import (
    load_project_manifest,
    public_project_state,
    read_project_manifest,
    source_identity,
    with_revision,
    write_project_manifest,
)
from baroque_transcribe.revision import (
    RevisionStatus,
    artifact_digest,
    artifact_inventory,
    event_heads,
    event_is_complete,
    event_lineage_id,
    event_parent_id,
    event_snapshot_sha256,
    file_inventory,
    revision_status,
    snapshot_for_folder,
    snapshot_for_project,
)
from baroque_transcribe.storage_io import (
    copy_file_atomic,
    sha256_file,
    sha256_json,
    tree_size,
    write_json_atomic,
)

EVENTS_DIR = "events"
RECOVERY_DIR = ".recovery"
RECOVERY_METADATA = "recovery.json"
SCORE_FILES = (
    NOTES_JSON,
    PEDAL_JSON,
    RAW_MID,
    QUANTIZED_MID,
    SCORE_XML,
    ANNOTATIONS_JSON,
    SOURCE_MID,
)
REQUIRED_SCORE_FILES = (NOTES_JSON, PEDAL_JSON, RAW_MID, QUANTIZED_MID, SCORE_XML)
STALE_SCORE_DERIVATIVES = (
    SCORE_VOICES_XML,
    NOTATION_MAP_JSON,
    PRESERVATION_JSON,
    "notation.html",
    "playback.html",
)


class RestoreError(ValueError):
    pass


@dataclass(frozen=True)
class EventSelection:
    project: str
    target: dict
    score: dict
    analysis: dict | None
    target_folder: Path
    score_folder: Path
    analysis_folder: Path | None


@dataclass(frozen=True)
class RestorePreview:
    project: str
    target_event_id: str
    score_event_id: str
    analysis_event_id: str
    source_kind: str
    note_count: int
    harmony_count: int
    changed_components: tuple[str, ...]
    status: str
    recovery_required: bool


@dataclass(frozen=True)
class UpdateResult:
    project: Path
    target_event_id: str
    recovery: Path | None = None
    retained_audio: bool = False


def event_by_id(events: list[dict], event_id: str) -> dict:
    for event in events:
        if str(event.get("id") or "") == event_id:
            return event
    raise RestoreError(f"Parse event {event_id!r} was not found.")


def _contained_child(root: Path, name: str, label: str) -> Path:
    root = Path(root)
    if (
        not name
        or name in {".", ".."}
        or "/" in name
        or "\\" in name
        or ":" in name
        or name == RECOVERY_DIR
        or name.startswith(".tmp-")
    ):
        raise RestoreError(f"{label} must be a plain name.")
    resolved_root = root.resolve()
    candidate = root / name
    resolved_candidate = candidate.resolve()
    if resolved_candidate.parent != resolved_root:
        raise RestoreError(f"{label} escapes its storage root.")
    return candidate


def _event_folder(store: Path, event_id: str) -> Path:
    return _contained_child(Path(store) / EVENTS_DIR, event_id, "Event id")


def _project_folder(projects_root: Path, project: str) -> Path:
    return _contained_child(Path(projects_root), project, "Project name")


def default_target_event(store: Path, events: list[dict], project: str) -> dict:
    relevant = [
        event
        for event in events
        if event.get("status") == "ok" and str(event.get("project") or "") == project
    ]
    if not relevant:
        raise RestoreError(f"No successful parse event exists for project {project!r}.")
    v2 = [event for event in relevant if event_lineage_id(event)]
    if v2:
        heads = [
            head
            for lineage in sorted({event_lineage_id(event) for event in v2})
            for head in event_heads(v2, lineage)
        ]
        if len(heads) != 1:
            names = ", ".join(str(item.get("id") or "") for item in heads)
            raise RestoreError(
                f"Project {project!r} has incomparable published heads: {names}. "
                "Select an event explicitly."
            )
        return heads[0]
    score_events = [
        event
        for event in relevant
        if event_is_complete(event)
        and (
            _event_folder(store, str(event.get("id") or "")) / NOTES_JSON
        ).is_file()
    ]
    return score_events[-1] if score_events else relevant[-1]


def resolve_event_selection(store: Path, events: list[dict], event_id: str) -> EventSelection:
    target = event_by_id(events, event_id)
    if target.get("status") != "ok":
        raise RestoreError(f"Event {event_id} did not complete successfully.")
    project = str(target.get("project") or "")
    if not project:
        raise RestoreError(f"Event {event_id} does not name a project.")
    if int(target.get("version", 1)) == 1:
        return _resolve_legacy_selection(store, events, target)
    by_id = {str(event.get("id") or ""): event for event in events}
    chain: list[dict] = []
    current = target
    seen: set[str] = set()
    score: dict | None = None
    while current is not None:
        ident = str(current.get("id") or "")
        if ident in seen:
            raise RestoreError(f"Event ancestry for {event_id} contains a cycle.")
        seen.add(ident)
        if str(current.get("project") or "") != project:
            raise RestoreError(f"Event ancestry for {event_id} crosses project boundaries.")
        chain.append(current)
        folder = _event_folder(store, ident)
        if event_is_complete(current) and (folder / NOTES_JSON).is_file():
            score = current
            break
        parent = event_parent_id(current)
        current = by_id.get(parent) if parent else None
    if score is None:
        # Immutable v1 events have no explicit complete-snapshot bit.
        folder = _event_folder(store, event_id)
        if int(target.get("version", 1)) == 1 and (folder / NOTES_JSON).is_file():
            score = target
        else:
            raise RestoreError(
                f"Event {event_id} has no complete score snapshot in its causal ancestry."
            )
    score_id = str(score.get("id") or "")
    score_folder = _event_folder(store, score_id)
    validate_complete_snapshot(score, score_folder)
    for name in SCORE_FILES:
        _contained_child(score_folder, name, "Artifact name")
    analysis = next(
        (
            item
            for item in chain
            if (_event_folder(store, str(item.get("id") or "")) / ANNOTATIONS_JSON).is_file()
        ),
        None,
    )
    analysis_folder = (
        _event_folder(store, str(analysis.get("id") or "")) if analysis is not None else None
    )
    if analysis_folder is not None:
        _contained_child(analysis_folder, ANNOTATIONS_JSON, "Artifact name")
    return EventSelection(
        project=project,
        target=target,
        score=score,
        analysis=analysis,
        target_folder=_event_folder(store, event_id),
        score_folder=score_folder,
        analysis_folder=analysis_folder,
    )


def _resolve_legacy_selection(
    store: Path,
    events: list[dict],
    target: dict,
) -> EventSelection:
    """Apply the original timestamp overlay rule to immutable v1 history."""

    project = str(target.get("project") or "")
    target_id = str(target.get("id") or "")
    target_folder = _event_folder(store, target_id)
    target_time = str(target.get("recorded_at") or "")
    if (target_folder / NOTES_JSON).is_file():
        score = target
    else:
        candidates = [
            event
            for event in events
            if int(event.get("version", 1)) == 1
            and event.get("status") == "ok"
            and str(event.get("project") or "") == project
            and (not target_time or str(event.get("recorded_at") or "") <= target_time)
            and (
                _event_folder(store, str(event.get("id") or "")) / NOTES_JSON
            ).is_file()
        ]
        if not candidates:
            raise RestoreError(
                f"Legacy event {target_id} has no earlier complete score snapshot."
            )
        score = candidates[-1]
    score_id = str(score.get("id") or "")
    score_folder = _event_folder(store, score_id)
    validate_complete_snapshot(score, score_folder)
    for name in SCORE_FILES:
        _contained_child(score_folder, name, "Artifact name")
    score_time = str(score.get("recorded_at") or "")
    if target_id != score_id and (target_folder / ANNOTATIONS_JSON).is_file():
        analysis = target
    else:
        overlays = [
            event
            for event in events
            if int(event.get("version", 1)) == 1
            and event.get("status") == "ok"
            and str(event.get("project") or "") == project
            and str(event.get("recorded_at") or "") >= score_time
            and (
                _event_folder(store, str(event.get("id") or ""))
                / ANNOTATIONS_JSON
            ).is_file()
        ]
        analysis = overlays[-1] if overlays else None
    analysis_folder = (
        _event_folder(store, str(analysis.get("id") or ""))
        if analysis is not None
        else None
    )
    if analysis_folder is not None:
        _contained_child(analysis_folder, ANNOTATIONS_JSON, "Artifact name")
    return EventSelection(
        project=project,
        target=target,
        score=score,
        analysis=analysis,
        target_folder=target_folder,
        score_folder=score_folder,
        analysis_folder=analysis_folder,
    )


def validate_complete_snapshot(event: dict, folder: Path) -> None:
    version = int(event.get("version", 1))
    missing = [
        name
        for name in REQUIRED_SCORE_FILES
        if not _contained_child(folder, name, "Artifact name").is_file()
    ]
    source = event.get("source") if isinstance(event.get("source"), dict) else {}
    if (
        version >= 2
        and source.get("kind") == "midi"
        and not _contained_child(folder, SOURCE_MID, "Artifact name").is_file()
    ):
        missing.append(SOURCE_MID)
    if missing:
        raise RestoreError(
            f"Event {event.get('id')} is not independently restorable; missing "
            f"{', '.join(sorted(set(missing)))}."
        )
    if version >= 2 and source.get("kind") == "midi":
        source_hash = str(source.get("sha256") or "")
        source_midi = _contained_child(folder, SOURCE_MID, "Artifact name")
        if not source_hash or sha256_file(source_midi) != source_hash:
            raise RestoreError(
                f"Event {event.get('id')} source.mid does not match its source SHA-256."
            )
    if version < 2:
        return
    if not event_is_complete(event):
        raise RestoreError(f"Event {event.get('id')} is not marked as a complete snapshot.")
    revision = event.get("revision") if isinstance(event.get("revision"), dict) else {}
    expected = revision.get("snapshot") if isinstance(revision.get("snapshot"), dict) else {}
    hashes = expected.get("artifacts") if isinstance(expected.get("artifacts"), dict) else {}
    for name, digest in hashes.items():
        path = _contained_child(folder, str(name), "Artifact name")
        if not path.is_file():
            raise RestoreError(f"Event {event.get('id')} is missing hashed artifact {name}.")
        if artifact_digest(path) != str(digest):
            raise RestoreError(f"Event {event.get('id')} has a hash mismatch for {name}.")
    files = expected.get("files") if isinstance(expected.get("files"), dict) else {}
    for name, digest in files.items():
        path = _contained_child(folder, str(name), "Artifact name")
        if not path.is_file():
            raise RestoreError(f"Event {event.get('id')} is missing byte-hashed artifact {name}.")
        if sha256_file(path) != str(digest):
            raise RestoreError(f"Event {event.get('id')} has a byte hash mismatch for {name}.")
    actual = snapshot_for_folder(folder, event.get("project_state") or {})
    if actual.sha256 != str(expected.get("sha256") or ""):
        raise RestoreError(f"Event {event.get('id')} has an aggregate snapshot mismatch.")


def preview_restore(
    *,
    store: Path,
    projects_root: Path,
    event_id: str,
) -> RestorePreview:
    from baroque_transcribe.parse_store import load_events

    events = load_events(store)
    selection = resolve_event_selection(store, events, event_id)
    project_dir = _project_folder(projects_root, selection.project)
    local = artifact_inventory(project_dir) if (project_dir / NOTES_JSON).is_file() else {}
    target = _effective_artifacts(selection)
    changed = {
        name
        for name in set(local) | set(target)
        if local.get(name) != target.get(name)
    }
    target_state = selection.target.get("project_state")
    if isinstance(target_state, dict) and (project_dir / MANIFEST).is_file():
        if public_project_state(load_project_manifest(project_dir)) != target_state:
            changed.add(MANIFEST)
    notes = _read_object(
        _contained_child(selection.score_folder, NOTES_JSON, "Artifact name")
    )
    annotation_path = (
        _contained_child(
            selection.analysis_folder,
            ANNOTATIONS_JSON,
            "Artifact name",
        )
        if selection.analysis_folder is not None
        else _contained_child(
            selection.score_folder,
            ANNOTATIONS_JSON,
            "Artifact name",
        )
    )
    annotations = _read_object(annotation_path)
    status = project_revision_status(
        selection.project, project_dir if project_dir.is_dir() else None, events, store
    )
    return RestorePreview(
        project=selection.project,
        target_event_id=event_id,
        score_event_id=str(selection.score.get("id") or ""),
        analysis_event_id=str(selection.analysis.get("id") or "") if selection.analysis else "",
        source_kind=_source_kind(selection.target),
        note_count=len(notes.get("notes") or []),
        harmony_count=_harmony_count(annotations),
        changed_components=tuple(sorted(changed)),
        status=status.state,
        recovery_required=(project_dir / NOTES_JSON).is_file(),
    )


def project_revision_status(
    project: str,
    project_dir: Path | None,
    events: list[dict],
    store: Path,
) -> RevisionStatus:
    return revision_status(project, project_dir, events, store)


def restore_missing_project(
    *,
    store: Path,
    projects_root: Path,
    event_id: str,
    with_audio: bool = False,
) -> UpdateResult:
    from baroque_transcribe.parse_store import load_events

    events = load_events(store)
    selection = resolve_event_selection(store, events, event_id)
    destination = _project_folder(projects_root, selection.project)
    if destination.exists():
        raise RestoreError(
            f"{destination} already exists. Use the explicit update operation to replace it safely."
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".tmp-restore-{destination.name}-{uuid.uuid4().hex[:8]}"
    try:
        stage.mkdir()
        _materialize(selection, stage, with_audio=with_audio)
        _validate_materialized(selection, stage)
        os.replace(stage, destination)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return UpdateResult(project=destination, target_event_id=event_id)


def update_project(
    *,
    store: Path,
    projects_root: Path,
    event_id: str,
    use_published_annotations: bool = False,
    reason: str = "published event update",
) -> UpdateResult:
    from baroque_transcribe.parse_store import load_events

    events = load_events(store)
    selection = resolve_event_selection(store, events, event_id)
    project = _project_folder(projects_root, selection.project)
    if not (project / NOTES_JSON).is_file():
        raise RestoreError(f"{project} is not an existing project; use restore instead.")
    preview_restore(store=store, projects_root=projects_root, event_id=event_id)
    recovery = create_recovery_copy(
        project,
        projects_root=Path(projects_root),
        target_event_id=event_id,
        reason=reason,
    )
    old_manifest = load_project_manifest(project)
    target_state = selection.target.get("project_state")
    source_matches = isinstance(target_state, dict) and _same_source_and_excerpt(
        old_manifest, target_state
    )
    stage = project.parent / f".tmp-update-{project.name}-{uuid.uuid4().hex[:8]}"
    prior = project.parent / f".tmp-prior-{project.name}-{uuid.uuid4().hex[:8]}"
    try:
        shutil.copytree(project, stage)
        local_annotations = stage / ANNOTATIONS_JSON
        for name in SCORE_FILES:
            if name == ANNOTATIONS_JSON and not use_published_annotations:
                continue
            (stage / name).unlink(missing_ok=True)
        for name in STALE_SCORE_DERIVATIVES:
            (stage / name).unlink(missing_ok=True)
        if not source_matches:
            (stage / SOURCE_WAV).unlink(missing_ok=True)
            (stage / WORKING_WAV).unlink(missing_ok=True)
        _materialize(
            selection,
            stage,
            preserve_annotations=(
                local_annotations
                if local_annotations.is_file() and not use_published_annotations
                else None
            ),
        )
        _validate_materialized(
            selection,
            stage,
            allow_local_annotations=(
                local_annotations.is_file() and not use_published_annotations
            ),
        )
        os.replace(project, prior)
        try:
            os.replace(stage, project)
            _validate_materialized(
                selection,
                project,
                allow_local_annotations=(
                    (project / ANNOTATIONS_JSON).is_file() and not use_published_annotations
                ),
            )
        except Exception:
            failed = project.parent / f".tmp-failed-{project.name}-{uuid.uuid4().hex[:8]}"
            if project.exists():
                os.replace(project, failed)
            os.replace(prior, project)
            raise
        shutil.rmtree(prior, ignore_errors=True)
    except Exception as exc:
        shutil.rmtree(stage, ignore_errors=True)
        if prior.exists() and not project.exists():
            os.replace(prior, project)
        raise RestoreError(
            f"Update failed; the prior project was restored. Recovery retained at {recovery}: {exc}"
        ) from exc
    return UpdateResult(
        project=project,
        target_event_id=event_id,
        recovery=recovery,
        retained_audio=source_matches
        and ((project / SOURCE_WAV).is_file() or (project / WORKING_WAV).is_file()),
    )


def adopt_exact_match(
    *,
    store: Path,
    projects_root: Path,
    project: str,
) -> RevisionStatus:
    from baroque_transcribe.parse_store import load_events

    events = load_events(store)
    folder = Path(projects_root) / Path(project).name
    status = revision_status(folder.name, folder, events, store)
    if status.state != "legacy_match" or not status.adoptable_event_id:
        raise RestoreError("The project is not an exact adoptable match.")
    event = event_by_id(events, status.adoptable_event_id)
    current_manifest = read_project_manifest(folder)
    manifest = with_revision(
        current_manifest,
        lineage_id=event_lineage_id(event) or status.adoptable_event_id,
        base_event_id=status.adoptable_event_id,
        base_snapshot_sha256=event_snapshot_sha256(event)
        or snapshot_for_folder(
            _event_folder(store, status.adoptable_event_id),
            public_project_state(current_manifest),
        ).sha256,
    )
    write_project_manifest(folder, manifest)
    return revision_status(folder.name, folder, events, store)


def create_recovery_copy(
    project: Path,
    *,
    projects_root: Path,
    target_event_id: str,
    reason: str,
) -> Path:
    project = Path(project)
    if not project.is_dir():
        raise RestoreError(f"Cannot recover missing project {project}.")
    when = datetime.now(timezone.utc)
    recovery_id = f"{when.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    parent = Path(projects_root) / RECOVERY_DIR / project.name
    parent.mkdir(parents=True, exist_ok=True)
    stage = parent / f".tmp-{recovery_id}"
    final = parent / recovery_id
    try:
        shutil.copytree(project, stage)
        inventory = _tree_inventory(stage)
        manifest = load_project_manifest(project)
        revision = manifest.get("revision") if isinstance(manifest.get("revision"), dict) else {}
        metadata = {
            "version": 1,
            "id": recovery_id,
            "project": project.name,
            "created_at": when.isoformat().replace("+00:00", "Z"),
            "reason": reason,
            "source_project": str(project),
            "prior_fingerprint": snapshot_for_project(project).sha256,
            "prior_base_event_id": str(revision.get("base_event_id") or ""),
            "target_event_id": target_event_id,
            "bytes": tree_size(stage),
            "files": inventory,
        }
        write_json_atomic(stage / RECOVERY_METADATA, metadata, sort_keys=True)
        _verify_inventory(stage, inventory)
        os.replace(stage, final)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return final


def list_recoveries(projects_root: Path) -> list[dict]:
    base = Path(projects_root) / RECOVERY_DIR
    found: list[dict] = []
    if not base.is_dir():
        return found
    for path in sorted(base.glob("*/*")):
        if not path.is_dir() or path.name.startswith(".tmp-"):
            continue
        payload = _read_object(path / RECOVERY_METADATA)
        payload["path"] = str(path)
        payload["actual_bytes"] = tree_size(path)
        found.append(payload)
    return found


def remove_recovery(
    projects_root: Path,
    *,
    project: str,
    recovery_id: str,
    confirm: str,
) -> Path:
    expected = f"DELETE {Path(project).name}/{Path(recovery_id).name}"
    if confirm != expected:
        raise RestoreError(f"Confirmation must be exactly: {expected}")
    base = (Path(projects_root) / RECOVERY_DIR).resolve()
    target = (base / Path(project).name / Path(recovery_id).name).resolve()
    if base not in target.parents or not (target / RECOVERY_METADATA).is_file():
        raise RestoreError("The selected recovery copy does not exist.")
    shutil.rmtree(target)
    return target


def rollback_recovery(
    projects_root: Path,
    *,
    project: str,
    recovery_id: str,
    confirm: str,
) -> UpdateResult:
    """Restore one verified recovery while retaining it and safeguarding current state."""

    project_name = Path(project).name
    recovery_name = Path(recovery_id).name
    expected = f"RESTORE {project_name}/{recovery_name}"
    if confirm != expected:
        raise RestoreError(f"Confirmation must be exactly: {expected}")
    base = (Path(projects_root) / RECOVERY_DIR).resolve()
    source = (base / project_name / recovery_name).resolve()
    if base not in source.parents:
        raise RestoreError("The selected recovery copy escapes the recovery root.")
    metadata = _read_object(source / RECOVERY_METADATA)
    if (
        str(metadata.get("project") or "") != project_name
        or str(metadata.get("id") or "") != recovery_name
    ):
        raise RestoreError("The selected recovery copy does not exist or is invalid.")
    inventory = metadata.get("files")
    if not isinstance(inventory, dict) or any(
        not isinstance(name, str) or not isinstance(digest, str)
        for name, digest in inventory.items()
    ):
        raise RestoreError("The selected recovery has no valid integrity inventory.")
    _verify_inventory(source, inventory)

    destination = _project_folder(Path(projects_root), project_name)
    safeguard = None
    if destination.is_dir():
        safeguard = create_recovery_copy(
            destination,
            projects_root=Path(projects_root),
            target_event_id=str(metadata.get("prior_base_event_id") or ""),
            reason=f"before rollback to recovery {project_name}/{recovery_name}",
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".tmp-rollback-{project_name}-{uuid.uuid4().hex[:8]}"
    prior = destination.parent / f".tmp-prior-{project_name}-{uuid.uuid4().hex[:8]}"
    try:
        shutil.copytree(source, stage)
        (stage / RECOVERY_METADATA).unlink(missing_ok=True)
        _verify_inventory(stage, inventory)
        if destination.exists():
            os.replace(destination, prior)
        try:
            os.replace(stage, destination)
            _verify_inventory(destination, inventory)
        except Exception:
            if destination.exists():
                failed = destination.parent / (
                    f".tmp-failed-rollback-{project_name}-{uuid.uuid4().hex[:8]}"
                )
                os.replace(destination, failed)
            if prior.exists():
                os.replace(prior, destination)
            raise
        shutil.rmtree(prior, ignore_errors=True)
    except Exception as exc:
        shutil.rmtree(stage, ignore_errors=True)
        if prior.exists() and not destination.exists():
            os.replace(prior, destination)
        raise RestoreError(
            f"Recovery rollback failed; current project was preserved: {exc}"
        ) from exc
    return UpdateResult(
        project=destination,
        target_event_id=str(metadata.get("prior_base_event_id") or ""),
        recovery=safeguard,
    )


def restore_projects_compatible(
    *,
    store: Path,
    projects_root: Path,
    projects: list[str] | None,
    with_audio: bool,
) -> tuple[list[Path], list[str]]:
    """Non-destructive bulk restore retained for the original CLI contract."""

    from baroque_transcribe.parse_store import load_events

    events = load_events(store)
    names = sorted(
        {str(event.get("project") or "") for event in events if event.get("status") == "ok"}
    )
    wanted = {Path(name).name for name in projects} if projects else set(names)
    restored: list[Path] = []
    skipped: list[str] = []
    for project in sorted(wanted):
        try:
            target = default_target_event(store, events, project)
            event_id = str(target.get("id") or "")
            selection = resolve_event_selection(store, events, event_id)
        except RestoreError:
            skipped.append(project)
            continue
        destination = _project_folder(projects_root, project)
        if (destination / NOTES_JSON).is_file():
            overlay = (
                _contained_child(
                    selection.analysis_folder,
                    ANNOTATIONS_JSON,
                    "Artifact name",
                )
                if selection.analysis_folder is not None
                else _contained_child(
                    selection.score_folder,
                    ANNOTATIONS_JSON,
                    "Artifact name",
                )
            )
            if overlay.is_file() and not (destination / ANNOTATIONS_JSON).is_file():
                copy_file_atomic(overlay, destination / ANNOTATIONS_JSON)
                restored.append(destination)
            else:
                skipped.append(project)
            continue
        if destination.exists():
            skipped.append(project)
            continue
        result = restore_missing_project(
            store=store,
            projects_root=projects_root,
            event_id=event_id,
            with_audio=with_audio,
        )
        restored.append(result.project)
    return restored, skipped


def _materialize(
    selection: EventSelection,
    destination: Path,
    *,
    preserve_annotations: Path | None = None,
    with_audio: bool = False,
) -> None:
    for name in SCORE_FILES:
        if name == ANNOTATIONS_JSON and preserve_annotations is not None:
            continue
        source = _contained_child(selection.score_folder, name, "Artifact name")
        if source.is_file():
            copy_file_atomic(source, destination / name)
    if selection.analysis_folder is not None and preserve_annotations is None:
        copy_file_atomic(
            _contained_child(
                selection.analysis_folder,
                ANNOTATIONS_JSON,
                "Artifact name",
            ),
            destination / ANNOTATIONS_JSON,
        )
    manifest = _restored_manifest(selection)
    if preserve_annotations is not None and preserve_annotations.is_file():
        warning = (
            "Local annotations were preserved. Select the published annotations explicitly "
            "to replace them."
        )
        manifest.setdefault("warnings", []).append(warning)
    if with_audio:
        from baroque_transcribe.parse_store import _restore_audio

        _restore_audio(selection.target, destination, manifest)
    generated = [
        name
        for name in (NOTES_JSON, PEDAL_JSON, RAW_MID, QUANTIZED_MID, SCORE_XML)
        if (destination / name).is_file()
    ]
    manifest["generation"] = {
        "version": 1,
        "artifacts": {
            name: sha256_file(destination / name)
            for name in generated
        },
    }
    write_project_manifest(destination, manifest)


def _restored_manifest(selection: EventSelection) -> dict:
    state = selection.target.get("project_state")
    if not isinstance(state, dict):
        from baroque_transcribe.parse_store import _restored_manifest as legacy_manifest

        copied = {
            name: _contained_child(
                selection.score_folder,
                name,
                "Artifact name",
            ).is_file()
            for name in SCORE_FILES
        }
        manifest = legacy_manifest(selection.score, selection.score_folder, copied)
    else:
        portable_source = state.get("source") if isinstance(state.get("source"), dict) else {}
        source = (
            selection.target.get("source")
            if isinstance(selection.target.get("source"), dict)
            else {}
        )
        kind = str(portable_source.get("kind") or source.get("kind") or "file")
        origin = str(source.get("url") or portable_source.get("name") or source.get("name") or "")
        notation = dict(state.get("notation") or {})
        notation.setdefault("quantized_mid", QUANTIZED_MID)
        notation.setdefault("score", SCORE_XML)
        notation.setdefault("raw_mid", RAW_MID)
        manifest = {
            "program": "baroque-transcribe (restored from parse history)",
            "origin": origin,
            "title": str(state.get("title") or source.get("title") or ""),
            "model": str(state.get("model") or ""),
            "import_settings_sha256": str(
                state.get("import_settings_sha256") or ""
            ),
            "source_kind": kind,
            "audio": dict(state.get("audio") or {}),
            "tempo": dict(state.get("tempo") or {}),
            "staff_split": state.get("staff_split"),
            "staff_assignment": str(state.get("staff_assignment") or ""),
            "notation": notation,
            "sync_offset_seconds": state.get("sync_offset_seconds", 0.0),
            "warnings": [],
        }
        if portable_source.get("sha256"):
            manifest["source_sha256"] = str(portable_source["sha256"])
        if kind == "youtube" and source.get("url"):
            manifest["youtube_url"] = str(source["url"])
        if kind in {"file", "midi"} and source.get("youtube_url"):
            manifest["youtube_url"] = str(source["youtube_url"])
        catalog = state.get("catalog")
        if isinstance(catalog, dict) and catalog:
            manifest["catalog"] = dict(catalog)
    notes = _read_object(
        _contained_child(selection.score_folder, NOTES_JSON, "Artifact name")
    )
    manifest["counts"] = {
        "notes": len(notes.get("notes") or []),
        "pedals": len(notes.get("pedals") or []),
    }
    manifest["restored_from"] = {
        "event": str(selection.target.get("id") or ""),
        "score_event": str(selection.score.get("id") or ""),
        "analysis_event": (
            str(selection.analysis.get("id") or "") if selection.analysis is not None else ""
        ),
        "machine": str(selection.target.get("machine") or ""),
        "recorded_at": str(selection.target.get("recorded_at") or ""),
    }
    manifest.setdefault("warnings", []).append(
        f"Restored from parse history event {selection.target.get('id')}; no model was run."
    )
    return with_revision(
        manifest,
        lineage_id=event_lineage_id(selection.target)
        or str(selection.target.get("id") or ""),
        base_event_id=str(selection.target.get("id") or ""),
        base_snapshot_sha256=event_snapshot_sha256(selection.target)
        or event_snapshot_sha256(selection.score)
        or _selection_snapshot_sha256(selection, public_project_state(manifest)),
    )


def _validate_materialized(
    selection: EventSelection,
    destination: Path,
    *,
    allow_local_annotations: bool = False,
) -> None:
    missing = [name for name in REQUIRED_SCORE_FILES if not (destination / name).is_file()]
    if (
        int(selection.score.get("version", 1)) >= 2
        and _source_kind(selection.target) == "midi"
        and not (destination / SOURCE_MID).is_file()
    ):
        missing.append(SOURCE_MID)
    if missing:
        raise RestoreError(f"Materialized project is missing {', '.join(sorted(set(missing)))}.")
    expected = _effective_artifacts(selection)
    for name, digest in expected.items():
        if name == ANNOTATIONS_JSON and allow_local_annotations:
            continue
        path = destination / name
        if not path.is_file() or artifact_digest(path) != digest:
            raise RestoreError(f"Materialized project failed validation for {name}.")
    expected_files = _effective_file_hashes(selection)
    for name, digest in expected_files.items():
        if name == ANNOTATIONS_JSON and allow_local_annotations:
            continue
        path = destination / name
        if not path.is_file() or sha256_file(path) != digest:
            raise RestoreError(
                f"Materialized project failed byte validation for {name}."
            )
    state = selection.target.get("project_state")
    if isinstance(state, dict):
        restored = public_project_state(load_project_manifest(destination))
        if restored != state:
            raise RestoreError("Materialized project settings do not match the selected event.")
    if not allow_local_annotations and event_snapshot_sha256(selection.target):
        actual = snapshot_for_project(destination)
        if actual.sha256 != event_snapshot_sha256(selection.target):
            raise RestoreError("Materialized project aggregate fingerprint does not match.")


def _effective_artifacts(selection: EventSelection) -> dict[str, str]:
    expected = dict(
        (
            selection.score.get("revision", {}).get("snapshot", {}).get("artifacts", {})
            if isinstance(selection.score.get("revision"), dict)
            else {}
        )
        or artifact_inventory(selection.score_folder)
    )
    if selection.analysis_folder is not None:
        expected[ANNOTATIONS_JSON] = artifact_digest(
            _contained_child(
                selection.analysis_folder,
                ANNOTATIONS_JSON,
                "Artifact name",
            )
        )
    return expected


def _effective_file_hashes(selection: EventSelection) -> dict[str, str]:
    snapshot = (
        selection.score.get("revision", {}).get("snapshot", {})
        if isinstance(selection.score.get("revision"), dict)
        else {}
    )
    expected = dict(snapshot.get("files") or file_inventory(selection.score_folder))
    if selection.analysis_folder is not None:
        analysis_snapshot = (
            selection.analysis.get("revision", {}).get("snapshot", {})
            if selection.analysis is not None
            and isinstance(selection.analysis.get("revision"), dict)
            else {}
        )
        analysis_files = (
            analysis_snapshot.get("files")
            if isinstance(analysis_snapshot.get("files"), dict)
            else {}
        )
        expected[ANNOTATIONS_JSON] = str(
            analysis_files.get(ANNOTATIONS_JSON)
            or sha256_file(
                _contained_child(
                    selection.analysis_folder,
                    ANNOTATIONS_JSON,
                    "Artifact name",
                )
            )
        )
    return expected


def _selection_snapshot_sha256(
    selection: EventSelection,
    project_state: dict,
) -> str:
    state_hash = sha256_json(project_state)
    return sha256_json(
        {
            "artifacts": dict(sorted(_effective_artifacts(selection).items())),
            "project_state_sha256": state_hash,
        }
    )


def _same_source_and_excerpt(local_manifest: dict, target_state: dict) -> bool:
    local_source = source_identity(local_manifest)
    target_source = dict(target_state.get("source") or {})
    if local_source != target_source:
        return False
    if str(target_source.get("kind") or "") in {"file", "midi"} and (
        not str(local_source.get("sha256") or "")
        or not str(target_source.get("sha256") or "")
    ):
        return False
    local_audio = public_project_state(local_manifest).get("audio") or {}
    target_audio = target_state.get("audio") if isinstance(target_state.get("audio"), dict) else {}
    keys = ("excerpt_start_seconds", "excerpt_end_seconds", "working_duration_seconds")
    return all(local_audio.get(key) == target_audio.get(key) for key in keys)


def _tree_inventory(folder: Path) -> dict[str, str]:
    return {
        path.relative_to(folder).as_posix(): sha256_file(path)
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.name != RECOVERY_METADATA
    }


def _verify_inventory(folder: Path, inventory: dict[str, str]) -> None:
    actual = _tree_inventory(folder)
    if actual != inventory:
        raise RestoreError("Recovery copy failed integrity verification.")


def _read_object(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _harmony_count(payload: dict) -> int:
    for key in ("harmonies", "annotations", "segments", "events"):
        value = payload.get(key)
        if isinstance(value, list):
            return len(value)
    return 0


def _source_kind(event: dict) -> str:
    state = event.get("project_state") if isinstance(event.get("project_state"), dict) else {}
    source = state.get("source") if isinstance(state.get("source"), dict) else {}
    if source.get("kind"):
        return str(source["kind"])
    described = event.get("source") if isinstance(event.get("source"), dict) else {}
    return str(described.get("kind") or "")
