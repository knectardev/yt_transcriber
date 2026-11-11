"""Causal project revisions and canonical storage fingerprints.

The module is deliberately independent of ``parse_store`` so event IO can use
it without a circular import. Event schema v2 stores complete score snapshots;
the parent link describes ancestry only.
"""

from __future__ import annotations

import json
import xml.etree.ElementTree as element_tree
from dataclasses import dataclass, field
from pathlib import Path

from baroque_transcribe.defaults import (
    ANNOTATIONS_JSON,
    NOTES_JSON,
    PEDAL_JSON,
    QUANTIZED_MID,
    RAW_MID,
    SCORE_XML,
    SOURCE_MID,
)
from baroque_transcribe.project_manifest import (
    load_project_manifest,
    public_project_state,
    revision_block,
)
from baroque_transcribe.storage_io import portable_name, sha256_file, sha256_json

EVENTS_DIR = "events"
SCORE_ARTIFACTS = (
    NOTES_JSON,
    PEDAL_JSON,
    RAW_MID,
    QUANTIZED_MID,
    SCORE_XML,
    ANNOTATIONS_JSON,
    SOURCE_MID,
)
LEGACY_SCORE_ARTIFACTS = (
    NOTES_JSON,
    PEDAL_JSON,
    RAW_MID,
    QUANTIZED_MID,
    SCORE_XML,
)


@dataclass(frozen=True)
class Snapshot:
    sha256: str
    artifacts: dict[str, str] = field(default_factory=dict)
    files: dict[str, str] = field(default_factory=dict)
    project_state_sha256: str = ""

    def to_dict(self) -> dict:
        return {
            "sha256": self.sha256,
            "project_state_sha256": self.project_state_sha256,
            "artifacts": dict(sorted(self.artifacts.items())),
            "files": dict(sorted(self.files.items())),
        }


@dataclass(frozen=True)
class RevisionStatus:
    state: str
    project: str
    local_sha256: str = ""
    lineage_id: str = ""
    base_event_id: str = ""
    published_event_ids: tuple[str, ...] = ()
    target_event_id: str = ""
    adoptable_event_id: str = ""
    message: str = ""

    @property
    def has_update(self) -> bool:
        return self.state in {"newer_published", "diverged", "published_conflict"}

    @property
    def has_local_changes(self) -> bool:
        return self.state in {"local_changes", "diverged", "legacy_unknown"}


def artifact_digest(path: Path) -> str:
    """Canonical component digest, excluding local paths and volatile XML IDs."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".musicxml", ".xml"}:
        try:
            root = element_tree.parse(path).getroot()
        except (OSError, element_tree.ParseError):
            return sha256_file(path)
        return sha256_json(_canonical_xml(root))
    if suffix != ".json":
        return sha256_file(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return sha256_file(path)
    if isinstance(payload, dict):
        payload = dict(payload)
        source = payload.get("source")
        if isinstance(source, str) and source and not _is_http(source):
            payload["source"] = portable_name(source)
    return sha256_json(payload)


def _canonical_xml(element: element_tree.Element):
    tag = element.tag.rsplit("}", 1)[-1]
    if tag == "encoding-date":
        return None
    attributes = {
        key.rsplit("}", 1)[-1]: value
        for key, value in element.attrib.items()
        if key.rsplit("}", 1)[-1] != "id"
    }
    children = [
        child
        for item in element
        if (child := _canonical_xml(item)) is not None
    ]
    return {
        "tag": tag,
        "attributes": dict(sorted(attributes.items())),
        "text": (element.text or "").strip(),
        "children": children,
    }


def artifact_inventory(folder: Path, names: tuple[str, ...] = SCORE_ARTIFACTS) -> dict[str, str]:
    inventory: dict[str, str] = {}
    for name in names:
        path = Path(folder) / name
        if path.is_file():
            inventory[name] = artifact_digest(path)
    return inventory


def file_inventory(folder: Path, names: tuple[str, ...] = SCORE_ARTIFACTS) -> dict[str, str]:
    return {
        name: sha256_file(Path(folder) / name)
        for name in names
        if (Path(folder) / name).is_file()
    }


def snapshot_for_folder(folder: Path, project_state: dict | None = None) -> Snapshot:
    artifacts = artifact_inventory(folder)
    files = file_inventory(folder)
    state_hash = sha256_json(project_state or {})
    return Snapshot(
        sha256=sha256_json(
            {
                "artifacts": dict(sorted(artifacts.items())),
                "project_state_sha256": state_hash,
            }
        ),
        artifacts=artifacts,
        files=files,
        project_state_sha256=state_hash,
    )


def snapshot_for_project(project: Path) -> Snapshot:
    manifest = load_project_manifest(project)
    return snapshot_for_folder(project, public_project_state(manifest))


def legacy_score_sha256(folder: Path) -> str:
    return sha256_json(artifact_inventory(folder, LEGACY_SCORE_ARTIFACTS))


def event_snapshot_sha256(event: dict) -> str:
    revision = event.get("revision")
    if not isinstance(revision, dict):
        return ""
    snapshot = revision.get("snapshot")
    return str(snapshot.get("sha256") or "") if isinstance(snapshot, dict) else ""


def event_lineage_id(event: dict) -> str:
    revision = event.get("revision")
    if not isinstance(revision, dict):
        return ""
    return str(revision.get("lineage_id") or "")


def event_parent_id(event: dict) -> str:
    revision = event.get("revision")
    if not isinstance(revision, dict):
        return ""
    return str(revision.get("parent_event_id") or "")


def event_is_complete(event: dict) -> bool:
    revision = event.get("revision")
    if isinstance(revision, dict) and "complete_snapshot" in revision:
        return bool(revision.get("complete_snapshot"))
    return str(event.get("action") or "") != "analyze"


def event_folder(store: Path, event: dict) -> Path:
    root = (Path(store) / EVENTS_DIR).resolve()
    ident = str(event.get("id") or "")
    if (
        not ident
        or ident in {".", ".."}
        or "/" in ident
        or "\\" in ident
        or ":" in ident
        or ident.startswith(".tmp-")
    ):
        raise ValueError("Event id must be a plain name.")
    candidate = Path(store) / EVENTS_DIR / ident
    if candidate.resolve().parent != root:
        raise ValueError("Event path escapes the event store.")
    return candidate


def event_heads(events: list[dict], lineage_id: str) -> list[dict]:
    members = [
        event
        for event in events
        if event_lineage_id(event) == lineage_id and event.get("status") == "ok"
    ]
    parents = {event_parent_id(event) for event in members if event_parent_id(event)}
    return [event for event in members if str(event.get("id") or "") not in parents]


def is_descendant(events: list[dict], event_id: str, ancestor_id: str) -> bool:
    by_id = {str(event.get("id") or ""): event for event in events}
    current = event_id
    seen: set[str] = set()
    while current and current not in seen:
        if current == ancestor_id:
            return True
        seen.add(current)
        event = by_id.get(current)
        if event is None:
            return False
        current = event_parent_id(event)
    return False


def revision_status(
    project: str,
    project_dir: Path | None,
    events: list[dict],
    store: Path,
) -> RevisionStatus:
    """Compare one local project with its causal published history."""

    relevant = [
        event
        for event in events
        if event.get("status") == "ok" and str(event.get("project") or "") == project
    ]
    score_events = [
        event
        for event in relevant
        if event_is_complete(event) and (event_folder(store, event) / NOTES_JSON).is_file()
    ]
    if project_dir is None or not (Path(project_dir) / NOTES_JSON).is_file():
        v2 = [event for event in relevant if event_lineage_id(event)]
        if v2:
            heads = [
                head
                for lineage in sorted({event_lineage_id(event) for event in v2})
                for head in event_heads(v2, lineage)
            ]
            if len(heads) > 1:
                head_ids = tuple(str(item.get("id") or "") for item in heads)
                return RevisionStatus(
                    state="published_conflict",
                    project=project,
                    published_event_ids=head_ids,
                    message="Several incomparable published revisions exist; select one explicitly.",
                )
        target = _preferred_unrestored_event(relevant, score_events)
        return RevisionStatus(
            state="not_restored" if target else "absent",
            project=project,
            published_event_ids=(
                (str(target.get("id") or ""),) if target is not None else ()
            ),
            target_event_id=str(target.get("id") or "") if target else "",
            message="Published history exists, but this project is not restored." if target else "",
        )

    local = snapshot_for_project(Path(project_dir))
    manifest = load_project_manifest(Path(project_dir))
    base = revision_block(manifest)
    base_id = str(base.get("base_event_id") or "")
    lineage_id = str(base.get("lineage_id") or "")
    by_id = {str(event.get("id") or ""): event for event in relevant}

    if base_id and lineage_id:
        base_event = by_id.get(base_id)
        if base_event is None:
            return RevisionStatus(
                state="legacy_unknown",
                project=project,
                local_sha256=local.sha256,
                lineage_id=lineage_id,
                base_event_id=base_id,
                message="The project's recorded base event is not present.",
            )
        other_lineages = sorted(
            {
                event_lineage_id(event)
                for event in relevant
                if event_lineage_id(event)
                and event_lineage_id(event) != lineage_id
            }
        )
        if other_lineages:
            heads: list[dict] = event_heads(relevant, lineage_id) or [base_event]
            for other_lineage in other_lineages:
                heads.extend(event_heads(relevant, other_lineage))
            head_ids = tuple(
                dict.fromkeys(str(item.get("id") or "") for item in heads)
            )
            return RevisionStatus(
                state="published_conflict",
                project=project,
                local_sha256=local.sha256,
                lineage_id=lineage_id,
                base_event_id=base_id,
                published_event_ids=head_ids,
                message="Published history contains independent project lineages.",
            )
        exact_descendants = [
            event
            for event in relevant
            if event_lineage_id(event) == lineage_id
            and str(event.get("id") or "") != base_id
            and event_snapshot_sha256(event) == local.sha256
        ]
        if exact_descendants:
            matched = exact_descendants[-1]
            ident = str(matched.get("id") or "")
            return RevisionStatus(
                state="legacy_match",
                project=project,
                local_sha256=local.sha256,
                lineage_id=lineage_id,
                base_event_id=base_id,
                published_event_ids=(ident,),
                target_event_id=ident,
                adoptable_event_id=ident,
                message=(
                    "The project exactly matches a published descendant, but its "
                    "revision marker was not advanced."
                ),
            )
        heads = event_heads(relevant, lineage_id)
        head_ids = tuple(str(event.get("id") or "") for event in heads)
        if not heads:
            heads = [base_event]
            head_ids = (base_id,)
        if len(heads) > 1:
            return RevisionStatus(
                state="published_conflict",
                project=project,
                local_sha256=local.sha256,
                lineage_id=lineage_id,
                base_event_id=base_id,
                published_event_ids=head_ids,
                message="Several incomparable published revisions exist.",
            )
        head = heads[0]
        head_id = str(head.get("id") or "")
        base_sha = event_snapshot_sha256(base_event) or str(base.get("base_snapshot_sha256") or "")
        local_changed = bool(base_sha and local.sha256 != base_sha)
        if head_id == base_id:
            state = "local_changes" if local_changed else "synchronized"
        elif is_descendant(relevant, head_id, base_id):
            state = "diverged" if local_changed else "newer_published"
        else:
            state = "published_conflict"
        return RevisionStatus(
            state=state,
            project=project,
            local_sha256=local.sha256,
            lineage_id=lineage_id,
            base_event_id=base_id,
            published_event_ids=head_ids,
            target_event_id=head_id,
            message=_status_message(state),
        )

    exact_v2 = [
        event for event in relevant if event_snapshot_sha256(event) == local.sha256
    ]
    if exact_v2:
        matched = exact_v2[-1]
        ident = str(matched.get("id") or "")
        return RevisionStatus(
            state="legacy_match",
            project=project,
            local_sha256=local.sha256,
            lineage_id=event_lineage_id(matched) or ident,
            base_event_id=ident,
            target_event_id=ident,
            adoptable_event_id=ident,
            message="This legacy project exactly matches a published event and can adopt it as its base.",
        )

    local_legacy = legacy_score_sha256(Path(project_dir))
    exact_v1 = [
        event
        for event in score_events
        if not event_lineage_id(event)
        and legacy_score_sha256(event_folder(store, event)) == local_legacy
    ]
    if exact_v1:
        matched = exact_v1[-1]
        ident = str(matched.get("id") or "")
        return RevisionStatus(
            state="legacy_match",
            project=project,
            local_sha256=local.sha256,
            lineage_id=ident,
            base_event_id=ident,
            target_event_id=ident,
            adoptable_event_id=ident,
            message="This legacy project exactly matches a published event and can adopt it as its base.",
        )
    if relevant:
        return RevisionStatus(
            state="legacy_unknown",
            project=project,
            local_sha256=local.sha256,
            published_event_ids=tuple(str(item.get("id") or "") for item in relevant),
            message="Local and published history exist, but their ancestry is unknown.",
        )
    return RevisionStatus(
        state="independent",
        project=project,
        local_sha256=local.sha256,
        message="This local project has no published ancestor.",
    )


def _preferred_unrestored_event(relevant: list[dict], score_events: list[dict]) -> dict | None:
    v2 = [event for event in relevant if event_lineage_id(event)]
    if v2:
        lineages = {event_lineage_id(event) for event in v2}
        heads = [head for lineage in lineages for head in event_heads(v2, lineage)]
        complete_heads = [head for head in heads if event_is_complete(head)]
        if complete_heads:
            return complete_heads[-1]
        if heads:
            # An analyze head restores from its complete ancestor; parse_store
            # resolves that base during materialization.
            return heads[-1]
    return score_events[-1] if score_events else None


def _status_message(state: str) -> str:
    return {
        "synchronized": "Local project matches the published head.",
        "local_changes": "This project has unpublished local changes.",
        "newer_published": "A newer published revision is available.",
        "diverged": "Both the local project and published history changed.",
        "published_conflict": "Published history has multiple incomparable heads.",
    }.get(state, "")


def _is_http(value: str) -> bool:
    text = value.strip().lower()
    return text.startswith("http://") or text.startswith("https://")
