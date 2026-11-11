"""Git-friendly history of each prepare, transcribe, and quantize.

One folder per run lives under ``data/parses/events/``. The folder is new
every time, and ``event.json`` is not rewritten, so two machines can commit
new runs and git can merge them. Score files and ``annotations.json`` are
copied into the folder. Audio stays in ``transcriptions/`` and is only
described here. A later harmonic edit is a new ``analyze`` event that
carries ``annotations.json`` alone.

``score.pdf`` is reserved. A later ``pdf`` command writes that file beside
the event, with a sibling ``pdf.json``, and leaves ``event.json`` as it was.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from baroque_transcribe.defaults import (
    ANNOTATIONS_JSON,
    DEFAULT_STAFF_SPLIT,
    DEFAULT_SUBDIVISION,
    DIAGNOSTICS_DIR,
    MANIFEST,
    NOTES_JSON,
    PEDAL_JSON,
    QUANTIZED_MID,
    RAW_MID,
    SCORE_XML,
    SOURCE_MID,
    SOURCE_WAV,
    WORKING_WAV,
)
from baroque_transcribe.playback import youtube_video_id
from baroque_transcribe.project_manifest import (
    public_project_state,
    read_project_manifest,
    revision_block,
    with_revision,
    write_project_manifest,
)
from baroque_transcribe.revision import (
    Snapshot,
    artifact_inventory,
    event_folder,
    event_heads,
    event_is_complete,
    event_lineage_id,
    event_parent_id,
    event_snapshot_sha256,
    is_descendant,
    revision_status,
    file_inventory,
    snapshot_for_folder,
)
from baroque_transcribe.storage_io import (
    copy_file_atomic,
    portable_name,
    sha256_file,
    write_json_atomic,
    write_text_atomic,
)

SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1
SCORE_PDF = "score.pdf"
NOTATION_HTML = "notation.html"
EVENTS_DIR = "events"

# These are the shareable products of a run. WAV, MP3, and FLAC stay out
# so a checkout does not carry the audio. annotations.json is the harmonic
# draft; a later edit is published as its own event so the file travels
# with git without rewriting an older event.json.
_STORED_SCORE_FILES = (
    NOTES_JSON,
    PEDAL_JSON,
    RAW_MID,
    QUANTIZED_MID,
    SCORE_XML,
    ANNOTATIONS_JSON,
    SOURCE_MID,
)
_REQUIRED_SCORE_FILES = (
    NOTES_JSON,
    PEDAL_JSON,
    RAW_MID,
    QUANTIZED_MID,
    SCORE_XML,
)
_COMPLETE_ACTIONS = {"transcribe", "quantize", "import"}


def default_store() -> Path:
    """``data/parses`` in the repository that contains this program."""

    return repo_root() / "data" / "parses"


def repo_root() -> Path:
    candidates = [Path.cwd(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "baroque_transcribe").is_dir():
            return candidate
    for candidate in candidates:
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return Path.cwd()


def resolve_store(store: Path | None) -> Path | None:
    """Use an explicit store, otherwise the repo store.

    Pytest calls the same pipeline functions as a real run. Those calls do
    not pass a store, and they must not write into the repository. An
    explicit store, including one passed from a test, is always used.
    """

    if store is not None:
        return store
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    return default_store()


def record_parse(
    *,
    action: str,
    status: str,
    source: str,
    project: str,
    store: Path | None = None,
    title: str = "",
    project_dir: Path | None = None,
    youtube_url: str | None = None,
    excerpt_start: float | None = None,
    excerpt_end: float | None = None,
    duration: float | None = None,
    full_duration: float | None = None,
    sample_rate: int | float | None = None,
    error: str | None = None,
    via: str | None = None,
    recorded_at: datetime | None = None,
    files: tuple[str, ...] | None = None,
) -> Path | None:
    """Write one new event folder. Return its path, or None if it could not be written.

    A failure here is reported and swallowed. The transcription result is
    already on disk and must not be discarded because the history could not
    be saved.
    """

    target = resolve_store(store)
    if target is None:
        return None
    when = recorded_at or datetime.now(timezone.utc)
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    staging: Path | None = None
    final: Path | None = None
    try:
        events = target / EVENTS_DIR
        events.mkdir(parents=True, exist_ok=True)
        ident, staging, final = _event_paths(events, when)
        staging.mkdir()
        copied = (
            _copy_scores(project_dir, staging, files)
            if status == "ok"
            else {name: False for name in _STORED_SCORE_FILES}
        )
        manifest = read_project_manifest(project_dir) if project_dir is not None else {}
        described_source = describe_source(source, title=title, youtube_url=youtube_url)
        manifest = _manifest_with_event_context(
            manifest,
            source=source,
            described_source=described_source,
            title=title,
            excerpt_start=excerpt_start,
            excerpt_end=excerpt_end,
            duration=duration,
            full_duration=full_duration,
            sample_rate=sample_rate,
        )
        if project_dir is not None and not manifest.get("tempo"):
            inferred_tempo = _score_tempo(project_dir / SCORE_XML)
            if inferred_tempo:
                manifest["tempo"] = inferred_tempo
                manifest.setdefault("staff_split", DEFAULT_STAFF_SPLIT)
        if project_dir is not None and not manifest.get("model"):
            try:
                notes_payload = json.loads(
                    (project_dir / NOTES_JSON).read_text(encoding="utf-8")
                )
                if isinstance(notes_payload, dict) and notes_payload.get("model"):
                    manifest["model"] = str(notes_payload["model"])
            except (OSError, json.JSONDecodeError):
                pass
        base = revision_block(manifest)
        parent_event_id = str(base.get("base_event_id") or "")
        lineage_id = str(base.get("lineage_id") or "") or parent_event_id or ident
        project_state = public_project_state(manifest)
        snapshot = None
        if status == "ok" and project_dir is not None:
            snapshot_source = (
                staging if action in _COMPLETE_ACTIONS else project_dir
            )
            snapshot = snapshot_for_folder(snapshot_source, project_state)
            if action == "analyze":
                snapshot = Snapshot(
                    sha256=snapshot.sha256,
                    artifacts=snapshot.artifacts,
                    files=file_inventory(staging),
                    project_state_sha256=snapshot.project_state_sha256,
                )
        complete_snapshot = action in _COMPLETE_ACTIONS and all(
            copied.get(name, False) for name in _REQUIRED_SCORE_FILES
        )
        if status == "ok" and action in _COMPLETE_ACTIONS and not complete_snapshot:
            missing = [
                name
                for name in _REQUIRED_SCORE_FILES
                if not copied.get(name, False)
            ]
            raise OSError(
                f"Cannot publish {action} event without a complete score snapshot: "
                f"{', '.join(missing)}."
            )
        payload = _payload(
            ident=ident,
            action=action,
            status=status,
            source=source,
            title=title,
            project=project,
            project_dir=project_dir if status == "ok" else None,
            youtube_url=youtube_url,
            excerpt_start=excerpt_start,
            excerpt_end=excerpt_end,
            duration=duration,
            full_duration=full_duration,
            sample_rate=sample_rate,
            error=error,
            via=via,
            recorded_at=when,
            copied=copied,
            lineage_id=lineage_id,
            parent_event_id=parent_event_id,
            complete_snapshot=complete_snapshot,
            snapshot=snapshot.to_dict() if snapshot is not None else {},
            project_state=project_state,
        )
        _write_json(staging / "event.json", payload)
        validate_event(payload)
        _validate_staged_event(staging, payload)
        os.replace(staging, final)
    except (OSError, ValueError) as exc:
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        print(f"warning: could not write parse history: {exc}", file=sys.stderr)
        return None
    if final is None:
        return None
    if status == "ok" and project_dir is not None and snapshot is not None:
        try:
            updated = with_revision(
                manifest,
                lineage_id=lineage_id,
                base_event_id=final.name,
                base_snapshot_sha256=snapshot.sha256,
            )
            write_project_manifest(project_dir, updated)
        except (OSError, ValueError) as exc:
            print(
                f"warning: parse event {final.name} was published, but the project revision marker was not updated: {exc}",
                file=sys.stderr,
            )
    return final


def load_events(store: Path | None = None) -> list[dict]:
    """Read every event, oldest first. A damaged file is skipped."""

    target = store if store is not None else default_store()
    events_dir = target / EVENTS_DIR
    if not events_dir.is_dir():
        return []
    resolved_events = events_dir.resolve()
    loaded: list[dict] = []
    for path in sorted(events_dir.glob("*/event.json")):
        if path.parent.name.startswith(".tmp-"):
            continue
        try:
            if path.resolve().parent.parent != resolved_events:
                print(
                    f"warning: skipped {path}: event path escapes the event store",
                    file=sys.stderr,
                )
                continue
        except OSError as exc:
            print(f"warning: skipped {path}: {exc}", file=sys.stderr)
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            validate_event(payload)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            print(f"warning: skipped {path}: {exc}", file=sys.stderr)
            continue
        loaded.append(payload)
    loaded.sort(key=lambda item: (str(item.get("recorded_at") or ""), str(item.get("id") or "")))
    return loaded


def validate_event(payload: object) -> dict:
    """Validate the event envelope while accepting immutable v1 history."""

    if not isinstance(payload, dict):
        raise ValueError("event.json is not a JSON object.")
    try:
        version = int(payload.get("version", LEGACY_SCHEMA_VERSION))
    except (TypeError, ValueError) as exc:
        raise ValueError("event.json has an invalid version.") from exc
    if version not in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}:
        raise ValueError(
            f"event.json uses schema version {version}; this program reads versions "
            f"{LEGACY_SCHEMA_VERSION} and {SCHEMA_VERSION}."
        )
    ident = str(payload.get("id") or "")
    if not ident:
        raise ValueError("event.json has no id.")
    _validate_plain_name(ident, "id")
    project = str(payload.get("project") or "")
    if project:
        _validate_plain_name(project, "project")
    if version == LEGACY_SCHEMA_VERSION:
        return payload
    revision = payload.get("revision")
    if not isinstance(revision, dict):
        raise ValueError("event.json v2 has no revision object.")
    if not str(revision.get("lineage_id") or ""):
        raise ValueError("event.json v2 has no lineage_id.")
    parent = revision.get("parent_event_id")
    if parent is not None and not isinstance(parent, str):
        raise ValueError("event.json v2 parent_event_id must be a string or null.")
    if parent:
        _validate_plain_name(parent, "revision.parent_event_id")
    complete_snapshot = revision.get("complete_snapshot")
    if not isinstance(complete_snapshot, bool):
        raise ValueError("event.json v2 complete_snapshot must be true or false.")
    if (
        payload.get("status") == "ok"
        and str(payload.get("action") or "") in _COMPLETE_ACTIONS
        and not complete_snapshot
    ):
        raise ValueError("A successful score event must contain a complete snapshot.")
    snapshot = revision.get("snapshot")
    if payload.get("status") == "ok":
        if not isinstance(snapshot, dict) or not str(snapshot.get("sha256") or ""):
            raise ValueError("A successful event.json v2 has no snapshot fingerprint.")
        artifacts = snapshot.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("event.json v2 snapshot artifacts must be an object.")
        for name, digest in artifacts.items():
            if name not in _STORED_SCORE_FILES or not isinstance(digest, str):
                raise ValueError(
                    "event.json v2 snapshot artifacts contain an invalid file or hash."
                )
        files = snapshot.get("files")
        if files is not None and (
            not isinstance(files, dict)
            or any(
                name not in _STORED_SCORE_FILES or not isinstance(digest, str)
                for name, digest in files.items()
            )
        ):
            raise ValueError(
                "event.json v2 snapshot files contain an invalid file or hash."
            )
    project_state = payload.get("project_state")
    if project_state is not None and not isinstance(project_state, dict):
        raise ValueError("event.json v2 project_state must be an object.")
    formats = payload.get("formats")
    if formats is not None:
        if not isinstance(formats, dict):
            raise ValueError("event.json v2 formats must be an object.")
        for entry in formats.values():
            if not isinstance(entry, dict):
                continue
            filename = str(entry.get("file") or "")
            if filename:
                _validate_plain_name(filename, "formats file")
    source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
    if payload.get("status") == "ok" and source.get("kind") == "midi":
        if not str(source.get("sha256") or ""):
            raise ValueError("A successful MIDI event has no source SHA-256.")
        formats = payload.get("formats") if isinstance(payload.get("formats"), dict) else {}
        source_midi = (
            formats.get("source_midi")
            if isinstance(formats.get("source_midi"), dict)
            else {}
        )
        if revision.get("complete_snapshot") and not source_midi.get("stored"):
            raise ValueError("A complete MIDI event does not store source.mid.")
        if revision.get("complete_snapshot"):
            state = project_state if isinstance(project_state, dict) else {}
            portable_source = (
                state.get("source") if isinstance(state.get("source"), dict) else {}
            )
            tempo = state.get("tempo") if isinstance(state.get("tempo"), dict) else {}
            notation = (
                state.get("notation") if isinstance(state.get("notation"), dict) else {}
            )
            if str(portable_source.get("sha256") or "") != str(source.get("sha256") or ""):
                raise ValueError("A complete MIDI event has inconsistent source hashes.")
            if not isinstance(tempo.get("tempo_map"), list) or not isinstance(
                tempo.get("time_signature_map"), list
            ):
                raise ValueError("A complete MIDI event has no full tempo or meter map.")
            if not str(state.get("staff_assignment") or ""):
                raise ValueError("A complete MIDI event has no staff assignment provenance.")
            if not str(notation.get("score_from") or ""):
                raise ValueError("A complete MIDI event has no score provenance.")
    return payload


def _validate_plain_name(value: str, field: str) -> None:
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or ":" in value
        or portable_name(value) != value
        or value == ".recovery"
        or value.startswith(".tmp-")
    ):
        raise ValueError(f"event.json {field} must be a plain name.")


def _validate_staged_event(folder: Path, payload: dict) -> None:
    if payload.get("status") != "ok":
        return
    revision = payload["revision"]
    snapshot = revision["snapshot"]
    artifacts = snapshot.get("artifacts") or {}
    files = snapshot.get("files") or {}
    formats = payload.get("formats") if isinstance(payload.get("formats"), dict) else {}
    stored_names = {
        str(entry.get("file") or "")
        for entry in formats.values()
        if isinstance(entry, dict) and entry.get("stored")
    }
    for name in sorted(stored_names):
        path = folder / name
        if not path.is_file():
            raise OSError(f"Staged event is missing {name}.")
        expected_file = str(files.get(name) or "")
        expected_component = str(artifacts.get(name) or "")
        if expected_file:
            if sha256_file(path) != expected_file:
                raise OSError(f"Staged event byte hash mismatch for {name}.")
        elif expected_component:
            from baroque_transcribe.revision import artifact_digest

            if artifact_digest(path) != expected_component:
                raise OSError(f"Staged event hash mismatch for {name}.")
    if revision.get("complete_snapshot"):
        missing = [name for name in artifacts if not (folder / name).is_file()]
        if missing:
            raise OSError(f"Complete staged event is missing: {', '.join(sorted(missing))}.")
        missing_files = [name for name in files if not (folder / name).is_file()]
        if missing_files:
            raise OSError(
                f"Complete staged event is missing byte-hashed files: "
                f"{', '.join(sorted(missing_files))}."
            )
        required = list(_REQUIRED_SCORE_FILES)
        source = payload.get("source") if isinstance(payload.get("source"), dict) else {}
        if source.get("kind") == "midi":
            required.append(SOURCE_MID)
        absent = [name for name in required if not (folder / name).is_file()]
        if absent:
            raise OSError(f"Complete staged event lacks required files: {', '.join(absent)}.")
        if source.get("kind") == "midi":
            source_hash = str(source.get("sha256") or "")
            if not source_hash or sha256_file(folder / SOURCE_MID) != source_hash:
                raise OSError("Staged source.mid does not match the MIDI source hash.")
        from baroque_transcribe.revision import snapshot_for_folder

        actual = snapshot_for_folder(folder, payload.get("project_state") or {})
        if actual.sha256 != str(snapshot.get("sha256") or ""):
            raise OSError("Complete staged event aggregate snapshot hash mismatch.")


def thumbnail_url(video_id: str) -> str:
    """Public YouTube thumbnail. No API key is required."""

    return f"https://i.ytimg.com/vi/{video_id}/hqdefault.jpg"


def transcribed_videos(store: Path | None = None) -> list[dict]:
    """One card per YouTube video that has a finished transcription.

    Several runs of the same video share a card. The newest video is first.
    A prepare that never produced notes is left out.
    """

    grouped: dict[str, dict] = {}
    for event in load_events(store):
        if not _transcribed_youtube(event):
            continue
        source = event.get("source") if isinstance(event.get("source"), dict) else {}
        video_id = str(source.get("video_id") or "")
        title = _display_title(str(source.get("title") or ""))
        recorded = str(event.get("recorded_at") or "")
        project = str(event.get("project") or "")
        url = str(source.get("url") or f"https://www.youtube.com/watch?v={video_id}")
        card = grouped.get(video_id)
        if card is None:
            grouped[video_id] = {
                "video_id": video_id,
                "title": title or video_id,
                "url": url,
                "thumbnail": thumbnail_url(video_id),
                "projects": [project] if project else [],
                "recorded_at": recorded,
            }
            continue
        if title and (_is_generic_title(card["title"]) or card["title"] == video_id):
            card["title"] = title
        if project and project not in card["projects"]:
            card["projects"].append(project)
        if recorded >= str(card["recorded_at"]):
            card["recorded_at"] = recorded
            card["url"] = url
    cards = sorted(grouped.values(), key=lambda item: str(item["recorded_at"]), reverse=True)
    for card in cards:
        card.pop("recorded_at", None)
    return cards


def _transcribed_youtube(event: dict) -> bool:
    if event.get("status") != "ok":
        return False
    source = event.get("source") if isinstance(event.get("source"), dict) else {}
    if not source.get("video_id"):
        return False
    formats = event.get("formats") if isinstance(event.get("formats"), dict) else {}
    notes = formats.get("notes_json") if isinstance(formats.get("notes_json"), dict) else {}
    if notes.get("present") or notes.get("stored"):
        return True
    return event.get("action") == "transcribe"


def _display_title(title: str) -> str:
    text = title.strip()
    if _is_generic_title(text):
        return ""
    return text


def _is_generic_title(title: str) -> bool:
    return title.strip().lower() in {"", "source"}


def import_projects(root: Path, store: Path | None = None) -> tuple[int, int]:
    """Copy existing projects into the store.

    A second import skips a project that was already imported with the same
    video (or file name), excerpt, and folder name. Returns ``(added, skipped)``.
    """

    target = resolve_store(store)
    if target is None:
        return 0, 0
    existing = {
        key
        for event in load_events(target)
        if (key := import_identity(event)) is not None
    }
    added = 0
    skipped = 0
    for manifest_path in _manifests(root):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: skipped {manifest_path}: {exc}", file=sys.stderr)
            continue
        if not isinstance(manifest, dict):
            print(f"warning: skipped {manifest_path}: manifest is not an object", file=sys.stderr)
            continue
        origin = str(manifest.get("origin") or "")
        youtube_url = str(manifest.get("youtube_url") or "") or None
        if not origin and not youtube_url:
            print(f"warning: skipped {manifest_path}: no origin", file=sys.stderr)
            continue
        project_dir = manifest_path.parent
        described = describe_source(origin, title=str(manifest.get("title") or ""), youtube_url=youtube_url)
        audio = manifest.get("audio") if isinstance(manifest.get("audio"), dict) else {}
        identity = (
            described.get("video_id") or described.get("name") or described.get("url") or "",
            _number(audio.get("excerpt_start_seconds")),
            _number(audio.get("excerpt_end_seconds")),
            project_dir.name,
        )
        if identity in existing:
            skipped += 1
            continue
        action = "transcribe" if (project_dir / NOTES_JSON).is_file() else "prepare"
        written = record_parse(
            action=action,
            status="ok",
            source=origin,
            project=project_dir.name,
            store=target,
            title=str(manifest.get("title") or ""),
            project_dir=project_dir,
            youtube_url=youtube_url,
            excerpt_start=_number(audio.get("excerpt_start_seconds")),
            excerpt_end=_number(audio.get("excerpt_end_seconds")),
            duration=_number(audio.get("working_duration_seconds")),
            full_duration=_number(audio.get("full_duration_seconds")),
            sample_rate=audio.get("sample_rate"),
            via="import",
            recorded_at=datetime.fromtimestamp(manifest_path.stat().st_mtime, timezone.utc),
        )
        if written is None:
            continue
        existing.add(identity)
        added += 1
    return added, skipped


def publish_analysis(project_dir: Path, store: Path | None = None) -> Path | None:
    """Publish ``annotations.json`` as a new history event.

    The event carries that file only. The notes and the score stay in the
    earlier run, and ``event.json`` files are not rewritten, so two machines
    can each publish an analysis and git merges the new folders.
    """

    annotations = project_dir / ANNOTATIONS_JSON
    if not annotations.is_file():
        return None
    target = resolve_store(store)
    if target is None:
        return None
    manifest = read_project_manifest(project_dir)
    events = load_events(target)
    status = revision_status(project_dir.name, project_dir, events, target)
    if status.state == "legacy_match" and status.adoptable_event_id:
        matched = next(
            (
                event
                for event in events
                if str(event.get("id") or "") == status.adoptable_event_id
            ),
            None,
        )
        if matched is not None:
            manifest = with_revision(
                manifest,
                lineage_id=event_lineage_id(matched)
                or status.adoptable_event_id,
                base_event_id=status.adoptable_event_id,
                base_snapshot_sha256=event_snapshot_sha256(matched)
                or snapshot_for_folder(
                    event_folder(target, matched),
                    public_project_state(manifest),
                ).sha256,
            )
            write_project_manifest(project_dir, manifest)
    base = revision_block(manifest)
    base_event_id = str(base.get("base_event_id") or "")
    if not base_event_id:
        print(
            "warning: analysis remains local because no exact published score base exists.",
            file=sys.stderr,
        )
        return None
    try:
        from baroque_transcribe.project_restore import resolve_event_selection

        selection = resolve_event_selection(target, events, base_event_id)
    except (OSError, ValueError) as exc:
        print(f"warning: analysis remains local: {exc}", file=sys.stderr)
        return None
    local_score = artifact_inventory(project_dir)
    local_score.pop(ANNOTATIONS_JSON, None)
    score_revision = (
        selection.score.get("revision")
        if isinstance(selection.score.get("revision"), dict)
        else {}
    )
    score_snapshot = (
        score_revision.get("snapshot")
        if isinstance(score_revision.get("snapshot"), dict)
        else {}
    )
    published_score = dict(
        score_snapshot.get("artifacts")
        if isinstance(score_snapshot.get("artifacts"), dict)
        else artifact_inventory(selection.score_folder)
    )
    published_score.pop(ANNOTATIONS_JSON, None)
    if local_score != published_score:
        print(
            "warning: analysis remains local because the project score no longer "
            "matches its causal published base.",
            file=sys.stderr,
        )
        return None
    audio = manifest.get("audio") if isinstance(manifest.get("audio"), dict) else {}
    return record_parse(
        action="analyze",
        status="ok",
        source=str(manifest.get("origin") or ""),
        title=str(manifest.get("title") or project_dir.name),
        project=project_dir.name,
        project_dir=project_dir,
        youtube_url=str(manifest.get("youtube_url") or "") or None,
        excerpt_start=_number(audio.get("excerpt_start_seconds")),
        excerpt_end=_number(audio.get("excerpt_end_seconds")),
        duration=_number(audio.get("working_duration_seconds")),
        full_duration=_number(audio.get("full_duration_seconds")),
        sample_rate=audio.get("sample_rate"),
        store=target,
        files=(ANNOTATIONS_JSON,),
    )


def restore_projects(
    store: Path | None = None,
    projects_root: Path | None = None,
    projects: list[str] | None = None,
    *,
    with_audio: bool = False,
) -> tuple[list[Path], list[str]]:
    """Non-destructively restore missing projects from validated event snapshots."""

    target = resolve_store(store)
    if target is None:
        return [], []
    dest_base = projects_root if projects_root is not None else repo_root() / "transcriptions"
    from baroque_transcribe.project_restore import restore_projects_compatible

    return restore_projects_compatible(
        store=target,
        projects_root=dest_base,
        projects=projects,
        with_audio=with_audio,
    )


def _restore_projects_legacy(
    store: Path | None = None,
    projects_root: Path | None = None,
    projects: list[str] | None = None,
    *,
    with_audio: bool = False,
) -> tuple[list[Path], list[str]]:
    """Rebuild project folders from the score files stored in the history.

    This is the receiving-machine half of the two-computer workflow. The
    stored ``notes.json``, pedal, MIDI, and MusicXML files are copied into
    ``transcriptions/<project>`` and a manifest is synthesized from the
    event (with the tempo and meter read back from the MusicXML). Nothing
    is downloaded and no model is run, so a pulled transcription can be
    opened locally without transcribing it again.     A project that already
    has a ``notes.json`` locally is left untouched, except that a missing
    ``annotations.json`` is filled from history. An existing analysis file
    is not replaced.

    With ``with_audio``, the excerpt audio is also fetched -- for a YouTube
    source that is the section download only, never a re-transcription.
    Without it the project simply has no ``working.wav``: the score and the
    synced video still play, the Listen tab has nothing to play.

    Returns ``(restored, skipped)``: the folders written, and the project
    names that already existed locally or have no stored run.
    """

    target = resolve_store(store)
    if target is None:
        return [], []
    dest_base = projects_root if projects_root is not None else repo_root() / "transcriptions"
    wanted = {Path(name).name for name in projects} if projects else None
    latest: dict[str, tuple[dict, Path]] = {}
    annotations: dict[str, tuple[dict, Path]] = {}
    for event in load_events(target):
        if event.get("status") != "ok":
            continue
        project = str(event.get("project") or "")
        if not project or (wanted is not None and project not in wanted):
            continue
        folder = target / EVENTS_DIR / str(event.get("id") or "")
        if (folder / NOTES_JSON).is_file():
            # load_events is oldest first, so a later run of the same
            # project replaces an earlier one here.
            latest[project] = (event, folder)
        if (folder / ANNOTATIONS_JSON).is_file():
            annotations[project] = (event, folder)
    restored: list[Path] = []
    skipped: list[str] = []
    if wanted is not None:
        skipped.extend(sorted(wanted - set(latest)))
    for project in sorted(latest):
        event, folder = latest[project]
        dest = dest_base / project
        newer_annotations = _newer_annotations(event, folder, annotations.get(project))
        if (dest / NOTES_JSON).is_file():
            if newer_annotations is not None and not (dest / ANNOTATIONS_JSON).is_file():
                _copy_score_file(newer_annotations, dest / ANNOTATIONS_JSON)
                restored.append(dest)
            elif (folder / ANNOTATIONS_JSON).is_file() and not (dest / ANNOTATIONS_JSON).is_file():
                _copy_score_file(folder / ANNOTATIONS_JSON, dest / ANNOTATIONS_JSON)
                restored.append(dest)
            else:
                skipped.append(project)
            continue
        dest.mkdir(parents=True, exist_ok=True)
        copied: dict[str, bool] = {}
        for name in _STORED_SCORE_FILES:
            source_file = folder / name
            copied[name] = source_file.is_file()
            if copied[name]:
                _copy_score_file(source_file, dest / name)
        if newer_annotations is not None:
            _copy_score_file(newer_annotations, dest / ANNOTATIONS_JSON)
            copied[ANNOTATIONS_JSON] = True
        manifest = _restored_manifest(event, folder, copied)
        if with_audio:
            _restore_audio(event, dest, manifest)
        _write_json(dest / MANIFEST, manifest)
        restored.append(dest)
    return restored, skipped


def _newer_annotations(
    score_event: dict,
    score_folder: Path,
    found: tuple[dict, Path] | None,
) -> Path | None:
    """Annotations published at or after the score run, from another event."""

    if found is None:
        return None
    event, folder = found
    if folder == score_folder:
        return None
    if str(event.get("recorded_at") or "") < str(score_event.get("recorded_at") or ""):
        return None
    path = folder / ANNOTATIONS_JSON
    return path if path.is_file() else None


def _restored_manifest(event: dict, event_folder: Path, copied: dict[str, bool]) -> dict:
    """A project manifest rebuilt from an event, for a restored project.

    The event does not store everything a live run writes (thresholds,
    tempo estimates, diagnostics). The fields the library, project page,
    and synced playback read are restored; the rest is left out rather
    than invented.
    """

    source = event.get("source") if isinstance(event.get("source"), dict) else {}
    audio = event.get("audio") if isinstance(event.get("audio"), dict) else {}
    working = audio.get("working_wav") if isinstance(audio.get("working_wav"), dict) else {}
    origin = str(source.get("url") or source.get("name") or "")
    notes_count: int | None = None
    pedals_count: int | None = None
    model = ""
    try:
        notes_payload = json.loads((event_folder / NOTES_JSON).read_text(encoding="utf-8"))
        notes_count = len(notes_payload.get("notes") or [])
        pedals_count = len(notes_payload.get("pedals") or [])
        model = str(notes_payload.get("model") or "")
    except (OSError, json.JSONDecodeError, AttributeError):
        pass
    warnings = [
        f"Restored from parse history event {event.get('id')} recorded on "
        f"{event.get('machine') or 'another machine'}. The audio was not "
        "downloaded again and no model was run. working.wav is missing "
        "unless the restore fetched it; the score and synced video play "
        "without it."
    ]
    tempo = _score_tempo(event_folder / SCORE_XML)
    if not tempo:
        warnings.append(
            "The tempo and meter could not be read from the restored score. "
            "Set them on the Notes tab and rebuild to enable synced playback."
        )
    manifest: dict = {
        "program": "baroque-transcribe (restored from parse history)",
        "origin": origin,
        "title": str(source.get("title") or ""),
        "model": model,
        "restored_from": {
            "event": str(event.get("id") or ""),
            "machine": str(event.get("machine") or ""),
            "recorded_at": str(event.get("recorded_at") or ""),
        },
        "counts": {"notes": notes_count, "pedals": pedals_count},
        "audio": {
            "source": SOURCE_WAV,
            "working": WORKING_WAV,
            "sample_rate": working.get("sample_rate"),
            "full_duration_seconds": (audio.get("source_wav") or {}).get("full_duration_seconds"),
            "excerpt_start_seconds": working.get("excerpt_start_seconds"),
            "excerpt_end_seconds": working.get("excerpt_end_seconds"),
            "working_duration_seconds": working.get("duration_seconds"),
        },
        "warnings": warnings,
        "notation": {
            "quantized_mid": QUANTIZED_MID,
            "score": SCORE_XML,
            "raw_mid": RAW_MID,
        },
    }
    if source.get("kind") == "youtube" and origin:
        manifest["youtube_url"] = origin
    if tempo:
        manifest["tempo"] = tempo
        manifest["staff_split"] = DEFAULT_STAFF_SPLIT
    return manifest


def _score_tempo(score_path: Path) -> dict:
    """Tempo, meter, and grid read back from a MusicXML score, or ``{}``.

    Synced playback re-derives measure times from the manifest tempo and
    the notes, so a restored project needs the values the score was made
    with. They are written in the score itself: the first ``<time>`` and
    the metronome ``<sound tempo=...>``.
    """

    if not score_path.is_file():
        return {}
    try:
        import xml.etree.ElementTree as element_tree

        root = element_tree.parse(score_path).getroot()
    except (OSError, element_tree.ParseError):  # type: ignore[attr-defined]
        return {}
    bpm: float | None = None
    signature = ""
    for element in root.iter():
        if bpm is None and element.tag == "sound" and element.get("tempo"):
            try:
                bpm = float(element.get("tempo"))  # type: ignore[arg-type]
            except ValueError:
                pass
        if bpm is None and element.tag == "per-minute" and element.text:
            try:
                bpm = float(element.text.strip())
            except ValueError:
                pass
        if not signature and element.tag == "time":
            beats = element.findtext("beats")
            beat_type = element.findtext("beat-type")
            if beats and beat_type:
                signature = f"{beats.strip()}/{beat_type.strip()}"
        if bpm is not None and signature:
            break
    if bpm is None or not signature:
        return {}
    return {
        "bpm": bpm,
        "downbeat_seconds": 0.0,
        "subdivision": DEFAULT_SUBDIVISION,
        "time_signature": signature,
        "time_signature_assumed": False,
        "source": "restored-from-score",
        "assumed_bpm": False,
    }


def _restore_audio(event: dict, dest: Path, manifest: dict) -> None:
    """Fetch the excerpt audio for a restored project, in place.

    Only a YouTube source can be fetched: a local file is stored in the
    history by file name only. This downloads audio; it never transcribes.
    A failure leaves the restored project without audio and says so in the
    manifest warnings instead of failing the restore.
    """

    source = event.get("source") if isinstance(event.get("source"), dict) else {}
    url = str(source.get("url") or "")
    warnings = manifest.setdefault("warnings", [])
    if source.get("kind") != "youtube" or not url:
        warnings.append("The excerpt audio was not fetched: the source is not a YouTube URL in the history.")
        return
    excerpt = event.get("excerpt") if isinstance(event.get("excerpt"), dict) else {}
    try:
        from baroque_transcribe.audio import prepare_audio

        prepare_audio(
            url,
            dest,
            start=_number(excerpt.get("start_seconds")),
            end=_number(excerpt.get("end_seconds")),
        )
    except Exception as exc:  # audio fetch must not discard a good restore
        warnings.append(f"The excerpt audio could not be fetched: {exc}")
        return
    manifest["warnings"] = [
        item for item in warnings if not item.startswith("Restored from parse history")
    ]
    warnings = manifest["warnings"]
    warnings.append(
        f"Restored from parse history event {event.get('id')}; the excerpt audio "
        "was downloaded (section only) and no model was run."
    )


def import_identity(event: dict) -> tuple | None:
    """Identity of an imported project, or None for a live run."""

    if event.get("via") != "import":
        return None
    source = event.get("source") if isinstance(event.get("source"), dict) else {}
    excerpt = event.get("excerpt") if isinstance(event.get("excerpt"), dict) else {}
    key = source.get("video_id") or source.get("name") or source.get("url") or ""
    return (
        key,
        _number(excerpt.get("start_seconds")),
        _number(excerpt.get("end_seconds")),
        str(event.get("project") or ""),
    )


def format_event(event: dict) -> str:
    source = event.get("source") if isinstance(event.get("source"), dict) else {}
    label = source.get("title") or source.get("video_id") or source.get("name") or source.get("url") or ""
    return (
        f"{event.get('recorded_at')}  {event.get('machine')}  "
        f"{event.get('action')}  {event.get('status')}  {event.get('project')}  {label}"
    )


def _payload(
    *,
    ident: str,
    action: str,
    status: str,
    source: str,
    title: str,
    project: str,
    project_dir: Path | None,
    youtube_url: str | None,
    excerpt_start: float | None,
    excerpt_end: float | None,
    duration: float | None,
    full_duration: float | None,
    sample_rate: int | float | None,
    error: str | None,
    via: str | None,
    recorded_at: datetime,
    copied: dict[str, bool],
    lineage_id: str,
    parent_event_id: str,
    complete_snapshot: bool,
    snapshot: dict,
    project_state: dict,
) -> dict:
    start = _number(excerpt_start)
    end = _number(excerpt_end)
    described_source = describe_source(source, title=title, youtube_url=youtube_url)
    portable_source = (
        project_state.get("source")
        if isinstance(project_state.get("source"), dict)
        else {}
    )
    if (
        described_source.get("kind") in {"file", "midi"}
        and portable_source.get("sha256")
    ):
        described_source["sha256"] = str(portable_source["sha256"])
    payload = {
        "version": SCHEMA_VERSION,
        "id": ident,
        "recorded_at": recorded_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "machine": platform.node(),
        "action": action,
        "status": status,
        "via": via,
        "project": portable_name(project) if project else "",
        "source": described_source,
        "excerpt": {"start_seconds": start, "end_seconds": end},
        "audio": {
            "source_wav": {
                "file": SOURCE_WAV,
                "stored": False,
                "full_duration_seconds": _number(full_duration),
            },
            "working_wav": {
                "file": WORKING_WAV,
                "stored": False,
                "duration_seconds": _number(duration),
                "sample_rate": int(sample_rate) if sample_rate is not None else None,
                "excerpt_start_seconds": start,
                "excerpt_end_seconds": end,
            },
            "diagnostics": _diagnostic_names(project_dir),
        },
        "formats": _formats(copied, project_dir),
        "project_state": project_state,
        "revision": {
            "lineage_id": lineage_id,
            "parent_event_id": parent_event_id or None,
            "complete_snapshot": complete_snapshot,
            "snapshot": snapshot,
        },
        "error": public_error(error, source) if error else None,
    }
    if via is None:
        payload.pop("via")
    return payload


def describe_source(source: str, title: str = "", youtube_url: str | None = None) -> dict:
    """Public description of an input. Local absolute paths are reduced to a file name."""

    name = portable_name(source) if source else ""
    extension = Path(name).suffix.lower()
    youtube = ""
    for candidate in (youtube_url or "", source or ""):
        if youtube_video_id(candidate):
            youtube = candidate.strip()
            break
    if source and not _is_http(source):
        source_kind = "midi" if extension in {".mid", ".midi"} else "file"
        described = {
            "kind": source_kind,
            "name": name,
            "extension": extension,
            "title": title,
        }
        if youtube:
            described["youtube_url"] = youtube
            described["video_id"] = youtube_video_id(youtube)
        try:
            path = Path(source).expanduser()
            if path.is_file():
                described["sha256"] = sha256_file(path)
        except OSError:
            pass
        return described
    if youtube:
        described: dict = {
            "kind": "youtube",
            "url": youtube,
            "video_id": youtube_video_id(youtube),
            "title": title,
        }
        if source and not _is_http(source):
            described["local_name"] = portable_name(source)
        return described
    if _is_http(source):
        return {"kind": "url", "url": source.strip(), "title": title}
    return {
        "kind": "file",
        "name": name,
        "extension": extension,
        "title": title,
    }


def _manifest_with_event_context(
    manifest: dict,
    *,
    source: str,
    described_source: dict,
    title: str,
    excerpt_start: float | None,
    excerpt_end: float | None,
    duration: float | None,
    full_duration: float | None,
    sample_rate: int | float | None,
) -> dict:
    """Fill portability fields missing from older/minimal project manifests."""

    updated = dict(manifest)
    if not updated.get("origin") and source:
        updated["origin"] = source
    if not updated.get("title") and title:
        updated["title"] = title
    kind = str(described_source.get("kind") or "")
    if not updated.get("source_kind") and kind:
        updated["source_kind"] = kind
    if kind == "youtube" and not updated.get("youtube_url"):
        updated["youtube_url"] = str(described_source.get("url") or "")
    if described_source.get("sha256") and not updated.get("source_sha256"):
        updated["source_sha256"] = str(described_source["sha256"])
    audio = dict(updated.get("audio") or {})
    values = {
        "sample_rate": int(sample_rate) if sample_rate is not None else None,
        "full_duration_seconds": _number(full_duration),
        "excerpt_start_seconds": _number(excerpt_start),
        "excerpt_end_seconds": _number(excerpt_end),
        "working_duration_seconds": _number(duration),
    }
    for key, value in values.items():
        if key not in audio and value is not None:
            audio[key] = value
    if audio:
        updated["audio"] = audio
    return updated


def public_error(message: str, source: str = "") -> str:
    """Short error text with the home directory and local absolute paths removed."""

    text = " ".join(message.split())
    home = Path.home()
    for variant in {str(home), str(home).replace("\\", "/")}:
        if variant:
            text = text.replace(variant, "~")
    if source and not _is_http(source):
        candidates = {source, str(Path(source).expanduser())}
        try:
            candidates.add(str(Path(source).expanduser().resolve()))
        except OSError:
            pass
        expanded = set(candidates)
        for item in candidates:
            expanded.add(item.replace("\\", "/"))
        for item in sorted(expanded, key=len, reverse=True):
            if item and item in text:
                text = text.replace(item, portable_name(item))
    return text[:500]


def _formats(copied: dict[str, bool], project_dir: Path | None) -> dict:
    notation_present = bool(project_dir and (project_dir / NOTATION_HTML).is_file())
    return {
        "notes_json": _format_entry(NOTES_JSON, copied.get(NOTES_JSON, False)),
        "pedal_json": _format_entry(PEDAL_JSON, copied.get(PEDAL_JSON, False)),
        "raw_midi": _format_entry(RAW_MID, copied.get(RAW_MID, False)),
        "quantized_midi": _format_entry(QUANTIZED_MID, copied.get(QUANTIZED_MID, False)),
        "musicxml": _format_entry(SCORE_XML, copied.get(SCORE_XML, False)),
        "annotations_json": _format_entry(ANNOTATIONS_JSON, copied.get(ANNOTATIONS_JSON, False)),
        "source_midi": _format_entry(SOURCE_MID, copied.get(SOURCE_MID, False)),
        "notation_html": {"file": NOTATION_HTML, "present": notation_present, "stored": False},
        "score_pdf": {
            "file": SCORE_PDF,
            "present": False,
            "stored": False,
            "status": "planned",
        },
    }


def _format_entry(filename: str, stored: bool) -> dict:
    return {"file": filename, "present": stored, "stored": stored}


def _copy_scores(
    project_dir: Path | None,
    dest: Path,
    files: tuple[str, ...] | None = None,
) -> dict[str, bool]:
    copied = {name: False for name in _STORED_SCORE_FILES}
    if project_dir is None:
        return copied
    for name in (files if files is not None else _STORED_SCORE_FILES):
        src = project_dir / name
        if src.is_file():
            _copy_score_file(src, dest / name)
            copied[name] = True
    return copied


def _copy_score_file(src: Path, dest: Path) -> None:
    """Copy a score file. A local path in a JSON ``source`` field is reduced to its name."""

    if src.suffix.lower() != ".json":
        copy_file_atomic(src, dest)
        return
    text = src.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        write_text_atomic(dest, text)
        return
    source = payload.get("source") if isinstance(payload, dict) else None
    if isinstance(source, str) and source and not _is_http(source):
        encoded = json.dumps(source, ensure_ascii=False)
        replacement = json.dumps(portable_name(source), ensure_ascii=False)
        if encoded in text:
            text = text.replace(encoded, replacement, 1)
    write_text_atomic(dest, text)


def _diagnostic_names(project_dir: Path | None) -> list[str]:
    if project_dir is None:
        return []
    folder = project_dir / DIAGNOSTICS_DIR
    if not folder.is_dir():
        return []
    return sorted(path.name for path in folder.iterdir() if path.is_file())


def _manifests(root: Path) -> list[Path]:
    skip = {".venv", "site-packages", "data", ".git"}
    found: list[Path] = []
    for path in root.expanduser().rglob("manifest.json"):
        if any(part in skip for part in path.parts):
            continue
        found.append(path)
    return sorted(found)


def _event_paths(events: Path, when: datetime) -> tuple[str, Path, Path]:
    for _ in range(5):
        ident = _event_id(when)
        final = events / ident
        staging = events / f".tmp-{ident}"
        if final.exists() or staging.exists():
            continue
        return ident, staging, final
    raise OSError("Could not allocate a parse-history folder.")


def _event_id(when: datetime) -> str:
    stamp = when.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _write_json(path: Path, payload: dict) -> None:
    write_json_atomic(path, payload, sort_keys=True)


def _number(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return round(float(value), 4)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _is_http(value: str) -> bool:
    text = value.strip().lower()
    return text.startswith("http://") or text.startswith("https://")
