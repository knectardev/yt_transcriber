"""Versioned project-manifest IO and its portable published projection."""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from baroque_transcribe.defaults import MANIFEST
from baroque_transcribe.storage_io import portable_name, sha256_file, write_json_atomic

SCHEMA_VERSION = 1


class ProjectManifestError(ValueError):
    pass


def manifest_path(project_or_path: Path) -> Path:
    path = Path(project_or_path)
    return path if path.name == MANIFEST else path / MANIFEST


def validate_project_manifest(payload: object) -> dict:
    """Validate the compatible envelope while leaving feature fields open."""

    if not isinstance(payload, dict):
        raise ProjectManifestError("manifest.json is not a JSON object.")
    version = payload.get("schema_version")
    if version is not None:
        try:
            parsed = int(version)
        except (TypeError, ValueError) as exc:
            raise ProjectManifestError("manifest.json has an invalid schema_version.") from exc
        if parsed != SCHEMA_VERSION:
            raise ProjectManifestError(
                f"manifest.json uses schema version {parsed}; this program reads version {SCHEMA_VERSION}."
            )
    revision = payload.get("revision")
    if revision is not None and not isinstance(revision, dict):
        raise ProjectManifestError("manifest.json revision must be an object.")
    if isinstance(revision, dict):
        for key in ("lineage_id", "base_event_id", "base_snapshot_sha256"):
            if key in revision and not isinstance(revision.get(key), str):
                raise ProjectManifestError(f"manifest.json revision.{key} must be text.")
    generation = payload.get("generation")
    if generation is not None and not isinstance(generation, dict):
        raise ProjectManifestError("manifest.json generation must be an object.")
    if isinstance(generation, dict):
        artifacts = generation.get("artifacts")
        if artifacts is not None and (
            not isinstance(artifacts, dict)
            or any(
                not isinstance(name, str) or not isinstance(digest, str)
                for name, digest in artifacts.items()
            )
        ):
            raise ProjectManifestError("manifest.json generation artifacts must be text hashes.")
    return payload


def read_project_manifest(project_or_path: Path, *, missing_ok: bool = True) -> dict:
    path = manifest_path(project_or_path)
    if not path.is_file():
        if missing_ok:
            return {}
        raise ProjectManifestError(f"Missing {path}.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProjectManifestError(f"Could not read {path}: {exc}") from exc
    return validate_project_manifest(payload)


def load_project_manifest(project_or_path: Path) -> dict:
    """Soft-failing reader for listing and UI paths."""

    try:
        return read_project_manifest(project_or_path)
    except ProjectManifestError:
        return {}


def write_project_manifest(project_or_path: Path, payload: dict) -> None:
    path = manifest_path(project_or_path)
    stored = dict(validate_project_manifest(dict(payload)))
    stored["schema_version"] = SCHEMA_VERSION
    write_json_atomic(path, stored)


def source_identity(manifest: dict) -> dict:
    """Portable source identity with no machine-specific absolute path."""

    kind = str(manifest.get("source_kind") or "").strip().lower()
    origin = str(manifest.get("youtube_url") or manifest.get("origin") or "").strip()
    video_id = _youtube_video_id(origin)
    local_origin = str(manifest.get("origin") or "").strip()
    local_suffix = Path(portable_name(local_origin)).suffix.lower()
    local_source = bool(local_origin and "://" not in local_origin)
    if (
        kind in {"file", "midi"}
        or local_source
        or local_suffix in {".mid", ".midi"}
    ):
        source_kind = (
            "midi"
            if kind == "midi" or local_suffix in {".mid", ".midi"}
            else "file"
        )
        identity: dict = {"kind": source_kind}
        source_hash = str(manifest.get("source_sha256") or "")
        if source_hash:
            identity["sha256"] = source_hash
        if local_origin:
            identity["name"] = portable_name(local_origin)
        if video_id:
            identity["attached_video_id"] = video_id
        return identity
    if video_id:
        return {"kind": "youtube", "video_id": video_id}
    if not kind:
        suffix = Path(portable_name(origin)).suffix.lower()
        kind = "midi" if suffix in {".mid", ".midi"} else "file"
    identity: dict = {"kind": kind or "file"}
    source_hash = str(manifest.get("source_sha256") or "")
    if source_hash:
        identity["sha256"] = source_hash
    if origin:
        identity["name"] = portable_name(origin) if "://" not in origin else origin
    return identity


def public_project_state(manifest: dict) -> dict:
    """The sanitized settings needed to compare and restore a project."""

    audio = manifest.get("audio") if isinstance(manifest.get("audio"), dict) else {}
    tempo = manifest.get("tempo") if isinstance(manifest.get("tempo"), dict) else {}
    notation = manifest.get("notation") if isinstance(manifest.get("notation"), dict) else {}
    settings = notation.get("settings") if isinstance(notation.get("settings"), dict) else {}
    state: dict = {
        "source": source_identity(manifest),
        "title": str(manifest.get("title") or ""),
        "model": str(manifest.get("model") or ""),
        "import_settings_sha256": str(
            manifest.get("import_settings_sha256") or ""
        ),
        "audio": {
            key: audio.get(key)
            for key in (
                "sample_rate",
                "full_duration_seconds",
                "excerpt_start_seconds",
                "excerpt_end_seconds",
                "working_duration_seconds",
            )
            if key in audio
        },
        "tempo": _portable_value(tempo),
        "staff_split": manifest.get("staff_split"),
        "staff_assignment": str(manifest.get("staff_assignment") or ""),
        "notation": {
            "style": str(notation.get("style") or ""),
            "settings": _portable_value(settings),
            "score_from": str(notation.get("score_from") or ""),
        },
        "sync_offset_seconds": manifest.get("sync_offset_seconds", 0.0),
    }
    return state


def revision_block(manifest: dict) -> dict:
    value = manifest.get("revision")
    return dict(value) if isinstance(value, dict) else {}


def with_revision(
    manifest: dict,
    *,
    lineage_id: str,
    base_event_id: str,
    base_snapshot_sha256: str,
) -> dict:
    updated = dict(manifest)
    updated["revision"] = {
        "lineage_id": str(lineage_id),
        "base_event_id": str(base_event_id),
        "base_snapshot_sha256": str(base_snapshot_sha256),
    }
    return updated


def generation_mismatches(project: Path, manifest: dict) -> list[str]:
    """Artifacts that no longer match the last completed generation marker."""

    generation = (
        manifest.get("generation")
        if isinstance(manifest.get("generation"), dict)
        else {}
    )
    artifacts = (
        generation.get("artifacts")
        if isinstance(generation.get("artifacts"), dict)
        else {}
    )
    mismatches: list[str] = []
    for name, expected in artifacts.items():
        path = Path(project) / str(name)
        try:
            matches = path.is_file() and sha256_file(path) == str(expected)
        except OSError:
            matches = False
        if not matches:
            mismatches.append(str(name))
    return sorted(mismatches)


def _portable_value(value):
    if isinstance(value, dict):
        return {str(key): _portable_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_portable_value(item) for item in value]
    if isinstance(value, Path):
        return value.name
    return value


def _youtube_video_id(value: str) -> str | None:
    if not value:
        return None
    try:
        parsed = urlparse(value)
    except ValueError:
        return None
    host = parsed.netloc.lower().split(":")[0]
    if host in {"youtu.be", "www.youtu.be"}:
        ident = parsed.path.strip("/").split("/")[0]
    elif host in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
        if parsed.path == "/watch":
            ident = (parse_qs(parsed.query).get("v") or [""])[0]
        elif parsed.path.startswith(("/embed/", "/shorts/")):
            ident = parsed.path.strip("/").split("/")[1]
        else:
            ident = ""
    else:
        return None
    return ident if len(ident) == 11 else None
