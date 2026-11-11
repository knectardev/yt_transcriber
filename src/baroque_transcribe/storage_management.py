"""Storage reporting and explicit, validated local project exports."""

from __future__ import annotations

import json
import os
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from baroque_transcribe.defaults import FRAGMENTS_DIR, SOURCE_WAV, WORKING_WAV
from baroque_transcribe.project_manifest import load_project_manifest, revision_block
from baroque_transcribe.storage_io import sha256_file, temporary_path, tree_size


@dataclass(frozen=True)
class StorageReport:
    active_projects: int
    recovery_copies: int
    parse_events: int
    fragments: int
    rebuildable_caches: int

    def to_dict(self) -> dict:
        return asdict(self)


def storage_report(root: Path) -> StorageReport:
    root = Path(root)
    transcriptions = root / "transcriptions"
    recovery = transcriptions / ".recovery"
    events = root / "data" / "parses" / "events"
    fragments = root / "data" / FRAGMENTS_DIR
    active_size = 0
    if transcriptions.is_dir():
        for path in transcriptions.iterdir():
            if path.is_dir() and path.name != ".recovery" and not path.name.startswith(".tmp-"):
                active_size += tree_size(path)
    cache_size = tree_size(root / ".local") + tree_size(root / "data" / "thumbnails")
    return StorageReport(
        active_projects=active_size,
        recovery_copies=tree_size(recovery),
        parse_events=tree_size(events),
        fragments=tree_size(fragments),
        rebuildable_caches=cache_size,
    )


def export_project(
    project: Path,
    destination: Path,
    *,
    include_audio: bool = False,
) -> Path:
    """Create an atomic ZIP backup and validate every member before publishing it."""

    project = Path(project).expanduser().resolve()
    if not project.is_dir():
        raise ValueError(f"Project folder not found: {project}")
    destination = Path(destination).expanduser()
    if destination.suffix.lower() != ".zip":
        destination = destination.with_suffix(".zip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Export already exists: {destination}")
    files = [
        path
        for path in sorted(project.rglob("*"))
        if path.is_file() and not path.name.startswith(".tmp-")
    ]
    if not include_audio:
        files = [path for path in files if not _is_audio(path, project)]
    inventory = {
        path.relative_to(project).as_posix(): sha256_file(path)
        for path in files
    }
    metadata = {
        "version": 1,
        "project": project.name,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "audio_included": include_audio,
        "revision": revision_block(load_project_manifest(project)),
        "files": inventory,
    }
    temporary = temporary_path(destination)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in files:
                archive.write(path, path.relative_to(project).as_posix())
            archive.writestr(
                "_baroque_export.json",
                json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            )
        _validate_export(temporary, inventory)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def _validate_export(path: Path, inventory: dict[str, str]) -> None:
    import hashlib

    with zipfile.ZipFile(path, "r") as archive:
        bad = archive.testzip()
        if bad:
            raise OSError(f"Export validation failed for {bad}.")
        stored = json.loads(archive.read("_baroque_export.json"))
        if stored.get("files") != inventory:
            raise OSError("Export inventory does not match.")
        for name, expected in inventory.items():
            actual = hashlib.sha256(archive.read(name)).hexdigest()
            if actual != expected:
                raise OSError(f"Export hash mismatch for {name}.")


def _is_audio(path: Path, project: Path) -> bool:
    relative = path.relative_to(project)
    if relative.name in {SOURCE_WAV, WORKING_WAV}:
        return True
    return relative.suffix.lower() in {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
