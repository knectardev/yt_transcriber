"""Small crash-safe filesystem helpers used by durable storage code."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from pathlib import Path


def portable_name(value: str | Path) -> str:
    """Final path component for either POSIX or Windows separators."""

    text = str(value).replace("\\", "/").rstrip("/")
    return text.rsplit("/", 1)[-1] if text else ""


def temporary_path(path: Path) -> Path:
    """A unique sibling path that keeps the destination suffix."""

    path = Path(path)
    return path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp{path.suffix}")


def write_text_atomic(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Replace ``path`` only after the complete text is on disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    try:
        temporary.write_text(text, encoding=encoding)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_bytes_atomic(path: Path, data: bytes) -> None:
    """Replace ``path`` only after the complete byte string is on disk."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(path)
    try:
        temporary.write_bytes(data)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: Path, payload: object, *, sort_keys: bool = False) -> None:
    """Write an indented JSON document through a sibling temporary file."""

    write_text_atomic(
        path,
        json.dumps(payload, indent=2, sort_keys=sort_keys, ensure_ascii=False) + "\n",
    )


def copy_file_atomic(source: Path, destination: Path) -> None:
    """Copy a file and atomically publish the completed destination."""

    source = Path(source)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = temporary_path(destination)
    try:
        shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of one file without loading it all into memory."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: object) -> str:
    """SHA-256 of a canonical JSON representation."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tree_size(path: Path) -> int:
    """Total bytes in regular files below ``path`` (or in one file)."""

    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    total = 0
    for candidate in path.rglob("*"):
        try:
            if candidate.is_file():
                total += candidate.stat().st_size
        except OSError:
            continue
    return total
