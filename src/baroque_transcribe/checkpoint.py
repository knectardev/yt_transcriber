"""Download the Kong et al. piano checkpoint without calling wget.

The published inference package shells out to wget, which is not a Windows
default. A Python download also lets us reject an HTML error page before
PyTorch tries to unpickle it.
"""

from __future__ import annotations

import os
import shutil
import urllib.request
from pathlib import Path

from baroque_transcribe.defaults import (
    CHECKPOINT_FILENAME,
    CHECKPOINT_MIN_BYTES,
    CHECKPOINT_URL,
)
from baroque_transcribe.progress import ProgressCallback, ProgressEvent, emit, fraction_for


class CheckpointError(RuntimeError):
    pass


def default_checkpoint_path() -> Path:
    override = os.environ.get("BAROQUE_TRANSCRIBE_CHECKPOINT")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".cache" / "baroque_transcribe" / CHECKPOINT_FILENAME


def is_checkpoint_payload(path: Path) -> bool:
    """True for a PyTorch save, false for an HTML error page.

    Checkpoints written by PyTorch 1.6 and later are zip files. The Kong et al.
    release was saved with PyTorch 1.4, so it is a protocol-2 pickle instead.
    """

    with path.open("rb") as handle:
        magic = handle.read(4)
    if magic.startswith(b"PK\x03\x04"):
        return True
    return magic[:2] == b"\x80\x02"


def ensure_checkpoint(path: Path | None = None, on_progress: ProgressCallback | None = None) -> Path:
    target = path or default_checkpoint_path()
    if target.exists() and target.stat().st_size >= CHECKPOINT_MIN_BYTES and is_checkpoint_payload(target):
        emit(
            on_progress,
            ProgressEvent(
                "checkpoint",
                "Model file is already on this PC.",
                fraction_for("checkpoint", 1.0),
            ),
        )
        return target
    if target.exists() and target.stat().st_size < CHECKPOINT_MIN_BYTES:
        target.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    request = urllib.request.Request(
        CHECKPOINT_URL,
        headers={"User-Agent": "baroque-transcribe/0.1"},
    )
    emit(
        on_progress,
        ProgressEvent(
            "checkpoint",
            f"Downloading the piano model (~165 MB) to {target}",
            fraction_for("checkpoint", 0.0),
        ),
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            written = 0
            next_report = 8 * 1024 * 1024
            while True:
                block = response.read(1024 * 1024)
                if not block:
                    break
                handle.write(block)
                written += len(block)
                stage_fraction = (written / total) if total else 0.0
                label = (
                    f"Downloading the model, {written / 1e6:.0f} / {total / 1e6:.0f} MB"
                    if total
                    else f"Downloading the model, {written / 1e6:.0f} MB"
                )
                echo = written >= next_report
                emit(
                    on_progress,
                    ProgressEvent("checkpoint", label, fraction_for("checkpoint", stage_fraction)),
                    echo=echo,
                )
                if echo:
                    next_report += 8 * 1024 * 1024
    except OSError as exc:
        if partial.exists():
            partial.unlink()
        raise CheckpointError(
            "Could not download the pretrained checkpoint from Zenodo. "
            f"Save {CHECKPOINT_FILENAME} yourself and pass --checkpoint, "
            "or set BAROQUE_TRANSCRIBE_CHECKPOINT to that file. "
            f"Original error: {exc}"
        ) from exc

    size = partial.stat().st_size
    if size < CHECKPOINT_MIN_BYTES or not is_checkpoint_payload(partial):
        partial.unlink(missing_ok=True)
        raise CheckpointError(
            f"The downloaded checkpoint is {size} bytes and does not look like a PyTorch file. "
            "Zenodo may have returned an error page. Try the download again later."
        )
    shutil.move(str(partial), target)
    emit(
        on_progress,
        ProgressEvent(
            "checkpoint",
            f"Model file ready ({target.stat().st_size / 1e6:.0f} MB).",
            fraction_for("checkpoint", 1.0),
        ),
    )
    return target
