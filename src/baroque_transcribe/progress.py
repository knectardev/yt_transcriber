"""Stage reports for a transcription.

The pipeline and the model stay free of any UI toolkit. A callback receives
each event. With no callback, the same sentence is printed so the command
line and the review page describe the same work.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from baroque_transcribe.storage_io import write_json_atomic

# Overall bar spans. Transcription is most of a CPU run; the other stages
# are short, so they occupy a thin slice of the bar.
STAGE_SPAN: dict[str, tuple[float, float]] = {
    "prepare": (0.00, 0.04),
    "checkpoint": (0.04, 0.14),
    "load": (0.14, 0.18),
    "transcribe": (0.18, 0.92),
    "write": (0.92, 1.00),
}

# The ByteDance model hears 10 seconds at a time and hops 5 seconds.
# piano_transcription_inference.PianoTranscription.enframe uses the same figures.
SEGMENT_SECONDS = 10.0
SEGMENT_HOP_DIVISOR = 2


@dataclass(frozen=True)
class ProgressEvent:
    stage: str
    label: str
    fraction: float | None = None


ProgressCallback = Callable[[ProgressEvent], None]


def emit(callback: ProgressCallback | None, event: ProgressEvent, *, echo: bool = True) -> None:
    if echo:
        print(event.label, flush=True)
    if callback is not None:
        callback(event)


def fraction_for(stage: str, stage_fraction: float) -> float:
    start, end = STAGE_SPAN[stage]
    clipped = min(1.0, max(0.0, stage_fraction))
    return start + (end - start) * clipped


def window_count(num_samples: int, *, sample_rate: int = 16_000, segment_seconds: float = SEGMENT_SECONDS) -> int:
    """How many overlapping windows the ByteDance model will run.

    Matches ``PianoTranscription.enframe``: pad up to a whole segment, then
    hop by half a segment.
    """

    if num_samples <= 0 or sample_rate <= 0 or segment_seconds <= 0:
        return 0
    segment_samples = int(sample_rate * segment_seconds)
    if segment_samples <= 0:
        return 0
    padded = -(-num_samples // segment_samples) * segment_samples  # ceil division
    hop = segment_samples // SEGMENT_HOP_DIVISOR
    if hop <= 0:
        return 0
    count = 0
    pointer = 0
    while pointer + segment_samples <= padded:
        count += 1
        pointer += hop
    return count


def remaining_seconds(windows_done: int, windows_total: int, elapsed_seconds: float) -> float | None:
    """Seconds left, measured from the windows that have already finished."""

    if windows_done <= 0 or elapsed_seconds < 0 or windows_total <= 0:
        return None
    if windows_done >= windows_total:
        return 0.0
    return elapsed_seconds / windows_done * (windows_total - windows_done)


def spoken_duration(seconds: float) -> str:
    whole = max(0, int(round(seconds)))
    if whole < 45:
        shown = max(1, whole)
        return "1 second" if shown == 1 else f"{shown} seconds"
    if whole < 90:
        return "1 minute"
    minutes = int(round(whole / 60))
    if minutes < 60:
        return "1 minute" if minutes == 1 else f"{minutes} minutes"
    hours = minutes // 60
    rest = minutes % 60
    hour_word = "hour" if hours == 1 else "hours"
    if rest == 0:
        return f"{hours} {hour_word}"
    minute_word = "minute" if rest == 1 else "minutes"
    return f"{hours} {hour_word} {rest} {minute_word}"


def eta_label(windows_done: int, windows_total: int, elapsed_seconds: float) -> str:
    """A short remaining-time sentence for the window currently finishing."""

    if windows_done <= 0 or windows_total <= 0:
        return "Timing the first window…"
    remaining = remaining_seconds(windows_done, windows_total, elapsed_seconds)
    if remaining is None:
        return "Timing the first window…"
    if remaining < 1:
        return "Finishing the last window…"
    return f"about {spoken_duration(remaining)} left"


def window_label(windows_done: int, windows_total: int, elapsed_seconds: float) -> str:
    timing = eta_label(windows_done, windows_total, elapsed_seconds)
    if windows_done <= 0:
        return f"{timing} ({windows_total} windows)"
    return f"Window {windows_done} of {windows_total}. {timing}"


def pace_path() -> Path:
    return Path.home() / ".cache" / "baroque_transcribe" / "pace.json"


def load_seconds_per_window(device: str, *, path: Path | None = None) -> float | None:
    file = path or pace_path()
    if not file.is_file():
        return None
    try:
        payload = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return None
    entry = payload.get(device)
    if not isinstance(entry, dict):
        return None
    value = entry.get("seconds_per_window")
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds <= 0:
        return None
    return seconds


def save_pace(device: str, seconds_per_window: float, *, path: Path | None = None) -> None:
    if seconds_per_window <= 0:
        return
    file = path or pace_path()
    try:
        file.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {}
        if file.is_file():
            try:
                loaded = json.loads(file.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload = loaded
            except (OSError, json.JSONDecodeError, TypeError):
                payload = {}
        payload[device] = {
            "seconds_per_window": seconds_per_window,
            "updated": datetime.now(timezone.utc).isoformat(),
        }
        write_json_atomic(file, payload)
    except OSError:
        return


def pace_preview(device: str, windows: int, *, path: Path | None = None) -> str | None:
    seconds = load_seconds_per_window(device, path=path)
    if seconds is None or windows <= 0:
        return None
    return (
        f"about {spoken_duration(seconds * windows)} on {device}, "
        "from the last run on this PC"
    )
