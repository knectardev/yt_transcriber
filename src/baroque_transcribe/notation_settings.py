"""Configurable thresholds for the readable notation pipeline.

These never change ``notes.json`` or ``raw.mid``. They only affect how the
quantized grid and the written score interpret those events.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields

from baroque_transcribe.defaults import (
    DEFAULT_CHORD_RELEASE_TOLERANCE_STEPS,
    DEFAULT_CLUSTER_MS,
    DEFAULT_EXTRA_VOICE_COST,
    DEFAULT_FILL_GAP_STEPS,
    DEFAULT_GRID_MODE,
    DEFAULT_KEEP_REST_STEPS,
    DEFAULT_NOTATION_STYLE,
    DEFAULT_OVERLAP_TRIM_STEPS,
    DEFAULT_SILENCE_REANCHOR_SECONDS,
)

GRID_MODES = ("fixed", "follow")
NOTATION_STYLES = ("literal", "readable")


class NotationSettingsError(ValueError):
    pass


@dataclass(frozen=True)
class NotationSettings:
    grid: str = DEFAULT_GRID_MODE
    style: str = DEFAULT_NOTATION_STYLE
    cluster_ms: float = DEFAULT_CLUSTER_MS
    fill_gap_steps: int = DEFAULT_FILL_GAP_STEPS
    keep_rest_steps: int = DEFAULT_KEEP_REST_STEPS
    overlap_trim_steps: int = DEFAULT_OVERLAP_TRIM_STEPS
    chord_release_tolerance_steps: int = DEFAULT_CHORD_RELEASE_TOLERANCE_STEPS
    extra_voice_cost: float = DEFAULT_EXTRA_VOICE_COST
    silence_reanchor_seconds: float = DEFAULT_SILENCE_REANCHOR_SECONDS

    def __post_init__(self) -> None:
        if self.grid not in GRID_MODES:
            raise NotationSettingsError(f"Grid mode must be one of {GRID_MODES}, not {self.grid!r}.")
        if self.style not in NOTATION_STYLES:
            raise NotationSettingsError(f"Notation style must be one of {NOTATION_STYLES}, not {self.style!r}.")
        if self.cluster_ms < 0:
            raise NotationSettingsError("cluster_ms must be non-negative.")
        for name in (
            "fill_gap_steps",
            "keep_rest_steps",
            "overlap_trim_steps",
            "chord_release_tolerance_steps",
        ):
            if getattr(self, name) < 0:
                raise NotationSettingsError(f"{name} must be non-negative.")
        if self.extra_voice_cost < 0:
            raise NotationSettingsError("extra_voice_cost must be non-negative.")
        if self.silence_reanchor_seconds <= 0:
            raise NotationSettingsError("silence_reanchor_seconds must be positive.")

    def to_dict(self) -> dict:
        return asdict(self)

    def cluster_window_seconds(self, quantum: float) -> float:
        """Onset clustering window: the smaller of ``cluster_ms`` and one third of a step."""

        capped = max(0.0, quantum / 3.0) if quantum > 0 else 0.0
        return min(self.cluster_ms / 1000.0, capped) if capped > 0 else self.cluster_ms / 1000.0


def settings_from_dict(data: dict | None) -> NotationSettings:
    """Build settings from a manifest fragment, ignoring unknown keys."""

    if not data:
        return NotationSettings()
    allowed = {item.name for item in fields(NotationSettings)}
    payload = {key: value for key, value in data.items() if key in allowed}
    return NotationSettings(**payload)


def settings_from_manifest(manifest: dict | None) -> NotationSettings:
    """Read grid/style/thresholds from a project manifest.

    Missing keys mean fixed + literal, so restored projects behave as before.
    """

    if not isinstance(manifest, dict):
        return NotationSettings()
    tempo = manifest.get("tempo") if isinstance(manifest.get("tempo"), dict) else {}
    notation = manifest.get("notation") if isinstance(manifest.get("notation"), dict) else {}
    stored = notation.get("settings") if isinstance(notation.get("settings"), dict) else {}
    payload = dict(stored)
    if "grid" in tempo:
        payload["grid"] = tempo["grid"]
    elif "grid" in notation:
        payload["grid"] = notation["grid"]
    if "style" in notation:
        payload["style"] = notation["style"]
    return settings_from_dict(payload)
