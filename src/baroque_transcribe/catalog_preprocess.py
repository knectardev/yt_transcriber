"""Batch-import catalog MIDI into Library projects (resume/skip already opened)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from baroque_transcribe.catalog import (
    CatalogError,
    CatalogManifest,
    entry_midi_path,
    entry_statuses,
    find_project_for_entry,
    load_catalog,
)
from baroque_transcribe.catalog_open import CATALOG_NOTATION, open_catalog_entry
from baroque_transcribe.catalog_sync import KeepAwake
from baroque_transcribe.defaults import NOTATION_MAP_JSON, NOTES_JSON, SCORE_VOICES_XML, SCORE_XML
from baroque_transcribe.midi_import import MidiImportError
from baroque_transcribe.notation_settings import NotationSettings

ProgressFn = Callable[[int, int, "PreprocessResult"], None]


@dataclass(frozen=True)
class PreprocessResult:
    entry_id: str
    action: str
    project: str = ""
    error: str = ""


@dataclass
class PreprocessReport:
    results: list[PreprocessResult] = field(default_factory=list)

    @property
    def opened(self) -> int:
        return sum(1 for row in self.results if row.action == "opened")

    @property
    def requantized(self) -> int:
        return sum(1 for row in self.results if row.action == "requantized")

    @property
    def skipped(self) -> int:
        return sum(1 for row in self.results if row.action == "skipped")

    @property
    def failed(self) -> int:
        return sum(1 for row in self.results if row.action == "failed")

    @property
    def ok(self) -> bool:
        return self.failed == 0

    def summary(self) -> str:
        return (
            f"{self.opened} opened, {self.requantized} requantized, "
            f"{self.skipped} skipped, {self.failed} failed"
        )


def project_is_scored(project: Path | None) -> bool:
    if project is None:
        return False
    return (project / NOTES_JSON).is_file() and (project / SCORE_XML).is_file()


def project_has_readable_voices(project: Path | None) -> bool:
    """Scored project with readable color-voice artifacts."""

    if not project_is_scored(project):
        return False
    assert project is not None
    return (project / SCORE_VOICES_XML).is_file() and (project / NOTATION_MAP_JSON).is_file()


def preprocess_catalog(
    catalog: CatalogManifest | None = None,
    *,
    manifest_path: Path | None = None,
    root: Path | None = None,
    entry_ids: list[str] | None = None,
    force: bool = False,
    keep_awake: bool = False,
    settings: NotationSettings | None = None,
    on_progress: ProgressFn | None = None,
) -> PreprocessReport:
    """Import or re-engrave catalog MIDI as readable (color-voice) projects.

    Skip when ``score.voices.musicxml`` and ``notation_map.json`` already exist,
    unless ``force`` is set (then re-quantize in place with readable settings).
    """

    loaded = catalog if catalog is not None else load_catalog(path=manifest_path, root=root)
    wanted = set(entry_ids) if entry_ids else None
    targets = [
        entry
        for entry in loaded.entries
        if wanted is None or entry.id in wanted
    ]
    if wanted is not None:
        known = {entry.id for entry in loaded.entries}
        missing = sorted(wanted - known)
        if missing:
            raise CatalogError(f"Unknown catalog entr{'y' if len(missing) == 1 else 'ies'}: {', '.join(missing)}")

    resolved = settings if settings is not None else CATALOG_NOTATION
    report = PreprocessReport()
    total = len(targets)
    with KeepAwake(keep_awake):
        for index, entry in enumerate(targets, start=1):
            result = _preprocess_one(
                entry.id,
                root=root,
                manifest_path=manifest_path,
                force=force,
                settings=resolved,
            )
            report.results.append(result)
            if on_progress is not None:
                on_progress(index, total, result)
    return report


def _readable_settings(project: Path, requested: NotationSettings) -> NotationSettings:
    """Keep the project's grid/thresholds; force readable style for color voices."""

    from baroque_transcribe.notation_settings import settings_from_manifest
    from baroque_transcribe.project_manifest import read_project_manifest

    manifest = read_project_manifest(project, missing_ok=True) or {}
    base = settings_from_manifest(manifest)
    # Keep the project's grid; force readable style for color-voice artifacts.
    return NotationSettings(
        grid=base.grid,
        style="readable",
        cluster_ms=requested.cluster_ms,
        fill_gap_steps=requested.fill_gap_steps,
        keep_rest_steps=requested.keep_rest_steps,
        overlap_trim_steps=requested.overlap_trim_steps,
        chord_release_tolerance_steps=requested.chord_release_tolerance_steps,
        extra_voice_cost=requested.extra_voice_cost,
        silence_reanchor_seconds=requested.silence_reanchor_seconds,
    )


def _preprocess_one(
    entry_id: str,
    *,
    root: Path | None,
    manifest_path: Path | None,
    force: bool,
    settings: NotationSettings,
) -> PreprocessResult:
    existing = find_project_for_entry(entry_id, root=root)
    prior_scored = project_is_scored(existing)
    prior_readable = project_has_readable_voices(existing)
    if prior_readable and not force:
        return PreprocessResult(
            entry_id=entry_id,
            action="skipped",
            project=str(existing) if existing else "",
        )
    midi = entry_midi_path(entry_id, root=root)
    if not midi.is_file():
        return PreprocessResult(
            entry_id=entry_id,
            action="failed",
            error=f"MIDI not downloaded ({midi})",
        )
    try:
        project = open_catalog_entry(
            entry_id,
            root=root,
            manifest_path=manifest_path,
            sync_if_missing=False,
            settings=settings,
        )
    except (CatalogError, MidiImportError, OSError, ValueError, RuntimeError) as exc:
        return PreprocessResult(entry_id=entry_id, action="failed", error=str(exc))
    if not (project / NOTES_JSON).is_file():
        return PreprocessResult(
            entry_id=entry_id,
            action="failed",
            project=str(project),
            error="Import finished without notes.json",
        )

    if force or not project_has_readable_voices(project):
        try:
            from baroque_transcribe.pipeline import requantize_project

            requantize_project(
                project,
                settings=_readable_settings(project, settings),
                diagnostics=False,
            )
        except Exception as exc:  # noqa: BLE001 — keep batch alive across engrave failures
            return PreprocessResult(
                entry_id=entry_id,
                action="failed",
                project=str(project),
                error=f"Readable re-engrave failed: {exc}",
            )

    if not project_has_readable_voices(project):
        return PreprocessResult(
            entry_id=entry_id,
            action="failed",
            project=str(project),
            error="Missing score.voices.musicxml or notation_map.json after engrave",
        )

    action = "requantized" if prior_scored else "opened"
    return PreprocessResult(entry_id=entry_id, action=action, project=str(project))


def entries_needing_preprocess(
    catalog: CatalogManifest | None = None,
    *,
    manifest_path: Path | None = None,
    root: Path | None = None,
) -> list[str]:
    loaded = catalog if catalog is not None else load_catalog(path=manifest_path, root=root)
    needed: list[str] = []
    for row in entry_statuses(loaded, root=root):
        if not row.local:
            continue
        existing = find_project_for_entry(row.entry.id, root=root)
        if not project_has_readable_voices(existing):
            needed.append(row.entry.id)
    return needed
