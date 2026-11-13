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
from baroque_transcribe.catalog_open import open_catalog_entry
from baroque_transcribe.catalog_sync import KeepAwake
from baroque_transcribe.defaults import NOTES_JSON, SCORE_XML
from baroque_transcribe.midi_import import MidiImportError

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
            f"{self.opened} opened, {self.skipped} skipped, {self.failed} failed"
        )


def project_is_scored(project: Path | None) -> bool:
    if project is None:
        return False
    return (project / NOTES_JSON).is_file() and (project / SCORE_XML).is_file()


def preprocess_catalog(
    catalog: CatalogManifest | None = None,
    *,
    manifest_path: Path | None = None,
    root: Path | None = None,
    entry_ids: list[str] | None = None,
    force: bool = False,
    keep_awake: bool = False,
    on_progress: ProgressFn | None = None,
) -> PreprocessReport:
    """Import every selected catalog MIDI that is not already a scored project."""

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

    report = PreprocessReport()
    total = len(targets)
    with KeepAwake(keep_awake):
        for index, entry in enumerate(targets, start=1):
            result = _preprocess_one(entry.id, root=root, manifest_path=manifest_path, force=force)
            report.results.append(result)
            if on_progress is not None:
                on_progress(index, total, result)
    return report


def _preprocess_one(
    entry_id: str,
    *,
    root: Path | None,
    manifest_path: Path | None,
    force: bool,
) -> PreprocessResult:
    existing = find_project_for_entry(entry_id, root=root)
    if project_is_scored(existing) and not force:
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
        )
    except (CatalogError, MidiImportError, OSError, ValueError) as exc:
        return PreprocessResult(entry_id=entry_id, action="failed", error=str(exc))
    if not project_is_scored(project):
        return PreprocessResult(
            entry_id=entry_id,
            action="failed",
            project=str(project),
            error="Import finished without notes.json/score.musicxml",
        )
    return PreprocessResult(entry_id=entry_id, action="opened", project=str(project))


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
        if not project_is_scored(existing):
            needed.append(row.entry.id)
    return needed
