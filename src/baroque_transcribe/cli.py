"""Command-line pipeline for solo piano transcription."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import socket
import sys
from pathlib import Path

from baroque_transcribe import __version__
from baroque_transcribe.defaults import (
    DEFAULT_GRID_MODE,
    DEFAULT_NOTATION_STYLE,
    DEFAULT_STAFF_SPLIT,
    DEFAULT_PLAYBACK_PORT,
    DEFAULT_SUBDIVISION,
    DEFAULT_UI_ADDRESS,
    DEFAULT_UI_PORT,
)

# Heavy modules (pipeline, notation, transcription, diagnostics) are imported
# inside the command handlers that need them, not here. Importing them at the
# top cost several seconds before any command ran -- including ``ui``, which
# only spawns a Streamlit subprocess that imports everything again itself.


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="baroque-transcribe",
        description=(
            "Transcribe a solo piano recording to an unquantized MIDI file and, "
            "separately, a quantized MIDI file and MusicXML score."
        ),
    )
    parser.add_argument("--version", action="version", version=f"baroque-transcribe {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare", help="Download or copy audio and write a normalized working file.")
    _add_input_arguments(prepare)

    transcribe = sub.add_parser("transcribe", help="Prepare audio, run a model, and write MIDI and MusicXML.")
    _add_input_arguments(transcribe)
    transcribe.add_argument(
        "--model",
        default="bytedance",
        choices=["bytedance", "basic-pitch"],
        help="Bytedance is the piano-specific model. basic-pitch is a lighter baseline. Default: bytedance.",
    )
    transcribe.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    transcribe.add_argument("--checkpoint", type=Path, default=None, help="Path to the ByteDance .pth checkpoint.")
    transcribe.add_argument("--tempo", default="auto", help="BPM for the notation grid, or 'auto'. Default: auto.")
    transcribe.add_argument(
        "--time-signature",
        default=None,
        help="Meter for the score, such as 4/4, 3/4, or 6/8. If omitted, 4/4 is used and recorded as an assumption.",
    )
    transcribe.add_argument(
        "--subdivision",
        type=int,
        default=DEFAULT_SUBDIVISION,
        choices=[8, 16, 32],
        help="Quantization grid: 8 eighths, 16 sixteenths, or 32 thirty-seconds. Default: 16.",
    )
    transcribe.add_argument(
        "--downbeat",
        default="0",
        help="Time of the first downbeat in the working audio. Seconds, M:SS, or H:MM:SS. Default: 0.",
    )
    transcribe.add_argument(
        "--staff-split",
        type=int,
        default=DEFAULT_STAFF_SPLIT,
        help="MIDI pitch at or above this value goes on the treble staff. Default: 60 (middle C).",
    )
    transcribe.add_argument("--onset-threshold", type=float, default=None)
    transcribe.add_argument("--offset-threshold", type=float, default=None)
    transcribe.add_argument("--frame-threshold", type=float, default=None)
    transcribe.add_argument("--minimum-note-ms", type=float, default=None, help="Basic Pitch only.")
    transcribe.add_argument("--raw-only", action="store_true", help="Stop after the unquantized MIDI file.")
    transcribe.add_argument("--no-diagnostics", action="store_true")
    transcribe.add_argument("--position", type=float, default=None, help="Draw a cursor at this time, in seconds.")
    _add_notation_arguments(transcribe)

    quantize = sub.add_parser(
        "quantize",
        help="Rebuild quantized MIDI and MusicXML from an edited notes.json.",
    )
    quantize.add_argument("project", type=Path, help="Transcription directory.")
    quantize.add_argument("--tempo", default=None, help="BPM. Omit to keep the previous tempo.")
    quantize.add_argument("--time-signature", default=None)
    quantize.add_argument("--subdivision", type=int, default=DEFAULT_SUBDIVISION, choices=[8, 16, 32])
    quantize.add_argument("--downbeat", default=None)
    quantize.add_argument("--staff-split", type=int, default=None)
    quantize.add_argument("--no-diagnostics", action="store_true")
    _add_notation_arguments(quantize)

    compare = sub.add_parser(
        "compare",
        help="Write side-by-side notation variants under project/comparison/ without changing the saved score.",
    )
    compare.add_argument("project", type=Path, help="Transcription directory.")
    compare.add_argument("--tempo", type=float, default=None, help="BPM override for the comparison only.")
    compare.add_argument("--time-signature", default=None)
    compare.add_argument("--subdivision", type=int, default=None, choices=[8, 16, 32])
    compare.add_argument("--downbeat", type=float, default=None)
    compare.add_argument("--staff-split", type=int, default=None)
    compare.add_argument(
        "--variants",
        default="literal_fixed,literal_follow,readable_follow,readable_follow_voices",
        help=(
            "Comma-separated variants: literal_fixed, literal_follow, "
            "readable_follow, readable_follow_voices."
        ),
    )

    analyze = sub.add_parser(
        "analyze",
        help="Write annotations.json with a harmonic draft for an existing project.",
    )
    analyze.add_argument("project", type=Path, help="Transcription directory.")
    analyze.add_argument("--key", default=None, help="Key tonic, such as C, F#, or Bb. Omit to estimate it.")
    analyze.add_argument("--mode", default=None, choices=["major", "minor"], help="Key mode. Default: major with --key, otherwise estimated.")
    analyze.add_argument("--force", action="store_true", help="Replace hand-edited harmonic labels as well.")

    diagnostics = sub.add_parser("diagnostics", help="Redraw plots and the comparison mix for an existing project.")
    diagnostics.add_argument("project", type=Path)
    diagnostics.add_argument("--position", type=float, default=None)

    sub.add_parser("doctor", help="Check FFmpeg, PyTorch, the checkpoint, and optional packages.")

    history = sub.add_parser(
        "history",
        help="List parse history, import existing transcription folders into it, or restore projects from it.",
    )
    history.add_argument(
        "history_command",
        nargs="?",
        choices=["import", "restore", "status", "preview", "update", "adopt"],
        default=None,
    )
    history.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=None,
        help="Folder of projects to import, or one project name to restore. Default: transcriptions, or every missing project.",
    )
    history.add_argument(
        "--with-audio",
        action="store_true",
        help="Restore only: also download the excerpt audio (section only). The model is never run.",
    )
    history.add_argument(
        "--published-annotations",
        action="store_true",
        help="Update only: replace local annotations with the selected event's causal overlay.",
    )
    history.add_argument(
        "--confirm",
        default="",
        help="Update only: repeat the exact target event id to confirm replacement.",
    )

    storage = sub.add_parser(
        "storage",
        help="Report storage, export a project backup, or manage durable recovery copies.",
    )
    storage.add_argument(
        "storage_command",
        choices=[
            "report",
            "export",
            "recoveries",
            "rollback-recovery",
            "delete-recovery",
        ],
    )
    storage.add_argument(
        "target",
        nargs="?",
        type=Path,
        help="Export: project folder/name. Recovery actions: project/recovery-id.",
    )
    storage.add_argument(
        "destination",
        nargs="?",
        type=Path,
        help="Export destination ZIP.",
    )
    storage.add_argument(
        "--with-audio",
        action="store_true",
        help="Export: include source, working, and diagnostic audio.",
    )
    storage.add_argument(
        "--confirm",
        default="",
        help="Recovery rollback/delete: exact confirmation shown by the recoveries report.",
    )

    fragments = sub.add_parser(
        "fragments",
        help="Extract a passage of a transcription as a reusable fragment, or list and index the saved fragments.",
    )
    fragment_sub = fragments.add_subparsers(dest="fragments_command", required=True)
    extract = fragment_sub.add_parser(
        "extract",
        help="Save whole measures of a project as a fragment under data/fragments. The project is not changed.",
    )
    extract.add_argument("project", type=Path, help="Transcription directory.")
    extract.add_argument(
        "--measures",
        required=True,
        help="Printed measures to keep, such as 12-15, or one measure such as 12.",
    )
    extract.add_argument("--title", default=None, help="Name of the fragment. Required unless --dry-run is given.")
    extract.add_argument("--description", default="")
    extract.add_argument("--practice-notes", default="")
    extract.add_argument("--tag", action="append", default=[], help="A tag. Repeat for more than one.")
    extract.add_argument("--composer", default="")
    extract.add_argument("--performer", default="")
    extract.add_argument("--dry-run", action="store_true", help="Describe the fragment without saving it.")
    _add_library_arguments(extract)

    listing = fragment_sub.add_parser("list", help="List the saved fragments, newest change first, with optional filters.")
    listing.add_argument("--text", default="", help="Words that must appear in the title, notes, source title, or tags.")
    listing.add_argument("--tag", action="append", default=[], help="A tag the fragment must carry. Repeat to require more.")
    listing.add_argument("--key", default="", help="Analyzed key, such as 'G minor', or a tonic alone, such as G.")
    listing.add_argument(
        "--progression",
        default="",
        help="Chord labels that must appear in a row, such as 'V7 i' or 'Dm7 G7'. Case matters: V is not v.",
    )
    _add_library_arguments(listing)

    reindex = fragment_sub.add_parser(
        "reindex",
        help="Read every fragment manifest and bring the search index up to date.",
    )
    reindex.add_argument("--rebuild", action="store_true", help="Delete the index file and build it again from the manifests.")
    _add_library_arguments(reindex)

    catalog = sub.add_parser(
        "catalog",
        help="Browse, sync, or open the Bach keyboard MIDI catalog.",
    )
    catalog_sub = catalog.add_subparsers(dest="catalog_command", required=True)
    catalog_list = catalog_sub.add_parser("list", help="List catalog entries with optional filters.")
    catalog_list.add_argument("--text", default="", help="Words that must appear in title, BWV, or collection.")
    catalog_list.add_argument("--collection", default="", help="Collection id, such as bwv-772-786-inventions.")
    catalog_list.add_argument("--bwv", default="", help="BWV number, such as 870 or 870a.")
    catalog_list.add_argument("--key", default="", help="Key filter substring, such as 'C major'.")
    catalog_list.add_argument(
        "--variants",
        default="include",
        choices=["include", "only", "exclude"],
        help="Whether to include variant/arrangement rows. Default: include.",
    )
    catalog_list.add_argument(
        "--status",
        default="all",
        help="all, missing, downloaded, imported, or 'hash mismatch'.",
    )
    catalog_list.add_argument(
        "--sort",
        default="bwv",
        choices=["bwv", "title", "collection"],
    )
    catalog_list.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Catalog JSON. Default: data/catalog/bach-keyboard.json.",
    )

    catalog_show = catalog_sub.add_parser("show", help="Show one catalog entry and its local status.")
    catalog_show.add_argument("entry_id", help="Stable catalog id, such as bwv-0870.")
    catalog_show.add_argument("--manifest", type=Path, default=None)

    catalog_sync = catalog_sub.add_parser(
        "sync",
        help="Download missing MIDI files into the local catalog/midi mirror.",
    )
    catalog_sync.add_argument("--all", action="store_true", help="Sync every missing entry.")
    catalog_sync.add_argument("--entry", action="append", default=[], help="Sync one entry id. Repeatable.")
    catalog_sync.add_argument(
        "--overnight",
        action="store_true",
        help="Use the polite 30-200s random delay between requests.",
    )
    catalog_sync.add_argument("--delay-min", type=float, default=None, help="Minimum delay seconds.")
    catalog_sync.add_argument("--delay-max", type=float, default=None, help="Maximum delay seconds.")
    catalog_sync.add_argument(
        "--keep-awake",
        action="store_true",
        help="Ask Windows not to sleep while this sync runs; cleared on exit.",
    )
    catalog_sync.add_argument("--force", action="store_true", help="Redownload even when a local file matches.")
    catalog_sync.add_argument(
        "--confirm",
        action="store_true",
        help="Required for bulk --all sync.",
    )
    catalog_sync.add_argument("--manifest", type=Path, default=None)

    catalog_open = catalog_sub.add_parser(
        "open",
        help="Download if needed and import/open the entry as a Library project.",
    )
    catalog_open.add_argument("entry_id", help="Stable catalog id.")
    catalog_open.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("transcriptions/catalog"),
        help="Project output folder stem. Default: transcriptions/catalog.",
    )
    catalog_open.add_argument("--manifest", type=Path, default=None)
    catalog_open.add_argument(
        "--allow-duplicate",
        action="store_true",
        help="Create another project even if this exact MIDI import was already published.",
    )

    catalog_preprocess = catalog_sub.add_parser(
        "preprocess",
        help="Import downloaded catalog MIDI into scored Library projects (resume/skip).",
    )
    catalog_preprocess.add_argument("--all", action="store_true", help="Preprocess every downloaded entry.")
    catalog_preprocess.add_argument("--entry", action="append", default=[], help="One entry id. Repeatable.")
    catalog_preprocess.add_argument(
        "--confirm",
        action="store_true",
        help="Required for bulk --all preprocess.",
    )
    catalog_preprocess.add_argument(
        "--keep-awake",
        action="store_true",
        help="Ask Windows not to sleep while this run lasts.",
    )
    catalog_preprocess.add_argument(
        "--force",
        action="store_true",
        help="Re-engrave with readable color voices even when score.voices already exists.",
    )
    catalog_preprocess.add_argument("--manifest", type=Path, default=None)

    catalog_movements = catalog_sub.add_parser(
        "movements-draft",
        help="Draft movement partitions for English/French suites and Partitas from tempo maps.",
    )
    catalog_movements.add_argument(
        "--write",
        action="store_true",
        help="Write data/catalog/movements-draft.json and a review markdown file.",
    )

    ui = sub.add_parser("ui", help="Open the local review window.")
    ui.add_argument("--port", type=int, default=DEFAULT_UI_PORT)
    ui.add_argument(
        "--address",
        default=DEFAULT_UI_ADDRESS,
        help="127.0.0.1 keeps the app on this PC. 0.0.0.0 also accepts browsers on the LAN.",
    )
    ui.add_argument("--project", type=Path, default=None)
    return parser


def _add_library_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--library",
        type=Path,
        default=None,
        help="Folder that holds the fragments. Default: data/fragments in this repository.",
    )
    parser.add_argument(
        "--index",
        type=Path,
        default=None,
        help="Search index file. Default: .local/library.sqlite in this repository. It is rebuilt from the fragments.",
    )


def _parse_measures(text: str) -> tuple[int, int]:
    """``12-15`` or ``12`` as a pair of printed measure numbers."""

    from baroque_transcribe.fragments import parse_measures

    return parse_measures(text)


def _fragments(args: argparse.Namespace) -> int:
    import sqlite3

    from baroque_transcribe.fragment_index import reconcile, refresh_folder, search_fragments
    from baroque_transcribe.fragments import describe_plan, extract_fragment, plan_fragment

    try:
        if args.fragments_command == "extract":
            start, end = _parse_measures(args.measures)
            if args.dry_run:
                for line in describe_plan(plan_fragment(args.project, start, end)):
                    print(line)
                print("Nothing was saved.")
                return 0
            if not args.title:
                raise ValueError("Give the fragment a name with --title.")
            folder = extract_fragment(
                args.project,
                start,
                end,
                title=args.title,
                description=args.description,
                practice_notes=args.practice_notes,
                tags=args.tag,
                composer=args.composer,
                performer=args.performer,
                root=args.library,
            )
            for line in describe_plan(plan_fragment(args.project, start, end)):
                print(line)
            print(f"Saved fragment {folder.name} in {folder}")
            try:
                _print_index_report(refresh_folder(folder, args.index))
            except (ValueError, OSError, sqlite3.Error) as exc:
                # The fragment is saved. The index is only a way to search, and
                # the next reindex will pick the fragment up.
                print(f"warning: the search index was not updated: {exc}", file=sys.stderr)
            return 0
        if args.fragments_command == "reindex":
            report = reconcile(args.library, args.index, mode="rebuild" if args.rebuild else "full")
            print(("Rebuilt the index. " if args.rebuild else "") + report.summary())
            _print_index_report(report, summary=False)
            return 0
        if args.fragments_command == "list":
            _print_index_report(reconcile(args.library, args.index), summary=False)
            tonic, _space, mode = " ".join(args.key.split()).partition(" ")
            found = search_fragments(
                args.index,
                text=args.text,
                tags=args.tag,
                key_tonic=tonic,
                key_mode=mode.lower(),
                progression=args.progression,
            )
            for item in found:
                where = f"m. {item.measure_start}-{item.measure_end}" if item.measure_start else ""
                details = [part for part in (item.key, where, f"{item.duration:.1f} s", item.source_title) if part]
                print(f"{item.id}  {item.title}  [{'; '.join(details)}]")
                if item.tags:
                    print(f"    tags: {', '.join(item.tags)}")
                if item.roman:
                    print(f"    harmony: {' '.join(item.roman)}")
            print(f"{len(found)} fragment{'' if len(found) == 1 else 's'}.")
            return 0
    except (ValueError, OSError, sqlite3.Error) as exc:
        # FragmentError, FragmentIndexError and QuantizeError are ValueErrors.
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


def _print_index_report(report, *, summary: bool = True) -> None:
    """Say what a reconcile did, and always name the folders it could not index."""

    if summary:
        print(f"Index: {report.summary()}")
    for problem in report.problems:
        note = " Its entry was removed from the index." if problem.was_indexed else ""
        print(f"warning: {problem.folder}: {problem.message}{note}", file=sys.stderr)


def _catalog(args: argparse.Namespace) -> int:
    from baroque_transcribe.catalog import (
        CatalogError,
        entry_statuses,
        get_entry,
        load_catalog,
        search_catalog,
    )
    from baroque_transcribe.catalog_open import open_catalog_entry
    from baroque_transcribe.catalog_sync import (
        CatalogSyncError,
        SyncSettings,
        setup_sync_logger,
        sync_entries,
        sync_missing,
    )
    from baroque_transcribe.defaults import (
        CATALOG_DELAY_MAX_SECONDS,
        CATALOG_DELAY_MIN_SECONDS,
    )
    from baroque_transcribe.midi_import import MidiImportError

    try:
        if args.catalog_command == "list":
            catalog = load_catalog(path=args.manifest)
            rows = search_catalog(
                catalog,
                text=args.text,
                collection_id=args.collection,
                bwv=args.bwv,
                key=args.key,
                variants=args.variants,
                status=args.status,
                sort=args.sort,
            )
            for row in rows:
                entry = row.entry
                print(
                    f"{entry.id}  BWV {entry.bwv}  {entry.title}  "
                    f"[{row.status_label}]  {entry.collection}"
                )
            print(f"{len(rows)} entr{'y' if len(rows) == 1 else 'ies'}.")
            return 0
        if args.catalog_command == "show":
            catalog = load_catalog(path=args.manifest)
            entry = get_entry(args.entry_id, path=args.manifest)
            status = next(
                row for row in entry_statuses(catalog) if row.entry.id == entry.id
            )
            print(entry.display_title)
            print(f"id: {entry.id}")
            print(f"collection: {entry.collection}")
            print(f"status: {status.status_label}")
            print(f"midi_url: {entry.midi_url}")
            if status.local_path:
                print(f"local: {status.local_path}")
            if status.imported_projects:
                print(f"projects: {', '.join(status.imported_projects)}")
            if entry.sha256:
                print(f"sha256: {entry.sha256}")
            return 0
        if args.catalog_command == "sync":
            catalog = load_catalog(path=args.manifest)
            delay_min = (
                CATALOG_DELAY_MIN_SECONDS
                if args.delay_min is None
                else float(args.delay_min)
            )
            delay_max = (
                CATALOG_DELAY_MAX_SECONDS
                if args.delay_max is None
                else float(args.delay_max)
            )
            if args.overnight and args.delay_min is None and args.delay_max is None:
                delay_min = CATALOG_DELAY_MIN_SECONDS
                delay_max = CATALOG_DELAY_MAX_SECONDS
            settings = SyncSettings(
                delay_min=delay_min,
                delay_max=delay_max,
                overnight=bool(args.overnight),
                keep_awake=bool(args.keep_awake or args.overnight),
                force=bool(args.force),
            )
            if args.entry:
                entries = [get_entry(entry_id, path=args.manifest) for entry_id in args.entry]
                print(
                    f"Syncing {len(entries)} catalog entr{'y' if len(entries) == 1 else 'ies'} "
                    f"with delays {settings.delay_min:g}-{settings.delay_max:g}s"
                    f"{' and keep-awake' if settings.keep_awake else ''}."
                )
                logger = setup_sync_logger()
                report = sync_entries(
                    entries,
                    settings=settings,
                    allowed_hosts=catalog.source.allowed_hosts,
                    logger=logger,
                    polite_gaps=bool(args.overnight) or settings.delay_max > 0,
                )
            elif args.all:
                if not args.confirm:
                    missing = [
                        row.entry.id
                        for row in entry_statuses(catalog)
                        if args.force or not row.local or row.hash_ok is False
                    ]
                    print(
                        f"Would sync {len(missing)} missing/mismatched files "
                        f"with {settings.delay_min:g}-{settings.delay_max:g}s delays"
                        f"{' and keep-awake' if settings.keep_awake else ''}."
                    )
                    print("Re-run with --confirm to start.")
                    return 2
                print(
                    f"Starting bulk catalog sync "
                    f"({settings.delay_min:g}-{settings.delay_max:g}s delays"
                    f"{', keep-awake' if settings.keep_awake else ''})."
                )
                logger = setup_sync_logger()

                def progress(index: int, total: int, result) -> None:
                    print(f"[{index}/{total}] {result.entry_id}: {result.action}"
                          + (f" ({result.error})" if result.error else ""))

                report = sync_missing(
                    catalog,
                    settings=settings,
                    logger=logger,
                    on_progress=progress,
                )
            else:
                print("Give --all or one or more --entry ids.", file=sys.stderr)
                return 2
            print(report.summary())
            return 0 if report.ok else 1
        if args.catalog_command == "open":
            project = open_catalog_entry(
                args.entry_id,
                output_dir=args.output,
                manifest_path=args.manifest,
                allow_duplicate=bool(args.allow_duplicate),
            )
            print(f"Opened catalog entry {args.entry_id} as {project}")
            return 0
        if args.catalog_command == "preprocess":
            from baroque_transcribe.catalog_preprocess import (
                entries_needing_preprocess,
                preprocess_catalog,
            )

            catalog = load_catalog(path=args.manifest)
            if args.entry:
                entry_ids = list(args.entry)
            elif args.all:
                entry_ids = None
                needed = entries_needing_preprocess(catalog)
                if not args.confirm:
                    print(
                        f"Would preprocess {len(needed)} downloaded entr"
                        f"{'y' if len(needed) == 1 else 'ies'} that lack readable "
                        "color-voice scores (score.voices + notation_map)."
                    )
                    print("Re-run with --confirm to start. Use --force to re-engrave all.")
                    return 2
            else:
                print("Give --all or one or more --entry ids.", file=sys.stderr)
                return 2
            print(
                "Starting catalog preprocess (readable / color voices)"
                + (" with force re-engrave" if args.force else "")
                + (" with keep-awake" if args.keep_awake else "")
                + "."
            )

            def progress(index: int, total: int, result) -> None:
                detail = result.project or result.error
                suffix = f" ({detail})" if detail else ""
                print(f"[{index}/{total}] {result.entry_id}: {result.action}{suffix}")

            report = preprocess_catalog(
                catalog,
                manifest_path=args.manifest,
                entry_ids=entry_ids,
                force=bool(args.force),
                keep_awake=bool(args.keep_awake),
                on_progress=progress,
            )
            print(report.summary())
            return 0 if report.ok else 1
        if args.catalog_command == "movements-draft":
            from baroque_transcribe.catalog_movements import (
                build_draft_overlay,
                movements_draft_path,
                review_markdown,
                write_draft_overlay,
            )

            payload = build_draft_overlay()
            count = len(payload.get("entries") or {})
            print(f"Drafted movements for {count} multi-movement catalog entries.")
            if args.write:
                path = write_draft_overlay(payload)
                review = path.with_name("movements-draft-REVIEW.md")
                review.write_text(review_markdown(payload), encoding="utf-8")
                print(f"Wrote {path}")
                print(f"Wrote {review}")
            else:
                print(f"Default output would be {movements_draft_path()}")
                print("Re-run with --write to save the draft and review markdown.")
            return 0
    except (CatalogError, CatalogSyncError, MidiImportError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


def _add_notation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--grid",
        choices=["fixed", "follow"],
        default=None,
        help=(
            f"Notation grid: fixed metronome or tempo-following. "
            f"Default for new runs: {DEFAULT_GRID_MODE}. Omit on quantize to keep the project setting."
        ),
    )
    parser.add_argument(
        "--notation",
        choices=["literal", "readable"],
        default=None,
        help=(
            f"Score style: literal event packing or readable interpretation. "
            f"Default for new runs: {DEFAULT_NOTATION_STYLE}. Readable voice work lands after the grid checkpoint."
        ),
    )


def _settings_from_args(args: argparse.Namespace, *, fallback_manifest: dict | None = None):
    from baroque_transcribe.notation_settings import NotationSettings, settings_from_manifest

    base = settings_from_manifest(fallback_manifest) if fallback_manifest is not None else NotationSettings()
    grid = getattr(args, "grid", None) or base.grid
    style = getattr(args, "notation", None) or base.style
    return NotationSettings(
        grid=grid,
        style=style,
        cluster_ms=base.cluster_ms,
        fill_gap_steps=base.fill_gap_steps,
        keep_rest_steps=base.keep_rest_steps,
        overlap_trim_steps=base.overlap_trim_steps,
        chord_release_tolerance_steps=base.chord_release_tolerance_steps,
        extra_voice_cost=base.extra_voice_cost,
        silence_reanchor_seconds=base.silence_reanchor_seconds,
    )


def _parse_compare_variants(text: str):
    from baroque_transcribe.notation_settings import NotationSettings

    mapping = {
        "literal_fixed": NotationSettings(grid="fixed", style="literal"),
        "literal_follow": NotationSettings(grid="follow", style="literal"),
        "readable_follow": NotationSettings(grid="follow", style="readable"),
        "readable_follow_voices": NotationSettings(grid="follow", style="readable"),
        "readable_fixed": NotationSettings(grid="fixed", style="readable"),
    }
    names = [part.strip() for part in text.split(",") if part.strip()]
    if not names:
        raise ValueError("Pass at least one comparison variant.")
    unknown = [name for name in names if name not in mapping]
    if unknown:
        raise ValueError(f"Unknown comparison variant(s): {', '.join(unknown)}.")
    return [(name, mapping[name]) for name in names]


def _add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", help="A WAV, MP3, or FLAC path, or a YouTube URL.")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output directory, for example transcriptions/baroque_improvisation.",
    )
    parser.add_argument("--start", default=None, help="Start of the excerpt. Seconds, M:SS, or H:MM:SS.")
    parser.add_argument("--end", default=None, help="End of the excerpt. Seconds, M:SS, or H:MM:SS.")
    parser.add_argument(
        "--cookies",
        type=Path,
        default=None,
        help="Netscape cookies.txt for yt-dlp, if YouTube asks you to sign in.",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help="Browser to read YouTube cookies from, for example chrome, edge, or firefox.",
    )


def _doctor() -> int:
    print(f"baroque-transcribe {__version__}")
    print(f"Python {platform.python_version()} on {platform.system()} {platform.release()}")
    ok = True
    from baroque_transcribe.audio import tool_path

    for tool in ("ffmpeg", "ffprobe"):
        found = tool_path(tool)
        print(f"{tool}: {found or 'NOT FOUND'}")
        if not found:
            ok = False
    try:
        import yt_dlp

        print(f"yt-dlp: {yt_dlp.version.__version__} (invoked as python -m yt_dlp)")
    except ImportError:
        print("yt-dlp: NOT INSTALLED")
        ok = False
    from baroque_transcribe.audio import _js_runtime_args

    runtimes = _js_runtime_args()
    print(f"youtube javascript runtime: {runtimes[1] if runtimes else 'NONE (install Node or Deno)'}")
    chrome_cookies = Path.home() / ".config" / "google-chrome" / "Default" / "Cookies"
    chrome_network = Path.home() / ".config" / "google-chrome" / "Default" / "Network" / "Cookies"
    windows_chrome = Path.home() / "AppData" / "Local" / "Google" / "Chrome" / "User Data"
    if chrome_cookies.exists() or chrome_network.exists() or windows_chrome.exists():
        print("chrome cookies: present")
    else:
        print("chrome cookies: not on this computer")
    try:
        import torch

        cuda = torch.cuda.is_available()
        print(f"torch: {torch.__version__}  cuda={cuda}")
        if cuda:
            print(f"gpu: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("torch: NOT INSTALLED")
        ok = False
    from baroque_transcribe.checkpoint import default_checkpoint_path

    checkpoint = default_checkpoint_path()
    if checkpoint.exists():
        print(f"checkpoint: {checkpoint} ({checkpoint.stat().st_size / 1e6:.0f} MB)")
    else:
        print(f"checkpoint: not downloaded yet ({checkpoint})")
    try:
        import piano_transcription_inference

        print(f"piano_transcription_inference: {piano_transcription_inference.__file__}")
    except ImportError:
        print("piano_transcription_inference: NOT INSTALLED")
        ok = False
    try:
        import music21

        print(f"music21: {music21.__version__}")
    except ImportError:
        print("music21: NOT INSTALLED")
        ok = False
    try:
        import logging

        logging.disable(logging.WARNING)
        from basic_pitch import FilenameSuffix, build_icassp_2022_model_path

        logging.disable(logging.NOTSET)
        onnx = build_icassp_2022_model_path(FilenameSuffix.onnx)
        print(f"basic-pitch onnx: {onnx} ({'present' if Path(onnx).exists() else 'missing'})")
    except ImportError:
        logging.disable(logging.NOTSET)
        print("basic-pitch: not installed (optional baseline)")
    try:
        import streamlit

        print(f"streamlit: {streamlit.__version__}")
    except ImportError:
        print("streamlit: not installed (optional local UI)")
    return 0 if ok else 1


def _history(
    command: str | None,
    root: Path | None,
    *,
    with_audio: bool = False,
    published_annotations: bool = False,
    confirm: str = "",
) -> int:
    from baroque_transcribe.parse_store import default_store, format_event, import_projects, load_events, restore_projects

    if command in {"status", "preview", "update", "adopt"}:
        from baroque_transcribe.library import disk_runs, transcriptions_root
        from baroque_transcribe.project_restore import (
            RestoreError,
            adopt_exact_match,
            preview_restore,
            update_project,
        )
        from baroque_transcribe.revision import revision_status

        store = default_store()
        projects_root = transcriptions_root()
        events = load_events(store)
        try:
            if command == "status":
                local = {run.name: run.folder for run in disk_runs(projects_root)}
                published = {
                    str(event.get("project") or "")
                    for event in events
                    if event.get("status") == "ok" and event.get("project")
                }
                names = {root.name} if root is not None else set(local) | published
                for name in sorted(names):
                    status = revision_status(name, local.get(name), events, store)
                    targets = ",".join(status.published_event_ids)
                    print(
                        f"{name}\t{status.state}\tbase={status.base_event_id or '-'}"
                        f"\ttarget={status.target_event_id or targets or '-'}"
                    )
                return 0
            if root is None:
                print(f"error: history {command} requires an event id or project name", file=sys.stderr)
                return 1
            value = root.name
            if command == "preview":
                preview = preview_restore(
                    store=store,
                    projects_root=projects_root,
                    event_id=value,
                )
                print(f"Project: {preview.project}")
                print(f"Target event: {preview.target_event_id}")
                print(f"Score event: {preview.score_event_id}")
                print(f"Analysis event: {preview.analysis_event_id or '-'}")
                print(f"Source: {preview.source_kind or 'unknown'}")
                print(f"Notes: {preview.note_count}; harmony entries: {preview.harmony_count}")
                print(
                    "Changed components: "
                    + (", ".join(preview.changed_components) or "none")
                )
                print(f"Recovery required: {'yes' if preview.recovery_required else 'no'}")
                return 0
            if command == "adopt":
                status = adopt_exact_match(
                    store=store,
                    projects_root=projects_root,
                    project=value,
                )
                print(f"{value}: {status.state}")
                return 0
            if confirm != value:
                print(
                    f"error: repeat the exact event id with --confirm {value}",
                    file=sys.stderr,
                )
                return 1
            result = update_project(
                store=store,
                projects_root=projects_root,
                event_id=value,
                use_published_annotations=published_annotations,
            )
            print(f"Updated {result.project} from {result.target_event_id}")
            print(f"Recovery retained at {result.recovery}")
            return 0
        except RestoreError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1

    if command == "restore":
        projects = [root.name] if root is not None else None
        restored, skipped = restore_projects(projects=projects, with_audio=with_audio)
        for folder in restored:
            print(f"Restored {folder}")
        if skipped:
            print(f"Skipped {len(skipped)} project(s) already on this computer or without a stored run: {', '.join(skipped)}")
        print(f"Restored {len(restored)} project(s) from {default_store() / 'events'}. No model was run.")
        return 0
    if command == "import":
        directory = root or Path("transcriptions")
        if not directory.is_dir():
            print(f"error: {directory} is not a directory", file=sys.stderr)
            return 1
        added, skipped = import_projects(directory)
        print(f"Imported {added} parse event(s). Skipped {skipped} already imported.")
        return 0
    events = load_events()
    if not events:
        print(f"No parse history yet. The store is {default_store()}.")
        return 0
    for event in events:
        print(format_event(event))
    print(f"{len(events)} event(s) in {default_store() / 'events'}")
    return 0


def _storage(args) -> int:
    from baroque_transcribe.library import find_project
    from baroque_transcribe.parse_store import repo_root
    from baroque_transcribe.project_restore import (
        RestoreError,
        list_recoveries,
        remove_recovery,
        rollback_recovery,
    )
    from baroque_transcribe.storage_management import export_project, storage_report

    root = repo_root()
    projects_root = root / "transcriptions"
    if args.storage_command == "report":
        report = storage_report(root)
        print(f"Active projects: {_format_bytes(report.active_projects)}")
        print(f"Recovery copies (durable): {_format_bytes(report.recovery_copies)}")
        print(f"Committed parse events: {_format_bytes(report.parse_events)}")
        print(f"Committed fragments: {_format_bytes(report.fragments)}")
        print(f"Local catalog MIDI: {_format_bytes(report.catalog_midi)}")
        print(f"Rebuildable caches: {_format_bytes(report.rebuildable_caches)}")
        return 0
    if args.storage_command == "recoveries":
        found = list_recoveries(projects_root)
        if not found:
            print("No recovery copies.")
            return 0
        for item in found:
            ident = f"{item.get('project')}/{item.get('id')}"
            print(
                f"{ident}\t{_format_bytes(int(item.get('actual_bytes') or 0))}"
                f"\t{item.get('created_at')}\t{item.get('reason')}"
            )
            print(f"  path: {item.get('path')}")
            print(
                f"  prior base: {item.get('prior_base_event_id') or '-'}"
                f"; target event: {item.get('target_event_id') or '-'}"
            )
            print(f"  rollback confirmation: RESTORE {ident}")
            print(f"  delete confirmation: DELETE {ident}")
        return 0
    if args.storage_command == "export":
        if args.target is None or args.destination is None:
            print("error: storage export requires PROJECT DESTINATION", file=sys.stderr)
            return 1
        project = find_project(str(args.target))
        if project is None:
            print(f"error: project not found: {args.target}", file=sys.stderr)
            return 1
        from baroque_transcribe.parse_store import default_store, load_events
        from baroque_transcribe.revision import revision_status

        status = revision_status(
            project.name,
            project,
            load_events(default_store()),
            default_store(),
        )
        if status.has_local_changes or status.state == "independent":
            print(
                f"warning: {project.name} has unpublished local work; "
                "this export is its independent backup.",
                file=sys.stderr,
            )
        try:
            exported = export_project(
                project,
                args.destination,
                include_audio=args.with_audio,
            )
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Validated project backup: {exported}")
        return 0
    if args.target is None or len(args.target.parts) < 2:
        print(
            f"error: storage {args.storage_command} requires PROJECT/RECOVERY-ID",
            file=sys.stderr,
        )
        return 1
    project, recovery_id = args.target.parts[-2:]
    if args.storage_command == "rollback-recovery":
        try:
            restored = rollback_recovery(
                projects_root,
                project=project,
                recovery_id=recovery_id,
                confirm=args.confirm,
            )
        except RestoreError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Restored verified recovery to: {restored.project}")
        if restored.recovery is not None:
            print(f"Current state retained as recovery: {restored.recovery}")
        return 0
    try:
        deleted = remove_recovery(
            projects_root,
            project=project,
            recovery_id=recovery_id,
            confirm=args.confirm,
        )
    except RestoreError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Deleted selected recovery copy: {deleted}")
    return 0


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024.0 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} GiB"


def _route_ipv4() -> str | None:
    """The IPv4 address the default route would use, when one can be found."""

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 1))
            address = sock.getsockname()[0]
    except OSError:
        return None
    if not address or address.startswith("169.254."):
        return None
    return address


def _launch_ui(port: int, project: Path | None, address: str) -> int:
    ui_path = Path(__file__).with_name("ui.py")
    if shutil.which("streamlit") is None and not _module_exists("streamlit"):
        print(
            "error: streamlit is not installed. Run: pip install streamlit",
            file=sys.stderr,
        )
        return 1
    bind = address.strip()
    if bind == "localhost":
        bind = "127.0.0.1"
    if bind not in {"127.0.0.1", "0.0.0.0"}:
        print("error: --address must be 127.0.0.1 or 0.0.0.0", file=sys.stderr)
        return 1
    # Drop leftover pytest markers from the parent shell so fragment/catalog
    # paths resolve to the real repo instead of raising FragmentError.
    os.environ.pop("PYTEST_CURRENT_TEST", None)
    if project is not None:
        os.environ["BAROQUE_UI_PROJECT"] = str(project)
    os.environ["BAROQUE_UI_ADDRESS"] = bind
    if bind == "0.0.0.0":
        os.environ["BAROQUE_PLAYBACK_PORT"] = os.environ.get("BAROQUE_PLAYBACK_PORT", str(DEFAULT_PLAYBACK_PORT))
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ui_path),
        "--server.port",
        str(port),
        "--server.address",
        bind,
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--client.toolbarMode",
        "minimal",
        *_UI_THEME,
    ]
    if bind == "0.0.0.0":
        for origin in _lan_origins(port):
            command.extend(["--server.corsAllowedOrigins", origin])
    import subprocess

    print(f"Review UI: http://127.0.0.1:{port}")
    if bind == "0.0.0.0":
        lan = _route_ipv4()
        if lan:
            print(f"LAN UI: http://{lan}:{port}")
        print(f"Playback port: {os.environ['BAROQUE_PLAYBACK_PORT']}")
        print("Stop the server with Ctrl+C. Logs stay in this window.")
    return subprocess.call(command)


def _lan_origins(port: int) -> list[str]:
    """Hostnames Streamlit should accept in addition to the machine's LAN address.

    The LAN address itself is allowed by Streamlit's own origin check. These
    names cover the computer hostname and ``BAROQUE_ALLOWED_HOSTS``.
    """

    names = {socket.gethostname().strip(), socket.getfqdn().strip()}
    extra = os.environ.get("BAROQUE_ALLOWED_HOSTS", "")
    names.update(part.strip() for part in extra.split(",") if part.strip())
    origins = []
    for name in sorted(names):
        if name and "://" not in name and "/" not in name:
            origins.append(f"http://{name}:{port}")
    return origins


# Warm paper palette shared with the score and playback pages.
_UI_THEME = (
    "--theme.base", "light",
    "--theme.primaryColor", "#b45309",
    "--theme.backgroundColor", "#faf7f2",
    "--theme.secondaryBackgroundColor", "#f1ebe2",
    "--theme.textColor", "#1c1917",
)


def _module_exists(name: str) -> bool:
    try:
        __import__(name)
    except ImportError:
        return False
    return True


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    # Light commands first: they must not pay for the transcription stack.
    if args.command == "doctor":
        return _doctor()
    if args.command == "history":
        return _history(
            args.history_command,
            args.root,
            with_audio=args.with_audio,
            published_annotations=args.published_annotations,
            confirm=args.confirm,
        )
    if args.command == "storage":
        return _storage(args)
    if args.command == "ui":
        return _launch_ui(args.port, args.project, args.address)
    if args.command == "fragments":
        return _fragments(args)
    if args.command == "catalog":
        return _catalog(args)

    from baroque_transcribe.analysis import AnalysisError
    from baroque_transcribe.audio import AudioError
    from baroque_transcribe.checkpoint import CheckpointError
    from baroque_transcribe.compare import CompareError, compare_project
    from baroque_transcribe.notation import NotationError
    from baroque_transcribe.pipeline import analyze_project, prepare_only, requantize_project, resolve_output_dir, run_transcription
    from baroque_transcribe.quantize import QuantizeError
    from baroque_transcribe.transcribe import TranscribeError

    try:
        if args.command == "prepare":
            prepare_only(
                args.input,
                args.output,
                args.start,
                args.end,
                cookies=args.cookies,
                cookies_from_browser=args.cookies_from_browser,
            )
            return 0
        if args.command == "transcribe":
            run_transcription(
                args.input,
                resolve_output_dir(args.input, args.output, None),
                start=args.start,
                end=args.end,
                model=args.model,
                device=args.device,
                checkpoint=args.checkpoint,
                tempo=args.tempo,
                time_signature=args.time_signature,
                subdivision=args.subdivision,
                downbeat=args.downbeat,
                staff_split=args.staff_split,
                onset_threshold=args.onset_threshold,
                offset_threshold=args.offset_threshold,
                frame_threshold=args.frame_threshold,
                minimum_note_ms=args.minimum_note_ms,
                diagnostics=not args.no_diagnostics,
                raw_only=args.raw_only,
                position=args.position,
                cookies=args.cookies,
                cookies_from_browser=args.cookies_from_browser,
                settings=_settings_from_args(args),
            )
            return 0
        if args.command == "analyze":
            analysis = analyze_project(args.project, key=args.key, mode=args.mode, force=args.force)
            print(
                f"Wrote harmonic analysis for {args.project}: "
                f"{analysis.key_tonic} {analysis.key_mode} ({analysis.key_source}), "
                f"{len(analysis.chords)} chord label(s), {len(analysis.sequences)} sequence(s)."
            )
            return 0
        if args.command == "compare":
            report = compare_project(
                args.project,
                variants=_parse_compare_variants(args.variants),
                tempo=args.tempo,
                time_signature=args.time_signature,
                subdivision=args.subdivision,
                downbeat=args.downbeat,
                staff_split=args.staff_split,
            )
            print(f"Wrote {len(report.variants)} comparison variant(s) to {report.output_dir}")
            for variant in report.variants:
                metrics = variant.metrics
                print(
                    f"  {variant.name}: on-beat {metrics.attacks_on_beat}/{metrics.attacks_total} "
                    f"({metrics.attacks_on_beat_ratio:.0%}), "
                    f"residual {metrics.mean_onset_residual_steps:.3f} steps, "
                    f"rests {metrics.rest_elements}, ties {metrics.tie_starts}, "
                    f"voices T/B {metrics.voices_treble}/{metrics.voices_bass}"
                )
            print(f"Report: {report.output_dir / 'report.json'}")
            return 0
        if args.command == "quantize":
            import json

            from baroque_transcribe.defaults import MANIFEST

            manifest_path = args.project / MANIFEST
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
            requantize_project(
                args.project,
                tempo=args.tempo,
                time_signature=args.time_signature,
                subdivision=args.subdivision,
                downbeat=args.downbeat,
                staff_split=args.staff_split,
                diagnostics=not args.no_diagnostics,
                settings=_settings_from_args(args, fallback_manifest=manifest),
            )
            return 0
        if args.command == "diagnostics":
            from baroque_transcribe.defaults import DIAGNOSTICS_DIR, NOTES_JSON, WORKING_WAV
            from baroque_transcribe.diagnostics import render_diagnostics
            from baroque_transcribe.schema import read_transcription

            transcription = read_transcription(args.project / NOTES_JSON)
            working = args.project / WORKING_WAV
            if not working.exists():
                raise FileNotFoundError(f"Missing {working}")
            render_diagnostics(working, transcription.notes, args.project / DIAGNOSTICS_DIR, position=args.position)
            print(f"Wrote diagnostics in {args.project / DIAGNOSTICS_DIR}")
            return 0
    except (
        AnalysisError,
        AudioError,
        CompareError,
        TranscribeError,
        NotationError,
        QuantizeError,
        CheckpointError,
        ValueError,
        FileNotFoundError,
    ) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    parser.error(f"Unknown command {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
