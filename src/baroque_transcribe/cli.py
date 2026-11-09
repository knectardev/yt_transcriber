"""Command-line pipeline for solo piano transcription."""

from __future__ import annotations

import argparse
import platform
import shutil
import sys
from pathlib import Path

from baroque_transcribe import __version__
from baroque_transcribe.defaults import (
    DEFAULT_STAFF_SPLIT,
    DEFAULT_SUBDIVISION,
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
    history.add_argument("history_command", nargs="?", choices=["import", "restore"], default=None)
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

    ui = sub.add_parser("ui", help="Open the local review window.")
    ui.add_argument("--port", type=int, default=DEFAULT_UI_PORT)
    ui.add_argument("--project", type=Path, default=None)
    return parser


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


def _history(command: str | None, root: Path | None, *, with_audio: bool = False) -> int:
    from baroque_transcribe.parse_store import default_store, format_event, import_projects, load_events, restore_projects

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


def _launch_ui(port: int, project: Path | None) -> int:
    ui_path = Path(__file__).with_name("ui.py")
    if shutil.which("streamlit") is None and not _module_exists("streamlit"):
        print(
            "error: streamlit is not installed. Run: pip install streamlit",
            file=sys.stderr,
        )
        return 1
    import os

    if project is not None:
        os.environ["BAROQUE_UI_PROJECT"] = str(project)
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(ui_path),
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
        "--client.toolbarMode",
        "minimal",
        *_UI_THEME,
    ]
    import subprocess

    print(f"Review UI: http://127.0.0.1:{port}")
    return subprocess.call(command)


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
        return _history(args.history_command, args.root, with_audio=args.with_audio)
    if args.command == "ui":
        return _launch_ui(args.port, args.project)

    from baroque_transcribe.analysis import AnalysisError
    from baroque_transcribe.audio import AudioError
    from baroque_transcribe.checkpoint import CheckpointError
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
        if args.command == "quantize":
            requantize_project(
                args.project,
                tempo=args.tempo,
                time_signature=args.time_signature,
                subdivision=args.subdivision,
                downbeat=args.downbeat,
                staff_split=args.staff_split,
                diagnostics=not args.no_diagnostics,
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
        AudioError,
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
