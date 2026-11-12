"""Join acquisition, transcription, MIDI, notation, and diagnostics."""

from __future__ import annotations

import platform
import shutil
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from baroque_transcribe import __version__
from baroque_transcribe.audio import (
    AudioError,
    PreparedAudio,
    load_working_audio,
    parse_timestamp,
    prepare_audio,
    slugify,
)
from baroque_transcribe.defaults import (
    BASIC_PITCH_FRAME_THRESHOLD,
    BASIC_PITCH_MIN_NOTE_MS,
    BASIC_PITCH_ONSET_THRESHOLD,
    BYTEDANCE_FRAME_THRESHOLD,
    BYTEDANCE_OFFSET_THRESHOLD,
    BYTEDANCE_ONSET_THRESHOLD,
    BYTEDANCE_PEDAL_OFFSET_THRESHOLD,
    DEFAULT_STAFF_SPLIT,
    DEFAULT_SUBDIVISION,
    DEFAULT_TIME_SIGNATURE,
    INCOMPLETE_MARKER,
    ANNOTATIONS_JSON,
    DIAGNOSTICS_DIR,
    MANIFEST,
    NOTES_JSON,
    NOTATION_MAP_JSON,
    PEDAL_JSON,
    PRESERVATION_JSON,
    QUANTIZED_MID,
    RAW_MID,
    SCORE_VOICES_XML,
    SCORE_XML,
    SOURCE_WAV,
    WORKING_WAV,
)
from baroque_transcribe.analysis import (
    AnalysisError,
    HarmonicAnalysis,
    analyze_harmony,
    load_analysis,
    preserve_edited,
    write_analysis,
)
from baroque_transcribe.diagnostics import render_diagnostics
from baroque_transcribe.midi_export import write_quantized_midi, write_raw_midi
from baroque_transcribe.notation import write_musicxml
from baroque_transcribe.parse_store import record_parse
from baroque_transcribe.notation_settings import NotationSettings, settings_from_manifest
from baroque_transcribe.quantize import Quantization, parse_time_signature, quantization_from_manifest, quantize_notes
from baroque_transcribe.progress import ProgressCallback, ProgressEvent, emit, fraction_for
from baroque_transcribe.project_manifest import read_project_manifest, write_project_manifest
from baroque_transcribe.schema import Transcription, read_transcription, write_pedal_file
from baroque_transcribe.storage_io import sha256_file
from baroque_transcribe.tempo import TempoEstimate, estimate_tempo
from baroque_transcribe.transcribe import transcribe_basic_pitch, transcribe_bytedance


@dataclass
class ProjectResult:
    output_dir: Path
    note_count: int
    pedal_count: int
    tempo_bpm: float | None
    time_signature: str
    warnings: list[str]


def _write_manifest(path: Path, payload: dict) -> None:
    write_project_manifest(path, payload)


def _read_manifest(path: Path) -> dict:
    return read_project_manifest(path)


def resolve_output_dir(source: str, output: Path | None, name: str | None) -> Path:
    if output is not None:
        return output
    if name:
        slug = slugify(name)
    else:
        slug = slugify(Path(source).stem if not source.startswith("http") else "transcription")
    return Path("transcriptions") / slug


# A later transcription must not replace these. Requantize still updates them
# in place, because that rebuilds the same project from its notes.json.
_SAVED_WORK = (
    NOTES_JSON,
    PEDAL_JSON,
    RAW_MID,
    QUANTIZED_MID,
    SCORE_XML,
    SOURCE_WAV,
    WORKING_WAV,
    MANIFEST,
    DIAGNOSTICS_DIR,
)


def holds_saved_work(path: Path) -> bool:
    return any((path / name).exists() for name in _SAVED_WORK)


def output_in_progress(path: Path) -> bool:
    """True while a transcription job owns this folder."""

    return (Path(path) / INCOMPLETE_MARKER).is_file()


def _output_taken(path: Path) -> bool:
    return holds_saved_work(path) or output_in_progress(path)


def unique_output_dir(path: Path, *, now: datetime | None = None) -> Path:
    """Return ``path`` when it is empty, otherwise a new sibling folder.

    The sibling is ``name-YYYYMMDD-HHMMSS``. An existing transcription, including
    ``quantized.mid``, stays where it is. A folder marked in progress is left
    alone as well, so a second run cannot write into it.
    """

    if not _output_taken(path):
        return path
    moment = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    candidate = path.parent / f"{path.name}-{moment}"
    suffix = 2
    while candidate.exists():
        candidate = path.parent / f"{path.name}-{moment}-{suffix}"
        suffix += 1
    return candidate


def unique_path(path: Path, *, now: datetime | None = None) -> Path:
    """Return ``path`` when nothing is there, otherwise a timestamped filename."""

    if not path.exists():
        return path
    moment = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    candidate = path.with_name(f"{path.stem}-{moment}{path.suffix}")
    suffix = 2
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}-{moment}-{suffix}{path.suffix}")
        suffix += 1
    return candidate


_OUTPUT_CLAIM_LOCK = threading.Lock()


def _claim_output_dir(output_dir: Path) -> Path:
    with _OUTPUT_CLAIM_LOCK:
        requested = output_dir
        claimed = unique_output_dir(output_dir)
    if claimed != requested:
        print(f"Keeping the transcription already in {requested}")
        print(f"Writing this run to {claimed}")
    return claimed


def _fresh_sibling(path: Path, *, now: datetime | None = None) -> Path:
    moment = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    candidate = path.parent / f"{path.name}-{moment}"
    suffix = 2
    while candidate.exists():
        candidate = path.parent / f"{path.name}-{moment}-{suffix}"
        suffix += 1
    return candidate


def claim_fresh_output_dir(path: Path, job_id: str) -> Path:
    """Create an empty folder for one job and mark it in progress.

    The marker keeps the library and any later claim from using the folder
    until :func:`release_incomplete` or :func:`discard_incomplete`.
    """

    owner = job_id.strip()
    if not owner or "/" in owner or "\\" in owner or ".." in owner:
        raise ValueError("Job id cannot be a path.")
    path = Path(path)
    with _OUTPUT_CLAIM_LOCK:
        dest = unique_output_dir(path)
        while True:
            try:
                dest.mkdir(parents=True, exist_ok=False)
                break
            except FileExistsError:
                dest = _fresh_sibling(dest)
        (dest / INCOMPLETE_MARKER).write_text(owner + "\n", encoding="utf-8")
        return dest


def release_incomplete(path: Path, job_id: str) -> None:
    """Drop the in-progress marker after a job has finished writing."""

    marker = Path(path) / INCOMPLETE_MARKER
    try:
        owner = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return
    if owner == job_id.strip():
        marker.unlink(missing_ok=True)


def discard_incomplete(path: Path, job_id: str) -> bool:
    """Delete a folder this job created, and only that folder.

    A finished project has no marker, so a late or repeated call does not
    remove it. A folder marked by a different job is left alone.
    """

    folder = Path(path).resolve()
    marker = folder / INCOMPLETE_MARKER
    try:
        owner = marker.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    if owner != job_id.strip() or not folder.is_dir() or folder.parent == folder:
        return False
    shutil.rmtree(folder)
    return True


def notate_transcription(
    transcription: Transcription,
    output_dir: Path,
    *,
    bpm: float,
    time_signature: tuple[int, int],
    subdivision: int,
    downbeat: float,
    staff_split: int,
    title: str,
    tempo_info: dict,
    time_signature_assumed: bool,
    settings: NotationSettings | None = None,
    write_score: bool = True,
) -> Quantization:
    resolved = settings or NotationSettings()
    quantization = quantize_notes(
        transcription.notes,
        bpm=bpm,
        numerator=time_signature[0],
        denominator=time_signature[1],
        subdivision=subdivision,
        downbeat=downbeat,
        settings=resolved,
    )
    staff_by_source = {
        index: int(note.staff)
        for index, note in enumerate(transcription.notes)
        if note.staff in (1, 2)
    } or None
    write_raw_midi(output_dir / RAW_MID, transcription.notes, transcription.pedals)
    write_quantized_midi(
        output_dir / QUANTIZED_MID,
        quantization,
        staff_split,
        staff_by_source=staff_by_source,
    )
    map_path = output_dir / NOTATION_MAP_JSON if resolved.style == "readable" else None
    voices_path = output_dir / SCORE_VOICES_XML if resolved.style == "readable" else None
    preservation_path = output_dir / PRESERVATION_JSON if resolved.style == "readable" else None
    if write_score:
        write_musicxml(
            output_dir / SCORE_XML,
            quantization,
            staff_split=staff_split,
            title=title,
            style=resolved.style,
            notation_map=map_path,
            pedals=transcription.pedals,
            voices_path=voices_path,
            preservation_report=preservation_path,
            staff_by_source=staff_by_source,
        )
    if resolved.style != "readable":
        for leftover in (output_dir / SCORE_VOICES_XML, output_dir / NOTATION_MAP_JSON, output_dir / PRESERVATION_JSON):
            if leftover.is_file():
                leftover.unlink()
    manifest = _read_manifest(output_dir / MANIFEST)
    tempo_block = {
        "bpm": bpm,
        "downbeat_seconds": downbeat,
        "subdivision": subdivision,
        "time_signature": f"{time_signature[0]}/{time_signature[1]}",
        "time_signature_assumed": time_signature_assumed,
        "grid": resolved.grid,
        **tempo_info,
    }
    previous_notation = (
        manifest.get("notation") if isinstance(manifest.get("notation"), dict) else {}
    )
    notation = {
        "quantized_mid": QUANTIZED_MID,
        "score": SCORE_XML,
        "raw_mid": RAW_MID,
        "style": resolved.style,
        "settings": resolved.to_dict(),
        "score_from": str(
            previous_notation.get("score_from")
            or ("notes_staff" if staff_by_source else "quantized_grid")
        ),
    }
    manifest.update(
        {
            "tempo": tempo_block,
            "staff_split": staff_split,
            "notation": notation,
        }
    )
    if quantization.warnings:
        warnings = list(manifest.get("warnings") or [])
        warnings.extend(quantization.warnings)
        manifest["warnings"] = warnings
    _write_manifest(output_dir / MANIFEST, manifest)
    if write_score:
        finalize_project_generation(output_dir)
    return quantization


def finalize_project_generation(output_dir: Path) -> None:
    """Publish integrity hashes only after every generated score artifact validates."""

    names = (NOTES_JSON, PEDAL_JSON, RAW_MID, QUANTIZED_MID, SCORE_XML)
    missing = [name for name in names if not (output_dir / name).is_file()]
    if missing:
        raise OSError(f"Cannot complete score generation; missing {', '.join(missing)}.")
    manifest = _read_manifest(output_dir / MANIFEST)
    manifest["generation"] = {
        "version": 1,
        "artifacts": {name: sha256_file(output_dir / name) for name in names},
    }
    _write_manifest(output_dir / MANIFEST, manifest)


def _analysis_summary(analysis: HarmonicAnalysis) -> dict:
    return {
        "file": ANNOTATIONS_JSON,
        "key": f"{analysis.key_tonic} {analysis.key_mode}",
        "key_source": analysis.key_source,
        "key_confidence": round(float(analysis.key_confidence), 4),
        "chords": len(analysis.chords),
        "sequences": len(analysis.sequences),
        "stale": False,
    }


def _try_generate_analysis(output_dir: Path, quantization) -> str | None:
    """Generate the harmonic draft; failure never discards a transcription.

    The caller records the history event after this returns, and that event
    copies ``annotations.json`` with the score. This call does not record
    a second event.
    """

    try:
        analyze_project(output_dir, quantization=quantization, record_history=False)
    except Exception as exc:
        return f"Harmonic analysis was not generated: {exc}"
    return None


def analyze_project(
    output_dir: Path,
    *,
    key: str | None = None,
    mode: str | None = None,
    force: bool = False,
    quantization=None,
    record_history: bool = True,
) -> HarmonicAnalysis:
    """Write ``annotations.json`` for an existing project from its notes and grid."""

    notes_path = output_dir / NOTES_JSON
    if not notes_path.is_file():
        raise FileNotFoundError(f"No {NOTES_JSON} in {output_dir}. Transcribe the audio first.")
    transcription = read_transcription(notes_path)
    manifest = _read_manifest(output_dir / MANIFEST)
    tempo_info = manifest.get("tempo") if isinstance(manifest.get("tempo"), dict) else {}
    if quantization is None:
        if not tempo_info.get("bpm") or not tempo_info.get("time_signature"):
            raise AnalysisError(
                "Harmonic analysis needs a tempo and time signature. Rebuild the score with a tempo first."
            )
        try:
            quantization = quantization_from_manifest(
                transcription.notes,
                tempo_info,
                settings=settings_from_manifest(manifest),
                manifest=manifest,
            )
        except Exception as exc:
            raise AnalysisError(f"Could not rebuild the notation grid: {exc}") from exc
    old = None if force else load_analysis(output_dir / ANNOTATIONS_JSON)
    chosen_key, chosen_mode = key, mode
    if chosen_key is None and old is not None and old.key_source == "manual":
        chosen_key, chosen_mode = old.key_tonic, old.key_mode
    analysis = analyze_harmony(quantization, key_tonic=chosen_key, key_mode=chosen_mode)
    if not force:
        analysis = preserve_edited(analysis, old)
    write_analysis(output_dir / ANNOTATIONS_JSON, analysis)
    manifest = _read_manifest(output_dir / MANIFEST)
    manifest["analysis"] = _analysis_summary(analysis)
    _write_manifest(output_dir / MANIFEST, manifest)
    if record_history:
        from baroque_transcribe.parse_store import publish_analysis

        publish_analysis(output_dir)
    return analysis


def run_transcription(
    source: str,
    output_dir: Path,
    *,
    start: str | float | None = None,
    end: str | float | None = None,
    model: str = "bytedance",
    device: str = "auto",
    checkpoint: Path | None = None,
    tempo: str | float | None = "auto",
    time_signature: str | None = None,
    subdivision: int = DEFAULT_SUBDIVISION,
    downbeat: str | float | None = 0.0,
    staff_split: int = DEFAULT_STAFF_SPLIT,
    onset_threshold: float | None = None,
    offset_threshold: float | None = None,
    frame_threshold: float | None = None,
    minimum_note_ms: float | None = None,
    diagnostics: bool = True,
    raw_only: bool = False,
    position: float | None = None,
    cookies: Path | None = None,
    cookies_from_browser: str | None = None,
    on_progress: ProgressCallback | None = None,
    history_store: Path | None = None,
    settings: NotationSettings | None = None,
    already_claimed: bool = False,
) -> ProjectResult:
    if not already_claimed:
        output_dir = _claim_output_dir(output_dir)
    start_seconds = None
    end_seconds = None
    try:
        start_seconds = parse_timestamp(start)
        end_seconds = parse_timestamp(end)
        downbeat_seconds = parse_timestamp(downbeat)
        if downbeat_seconds is None:
            downbeat_seconds = 0.0
        prepared = prepare_audio(
            source,
            output_dir,
            start=start_seconds,
            end=end_seconds,
            cookies=cookies,
            cookies_from_browser=cookies_from_browser,
            on_progress=on_progress,
        )
    except AudioError as exc:
        record_parse(
            action="transcribe",
            status="error",
            source=source,
            project=output_dir.name,
            error=str(exc),
            excerpt_start=start_seconds,
            excerpt_end=end_seconds,
            store=history_store,
        )
        raise
    audio, sample_rate = load_working_audio(prepared.working_wav)
    warnings: list[str] = []
    if prepared.full_duration - prepared.duration > 0.05:
        warnings.append(
            "MIDI times start at the beginning of working.wav, not the original file. "
            f"The excerpt begins at {prepared.start:.3f}s."
        )
    if prepared.duration > 20 * 60:
        warnings.append(
            "This excerpt is longer than 20 minutes. CPU transcription will take a while, "
            "and a long YouTube file often contains speech as well as piano."
        )

    model_name = model.lower().replace("_", "-")
    if model_name in {"bytedance", "piano", "kong"}:
        notes, pedals, resolved_device = transcribe_bytedance(
            audio,
            sample_rate,
            device=device,
            checkpoint=checkpoint,
            onset_threshold=BYTEDANCE_ONSET_THRESHOLD if onset_threshold is None else onset_threshold,
            offset_threshold=BYTEDANCE_OFFSET_THRESHOLD if offset_threshold is None else offset_threshold,
            frame_threshold=BYTEDANCE_FRAME_THRESHOLD if frame_threshold is None else frame_threshold,
            on_progress=on_progress,
        )
        model_id = "bytedance-note-pedal"
        thresholds = {
            "onset": BYTEDANCE_ONSET_THRESHOLD if onset_threshold is None else onset_threshold,
            "offset": BYTEDANCE_OFFSET_THRESHOLD if offset_threshold is None else offset_threshold,
            "frame": BYTEDANCE_FRAME_THRESHOLD if frame_threshold is None else frame_threshold,
            "pedal_offset": BYTEDANCE_PEDAL_OFFSET_THRESHOLD,
        }
    elif model_name in {"basic-pitch", "basicpitch", "basic_pitch"}:
        notes, pedals = transcribe_basic_pitch(
            prepared.working_wav,
            onset_threshold=BASIC_PITCH_ONSET_THRESHOLD if onset_threshold is None else onset_threshold,
            frame_threshold=BASIC_PITCH_FRAME_THRESHOLD if frame_threshold is None else frame_threshold,
            minimum_note_ms=BASIC_PITCH_MIN_NOTE_MS if minimum_note_ms is None else minimum_note_ms,
            on_progress=on_progress,
        )
        resolved_device = "onnxruntime"
        model_id = "basic-pitch"
        thresholds = {
            "onset": BASIC_PITCH_ONSET_THRESHOLD if onset_threshold is None else onset_threshold,
            "frame": BASIC_PITCH_FRAME_THRESHOLD if frame_threshold is None else frame_threshold,
            "minimum_note_ms": BASIC_PITCH_MIN_NOTE_MS if minimum_note_ms is None else minimum_note_ms,
            "melodia_trick": False,
        }
        warnings.append(
            "Basic Pitch does not detect the sustain pedal and is less reliable "
            "than the ByteDance piano model on dense counterpoint."
        )
    else:
        raise ValueError("Model must be 'bytedance' or 'basic-pitch'.")

    if not notes:
        warnings.append(
            "The model returned no notes. The excerpt may be speech, silence, "
            "or a timbre the model does not recognize. No notes were added to fill the gap."
        )

    emit(
        on_progress,
        ProgressEvent(
            "write",
            "Writing notes and raw MIDI." if raw_only else "Writing notes, MIDI, and the score.",
            fraction_for("write", 0.15),
        ),
    )
    transcription = Transcription(notes=notes, pedals=pedals, model=model_id, source=prepared.origin)
    transcription.write(output_dir / NOTES_JSON)
    write_pedal_file(output_dir / PEDAL_JSON, pedals, model_id)
    write_raw_midi(output_dir / RAW_MID, notes, pedals)

    tempo_estimate = estimate_tempo(audio, sample_rate)
    chosen_bpm, tempo_assumed = _choose_tempo(tempo, tempo_estimate, warnings)
    if time_signature:
        signature = parse_time_signature(time_signature)
        signature_assumed = False
    else:
        signature = DEFAULT_TIME_SIGNATURE
        signature_assumed = True
        warnings.append(
            "No time signature was given. The score uses 4/4 as a placeholder. "
            "Re-run quantize with --time-signature once you know the meter."
        )

    manifest = {
        "program": f"baroque-transcribe {__version__}",
        "python": platform.python_version(),
        "system": platform.platform(),
        "origin": prepared.origin,
        "source_sha256": sha256_file(prepared.source_wav),
        "title": prepared.title,
        "model": model_id,
        "device": resolved_device,
        "thresholds": thresholds,
        "audio": {
            "source": "source.wav",
            "working": WORKING_WAV,
            "sample_rate": prepared.sample_rate,
            "full_duration_seconds": round(prepared.full_duration, 4),
            "excerpt_start_seconds": round(prepared.start, 4),
            "excerpt_end_seconds": round(prepared.end, 4),
            "working_duration_seconds": round(prepared.duration, 4),
            "peak_before_normalize": prepared.original_peak,
        },
        "counts": {"notes": len(notes), "pedals": len(pedals)},
        "tempo_estimate": tempo_estimate.to_dict(),
        "warnings": warnings,
        "accuracy": (
            "This is a model decoding of the recording, not a verified note-for-note "
            "transcription. Ornamentation, repeated notes, and notes hidden by the pedal "
            "are the usual failure points. Listen to diagnostics/comparison.wav and edit notes.json."
        ),
    }
    _write_manifest(output_dir / MANIFEST, manifest)

    if not raw_only:
        if chosen_bpm is None:
            warnings.append("Tempo could not be estimated. Notation was skipped. Pass --tempo.")
            manifest["warnings"] = warnings
            _write_manifest(output_dir / MANIFEST, manifest)
        else:
            quantization = notate_transcription(
                transcription,
                output_dir,
                bpm=chosen_bpm,
                time_signature=signature,
                subdivision=subdivision,
                downbeat=downbeat_seconds,
                staff_split=staff_split,
                title=prepared.title,
                tempo_info={
                    "source": "manual" if tempo not in (None, "auto") else "estimated",
                    "estimate": tempo_estimate.to_dict(),
                    "assumed_bpm": tempo_assumed,
                },
                time_signature_assumed=signature_assumed,
                settings=settings or NotationSettings(),
            )
            analysis_error = _try_generate_analysis(output_dir, quantization)
            if analysis_error:
                warnings.append(analysis_error)
            manifest = _read_manifest(output_dir / MANIFEST)
            manifest["warnings"] = warnings
            _write_manifest(output_dir / MANIFEST, manifest)

    if diagnostics:
        emit(
            on_progress,
            ProgressEvent(
                "write",
                "Writing diagnostics. This is usually under a minute.",
                fraction_for("write", 0.55),
            ),
        )
        render_diagnostics(
            prepared.working_wav,
            notes,
            output_dir / DIAGNOSTICS_DIR,
            position=position,
        )
    emit(
        on_progress,
        ProgressEvent(
            "write",
            "Score and diagnostics are ready." if diagnostics else "Notes and MIDI are ready.",
            fraction_for("write", 1.0),
        ),
    )

    print(f"Notes: {len(notes)}")
    print(f"Pedal events: {len(pedals)}")
    print(f"Output: {output_dir}")
    print(f"Review in the browser with: baroque-transcribe ui --project {output_dir}")
    for warning in warnings:
        print(f"warning: {warning}")
    record_parse(
        action="transcribe",
        status="ok",
        source=prepared.origin,
        title=prepared.title,
        project=output_dir.name,
        project_dir=output_dir,
        excerpt_start=prepared.start,
        excerpt_end=prepared.end,
        duration=prepared.duration,
        full_duration=prepared.full_duration,
        sample_rate=prepared.sample_rate,
        store=history_store,
    )
    return ProjectResult(
        output_dir=output_dir,
        note_count=len(notes),
        pedal_count=len(pedals),
        tempo_bpm=chosen_bpm,
        time_signature=f"{signature[0]}/{signature[1]}",
        warnings=warnings,
    )


def requantize_project(
    output_dir: Path,
    *,
    tempo: str | float | None = None,
    time_signature: str | None = None,
    subdivision: int = DEFAULT_SUBDIVISION,
    downbeat: str | float | None = None,
    staff_split: int | None = None,
    diagnostics: bool = True,
    history_store: Path | None = None,
    settings: NotationSettings | None = None,
) -> ProjectResult:
    notes_path = output_dir / NOTES_JSON
    if not notes_path.exists():
        raise FileNotFoundError(f"No {NOTES_JSON} in {output_dir}. Transcribe the audio first.")
    transcription = read_transcription(notes_path)
    manifest = _read_manifest(output_dir / MANIFEST)
    previous = manifest.get("tempo") or {}
    warnings = list(manifest.get("warnings") or [])

    if tempo in (None, "auto"):
        if previous.get("bpm"):
            bpm = float(previous["bpm"])
            assumed = bool(previous.get("assumed_bpm"))
        else:
            working = output_dir / WORKING_WAV
            if not working.exists():
                raise FileNotFoundError("Pass --tempo. There is no previous tempo and no working.wav to estimate from.")
            audio, sample_rate = load_working_audio(working)
            estimate = estimate_tempo(audio, sample_rate)
            bpm, assumed = _choose_tempo("auto", estimate, warnings)
            if bpm is None:
                raise ValueError("Tempo could not be estimated. Pass --tempo with a BPM.")
    else:
        bpm = float(tempo)
        assumed = False

    if time_signature:
        signature = parse_time_signature(time_signature)
        assumed_meter = False
    elif previous.get("time_signature"):
        signature = parse_time_signature(str(previous["time_signature"]))
        assumed_meter = bool(previous.get("time_signature_assumed"))
    else:
        signature = DEFAULT_TIME_SIGNATURE
        assumed_meter = True

    if downbeat is None:
        downbeat_seconds = float(previous.get("downbeat_seconds") or 0.0)
    else:
        parsed = parse_timestamp(downbeat)
        downbeat_seconds = 0.0 if parsed is None else parsed
    split = (
        int(manifest.get("staff_split") or DEFAULT_STAFF_SPLIT)
        if staff_split is None
        else staff_split
    )
    title = str(manifest.get("title") or output_dir.name)

    write_raw_midi(output_dir / RAW_MID, transcription.notes, transcription.pedals)
    write_pedal_file(output_dir / PEDAL_JSON, transcription.pedals, transcription.model)
    resolved = settings or settings_from_manifest(manifest)
    preserved_tempo = {
        key: previous[key]
        for key in ("tempo_map", "time_signature_map", "constant_tempo")
        if key in previous
    }
    quantization = notate_transcription(
        transcription,
        output_dir,
        bpm=bpm,
        time_signature=signature,
        subdivision=subdivision,
        downbeat=downbeat_seconds,
        staff_split=split,
        title=title,
        tempo_info={
            **preserved_tempo,
            "source": (
                str(previous.get("source") or "previous-or-estimated")
                if tempo in (None, "auto")
                else "manual"
            ),
            "assumed_bpm": assumed,
        },
        time_signature_assumed=assumed_meter,
        settings=resolved,
    )
    analysis_error = _try_generate_analysis(output_dir, quantization)
    if analysis_error:
        warnings.append(analysis_error)
        manifest = _read_manifest(output_dir / MANIFEST)
        manifest["warnings"] = warnings
        _write_manifest(output_dir / MANIFEST, manifest)
    print(f"Rewrote MIDI and MusicXML for {len(transcription.notes)} notes in {output_dir}")
    audio_info = manifest.get("audio") if isinstance(manifest.get("audio"), dict) else {}
    record_parse(
        action="quantize",
        status="ok",
        source=str(manifest.get("origin") or ""),
        title=title,
        project=output_dir.name,
        project_dir=output_dir,
        youtube_url=str(manifest.get("youtube_url") or "") or None,
        excerpt_start=audio_info.get("excerpt_start_seconds"),
        excerpt_end=audio_info.get("excerpt_end_seconds"),
        duration=audio_info.get("working_duration_seconds"),
        full_duration=audio_info.get("full_duration_seconds"),
        sample_rate=audio_info.get("sample_rate"),
        store=history_store,
    )
    if diagnostics and (output_dir / WORKING_WAV).exists():
        render_diagnostics(output_dir / WORKING_WAV, transcription.notes, output_dir / DIAGNOSTICS_DIR)
    return ProjectResult(
        output_dir=output_dir,
        note_count=len(transcription.notes),
        pedal_count=len(transcription.pedals),
        tempo_bpm=bpm,
        time_signature=f"{signature[0]}/{signature[1]}",
        warnings=warnings,
    )


def _choose_tempo(
    requested: str | float | None,
    estimate: TempoEstimate,
    warnings: list[str],
) -> tuple[float | None, bool]:
    if requested not in (None, "auto"):
        bpm = float(requested)
        if bpm <= 0:
            raise ValueError("Tempo must be a positive BPM, or 'auto'.")
        return bpm, False
    if estimate.bpm is None:
        warnings.append("Tempo estimation failed.")
        return None, True
    warnings.append(
        f"Tempo estimated at {estimate.bpm:.2f} BPM ({estimate.method}). "
        f"Also consider {estimate.candidates}. Rubato will not survive a single tempo."
    )
    return float(estimate.bpm), True


def prepare_only(
    source: str,
    output_dir: Path,
    start,
    end,
    *,
    cookies: Path | None = None,
    cookies_from_browser: str | None = None,
    history_store: Path | None = None,
) -> PreparedAudio:
    output_dir = _claim_output_dir(output_dir)
    start_seconds = None
    end_seconds = None
    try:
        start_seconds = parse_timestamp(start)
        end_seconds = parse_timestamp(end)
        prepared = prepare_audio(
            source,
            output_dir,
            start=start_seconds,
            end=end_seconds,
            cookies=cookies,
            cookies_from_browser=cookies_from_browser,
        )
    except AudioError as exc:
        record_parse(
            action="prepare",
            status="error",
            source=source,
            project=output_dir.name,
            error=str(exc),
            excerpt_start=start_seconds,
            excerpt_end=end_seconds,
            store=history_store,
        )
        raise
    manifest = {
        "program": f"baroque-transcribe {__version__}",
        "origin": prepared.origin,
        "source_sha256": sha256_file(prepared.source_wav),
        "title": prepared.title,
        "audio": {
            "source": "source.wav",
            "working": WORKING_WAV,
            "sample_rate": prepared.sample_rate,
            "full_duration_seconds": round(prepared.full_duration, 4),
            "excerpt_start_seconds": round(prepared.start, 4),
            "excerpt_end_seconds": round(prepared.end, 4),
            "working_duration_seconds": round(prepared.duration, 4),
            "peak_before_normalize": prepared.original_peak,
        },
        "warnings": [
            "Audio has been prepared. No transcription has been run yet."
        ],
    }
    _write_manifest(output_dir / MANIFEST, manifest)
    print(f"Prepared {prepared.duration:.2f}s of audio in {output_dir}")
    record_parse(
        action="prepare",
        status="ok",
        source=prepared.origin,
        title=prepared.title,
        project=output_dir.name,
        project_dir=output_dir,
        excerpt_start=prepared.start,
        excerpt_end=prepared.end,
        duration=prepared.duration,
        full_duration=prepared.full_duration,
        sample_rate=prepared.sample_rate,
        store=history_store,
    )
    return prepared
