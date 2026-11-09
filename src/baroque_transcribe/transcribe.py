"""Run a pretrained piano transcription model.

ByteDance's high-resolution piano model is the default. It was trained on
close-miked solo piano and predicts note onset, offset, velocity, and sustain
pedal. Basic Pitch is a lighter instrument-agnostic baseline. Neither model
is asked to harmonize or to invent a Baroque texture; both only decode what
they hear, and both can still be wrong.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from baroque_transcribe.checkpoint import ensure_checkpoint
from baroque_transcribe.defaults import (
    BASIC_PITCH_FRAME_THRESHOLD,
    BASIC_PITCH_MIN_NOTE_MS,
    BASIC_PITCH_ONSET_THRESHOLD,
    BYTEDANCE_FRAME_THRESHOLD,
    BYTEDANCE_OFFSET_THRESHOLD,
    BYTEDANCE_ONSET_THRESHOLD,
    BYTEDANCE_PEDAL_OFFSET_THRESHOLD,
    MODEL_SAMPLE_RATE,
)
from baroque_transcribe.progress import (
    ProgressCallback,
    ProgressEvent,
    emit,
    fraction_for,
    pace_preview,
    save_pace,
    window_count,
    window_label,
)
from baroque_transcribe.schema import NoteEvent, PedalEvent, clean_notes, clean_pedals


class TranscribeError(RuntimeError):
    pass


def resolve_device(requested: str) -> str:
    try:
        import torch
    except ImportError as exc:
        raise TranscribeError(
            "PyTorch is not installed. Install a CPU or CUDA build of torch before transcribing. "
            "See the README."
        ) from exc
    choice = requested.lower()
    if choice == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if choice == "cuda" and not torch.cuda.is_available():
        raise TranscribeError(
            "CUDA was requested, but this PyTorch build cannot see a GPU. "
            "Reinstall the CUDA wheel that matches your driver, or pass --device cpu."
        )
    if choice not in {"cpu", "cuda"}:
        raise TranscribeError("Device must be auto, cpu, or cuda.")
    return choice


def _forward_with_progress(model, segments, batch_size: int, on_window) -> dict:
    """The library's segment loop, with a callback after each window.

    ``piano_transcription_inference.inference`` binds ``forward`` in its own
    module global, so a one-call patch of that name is what ``transcribe``
    actually runs. The arithmetic matches ``pytorch_utils.forward``.
    """

    import torch
    from piano_transcription_inference.pytorch_utils import append_to_dict, move_data_to_device

    output_dict: dict = {}
    device = next(model.parameters()).device
    total = int(np.ceil(len(segments) / batch_size)) if batch_size else 0
    if on_window is not None and total:
        on_window(0, total, 0.0)
    pointer = 0
    started = time.perf_counter()
    while pointer < len(segments):
        batch_waveform = move_data_to_device(segments[pointer : pointer + batch_size], device)
        pointer += batch_size
        with torch.no_grad():
            model.eval()
            batch_output = model(batch_waveform)
        for key in batch_output.keys():
            append_to_dict(output_dict, key, batch_output[key].data.cpu().numpy())
        if on_window is not None and total:
            done = min(total, pointer // batch_size)
            on_window(done, total, time.perf_counter() - started)
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
    return output_dict


def transcribe_bytedance(
    audio: np.ndarray,
    sample_rate: int,
    *,
    device: str,
    checkpoint: Path | None = None,
    onset_threshold: float = BYTEDANCE_ONSET_THRESHOLD,
    offset_threshold: float = BYTEDANCE_OFFSET_THRESHOLD,
    frame_threshold: float = BYTEDANCE_FRAME_THRESHOLD,
    pedal_offset_threshold: float = BYTEDANCE_PEDAL_OFFSET_THRESHOLD,
    on_progress: ProgressCallback | None = None,
) -> tuple[list[NoteEvent], list[PedalEvent], str]:
    if sample_rate != MODEL_SAMPLE_RATE:
        raise TranscribeError(f"ByteDance inference expects {MODEL_SAMPLE_RATE} Hz audio, got {sample_rate}.")
    if audio.ndim != 1:
        raise TranscribeError("Pass mono audio to the piano model.")

    # The Zenodo checkpoint predates PyTorch 2.6, which refuses the old pickle
    # by default. The file is the Kong et al. release, downloaded by this project.
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        from piano_transcription_inference import PianoTranscription
    except ImportError as exc:
        raise TranscribeError(
            "piano-transcription-inference is not installed. Install the requirements.txt packages."
        ) from exc

    checkpoint_path = ensure_checkpoint(checkpoint, on_progress=on_progress)
    resolved = resolve_device(device)
    windows = window_count(len(audio), sample_rate=sample_rate)
    preview = pace_preview(resolved, windows)
    if preview:
        emit(
            on_progress,
            ProgressEvent(
                "transcribe",
                f"{windows} windows. {preview}.",
                fraction_for("transcribe", 0.0),
            ),
        )
    load_label = f"Loading the ByteDance piano model on {resolved}."
    if resolved == "cpu":
        load_label += " On CPU this is usually under a minute."
    emit(on_progress, ProgressEvent("load", load_label, fraction_for("load", 0.2)))
    engine = PianoTranscription(device=resolved, checkpoint_path=str(checkpoint_path))
    engine.onset_threshold = onset_threshold
    # The upstream attribute name is misspelled. Set both so a future release
    # that fixes the name still receives the threshold.
    engine.offset_threshold = offset_threshold
    engine.offset_threshod = offset_threshold
    engine.frame_threshold = frame_threshold
    engine.pedal_offset_threshold = pedal_offset_threshold

    emit(
        on_progress,
        ProgressEvent("load", f"Model loaded on {resolved}.", fraction_for("load", 1.0)),
    )
    waveform = np.ascontiguousarray(audio, dtype=np.float32)

    def on_window(done: int, total: int, elapsed: float) -> None:
        stage_fraction = (done / total) if total else 0.0
        emit(
            on_progress,
            ProgressEvent(
                "transcribe",
                window_label(done, total, elapsed),
                fraction_for("transcribe", stage_fraction),
            ),
        )
        if done == total and total > 0 and elapsed > 0:
            save_pace(resolved, elapsed / total)

    import piano_transcription_inference.inference as inference_module

    original_forward = inference_module.forward

    def patched_forward(model, segments, batch_size):
        return _forward_with_progress(model, segments, batch_size, on_window)

    inference_module.forward = patched_forward
    try:
        result = engine.transcribe(waveform, None)
    finally:
        inference_module.forward = original_forward
    notes = [
        NoteEvent(
            pitch=int(event["midi_note"]),
            onset=float(event["onset_time"]),
            offset=float(event["offset_time"]),
            velocity=int(event["velocity"]),
        )
        for event in result.get("est_note_events") or []
    ]
    pedals = [
        PedalEvent(onset=float(event["onset_time"]), offset=float(event["offset_time"]))
        for event in result.get("est_pedal_events") or []
    ]
    return clean_notes(notes), clean_pedals(pedals), resolved


def transcribe_basic_pitch(
    audio_path: Path,
    *,
    onset_threshold: float = BASIC_PITCH_ONSET_THRESHOLD,
    frame_threshold: float = BASIC_PITCH_FRAME_THRESHOLD,
    minimum_note_ms: float = BASIC_PITCH_MIN_NOTE_MS,
    on_progress: ProgressCallback | None = None,
) -> tuple[list[NoteEvent], list[PedalEvent]]:
    try:
        from basic_pitch import FilenameSuffix, build_icassp_2022_model_path
        from basic_pitch.inference import predict
    except ImportError as exc:
        raise TranscribeError(
            "Basic Pitch is not installed. On Python 3.12, TensorFlow is not a usable "
            "dependency of basic-pitch 0.4, so install the ONNX runtime instead:\n"
            "  pip install onnxruntime scikit-learn mir_eval resampy\n"
            "  pip install basic-pitch==0.4.0 --no-deps"
        ) from exc

    model_path = build_icassp_2022_model_path(FilenameSuffix.onnx)
    if not Path(model_path).exists():
        raise TranscribeError(
            f"The Basic Pitch ONNX file is missing at {model_path}. "
            "Reinstall basic-pitch==0.4.0 --no-deps so its saved_models directory is present."
        )
    emit(
        on_progress,
        ProgressEvent(
            "checkpoint",
            "Basic Pitch uses its ONNX file, not the ByteDance checkpoint.",
            fraction_for("checkpoint", 1.0),
        ),
    )
    emit(
        on_progress,
        ProgressEvent("load", "Loading Basic Pitch.", fraction_for("load", 0.5)),
    )
    emit(
        on_progress,
        ProgressEvent(
            "transcribe",
            "Running Basic Pitch. This step does not report windows.",
            fraction_for("transcribe", 0.1),
        ),
    )
    _model_output, _midi, note_events = predict(
        str(audio_path),
        model_or_model_path=str(model_path),
        onset_threshold=onset_threshold,
        frame_threshold=frame_threshold,
        minimum_note_length=minimum_note_ms,
        minimum_frequency=27.5,
        maximum_frequency=4200.0,
        multiple_pitch_bends=False,
        melodia_trick=False,
    )
    notes = []
    for start, end, pitch, amplitude, _bends in note_events:
        velocity = int(round(float(amplitude) * 127))
        notes.append(NoteEvent(pitch=int(pitch), onset=float(start), offset=float(end), velocity=velocity))
    emit(
        on_progress,
        ProgressEvent("transcribe", "Basic Pitch finished.", fraction_for("transcribe", 1.0)),
    )
    return clean_notes(notes), []
