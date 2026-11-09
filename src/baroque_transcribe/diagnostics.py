"""Plots and a rough resynthesis for checking the transcription by ear.

The resynthesis is a diagnostic click-and-tone rendering of the detected
notes. It is not a piano sample and it is not evidence that every note is
correct. Play ``comparison.wav`` against the recording, then fix ``notes.json``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
from scipy import signal

from baroque_transcribe.schema import NoteEvent


def _load_mono(path: Path) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, always_2d=False)
    array = np.asarray(audio, dtype=np.float64)
    if array.ndim > 1:
        array = array.mean(axis=1)
    return array, int(sample_rate)


def synthesize_notes(notes: list[NoteEvent], sample_rate: int, duration: float) -> np.ndarray:
    length = max(1, int(round(duration * sample_rate)))
    mix = np.zeros(length, dtype=np.float64)
    for note in notes:
        start = int(round(note.onset * sample_rate))
        end = int(round(note.offset * sample_rate))
        start = max(0, min(length, start))
        end = max(start + 1, min(length, end))
        count = end - start
        times = np.arange(count, dtype=np.float64) / sample_rate
        frequency = 440.0 * 2.0 ** ((note.pitch - 69) / 12.0)
        envelope = np.exp(-2.5 * times)
        tone = np.sin(2 * np.pi * frequency * times)
        tone += 0.35 * np.sin(2 * np.pi * frequency * 2 * times)
        tone += 0.12 * np.sin(2 * np.pi * frequency * 3 * times)
        mix[start:end] += envelope * tone * (note.velocity / 127.0)
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1e-8:
        mix *= 0.9 / peak
    return mix.astype(np.float32)


def _draw_cursor(axis: plt.Axes, position: float | None) -> None:
    if position is not None:
        axis.axvline(position, color="#c2410c", linewidth=1.0, linestyle="--", label="position")


def render_diagnostics(
    working_wav: Path,
    notes: list[NoteEvent],
    output_dir: Path,
    *,
    position: float | None = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    audio, sample_rate = _load_mono(working_wav)
    duration = len(audio) / sample_rate
    times = np.arange(len(audio)) / sample_rate

    waveform_path = output_dir / "waveform.png"
    figure, axis = plt.subplots(figsize=(12, 3.2))
    axis.plot(times, audio, color="#1f2937", linewidth=0.4)
    _draw_cursor(axis, position)
    axis.set_title("Working audio")
    axis.set_xlabel("Seconds")
    axis.set_ylabel("Amplitude")
    axis.set_xlim(0, duration)
    figure.tight_layout()
    figure.savefig(waveform_path, dpi=120)
    plt.close(figure)

    frequencies, spec_times, spectrum = signal.spectrogram(
        audio,
        fs=sample_rate,
        nperseg=1024,
        noverlap=768,
        scaling="spectrum",
    )
    spectrogram = 20 * np.log10(spectrum + 1e-10)
    spectrogram_path = output_dir / "spectrogram.png"
    figure, axis = plt.subplots(figsize=(12, 4))
    image = axis.pcolormesh(spec_times, frequencies, spectrogram, shading="auto", cmap="magma")
    axis.set_ylim(0, 4000)
    _draw_cursor(axis, position)
    axis.set_title("Spectrogram")
    axis.set_xlabel("Seconds")
    axis.set_ylabel("Hz")
    figure.colorbar(image, ax=axis, label="dB")
    figure.tight_layout()
    figure.savefig(spectrogram_path, dpi=120)
    plt.close(figure)

    piano_path = output_dir / "piano_roll.png"
    figure, axis = plt.subplots(figsize=(12, 4.5))
    if notes:
        for note in notes:
            color = plt.cm.viridis(note.velocity / 127.0)
            axis.barh(
                note.pitch,
                max(0.01, note.duration),
                left=note.onset,
                height=0.8,
                color=color,
                align="center",
            )
        pitches = [note.pitch for note in notes]
        axis.set_ylim(min(pitches) - 2, max(pitches) + 2)
    else:
        axis.text(0.5, 0.5, "No notes detected", transform=axis.transAxes, ha="center")
        axis.set_ylim(21, 108)
    _draw_cursor(axis, position)
    axis.set_xlim(0, duration)
    axis.set_title("Detected notes")
    axis.set_xlabel("Seconds")
    axis.set_ylabel("MIDI pitch")
    figure.tight_layout()
    figure.savefig(piano_path, dpi=120)
    plt.close(figure)

    overview_path = output_dir / "overview.png"
    figure, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(times, audio, color="#1f2937", linewidth=0.4)
    axes[0].set_ylabel("Waveform")
    axes[1].pcolormesh(spec_times, frequencies, spectrogram, shading="auto", cmap="magma")
    axes[1].set_ylim(0, 4000)
    axes[1].set_ylabel("Hz")
    if notes:
        for detected in notes:
            axes[2].barh(
                detected.pitch,
                max(0.01, detected.duration),
                left=detected.onset,
                height=0.8,
                color="#1d4ed8",
                align="center",
            )
    axes[2].set_ylabel("MIDI")
    axes[2].set_xlabel("Seconds")
    for axis in axes:
        _draw_cursor(axis, position)
        axis.set_xlim(0, duration)
    figure.suptitle("Transcription check")
    figure.tight_layout()
    figure.savefig(overview_path, dpi=120)
    plt.close(figure)

    resynthesis = synthesize_notes(notes, sample_rate, duration)
    resynthesis_path = output_dir / "resynthesis.wav"
    sf.write(resynthesis_path, resynthesis, sample_rate)
    length = min(len(audio), len(resynthesis))
    mixed = 0.72 * audio[:length] + 0.28 * resynthesis[:length]
    peak = float(np.max(np.abs(mixed))) if length else 0.0
    if peak > 0.98:
        mixed *= 0.98 / peak
    comparison_path = output_dir / "comparison.wav"
    sf.write(comparison_path, mixed.astype(np.float32), sample_rate)

    return {
        "waveform": str(waveform_path),
        "spectrogram": str(spectrogram_path),
        "piano_roll": str(piano_path),
        "overview": str(overview_path),
        "resynthesis": str(resynthesis_path),
        "comparison": str(comparison_path),
    }
