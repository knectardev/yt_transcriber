"""Estimate a single tempo from the working audio.

Baroque improvisation is often flexible. The estimate is a starting point for
the notation grid, not a claim that the performance was metronomic. Half-tempo
and double-tempo candidates are reported because onset autocorrelation often
locks onto the wrong metrical level.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass
class TempoEstimate:
    bpm: float | None
    candidates: list[float]
    method: str

    def to_dict(self) -> dict:
        return {"bpm": self.bpm, "candidates": self.candidates, "method": self.method}


def estimate_tempo(samples: np.ndarray, sample_rate: int) -> TempoEstimate:
    audio = np.asarray(samples, dtype=np.float64)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if audio.size < sample_rate:
        return TempoEstimate(None, [], "audio shorter than one second")

    hop = 512
    _frequencies, _times, spectrum = signal.stft(
        audio,
        fs=sample_rate,
        nperseg=2048,
        noverlap=2048 - hop,
        boundary=None,
    )
    magnitude = np.abs(spectrum)
    if magnitude.shape[1] < 8:
        return TempoEstimate(None, [], "not enough frames")

    flux = np.maximum(np.diff(magnitude, axis=1), 0.0).sum(axis=0)
    window = np.hanning(9)
    window /= window.sum()
    flux = np.convolve(flux, window, mode="same")
    flux = flux - float(flux.mean())
    if not np.any(np.abs(flux) > 0):
        return TempoEstimate(None, [], "no onset energy")

    correlation = np.correlate(flux, flux, mode="full")
    correlation = correlation[len(correlation) // 2 :]
    frame_rate = sample_rate / hop
    min_bpm, max_bpm = 40.0, 200.0
    min_lag = max(1, int(frame_rate * 60.0 / max_bpm))
    max_lag = min(len(correlation) - 1, int(frame_rate * 60.0 / min_bpm))
    if max_lag <= min_lag:
        return TempoEstimate(None, [], "onset series too short for 40-200 BPM")

    lag = min_lag + int(np.argmax(correlation[min_lag:max_lag]))
    bpm = 60.0 * frame_rate / lag
    candidates: list[float] = []
    for factor in (0.5, 1.0, 2.0):
        candidate = bpm * factor
        if min_bpm <= candidate <= max_bpm:
            rounded = round(float(candidate), 2)
            if rounded not in candidates:
                candidates.append(rounded)
    return TempoEstimate(round(float(bpm), 2), candidates, "spectral-flux autocorrelation")
