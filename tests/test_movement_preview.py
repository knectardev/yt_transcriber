"""Per-movement preview clips for standalone section playback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from baroque_transcribe.defaults import NOTES_JSON
from baroque_transcribe.preview_audio import (
    ensure_movement_preview_wav,
    movement_preview_filename,
)
from baroque_transcribe.schema import NoteEvent, Transcription


def test_movement_preview_filename_sanitizes():
    assert movement_preview_filename("allemande") == "mov-allemande.wav"
    assert movement_preview_filename("courantes (i)").startswith("mov-courantes")


def test_synthesize_preview_by_voice_uses_two_timbres():
    from baroque_transcribe.preview_synth import synthesize_preview, synthesize_preview_by_voice
    from baroque_transcribe.schema import NoteEvent

    notes = [
        (NoteEvent(72, 0.0, 0.4, 90), 1),
        (NoteEvent(48, 0.0, 0.4, 90), 2),
    ]
    dual = synthesize_preview_by_voice(
        notes,
        8_000,
        0.5,
        sounds_by_voice={1: "simple", 2: "clav"},
    )
    single = synthesize_preview(
        [NoteEvent(72, 0.0, 0.4, 90), NoteEvent(48, 0.0, 0.4, 90)],
        8_000,
        0.5,
        sound="simple",
    )
    assert dual.shape == single.shape
    assert float(np.max(np.abs(dual - single))) > 0.01


def test_ensure_movement_preview_renders_local_zero(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "baroque_transcribe.preview_audio.project_audio_root",
        lambda local=None: tmp_path / "audio",
    )
    project = tmp_path / "proj"
    project.mkdir()
    Transcription(
        notes=[
            NoteEvent(pitch=60, onset=0.0, offset=0.4, velocity=80),
            NoteEvent(pitch=64, onset=1.0, offset=1.4, velocity=80),
            NoteEvent(pitch=67, onset=2.0, offset=2.4, velocity=80),
        ]
    ).write(project / NOTES_JSON)

    clip = ensure_movement_preview_wav(
        project,
        movement_id="allemande",
        start_seconds=1.0,
        end_seconds=2.5,
        sound="simple",
        local=tmp_path,
        sample_rate=8_000,
    )
    assert clip.is_file()
    data, got_rate = sf.read(clip)
    assert got_rate == 8_000
    assert len(data) == pytest.approx(8_000 * 1.5, abs=2)
    # Suite note at 1.0s is remapped to local 0 — first frames should speak.
    assert float(np.max(np.abs(data[:800]))) > 0.01
