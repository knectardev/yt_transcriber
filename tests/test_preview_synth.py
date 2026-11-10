"""Synthetic preview timbres for the fragment library."""

from __future__ import annotations

import numpy as np
import pytest

from baroque_transcribe.preview_synth import (
    DEFAULT_PREVIEW_SOUND,
    PREVIEW_SOUNDS,
    preview_sound_ids,
    resolve_preview_sound,
    synthesize_preview,
)
from baroque_transcribe.schema import NoteEvent


def test_preview_sound_catalog_covers_pianos_and_electric_pianos():
    ids = preview_sound_ids()
    assert DEFAULT_PREVIEW_SOUND in ids
    assert "rhodes" in ids and "wurli" in ids and "dx_ep" in ids
    assert {sound.kind for sound in PREVIEW_SOUNDS} == {"piano", "electric"}
    assert resolve_preview_sound("bright_piano").label == "Bright piano"
    with pytest.raises(ValueError, match="Unknown preview sound"):
        resolve_preview_sound("bagpipes")


def test_synthesize_preview_renders_each_named_sound():
    notes = [
        NoteEvent(60, 0.0, 0.4, 90),
        NoteEvent(64, 0.2, 0.6, 70),
        NoteEvent(67, 0.4, 0.9, 80),
    ]
    for sound in PREVIEW_SOUNDS:
        audio = synthesize_preview(notes, 16_000, 1.0, sound=sound.id)
        assert audio.dtype == np.float32
        assert audio.shape == (16_000,)
        assert float(np.max(np.abs(audio))) > 0.1


def test_piano_roll_svg_is_a_compact_inline_strip():
    from baroque_transcribe.preview_synth import piano_roll_svg

    notes = [NoteEvent(60, 0.0, 0.5, 80), NoteEvent(67, 0.25, 0.9, 100)]
    svg = piano_roll_svg(notes, 1.0, width=200, height=34)
    assert svg.startswith("<svg")
    assert 'height="34"' in svg
    assert svg.count("<rect") >= 3
    empty = piano_roll_svg([], 1.0)
    assert "<svg" in empty
