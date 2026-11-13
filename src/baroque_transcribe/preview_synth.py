"""Synthetic piano and electric-piano timbres for fragment library previews.

These are additive / lightly FM tones for quick listening in the UI. They are
not sampled instruments and are not a substitute for the source recording.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baroque_transcribe.schema import NoteEvent


@dataclass(frozen=True)
class PreviewSound:
    id: str
    label: str
    kind: str  # "piano" or "electric"
    # Relative amplitudes of partials 1..n (fundamental first).
    partials: tuple[float, ...]
    # How quickly each partial decays, relative to the note length curve.
    partial_decay: tuple[float, ...]
    attack: float
    release: float
    brightness: float = 1.0
    inharmonicity: float = 0.0
    fm_ratio: float = 0.0
    fm_index: float = 0.0
    soft_clip: float = 0.0


PREVIEW_SOUNDS: tuple[PreviewSound, ...] = (
    PreviewSound(
        "simple",
        "Simple tone",
        "piano",
        partials=(1.0, 0.35, 0.12),
        partial_decay=(2.5, 3.5, 5.0),
        attack=0.005,
        release=0.04,
    ),
    PreviewSound(
        "soft_piano",
        "Soft piano",
        "piano",
        partials=(1.0, 0.45, 0.22, 0.1, 0.05),
        partial_decay=(2.2, 3.0, 4.2, 5.5, 7.0),
        attack=0.008,
        release=0.08,
        brightness=0.85,
        inharmonicity=0.0004,
    ),
    PreviewSound(
        "concert_grand",
        "Concert grand",
        "piano",
        partials=(1.0, 0.55, 0.32, 0.18, 0.1, 0.06, 0.03),
        partial_decay=(2.0, 2.8, 3.8, 5.0, 6.5, 8.0, 10.0),
        attack=0.003,
        release=0.1,
        brightness=1.15,
        inharmonicity=0.0006,
    ),
    PreviewSound(
        "bright_piano",
        "Bright piano",
        "piano",
        partials=(1.0, 0.7, 0.45, 0.28, 0.16, 0.1, 0.06, 0.04),
        partial_decay=(1.8, 2.4, 3.2, 4.2, 5.5, 7.0, 8.5, 10.0),
        attack=0.002,
        release=0.07,
        brightness=1.35,
        inharmonicity=0.0008,
    ),
    PreviewSound(
        "upright",
        "Upright piano",
        "piano",
        partials=(1.0, 0.5, 0.28, 0.16, 0.09, 0.05),
        partial_decay=(2.6, 3.4, 4.6, 6.0, 7.5, 9.0),
        attack=0.006,
        release=0.06,
        brightness=0.95,
        inharmonicity=0.0012,
        soft_clip=0.15,
    ),
    PreviewSound(
        "honky_tonk",
        "Honky-tonk piano",
        "piano",
        partials=(1.0, 0.6, 0.35, 0.2, 0.12, 0.07),
        partial_decay=(2.1, 2.9, 3.9, 5.2, 6.8, 8.5),
        attack=0.003,
        release=0.05,
        brightness=1.2,
        inharmonicity=0.0025,
        soft_clip=0.25,
    ),
    PreviewSound(
        "rhodes",
        "Rhodes electric piano",
        "electric",
        partials=(1.0, 0.15, 0.35, 0.08, 0.18),
        partial_decay=(1.6, 2.2, 3.0, 4.0, 5.0),
        attack=0.004,
        release=0.12,
        brightness=1.05,
        fm_ratio=14.0,
        fm_index=1.4,
    ),
    PreviewSound(
        "rhodes_soft",
        "Soft Rhodes",
        "electric",
        partials=(1.0, 0.12, 0.28, 0.06, 0.12),
        partial_decay=(1.8, 2.4, 3.2, 4.2, 5.2),
        attack=0.01,
        release=0.16,
        brightness=0.9,
        fm_ratio=12.0,
        fm_index=0.9,
    ),
    PreviewSound(
        "wurli",
        "Wurlitzer",
        "electric",
        partials=(1.0, 0.4, 0.55, 0.18, 0.22, 0.1),
        partial_decay=(2.0, 2.8, 3.4, 4.6, 5.5, 7.0),
        attack=0.002,
        release=0.07,
        brightness=1.25,
        soft_clip=0.35,
        fm_ratio=3.0,
        fm_index=0.6,
    ),
    PreviewSound(
        "dx_ep",
        "DX electric piano",
        "electric",
        partials=(1.0, 0.08, 0.45, 0.05, 0.2),
        partial_decay=(1.4, 2.0, 2.8, 3.8, 5.0),
        attack=0.003,
        release=0.14,
        brightness=1.1,
        fm_ratio=1.0,
        fm_index=2.2,
    ),
    PreviewSound(
        "bright_ep",
        "Bright electric piano",
        "electric",
        partials=(1.0, 0.25, 0.5, 0.15, 0.28, 0.1),
        partial_decay=(1.5, 2.1, 2.9, 3.8, 4.8, 6.0),
        attack=0.002,
        release=0.1,
        brightness=1.3,
        fm_ratio=8.0,
        fm_index=1.8,
    ),
    PreviewSound(
        "clav",
        "Clavinet",
        "electric",
        partials=(1.0, 0.7, 0.55, 0.4, 0.25, 0.15, 0.1),
        partial_decay=(4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0),
        attack=0.001,
        release=0.03,
        brightness=1.4,
        soft_clip=0.45,
    ),
)

_BY_ID = {sound.id: sound for sound in PREVIEW_SOUNDS}
DEFAULT_PREVIEW_SOUND = "concert_grand"


def preview_sound_ids() -> list[str]:
    return [sound.id for sound in PREVIEW_SOUNDS]


def preview_sound_label(sound_id: str) -> str:
    sound = _BY_ID.get(str(sound_id or "").strip())
    return sound.label if sound is not None else PREVIEW_SOUNDS[0].label


def resolve_preview_sound(sound_id: str | None) -> PreviewSound:
    chosen = str(sound_id or DEFAULT_PREVIEW_SOUND).strip()
    if chosen in _BY_ID:
        return _BY_ID[chosen]
    if chosen in {"", "default", "piano"}:
        return _BY_ID[DEFAULT_PREVIEW_SOUND]
    raise ValueError(f"Unknown preview sound {sound_id!r}.")


def synthesize_preview(
    notes: list[NoteEvent],
    sample_rate: int,
    duration: float,
    *,
    sound: str | PreviewSound = DEFAULT_PREVIEW_SOUND,
) -> np.ndarray:
    """Render ``notes`` with one of the named preview timbres."""

    preset = sound if isinstance(sound, PreviewSound) else resolve_preview_sound(sound)
    length = max(1, int(round(duration * sample_rate)))
    mix = np.zeros(length, dtype=np.float64)
    for note in notes:
        start = int(round(note.onset * sample_rate))
        end = int(round(note.offset * sample_rate))
        start = max(0, min(length, start))
        end = max(start + 1, min(length, end))
        count = end - start
        if count <= 0:
            continue
        mix[start:end] += _render_note(note, count, sample_rate, preset)
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1e-8:
        mix *= 0.9 / peak
    return mix.astype(np.float32)


def synthesize_preview_by_voice(
    notes: list[tuple[NoteEvent, int]],
    sample_rate: int,
    duration: float,
    *,
    sounds_by_voice: dict[int, str | PreviewSound],
    default_sound: str | PreviewSound = DEFAULT_PREVIEW_SOUND,
    volumes_by_voice: dict[int, float] | None = None,
) -> np.ndarray:
    """Render notes with a timbre chosen by interpreted global voice id.

    ``notes`` is ``(NoteEvent, voice_id)``. Voice 1/2 map to the two instrument
    controls; any higher voice id falls back to voice 1's sound when present,
    else ``default_sound``. Chord tones that share one voice id share one timbre.

    ``volumes_by_voice`` is linear gain in ``0..1`` (default 0.5 per voice). Each
    voice is peak-normalized on its own stem, then scaled by its volume, then
    summed — so the two sliders stay independent.
    """

    resolved: dict[int, PreviewSound] = {}
    for voice_id, sound in sounds_by_voice.items():
        resolved[int(voice_id)] = (
            sound if isinstance(sound, PreviewSound) else resolve_preview_sound(sound)
        )
    fallback = (
        default_sound
        if isinstance(default_sound, PreviewSound)
        else resolve_preview_sound(default_sound)
    )
    primary = resolved.get(1, fallback)
    volumes = {1: 0.5, 2: 0.5}
    if volumes_by_voice:
        for voice_id, gain in volumes_by_voice.items():
            try:
                volumes[int(voice_id)] = max(0.0, min(1.0, float(gain)))
            except (TypeError, ValueError):
                continue
    length = max(1, int(round(duration * sample_rate)))
    stems: dict[int, np.ndarray] = {}
    for note, voice_id in notes:
        vid = int(voice_id)
        if vid <= 0:
            preset = primary
            stem_id = 1
        elif vid >= 3:
            preset = primary
            stem_id = 1
        elif vid in resolved:
            preset = resolved[vid]
            stem_id = vid
        else:
            preset = resolved.get(2, primary)
            stem_id = 2 if 2 in resolved else 1
        start = int(round(note.onset * sample_rate))
        end = int(round(note.offset * sample_rate))
        start = max(0, min(length, start))
        end = max(start + 1, min(length, end))
        count = end - start
        if count <= 0:
            continue
        stem = stems.get(stem_id)
        if stem is None:
            stem = np.zeros(length, dtype=np.float64)
            stems[stem_id] = stem
        stem[start:end] += _render_note(note, count, sample_rate, preset)
    mix = np.zeros(length, dtype=np.float64)
    for stem_id, stem in stems.items():
        peak = float(np.max(np.abs(stem))) if stem.size else 0.0
        if peak > 1e-8:
            stem = stem * (0.9 / peak)
        gain = volumes.get(stem_id, volumes.get(1, 0.5))
        mix += stem * gain
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1.0:
        mix *= 0.98 / peak
    return mix.astype(np.float32)


def _render_note(note: NoteEvent, count: int, sample_rate: int, preset: PreviewSound) -> np.ndarray:
    times = np.arange(count, dtype=np.float64) / sample_rate
    frequency = 440.0 * 2.0 ** ((note.pitch - 69) / 12.0)
    velocity = max(1, min(127, int(note.velocity))) / 127.0
    tone = np.zeros(count, dtype=np.float64)
    phase = 2.0 * np.pi * frequency * times
    if preset.fm_index > 0.0 and preset.fm_ratio > 0.0:
        modulator = np.sin(2.0 * np.pi * frequency * preset.fm_ratio * times)
        carrier = np.sin(phase + preset.fm_index * velocity * modulator)
        tone += carrier
    for index, (amp, decay) in enumerate(zip(preset.partials, preset.partial_decay), start=1):
        stretch = 1.0 + preset.inharmonicity * index * index
        partial_hz = frequency * index * stretch
        if partial_hz >= 0.48 * sample_rate:
            continue
        strength = amp * (preset.brightness ** (index - 1))
        envelope = np.exp(-decay * (0.55 + 0.45 * (1.0 - velocity)) * times)
        tone += strength * envelope * np.sin(2.0 * np.pi * partial_hz * times)
    amp_env = _amplitude_envelope(times, note.duration, preset.attack, preset.release)
    voice = tone * amp_env * (0.35 + 0.65 * velocity)
    if preset.soft_clip > 0.0:
        drive = 1.0 + preset.soft_clip * (0.5 + 0.5 * velocity)
        voice = np.tanh(voice * drive) / np.tanh(drive)
    return voice


def _amplitude_envelope(times: np.ndarray, duration: float, attack: float, release: float) -> np.ndarray:
    attack = max(1e-4, float(attack))
    release = max(1e-4, float(release))
    held = max(duration, attack + release)
    env = np.ones_like(times)
    attack_mask = times < attack
    env[attack_mask] = times[attack_mask] / attack
    release_start = max(attack, held - release)
    release_mask = times >= release_start
    env[release_mask] = np.clip(1.0 - (times[release_mask] - release_start) / release, 0.0, 1.0)
    return env


def piano_roll_svg(
    notes: list[NoteEvent],
    duration: float,
    *,
    width: int = 340,
    height: int = 34,
) -> str:
    """A dense inline piano-roll strip for a fragment library row."""

    from html import escape

    width = max(80, int(width))
    height = max(18, int(height))
    span = max(0.05, float(duration))
    if not notes:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" aria-hidden="true">'
            f'<rect width="{width}" height="{height}" rx="4" fill="#f5f1ea"/>'
            f"</svg>"
        )
    pitches = [note.pitch for note in notes]
    low = min(pitches)
    high = max(pitches)
    if high <= low:
        high = low + 1
    pad = 2.0
    usable_h = max(1.0, height - 2 * pad)
    usable_w = max(1.0, width - 2 * pad)
    note_h = max(2.0, min(5.0, usable_h / max(4, high - low + 1)))
    bars: list[str] = [
        f'<rect width="{width}" height="{height}" rx="4" fill="#f5f1ea"/>',
    ]
    for note in notes:
        x = pad + (max(0.0, note.onset) / span) * usable_w
        w = max(2.0, ((max(note.offset, note.onset + 0.02) - max(0.0, note.onset)) / span) * usable_w)
        y = pad + (1.0 - (note.pitch - low) / (high - low)) * (usable_h - note_h)
        opacity = 0.35 + 0.65 * (max(1, min(127, note.velocity)) / 127.0)
        bars.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{note_h:.2f}" '
            f'rx="1" fill="#b45309" fill-opacity="{opacity:.3f}"/>'
        )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="{escape("Piano roll")}">'
        + "".join(bars)
        + "</svg>"
    )
