"""Cached synthetic preview audio for projects and fragments' shared pattern."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from baroque_transcribe.defaults import LOCAL_DIR, NOTATION_MAP_JSON, NOTES_JSON, PROJECT_AUDIO_DIR
from baroque_transcribe.parse_store import repo_root
from baroque_transcribe.preview_synth import (
    DEFAULT_PREVIEW_SOUND,
    resolve_preview_sound,
    synthesize_preview,
    synthesize_preview_by_voice,
)
from baroque_transcribe.schema import NoteEvent, read_transcription

PREVIEW_SAMPLE_RATE = 22_050


class PreviewAudioError(ValueError):
    pass


def project_audio_root(local: Path | None = None) -> Path:
    base = local if local is not None else repo_root() / LOCAL_DIR
    return Path(base) / PROJECT_AUDIO_DIR


def voice_assignments_from_notation_map(project: Path) -> dict[int, int] | None:
    """``source_index → global voice id`` from readable ``notation_map.json``, if present."""

    path = Path(project) / NOTATION_MAP_JSON
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    rows = payload.get("notes") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        return None
    mapping: dict[int, int] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            mapping[int(row["source_index"])] = int(row["voice"])
        except (KeyError, TypeError, ValueError):
            continue
    return mapping or None


def _resolve_pair(sound: str | None, sound2: str | None) -> tuple[str, str | None]:
    primary = resolve_preview_sound(sound or DEFAULT_PREVIEW_SOUND).id
    if sound2 is None or str(sound2).strip() == "":
        return primary, None
    try:
        secondary = resolve_preview_sound(sound2).id
    except ValueError:
        return primary, None
    return primary, secondary


def _clamp_volume(value: float | int | str | None, default: float = 0.5) -> float:
    if value is None or value == "":
        return default
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def ensure_project_preview_wav(
    project: Path,
    *,
    sound: str | None = None,
    sound2: str | None = None,
    volume: float | None = None,
    volume2: float | None = None,
    local: Path | None = None,
    sample_rate: int = PREVIEW_SAMPLE_RATE,
) -> Path:
    """Render a cached WAV of the project's ``notes.json`` performance times.

    When ``sound2`` is set and a readable notation map exists, voice 1 uses
    ``sound`` and voice 2 uses ``sound2``, each scaled by ``volume`` /
    ``volume2`` (default 0.5). Otherwise a single timbre is used.
    """

    import soundfile as sf

    project = Path(project)
    notes_path = project / NOTES_JSON
    if not notes_path.is_file():
        raise PreviewAudioError(f"This project has no {NOTES_JSON} to preview.")
    if sample_rate < 8_000:
        raise PreviewAudioError("The preview sample rate is too low.")
    primary, secondary = _resolve_pair(sound, sound2)
    vol1 = _clamp_volume(volume, 0.5)
    vol2 = _clamp_volume(volume2, 0.5)
    voice_map = voice_assignments_from_notation_map(project) if secondary else None
    use_voices = bool(secondary and voice_map)

    transcription = read_transcription(notes_path)
    if not transcription.notes:
        raise PreviewAudioError("This project has no notes to preview.")
    duration = max(note.offset for note in transcription.notes)
    digest = hashlib.sha256(notes_path.read_bytes()).hexdigest()[:16]
    cache = project_audio_root(local)
    cache.mkdir(parents=True, exist_ok=True)
    if use_voices:
        map_digest = hashlib.sha256((project / NOTATION_MAP_JSON).read_bytes()).hexdigest()[:12]
        target = (
            cache
            / (
                f"{project.name}-performance-{primary}+{secondary}"
                f"-v{vol1:.2f}+{vol2:.2f}-{digest}-{map_digest}.wav"
            )
        )
    else:
        target = cache / f"{project.name}-performance-{primary}-{digest}.wav"
    if target.is_file() and target.stat().st_size > 0:
        return target

    if use_voices and voice_map is not None:
        paired: list[tuple[NoteEvent, int]] = []
        for index, note in enumerate(transcription.notes):
            paired.append((note, int(voice_map.get(index, 1))))
        audio = synthesize_preview_by_voice(
            paired,
            int(sample_rate),
            float(duration),
            sounds_by_voice={1: primary, 2: secondary or primary},
            default_sound=primary,
            volumes_by_voice={1: vol1, 2: vol2},
        )
        pattern = f"{project.name}-performance-{primary}+{secondary}-*.wav"
    else:
        audio = synthesize_preview(
            transcription.notes, int(sample_rate), float(duration), sound=primary
        )
        pattern = f"{project.name}-performance-{primary}-*.wav"
    for old in cache.glob(pattern):
        if old != target:
            old.unlink(missing_ok=True)
    temporary = target.with_name(target.stem + ".tmp.wav")
    try:
        sf.write(temporary, audio, int(sample_rate))
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def movement_preview_filename(movement_id: str) -> str:
    """URL/path leaf for a movement preview, e.g. ``mov-allemande.wav``."""

    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in movement_id).strip("-")
    return f"mov-{(safe or 'section')}.wav"


def ensure_movement_preview_wav(
    project: Path,
    *,
    movement_id: str,
    start_seconds: float,
    end_seconds: float,
    sound: str | None = None,
    sound2: str | None = None,
    volume: float | None = None,
    volume2: float | None = None,
    local: Path | None = None,
    sample_rate: int = PREVIEW_SAMPLE_RATE,
) -> Path:
    """Cached WAV clip for one movement, starting at local time 0.

    Renders only notes inside the movement window (shifted to time 0). When
    ``sound2`` and a notation map are available, uses per-voice timbres and
    volumes (default 0.5 each).
    """

    import soundfile as sf

    if end_seconds <= start_seconds:
        raise PreviewAudioError("Movement end must be after start.")
    if sample_rate < 8_000:
        raise PreviewAudioError("The preview sample rate is too low.")
    primary, secondary = _resolve_pair(sound, sound2)
    vol1 = _clamp_volume(volume, 0.5)
    vol2 = _clamp_volume(volume2, 0.5)
    voice_map = voice_assignments_from_notation_map(project) if secondary else None
    use_voices = bool(secondary and voice_map)

    project = Path(project)
    notes_path = project / NOTES_JSON
    if not notes_path.is_file():
        raise PreviewAudioError(f"This project has no {NOTES_JSON} to preview.")
    digest = hashlib.sha256(notes_path.read_bytes()).hexdigest()[:16]
    cache = project_audio_root(local)
    cache.mkdir(parents=True, exist_ok=True)
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in movement_id).strip("-") or "section"
    if use_voices:
        map_digest = hashlib.sha256((project / NOTATION_MAP_JSON).read_bytes()).hexdigest()[:12]
        target = (
            cache
            / (
                f"{project.name}-mov-{safe}-{primary}+{secondary}"
                f"-v{vol1:.2f}+{vol2:.2f}-{digest}-{map_digest}"
                f"-{start_seconds:.3f}-{end_seconds:.3f}.wav"
            )
        )
    else:
        target = (
            cache
            / f"{project.name}-mov-{safe}-{primary}-{digest}-{start_seconds:.3f}-{end_seconds:.3f}.wav"
        )
    if target.is_file() and target.stat().st_size > 0:
        return target

    # Single-timbre: prefer slicing a cached full-suite preview when present.
    if not use_voices:
        suite = cache / f"{project.name}-performance-{primary}-{digest}.wav"
        if suite.is_file() and suite.stat().st_size > 0:
            info = sf.info(str(suite))
            rate = int(info.samplerate) or int(sample_rate)
            start_frame = max(0, int(round(float(start_seconds) * rate)))
            end_frame = max(start_frame + 1, int(round(float(end_seconds) * rate)))
            with sf.SoundFile(str(suite), mode="r") as handle:
                if start_frame < len(handle):
                    handle.seek(start_frame)
                    frames = min(end_frame, len(handle)) - start_frame
                    sliced = handle.read(frames, dtype="float32", always_2d=False)
                    if getattr(sliced, "size", 0) > 0:
                        temporary = target.with_name(target.stem + ".tmp.wav")
                        try:
                            sf.write(temporary, sliced, rate)
                            temporary.replace(target)
                            return target
                        except BaseException:
                            temporary.unlink(missing_ok=True)
                            raise

    transcription = read_transcription(notes_path)
    duration = float(end_seconds - start_seconds)
    if use_voices and voice_map is not None:
        paired: list[tuple[NoteEvent, int]] = []
        for index, note in enumerate(transcription.notes):
            if note.offset <= start_seconds + 1e-9:
                continue
            if note.onset >= end_seconds - 1e-9:
                continue
            onset = max(0.0, float(note.onset) - start_seconds)
            offset = min(duration, max(onset + 0.01, float(note.offset) - start_seconds))
            paired.append(
                (
                    NoteEvent(
                        pitch=int(note.pitch),
                        onset=onset,
                        offset=offset,
                        velocity=int(note.velocity),
                        staff=note.staff,
                    ),
                    int(voice_map.get(index, 1)),
                )
            )
        if not paired:
            raise PreviewAudioError("This movement has no notes to preview.")
        audio = synthesize_preview_by_voice(
            paired,
            int(sample_rate),
            duration,
            sounds_by_voice={1: primary, 2: secondary or primary},
            default_sound=primary,
            volumes_by_voice={1: vol1, 2: vol2},
        )
    else:
        window: list[NoteEvent] = []
        for note in transcription.notes:
            if note.offset <= start_seconds + 1e-9:
                continue
            if note.onset >= end_seconds - 1e-9:
                continue
            onset = max(0.0, float(note.onset) - start_seconds)
            offset = min(duration, max(onset + 0.01, float(note.offset) - start_seconds))
            window.append(
                NoteEvent(
                    pitch=int(note.pitch),
                    onset=onset,
                    offset=offset,
                    velocity=int(note.velocity),
                    staff=note.staff,
                )
            )
        if not window:
            raise PreviewAudioError("This movement has no notes to preview.")
        audio = synthesize_preview(window, int(sample_rate), duration, sound=primary)
    temporary = target.with_name(target.stem + ".tmp.wav")
    try:
        sf.write(temporary, audio, int(sample_rate))
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target
