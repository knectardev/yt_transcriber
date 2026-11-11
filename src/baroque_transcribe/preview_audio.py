"""Cached synthetic preview audio for projects and fragments' shared pattern."""

from __future__ import annotations

import hashlib
from pathlib import Path

from baroque_transcribe.defaults import LOCAL_DIR, NOTES_JSON, PROJECT_AUDIO_DIR
from baroque_transcribe.parse_store import repo_root
from baroque_transcribe.preview_synth import DEFAULT_PREVIEW_SOUND, resolve_preview_sound, synthesize_preview
from baroque_transcribe.schema import read_transcription

PREVIEW_SAMPLE_RATE = 22_050


class PreviewAudioError(ValueError):
    pass


def project_audio_root(local: Path | None = None) -> Path:
    base = local if local is not None else repo_root() / LOCAL_DIR
    return Path(base) / PROJECT_AUDIO_DIR


def ensure_project_preview_wav(
    project: Path,
    *,
    sound: str | None = None,
    local: Path | None = None,
    sample_rate: int = PREVIEW_SAMPLE_RATE,
) -> Path:
    """Render a cached WAV of the project's ``notes.json`` performance times.

    The file lives under ``.local/project_audio/`` and is named with a hash of
    the notes so edits rebuild automatically. Failure raises ``PreviewAudioError``.
    """

    import soundfile as sf

    project = Path(project)
    notes_path = project / NOTES_JSON
    if not notes_path.is_file():
        raise PreviewAudioError(f"This project has no {NOTES_JSON} to preview.")
    if sample_rate < 8_000:
        raise PreviewAudioError("The preview sample rate is too low.")
    try:
        preset = resolve_preview_sound(sound or DEFAULT_PREVIEW_SOUND)
    except ValueError as exc:
        raise PreviewAudioError(str(exc)) from exc

    transcription = read_transcription(notes_path)
    if not transcription.notes:
        raise PreviewAudioError("This project has no notes to preview.")
    duration = max(note.offset for note in transcription.notes)
    digest = hashlib.sha256(notes_path.read_bytes()).hexdigest()[:16]
    cache = project_audio_root(local)
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"{project.name}-performance-{preset.id}-{digest}.wav"
    if target.is_file() and target.stat().st_size > 0:
        return target

    audio = synthesize_preview(transcription.notes, int(sample_rate), float(duration), sound=preset)
    for old in cache.glob(f"{project.name}-performance-{preset.id}-*.wav"):
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
