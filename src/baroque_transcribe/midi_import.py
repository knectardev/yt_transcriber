"""Import a MIDI file as a first-class Library project.

Note times come from pretty_midi's absolute seconds (tempo map applied).
When the file already encodes separate staves/tracks, that structure is kept
for the score instead of re-splitting at middle C.
"""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pretty_midi

from baroque_transcribe.defaults import (
    DEFAULT_STAFF_SPLIT,
    DEFAULT_SUBDIVISION,
    MANIFEST,
    NOTES_JSON,
    PEDAL_JSON,
    RAW_MID,
    SCORE_XML,
    SOURCE_MID,
)
from baroque_transcribe.midi_export import write_raw_midi
from baroque_transcribe.notation_settings import NotationSettings
from baroque_transcribe.pipeline import notate_transcription, unique_output_dir
from baroque_transcribe.schema import (
    NoteEvent,
    PedalEvent,
    Transcription,
    clean_notes,
    clean_pedals,
    write_pedal_file,
)

_RIGHT_NAME = re.compile(r"(right|treble|rh\b|hand\s*1|staff\s*1)", re.I)
_LEFT_NAME = re.compile(r"(left|bass|lh\b|hand\s*2|staff\s*2)", re.I)


class MidiImportError(ValueError):
    pass


@dataclass(frozen=True)
class MidiTiming:
    tempo_map: list[dict]
    time_signature_map: list[dict]
    end_time: float
    bpm: float
    time_signature: tuple[int, int]
    constant_tempo: bool
    track_staffs: bool
    has_time_signature_events: bool


def _staff_for_instrument(index: int, name: str, track_count: int) -> int | None:
    if track_count < 2:
        return None
    if _RIGHT_NAME.search(name or ""):
        return 1
    if _LEFT_NAME.search(name or ""):
        return 2
    if track_count == 2:
        return 1 if index == 0 else 2
    return 1 if index == 0 else 2


def parse_midi(path: Path) -> tuple[Transcription, MidiTiming]:
    """Read notes, pedals, and timing maps from a MIDI file."""

    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise MidiImportError(f"MIDI file not found: {path}")
    if path.suffix.lower() not in {".mid", ".midi"}:
        raise MidiImportError("Choose a .mid or .midi file.")
    try:
        midi = pretty_midi.PrettyMIDI(str(path))
    except Exception as exc:
        raise MidiImportError(f"Could not read {path.name} as MIDI: {exc}") from exc

    melodic = [inst for inst in midi.instruments if not inst.is_drum and inst.notes]
    track_staffs = len(melodic) >= 2
    notes: list[NoteEvent] = []
    for index, instrument in enumerate(melodic):
        staff = _staff_for_instrument(index, instrument.name or "", len(melodic))
        for note in instrument.notes:
            notes.append(
                NoteEvent(
                    pitch=int(note.pitch),
                    onset=float(note.start),
                    offset=float(note.end),
                    velocity=int(note.velocity),
                    staff=staff if track_staffs else None,
                )
            )

    pedals: list[PedalEvent] = []
    for instrument in midi.instruments:
        down: float | None = None
        for change in sorted(instrument.control_changes, key=lambda item: item.time):
            if change.number != 64:
                continue
            if change.value >= 64 and down is None:
                down = float(change.time)
            elif change.value < 64 and down is not None:
                pedals.append(PedalEvent(onset=down, offset=float(change.time)))
                down = None
        if down is not None:
            pedals.append(PedalEvent(onset=down, offset=max(down + 0.01, float(midi.get_end_time()))))

    cleaned = clean_notes(notes)
    if not cleaned:
        raise MidiImportError(f"{path.name} has no playable piano notes.")

    tempo_times, tempo_values = midi.get_tempo_changes()
    tempo_map = [
        {"time": float(time), "bpm": float(bpm)}
        for time, bpm in zip(tempo_times.tolist(), tempo_values.tolist(), strict=False)
    ]
    if not tempo_map:
        tempo_map = [{"time": 0.0, "bpm": 120.0}]
    constant_tempo = len({round(item["bpm"], 6) for item in tempo_map}) == 1

    has_time_signature_events = bool(midi.time_signature_changes)
    time_signature_map = [
        {
            "time": float(mark.time),
            "numerator": int(mark.numerator),
            "denominator": int(mark.denominator),
        }
        for mark in midi.time_signature_changes
    ]
    if not time_signature_map:
        time_signature_map = [{"time": 0.0, "numerator": 4, "denominator": 4}]

    first_ts = time_signature_map[0]
    timing = MidiTiming(
        tempo_map=tempo_map,
        time_signature_map=time_signature_map,
        end_time=float(midi.get_end_time()),
        bpm=float(tempo_map[0]["bpm"]),
        time_signature=(int(first_ts["numerator"]), int(first_ts["denominator"])),
        constant_tempo=constant_tempo,
        track_staffs=track_staffs,
        has_time_signature_events=has_time_signature_events,
    )
    transcription = Transcription(
        notes=cleaned,
        pedals=clean_pedals(pedals),
        model="midi-import",
        source=path.name,
    )
    return transcription, timing


def _write_score_from_midi(midi_path: Path, dest: Path, *, title: str) -> Path:
    """Prefer music21's track→staff rendering of the source MIDI."""

    from music21 import converter, metadata

    from baroque_transcribe.notation import NotationError

    try:
        parsed = converter.parse(str(midi_path))
        if parsed.metadata is None:
            parsed.metadata = metadata.Metadata()
        parsed.metadata.title = title
        written = parsed.write("musicxml", fp=str(dest))
    except Exception as exc:
        raise MidiImportError(f"Could not write MusicXML from {midi_path.name}: {exc}") from exc
    written_path = Path(str(written)) if written else dest
    if written_path != dest and written_path.exists():
        written_path.replace(dest)
    if not dest.is_file():
        raise NotationError(f"music21 did not write {dest}.")
    return dest


def import_midi_project(
    midi_path: Path,
    output_dir: Path,
    *,
    title: str | None = None,
    youtube_url: str | None = None,
    bpm: float | None = None,
    time_signature: tuple[int, int] | None = None,
    subdivision: int = DEFAULT_SUBDIVISION,
    downbeat: float = 0.0,
    staff_split: int = DEFAULT_STAFF_SPLIT,
    settings: NotationSettings | None = None,
) -> Path:
    """Create a Library project from a MIDI file and return its folder."""

    midi_path = Path(midi_path).expanduser().resolve()
    transcription, timing = parse_midi(midi_path)
    resolved_title = (title or Path(midi_path).stem or "MIDI import").strip() or "MIDI import"
    project = unique_output_dir(Path(output_dir))
    project.mkdir(parents=True, exist_ok=True)

    stored_midi = project / SOURCE_MID
    if midi_path.resolve() != stored_midi.resolve():
        shutil.copy2(midi_path, stored_midi)

    transcription.source = midi_path.name
    transcription.write(project / NOTES_JSON)
    write_pedal_file(project / PEDAL_JSON, transcription.pedals, transcription.model)
    write_raw_midi(project / RAW_MID, transcription.notes, transcription.pedals)

    resolved_bpm = float(bpm) if bpm is not None else timing.bpm
    resolved_meter = time_signature or timing.time_signature
    meter_assumed = time_signature is None and not timing.has_time_signature_events

    warnings: list[str] = []
    if not timing.constant_tempo:
        warnings.append(
            "This MIDI has tempo changes. Note times keep pretty_midi absolute timing; "
            "the notation grid uses the first tempo as a representative BPM."
        )
    if len(timing.time_signature_map) > 1:
        warnings.append(
            "This MIDI has multiple time signatures. The notation grid uses the first one."
        )

    manifest = {
        "origin": str(midi_path),
        "title": resolved_title,
        "model": "midi-import",
        "source_kind": "midi",
        "counts": {"notes": len(transcription.notes), "pedals": len(transcription.pedals)},
        "audio": {
            "working_duration_seconds": round(timing.end_time, 5),
            "excerpt_start_seconds": 0.0,
            "excerpt_end_seconds": None,
        },
        "staff_assignment": "tracks" if timing.track_staffs else "pitch_split",
        "warnings": warnings,
    }
    if youtube_url:
        manifest["youtube_url"] = youtube_url.strip()
    (project / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    tempo_info = {
        "source": "midi",
        "tempo_map": timing.tempo_map,
        "time_signature_map": timing.time_signature_map,
        "constant_tempo": timing.constant_tempo,
    }
    notate_transcription(
        transcription,
        project,
        bpm=resolved_bpm,
        time_signature=resolved_meter,
        subdivision=int(subdivision),
        downbeat=float(downbeat),
        staff_split=int(staff_split),
        title=resolved_title,
        tempo_info=tempo_info,
        time_signature_assumed=bool(meter_assumed),
        settings=settings,
    )

    if timing.track_staffs:
        # Replace the pitch-split score with music21's track→staff rendering.
        _write_score_from_midi(stored_midi, project / SCORE_XML, title=resolved_title)
        updated = json.loads((project / MANIFEST).read_text(encoding="utf-8"))
        notation = dict(updated.get("notation") or {})
        notation["score_from"] = "midi_tracks"
        updated["notation"] = notation
        (project / MANIFEST).write_text(json.dumps(updated, indent=2) + "\n", encoding="utf-8")

    try:
        from baroque_transcribe.parse_store import record_parse

        record_parse(
            action="import",
            status="ok",
            source=str(midi_path),
            project=project.name,
            project_dir=project,
            title=resolved_title,
            youtube_url=youtube_url,
            duration=timing.end_time,
            via="midi-import",
        )
    except Exception:
        pass

    return project
