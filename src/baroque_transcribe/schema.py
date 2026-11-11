"""Editable intermediate representation for a piano transcription.

``notes.json`` is the source of truth after the model runs. Quantization,
MusicXML, diagnostics, and any later contrapuntal analysis all read it.
Hand-editing the file and running ``quantize`` again is the correction path.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from baroque_transcribe.storage_io import write_json_atomic

SCHEMA_VERSION = 1


@dataclass
class NoteEvent:
    pitch: int
    onset: float
    offset: float
    velocity: int
    # Optional MusicXML/MIDI staff (1 = treble, 2 = bass). When set on import,
    # notation prefers this over the mid-C staff_split heuristic.
    staff: int | None = None

    @property
    def duration(self) -> float:
        return self.offset - self.onset


@dataclass
class PedalEvent:
    onset: float
    offset: float

    @property
    def duration(self) -> float:
        return self.offset - self.onset


@dataclass
class Transcription:
    notes: list[NoteEvent] = field(default_factory=list)
    pedals: list[PedalEvent] = field(default_factory=list)
    model: str = ""
    source: str = ""

    def to_dict(self) -> dict:
        notes: list[dict] = []
        for note in self.notes:
            payload = {
                "pitch": note.pitch,
                "onset": note.onset,
                "offset": note.offset,
                "velocity": note.velocity,
            }
            if note.staff in (1, 2):
                payload["staff"] = int(note.staff)
            notes.append(payload)
        return {
            "version": SCHEMA_VERSION,
            "model": self.model,
            "source": self.source,
            "notes": notes,
            "pedals": [asdict(pedal) for pedal in self.pedals],
        }

    def write(self, path: Path) -> None:
        write_json_atomic(path, self.to_dict())


def clean_notes(notes: list[NoteEvent]) -> list[NoteEvent]:
    """Drop impossible pitches and clamp MIDI velocities into 1..127.

    A velocity of 0 is a note-off in MIDI, so a detected note never keeps it.
    Durations under 10 ms are extended rather than deleted: ornaments and
    repeated notes are often that short, and dropping them would be an
    editorial decision.
    """

    cleaned: list[NoteEvent] = []
    for note in notes:
        pitch = int(note.pitch)
        if pitch < 21 or pitch > 108:
            continue
        # Five decimals is 0.01 ms, finer than the model, and drops float32 noise
        # such as 0.649999976 so the JSON is practical to edit by hand.
        onset = round(float(note.onset), 5)
        offset = round(float(note.offset), 5)
        if offset < onset:
            onset, offset = offset, onset
        if onset < 0:
            if offset <= 0:
                continue
            onset = 0.0
        if offset - onset < 0.01:
            offset = onset + 0.01
        velocity = min(127, max(1, int(note.velocity)))
        staff = int(note.staff) if note.staff in (1, 2) else None
        cleaned.append(NoteEvent(pitch, onset, offset, velocity, staff=staff))
    cleaned.sort(key=lambda item: (item.onset, item.pitch, item.offset))
    return cleaned


def clean_pedals(pedals: list[PedalEvent]) -> list[PedalEvent]:
    cleaned: list[PedalEvent] = []
    for pedal in pedals:
        onset = round(float(pedal.onset), 5)
        offset = round(float(pedal.offset), 5)
        if offset < onset:
            onset, offset = offset, onset
        if offset <= 0:
            continue
        onset = max(0.0, onset)
        if offset - onset < 0.01:
            continue
        cleaned.append(PedalEvent(onset, offset))
    cleaned.sort(key=lambda item: item.onset)
    return cleaned


def transcription_from_dict(data: dict) -> Transcription:
    version = int(data.get("version", SCHEMA_VERSION))
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"notes.json uses schema version {version}; this program reads version {SCHEMA_VERSION}."
        )
    notes = []
    for item in data.get("notes", []):
        staff_raw = item.get("staff")
        staff = int(staff_raw) if staff_raw in (1, 2, "1", "2") else None
        notes.append(
            NoteEvent(
                pitch=int(item["pitch"]),
                onset=float(item["onset"]),
                offset=float(item["offset"]),
                velocity=int(item["velocity"]),
                staff=staff,
            )
        )
    pedals = [
        PedalEvent(onset=float(item["onset"]), offset=float(item["offset"]))
        for item in data.get("pedals", [])
    ]
    return Transcription(
        notes=clean_notes(notes),
        pedals=clean_pedals(pedals),
        model=str(data.get("model", "")),
        source=str(data.get("source", "")),
    )


def read_transcription(path: Path) -> Transcription:
    data = json.loads(path.read_text(encoding="utf-8"))
    return transcription_from_dict(data)


def write_pedal_file(path: Path, pedals: list[PedalEvent], model: str) -> None:
    payload = {
        "version": SCHEMA_VERSION,
        "model": model,
        "control": 64,
        "description": (
            "Sustain pedal down/up times in seconds, relative to working.wav. "
            "The same events are also stored as MIDI CC 64 inside raw.mid."
        ),
        "pedals": [asdict(pedal) for pedal in pedals],
    }
    write_json_atomic(path, payload)
