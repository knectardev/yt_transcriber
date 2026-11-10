"""Shared builders for the fragment tests. Not a test module."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

from baroque_transcribe.analysis import analyze_harmony, write_analysis
from baroque_transcribe.defaults import ANNOTATIONS_JSON, MANIFEST, NOTES_JSON, QUANTIZED_MID, SCORE_XML
from baroque_transcribe.midi_export import write_quantized_midi
from baroque_transcribe.notation import write_musicxml
from baroque_transcribe.notation_settings import NotationSettings, settings_from_manifest
from baroque_transcribe.quantize import Quantization, quantization_from_manifest
from baroque_transcribe.schema import NoteEvent, PedalEvent, Transcription, clean_notes, read_transcription

FIXTURES = Path(__file__).parent / "fixtures"
YOUTUBE = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
LOCAL_SOURCE = r"C:\Users\someone\Music\private take.wav"


def n(pitch: int, onset: float, offset: float, velocity: int = 80) -> NoteEvent:
    return NoteEvent(pitch=pitch, onset=onset, offset=offset, velocity=velocity)


def make_project(
    tmp_path: Path,
    notes: list[NoteEvent],
    *,
    name: str = "parent",
    bpm: float = 120.0,
    time_signature: str = "4/4",
    subdivision: int = 16,
    downbeat: float = 0.0,
    grid: str = "fixed",
    style: str = "literal",
    excerpt: float = 0.0,
    sync: float = 0.0,
    pedals: list[PedalEvent] | None = None,
    youtube: str | None = YOUTUBE,
    analyze: bool = False,
    build: bool = True,
) -> Path:
    """A parent project with the files a real run leaves, minus the audio.

    At the default 120 BPM in 4/4 with sixteenths, one step is 0.125 s and
    one measure is 2 s, so step numbers are easy to read off a time.
    """

    project = tmp_path / name
    project.mkdir(parents=True)
    origin = youtube or LOCAL_SOURCE
    Transcription(notes=clean_notes(notes), pedals=list(pedals or []), model="fixture", source=origin).write(
        project / NOTES_JSON
    )
    settings = NotationSettings(grid=grid, style=style)
    manifest = {
        "origin": origin,
        "title": "Parent take",
        "audio": {"excerpt_start_seconds": excerpt},
        "sync_offset_seconds": sync,
        "staff_split": 60,
        "tempo": {
            "bpm": bpm,
            "downbeat_seconds": downbeat,
            "subdivision": subdivision,
            "time_signature": time_signature,
            "grid": grid,
        },
        "notation": {"style": style, "settings": settings.to_dict()},
    }
    if youtube:
        manifest["youtube_url"] = youtube
    (project / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if build:
        quantization = parent_quantization(project)
        if analyze:
            write_analysis(project / ANNOTATIONS_JSON, analyze_harmony(quantization))
        write_quantized_midi(project / QUANTIZED_MID, quantization, 60)
        write_musicxml(
            project / SCORE_XML,
            quantization,
            staff_split=60,
            title="Parent take",
            style=style,
            pedals=read_transcription(project / NOTES_JSON).pedals,
        )
    return project


def fixture_project(tmp_path: Path, *, grid: str, style: str, name: str = "parent", **kwargs) -> Path:
    """The Tip No. 4 excerpt: twelve or so measures of real two-hand playing."""

    meta = json.loads((FIXTURES / "tip4_measures_24_35.meta.json").read_text(encoding="utf-8"))
    notes = read_transcription(FIXTURES / "tip4_measures_24_35.json").notes
    return make_project(tmp_path, notes, name=name, bpm=float(meta["source_bpm"]), grid=grid, style=style, **kwargs)


def parent_quantization(project: Path) -> Quantization:
    manifest = json.loads((project / MANIFEST).read_text(encoding="utf-8"))
    return quantization_from_manifest(
        read_transcription(project / NOTES_JSON).notes,
        manifest["tempo"],
        settings=settings_from_manifest(manifest),
        manifest=manifest,
    )


def fake_manifest(
    fragment_id: str,
    *,
    title: str = "Fragment",
    tags=(),
    tonic: str = "G",
    mode: str = "minor",
    roman=(),
    symbols=(),
    description: str = "",
    practice_notes: str = "",
    composer: str = "",
    performer: str = "",
    video_id: str = "dQw4w9WgXcQ",
    source_title: str = "Parent take",
    project: str = "parent",
    measure_start: int = 1,
    measure_end: int = 2,
    start: float = 10.0,
    duration: float = 4.0,
    sync: float = 0.0,
    modified: str = "2026-11-10T09:00:00Z",
) -> dict:
    """A manifest with the shape ``build_manifest`` writes, without running an extraction."""

    from baroque_transcribe.fragments import FRAGMENT_SCHEMA_VERSION, validate_manifest

    return validate_manifest(
        {
            "schema_version": FRAGMENT_SCHEMA_VERSION,
            "id": fragment_id,
            "title": title,
            "description": description,
            "practice_notes": practice_notes,
            "tags": list(tags),
            "composer": composer,
            "performer": performer,
            "source": {
                "kind": "youtube",
                "url": f"https://www.youtube.com/watch?v={video_id}",
                "video_id": video_id,
                "title": source_title,
            },
            "parent": {
                "project": project,
                "event_id": "",
                "notes_fingerprint": "0123456789abcdef",
                "analysis_sha256": "",
                "measure_start": measure_start,
                "measure_end": measure_end,
                "step_start": (measure_start - 1) * 16,
                "step_end": measure_end * 16,
            },
            "time": {
                "working_audio": {"start_seconds": start, "end_seconds": start + duration},
                "source_video": {
                    "start_seconds": 30.0 + start,
                    "end_seconds": 30.0 + start + duration,
                    "excerpt_start_seconds": 30.0,
                },
                "duration_seconds": duration,
            },
            "sync": {"offset_seconds": sync, "parent_offset_at_extraction": 0.0},
            "music": {
                "analyzed_key": {"tonic": tonic, "mode": mode, "source": "automatic", "confidence": 0.9},
                "written_key_signature": {"tonic": tonic, "mode": mode},
                "time_signature": "4/4",
                "bpm": 120.0,
                "subdivision": 16,
                "staff_split": 60,
                "measures": measure_end - measure_start + 1,
                "parent_grid_mode": "fixed",
                "notation_style": "literal",
            },
            "harmony": {"roman": list(roman), "symbols": list(symbols)},
            "boundary": {
                "notes": 12,
                "owned": 12,
                "carried_in": 0,
                "cut_off": 0,
                "unprinted_overlaps": 0,
                "pedal_down_at_start": False,
            },
            "artifacts": {
                "notes": "notes.json",
                "grid": "grid.json",
                "performance_midi": "performance.mid",
                "quantized_midi": "quantized.mid",
                "musicxml": "score.musicxml",
                "harmony": None,
            },
            "warnings": [],
            "created_at": "2026-11-10T09:00:00Z",
            "modified_at": modified,
        }
    )


_CLOCK = [1_800_000_000]


def write_manifest(folder: Path, payload) -> Path:
    """Write ``fragment.json`` (a dict, or raw text) with a modification time later than any before.

    Each write gets its own second, so a test never depends on how fine the
    file system's clock is.
    """

    import os

    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "fragment.json"
    text = payload if isinstance(payload, str) else json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    _CLOCK[0] += 1
    os.utime(path, (_CLOCK[0], _CLOCK[0]))
    return path


def fake_fragment(root: Path, fragment_id: str, *, folder: str | None = None, **fields) -> Path:
    """A fragment folder that holds only a valid manifest. Enough for the index."""

    target = root / (folder or fragment_id)
    write_manifest(target, fake_manifest(fragment_id, **fields))
    return target


def tree_digest(folder: Path) -> dict[str, str]:
    """Content hash of every file under a folder, by relative path."""

    return {
        path.relative_to(folder).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.rglob("*"))
        if path.is_file()
    }


ScoreNote = tuple[Fraction, int, Fraction, str]


def score_notes(path: Path) -> dict[tuple[int, int], list[ScoreNote]]:
    """Every printed note as ``(offset in measure, MIDI pitch, quarter length, tie)``.

    Keyed by ``(staff, measure number)``. Rests are not included. ``tie`` is
    ``start``, ``continue``, ``stop``, or an empty string.
    """

    from music21 import converter

    parsed = converter.parse(str(path))
    found: dict[tuple[int, int], list[ScoreNote]] = {}
    for staff, part in enumerate(parsed.parts, start=1):
        for measure in part.getElementsByClass("Measure"):
            entries = []
            for element in measure.recurse().notes:
                offset = Fraction(element.getOffsetInHierarchy(measure)).limit_denominator(960)
                length = Fraction(element.quarterLength).limit_denominator(960)
                inner = element.notes if element.isChord else [element]
                for item in inner:
                    kind = str(item.tie.type) if item.tie is not None else ""
                    entries.append((offset, int(item.pitch.midi), length, kind))
            found[(staff, int(measure.number))] = sorted(entries)
    return found


def score_measures(path: Path) -> dict[tuple[int, int], list[tuple[Fraction, int, Fraction]]]:
    """``score_notes`` without the ties: pitch, position and written duration only."""

    return {
        key: sorted((offset, pitch, length) for offset, pitch, length, _tie in entries)
        for key, entries in score_notes(path).items()
    }


def score_ties(path: Path) -> dict[tuple[int, int], list[tuple[Fraction, int, str]]]:
    """``(offset in measure, MIDI pitch, tie type)`` for every tied printed note."""

    return {
        key: sorted((offset, pitch, tie) for offset, pitch, _length, tie in entries if tie)
        for key, entries in score_notes(path).items()
    }


def cut_at_barlines(entries: list[ScoreNote], *, first: bool, last: bool, bar_length: Fraction) -> list[ScoreNote]:
    """What a parent measure becomes when the fragment begins or ends at its barline.

    This is the documented boundary change. A note tied in from the previous
    measure begins at the first barline instead. A note tied into the next
    measure ends at the last barline instead. Pitch, position and written
    duration inside the measure are untouched.
    """

    result = []
    for offset, pitch, length, tie in entries:
        if first and offset == 0 and tie in {"stop", "continue"}:
            tie = "" if tie == "stop" else "start"
        if last and offset + length == bar_length and tie in {"start", "continue"}:
            tie = "" if tie == "start" else "stop"
        result.append((offset, pitch, length, tie))
    return sorted(result)


def measure_count(path: Path) -> list[int]:
    """Number of measures in which each staff has something written, note or rest.

    Read from the file itself. music21 pads the shorter staff when it parses
    a piano part, so a parsed score always reports equal lengths.
    """

    import xml.etree.ElementTree as ET

    counts = [0, 0]
    for measure in ET.parse(str(path)).getroot().iter("measure"):
        staves = {int(element.findtext("staff") or 1) for element in measure.iter("note")}
        for staff in (1, 2):
            if staff in staves:
                counts[staff - 1] += 1
    return counts


def parent_interpretation(project: Path):
    """The readable interpretation a parent's score was written from."""

    from baroque_transcribe.engrave import interpret_quantization

    parent = parent_quantization(project)
    manifest = json.loads((project / MANIFEST).read_text(encoding="utf-8"))
    return interpret_quantization(
        parent,
        staff_split=int(manifest.get("staff_split") or 60),
        settings=parent.settings,
        pedals=read_transcription(project / NOTES_JSON).pedals,
    )


def displaced_measures(project: Path) -> set[tuple[int, int]]:
    """``(staff, measure)`` pairs where a readable parent's file misplaces notes.

    The readable interpretation can leave two events of one voice overlapping.
    The writer lays a voice out end to end, so the later event is printed
    late and the voice runs past its barline. music21 then ties the surplus
    into the next measure, which pushes that measure's notes late too. This
    is how whole-transcription scores are written today; it is not part of
    the fragment work. In these measures the parent's file is not a reference
    for where a note belongs, so a fragment is checked against the parent's
    interpretation instead.

    A literal parent has no such measures: its voices are packed so that
    events never overlap.
    """

    from baroque_transcribe.sync_map import printed_measure_span

    parent = parent_quantization(project)
    if parent.settings.style != "readable":
        return set()
    first, count = printed_measure_span(parent)
    per = parent.steps_per_measure
    found: set[tuple[int, int]] = set()
    for staff, voices in parent_interpretation(project).events_by_staff.items():
        for events in voices:
            for index in range(count):
                start = first + index * per
                end = start + per
                cursor = start
                clips = sorted(
                    (max(event.onset_step, start), min(event.offset_step, end))
                    for event in events
                    if event.offset_step > start and event.onset_step < end
                )
                for clip_start, clip_end in clips:
                    if clip_start < cursor:
                        found.add((staff, index + 1))
                    cursor = clip_end
    # The surplus lands in the following measure, and can run on from there.
    spread = set(found)
    for staff, number in found:
        spread.add((staff, number + 1))
        spread.add((staff, number + 2))
    return spread


def total_measures(path: Path) -> int:
    """Number of ``<measure>`` elements in the file."""

    import xml.etree.ElementTree as ET

    return sum(1 for _measure in ET.parse(str(path)).getroot().iter("measure"))


def midi_notes(path: Path) -> list[list[tuple[int, float, float, int]]]:
    """``(pitch, start, end, velocity)`` per track, sorted."""

    import pretty_midi

    midi = pretty_midi.PrettyMIDI(str(path))
    return [
        sorted((int(note.pitch), float(note.start), float(note.end), int(note.velocity)) for note in instrument.notes)
        for instrument in midi.instruments
    ]
