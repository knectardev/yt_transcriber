"""Build a two-staff MusicXML score from the quantized grid.

Notes that start and end together on the same staff become a chord. Notes
with independent durations stay in separate voices. The staff split is a
pitch threshold, not a contrapuntal analysis: a middle-voice crossing will
need a hand edit in MuseScore. That analysis is deliberately not guessed here.
"""

from __future__ import annotations

import json
import math
import shutil
from dataclasses import dataclass
from fractions import Fraction
from html import escape
from pathlib import Path

from music21 import chord, clef, instrument, key as m21key, layout, meter, note, pitch, stream, tempo, tie

from baroque_transcribe.analysis import load_analysis, spell_pitch_name
from baroque_transcribe.defaults import ANNOTATIONS_JSON, NOTES_JSON, SCORE_XML
from baroque_transcribe.quantize import GridNote, Quantization


class NotationError(RuntimeError):
    pass


# Stable palette for the voice-continuity diagnostic. Same (staff, voice) keeps
# the same colour across the whole score.
VOICE_COLORS = (
    "#c62828",
    "#1565c0",
    "#2e7d32",
    "#6a1b9a",
    "#ef6c00",
    "#00838f",
    "#ad1457",
    "#4527a0",
)

# Durations (in grid steps at subdivision 16) that read as one symbol.
_CONVENTIONAL_STEPS = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32}


def voice_color(staff: int, voice: int) -> str:
    return VOICE_COLORS[(staff * 8 + voice - 1) % len(VOICE_COLORS)]


def nearest_second(seconds: float) -> int:
    if seconds >= 0:
        return int(math.floor(seconds + 0.5))
    return -int(math.floor(-seconds + 0.5))


def format_measure_time(seconds: float) -> str:
    """Clock time in parentheses, rounded to the nearest second. ``(1:05)``."""

    whole = nearest_second(seconds)
    sign = "-" if whole < 0 else ""
    whole = abs(whole)
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"({sign}{hours}:{minutes:02d}:{secs:02d})"
    return f"({sign}{minutes}:{secs:02d})"


def measure_time_labels(
    score_path: Path,
    *,
    time_offset_seconds: float = 0.0,
) -> dict[str, str]:
    """Map each printed measure number to its start time.

    Time is measured from the beginning of the score, which is the beginning
    of the transcribed audio when the downbeat is at the start.
    ``time_offset_seconds`` shifts the printed clock (for excerpt diagnostics
    that should show absolute performance time).
    """

    from music21 import converter
    from music21 import tempo as score_tempo

    parsed = converter.parse(str(score_path))
    parts = list(parsed.parts)
    if not parts:
        return {}
    part = max(parts, key=lambda item: len(list(item.getElementsByClass("Measure"))))
    bpm = 120.0
    elapsed = 0.0
    labels: dict[str, str] = {}
    for measure in part.getElementsByClass("Measure"):
        for mark in measure.recurse().getElementsByClass(score_tempo.MetronomeMark):
            if mark.number:
                bpm = float(mark.number)
        if measure.number is None or bpm <= 0:
            continue
        labels[str(measure.number)] = format_measure_time(elapsed + time_offset_seconds)
        expected = float(measure.barDuration.quarterLength)
        filled = float(measure.duration.quarterLength)
        quarters = filled if 0 < filled < expected - 1e-6 else expected
        elapsed += quarters * 60.0 / bpm
    return labels


@dataclass
class GridEvent:
    onset: int
    offset: int
    pitches: list[int]
    velocity: int


def _collapse(notes: list[GridNote]) -> list[GridEvent]:
    grouped: dict[tuple[int, int], list[GridNote]] = {}
    for item in notes:
        grouped.setdefault((item.onset_step, item.offset_step), []).append(item)
    events = []
    for (onset, offset), members in grouped.items():
        members.sort(key=lambda item: item.pitch)
        velocity = int(round(sum(item.velocity for item in members) / len(members)))
        events.append(
            GridEvent(
                onset=onset,
                offset=offset,
                pitches=[item.pitch for item in members],
                velocity=velocity,
            )
        )
    events.sort(key=lambda item: (item.onset, max(item.pitches)))
    return events


def _pack_voices(events: list[GridEvent]) -> list[list[GridEvent]]:
    voices: list[list[GridEvent]] = []
    ends: list[int] = []
    for event in sorted(events, key=lambda item: (item.onset, -max(item.pitches))):
        placed = False
        for index, end in enumerate(ends):
            if event.onset >= end:
                voices[index].append(event)
                ends[index] = event.offset
                placed = True
                break
        if not placed:
            voices.append([event])
            ends.append(event.offset)
    return voices or [[]]


def _steps_to_ql(steps: int, subdivision: int) -> Fraction:
    return Fraction(steps * 4, subdivision)


def _tie(element: note.GeneralNote, kind: str) -> None:
    mark = tie.Tie(kind)
    if isinstance(element, chord.Chord):
        for inner in element.notes:
            inner.tie = tie.Tie(kind)
    else:
        element.tie = mark


def _midi_pitch(
    midi_note: int,
    *,
    key_tonic: str | None = None,
    key_mode: str | None = None,
    previous_midi: int | None = None,
) -> pitch.Pitch:
    if key_tonic and key_mode:
        rising = None if previous_midi is None else midi_note >= previous_midi
        name = spell_pitch_name(midi_note, key_tonic, key_mode, rising=rising)
        octave = (int(midi_note) // 12) - 1
        spelled = pitch.Pitch(f"{name}{octave}")
        # Guard against rare octave slips from spelling.
        if spelled.midi != int(midi_note):
            spelled.midi = int(midi_note)
        return spelled
    spelled = pitch.Pitch()
    spelled.midi = midi_note
    return spelled


def _general_note(
    pitches: list[int],
    quarter_length: Fraction,
    velocity: int,
    *,
    key_tonic: str | None = None,
    key_mode: str | None = None,
    color: str | None = None,
    previous_midi: int | None = None,
) -> note.GeneralNote:
    if len(pitches) == 1:
        element: note.GeneralNote = note.Note(
            _midi_pitch(pitches[0], key_tonic=key_tonic, key_mode=key_mode, previous_midi=previous_midi)
        )
    else:
        spelled = []
        prev = previous_midi
        for midi_note in pitches:
            spelled.append(
                _midi_pitch(midi_note, key_tonic=key_tonic, key_mode=key_mode, previous_midi=prev)
            )
            prev = midi_note
        element = chord.Chord(spelled)
    element.quarterLength = quarter_length
    element.volume.velocity = velocity
    if color:
        element.style.color = color
        if isinstance(element, chord.Chord):
            for inner in element.notes:
                inner.style.color = color
    return element


def _is_conventional(steps: int, subdivision: int) -> bool:
    if steps <= 0:
        return False
    # Normalize to a 16th-note grid so the table stays readable.
    scaled = steps * 16 // subdivision if subdivision else steps
    return scaled in _CONVENTIONAL_STEPS


def _split_span(
    start: int,
    end: int,
    *,
    subdivision: int,
    steps_per_measure: int,
    beat_steps: int,
) -> list[tuple[int, int]]:
    """Selectively split a sustained span at the mid-bar when that improves reading."""

    if end <= start:
        return []
    duration = end - start
    mid = (start // steps_per_measure) * steps_per_measure + steps_per_measure // 2
    if start < mid < end and not _is_conventional(duration, subdivision):
        # Keep a short pickup into the mid-bar intact when it is already clear.
        if mid - start >= beat_steps or not _is_conventional(mid - start, subdivision):
            return [(start, mid), (mid, end)]
    return [(start, end)]


def _render_voice(
    events: list[GridEvent],
    start: int,
    end: int,
    subdivision: int,
    *,
    steps_per_measure: int | None = None,
    beat_steps: int | None = None,
    key_tonic: str | None = None,
    key_mode: str | None = None,
    color: str | None = None,
    previous_midi: int | None = None,
) -> list[note.GeneralNote]:
    active: list[tuple[GridEvent, int, int]] = []
    for event in events:
        if event.offset <= start or event.onset >= end:
            continue
        active.append((event, max(event.onset, start), min(event.offset, end)))
    active.sort(key=lambda item: item[1])
    rendered: list[note.GeneralNote] = []
    cursor = start
    per_measure = steps_per_measure or (end - start)
    beats = beat_steps or max(1, subdivision // 4)
    last_midi = previous_midi
    for event, clip_start, clip_end in active:
        if clip_start > cursor:
            rest = note.Rest(quarterLength=_steps_to_ql(clip_start - cursor, subdivision))
            rendered.append(rest)
        spans = _split_span(
            clip_start,
            clip_end,
            subdivision=subdivision,
            steps_per_measure=per_measure,
            beat_steps=beats,
        )
        for index, (span_start, span_end) in enumerate(spans):
            element = _general_note(
                event.pitches,
                _steps_to_ql(span_end - span_start, subdivision),
                event.velocity,
                key_tonic=key_tonic,
                key_mode=key_mode,
                color=color,
                previous_midi=last_midi,
            )
            continues_left = event.onset < start or span_start > clip_start or index > 0
            continues_right = event.offset > end or span_end < clip_end or index < len(spans) - 1
            # Measure-boundary ties from the caller still apply.
            if event.onset < start and index == 0:
                continues_left = True
            if event.offset > end and index == len(spans) - 1:
                continues_right = True
            if continues_left and continues_right:
                _tie(element, "continue")
            elif continues_left:
                _tie(element, "stop")
            elif continues_right:
                _tie(element, "start")
            rendered.append(element)
            last_midi = max(event.pitches)
        cursor = clip_end
    if cursor < end:
        rendered.append(note.Rest(quarterLength=_steps_to_ql(end - cursor, subdivision)))
    return rendered


def _staff_events(events: list[GridEvent], staff_split: int, treble: bool) -> list[GridEvent]:
    selected: list[GridEvent] = []
    for event in events:
        pitches = [
            pitch
            for pitch in event.pitches
            if (pitch >= staff_split if treble else pitch < staff_split)
        ]
        if pitches:
            selected.append(
                GridEvent(event.onset, event.offset, pitches, event.velocity)
            )
    return selected


def _opening_key(key_tonic: str | None, key_mode: str | None):
    if not key_tonic or not key_mode:
        return None
    try:
        return m21key.Key(key_tonic, key_mode)
    except Exception:
        return None


def _beam_measure(measure: stream.Measure) -> None:
    try:
        measure.makeBeams(inPlace=True)
    except Exception:
        # Incomplete or multi-voice edge cases can confuse makeBeams; leave as-is.
        pass


def _fill_part(
    part: stream.Part,
    events: list[GridEvent],
    quantization: Quantization,
    opening_clef: clef.Clef,
    *,
    key_tonic: str | None = None,
    key_mode: str | None = None,
    include_tempo: bool = True,
    measure_number_offset: int = 0,
) -> None:
    per_measure = quantization.steps_per_measure
    first = -quantization.lead_steps
    last_note = max((event.offset for event in events), default=per_measure)
    last_bar = (max(last_note - 1, first) // per_measure) * per_measure
    if last_bar < first:
        last_bar = first
    voices = _pack_voices(events)
    signature = meter.TimeSignature(f"{quantization.numerator}/{quantization.denominator}")
    mark = tempo.MetronomeMark(number=quantization.bpm)
    opening_key = _opening_key(key_tonic, key_mode)
    beat_steps = max(1, round(quantization.subdivision / quantization.denominator))
    measure_index = 0
    start = first
    while start <= last_bar:
        measure = stream.Measure(number=measure_index + 1 + measure_number_offset)
        if measure_index == 0:
            measure.insert(0, opening_clef)
            measure.insert(0, signature)
            if include_tempo:
                measure.insert(0, mark)
            if opening_key is not None:
                measure.insert(0, opening_key)
        voice_notes = [
            _render_voice(
                voice,
                start,
                start + per_measure,
                quantization.subdivision,
                steps_per_measure=per_measure,
                beat_steps=beat_steps,
                key_tonic=key_tonic,
                key_mode=key_mode,
            )
            for voice in voices
        ]
        if len(voice_notes) == 1:
            for element in voice_notes[0]:
                measure.append(element)
        else:
            for voice_id, elements in enumerate(voice_notes, start=1):
                voice = stream.Voice(id=str(voice_id))
                for element in elements:
                    voice.append(element)
                measure.insert(0, voice)
        _beam_measure(measure)
        part.append(measure)
        start += per_measure
        measure_index += 1


def _voice_events_to_grid(events: list) -> list[GridEvent]:
    return [
        GridEvent(
            onset=event.onset_step,
            offset=event.offset_step,
            pitches=list(event.pitches),
            velocity=event.velocity,
        )
        for event in events
    ]


def _hidden_rests_to_forward(musicxml_text: str) -> str:
    """Replace print-object=no rests with ``<forward>`` so OSMD cannot draw them.

    music21 can emit ``print-object="no"`` rests, but OpenSheetMusicDisplay still
    paints those glyphs (often far above the staff). ``<forward>`` keeps timing.
    """

    import re

    pattern = re.compile(
        r'<note\b[^>]*print-object="no"[^>]*>\s*<rest\s*/>.*?</note>',
        re.S,
    )

    def replace(match: re.Match[str]) -> str:
        chunk = match.group(0)
        duration = re.search(r"<duration>(\d+)</duration>", chunk)
        if duration is None:
            return chunk
        return f"<forward>\n        <duration>{duration.group(1)}</duration>\n      </forward>"

    return pattern.sub(replace, musicxml_text)


def _export_voice_number(staff: int, voice_id: int) -> int:
    """Disjoint MusicXML voice numbers for each staff of a PartStaff piano part.

    OSMD lays out voices globally inside one part. If the bass reuses 1..n, those
    layers shove treble rests far above the staff. Keep treble on 1.. and bass on
    9.. so the staves do not share voice layers.
    """

    return voice_id if staff == 1 else voice_id + 8


def _fill_part_readable(
    part: stream.Part,
    staff_voices: list[list],
    quantization: Quantization,
    opening_clef: clef.Clef,
    *,
    staff: int,
    key_tonic: str | None = None,
    key_mode: str | None = None,
    colorize_voices: bool = False,
    include_tempo: bool = True,
    measure_number_offset: int = 0,
) -> None:
    """Fill a part from interpreted voices, omitting absent-voice measures."""

    per_measure = quantization.steps_per_measure
    first = -quantization.lead_steps
    last_note = max(
        (event.offset_step for voice in staff_voices for event in voice),
        default=per_measure,
    )
    last_bar = (max(last_note - 1, first) // per_measure) * per_measure
    if last_bar < first:
        last_bar = first
    signature = meter.TimeSignature(f"{quantization.numerator}/{quantization.denominator}")
    mark = tempo.MetronomeMark(number=quantization.bpm)
    opening_key = _opening_key(key_tonic, key_mode)
    beat_steps = max(1, round(quantization.subdivision / quantization.denominator))
    measure_index = 0
    start = first
    while start <= last_bar:
        end = start + per_measure
        measure = stream.Measure(number=measure_index + 1 + measure_number_offset)
        if measure_index == 0:
            measure.insert(0, opening_clef)
            measure.insert(0, signature)
            # Tempo only on the upper staff so PartStaff export does not
            # print a duplicate metronome mark on the bass staff.
            if include_tempo:
                measure.insert(0, mark)
            if opening_key is not None:
                measure.insert(0, opening_key)
        active: list[tuple[int, list[note.GeneralNote]]] = []
        for voice_id, events in enumerate(staff_voices, start=1):
            sounding = any(event.onset_step < end and event.offset_step > start for event in events)
            if not sounding:
                continue
            color = voice_color(staff, voice_id) if colorize_voices else None
            rendered = _render_voice(
                _voice_events_to_grid(events),
                start,
                end,
                quantization.subdivision,
                steps_per_measure=per_measure,
                beat_steps=beat_steps,
                key_tonic=key_tonic,
                key_mode=key_mode,
                color=color,
            )
            active.append((voice_id, rendered))
        if not active:
            measure.append(note.Rest(quarterLength=_steps_to_ql(per_measure, quantization.subdivision)))
        elif len(active) == 1:
            # Still wrap in a Voice so PartStaff export keeps bass on 9+ and
            # treble on 1+, avoiding OSMD global voice-layer collisions.
            # Hide gap rests on print: OSMD often floats them above stem-down
            # lines even for a single voice on the staff.
            export_id = _export_voice_number(staff, active[0][0])
            voice = stream.Voice(id=str(export_id))
            for element in active[0][1]:
                if isinstance(element, note.Rest):
                    element.style.hideObjectOnPrint = True
                voice.append(element)
            measure.insert(0, voice)
        else:
            # Multi-voice gap rests are required for MusicXML timing, but OSMD
            # and MuseScore displace them far above/below the staff. Hide every
            # rest in a multi-voice measure on print; notes still show the lines.
            for voice_id, elements in active:
                export_id = _export_voice_number(staff, voice_id)
                voice = stream.Voice(id=str(export_id))
                for element in elements:
                    if isinstance(element, note.Rest):
                        element.style.hideObjectOnPrint = True
                    voice.append(element)
                measure.insert(0, voice)
        _beam_measure(measure)
        part.append(measure)
        start = end
        measure_index += 1


def _resolve_key(
    quantization: Quantization,
    key: tuple[str, str] | None,
    project_dir: Path | None,
) -> tuple[str, str] | None:
    if key is not None:
        return key
    if project_dir is not None:
        stored = load_analysis(project_dir / ANNOTATIONS_JSON)
        if stored is not None:
            return stored.key_tonic, stored.key_mode
    from baroque_transcribe.analysis import estimate_key

    tonic, mode, _confidence = estimate_key(quantization)
    return tonic, mode


def write_musicxml(
    path: Path,
    quantization: Quantization,
    *,
    staff_split: int = 60,
    title: str = "Piano transcription",
    style: str = "literal",
    key: tuple[str, str] | None = None,
    interpretation=None,
    notation_map: Path | None = None,
    pedals=None,
    colorize_voices: bool = False,
    preservation_report: Path | None = None,
    measure_number_offset: int = 0,
    voices_path: Path | None = None,
) -> Path:
    """Write a conventional two-staff piano MusicXML score.

    ``style`` is ``literal`` (event packing) or ``readable`` (interpreted
    voices and written durations). ``colorize_voices`` paints each interpreted
    voice. When ``voices_path`` is set with readable style, a coloured sibling
    file is also written from the same interpretation.
    """

    if style not in {"literal", "readable"}:
        raise NotationError(f"Notation style must be 'literal' or 'readable', not {style!r}.")
    from music21 import metadata

    resolved_key = _resolve_key(quantization, key, path.parent if path else None)
    key_tonic = resolved_key[0] if resolved_key else None
    key_mode = resolved_key[1] if resolved_key else None

    def build(colorize: bool) -> stream.Score:
        score = stream.Score()
        score.metadata = metadata.Metadata()
        score.metadata.title = title
        if colorize:
            movement = "Voice colours mark interpreted contrapuntal lines."
        elif style == "readable":
            movement = "Readable interpretation. Compare it with raw.mid before treating a note as certain."
        elif quantization.grid_mode == "follow":
            movement = "Tempo-following grid. Compare it with raw.mid before treating a note as certain."
        else:
            movement = "Quantized for notation. Compare it with raw.mid before treating a note as certain."
        score.metadata.movementName = movement
        score.insert(0, instrument.Piano())

        right = stream.PartStaff(id="P1-Staff1")
        left = stream.PartStaff(id="P1-Staff2")
        right.partName = "Piano"
        right.partAbbreviation = "Pno."
        left.partName = "Piano"
        left.partAbbreviation = "Pno."

        if style == "readable":
            _fill_part_readable(
                right,
                resolved.events_for_staff(1),
                quantization,
                clef.TrebleClef(),
                staff=1,
                key_tonic=key_tonic,
                key_mode=key_mode,
                colorize_voices=colorize,
                include_tempo=True,
                measure_number_offset=measure_number_offset,
            )
            _fill_part_readable(
                left,
                resolved.events_for_staff(2),
                quantization,
                clef.BassClef(),
                staff=2,
                key_tonic=key_tonic,
                key_mode=key_mode,
                colorize_voices=colorize,
                include_tempo=False,
                measure_number_offset=measure_number_offset,
            )
        else:
            events = _collapse(quantization.grid)
            _fill_part(
                right,
                _staff_events(events, staff_split, True),
                quantization,
                clef.TrebleClef(),
                key_tonic=key_tonic,
                key_mode=key_mode,
                include_tempo=True,
                measure_number_offset=measure_number_offset,
            )
            _fill_part(
                left,
                _staff_events(events, staff_split, False),
                quantization,
                clef.BassClef(),
                key_tonic=key_tonic,
                key_mode=key_mode,
                include_tempo=False,
                measure_number_offset=measure_number_offset,
            )

        group = layout.StaffGroup(
            [right, left],
            name="Piano",
            abbreviation="Pno.",
            symbol="brace",
            barTogether=True,
        )
        score.insert(0, group)
        score.append(right)
        score.append(left)
        return score

    resolved = interpretation
    if style == "readable":
        from baroque_transcribe.engrave import (
            interpret_quantization,
            validate_note_preservation,
            write_notation_map,
            write_preservation_report,
        )

        resolved = interpretation or interpret_quantization(
            quantization,
            staff_split=staff_split,
            settings=quantization.settings,
            pedals=pedals,
        )
        report = validate_note_preservation(quantization, resolved)
        if preservation_report is not None:
            write_preservation_report(preservation_report, report)
        if not report.ok:
            raise NotationError(
                f"Note preservation failed: {report.matched}/{report.total_source} matched, "
                f"{len(report.unmatched)} unmatched."
            )
        if notation_map is not None:
            write_notation_map(notation_map, resolved, quantization)
    elif colorize_voices:
        raise NotationError("Voice colouring requires style='readable'.")

    def export(score_obj: stream.Score, dest: Path) -> Path:
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            written = score_obj.write("musicxml", fp=str(dest))
        except Exception as exc:
            raise NotationError(f"MusicXML export failed: {exc}") from exc
        written_path = Path(str(written)) if written else dest
        if written_path != dest and written_path.exists():
            written_path.replace(dest)
        if not dest.exists():
            raise NotationError(f"music21 did not write {dest}.")
        if style == "readable":
            text = dest.read_text(encoding="utf-8")
            converted = _hidden_rests_to_forward(text)
            if converted != text:
                dest.write_text(converted, encoding="utf-8")
        return dest

    export(build(colorize_voices), path)
    if style == "readable" and voices_path is not None:
        export(build(True), voices_path)
    return path


_NOTATION_LIBRARY = "opensheetmusicdisplay.min.js"


def _notation_library() -> Path:
    path = Path(__file__).resolve().parent / "vendor" / _NOTATION_LIBRARY
    if not path.is_file():
        raise NotationError(f"Notation library is missing at {path}.")
    return path


def musicxml_for_midi(midi_path: Path) -> Path:
    """Find or write the MusicXML for a MIDI file without running the piano model.

    A finished project already has ``score.musicxml`` beside ``quantized.mid``.
    That score is returned unchanged. Any other MIDI file is read once and
    written to a ``.musicxml`` file next to it.
    """

    midi_path = midi_path.expanduser().resolve()
    if not midi_path.is_file():
        raise NotationError(f"MIDI file not found: {midi_path}")
    if midi_path.suffix.lower() not in {".mid", ".midi"}:
        raise NotationError("Choose a .mid or .midi file.")
    project_score = midi_path.parent / SCORE_XML
    if project_score.is_file() and (midi_path.parent / NOTES_JSON).is_file():
        return project_score

    dest = midi_path.with_suffix(".musicxml")
    if dest.is_file() and dest.stat().st_mtime >= midi_path.stat().st_mtime:
        return dest
    from music21 import converter

    try:
        parsed = converter.parse(str(midi_path))
        written = parsed.write("musicxml", fp=str(dest))
    except Exception as exc:
        raise NotationError(f"Could not read {midi_path.name} as notation: {exc}") from exc
    written_path = Path(str(written)) if written else dest
    if written_path != dest and written_path.exists():
        written_path.replace(dest)
    if not dest.exists():
        raise NotationError(f"music21 did not write {dest}.")
    return dest


def write_notation_page(
    score_path: Path,
    dest: Path,
    *,
    title: str | None = None,
    note: str | None = None,
    home_url: str | None = None,
    harmony: dict | None = None,
    facts: list[dict[str, str]] | None = None,
    folder: str | None = None,
    tools_url: str | None = None,
    playback_url: str | None = None,
    voices_score_path: Path | None = None,
    time_offset_seconds: float = 0.0,
) -> Path:
    """Write an HTML page that draws the two-staff score in a browser tab.

    The page embeds ``score.musicxml`` and a local copy of the drawing
    library. A ``file://`` tab cannot rely on a network script: if that
    script does not load, the page dies with ``opensheetmusicdisplay is not
    defined``. The upper staff is the right-hand track from ``quantized.mid``
    and the lower staff is the left-hand track.

    ``home_url`` adds a link back to the library page. ``?embed=1`` on the
    page address hides the heading, for a page that frames this one.

    The rest of the heading is for a score that belongs to a project.
    ``facts`` are the chips from ``library.run_facts`` and ``folder`` is the
    project folder, shown under the title. ``playback_url`` links to the
    video with the score following it, and ``tools_url`` to the page with the
    audio, the note table, the harmony editor, and the files.
    """

    if not score_path.is_file():
        raise NotationError(f"No score to display at {score_path}.")
    library = _notation_library()
    dest.parent.mkdir(parents=True, exist_ok=True)
    local_library = dest.parent / _NOTATION_LIBRARY
    if library.resolve() != local_library.resolve():
        shutil.copyfile(library, local_library)
    xml = score_path.read_text(encoding="utf-8")
    # Keep a literal "</script>" in the MusicXML from closing the HTML script early.
    payload = json.dumps(xml).replace("<", "\\u003c")
    voices_payload = "null"
    if voices_score_path is None:
        candidate = score_path.parent / "score.voices.musicxml"
        if candidate.is_file() and score_path.name == SCORE_XML:
            voices_score_path = candidate
    if voices_score_path is not None and voices_score_path.is_file():
        voices_payload = json.dumps(voices_score_path.read_text(encoding="utf-8")).replace("<", "\\u003c")
    try:
        measure_times = json.dumps(
            measure_time_labels(score_path, time_offset_seconds=time_offset_seconds)
        )
    except Exception:
        measure_times = "{}"
    if harmony is None:
        stored = load_analysis(score_path.parent / ANNOTATIONS_JSON)
        harmony = stored.to_dict() if stored is not None else None
    harmony_payload = json.dumps(harmony).replace("<", "\\u003c")
    heading = escape(title or score_path.stem)
    blurb = escape(
        note
        or "Upper staff: right-hand track. Lower staff: left-hand track. "
        "The split follows pitch, with middle C on the upper staff."
    )
    home_link = "".join(
        f'<a class="home" href="{escape(address, quote=True)}">{label}</a>'
        for address, label in (
            (home_url, "&larr; Library"),
            (playback_url, "Play with score"),
            (tools_url, "Edit &amp; files"),
        )
        if address
    )
    chips = "".join(
        '<span class="fact">'
        + escape(fact.get("before", ""))
        + (f'<b>{escape(fact["value"])}</b>' if fact.get("value") else "")
        + escape(fact.get("after", ""))
        + "</span>"
        for fact in facts or []
    )
    details = (f'<div class="facts">{chips}</div>' if chips else "") + (
        f'<div class="path">{escape(folder)}</div>' if folder else ""
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{heading}</title>
  <style>
    body {{
      margin: 0; color: #1c1917; background: #faf7f2;
      font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }}
    header {{ padding: 1rem 1.5rem 0.75rem; border-bottom: 1px solid #e7e2da; background: #fffdf9; }}
    .home {{ display: inline-block; margin-bottom: 0.5rem; font-size: 0.9rem; color: #7c3a07; text-decoration: none; font-weight: 500; }}
    .home:hover {{ text-decoration: underline; }}
    .home + .home {{ margin-left: 1.1rem; }}
    .facts {{ display: flex; flex-wrap: wrap; gap: 0.35rem 0.5rem; margin: 0.15rem 0 0.4rem; }}
    .fact {{ font-size: 0.78rem; padding: 0.14rem 0.55rem; border-radius: 999px; white-space: nowrap; background: #fdf3e2; border: 1px solid #e7e2da; font-variant-numeric: tabular-nums; }}
    .fact b {{ color: #7c3a07; font-weight: 600; }}
    .path {{ margin-bottom: 0.45rem; color: #6b6560; font-size: 0.78rem; font-family: ui-monospace, Consolas, monospace; word-break: break-all; }}
    h1 {{ font-family: Georgia, "Iowan Old Style", "Times New Roman", serif; font-size: 1.4rem; font-weight: 600; margin: 0 0 0.35rem; }}
    p {{ margin: 0; max-width: 42rem; line-height: 1.45; color: #6b6560; font-size: 0.92rem; }}
    body.embed header {{ display: none; }}
    #score {{ padding: 0.5rem 1rem 2rem; background: white; }}
    #score svg {{ overflow: visible; }}
    .score-controls {{ margin-top: 0.5rem; display: flex; flex-wrap: wrap; gap: 1rem; font-size: 0.9rem; color: #44403c; }}
    .score-controls label {{ display: flex; gap: 0.3rem; align-items: center; cursor: pointer; }}
    .score-controls label[hidden] {{ display: none !important; }}
    .voice-legend {{ margin-top: 0.35rem; font-size: 0.8rem; color: #6b6560; }}
    .voice-legend[hidden] {{ display: none !important; }}
    .harmony-label {{ position: absolute; font-size: 11px; font-weight: 700; color: #7c3a07; background: rgba(255,253,249,0.92); padding: 0 2px; pointer-events: none; white-space: nowrap; z-index: 4; }}
    .harmony-label.uncertain {{ color: #b45309; font-weight: 400; }}
    .harmony-seq {{ position: absolute; border-top: 2px solid #a16207; color: #854d0e; font-size: 10px; font-style: italic; padding-top: 1px; pointer-events: none; white-space: nowrap; overflow: hidden; z-index: 4; }}
  </style>
</head>
<body>
  <header>
    {home_link}
    <h1>{heading}</h1>
    {details}<p>{blurb}</p>
    <div class="score-controls" id="score-controls">
      <label id="harmony-toggle" hidden><input type="checkbox" id="show-harmony" checked> Harmony</label>
      <label id="lead-toggle" hidden><input type="checkbox" id="lead-sheet"> Lead-sheet chord symbols</label>
      <label id="color-toggle" hidden><input type="checkbox" id="color-voices"> Color voices</label>
    </div>
    <div class="voice-legend" id="voice-legend" hidden>Colours mark interpreted contrapuntal voices, not hands.</div>
  </header>
  <div id="score"></div>
  <script src="{_NOTATION_LIBRARY}"></script>
  <script>
    if (new URLSearchParams(window.location.search).get("embed") === "1") document.body.classList.add("embed");
    const xmlPlain = {payload};
    const xmlVoices = {voices_payload};
    const measureTimes = {measure_times};
    const harmony = {harmony_payload};
    const colorStorageKey = "baroque-transcribe.color-voices";
    if (harmony) {{
      document.getElementById("harmony-toggle").hidden = false;
      document.getElementById("lead-toggle").hidden = false;
    }}
    if (xmlVoices) document.getElementById("color-toggle").hidden = false;
    const library = window.opensheetmusicdisplay;
    function stampMeasureTimes(osmd) {{
      const pages = (osmd.GraphicSheet && osmd.GraphicSheet.MusicPages) || [];
      for (const page of pages) {{
        for (const system of page.MusicSystems || []) {{
          for (const label of system.MeasureNumberLabels || []) {{
            const node = label.SVGNode;
            if (!node) continue;
            const text = node.tagName === "text" ? node : node.querySelector("text, tspan");
            if (!text) continue;
            const raw = (text.textContent || "").trim();
            const stamp = measureTimes[raw];
            if (!stamp || raw.indexOf("(") !== -1) continue;
            text.textContent = raw + " " + stamp;
          }}
        }}
      }}
    }}
    if (!library) {{
      document.getElementById("score").textContent = "The notation library did not load from {_NOTATION_LIBRARY}.";
    }} else {{
    const osmd = new library.OpenSheetMusicDisplay("score", {{
      autoResize: true,
      backend: "svg",
      drawTitle: false,
      drawComposer: false,
      drawPartNames: true,
      drawMeasureNumbers: true,
      measureNumberInterval: 1
    }});
    function measureBoxes() {{
      const boxes = [];
      const systems = [];
      const list = (osmd.GraphicSheet && osmd.GraphicSheet.MeasureList) || [];
      const scale = 10 * (osmd.Zoom || 1);
      for (let index = 0; index < list.length; index += 1) {{
        const staffs = list[index] || [];
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        let pageNumber = null;
        let systemKey = "";
        let measureNumber = index + 1;
        for (let staff = 0; staff < staffs.length; staff += 1) {{
          const measure = staffs[staff];
          if (!measure || !measure.PositionAndShape) continue;
          const source = measure.parentSourceMeasure;
          if (source && source.MeasureNumber) measureNumber = source.MeasureNumber;
          const box = measure.PositionAndShape;
          const x = box.AbsolutePosition.x + box.BorderLeft;
          const y = box.AbsolutePosition.y + box.BorderTop;
          const right = x + box.Size.width;
          const bottom = y + box.Size.height;
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (right > maxX) maxX = right;
          if (bottom > maxY) maxY = bottom;
          const system = measure.ParentStaffLine && measure.ParentStaffLine.ParentMusicSystem;
          const page = system && system.Parent;
          if (page && page.PageNumber) pageNumber = page.PageNumber;
          if (system) {{
            let systemIndex = systems.indexOf(system);
            if (systemIndex === -1) {{ systems.push(system); systemIndex = systems.length - 1; }}
            systemKey = (pageNumber || 1) + ":" + systemIndex;
          }}
        }}
        if (!isFinite(minX) || maxX <= minX || maxY <= minY) continue;
        const pageEl = document.getElementById("osmdCanvasPage" + pageNumber) || document.getElementById("score");
        boxes.push({{ measure: measureNumber, pageEl: pageEl, systemKey: systemKey, minX: minX, maxX: maxX, minY: minY, maxY: maxY, scale: scale }});
      }}
      return boxes;
    }}
    function drawHarmony() {{
      document.querySelectorAll(".harmony-label, .harmony-seq").forEach(function (node) {{ node.remove(); }});
      if (!harmony || !document.getElementById("show-harmony").checked) return;
      const leadSheet = document.getElementById("lead-sheet").checked;
      const boxes = measureBoxes();
      (harmony.chords || []).forEach(function (chord) {{
        const box = boxes.find(function (item) {{ return item.measure === chord.measure; }});
        if (!box || !box.pageEl) return;
        box.pageEl.style.position = "relative";
        const label = document.createElement("div");
        label.className = "harmony-label" + (chord.confidence < 0.5 ? " uncertain" : "");
        label.textContent = leadSheet ? chord.symbol : chord.roman;
        label.title = (leadSheet ? chord.roman : chord.symbol) + " - " + Math.round((chord.confidence || 0) * 100) + "% confidence";
        label.style.left = ((box.minX + chord.measure_position * (box.maxX - box.minX)) * box.scale) + "px";
        label.style.top = (box.maxY * box.scale + 5) + "px";
        box.pageEl.appendChild(label);
      }});
      (harmony.sequences || []).forEach(function (sequence) {{
        const bySystem = {{}};
        boxes.filter(function (box) {{ return box.measure >= sequence.start_measure && box.measure <= sequence.end_measure; }})
          .forEach(function (box) {{
            const key = box.systemKey || ((box.pageEl.id || "score") + ":" + Math.round(box.maxY * 2));
            if (!bySystem[key]) bySystem[key] = [];
            bySystem[key].push(box);
          }});
        Object.values(bySystem).forEach(function (group) {{
          group.sort(function (a, b) {{ return a.minX - b.minX; }});
          const first = group[0];
          const last = group[group.length - 1];
          const baseY = Math.max.apply(null, group.map(function (box) {{ return box.maxY; }}));
          first.pageEl.style.position = "relative";
          const span = document.createElement("div");
          span.className = "harmony-seq";
          span.textContent = sequence.description;
          span.title = sequence.description + " - measures " + sequence.start_measure + "-" + sequence.end_measure;
          span.style.left = (first.minX * first.scale) + "px";
          span.style.width = Math.max(30, (last.maxX - first.minX) * first.scale) + "px";
          span.style.top = (baseY * first.scale + 20) + "px";
          group[0].pageEl.appendChild(span);
        }});
      }});
    }}
    document.getElementById("show-harmony").addEventListener("change", drawHarmony);
    document.getElementById("lead-sheet").addEventListener("change", drawHarmony);
    const drawScore = osmd.render.bind(osmd);
    osmd.render = function () {{
      const drawn = drawScore();
      stampMeasureTimes(osmd);
      drawHarmony();
      return drawn;
    }};
    function preferredXml() {{
      const box = document.getElementById("color-voices");
      return (box && box.checked && xmlVoices) ? xmlVoices : xmlPlain;
    }}
    function loadScore() {{
      return osmd.load(preferredXml()).then(function () {{ osmd.render(); }});
    }}
    if (xmlVoices) {{
      const box = document.getElementById("color-voices");
      try {{ box.checked = window.localStorage.getItem(colorStorageKey) === "1"; }} catch (error) {{}}
      document.getElementById("voice-legend").hidden = !box.checked;
      box.addEventListener("change", function () {{
        try {{ window.localStorage.setItem(colorStorageKey, box.checked ? "1" : "0"); }} catch (error) {{}}
        document.getElementById("voice-legend").hidden = !box.checked;
        loadScore().catch(function (error) {{
          document.getElementById("score").textContent = "Could not draw the score. " + error;
        }});
      }});
    }}
    loadScore().catch(function (error) {{
      document.getElementById("score").textContent = "Could not draw the score. " + error;
    }});
    }}
  </script>
</body>
</html>
""",
        encoding="utf-8",
    )
    return dest
