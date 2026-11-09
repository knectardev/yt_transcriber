"""Harmonic analysis over the quantized notation grid.

The transcription pipeline stops at notes, pedal, and a pitch-based staff
split. This module is a separate pass that reads the quantized grid and
writes ``annotations.json`` beside ``notes.json``. It never rewrites the
notes or the raw MIDI.

Voice separation, imitation, and cadence detection are still reserved for
a later phase. Harmonic progression and sequence detection are implemented
here as an automatic draft: every label carries a confidence and can be
edited by hand, and edited labels survive regeneration.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

from baroque_transcribe.quantize import Quantization
from baroque_transcribe.schema import Transcription

SCHEMA_VERSION = 1


class AnalysisError(ValueError):
    pass


@dataclass
class Voice:
    notes: list = field(default_factory=list)
    label: str = ""


@dataclass
class AnalysisAnnotation:
    kind: str
    start: float
    end: float
    description: str


# --------------------------------------------------------------------------
# Harmonic annotation model
# --------------------------------------------------------------------------


@dataclass
class ChordEvent:
    measure: int
    beat: float
    measure_position: float
    start_step: int
    end_step: int
    root_pc: int
    root_name: str
    quality: str
    bass_pc: int
    bass_name: str
    roman: str
    symbol: str
    inversion: int
    confidence: float
    edited: bool = False
    edited_fields: list[str] = field(default_factory=list)


@dataclass
class SequenceEvent:
    start_measure: int
    end_measure: int
    kind: str
    description: str
    repetitions: int
    confidence: float
    edited: bool = False


@dataclass
class HarmonicAnalysis:
    key_tonic: str
    key_mode: str
    key_source: str
    key_confidence: float
    chords: list[ChordEvent] = field(default_factory=list)
    sequences: list[SequenceEvent] = field(default_factory=list)
    grid: dict = field(default_factory=dict)
    notes_fingerprint: str = ""
    version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "key": {
                "tonic": self.key_tonic,
                "mode": self.key_mode,
                "source": self.key_source,
                "confidence": round(float(self.key_confidence), 4),
            },
            "grid": dict(self.grid),
            "notes_fingerprint": self.notes_fingerprint,
            "chords": [asdict(item) for item in self.chords],
            "sequences": [asdict(item) for item in self.sequences],
        }


def analysis_from_dict(data: dict) -> HarmonicAnalysis:
    if not isinstance(data, dict):
        raise AnalysisError("annotations.json is not an object.")
    version = int(data.get("version", SCHEMA_VERSION))
    if version != SCHEMA_VERSION:
        raise AnalysisError(
            f"annotations.json uses schema version {version}; this program reads version {SCHEMA_VERSION}."
        )
    key = data.get("key") if isinstance(data.get("key"), dict) else {}
    chords = []
    for item in data.get("chords") or []:
        chords.append(
            ChordEvent(
                measure=int(item["measure"]),
                beat=float(item.get("beat") or 0.0),
                measure_position=float(item.get("measure_position") or 0.0),
                start_step=int(item["start_step"]),
                end_step=int(item["end_step"]),
                root_pc=int(item["root_pc"]) % 12,
                root_name=str(item.get("root_name") or ""),
                quality=str(item.get("quality") or "major"),
                bass_pc=int(item["bass_pc"]) % 12,
                bass_name=str(item.get("bass_name") or ""),
                roman=str(item.get("roman") or ""),
                symbol=str(item.get("symbol") or ""),
                inversion=int(item.get("inversion") or 0),
                confidence=float(item.get("confidence") or 0.0),
                edited=bool(item.get("edited")),
                edited_fields=[str(name) for name in item.get("edited_fields") or []],
            )
        )
    sequences = []
    for item in data.get("sequences") or []:
        sequences.append(
            SequenceEvent(
                start_measure=int(item["start_measure"]),
                end_measure=int(item["end_measure"]),
                kind=str(item.get("kind") or "sequence"),
                description=str(item.get("description") or ""),
                repetitions=int(item.get("repetitions") or 0),
                confidence=float(item.get("confidence") or 0.0),
                edited=bool(item.get("edited")),
            )
        )
    return HarmonicAnalysis(
        key_tonic=str(key.get("tonic") or "C"),
        key_mode=str(key.get("mode") or "major"),
        key_source=str(key.get("source") or "automatic"),
        key_confidence=float(key.get("confidence") or 0.0),
        chords=chords,
        sequences=sequences,
        grid=dict(data.get("grid") or {}),
        notes_fingerprint=str(data.get("notes_fingerprint") or ""),
        version=version,
    )


def read_analysis(path: Path) -> HarmonicAnalysis:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnalysisError(f"Could not read {path.name}: {exc}") from exc
    return analysis_from_dict(data)


def load_analysis(path: Path) -> HarmonicAnalysis | None:
    """Analysis for display: missing or damaged annotations simply omit the layer."""

    if not path.is_file():
        return None
    try:
        return read_analysis(path)
    except (OSError, AnalysisError, KeyError, TypeError, ValueError):
        return None


def write_analysis(path: Path, analysis: HarmonicAnalysis) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(analysis.to_dict(), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def notes_fingerprint(quantization: Quantization) -> str:
    payload = [
        [item.pitch, round(item.source_onset, 5), round(item.source_offset, 5), item.velocity]
        for item in sorted(quantization.grid, key=lambda note: note.source_index)
    ]
    return hashlib.sha256(json.dumps(payload).encode("utf-8")).hexdigest()[:16]


def analysis_is_stale(analysis: HarmonicAnalysis, quantization: Quantization) -> bool:
    if analysis.notes_fingerprint != notes_fingerprint(quantization):
        return True
    grid = analysis.grid
    return (
        float(grid.get("bpm") or 0) != float(quantization.bpm)
        or int(grid.get("numerator") or 0) != quantization.numerator
        or int(grid.get("denominator") or 0) != quantization.denominator
        or int(grid.get("subdivision") or 0) != quantization.subdivision
    )


def preserve_edited(new: HarmonicAnalysis, old: HarmonicAnalysis | None) -> HarmonicAnalysis:
    """Carry hand edits from ``old`` into a regenerated draft."""

    if old is None:
        return new
    old_chords = {(item.start_step, item.end_step): item for item in old.chords if item.edited}
    for chord in new.chords:
        previous = old_chords.get((chord.start_step, chord.end_step))
        if previous is None:
            continue
        fields = previous.edited_fields or ["roman", "symbol"]
        for name in fields:
            if hasattr(chord, name):
                setattr(chord, name, getattr(previous, name))
        chord.edited = True
        chord.edited_fields = list(fields)
    old_sequences = {(item.start_measure, item.end_measure): item for item in old.sequences if item.edited}
    for sequence in new.sequences:
        previous = old_sequences.get((sequence.start_measure, sequence.end_measure))
        if previous is not None:
            sequence.description = previous.description
            sequence.kind = previous.kind
            sequence.edited = True
    return new


# --------------------------------------------------------------------------
# Key and chord detection
# --------------------------------------------------------------------------

_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_NAME_TO_PC = {}
for _index, _name in enumerate(_SHARP_NAMES):
    _NAME_TO_PC[_name] = _index
for _index, _name in enumerate(_FLAT_NAMES):
    _NAME_TO_PC[_name] = _index
_NAME_TO_PC.update({"B#": 0, "E#": 5, "Cb": 11, "Fb": 4})

_MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
_MAJOR_STEPS = [0, 2, 4, 5, 7, 9, 11]
_MINOR_STEPS = [0, 2, 3, 5, 7, 8, 10]
_NUMERALS = ["I", "II", "III", "IV", "V", "VI", "VII"]

_QUALITIES: dict[str, tuple[set[int], str]] = {
    "major": ({0, 4, 7}, ""),
    "minor": ({0, 3, 7}, "m"),
    "diminished": ({0, 3, 6}, "dim"),
    "augmented": ({0, 4, 8}, "aug"),
    "dominant7": ({0, 4, 7, 10}, "7"),
    "major7": ({0, 4, 7, 11}, "maj7"),
    "minor7": ({0, 3, 7, 10}, "m7"),
    "halfdiminished7": ({0, 3, 6, 10}, "m7b5"),
    "diminished7": ({0, 3, 6, 9}, "dim7"),
    "minorMajor7": ({0, 3, 7, 11}, "mMaj7"),
    "major6": ({0, 4, 7, 9}, "6"),
    "minor6": ({0, 3, 7, 9}, "m6"),
    "sus4": ({0, 5, 7}, "sus4"),
    "sus2": ({0, 2, 7}, "sus2"),
}

CHORD_QUALITIES = tuple(_QUALITIES)


def parse_tonic(text: str) -> tuple[str, int]:
    name = str(text or "").strip()
    if not name:
        raise AnalysisError("Key tonic is empty.")
    name = name[0].upper() + name[1:]
    if name not in _NAME_TO_PC:
        raise AnalysisError(f"Could not read key tonic {text!r}. Use a name such as C, F#, or Bb.")
    return name, _NAME_TO_PC[name]


def _scale_pcs(tonic_pc: int, mode: str) -> list[int]:
    steps = _MAJOR_STEPS if mode == "major" else _MINOR_STEPS
    return [(tonic_pc + step) % 12 for step in steps]


def _scale_names(tonic: str, mode: str) -> dict[int, str]:
    tonic_pc = _NAME_TO_PC[tonic]
    letters = "ABCDEFG"
    start = letters.index(tonic[0].upper())
    steps = _MAJOR_STEPS if mode == "major" else _MINOR_STEPS
    names: dict[int, str] = {}
    for degree, step in enumerate(steps):
        letter = letters[(start + degree) % 7]
        target = (tonic_pc + step) % 12
        natural = _NAME_TO_PC[letter]
        diff = (target - natural + 6) % 12 - 6
        accidental = {0: "", 1: "#", -1: "b", 2: "##", -2: "bb"}.get(diff, "")
        names[target] = f"{letter}{accidental}"
    return names


def _pc_name(pc: int, tonic: str, mode: str) -> str:
    scale = _scale_names(tonic, mode)
    if pc in scale:
        return scale[pc]
    flat_keys = {"F", "Bb", "Eb", "Ab", "Db", "Gb", "Dm", "Gm", "Cm", "Fm"}
    use_flats = tonic in flat_keys or tonic.endswith("b")
    return (_FLAT_NAMES if use_flats else _SHARP_NAMES)[pc % 12]


def estimate_key(quantization: Quantization) -> tuple[str, str, float]:
    """Estimate tonic and mode from duration-weighted pitch classes."""

    weights = [0.0] * 12
    for item in quantization.grid:
        weight = max(1, item.offset_step - item.onset_step) * (0.6 + 0.4 * item.velocity / 127.0)
        weights[item.pitch % 12] += weight
    if sum(weights) <= 0:
        return "C", "major", 0.0
    total = sum(weights)
    vector = [value / total for value in weights]

    def correlation(profile: list[float]) -> float:
        mean_v = sum(vector) / 12
        mean_p = sum(profile) / 12
        numerator = sum((vector[i] - mean_v) * (profile[i] - mean_p) for i in range(12))
        denom_v = math.sqrt(sum((value - mean_v) ** 2 for value in vector))
        denom_p = math.sqrt(sum((value - mean_p) ** 2 for value in profile))
        return numerator / (denom_v * denom_p) if denom_v and denom_p else 0.0

    scored: list[tuple[float, int, str]] = []
    for tonic_pc in range(12):
        for mode, profile in (("major", _MAJOR_PROFILE), ("minor", _MINOR_PROFILE)):
            rotated = [profile[(pc - tonic_pc) % 12] for pc in range(12)]
            scored.append((correlation(rotated), tonic_pc, mode))
    scored.sort(reverse=True)
    best, second = scored[0], scored[1]
    names = _SHARP_NAMES
    tonic = names[best[1]]
    # Present flat tonics with their usual spelling when the pitch class has one.
    if best[1] in (1, 3, 6, 8, 10) and best[2] == "major" and best[1] in (1, 6):
        tonic = _FLAT_NAMES[best[1]]
    confidence = max(0.0, min(1.0, 0.35 + 0.45 * max(0.0, best[0]) + min(0.2, max(0.0, best[0] - second[0]) * 2.0)))
    return tonic, best[2], round(confidence, 4)


def _segment_weights(quantization: Quantization, start: int, end: int) -> tuple[dict[int, float], int | None]:
    weights: dict[int, float] = {}
    bass_candidates: list[tuple[int, float]] = []
    for item in quantization.grid:
        overlap = min(item.offset_step, end) - max(item.onset_step, start)
        if overlap <= 0:
            continue
        weight = overlap * (0.6 + 0.4 * item.velocity / 127.0)
        weights[item.pitch % 12] = weights.get(item.pitch % 12, 0.0) + weight
        if item.onset_step <= start or item.onset_step < start + max(1, (end - start) // 4):
            bass_candidates.append((item.pitch, weight))
    bass_pc = None
    if bass_candidates:
        lowest = min(pitch for pitch, _weight in bass_candidates)
        bass_pc = lowest % 12
    elif weights:
        # No attack near the start: the lowest prominent pitch class still
        # sustaining is the best available bass estimate.
        bass_pc = min(weights, key=lambda pc: (-weights[pc], pc))
    return weights, bass_pc


@dataclass
class _Candidate:
    root_pc: int
    quality: str
    bass_pc: int
    confidence: float


def _best_chord(weights: dict[int, float], bass_pc: int | None) -> _Candidate | None:
    total = sum(weights.values())
    if total <= 0 or bass_pc is None:
        return None
    best: _Candidate | None = None
    for root in range(12):
        for quality, (intervals, _suffix) in _QUALITIES.items():
            tones = {(root + interval) % 12 for interval in intervals}
            matched = sum(weights.get(pc, 0.0) for pc in tones)
            unexplained = sum(
                value for pc, value in weights.items() if pc not in tones and value / total > 0.08
            )
            missing = sum(1 for pc in tones if weights.get(pc, 0.0) / total < 0.03)
            score = matched / total - 0.7 * unexplained / total - 0.06 * missing
            if bass_pc == root:
                score += 0.12
            elif bass_pc in tones:
                score += 0.05
            else:
                score -= 0.08
            score -= 0.012 * max(0, len(tones) - 3)
            confidence = max(0.0, min(1.0, (score - 0.25) / 0.85))
            candidate = _Candidate(root, quality, bass_pc, round(confidence, 4))
            if best is None or candidate.confidence > best.confidence:
                best = candidate
    return best


def _inversion(root_pc: int, bass_pc: int, quality: str) -> int:
    intervals = _QUALITIES[quality][0]
    ordered = sorted(intervals)
    relative = (bass_pc - root_pc) % 12
    return ordered.index(relative) if relative in ordered else 0


def _roman(root_pc: int, quality: str, bass_pc: int, tonic: str, mode: str) -> str:
    tonic_pc = _NAME_TO_PC[tonic]
    scale = _scale_pcs(tonic_pc, mode)
    inversion = _inversion(root_pc, bass_pc, quality)
    # Six-chords carry four pitch classes but are not seventh chords.
    seventh = quality in {
        "dominant7", "major7", "minor7", "halfdiminished7", "diminished7", "minorMajor7",
    }
    suffix = ""
    if seventh:
        suffix = ("7", "6/5", "4/3", "4/2")[inversion]
    elif inversion == 1:
        suffix = "6"
    elif inversion == 2:
        suffix = "6/4"
    quality_mark = ""
    if quality in {"diminished", "diminished7"}:
        quality_mark = "°"
    elif quality == "halfdiminished7":
        quality_mark = "ø"
    elif quality == "augmented":
        quality_mark = "+"

    def base_numeral(degree_index: int, accidental: str = "") -> str:
        numeral = _NUMERALS[degree_index]
        if quality in {"minor", "minor7", "minor6", "minorMajor7", "diminished", "diminished7", "halfdiminished7"}:
            numeral = numeral.lower()
        return f"{accidental}{numeral}{quality_mark}{suffix}"

    if root_pc in scale:
        return base_numeral(scale.index(root_pc))
    # Secondary dominant: a major/dominant chord a fifth above a diatonic root.
    if quality in {"major", "dominant7"}:
        target_pc = (root_pc - 7) % 12
        if target_pc in scale and target_pc != tonic_pc:
            target = _NUMERALS[scale.index(target_pc)]
            return f"V{suffix}/{target}"
    # Chromatic root: borrow the nearest scale degree and mark the accidental.
    best_degree = min(range(7), key=lambda index: min((root_pc - scale[index]) % 12, (scale[index] - root_pc) % 12))
    diff = (root_pc - scale[best_degree] + 6) % 12 - 6
    accidental = "b" if diff < 0 else "#" if diff > 0 else ""
    return base_numeral(best_degree, accidental)


def _symbol(root_pc: int, quality: str, bass_pc: int, tonic: str, mode: str) -> str:
    root_name = _pc_name(root_pc, tonic, mode)
    suffix = _QUALITIES[quality][1]
    symbol = f"{root_name}{suffix}"
    if bass_pc != root_pc:
        symbol += f"/{_pc_name(bass_pc, tonic, mode)}"
    return symbol


def _chord_event(
    quantization: Quantization,
    start: int,
    end: int,
    measure: int,
    measure_start: int,
    candidate: _Candidate,
    tonic: str,
    mode: str,
) -> ChordEvent:
    per_measure = quantization.steps_per_measure
    beat_steps = max(1, round(quantization.subdivision / quantization.denominator))
    return ChordEvent(
        measure=measure,
        beat=round((start - measure_start) / beat_steps, 3),
        measure_position=round((start - measure_start) / per_measure, 4) if per_measure else 0.0,
        start_step=start,
        end_step=end,
        root_pc=candidate.root_pc,
        root_name=_pc_name(candidate.root_pc, tonic, mode),
        quality=candidate.quality,
        bass_pc=candidate.bass_pc,
        bass_name=_pc_name(candidate.bass_pc, tonic, mode),
        roman=_roman(candidate.root_pc, candidate.quality, candidate.bass_pc, tonic, mode),
        symbol=_symbol(candidate.root_pc, candidate.quality, candidate.bass_pc, tonic, mode),
        inversion=_inversion(candidate.root_pc, candidate.bass_pc, candidate.quality),
        confidence=candidate.confidence,
    )


def _detect_measure_chords(
    quantization: Quantization,
    measure_start: int,
    measure_end: int,
    measure_number: int,
    tonic: str,
    mode: str,
) -> list[ChordEvent]:
    beat_steps = max(1, round(quantization.subdivision / quantization.denominator))

    def candidate_for(start: int, end: int) -> _Candidate | None:
        weights, bass_pc = _segment_weights(quantization, start, end)
        return _best_chord(weights, bass_pc)

    def build(start: int, end: int) -> list[ChordEvent]:
        whole = candidate_for(start, end)
        if whole is None:
            return []
        if end - start <= beat_steps:
            return [_chord_event(quantization, start, end, measure_number, measure_start, whole, tonic, mode)]
        best_split: tuple[float, int, _Candidate, _Candidate] | None = None
        for split in range(start + beat_steps, end, beat_steps):
            left = candidate_for(start, split)
            right = candidate_for(split, end)
            if left is None or right is None:
                continue
            if (left.root_pc, left.quality) == (right.root_pc, right.quality):
                continue
            if left.confidence < 0.35 or right.confidence < 0.35:
                continue
            combined = (left.confidence * (split - start) + right.confidence * (end - split)) / (end - start)
            improvement = combined - whole.confidence
            if best_split is None or improvement > best_split[0]:
                best_split = (improvement, split, left, right)
        if best_split is not None and best_split[0] > 0.08:
            split = best_split[1]
            return build(start, split) + build(split, end)
        return [_chord_event(quantization, start, end, measure_number, measure_start, whole, tonic, mode)]

    return build(measure_start, measure_end)


def detect_chords(
    quantization: Quantization,
    tonic: str,
    mode: str,
) -> list[ChordEvent]:
    from baroque_transcribe.sync_map import printed_measure_span

    first, count = printed_measure_span(quantization)
    chords: list[ChordEvent] = []
    per_measure = quantization.steps_per_measure
    for index in range(count):
        start = first + index * per_measure
        chords.extend(
            _detect_measure_chords(quantization, start, start + per_measure, index + 1, tonic, mode)
        )
    return chords


def detect_sequences(chords: list[ChordEvent], tonic: str, mode: str) -> list[SequenceEvent]:
    """Find repeated root-motion patterns over the primary chord of each measure."""

    primary: dict[int, ChordEvent] = {}
    for chord in chords:
        current = primary.get(chord.measure)
        if current is None or (chord.end_step - chord.start_step) * (0.5 + chord.confidence) > (
            current.end_step - current.start_step
        ) * (0.5 + current.confidence):
            primary[chord.measure] = chord
    measures = sorted(primary)
    if len(measures) < 3:
        return []
    events = [primary[measure] for measure in measures]
    tonic_pc = _NAME_TO_PC[tonic]
    scale = _scale_pcs(tonic_pc, mode)

    def describe(interval: int) -> tuple[str, str] | None:
        return {
            5: ("descending-fifth", "Descending fifth sequence"),
            7: ("ascending-fifth", "Ascending fifth sequence"),
            2: ("ascending-step", "Ascending step sequence"),
            10: ("descending-step", "Descending step sequence"),
        }.get(interval)

    found: list[SequenceEvent] = []
    occupied: set[int] = set()

    def add(start_index: int, end_index: int, kind: str, description: str) -> None:
        span = set(range(start_index, end_index + 1))
        if span & occupied:
            return
        occupied.update(span)
        group = events[start_index : end_index + 1]
        found.append(
            SequenceEvent(
                start_measure=group[0].measure,
                end_measure=group[-1].measure,
                kind=kind,
                description=description,
                repetitions=len(group) - 1,
                confidence=round(sum(item.confidence for item in group) / len(group), 4),
            )
        )

    # Diatonic stepwise root motion first: its semitone sizes differ
    # (tone/semitone), so a constant-interval pass would otherwise claim a
    # shorter prefix of the same span (C-D-E inside C-D-E-F).
    degrees: list[int | None] = [
        scale.index(event.root_pc) if event.root_pc in scale else None for event in events
    ]
    diatonic: list[tuple[int, int, int]] = []
    for start in range(len(events) - 2):
        for direction in (1, -1):
            length = 0
            for index in range(start, len(events) - 1):
                current, following = degrees[index], degrees[index + 1]
                if current is None or following is None:
                    break
                if (following - current) % 7 != direction % 7:
                    break
                length += 1
            if length >= 2:
                diatonic.append((length, start, direction))
    for length, start, direction in sorted(diatonic, reverse=True):
        kind = "ascending-step" if direction == 1 else "descending-step"
        text = "Ascending step sequence" if direction == 1 else "Descending step sequence"
        add(start, start + length, kind, text)

    # Longest constant root-motion windows first.
    windows: list[tuple[int, int, int]] = []
    for start in range(len(events) - 2):
        intervals = [(events[i + 1].root_pc - events[i].root_pc) % 12 for i in range(start, len(events) - 1)]
        for interval in {5, 7, 2, 10}:
            length = 0
            for value in intervals:
                if value != interval:
                    break
                length += 1
            if length >= 2:
                windows.append((length, start, interval))
    for length, start, interval in sorted(windows, reverse=True):
        named = describe(interval)
        if named:
            add(start, start + length, named[0], named[1])

    # A one- or two-measure harmonic module stated at least three times.
    for period in (1, 2):
        for start in range(len(events)):
            pattern = [(item.root_pc, item.quality) for item in events[start : start + period]]
            if len(pattern) < period:
                continue
            count = 0
            index = start
            while index + period <= len(events) and [
                (item.root_pc, item.quality) for item in events[index : index + period]
            ] == pattern:
                count += 1
                index += period
            if count >= 3:
                add(start, index - 1, "repeated-module", "Repeated harmonic module")
    found.sort(key=lambda item: item.start_measure)
    return found


def analyze_harmony(
    quantization: Quantization,
    *,
    key_tonic: str | None = None,
    key_mode: str | None = None,
) -> HarmonicAnalysis:
    if key_tonic is not None or key_mode is not None:
        tonic, _pc = parse_tonic(key_tonic or "C")
        mode = (key_mode or "major").lower()
        if mode not in {"major", "minor"}:
            raise AnalysisError("Key mode must be major or minor.")
        source = "manual"
        confidence = 1.0
    else:
        tonic, mode, confidence = estimate_key(quantization)
        source = "automatic"
    chords = detect_chords(quantization, tonic, mode)
    sequences = detect_sequences(chords, tonic, mode)
    return HarmonicAnalysis(
        key_tonic=tonic,
        key_mode=mode,
        key_source=source,
        key_confidence=confidence,
        chords=chords,
        sequences=sequences,
        grid={
            "bpm": quantization.bpm,
            "numerator": quantization.numerator,
            "denominator": quantization.denominator,
            "subdivision": quantization.subdivision,
            "downbeat_seconds": quantization.downbeat,
            "steps_per_measure": quantization.steps_per_measure,
            "first_step": -quantization.lead_steps,
        },
        notes_fingerprint=notes_fingerprint(quantization),
    )


# --------------------------------------------------------------------------
# Reserved analysis passes
# --------------------------------------------------------------------------


class CounterpointAnalyzer:
    """Later voice and imitation passes; harmony and sequences are implemented above."""

    def split_voices(self, transcription: Transcription) -> list[Voice]:
        """Separate independent melodic lines, including voice crossings."""

        raise NotImplementedError("Voice separation is reserved for a later phase.")

    def find_imitation(self, voices: list[Voice]) -> list[AnalysisAnnotation]:
        raise NotImplementedError("Imitation detection is reserved for a later phase.")

    def harmonic_progression(
        self, quantization: Quantization, *, key_tonic: str | None = None, key_mode: str | None = None
    ) -> list[ChordEvent]:
        return analyze_harmony(quantization, key_tonic=key_tonic, key_mode=key_mode).chords

    def find_sequences(self, chords: list[ChordEvent], tonic: str, mode: str) -> list[SequenceEvent]:
        return detect_sequences(chords, tonic, mode)

    def find_cadences(self, transcription: Transcription) -> list[AnalysisAnnotation]:
        raise NotImplementedError("Cadence detection is reserved for a later phase.")
