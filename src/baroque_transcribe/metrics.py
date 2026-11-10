"""Notation quality metrics for comparing literal and readable scores.

These numbers are a reading aid for the comparison command. They do not change
any project files.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
import xml.etree.ElementTree as element_tree

from baroque_transcribe.notation import _collapse, _pack_voices, _staff_events
from baroque_transcribe.quantize import Quantization


@dataclass
class NotationMetrics:
    note_count: int
    attack_clusters: int
    attacks_on_beat: int
    attacks_total: int
    attacks_on_beat_ratio: float
    mean_onset_residual_steps: float
    rest_elements: int
    note_elements: int
    tie_starts: int
    voices_treble: int
    voices_bass: int
    measures_with_3plus_treble_voices: int
    split_onset_groups: int
    colliding_attacks: int
    grid_mode: str
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _local_tag(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def score_xml_counts(score_path: Path) -> dict[str, int]:
    """Count rests, pitched notes and tie starts in a MusicXML file."""

    root = element_tree.parse(score_path).getroot()
    rests = 0
    notes = 0
    ties = 0
    for element in root.iter():
        tag = _local_tag(element.tag)
        if tag != "note":
            continue
        if any(_local_tag(child.tag) == "rest" for child in element):
            rests += 1
            continue
        if any(_local_tag(child.tag) == "pitch" for child in element):
            notes += 1
        for child in element:
            if _local_tag(child.tag) == "tie" and child.get("type") == "start":
                ties += 1
                break
    return {"rest_elements": rests, "note_elements": notes, "tie_starts": ties}


def grid_metrics(quantization: Quantization, *, staff_split: int = 60) -> dict:
    """Metrics that come from the quantized grid alone."""

    grid = quantization.grid
    attacks = sorted({item.onset_step for item in grid})
    # Pulse is the modal step interval between attacks (often 2 on a 16th grid
    # for eighth-note music). "On the beat" means the attack lands on that pulse.
    if len(attacks) >= 2:
        advances = [b - a for a, b in zip(attacks, attacks[1:])]
        pulse = Counter(advances).most_common(1)[0][0]
        pulse = max(1, int(pulse))
    else:
        pulse = max(1, quantization.subdivision // 8)
    on_beat = sum(1 for step in attacks if (step + quantization.lead_steps) % pulse == 0)
    residuals: list[float] = []
    for item in grid:
        expected = quantization.step_to_audio(item.onset_step)
        local_q = quantization.quantum
        if quantization.grid_mode == "follow":
            local_q = max(
                quantization.quantum * 0.5,
                abs(quantization.step_to_audio(item.onset_step + 1) - expected),
            )
        residuals.append(abs(item.source_onset - expected) / local_q if local_q else 0.0)
    mean_residual = sum(residuals) / len(residuals) if residuals else 0.0

    events = _collapse(grid)
    treble = _pack_voices(_staff_events(events, staff_split, True))
    bass = _pack_voices(_staff_events(events, staff_split, False))
    per = quantization.steps_per_measure
    first = -quantization.lead_steps
    last = max((item.offset_step for item in grid), default=first + per)
    measures_3plus = 0
    measure = first
    while measure < last:
        end = measure + per
        active = sum(
            1
            for voice in treble
            if any(event.onset < end and event.offset > measure for event in voice)
        )
        if active >= 3:
            measures_3plus += 1
        measure = end

    by_onset: dict[tuple[int, bool], set[int]] = {}
    for item in grid:
        key = (item.onset_step, item.pitch >= staff_split)
        by_onset.setdefault(key, set()).add(item.offset_step)
    split_groups = sum(1 for offsets in by_onset.values() if len(offsets) > 1)

    occupied: Counter[int] = Counter()
    for cluster in quantization.clusters:
        # Clusters already share a step in follow mode; count fixed-mode
        # collisions as distinct pitches on one step.
        pass
    for item in grid:
        occupied[item.onset_step] += 1
    colliding = sum(1 for count in occupied.values() if count > 1)

    return {
        "note_count": len(grid),
        "attack_clusters": len(quantization.clusters) if quantization.clusters else len(attacks),
        "attacks_on_beat": on_beat,
        "attacks_total": len(attacks),
        "attacks_on_beat_ratio": round(on_beat / len(attacks), 4) if attacks else 0.0,
        "mean_onset_residual_steps": round(mean_residual, 4),
        "voices_treble": len(treble),
        "voices_bass": len(bass),
        "measures_with_3plus_treble_voices": measures_3plus,
        "split_onset_groups": split_groups,
        "colliding_attacks": colliding,
        "grid_mode": quantization.grid_mode,
        "warnings": list(quantization.warnings),
    }


def measure_notation(
    quantization: Quantization,
    score_path: Path | None = None,
    *,
    staff_split: int = 60,
    interpretation=None,
) -> NotationMetrics:
    """Combine grid metrics with optional MusicXML element counts."""

    payload = grid_metrics(quantization, staff_split=staff_split)
    if interpretation is not None:
        payload["voices_treble"] = len(interpretation.events_for_staff(1))
        payload["voices_bass"] = len(interpretation.events_for_staff(2))
        payload["warnings"] = list(payload["warnings"]) + list(interpretation.warnings)
    xml_counts = {"rest_elements": 0, "note_elements": 0, "tie_starts": 0}
    if score_path is not None and score_path.is_file():
        xml_counts = score_xml_counts(score_path)
    return NotationMetrics(
        note_count=payload["note_count"],
        attack_clusters=payload["attack_clusters"],
        attacks_on_beat=payload["attacks_on_beat"],
        attacks_total=payload["attacks_total"],
        attacks_on_beat_ratio=payload["attacks_on_beat_ratio"],
        mean_onset_residual_steps=payload["mean_onset_residual_steps"],
        rest_elements=xml_counts["rest_elements"],
        note_elements=xml_counts["note_elements"],
        tie_starts=xml_counts["tie_starts"],
        voices_treble=payload["voices_treble"],
        voices_bass=payload["voices_bass"],
        measures_with_3plus_treble_voices=payload["measures_with_3plus_treble_voices"],
        split_onset_groups=payload["split_onset_groups"],
        colliding_attacks=payload["colliding_attacks"],
        grid_mode=payload["grid_mode"],
        warnings=payload["warnings"],
    )
