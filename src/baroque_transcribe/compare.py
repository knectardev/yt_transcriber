"""Write side-by-side notation variants without touching the project score.

Outputs land in ``<project>/comparison/``. The project's ``score.musicxml``,
``quantized.mid``, ``notes.json`` and ``raw.mid`` are left alone.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from baroque_transcribe.defaults import (
    COMPARISON_DIR,
    DEFAULT_STAFF_SPLIT,
    MANIFEST,
    NOTES_JSON,
    RAW_MID,
)
from baroque_transcribe.metrics import NotationMetrics, measure_notation
from baroque_transcribe.notation import write_musicxml, write_notation_page
from baroque_transcribe.notation_settings import NotationSettings
from baroque_transcribe.quantize import Quantization, quantize_notes
from baroque_transcribe.schema import read_transcription


class CompareError(ValueError):
    pass


@dataclass
class VariantResult:
    name: str
    score_path: Path
    page_path: Path
    metrics: NotationMetrics
    quantization: Quantization


@dataclass
class CompareReport:
    project: Path
    output_dir: Path
    variants: list[VariantResult]
    notes_fingerprint: str
    raw_midi_unchanged: bool

    preservation: dict[str, dict] | None = None

    def to_dict(self) -> dict:
        payload = {
            "project": str(self.project),
            "output_dir": str(self.output_dir),
            "notes_fingerprint": self.notes_fingerprint,
            "raw_midi_unchanged": self.raw_midi_unchanged,
            "variants": {
                variant.name: {
                    "score": variant.score_path.name,
                    "page": variant.page_path.name,
                    "metrics": variant.metrics.to_dict(),
                    "warnings": variant.quantization.warnings,
                }
                for variant in self.variants
            },
        }
        if self.preservation:
            payload["preservation"] = self.preservation
        return payload


def _read_manifest(project: Path) -> dict:
    path = project / MANIFEST
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _tempo_overrides(
    manifest: dict,
    *,
    tempo: float | None,
    time_signature: str | None,
    subdivision: int | None,
    downbeat: float | None,
) -> dict:
    previous = manifest.get("tempo") if isinstance(manifest.get("tempo"), dict) else {}
    if tempo is None and not previous.get("bpm"):
        raise CompareError("Pass --tempo. This project has no tempo in its manifest.")
    signature = time_signature or previous.get("time_signature") or "4/4"
    return {
        "bpm": float(tempo if tempo is not None else previous["bpm"]),
        "time_signature": str(signature),
        "subdivision": int(subdivision if subdivision is not None else previous.get("subdivision") or 16),
        "downbeat_seconds": float(
            downbeat if downbeat is not None else previous.get("downbeat_seconds") or 0.0
        ),
    }


def _notes_fingerprint(notes_path: Path) -> str:
    import hashlib

    return hashlib.sha256(notes_path.read_bytes()).hexdigest()[:16]


def compare_project(
    project: Path,
    *,
    variants: list[tuple[str, NotationSettings]] | None = None,
    tempo: float | None = None,
    time_signature: str | None = None,
    subdivision: int | None = None,
    downbeat: float | None = None,
    staff_split: int | None = None,
    title: str | None = None,
) -> CompareReport:
    """Build variant scores under ``project/comparison/``."""

    project = project.expanduser().resolve()
    notes_path = project / NOTES_JSON
    if not notes_path.is_file():
        raise CompareError(f"No {NOTES_JSON} in {project}.")
    before_notes = notes_path.read_bytes()
    raw_path = project / RAW_MID
    before_raw = raw_path.read_bytes() if raw_path.is_file() else None

    manifest = _read_manifest(project)
    tempo_info = _tempo_overrides(
        manifest,
        tempo=tempo,
        time_signature=time_signature,
        subdivision=subdivision,
        downbeat=downbeat,
    )
    from baroque_transcribe.quantize import parse_time_signature

    numerator, denominator = parse_time_signature(str(tempo_info["time_signature"]))
    split = DEFAULT_STAFF_SPLIT if staff_split is None else staff_split
    transcription = read_transcription(notes_path)
    heading = title or str(manifest.get("title") or project.name)

    if variants is None:
        variants = [
            ("literal_fixed", NotationSettings(grid="fixed", style="literal")),
            ("literal_follow", NotationSettings(grid="follow", style="literal")),
            ("readable_follow", NotationSettings(grid="follow", style="readable")),
        ]

    output_dir = project / COMPARISON_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[VariantResult] = []
    preservation_summary: dict[str, dict] = {}
    for name, settings in variants:
        colorize = name.endswith("_voices")
        quantization = quantize_notes(
            transcription.notes,
            bpm=float(tempo_info["bpm"]),
            numerator=numerator,
            denominator=denominator,
            subdivision=int(tempo_info["subdivision"]),
            downbeat=float(tempo_info["downbeat_seconds"]),
            settings=settings,
        )
        score_path = output_dir / f"{name}.musicxml"
        map_path = output_dir / f"{name}.notation_map.json" if settings.style == "readable" else None
        preservation_path = (
            output_dir / f"{name}.preservation.json" if settings.style == "readable" else None
        )
        voices_path = None
        if settings.style == "readable" and not colorize:
            voices_path = output_dir / f"{name}.voices.musicxml"
        write_musicxml(
            score_path,
            quantization,
            staff_split=split,
            title=f"{heading} ({name})",
            style=settings.style,
            notation_map=map_path,
            pedals=transcription.pedals,
            colorize_voices=colorize,
            preservation_report=preservation_path,
            voices_path=voices_path,
        )
        legend = None
        if colorize:
            from baroque_transcribe.notation import VOICE_COLORS, voice_color

            legend = (
                "Voice colours: "
                + ", ".join(
                    f"staff{staff}/v{voice}={voice_color(staff, voice)}"
                    for staff in (1, 2)
                    for voice in (1, 2, 3, 4)
                )
            )
            del VOICE_COLORS
        page_path = write_notation_page(
            score_path,
            output_dir / f"{name}.html",
            title=f"{heading} ({name})",
            note=(
                f"Comparison variant '{name}'. Grid={settings.grid}, style={settings.style}"
                f"{', voice colours on' if colorize else ''}. "
                "This file does not replace the project score."
                + (f" {legend}" if legend else "")
            ),
            voices_score_path=voices_path if voices_path and voices_path.is_file() else None,
        )
        interpretation = None
        if settings.style == "readable":
            from baroque_transcribe.engrave import (
                interpret_quantization,
                validate_against_source_notes,
            )

            interpretation = interpret_quantization(
                quantization,
                staff_split=split,
                settings=settings,
                pedals=transcription.pedals,
            )
            preservation_summary[name] = validate_against_source_notes(
                transcription.notes, quantization, interpretation
            ).to_dict()
        metrics = measure_notation(
            quantization,
            score_path,
            staff_split=split,
            interpretation=interpretation,
        )
        results.append(
            VariantResult(
                name=name,
                score_path=score_path,
                page_path=page_path,
                metrics=metrics,
                quantization=quantization,
            )
        )

    after_notes = notes_path.read_bytes()
    after_raw = raw_path.read_bytes() if raw_path.is_file() else None
    if after_notes != before_notes:
        raise CompareError("compare_project changed notes.json; that must never happen.")
    raw_unchanged = before_raw == after_raw
    report = CompareReport(
        project=project,
        output_dir=output_dir,
        variants=results,
        notes_fingerprint=_notes_fingerprint(notes_path),
        raw_midi_unchanged=raw_unchanged,
        preservation=preservation_summary or None,
    )
    (output_dir / "report.json").write_text(json.dumps(report.to_dict(), indent=2) + "\n", encoding="utf-8")
    return report
