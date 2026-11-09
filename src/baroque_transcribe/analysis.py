"""Places where contrapuntal analysis can attach later.

Nothing in this module inspects the music yet. The transcription pipeline
stops at notes, pedal, and a pitch-based staff split. Voice leading, imitation,
sequences, harmonic progressions, and cadences need a separate pass that
reads ``notes.json`` and writes annotations beside it, without rewriting
``raw.mid``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from baroque_transcribe.schema import Transcription


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


class CounterpointAnalyzer:
    """Interface for a later analysis pass. Every method still raises."""

    def split_voices(self, transcription: Transcription) -> list[Voice]:
        """Separate independent melodic lines, including voice crossings."""

        raise NotImplementedError("Voice separation is reserved for a later phase.")

    def find_imitation(self, voices: list[Voice]) -> list[AnalysisAnnotation]:
        raise NotImplementedError("Imitation detection is reserved for a later phase.")

    def find_sequences(self, voices: list[Voice]) -> list[AnalysisAnnotation]:
        raise NotImplementedError("Sequence detection is reserved for a later phase.")

    def harmonic_progression(self, transcription: Transcription) -> list[AnalysisAnnotation]:
        raise NotImplementedError("Harmonic analysis is reserved for a later phase.")

    def find_cadences(self, transcription: Transcription) -> list[AnalysisAnnotation]:
        raise NotImplementedError("Cadence detection is reserved for a later phase.")
