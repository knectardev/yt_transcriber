"""Snap note times onto a metrical grid without changing pitches.

The grid is defined by a tempo, a time signature, a subdivision, and the
audio time of the first downbeat. Pitches and velocities are copied. This is
a rhythmic reading of the performance, not a new arrangement.

Two grid modes are supported:

- ``fixed``: every note snaps to ``downbeat + n * quantum``. This is the
  historical behaviour and the default for existing projects.
- ``follow``: attack clusters are tracked with a phase-locked local tempo so
  expressive timing lands on musically intended beats. Sounding durations are
  still derived from the original offsets; only the step labels change.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from baroque_transcribe.defaults import DEFAULT_SUBDIVISION, DEFAULT_TIME_SIGNATURE
from baroque_transcribe.notation_settings import NotationSettings, settings_from_manifest
from baroque_transcribe.schema import NoteEvent


class QuantizeError(ValueError):
    pass


@dataclass
class GridNote:
    pitch: int
    onset_step: int
    offset_step: int
    velocity: int
    source_index: int
    source_onset: float
    source_offset: float


@dataclass
class AttackCluster:
    """Notes that share one notated attack after conservative onset clustering."""

    time: float
    members: list[int]
    seed_onset: float


@dataclass
class Quantization:
    bpm: float
    numerator: int
    denominator: int
    subdivision: int
    downbeat: float
    quantum: float
    steps_per_measure: int
    lead_steps: int
    notes: list[NoteEvent]
    grid: list[GridNote]
    grid_mode: str = "fixed"
    step_audio_times: dict[int, float] = field(default_factory=dict)
    settings: NotationSettings = field(default_factory=NotationSettings)
    warnings: list[str] = field(default_factory=list)
    clusters: list[AttackCluster] = field(default_factory=list)

    @property
    def quarter_seconds(self) -> float:
        return 60.0 / self.bpm

    @property
    def measure_seconds(self) -> float:
        return self.steps_per_measure * self.quantum

    def step_to_audio(self, step: int) -> float:
        """Working-audio seconds for a grid step.

        In ``follow`` mode this interpolates the tracked attack times so video
        sync and measure anchors follow the performance. In ``fixed`` mode it
        is the metronome time.
        """

        if self.grid_mode == "fixed" or not self.step_audio_times:
            return self.downbeat + step * self.quantum
        known = self.step_audio_times
        if step in known:
            return known[step]
        steps = sorted(known)
        if not steps:
            return self.downbeat + step * self.quantum
        if step < steps[0]:
            return known[steps[0]] - (steps[0] - step) * self._local_quantum(steps[0])
        if step > steps[-1]:
            return known[steps[-1]] + (step - steps[-1]) * self._local_quantum(steps[-1])
        lo = max(value for value in steps if value <= step)
        hi = min(value for value in steps if value >= step)
        if hi == lo:
            return known[lo]
        fraction = (step - lo) / (hi - lo)
        return known[lo] + fraction * (known[hi] - known[lo])

    def _local_quantum(self, around_step: int) -> float:
        steps = sorted(self.step_audio_times)
        if len(steps) < 2:
            return self.quantum
        if around_step <= steps[0]:
            a, b = steps[0], steps[1]
        elif around_step >= steps[-1]:
            a, b = steps[-2], steps[-1]
        else:
            lo = max(value for value in steps if value <= around_step)
            higher = [value for value in steps if value > lo]
            a, b = lo, higher[0]
        span = b - a
        if span <= 0:
            return self.quantum
        return max(self.quantum * 0.5, (self.step_audio_times[b] - self.step_audio_times[a]) / span)

    def step_to_score(self, step: int) -> float:
        """Seconds from the start of the notated score, with the downbeat on a barline."""

        return (step + self.lead_steps) * self.quantum

    def audio_to_step(self, seconds: float) -> int:
        """Nearest grid step for an audio time."""

        if self.grid_mode == "fixed" or not self.step_audio_times:
            return int(round((seconds - self.downbeat) / self.quantum))
        steps = sorted(self.step_audio_times)
        if not steps:
            return int(round((seconds - self.downbeat) / self.quantum))
        # Search a window around the extrapolated step.
        approx = int(round((seconds - self.downbeat) / self.quantum))
        lo = min(steps[0], approx - 8)
        hi = max(steps[-1], approx + 8)
        best_step = approx
        best_err = abs(self.step_to_audio(approx) - seconds)
        for step in range(lo, hi + 1):
            err = abs(self.step_to_audio(step) - seconds)
            if err < best_err:
                best_err = err
                best_step = step
        return best_step


def parse_time_signature(text: str) -> tuple[int, int]:
    parts = text.replace(" ", "").split("/")
    if len(parts) != 2:
        raise QuantizeError(f"Time signature must look like 4/4 or 6/8, not {text!r}.")
    try:
        numerator, denominator = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise QuantizeError(f"Time signature must look like 4/4 or 6/8, not {text!r}.") from exc
    if numerator < 1 or denominator not in {1, 2, 4, 8, 16, 32}:
        raise QuantizeError(f"Unsupported time signature {numerator}/{denominator}.")
    return numerator, denominator


def steps_per_measure(numerator: int, denominator: int, subdivision: int) -> int:
    if subdivision not in {8, 16, 32}:
        raise QuantizeError("Subdivision must be 8, 16, or 32.")
    if (numerator * subdivision) % denominator != 0:
        raise QuantizeError(
            f"{numerator}/{denominator} does not land on a {subdivision}-note grid. "
            "Use a finer subdivision or a different time signature."
        )
    return numerator * subdivision // denominator


def cluster_onsets(
    notes: list[NoteEvent],
    *,
    window_seconds: float,
) -> list[AttackCluster]:
    """Group near-simultaneous onsets without absorbing runs or ornaments.

    A note joins a cluster only when it falls inside the window from the
    cluster seed and does not share a pitch with any member already there.
    Same-pitch attacks always start a new cluster.
    """

    if not notes:
        return []
    ordered = sorted(range(len(notes)), key=lambda index: (notes[index].onset, notes[index].pitch, index))
    clusters: list[AttackCluster] = []
    for index in ordered:
        note = notes[index]
        placed = False
        if window_seconds > 0 and clusters:
            candidate = clusters[-1]
            if note.onset - candidate.seed_onset <= window_seconds:
                pitches = {notes[member].pitch for member in candidate.members}
                if note.pitch not in pitches:
                    candidate.members.append(index)
                    placed = True
        if not placed:
            clusters.append(AttackCluster(time=note.onset, members=[index], seed_onset=note.onset))
    for cluster in clusters:
        onsets = [notes[index].onset for index in cluster.members]
        cluster.time = sum(onsets) / len(onsets)
    return clusters


def track_attack_steps(
    attack_times: list[float],
    *,
    bpm: float,
    subdivision: int,
    downbeat: float,
    silence_reanchor_seconds: float,
) -> tuple[list[int], dict[int, float], list[str]]:
    """Assign an integer step to each attack time, following local tempo.

    Returns ``(steps, step_audio_times, warnings)``. ``step_audio_times`` maps
    each observed step to the attack time that landed on it.
    """

    warnings: list[str] = []
    if not attack_times:
        return [], {}, warnings
    quantum = (60.0 / bpm) * 4.0 / subdivision
    steps: list[int] = []
    step_times: dict[int, float] = {}
    first = int(round((attack_times[0] - downbeat) / quantum))
    steps.append(first)
    step_times[first] = attack_times[0]
    phase = attack_times[0]
    local_q = quantum
    last_step = first
    collisions = 0

    for time in attack_times[1:]:
        gap = time - phase
        if gap >= silence_reanchor_seconds:
            local_q = quantum
            candidate = int(round((time - downbeat) / quantum))
            if candidate <= last_step:
                candidate = last_step + max(1, int(round(gap / max(local_q, 1e-6))))
            steps.append(candidate)
            step_times[candidate] = time
            phase = time
            last_step = candidate
            continue

        expected = max(1, int(round(gap / max(local_q, 1e-6))))
        previous_advance = steps[-1] - steps[-2] if len(steps) >= 2 else expected
        best: tuple[float, int, float] | None = None
        for advance in range(max(1, expected - 2), expected + 3):
            predicted = phase + advance * local_q
            residual = time - predicted
            # Prefer a residual that keeps local tempo near the recent pulse,
            # and lightly prefer continuing the previous step advance.
            local_cost = abs((gap / advance) - local_q) / max(local_q, 1e-6)
            continuity = 0.02 * quantum * abs(advance - previous_advance)
            score = abs(residual) + 0.2 * local_cost * local_q + continuity
            if best is None or score < best[0]:
                best = (score, advance, residual)
        assert best is not None
        advance = best[1]
        residual = best[2]
        new_step = last_step + advance
        if new_step in step_times:
            collisions += 1
            new_step = last_step + max(advance, 1)
            while new_step in step_times:
                new_step += 1
        alpha = 0.55
        beta = 0.18
        # Track local step duration from the observed gap, clamped so a single
        # outlier cannot double or halve the grid.
        observed_q = gap / advance
        local_q = min(quantum * 1.35, max(quantum * 0.65, (1.0 - beta) * local_q + beta * observed_q))
        # Lock the observed attack onto its step for the next gap.
        phase = time
        steps.append(new_step)
        step_times[new_step] = time
        last_step = new_step

    if collisions:
        warnings.append(
            f"{collisions} attack(s) landed on an already occupied grid step. "
            "Consider a finer subdivision."
        )
    return steps, step_times, warnings


def _lead_steps(grid: list[GridNote], per_measure: int) -> int:
    if grid:
        earliest = min(item.onset_step for item in grid)
    else:
        earliest = 0
    first_bar = (min(earliest, 0) // per_measure) * per_measure
    return -first_bar


def _fixed_grid(
    notes: list[NoteEvent],
    *,
    bpm: float,
    downbeat: float,
    quantum: float,
) -> list[GridNote]:
    grid: list[GridNote] = []
    for index, note in enumerate(notes):
        onset_step = int(round((note.onset - downbeat) / quantum))
        offset_step = int(round((note.offset - downbeat) / quantum))
        if offset_step <= onset_step:
            offset_step = onset_step + 1
        grid.append(
            GridNote(
                pitch=int(note.pitch),
                onset_step=onset_step,
                offset_step=offset_step,
                velocity=int(note.velocity),
                source_index=index,
                source_onset=float(note.onset),
                source_offset=float(note.offset),
            )
        )
    grid.sort(key=lambda item: (item.onset_step, item.pitch, item.offset_step))
    return grid


def _follow_grid(
    notes: list[NoteEvent],
    *,
    bpm: float,
    subdivision: int,
    downbeat: float,
    quantum: float,
    settings: NotationSettings,
) -> tuple[list[GridNote], list[AttackCluster], dict[int, float], list[str]]:
    window = settings.cluster_window_seconds(quantum)
    clusters = cluster_onsets(notes, window_seconds=window)
    attack_times = [cluster.time for cluster in clusters]
    attack_steps, step_times, warnings = track_attack_steps(
        attack_times,
        bpm=bpm,
        subdivision=subdivision,
        downbeat=downbeat,
        silence_reanchor_seconds=settings.silence_reanchor_seconds,
    )
    # Temporary quantization used only for offset snapping through step_to_audio.
    probe = Quantization(
        bpm=bpm,
        numerator=4,
        denominator=4,
        subdivision=subdivision,
        downbeat=downbeat,
        quantum=quantum,
        steps_per_measure=subdivision,
        lead_steps=0,
        notes=[],
        grid=[],
        grid_mode="follow",
        step_audio_times=step_times,
        settings=settings,
    )
    by_source: dict[int, GridNote] = {}
    occupied: dict[tuple[int, int], int] = {}
    for cluster, onset_step in zip(clusters, attack_steps):
        for source_index in cluster.members:
            note = notes[source_index]
            offset_step = probe.audio_to_step(note.offset)
            if offset_step <= onset_step:
                offset_step = onset_step + 1
            key = (onset_step, note.pitch)
            if key in occupied:
                warnings.append(
                    f"Pitch {note.pitch} appears twice at step {onset_step}; "
                    "kept both notes on adjacent steps."
                )
                onset_step = onset_step + 1
                offset_step = max(offset_step, onset_step + 1)
            occupied[(onset_step, note.pitch)] = source_index
            by_source[source_index] = GridNote(
                pitch=int(note.pitch),
                onset_step=onset_step,
                offset_step=offset_step,
                velocity=int(note.velocity),
                source_index=source_index,
                source_onset=float(note.onset),
                source_offset=float(note.offset),
            )
    grid = sorted(by_source.values(), key=lambda item: (item.onset_step, item.pitch, item.offset_step))
    # Densify observed step times with every assigned onset so interpolation stays local.
    for item in grid:
        step_times.setdefault(item.onset_step, item.source_onset)
    return grid, clusters, step_times, warnings


def quantize_notes(
    notes: list[NoteEvent],
    *,
    bpm: float,
    numerator: int,
    denominator: int,
    subdivision: int,
    downbeat: float = 0.0,
    grid: str | None = None,
    settings: NotationSettings | None = None,
) -> Quantization:
    if bpm <= 0:
        raise QuantizeError("Tempo must be a positive BPM.")
    resolved = settings or NotationSettings()
    mode = grid or resolved.grid
    if mode not in {"fixed", "follow"}:
        raise QuantizeError(f"Grid mode must be 'fixed' or 'follow', not {mode!r}.")
    if mode != resolved.grid:
        resolved = NotationSettings(
            grid=mode,
            style=resolved.style,
            cluster_ms=resolved.cluster_ms,
            fill_gap_steps=resolved.fill_gap_steps,
            keep_rest_steps=resolved.keep_rest_steps,
            overlap_trim_steps=resolved.overlap_trim_steps,
            chord_release_tolerance_steps=resolved.chord_release_tolerance_steps,
            extra_voice_cost=resolved.extra_voice_cost,
            silence_reanchor_seconds=resolved.silence_reanchor_seconds,
        )
    per_measure = steps_per_measure(numerator, denominator, subdivision)
    quarter = 60.0 / bpm
    quantum = quarter * 4.0 / subdivision
    warnings: list[str] = []
    clusters: list[AttackCluster] = []
    step_audio_times: dict[int, float] = {}

    if mode == "follow":
        grid_notes, clusters, step_audio_times, warnings = _follow_grid(
            notes,
            bpm=bpm,
            subdivision=subdivision,
            downbeat=downbeat,
            quantum=quantum,
            settings=resolved,
        )
    else:
        grid_notes = _fixed_grid(notes, bpm=bpm, downbeat=downbeat, quantum=quantum)

    snapped = [
        NoteEvent(
            pitch=item.pitch,
            onset=downbeat + item.onset_step * quantum,
            offset=downbeat + item.offset_step * quantum,
            velocity=item.velocity,
        )
        for item in grid_notes
    ]
    lead = _lead_steps(grid_notes, per_measure)
    if mode == "fixed":
        for item in grid_notes:
            step_audio_times[item.onset_step] = downbeat + item.onset_step * quantum
    return Quantization(
        bpm=float(bpm),
        numerator=numerator,
        denominator=denominator,
        subdivision=subdivision,
        downbeat=float(downbeat),
        quantum=quantum,
        steps_per_measure=per_measure,
        lead_steps=lead,
        notes=snapped,
        grid=grid_notes,
        grid_mode=mode,
        step_audio_times=step_audio_times,
        settings=resolved,
        warnings=warnings,
        clusters=clusters,
    )


def quantization_from_manifest(
    notes: list[NoteEvent],
    tempo_info: dict,
    *,
    settings: NotationSettings | None = None,
    manifest: dict | None = None,
) -> Quantization:
    """Rebuild the grid the same way every call site does.

    ``tempo_info`` is the ``manifest['tempo']`` object. Missing meter falls
    back to 4/4 and the default subdivision.
    """

    if not tempo_info.get("bpm"):
        raise QuantizeError("Tempo information has no BPM.")
    signature_text = str(tempo_info.get("time_signature") or f"{DEFAULT_TIME_SIGNATURE[0]}/{DEFAULT_TIME_SIGNATURE[1]}")
    numerator, denominator = parse_time_signature(signature_text)
    resolved = settings
    if resolved is None:
        resolved = settings_from_manifest(manifest if manifest is not None else {"tempo": tempo_info, "notation": {}})
        # Prefer an explicit grid key on the tempo block when present.
        if tempo_info.get("grid"):
            resolved = NotationSettings(
                grid=str(tempo_info["grid"]),
                style=resolved.style,
                cluster_ms=resolved.cluster_ms,
                fill_gap_steps=resolved.fill_gap_steps,
                keep_rest_steps=resolved.keep_rest_steps,
                overlap_trim_steps=resolved.overlap_trim_steps,
                chord_release_tolerance_steps=resolved.chord_release_tolerance_steps,
                extra_voice_cost=resolved.extra_voice_cost,
                silence_reanchor_seconds=resolved.silence_reanchor_seconds,
            )
    return quantize_notes(
        notes,
        bpm=float(tempo_info["bpm"]),
        numerator=numerator,
        denominator=denominator,
        subdivision=int(tempo_info.get("subdivision") or DEFAULT_SUBDIVISION),
        downbeat=float(tempo_info.get("downbeat_seconds") or 0.0),
        settings=resolved,
    )
