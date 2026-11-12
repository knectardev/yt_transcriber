"""Transcription jobs for the review UI.

Jobs live in this process. A browser can disconnect without stopping one.
Restarting the process does not resume a job. Two submissions of the same
source and settings, including a Streamlit rerun, share one running job.
"""

from __future__ import annotations

import secrets
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from baroque_transcribe.progress import ProgressEvent

ProgressFn = Callable[[ProgressEvent], None]
Runner = Callable[[str, ProgressFn], Path]


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def submission_key(source: str, output: str, settings: dict[str, object]) -> str:
    """Stable identity of one submission. The order of settings does not matter."""

    parts = [source.strip(), str(Path(output))]
    for name in sorted(settings):
        parts.append(f"{name}={settings[name]}")
    return "\n".join(parts)


@dataclass
class JobRecord:
    id: str
    status: str
    source: str
    requested_output: str
    output_dir: str | None = None
    label: str = "Starting"
    fraction: float | None = None
    stages: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    created_at: str = ""
    updated_at: str = ""

    def copy(self) -> "JobRecord":
        return JobRecord(
            id=self.id,
            status=self.status,
            source=self.source,
            requested_output=self.requested_output,
            output_dir=self.output_dir,
            label=self.label,
            fraction=self.fraction,
            stages=dict(self.stages),
            error=self.error,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class JobBoard:
    """The jobs started by every browser connected to this process."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._active: dict[str, str] = {}

    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return None if job is None else job.copy()

    def list_jobs(self) -> list[JobRecord]:
        with self._lock:
            jobs = [job.copy() for job in self._jobs.values()]
        jobs.sort(key=lambda job: job.created_at, reverse=True)
        return jobs

    def start(
        self,
        *,
        source: str,
        output: str,
        settings: dict[str, object],
        run: Runner,
    ) -> JobRecord:
        """Start ``run`` on a thread, or return the job already running for this submission."""

        key = submission_key(source, output, settings)
        with self._lock:
            existing_id = self._active.get(key)
            if existing_id is not None:
                existing = self._jobs.get(existing_id)
                if existing is not None and existing.status == "running":
                    return existing.copy()
            stamp = _now()
            job = JobRecord(
                id=secrets.token_hex(8),
                status="running",
                source=source,
                requested_output=output,
                created_at=stamp,
                updated_at=stamp,
            )
            self._jobs[job.id] = job
            self._active[key] = job.id
            job_id = job.id
            snapshot = job.copy()

        def on_progress(event: ProgressEvent) -> None:
            with self._lock:
                current = self._jobs.get(job_id)
                if current is None or current.status != "running":
                    return
                current.label = event.label
                current.fraction = event.fraction
                current.stages[event.stage] = event.label
                current.updated_at = _now()

        def target() -> None:
            try:
                written = run(job_id, on_progress)
            except Exception as exc:
                with self._lock:
                    current = self._jobs[job_id]
                    current.status = "failed"
                    current.error = str(exc) or exc.__class__.__name__
                    current.label = "Failed"
                    current.updated_at = _now()
                    if self._active.get(key) == job_id:
                        self._active.pop(key, None)
                return
            with self._lock:
                current = self._jobs[job_id]
                current.status = "succeeded"
                current.output_dir = str(written)
                current.label = "Finished"
                current.fraction = 1.0
                current.updated_at = _now()
                if self._active.get(key) == job_id:
                    self._active.pop(key, None)

        threading.Thread(target=target, name=f"transcribe-{job_id}", daemon=False).start()
        return snapshot


_BOARD = JobBoard()


def get_job(job_id: str) -> JobRecord | None:
    return _BOARD.get(job_id)


def list_jobs() -> list[JobRecord]:
    return _BOARD.list_jobs()


def reset_jobs_for_tests() -> None:
    global _BOARD
    _BOARD = JobBoard()


def start_transcription(
    source: str,
    output: Path,
    *,
    start: str | None,
    end: str | None,
    model: str,
    device: str,
    tempo: str,
    meter: str | None,
    subdivision: int,
    downbeat: str,
    staff_split: int,
) -> JobRecord:
    """Transcribe on a background thread.

    The folder is created before the model runs and removed if the run fails,
    so a stopped job does not leave a half-written project in the library.
    """

    settings: dict[str, object] = {
        "start": start or "",
        "end": end or "",
        "model": model,
        "device": device,
        "tempo": tempo or "auto",
        "meter": meter or "",
        "subdivision": int(subdivision),
        "downbeat": downbeat or "0",
        "staff_split": int(staff_split),
    }

    def run(job_id: str, on_progress: ProgressFn) -> Path:
        from baroque_transcribe.pipeline import (
            claim_fresh_output_dir,
            discard_incomplete,
            release_incomplete,
            run_transcription,
        )

        claimed = claim_fresh_output_dir(Path(output), job_id)
        try:
            result = run_transcription(
                source,
                claimed,
                start=start,
                end=end,
                model=model,
                device=device,
                tempo=tempo or "auto",
                time_signature=meter,
                subdivision=int(subdivision),
                downbeat=downbeat or "0",
                staff_split=int(staff_split),
                on_progress=on_progress,
                already_claimed=True,
            )
        except Exception:
            discard_incomplete(claimed, job_id)
            raise
        release_incomplete(claimed, job_id)
        return result.output_dir

    return _BOARD.start(source=source, output=str(Path(output)), settings=settings, run=run)
