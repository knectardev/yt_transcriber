"""In-process transcription jobs and the folders they own."""

from __future__ import annotations

import threading
import time
from pathlib import Path

from baroque_transcribe.defaults import INCOMPLETE_MARKER
from baroque_transcribe.jobs import JobBoard, get_job, reset_jobs_for_tests, start_transcription
from baroque_transcribe.library import disk_runs, find_project
from baroque_transcribe.pipeline import (
    ProjectResult,
    claim_fresh_output_dir,
    discard_incomplete,
    release_incomplete,
)
from baroque_transcribe.progress import ProgressEvent


def _wait(predicate, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("timed out")


def test_one_submission_runs_once_until_it_finishes():
    board = JobBoard()
    started = threading.Event()
    release = threading.Event()
    calls: list[str] = []

    def run(job_id: str, _on_progress) -> Path:
        calls.append(job_id)
        started.set()
        assert release.wait(5)
        return Path("transcriptions") / "piece"

    first = board.start(source="https://youtu.be/abcdefghijk", output="transcriptions/piece", settings={"start": ""}, run=run)
    assert started.wait(5)
    second = board.start(source="https://youtu.be/abcdefghijk", output="transcriptions/piece", settings={"start": ""}, run=run)
    assert second.id == first.id
    assert calls == [first.id]
    release.set()
    finished = _wait(lambda: (seen := board.get(first.id)) and seen.status == "succeeded" and seen)
    assert finished.output_dir.endswith("piece")
    third = board.start(source="https://youtu.be/abcdefghijk", output="transcriptions/piece", settings={"start": ""}, run=run)
    assert third.id != first.id
    release.set()


def test_a_failed_job_reports_the_error():
    board = JobBoard()

    def run(_job_id: str, on_progress) -> Path:
        on_progress(ProgressEvent("prepare", "Preparing", 0.1))
        raise RuntimeError("the download stopped")

    job = board.start(source="file", output="transcriptions/piece", settings={}, run=run)
    failed = _wait(lambda: (seen := board.get(job.id)) and seen.status == "failed" and seen)
    assert failed.error == "the download stopped"
    assert failed.stages["prepare"] == "Preparing"
    assert board.get(job.id).id == job.id


def test_claim_keeps_two_jobs_out_of_one_folder(tmp_path: Path):
    first = claim_fresh_output_dir(tmp_path / "piece", "job-a")
    second = claim_fresh_output_dir(tmp_path / "piece", "job-b")
    assert first != second
    assert (first / INCOMPLETE_MARKER).read_text(encoding="utf-8").strip() == "job-a"
    assert discard_incomplete(second, "job-a") is False
    assert second.is_dir()
    (first / "notes.json").write_text("{}", encoding="utf-8")
    assert disk_runs(tmp_path) == []
    assert find_project(first.name, root=tmp_path) is None
    assert discard_incomplete(first, "job-a") is True
    assert not first.exists()
    release_incomplete(second, "job-b")
    (second / "notes.json").write_text("{}", encoding="utf-8")
    assert discard_incomplete(second, "job-b") is False
    assert (second / "notes.json").is_file()


def test_a_failed_transcription_deletes_only_its_incomplete_folder(tmp_path: Path, monkeypatch):
    reset_jobs_for_tests()
    output = tmp_path / "piece"

    def boom(_source, output_dir, **_kwargs):
        (output_dir / "notes.json").write_text("partial", encoding="utf-8")
        (output_dir / "source.wav").write_bytes(b"not a real wav")
        raise RuntimeError("stopped")

    monkeypatch.setattr("baroque_transcribe.pipeline.run_transcription", boom)
    job = start_transcription(
        "https://youtu.be/abcdefghijk",
        output,
        start=None,
        end=None,
        model="bytedance",
        device="cpu",
        tempo="auto",
        meter=None,
        subdivision=16,
        downbeat="0",
        staff_split=60,
    )
    failed = _wait(lambda: (seen := get_job(job.id)) and seen.status == "failed" and seen)
    assert failed.error == "stopped"
    assert not output.exists()
    assert list(tmp_path.glob("piece*")) == []


def test_a_finished_transcription_keeps_the_project_and_drops_the_marker(tmp_path: Path, monkeypatch):
    reset_jobs_for_tests()
    output = tmp_path / "piece"

    def finish(_source, output_dir, **_kwargs):
        (output_dir / "notes.json").write_text("{}", encoding="utf-8")
        return ProjectResult(
            output_dir=output_dir,
            note_count=1,
            pedal_count=0,
            tempo_bpm=120.0,
            time_signature="4/4",
            warnings=[],
        )

    monkeypatch.setattr("baroque_transcribe.pipeline.run_transcription", finish)
    job = start_transcription(
        "https://youtu.be/abcdefghijk",
        output,
        start="0",
        end=None,
        model="bytedance",
        device="cpu",
        tempo="auto",
        meter=None,
        subdivision=16,
        downbeat="0",
        staff_split=60,
    )
    done = _wait(lambda: (seen := get_job(job.id)) and seen.status == "succeeded" and seen)
    assert Path(done.output_dir) == output
    assert (output / "notes.json").is_file()
    assert not (output / INCOMPLETE_MARKER).exists()
    again = start_transcription(
        "https://youtu.be/abcdefghijk",
        output,
        start="0",
        end=None,
        model="bytedance",
        device="cpu",
        tempo="auto",
        meter=None,
        subdivision=16,
        downbeat="0",
        staff_split=60,
    )
    assert again.id != job.id
