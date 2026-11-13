"""Polite catalog download scheduler: delays, backoff, resume, keep-awake."""

from __future__ import annotations

from pathlib import Path

import pretty_midi

from baroque_transcribe.catalog import CatalogEntry
from baroque_transcribe.catalog_sync import (
    KeepAwake,
    SyncSettings,
    compute_wait,
    ensure_local_midi,
    parse_retry_after,
    sample_delay,
    sync_entries,
)
from baroque_transcribe.defaults import SOURCE_MID


def _write_midi(path: Path) -> bytes:
    midi = pretty_midi.PrettyMIDI(initial_tempo=90)
    inst = pretty_midi.Instrument(program=0, name="Piano")
    inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.4))
    midi.instruments.append(inst)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    return path.read_bytes()


class _FakeResponse:
    def __init__(self, payload: bytes, status: int = 200, headers: dict | None = None):
        self._payload = payload
        self.status = status
        self.headers = headers or {}

    def read(self):
        return self._payload

    def getcode(self):
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


def _entry(entry_id: str = "bwv-0772") -> CatalogEntry:
    return CatalogEntry(
        id=entry_id,
        bwv="772",
        title="Invention No. 1",
        collection_id="bwv-772-786-inventions",
        collection="Inventions",
        order_in_collection=1,
        midi_url=f"https://example.test/{entry_id}.mid",
        source_filename=f"{entry_id}.mid",
        sha256="",
    )


def test_sample_delay_and_wait_bounds():
    values = [sample_delay(30, 200, rng=lambda low, high: low) for _ in range(3)]
    assert values == [30, 30, 30]
    values = [sample_delay(30, 200, rng=lambda low, high: high) for _ in range(3)]
    assert values == [200, 200, 200]
    assert compute_wait(stochastic=40, retry_after=120, backoff=10) == 120
    assert compute_wait(stochastic=40, retry_after=None, backoff=55) == 55
    assert parse_retry_after("90") == 90.0


def test_sync_resumes_and_uses_polite_gaps(tmp_path: Path):
    payload_path = tmp_path / "sample.mid"
    payload = _write_midi(payload_path)
    slept: list[float] = []
    calls = {"n": 0}

    def opener(request, timeout=0):
        calls["n"] += 1
        return _FakeResponse(payload)

    entries = [_entry("bwv-0772"), _entry("bwv-0773")]
    settings = SyncSettings(delay_min=30, delay_max=30, keep_awake=False)
    report = sync_entries(
        entries,
        root=tmp_path,
        settings=settings,
        allowed_hosts={"example.test"},
        sleeper=slept.append,
        rng=lambda low, high: low,
        opener=opener,
        validate=True,
        polite_gaps=True,
    )
    assert len(report.downloaded) == 2
    assert slept  # gap before second download (may be chunked into 1s steps)
    assert sum(slept) >= 30
    assert (tmp_path / "catalog" / "midi" / "bwv-0772" / SOURCE_MID).is_file()

    slept.clear()
    calls["n"] = 0
    report2 = sync_entries(
        entries,
        root=tmp_path,
        settings=settings,
        allowed_hosts={"example.test"},
        sleeper=slept.append,
        opener=opener,
        validate=True,
        polite_gaps=True,
    )
    assert len(report2.skipped) == 2
    assert calls["n"] == 0


def test_access_denied_stops_bulk_sync(tmp_path: Path):
    def opener_forbidden(request, timeout=0):
        return _FakeResponse(b"nope", status=403)

    report = sync_entries(
        [_entry("bwv-0772"), _entry("bwv-0773")],
        root=tmp_path,
        settings=SyncSettings(delay_min=0, delay_max=0, max_attempts=1),
        allowed_hosts={"example.test"},
        sleeper=lambda _s: None,
        opener=opener_forbidden,
        validate=False,
        polite_gaps=False,
    )
    assert report.stopped_reason
    assert len(report.failed) == 1
    assert len(report.downloaded) == 0


def test_ensure_local_midi_immediate(tmp_path: Path):
    payload = _write_midi(tmp_path / "seed.mid")

    def opener(request, timeout=0):
        return _FakeResponse(payload)

    path = ensure_local_midi(
        _entry("bwv-0772"),
        root=tmp_path,
        allowed_hosts={"example.test"},
        opener=opener,
        sleeper=lambda _s: None,
        validate=True,
    )
    assert path.is_file()
    assert path.read_bytes().startswith(b"MThd")


def test_keep_awake_context_does_not_raise():
    with KeepAwake(False):
        pass
    with KeepAwake(True):
        pass


def test_host_allowlist(tmp_path: Path):
    report = sync_entries(
        [_entry()],
        root=tmp_path,
        settings=SyncSettings(delay_min=0, delay_max=0, max_attempts=1),
        allowed_hosts={"other.test"},
        sleeper=lambda _s: None,
        opener=lambda *a, **k: _FakeResponse(b"MThd"),
        validate=False,
        polite_gaps=False,
    )
    assert report.failed
    assert "allowlist" in report.failed[0].error
