"""Polite, resumable download of catalog MIDI files into the local mirror.

Bulk sync is single-worker: one HTTPS GET at a time, a fresh uniform-random
delay between 30 and 200 seconds by default, Retry-After / exponential backoff
on transient failures, and immediate persistence of each verified file so a
restart skips completed entries. Optional ``keep_awake`` asks Windows (or a
POSIX no-op fallback) not to sleep for the duration of the run.
"""

from __future__ import annotations

import ctypes
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from baroque_transcribe.catalog import (
    CatalogEntry,
    CatalogError,
    CatalogManifest,
    catalog_midi_root,
    entry_midi_path,
    load_catalog,
)
from baroque_transcribe.defaults import (
    CATALOG_DELAY_MAX_SECONDS,
    CATALOG_DELAY_MIN_SECONDS,
    CATALOG_DOWNLOAD_TIMEOUT_SECONDS,
    CATALOG_LOG_DIR,
    CATALOG_MAX_ATTEMPTS,
    CATALOG_MAX_REQUESTS_PER_HOUR,
    CATALOG_SYNC_USER_AGENT,
    SOURCE_MID,
)
from baroque_transcribe.parse_store import repo_root
from baroque_transcribe.storage_io import sha256_file, write_bytes_atomic

RandomFn = Callable[[float, float], float]
SleepFn = Callable[[float], None]
ClockFn = Callable[[], float]
Logger = logging.Logger


class CatalogSyncError(CatalogError):
    pass


class CatalogAccessDenied(CatalogSyncError):
    """HTTP 403 / robots-style refusal; stop rather than bypass."""


@dataclass
class SyncResult:
    entry_id: str
    action: str
    path: Path | None = None
    error: str = ""
    attempts: int = 0
    waited_seconds: float = 0.0


@dataclass
class SyncReport:
    downloaded: list[SyncResult] = field(default_factory=list)
    skipped: list[SyncResult] = field(default_factory=list)
    failed: list[SyncResult] = field(default_factory=list)
    stopped_reason: str = ""

    @property
    def ok(self) -> bool:
        return not self.failed and not self.stopped_reason

    def summary(self) -> str:
        parts = [
            f"{len(self.downloaded)} downloaded",
            f"{len(self.skipped)} skipped",
            f"{len(self.failed)} failed",
        ]
        if self.stopped_reason:
            parts.append(f"stopped: {self.stopped_reason}")
        return ", ".join(parts)


@dataclass
class SyncSettings:
    delay_min: float = CATALOG_DELAY_MIN_SECONDS
    delay_max: float = CATALOG_DELAY_MAX_SECONDS
    max_attempts: int = CATALOG_MAX_ATTEMPTS
    max_requests_per_hour: int = CATALOG_MAX_REQUESTS_PER_HOUR
    timeout: float = CATALOG_DOWNLOAD_TIMEOUT_SECONDS
    overnight: bool = False
    keep_awake: bool = False
    force: bool = False
    user_agent: str = CATALOG_SYNC_USER_AGENT


class KeepAwake:
    """Best-effort request that the machine not sleep while syncing."""

    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_AWAYMODE_REQUIRED = 0x00000040

    def __init__(self, enabled: bool = False) -> None:
        self.enabled = bool(enabled)
        self._active = False

    def __enter__(self) -> "KeepAwake":
        if not self.enabled:
            return self
        if os.name == "nt":
            try:
                mask = self.ES_CONTINUOUS | self.ES_SYSTEM_REQUIRED | self.ES_AWAYMODE_REQUIRED
                ctypes.windll.kernel32.SetThreadExecutionState(mask)
                self._active = True
            except Exception:
                self._active = False
        return self

    def __exit__(self, *_exc) -> None:
        if not self._active:
            return
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(self.ES_CONTINUOUS)
        except Exception:
            pass
        self._active = False


def sample_delay(
    delay_min: float,
    delay_max: float,
    *,
    rng: RandomFn | None = None,
) -> float:
    low = float(delay_min)
    high = float(delay_max)
    if high < low:
        raise CatalogSyncError("delay_max must be >= delay_min.")
    if low < 0 or high < 0:
        raise CatalogSyncError("Delays must be non-negative.")
    picker = rng or random.uniform
    return float(picker(low, high))


def compute_wait(
    *,
    stochastic: float,
    retry_after: float | None,
    backoff: float,
) -> float:
    """Wait is the maximum of stochastic delay, Retry-After, and backoff."""

    values = [max(0.0, float(stochastic)), max(0.0, float(backoff))]
    if retry_after is not None:
        values.append(max(0.0, float(retry_after)))
    return max(values)


def parse_retry_after(value: str | None, *, now: float | None = None) -> float | None:
    if not value:
        return None
    text = value.strip()
    if not text:
        return None
    if text.isdigit():
        return float(text)
    try:
        from email.utils import parsedate_to_datetime

        when = parsedate_to_datetime(text)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        basis = now if now is not None else time.time()
        return max(0.0, when.timestamp() - basis)
    except (TypeError, ValueError, OverflowError, OSError):
        return None


def setup_sync_logger(root: Path | None = None) -> Logger:
    base = Path(root) if root is not None else repo_root()
    folder = base / CATALOG_LOG_DIR
    folder.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = folder / f"sync-{stamp}.log"
    logger = logging.getLogger(f"baroque_transcribe.catalog_sync.{stamp}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    logger.info("Catalog sync log %s", path)
    return logger


Hosts = list[str] | tuple[str, ...] | set[str]


def ensure_local_midi(
    entry: CatalogEntry,
    *,
    root: Path | None = None,
    settings: SyncSettings | None = None,
    allowed_hosts: Hosts | None = None,
    sleeper: SleepFn | None = None,
    rng: RandomFn | None = None,
    opener=None,
    logger: Logger | None = None,
    validate: bool = True,
) -> Path:
    """Download one entry immediately (no inter-request delay). Used for open-on-demand."""

    cfg = settings or SyncSettings(delay_min=0, delay_max=0, overnight=False)
    report = sync_entries(
        [entry],
        root=root,
        settings=SyncSettings(
            delay_min=0,
            delay_max=0,
            max_attempts=cfg.max_attempts,
            max_requests_per_hour=cfg.max_requests_per_hour,
            timeout=cfg.timeout,
            overnight=False,
            keep_awake=False,
            force=cfg.force,
            user_agent=cfg.user_agent,
        ),
        allowed_hosts=allowed_hosts,
        sleeper=sleeper or (lambda _seconds: None),
        rng=rng,
        opener=opener,
        logger=logger,
        validate=validate,
        polite_gaps=False,
    )
    if report.downloaded:
        path = report.downloaded[0].path
        if path is None:
            raise CatalogSyncError(f"Download of {entry.id} produced no file.")
        return path
    if report.skipped:
        path = report.skipped[0].path or entry_midi_path(entry.id, root=root)
        if path.is_file():
            return path
    detail = report.failed[0].error if report.failed else report.stopped_reason or "unknown error"
    raise CatalogSyncError(f"Could not download {entry.id}: {detail}")


def sync_missing(
    catalog: CatalogManifest | None = None,
    *,
    path: Path | None = None,
    root: Path | None = None,
    settings: SyncSettings | None = None,
    sleeper: SleepFn | None = None,
    rng: RandomFn | None = None,
    clock: ClockFn | None = None,
    opener=None,
    logger: Logger | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_progress: Callable[[int, int, SyncResult], None] | None = None,
    validate: bool = True,
) -> SyncReport:
    loaded = catalog if catalog is not None else load_catalog(path=path, root=root)
    cfg = settings or SyncSettings(overnight=True)
    missing = [
        entry
        for entry in loaded.entries
        if cfg.force or not _local_ok(entry, root=root)
    ]
    return sync_entries(
        missing,
        root=root,
        settings=cfg,
        allowed_hosts=loaded.source.allowed_hosts,
        sleeper=sleeper,
        rng=rng,
        clock=clock,
        opener=opener,
        logger=logger,
        should_stop=should_stop,
        on_progress=on_progress,
        validate=validate,
        polite_gaps=True,
    )


def sync_entries(
    entries: list[CatalogEntry],
    *,
    root: Path | None = None,
    settings: SyncSettings | None = None,
    allowed_hosts: Hosts | None = None,
    sleeper: SleepFn | None = None,
    rng: RandomFn | None = None,
    clock: ClockFn | None = None,
    opener=None,
    logger: Logger | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_progress: Callable[[int, int, SyncResult], None] | None = None,
    validate: bool = True,
    polite_gaps: bool = True,
) -> SyncReport:
    cfg = settings or SyncSettings()
    sleep = sleeper or time.sleep
    now = clock or time.time
    log = logger or logging.getLogger("baroque_transcribe.catalog_sync")
    hosts = {
        str(host).strip().lower()
        for host in (allowed_hosts or ())
        if str(host).strip()
    }
    report = SyncReport()
    request_times: list[float] = []
    total = len(entries)

    with KeepAwake(cfg.keep_awake):
        if cfg.keep_awake:
            log.info("Keep-awake requested for this sync process.")
        for index, entry in enumerate(entries, start=1):
            if should_stop and should_stop():
                report.stopped_reason = "interrupted"
                log.warning("Sync interrupted before %s", entry.id)
                break
            if not cfg.force and _local_ok(entry, root=root):
                result = SyncResult(
                    entry_id=entry.id,
                    action="skipped",
                    path=entry_midi_path(entry.id, root=root),
                )
                report.skipped.append(result)
                if on_progress:
                    on_progress(index, total, result)
                continue

            if polite_gaps and (report.downloaded or report.failed or index > 1):
                wait = sample_delay(cfg.delay_min, cfg.delay_max, rng=rng)
                log.info("Waiting %.1fs before %s", wait, entry.id)
                _interruptible_sleep(wait, sleep, should_stop)
                if should_stop and should_stop():
                    report.stopped_reason = "interrupted"
                    break

            try:
                result = _download_entry(
                    entry,
                    root=root,
                    settings=cfg,
                    allowed_hosts=hosts,
                    sleeper=sleep,
                    rng=rng,
                    clock=now,
                    opener=opener,
                    logger=log,
                    validate=validate,
                    request_times=request_times,
                    should_stop=should_stop,
                )
            except CatalogAccessDenied as exc:
                result = SyncResult(entry_id=entry.id, action="failed", error=str(exc))
                report.failed.append(result)
                report.stopped_reason = str(exc)
                log.error("Access denied for %s: %s", entry.id, exc)
                if on_progress:
                    on_progress(index, total, result)
                break
            except Exception as exc:  # noqa: BLE001 - surface as failed item
                result = SyncResult(entry_id=entry.id, action="failed", error=str(exc))
                report.failed.append(result)
                log.error("Failed %s: %s", entry.id, exc)
                if on_progress:
                    on_progress(index, total, result)
                continue

            if result.action == "downloaded":
                report.downloaded.append(result)
            elif result.action == "skipped":
                report.skipped.append(result)
            else:
                report.failed.append(result)
            if on_progress:
                on_progress(index, total, result)
    return report


def fill_manifest_hashes(
    entries: list[dict],
    *,
    overnight: bool = True,
    root: Path | None = None,
) -> None:
    """Curator helper used by ``scripts/refresh_bach_catalog.py --with-hashes``."""

    catalog_entries = [
        CatalogEntry(
            id=str(item["id"]),
            bwv=str(item.get("bwv") or ""),
            title=str(item.get("title") or ""),
            collection_id=str(item.get("collection_id") or ""),
            collection=str(item.get("collection") or ""),
            order_in_collection=int(item.get("order_in_collection") or 0),
            key=str(item.get("key") or ""),
            is_variant=bool(item.get("is_variant")),
            is_arrangement=bool(item.get("is_arrangement")),
            source_page=str(item.get("source_page") or ""),
            midi_url=str(item["midi_url"]),
            source_filename=str(item.get("source_filename") or ""),
            sha256="",
            import_defaults=dict(item.get("import_defaults") or {}),
        )
        for item in entries
    ]
    settings = SyncSettings(overnight=overnight, keep_awake=True, force=True)
    logger = setup_sync_logger(root)
    report = sync_entries(
        catalog_entries,
        root=root,
        settings=settings,
        allowed_hosts={"tobis-notenarchiv.de", "www.tobis-notenarchiv.de"},
        logger=logger,
        validate=True,
    )
    by_id = {item["id"]: item for item in entries}
    for result in report.downloaded + report.skipped:
        path = result.path or entry_midi_path(result.entry_id, root=root)
        if not path.is_file():
            continue
        item = by_id[result.entry_id]
        item["sha256"] = sha256_file(path)
        item["size_bytes"] = path.stat().st_size
        try:
            from baroque_transcribe.midi_import import parse_midi

            transcription, timing = parse_midi(path)
            item["duration_seconds"] = round(timing.end_time, 5)
            item["note_count"] = len(transcription.notes)
            item["track_count"] = 2 if timing.track_staffs else 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Audit parse failed for %s: %s", result.entry_id, exc)
    if report.failed or report.stopped_reason:
        raise CatalogSyncError(report.summary())


def _local_ok(entry: CatalogEntry, *, root: Path | None) -> bool:
    path = entry_midi_path(entry.id, root=root)
    if not path.is_file():
        return False
    if not entry.sha256:
        return True
    try:
        return sha256_file(path).lower() == entry.sha256.lower()
    except OSError:
        return False


def _download_entry(
    entry: CatalogEntry,
    *,
    root: Path | None,
    settings: SyncSettings,
    allowed_hosts: set[str],
    sleeper: SleepFn,
    rng: RandomFn | None,
    clock: ClockFn,
    opener,
    logger: Logger,
    validate: bool,
    request_times: list[float],
    should_stop: Callable[[], bool] | None,
) -> SyncResult:
    _assert_allowed_url(entry.midi_url, allowed_hosts)
    dest = entry_midi_path(entry.id, root=root)
    dest.parent.mkdir(parents=True, exist_ok=True)
    attempts = 0
    waited = 0.0
    last_error = ""
    open_url = opener or urlopen

    while attempts < settings.max_attempts:
        if should_stop and should_stop():
            return SyncResult(
                entry_id=entry.id,
                action="failed",
                error="interrupted",
                attempts=attempts,
                waited_seconds=waited,
            )
        _throttle(request_times, settings.max_requests_per_hour, clock, sleeper, should_stop)
        attempts += 1
        request_times.append(clock())
        request = Request(
            entry.midi_url,
            headers={
                "User-Agent": settings.user_agent,
                "Accept": "audio/midi,audio/x-midi,application/octet-stream,*/*",
            },
        )
        try:
            with open_url(request, timeout=settings.timeout) as response:
                status = getattr(response, "status", None) or response.getcode()
                if status in {403, 451}:
                    raise CatalogAccessDenied(
                        f"Server refused download of {entry.id} with HTTP {status}."
                    )
                if status and int(status) >= 400:
                    raise CatalogSyncError(f"HTTP {status} for {entry.id}")
                payload = response.read()
            _validate_midi_bytes(payload, entry=entry, validate=validate)
            write_bytes_atomic(dest, payload)
            digest = sha256_file(dest)
            if entry.sha256 and digest.lower() != entry.sha256.lower():
                dest.unlink(missing_ok=True)
                raise CatalogSyncError(
                    f"SHA-256 mismatch for {entry.id}: got {digest}, expected {entry.sha256}."
                )
            logger.info("Downloaded %s -> %s (%s bytes)", entry.id, dest, len(payload))
            return SyncResult(
                entry_id=entry.id,
                action="downloaded",
                path=dest,
                attempts=attempts,
                waited_seconds=waited,
            )
        except CatalogAccessDenied:
            raise
        except HTTPError as exc:
            if exc.code in {403, 451}:
                raise CatalogAccessDenied(
                    f"Server refused download of {entry.id} with HTTP {exc.code}."
                ) from exc
            retry_after = parse_retry_after(exc.headers.get("Retry-After") if exc.headers else None)
            last_error = f"HTTP {exc.code}: {exc.reason}"
            if exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                break
            wait = compute_wait(
                stochastic=sample_delay(settings.delay_min, settings.delay_max, rng=rng),
                retry_after=retry_after,
                backoff=_backoff_seconds(attempts, rng=rng),
            )
            waited += wait
            logger.warning("Retrying %s after %.1fs (%s)", entry.id, wait, last_error)
            _interruptible_sleep(wait, sleeper, should_stop)
        except (URLError, TimeoutError, CatalogSyncError, OSError, ValueError) as exc:
            last_error = str(exc)
            wait = compute_wait(
                stochastic=sample_delay(settings.delay_min, settings.delay_max, rng=rng),
                retry_after=None,
                backoff=_backoff_seconds(attempts, rng=rng),
            )
            waited += wait
            logger.warning("Retrying %s after %.1fs (%s)", entry.id, wait, last_error)
            _interruptible_sleep(wait, sleeper, should_stop)
    return SyncResult(
        entry_id=entry.id,
        action="failed",
        error=last_error or "download failed",
        attempts=attempts,
        waited_seconds=waited,
    )


def _assert_allowed_url(url: str, allowed_hosts: set[str]) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise CatalogSyncError("Catalog downloads must use HTTPS.")
    host = (parsed.hostname or "").lower()
    if allowed_hosts and host not in allowed_hosts:
        raise CatalogSyncError(f"Host {host!r} is not in the catalog allowlist.")


def _validate_midi_bytes(payload: bytes, *, entry: CatalogEntry, validate: bool) -> None:
    if len(payload) < 14 or not payload.startswith(b"MThd"):
        raise CatalogSyncError(f"{entry.id} did not look like a MIDI file.")
    if not validate:
        return
    staging_root = catalog_midi_root() / ".tmp-validate"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging = staging_root / f"{entry.id}-{os.getpid()}.mid"
    try:
        write_bytes_atomic(staging, payload)
        from baroque_transcribe.midi_import import parse_midi

        parse_midi(staging)
    finally:
        staging.unlink(missing_ok=True)


def _backoff_seconds(attempt: int, *, rng: RandomFn | None = None) -> float:
    base = min(600.0, 2.0 ** max(0, attempt - 1))
    jitter = sample_delay(0.0, min(30.0, base), rng=rng) if base else 0.0
    return base + jitter


def _throttle(
    request_times: list[float],
    max_per_hour: int,
    clock: ClockFn,
    sleeper: SleepFn,
    should_stop: Callable[[], bool] | None,
) -> None:
    if max_per_hour <= 0:
        return
    while True:
        now = clock()
        request_times[:] = [stamp for stamp in request_times if now - stamp < 3600.0]
        if len(request_times) < max_per_hour:
            return
        wait = max(0.1, 3600.0 - (now - request_times[0]) + 0.1)
        _interruptible_sleep(wait, sleeper, should_stop)
        if should_stop and should_stop():
            return


def _interruptible_sleep(
    seconds: float,
    sleeper: SleepFn,
    should_stop: Callable[[], bool] | None,
) -> None:
    remaining = max(0.0, float(seconds))
    while remaining > 0:
        if should_stop and should_stop():
            return
        step = min(1.0, remaining)
        sleeper(step)
        remaining -= step
