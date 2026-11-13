"""A SQLite index of the fragment library, for searching.

The fragment folders are the library. This file is only a way to query
them, and it can be deleted at any time: every row is derived from a
``fragment.json``, and nothing is stored here that is not stored there.
The index lives in ``.local/library.sqlite`` and is not committed.

Reconciling makes the index agree with the folders. There is one rule for
what the index should contain, and it looks only at the folders as they are
now, never at what the index held before:

* A folder whose manifest cannot be read, or is not valid, is not indexed.
  It is reported as a problem.
* When two folders carry the same fragment id, one is indexed: the folder
  named after the id, or else the first by name. The others are reported.
* Every other folder with a manifest is indexed.

Any row that the rule does not produce is deleted in the same transaction,
whatever the reason: the folder was removed, the manifest was damaged, or
another folder now owns the id. So a manifest that was indexed and has since
become unreadable loses its row at the next reconcile, and is reported.

The three ways to reconcile differ only in how much they read:

* ``incremental`` reads a manifest only when its modification time or size
  changed since it was last read. Everything else is taken from what was
  recorded at that reading.
* ``full`` reads and hashes every manifest.
* ``rebuild`` deletes the index file and reads every manifest.

All three apply the same rule to the same set of folders, so they end with
the same rows.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

from baroque_transcribe.defaults import FRAGMENT_JSON, LIBRARY_INDEX
from baroque_transcribe.fragments import FragmentError, fragments_root, local_root, validate_manifest

INDEX_SCHEMA_VERSION = 2
# The library page reconciles at most this often while it is being redrawn.
RECONCILE_INTERVAL_SECONDS = 10.0

_SCHEMA = """
CREATE TABLE fragments (
    id TEXT PRIMARY KEY,
    rel_dir TEXT NOT NULL UNIQUE,
    manifest_sha256 TEXT NOT NULL,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    practice_notes TEXT NOT NULL,
    composer TEXT NOT NULL,
    performer TEXT NOT NULL,
    source_kind TEXT NOT NULL,
    source_title TEXT NOT NULL,
    source_url TEXT NOT NULL,
    video_id TEXT NOT NULL,
    parent_project TEXT NOT NULL,
    parent_event_id TEXT NOT NULL,
    catalog_entry_id TEXT NOT NULL,
    catalog_collection TEXT NOT NULL,
    video_start REAL NOT NULL,
    video_end REAL NOT NULL,
    working_start REAL NOT NULL,
    working_end REAL NOT NULL,
    duration REAL NOT NULL,
    sync_offset REAL NOT NULL,
    analyzed_key_tonic TEXT NOT NULL,
    analyzed_key_mode TEXT NOT NULL,
    time_signature TEXT NOT NULL,
    bpm REAL NOT NULL,
    measures INTEGER NOT NULL,
    measure_start INTEGER NOT NULL,
    measure_end INTEGER NOT NULL,
    note_count INTEGER NOT NULL,
    harmony_roman TEXT NOT NULL,
    harmony_symbols TEXT NOT NULL,
    created_at TEXT NOT NULL,
    modified_at TEXT NOT NULL
);
CREATE TABLE tags (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL UNIQUE COLLATE NOCASE
);
CREATE TABLE fragment_tags (
    fragment_id TEXT NOT NULL REFERENCES fragments(id) ON DELETE CASCADE,
    tag_id INTEGER NOT NULL REFERENCES tags(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (fragment_id, tag_id)
);
CREATE INDEX fragment_tags_by_tag ON fragment_tags(tag_id);
-- Every folder seen at the last reconcile, valid or not, with the state of
-- its manifest file when it was read. This is what lets an incremental pass
-- skip a manifest that has not changed, and what the library page shows as
-- problems without scanning again.
CREATE TABLE folders (
    rel_dir TEXT PRIMARY KEY,
    mtime_ns INTEGER NOT NULL,
    size INTEGER NOT NULL,
    fragment_id TEXT,
    invalid TEXT,
    problem TEXT
);
"""

_COLUMNS = (
    "id",
    "rel_dir",
    "manifest_sha256",
    "title",
    "description",
    "practice_notes",
    "composer",
    "performer",
    "source_kind",
    "source_title",
    "source_url",
    "video_id",
    "parent_project",
    "parent_event_id",
    "catalog_entry_id",
    "catalog_collection",
    "video_start",
    "video_end",
    "working_start",
    "working_end",
    "duration",
    "sync_offset",
    "analyzed_key_tonic",
    "analyzed_key_mode",
    "time_signature",
    "bpm",
    "measures",
    "measure_start",
    "measure_end",
    "note_count",
    "harmony_roman",
    "harmony_symbols",
    "created_at",
    "modified_at",
)

SORTS = {
    "modified": "f.modified_at DESC, f.title COLLATE NOCASE, f.id",
    "created": "f.created_at DESC, f.title COLLATE NOCASE, f.id",
    "title": "f.title COLLATE NOCASE, f.id",
    "source": "f.source_title COLLATE NOCASE, f.video_start, f.id",
    "duration": "f.duration, f.title COLLATE NOCASE, f.id",
}


class FragmentIndexError(ValueError):
    pass


@dataclass(frozen=True)
class IndexProblem:
    """A folder that is not in the index, and why.

    ``was_indexed`` is True when the folder had a row before this reconcile
    and the row was removed by it.
    """

    folder: str
    message: str
    was_indexed: bool = False


@dataclass
class ReconcileReport:
    mode: str
    folders: int = 0
    indexed: int = 0
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    problems: list[IndexProblem] = field(default_factory=list)
    # How many manifests were opened and parsed by this pass.
    manifests_read: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.added or self.updated or self.removed)

    def summary(self) -> str:
        parts = [f"{self.indexed} fragment{'' if self.indexed == 1 else 's'} indexed"]
        if self.added:
            parts.append(f"{len(self.added)} added")
        if self.updated:
            parts.append(f"{len(self.updated)} updated")
        if self.removed:
            parts.append(f"{len(self.removed)} removed")
        if self.problems:
            count = len(self.problems)
            parts.append(f"{count} folder{'' if count == 1 else 's'} could not be indexed")
        return ", ".join(parts) + "."


@dataclass(frozen=True)
class IndexedFragment:
    id: str
    rel_dir: str
    title: str
    description: str
    practice_notes: str
    composer: str
    performer: str
    source_kind: str
    source_title: str
    source_url: str
    video_id: str
    parent_project: str
    catalog_entry_id: str
    catalog_collection: str
    video_start: float
    video_end: float
    duration: float
    sync_offset: float
    key_tonic: str
    key_mode: str
    time_signature: str
    bpm: float
    measures: int
    measure_start: int
    measure_end: int
    note_count: int
    roman: tuple[str, ...]
    symbols: tuple[str, ...]
    tags: tuple[str, ...]
    created_at: str
    modified_at: str

    @property
    def key(self) -> str:
        return f"{self.key_tonic} {self.key_mode}".strip()


# --------------------------------------------------------------------------
# The database file
# --------------------------------------------------------------------------


def index_path(db: Path | None = None) -> Path:
    """``.local/library.sqlite`` in this repository, or an explicit file."""

    return Path(db) if db is not None else local_root() / LIBRARY_INDEX


def _open(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path), timeout=10.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def _has_schema(connection: sqlite3.Connection) -> bool:
    version = int(connection.execute("PRAGMA user_version").fetchone()[0])
    if version != INDEX_SCHEMA_VERSION:
        return False
    tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    return {"fragments", "tags", "fragment_tags", "folders"} <= tables


def _create(path: Path) -> sqlite3.Connection:
    """A new, empty index at ``path``, replacing whatever is there."""

    path.parent.mkdir(parents=True, exist_ok=True)
    removed = True
    for leftover in (path, path.with_name(path.name + "-journal"), path.with_name(path.name + "-wal")):
        try:
            leftover.unlink(missing_ok=True)
        except OSError:
            # Windows refuses to delete a file another reader has open.
            removed = False
    connection = _open(path)
    if not removed:
        # Empty the file in place instead.
        connection.execute("PRAGMA foreign_keys = OFF")
        names = connection.execute(
            "SELECT type, name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%' AND type IN ('table', 'index')"
        ).fetchall()
        for row in names:
            if row["type"] == "table":
                connection.execute(f'DROP TABLE IF EXISTS "{row["name"]}"')
        connection.execute("PRAGMA foreign_keys = ON")
    connection.executescript(_SCHEMA)
    connection.execute(f"PRAGMA user_version = {INDEX_SCHEMA_VERSION}")
    return connection


def _connect(path: Path, *, rebuild: bool = False) -> sqlite3.Connection:
    """Open the index, creating it when it is missing, unreadable, or from another version.

    The file holds nothing that the fragment folders do not, so the answer
    to a damaged or outdated index is a new one.
    """

    if rebuild or not path.is_file():
        return _create(path)
    connection: sqlite3.Connection | None = None
    try:
        connection = _open(path)
        if _has_schema(connection):
            return connection
    except sqlite3.DatabaseError:
        pass
    if connection is not None:
        connection.close()
    return _create(path)


@contextmanager
def _reading(db: Path | None) -> Iterator[sqlite3.Connection | None]:
    """A connection for a query, or None when there is no usable index yet."""

    path = index_path(db)
    if not path.is_file():
        yield None
        return
    connection: sqlite3.Connection | None = None
    try:
        connection = _open(path)
        usable = _has_schema(connection)
    except sqlite3.DatabaseError:
        usable = False
    try:
        yield connection if usable else None
    finally:
        if connection is not None:
            connection.close()


# --------------------------------------------------------------------------
# Reading one folder
# --------------------------------------------------------------------------


@dataclass
class _Folder:
    rel_dir: str
    mtime_ns: int
    size: int
    fragment_id: str | None = None
    invalid: str | None = None
    row: dict | None = None
    tags: list[str] | None = None


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _real(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def _whole(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return int(value)


def harmony_text(labels) -> str:
    """Chord labels as one searchable string: `` V7 i iv `` with a space at each end.

    A label repeated straight away is stored once, so a progression search
    is not defeated by a chord that was re-labelled in the next measure.
    """

    tokens: list[str] = []
    for label in labels or []:
        token = "".join(str(label).split())
        if token and (not tokens or tokens[-1] != token):
            tokens.append(token)
    return f" {' '.join(tokens)} " if tokens else ""


def _manifest_row(manifest: dict, rel_dir: str, digest: str) -> dict:
    source = manifest.get("source") if isinstance(manifest.get("source"), dict) else {}
    parent = manifest.get("parent") if isinstance(manifest.get("parent"), dict) else {}
    time_block = manifest["time"]
    sync = manifest.get("sync") if isinstance(manifest.get("sync"), dict) else {}
    music = manifest.get("music") if isinstance(manifest.get("music"), dict) else {}
    key = music.get("analyzed_key") if isinstance(music.get("analyzed_key"), dict) else {}
    harmony = manifest.get("harmony") if isinstance(manifest.get("harmony"), dict) else {}
    boundary = manifest.get("boundary") if isinstance(manifest.get("boundary"), dict) else {}
    return {
        "id": manifest["id"],
        "rel_dir": rel_dir,
        "manifest_sha256": digest,
        "title": manifest["title"],
        "description": _text(manifest.get("description")),
        "practice_notes": _text(manifest.get("practice_notes")),
        "composer": _text(manifest.get("composer")),
        "performer": _text(manifest.get("performer")),
        "source_kind": _text(source.get("kind")),
        "source_title": _text(source.get("title")) or _text(source.get("name")),
        "source_url": _text(source.get("url")),
        "video_id": _text(source.get("video_id")),
        "parent_project": _text(parent.get("project")),
        "parent_event_id": _text(parent.get("event_id")),
        "catalog_entry_id": _text(
            (manifest.get("catalog") or {}).get("entry_id")
            if isinstance(manifest.get("catalog"), dict)
            else ""
        ),
        "catalog_collection": _text(
            (manifest.get("catalog") or {}).get("collection")
            if isinstance(manifest.get("catalog"), dict)
            else ""
        ),
        "video_start": _real(time_block["source_video"]["start_seconds"]),
        "video_end": _real(time_block["source_video"]["end_seconds"]),
        "working_start": _real(time_block["working_audio"]["start_seconds"]),
        "working_end": _real(time_block["working_audio"]["end_seconds"]),
        "duration": _real(time_block["duration_seconds"]),
        "sync_offset": _real(sync.get("offset_seconds")),
        "analyzed_key_tonic": _text(key.get("tonic")),
        "analyzed_key_mode": _text(key.get("mode")),
        "time_signature": _text(music.get("time_signature")),
        "bpm": _real(music.get("bpm")),
        "measures": _whole(music.get("measures")),
        "measure_start": _whole(parent.get("measure_start")),
        "measure_end": _whole(parent.get("measure_end")),
        "note_count": _whole(boundary.get("notes")),
        "harmony_roman": harmony_text(harmony.get("roman") if isinstance(harmony.get("roman"), list) else []),
        "harmony_symbols": harmony_text(harmony.get("symbols") if isinstance(harmony.get("symbols"), list) else []),
        "created_at": _text(manifest.get("created_at")),
        "modified_at": _text(manifest.get("modified_at")),
    }


def _clean_tags(tags) -> list[str]:
    from baroque_transcribe.fragments import normalize_tags

    return normalize_tags(tags)


def _read_folder(base: Path, folder: _Folder) -> None:
    """Open and parse one manifest. The outcome is stored on ``folder``."""

    folder.fragment_id = None
    folder.invalid = None
    folder.row = None
    folder.tags = None
    path = base / folder.rel_dir / FRAGMENT_JSON
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        folder.invalid = f"Could not read {FRAGMENT_JSON}: {exc.strerror or exc}"
        return
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        folder.invalid = f"{FRAGMENT_JSON} is not valid JSON: {exc}"
        return
    try:
        manifest = validate_manifest(payload)
    except FragmentError as exc:
        folder.invalid = str(exc)
        return
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    folder.fragment_id = str(manifest["id"])
    folder.row = _manifest_row(manifest, folder.rel_dir, digest)
    folder.tags = _clean_tags(manifest.get("tags"))


_NO_MANIFEST = f"This folder has no {FRAGMENT_JSON}, so it is not a fragment."


def _scan(base: Path) -> list[_Folder]:
    """Every folder under the library root, with its manifest's time and size."""

    found: list[_Folder] = []
    if not base.is_dir():
        return found
    for entry in sorted(base.iterdir(), key=lambda item: item.name):
        # Names that begin with a dot are the staging folders of an extraction in progress.
        if entry.name.startswith(".") or not entry.is_dir():
            continue
        try:
            status = (entry / FRAGMENT_JSON).stat()
        except OSError:
            # A folder without a manifest is not a fragment. It is reported, so
            # a fragment whose manifest was deleted does not vanish silently.
            found.append(_Folder(rel_dir=entry.name, mtime_ns=0, size=0, invalid=_NO_MANIFEST))
            continue
        found.append(_Folder(rel_dir=entry.name, mtime_ns=int(status.st_mtime_ns), size=int(status.st_size)))
    return found


# --------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------


def _resolve(folders: list[_Folder]) -> tuple[dict[str, _Folder], dict[str, str]]:
    """Which folder each fragment id is indexed from, and why the rest are not.

    Depends only on each folder's name and on its id or its error. It does
    not look at the index.
    """

    winners: dict[str, _Folder] = {}
    problems: dict[str, str] = {}
    by_id: dict[str, list[_Folder]] = {}
    for folder in folders:
        if folder.invalid is not None or folder.fragment_id is None:
            problems[folder.rel_dir] = folder.invalid or "The manifest has no fragment id."
        else:
            by_id.setdefault(folder.fragment_id, []).append(folder)
    for fragment_id, claimants in by_id.items():
        claimants.sort(key=lambda item: (item.rel_dir != fragment_id, item.rel_dir))
        winners[fragment_id] = claimants[0]
        for loser in claimants[1:]:
            problems[loser.rel_dir] = (
                f"This folder has the same fragment id as {claimants[0].rel_dir} ({fragment_id}). "
                "Only that one is in the library. Delete one of the two, or give this one a new id."
            )
    return winners, problems


# --------------------------------------------------------------------------
# Reconcile
# --------------------------------------------------------------------------

_LOCK = threading.Lock()
_LAST_RECONCILE: dict[str, float] = {}


def reconcile(
    root: Path | None = None,
    db: Path | None = None,
    *,
    mode: str = "incremental",
    reread: tuple[str, ...] | list[str] = (),
) -> ReconcileReport:
    """Make the index agree with the fragment folders.

    ``mode`` is ``incremental``, ``full`` or ``rebuild``. ``reread`` names
    folders to read even when their manifest looks unchanged; the program
    passes the folder it has just written.
    """

    if mode not in {"incremental", "full", "rebuild"}:
        raise FragmentIndexError(f"Unknown reconcile mode {mode!r}.")
    base = fragments_root(root)
    path = index_path(db)
    forced = {str(name) for name in reread}
    report = ReconcileReport(mode=mode)
    with _LOCK:
        connection = _connect(path, rebuild=mode == "rebuild")
        try:
            connection.execute("BEGIN IMMEDIATE")
            try:
                _reconcile(connection, base, report, read_all=mode != "incremental", forced=forced)
            except BaseException:
                connection.execute("ROLLBACK")
                raise
            connection.execute("COMMIT")
        finally:
            connection.close()
        _LAST_RECONCILE[str(path.resolve())] = time.monotonic()
    return report


def _reconcile(
    connection: sqlite3.Connection,
    base: Path,
    report: ReconcileReport,
    *,
    read_all: bool,
    forced: set[str],
) -> None:
    folders = _scan(base)
    report.folders = len(folders)
    remembered = {row["rel_dir"]: row for row in connection.execute("SELECT * FROM folders")}
    indexed = {
        row["id"]: (row["rel_dir"], row["manifest_sha256"])
        for row in connection.execute("SELECT id, rel_dir, manifest_sha256 FROM fragments")
    }

    # 1. Learn each folder's id or error: from the last reading when the
    #    manifest file is unchanged, otherwise by reading it now.
    for folder in folders:
        if folder.invalid == _NO_MANIFEST:
            continue  # nothing to read
        seen = remembered.get(folder.rel_dir)
        unchanged = (
            seen is not None
            and int(seen["mtime_ns"]) == folder.mtime_ns
            and int(seen["size"]) == folder.size
            and (seen["fragment_id"] is not None or seen["invalid"] is not None)
        )
        if unchanged and not read_all and folder.rel_dir not in forced:
            folder.fragment_id = seen["fragment_id"]
            folder.invalid = seen["invalid"]
        else:
            _read_folder(base, folder)
            report.manifests_read += 1

    # 2. Apply the rule. It sees the folders only.
    winners, problems = _resolve(folders)

    # 3. A winner that was not read in step 1 is read now when the index does
    #    not already hold it from that folder. That is a folder which used to
    #    lose a duplicate id and now owns it. The id can turn out to be
    #    different or the manifest unreadable (the file changed without its
    #    time or size changing), so the rule is applied again until it settles.
    while True:
        pending = [
            folder
            for fragment_id, folder in winners.items()
            if folder.row is None and indexed.get(fragment_id, ("", ""))[0] != folder.rel_dir
        ]
        if not pending:
            break
        for folder in pending:
            _read_folder(base, folder)
            report.manifests_read += 1
        winners, problems = _resolve(folders)

    # 4. Delete every row the rule did not produce.
    for fragment_id in sorted(indexed):
        if fragment_id not in winners:
            connection.execute("DELETE FROM fragments WHERE id = ?", (fragment_id,))
            report.removed.append(fragment_id)

    # 5. Write the rows that are new or different.
    placeholders = ", ".join("?" for _ in _COLUMNS)
    assignments = ", ".join(f"{name} = excluded.{name}" for name in _COLUMNS if name != "id")
    for fragment_id in sorted(winners):
        folder = winners[fragment_id]
        if folder.row is None:
            continue  # unchanged since it was indexed from this folder
        previous = indexed.get(fragment_id)
        if previous == (folder.rel_dir, folder.row["manifest_sha256"]):
            continue
        # A folder that was renamed gives up its old name before the new row lands.
        connection.execute(
            "DELETE FROM fragments WHERE rel_dir = ? AND id <> ?",
            (folder.rel_dir, fragment_id),
        )
        connection.execute(
            f"INSERT INTO fragments ({', '.join(_COLUMNS)}) VALUES ({placeholders}) "
            f"ON CONFLICT(id) DO UPDATE SET {assignments}",
            [folder.row[name] for name in _COLUMNS],
        )
        _write_tags(connection, fragment_id, folder.tags or [])
        (report.updated if previous is not None else report.added).append(fragment_id)

    _settle_tags(connection)

    # 6. Remember what was seen, including the folders that are not indexed.
    wanted = {
        folder.rel_dir: (folder.mtime_ns, folder.size, folder.fragment_id, folder.invalid, problems.get(folder.rel_dir))
        for folder in folders
    }
    current = {
        name: (int(row["mtime_ns"]), int(row["size"]), row["fragment_id"], row["invalid"], row["problem"])
        for name, row in remembered.items()
    }
    for name in sorted(set(current) - set(wanted)):
        connection.execute("DELETE FROM folders WHERE rel_dir = ?", (name,))
    for name in sorted(wanted):
        if current.get(name) != wanted[name]:
            connection.execute(
                "INSERT INTO folders (rel_dir, mtime_ns, size, fragment_id, invalid, problem) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(rel_dir) DO UPDATE SET mtime_ns = excluded.mtime_ns, size = excluded.size, "
                "fragment_id = excluded.fragment_id, invalid = excluded.invalid, problem = excluded.problem",
                (name, *wanted[name]),
            )

    # A problem folder "was indexed" when a row came from it before this pass.
    # It cannot be the source of a row now: a folder is a winner or a problem.
    indexed_before = {rel_dir for rel_dir, _digest in indexed.values()}
    report.problems = [
        IndexProblem(folder=name, message=problems[name], was_indexed=name in indexed_before)
        for name in sorted(problems)
    ]
    report.indexed = int(connection.execute("SELECT COUNT(*) FROM fragments").fetchone()[0])


def _write_tags(connection: sqlite3.Connection, fragment_id: str, tags: list[str]) -> None:
    connection.execute("DELETE FROM fragment_tags WHERE fragment_id = ?", (fragment_id,))
    for position, name in enumerate(tags):
        connection.execute("INSERT OR IGNORE INTO tags (name) VALUES (?)", (name,))
        tag_id = connection.execute("SELECT id FROM tags WHERE name = ?", (name,)).fetchone()[0]
        connection.execute(
            "INSERT OR IGNORE INTO fragment_tags (fragment_id, tag_id, name, position) VALUES (?, ?, ?, ?)",
            (fragment_id, tag_id, name, position),
        )


def _settle_tags(connection: sqlite3.Connection) -> None:
    """Drop tags nothing uses, and spell each tag one way.

    ``Cadence`` and ``cadence`` are one tag. Its spelling is the one used by
    the fragment that was created first among those that carry it (the id
    settles a tie). That is read from the manifests, so it does not depend on
    the order in which fragments happened to be indexed.
    """

    connection.execute("DELETE FROM tags WHERE id NOT IN (SELECT tag_id FROM fragment_tags)")
    rows = connection.execute(
        "SELECT t.id, t.name, ("
        "  SELECT ft.name FROM fragment_tags ft JOIN fragments f ON f.id = ft.fragment_id "
        "  WHERE ft.tag_id = t.id ORDER BY f.created_at, ft.fragment_id LIMIT 1"
        ") AS spelling FROM tags t"
    ).fetchall()
    for row in rows:
        if row["spelling"] is not None and row["spelling"] != row["name"]:
            connection.execute("UPDATE tags SET name = ? WHERE id = ?", (row["spelling"], row["id"]))


def reconcile_if_due(
    root: Path | None = None,
    db: Path | None = None,
    *,
    interval: float = RECONCILE_INTERVAL_SECONDS,
) -> ReconcileReport | None:
    """An incremental reconcile, unless one ran in this process within ``interval`` seconds.

    For a page that is redrawn on every click. Returns None when nothing was
    done. A missing index file is never skipped.
    """

    path = index_path(db)
    last = _LAST_RECONCILE.get(str(path.resolve()))
    if path.is_file() and last is not None and time.monotonic() - last < interval:
        return None
    return reconcile(root, db, mode="incremental")


def refresh_folder(folder: Path, db: Path | None = None) -> ReconcileReport:
    """Bring the index up to date after the program has written one fragment."""

    folder = Path(folder)
    return reconcile(folder.parent, db, mode="incremental", reread=(folder.name,))


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------


def stored_problems(db: Path | None = None) -> list[IndexProblem]:
    """The folders the last reconcile could not index. Reads the index only."""

    with _reading(db) as connection:
        if connection is None:
            return []
        rows = connection.execute(
            "SELECT rel_dir, problem FROM folders WHERE problem IS NOT NULL ORDER BY rel_dir"
        ).fetchall()
    return [IndexProblem(folder=row["rel_dir"], message=row["problem"]) for row in rows]


def progression_tokens(text: str) -> list[str]:
    """``V7 - i``, ``V7, i`` and ``V7 → i`` all mean the labels ``V7`` then ``i``."""

    cleaned = str(text or "")
    for separator in (",", ";", "|", "→", "->", "–", "—", " - "):
        cleaned = cleaned.replace(separator, " ")
    return [token for token in cleaned.split() if token]


def _like(text: str) -> str:
    escaped = text.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return f"%{escaped}%"


def _fragment(row: sqlite3.Row, tags: list[str]) -> IndexedFragment:
    return IndexedFragment(
        id=row["id"],
        rel_dir=row["rel_dir"],
        title=row["title"],
        description=row["description"],
        practice_notes=row["practice_notes"],
        composer=row["composer"],
        performer=row["performer"],
        source_kind=row["source_kind"],
        source_title=row["source_title"],
        source_url=row["source_url"],
        video_id=row["video_id"],
        parent_project=row["parent_project"],
        catalog_entry_id=row["catalog_entry_id"] if "catalog_entry_id" in row.keys() else "",
        catalog_collection=row["catalog_collection"] if "catalog_collection" in row.keys() else "",
        video_start=float(row["video_start"]),
        video_end=float(row["video_end"]),
        duration=float(row["duration"]),
        sync_offset=float(row["sync_offset"]),
        key_tonic=row["analyzed_key_tonic"],
        key_mode=row["analyzed_key_mode"],
        time_signature=row["time_signature"],
        bpm=float(row["bpm"]),
        measures=int(row["measures"]),
        measure_start=int(row["measure_start"]),
        measure_end=int(row["measure_end"]),
        note_count=int(row["note_count"]),
        roman=tuple(row["harmony_roman"].split()),
        symbols=tuple(row["harmony_symbols"].split()),
        tags=tuple(tags),
        created_at=row["created_at"],
        modified_at=row["modified_at"],
    )


def search_fragments(
    db: Path | None = None,
    *,
    text: str = "",
    tags: tuple[str, ...] | list[str] = (),
    key_tonic: str = "",
    key_mode: str = "",
    progression: str = "",
    source: str = "",
    catalog_entry_id: str = "",
    sort: str = "modified",
) -> list[IndexedFragment]:
    """Fragments that match every filter given. An empty filter matches everything.

    ``text`` is split into words; each word must appear in the title, the
    description, the practice notes, the composer, the performer, the source
    title, or a tag. ``tags`` must all be present. ``progression`` is a run
    of chord labels that must appear next to each other, in order, in the
    Roman numerals or in the chord symbols; it is case-sensitive, because
    ``V`` and ``v`` are different chords. ``source`` is a video id or a
    parent project name. ``catalog_entry_id`` matches curated Bach catalog provenance.
    """

    if sort not in SORTS:
        raise FragmentIndexError(f"Unknown sort {sort!r}. Choose from {', '.join(SORTS)}.")
    where: list[str] = []
    values: list[object] = []
    for word in str(text or "").split():
        pattern = _like(word)
        where.append(
            "(f.title LIKE ? ESCAPE '\\' OR f.description LIKE ? ESCAPE '\\' OR f.practice_notes LIKE ? ESCAPE '\\' "
            "OR f.composer LIKE ? ESCAPE '\\' OR f.performer LIKE ? ESCAPE '\\' OR f.source_title LIKE ? ESCAPE '\\' "
            "OR f.catalog_entry_id LIKE ? ESCAPE '\\' OR f.catalog_collection LIKE ? ESCAPE '\\' "
            "OR EXISTS (SELECT 1 FROM fragment_tags ft JOIN tags t ON t.id = ft.tag_id "
            "WHERE ft.fragment_id = f.id AND t.name LIKE ? ESCAPE '\\'))"
        )
        values.extend([pattern] * 9)
    for tag in _clean_tags(list(tags)):
        where.append(
            "EXISTS (SELECT 1 FROM fragment_tags ft JOIN tags t ON t.id = ft.tag_id "
            "WHERE ft.fragment_id = f.id AND t.name = ? COLLATE NOCASE)"
        )
        values.append(tag)
    if key_tonic:
        where.append("f.analyzed_key_tonic = ?")
        values.append(key_tonic)
    if key_mode:
        where.append("f.analyzed_key_mode = ?")
        values.append(key_mode)
    tokens = progression_tokens(progression)
    if tokens:
        needle = f" {' '.join(tokens)} "
        where.append("(instr(f.harmony_roman, ?) > 0 OR instr(f.harmony_symbols, ?) > 0)")
        values.extend([needle, needle])
    if source:
        where.append("(f.video_id = ? OR f.parent_project = ?)")
        values.extend([source, source])
    if catalog_entry_id:
        where.append("f.catalog_entry_id = ?")
        values.append(str(catalog_entry_id))
    query = "SELECT f.* FROM fragments f"
    if where:
        query += " WHERE " + " AND ".join(where)
    query += f" ORDER BY {SORTS[sort]}"
    with _reading(db) as connection:
        if connection is None:
            return []
        rows = connection.execute(query, values).fetchall()
        return [_fragment(row, _tags_of(connection, row["id"])) for row in rows]


def _tags_of(connection: sqlite3.Connection, fragment_id: str) -> list[str]:
    return [
        row[0]
        for row in connection.execute(
            "SELECT t.name FROM fragment_tags ft JOIN tags t ON t.id = ft.tag_id "
            "WHERE ft.fragment_id = ? ORDER BY ft.position",
            (fragment_id,),
        )
    ]


def get_fragment(fragment_id: str, db: Path | None = None) -> IndexedFragment | None:
    with _reading(db) as connection:
        if connection is None:
            return None
        row = connection.execute("SELECT * FROM fragments WHERE id = ?", (fragment_id,)).fetchone()
        return _fragment(row, _tags_of(connection, row["id"])) if row is not None else None


def fragment_count(db: Path | None = None) -> int:
    with _reading(db) as connection:
        if connection is None:
            return 0
        return int(connection.execute("SELECT COUNT(*) FROM fragments").fetchone()[0])


def all_tags(db: Path | None = None) -> list[tuple[str, int]]:
    """Every tag in use with the number of fragments that carry it, by name."""

    with _reading(db) as connection:
        if connection is None:
            return []
        rows = connection.execute(
            "SELECT t.name, COUNT(*) FROM tags t JOIN fragment_tags ft ON ft.tag_id = t.id "
            "GROUP BY t.id ORDER BY t.name COLLATE NOCASE"
        ).fetchall()
    return [(row[0], int(row[1])) for row in rows]


def all_keys(db: Path | None = None) -> list[tuple[str, str, int]]:
    """Every analyzed key in the library with its count."""

    with _reading(db) as connection:
        if connection is None:
            return []
        rows = connection.execute(
            "SELECT analyzed_key_tonic, analyzed_key_mode, COUNT(*) FROM fragments "
            "WHERE analyzed_key_tonic <> '' GROUP BY analyzed_key_tonic, analyzed_key_mode "
            "ORDER BY analyzed_key_tonic, analyzed_key_mode"
        ).fetchall()
    return [(row[0], row[1], int(row[2])) for row in rows]


def all_sources(db: Path | None = None) -> list[tuple[str, str, int]]:
    """``(video id or parent project, title, count)`` for every source in the library."""

    with _reading(db) as connection:
        if connection is None:
            return []
        rows = connection.execute(
            "SELECT CASE WHEN video_id <> '' THEN video_id ELSE parent_project END AS source, "
            "MIN(source_title), COUNT(*) FROM fragments GROUP BY source ORDER BY MIN(source_title) COLLATE NOCASE"
        ).fetchall()
    return [(row[0], row[1], int(row[2])) for row in rows]


def dump_index(db: Path | None = None) -> dict:
    """Everything the index says about the library, in a form that can be compared.

    Tag row numbers are left out: they record the order tags were first
    seen, which is history and not content.
    """

    with _reading(db) as connection:
        if connection is None:
            return {"fragments": [], "tags": [], "problems": []}
        fragments = []
        for row in connection.execute(f"SELECT {', '.join(_COLUMNS)} FROM fragments ORDER BY id"):
            entry = {name: row[name] for name in _COLUMNS}
            entry["tags"] = _tags_of(connection, row["id"])
            fragments.append(entry)
        tags = [row[0] for row in connection.execute("SELECT name FROM tags ORDER BY name COLLATE NOCASE, name")]
        problems = [
            [row[0], row[1]]
            for row in connection.execute(
                "SELECT rel_dir, problem FROM folders WHERE problem IS NOT NULL ORDER BY rel_dir"
            )
        ]
    return {"fragments": fragments, "tags": tags, "problems": problems}
