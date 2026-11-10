"""The SQLite index of the fragment library: reconcile, stale entries, and search."""

from __future__ import annotations

import json
import random
import shutil
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from baroque_transcribe import fragment_index as X
from baroque_transcribe import fragments as F
from baroque_transcribe.cli import main
from baroque_transcribe.defaults import FRAGMENT_JSON

from fragment_helpers import fake_fragment, fake_manifest, make_project, n, write_manifest

A = "0a000000-0000-4000-8000-000000000001"
B = "0b000000-0000-4000-8000-000000000002"
C = "0c000000-0000-4000-8000-000000000003"
D = "0d000000-0000-4000-8000-000000000004"
E = "0e000000-0000-4000-8000-000000000005"


@pytest.fixture
def library(tmp_path: Path) -> tuple[Path, Path]:
    """``(fragments root, index file)``, both empty."""

    root = tmp_path / "data" / "fragments"
    root.mkdir(parents=True)
    return root, tmp_path / ".local" / "library.sqlite"


def three(root: Path) -> None:
    fake_fragment(
        root,
        A,
        title="Circle of fifths",
        tags=["Sequence", "bass"],
        roman=["i", "iv7", "VII7", "III", "VI"],
        symbols=["Gm", "Cm7", "F7", "Bb", "Eb"],
        composer="Bach",
        modified="2026-11-10T09:00:00Z",
    )
    fake_fragment(
        root,
        B,
        title="Cadential six-four",
        tags=["cadence"],
        roman=["i6/4", "V7", "V7", "i"],
        symbols=["Gm/D", "D7", "D7", "Gm"],
        modified="2026-11-11T09:00:00Z",
    )
    fake_fragment(
        root,
        C,
        title="Deceptive",
        tags=["Cadence", "sequence"],
        tonic="D",
        mode="major",
        roman=["ii7", "V7", "vi"],
        symbols=["Em7", "A7", "Bm"],
        description="Keep it 100% legato_line.",
        video_id="aaaaaaaaaaa",
        source_title="Another take",
        modified="2026-11-12T09:00:00Z",
    )


def ids(found) -> list[str]:
    return [item.id for item in found]


def problem_pairs(report) -> list[tuple[str, str]]:
    return [(problem.folder, problem.message) for problem in report.problems]


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------


def test_index_is_built_from_real_fragments(tmp_path: Path, library):
    root, db = library
    notes = [n(55, 0.0, 2.0), n(62, 0.0, 0.5), n(70, 0.5, 1.0), n(74, 1.0, 2.0)]
    notes += [n(48, 2.0, 4.0), n(63, 2.0, 2.5), n(67, 2.5, 3.0), n(72, 3.0, 4.0)]
    notes += [n(50, 4.0, 6.0), n(66, 4.0, 4.5), n(69, 4.5, 5.0), n(74, 5.0, 6.0)]
    project = make_project(tmp_path, notes, excerpt=30.0, sync=0.4, analyze=True)
    first = F.extract_fragment(
        project,
        1,
        2,
        title="Opening",
        tags=["Sequence"],
        composer="Improvised",
        root=root,
        now=datetime(2026, 11, 10, 9, 0, tzinfo=timezone.utc),
    )
    second = F.extract_fragment(
        project,
        3,
        3,
        title="Dominant",
        tags=["cadence", "sequence"],
        root=root,
        now=datetime(2026, 11, 10, 9, 5, tzinfo=timezone.utc),
    )

    assert not db.exists()
    report = X.reconcile(root, db)
    assert db.is_file()
    assert (report.folders, report.indexed, report.manifests_read) == (2, 2, 2)
    assert sorted(report.added) == sorted([first.name, second.name])
    assert not report.updated and not report.removed and not report.problems

    manifest = F.read_manifest(first)
    row = X.get_fragment(first.name, db)
    assert row is not None
    assert (row.title, row.composer, row.rel_dir) == ("Opening", "Improvised", first.name)
    assert row.tags == ("Sequence",)
    # One tag, spelled as the fragment that was created first spells it.
    assert X.get_fragment(second.name, db).tags == ("cadence", "Sequence")
    assert row.key == f"{manifest['music']['analyzed_key']['tonic']} {manifest['music']['analyzed_key']['mode']}"
    assert (row.measure_start, row.measure_end, row.measures) == (1, 2, 2)
    assert row.video_start == manifest["time"]["source_video"]["start_seconds"]
    assert row.video_end == manifest["time"]["source_video"]["end_seconds"]
    assert row.duration == manifest["time"]["duration_seconds"]
    assert row.sync_offset == 0.4
    assert row.video_id == "dQw4w9WgXcQ" and row.parent_project == project.name
    assert row.note_count == manifest["boundary"]["notes"]
    assert row.roman  # the parent was analyzed
    assert X.get_fragment("missing", db) is None
    assert X.fragment_count(db) == 2

    # Editing through the program and refreshing that one folder is enough.
    F.update_fragment_metadata(first, title="Opening, revised", tags=["bass"])
    F.set_fragment_sync(first, 1.5)
    refreshed = X.refresh_folder(first, db)
    assert refreshed.updated == [first.name] and refreshed.manifests_read == 1
    row = X.get_fragment(first.name, db)
    assert (row.title, row.tags, row.sync_offset) == ("Opening, revised", ("bass",), 1.5)
    # The sync adjustment did not move the indexed interval.
    assert row.video_start == manifest["time"]["source_video"]["start_seconds"]
    assert row.duration == manifest["time"]["duration_seconds"]


def test_queries_without_an_index_return_nothing_and_create_nothing(library):
    _root, db = library
    assert X.search_fragments(db) == []
    assert X.all_tags(db) == [] and X.all_keys(db) == [] and X.all_sources(db) == []
    assert X.stored_problems(db) == []
    assert X.fragment_count(db) == 0
    assert not db.exists() and not db.parent.exists()


def test_staging_folders_and_stray_files_are_ignored(library):
    root, db = library
    three(root)
    fake_fragment(root, D, folder=".tmp-" + D, title="Half written")
    (root / "README.txt").write_text("notes", encoding="utf-8")
    report = X.reconcile(root, db)
    assert report.indexed == 3 and not report.problems
    assert D not in ids(X.search_fragments(db))


# --------------------------------------------------------------------------
# Search
# --------------------------------------------------------------------------


def test_search_filters(library):
    root, db = library
    three(root)
    X.reconcile(root, db)

    assert ids(X.search_fragments(db)) == [C, B, A]  # most recently modified first
    assert ids(X.search_fragments(db, sort="title")) == [B, A, C]
    with pytest.raises(X.FragmentIndexError):
        X.search_fragments(db, sort="nonsense")

    # Text: every word must appear somewhere, in any case.
    assert ids(X.search_fragments(db, text="FIFTHS")) == [A]
    assert ids(X.search_fragments(db, text="bach")) == [A]  # composer
    assert ids(X.search_fragments(db, text="cadence")) == [C, B]  # a tag
    assert ids(X.search_fragments(db, text="another")) == [C]  # source title
    assert ids(X.search_fragments(db, text="circle bass")) == [A]
    assert ids(X.search_fragments(db, text="circle cadence")) == []
    # LIKE wildcards in the text are ordinary characters.
    assert ids(X.search_fragments(db, text="100%")) == [C]
    assert ids(X.search_fragments(db, text="_")) == [C]
    assert ids(X.search_fragments(db, text="%")) == [C]

    # Tags: all of them, whatever the capitals.
    assert ids(X.search_fragments(db, tags=["sequence"])) == [C, A]
    assert ids(X.search_fragments(db, tags=["SEQUENCE", "cadence"])) == [C]
    assert ids(X.search_fragments(db, tags=["sequence", "bass", "cadence"])) == []

    # Key.
    assert ids(X.search_fragments(db, key_tonic="G", key_mode="minor")) == [B, A]
    assert ids(X.search_fragments(db, key_tonic="G")) == [B, A]
    assert ids(X.search_fragments(db, key_tonic="G", key_mode="major")) == []
    assert ids(X.search_fragments(db, key_mode="major")) == [C]

    # Progression: whole labels, next to each other, in order, case-sensitive.
    assert ids(X.search_fragments(db, progression="V7 i")) == [B]  # stored as i6/4 V7 V7 i
    assert ids(X.search_fragments(db, progression="V7 - i")) == [B]
    assert ids(X.search_fragments(db, progression="V7, i")) == [B]
    assert ids(X.search_fragments(db, progression="V7 → i")) == [B]
    assert ids(X.search_fragments(db, progression="V7 vi")) == [C]
    assert ids(X.search_fragments(db, progression="V7")) == [C, B]
    assert ids(X.search_fragments(db, progression="v7 i")) == []  # a minor dominant is another chord
    assert ids(X.search_fragments(db, progression="i V7")) == []  # order matters
    assert ids(X.search_fragments(db, progression="VII")) == []  # VII7 is not VII
    assert ids(X.search_fragments(db, progression="iv7 VII7 III")) == [A]
    assert ids(X.search_fragments(db, progression="iv7 III")) == []  # not adjacent
    assert ids(X.search_fragments(db, progression="Cm7 F7")) == [A]  # chord symbols too
    assert ids(X.search_fragments(db, progression="D7 Gm")) == [B]

    # Source, and filters together.
    assert ids(X.search_fragments(db, source="aaaaaaaaaaa")) == [C]
    assert ids(X.search_fragments(db, source="parent")) == [C, B, A]
    assert ids(X.search_fragments(db, text="cadential", tags=["cadence"], key_tonic="G", progression="V7 i")) == [B]
    assert ids(X.search_fragments(db, text="cadential", tags=["cadence"], key_tonic="D")) == []

    assert X.all_keys(db) == [("D", "major", 1), ("G", "minor", 2)]
    assert X.all_sources(db) == [("aaaaaaaaaaa", "Another take", 1), ("dQw4w9WgXcQ", "Parent take", 2)]


def test_a_tag_is_spelled_one_way(library):
    root, db = library
    three(root)
    X.reconcile(root, db)
    # A and C both use it. They were created at the same moment, so the smaller id decides.
    assert X.all_tags(db) == [("bass", 1), ("cadence", 2), ("Sequence", 2)]
    assert X.get_fragment(C, db).tags == ("cadence", "Sequence")

    # Remove A's use of it: the spelling follows the fragment that is left.
    write_manifest(root / A, fake_manifest(A, title="Circle of fifths", tags=["bass"]))
    X.reconcile(root, db)
    assert X.all_tags(db) == [("bass", 1), ("cadence", 2), ("sequence", 1)]
    # A tag nobody uses is gone.
    shutil.rmtree(root / A)
    X.reconcile(root, db)
    assert X.all_tags(db) == [("cadence", 2), ("sequence", 1)]


def test_harmony_text():
    assert X.harmony_text(["V7", "V7", "i", "V7"]) == " V7 i V7 "
    assert X.harmony_text([" V 7 ", "", "i"]) == " V7 i "
    assert X.harmony_text([]) == ""
    assert X.progression_tokens("V7 - i, iv; ii -> V | I") == ["V7", "i", "iv", "ii", "V", "I"]
    assert X.progression_tokens("i6/4 V4-3") == ["i6/4", "V4-3"]  # a hyphen inside a label is kept


# --------------------------------------------------------------------------
# A manifest that was indexed and is no longer usable
# --------------------------------------------------------------------------


def _not_json(folder: Path) -> None:
    write_manifest(folder, "{ this is not json")


def _wrong_version(folder: Path) -> None:
    payload = json.loads((folder / FRAGMENT_JSON).read_text(encoding="utf-8"))
    payload["schema_version"] = 99
    write_manifest(folder, payload)


def _no_title(folder: Path) -> None:
    payload = json.loads((folder / FRAGMENT_JSON).read_text(encoding="utf-8"))
    del payload["title"]
    write_manifest(folder, payload)


def _deleted(folder: Path) -> None:
    (folder / FRAGMENT_JSON).unlink()


def _unreadable(folder: Path) -> None:
    # A directory where the file should be: it can be found and cannot be read.
    (folder / FRAGMENT_JSON).unlink()
    (folder / FRAGMENT_JSON).mkdir()


DAMAGE = {
    "not-json": (_not_json, "not valid JSON"),
    "wrong-version": (_wrong_version, "schema version 99"),
    "no-title": (_no_title, "no title"),
    "manifest-deleted": (_deleted, "has no fragment.json"),
    "unreadable": (_unreadable, "Could not read"),
}


@pytest.mark.parametrize("mode", ["incremental", "full"])
@pytest.mark.parametrize("damage", sorted(DAMAGE))
def test_indexed_manifest_that_becomes_unusable_is_removed_and_reported(library, damage: str, mode: str):
    root, db = library
    three(root)
    first = X.reconcile(root, db)
    assert first.indexed == 3 and not first.problems
    assert ids(X.search_fragments(db, tags=["cadence"])) == [C, B]
    assert ("cadence", 2) in X.all_tags(db)

    spoil, expected = DAMAGE[damage]
    spoil(root / B)
    report = X.reconcile(root, db, mode=mode)

    # The stale row is gone, in the same pass.
    assert report.removed == [B]
    assert not report.added and not report.updated
    assert report.indexed == 2
    assert X.get_fragment(B, db) is None
    assert ids(X.search_fragments(db)) == [C, A]
    assert ids(X.search_fragments(db, tags=["cadence"])) == [C]
    assert ids(X.search_fragments(db, progression="V7 i")) == []
    # Its tag use went with it. The one fragment left spells the tag its own way.
    assert X.all_tags(db) == [("bass", 1), ("Cadence", 1), ("Sequence", 2)]
    # And the problem is reported, as one that cost the library an entry.
    assert len(report.problems) == 1
    problem = report.problems[0]
    assert problem.folder == B and expected in problem.message and problem.was_indexed is True
    assert [(p.folder, p.message) for p in X.stored_problems(db)] == [(B, problem.message)]

    # It keeps being reported while it stays broken, without removing anything more.
    again = X.reconcile(root, db, mode=mode)
    assert again.removed == [] and again.indexed == 2
    assert problem_pairs(again) == [(B, problem.message)]
    assert again.problems[0].was_indexed is False

    # The result is what a rebuild from the same folders gives.
    rebuilt_db = db.with_name("rebuilt.sqlite")
    rebuilt = X.reconcile(root, rebuilt_db, mode="rebuild")
    assert X.dump_index(db) == X.dump_index(rebuilt_db)
    assert problem_pairs(rebuilt) == problem_pairs(again)

    # Repairing the manifest brings the fragment back.
    if (root / B / FRAGMENT_JSON).is_dir():
        (root / B / FRAGMENT_JSON).rmdir()
    write_manifest(root / B, fake_manifest(B, title="Cadential six-four", tags=["cadence"], roman=["V7", "i"]))
    repaired = X.reconcile(root, db, mode=mode)
    assert repaired.added == [B] and not repaired.problems and repaired.indexed == 3
    assert X.stored_problems(db) == []
    assert ids(X.search_fragments(db, progression="V7 i")) == [B]


def test_deleted_folder_leaves_no_row_and_no_problem(library):
    root, db = library
    three(root)
    X.reconcile(root, db)
    shutil.rmtree(root / B)
    report = X.reconcile(root, db)
    assert report.removed == [B] and not report.problems
    assert ids(X.search_fragments(db)) == [C, A]


def test_duplicate_id_indexes_one_folder_and_reports_the_other(library):
    root, db = library
    three(root)
    shutil.copytree(root / A, root / "copy-of-a")
    report = X.reconcile(root, db)
    assert report.indexed == 3
    assert X.get_fragment(A, db).rel_dir == A  # the folder named after the id
    assert [problem.folder for problem in report.problems] == ["copy-of-a"]
    assert A in report.problems[0].message and "same fragment id" in report.problems[0].message

    # With the original gone the copy is the fragment. Its manifest did not
    # change, so an incremental pass has to notice that on its own.
    shutil.rmtree(root / A)
    after = X.reconcile(root, db)
    assert after.updated == [A] and not after.removed and not after.problems
    assert X.get_fragment(A, db).rel_dir == "copy-of-a"
    assert X.get_fragment(A, db).tags == ("Sequence", "bass")

    # Neither folder is named after the id: the first by name is used.
    shutil.copytree(root / "copy-of-a", root / "another-copy")
    third = X.reconcile(root, db)
    assert X.get_fragment(A, db).rel_dir == "another-copy"
    assert [problem.folder for problem in third.problems] == ["copy-of-a"]
    assert third.problems[0].was_indexed is True


# --------------------------------------------------------------------------
# Incremental reconcile and a rebuild end in the same place
# --------------------------------------------------------------------------


def assert_same_as_rebuild(root: Path, db: Path, full_db: Path, scratch: Path, step: object) -> X.ReconcileReport:
    """Reconcile three ways and require one answer. Returns the incremental report."""

    incremental = X.reconcile(root, db, mode="incremental")
    full = X.reconcile(root, full_db, mode="full")
    rebuilt = X.reconcile(root, scratch, mode="rebuild")
    assert X.dump_index(db) == X.dump_index(scratch), f"incremental differs from a rebuild after {step}"
    assert X.dump_index(full_db) == X.dump_index(scratch), f"full differs from a rebuild after {step}"
    assert problem_pairs(incremental) == problem_pairs(rebuilt) == problem_pairs(full), step
    assert incremental.indexed == rebuilt.indexed == full.indexed, step
    # The two that keep an index between steps also agree on what changed.
    assert (sorted(incremental.added), sorted(incremental.updated), sorted(incremental.removed)) == (
        sorted(full.added),
        sorted(full.updated),
        sorted(full.removed),
    ), step
    assert [p.was_indexed for p in incremental.problems] == [p.was_indexed for p in full.problems], step
    return incremental


def test_incremental_and_rebuild_agree_through_a_scripted_history(library, tmp_path: Path):
    root, db = library
    full_db, scratch = tmp_path / "full.sqlite", tmp_path / "scratch.sqlite"

    def check(step: str) -> None:
        assert_same_as_rebuild(root, db, full_db, scratch, step)

    three(root)
    fake_fragment(root, D, title="Pedal point", tags=["pedal"], roman=["I", "IV6/4", "I"])
    check("four fragments")

    write_manifest(root / A, fake_manifest(A, title="Circle of fifths, again", tags=["Sequence", "bass", "slow"]))
    check("an edited title and a new tag")

    shutil.copytree(root / A, root / "zzz-copy")
    check("a second folder with the same id")

    shutil.rmtree(root / A)
    check("the first of the two removed")
    assert X.get_fragment(A, db).rel_dir == "zzz-copy"

    _not_json(root / B)
    check("a damaged manifest")
    assert X.get_fragment(B, db) is None

    (root / C).rename(root / "renamed-c")
    check("a renamed folder")
    assert X.get_fragment(C, db).rel_dir == "renamed-c"

    payload = fake_manifest(E, title="Pedal point", tags=["pedal"], roman=["I", "IV6/4", "I"])
    write_manifest(root / D, payload)
    check("a manifest that now carries a new id")
    assert X.get_fragment(D, db) is None and X.get_fragment(E, db).rel_dir == D

    fake_fragment(root, C, title="Deceptive, the real one", tags=["cadence"])
    check("a folder named after an id that another folder already claims")
    assert X.get_fragment(C, db).rel_dir == C

    write_manifest(root / B, fake_manifest(B, title="Cadential six-four", tags=["CADENCE"], roman=["V7", "i"]))
    check("the damaged manifest repaired, with a tag in capitals")

    (root / "not-a-fragment").mkdir()
    check("a folder with no manifest")

    _deleted(root / D)
    check("a manifest deleted from its folder")

    for folder in list(root.iterdir()):
        if folder.name not in {B}:
            shutil.rmtree(folder)
    check("everything but one removed")
    assert ids(X.search_fragments(db)) == [B]

    shutil.rmtree(root / B)
    check("an empty library")
    assert X.dump_index(db) == {"fragments": [], "tags": [], "problems": []}


def test_incremental_and_rebuild_agree_through_random_histories(library, tmp_path: Path):
    root, db = library
    full_db, scratch = tmp_path / "full.sqlite", tmp_path / "scratch.sqlite"
    rng = random.Random(4242)
    pool = [f"{index:02d}000000-0000-4000-8000-00000000000{index % 10}" for index in range(1, 9)]
    tag_pool = ["cadence", "Cadence", "CADENCE", "sequence", "Sequence", "bass", "pedal"]
    chords = ["i", "iv", "V7", "V", "VI", "ii7", "I", "vi"]

    def manifest_for(fragment_id: str) -> dict:
        return fake_manifest(
            fragment_id,
            title=f"Fragment {rng.randint(1, 999)}",
            tags=rng.sample(tag_pool, rng.randint(0, 3)),
            roman=[rng.choice(chords) for _ in range(rng.randint(0, 5))],
            tonic=rng.choice(["G", "D", "C"]),
            mode=rng.choice(["major", "minor"]),
        )

    def folders() -> list[Path]:
        return sorted(path for path in root.iterdir() if path.is_dir())

    seen = {"duplicate": 0, "stale removed": 0, "took over": 0, "no manifest": 0, "added": 0, "updated": 0}
    for step in range(120):
        existing = folders()
        action = rng.choice(["add", "add", "edit", "delete", "damage", "copy", "rename", "reid", "strip"])
        if action == "add" or not existing:
            fragment_id = rng.choice(pool)
            name = fragment_id if rng.random() < 0.6 else f"folder-{rng.randint(1, 30)}"
            write_manifest(root / name, manifest_for(fragment_id))
        else:
            target = rng.choice(existing)
            manifest_path = target / FRAGMENT_JSON
            if action == "delete":
                shutil.rmtree(target)
            elif action == "damage":
                write_manifest(target, rng.choice(["{", "[]", json.dumps({"schema_version": 1}), ""]))
            elif action == "copy":
                destination = root / f"folder-{rng.randint(1, 30)}"
                if not destination.exists():
                    shutil.copytree(target, destination)
            elif action == "rename":
                destination = root / rng.choice(pool + [f"folder-{rng.randint(1, 30)}"])
                if not destination.exists():
                    target.rename(destination)
            elif action == "strip":
                if manifest_path.is_file():
                    manifest_path.unlink()
            else:  # edit, reid
                current = None
                if manifest_path.is_file():
                    try:
                        current = json.loads(manifest_path.read_text(encoding="utf-8")).get("id")
                    except (json.JSONDecodeError, AttributeError):
                        current = None
                fragment_id = rng.choice(pool) if action == "reid" or not current else current
                write_manifest(target, manifest_for(fragment_id))
        before = {row["id"]: row["rel_dir"] for row in X.dump_index(db)["fragments"]}
        report = assert_same_as_rebuild(root, db, full_db, scratch, f"step {step}: {action}")
        after = {row["id"]: row["rel_dir"] for row in X.dump_index(db)["fragments"]}
        seen["duplicate"] += any("same fragment id" in p.message for p in report.problems)
        seen["no manifest"] += any("has no fragment.json" in p.message for p in report.problems)
        seen["stale removed"] += any(p.was_indexed for p in report.problems) and bool(report.removed)
        seen["took over"] += any(key in before and before[key] != after[key] for key in after)
        seen["added"] += bool(report.added)
        seen["updated"] += bool(report.updated)

    # The history did exercise the cases that matter, each more than once.
    assert all(count >= 3 for count in seen.values()), seen


def test_stat_identical_edit_is_caught_by_a_full_reconcile_and_by_a_refresh(library):
    """The one thing an incremental pass cannot see: new content, same time and size."""

    import os

    root, db = library
    three(root)
    X.reconcile(root, db)
    path = root / A / FRAGMENT_JSON
    status = path.stat()
    text = path.read_text(encoding="utf-8")
    changed = text.replace("Circle of fifths", "Circle of FIFTHS")
    assert len(changed.encode("utf-8")) == len(text.encode("utf-8")) and changed != text
    path.write_text(changed, encoding="utf-8")
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))
    assert (path.stat().st_mtime_ns, path.stat().st_size) == (status.st_mtime_ns, status.st_size)

    assert X.reconcile(root, db).updated == []
    assert X.get_fragment(A, db).title == "Circle of fifths"
    # Reading the folder the program has just written does not rely on the clock.
    assert X.refresh_folder(root / A, db).updated == [A]
    assert X.get_fragment(A, db).title == "Circle of FIFTHS"

    path.write_text(text, encoding="utf-8")
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))
    assert X.reconcile(root, db).updated == []
    assert X.reconcile(root, db, mode="full").updated == [A]
    assert X.get_fragment(A, db).title == "Circle of fifths"


# --------------------------------------------------------------------------
# How much work a reconcile does
# --------------------------------------------------------------------------


def test_incremental_reconcile_reads_only_what_changed(library, monkeypatch):
    root, db = library
    three(root)
    opened: list[str] = []
    real = X._read_folder

    def counting(base, folder):
        opened.append(folder.rel_dir)
        return real(base, folder)

    monkeypatch.setattr(X, "_read_folder", counting)

    assert X.reconcile(root, db).manifests_read == 3
    assert sorted(opened) == [A, B, C]

    opened.clear()
    before = db.read_bytes()
    unchanged = X.reconcile(root, db)
    assert unchanged.manifests_read == 0 and opened == []
    assert not unchanged.changed and unchanged.indexed == 3
    assert db.read_bytes() == before  # nothing was written either

    write_manifest(root / B, fake_manifest(B, title="Cadential six-four, revised", tags=["cadence"]))
    fake_fragment(root, D, title="New")
    opened.clear()
    report = X.reconcile(root, db)
    assert sorted(opened) == [B, D] and report.manifests_read == 2
    assert report.updated == [B] and report.added == [D]

    opened.clear()
    full = X.reconcile(root, db, mode="full")
    assert sorted(opened) == [A, B, C, D] and full.manifests_read == 4
    assert not full.changed  # everything was hashed and nothing differed


def test_page_reconcile_is_throttled(library, monkeypatch):
    root, db = library
    three(root)
    now = [1000.0]
    monkeypatch.setattr(X.time, "monotonic", lambda: now[0])
    calls: list[str] = []
    real = X.reconcile

    def spying(*args, **kwargs):
        calls.append(kwargs.get("mode", "incremental"))
        return real(*args, **kwargs)

    monkeypatch.setattr(X, "reconcile", spying)

    # No index file yet: it runs, whatever the clock says.
    first = X.reconcile_if_due(root, db)
    assert first is not None and first.indexed == 3 and calls == ["incremental"]

    # A page that is redrawn on every click does not scan again.
    fake_fragment(root, D, title="Added between redraws")
    for _ in range(25):
        now[0] += 0.2
        assert X.reconcile_if_due(root, db) is None
    assert calls == ["incremental"]
    assert X.fragment_count(db) == 3

    # After the interval it does, and it reads only the new manifest.
    now[0] += X.RECONCILE_INTERVAL_SECONDS
    due = X.reconcile_if_due(root, db)
    assert due is not None and due.added == [D] and due.manifests_read == 1
    assert len(calls) == 2

    # A reconcile from anywhere else resets the clock.
    now[0] += 9.0
    X.reconcile(root, db, mode="full")
    now[0] += 9.0
    assert X.reconcile_if_due(root, db) is None

    # A deleted index is rebuilt at the next draw, inside the interval.
    db.unlink()
    again = X.reconcile_if_due(root, db)
    assert again is not None and again.indexed == 4 and again.manifests_read == 4


# --------------------------------------------------------------------------
# The index file itself
# --------------------------------------------------------------------------


def test_damaged_or_outdated_index_file_is_replaced(library):
    root, db = library
    three(root)
    X.reconcile(root, db)
    expected = X.dump_index(db)

    db.write_bytes(b"this is not a database at all, just some bytes" * 50)
    assert X.search_fragments(db) == []  # a query does not repair, and does not raise
    report = X.reconcile(root, db)
    assert report.indexed == 3 and report.manifests_read == 3
    assert X.dump_index(db) == expected

    connection = sqlite3.connect(str(db))
    connection.execute("PRAGMA user_version = 77")
    connection.commit()
    connection.close()
    assert X.search_fragments(db) == []
    report = X.reconcile(root, db)
    assert report.indexed == 3 and sorted(report.added) == [A, B, C]
    assert X.dump_index(db) == expected

    # A rebuild works while another connection has the file open.
    reader = sqlite3.connect(str(db))
    reader.execute("SELECT COUNT(*) FROM fragments").fetchone()
    rebuilt = X.reconcile(root, db, mode="rebuild")
    reader.close()
    assert rebuilt.indexed == 3 and X.dump_index(db) == expected

    with pytest.raises(X.FragmentIndexError):
        X.reconcile(root, db, mode="sometimes")


def test_index_holds_nothing_the_manifests_do_not(library):
    """Deleting the index loses nothing."""

    root, db = library
    three(root)
    X.reconcile(root, db)
    before = X.dump_index(db)
    db.unlink()
    X.reconcile(root, db)
    assert X.dump_index(db) == before
    manifests = {path.name: json.loads((path / FRAGMENT_JSON).read_text(encoding="utf-8")) for path in root.iterdir()}
    for row in before["fragments"]:
        manifest = manifests[row["rel_dir"]]
        assert row["title"] == manifest["title"]
        assert row["harmony_roman"] == X.harmony_text(manifest["harmony"]["roman"])
        assert [tag.casefold() for tag in row["tags"]] == [tag.casefold() for tag in manifest["tags"]]


# --------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------


def test_cli_reindex_and_list(library, capsys):
    root, db = library
    three(root)
    base = ["--library", str(root), "--index", str(db)]

    assert main(["fragments", "reindex", *base]) == 0
    assert "3 fragments indexed, 3 added." in capsys.readouterr().out

    assert main(["fragments", "list", *base]) == 0
    out = capsys.readouterr().out
    assert out.index("Deceptive") < out.index("Cadential six-four") < out.index("Circle of fifths")
    assert "3 fragments." in out and "tags: Sequence, bass" in out and "harmony: i iv7 VII7 III VI" in out

    assert main(["fragments", "list", "--progression", "V7 i", "--key", "G minor", *base]) == 0
    out = capsys.readouterr().out
    assert "Cadential six-four" in out and "Deceptive" not in out and "1 fragment." in out
    assert main(["fragments", "list", "--tag", "sequence", "--text", "deceptive", "--key", "D", *base]) == 0
    assert "1 fragment." in capsys.readouterr().out

    # A manifest that goes bad is dropped and named, by a plain listing too.
    _not_json(root / B)
    assert main(["fragments", "list", *base]) == 0
    captured = capsys.readouterr()
    assert "2 fragments." in captured.out and "Cadential" not in captured.out
    assert f"warning: {B}:" in captured.err and "removed from the index" in captured.err

    assert main(["fragments", "reindex", "--rebuild", *base]) == 0
    captured = capsys.readouterr()
    assert "Rebuilt the index. 2 fragments indexed, 2 added, 1 folder could not be indexed." in captured.out
    assert f"warning: {B}:" in captured.err and "removed from the index" not in captured.err


def test_cli_extract_updates_the_index(tmp_path: Path, library, capsys):
    root, db = library
    notes = [n(60, 0.0, 0.5), n(64, 0.5, 1.0), n(67, 2.0, 2.5), n(72, 2.5, 3.5)]
    project = make_project(tmp_path, notes)
    arguments = ["fragments", "extract", str(project), "--measures", "1-2", "--title", "Indexed at once"]
    assert main([*arguments, "--library", str(root), "--index", str(db)]) == 0
    assert "Index: 1 fragment indexed, 1 added." in capsys.readouterr().out
    assert [item.title for item in X.search_fragments(db, text="indexed")] == ["Indexed at once"]

    # Without a usable index location the fragment is still saved.
    assert main([*arguments, "--library", str(root)]) == 0
    captured = capsys.readouterr()
    assert "Saved fragment" in captured.out and "the search index was not updated" in captured.err
    assert len(F.fragment_folders(root)) == 2
