import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

from baroque_transcribe.parse_store import (
    load_events,
    publish_analysis,
    record_parse,
    validate_event,
)
from baroque_transcribe.project_manifest import load_project_manifest, write_project_manifest
from baroque_transcribe.project_restore import (
    RestoreError,
    _same_source_and_excerpt,
    adopt_exact_match,
    list_recoveries,
    preview_restore,
    remove_recovery,
    rollback_recovery,
    restore_missing_project,
    update_project,
)
from baroque_transcribe.revision import (
    artifact_digest,
    revision_status,
    snapshot_for_folder,
)
from baroque_transcribe.storage_management import export_project, storage_report
from baroque_transcribe.storage_io import sha256_file, write_json_atomic


def _project(path: Path, *, pitch: int = 60, annotation: str = "") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    existing = load_project_manifest(path)
    revision = existing.get("revision") if isinstance(existing.get("revision"), dict) else None
    manifest = {
        "origin": "https://youtu.be/abcdefghijk",
        "youtube_url": "https://youtu.be/abcdefghijk",
        "title": "Revision piece",
        "model": "test",
        "audio": {
            "excerpt_start_seconds": 1.0,
            "excerpt_end_seconds": 3.0,
            "working_duration_seconds": 2.0,
        },
        "tempo": {
            "bpm": 90.0,
            "downbeat_seconds": 0.0,
            "subdivision": 16,
            "time_signature": "4/4",
            "time_signature_assumed": False,
        },
        "staff_split": 60,
        "notation": {
            "style": "literal",
            "settings": {},
            "score_from": "quantized_grid",
        },
    }
    if revision is not None:
        manifest["revision"] = revision
    write_project_manifest(
        path,
        manifest,
    )
    (path / "notes.json").write_text(
        json.dumps(
            {
                "version": 1,
                "model": "test",
                "source": "private/source.wav",
                "notes": [
                    {
                        "pitch": pitch,
                        "onset": 0.0,
                        "offset": 0.5,
                        "velocity": 80,
                    }
                ],
                "pedals": [],
            }
        ),
        encoding="utf-8",
    )
    (path / "pedal.json").write_text('{"version": 1, "pedals": []}\n', encoding="utf-8")
    (path / "raw.mid").write_bytes(b"raw")
    (path / "quantized.mid").write_bytes(b"quantized")
    (path / "score.musicxml").write_text("<score-partwise/>\n", encoding="utf-8")
    (path / "working.wav").write_bytes(b"local audio")
    if annotation:
        (path / "annotations.json").write_text(annotation, encoding="utf-8")
    return path


def _publish(
    project: Path,
    store: Path,
    action: str = "transcribe",
    *,
    project_name: str | None = None,
    recorded_at: datetime | None = None,
) -> Path:
    event = record_parse(
        action=action,
        status="ok",
        source="https://youtu.be/abcdefghijk",
        title="Revision piece",
        project=project_name or project.name,
        project_dir=project,
        store=store,
        excerpt_start=1.0,
        excerpt_end=3.0,
        duration=2.0,
        recorded_at=recorded_at,
    )
    assert event is not None
    return event


def test_v2_events_are_complete_causal_snapshots(tmp_path):
    store = tmp_path / "store"
    project = _project(tmp_path / "piece", pitch=60)
    first = _publish(project, store)
    _project(project, pitch=62)
    second = _publish(project, store, action="quantize")

    one = json.loads((first / "event.json").read_text(encoding="utf-8"))
    two = json.loads((second / "event.json").read_text(encoding="utf-8"))
    assert one["version"] == 2
    assert two["revision"]["lineage_id"] == one["revision"]["lineage_id"]
    assert two["revision"]["parent_event_id"] == one["id"]
    assert two["revision"]["complete_snapshot"] is True
    assert set(two["revision"]["snapshot"]["files"]) >= {
        "notes.json",
        "pedal.json",
        "raw.mid",
        "quantized.mid",
        "score.musicxml",
    }
    assert all(
        (second / name).is_file()
        for name in ("notes.json", "pedal.json", "raw.mid", "quantized.mid", "score.musicxml")
    )
    assert "private/" not in (second / "notes.json").read_text(encoding="utf-8")

    status = revision_status(project.name, project, load_events(store), store)
    assert status.state == "synchronized"
    payload = json.loads((project / "notes.json").read_text(encoding="utf-8"))
    payload["notes"][0]["pitch"] = 64
    (project / "notes.json").write_text(json.dumps(payload), encoding="utf-8")
    assert revision_status(project.name, project, load_events(store), store).state == "local_changes"
    shutil.rmtree(first)
    restored = restore_missing_project(
        store=store,
        projects_root=tmp_path / "independent-restore",
        event_id=second.name,
    )
    assert (
        json.loads((restored.project / "notes.json").read_text(encoding="utf-8"))[
            "notes"
        ][0]["pitch"]
        == 62
    )


def test_analyze_event_restores_its_causal_score_overlay(tmp_path):
    store = tmp_path / "store"
    project = _project(tmp_path / "piece")
    score = _publish(project, store)
    (project / "annotations.json").write_text(
        '{"version": 1, "chords": [{"roman": "i"}]}\n',
        encoding="utf-8",
    )
    analysis = publish_analysis(project, store)
    assert analysis is not None

    restored = restore_missing_project(
        store=store,
        projects_root=tmp_path / "restored",
        event_id=analysis.name,
    )
    assert (restored.project / "notes.json").is_file()
    assert '"roman": "i"' in (restored.project / "annotations.json").read_text(
        encoding="utf-8"
    )
    manifest = load_project_manifest(restored.project)
    assert manifest["restored_from"]["score_event"] == score.name
    assert manifest["restored_from"]["analysis_event"] == analysis.name
    changed = json.loads((project / "notes.json").read_text(encoding="utf-8"))
    changed["notes"][0]["pitch"] = 71
    (project / "notes.json").write_text(json.dumps(changed), encoding="utf-8")
    (project / "annotations.json").write_text('{"locally_changed": true}\n', encoding="utf-8")
    before = len(load_events(store))
    assert publish_analysis(project, store) is None
    assert len(load_events(store)) == before


def test_update_preserves_local_annotations_audio_and_permanent_recovery(tmp_path):
    store = tmp_path / "store"
    publisher = _project(tmp_path / "publisher" / "piece", pitch=60)
    first = _publish(publisher, store)
    projects_root = tmp_path / "transcriptions"
    local = projects_root / "piece"
    shutil.copytree(publisher, local)
    (local / "annotations.json").write_text('{"local": true}\n', encoding="utf-8")

    _project(publisher, pitch=67, annotation='{"published": true}\n')
    second = _publish(publisher, store, action="quantize")
    preview = preview_restore(
        store=store,
        projects_root=projects_root,
        event_id=second.name,
    )
    assert preview.recovery_required is True
    assert "notes.json" in preview.changed_components

    result = update_project(
        store=store,
        projects_root=projects_root,
        event_id=second.name,
    )
    notes = json.loads((local / "notes.json").read_text(encoding="utf-8"))
    assert notes["notes"][0]["pitch"] == 67
    assert json.loads((local / "annotations.json").read_text(encoding="utf-8")) == {
        "local": True
    }
    assert (local / "working.wav").read_bytes() == b"local audio"
    assert result.recovery is not None and result.recovery.is_dir()
    assert (result.recovery / "recovery.json").is_file()
    assert len(list_recoveries(projects_root)) == 1

    with pytest.raises(RestoreError):
        remove_recovery(
            projects_root,
            project="piece",
            recovery_id=result.recovery.name,
            confirm="yes",
        )
    removed = remove_recovery(
        projects_root,
        project="piece",
        recovery_id=result.recovery.name,
        confirm=f"DELETE piece/{result.recovery.name}",
    )
    assert removed == result.recovery
    assert not removed.exists()
    assert first.is_dir() and second.is_dir()


def test_explicit_recovery_rollback_is_verified_and_retains_both_states(tmp_path):
    store = tmp_path / "store"
    publisher = _project(tmp_path / "publisher" / "piece", pitch=60)
    _publish(publisher, store)
    projects_root = tmp_path / "transcriptions"
    local = projects_root / "piece"
    shutil.copytree(publisher, local)
    _project(publisher, pitch=72)
    target = _publish(publisher, store, action="quantize")
    updated = update_project(
        store=store,
        projects_root=projects_root,
        event_id=target.name,
    )
    assert updated.recovery is not None

    with pytest.raises(RestoreError, match="Confirmation must be exactly"):
        rollback_recovery(
            projects_root,
            project="piece",
            recovery_id=updated.recovery.name,
            confirm="yes",
        )
    rolled_back = rollback_recovery(
        projects_root,
        project="piece",
        recovery_id=updated.recovery.name,
        confirm=f"RESTORE piece/{updated.recovery.name}",
    )
    notes = json.loads((local / "notes.json").read_text(encoding="utf-8"))
    assert notes["notes"][0]["pitch"] == 60
    assert updated.recovery.is_dir()
    assert rolled_back.recovery is not None and rolled_back.recovery.is_dir()
    assert len(list_recoveries(projects_root)) == 2


def test_hash_mismatch_blocks_restore_before_writing(tmp_path):
    store = tmp_path / "store"
    project = _project(tmp_path / "piece")
    event = _publish(project, store)
    (event / "notes.json").write_text('{"tampered": true}\n', encoding="utf-8")
    root = tmp_path / "transcriptions"
    with pytest.raises(RestoreError, match="hash mismatch"):
        restore_missing_project(store=store, projects_root=root, event_id=event.name)
    assert not (root / "piece").exists()


def test_byte_hash_blocks_semantically_equivalent_event_tampering(tmp_path):
    store = tmp_path / "store"
    event = _publish(_project(tmp_path / "piece"), store)
    (event / "score.musicxml").write_text(
        "<score-partwise id='changed'/>\n",
        encoding="utf-8",
    )

    with pytest.raises(RestoreError, match="byte hash mismatch"):
        restore_missing_project(
            store=store,
            projects_root=tmp_path / "transcriptions",
            event_id=event.name,
        )


def test_musicxml_fingerprint_ignores_volatile_ids_but_integrity_does_not(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "score.musicxml").write_text(
        "<score-partwise id='one'><identification><encoding>"
        "<encoding-date>2026-01-01</encoding-date></encoding></identification>"
        "<part id='part-one'/></score-partwise>",
        encoding="utf-8",
    )
    (second / "score.musicxml").write_text(
        "<score-partwise id='two'><identification><encoding>"
        "<encoding-date>2026-11-10</encoding-date></encoding></identification>"
        "<part id='part-two'/></score-partwise>",
        encoding="utf-8",
    )

    assert artifact_digest(first / "score.musicxml") == artifact_digest(
        second / "score.musicxml"
    )
    assert snapshot_for_folder(first).sha256 == snapshot_for_folder(second).sha256
    assert sha256_file(first / "score.musicxml") != sha256_file(
        second / "score.musicxml"
    )


def test_local_audio_retention_requires_matching_source_hashes():
    local = {
        "origin": r"C:\recordings\piece.wav",
        "source_kind": "file",
        "audio": {
            "excerpt_start_seconds": 0.0,
            "excerpt_end_seconds": 10.0,
            "working_duration_seconds": 10.0,
        },
    }
    target = {
        "source": {"kind": "file", "name": "piece.wav"},
        "audio": dict(local["audio"]),
    }
    assert _same_source_and_excerpt(local, target) is False
    local["source_sha256"] = "a" * 64
    target["source"]["sha256"] = "a" * 64
    assert _same_source_and_excerpt(local, target) is True
    target["source"]["sha256"] = "b" * 64
    assert _same_source_and_excerpt(local, target) is False


def test_event_names_cannot_escape_storage_roots():
    payload = {
        "version": 2,
        "id": "../outside",
        "recorded_at": "2026-01-01T00:00:00Z",
        "action": "prepare",
        "status": "ok",
        "project": "piece",
        "revision": {
            "lineage_id": "lineage",
            "parent_event_id": None,
            "complete_snapshot": False,
            "snapshot": {"sha256": "snapshot", "artifacts": {}, "files": {}},
        },
        "project_state": {},
    }
    with pytest.raises(ValueError, match="plain name"):
        validate_event(payload)


def test_export_and_storage_report_keep_recovery_separate(tmp_path):
    root = tmp_path / "repo"
    project = _project(root / "transcriptions" / "piece")
    recovery = root / "transcriptions" / ".recovery" / "piece" / "r1"
    shutil.copytree(project, recovery)
    (root / ".local").mkdir()
    (root / ".local" / "cache.bin").write_bytes(b"cache")

    exported = export_project(project, root / "backups" / "piece")
    assert exported.suffix == ".zip"
    assert recovery.is_dir()
    report = storage_report(root)
    assert report.active_projects > 0
    assert report.recovery_copies > 0
    assert report.rebuildable_caches == len(b"cache")


def test_atomic_json_and_staged_event_failures_leave_prior_state(
    tmp_path, monkeypatch
):
    target = tmp_path / "state.json"
    target.write_text('{"old": true}\n', encoding="utf-8")

    def fail_replace(source, destination):
        raise OSError("interrupted")

    monkeypatch.setattr("baroque_transcribe.storage_io.os.replace", fail_replace)
    with pytest.raises(OSError, match="interrupted"):
        write_json_atomic(target, {"new": True})
    assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
    assert not list(tmp_path.glob(".*.tmp.json"))

    monkeypatch.undo()
    project = _project(tmp_path / "piece")
    store = tmp_path / "store"

    def fail_validation(folder, payload):
        raise OSError("invalid staged event")

    monkeypatch.setattr(
        "baroque_transcribe.parse_store._validate_staged_event",
        fail_validation,
    )
    assert (
        record_parse(
            action="transcribe",
            status="ok",
            source="https://youtu.be/abcdefghijk",
            project=project.name,
            project_dir=project,
            store=store,
        )
        is None
    )
    assert load_events(store) == []
    assert not list((store / "events").glob(".tmp-*"))
    assert "revision" not in load_project_manifest(project)


def test_score_event_without_complete_artifacts_is_not_published(tmp_path):
    project = _project(tmp_path / "piece")
    (project / "pedal.json").unlink()
    store = tmp_path / "store"

    assert (
        record_parse(
            action="transcribe",
            status="ok",
            source="https://youtu.be/abcdefghijk",
            project=project.name,
            project_dir=project,
            store=store,
        )
        is None
    )
    assert load_events(store) == []
    assert not list((store / "events").glob(".tmp-*"))
    assert "revision" not in load_project_manifest(project)


def test_exact_published_head_recovers_from_failed_revision_marker(
    tmp_path, monkeypatch
):
    store = tmp_path / "store"
    project = _project(tmp_path / "piece", pitch=60)
    _publish(project, store)
    _project(project, pitch=64)

    def fail_marker(*args, **kwargs):
        raise OSError("marker interrupted")

    monkeypatch.setattr(
        "baroque_transcribe.parse_store.write_project_manifest",
        fail_marker,
    )
    event = record_parse(
        action="quantize",
        status="ok",
        source="https://youtu.be/abcdefghijk",
        project=project.name,
        project_dir=project,
        store=store,
    )
    assert event is not None
    status = revision_status(project.name, project, load_events(store), store)
    assert status.state == "legacy_match"
    assert status.adoptable_event_id == event.name

    monkeypatch.undo()
    assert adopt_exact_match(
        store=store,
        projects_root=project.parent,
        project=project.name,
    ).state == "synchronized"


def test_revision_states_use_ancestry_not_timestamps(tmp_path):
    store = tmp_path / "store"
    publisher = _project(tmp_path / "publisher" / "piece", pitch=60)
    first = _publish(
        publisher,
        store,
        recorded_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    projects = tmp_path / "transcriptions"
    local = projects / "piece"
    shutil.copytree(publisher, local)

    _project(publisher, pitch=62)
    second = _publish(
        publisher,
        store,
        action="quantize",
        recorded_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    events = load_events(store)
    status = revision_status("piece", local, events, store)
    assert status.state == "newer_published"
    assert status.target_event_id == second.name

    notes = json.loads((local / "notes.json").read_text(encoding="utf-8"))
    notes["notes"][0]["pitch"] = 65
    (local / "notes.json").write_text(json.dumps(notes), encoding="utf-8")
    assert revision_status("piece", local, events, store).state == "diverged"
    assert revision_status("piece", None, events, store).state == "not_restored"
    assert (
        revision_status(
            "independent",
            _project(projects / "independent"),
            events,
            store,
        ).state
        == "independent"
    )

    branch_a = tmp_path / "branches" / "a"
    branch_b = tmp_path / "branches" / "b"
    shutil.copytree(publisher, branch_a)
    shutil.copytree(publisher, branch_b)
    # Both branches point at second and publish incomparable children.
    _project(branch_a, pitch=67)
    _project(branch_b, pitch=69)
    _publish(branch_a, store, action="quantize", project_name="piece")
    _publish(branch_b, store, action="quantize", project_name="piece")
    assert revision_status("piece", local, load_events(store), store).state == (
        "published_conflict"
    )
    assert revision_status("piece", None, load_events(store), store).state == (
        "published_conflict"
    )
    assert first.is_dir()


def test_independent_lineages_with_same_project_name_are_conflict(tmp_path):
    store = tmp_path / "store"
    first_publisher = _project(tmp_path / "publishers" / "one", pitch=60)
    first = _publish(first_publisher, store, project_name="shared-name")
    projects = tmp_path / "transcriptions"
    local = projects / "shared-name"
    shutil.copytree(first_publisher, local)

    second_publisher = _project(tmp_path / "publishers" / "two", pitch=72)
    second = _publish(second_publisher, store, project_name="shared-name")

    status = revision_status("shared-name", local, load_events(store), store)
    assert status.state == "published_conflict"
    assert set(status.published_event_ids) == {first.name, second.name}


def test_v1_event_remains_readable_and_exactly_adoptable(tmp_path):
    store = tmp_path / "store"
    event_id = "20200101T000000Z-legacy"
    event_folder = store / "events" / event_id
    legacy_project = _project(tmp_path / "transcriptions" / "legacy")
    event_folder.mkdir(parents=True)
    for name in ("notes.json", "pedal.json", "raw.mid", "quantized.mid", "score.musicxml"):
        shutil.copy2(legacy_project / name, event_folder / name)
    (event_folder / "event.json").write_text(
        json.dumps(
            {
                "version": 1,
                "id": event_id,
                "recorded_at": "2020-01-01T00:00:00Z",
                "machine": "old",
                "action": "transcribe",
                "status": "ok",
                "project": "legacy",
                "source": {"kind": "file", "name": "source.wav"},
            }
        ),
        encoding="utf-8",
    )
    events = load_events(store)
    assert events[0]["version"] == 1
    status = revision_status("legacy", legacy_project, events, store)
    assert status.state == "legacy_match"
    assert status.adoptable_event_id == event_id
    unrelated = _project(tmp_path / "other" / "legacy", pitch=71)
    assert revision_status("legacy", unrelated, events, store).state == "legacy_unknown"
    corrupt_project = tmp_path / "corrupt" / "legacy"
    shutil.copytree(legacy_project, corrupt_project)
    corrupt_manifest = corrupt_project / "manifest.json"
    corrupt_manifest.write_text('{"title": "unfinished"', encoding="utf-8")
    with pytest.raises(ValueError, match="Could not read"):
        adopt_exact_match(
            store=store,
            projects_root=corrupt_project.parent,
            project="legacy",
        )
    assert corrupt_manifest.read_text(encoding="utf-8") == '{"title": "unfinished"'
    restored = restore_missing_project(
        store=store,
        projects_root=tmp_path / "legacy-restore",
        event_id=event_id,
    )
    assert (restored.project / "notes.json").is_file()
    adopted = adopt_exact_match(
        store=store,
        projects_root=legacy_project.parent,
        project="legacy",
    )
    assert adopted.state == "synchronized"


def test_v1_restore_applies_later_analysis_and_tracks_local_edits(tmp_path):
    store = tmp_path / "store"
    source = _project(tmp_path / "source" / "legacy")
    score_id = "20200101T000000Z-score"
    score_folder = store / "events" / score_id
    score_folder.mkdir(parents=True)
    for name in ("notes.json", "pedal.json", "raw.mid", "quantized.mid", "score.musicxml"):
        shutil.copy2(source / name, score_folder / name)
    (score_folder / "event.json").write_text(
        json.dumps(
            {
                "version": 1,
                "id": score_id,
                "recorded_at": "2020-01-01T00:00:00Z",
                "action": "transcribe",
                "status": "ok",
                "project": "legacy",
                "source": {"kind": "file", "name": "source.wav"},
            }
        ),
        encoding="utf-8",
    )
    analysis_id = "20200102T000000Z-analysis"
    analysis_folder = store / "events" / analysis_id
    analysis_folder.mkdir()
    (analysis_folder / "annotations.json").write_text(
        '{"version": 1, "chords": [{"roman": "V"}]}\n',
        encoding="utf-8",
    )
    (analysis_folder / "event.json").write_text(
        json.dumps(
            {
                "version": 1,
                "id": analysis_id,
                "recorded_at": "2020-01-02T00:00:00Z",
                "action": "analyze",
                "status": "ok",
                "project": "legacy",
                "source": {"kind": "file", "name": "source.wav"},
            }
        ),
        encoding="utf-8",
    )

    root = tmp_path / "restored"
    restored = restore_missing_project(
        store=store,
        projects_root=root,
        event_id=score_id,
    ).project
    assert '"roman": "V"' in (restored / "annotations.json").read_text(
        encoding="utf-8"
    )
    events = load_events(store)
    assert revision_status("legacy", restored, events, store).state == "synchronized"
    notes = json.loads((restored / "notes.json").read_text(encoding="utf-8"))
    notes["notes"][0]["pitch"] = 73
    (restored / "notes.json").write_text(json.dumps(notes), encoding="utf-8")
    assert revision_status("legacy", restored, events, store).state == "local_changes"


def test_failed_update_rolls_back_and_retains_recovery(tmp_path, monkeypatch):
    store = tmp_path / "store"
    publisher = _project(tmp_path / "publisher" / "piece", pitch=60)
    _publish(publisher, store)
    projects_root = tmp_path / "transcriptions"
    local = projects_root / "piece"
    shutil.copytree(publisher, local)
    _project(publisher, pitch=72)
    target = _publish(publisher, store, action="quantize")
    original_replace = __import__(
        "baroque_transcribe.project_restore", fromlist=["os"]
    ).os.replace

    def interrupt(source, destination):
        source_path = Path(source)
        destination_path = Path(destination)
        if source_path.name.startswith(".tmp-update-") and destination_path == local:
            raise OSError("simulated interruption")
        return original_replace(source, destination)

    monkeypatch.setattr("baroque_transcribe.project_restore.os.replace", interrupt)
    with pytest.raises(RestoreError, match="prior project was restored"):
        update_project(
            store=store,
            projects_root=projects_root,
            event_id=target.name,
        )
    notes = json.loads((local / "notes.json").read_text(encoding="utf-8"))
    assert notes["notes"][0]["pitch"] == 60
    recoveries = list_recoveries(projects_root)
    assert len(recoveries) == 1
    assert recoveries[0]["target_event_id"] == target.name


def test_history_cli_previews_and_requires_exact_update_confirmation(
    tmp_path, monkeypatch, capsys
):
    from baroque_transcribe.cli import main

    store = tmp_path / "store"
    publisher = _project(tmp_path / "publisher" / "piece", pitch=60)
    _publish(publisher, store)
    projects_root = tmp_path / "transcriptions"
    shutil.copytree(publisher, projects_root / "piece")
    _project(publisher, pitch=66)
    target = _publish(publisher, store, action="quantize")
    monkeypatch.setattr("baroque_transcribe.parse_store.default_store", lambda: store)
    monkeypatch.setattr(
        "baroque_transcribe.library.transcriptions_root",
        lambda root=None: Path(root) if root is not None else projects_root,
    )

    assert main(["history", "status", "piece"]) == 0
    assert "newer_published" in capsys.readouterr().out
    assert main(["history", "preview", target.name]) == 0
    assert f"Target event: {target.name}" in capsys.readouterr().out
    assert main(["history", "update", target.name]) == 1
    assert "repeat the exact event id" in capsys.readouterr().err
    assert (
        main(
            [
                "history",
                "update",
                target.name,
                "--confirm",
                target.name,
            ]
        )
        == 0
    )
    assert "Recovery retained" in capsys.readouterr().out


def test_midi_and_musicxml_failures_preserve_prior_artifacts(tmp_path, monkeypatch):
    import pretty_midi

    from baroque_transcribe.midi_export import write_raw_midi
    from baroque_transcribe.notation import NotationError, write_musicxml
    from baroque_transcribe.quantize import quantize_notes
    from baroque_transcribe.schema import NoteEvent

    notes = [NoteEvent(pitch=60, onset=0.0, offset=0.5, velocity=80)]
    midi_path = tmp_path / "raw.mid"
    midi_path.write_bytes(b"prior midi")

    def fail_midi(self, filename):
        raise OSError("MIDI interrupted")

    monkeypatch.setattr(pretty_midi.PrettyMIDI, "write", fail_midi)
    with pytest.raises(OSError, match="MIDI interrupted"):
        write_raw_midi(midi_path, notes, [])
    assert midi_path.read_bytes() == b"prior midi"

    monkeypatch.undo()
    score_path = tmp_path / "score.musicxml"
    score_path.write_text("<prior/>\n", encoding="utf-8")
    quantization = quantize_notes(
        notes,
        bpm=120,
        numerator=4,
        denominator=4,
        subdivision=16,
        downbeat=0.0,
    )

    def fail_score(self, fmt=None, fp=None, **kwargs):
        raise OSError("MusicXML interrupted")

    monkeypatch.setattr("music21.stream.Score.write", fail_score)
    with pytest.raises(NotationError, match="MusicXML export failed"):
        write_musicxml(score_path, quantization)
    assert score_path.read_text(encoding="utf-8") == "<prior/>\n"
