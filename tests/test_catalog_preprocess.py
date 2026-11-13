"""Catalog batch preprocess and movement drafts."""

from __future__ import annotations

from pathlib import Path

import pretty_midi
import pytest

from baroque_transcribe.catalog import validate_catalog
from baroque_transcribe.catalog_movements import (
    build_draft_overlay,
    propose_movements_from_midi,
)
from baroque_transcribe.catalog_preprocess import (
    preprocess_catalog,
    project_has_readable_voices,
    project_is_scored,
)
from baroque_transcribe.defaults import (
    NOTATION_MAP_JSON,
    NOTES_JSON,
    SCORE_VOICES_XML,
    SCORE_XML,
    SOURCE_MID,
)
from baroque_transcribe.playback import detail_url, shared_playback_server


def _mini_midi(path: Path, *, tempo_changes: list[tuple[float, float]] | None = None) -> Path:
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    if tempo_changes:
        # pretty_midi stores tempo via get_tempo_changes from internal maps;
        # write successive notes and adjust tempo with change_tempo if available.
        inst = pretty_midi.Instrument(program=0)
        for index, (start, _bpm) in enumerate(tempo_changes):
            inst.notes.append(
                pretty_midi.Note(velocity=80, pitch=60 + index, start=start, end=start + 0.4)
            )
        midi.instruments.append(inst)
        for start, bpm in tempo_changes[1:]:
            midi._tick_scales.append((midi.time_to_tick(start), 60.0 / (bpm * midi.resolution)))
    else:
        inst = pretty_midi.Instrument(program=0)
        inst.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5))
        midi.instruments.append(inst)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    return path


def _manifest_with(entries: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "id": "test",
        "title": "Test",
        "composer": "Bach",
        "entry_count": len(entries),
        "source": {
            "name": "Tobis",
            "home_url": "https://example.test/",
            "keyboard_index_url": "https://example.test/k/",
            "terms_en_url": "https://example.test/en",
            "terms_de_url": "https://example.test/de",
            "license_notes": "nc",
            "attribution": "MIDI files from Tobis Notenarchiv.",
            "retrieved_at": "2026-01-01",
            "allowed_hosts": ["example.test"],
            "collections": [],
        },
        "entries": entries,
    }


def test_propose_movements_from_tempo_map(tmp_path: Path):
    midi = _mini_midi(
        tmp_path / "suite.mid",
        tempo_changes=[(0.0, 120.0), (2.0, 80.0), (4.0, 100.0)],
    )
    # Rebuild with explicit tempo changes via pretty_midi API if tick scales flaky.
    pm = pretty_midi.PrettyMIDI(str(midi))
    if len(pm.get_tempo_changes()[0]) < 2:
        pytest.skip("pretty_midi did not retain tempo changes in this environment")
    row = propose_movements_from_midi(
        midi, entry_id="bwv-0806", collection_id="bwv-806-811-english-suites"
    )
    assert row.status == "draft"
    assert len(row.movements) >= 2
    assert row.movements[0].start_seconds == pytest.approx(0.0)
    assert row.movements[-1].end_seconds > row.movements[0].start_seconds


def test_preprocess_skips_readable_voice_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    entry = {
        "id": "bwv-0772",
        "bwv": "772",
        "title": "Invention No. 1",
        "collection_id": "inv",
        "collection": "Inventions",
        "order_in_collection": 1,
        "key": "",
        "is_variant": False,
        "is_arrangement": False,
        "source_page": "https://example.test/inventions/",
        "midi_url": "https://example.test/BWV_0772.mid",
        "source_filename": "BWV_0772.mid",
        "sha256": "",
    }
    catalog = validate_catalog(_manifest_with([entry]))
    midi = tmp_path / "catalog" / "midi" / "bwv-0772" / SOURCE_MID
    _mini_midi(midi)
    project = tmp_path / "transcriptions" / "already"
    project.mkdir(parents=True)
    (project / NOTES_JSON).write_text('{"schema_version":1,"notes":[]}', encoding="utf-8")
    (project / SCORE_XML).write_text("<score/>", encoding="utf-8")
    (project / SCORE_VOICES_XML).write_text("<score-voices/>", encoding="utf-8")
    (project / NOTATION_MAP_JSON).write_text('{"notes":[]}', encoding="utf-8")

    monkeypatch.setattr(
        "baroque_transcribe.catalog_preprocess.find_project_for_entry",
        lambda entry_id, root=None: project,
    )
    monkeypatch.setattr(
        "baroque_transcribe.catalog_preprocess.entry_midi_path",
        lambda entry_id, root=None: midi,
    )
    opened = []

    def fake_open(entry_id, **kwargs):
        opened.append(entry_id)
        return project

    monkeypatch.setattr("baroque_transcribe.catalog_preprocess.open_catalog_entry", fake_open)
    report = preprocess_catalog(catalog, root=tmp_path, entry_ids=["bwv-0772"])
    assert report.skipped == 1
    assert opened == []
    assert project_is_scored(project)
    assert project_has_readable_voices(project)


def test_preprocess_force_requantizes_readable_project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    entry = {
        "id": "bwv-0772",
        "bwv": "772",
        "title": "Invention No. 1",
        "collection_id": "inv",
        "collection": "Inventions",
        "order_in_collection": 1,
        "key": "",
        "is_variant": False,
        "is_arrangement": False,
        "source_page": "https://example.test/inventions/",
        "midi_url": "https://example.test/BWV_0772.mid",
        "source_filename": "BWV_0772.mid",
        "sha256": "",
    }
    catalog = validate_catalog(_manifest_with([entry]))
    midi = tmp_path / "catalog" / "midi" / "bwv-0772" / SOURCE_MID
    _mini_midi(midi)
    project = tmp_path / "transcriptions" / "already"
    project.mkdir(parents=True)
    (project / NOTES_JSON).write_text('{"schema_version":1,"notes":[]}', encoding="utf-8")
    (project / SCORE_XML).write_text("<score/>", encoding="utf-8")
    (project / SCORE_VOICES_XML).write_text("<score-voices/>", encoding="utf-8")
    (project / NOTATION_MAP_JSON).write_text('{"notes":[]}', encoding="utf-8")
    (project / "manifest.json").write_text(
        '{"title":"Invention","tempo":{"bpm":120,"time_signature":"4/4","subdivision":16,"downbeat_seconds":0}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "baroque_transcribe.catalog_preprocess.find_project_for_entry",
        lambda entry_id, root=None: project,
    )
    monkeypatch.setattr(
        "baroque_transcribe.catalog_preprocess.entry_midi_path",
        lambda entry_id, root=None: midi,
    )
    monkeypatch.setattr(
        "baroque_transcribe.catalog_preprocess.open_catalog_entry",
        lambda entry_id, **kwargs: project,
    )
    calls = []

    def fake_requantize(path, **kwargs):
        calls.append((path, kwargs.get("settings")))
        (path / SCORE_VOICES_XML).write_text("<score-voices/>", encoding="utf-8")
        (path / NOTATION_MAP_JSON).write_text('{"notes":[]}', encoding="utf-8")
        (path / SCORE_XML).write_text("<score/>", encoding="utf-8")

    monkeypatch.setattr(
        "baroque_transcribe.pipeline.requantize_project",
        fake_requantize,
    )
    report = preprocess_catalog(catalog, root=tmp_path, entry_ids=["bwv-0772"], force=True)
    assert report.requantized == 1
    assert len(calls) == 1
    assert calls[0][1].style == "readable"


def test_detail_url_appends_movement_query(tmp_path: Path):
    project = tmp_path / "piece"
    project.mkdir()
    shared_playback_server("http://127.0.0.1:8765")
    url = detail_url(project, movement_id="allemande")
    assert "mov=allemande" in url


def test_measures_for_time_span_includes_measure_underway():
    from baroque_transcribe.movement_scores import measures_for_time_span
    from baroque_transcribe.sync_map import MeasureAnchor

    anchors = [
        MeasureAnchor(measure=1, time=0.0, collided=False, opens_before_attack=False),
        MeasureAnchor(measure=2, time=2.0, collided=False, opens_before_attack=False),
        MeasureAnchor(measure=3, time=4.0, collided=False, opens_before_attack=False),
        MeasureAnchor(measure=4, time=6.0, collided=False, opens_before_attack=False),
        MeasureAnchor(measure=5, time=8.0, collided=False, opens_before_attack=False),
    ]
    # Section starts mid-measure 2 → still include measure 2.
    first, last = measures_for_time_span(anchors, 3.0, 7.0)
    assert first == 2
    assert last == 4


def test_remap_movement_timeline_restarts_at_zero():
    from baroque_transcribe.movement_scores import remap_movement_timeline
    from baroque_transcribe.sync_map import MeasureAnchor, NoteAnchor

    anchors = [
        MeasureAnchor(measure=38, time=90.0, collided=False, opens_before_attack=False),
        MeasureAnchor(measure=39, time=92.5, collided=False, opens_before_attack=False),
        MeasureAnchor(measure=40, time=95.0, collided=False, opens_before_attack=False),
        MeasureAnchor(measure=41, time=97.5, collided=False, opens_before_attack=False),
    ]
    notes = [
        NoteAnchor(source_index=1, pitch=60, onset=92.5, offset=93.0, measure=39),
        NoteAnchor(source_index=2, pitch=62, onset=95.0, offset=95.5, measure=40),
        NoteAnchor(source_index=3, pitch=64, onset=100.0, offset=100.5, measure=41),
    ]
    remapped, remapped_notes, duration = remap_movement_timeline(
        anchors=anchors,
        note_anchors=notes,
        start_seconds=92.5,
        end_seconds=100.0,
        source_first=39,
        source_last=40,
    )
    assert duration == pytest.approx(7.5)
    assert [row.measure for row in remapped] == [1, 2]
    assert remapped[0].time == pytest.approx(0.0)
    assert remapped[1].time == pytest.approx(2.5)
    assert remapped_notes[0].onset == pytest.approx(0.0)
    assert remapped_notes[0].measure == 1
    assert all(note.onset < duration + 1e-6 for note in remapped_notes)
