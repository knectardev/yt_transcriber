"""Bach catalog metadata, search filters, and materialization helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pretty_midi
import pytest

from baroque_transcribe.catalog import (
    CatalogError,
    catalog_provenance,
    entry_statuses,
    load_catalog,
    search_catalog,
    validate_catalog,
)
from baroque_transcribe.catalog_open import open_catalog_entry
from baroque_transcribe.defaults import NOTES_JSON, SOURCE_MID
from baroque_transcribe.fragments import catalog_from_parent, extract_fragment, plan_fragment
from baroque_transcribe.library import read_run
from baroque_transcribe.playback import load_playback
from baroque_transcribe.project_manifest import load_project_manifest, public_project_state


def _mini_midi(path: Path) -> Path:
    midi = pretty_midi.PrettyMIDI(initial_tempo=100)
    right = pretty_midi.Instrument(program=0, name="Right hand")
    left = pretty_midi.Instrument(program=0, name="Left hand")
    right.notes.append(pretty_midi.Note(velocity=80, pitch=72, start=0.0, end=0.5))
    left.notes.append(pretty_midi.Note(velocity=70, pitch=48, start=0.0, end=0.5))
    midi.instruments.extend([right, left])
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    return path


def _manifest_with(entries: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "id": "bach-keyboard",
        "title": "Test catalog",
        "composer": "Johann Sebastian Bach",
        "source": {
            "name": "Tobis Notenarchiv",
            "home_url": "https://tobis-notenarchiv.de/wp/",
            "keyboard_index_url": "https://example.test/",
            "terms_en_url": "https://example.test/en",
            "terms_de_url": "https://example.test/de",
            "license_notes": "noncommercial",
            "attribution": "MIDI files from Tobis Notenarchiv.",
            "retrieved_at": "2026-11-12T00:00:00Z",
            "allowed_hosts": ["example.test"],
            "collections": [
                {
                    "id": "bwv-772-786-inventions",
                    "title": "Inventions",
                    "page_url": "https://example.test/inventions/",
                }
            ],
        },
        "entry_count": len(entries),
        "entries": entries,
    }


def test_repo_catalog_loads_194_entries():
    catalog = load_catalog()
    assert catalog.entry_count == 194
    assert catalog.entries[0].id.startswith("bwv-")
    assert all(entry.midi_url.startswith("https://") for entry in catalog.entries)
    found = search_catalog(catalog, text="well-tempered", collection_id="bwv-870-893-wtc-ii")
    assert found
    assert all(row.entry.collection_id == "bwv-870-893-wtc-ii" for row in found)


def test_validate_and_search_filters(tmp_path: Path):
    entries = [
        {
            "id": "bwv-0772",
            "bwv": "772",
            "title": "Invention No. 1",
            "collection_id": "bwv-772-786-inventions",
            "collection": "Inventions",
            "order_in_collection": 1,
            "key": "C major",
            "is_variant": False,
            "is_arrangement": False,
            "source_page": "https://example.test/inventions/",
            "midi_url": "https://example.test/BWV_0772.mid",
            "source_filename": "BWV_0772.mid",
            "sha256": "",
        },
        {
            "id": "bwv-0772a",
            "bwv": "772a",
            "title": "Invention No. 1 – variant",
            "collection_id": "bwv-772-786-inventions",
            "collection": "Inventions",
            "order_in_collection": 2,
            "key": "C major",
            "is_variant": True,
            "is_arrangement": False,
            "source_page": "https://example.test/inventions/",
            "midi_url": "https://example.test/BWV_0772a.mid",
            "source_filename": "BWV_0772a.mid",
            "sha256": "",
        },
    ]
    catalog = validate_catalog(_manifest_with(entries))
    assert [row.entry.id for row in search_catalog(catalog, variants="only")] == ["bwv-0772a"]
    assert [row.entry.id for row in search_catalog(catalog, variants="exclude")] == ["bwv-0772"]
    assert [row.entry.id for row in search_catalog(catalog, bwv="772")] == ["bwv-0772"]
    assert [row.entry.id for row in search_catalog(catalog, text="variant")] == ["bwv-0772a"]
    with pytest.raises(CatalogError):
        validate_catalog({"schema_version": 1, "source": {}, "entries": []})


def test_open_catalog_entry_imports_and_extracts(tmp_path: Path, monkeypatch):
    root = tmp_path / "repo"
    midi = _mini_midi(root / "catalog" / "midi" / "bwv-0772" / SOURCE_MID)
    entry = {
        "id": "bwv-0772",
        "bwv": "772",
        "title": "Invention No. 1",
        "collection_id": "bwv-772-786-inventions",
        "collection": "Inventions",
        "order_in_collection": 1,
        "key": "C major",
        "is_variant": False,
        "is_arrangement": False,
        "source_page": "https://example.test/inventions/",
        "midi_url": "https://example.test/BWV_0772.mid",
        "source_filename": "BWV_0772.mid",
        "sha256": "",
        "import_defaults": {},
    }
    manifest_path = root / "data" / "catalog" / "bach-keyboard.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(_manifest_with([entry]), indent=2),
        encoding="utf-8",
    )
    monkeypatch.chdir(root)
    project = open_catalog_entry(
        "bwv-0772",
        output_dir=root / "transcriptions" / "catalog",
        root=root,
        manifest_path=manifest_path,
        sync_if_missing=False,
    )
    run = read_run(project)
    assert run is not None
    assert run.source_kind == "midi"
    manifest = load_project_manifest(project)
    assert manifest["catalog"]["entry_id"] == "bwv-0772"
    assert public_project_state(manifest)["catalog"]["bwv"] == "772"
    assert catalog_from_parent(manifest)["entry_id"] == "bwv-0772"
    context = load_playback(project)
    assert context.media_kind in {"audio", "silent"}
    planned = plan_fragment(project, 1, 1)
    fragment = extract_fragment(
        project,
        1,
        1,
        title="BWV 772 m.1",
        composer="Johann Sebastian Bach",
        tags=["bwv-772"],
        root=root / "data" / "fragments",
    )
    payload = json.loads((fragment / "fragment.json").read_text(encoding="utf-8"))
    assert payload["catalog"]["entry_id"] == "bwv-0772"
    assert (project / NOTES_JSON).is_file()
    assert midi.is_file()
    again = open_catalog_entry(
        "bwv-0772",
        output_dir=root / "transcriptions" / "catalog-again",
        root=root,
        manifest_path=manifest_path,
        sync_if_missing=False,
    )
    assert again.resolve() == project.resolve()


def test_catalog_provenance_helper():
    from baroque_transcribe.catalog import CatalogEntry

    entry = CatalogEntry(
        id="bwv-0870",
        bwv="870",
        title="Prelude and Fugue No. 1",
        collection_id="bwv-870-893-wtc-ii",
        collection="WTC II",
        order_in_collection=1,
        midi_url="https://example.test/x.mid",
        sha256="abc",
    )
    assert catalog_provenance(entry)["pinned_sha256"] == "abc"


def test_entry_statuses_detect_missing(tmp_path: Path):
    entries = [
        {
            "id": "bwv-0772",
            "bwv": "772",
            "title": "Invention No. 1",
            "collection_id": "bwv-772-786-inventions",
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
    ]
    catalog = validate_catalog(_manifest_with(entries))
    rows = entry_statuses(catalog, root=tmp_path)
    assert rows[0].status_label == "not downloaded"


def test_playback_library_home_for_catalog_and_other_projects(tmp_path: Path):
    from baroque_transcribe.playback import library_home_for_project
    from baroque_transcribe.project_manifest import write_project_manifest

    catalog_project = tmp_path / "catalog-work"
    other = tmp_path / "youtube-work"
    catalog_project.mkdir()
    other.mkdir()
    write_project_manifest(
        catalog_project,
        {
            "schema_version": 1,
            "title": "BWV 772",
            "catalog": {"entry_id": "bwv-0772"},
        },
    )
    write_project_manifest(other, {"schema_version": 1, "title": "Other"})
    base = "http://192.168.40.137:8765"
    assert library_home_for_project(catalog_project, base) == (f"{base}/catalog", "Bach catalog")
    assert library_home_for_project(other, base) == (f"{base}/projects", "Projects")
