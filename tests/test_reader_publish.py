"""Incremental reader publish into the static reader/ partition."""

from __future__ import annotations

import json
from pathlib import Path

import pretty_midi
import pytest

from baroque_transcribe.defaults import MANIFEST, SCORE_XML, SCORE_VOICES_XML
from baroque_transcribe.reader_publish import (
    ensure_shell_assets,
    list_published,
    publish_scores,
    remove_published,
)


def _mini_midi(path: Path) -> Path:
    midi = pretty_midi.PrettyMIDI(initial_tempo=100)
    part = pretty_midi.Instrument(program=0, name="Piano")
    part.notes.append(pretty_midi.Note(velocity=80, pitch=60, start=0.0, end=0.5))
    part.notes.append(pretty_midi.Note(velocity=80, pitch=64, start=0.5, end=1.0))
    midi.instruments.append(part)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    return path


def _mini_musicxml(measure_label: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <part-list><score-part id="P1"><part-name>Music</part-name></score-part></part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>1</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>1</duration><type>quarter</type></note>
      <direction><direction-type><words>{measure_label}</words></direction-type></direction>
    </measure>
  </part>
</score-partwise>
"""


def _write_catalog(root: Path, entries: list[dict]) -> Path:
    path = root / "data" / "catalog" / "bach-keyboard.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
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
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _entry(entry_id: str, bwv: str, title: str) -> dict:
    return {
        "id": entry_id,
        "bwv": bwv,
        "title": title,
        "collection_id": "bwv-772-786-inventions",
        "collection": "Inventions",
        "order_in_collection": int(bwv) if bwv.isdigit() else 1,
        "key": "C major",
        "is_variant": False,
        "is_arrangement": False,
        "source_page": "https://example.test/inventions/",
        "midi_url": f"https://example.test/{entry_id}.mid",
        "source_filename": f"{entry_id}.mid",
    }


def _seed_project(root: Path, entry_id: str, label: str) -> Path:
    project = root / "transcriptions" / f"proj-{entry_id}"
    project.mkdir(parents=True, exist_ok=True)
    (project / SCORE_XML).write_text(_mini_musicxml(label), encoding="utf-8")
    (project / SCORE_VOICES_XML).write_text(_mini_musicxml(label + "-voices"), encoding="utf-8")
    (project / MANIFEST).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "catalog": {
                    "entry_id": entry_id,
                    "bwv": entry_id.replace("bwv-0", "").lstrip("0") or "772",
                },
            }
        ),
        encoding="utf-8",
    )
    _mini_midi(project / "source.mid")
    return project


@pytest.fixture
def reader_workspace(tmp_path: Path) -> Path:
    _write_catalog(
        tmp_path,
        [
            _entry("bwv-0772", "772", "Invention No. 1"),
            _entry("bwv-0773", "773", "Invention No. 2"),
        ],
    )
    _seed_project(tmp_path, "bwv-0772", "A")
    _seed_project(tmp_path, "bwv-0773", "B")
    # Minimal shell so ensure_shell_assets has a destination tree.
    (tmp_path / "reader").mkdir(parents=True, exist_ok=True)
    # Point OSMD source by copying real vendor next to installed package path via ensure.
    return tmp_path


def test_incremental_publish_preserves_other_entries(reader_workspace: Path):
    dist = reader_workspace / "reader"
    first = publish_scores(["bwv-0772", "bwv-0773"], dist_root=dist, root=reader_workspace)
    assert first.added == ["bwv-0772", "bwv-0773"]
    assert not first.errors
    ids = {row["entryId"] for row in list_published(dist)}
    assert ids == {"bwv-0772", "bwv-0773"}

    package_a = dist / "catalog" / "packages" / "bwv-0772.json"
    before_hash = json.loads(package_a.read_text(encoding="utf-8"))["contentHash"]

    second = publish_scores(["bwv-0772"], dist_root=dist, root=reader_workspace)
    assert "bwv-0772" in second.unchanged or "bwv-0772" in second.updated
    assert "bwv-0773" not in second.added + second.updated + second.removed
    ids_after = {row["entryId"] for row in list_published(dist)}
    assert ids_after == {"bwv-0772", "bwv-0773"}
    assert (dist / "catalog" / "packages" / "bwv-0773.json").is_file()
    assert json.loads(package_a.read_text(encoding="utf-8"))["contentHash"] == before_hash


def test_publish_update_and_remove(reader_workspace: Path):
    dist = reader_workspace / "reader"
    publish_scores(["bwv-0772", "bwv-0773"], dist_root=dist, root=reader_workspace)

    project = reader_workspace / "transcriptions" / "proj-bwv-0772"
    (project / SCORE_VOICES_XML).write_text(_mini_musicxml("A-changed"), encoding="utf-8")
    (project / SCORE_XML).write_text(_mini_musicxml("A-changed-plain"), encoding="utf-8")

    updated = publish_scores(["bwv-0772"], dist_root=dist, root=reader_workspace)
    assert updated.updated == ["bwv-0772"]
    assert {row["entryId"] for row in list_published(dist)} == {"bwv-0772", "bwv-0773"}

    removed = remove_published(["bwv-0773"], dist_root=dist, root=reader_workspace)
    assert removed.removed == ["bwv-0773"]
    assert {row["entryId"] for row in list_published(dist)} == {"bwv-0772"}
    assert not (dist / "catalog" / "packages" / "bwv-0773.json").is_file()
    assert (dist / "catalog" / "packages" / "bwv-0772.json").is_file()


def test_ensure_shell_assets_copies_osmd(reader_workspace: Path):
    dist = reader_workspace / "reader"
    dest = ensure_shell_assets(dist, root=reader_workspace)
    assert dest.is_file()
    assert dest.name == "opensheetmusicdisplay.min.js"
    assert dest.stat().st_size > 1000


def test_publish_does_not_create_new_project_folder(reader_workspace: Path):
    dist = reader_workspace / "reader"
    before = {path.name for path in (reader_workspace / "transcriptions").iterdir()}
    publish_scores(["bwv-0772"], dist_root=dist, root=reader_workspace)
    after = {path.name for path in (reader_workspace / "transcriptions").iterdir()}
    assert before == after
