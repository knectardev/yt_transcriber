"""Fast MusicXML movement slicing (no music21 parse)."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from baroque_transcribe.movement_scores import slice_musicxml_measures


def _minimal_score(path: Path) -> None:
    path.write_text(
        """<?xml version="1.0" encoding="utf-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN"
  "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="4.0">
  <part-list>
    <score-part id="P1"><part-name>Piano</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes>
        <divisions>4</divisions>
        <key><fifths>0</fifths></key>
        <time><beats>4</beats><beat-type>4</beat-type></time>
        <staves>1</staves>
        <clef><sign>G</sign><line>2</line></clef>
      </attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><type>quarter</type></note>
    </measure>
    <measure number="2">
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>4</duration><type>quarter</type></note>
    </measure>
    <measure number="3">
      <attributes><key><fifths>2</fifths></key></attributes>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>4</duration><type>quarter</type></note>
    </measure>
    <measure number="4">
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>4</duration><type>quarter</type></note>
    </measure>
  </part>
</score-partwise>
""",
        encoding="utf-8",
    )


def test_slice_musicxml_carries_attributes_and_renumbers(tmp_path: Path):
    source = tmp_path / "score.musicxml"
    dest = tmp_path / "excerpt.musicxml"
    _minimal_score(source)

    slice_musicxml_measures(source, dest, first=3, last=4, title="Sarabande")

    root = ET.parse(dest).getroot()
    assert root.findtext("movement-title") == "Sarabande"
    part = root.find("part")
    assert part is not None
    measures = list(part.findall("measure"))
    assert [m.get("number") for m in measures] == ["1", "2"]
    attrs = measures[0].find("attributes")
    assert attrs is not None
    assert attrs.findtext("divisions") == "4"
    assert attrs.findtext("key/fifths") == "2"
    assert attrs.find("clef") is not None
    assert measures[1].find("attributes") is None
