#!/usr/bin/env python3
"""Curator-only refresh of the Bach keyboard catalog metadata.

Parses only the 16 allowlisted Tobis Notenarchiv keyboard-work pages and
writes ``data/catalog/bach-keyboard.json``. Runtime browsing uses that
checked-in manifest; it does not scrape the website.

Optional ``--with-hashes`` downloads each MIDI file with the polite overnight
scheduler and fills SHA-256 / size / parse-audit fields. Without it, those
fields stay empty until a later sync or hash refresh.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin
from urllib.request import Request, urlopen

REPO = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO / "data" / "catalog" / "bach-keyboard.json"

COLLECTIONS: list[tuple[str, str, str]] = [
    (
        "bwv-772-786-inventions",
        "BWV 772-786 Inventions",
        "https://tobis-notenarchiv.de/wp/en/bach-archive/instrumental-works/keyboard-works/inventions/",
    ),
    (
        "bwv-787-801-sinfonias",
        "BWV 787-801 Sinfonias",
        "https://tobis-notenarchiv.de/wp/en/bach-archive/instrumental-works/keyboard-works/sinfonias/",
    ),
    (
        "bwv-802-805-four-duets",
        "BWV 802-805 Four Duets",
        "https://tobis-notenarchiv.de/wp/en/bach-archive/instrumental-works/keyboard-works/duets/",
    ),
    (
        "bwv-806-811-english-suites",
        "BWV 806-811 English Suites",
        "https://tobis-notenarchiv.de/wp/en/bach-archive/instrumental-works/keyboard-works/english-suites/",
    ),
    (
        "bwv-812-817-french-suites",
        "BWV 812-817 French Suites",
        "https://tobis-notenarchiv.de/wp/en/bach-archive/instrumental-works/keyboard-works/french-suites/",
    ),
    (
        "bwv-818-824-miscellaneous-suites",
        "BWV 818-824 Miscellaneous Suites",
        "https://tobis-notenarchiv.de/wp/en/bach-archive/instrumental-works/keyboard-works/miscellaneous-suites/",
    ),
    (
        "bwv-825-831-partitas",
        "BWV 825-831 Partitas",
        "https://tobis-notenarchiv.de/wp/en/bach-archive/instrumental-works/keyboard-works/partitas/",
    ),
    (
        "bwv-832-845-suites-and-suite-movements",
        "BWV 832-845 Suites and Suite Movements",
        "https://tobis-notenarchiv.de/wp/en/bach-archive/instrumental-works/keyboard-works/suites-and-suite-movements/",
    ),
    (
        "bwv-846-869-wtc-i",
        "BWV 846-869 The Well-Tempered Clavier Book I",
        "https://tobis-notenarchiv.de/wp/en/bach-archive/instrumental-works/keyboard-works/the-well-tempered-clavier-book-i/",
    ),
    (
        "bwv-870-893-wtc-ii",
        "BWV 870-893 The Well-Tempered Clavier Book II",
        "https://tobis-notenarchiv.de/wp/en/bach-archive/instrumental-works/keyboard-works/the-well-tempered-clavier-book-ii/",
    ),
    (
        "bwv-894-902-preludes-and-fugues",
        "BWV 894-902 Preludes and Fugues/Fughettas",
        "https://tobis-notenarchiv.de/wp/en/bach-archive/instrumental-works/keyboard-works/preludes-and-fugues/",
    ),
    (
        "bwv-903-909-fantasias-and-fugues",
        "BWV 903-909 Fantasias and Fugues/Fughettas",
        "https://tobis-notenarchiv.de/wp/en/bach-archive/instrumental-works/keyboard-works/fantasias-and-fugues/",
    ),
    (
        "bwv-910-923-toccatas-fantasias-and-preludes",
        "BWV 910-923 Toccatas, Fantasias and Preludes",
        "https://tobis-notenarchiv.de/wp/en/bach-archive/instrumental-works/keyboard-works/toccatas-fantasias-and-preludes/",
    ),
    (
        "bwv-924-932-little-preludes-wf",
        "BWV 924-932 Little Preludes (Clavierbüchlein for W.F. Bach)",
        "https://tobis-notenarchiv.de/wp/en/bach-archive/instrumental-works/keyboard-works/little-preludes-clavier-buechlein-for-w-f-bach/",
    ),
    (
        "bwv-933-938-six-little-preludes",
        "BWV 933-938 Six Little Preludes",
        "https://tobis-notenarchiv.de/wp/en/bach-archive/instrumental-works/keyboard-works/six-little-preludes/",
    ),
    (
        "bwv-939-943-five-little-preludes",
        "BWV 939-943 Five Little Preludes",
        "https://tobis-notenarchiv.de/wp/en/bach-archive/instrumental-works/keyboard-works/five-little-preludes/",
    ),
]

USER_AGENT = "baroque-transcribe-catalog-refresh/0.1 (local curator; personal noncommercial use)"
BWV_RE = re.compile(r"^BWV\s+(\d+[a-z]?)\b", re.I)
BWV_FILE_RE = re.compile(r"BWV[_\s-]?(\d+[a-z]?)", re.I)
KEY_RE = re.compile(
    r"\bin\s+([A-Ga-g](?:\s*(?:flat|sharp|#|b))?\s*(?:major|minor|Dur|Moll))\b",
    re.I,
)
MIDI_RE = re.compile(r"\.midi?($|\?)", re.I)


class TableParser(HTMLParser):
    def __init__(self, base: str) -> None:
        super().__init__()
        self.base = base
        self.in_tr = False
        self.in_td = False
        self.row_cells: list[dict] = []
        self.cell_text: list[str] = []
        self.cell_links: list[str] = []
        self.rows: list[list[dict]] = []

    def handle_starttag(self, tag: str, attrs) -> None:  # type: ignore[override]
        attrs_dict = dict(attrs)
        if tag == "tr":
            self.in_tr = True
            self.row_cells = []
        elif tag == "td" and self.in_tr:
            self.in_td = True
            self.cell_text = []
            self.cell_links = []
        elif tag == "a" and self.in_td:
            href = attrs_dict.get("href", "")
            if href:
                self.cell_links.append(urljoin(self.base, href))

    def handle_endtag(self, tag: str) -> None:  # type: ignore[override]
        if tag == "td" and self.in_td:
            self.in_td = False
            self.row_cells.append(
                {"text": "".join(self.cell_text).strip(), "links": list(self.cell_links)}
            )
        elif tag == "tr" and self.in_tr:
            self.in_tr = False
            if self.row_cells:
                self.rows.append(self.row_cells)

    def handle_data(self, data: str) -> None:  # type: ignore[override]
        if self.in_td:
            self.cell_text.append(data)


def _fetch(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", "replace")


def _normalize_key(raw: str) -> str:
    key = re.sub(r"\s+", " ", raw).strip()
    key = key.replace(" Dur", " major").replace(" Moll", " minor")
    return key


def _stable_id(filename: str) -> str:
    stem = re.sub(r"\.midi?$", "", filename, flags=re.I).lower().replace("_", "-")
    return stem


def scrape_collection(collection_id: str, collection_title: str, page: str) -> list[dict]:
    html = _fetch(page)
    parser = TableParser(page)
    parser.feed(html)
    entries: list[dict] = []
    order = 0
    for row in parser.rows:
        texts = [cell["text"] for cell in row if cell["text"]]
        midis: list[str] = []
        for cell in row:
            for link in cell["links"]:
                if MIDI_RE.search(link):
                    midis.append(link)
        if not midis:
            continue
        label = " — ".join(texts) if texts else midis[0]
        title_cell = texts[0] if texts else ""
        match = BWV_RE.search(title_cell) or BWV_RE.search(label)
        filename = midis[0].rsplit("/", 1)[-1]
        if match:
            bwv = match.group(1)
            title = title_cell[match.end() :].strip(" —-\t") or label
        else:
            file_match = BWV_FILE_RE.search(filename)
            bwv = file_match.group(1) if file_match else ""
            title = title_cell or label
        key_match = KEY_RE.search(label)
        key = _normalize_key(key_match.group(1)) if key_match else ""
        lower = label.lower()
        is_variant = bool(
            (bwv and re.search(r"[a-z]$", bwv))
            or "variant" in lower
            or "alternate" in lower
            or "early version" in lower
        )
        is_arrangement = "arrangement" in lower
        order += 1
        entries.append(
            {
                "id": _stable_id(filename),
                "bwv": bwv,
                "title": title,
                "collection_id": collection_id,
                "collection": collection_title,
                "order_in_collection": order,
                "key": key,
                "is_variant": is_variant,
                "is_arrangement": is_arrangement,
                "source_page": page,
                "midi_url": midis[0],
                "source_filename": filename,
                "sha256": "",
                "size_bytes": None,
                "duration_seconds": None,
                "note_count": None,
                "track_count": None,
                "import_defaults": {},
            }
        )
    return entries


def _dedupe_ids(entries: list[dict]) -> None:
    seen: dict[str, int] = {}
    for entry in entries:
        entry_id = entry["id"]
        if entry_id in seen:
            seen[entry_id] += 1
            entry["id"] = f"{entry_id}-{seen[entry_id]}"
        else:
            seen[entry_id] = 1


def build_manifest(entries: list[dict]) -> dict:
    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "schema_version": 1,
        "id": "bach-keyboard",
        "title": "J.S. Bach keyboard works (BWV 772–943)",
        "composer": "Johann Sebastian Bach",
        "source": {
            "name": "Tobis Notenarchiv",
            "home_url": "https://tobis-notenarchiv.de/wp/",
            "keyboard_index_url": (
                "https://tobis-notenarchiv.de/wp/en/bach-archive/instrumental-works/keyboard-works/"
            ),
            "terms_en_url": "https://tobis-notenarchiv.de/wp/en/terms-of-use/",
            "terms_de_url": (
                "https://tobis-notenarchiv.de/wp/urheberrecht-und-nutzungsbeschraenkungen/"
            ),
            "license_notes": (
                "English terms page currently shows CC BY-NC; German terms page currently "
                "shows CC BY-NC-SA 4.0. Treat as noncommercial with attribution required. "
                "MIDI bytes are not redistributed with this repository; they are downloaded "
                "into a local catalog mirror for personal noncommercial use."
            ),
            "attribution": "MIDI files from Tobis Notenarchiv (tobis-notenarchiv.de).",
            "retrieved_at": now,
            "allowed_hosts": [
                "tobis-notenarchiv.de",
                "www.tobis-notenarchiv.de",
            ],
            "collections": [
                {"id": collection_id, "title": title, "page_url": page}
                for collection_id, title, page in COLLECTIONS
            ],
        },
        "entry_count": len(entries),
        "entries": entries,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=MANIFEST_PATH,
        help="Where to write the catalog manifest JSON.",
    )
    parser.add_argument(
        "--with-hashes",
        action="store_true",
        help="Download each MIDI politely and fill sha256/size/audit fields (slow).",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required with --with-hashes to start the overnight download.",
    )
    args = parser.parse_args(argv)

    entries: list[dict] = []
    for collection_id, title, page in COLLECTIONS:
        print(f"Scraping {title} …", file=sys.stderr)
        entries.extend(scrape_collection(collection_id, title, page))
    _dedupe_ids(entries)

    counts = Counter(entry["collection_id"] for entry in entries)
    print(f"Collected {len(entries)} MIDI rows across {len(counts)} collections.", file=sys.stderr)
    for collection_id, _title, _page in COLLECTIONS:
        print(f"  {collection_id}: {counts.get(collection_id, 0)}", file=sys.stderr)

    if args.with_hashes:
        if not args.confirm:
            print("Refusing --with-hashes without --confirm.", file=sys.stderr)
            return 2
        sys.path.insert(0, str(REPO / "src"))
        from baroque_transcribe.catalog_sync import fill_manifest_hashes

        fill_manifest_hashes(entries, overnight=True)

    manifest = build_manifest(entries)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"Wrote {args.output} ({len(entries)} entries).", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
