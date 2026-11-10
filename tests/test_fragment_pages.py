"""The pages of a saved fragment, and passage selection on a transcription's page.

The numbers handed to the page are checked in Python. What the page then
does with them (which measure it highlights, where a loop starts and ends,
what a sync adjustment moves) is checked by running the page's own script
under Node with a stand-in player. Those tests are skipped without Node.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from http.client import HTTPConnection
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

import baroque_transcribe.fragment_pages as P
import baroque_transcribe.fragments as F
from baroque_transcribe.defaults import FRAGMENT_JSON, MANIFEST
from baroque_transcribe.playback import (
    PlaybackServer,
    SyncError,
    extract_page_url,
    load_playback,
    score_end_time,
    write_playback_page,
)
from baroque_transcribe.sync_map import measure_at_working_time, measure_seek_video, working_time
from fragment_helpers import make_project, n, parent_quantization, tree_digest

HARNESS = Path(__file__).with_name("playback_page_harness.js")
HOME = "http://127.0.0.1:8765/"
EXCERPT, SIGMA = 30.0, 0.4
# Measures 2 and 3 of ``piece``: the window opens on the early attack at 1.95 s
# and closes on the written end barline of the parent's last measure at 6.0 s.
T0, DURATION = 1.95, 4.05
ORIGIN = EXCERPT + T0


def piece():
    return [
        n(60, 0.0, 0.5),
        n(62, 0.5, 1.0),
        n(64, 1.0, 1.5),
        n(65, 1.5, 1.9),
        # Measure 2 opens 50 ms before the metronome barline at 2.0 s.
        n(72, 1.95, 2.4),
        n(67, 2.5, 3.0),
        n(69, 3.0, 3.9),
        # Measure 3 opens 50 ms after the metronome barline at 4.0 s.
        n(74, 4.05, 4.5),
        n(76, 4.5, 5.0),
    ]


def saved(tmp_path: Path, **kwargs) -> tuple[Path, Path]:
    """A parent with nonzero excerpt start and sync offset, and measures 2-3 of it saved."""

    options = {"excerpt": EXCERPT, "sync": SIGMA, "analyze": True}
    options.update(kwargs)
    project = make_project(tmp_path, piece(), **options)
    folder = F.extract_fragment(project, 2, 3, title="Two bars", tags=["early"], root=tmp_path / "data" / "fragments")
    return project, folder


def embedded_json(text: str, prefix: str) -> object:
    """The JSON value that follows ``prefix`` in a generated page.

    The page is written with the host's newlines, so splitting on ``;\\n``
    is not reliable. ``raw_decode`` reads one value and stops.
    """

    _, rest = text.split(prefix, 1)
    return json.JSONDecoder().raw_decode(rest.lstrip())[0]


def page_config(page: Path) -> dict:
    """The configuration object a generated playback page starts from."""

    return embedded_json(page.read_text(encoding="utf-8"), "const config = ")


def run_page(tmp_path: Path, page: Path, scenario: str, search: str = "") -> dict:
    """Run a page's script under Node and return what the scenario returned."""

    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is not installed, so the page's script cannot be run.")
    script = tmp_path / "scenario.js"
    script.write_text("(function () {\n" + scenario + "\n})()", encoding="utf-8")
    done = subprocess.run(
        [node, str(HARNESS), str(page), str(script), search],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def get(server: PlaybackServer, path: str) -> tuple[int, str | None]:
    """Status and Location of one GET, without following a redirect."""

    connection = HTTPConnection("127.0.0.1", server.port, timeout=30)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        response.read()
        return response.status, response.getheader("Location")
    finally:
        connection.close()


@pytest.fixture
def server(tmp_path: Path):
    running = PlaybackServer(home_url=HOME)
    running.fragment_pages = tmp_path / ".local" / "fragment_pages"
    running.transcriptions = tmp_path
    running.start()
    yield running
    running.close()


# --------------------------------------------------------------------------
# The numbers handed to the page
# --------------------------------------------------------------------------


def test_fragment_page_is_placed_in_the_video_by_origin_and_offset(tmp_path: Path):
    project, folder = saved(tmp_path)
    context = P.fragment_context(folder)
    parent = load_playback(project)

    assert context.video_id == parent.video_id == "dQw4w9WgXcQ"
    # E + t0, with no sync offset in it; the offset is its own field.
    assert context.excerpt_start == pytest.approx(ORIGIN)
    assert context.sync_offset == SIGMA
    # Anchors are on the fragment clock, and the passage is 0 .. duration on it.
    assert [(anchor.measure, round(anchor.time, 5)) for anchor in context.anchors] == [(1, 0.0), (2, 2.1)]
    assert context.passage == (0.0, pytest.approx(DURATION))
    assert context.end_time == pytest.approx(DURATION)
    assert context.title == "Two bars"

    # Highlight: at any video time in the passage the fragment's page lights
    # the bar the parent's page lights, renumbered from 1.
    start = ORIGIN + SIGMA
    for index in range(80):
        video = start + 0.013 + index * 0.05
        here = measure_at_working_time(
            context.anchors,
            working_time(video, excerpt_start=context.excerpt_start, sync_offset=context.sync_offset),
        )
        there = measure_at_working_time(
            parent.anchors,
            working_time(video, excerpt_start=parent.excerpt_start, sync_offset=parent.sync_offset),
        )
        assert here is not None and here + 1 == there, video

    # Seek: clicking fragment measure k goes where clicking parent measure k + 1 goes.
    for index, anchor in enumerate(context.anchors):
        assert measure_seek_video(
            anchor, excerpt_start=context.excerpt_start, sync_offset=context.sync_offset
        ) == pytest.approx(
            measure_seek_video(
                parent.anchors[index + 1], excerpt_start=parent.excerpt_start, sync_offset=parent.sync_offset
            )
        )

    # Attack dots: every fragment note is at the video time of the same note in the parent.
    parent_attacks = {
        (note.pitch, round(note.onset + parent.excerpt_start + parent.sync_offset, 4)) for note in parent.note_anchors
    }
    assert context.note_anchors
    for note in context.note_anchors:
        assert (note.pitch, round(note.onset + context.excerpt_start + context.sync_offset, 4)) in parent_attacks
    assert [note.measure for note in context.note_anchors] == [1, 1, 1, 2, 2]


def test_fragment_page_config_and_where_it_is_written(tmp_path: Path):
    _project, folder = saved(tmp_path)
    manifest = F.read_manifest(folder)
    before = tree_digest(folder)
    pages = tmp_path / ".local" / "fragment_pages"
    source = "http://127.0.0.1:9/p/token/?w=1.950"

    page = P.write_fragment_playback(folder, pages, home_url=HOME, source_url=source)

    # The page is machine-local. The fragment folder holds what it held.
    assert page == pages / f"{folder.name}.playback.html"
    assert tree_digest(folder) == before
    assert sorted(path.name for path in pages.iterdir()) == [
        f"{folder.name}.playback.html",
        "opensheetmusicdisplay.min.js",
    ]

    config = page_config(page)
    assert config["excerptStart"] == pytest.approx(ORIGIN)
    assert config["syncOffset"] == SIGMA
    assert config["passage"] == {"start": 0.0, "end": pytest.approx(DURATION)}
    assert config["endTime"] == pytest.approx(DURATION)
    assert [(anchor["measure"], round(anchor["time"], 5)) for anchor in config["anchors"]] == [(1, 0.0), (2, 2.1)]
    # The printed place of each bar is its video time.
    assert [anchor["place"] for anchor in config["anchors"]] == ["(0:32)", "(0:34)"]
    assert config["homeUrl"] == HOME + "fragments" and config["homeLabel"] == "Fragments"
    assert config["toolsUrl"] == f"{HOME}fragment?id={manifest['id']}" and config["toolsLabel"] == "Details & files"
    assert config["sourceUrl"] == source
    # A fragment is not extracted from again.
    assert config["extractUrl"] == ""
    assert {"before": "measures ", "value": "2\u20133", "after": ""} in config["facts"]
    assert {"before": "analyzed key ", "value": P.key_label(manifest), "after": ""} in config["facts"]

    # Harmony is placed by bar and position in the bar. It has no clock of its own.
    harmony = config["harmony"]
    assert harmony["chords"] and harmony["stale"] is False
    assert {chord["measure"] for chord in harmony["chords"]} <= {1, 2}
    assert all(0.0 <= chord["measure_position"] < 1.0 for chord in harmony["chords"])
    # Labels sit on a bar, not on a clock. The grid block may mention
    # downbeat_seconds; the chords and sequences themselves must not.
    for item in harmony["chords"] + harmony["sequences"]:
        assert not any("seconds" in key for key in item)

    text = page.read_text(encoding="utf-8")
    assert "<h1>Two bars</h1>" in text
    assert text.count('target="_blank"') == 1  # only YouTube leaves the tab


def test_sync_adjustment_moves_the_page_and_not_the_passage(tmp_path: Path):
    _project, folder = saved(tmp_path)
    pages = tmp_path / "pages"
    before = page_config(P.write_fragment_playback(folder, pages))

    F.set_fragment_sync(folder, 1.25)
    after = page_config(P.write_fragment_playback(folder, pages))

    assert after["syncOffset"] == 1.25
    for key in ("excerptStart", "passage", "endTime", "anchors", "noteAnchors", "harmony"):
        if key == "anchors":
            # Only the printed video time of each bar follows the offset.
            assert [(a["measure"], a["time"]) for a in after[key]] == [(a["measure"], a["time"]) for a in before[key]]
            assert [a["place"] for a in after[key]] == ["(0:33)", "(0:35)"]
        else:
            assert after[key] == before[key], key

    def loop(config: dict) -> tuple[float, float]:
        origin = config["excerptStart"] + config["syncOffset"]
        return config["passage"]["start"] + origin, config["passage"]["end"] + origin

    moved = 1.25 - SIGMA
    assert loop(after)[0] - loop(before)[0] == pytest.approx(moved)
    assert loop(after)[1] - loop(before)[1] == pytest.approx(moved)
    assert loop(after)[1] - loop(after)[0] == pytest.approx(DURATION)
    assert loop(after) == pytest.approx(F.video_window(F.read_manifest(folder)))


def test_transcription_page_gets_the_selection_controls_and_stays_as_it_was(tmp_path: Path):
    project, _folder = saved(tmp_path)
    context = load_playback(project)
    # The last measure has no next anchor; a loop over it ends at the written end barline.
    assert context.end_time == pytest.approx(6.0) == pytest.approx(score_end_time(parent_quantization(project)))
    assert context.passage is None

    extract = extract_page_url(HOME, project)
    assert extract.startswith(HOME + "extract?project=")
    assert extract_page_url("", project) == ""
    page = write_playback_page(context, tmp_path / "pages" / "playback.html", home_url=HOME, extract_url=extract)
    config = page_config(page)
    assert config["extractUrl"] == extract
    assert config["passage"] is None and config["endTime"] == pytest.approx(6.0)
    # What the page was given before is what it is given now.
    assert config["excerptStart"] == EXCERPT and config["syncOffset"] == SIGMA
    assert config["sourceUrl"] == "" and config["homeLabel"] == "" and config["toolsLabel"] == ""
    assert [round(anchor["time"], 5) for anchor in config["anchors"]] == [0.0, 1.95, 4.05]

    text = page.read_text(encoding="utf-8")
    for needle in ('id="select-toggle"', 'id="loop"', 'id="extract-link"', "Select passage", "openingCue", "keepInsideLoop"):
        assert needle in text
    assert text.count('target="_blank"') == 1

    # Without a library site there is no form to send a selection to.
    bare = page_config(write_playback_page(context, tmp_path / "pages" / "bare.html"))
    assert bare["extractUrl"] == ""


# --------------------------------------------------------------------------
# What the page does with them
# --------------------------------------------------------------------------


def test_fragment_page_loops_its_passage_on_the_video_clock(tmp_path: Path):
    _project, folder = saved(tmp_path)
    page = P.write_fragment_playback(folder, tmp_path / "pages", home_url=HOME)
    out = run_page(
        tmp_path,
        page,
        """
        const out = {};
        out.loopOn = H.el("loop").checked;
        out.label = H.el("passage-label").textContent;
        out.cue = openingCue();
        out.youtube = H.el("youtube-link").href;
        out.homeText = H.el("home-link").textContent;
        out.toolsText = H.el("tools-link").textContent;
        out.extractHidden = H.el("extract-link").hidden;
        const first = loopBounds();
        out.window = [toVideo(first.start), toVideo(first.end)];
        H.draw();
        H.player.state = 1;
        H.at(out.window[0] + 1.0);
        out.measureEarly = H.active();
        H.at(out.window[0] + 2.5);
        out.measureLate = H.active();
        out.seeksInside = H.player.seeks.length;
        H.at(out.window[1] + 0.05);
        out.seeksPastEnd = H.player.seeks.slice();

        // Notation earlier: the score is shifted 0.1 s later in the video.
        H.click("earlier");
        const second = loopBounds();
        out.nudged = [toVideo(second.start), toVideo(second.end)];
        out.onFragmentClock = [second.start, second.end];
        out.posts = H.posts;
        out.offsetLabel = H.el("offset-label").textContent;

        // Paused, the player can be anywhere. With Loop off, so can a playing one.
        const count = H.player.seeks.length;
        H.player.state = 2;
        H.at(5);
        out.pausedOutside = H.player.seeks.length - count;
        H.change("loop", false);
        H.player.state = 1;
        H.at(5);
        out.loopOffOutside = H.player.seeks.length - count;

        // A bar inside the fragment can be looped on its own.
        H.click("select-toggle");
        H.clickMeasure(2);
        const inner = loopBounds();
        out.innerBar = [inner.start, inner.end];
        return out;
        """,
    )
    start, end = ORIGIN + SIGMA, ORIGIN + SIGMA + DURATION
    assert out["loopOn"] is True
    assert out["label"] == "Whole fragment \u00b7 4.0 s" or out["label"] == "Whole fragment \u00b7 4.1 s"
    assert out["cue"] == pytest.approx(start)
    assert out["youtube"] == "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=32s"
    assert out["homeText"] == "\u2190 Fragments" and out["toolsText"] == "Details & files"
    assert out["extractHidden"] is True
    assert out["window"] == pytest.approx([start, end])
    assert (out["measureEarly"], out["measureLate"]) == (1, 2)
    assert out["seeksInside"] == 0
    assert out["seeksPastEnd"] == pytest.approx([start])
    # Both ends moved together; the passage itself is what it was.
    assert out["nudged"] == pytest.approx([start + 0.1, end + 0.1])
    assert out["onFragmentClock"] == pytest.approx([0.0, DURATION])
    assert out["posts"] == [{"address": "sync-offset", "body": {"offset": pytest.approx(SIGMA + 0.1)}}]
    assert out["offsetLabel"] == "0.50 s"
    assert out["pausedOutside"] == 0 and out["loopOffOutside"] == 0
    # The fragment's last bar ends where the fragment ends.
    assert out["innerBar"] == pytest.approx([2.1, DURATION])


def test_transcription_page_selects_loops_and_links_a_passage(tmp_path: Path):
    project, _folder = saved(tmp_path)
    extract = HOME + "extract?project=parent"
    page = write_playback_page(
        load_playback(project), tmp_path / "pages" / "playback.html", home_url=HOME, extract_url=extract
    )
    shift = EXCERPT + SIGMA
    out = run_page(
        tmp_path,
        page,
        """
        const out = {};
        out.cue = openingCue();
        out.loopOn = H.el("loop").checked;
        out.label = H.el("passage-label").textContent;
        out.extractHidden = H.el("extract-link").hidden;
        out.noBounds = loopBounds();
        H.draw();

        // As before: a click on a measure seeks the video to it.
        H.clickMeasure(2);
        out.seekOnClick = H.player.seeks.slice();

        // Selecting: clicks mark measures and leave the video alone.
        H.click("select-toggle");
        H.clickMeasure(3);
        out.afterFirst = { selected: H.selected(), label: H.el("passage-label").textContent };
        H.clickMeasure(2);
        out.afterSecond = {
          selected: H.selected(),
          label: H.el("passage-label").textContent,
          href: H.el("extract-link").href,
          extractHidden: H.el("extract-link").hidden,
          seeks: H.player.seeks.length
        };
        const chosen = loopBounds();
        out.bounds = [chosen.start, chosen.end];

        H.click("first-earlier");
        out.widened = H.selected();
        H.click("first-earlier");
        out.clampedLow = H.selected();
        H.click("last-later");
        out.clampedHigh = H.selected();
        H.click("last-earlier");
        out.narrowed = H.selected();
        out.href = H.el("extract-link").href;
        const inner = loopBounds();
        out.inner = [inner.start, inner.end];

        // Back to jumping. Playing past the passage with Loop off is left alone.
        H.click("select-toggle");
        H.player.state = 1;
        const count = H.player.seeks.length;
        H.at(inner.end + __SHIFT__ + 0.2);
        out.loopOff = H.player.seeks.length - count;
        H.change("loop", true);
        H.at(inner.end + __SHIFT__ + 0.2);
        out.loopSeek = H.player.seeks[H.player.seeks.length - 1];
        out.selectionKept = H.selected();

        H.click("clear-passage");
        out.cleared = { selected: H.selected(), bounds: loopBounds(), extractHidden: H.el("extract-link").hidden };
        return out;
        """.replace("__SHIFT__", repr(shift)),
    )
    # A transcription's page opens where it always did, with nothing selected and no loop.
    assert out["cue"] == EXCERPT
    assert out["loopOn"] is False and out["extractHidden"] is True and out["noBounds"] is None
    assert out["label"] == "No passage selected"
    assert out["seekOnClick"] == pytest.approx([1.95 + shift])

    assert out["afterFirst"]["selected"] == [3]
    assert "click the last measure" in out["afterFirst"]["label"]
    assert out["afterSecond"]["selected"] == [2, 3]
    assert out["afterSecond"]["label"] == "Measures 2\u20133 \u00b7 (0:32)\u2013(0:36) \u00b7 4.0 s" or out["afterSecond"][
        "label"
    ] == "Measures 2\u20133 \u00b7 (0:32)\u2013(0:36) \u00b7 4.1 s"
    assert out["afterSecond"]["href"] == extract + "&measures=2-3"
    assert out["afterSecond"]["extractHidden"] is False
    assert out["afterSecond"]["seeks"] == 1  # only the click made before selecting
    # The same window an extraction of measures 2-3 gets.
    assert out["bounds"] == pytest.approx([T0, T0 + DURATION])

    assert out["widened"] == [1, 2, 3] and out["clampedLow"] == [1, 2, 3] and out["clampedHigh"] == [1, 2, 3]
    assert out["narrowed"] == [1, 2] and out["href"] == extract + "&measures=1-2"
    assert out["inner"] == pytest.approx([0.0, 4.05])  # up to the anchor of measure 3

    assert out["loopOff"] == 0
    assert out["loopSeek"] == pytest.approx(0.0 + shift)
    assert out["selectionKept"] == [1, 2]
    assert out["cleared"] == {"selected": [], "bounds": None, "extractHidden": True}


@pytest.mark.parametrize(
    ("search", "selected", "cue"),
    [
        ("?sel=2-3&w=1.95", [2, 3], 1.95 + EXCERPT + SIGMA),
        ("?sel=2", [2], EXCERPT),
        ("?sel=3-2", [2, 3], EXCERPT),
        ("?sel=2-99", [2, 3], EXCERPT),
        ("?sel=9-12", [], EXCERPT),
        ("?sel=abc&w=nope", [], EXCERPT),
        ("?embed=1", [], EXCERPT),
    ],
)
def test_page_address_can_carry_a_selection_and_a_place(tmp_path: Path, search, selected, cue):
    project, _folder = saved(tmp_path)
    page = write_playback_page(load_playback(project), tmp_path / "pages" / "playback.html", home_url=HOME)
    out = run_page(
        tmp_path,
        page,
        """
        H.draw();
        return { selected: H.selected(), cue: openingCue(), seeks: H.player.seeks.length };
        """,
        search,
    )
    assert out["selected"] == selected
    assert out["cue"] == pytest.approx(cue)
    assert out["seeks"] == 0


# --------------------------------------------------------------------------
# Served
# --------------------------------------------------------------------------


def test_server_serves_a_fragment_and_stores_its_sync_offset(tmp_path: Path, server: PlaybackServer):
    project, folder = saved(tmp_path)
    parent_before = tree_digest(project)
    before = F.read_manifest(folder)
    files_before = tree_digest(folder)
    token = server.register_fragment(folder)
    assert server.register_fragment(folder) == token
    address = server.fragment_url(token)
    assert f"/f/{token}/" in address

    with urlopen(address) as response:
        body = response.read().decode("utf-8")
    assert "<h1>Two bars</h1>" in body
    config = embedded_json(body, "const config = ")
    assert config["syncOffset"] == SIGMA and config["passage"]["end"] == pytest.approx(DURATION)
    # The parent is on this computer, so the page links to it at the passage.
    parent_token = server.register(project)
    assert config["sourceUrl"] == f"{server.page_url(parent_token)}?w={T0:.3f}"
    assert config["toolsUrl"] == f"{HOME}fragment?id={before['id']}"

    with urlopen(address + "sync-offset") as response:
        assert json.loads(response.read())["offset"] == SIGMA
    request = Request(address + "sync-offset", data=json.dumps({"offset": -0.75}).encode("utf-8"), method="POST")
    request.add_header("Content-Type", "application/json")
    with urlopen(request) as response:
        assert json.loads(response.read())["offset"] == -0.75
    with urlopen(address + "sync-offset") as response:
        assert json.loads(response.read())["offset"] == -0.75

    # Only the manifest changed, and in it only the offset and the modification time.
    after = F.read_manifest(folder)
    files_after = tree_digest(folder)
    assert sorted(files_after) == sorted(files_before)  # no page was written into the fragment
    assert {name for name in files_after if files_after[name] != files_before[name]} == {FRAGMENT_JSON}
    assert {key for key in after if after[key] != before[key]} <= {"sync", "modified_at"}
    assert after["sync"] == {"offset_seconds": -0.75, "parent_offset_at_extraction": SIGMA}
    assert after["time"] == before["time"]
    assert tree_digest(project) == parent_before

    # The page is written again on each request, so it shows the new offset.
    with urlopen(address + "playback.html") as response:
        body = response.read().decode("utf-8")
    assert embedded_json(body, "const config = ")["syncOffset"] == -0.75

    with urlopen(address + "opensheetmusicdisplay.min.js") as response:
        assert len(response.read()) > 1000
    assert sorted(path.name for path in server.fragment_pages.iterdir()) == [
        f"{folder.name}.playback.html",
        "opensheetmusicdisplay.min.js",
    ]


@pytest.mark.parametrize("body", ['{"offset": NaN}', '{"offset": "soon"}', "{}", "not json", '{"offset": Infinity}'])
def test_server_refuses_an_offset_that_is_not_a_number(tmp_path: Path, server: PlaybackServer, body: str):
    _project, folder = saved(tmp_path)
    before = (folder / FRAGMENT_JSON).read_bytes()
    address = server.fragment_url(server.register_fragment(folder))
    request = Request(address + "sync-offset", data=body.encode("utf-8"), method="POST")
    with pytest.raises(HTTPError) as caught:
        urlopen(request)
    assert caught.value.code == 400
    assert (folder / FRAGMENT_JSON).read_bytes() == before
    F.read_manifest(folder)


def test_server_knows_only_the_fragment_addresses_it_gave_out(tmp_path: Path, server: PlaybackServer):
    project, folder = saved(tmp_path)
    token = server.register_fragment(folder)
    project_token = server.register(project)
    assert get(server, "/f/unknown/")[0] == 404
    assert get(server, f"/f/{token}/fragment.json")[0] == 404
    assert get(server, f"/f/{token}/../{folder.name}/fragment.json")[0] == 404
    # A fragment's token is not a project's, and the other way round.
    assert get(server, f"/p/{token}/")[0] == 404
    assert get(server, f"/f/{project_token}/")[0] == 404
    request = Request(server.fragment_url(token) + "playback.html", data=b'{"offset": 1}', method="POST")
    with pytest.raises(HTTPError) as caught:
        urlopen(request)
    assert caught.value.code == 404
    # The project's own page is served as it was.
    with urlopen(server.page_url(project_token)) as response:
        assert b"<h1>Parent take</h1>" in response.read()


def test_fragment_of_a_local_recording_has_a_score_page_and_no_player(tmp_path: Path, server: PlaybackServer):
    _project, folder = saved(tmp_path, youtube=None)
    manifest = F.read_manifest(folder)
    assert manifest["source"]["kind"] == "file" and P.fragment_video_id(manifest) is None
    assert not P.fragment_has_video(folder) and P.fragment_has_score(folder)
    assert P.youtube_place_url(manifest) == ""
    with pytest.raises(SyncError, match="local recording"):
        P.fragment_context(folder)
    before = tree_digest(folder)

    token = server.register_fragment(folder)
    assert get(server, f"/f/{token}/") == (302, f"/f/{token}/notation.html")
    assert get(server, f"/f/{token}/?embed=1") == (302, f"/f/{token}/notation.html?embed=1")
    with urlopen(server.fragment_url(token)) as response:
        body = response.read().decode("utf-8")
    assert "<h1>Two bars</h1>" in body
    assert "Play with score" not in body
    assert f'href="{HOME}fragments">&larr; Fragments</a>' in body
    assert f'href="{HOME}fragment?id={manifest["id"]}">Details &amp; files</a>' in body
    assert "in the recording" in body and "in the video" not in body
    # The harmony travels with the fragment and is drawn by bar.
    harmony = embedded_json(body, "const harmony = ")
    assert harmony["chords"] and {chord["measure"] for chord in harmony["chords"]} <= {1, 2}
    # Measure 1 is labelled with its place in the recording: E + t0 + offset.
    times = embedded_json(body, "const measureTimes = ")
    assert times["1"] == "(0:32)"
    # Nothing was written into the fragment to show it.
    assert tree_digest(folder) == before


def test_fragment_with_a_video_has_both_pages(tmp_path: Path, server: PlaybackServer):
    _project, folder = saved(tmp_path)
    token = server.register_fragment(folder)
    with urlopen(server.fragment_url(token) + "notation.html") as response:
        body = response.read().decode("utf-8")
    assert f'href="{server.fragment_url(token)}">Play with score</a>' in body
    assert "in the video" in body


def test_fragment_that_cannot_be_drawn_sends_the_reader_to_its_details(tmp_path: Path, server: PlaybackServer):
    _project, folder = saved(tmp_path)
    ident = F.read_manifest(folder)["id"]
    token = server.register_fragment(folder)
    (folder / "score.musicxml").unlink()
    for page in ("", "notation.html"):
        assert get(server, f"/f/{token}/{page}") == (302, f"{HOME}fragment?id={ident}")
    server.home_url = ""
    assert get(server, f"/f/{token}/")[0] == 404


# --------------------------------------------------------------------------
# The way back to the source
# --------------------------------------------------------------------------


def test_parent_place_finds_the_passage_in_the_parent_as_it_is_now(tmp_path: Path):
    project, folder = saved(tmp_path)
    assert P.parent_place(folder, transcriptions=tmp_path) == (project.resolve(), pytest.approx(T0))
    manifest = F.read_manifest(folder)
    assert P.youtube_place_url(manifest) == "https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=32s"

    # Not on this computer.
    assert P.parent_place(folder, transcriptions=tmp_path / "elsewhere") is None

    # Transcribed again from an earlier start: the same video time is later in the working audio.
    path = project / MANIFEST
    parent = json.loads(path.read_text(encoding="utf-8"))
    parent["audio"]["excerpt_start_seconds"] = 20.0
    path.write_text(json.dumps(parent), encoding="utf-8")
    assert P.parent_place(folder, transcriptions=tmp_path) == (project.resolve(), pytest.approx(T0 + 10.0))

    # Transcribed again from a start after the passage: the passage is not in it.
    parent["audio"]["excerpt_start_seconds"] = 60.0
    path.write_text(json.dumps(parent), encoding="utf-8")
    assert P.parent_place(folder, transcriptions=tmp_path) is None

    # A project of the same name that is a different video.
    parent["audio"]["excerpt_start_seconds"] = EXCERPT
    parent["origin"] = parent["youtube_url"] = "https://www.youtube.com/watch?v=aaaaaaaaaaa"
    path.write_text(json.dumps(parent), encoding="utf-8")
    assert P.parent_place(folder, transcriptions=tmp_path) is None


def test_notation_page_wording_is_unchanged_unless_asked(tmp_path: Path):
    from baroque_transcribe.notation import write_notation_page

    project, _folder = saved(tmp_path)
    plain = write_notation_page(
        project / "score.musicxml", tmp_path / "out" / "a.html", home_url=HOME, tools_url=HOME + "project?project=x"
    ).read_text(encoding="utf-8")
    assert f'<a class="home" href="{HOME}">&larr; Library</a>' in plain
    assert ">Edit &amp; files</a>" in plain
    worded = write_notation_page(
        project / "score.musicxml",
        tmp_path / "out" / "b.html",
        home_url=HOME,
        tools_url=HOME + "x",
        home_label="Fragments",
        tools_label="Details & files",
    ).read_text(encoding="utf-8")
    assert ">&larr; Fragments</a>" in worded and ">Details &amp; files</a>" in worded
    assert "&larr; Library" not in worded


def test_library_addresses_point_at_the_streamlit_pages(tmp_path: Path):
    from baroque_transcribe.fragment_ui import extract_href, fragment_href, fragments_href

    _project, folder = saved(tmp_path)
    ident = F.read_manifest(folder)["id"]
    assert fragments_href() == "/fragments"
    assert fragment_href(ident) == f"/fragment?id={ident}"
    href = extract_href(_project, "2-3")
    assert href.startswith("/extract?project=") and href.endswith("&measures=2-3")
