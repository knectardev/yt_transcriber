"""Local pages for transcribing a recording and reviewing the result.

Run with ``baroque-transcribe ui``. Transcription still happens on this machine.

Two pages:

* ``/`` is the home page. It has the controls that start a transcription and
  a grid of the videos and recordings that already have one.
* ``/project?project=<folder>`` is the page for one transcription: the video
  with the score following it, the audio and plots, the note table, and the
  files. The note table writes ``notes.json`` and rebuilds the MIDI and
  MusicXML from it.
"""

from __future__ import annotations

import base64
import json
import os
import webbrowser
from datetime import datetime
from html import escape
from pathlib import Path
from urllib.parse import quote

from baroque_transcribe.defaults import (
    DIAGNOSTICS_DIR,
    MANIFEST,
    NOTES_JSON,
    PEDAL_JSON,
    QUANTIZED_MID,
    RAW_MID,
    SCORE_XML,
    WORKING_WAV,
)
from baroque_transcribe.library import Card, Run, card_for, find_project, library_cards, load_thumbnail, project_param, read_run
# ``notation`` (music21), ``pipeline`` (model/diagnostics stack), and ``pandas``
# are imported inside the handlers that use them. At the top they added
# seconds to every first page load, including the home-page thumbnail grid.
from baroque_transcribe.playback import (
    SyncError,
    notation_page_url,
    open_synced_playback,
    shared_playback_server,
    write_youtube_url,
    youtube_project_for_midi,
    youtube_video_id,
)
from baroque_transcribe.progress import ProgressEvent
from baroque_transcribe.schema import NoteEvent, PedalEvent, Transcription, read_transcription

_STAGE_ORDER = ("prepare", "checkpoint", "load", "transcribe", "write")
_STAGE_TITLES = {
    "prepare": "Prepare audio",
    "checkpoint": "Model file",
    "load": "Load model",
    "transcribe": "Transcribe",
    "write": "Write score and diagnostics",
}
_SOURCE_KINDS = ("YouTube URL", "Local file", "Upload", "MIDI file")
_DEFAULT_OUTPUT = "transcriptions/performance"
_PAGES: dict[str, object] = {}


_TRANSCRIBE_FRAGMENT = None

_STYLE = """
<style>
  :root {
    --bt-ink: #1c1917; --bt-muted: #6b6560; --bt-line: #e7e2da; --bt-panel: #fffdf9;
    --bt-accent: #b45309; --bt-accent-ink: #7c3a07; --bt-soft: #fdf3e2;
    --bt-serif: Georgia, "Iowan Old Style", "Times New Roman", serif;
  }
  header[data-testid="stHeader"] { background: transparent; }
  .block-container { padding-top: 1.1rem; padding-bottom: 3rem; max-width: 1240px; }
  h1, h2, h3 { font-family: var(--bt-serif); letter-spacing: -0.01em; }
  .bt-brand {
    display: flex; align-items: baseline; gap: 1rem; flex-wrap: wrap;
    padding: 0.2rem 0 0.9rem; margin-bottom: 1.2rem; border-bottom: 1px solid var(--bt-line);
  }
  .bt-brand .bt-wordmark {
    font-family: var(--bt-serif); font-size: 1.45rem; font-weight: 600; color: var(--bt-ink); text-decoration: none;
  }
  .bt-brand .bt-tag { color: var(--bt-muted); font-size: 0.92rem; }
  .bt-brand .bt-crumb { color: var(--bt-muted); font-size: 0.92rem; }
  .bt-brand .bt-crumb a { color: var(--bt-accent-ink); text-decoration: none; font-weight: 500; }
  .bt-brand .bt-crumb a:hover { text-decoration: underline; }
  .bt-section { display: flex; align-items: baseline; justify-content: space-between; gap: 1rem; margin: 1.6rem 0 0.7rem; }
  .bt-section h2 { margin: 0; font-size: 1.35rem; font-weight: 600; }
  .bt-section .bt-count { color: var(--bt-muted); font-size: 0.9rem; }
  .bt-lede { color: var(--bt-muted); font-size: 0.95rem; margin: 0 0 0.6rem; max-width: 60rem; line-height: 1.5; }
  .bt-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 1.25rem 1.1rem; margin-top: 0.4rem;
  }
  .bt-card {
    display: flex; flex-direction: column; background: var(--bt-panel); border: 1px solid var(--bt-line);
    border-radius: 14px; overflow: hidden; transition: transform 0.15s ease, box-shadow 0.15s ease;
  }
  .bt-card:hover { transform: translateY(-2px); box-shadow: 0 12px 28px -18px rgba(28, 25, 23, 0.45); }
  .bt-thumb { position: relative; display: block; aspect-ratio: 16 / 9; background: #1c1917; overflow: hidden; }
  .bt-thumb img { width: 100%; height: 100%; object-fit: cover; display: block; }
  .bt-thumb.bt-placeholder {
    display: flex; align-items: center; justify-content: center; color: #fbbf24; font-size: 2.6rem;
    background: linear-gradient(135deg, #292524, #44403c);
  }
  .bt-thumb .bt-badge {
    position: absolute; right: 0.55rem; bottom: 0.55rem; padding: 0.15rem 0.5rem; border-radius: 999px;
    background: rgba(28, 25, 23, 0.82); color: white; font-size: 0.74rem; font-variant-numeric: tabular-nums;
  }
  .bt-thumb .bt-play {
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: center; opacity: 0; transition: opacity 0.15s;
  }
  .bt-thumb .bt-play span {
    width: 3rem; height: 3rem; border-radius: 999px; background: rgba(180, 83, 9, 0.92); color: white;
    display: flex; align-items: center; justify-content: center; font-size: 1.2rem; padding-left: 0.2rem;
  }
  .bt-card:hover .bt-play { opacity: 1; }
  .bt-body { display: flex; flex-direction: column; gap: 0.4rem; padding: 0.75rem 0.9rem 0.9rem; }
  .bt-title {
    font-family: var(--bt-serif); font-size: 1.02rem; font-weight: 600; line-height: 1.3; color: var(--bt-ink);
    text-decoration: none; display: -webkit-box; -webkit-line-clamp: 2; -webkit-box-orient: vertical; overflow: hidden;
  }
  .bt-title:hover { color: var(--bt-accent-ink); }
  .bt-meta { color: var(--bt-muted); font-size: 0.82rem; }
  .bt-runs { display: flex; flex-wrap: wrap; gap: 0.35rem; margin-top: 0.15rem; }
  .bt-chip {
    font-size: 0.74rem; padding: 0.14rem 0.55rem; border-radius: 999px; border: 1px solid var(--bt-line);
    color: var(--bt-accent-ink); background: var(--bt-soft); text-decoration: none; max-width: 100%;
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
  }
  .bt-chip:hover { border-color: var(--bt-accent); }
  .bt-chip.bt-off { color: var(--bt-muted); background: transparent; border-style: dashed; }
  .bt-empty {
    padding: 2rem; border: 1px dashed var(--bt-line); border-radius: 14px; color: var(--bt-muted); text-align: center;
  }
  .bt-head { display: grid; grid-template-columns: auto 1fr; gap: 1.1rem; align-items: start; margin-bottom: 0.6rem; }
  .bt-head .bt-poster { width: 200px; aspect-ratio: 16 / 9; border-radius: 10px; object-fit: cover; background: #1c1917; }
  .bt-head .bt-poster.bt-placeholder {
    display: flex; align-items: center; justify-content: center; color: #fbbf24; font-size: 2rem;
    background: linear-gradient(135deg, #292524, #44403c);
  }
  .bt-head h1 { margin: 0 0 0.3rem; font-size: 1.7rem; font-weight: 600; line-height: 1.2; }
  .bt-facts { display: flex; flex-wrap: wrap; gap: 0.4rem 0.5rem; margin: 0.3rem 0 0.4rem; }
  .bt-fact {
    font-size: 0.8rem; padding: 0.18rem 0.6rem; border-radius: 999px; background: var(--bt-soft);
    border: 1px solid var(--bt-line); color: var(--bt-ink); font-variant-numeric: tabular-nums;
  }
  .bt-fact b { color: var(--bt-accent-ink); font-weight: 600; }
  .bt-path { color: var(--bt-muted); font-size: 0.82rem; font-family: ui-monospace, Consolas, monospace; word-break: break-all; }
  @media (max-width: 720px) { .bt-head { grid-template-columns: 1fr; } .bt-head .bt-poster { width: 100%; } }
</style>
"""


def run() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Piano transcription",
        page_icon="🎹",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.html(_STYLE)
    shared_playback_server(_home_url())
    home = st.Page(_home_page, title="Library", url_path="", default=True)
    project = st.Page(_project_page, title="Project", url_path="project")
    _PAGES["home"] = home
    _PAGES["project"] = project
    navigation = st.navigation([home, project], position="hidden")
    _open_launch_project()
    navigation.run()


def _home_url() -> str:
    import streamlit as st

    host = ""
    try:
        host = str(st.context.headers.get("Host") or "")
    except Exception:
        host = ""
    if not host:
        host = f"127.0.0.1:{st.get_option('server.port')}"
    return f"http://{host}/"


def _open_launch_project() -> None:
    """``baroque-transcribe ui --project X`` opens X once, on the first run of the session."""

    import streamlit as st

    if st.session_state.get("launched"):
        return
    st.session_state["launched"] = True
    requested = os.environ.get("BAROQUE_UI_PROJECT")
    if requested and find_project(requested) is not None and not st.query_params.get("project"):
        st.session_state["open_project"] = requested
        st.switch_page(_PAGES["project"])


# --------------------------------------------------------------------------- home


def _home_page() -> None:
    import streamlit as st

    _brand()
    message = st.session_state.pop("transcription_message", None)
    if message:
        st.success(message)
    _flash()

    st.html(
        '<div class="bt-section"><h2>Transcribe a recording</h2></div>'
        '<p class="bt-lede">Solo piano from YouTube or a file on this computer. The raw MIDI keeps the timing '
        "of the performance. The MusicXML score is a separate rhythmic reading for MuseScore.</p>"
    )
    with st.container(border=True):
        _transcribe_fragment()()

    # Card metadata only: titles, counts, and the cached thumbnail JPEG
    # (fetched from YouTube once per video, then read from disk).
    # The notes, audio, and score are opened on the project page.
    cards = _with_local_thumbnails(library_cards(for_grid=True))
    count = sum(1 for card in cards for run in card.runs if run.available)
    st.html(
        '<div class="bt-section"><h2>Library</h2>'
        f'<span class="bt-count">{count} transcription{"" if count == 1 else "s"} on this computer</span></div>'
    )
    if not cards:
        st.html('<div class="bt-empty">Nothing has been transcribed yet. The first result will appear here.</div>')
    else:
        # Same document as the Library heading. An iframe only receives a
        # height after its own measurement message, so the grid stayed blank.
        _show_cards(cards)


def _transcribe_fragment():
    """The form, isolated so typing in it does not redraw the library grid."""

    global _TRANSCRIBE_FRAGMENT
    import streamlit as st

    if _TRANSCRIBE_FRAGMENT is None:
        _TRANSCRIBE_FRAGMENT = st.fragment(_transcribe_panel)
    return _TRANSCRIBE_FRAGMENT


def _with_local_thumbnails(cards: list[Card]) -> list[Card]:
    """Attach each card's JPEG as a data URI, and nothing else.

    ``load_thumbnail`` reads the JPEG from the ``data/thumbnails`` cache and
    fetches it from YouTube only the first time a video is seen, so a new
    transcription's card gets its thumbnail without anyone having to place
    a file anywhere by hand. The JPEGs committed under ``static/thumbnails``
    are kept as a fallback for when the cache is empty and the fetch fails.
    A card whose thumbnail cannot be fetched simply renders without one.
    """

    folder = Path(__file__).resolve().parent / "static" / "thumbnails"
    for card in cards:
        card.thumbnail = None
        if not card.video_id:
            continue
        data = load_thumbnail(card.video_id)
        if not data:
            try:
                data = (folder / f"{card.video_id}.jpg").read_bytes()
            except OSError:
                data = b""
        if data:
            card.thumbnail = "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
    return cards


def _show_cards(cards: list[Card]) -> None:
    """Draw the library from JPEGs already read into memory."""

    import streamlit as st

    columns = st.columns(min(3, len(cards)), gap="medium")
    for index, card in enumerate(cards):
        with columns[index % len(columns)]:
            _show_card(card)


def _show_card(card: Card) -> bool:
    import streamlit as st

    primary = card.primary
    href = _project_href(primary.folder) if primary else None
    with st.container(border=True):
        if card.thumbnail:
            st.image(card.thumbnail, width="stretch", link=href)
        title = card.title.replace("[", "\\[").replace("]", "\\]")
        st.markdown(f"**[{title}]({href})**" if href else f"**{title}**")
        bits: list[str] = []
        if primary and primary.duration:
            bits.append(_clock(primary.duration))
        if primary and primary.notes is not None:
            bits.append(f"{primary.notes:,} notes")
        if primary and primary.modified:
            bits.append(_relative(primary.modified))
        elif card.runs and primary is None:
            bits.append("not on this computer")
        if bits:
            st.caption(" · ".join(bits))
        if len(card.runs) > 1 or primary is None:
            chips: list[str] = []
            for run in card.runs:
                label = run.name.replace("[", "\\[").replace("]", "\\]")
                if run.available:
                    chips.append(f"[{label}]({_project_href(run.folder)})")
                else:
                    chips.append(label)
            if chips:
                st.markdown(" · ".join(chips))
        for run in card.runs:
            if run.available:
                continue
            # The score files for this run are in the parse history (they
            # arrived with the repo); only the project folder is missing.
            if st.button(
                f"Restore {run.name} from history",
                key=f"restore-{card.key}-{run.name}",
                help="Copy the stored notes, MIDI, and score into a local project folder. Nothing is downloaded and no model is run.",
            ):
                from baroque_transcribe.parse_store import restore_projects

                restored, _skipped = restore_projects(projects=[run.name])
                if restored:
                    st.session_state["flash"] = ("success", f"Restored {run.name} from the parse history. No model was run.")
                else:
                    st.session_state["flash"] = ("info", f"{run.name} has no stored score files to restore.")
                st.rerun()
    return bool(card.thumbnail)


def _cards_document(cards: list[Card]) -> str:
    """One HTML document whose images are the bytes read above."""

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><script>
fetch('http://127.0.0.1:7249/ingest/38e55499-7058-440e-99f5-c0e7258b5b2a',{{method:'POST',headers:{{'Content-Type':'application/json','X-Debug-Session-Id':'cb4792'}},body:JSON.stringify({{sessionId:'cb4792',hypothesisId:'E',location:'cards-iframe',message:'iframe document started',data:{{imgs:document.images.length}},timestamp:Date.now()}})}}).catch(()=>{{}});
addEventListener('load',function(){{fetch('http://127.0.0.1:7249/ingest/38e55499-7058-440e-99f5-c0e7258b5b2a',{{method:'POST',headers:{{'Content-Type':'application/json','X-Debug-Session-Id':'cb4792'}},body:JSON.stringify({{sessionId:'cb4792',hypothesisId:'E',location:'cards-iframe',message:'iframe document loaded',data:{{imgs:[...document.images].map(function(img){{return img.naturalWidth}})}},timestamp:Date.now()}})}}).catch(()=>{{}});}});
</script><style>
  html, body {{ margin: 0; background: #faf7f2; }}
  body {{ font-family: Georgia, "Iowan Old Style", "Times New Roman", serif; color: #1c1917; }}
  .bt-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(250px, 1fr)); gap: 1.25rem 1.1rem; }}
  .bt-card {{ display: flex; flex-direction: column; background: #fffdf9; border: 1px solid #e7e2da; border-radius: 14px; overflow: hidden; }}
  .bt-thumb {{ position: relative; display: block; aspect-ratio: 16 / 9; background: #1c1917; overflow: hidden; }}
  .bt-thumb img {{ width: 100%; height: 100%; object-fit: cover; display: block; }}
  .bt-thumb.bt-placeholder {{ display: flex; align-items: center; justify-content: center; color: #fbbf24; font-size: 2.6rem; background: linear-gradient(135deg, #292524, #44403c); }}
  .bt-badge {{ position: absolute; right: 0.55rem; bottom: 0.55rem; padding: 0.15rem 0.5rem; border-radius: 999px; background: rgba(28, 25, 23, 0.82); color: white; font-size: 0.74rem; }}
  .bt-body {{ display: flex; flex-direction: column; gap: 0.4rem; padding: 0.75rem 0.9rem 0.9rem; }}
  .bt-title {{ font-size: 1.02rem; font-weight: 600; line-height: 1.3; color: #1c1917; text-decoration: none; }}
  .bt-meta {{ color: #6b6560; font-size: 0.82rem; font-family: "Segoe UI", sans-serif; }}
  .bt-runs {{ display: flex; flex-wrap: wrap; gap: 0.35rem; }}
  .bt-chip {{ font-family: "Segoe UI", sans-serif; font-size: 0.74rem; padding: 0.14rem 0.55rem; border-radius: 999px; border: 1px solid #e7e2da; color: #7c3a07; background: #fdf3e2; text-decoration: none; }}
  .bt-chip.bt-off {{ color: #6b6560; background: transparent; border-style: dashed; }}
</style></head><body>{_cards_html(cards)}</body></html>"""


def _transcribe_panel() -> None:
    import streamlit as st

    if "source_kind" not in st.session_state:
        st.session_state["source_kind"] = _SOURCE_KINDS[0]
    kind = st.segmented_control("Source", list(_SOURCE_KINDS), key="source_kind")
    kind = kind or _SOURCE_KINDS[0]
    if kind == "MIDI file":
        _midi_panel()
        return

    youtube = local_path = ""
    upload = None
    main, side = st.columns([3, 2], gap="medium")
    with main:
        if kind == "YouTube URL":
            youtube = st.text_input("YouTube URL", placeholder="https://youtu.be/…", key="youtube_url")
        elif kind == "Local file":
            local_path = st.text_input("WAV, MP3, or FLAC path", key="local_path")
        else:
            upload = st.file_uploader("WAV, MP3, or FLAC", type=["wav", "mp3", "flac"], key="upload")
    with side:
        output = st.text_input(
            "Output folder",
            value=_default_output(),
            key="output_dir",
            help=(
                "A folder that already has a transcription is left alone. "
                "The next run is saved beside it, with the date and time in the folder name."
            ),
        )

    with st.expander("Excerpt and notation settings", expanded=False):
        a, b, c, d = st.columns(4)
        start = a.text_input("Start", value="0", key="start", help="Seconds, M:SS, or H:MM:SS.")
        end = b.text_input("End", value="", key="end", help="Leave empty to keep going until the file ends.")
        model = c.selectbox("Model", ["bytedance", "basic-pitch"], key="model")
        device = d.selectbox("Device", ["auto", "cpu", "cuda"], key="device")
        e, f, g, h, i = st.columns(5)
        tempo = e.text_input("Tempo", value="auto", key="tempo", help="BPM, or auto.")
        meter = f.text_input("Time signature", value="", key="meter", help="Leave empty to assume 4/4.")
        subdivision = g.selectbox("Subdivision", [16, 8, 32], index=0, key="subdivision")
        downbeat = h.text_input("Downbeat", value="0", key="downbeat", help="Seconds into the excerpt of the first strong beat.")
        staff_split = i.number_input("Treble from MIDI pitch", min_value=21, max_value=108, value=60, key="staff_split")

    button_col, hint_col = st.columns([1, 4], vertical_alignment="center")
    run_clicked = button_col.button("Transcribe", type="primary", width="stretch")
    hint_col.caption("The first run downloads the piano model (about 165 MB). A long video takes several minutes on CPU.")

    if not run_clicked:
        return
    project = Path(output)
    source = _resolve_source(kind, youtube, local_path, upload, project)
    if source is None:
        st.error("Choose a file or a URL first.")
        return
    if kind == "YouTube URL" and youtube_video_id(source) is None:
        st.error("That is not a YouTube URL.")
        return
    try:
        written = _run_with_progress(
            source,
            project,
            start=start or None,
            end=end or None,
            model=model,
            device=device,
            tempo=tempo or "auto",
            meter=meter or None,
            subdivision=int(subdivision),
            downbeat=downbeat or "0",
            staff_split=int(staff_split),
        )
    except Exception as exc:
        st.exception(exc)
        return
    message = f"Wrote {written}"
    if written != project:
        message += f". The transcription already in {project} was left unchanged."
    st.session_state["transcription_message"] = message
    st.session_state["open_project"] = str(written)
    st.switch_page(_PAGES["project"])


def _midi_panel() -> None:
    import streamlit as st

    st.caption("Open notation from a MIDI file. The piano model is not run.")
    main, side = st.columns([3, 2], gap="medium")
    with main:
        midi_path = st.text_input(
            "MIDI path",
            placeholder=r"transcriptions\piece\quantized.mid",
            key="midi_path",
            help="A quantized.mid from a finished project opens that project's page.",
        )
        midi_upload = st.file_uploader("Or choose a .mid file", type=["mid", "midi"], key="midi_upload")
    with side:
        output = st.text_input("Output folder", value=_default_output(), key="output_dir")
        midi_youtube = st.text_input(
            "YouTube URL to attach",
            key="midi_youtube_url",
            placeholder="https://youtu.be/…",
            help=(
                "The MIDI file does not contain the video address. "
                "A project folder saves it in manifest.json for synced playback."
            ),
        )
    if not st.button("Open notation", type="primary"):
        return
    chosen = _midi_from_input(midi_path, midi_upload, Path(output))
    attached = midi_youtube.strip()
    if attached and youtube_video_id(attached) is None:
        st.error("That is not a YouTube URL.")
        return
    if chosen is None:
        st.error("Choose a MIDI file first.")
        return
    _open_midi(chosen, attached, Path(output))


def _open_midi(midi: Path, attached: str, output: Path) -> None:
    """Open a project page for a project MIDI, or draw any other MIDI in a new tab."""

    import streamlit as st

    from baroque_transcribe.notation import NotationError, musicxml_for_midi, write_notation_page

    midi = midi.expanduser().resolve()
    try:
        score = musicxml_for_midi(midi)
        project_score = midi.parent / SCORE_XML
        used_existing = score == project_score and (midi.parent / NOTES_JSON).is_file()
        saved_video = False
        if attached:
            remembered = youtube_project_for_midi(midi, output)
            if remembered is not None:
                write_youtube_url(remembered, attached)
                saved_video = True
        if used_existing:
            if saved_video:
                st.session_state["flash"] = ("success", "The YouTube address was saved with the project.")
            st.session_state["open_project"] = str(midi.parent)
            st.switch_page(_PAGES["project"])
            return
        page = write_notation_page(
            score,
            midi.parent / "notation.html",
            title=midi.stem,
            note="Each track in the MIDI file is one staff. The piano model was not run.",
        )
        webbrowser.open_new_tab(page.as_uri())
    except (NotationError, OSError, SyncError, json.JSONDecodeError) as exc:
        st.error(str(exc))
        return
    message = f"Opened notation for {midi.name} in a new tab. The piano model was not run."
    if saved_video:
        message += " The YouTube address was saved with the project."
    st.success(message)


def _cards_html(cards: list[Card]) -> str:
    parts = ['<section class="bt-grid">']
    for card in cards:
        parts.append(_card_html(card))
    parts.append("</section>")
    return "".join(parts)


def _card_html(card: Card) -> str:
    primary = card.primary
    href = _project_href(primary.folder) if primary else None
    title = escape(card.title)
    available = [run for run in card.runs if run.available]
    badge = f"{len(available)} run{'' if len(available) == 1 else 's'}" if len(card.runs) > 1 else ""
    if primary and primary.duration:
        badge = _clock(primary.duration) + (f" · {badge}" if badge else "")
    if card.thumbnail:
        inner = f'<img src="{escape(card.thumbnail, quote=True)}" alt="">'
        thumb_class = "bt-thumb"
    else:
        inner = "&#9835;"
        thumb_class = "bt-thumb bt-placeholder"
    play = '<span class="bt-play"><span>&#9654;</span></span>' if href else ""
    badge_html = f'<span class="bt-badge">{escape(badge)}</span>' if badge else ""
    if href:
        thumb = f'<a class="{thumb_class}" href="{href}" target="_top">{inner}{play}{badge_html}</a>'
        heading = f'<a class="bt-title" href="{href}" target="_top">{title}</a>'
    else:
        thumb = f'<div class="{thumb_class}">{inner}{badge_html}</div>'
        heading = f'<span class="bt-title">{title}</span>'
    meta_bits: list[str] = []
    if primary:
        if primary.notes is not None:
            meta_bits.append(f"{primary.notes:,} notes")
        if primary.modified:
            meta_bits.append(_relative(primary.modified))
    elif card.runs:
        meta_bits.append("not on this computer")
    meta = f'<div class="bt-meta">{escape(" · ".join(meta_bits))}</div>' if meta_bits else ""
    chips: list[str] = []
    for run in card.runs:
        label = escape(run.name)
        if run.available:
            chips.append(
                f'<a class="bt-chip" href="{_project_href(run.folder)}" target="_top" title="{label}">{label}</a>'
            )
        else:
            chips.append(f'<span class="bt-chip bt-off" title="{label} is not on this computer">{label}</span>')
    runs = f'<div class="bt-runs">{"".join(chips)}</div>' if len(card.runs) > 1 or not primary else ""
    return f'<article class="bt-card">{thumb}<div class="bt-body">{heading}{meta}{runs}</div></article>'


# ------------------------------------------------------------------------ project


def _project_page() -> None:
    import streamlit as st

    requested = st.query_params.get("project") or st.session_state.pop("open_project", None)
    project = find_project(requested)
    if project is None:
        _brand(crumb="Project")
        st.warning("That transcription is not on this computer, or the address is incomplete.")
        st.page_link(_PAGES["home"], label="Back to the library", icon=":material/arrow_back:")
        return
    st.query_params["project"] = project_param(project)
    run = read_run(project)
    if run is None:
        st.switch_page(_PAGES["home"])
        return
    card = card_for(project)
    title = run.title or (card.title if card else "") or project.name

    _brand(crumb=title)
    message = st.session_state.pop("transcription_message", None)
    if message:
        st.success(message)
    _flash()
    pages = _score_pages(project, run, title)
    _project_header(run, card, title, pages)

    manifest = _manifest(project)
    score_tab_label = "Video and score" if run.video_id else "Score"
    tabs = st.tabs([score_tab_label, "Listen", "Notes", "Files"])
    with tabs[0]:
        _score_tab(project, run, pages)
    with tabs[1]:
        _listen_tab(project, manifest)
    with tabs[2]:
        _notes_tab(project, manifest)
    with tabs[3]:
        _files_tab(project, run, card, manifest)


def _score_pages(project: Path, run: Run, title: str) -> dict[str, str]:
    """Addresses of the served score pages for this project, plus any reason one is missing."""

    pages: dict[str, str] = {}
    if not run.has_score:
        return pages
    if run.video_id:
        try:
            pages["playback"] = open_synced_playback(project, title=title)
        except (SyncError, OSError, ValueError) as exc:
            pages["playback_error"] = str(exc)
    try:
        pages["notation"] = notation_page_url(project, title=title)
    except (SyncError, OSError) as exc:
        pages["notation_error"] = str(exc)
    return pages


def _project_header(run: Run, card: Card | None, title: str, pages: dict[str, str]) -> None:
    import streamlit as st

    facts: list[str] = []
    if run.notes is not None:
        facts.append(f"<b>{run.notes:,}</b> notes")
    if run.pedals is not None:
        facts.append(f"<b>{run.pedals:,}</b> pedal events")
    if run.duration:
        facts.append(f"<b>{_clock(run.duration)}</b> excerpt")
    if run.excerpt_start:
        facts.append(f"from <b>{_clock(run.excerpt_start)}</b> in the video")
    if run.bpm:
        facts.append(f"<b>{run.bpm:g}</b> BPM")
    if run.time_signature:
        facts.append(f"<b>{escape(run.time_signature)}</b>")
    if run.model:
        facts.append(escape(run.model))
    if run.modified:
        facts.append(escape(run.modified.strftime("%b %d, %Y %H:%M")))
    if card and card.thumbnail:
        poster = f'<img class="bt-poster" src="{escape(card.thumbnail, quote=True)}" alt="">'
    else:
        poster = '<div class="bt-poster bt-placeholder">&#9835;</div>'
    st.html(
        f'<div class="bt-head">{poster}<div><h1>{escape(title)}</h1>'
        f'<div class="bt-facts">{"".join(f"<span class=\'bt-fact\'>{fact}</span>" for fact in facts)}</div>'
        f'<div class="bt-path">{escape(str(run.folder))}</div></div></div>'
    )
    actions = st.columns([1.2, 1.2, 1.2, 4], gap="small")
    column = 0
    if "playback" in pages:
        actions[column].link_button(
            "Play with score", pages["playback"], type="primary", icon=":material/play_arrow:", width="stretch"
        )
        column += 1
    if "notation" in pages:
        actions[column].link_button("Score only", pages["notation"], icon=":material/music_note:", width="stretch")
        column += 1
    if card and card.url:
        actions[column].link_button("YouTube", card.url, icon=":material/open_in_new:", width="stretch")
    actions[-1].caption("Play with score and Score only open full-size in a new tab.")


def _score_tab(project: Path, run: Run, pages: dict[str, str]) -> None:
    import streamlit as st
    import streamlit.components.v1 as components

    if not run.has_score:
        st.info("This project has no score yet. Set a tempo on the Notes tab and rebuild.")
        return
    if "playback" in pages:
        components.iframe(pages["playback"] + "?embed=1", height=820, scrolling=False)
        st.caption(
            "The highlighted measure follows the video. Click a measure to jump there. "
            "Notation earlier and later shift the highlight when the score runs ahead of or behind the sound."
        )
        return
    if "playback_error" in pages:
        st.warning(pages["playback_error"])
    if "notation" in pages:
        components.iframe(pages["notation"] + "?embed=1", height=820, scrolling=True)
    elif "notation_error" in pages:
        st.warning(pages["notation_error"])
    if not run.video_id:
        _attach_video_form(project)


def _attach_video_form(project: Path) -> None:
    import streamlit as st

    with st.expander("Attach a YouTube video so the score can follow it"):
        url = st.text_input("YouTube URL", key=f"attach-{project.name}", placeholder="https://youtu.be/…")
        if st.button("Attach", key=f"attach-button-{project.name}"):
            try:
                write_youtube_url(project, url)
            except (SyncError, OSError) as exc:
                st.error(str(exc))
            else:
                st.session_state["flash"] = ("success", "The YouTube address was saved with the project.")
                st.rerun()


def _listen_tab(project: Path, manifest: dict) -> None:
    import streamlit as st

    diagnostics = project / DIAGNOSTICS_DIR
    working = project / WORKING_WAV
    comparison = diagnostics / "comparison.wav"
    left, right = st.columns(2, gap="large")
    if working.exists():
        left.markdown("**Working audio**")
        left.audio(working.read_bytes(), format="audio/wav")
        left.caption("Trimmed, mono, 16 kHz, peak-normalized. MIDI time zero is the start of this file.")
    else:
        left.info("working.wav is not in this folder, so there is nothing to play here.")
    if comparison.exists():
        right.markdown("**Recording with the detected notes**")
        right.audio(comparison.read_bytes(), format="audio/wav")
        right.caption("A tone at each detected pitch, mixed over the recording. It checks timing, not piano sound.")
    overview = diagnostics / "overview.png"
    if overview.exists():
        st.image(str(overview), caption="Waveform, spectrogram, and detected notes", width="stretch")
    for name, caption in (("piano_roll.png", "Piano roll"), ("spectrogram.png", "Spectrogram")):
        path = diagnostics / name
        if path.exists() and not overview.exists():
            st.image(str(path), caption=caption, width="stretch")


def _notes_tab(project: Path, manifest: dict) -> None:
    import pandas as pd
    import streamlit as st

    transcription = read_transcription(project / NOTES_JSON)
    tempo_info = manifest.get("tempo") or {}
    st.caption(
        "Edit a pitch, time, or velocity, then rebuild. Times are seconds from the start of the working audio. "
        "Rows with an empty pitch, onset, or offset are dropped."
    )
    frame = pd.DataFrame(
        [
            {"pitch": note.pitch, "onset": note.onset, "offset": note.offset, "velocity": note.velocity}
            for note in transcription.notes
        ]
    )
    edited = st.data_editor(frame, num_rows="dynamic", width="stretch", height=420, key=f"notes-{project.name}")
    a, b, c, d = st.columns(4)
    tempo = a.text_input("Tempo", value=str(tempo_info.get("bpm") or "auto"), key=f"tempo-{project.name}")
    meter = b.text_input("Time signature", value=str(tempo_info.get("time_signature") or "4/4"), key=f"meter-{project.name}")
    options = [16, 8, 32]
    current = int(tempo_info.get("subdivision") or 16)
    subdivision = c.selectbox(
        "Subdivision",
        options,
        index=options.index(current) if current in options else 0,
        key=f"subdivision-{project.name}",
    )
    staff_split = d.number_input(
        "Treble from MIDI pitch",
        min_value=21,
        max_value=108,
        value=int(manifest.get("staff_split") or 60),
        key=f"split-{project.name}",
    )
    if not st.button("Save notes and rebuild MIDI and MusicXML", type="primary", key=f"rebuild-{project.name}"):
        return
    notes = []
    for row in edited.to_dict(orient="records"):
        if any(_blank(row.get(key)) for key in ("pitch", "onset", "offset")):
            continue
        notes.append(
            NoteEvent(
                pitch=int(row["pitch"]),
                onset=float(row["onset"]),
                offset=float(row["offset"]),
                velocity=int(row.get("velocity") or 80),
            )
        )
    updated = Transcription(
        notes=notes,
        pedals=[PedalEvent(onset=item.onset, offset=item.offset) for item in transcription.pedals],
        model=transcription.model,
        source=transcription.source,
    )
    updated.write(project / NOTES_JSON)
    try:
        from baroque_transcribe.pipeline import requantize_project

        requantize_project(
            project,
            tempo=None if tempo in {"", "auto"} else tempo,
            time_signature=meter or None,
            subdivision=int(subdivision),
            staff_split=int(staff_split),
        )
    except Exception as exc:
        st.exception(exc)
    else:
        st.session_state["flash"] = ("success", "Rebuilt raw.mid, quantized.mid, and score.musicxml from the edited notes.")
        st.rerun()


def _files_tab(project: Path, run: Run, card: Card | None, manifest: dict) -> None:
    import pandas as pd
    import streamlit as st

    st.markdown("**Download**")
    columns = st.columns(5)
    for column, filename, label in zip(
        columns,
        (RAW_MID, QUANTIZED_MID, SCORE_XML, NOTES_JSON, PEDAL_JSON),
        ("Raw MIDI", "Quantized MIDI", "MusicXML", "notes.json", "pedal.json"),
    ):
        path = project / filename
        if path.exists():
            column.download_button(label, path.read_bytes(), file_name=filename, key=f"dl-{filename}-{project.name}", width="stretch")
        else:
            column.caption(f"No {filename}")
    st.caption(
        "raw.mid keeps the performance timing at a 120 BPM clock. quantized.mid and score.musicxml are the "
        "rhythmic reading. Open the MusicXML in MuseScore."
    )

    warnings = manifest.get("warnings") or []
    if warnings:
        st.markdown("**Notes from the run**")
        for warning in warnings:
            st.warning(warning)
    accuracy = manifest.get("accuracy")
    if accuracy:
        st.caption(accuracy)

    with st.expander("Run details"):
        details = {
            "Folder": str(project),
            "Source": str(manifest.get("origin") or ""),
            "Program": str(manifest.get("program") or ""),
            "Model": str(manifest.get("model") or run.model),
            "Device": str(manifest.get("device") or ""),
            "Thresholds": json.dumps(manifest.get("thresholds") or {}),
            "Tempo": json.dumps(manifest.get("tempo") or {}),
        }
        st.table(pd.DataFrame({"Field": list(details), "Value": list(details.values())}).set_index("Field"))

    if card and len(card.runs) > 1:
        st.markdown("**Other runs of this recording**")
        chips: list[str] = []
        for other in card.runs:
            if other.folder.resolve() == project.resolve():
                continue
            label = escape(other.name)
            if other.available:
                detail = []
                if other.notes is not None:
                    detail.append(f"{other.notes:,} notes")
                if other.modified:
                    detail.append(_relative(other.modified))
                suffix = f" · {escape(', '.join(detail))}" if detail else ""
                chips.append(f'<a class="bt-chip" href="{_project_href(other.folder)}">{label}{suffix}</a>')
            else:
                chips.append(f'<span class="bt-chip bt-off">{label} · not on this computer</span>')
        st.html(f'<div class="bt-runs">{"".join(chips)}</div>')


# ------------------------------------------------------------------------- shared


def _brand(crumb: str | None = None) -> None:
    import streamlit as st

    if crumb:
        trail = f'<span class="bt-crumb"><a href="/">Library</a> &nbsp;/&nbsp; {escape(crumb)}</span>'
    else:
        trail = '<span class="bt-tag">Solo piano recordings into MIDI and a score, on this computer.</span>'
    st.html(f'<div class="bt-brand"><a class="bt-wordmark" href="/">Piano transcription</a>{trail}</div>')


def _flash() -> None:
    import streamlit as st

    flash = st.session_state.pop("flash", None)
    if not flash:
        return
    level, text = flash
    getattr(st, level, st.info)(text)


def _default_output() -> str:
    requested = os.environ.get("BAROQUE_UI_PROJECT")
    return requested or _DEFAULT_OUTPUT


def _project_href(folder: Path) -> str:
    return f"/project?project={quote(project_param(folder))}"


def _manifest(project: Path) -> dict:
    path = project / MANIFEST
    if not path.is_file():
        return {}
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return loaded if isinstance(loaded, dict) else {}


def _clock(seconds: float) -> str:
    whole = int(round(seconds))
    minutes, secs = divmod(whole, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _relative(moment: datetime) -> str:
    delta = datetime.now() - moment
    days = delta.days
    if days <= 0:
        hours = delta.seconds // 3600
        if hours < 1:
            return "just now"
        return f"{hours} hour{'' if hours == 1 else 's'} ago"
    if days == 1:
        return "yesterday"
    if days < 14:
        return f"{days} days ago"
    return moment.strftime("%b %d, %Y")


def _midi_from_input(text: str, upload, project: Path) -> Path | None:
    from baroque_transcribe.pipeline import unique_path

    if upload is not None:
        folder = project / "_midi"
        folder.mkdir(parents=True, exist_ok=True)
        dest = unique_path(folder / Path(upload.name).name)
        dest.write_bytes(upload.getbuffer())
        return dest
    raw = text.strip()
    if not raw:
        return None
    return Path(raw)


def _run_with_progress(
    source: str,
    project: Path,
    *,
    start: str | None,
    end: str | None,
    model: str,
    device: str,
    tempo: str,
    meter: str | None,
    subdivision: int,
    downbeat: str,
    staff_split: int,
) -> Path:
    import streamlit as st

    from baroque_transcribe.pipeline import run_transcription

    labels: dict[str, str] = {}
    with st.status("Starting transcription…", expanded=True) as status:
        slots = {stage: st.empty() for stage in _STAGE_ORDER}
        for stage in _STAGE_ORDER:
            slots[stage].markdown(f"{_STAGE_TITLES[stage]} — waiting")
        bar = st.progress(0.0, text="Starting")

        def on_progress(event: ProgressEvent) -> None:
            labels[event.stage] = event.label
            active = _STAGE_ORDER.index(event.stage) if event.stage in _STAGE_ORDER else 0
            for index, stage in enumerate(_STAGE_ORDER):
                title = _STAGE_TITLES[stage]
                text = labels.get(stage)
                if stage == event.stage:
                    slots[stage].markdown(f"**{title}** — {text}")
                elif text and index < active:
                    slots[stage].markdown(f"Done. {title} — {text}")
                elif text:
                    slots[stage].markdown(f"{title} — {text}")
                else:
                    slots[stage].markdown(f"{title} — waiting")
            if event.fraction is not None:
                bar.progress(min(1.0, max(0.0, event.fraction)), text=event.label)
            status.update(label=event.label, state="running")

        try:
            written = run_transcription(
                source,
                project,
                start=start,
                end=end,
                model=model,
                device=device,
                tempo=tempo,
                time_signature=meter,
                subdivision=subdivision,
                downbeat=downbeat,
                staff_split=staff_split,
                on_progress=on_progress,
            )
        except Exception:
            status.update(label="Transcription failed", state="error")
            raise
        status.update(label="Transcription finished", state="complete")
        bar.progress(1.0, text="Transcription finished")
        return written.output_dir


def _blank(value: object) -> bool:
    if value is None or value == "":
        return True
    try:
        return float(value) != float(value)  # NaN
    except (TypeError, ValueError):
        return False


def _resolve_source(kind: str, youtube: str, local_path: str, upload, project: Path) -> str | None:
    if kind == "YouTube URL":
        return youtube.strip() or None
    if kind == "Local file":
        return local_path.strip() or None
    if upload is None:
        return None
    from baroque_transcribe.pipeline import unique_path

    inbox = project / "_uploads"
    inbox.mkdir(parents=True, exist_ok=True)
    dest = unique_path(inbox / upload.name)
    dest.write_bytes(upload.getbuffer())
    return str(dest)


if __name__ == "__main__":
    run()
