"""Streamlit pages for the fragment library.

The playback page marks a passage and sends it here. These pages save it,
find it again, and edit the fields that travel with it. The video and the
score stay on the localhost pages that ``playback`` serves; this module
links to them and never copies audio into a fragment.
"""

from __future__ import annotations

from html import escape
from pathlib import Path
from urllib.parse import quote

from baroque_transcribe.defaults import (
    FRAGMENT_GRID_JSON,
    FRAGMENT_HARMONY_JSON,
    FRAGMENT_JSON,
    FRAGMENT_PERFORMANCE_MID,
    NOTES_JSON,
    QUANTIZED_MID,
    SCORE_XML,
)
from baroque_transcribe.fragment_index import (
    IndexedFragment,
    ReconcileReport,
    all_keys,
    all_sources,
    all_tags,
    fragment_count,
    reconcile,
    reconcile_if_due,
    refresh_folder,
    search_fragments,
    stored_problems,
)
from baroque_transcribe.fragment_pages import (
    fragment_detail_url,
    fragment_has_video,
    parent_place,
    youtube_place_url,
)
from baroque_transcribe.fragments import (
    FragmentError,
    describe_plan,
    ensure_fragment_preview_wav,
    extract_fragment,
    find_fragment,
    normalize_tags,
    parse_measures,
    plan_fragment,
    preview_notes,
    read_manifest,
    update_fragment_metadata,
)
from baroque_transcribe.preview_synth import (
    DEFAULT_PREVIEW_SOUND,
    PREVIEW_SOUNDS,
    piano_roll_svg,
    preview_sound_label,
)
from baroque_transcribe.library import find_project, format_clock, project_param, read_run

# Filled by ``ui.run`` once the Streamlit pages exist, so these handlers can
# switch to them. Circular imports are avoided: this module does not import
# ``ui``, and ``ui`` only calls ``bind_pages`` after creating the pages.
PAGES: dict[str, object] = {}

_SORT_LABELS = {
    "modified": "Recently edited",
    "created": "Recently saved",
    "title": "Title",
    "source": "Source",
    "duration": "Duration",
}


def bind_pages(pages: dict[str, object]) -> None:
    PAGES.update(pages)


def prepare_index() -> None:
    """Read every manifest once when the UI process starts.

    Later rerenders use the cheap incremental pass, so typing in a search
    box does not open the files again.
    """

    import streamlit as st

    if st.session_state.get("fragment_index_ready"):
        return
    try:
        reconcile(mode="full")
    except (FragmentError, OSError) as exc:
        st.session_state["fragment_index_error"] = str(exc)
    st.session_state["fragment_index_ready"] = True


# --------------------------------------------------------------------------
# Addresses
# --------------------------------------------------------------------------


def fragments_href() -> str:
    return "/fragments"


def fragment_href(fragment_id: str) -> str:
    return f"/fragment?id={quote(str(fragment_id), safe='')}"


def extract_href(project: Path, measures: str | None = None) -> str:
    address = f"/extract?project={quote(project_param(project))}"
    return f"{address}&measures={quote(measures)}" if measures else address


def last_credits(project: Path) -> tuple[str, str]:
    """Composer and performer from the newest fragment of this recording, if any."""

    run = read_run(project)
    source = (run.video_id if run and run.video_id else project.name)
    rows = search_fragments(source=source, sort="created")
    if not rows:
        return "", ""
    return rows[0].composer, rows[0].performer


# --------------------------------------------------------------------------
# Library
# --------------------------------------------------------------------------


def fragments_page() -> None:
    import streamlit as st

    from baroque_transcribe.ui import _brand, _flash

    _brand(crumb="Fragments")
    _flash()
    report = reconcile_if_due()
    if report is not None and report.changed:
        st.caption(report.summary())
    _show_index_error()
    _show_problems()

    count = fragment_count()
    st.html(
        '<div class="bt-section bt-section-tight"><h2>Fragment library</h2>'
        f'<span class="bt-count">{count} fragment{"" if count == 1 else "s"}</span></div>'
    )
    rows = _library_toolbar()
    if not rows:
        if count == 0:
            st.html(
                '<div class="bt-empty">Nothing has been saved yet. Open a transcription, '
                "choose Select passage, and extract a fragment.</div>"
            )
        else:
            st.info("Nothing matches those filters.")
        _index_buttons()
        return
    _show_fragment_cards(rows)
    _index_buttons()


def _library_toolbar() -> list[IndexedFragment]:
    """Search, filters, and preview sound in two tight rows (labels collapsed)."""

    import streamlit as st

    st.html(
        "<style>"
        "section.main .block-container div[data-testid='stVerticalBlock'] { gap: 0.35rem !important; }"
        "section.main div[data-testid='stWidgetLabel'] p { font-size: 0.72rem !important; }"
        "</style>"
    )
    sound_ids = [sound.id for sound in PREVIEW_SOUNDS]
    if "frag-preview-sound" not in st.session_state:
        st.session_state["frag-preview-sound"] = (
            DEFAULT_PREVIEW_SOUND if DEFAULT_PREVIEW_SOUND in sound_ids else sound_ids[0]
        )
    if "frag-preview-mode" not in st.session_state:
        st.session_state["frag-preview-mode"] = "performance"

    search_col, sound_col, mode_col = st.columns([3.2, 1.5, 1.3], gap="small")
    text = search_col.text_input(
        "Search",
        placeholder="Search title, composer, tag…",
        key="frag-search",
        label_visibility="collapsed",
    )
    sound_col.selectbox(
        "Sound",
        sound_ids,
        format_func=preview_sound_label,
        key="frag-preview-sound",
        label_visibility="collapsed",
        help="Preview instrument",
    )
    mode_col.selectbox(
        "Playback",
        ["performance", "quantized"],
        format_func=lambda value: "As played" if value == "performance" else "Quantized",
        key="frag-preview-mode",
        label_visibility="collapsed",
        help="As played keeps performance timing; Quantized follows the written grid.",
    )

    key_col, tag_col, source_col, sort_col, chord_col = st.columns([1, 1, 1, 1, 1.2], gap="small")
    keys = all_keys()
    key_labels = ["Any key"] + [f"{tonic} {mode}" for tonic, mode, _count in keys]
    chosen_key = key_col.selectbox("Key", key_labels, key="frag-key", label_visibility="collapsed")
    tag_names = [name for name, _count in all_tags()]
    chosen_tags = tag_col.multiselect(
        "Tags",
        tag_names,
        key="frag-tags",
        label_visibility="collapsed",
        placeholder="Tags",
    )
    sources = all_sources()
    source_labels = ["Any source"] + [f"{title or ident}" for ident, title, _count in sources]
    chosen_source = source_col.selectbox(
        "Source", source_labels, key="frag-source", label_visibility="collapsed"
    )
    sort_keys = list(_SORT_LABELS)
    sort = sort_col.selectbox(
        "Sort",
        sort_keys,
        format_func=lambda key: _SORT_LABELS[key],
        key="frag-sort",
        label_visibility="collapsed",
    )
    progression = chord_col.text_input(
        "Chords",
        placeholder="Chords: V7 i",
        key="frag-progression",
        label_visibility="collapsed",
        help="Adjacent chord labels in order. V and v are different.",
    )
    tonic = mode = ""
    if chosen_key != "Any key":
        tonic, mode = chosen_key.rsplit(" ", 1)
    source = ""
    if chosen_source != "Any source":
        source = sources[source_labels.index(chosen_source) - 1][0]
    return search_fragments(
        text=text,
        tags=chosen_tags,
        key_tonic=tonic,
        key_mode=mode,
        progression=progression,
        source=source,
        sort=sort,
    )


_ROW_HEIGHT = 46
_LIST_MAX_HEIGHT = 720


def _show_fragment_cards(rows: list[IndexedFragment]) -> None:
    """One dense HTML list: Play, title, piano-roll, facts — Noiiz-like row height."""

    import streamlit as st
    import streamlit.components.v1 as components

    sound = str(st.session_state.get("frag-preview-sound") or DEFAULT_PREVIEW_SOUND)
    mode = str(st.session_state.get("frag-preview-mode") or "performance")
    items: list[str] = []
    for row in rows:
        items.append(_fragment_row_html(row, sound=sound, mode=mode))
    body = "\n".join(items)
    height = min(_LIST_MAX_HEIGHT, 8 + len(rows) * _ROW_HEIGHT)
    components.html(
        f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><style>
  html, body {{ margin: 0; padding: 0; background: transparent; }}
  body {{
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1c1917;
  }}
  .bt-list {{ display: flex; flex-direction: column; }}
  .bt-row {{
    display: grid;
    grid-template-columns: 3.4rem minmax(7rem, 0.9fr) minmax(8rem, 1.4fr) minmax(7rem, 0.9fr) auto;
    gap: 0.55rem; align-items: center;
    min-height: {_ROW_HEIGHT - 1}px; padding: 0 0.15rem;
    border-bottom: 1px solid #e7e2da;
    box-sizing: border-box;
  }}
  .bt-row button.play {{
    font: inherit; font-size: 0.72rem; font-weight: 650; padding: 0.18rem 0.55rem;
    border-radius: 999px; border: 1px solid #e7e2da; background: #b45309; color: #fff; cursor: pointer;
    width: 3.2rem;
  }}
  .bt-row button.play:hover {{ background: #7c3a07; }}
  .bt-row button.play:disabled {{ opacity: 0.45; cursor: default; background: #a8a29e; }}
  .bt-row .title {{
    font-family: Georgia, "Iowan Old Style", "Times New Roman", serif;
    font-size: 0.92rem; font-weight: 600; color: #1c1917; text-decoration: none;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  .bt-row .title:hover {{ color: #7c3a07; }}
  .bt-row .roll {{ min-width: 0; line-height: 0; }}
  .bt-row .roll svg {{ width: 100%; height: 34px; display: block; }}
  .bt-row .meta {{
    color: #6b6560; font-size: 0.72rem; line-height: 1.25;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }}
  .bt-row .chip {{
    font-size: 0.68rem; padding: 0.1rem 0.45rem; border-radius: 999px; border: 1px solid #e7e2da;
    color: #7c3a07; background: #fdf3e2; text-decoration: none; white-space: nowrap;
  }}
  .bt-row .chip:hover {{ border-color: #b45309; }}
</style></head><body>
<div class="bt-list">
{body}
</div>
<script>
(function () {{
  const volume = 0.2;
  function stopAll(except) {{
    document.querySelectorAll("audio").forEach(function (audio) {{
      if (audio === except) return;
      audio.pause();
      audio.currentTime = 0;
    }});
    document.querySelectorAll("button.play").forEach(function (button) {{
      if (!button.disabled) button.textContent = "Play";
    }});
  }}
  document.querySelectorAll("button.play").forEach(function (button) {{
    const audio = document.getElementById(button.getAttribute("data-audio"));
    if (!audio) return;
    audio.volume = volume;
    button.addEventListener("click", function () {{
      audio.volume = volume;
      if (audio.paused) {{
        stopAll(audio);
        audio.play();
        button.textContent = "Stop";
      }} else {{
        audio.pause();
        audio.currentTime = 0;
        button.textContent = "Play";
      }}
    }});
    audio.addEventListener("ended", function () {{
      audio.currentTime = 0;
      button.textContent = "Play";
    }});
  }});
}})();
</script>
</body></html>""",
        height=height,
        scrolling=len(rows) * _ROW_HEIGHT > _LIST_MAX_HEIGHT,
    )


def _fragment_row_html(row: IndexedFragment, *, sound: str, mode: str) -> str:
    import base64

    folder = find_fragment(row.id)
    player = fragment_detail_url(folder or Path(row.rel_dir))
    details = fragment_href(row.id)
    measures = (
        f"mm. {row.measure_start}"
        if row.measure_start == row.measure_end
        else f"mm. {row.measure_start}–{row.measure_end}"
    )
    meta_bits = [
        escape(part)
        for part in (
            row.source_title,
            measures,
            f"{row.duration:.1f}s",
            row.key or "",
            ", ".join(row.tags[:2]),
        )
        if part
    ]
    audio_id = f"frag-audio-{row.id}"
    play = (
        f'<button type="button" class="play" disabled title="Preview unavailable">—</button>'
        f'<audio id="{escape(audio_id, quote=True)}"></audio>'
    )
    roll = piano_roll_svg([], float(row.duration or 1.0))
    if folder is not None:
        try:
            notes, duration = preview_notes(folder, mode)
            roll = piano_roll_svg(notes, duration)
            wav = ensure_fragment_preview_wav(folder, mode=mode, sound=sound)
            payload = base64.b64encode(wav.read_bytes()).decode("ascii")
            play = (
                f'<button type="button" class="play" data-audio="{escape(audio_id, quote=True)}">Play</button>'
                f'<audio id="{escape(audio_id, quote=True)}" preload="metadata" '
                f'src="data:audio/wav;base64,{payload}"></audio>'
            )
        except (FragmentError, OSError, ValueError):
            pass
    return (
        f'<div class="bt-row">'
        f"<div>{play}</div>"
        f'<a class="title" href="{escape(player, quote=True)}" target="_top" title="{escape(row.title)}">'
        f"{escape(row.title)}</a>"
        f'<div class="roll">{roll}</div>'
        f'<div class="meta" title="{" · ".join(meta_bits)}">{" · ".join(meta_bits)}</div>'
        f'<a class="chip" href="{escape(details, quote=True)}" target="_top">Details</a>'
        f"</div>"
    )


def _index_buttons() -> None:
    import streamlit as st

    check, rebuild = st.columns(2)
    if check.button("Check for changes", key="frag-check"):
        _apply_report(reconcile(mode="full"))
        st.rerun()
    if rebuild.button("Rebuild library index", key="frag-rebuild"):
        _apply_report(reconcile(mode="rebuild"))
        st.rerun()


def _apply_report(report: ReconcileReport) -> None:
    import streamlit as st

    st.session_state["flash"] = ("success" if not report.problems else "warning", report.summary())


def _show_problems() -> None:
    import streamlit as st

    problems = stored_problems()
    if not problems:
        return
    for problem in problems:
        st.warning(f"{problem.folder}: {problem.message}")


def _show_index_error() -> None:
    import streamlit as st

    message = st.session_state.get("fragment_index_error")
    if message:
        st.error(f"The search index could not be opened: {message}")


# --------------------------------------------------------------------------
# Extract
# --------------------------------------------------------------------------


def extract_page() -> None:
    import streamlit as st

    from baroque_transcribe.ui import _brand, _flash

    requested = st.query_params.get("project")
    project = find_project(requested)
    if project is None:
        _brand(crumb="Extract fragment")
        st.warning("That transcription is not on this computer, or the address is incomplete.")
        st.page_link(PAGES["home"], label="Back to the library", icon=":material/arrow_back:")
        return
    run = read_run(project)
    heading = (run.title if run else "") or project.name
    _brand(crumb=heading, crumb_href=f"/project?project={quote(project_param(project))}", leaf="Extract fragment")
    _flash()

    try:
        default_start, default_end = parse_measures(str(st.query_params.get("measures") or "1"))
    except FragmentError:
        default_start, default_end = 1, 1
    stamp = f"{project.name}:{st.query_params.get('measures') or ''}"
    if st.session_state.get("extract-range-stamp") != stamp:
        st.session_state["extract-start"] = default_start
        st.session_state["extract-end"] = default_end
        st.session_state["extract-range-stamp"] = stamp
    a, b = st.columns(2)
    start = int(a.number_input("First measure", min_value=1, value=default_start, key="extract-start"))
    end = int(b.number_input("Last measure", min_value=1, value=default_end, key="extract-end"))
    if end < start:
        start, end = end, start

    try:
        plan = plan_fragment(project, start, end)
    except (FragmentError, OSError, ValueError) as exc:
        st.error(str(exc))
        return

    for line in describe_plan(plan):
        st.markdown(line)
    st.caption(
        f"Playback window: {plan.working_start:.2f}–{plan.working_end:.2f} s of the working audio"
        + (
            f" ({format_clock(plan.video_start)}–{format_clock(plan.video_end)} in the video, before any sync adjustment)."
            if run and run.video_id
            else "."
        )
    )
    if plan.unprinted:
        st.caption(
            f"{plan.unprinted} overlapping note{'' if plan.unprinted == 1 else 's'} from the bar before "
            "are kept in the MIDI and omitted from the printed score."
        )

    composer, performer = last_credits(project)
    title = st.text_input(
        "Title",
        value=f"Measures {plan.measure_start}–{plan.measure_end}"
        if plan.measure_start != plan.measure_end
        else f"Measure {plan.measure_start}",
        key=f"extract-title-{project.name}",
    )
    description = st.text_area("Description", key=f"extract-description-{project.name}")
    practice = st.text_area("Practice notes", key=f"extract-practice-{project.name}")
    tags = st.text_input("Tags", placeholder="sequence, cadence", key=f"extract-tags-{project.name}")
    c1, c2 = st.columns(2)
    composer = c1.text_input("Composer", value=composer, key=f"extract-composer-{project.name}")
    performer = c2.text_input("Performer", value=performer, key=f"extract-performer-{project.name}")

    if not st.button("Save fragment", type="primary", key="extract-save"):
        return
    try:
        folder = extract_fragment(
            project,
            start,
            end,
            title=title,
            description=description,
            practice_notes=practice,
            tags=normalize_tags(tags),
            composer=composer,
            performer=performer,
        )
    except (FragmentError, OSError) as exc:
        st.error(str(exc))
        return
    try:
        refresh_folder(folder)
    except (FragmentError, OSError) as exc:
        st.session_state["flash"] = ("warning", f"Saved, but the search index was not updated: {exc}")
    else:
        st.session_state["flash"] = ("success", f"Saved “{read_manifest(folder)['title']}”.")
    st.session_state["open_fragment"] = read_manifest(folder)["id"]
    st.switch_page(PAGES["fragment"])


# --------------------------------------------------------------------------
# One fragment
# --------------------------------------------------------------------------


def fragment_page() -> None:
    import streamlit as st

    from baroque_transcribe.ui import _brand, _flash

    ident = str(st.query_params.get("id") or st.session_state.pop("open_fragment", "") or "")
    folder = find_fragment(ident) if ident else None
    if folder is None:
        _brand(crumb="Fragments", crumb_href=fragments_href(), leaf="Fragment")
        st.warning("That fragment is not in this library.")
        st.page_link(PAGES["fragments"], label="Back to the fragment library", icon=":material/arrow_back:")
        return
    try:
        manifest = read_manifest(folder)
    except FragmentError as exc:
        _brand(crumb="Fragments", crumb_href=fragments_href())
        st.error(str(exc))
        return
    st.query_params["id"] = str(manifest["id"])
    _brand(crumb="Fragments", crumb_href=fragments_href(), leaf=str(manifest["title"]))
    _flash()

    player = fragment_detail_url(folder)
    st.html(f'<a class="bt-back" href="{escape(player, quote=True)}">&larr; Open score page</a>')
    try:
        import streamlit.components.v1 as components

        embed = player if player.endswith("/") else player + "/"
        if fragment_has_video(folder):
            components.iframe(embed + "?embed=1", height=640, scrolling=True)
        else:
            components.iframe(embed + "notation.html?embed=1", height=520, scrolling=True)
            st.info("This fragment was taken from a local recording, so there is no video to play here.")
    except Exception:
        st.caption("The score page could not be framed here. Use Open score page above.")

    _fragment_links(folder, manifest)
    _fragment_form(folder, manifest)
    _fragment_downloads(folder, manifest)


def _fragment_links(folder: Path, manifest: dict) -> None:
    import streamlit as st

    chips: list[str] = []
    youtube = youtube_place_url(manifest)
    if youtube:
        chips.append(
            f'<a class="bt-chip" href="{escape(youtube, quote=True)}" target="_blank" rel="noopener">Open on YouTube</a>'
        )
    place = parent_place(folder)
    if place is not None:
        from baroque_transcribe.playback import detail_url

        project, working = place
        address = f"{detail_url(project)}?w={working:.3f}"
        chips.append(f'<a class="bt-chip" href="{escape(address, quote=True)}">Open source transcription</a>')
    parent = manifest.get("parent") if isinstance(manifest.get("parent"), dict) else {}
    name = str(parent.get("project") or "")
    if name and find_project(name) is None and not chips:
        st.caption("The parent transcription is not on this computer. The fragment still plays on its own.")
    if chips:
        st.html(f'<div class="bt-runs">{"".join(chips)}</div>')


def _fragment_form(folder: Path, manifest: dict) -> None:
    import streamlit as st

    ident = str(manifest["id"])
    title = st.text_input("Title", value=str(manifest.get("title") or ""), key=f"meta-title-{ident}")
    description = st.text_area("Description", value=str(manifest.get("description") or ""), key=f"meta-desc-{ident}")
    practice = st.text_area(
        "Practice notes", value=str(manifest.get("practice_notes") or ""), key=f"meta-practice-{ident}"
    )
    tags = st.text_input("Tags", value=", ".join(manifest.get("tags") or []), key=f"meta-tags-{ident}")
    c1, c2 = st.columns(2)
    composer = c1.text_input("Composer", value=str(manifest.get("composer") or ""), key=f"meta-composer-{ident}")
    performer = c2.text_input("Performer", value=str(manifest.get("performer") or ""), key=f"meta-performer-{ident}")
    if not st.button("Save details", type="primary", key=f"meta-save-{ident}"):
        return
    try:
        update_fragment_metadata(
            folder,
            title=title,
            description=description,
            practice_notes=practice,
            tags=normalize_tags(tags),
            composer=composer,
            performer=performer,
        )
        refresh_folder(folder)
    except (FragmentError, OSError) as exc:
        st.error(str(exc))
        return
    st.session_state["flash"] = ("success", "Details saved.")
    st.rerun()


def _fragment_downloads(folder: Path, manifest: dict) -> None:
    import streamlit as st

    artifacts = manifest.get("artifacts") if isinstance(manifest.get("artifacts"), dict) else {}
    names = [
        (artifacts.get("performance_midi") or FRAGMENT_PERFORMANCE_MID, "Performance MIDI"),
        (artifacts.get("quantized_midi") or QUANTIZED_MID, "Quantized MIDI"),
        (artifacts.get("musicxml") or SCORE_XML, "MusicXML"),
        (artifacts.get("notes") or NOTES_JSON, "notes.json"),
        (artifacts.get("grid") or FRAGMENT_GRID_JSON, "grid.json"),
        (artifacts.get("harmony") or FRAGMENT_HARMONY_JSON, "harmony.json"),
        (FRAGMENT_JSON, "fragment.json"),
    ]
    st.markdown("**Download**")
    columns = st.columns(len(names))
    ident = str(manifest["id"])
    for column, (filename, label) in zip(columns, names):
        path = folder / str(filename)
        if path.is_file():
            column.download_button(
                label, path.read_bytes(), file_name=filename, key=f"dl-{filename}-{ident}", width="stretch"
            )
        else:
            column.caption(f"No {filename}")
    boundary = manifest.get("boundary") if isinstance(manifest.get("boundary"), dict) else {}
    if boundary.get("carried_in"):
        st.caption(
            "A note already sounding at the first barline is struck again at the start of the MIDI, "
            "at its original loudness. That is an approximation: the real attack came earlier."
        )
    warnings = manifest.get("warnings") or []
    for warning in warnings:
        st.warning(warning)


# --------------------------------------------------------------------------
# Project tab
# --------------------------------------------------------------------------


def project_tab(project: Path) -> None:
    """The Fragments tab of one transcription's Edit & files page."""

    import streamlit as st

    reconcile_if_due()
    rows = search_fragments(source=project.name, sort="created")
    st.caption(
        "Save a run of measures as its own score. On a video-and-score page, Select passage "
        "then Extract fragment. Measure numbers work here too, including for a recording with no video."
    )
    a, b, c = st.columns([1, 1, 2])
    start = int(a.number_input("First measure", min_value=1, value=1, key=f"proj-frag-start-{project.name}"))
    end = int(b.number_input("Last measure", min_value=1, value=start, key=f"proj-frag-end-{project.name}"))
    if end < start:
        start, end = end, start
    measures = f"{start}-{end}" if end != start else str(start)
    if c.button("Extract these measures…", key=f"proj-frag-go-{project.name}"):
        open_extract(project, measures)
    st.html(
        f'<a class="bt-chip" href="{escape(extract_href(project, measures), quote=True)}">'
        f"Open extract form for measures {escape(measures)}</a>"
    )

    if not rows:
        st.info("No fragments have been saved from this transcription yet.")
        return
    st.markdown(f"**{len(rows)} fragment{'' if len(rows) == 1 else 's'} from this transcription**")
    _compact_preview_selectors()
    _show_fragment_cards(rows)


def _compact_preview_selectors() -> None:
    """Sound / playback mode only — for the project Fragments tab."""

    import streamlit as st

    sound_ids = [sound.id for sound in PREVIEW_SOUNDS]
    if "frag-preview-sound" not in st.session_state:
        st.session_state["frag-preview-sound"] = (
            DEFAULT_PREVIEW_SOUND if DEFAULT_PREVIEW_SOUND in sound_ids else sound_ids[0]
        )
    if "frag-preview-mode" not in st.session_state:
        st.session_state["frag-preview-mode"] = "performance"
    sound_col, mode_col = st.columns([2, 1.2], gap="small")
    sound_col.selectbox(
        "Sound",
        sound_ids,
        format_func=preview_sound_label,
        key="frag-preview-sound",
        label_visibility="collapsed",
    )
    mode_col.selectbox(
        "Playback",
        ["performance", "quantized"],
        format_func=lambda value: "As played" if value == "performance" else "Quantized",
        key="frag-preview-mode",
        label_visibility="collapsed",
    )


# The extract form reads query params. Setting them then switching is the
# Streamlit way to send the measure range from the project tab; a plain
# anchor (the chip above) does the same without a widget rerun.
def open_extract(project: Path, measures: str) -> None:
    import streamlit as st

    st.query_params["project"] = project_param(project)
    st.query_params["measures"] = measures
    st.switch_page(PAGES["extract"])
