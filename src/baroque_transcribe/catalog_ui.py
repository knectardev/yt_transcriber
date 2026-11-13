"""Streamlit page for the Bach keyboard MIDI catalog."""

from __future__ import annotations

from html import escape
from pathlib import Path

from baroque_transcribe.catalog import (
    CatalogError,
    all_collections,
    all_keys,
    entry_statuses_with_fragments,
    load_catalog,
    search_catalog,
)
from baroque_transcribe.catalog_movements import movements_for_entry
from baroque_transcribe.catalog_open import open_catalog_entry
from baroque_transcribe.catalog_sync import (
    CatalogSyncError,
    SyncSettings,
    ensure_local_midi,
    setup_sync_logger,
    sync_missing,
)
from baroque_transcribe.defaults import (
    CATALOG_DELAY_MAX_SECONDS,
    CATALOG_DELAY_MIN_SECONDS,
    NOTES_JSON,
)
from baroque_transcribe.library import find_project
from baroque_transcribe.midi_import import MidiImportError
from baroque_transcribe.playback import detail_url

PAGES: dict[str, object] = {}


def bind_pages(pages: dict[str, object]) -> None:
    PAGES.update(pages)


def catalog_href() -> str:
    return "/catalog"


def catalog_page() -> None:
    import streamlit as st

    from baroque_transcribe.ui import _brand, _flash, _navigate_same_tab

    pending = st.session_state.pop("catalog_open_url", None)
    if pending:
        _brand(crumb="Bach catalog", current="catalog")
        _navigate_same_tab(str(pending))
        return

    _brand(crumb="Bach catalog", current="catalog")
    _flash()
    try:
        catalog = load_catalog()
    except CatalogError as exc:
        st.error(str(exc))
        return

    statuses = entry_statuses_with_fragments(catalog)
    downloaded = sum(1 for row in statuses if row.local)
    imported = sum(1 for row in statuses if row.imported_projects)
    st.html(
        '<div class="bt-section bt-section-tight"><h2>Bach catalog</h2>'
        f'<span class="bt-count">{catalog.entry_count} works · {downloaded} downloaded'
        f" · {imported} opened as projects</span></div>"
        '<p class="bt-lede">This is the full Bach keyboard list. Download keeps MIDI in the local mirror; '
        '<b>Open score</b> creates a project. Only opened works also appear under '
        '<a href="/projects">Projects → Opened Bach</a>. '
        "Multi-movement suites can expand to draft section jumps (tempo-map based — pending your review).</p>"
    )
    st.markdown(
        f"{catalog.source.attribution} Noncommercial use with attribution. "
        f"[Terms (EN)]({catalog.source.terms_en_url}) · "
        f"[Terms (DE)]({catalog.source.terms_de_url})"
    )

    rows = _toolbar(catalog, statuses)
    _sync_panel(catalog, statuses)

    if not rows:
        st.info("Nothing matches those filters.")
        return

    for row in rows:
        _row(row, catalog=catalog)


def _toolbar(catalog, statuses):
    import streamlit as st

    text = st.text_input(
        "Search",
        placeholder="Search BWV, title, collection…",
        key="catalog-search",
        label_visibility="collapsed",
    )
    c1, c2, c3, c4, c5 = st.columns([1.4, 1, 1, 1, 1], gap="small")
    collections = all_collections(catalog)
    collection_labels = ["Any collection"] + [
        f"{title} ({count})" for _cid, title, count in collections
    ]
    chosen_collection = c1.selectbox(
        "Collection", collection_labels, key="catalog-collection", label_visibility="collapsed"
    )
    keys = all_keys(catalog)
    key_labels = ["Any key"] + [f"{name} ({count})" for name, count in keys]
    chosen_key = c2.selectbox("Key", key_labels, key="catalog-key", label_visibility="collapsed")
    variants = c3.selectbox(
        "Variants",
        ["include", "exclude", "only"],
        format_func=lambda value: {
            "include": "Variants included",
            "exclude": "Hide variants",
            "only": "Variants only",
        }[value],
        key="catalog-variants",
        label_visibility="collapsed",
    )
    status = c4.selectbox(
        "Status",
        ["all", "missing", "downloaded", "imported", "hash mismatch"],
        format_func=lambda value: value.replace("hash mismatch", "Hash mismatch").title()
        if value != "all"
        else "Any status",
        key="catalog-status",
        label_visibility="collapsed",
    )
    sort = c5.selectbox(
        "Sort",
        ["bwv", "title", "collection"],
        format_func=lambda value: {"bwv": "BWV", "title": "Title", "collection": "Collection"}[value],
        key="catalog-sort",
        label_visibility="collapsed",
    )
    collection_id = ""
    if chosen_collection != "Any collection":
        collection_id = collections[collection_labels.index(chosen_collection) - 1][0]
    key = ""
    if chosen_key != "Any key":
        key = keys[key_labels.index(chosen_key) - 1][0]
    return search_catalog(
        catalog,
        text=text,
        collection_id=collection_id,
        key=key,
        variants=variants,
        status=status,
        statuses=statuses,
        sort=sort,
    )


def _sync_panel(catalog, statuses) -> None:
    import streamlit as st

    missing = [row for row in statuses if not row.local or row.hash_ok is False]
    with st.expander(
        f"Sync missing files ({len(missing)} remaining)",
        expanded=False,
    ):
        st.caption(
            "Bulk sync downloads one file at a time with a random 30–200 second delay. "
            "Use keep-awake for overnight runs. Verified files are skipped on restart."
        )
        keep_awake = st.checkbox("Keep this PC awake while syncing", value=True, key="catalog-keep-awake")
        confirm = st.checkbox(
            "I understand this contacts Tobis Notenarchiv for personal noncommercial use",
            value=False,
            key="catalog-sync-confirm",
        )
        if st.button(
            "Sync missing files overnight",
            type="primary",
            disabled=not missing or not confirm,
            key="catalog-sync-start",
        ):
            settings = SyncSettings(
                delay_min=CATALOG_DELAY_MIN_SECONDS,
                delay_max=CATALOG_DELAY_MAX_SECONDS,
                overnight=True,
                keep_awake=keep_awake,
            )
            logger = setup_sync_logger()
            progress = st.progress(0.0, text="Starting…")

            def on_progress(index: int, total: int, result) -> None:
                progress.progress(
                    index / max(1, total),
                    text=f"{index}/{total} {result.entry_id}: {result.action}",
                )

            try:
                report = sync_missing(
                    catalog,
                    settings=settings,
                    logger=logger,
                    on_progress=on_progress,
                )
            except CatalogSyncError as exc:
                st.error(str(exc))
                return
            st.session_state["flash"] = (
                "success" if report.ok else "warning",
                report.summary(),
            )
            st.rerun()


def _row(row, *, catalog) -> None:
    import streamlit as st

    entry = row.entry
    movements = movements_for_entry(entry.id, include_draft=True)
    with st.container(border=True):
        left, right = st.columns([4, 2], vertical_alignment="center")
        with left:
            meta = " · ".join(
                part
                for part in (
                    f"BWV {entry.bwv}" if entry.bwv else "",
                    entry.key,
                    row.status_label,
                    f"{row.fragment_count} fragment{'s' if row.fragment_count != 1 else ''}"
                    if row.fragment_count
                    else "",
                    "variant" if entry.is_variant else "",
                    (
                        f"{len(movements.movements)} sections"
                        if movements is not None and len(movements.movements) > 1
                        else ""
                    ),
                )
                if part
            )
            st.markdown(f"**{escape(entry.display_title)}**")
            st.caption(f"{escape(entry.collection)} · {escape(meta)}")
        with right:
            a, b, c = st.columns(3)
            _download_button(a, row=row, catalog=catalog)
            _open_score_button(b, row=row)
            if c.button("Fragments", key=f"cat-frags-{entry.id}"):
                st.session_state["frag-search"] = entry.id
                st.session_state["catalog_fragment_filter"] = entry.id
                if "home" in PAGES:
                    st.switch_page(PAGES["home"])
                elif "fragments" in PAGES:
                    st.switch_page(PAGES["fragments"])
        if movements is not None and len(movements.movements) > 1:
            label = f"Sections ({len(movements.movements)})"
            if movements.is_draft:
                label += " — draft"
            with st.expander(label, expanded=False):
                if movements.is_draft:
                    st.caption(
                        "Draft partitions from the MIDI tempo map — confirm before relying on them."
                    )
                ready = _ready_project(row)
                for movement in movements.movements:
                    cols = st.columns([4, 1])
                    cols[0].markdown(
                        f"**{escape(movement.title)}**  \n"
                        f"{movement.start_seconds:.0f}s–{movement.end_seconds:.0f}s"
                    )
                    if ready is not None:
                        url = detail_url(ready, movement_id=movement.id)
                        cols[1].html(
                            f'<a class="bt-catalog-open" href="{escape(url, quote=True)}" '
                            f'target="_top">Open</a>'
                        )
                    elif cols[1].button("Open", key=f"cat-mov-{entry.id}-{movement.id}"):
                        try:
                            project = open_catalog_entry(entry.id)
                        except (
                            CatalogError,
                            CatalogSyncError,
                            MidiImportError,
                            OSError,
                            ValueError,
                        ) as exc:
                            st.error(str(exc))
                        else:
                            st.session_state["catalog_open_url"] = detail_url(
                                project, movement_id=movement.id
                            )
                            st.rerun()


def _download_button(column, *, row, catalog) -> None:
    import streamlit as st

    entry = row.entry
    if row.local and row.hash_ok is not False:
        column.button("Downloaded", key=f"cat-dl-{entry.id}", disabled=True)
        return
    label = "Redownload" if row.hash_ok is False else "Download"
    if column.button(label, key=f"cat-dl-{entry.id}"):
        try:
            ensure_local_midi(entry, allowed_hosts=catalog.source.allowed_hosts)
        except (CatalogError, CatalogSyncError, OSError) as exc:
            st.error(str(exc))
        else:
            st.session_state["flash"] = ("success", f"Downloaded {entry.id}.")
            st.rerun()


def _open_score_button(column, *, row) -> None:
    """One click → same-tab score. Materialize first when needed, then redirect at page top."""

    import streamlit as st

    entry = row.entry
    ready = _ready_project(row)
    with column:
        if ready is not None:
            url = detail_url(ready)
            st.html(
                f'<a class="bt-catalog-open" href="{escape(url, quote=True)}" '
                'target="_top">Open score</a>'
            )
            return
        if st.button("Open score", key=f"cat-open-{entry.id}"):
            try:
                project = open_catalog_entry(entry.id)
            except (CatalogError, CatalogSyncError, MidiImportError, OSError, ValueError) as exc:
                st.error(str(exc))
            else:
                st.session_state["catalog_open_url"] = detail_url(project)
                st.rerun()


def _ready_project(row) -> Path | None:
    """Existing imported project that can open without a Streamlit round-trip."""

    for name in row.imported_projects:
        folder = find_project(name)
        if folder is not None and (folder / NOTES_JSON).is_file():
            return folder
    return None
