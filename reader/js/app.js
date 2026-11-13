import { loadCatalogIndex, loadCatalogLibrary, loadPackage, mergeCatalogViews } from "./catalog.js";
import { loadReadingPosition, saveReadingPosition } from "./position.js";
import { createScoreController } from "./score.js";
import { loadSettings, saveSettings } from "./settings.js";

const catalogList = document.getElementById("catalog-list");
const catalogEmpty = document.getElementById("catalog-empty");
const errorBanner = document.getElementById("error-banner");
const pieceTitle = document.getElementById("piece-title");
const systemLabel = document.getElementById("system-label");
const zoomInput = document.getElementById("zoom");
const overlapInput = document.getElementById("overlap");
const btnPrev = document.getElementById("btn-prev");
const btnNext = document.getElementById("btn-next");
const btnBack = document.getElementById("btn-back");
const btnPerformance = document.getElementById("btn-performance");

let settings = loadSettings();
let catalogRows = [];
let score = null;
let activeEntryId = null;
let activeMovementId = null;

function showError(message) {
  if (!errorBanner) return;
  errorBanner.hidden = !message;
  errorBanner.textContent = message || "";
}

function applyChrome() {
  document.body.classList.toggle("performance", Boolean(settings.performanceMode));
  if (btnPerformance) {
    btnPerformance.textContent = settings.performanceMode ? "Exit performance" : "Performance";
  }
  if (zoomInput) zoomInput.value = String(settings.zoom);
  if (overlapInput) overlapInput.value = String(settings.overlap);
}

function pieceQueryId(entry) {
  return String(entry.entryId || entry.id || "").replace(/^catalog:/, "");
}

function findEntry(targetId) {
  if (!targetId) return null;
  const needle = String(targetId);
  return (
    catalogRows.find(
      (item) =>
        item.id === needle ||
        item.entryId === needle ||
        `catalog:${item.entryId}` === needle ||
        pieceQueryId(item) === needle.replace(/^catalog:/, "")
    ) || null
  );
}

function findSection(entry, movementId) {
  if (!entry || !movementId) return null;
  const sections = entry.sections || [];
  return sections.find((section) => section.id === movementId) || null;
}

function pieceUrl(entry, { movementId = null, anchorMeasure = null } = {}) {
  const params = new URLSearchParams();
  params.set("id", pieceQueryId(entry));
  if (movementId) params.set("mov", movementId);
  const measure = Number(anchorMeasure);
  if (Number.isFinite(measure) && measure > 1) {
    params.set("m", String(Math.floor(measure)));
  }
  return `?${params.toString()}`;
}

function setPieceLocation(entry, { movementId = null, anchorMeasure = null, replace = false } = {}) {
  if (!entry) return;
  const url = pieceUrl(entry, { movementId, anchorMeasure });
  const state = {
    entryId: pieceQueryId(entry),
    movementId: movementId || null,
    anchorMeasure: Number(anchorMeasure) || 1,
  };
  if (replace) {
    window.history.replaceState(state, "", url);
  } else {
    const current = `${window.location.pathname}${window.location.search}`;
    const next = `${window.location.pathname}${url}`;
    if (current === next) window.history.replaceState(state, "", url);
    else window.history.pushState(state, "", url);
  }
  const section = movementId ? findSection(entry, movementId) : null;
  document.title = section
    ? `${entry.displayTitle || entry.title} — ${section.title}`
    : entry.displayTitle || entry.title || pieceQueryId(entry);
}

function setCatalogLocation({ replace = false } = {}) {
  activeEntryId = null;
  activeMovementId = null;
  const url = window.location.pathname || "./";
  if (replace) window.history.replaceState({ catalog: true }, "", url);
  else window.history.pushState({ catalog: true }, "", url);
  document.title = "Performance reader";
}

function syncLocationFromScore({ replace = true } = {}) {
  if (!activeEntryId || !score) return;
  const entry = findEntry(activeEntryId);
  if (!entry) return;
  setPieceLocation(entry, {
    movementId: activeMovementId,
    anchorMeasure: score.getAnchor(),
    replace,
  });
}

function renderCatalog() {
  catalogList.innerHTML = "";
  if (!catalogRows.length) {
    catalogEmpty.hidden = false;
    catalogEmpty.textContent =
      "No catalog metadata yet. On the desktop run: baroque-transcribe reader publish --collection bwv-806-811-english-suites";
    return;
  }
  catalogEmpty.hidden = true;
  let currentCollection = null;
  for (const entry of catalogRows) {
    if (entry.collection && entry.collection !== currentCollection) {
      currentCollection = entry.collection;
      const heading = document.createElement("li");
      heading.className = "collection-heading";
      heading.textContent = currentCollection;
      catalogList.appendChild(heading);
    }
    const li = document.createElement("li");
    li.className = entry.published ? "published" : "unpublished";
    const open = document.createElement(entry.published ? "a" : "div");
    open.className = "open-piece";
    if (entry.published) {
      const firstMov = (entry.sections || [])[0];
      open.href = pieceUrl(entry, { movementId: firstMov ? firstMov.id : null });
      open.addEventListener("click", (event) => {
        event.preventDefault();
        openEntry(entry, {
          movementId: firstMov ? firstMov.id : null,
          restoreAnchor: true,
          updateHistory: true,
        });
      });
    }
    open.innerHTML = `<div class="bwv">BWV ${escapeHtml(entry.bwv || "")}</div>
      <div>${escapeHtml(entry.title || entry.displayTitle || entry.entryId)}</div>`;
    const status = document.createElement("span");
    status.className = "status";
    if (entry.published) {
      const n = (entry.sections || []).length;
      status.textContent = n > 1 ? `Published · ${n} sections` : "Published";
    } else {
      status.textContent = "Catalog only — not published offline yet";
    }
    li.append(open, status);
    if (entry.published && (entry.sections || []).length > 1) {
      const sections = document.createElement("ul");
      sections.className = "section-list";
      for (const section of entry.sections) {
        const item = document.createElement("li");
        const link = document.createElement("a");
        link.href = pieceUrl(entry, { movementId: section.id });
        link.textContent = section.title || section.id;
        link.addEventListener("click", (event) => {
          event.preventDefault();
          openEntry(entry, { movementId: section.id, updateHistory: true });
        });
        item.appendChild(link);
        sections.appendChild(item);
      }
      li.appendChild(sections);
    }
    catalogList.appendChild(li);
  }
}

async function openEntry(
  entry,
  {
    movementId = null,
    restoreAnchor = false,
    anchorMeasure = null,
    updateHistory = true,
    replaceHistory = false,
  } = {}
) {
  showError("");
  if (!entry.published) {
    showError("This catalog entry is not published to the reader yet.");
    return;
  }
  try {
    let pkg;
    let resolvedMovement = movementId;
    if ((entry.sections || []).length) {
      if (!resolvedMovement) resolvedMovement = entry.sections[0].id;
      const section = findSection(entry, resolvedMovement);
      if (!section) throw new Error(`Unknown section: ${resolvedMovement}`);
      pkg = await loadPackage({
        entryId: entry.entryId,
        movementId: section.id,
        packagePath: section.packagePath,
      });
    } else {
      pkg = await loadPackage(entry);
      resolvedMovement = null;
    }
    const xml = pkg.musicxmlVoices || pkg.musicxml;
    if (!xml) throw new Error("Package has no MusicXML. Open a section of this suite.");
    const pieceId = pkg.id || entry.id || `catalog:${entry.entryId}`;
    activeEntryId = pieceQueryId(entry);
    activeMovementId = resolvedMovement;
    document.body.classList.add("score-open");
    btnBack.hidden = false;
    pieceTitle.textContent = pkg.displayTitle || pkg.title || entry.title || entry.entryId;
    ensureScore().setPieceId(pieceId);
    let anchor = anchorMeasure;
    if (anchor == null && restoreAnchor) {
      const saved = loadReadingPosition();
      if (saved && saved.pieceId === pieceId) anchor = saved.anchorMeasure;
    }
    await ensureScore().loadMusicXml(xml, { anchorMeasure: anchor || 1 });
    saveReadingPosition(pieceId, ensureScore().getAnchor());
    if (updateHistory) {
      setPieceLocation(entry, {
        movementId: resolvedMovement,
        anchorMeasure: ensureScore().getAnchor(),
        replace: replaceHistory,
      });
    } else {
      document.title = pkg.displayTitle || pkg.title || pieceQueryId(entry);
    }
  } catch (error) {
    showError(error.message || String(error));
  }
}

function showCatalog({ updateHistory = true } = {}) {
  if (score) score.persist();
  document.body.classList.remove("score-open");
  btnBack.hidden = true;
  pieceTitle.textContent = "Performance reader";
  systemLabel.textContent = "Bach keyboard catalog";
  if (updateHistory) setCatalogLocation();
  else {
    activeEntryId = null;
    activeMovementId = null;
    document.title = "Performance reader";
  }
}

function ensureScore() {
  if (score) return score;
  score = createScoreController({
    scoreElement: document.getElementById("score"),
    paneElement: document.getElementById("score-pane"),
    getOverlap: () => settings.overlap,
    getZoom: () => settings.zoom,
    pieceId: null,
    onSystemsChanged(systems, index) {
      const total = systems.length;
      systemLabel.textContent = total ? `System ${index + 1} / ${total}` : "System —";
      btnPrev.disabled = index <= 0;
      btnNext.disabled = index >= total - 1;
      syncLocationFromScore({ replace: true });
    },
  });
  return score;
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function wireControls() {
  btnPrev.addEventListener("click", () => {
    ensureScore().previousSystem();
    syncLocationFromScore({ replace: true });
  });
  btnNext.addEventListener("click", () => {
    ensureScore().nextSystem();
    syncLocationFromScore({ replace: true });
  });
  btnBack.addEventListener("click", () => showCatalog({ updateHistory: true }));
  btnPerformance.addEventListener("click", () => {
    settings = saveSettings({ performanceMode: !settings.performanceMode });
    applyChrome();
  });
  zoomInput.addEventListener("input", () => {
    settings = saveSettings({ zoom: Number(zoomInput.value) });
    if (document.body.classList.contains("score-open")) {
      ensureScore().applyZoom(settings.zoom);
      syncLocationFromScore({ replace: true });
    }
  });
  overlapInput.addEventListener("input", () => {
    settings = saveSettings({ overlap: Number(overlapInput.value) });
    if (document.body.classList.contains("score-open") && score) {
      score.goToSystem(score.getSystemIndex());
      syncLocationFromScore({ replace: true });
    }
  });
  window.addEventListener("keydown", (event) => {
    if (!document.body.classList.contains("score-open")) return;
    if (event.key === "ArrowRight" || event.key === "PageDown") {
      event.preventDefault();
      ensureScore().nextSystem();
      syncLocationFromScore({ replace: true });
    } else if (event.key === "ArrowLeft" || event.key === "PageUp") {
      event.preventDefault();
      ensureScore().previousSystem();
      syncLocationFromScore({ replace: true });
    } else if (event.key === "Escape" && settings.performanceMode) {
      settings = saveSettings({ performanceMode: false });
      applyChrome();
    }
  });
  window.addEventListener("resize", () => {
    if (score) score.handleResize();
  });
  window.addEventListener("beforeunload", () => {
    if (score) score.persist();
  });
  window.addEventListener("popstate", async () => {
    const params = new URLSearchParams(window.location.search);
    const wantId = params.get("id");
    if (!wantId) {
      showCatalog({ updateHistory: false });
      return;
    }
    const entry = findEntry(wantId);
    if (!entry) {
      showError(`Unknown piece id: ${wantId}`);
      return;
    }
    const measure = Number(params.get("m"));
    await openEntry(entry, {
      movementId: params.get("mov"),
      restoreAnchor: false,
      anchorMeasure: Number.isFinite(measure) && measure > 0 ? measure : 1,
      updateHistory: false,
    });
  });
}

async function boot() {
  applyChrome();
  wireControls();
  try {
    const [published, library] = await Promise.all([loadCatalogIndex("."), loadCatalogLibrary(".")]);
    catalogRows = mergeCatalogViews(library, published);
    renderCatalog();
  } catch (error) {
    showError(error.message || String(error));
    catalogEmpty.hidden = false;
  }

  const params = new URLSearchParams(window.location.search);
  const wantId = params.get("id");
  const wantMov = params.get("mov");
  const wantMeasure = Number(params.get("m"));
  if (wantId) {
    const entry = findEntry(wantId);
    if (entry) {
      await openEntry(entry, {
        movementId: wantMov,
        restoreAnchor: !(Number.isFinite(wantMeasure) && wantMeasure > 0),
        anchorMeasure: Number.isFinite(wantMeasure) && wantMeasure > 0 ? wantMeasure : null,
        updateHistory: true,
        replaceHistory: true,
      });
    } else {
      showError(`Unknown piece id: ${wantId}`);
    }
  }
}

boot();
