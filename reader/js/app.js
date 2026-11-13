import { loadCatalogIndex, loadPackage } from "./catalog.js";
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
let catalog = { entries: [] };
let score = null;
let activeEntryId = null;

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
    catalog.entries.find(
      (item) =>
        item.id === needle ||
        item.entryId === needle ||
        `catalog:${item.entryId}` === needle ||
        pieceQueryId(item) === needle.replace(/^catalog:/, "")
    ) || null
  );
}

function pieceUrl(entry, anchorMeasure) {
  const params = new URLSearchParams();
  params.set("id", pieceQueryId(entry));
  const measure = Number(anchorMeasure);
  if (Number.isFinite(measure) && measure > 1) {
    params.set("m", String(Math.floor(measure)));
  }
  return `?${params.toString()}`;
}

function setPieceLocation(entry, anchorMeasure, { replace = false } = {}) {
  if (!entry) return;
  const url = pieceUrl(entry, anchorMeasure);
  const state = {
    entryId: pieceQueryId(entry),
    anchorMeasure: Number(anchorMeasure) || 1,
  };
  if (replace) {
    window.history.replaceState(state, "", url);
  } else {
    const current = `${window.location.pathname}${window.location.search}`;
    const next = `${window.location.pathname}${url}`;
    if (current === next) {
      window.history.replaceState(state, "", url);
    } else {
      window.history.pushState(state, "", url);
    }
  }
  document.title = entry.displayTitle || entry.title || pieceQueryId(entry);
}

function setCatalogLocation({ replace = false } = {}) {
  activeEntryId = null;
  const url = window.location.pathname || "./";
  if (replace) {
    window.history.replaceState({ catalog: true }, "", url);
  } else {
    window.history.pushState({ catalog: true }, "", url);
  }
  document.title = "Performance reader";
}

function syncLocationFromScore({ replace = true } = {}) {
  if (!activeEntryId || !score) return;
  const entry = findEntry(activeEntryId);
  if (!entry) return;
  setPieceLocation(entry, score.getAnchor(), { replace });
}

function renderCatalog() {
  catalogList.innerHTML = "";
  if (!catalog.entries.length) {
    catalogEmpty.hidden = false;
    catalogEmpty.textContent =
      "No scores published yet. On the desktop run: baroque-transcribe reader publish --id bwv-0772";
    return;
  }
  catalogEmpty.hidden = true;
  for (const entry of catalog.entries) {
    const li = document.createElement("li");
    const open = document.createElement("a");
    open.className = "open-piece";
    open.href = pieceUrl(entry);
    open.innerHTML = `<div class="bwv">BWV ${escapeHtml(entry.bwv || "")}</div>
      <div>${escapeHtml(entry.title || entry.displayTitle || entry.entryId)}</div>`;
    open.addEventListener("click", (event) => {
      event.preventDefault();
      openEntry(entry, { restoreAnchor: true, updateHistory: true });
    });
    const status = document.createElement("span");
    status.className = "status";
    status.textContent = "Available from hosted catalog";
    li.append(open, status);
    catalogList.appendChild(li);
  }
}

async function openEntry(
  entry,
  { restoreAnchor = false, anchorMeasure = null, updateHistory = true, replaceHistory = false } = {}
) {
  showError("");
  try {
    const pkg = await loadPackage(entry);
    const xml = pkg.musicxmlVoices || pkg.musicxml;
    if (!xml) throw new Error("Package has no MusicXML.");
    const pieceId = pkg.id || entry.id || `catalog:${entry.entryId}`;
    activeEntryId = pieceQueryId(entry);
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
      setPieceLocation(entry, ensureScore().getAnchor(), { replace: replaceHistory });
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
  systemLabel.textContent = "Published Bach scores";
  if (updateHistory) setCatalogLocation();
  else {
    activeEntryId = null;
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
    catalog = await loadCatalogIndex(".");
    renderCatalog();
  } catch (error) {
    showError(error.message || String(error));
    catalogEmpty.hidden = false;
  }

  const params = new URLSearchParams(window.location.search);
  const wantId = params.get("id");
  const wantMeasure = Number(params.get("m"));
  if (wantId) {
    const entry = findEntry(wantId);
    if (entry) {
      await openEntry(entry, {
        restoreAnchor: !(Number.isFinite(wantMeasure) && wantMeasure > 0),
        anchorMeasure: Number.isFinite(wantMeasure) && wantMeasure > 0 ? wantMeasure : null,
        updateHistory: true,
        replaceHistory: true,
      });
      return;
    }
    showError(`Unknown piece id: ${wantId}`);
    return;
  }

  // Deep link wins over local restore; without ?id, stay on catalog so shared
  // root URL remains a browse page. Local position still restores after open.
}

boot();
