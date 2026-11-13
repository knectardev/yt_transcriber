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
    const open = document.createElement("button");
    open.type = "button";
    open.className = "open-piece";
    open.innerHTML = `<div class="bwv">BWV ${escapeHtml(entry.bwv || "")}</div>
      <div>${escapeHtml(entry.title || entry.displayTitle || entry.entryId)}</div>`;
    open.addEventListener("click", () => openEntry(entry, { restoreAnchor: true }));
    const status = document.createElement("span");
    status.className = "status";
    status.textContent = "Available from hosted catalog";
    li.append(open, status);
    catalogList.appendChild(li);
  }
}

async function openEntry(entry, { restoreAnchor = false, anchorMeasure = null } = {}) {
  showError("");
  try {
    const pkg = await loadPackage(entry);
    const xml = pkg.musicxmlVoices || pkg.musicxml;
    if (!xml) throw new Error("Package has no MusicXML.");
    const pieceId = pkg.id || entry.id || `catalog:${entry.entryId}`;
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
  } catch (error) {
    showError(error.message || String(error));
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
  btnPrev.addEventListener("click", () => ensureScore().previousSystem());
  btnNext.addEventListener("click", () => ensureScore().nextSystem());
  btnBack.addEventListener("click", () => {
    if (score) score.persist();
    document.body.classList.remove("score-open");
    btnBack.hidden = true;
    pieceTitle.textContent = "Performance reader";
    systemLabel.textContent = "Published Bach scores";
  });
  btnPerformance.addEventListener("click", () => {
    settings = saveSettings({ performanceMode: !settings.performanceMode });
    applyChrome();
  });
  zoomInput.addEventListener("input", () => {
    settings = saveSettings({ zoom: Number(zoomInput.value) });
    if (document.body.classList.contains("score-open")) {
      ensureScore().applyZoom(settings.zoom);
    }
  });
  overlapInput.addEventListener("input", () => {
    settings = saveSettings({ overlap: Number(overlapInput.value) });
    if (document.body.classList.contains("score-open") && score) {
      score.goToSystem(score.getSystemIndex());
    }
  });
  window.addEventListener("keydown", (event) => {
    if (!document.body.classList.contains("score-open")) return;
    if (event.key === "ArrowRight" || event.key === "PageDown") {
      event.preventDefault();
      ensureScore().nextSystem();
    } else if (event.key === "ArrowLeft" || event.key === "PageUp") {
      event.preventDefault();
      ensureScore().previousSystem();
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
  const saved = loadReadingPosition();
  const targetId = wantId || (saved && saved.pieceId);
  if (targetId) {
    const entry = catalog.entries.find(
      (item) => item.id === targetId || item.entryId === targetId || `catalog:${item.entryId}` === targetId
    );
    if (entry) {
      await openEntry(entry, {
        restoreAnchor: !wantId || (saved && saved.pieceId === (entry.id || `catalog:${entry.entryId}`)),
      });
    }
  }
}

boot();
