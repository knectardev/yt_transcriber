import { loadCatalogIndex, loadCatalogLibrary, loadPackage, mergeCatalogViews } from "./catalog.js";
import { createGestureController } from "./gestures.js";
import { loadReadingPosition, saveReadingPosition } from "./position.js";
import { createScoreController } from "./score.js";
import { loadSettings, saveSettings } from "./settings.js";

const catalogList = document.getElementById("catalog-list");
const catalogEmpty = document.getElementById("catalog-empty");
const errorBanner = document.getElementById("error-banner");
const pieceTitle = document.getElementById("piece-title");
const sectionLabel = document.getElementById("section-label");
const systemLabel = document.getElementById("system-label");
const zoomInput = document.getElementById("zoom");
const overlapInput = document.getElementById("overlap");
const colorToggle = document.getElementById("color-toggle");
const colorVoices = document.getElementById("color-voices");
const btnPrev = document.getElementById("btn-prev");
const btnNext = document.getElementById("btn-next");
const btnPrevSection = document.getElementById("btn-prev-section");
const btnNextSection = document.getElementById("btn-next-section");
const sectionPickerWrap = document.getElementById("section-picker-wrap");
const sectionPicker = document.getElementById("section-picker");
const btnCatalog = document.getElementById("btn-catalog");
const btnPerformance = document.getElementById("btn-performance");
const btnCameraPanel = document.getElementById("btn-camera-panel");
const cameraPanel = document.getElementById("camera-panel");
const cameraSelect = document.getElementById("camera-select");
const btnCameraStart = document.getElementById("btn-camera-start");
const btnCameraStop = document.getElementById("btn-camera-stop");
const btnCalibrate = document.getElementById("btn-calibrate");
const gestureAxis = document.getElementById("gesture-axis");
const gestureDirection = document.getElementById("gesture-direction");
const gestureSensitivity = document.getElementById("gesture-sensitivity");
const gestureHold = document.getElementById("gesture-hold");
const gestureSensitivityValue = document.getElementById("gesture-sensitivity-value");
const gestureHoldValue = document.getElementById("gesture-hold-value");
const cameraStatus = document.getElementById("camera-status");
const cameraPreview = document.getElementById("camera-preview");
const trackingPill = document.getElementById("tracking-pill");

function setCameraStatus(message, tone = "") {
  if (!cameraStatus) return;
  cameraStatus.textContent = message || "";
  cameraStatus.classList.toggle("is-warning", tone === "warning");
  cameraStatus.classList.toggle("is-error", tone === "error");
}

function syncGestureReadouts() {
  if (gestureSensitivityValue && gestureSensitivity) {
    gestureSensitivityValue.textContent = `${gestureSensitivity.value}°`;
  }
  if (gestureHoldValue && gestureHold) {
    gestureHoldValue.textContent = `${gestureHold.value} ms`;
  }
}

function setCameraPanelOpen(open) {
  if (!cameraPanel || !btnCameraPanel) return;
  cameraPanel.hidden = !open;
  btnCameraPanel.setAttribute("aria-expanded", open ? "true" : "false");
}

let settings = loadSettings();
let catalogRows = [];
let score = null;
let activeEntryId = null;
let activeMovementId = null;
let activePackage = null;
let gestures = null;

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
  if (colorVoices) colorVoices.checked = Boolean(settings.colorVoices);
  if (gestureAxis) gestureAxis.value = settings.gestureAxis;
  if (gestureDirection) gestureDirection.value = settings.gestureDirection;
  if (gestureSensitivity) gestureSensitivity.value = String(settings.gestureSensitivity);
  if (gestureHold) gestureHold.value = String(settings.gestureHoldMs);
  syncGestureReadouts();
  if (settings.performanceMode) setCameraPanelOpen(false);
}

function applyGestureOptions() {
  if (!gestures) return;
  gestures.configure({
    axis: settings.gestureAxis,
    nextOnPositive: settings.gestureDirection !== "pos-prev",
    sensitivityDeg: settings.gestureSensitivity,
    minDurationMs: settings.gestureHoldMs,
  });
}

function ensureGestures() {
  if (gestures) return gestures;
  gestures = createGestureController({
    onFire(direction) {
      if (!document.body.classList.contains("score-open") || !score) return;
      if (direction === "next") score.nextSystem();
      else score.previousSystem();
      syncLocationFromScore({ replace: true });
      setCameraStatus(
        direction === "next" ? "Turned to next system." : "Turned to previous system.",
      );
    },
    onStatus(status) {
      if (trackingPill) {
        trackingPill.hidden = false;
        if (!status.running) {
          trackingPill.textContent = "Camera off";
          trackingPill.className = "tracking-pill setup-only";
          trackingPill.title = "Open camera setup";
        } else if (!status.calibrated) {
          trackingPill.textContent = "Needs calibrate";
          trackingPill.className = "tracking-pill setup-only needs-calibrate";
          trackingPill.title = "Open camera setup and set neutral pose";
        } else if (status.tracking) {
          trackingPill.textContent = "Tracking";
          trackingPill.className = "tracking-pill setup-only on";
          trackingPill.title = "Head tracking active";
        } else {
          trackingPill.textContent = "Face lost";
          trackingPill.className = "tracking-pill setup-only lost";
          trackingPill.title = "Keep your face in the camera preview";
        }
      }
      if (status.error) setCameraStatus(status.error, "error");
      else if (!status.running) {
        setCameraStatus("Camera off. Start the camera, then calibrate your neutral sitting pose.");
      } else if (!status.calibrated) {
        setCameraStatus("Hold still in your playing posture, then press Set neutral pose.", "warning");
      } else if (status.tracking) {
        setCameraStatus("Tracking. Make a deliberate tilt/turn, then return to neutral.");
      } else {
        setCameraStatus("Looking for your face…", "warning");
      }
      if (btnCameraStart) btnCameraStart.disabled = Boolean(status.running);
      if (btnCameraStop) btnCameraStop.disabled = !status.running;
      if (btnCalibrate) btnCalibrate.disabled = !status.running;
    },
  });
  applyGestureOptions();
  return gestures;
}

async function refreshCameraList() {
  if (!cameraSelect) return;
  try {
    const devices = await ensureGestures().listCameras();
    cameraSelect.innerHTML = "";
    if (!devices.length) {
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "No camera found";
      cameraSelect.appendChild(option);
      return;
    }
    for (const device of devices) {
      const option = document.createElement("option");
      option.value = device.deviceId;
      option.textContent = device.label || `Camera ${cameraSelect.options.length + 1}`;
      if (device.deviceId === settings.cameraDeviceId) option.selected = true;
      cameraSelect.appendChild(option);
    }
  } catch (_error) {
    cameraSelect.innerHTML = "<option value=\"\">Camera list unavailable</option>";
  }
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
  activePackage = null;
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

function selectedXml(pkg) {
  if (!pkg) return "";
  if (settings.colorVoices && pkg.musicxmlVoices) return pkg.musicxmlVoices;
  return pkg.musicxml || pkg.musicxmlVoices || "";
}

function updateSectionChrome(entry, movementId) {
  const sections = entry && Array.isArray(entry.sections) ? entry.sections : [];
  const hasSections = sections.length > 1;
  const section = movementId ? findSection(entry, movementId) : null;
  const sectionIndex = section ? sections.findIndex((row) => row.id === section.id) : -1;

  if (sectionLabel) {
    if (hasSections && section) {
      sectionLabel.hidden = false;
      sectionLabel.textContent = `${section.title} · section ${sectionIndex + 1} of ${sections.length}`;
    } else if (entry && /suite/i.test(entry.title || entry.displayTitle || "")) {
      sectionLabel.hidden = false;
      sectionLabel.textContent = `${entry.collection || "Suite"} · complete work (no section split published)`;
    } else {
      sectionLabel.hidden = true;
      sectionLabel.textContent = "";
    }
  }

  if (pieceTitle && entry) {
    pieceTitle.textContent = entry.displayTitle || entry.title || entry.entryId;
  }

  if (btnPrevSection && btnNextSection && sectionPickerWrap && sectionPicker) {
    btnPrevSection.hidden = !hasSections;
    btnNextSection.hidden = !hasSections;
    sectionPickerWrap.hidden = !hasSections;
    if (hasSections) {
      btnPrevSection.disabled = sectionIndex <= 0;
      btnNextSection.disabled = sectionIndex < 0 || sectionIndex >= sections.length - 1;
      sectionPicker.innerHTML = "";
      for (const row of sections) {
        const option = document.createElement("option");
        option.value = row.id;
        option.textContent = row.title || row.id;
        if (row.id === movementId) option.selected = true;
        sectionPicker.appendChild(option);
      }
    }
  }
}

function updateColorToggle(pkg) {
  // Show whenever a colored engraving exists, even if plain MusicXML is absent.
  const hasVoices = Boolean(pkg && pkg.musicxmlVoices);
  const hasPlain = Boolean(pkg && pkg.musicxml);
  const canToggle = hasVoices && hasPlain;
  if (colorToggle) colorToggle.hidden = !canToggle;
  if (colorVoices) {
    colorVoices.disabled = !canToggle;
    colorVoices.checked = Boolean(settings.colorVoices);
  }
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
    const xml = selectedXml(pkg);
    if (!xml) throw new Error("Package has no MusicXML. Open a section of this suite.");
    const pieceId = pkg.id || entry.id || `catalog:${entry.entryId}`;
    activeEntryId = pieceQueryId(entry);
    activeMovementId = resolvedMovement;
    activePackage = pkg;
    document.body.classList.add("score-open");
    if (btnCatalog) btnCatalog.hidden = false;
    updateSectionChrome(entry, resolvedMovement);
    updateColorToggle(pkg);
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

async function reloadWithColorPreference() {
  if (!activePackage || !score) return;
  const xml = selectedXml(activePackage);
  if (!xml) return;
  const anchor = score.getAnchor();
  await score.loadMusicXml(xml, { anchorMeasure: anchor || 1 });
}

function showCatalog({ updateHistory = true } = {}) {
  if (score) score.persist();
  document.body.classList.remove("score-open");
  if (btnCatalog) btnCatalog.hidden = true;
  pieceTitle.textContent = "Performance reader";
  systemLabel.textContent = "Bach keyboard catalog";
  if (sectionLabel) {
    sectionLabel.hidden = true;
    sectionLabel.textContent = "";
  }
  updateSectionChrome(null, null);
  updateColorToggle(null);
  if (updateHistory) setCatalogLocation();
  else {
    activeEntryId = null;
    activeMovementId = null;
    activePackage = null;
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

function currentSections() {
  const entry = findEntry(activeEntryId);
  return entry && Array.isArray(entry.sections) ? entry.sections : [];
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
  btnPrevSection.addEventListener("click", () => {
    const entry = findEntry(activeEntryId);
    const sections = currentSections();
    const index = sections.findIndex((row) => row.id === activeMovementId);
    if (!entry || index <= 0) return;
    openEntry(entry, { movementId: sections[index - 1].id, updateHistory: true });
  });
  btnNextSection.addEventListener("click", () => {
    const entry = findEntry(activeEntryId);
    const sections = currentSections();
    const index = sections.findIndex((row) => row.id === activeMovementId);
    if (!entry || index < 0 || index >= sections.length - 1) return;
    openEntry(entry, { movementId: sections[index + 1].id, updateHistory: true });
  });
  sectionPicker.addEventListener("change", () => {
    const entry = findEntry(activeEntryId);
    if (!entry || !sectionPicker.value) return;
    openEntry(entry, { movementId: sectionPicker.value, updateHistory: true });
  });
  if (btnCatalog) {
    btnCatalog.addEventListener("click", (event) => {
      event.preventDefault();
      showCatalog({ updateHistory: true });
    });
  }
  btnPerformance.addEventListener("click", () => {
    settings = saveSettings({ performanceMode: !settings.performanceMode });
    applyChrome();
  });
  if (colorVoices) {
    colorVoices.addEventListener("change", async () => {
      settings = saveSettings({ colorVoices: colorVoices.checked });
      await reloadWithColorPreference();
    });
  }
  if (btnCameraPanel && cameraPanel) {
    btnCameraPanel.addEventListener("click", async () => {
      const open = cameraPanel.hidden;
      setCameraPanelOpen(open);
      if (open) await refreshCameraList();
    });
  }
  if (trackingPill) {
    trackingPill.addEventListener("click", async () => {
      setCameraPanelOpen(true);
      await refreshCameraList();
      if (gestures && !btnCalibrate?.disabled) {
        // Focus the calibrate action when setup is incomplete.
        btnCalibrate?.focus();
      }
    });
  }
  if (btnCameraStart) {
    btnCameraStart.addEventListener("click", async () => {
      try {
        const controller = ensureGestures();
        applyGestureOptions();
        const preferred = (cameraSelect && cameraSelect.value) || settings.cameraDeviceId || "";
        await controller.start({ videoElement: cameraPreview, preferredDeviceId: preferred });
        settings = saveSettings({ cameraDeviceId: preferred });
        await refreshCameraList();
      } catch (error) {
        setCameraStatus(error.message || String(error), "error");
      }
    });
  }
  if (btnCameraStop) {
    btnCameraStop.addEventListener("click", () => {
      if (gestures) gestures.stop();
      setCameraStatus("Camera off. Start the camera, then calibrate your neutral sitting pose.");
    });
  }
  if (btnCalibrate) {
    btnCalibrate.addEventListener("click", () => {
      if (gestures) gestures.calibrate();
      setCameraStatus("Neutral pose saved. Try a deliberate tilt, then return to center.");
    });
  }
  if (gestureAxis) {
    gestureAxis.addEventListener("change", () => {
      settings = saveSettings({ gestureAxis: gestureAxis.value });
      applyGestureOptions();
    });
  }
  if (gestureDirection) {
    gestureDirection.addEventListener("change", () => {
      settings = saveSettings({ gestureDirection: gestureDirection.value });
      applyGestureOptions();
    });
  }
  if (gestureSensitivity) {
    gestureSensitivity.addEventListener("input", () => {
      settings = saveSettings({ gestureSensitivity: Number(gestureSensitivity.value) });
      syncGestureReadouts();
      applyGestureOptions();
    });
  }
  if (gestureHold) {
    gestureHold.addEventListener("input", () => {
      settings = saveSettings({ gestureHoldMs: Number(gestureHold.value) });
      syncGestureReadouts();
      applyGestureOptions();
    });
  }
  if (cameraSelect) {
    cameraSelect.addEventListener("change", () => {
      settings = saveSettings({ cameraDeviceId: cameraSelect.value });
    });
  }
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
    if (gestures) gestures.stop();
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
