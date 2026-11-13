import {
  measureBoxesFromOsmd,
  scrollTopForSystem,
  systemIndexForMeasure,
  systemsFromBoxes,
} from "./system-nav.js";
import { saveReadingPosition } from "./position.js";

export function createScoreController({
  scoreElement,
  paneElement,
  getOverlap,
  getZoom,
  onSystemsChanged,
  pieceId,
}) {
  let osmd = null;
  let systems = [];
  let systemIndex = 0;
  let currentPieceId = pieceId || null;
  let resizeTimer = null;

  async function loadMusicXml(xml, { anchorMeasure = 1 } = {}) {
    const library = window.opensheetmusicdisplay;
    if (!library || !library.OpenSheetMusicDisplay) {
      throw new Error("OpenSheetMusicDisplay failed to load.");
    }
    scoreElement.innerHTML = "";
    osmd = new library.OpenSheetMusicDisplay(scoreElement, {
      autoResize: true,
      backend: "svg",
      drawTitle: false,
      drawComposer: false,
      drawPartNames: true,
      drawMeasureNumbers: true,
      measureNumberInterval: 1,
    });
    osmd.Zoom = getZoom();
    await osmd.load(xml);
    osmd.render();
    rebuildSystems();
    goToMeasure(anchorMeasure, { save: false });
  }

  function rebuildSystems() {
    const boxes = measureBoxesFromOsmd(osmd);
    systems = systemsFromBoxes(boxes);
    if (systemIndex >= systems.length) {
      systemIndex = Math.max(0, systems.length - 1);
    }
    if (onSystemsChanged) onSystemsChanged(systems, systemIndex);
  }

  function currentAnchor() {
    const system = systems[systemIndex];
    return system ? system.anchorMeasure : 1;
  }

  function persist() {
    if (!currentPieceId) return;
    saveReadingPosition(currentPieceId, currentAnchor());
  }

  function goToSystem(index, { save = true } = {}) {
    if (!systems.length || !paneElement) return;
    systemIndex = Math.max(0, Math.min(systems.length - 1, index));
    paneElement.scrollTop = scrollTopForSystem(systems, systemIndex, getOverlap(), 0);
    if (onSystemsChanged) onSystemsChanged(systems, systemIndex);
    if (save) persist();
  }

  function goToMeasure(measureNumber, options) {
    goToSystem(systemIndexForMeasure(systems, measureNumber), options);
  }

  function nextSystem() {
    goToSystem(systemIndex + 1);
  }

  function previousSystem() {
    goToSystem(systemIndex - 1);
  }

  function setPieceId(id) {
    currentPieceId = id;
  }

  function applyZoom(zoom) {
    if (!osmd) return;
    const anchor = currentAnchor();
    osmd.Zoom = zoom;
    osmd.render();
    rebuildSystems();
    goToMeasure(anchor, { save: true });
  }

  function handleResize() {
    if (!osmd) return;
    window.clearTimeout(resizeTimer);
    resizeTimer = window.setTimeout(() => {
      const anchor = currentAnchor();
      osmd.render();
      rebuildSystems();
      goToMeasure(anchor, { save: true });
    }, 120);
  }

  return {
    loadMusicXml,
    rebuildSystems,
    goToSystem,
    goToMeasure,
    nextSystem,
    previousSystem,
    setPieceId,
    applyZoom,
    handleResize,
    persist,
    getSystemIndex: () => systemIndex,
    getSystems: () => systems,
    getAnchor: currentAnchor,
  };
}
