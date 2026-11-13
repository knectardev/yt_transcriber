/**
 * Musical-system navigation helpers.
 * measureBoxes shaped like playback/notation OSMD overlays:
 * { measure, systemKey, minY, maxY, pageEl?, scale? }
 */

export function systemsFromBoxes(boxes) {
  const order = [];
  const byKey = new Map();
  for (const box of boxes || []) {
    if (!box || !box.systemKey) continue;
    let system = byKey.get(box.systemKey);
    if (!system) {
      system = {
        key: box.systemKey,
        minY: box.minY,
        maxY: box.maxY,
        measures: [],
        pageEl: box.pageEl || null,
        scale: box.scale || 10,
      };
      byKey.set(box.systemKey, system);
      order.push(system);
    } else {
      system.minY = Math.min(system.minY, box.minY);
      system.maxY = Math.max(system.maxY, box.maxY);
      if (box.scale) system.scale = box.scale;
      if (box.pageEl) system.pageEl = box.pageEl;
    }
    if (!system.measures.includes(box.measure)) {
      system.measures.push(box.measure);
    }
  }
  for (const system of order) {
    system.measures.sort((a, b) => a - b);
    system.anchorMeasure = system.measures[0] || 1;
  }
  return order;
}

export function systemIndexForMeasure(systems, measureNumber) {
  const target = Number(measureNumber) || 1;
  let best = 0;
  for (let index = 0; index < systems.length; index += 1) {
    const measures = systems[index].measures || [];
    if (measures.includes(target)) return index;
    if ((systems[index].anchorMeasure || 0) <= target) best = index;
  }
  return best;
}

/**
 * Pixel scrollTop so system `index` sits near the top of the pane,
 * keeping `overlap` of the previous system's height still visible.
 */
export function scrollTopForSystem(systems, index, overlap, paneTopInPage) {
  if (!systems.length) return 0;
  const clamped = Math.max(0, Math.min(systems.length - 1, index));
  const system = systems[clamped];
  const scale = system.scale || 10;
  const topPx = system.minY * scale;
  let overlapPx = 0;
  if (clamped > 0 && overlap > 0) {
    const prev = systems[clamped - 1];
    const prevHeight = Math.max(0, (prev.maxY - prev.minY) * (prev.scale || scale));
    overlapPx = prevHeight * overlap;
  }
  const pageOffset = typeof paneTopInPage === "number" ? paneTopInPage : 0;
  return Math.max(0, topPx - overlapPx - pageOffset);
}

export function measureBoxesFromOsmd(osmd) {
  const boxes = [];
  const systems = [];
  const list = (osmd && osmd.GraphicSheet && osmd.GraphicSheet.MeasureList) || [];
  const scale = 10 * ((osmd && osmd.Zoom) || 1);
  for (let index = 0; index < list.length; index += 1) {
    const staffs = list[index] || [];
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    let pageNumber = null;
    let systemKey = "";
    let measureNumber = index + 1;
    for (let staff = 0; staff < staffs.length; staff += 1) {
      const measure = staffs[staff];
      if (!measure || !measure.PositionAndShape) continue;
      const source = measure.parentSourceMeasure;
      if (source && source.MeasureNumber) measureNumber = source.MeasureNumber;
      const box = measure.PositionAndShape;
      const x = box.AbsolutePosition.x + box.BorderLeft;
      const y = box.AbsolutePosition.y + box.BorderTop;
      const right = x + box.Size.width;
      const bottom = y + box.Size.height;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (right > maxX) maxX = right;
      if (bottom > maxY) maxY = bottom;
      const system = measure.ParentStaffLine && measure.ParentStaffLine.ParentMusicSystem;
      const page = system && system.Parent;
      if (page && page.PageNumber) pageNumber = page.PageNumber;
      if (system) {
        let systemIndex = systems.indexOf(system);
        if (systemIndex === -1) {
          systems.push(system);
          systemIndex = systems.length - 1;
        }
        systemKey = (pageNumber || 1) + ":" + systemIndex;
      }
    }
    if (!isFinite(minX) || maxX <= minX || maxY <= minY) continue;
    const pageEl =
      (typeof document !== "undefined" &&
        (document.getElementById("osmdCanvasPage" + pageNumber) ||
          document.getElementById("score"))) ||
      null;
    boxes.push({
      measure: measureNumber,
      pageEl,
      systemKey,
      minX,
      maxX,
      minY,
      maxY,
      scale,
    });
  }
  return boxes;
}
