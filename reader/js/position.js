/** Persist and restore the current piece + anchor measure. */

export const POSITION_KEY = "baroque-performance.reading-position";

export function loadReadingPosition() {
  try {
    const raw = window.localStorage.getItem(POSITION_KEY);
    if (!raw) return null;
    const data = JSON.parse(raw);
    if (!data || typeof data.pieceId !== "string") return null;
    const anchor = Number(data.anchorMeasure);
    return {
      pieceId: data.pieceId,
      anchorMeasure: Number.isFinite(anchor) && anchor > 0 ? anchor : 1,
      updatedAt: data.updatedAt || null,
    };
  } catch (_error) {
    return null;
  }
}

export function saveReadingPosition(pieceId, anchorMeasure) {
  if (!pieceId) return;
  const payload = {
    pieceId: String(pieceId),
    anchorMeasure: Math.max(1, Math.floor(Number(anchorMeasure) || 1)),
    updatedAt: new Date().toISOString(),
  };
  try {
    window.localStorage.setItem(POSITION_KEY, JSON.stringify(payload));
  } catch (_error) {
    /* ignore quota / private mode */
  }
}

export function clearReadingPosition() {
  try {
    window.localStorage.removeItem(POSITION_KEY);
  } catch (_error) {
    /* ignore */
  }
}
