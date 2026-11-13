/** Local performance-reader preferences (zoom, system overlap). */

export const SETTINGS_KEY = "baroque-performance.settings";

const DEFAULTS = {
  zoom: 1,
  overlap: 0.15,
  performanceMode: false,
};

export function loadSettings() {
  try {
    const raw = window.localStorage.getItem(SETTINGS_KEY);
    if (!raw) return { ...DEFAULTS };
    const data = JSON.parse(raw);
    return {
      zoom: clamp(Number(data.zoom), 0.45, 1.8, DEFAULTS.zoom),
      overlap: clamp(Number(data.overlap), 0, 0.45, DEFAULTS.overlap),
      performanceMode: Boolean(data.performanceMode),
    };
  } catch (_error) {
    return { ...DEFAULTS };
  }
}

export function saveSettings(partial) {
  const next = { ...loadSettings(), ...partial };
  next.zoom = clamp(Number(next.zoom), 0.45, 1.8, DEFAULTS.zoom);
  next.overlap = clamp(Number(next.overlap), 0, 0.45, DEFAULTS.overlap);
  next.performanceMode = Boolean(next.performanceMode);
  try {
    window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(next));
  } catch (_error) {
    /* ignore */
  }
  return next;
}

function clamp(value, min, max, fallback) {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, value));
}
