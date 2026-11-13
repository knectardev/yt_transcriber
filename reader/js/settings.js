/** Local performance-reader preferences (zoom, system overlap). */

export const SETTINGS_KEY = "baroque-performance.settings";
export const COLOR_VOICES_KEY = "baroque-transcribe.color-voices";

const DEFAULTS = {
  zoom: 1,
  overlap: 0.15,
  performanceMode: false,
  colorVoices: true,
  gestureAxis: "roll",
  gestureDirection: "pos-next",
  gestureSensitivity: 14,
  gestureHoldMs: 280,
  cameraDeviceId: "",
};

export function loadSettings() {
  try {
    const raw = window.localStorage.getItem(SETTINGS_KEY);
    const data = raw ? JSON.parse(raw) : {};
    let colorVoices = DEFAULTS.colorVoices;
    try {
      const legacy = window.localStorage.getItem(COLOR_VOICES_KEY);
      if (legacy === "1") colorVoices = true;
      else if (legacy === "0") colorVoices = false;
      else if (typeof data.colorVoices === "boolean") colorVoices = data.colorVoices;
    } catch (_error) {
      if (typeof data.colorVoices === "boolean") colorVoices = data.colorVoices;
    }
    return {
      zoom: clamp(Number(data.zoom), 0.45, 1.8, DEFAULTS.zoom),
      overlap: clamp(Number(data.overlap), 0, 0.45, DEFAULTS.overlap),
      performanceMode: Boolean(data.performanceMode),
      colorVoices,
      gestureAxis: data.gestureAxis === "yaw" ? "yaw" : "roll",
      gestureDirection: data.gestureDirection === "pos-prev" ? "pos-prev" : "pos-next",
      gestureSensitivity: clamp(Number(data.gestureSensitivity), 8, 28, DEFAULTS.gestureSensitivity),
      gestureHoldMs: clamp(Number(data.gestureHoldMs), 120, 800, DEFAULTS.gestureHoldMs),
      cameraDeviceId: typeof data.cameraDeviceId === "string" ? data.cameraDeviceId : "",
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
  next.colorVoices = Boolean(next.colorVoices);
  next.gestureAxis = next.gestureAxis === "yaw" ? "yaw" : "roll";
  next.gestureDirection = next.gestureDirection === "pos-prev" ? "pos-prev" : "pos-next";
  next.gestureSensitivity = clamp(Number(next.gestureSensitivity), 8, 28, DEFAULTS.gestureSensitivity);
  next.gestureHoldMs = clamp(Number(next.gestureHoldMs), 120, 800, DEFAULTS.gestureHoldMs);
  next.cameraDeviceId = typeof next.cameraDeviceId === "string" ? next.cameraDeviceId : "";
  try {
    window.localStorage.setItem(SETTINGS_KEY, JSON.stringify(next));
    window.localStorage.setItem(COLOR_VOICES_KEY, next.colorVoices ? "1" : "0");
  } catch (_error) {
    /* ignore */
  }
  return next;
}

function clamp(value, min, max, fallback) {
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, value));
}
