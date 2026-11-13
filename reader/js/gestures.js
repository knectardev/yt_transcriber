/**
 * Head-gesture page turning via MediaPipe Face Landmarker.
 * Frames stay in-browser; no recording or upload.
 */

const DEFAULTS = {
  axis: "roll", // roll | yaw
  /** When true, positive axis delta means next system (swap for mirroring). */
  nextOnPositive: true,
  sensitivityDeg: 14,
  minDurationMs: 280,
  cooldownMs: 900,
  neutralBandDeg: 7,
  smoothing: 0.35,
};

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

/** Extract approximate yaw/pitch/roll (degrees) from a column-major 4x4 matrix. */
export function eulerFromMatrix(matrix) {
  const m = matrix && matrix.data ? matrix.data : matrix;
  if (!m || m.length < 16) return { yaw: 0, pitch: 0, roll: 0 };
  // Column-major rotation: R = [[m0,m4,m8],[m1,m5,m9],[m2,m6,m10]]
  const r00 = m[0];
  const r10 = m[1];
  const r20 = m[2];
  const r21 = m[6];
  const r22 = m[10];
  const pitch = Math.atan2(-r20, Math.sqrt(r00 * r00 + r10 * r10));
  const yaw = Math.atan2(m[4], m[0]);
  const roll = Math.atan2(r21, r22);
  const toDeg = 180 / Math.PI;
  return {
    yaw: yaw * toDeg,
    pitch: pitch * toDeg,
    roll: roll * toDeg,
  };
}

export function createGestureController(options = {}) {
  const opts = { ...DEFAULTS, ...options };
  let faceLandmarker = null;
  let stream = null;
  let video = null;
  let rafId = 0;
  let running = false;
  let calibrated = false;
  let neutral = { yaw: 0, pitch: 0, roll: 0 };
  let smoothed = { yaw: 0, pitch: 0, roll: 0 };
  let armed = true;
  let holdStarted = 0;
  let holdSign = 0;
  let lastFireAt = 0;
  let tracking = false;
  let lastError = "";
  let deviceId = "";

  const listeners = {
    onFire: options.onFire || null,
    onStatus: options.onStatus || null,
    onPose: options.onPose || null,
  };

  function emitStatus(extra = {}) {
    if (!listeners.onStatus) return;
    listeners.onStatus({
      running,
      calibrated,
      tracking,
      armed,
      deviceId,
      axis: opts.axis,
      nextOnPositive: opts.nextOnPositive,
      sensitivityDeg: opts.sensitivityDeg,
      minDurationMs: opts.minDurationMs,
      cooldownMs: opts.cooldownMs,
      error: lastError,
      ...extra,
    });
  }

  function axisValue(pose) {
    if (opts.axis === "yaw") return pose.yaw;
    if (opts.axis === "pitch") return pose.pitch;
    return pose.roll;
  }

  function configure(partial) {
    Object.assign(opts, partial || {});
    opts.sensitivityDeg = clamp(Number(opts.sensitivityDeg) || DEFAULTS.sensitivityDeg, 6, 40);
    opts.minDurationMs = clamp(Number(opts.minDurationMs) || DEFAULTS.minDurationMs, 80, 2000);
    opts.cooldownMs = clamp(Number(opts.cooldownMs) || DEFAULTS.cooldownMs, 200, 4000);
    opts.neutralBandDeg = clamp(Number(opts.neutralBandDeg) || DEFAULTS.neutralBandDeg, 2, 20);
    emitStatus();
  }

  async function listCameras() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return [];
    const devices = await navigator.mediaDevices.enumerateDevices();
    return devices.filter((device) => device.kind === "videoinput");
  }

  async function ensureLandmarker() {
    if (faceLandmarker) return faceLandmarker;
    const mod = await import("../vendor/mediapipe/vision_bundle.mjs");
    const vision = await mod.FilesetResolver.forVisionTasks("./vendor/mediapipe/wasm");
    faceLandmarker = await mod.FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "./vendor/mediapipe/face_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFaceBlendshapes: false,
      outputFacialTransformationMatrixes: true,
    });
    return faceLandmarker;
  }

  async function start({ videoElement, preferredDeviceId = "" } = {}) {
    lastError = "";
    if (!window.isSecureContext) {
      lastError = "Camera requires HTTPS (or localhost).";
      emitStatus();
      throw new Error(lastError);
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      lastError = "Camera API unavailable in this browser.";
      emitStatus();
      throw new Error(lastError);
    }
    await ensureLandmarker();
    video = videoElement;
    if (!video) throw new Error("Missing video element.");
    if (stream) stopTracks();
    const constraints = {
      audio: false,
      video: preferredDeviceId
        ? { deviceId: { exact: preferredDeviceId }, facingMode: "user" }
        : { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
    };
    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
    } catch (error) {
      lastError = error && error.name === "NotAllowedError"
        ? "Camera permission denied."
        : error.message || String(error);
      emitStatus();
      throw error;
    }
    deviceId = preferredDeviceId || (stream.getVideoTracks()[0] && stream.getVideoTracks()[0].getSettings().deviceId) || "";
    video.srcObject = stream;
    video.playsInline = true;
    video.muted = true;
    await video.play();
    running = true;
    tracking = false;
    armed = true;
    holdStarted = 0;
    holdSign = 0;
    emitStatus();
    loop();
  }

  function stopTracks() {
    if (!stream) return;
    for (const track of stream.getTracks()) track.stop();
    stream = null;
  }

  function stop() {
    running = false;
    tracking = false;
    if (rafId) cancelAnimationFrame(rafId);
    rafId = 0;
    stopTracks();
    if (video) video.srcObject = null;
    emitStatus();
  }

  function calibrate() {
    neutral = { ...smoothed };
    calibrated = true;
    armed = true;
    holdStarted = 0;
    holdSign = 0;
    emitStatus({ calibrated: true });
  }

  function processPose(pose) {
    const alpha = opts.smoothing;
    smoothed = {
      yaw: smoothed.yaw * (1 - alpha) + pose.yaw * alpha,
      pitch: smoothed.pitch * (1 - alpha) + pose.pitch * alpha,
      roll: smoothed.roll * (1 - alpha) + pose.roll * alpha,
    };
    if (listeners.onPose) listeners.onPose({ ...smoothed, raw: pose, calibrated, tracking });
    if (!calibrated) return;

    const delta = axisValue(smoothed) - axisValue(neutral);
    const now = performance.now();
    const abs = Math.abs(delta);
    const sign = delta >= 0 ? 1 : -1;

    if (!armed) {
      if (abs <= opts.neutralBandDeg) armed = true;
      return;
    }
    if (now - lastFireAt < opts.cooldownMs) return;

    if (abs < opts.sensitivityDeg) {
      holdStarted = 0;
      holdSign = 0;
      return;
    }
    if (holdSign !== sign) {
      holdSign = sign;
      holdStarted = now;
      return;
    }
    if (now - holdStarted < opts.minDurationMs) return;

    const positiveIsNext = Boolean(opts.nextOnPositive);
    const direction = sign > 0
      ? positiveIsNext ? "next" : "prev"
      : positiveIsNext ? "prev" : "next";
    lastFireAt = now;
    armed = false;
    holdStarted = 0;
    holdSign = 0;
    if (listeners.onFire) listeners.onFire(direction, { delta, axis: opts.axis });
    emitStatus({ lastDirection: direction });
  }

  function loop() {
    if (!running || !video || !faceLandmarker) return;
    rafId = requestAnimationFrame(loop);
    if (video.readyState < 2) return;
    const result = faceLandmarker.detectForVideo(video, performance.now());
    const matrices = result && result.facialTransformationMatrixes;
    if (!matrices || !matrices.length) {
      tracking = false;
      emitStatus();
      return;
    }
    tracking = true;
    const pose = eulerFromMatrix(matrices[0]);
    processPose(pose);
  }

  return {
    start,
    stop,
    calibrate,
    configure,
    listCameras,
    getOptions: () => ({ ...opts }),
    isRunning: () => running,
    isCalibrated: () => calibrated,
    isTracking: () => tracking,
  };
}
