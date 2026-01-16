import {
  PoseLandmarker,
  FilesetResolver,
  DrawingUtils
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const statusEl = document.getElementById("status");
const logEl = document.getElementById("log");

const btnStart = document.getElementById("btnStart");
const btnStop = document.getElementById("btnStop");

let poseLandmarker = null;
let drawingUtils = null;
let rafId = null;
let stream = null;

function log(msg) {
  logEl.textContent = msg;
}

async function setupCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { facingMode: "environment" },
    audio: false,
  });
  video.srcObject = stream;

  await new Promise((resolve) => {
    video.onloadedmetadata = () => resolve();
  });

  // Nastav skutečné pixel rozměry canvasu
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
}

async function loadModel() {
  statusEl.textContent = "Loading model...";
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm"
  );

  poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
      delegate: "GPU",
    },
    runningMode: "VIDEO",
    numPoses: 1,
  });

  drawingUtils = new DrawingUtils(ctx);
  statusEl.textContent = "Model loaded.";
}

function shoulderPx(lm) {
  const L = lm[11], R = lm[12]; // shoulders
  const dx = (L.x - R.x) * canvas.width;
  const dy = (L.y - R.y) * canvas.height;
  return Math.hypot(dx, dy);
}

function draw(results) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (!results?.landmarks?.length) {
    statusEl.textContent = "No person detected.";
    return;
  }

  const lm = results.landmarks[0];

  drawingUtils.drawConnectors(lm, PoseLandmarker.POSE_CONNECTIONS);
  drawingUtils.drawLandmarks(lm, { radius: 3 });

  const spx = shoulderPx(lm);
  statusEl.textContent = `Pose OK | shoulder_px ≈ ${spx.toFixed(1)}`;
}

function loop() {
  const now = performance.now();
  const results = poseLandmarker.detectForVideo(video, now);
  draw(results);
  rafId = requestAnimationFrame(loop);
}

function stopAll() {
  if (rafId) cancelAnimationFrame(rafId);
  rafId = null;

  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;
  statusEl.textContent = "Stopped.";
}

btnStart.addEventListener("click", async () => {
  try {
    btnStart.disabled = true;
    statusEl.textContent = "Requesting camera...";

    await setupCamera();
    await loadModel();
    await video.play();

    btnStop.disabled = false;
    log("Tip: stůj čelně, ruce dolů, celé tělo/torzo v záběru.");
    loop();
  } catch (e) {
    console.error(e);
    statusEl.textContent = "Error: " + (e?.message || e);
    btnStart.disabled = false;
  }
});

btnStop.addEventListener("click", () => {
  stopAll();
  btnStart.disabled = false;
  btnStop.disabled = true;
});
