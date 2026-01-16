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
const sizeWindow = [];
const WIN = 15; // cca 0.25s při ~60fps (reálně méně na mobilu)


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

function smoothSize(newSize) {
  if (!newSize) return null;
  sizeWindow.push(newSize);
  if (sizeWindow.length > WIN) sizeWindow.shift();

  // majorita
  const counts = {};
  for (const s of sizeWindow) counts[s] = (counts[s] || 0) + 1;

  let best = null, bestC = -1;
  for (const [k, v] of Object.entries(counts)) {
    if (v > bestC) { bestC = v; best = k; }
  }
  return { size: best, stability: bestC / sizeWindow.length };
}


function distPx(a, b) {
  const dx = (a.x - b.x) * canvas.width;
  const dy = (a.y - b.y) * canvas.height;
  return Math.hypot(dx, dy);
}

function mid(a, b) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

// Jednoduchý odhad velikosti podle poměru šířky ramen k délce trupu
function estimateTshirtSize(lm) {
  // MediaPipe indices
  const L_SH = lm[11], R_SH = lm[12];
  const L_HIP = lm[23], R_HIP = lm[24];

  // základní kvalita: pokud některé body “mizí”, vrať null
  const visOk = (p) => (p.visibility ?? 1) > 0.6;
  if (![L_SH, R_SH, L_HIP, R_HIP].every(visOk)) return null;

  const shoulderW = distPx(L_SH, R_SH);
  const hipW = distPx(L_HIP, R_HIP);

  const midShoulder = mid(L_SH, R_SH);
  const midHip = mid(L_HIP, R_HIP);
  const torsoL = distPx(midShoulder, midHip);

  if (torsoL < 1) return null;

  const w = shoulderW / torsoL; // hlavní bezrozměrná metrika
  const h = hipW / torsoL;      // pomocná

  // (Volitelně) malá korekce: když jsou boky výrazně širší než ramena,
  // posuň w mírně nahoru, aby to netlačilo k menším velikostem.
  const corr = Math.max(0, (hipW - shoulderW) / torsoL) * 0.15;
  const score = w + corr;

  // Prahy — MUSÍŠ je doladit na vlastních datech.
  // Tyhle jsou rozumný start pro “regular fit” dospělí, čelní pohled.
  let size;
  if (score < 0.72) size = "S";
  else if (score < 0.80) size = "M";
  else if (score < 0.88) size = "L";
  else size = "XL";

  return {
    size,
    score,
    shoulderW,
    hipW,
    torsoL,
    w,
    h
  };
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
    statusEl.textContent = "Namířit na osobu";
    return;
  }

  const lm = results.landmarks[0];
  const est = estimateTshirtSize(lm);
  const sm = smoothSize(est?.size);

  if (!est || !sm) {
    statusEl.textContent = "Detekce funkční ale nedostatek dat k výpočtu";
    return;
  }

  statusEl.textContent =
    `Size: ${sm.size}  | stability ${(sm.stability*100).toFixed(0)}%  | score=${est.score.toFixed(3)}  (w=${est.w.toFixed(3)}, h=${est.h.toFixed(3)})`;

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
