/* ==========================================================================
   Godavari · Crowd Intelligence — dashboard client
   ---------------------------------------------------------------
   · WebSocket snapshot in (~1.7 Hz) -> whole console re-rendered
   · Overlay canvas draws density heatmap, zones, flow vectors and drafts
   · No frameworks, no external assets: works fully offline.
   ========================================================================== */
"use strict";

/* ----------------------------------------------------------- helpers --- */
const $ = (id) => document.getElementById(id);
const DPR = () => window.devicePixelRatio || 1;

const LEVEL_COLOR = {
  OK: "#34d399", BUSY: "#fbbf24", WARNING: "#fb923c", OVERCROWDED: "#f87171",
};
const LEVEL_ORDER = { OK: 0, BUSY: 1, WARNING: 2, OVERCROWDED: 3 };
const RING_C = 2 * Math.PI * 19; // gauge circumference (r = 19 in a 46 box)

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const esc = (s) => String(s ?? "").replace(/[&<>"']/g,
  (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function fmtClock(d) {
  return d.toLocaleTimeString([], { hour12: false });
}
function fmtUptime(sec) {
  sec = Math.max(0, Math.round(sec || 0));
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = sec % 60;
  return h ? `${h}h ${m}m` : m ? `${m}m ${s}s` : `${s}s`;
}
function pushMax(arr, item, max) {
  arr.push(item);
  if (arr.length > max) arr.splice(0, arr.length - max);
}

/* ------------------------------------------------------------- state --- */
const state = {
  cameras: [],
  alerts: [],
  sys: {},
  selectedId: null,
  view: "live",
  overlays: { heatmap: true, zones: true, flow: true },
  draw: { active: false, pts: [] },
  soundOn: true,
  bootedAlerts: false,
  hist: { total: [], tracks: [], fps: [], risk: [], cams: {}, zones: {}, markers: [] },
  seen: new Set(),
  zoneRows: new Map(),
  wallKey: "",
  alertsKey: "",
  audioCtx: null,
  ov: { w: 0, h: 0 },
  kpi: {},
};

let ws = null;
let heatOff = null;
let heatLut = null;

/* --------------------------------------------------------------- boot --- */
window.addEventListener("DOMContentLoaded", () => {
  buildKpis();
  bindUI();
  connectWS();
  refreshSystem();
  refreshAlerts();
  refreshSamples();
  tickClock();
  setInterval(tickClock, 1000);
  setInterval(refreshWallImages, 1300);
  window.addEventListener("resize", sizeOverlay);
  if (window.ResizeObserver) {
    new ResizeObserver(() => sizeOverlay()).observe($("stage"));
  }
});

function tickClock() {
  const el = $("clock");
  if (el) el.textContent = fmtClock(new Date());
}

/* ------------------------------------------------------------- wiring --- */
function bindUI() {
  $("btn-add").onclick = () => {
    const d = $("add-dialog");
    d.classList.toggle("hidden");
    if (!d.classList.contains("hidden")) $("cam-name").focus();
  };
  $("btn-add-close").onclick = () => $("add-dialog").classList.add("hidden");

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.onclick = () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("on"));
      tab.classList.add("on");
      ["upload", "sample", "source"].forEach((m) =>
        $("pane-" + m).classList.toggle("hidden", m !== tab.dataset.mode));
    };
  });

  $("btn-start").onclick = addCamera;
  $("chk-draw").onchange = toggleDraw;
  $("btn-finish").onclick = finishZone;
  $("btn-grid").onclick = autoGrid;
  $("btn-clear-zones").onclick = () => {
    const cam = selectedCam();
    if (!cam) return;
    if ((cam.zones || []).length && !confirm("Delete all zones for this camera?")) return;
    saveZones([]);
  };
  $("btn-del-cam").onclick = deleteCamera;
  $("chk-sound").onchange = (e) => { state.soundOn = e.target.checked; };

  document.querySelectorAll("#view-seg button").forEach((b) => {
    b.onclick = () => setView(b.dataset.view);
  });
  document.querySelectorAll(".tgl").forEach((b) => {
    b.onclick = () => {
      const k = b.dataset.ov;
      state.overlays[k] = !state.overlays[k];
      b.classList.toggle("on", state.overlays[k]);
      drawOverlay();
    };
  });

  // drag & drop upload
  const drop = $("drop");
  const fileInput = $("file-input");
  ["dragenter", "dragover"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) =>
    drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files.length) {
      fileInput.files = e.dataTransfer.files;
      $("file-label").textContent = e.dataTransfer.files[0].name;
    }
  });
  fileInput.addEventListener("change", () => {
    $("file-label").textContent = fileInput.files[0] ? fileInput.files[0].name : "";
  });

  $("overlay").addEventListener("click", overlayClick);

  window.addEventListener("keydown", (e) => {
    if (e.target.matches("input, select, textarea")) return;
    if (e.key === "Escape") cancelDraw();
    else if (e.key === "h" || e.key === "H") pokeToggle("heatmap");
    else if (e.key === "z" || e.key === "Z") pokeToggle("zones");
    else if (e.key === "f" || e.key === "F") pokeToggle("flow");
  });

  document.body.addEventListener("click", initAudioOnce, { once: true });
}

function pokeToggle(k) {
  const btn = document.querySelector(`.tgl[data-ov="${k}"]`);
  if (btn) btn.click();
}

function initAudioOnce() {
  if (!state.audioCtx) {
    try { state.audioCtx = new (window.AudioContext || window.webkitAudioContext)(); }
    catch { /* audio unsupported */ }
  }
}

function beep(severity) {
  if (!state.soundOn || !state.audioCtx) return;
  const ctx = state.audioCtx;
  const now = ctx.currentTime;
  const tones = severity === "OVERCROWDED" ? [880, 660, 880] : [660];
  tones.forEach((freq, i) => {
    const o = ctx.createOscillator();
    const g = ctx.createGain();
    o.type = "square";
    o.frequency.value = freq;
    const t = now + i * 0.16;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.07, t + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.14);
    o.connect(g).connect(ctx.destination);
    o.start(t);
    o.stop(t + 0.16);
  });
}

/* ------------------------------------------------------------ network --- */
function connectWS() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.onmessage = (msg) => {
    let data;
    try { data = JSON.parse(msg.data); } catch { return; }
    if (data.type !== "snapshot") return;
    state.cameras = data.cameras || [];
    state.sys = data.system || {};
    if (!state.selectedId && state.cameras.length) {
      state.selectedId = state.cameras[0].id;
    }
    mergeAlerts(data.alerts);
    recordHistory();
    renderAll();
  };
  ws.onclose = () => {
    $("live-pill").classList.add("off");
    setTimeout(connectWS, 1800);
  };
  ws.onerror = () => { try { ws.close(); } catch { /* noop */ } };
}

async function refreshSystem() {
  try {
    const r = await fetch("/api/system");
    state.sys = await r.json();
    renderTop();
    renderEngine();
  } catch { /* server not ready */ }
}

async function refreshAlerts() {
  try {
    const r = await fetch("/api/alerts?limit=40");
    const list = await r.json();
    list.forEach((a) => state.seen.add(a.id));
    state.alerts = list;
    renderAlerts();
    state.bootedAlerts = true;
  } catch { /* ignore */ }
}

async function refreshSamples() {
  try {
    const r = await fetch("/api/samples");
    const samples = await r.json();
    const sel = $("sample-select");
    sel.innerHTML = "";
    if (!samples.length) {
      const o = document.createElement("option");
      o.value = ""; o.textContent = "No samples bundled";
      sel.appendChild(o);
    } else {
      // Group by folder so the bundled demo and the crowd-footage library
      // do not turn into one long unsorted list.
      const groups = new Map();
      samples.forEach((s) => {
        const i = s.lastIndexOf("/");
        const dir = i === -1 ? "Bundled samples" : s.slice(0, i);
        if (!groups.has(dir)) groups.set(dir, []);
        groups.get(dir).push(s);
      });
      const pretty = (s) => s.slice(s.lastIndexOf("/") + 1).replace(/\.[^.]+$/, "").replace(/_/g, " ");
      groups.forEach((items, dir) => {
        const parent = dir === "Bundled samples"
          ? sel
          : sel.appendChild(Object.assign(document.createElement("optgroup"), {
            label: dir.charAt(0).toUpperCase() + dir.slice(1),
          }));
        items.forEach((s) => {
          const o = document.createElement("option");
          o.value = s; o.textContent = pretty(s);
          parent.appendChild(o);
        });
      });
    }
  } catch { /* ignore */ }
}

/* -------------------------------------------------------------- data --- */
function selectedCam() {
  return state.cameras.find((c) => c.id === state.selectedId) || null;
}

function recordHistory() {
  const now = Date.now();
  let total = 0, tracks = 0, fps = 0, risk = 0, fpsN = 0;

  for (const cam of state.cameras) {
    const m = cam.metrics || {};
    total += m.count || 0;
    tracks += m.active_tracks || 0;
    if (m.infer_fps) { fps += m.infer_fps; fpsN += 1; }
    if (!state.hist.cams[cam.id]) state.hist.cams[cam.id] = [];
    pushMax(state.hist.cams[cam.id], m.count || 0, 90);
    for (const z of m.zones || []) {
      if (!state.hist.zones[z.id]) state.hist.zones[z.id] = [];
      pushMax(state.hist.zones[z.id], z.count || 0, 90);
      if (LEVEL_ORDER[z.level] >= 2) risk += 1;
    }
  }
  pushMax(state.hist.total, { t: now, v: total }, 300);
  pushMax(state.hist.tracks, tracks, 90);
  pushMax(state.hist.fps, fpsN ? fps / fpsN : 0, 90);
  pushMax(state.hist.risk, risk, 90);
  if (state.hist.markers.length) {
    const cutoff = now - 200000;
    state.hist.markers = state.hist.markers.filter((m) => m.t >= cutoff);
  }
}

function mergeAlerts(list) {
  const incoming = list || [];
  for (const a of incoming) {
    if (state.seen.has(a.id)) continue;
    state.seen.add(a.id);
    if (state.bootedAlerts) {
      if (a.level === "OVERCROWDED") {
        beep("OVERCROWDED");
        toast(a);
        flash();
        state.hist.markers.push({ t: Date.now(), level: a.level });
      } else if (a.level === "WARNING") {
        beep("WARNING");
      }
    }
  }
  state.alerts = incoming;
}

/* ------------------------------------------------------------ render --- */
function renderAll() {
  renderTop();
  renderCams();
  renderStage();
  renderZones();
  renderAlerts();
  renderKpis();
  drawTrend();
  renderEngine();
  if (state.view === "wall") renderWall();
}

function renderTop() {
  const sys = state.sys || {};
  const running = sys.cameras_running || 0;
  const total = sys.cameras_total || 0;

  $("live-pill").classList.toggle("off", running === 0);

  const chips = [
    `<span class="chip"><i class="cdot"></i><b>${running}/${total}</b> sources live</span>`,
    `<span class="chip">model <b>${esc(sys.model || "loading…")}</b></span>`,
    `<span class="chip">${sys.device === "gpu" ? "GPU" : "CPU"} <b>${esc(sys.torch || "—")}</b></span>`,
    `<span class="chip">uptime <b>${fmtUptime(sys.uptime_s || 0)}</b></span>`,
  ];
  $("topmeta").innerHTML = chips.join("");
}

function renderEngine() {
  const sys = state.sys || {};
  const rows = [
    ["Model", sys.model || "loading…", !sys.load_error],
    ["Device", sys.device === "gpu" ? "CUDA GPU" : "CPU", sys.device === "gpu"],
    ["CUDA", sys.cuda_available ? "available" : "not available", !!sys.cuda_available],
    ["Sources", `${sys.cameras_running || 0} / ${sys.cameras_total || 0}`, true],
    ["Dashboards", String(sys.ws_clients ?? 0), true],
    ["Uptime", fmtUptime(sys.uptime_s || 0), true],
  ];
  $("sys-kv").innerHTML = rows.map(([k, v, good]) =>
    `<dt>${k}</dt><dd class="${good ? "good" : ""}">${esc(v)}</dd>`).join("");
  const es = $("engine-state");
  es.textContent = sys.loaded ? "READY" : "LOADING";
  es.style.color = sys.loaded ? "#6ee7b7" : "";
}

function renderCams() {
  const ul = $("cam-list");
  if (!state.cameras.length) {
    ul.innerHTML = `<div class="empty-zone">No sources yet — add one to start detecting people.</div>`;
    return;
  }
  ul.innerHTML = state.cameras.map((cam) => {
    const m = cam.metrics || {};
    const sel = cam.id === state.selectedId ? "sel" : "";
    const lvl = m.level || "";
    return `<div class="cam-item ${sel}" data-id="${cam.id}">
      <i class="dot ${esc(cam.status)}"></i>
      <div style="min-width:0">
        <div class="name">${esc(cam.name)}</div>
        <div class="meta"><b>${m.count ?? 0}</b> ppl · ${esc(cam.status)}</div>
      </div>
      <div style="display:flex;flex-direction:column;align-items:flex-end;gap:4px">
        ${lvl ? `<span class="lvl ${lvl}">${lvl}</span>` : ""}
        <canvas class="spark" width="52" height="18"></canvas>
      </div>
    </div>`;
  }).join("");

  ul.querySelectorAll(".cam-item").forEach((item) => {
    item.onclick = () => selectCamera(item.dataset.id);
    const cam = state.cameras.find((c) => c.id === item.dataset.id);
    const spark = item.querySelector("canvas.spark");
    const hist = state.hist.cams[cam.id] || [];
    drawSpark(spark, hist, cam.metrics ? LEVEL_COLOR[cam.metrics.level] || "#22d3ee" : "#22d3ee");
  });
}

function renderStage() {
  const cam = selectedCam();
  const m = (cam && cam.metrics) || {};
  const has = !!cam;

  $("stage-empty").classList.toggle("hidden", has);
  $("wall-wrap").classList.toggle("hidden", state.view !== "wall" || !has);
  $("stage").classList.toggle("hidden", state.view === "wall" && has);

  $("cam-title").textContent = cam ? cam.name : "No camera selected";
  $("cam-sub").textContent = cam
    ? `${String(cam.source_type || "").toUpperCase()} · ${cam.status}${cam.detail ? " · " + cam.detail : ""}`
    : "Add a source to begin monitoring";
  $("st-dot").className = "dot " + (cam ? cam.status : "");

  const feed = $("feed");
  const stage = $("stage");
  if (cam) {
    if (feed.dataset.cam !== cam.id) {
      feed.dataset.cam = cam.id;
      feed.src = `/api/cameras/${cam.id}/stream?t=${Date.now()}`;
    }
    if (m.frame_size && m.frame_size[0]) {
      stage.style.aspectRatio = `${m.frame_size[0]} / ${m.frame_size[1]}`;
    }
  } else {
    feed.dataset.cam = "";
    feed.removeAttribute("src");
  }

  $("hud-count").textContent = m.count ?? 0;
  const lvlEl = $("hud-level");
  lvlEl.className = "lvl " + (m.level || "IDLE");
  lvlEl.textContent = m.level || (cam ? String(cam.status).toUpperCase() : "IDLE");
  $("hud-fps").textContent = `${m.infer_fps ?? 0} fps`;
  $("hud-src").textContent = cam ? `${cam.source_type.toUpperCase()} · ${cam.id}` : "—";

  sizeOverlay();
}

function sizeOverlay() {
  const stage = $("stage");
  const cv = $("overlay");
  if (!stage || !cv) return;
  if (state.view !== "live" || stage.classList.contains("hidden")) return;
  const rect = stage.getBoundingClientRect();
  if (!rect.width) return;
  const dpr = DPR();
  const w = Math.round(rect.width), h = Math.round(rect.height);
  state.ov = { w, h };
  if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
    cv.width = Math.round(w * dpr);
    cv.height = Math.round(h * dpr);
  }
  drawOverlay();
}

/* ------------------------------------------------------ overlay draw --- */
function drawOverlay() {
  const cv = $("overlay");
  if (!cv) return;
  const ctx = cv.getContext("2d");
  const { w: W, h: H } = state.ov;
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, cv.width, cv.height);
  if (!W || !H || state.view !== "live") return;
  ctx.setTransform(DPR(), 0, 0, DPR(), 0, 0);
  ctx.__W = W;
  ctx.__H = H;

  const cam = selectedCam();
  const m = (cam && cam.metrics) || {};

  if (state.overlays.heatmap && m.heatmap) drawHeat(ctx, m.heatmap, W, H);
  if (cam && state.overlays.zones) drawZones(ctx, cam, m, W, H);
  if (cam && state.overlays.flow) drawFlow(ctx, cam, m, W, H);
  if (state.draw.active) drawDraft(ctx, W, H);
}

function buildHeatLut() {
  const stops = [
    [0.00, 24, 86, 255],
    [0.28, 0, 200, 255],
    [0.50, 16, 255, 160],
    [0.72, 255, 224, 60],
    [0.88, 255, 120, 40],
    [1.00, 255, 50, 60],
  ];
  const lut = new Uint8ClampedArray(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let a = stops[0], b = stops[stops.length - 1];
    for (let s = 1; s < stops.length; s++) {
      if (t <= stops[s][0]) { a = stops[s - 1]; b = stops[s]; break; }
    }
    const span = b[0] - a[0] || 1;
    const k = clamp((t - a[0]) / span, 0, 1);
    lut[i * 4] = a[1] + (b[1] - a[1]) * k;
    lut[i * 4 + 1] = a[2] + (b[2] - a[2]) * k;
    lut[i * 4 + 2] = a[3] + (b[3] - a[3]) * k;
    lut[i * 4 + 3] = t < 0.02 ? 0 : Math.min(238, Math.pow(t, 0.82) * 255);
  }
  return lut;
}

function drawHeat(ctx, hm, W, H) {
  if (!hm || !hm.b64) return;
  if (!heatLut) heatLut = buildHeatLut();
  let bin;
  try { bin = atob(hm.b64); } catch { return; }
  const cols = hm.cols, rows = hm.rows;
  if (!heatOff || heatOff.width !== cols || heatOff.height !== rows) {
    heatOff = document.createElement("canvas");
    heatOff.width = cols;
    heatOff.height = rows;
  }
  const hctx = heatOff.getContext("2d");
  const img = hctx.createImageData(cols, rows);
  const d = img.data;
  for (let i = 0; i < bin.length; i++) {
    const v = bin.charCodeAt(i) * 4;
    d[i * 4] = heatLut[v];
    d[i * 4 + 1] = heatLut[v + 1];
    d[i * 4 + 2] = heatLut[v + 2];
    d[i * 4 + 3] = heatLut[v + 3];
  }
  hctx.putImageData(img, 0, 0);

  ctx.save();
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";
  ctx.globalAlpha = 0.84;
  ctx.filter = "blur(10px)";
  ctx.drawImage(heatOff, 0, 0, W, H);
  ctx.restore();
}

function polyPath(ctx, polygon, W, H) {
  ctx.beginPath();
  polygon.forEach((p, i) => {
    const x = p[0] * W, y = p[1] * H;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.closePath();
}

function chip(ctx, text, x, y, color, sub) {
  ctx.font = '600 11px "Segoe UI", system-ui, sans-serif';
  const padX = 7, padY = 5;
  const tw = ctx.measureText(text).width;
  const subText = sub || "";
  ctx.font = '500 10px "Segoe UI", system-ui, sans-serif';
  const sw = subText ? ctx.measureText(subText).width + 8 : 0;
  const w = tw + sw + padX * 2 + 4;
  const h = 22;
  const bx = clamp(x, 6, ctx.__W - w - 6);
  const by = clamp(y, 6, ctx.__H - h - 6);

  ctx.fillStyle = "rgba(4, 9, 17, 0.82)";
  roundRect(ctx, bx, by, w, h, 7);
  ctx.fill();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1;
  ctx.globalAlpha = 0.55;
  roundRect(ctx, bx, by, w, h, 7);
  ctx.stroke();
  ctx.globalAlpha = 1;

  ctx.fillStyle = color;
  ctx.font = '600 11px "Segoe UI", system-ui, sans-serif';
  ctx.textBaseline = "middle";
  ctx.fillText(text, bx + padX, by + h / 2 + 0.5);
  if (subText) {
    ctx.fillStyle = "rgba(190, 214, 235, 0.82)";
    ctx.font = '500 10px "Segoe UI", system-ui, sans-serif';
    ctx.fillText(subText, bx + padX + tw + 4, by + h / 2 + 0.5);
  }
}

function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawZones(ctx, cam, m, W, H) {
  const zones = cam.zones || [];
  const mets = m.zones || [];
  ctx.__W = W; ctx.__H = H;
  let lane = 0; // stagger labels that would sit under the HUD

  for (const z of zones) {
    const met = mets.find((x) => x.id === z.id) || {};
    const level = met.level || "OK";
    const color = LEVEL_COLOR[level] || LEVEL_COLOR.OK;

    polyPath(ctx, z.polygon, W, H);
    ctx.fillStyle = hexA(color, level === "OVERCROWDED" ? 0.17 : 0.09);
    ctx.fill();

    ctx.save();
    ctx.strokeStyle = hexA(color, 0.85);
    ctx.lineWidth = level === "OVERCROWDED" ? 2 : 1.4;
    if (level === "OVERCROWDED" || level === "WARNING") {
      ctx.shadowColor = hexA(color, 0.8);
      ctx.shadowBlur = 14;
    }
    ctx.stroke();
    ctx.restore();

    let minX = 1e9, minY = 1e9;
    for (const p of z.polygon) {
      minX = Math.min(minX, p[0] * W);
      minY = Math.min(minY, p[1] * H);
    }
    const cap = met.capacity ?? z.capacity;
    const occ = met.occupancy_pct ?? 0;
    let labelY = minY + 4;
    if (labelY < 78) { labelY = 78 + lane * 26; lane += 1; }
    chip(ctx, `${z.name}  ${met.count ?? 0}/${cap}`, minX + 4, labelY, color,
      `${occ.toFixed(0)}% · ${level}`);
  }
}

function drawFlow(ctx, cam, m, W, H) {
  const zones = cam.zones || [];
  const mets = m.zones || [];
  for (const met of mets) {
    const flow = met.flow || [0, 0];
    const speed = Math.hypot(flow[0], flow[1]);
    if (speed < 0.0035) continue;
    // live = tracks moving right now; otherwise the server is coasting the
    // last known vector, drawn dimmer so it never blinks off.
    const alpha = met.moving ? 0.95 : 0.42;
    const zone = zones.find((z) => z.id === met.id);
    if (!zone) continue;

    let cx = 0, cy = 0;
    for (const p of zone.polygon) { cx += p[0] * W; cy += p[1] * H; }
    cx /= zone.polygon.length; cy /= zone.polygon.length;

    const ang = Math.atan2(flow[1], flow[0]);
    const len = clamp(speed * 780, 40, 104);
    const x1 = cx - Math.cos(ang) * len / 2;
    const y1 = cy - Math.sin(ang) * len / 2;
    const x2 = cx + Math.cos(ang) * len / 2;
    const y2 = cy + Math.sin(ang) * len / 2;

    ctx.save();
    // anchor dot so the arrow always has a clear origin on busy footage
    ctx.beginPath();
    ctx.arc(cx, cy, 3.2, 0, Math.PI * 2);
    ctx.fillStyle = `rgba(8, 20, 28, ${0.55 * alpha})`;
    ctx.fill();
    ctx.strokeStyle = `rgba(103, 232, 249, ${0.7 * alpha})`;
    ctx.lineWidth = 1.2;
    ctx.stroke();

    ctx.strokeStyle = `rgba(103, 232, 249, ${alpha})`;
    ctx.fillStyle = `rgba(103, 232, 249, ${alpha})`;
    ctx.lineWidth = met.moving ? 2.8 : 1.8;
    ctx.lineCap = "round";
    ctx.shadowColor = `rgba(34, 211, 238, ${alpha * 0.9})`;
    ctx.shadowBlur = met.moving ? 14 : 7;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.stroke();

    const head = 11;
    ctx.beginPath();
    ctx.moveTo(x2, y2);
    ctx.lineTo(x2 - head * Math.cos(ang - 0.42), y2 - head * Math.sin(ang - 0.42));
    ctx.lineTo(x2 - head * Math.cos(ang + 0.42), y2 - head * Math.sin(ang + 0.42));
    ctx.closePath();
    ctx.fill();

    if (met.moving) {
      const txt = `${(speed * 100).toFixed(0)}%/s`;
      ctx.shadowBlur = 0;
      ctx.font = "600 10px ui-monospace, SFMono-Regular, Menlo, monospace";
      // sit beside the shaft, never on top of the arrow head
      const tx = cx - Math.sin(ang) * 17;
      const ty = cy + Math.cos(ang) * 17;
      const tw = ctx.measureText(txt).width;
      ctx.fillStyle = "rgba(4, 12, 18, 0.62)";
      ctx.beginPath();
      ctx.roundRect(tx - tw / 2 - 4, ty - 8, tw + 8, 15, 7);
      ctx.fill();
      ctx.fillStyle = "rgba(165, 243, 252, 0.95)";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(txt, tx, ty);
      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
    }
    ctx.restore();
  }
}

function drawDraft(ctx, W, H) {
  const pts = state.draw.pts;
  if (!pts.length) return;
  ctx.save();
  ctx.lineWidth = 2;
  ctx.strokeStyle = "#fb923c";
  ctx.fillStyle = "rgba(251, 146, 60, 0.16)";
  ctx.setLineDash([6, 4]);
  polyPath(ctx, pts.concat(pts.length >= 3 ? [pts[0]] : []), W, H);
  if (pts.length >= 3) ctx.fill();
  ctx.stroke();
  ctx.setLineDash([]);
  pts.forEach((p, i) => {
    ctx.beginPath();
    ctx.arc(p[0] * W, p[1] * H, i === 0 ? 5 : 3.6, 0, Math.PI * 2);
    ctx.fillStyle = i === 0 ? "#fb923c" : "#fff";
    ctx.fill();
  });
  const last = pts[pts.length - 1];
  chip(ctx, `${pts.length} pts`, last[0] * W + 8, last[1] * H + 8, "#fb923c", "click to add");
  ctx.restore();
}

function compass(vx, vy) {
  // screen y grows downwards, so negate to get compass-style naming
  const deg = (Math.atan2(-vy, vx) * 180 / Math.PI + 360) % 360;
  const names = ["→ E", "↗ NE", "↑ N", "↖ NW", "← W", "↙ SW", "↓ S", "↘ SE"];
  return names[Math.round(deg / 45) % 8];
}

function hexA(hex, a) {
  const h = hex.replace("#", "");
  const n = parseInt(h.length === 3 ? h.split("").map((c) => c + c).join("") : h, 16);
  return `rgba(${(n >> 16) & 255}, ${(n >> 8) & 255}, ${n & 255}, ${a})`;
}

/* ------------------------------------------------------------- zones --- */
function renderZones() {
  const cam = selectedCam();
  const list = $("zone-list");
  const zones = (cam && cam.zones) || [];
  $("zone-count").textContent = zones.length ? String(zones.length) : "0";

  if (!zones.length) {
    list.innerHTML = `<div class="empty-zone">No zones defined for this camera.<br>
      Use <b>Auto-grid</b> or enable <b>Draw zone</b> to watch density per area.</div>`;
    state.zoneRows.clear();
    return;
  }

  const mets = (cam.metrics && cam.metrics.zones) || [];
  const placeholder = list.querySelector(".empty-zone");
  if (placeholder) placeholder.remove();
  const seen = new Set(zones.map((z) => z.id));
  for (const [id, row] of state.zoneRows) {
    if (!seen.has(id)) { row.el.remove(); state.zoneRows.delete(id); }
  }

  zones.forEach((z) => {
    let row = state.zoneRows.get(z.id);
    if (!row) {
      row = createZoneRow(z);
      state.zoneRows.set(z.id, row);
      list.appendChild(row.el);
    }
    updateZoneRow(row, z, mets.find((m) => m.id === z.id) || {});
  });
}

function createZoneRow(z) {
  const el = document.createElement("div");
  el.className = "zone-row";
  el.innerHTML = `
    <div class="gauge">
      <svg viewBox="0 0 46 46">
        <circle class="bg" cx="23" cy="23" r="19"/>
        <circle class="fg" cx="23" cy="23" r="19"
                stroke-dasharray="${RING_C.toFixed(1)}" stroke-dashoffset="${RING_C.toFixed(1)}"/>
      </svg>
      <span class="g-txt">0</span>
    </div>
    <div class="zone-main">
      <div class="zone-top"><span class="zone-name"></span><span class="lvl">OK</span></div>
      <div class="zone-sub"><span class="zs"></span><span class="fb"></span></div>
      <canvas class="zone-mini" width="120" height="20"></canvas>
    </div>
    <div class="zone-side">
      <input type="number" min="1" step="1" title="Zone capacity (people)"/>
      <span class="zlabel">capacity</span>
      <button class="zone-del" title="Remove zone">✕</button>
    </div>`;

  const row = {
    el,
    fg: el.querySelector("circle.fg"),
    gtxt: el.querySelector(".g-txt"),
    name: el.querySelector(".zone-name"),
    level: el.querySelector(".lvl"),
    sub: el.querySelector(".zs"),
    flow: el.querySelector(".fb"),
    mini: el.querySelector(".zone-mini"),
    input: el.querySelector("input"),
    del: el.querySelector(".zone-del"),
  };

  row.input.onchange = () => {
    const v = Math.max(1, parseInt(row.input.value, 10) || 1);
    updateZoneCapacity(z.id, v);
  };
  row.input.onkeydown = (e) => { if (e.key === "Enter") row.input.blur(); };
  row.del.onclick = () => {
    const cam = selectedCam();
    if (!cam) return;
    saveZones((cam.zones || []).filter((x) => x.id !== z.id));
  };
  return row;
}

function updateZoneRow(row, z, met) {
  const level = met.level || "OK";
  const occ = clamp(met.occupancy_pct ?? 0, 0, 100);
  const color = LEVEL_COLOR[level] || LEVEL_COLOR.OK;
  const cap = met.capacity ?? z.capacity;

  row.el.className = "zone-row lvl-" + level;
  row.fg.style.stroke = color;
  row.fg.style.strokeDashoffset = String(RING_C * (1 - occ / 100));
  row.gtxt.textContent = met.count ?? 0;
  row.gtxt.style.color = color;
  row.name.textContent = z.name;
  row.level.className = "lvl " + level;
  row.level.textContent = level;
  row.sub.innerHTML = `<b>${met.count ?? 0}</b> / ${cap} · ${occ.toFixed(0)}%`;

  const flow = met.flow || [0, 0];
  const spd = Math.hypot(flow[0], flow[1]);
  if (spd > 0.0035) {
    const dir = compass(flow[0], flow[1]);
    row.flow.textContent = `flow ${(spd * 100).toFixed(0)}%/s ${dir}`;
    row.flow.dataset.moving = met.moving ? "1" : "0";
  } else {
    row.flow.textContent = "static";
    row.flow.dataset.moving = "0";
  }

  if (document.activeElement !== row.input && String(row.input.value) !== String(cap)) {
    row.input.value = cap;
  }
  const hist = state.hist.zones[z.id] || [];
  drawSpark(row.mini, hist, color);
}

async function saveZones(zones) {
  const cam = selectedCam();
  if (!cam) return;
  const payload = zones.map((z) => ({
    id: z.id, name: z.name, polygon: z.polygon,
    capacity: z.capacity, warn_pct: z.warn_pct ?? 75,
  }));
  try {
    const r = await fetch(`/api/cameras/${cam.id}/zones`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ zones: payload }),
    });
    if (!r.ok) throw new Error(await r.text());
    state.zoneRows.clear();
    $("zone-list").innerHTML = "";
  } catch (e) {
    alert("Saving zones failed: " + e.message);
  }
}

async function updateZoneCapacity(zoneId, capacity) {
  const cam = selectedCam();
  if (!cam) return;
  const zones = (cam.zones || []).map((z) =>
    z.id === zoneId ? { ...z, capacity } : z);
  await saveZones(zones);
}

function gridZones(cols, rows, capacity) {
  const out = [];
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const x0 = c / cols, y0 = r / rows, x1 = (c + 1) / cols, y1 = (r + 1) / rows;
      out.push({
        name: `Zone ${r * cols + c + 1}`,
        polygon: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        capacity,
      });
    }
  }
  return out;
}

async function autoGrid() {
  const cam = selectedCam();
  if (!cam) { alert("Select a camera first."); return; }
  const cols = clamp(parseInt(prompt("Grid columns", "3"), 10) || 3, 1, 6);
  const rows = clamp(parseInt(prompt("Grid rows", "2"), 10) || 2, 1, 6);
  const capacity = Math.max(1, parseInt(prompt("Capacity per zone (people)", "20"), 10) || 20);
  await saveZones(gridZones(cols, rows, capacity));
}

/* ------------------------------------------------------------ drawing --- */
function toggleDraw(e) {
  state.draw.active = e.target.checked;
  state.draw.pts = [];
  $("stage").classList.toggle("drawing", state.draw.active);
  $("draw-hint").classList.toggle("hidden", !state.draw.active);
  $("new-zone-form").classList.toggle("hidden", !state.draw.active);
  if (state.draw.active) $("zone-name").focus();
  drawOverlay();
}

function cancelDraw() {
  state.draw.pts = [];
  state.draw.active = false;
  $("chk-draw").checked = false;
  $("stage").classList.remove("drawing");
  $("draw-hint").classList.add("hidden");
  $("new-zone-form").classList.add("hidden");
  drawOverlay();
}

function overlayClick(ev) {
  if (!state.draw.active) return;
  const rect = $("stage").getBoundingClientRect();
  const x = clamp((ev.clientX - rect.left) / rect.width, 0, 1);
  const y = clamp((ev.clientY - rect.top) / rect.height, 0, 1);
  state.draw.pts.push([+x.toFixed(4), +y.toFixed(4)]);
  drawOverlay();
}

async function finishZone() {
  if (state.draw.pts.length < 3) {
    alert("Place at least 3 points on the frame first.");
    return;
  }
  const cam = selectedCam();
  if (!cam) return;
  const name = $("zone-name").value.trim() || `Zone ${new Date().toLocaleTimeString()}`;
  const capacity = Math.max(1, parseInt($("zone-cap").value, 10) || 20);
  const zones = [...(cam.zones || []), {
    name, polygon: state.draw.pts.slice(), capacity, warn_pct: 75,
  }];
  cancelDraw();
  $("zone-name").value = "";
  await saveZones(zones);
}

/* ------------------------------------------------------------ cameras --- */
function selectCamera(id) {
  if (state.selectedId === id) return;
  state.selectedId = id;
  state.zoneRows.clear();
  $("zone-list").innerHTML = "";
  $("feed").dataset.cam = "";
  renderAll();
}

function setView(view) {
  state.view = view;
  document.querySelectorAll("#view-seg button").forEach((b) =>
    b.classList.toggle("on", b.dataset.view === view));
  const foot = document.querySelector(".stage-foot");
  if (foot) foot.classList.toggle("hidden", view === "wall");
  renderStage();
  if (view === "wall") { renderWall(); refreshWallImages(); }
}

function renderWall() {
  const wall = $("wall");
  const cams = state.cameras;
  const key = cams.map((c) => c.id).join("|");
  if (state.wallKey !== key) {
    state.wallKey = key;
    wall.innerHTML = cams.map((c) => `
      <div class="tile" data-id="${c.id}">
        <img alt="" data-tile="${c.id}"/>
        <div class="t-top"><span class="lvl ${c.metrics ? c.metrics.level : "IDLE"}">${c.metrics ? c.metrics.level : "—"}</span></div>
        <div class="t-hud">
          <span class="t-name">${esc(c.name)}</span>
          <span class="t-count" data-count="${c.id}">0</span>
        </div>
      </div>`).join("");
    wall.querySelectorAll(".tile").forEach((t) => {
      t.onclick = () => { selectCamera(t.dataset.id); setView("live"); };
    });
  }
  cams.forEach((c) => {
    const node = wall.querySelector(`[data-count="${c.id}"]`);
    if (node) node.textContent = (c.metrics && c.metrics.count) ?? 0;
    const tile = wall.querySelector(`.tile[data-id="${c.id}"]`);
    if (tile) {
      const lvl = c.metrics ? c.metrics.level : "IDLE";
      tile.classList.toggle("off", c.status !== "running");
      const badge = tile.querySelector(".t-top .lvl");
      if (badge) { badge.className = "lvl " + lvl; badge.textContent = lvl; }
    }
  });
}

function refreshWallImages() {
  if (state.view !== "wall") return;
  document.querySelectorAll("#wall img[data-tile]").forEach((img) => {
    img.src = `/api/cameras/${img.dataset.tile}/snapshot?t=${Date.now()}`;
  });
}

async function addCamera() {
  const mode = document.querySelector(".tab.on").dataset.mode;
  const name = $("cam-name").value.trim();
  const fd = new FormData();
  if (name) fd.append("name", name);

  $("btn-start").disabled = true;
  $("add-progress").textContent = "starting…";
  try {
    if (mode === "upload") {
      const f = $("file-input").files[0];
      if (!f) { alert("Choose a video file first."); return; }
      fd.append("file", f);
      fd.append("loop", $("loop-input").checked ? "true" : "false");
      $("add-progress").textContent = `uploading ${f.name}…`;
    } else if (mode === "sample") {
      const s = $("sample-select").value;
      if (!s) { alert("No sample available."); return; }
      fd.append("sample", s);
      fd.append("loop", $("loop-input").checked ? "true" : "false");
    } else {
      const src = $("source-input").value.trim();
      if (!src) { alert("Enter an rtsp:// url or a webcam index (0)."); return; }
      fd.append("source", src);
    }

    const r = await fetch("/api/cameras", { method: "POST", body: fd });
    if (!r.ok) throw new Error(await r.text());
    const cam = await r.json();
    $("add-progress").textContent = "live";
    $("add-dialog").classList.add("hidden");
    state.selectedId = cam.id;
    state.zoneRows.clear();
    $("zone-list").innerHTML = "";
    setTimeout(() => { state.selectedId = cam.id; renderAll(); }, 250);
  } catch (e) {
    alert("Could not add camera: " + e.message);
    $("add-progress").textContent = "";
  } finally {
    $("btn-start").disabled = false;
  }
}

async function deleteCamera() {
  const cam = selectedCam();
  if (!cam) return;
  if (!confirm(`Remove "${cam.name}" from monitoring?`)) return;
  await fetch(`/api/cameras/${cam.id}`, { method: "DELETE" });
  state.selectedId = null;
  state.zoneRows.clear();
  $("zone-list").innerHTML = "";
  renderAll();
}

/* ------------------------------------------------------------- chips --- */
function buildKpis() {
  const defs = [
    { key: "people", label: "People, all sources", unit: "" },
    { key: "tracks", label: "Tracked identities", unit: "" },
    { key: "fps", label: "Inference rate", unit: "fps" },
    { key: "risk", label: "Zones at risk", unit: "" },
  ];
  const host = $("kpis");
  host.innerHTML = defs.map((d) => `
    <div class="kpi">
      <div class="k-accent"></div>
      <div class="k-label">${d.label}</div>
      <div class="k-value"><span data-k="${d.key}">0</span><small>${d.unit}</small></div>
      <canvas data-kc="${d.key}"></canvas>
    </div>`).join("");
  defs.forEach((d) => {
    state.kpi[d.key] = {
      value: host.querySelector(`[data-k="${d.key}"]`),
      canvas: host.querySelector(`[data-kc="${d.key}"]`),
    };
  });
}

function renderKpis() {
  const h = state.hist;
  const people = h.total.length ? h.total[h.total.length - 1].v : 0;
  const tracks = h.tracks[h.tracks.length - 1] || 0;
  const fps = h.fps[h.fps.length - 1] || 0;
  const risk = h.risk[h.risk.length - 1] || 0;

  setKpi("people", people, "#22d3ee");
  setKpi("tracks", Math.round(tracks), "#818cf8");
  setKpi("fps", fps.toFixed(1), "#34d399");
  setKpi("risk", risk, risk > 0 ? "#f87171" : "#34d399");
}

function setKpi(key, value, color) {
  const k = state.kpi[key];
  if (!k) return;
  k.value.textContent = value;
  const arr = key === "people"
    ? state.hist.total.map((p) => p.v)
    : state.hist[key] || [];
  drawSpark(k.canvas, arr, color, { fill: true, height: 30 });
}

function drawSpark(cv, arr, color, opts = {}) {
  if (!cv) return;
  const dpr = DPR();
  const cssW = cv.clientWidth || cv.width || 60;
  const cssH = opts.height || cv.clientHeight || 20;
  if (cv.width !== Math.round(cssW * dpr) || cv.height !== Math.round(cssH * dpr)) {
    cv.width = Math.round(cssW * dpr);
    cv.height = Math.round(cssH * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);
  if (arr.length < 2) return;

  const n = arr.length;
  const max = Math.max(1, ...arr) * 1.12;
  const pad = 2;
  const x = (i) => pad + (i / (n - 1)) * (cssW - pad * 2);
  const y = (v) => cssH - pad - (v / max) * (cssH - pad * 2);

  ctx.beginPath();
  arr.forEach((v, i) => (i ? ctx.lineTo(x(i), y(v)) : ctx.moveTo(x(i), y(v))));
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.4;
  ctx.lineJoin = "round";
  ctx.globalAlpha = 0.95;
  ctx.stroke();

  if (opts.fill) {
    ctx.lineTo(x(n - 1), cssH);
    ctx.lineTo(x(0), cssH);
    ctx.closePath();
    const g = ctx.createLinearGradient(0, 0, 0, cssH);
    g.addColorStop(0, hexA(color, 0.3));
    g.addColorStop(1, hexA(color, 0));
    ctx.fillStyle = g;
    ctx.fill();
  }
  ctx.globalAlpha = 1;
}

/* ------------------------------------------------------------- trend --- */
function drawTrend() {
  const cv = $("trend");
  if (!cv) return;
  const dpr = DPR();
  const cssW = cv.clientWidth || 600;
  const cssH = 120;
  if (cv.width !== Math.round(cssW * dpr) || cv.height !== Math.round(cssH * dpr)) {
    cv.width = Math.round(cssW * dpr);
    cv.height = Math.round(cssH * dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  const pad = { l: 26, r: 8, t: 10, b: 6 };
  const innerW = cssW - pad.l - pad.r;
  const innerH = cssH - pad.t - pad.b;
  const pts = state.hist.total;

  const maxV = Math.max(4, ...pts.map((p) => p.v));
  const niceMax = Math.ceil(maxV / 4) * 4;

  // grid + y labels
  ctx.font = '9px "Segoe UI", system-ui, sans-serif';
  ctx.fillStyle = "rgba(124, 145, 170, 0.75)";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const y = pad.t + innerH * (i / 4);
    ctx.strokeStyle = "rgba(148, 163, 184, 0.08)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(pad.l, y + 0.5);
    ctx.lineTo(cssW - pad.r, y + 0.5);
    ctx.stroke();
    ctx.fillText(String(Math.round(niceMax * (1 - i / 4))), pad.l - 7, y);
  }

  if (pts.length < 2) {
    ctx.fillStyle = "rgba(124, 145, 170, 0.6)";
    ctx.textAlign = "center";
    ctx.font = '11px "Segoe UI", system-ui, sans-serif';
    ctx.fillText("Collecting telemetry…", cssW / 2, cssH / 2);
    return;
  }

  const t0 = pts[0].t;
  const t1 = pts[pts.length - 1].t || (t0 + 1);
  const x = (t) => pad.l + ((t - t0) / Math.max(1, t1 - t0)) * innerW;
  const y = (v) => pad.t + innerH - (v / niceMax) * innerH;

  // alert markers
  for (const mk of state.hist.markers) {
    if (mk.t < t0) continue;
    const mx = x(mk.t);
    ctx.strokeStyle = mk.level === "OVERCROWDED"
      ? "rgba(248, 113, 113, 0.45)" : "rgba(251, 146, 60, 0.3)";
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(mx, pad.t);
    ctx.lineTo(mx, pad.t + innerH);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // area + line
  ctx.beginPath();
  pts.forEach((p, i) => (i ? ctx.lineTo(x(p.t), y(p.v)) : ctx.moveTo(x(p.t), y(p.v))));
  ctx.strokeStyle = "#22d3ee";
  ctx.lineWidth = 1.8;
  ctx.lineJoin = "round";
  ctx.shadowColor = "rgba(34, 211, 238, 0.35)";
  ctx.shadowBlur = 10;
  ctx.stroke();
  ctx.shadowBlur = 0;

  ctx.lineTo(x(t1), pad.t + innerH);
  ctx.lineTo(x(t0), pad.t + innerH);
  ctx.closePath();
  const g = ctx.createLinearGradient(0, pad.t, 0, pad.t + innerH);
  g.addColorStop(0, "rgba(34, 211, 238, 0.26)");
  g.addColorStop(0.7, "rgba(129, 140, 248, 0.08)");
  g.addColorStop(1, "rgba(129, 140, 248, 0)");
  ctx.fillStyle = g;
  ctx.fill();

  // head dot + value
  const last = pts[pts.length - 1];
  ctx.beginPath();
  ctx.arc(x(last.t), y(last.v), 3, 0, Math.PI * 2);
  ctx.fillStyle = "#a5f3fc";
  ctx.shadowColor = "#22d3ee";
  ctx.shadowBlur = 8;
  ctx.fill();
  ctx.shadowBlur = 0;
}

/* ------------------------------------------------------------ alerts --- */
function renderAlerts() {
  const feed = $("alert-feed");
  const list = state.alerts.slice(0, 30);
  const key = list.map((a) => a.id).join("|");
  if (key === state.alertsKey) return;
  state.alertsKey = key;

  if (!list.length) {
    feed.innerHTML = `<div class="alert-empty">No alerts yet.<br>
      Warnings appear at 75% of a zone's capacity; the capacity alert fires
      when a zone stays over 100% for 5 seconds.</div>`;
    return;
  }
  feed.innerHTML = list.map((a) => {
    const when = new Date(a.ts * 1000).toLocaleTimeString([], { hour12: false });
    const lvl = a.level || "WARNING";
    return `<div class="alert ${lvl}">
      <div class="a-top">
        <span class="lvl ${lvl}">${lvl}</span>
        <span class="a-when">${when}</span>
        <span class="a-zone">${esc(a.cam_name || "")} · ${esc(a.zone || "")}</span>
      </div>
      <div class="a-msg">${esc(a.message || "")}</div>
    </div>`;
  }).join("");
}

function toast(a) {
  const host = $("toasts");
  const el = document.createElement("div");
  el.className = "toast";
  el.innerHTML = `
    <div class="t-ico">⚠</div>
    <div>
      <div class="t-title">Capacity alert</div>
      <div class="t-body"><b>${esc(a.zone || "")}</b> — ${esc(a.message || "")}</div>
    </div>`;
  el.onclick = () => el.remove();
  host.appendChild(el);
  setTimeout(() => {
    el.classList.add("out");
    setTimeout(() => el.remove(), 260);
  }, 6500);
}

function flash() {
  const f = $("flush");
  if (!f) return;
  f.classList.add("on");
  setTimeout(() => f.classList.remove("on"), 140);
}
