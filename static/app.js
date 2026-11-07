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

/* One camera can be inspected in several ways. Boxes are burnt into the JPEG
   by the worker, so switching a view may also tell the server to stop drawing
   them; everything else is drawn locally on the overlay canvas. */
const CAMERA_VIEWS = {
  boxes:   { detections: true,  zones: true, heatmap: false, flow: false, countLines: true },
  actual:  { detections: false, zones: false, heatmap: false, flow: false, countLines: false },
  heatmap: { detections: false, zones: true, heatmap: true,  flow: false, countLines: false },
  areas:   { detections: false, zones: true, heatmap: false, flow: false, countLines: true },
  flow:    { detections: false, zones: false, heatmap: false, flow: true,  countLines: false },
};
const VIEW_KEYS = { "1": "boxes", "2": "actual", "3": "heatmap", "4": "areas", "5": "flow" };

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
  overlays: { heatmap: false, zones: true, flow: false, countLines: true, detections: true },
  heatOpacity: 0.58,
  focusZoneId: null,
  draw: { active: false, pts: [] },
  lineDraw: { active: false, pts: [] },
  calibrationDraw: { active: false, pts: [] },
  panel: "cameras",
  inspector: "alerts",
  camView: "boxes",
  viewSyncKey: null,
  mosaic: false,
  mosaicTimer: null,
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
  heatmapArchive: { camera: "", items: [], playing: false, timer: null },
};

let ws = null;
let heatOff = null;
let heatLut = null;

/* --------------------------------------------------------------- boot --- */
window.addEventListener("DOMContentLoaded", () => {
  buildKpis();
  restoreLayout();
  bindUI();
  connectWS();
  refreshSystem();
  refreshAlerts();
  refreshSamples();
  refreshAnalytics();
  setInterval(refreshAnalytics, 30000);
  tickClock();
  setInterval(tickClock, 1000);
  setInterval(refreshWallImages, 1300);
  window.addEventListener("resize", sizeOverlay);
  window.addEventListener("resize", () => {
    const range = $("heatmap-range");
    if (range && !range.disabled) renderArchivedHeatmap(Number(range.value));
  });
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
  $("btn-analytics").onclick = refreshAnalytics;
  $("analytics-period").onchange = refreshAnalytics;
  $("btn-fullscreen").onclick = async () => {
    if (document.fullscreenElement) await document.exitFullscreen();
    else {
      setView("wall");
      await $("wall-wrap").requestFullscreen?.();
    }
  };
  $("btn-count-line").onclick = startCountLineDrawing;
  $("btn-calibrate").onclick = startCalibrationDrawing;
  $("validation-source").onchange = renderValidation;
  $("btn-validation-sample").onclick = recordValidationSample;
  $("btn-validation-promote").onclick = promoteValidationSource;
  $("btn-validation-revoke").onclick = revokeValidationSource;
  $("heatmap-camera").onchange = loadHeatmapHistory;
  $("heatmap-range").oninput = (e) => renderArchivedHeatmap(Number(e.target.value));
  $("btn-heatmap-play").onclick = toggleHeatmapPlayback;
  $("btn-finish").onclick = finishZone;
  $("btn-new-area").onclick = startAreaDraw;
  $("btn-undo-pt").onclick = undoPoint;
  $("btn-cancel-draw").onclick = cancelDraw;
  $("btn-areas").onclick = () => switchInspector("areas");
  $("btn-edit-areas").onclick = () => switchInspector("areas");
  $("btn-grid").onclick = autoGrid;
  $("btn-clear-zones").onclick = () => {
    const cam = selectedCam();
    if (!cam) return;
    if ((cam.zones || []).length && !confirm("Delete all areas for this camera?")) return;
    saveZones([]);
  };
  $("btn-del-cam").onclick = deleteCamera;
  $("camera-profile").onchange = async (event) => {
    const cam = selectedCam();
    if (!cam) return;
    const select = event.target;
    select.disabled = true;
    try {
      const response = await fetch(`/api/cameras/${encodeURIComponent(cam.id)}/profile`, {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile: select.value }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not update camera profile");
      cam.profile = data.profile;
      toastOk("Camera profile saved", `${esc(cam.name)} now uses ${esc(data.profile)}.`);
    } catch (err) {
      alert(err.message);
      select.value = cam.profile || "global";
    } finally { select.disabled = false; }
  };
  $("chk-sound").onchange = (e) => { state.soundOn = e.target.checked; };
  $("detector-profile").onchange = async (e) => {
    const select = e.target;
    select.disabled = true;
    try {
      const response = await fetch("/api/detector-profile", {
        method: "PUT", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ profile: select.value }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Could not change detection profile");
      state.sys = { ...(state.sys || {}), ...data };
      renderEngine();
      $("detector-profile-note").textContent = `Active · ${data.imgsz}px · confidence ${data.confidence}. Expect more compute load and some extra false detections.`;
    } catch (err) {
      alert(err.message);
      select.value = state.sys?.profile || "balanced";
    } finally { select.disabled = false; }
  };

  document.querySelectorAll("#view-seg button").forEach((b) => {
    b.onclick = () => setView(b.dataset.view);
  });
  document.querySelectorAll(".camview").forEach((b) => {
    b.onclick = () => (b.dataset.camview === "all" ? setMosaic(true) : applyCameraView(b.dataset.camview));
  });
  document.querySelectorAll(".mtile").forEach((t) => {
    const open = () => { applyCameraView(t.dataset.mview); setMosaic(false); };
    t.onclick = open;
    t.onkeydown = (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); } };
  });
  $("btn-side-toggle").onclick = () => setSidebar($("app").dataset.side === "closed");
  $("btn-side-toggle-2").onclick = () => setSidebar($("app").dataset.side === "closed");
  document.querySelectorAll(".side-tab").forEach((b) => {
    b.onclick = () => switchPanel(b.dataset.panel);
  });
  document.querySelectorAll(".ins-tab").forEach((b) => {
    b.onclick = () => switchInspector(b.dataset.ins);
  });
  $("heat-opacity").oninput = (e) => {
    state.heatOpacity = Number(e.target.value) / 100;
    $("heat-opacity-value").textContent = `${e.target.value}%`;
    drawOverlay();
  };

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
    else if (e.key === "Enter" && state.draw.active) finishZone();
    else if (e.key === "0") setMosaic(true);
    else if (VIEW_KEYS[e.key]) applyCameraView(VIEW_KEYS[e.key]);
  });

  document.body.addEventListener("click", initAudioOnce, { once: true });
}

/* ---- camera views ----------------------------------------------------- */
/* All views at once: the same camera tiled five ways, so an operator can read
   boxes, the clean picture, density and areas together, then click a tile to
   inspect that one view full size. */
function setMosaic(on) {
  const host = $("mosaic");
  if (!host) return;
  if (!on) {
    if (!state.mosaic) return;
    state.mosaic = false;
    host.classList.add("hidden");
    if (state.mosaicTimer) { clearInterval(state.mosaicTimer); state.mosaicTimer = null; }
    const raw = $("feed-raw");
    if (raw) { raw.removeAttribute("src"); raw.dataset.cam = ""; }
    renderStage();
    syncViewButtons();
    drawOverlay();
    return;
  }
  if (state.view === "wall") setView("live");
  const cam = selectedCam();
  if (!cam) { alert("Select a camera first — all views show one camera side by side."); return; }
  state.mosaic = true;
  state.viewSyncKey = null;
  host.classList.remove("hidden");
  // The boxes tile is the annotated stream, so that camera must be drawing them.
  if (cam.display && cam.display.detections !== true) updateDetectionDisplay(true);
  setFeedRaw(cam);
  syncViewButtons();
  renderStage();
  if (!state.mosaicTimer) state.mosaicTimer = setInterval(paintMosaic, 180);
  paintMosaic();
}

function setFeedRaw(cam) {
  const raw = $("feed-raw");
  if (!raw || !cam) return;
  if (raw.dataset.cam !== cam.id) {
    raw.dataset.cam = cam.id;
    raw.src = `/api/cameras/${cam.id}/stream?raw=1&t=${Date.now()}`;
  }
}

function drawContain(ctx, img, w, h) {
  const iw = img && img.naturalWidth, ih = img && img.naturalHeight;
  if (!iw || !ih) return;
  const s = Math.min(w / iw, h / ih);
  const dw = iw * s, dh = ih * s;
  ctx.drawImage(img, (w - dw) / 2, (h - dh) / 2, dw, dh);
}

function paintMosaic() {
  const host = $("mosaic");
  if (!host || host.classList.contains("hidden")) return;
  const cam = selectedCam();
  if (!cam) return;
  const m = cam.metrics || {};
  const dpr = DPR();
  const feed = $("feed");
  const raw = $("feed-raw");
  const rawReady = raw && raw.complete && raw.naturalWidth > 0;

  const head = $("mosaic-head");
  if (head) {
    const hl = headlineCount(m);
    const count = hl.prefix + hl.value;
    const trust = m.estimation_mode ? ` · ${String(m.estimation_mode).replace(/_/g, " ")}` : "";
    const txt = `${cam.name} · ${count} people${trust} · click any tile to open it`;
    if (head.textContent !== txt) head.textContent = txt;
  }

  host.querySelectorAll(".mtile").forEach((tile) => {
    const view = tile.dataset.mview;
    const cv = tile.querySelector("canvas");
    const w = tile.clientWidth, h = tile.clientHeight;
    if (!cv || !w || !h) return;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#02060d";
    ctx.fillRect(0, 0, w, h);

    const src = view === "boxes" ? feed : (rawReady ? raw : feed);
    drawContain(ctx, src, w, h);
    ctx.__W = w;
    ctx.__H = h;

    if (view === "heatmap") {
      if (m.heatmap) drawHeat(ctx, m.heatmap, w, h, state.heatOpacity);
      drawZones(ctx, cam, m, w, h, { compact: true, thin: true });
    } else if (view === "areas") {
      if ((cam.zones || []).length) {
        ctx.fillStyle = "rgba(3, 8, 16, 0.62)";
        ctx.fillRect(0, 0, w, h);
      }
      drawZones(ctx, cam, m, w, h, { compact: true, isolated: true });
    } else if (view === "flow") {
      drawZones(ctx, cam, m, w, h, { compact: true, thin: true });
      drawFlow(ctx, cam, m, w, h);
    }
  });
}

function applyCameraView(name, opts = {}) {
  const view = CAMERA_VIEWS[name];
  if (!view) return;
  if (state.mosaic) setMosaic(false);
  const wantBoxes = view.detections;
  const hadBoxes = !!state.overlays.detections;
  state.camView = name;
  Object.assign(state.overlays, view);

  document.querySelectorAll(".camview").forEach((b) => {
    const on = b.dataset.camview === name;
    b.classList.toggle("on", on);
    b.setAttribute("aria-selected", String(on));
  });
  const stage = $("stage");
  if (stage) stage.classList.toggle("view-areas", name === "areas");
  const opWrap = $("heat-op-wrap");
  if (opWrap) opWrap.classList.toggle("hidden", name !== "heatmap");

  if (wantBoxes !== hadBoxes) updateDetectionDisplay(wantBoxes);
  if (!opts.silent) {
    try { localStorage.setItem("godavari.camview", name); } catch { /* private mode */ }
  }
  drawOverlay();
}

function syncViewButtons() {
  document.querySelectorAll(".camview").forEach((b) => {
    const on = state.mosaic ? b.dataset.camview === "all" : b.dataset.camview === state.camView;
    b.classList.toggle("on", on);
    b.setAttribute("aria-selected", String(on));
  });
  const opWrap = $("heat-op-wrap");
  if (opWrap) opWrap.classList.toggle("hidden", state.camView !== "heatmap");
  const stage = $("stage");
  if (stage) stage.classList.toggle("view-areas", state.camView === "areas");
}

/* Sidebar: three panels (sources, engine, insights) behind a collapsible rail,
   so the main view stays on the camera instead of competing with settings. */
function setSidebar(open) {
  $("app").dataset.side = open ? "open" : "closed";
  $("btn-side-toggle").setAttribute("aria-expanded", String(open));
  const t = $("btn-side-toggle");
  t.textContent = open ? "«" : "»";
  t.title = open ? "Collapse sidebar" : "Expand sidebar";
  try { localStorage.setItem("godavari.sidebar", open ? "open" : "closed"); } catch { /* private mode */ }
}

function switchPanel(name) {
  if (!name) return;
  // Choosing a section while collapsed expands the rail, otherwise the click
  // would appear to do nothing.
  if ($("app").dataset.side === "closed") setSidebar(true);
  state.panel = name;
  document.querySelectorAll(".side-tab").forEach((b) => {
    const on = b.dataset.panel === name;
    b.classList.toggle("on", on);
    b.setAttribute("aria-selected", String(on));
  });
  document.querySelectorAll(".side-panel").forEach((p) =>
    p.classList.toggle("on", p.id === "panel-" + name));
  try { localStorage.setItem("godavari.panel", name); } catch { /* private mode */ }
}

function switchInspector(name) {
  if (!name) return;
  state.inspector = name;
  document.querySelectorAll(".ins-tab").forEach((b) => {
    const on = b.dataset.ins === name;
    b.classList.toggle("on", on);
    b.setAttribute("aria-selected", String(on));
  });
  document.querySelectorAll(".ins-panel").forEach((p) =>
    p.classList.toggle("on", p.id === "insp-" + name));
  try { localStorage.setItem("godavari.inspector", name); } catch { /* private mode */ }
}

function restoreLayout() {
  let side = "open", panel = "cameras", inspector = "alerts", view = "boxes";
  try {
    side = localStorage.getItem("godavari.sidebar") || "open";
    panel = localStorage.getItem("godavari.panel") || "cameras";
    inspector = localStorage.getItem("godavari.inspector") || "alerts";
    view = localStorage.getItem("godavari.camview") || "boxes";
  } catch { /* private mode */ }
  // Panels first: switchPanel() expands a collapsed rail on purpose, so calling
  // it after setSidebar() would throw away the saved collapsed state on every
  // page load.
  switchPanel(panel);
  switchInspector(inspector);
  setSidebar(side !== "closed");
  applyCameraView(CAMERA_VIEWS[view] ? view : "boxes", { silent: true });
}

async function updateDetectionDisplay(enabled) {
  const cam = selectedCam();
  if (!cam) return;
  cam.display = { ...(cam.display || {}), detections: enabled };
  try {
    const r = await fetch(`/api/cameras/${cam.id}/display`, {
      method: "PUT", headers: { "content-type": "application/json" },
      body: JSON.stringify({ detections: enabled }),
    });
    if (!r.ok) throw new Error(await r.text());
  } catch (err) { toastErr("Display setting not saved", err.message); }
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
  // The websocket bypasses HTTP middleware, so when the deployment has auth
  // enabled the dashboard token has to travel in the query string.
  const token = new URLSearchParams(location.search).get("token");
  const query = token ? `?token=${encodeURIComponent(token)}` : "";
  ws = new WebSocket(`${proto}://${location.host}/ws${query}`);
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

async function refreshAnalytics() {
  const host = $("analytics-summary");
  if (!host) return;
  try {
    const hours = Number($("analytics-period")?.value || 24);
    const r = await fetch(`/api/analytics?hours=${hours}`);
    const data = await r.json();
    $("analytics-window").textContent = `${hours}h`;
    const eventReport = document.querySelector('a[href^="/api/reports/event.pdf"]');
    if (eventReport) eventReport.href = `/api/reports/event.pdf?hours=${hours}`;
    const cards = Object.entries(data.forecasts || {}).map(([id, f]) => {
      const cam = state.cameras.find((c) => c.id === id);
      const growth = f.growth_per_min;
      return `<article class="analytics-tile"><b>${esc(cam?.name || id)}</b>
        <span>${f.count} now · ${growth > 0 ? "+" : ""}${growth}/min</span>
        <small>5m ${f.forecast[5]} · 15m ${f.forecast[15]} · 30m ${f.forecast[30]}</small>
        <small>${esc(f.confidence)} · ${data.samples} samples in window</small></article>`;
    });
    const bottlenecks = (data.bottlenecks || []).length;
    cards.push(`<article class="analytics-tile"><b>Pressure signals</b><span>${bottlenecks} possible bottlenecks</span><small>High occupancy with low observed movement</small><small>${esc(data.comparison || "")}</small></article>`);
    (data.growth_alerts || []).forEach((a) => cards.push(`<article class="analytics-tile growth-alert"><b>Rapid crowd growth · ${esc(a.cam_name)}</b><span>+${a.growth_per_min}/min · ${a.count} people</span><small>${esc(a.basis)}</small></article>`));
    (data.queues || []).forEach((q) => cards.push(`<article class="analytics-tile"><b>Queue · ${esc(q.zone)}</b><span>${q.length} people</span><small>${q.wait_minutes == null ? "Wait estimate needs exit history" : `Estimated wait ${q.wait_minutes} min`}</small><small>${esc(q.basis)} · ${esc(state.cameras.find((c) => c.id === q.cam_id)?.name || q.cam_id)}</small></article>`));
    (data.transitions || []).sort((a, b) => b.count - a.count).slice(0, 6).forEach((t) => {
      cards.push(`<article class="analytics-tile"><b>Area movement</b><span>${esc(t.from)} → ${esc(t.to)}</span><small>${t.count} tracked transitions · ${esc(state.cameras.find((c) => c.id === t.cam_id)?.name || t.cam_id)}</small></article>`);
    });
    if (data.performance) cards.push(`<article class="analytics-tile"><b>Performance</b><span>${data.performance.running}/${data.performance.cameras} cameras live</span><small>${data.performance.mean_infer_fps} mean inference FPS · ${data.performance.degraded} degraded</small><small>${data.performance.sample_coverage} samples in this window</small></article>`);
    state.cameras.forEach((cam) => (cam.metrics?.count_lines || []).forEach((line) => {
      cards.push(`<article class="analytics-tile"><b>Gate · ${esc(line.name)}</b><span>${line.entry || 0} in · ${line.exit || 0} out</span><small>Track-based directional crossings</small></article>`);
    }));
    host.innerHTML = cards.join("");
  } catch {
    host.innerHTML = `<div class="empty-zone">Analytics service is temporarily unavailable.</div>`;
  }
}

function syncHeatmapCameras() {
  const select = $("heatmap-camera");
  if (!select) return;
  const key = state.cameras.map((c) => `${c.id}:${c.name}`).join("|");
  if (select.dataset.key !== key) {
    select.dataset.key = key;
    select.innerHTML = state.cameras.map((c) => `<option value="${esc(c.id)}">${esc(c.name)}</option>`).join("");
  }
  if (state.cameras.some((c) => c.id === state.selectedId)) select.value = state.selectedId;
  if (select.value && state.heatmapArchive.camera !== select.value) loadHeatmapHistory();
}

async function loadHeatmapHistory() {
  const camId = $("heatmap-camera").value;
  if (!camId) return;
  state.heatmapArchive.camera = camId;
  state.heatmapArchive.items = [];
  if (state.heatmapArchive.timer) clearInterval(state.heatmapArchive.timer);
  state.heatmapArchive.playing = false;
  $("btn-heatmap-play").textContent = "Play";
  try {
    const r = await fetch(`/api/heatmaps?cam_id=${encodeURIComponent(camId)}&hours=24`);
    if (!r.ok) throw new Error("Heatmap history unavailable");
    state.heatmapArchive.items = await r.json();
    const range = $("heatmap-range");
    range.max = Math.max(0, state.heatmapArchive.items.length - 1);
    range.value = range.max;
    range.disabled = state.heatmapArchive.items.length < 1;
    if (range.disabled) $("heatmap-time").textContent = "Heatmap snapshots are saved every five minutes";
    else renderArchivedHeatmap(Number(range.value));
  } catch { $("heatmap-time").textContent = "Could not load heatmap history"; }
}

function renderArchivedHeatmap(index) {
  const item = state.heatmapArchive.items[index];
  const canvas = $("heatmap-history");
  if (!canvas) return;
  const rect = canvas.getBoundingClientRect(), dpr = DPR();
  if (!rect.width) return;
  canvas.width = Math.round(rect.width * dpr);
  canvas.height = Math.round(rect.height * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  ctx.fillStyle = "#070b13";
  ctx.fillRect(0, 0, rect.width, rect.height);
  if (item) {
    drawHeat(ctx, item.heatmap, rect.width, rect.height);
    $("heatmap-time").textContent = new Date(item.ts * 1000).toLocaleString();
    $("heatmap-range").value = index;
  } else $("heatmap-time").textContent = "No archived heatmaps yet";
}

function toggleHeatmapPlayback() {
  const archive = state.heatmapArchive;
  if (archive.playing) {
    clearInterval(archive.timer); archive.timer = null; archive.playing = false;
    $("btn-heatmap-play").textContent = "Play"; return;
  }
  if (archive.items.length < 2) return;
  archive.playing = true;
  $("btn-heatmap-play").textContent = "Pause";
  archive.timer = setInterval(() => {
    let i = Number($("heatmap-range").value) + 1;
    if (i >= archive.items.length) i = 0;
    renderArchivedHeatmap(i);
  }, 1000);
}

function startCountLineDrawing() {
  const cam = selectedCam();
  if (!cam) { alert("Select a camera first."); return; }
  if (state.mosaic) setMosaic(false);
  if (state.lineDraw.active) { cancelDraw(); return; }
  state.draw.active = false;
  state.draw.pts = [];
  state.lineDraw = { active: true, pts: [] };
  $("stage").classList.add("drawing");
  $("draw-banner").classList.remove("hidden");
  $("area-fields").classList.add("hidden");
  $("gate-fields").classList.remove("hidden");
  $("btn-finish").classList.add("hidden");
  $("btn-undo-pt").classList.add("hidden");
  $("btn-count-line").disabled = true;
  $("btn-new-area").disabled = true;
  $("gate-name").value = `Gate ${(cam.count_lines || []).length + 1}`;
  $("draw-banner-text").textContent = "Click the two ends of the gate line on the video — it saves on the second click.";
  switchInspector("areas");
  drawOverlay();
}

async function saveCountLine(points) {
  const cam = selectedCam();
  if (!cam) return;
  const input = $("gate-name");
  const name = (input && input.value.trim()) || `Gate ${(cam.count_lines || []).length + 1}`;
  const line = { id: crypto.randomUUID().slice(0, 8), name,
    a: points[0], b: points[1], entry_side: "positive" };
  const lines = [...(cam.count_lines || []), line];
  try {
    const res = await fetch(`/api/cameras/${cam.id}/count-lines`, { method: "PUT", headers: { "content-type": "application/json" }, body: JSON.stringify({ lines }) });
    if (!res.ok) throw new Error(await res.text());
    cam.count_lines = lines;
    toastOk("Gate line saved", `${esc(name)} now counts entries and exits.`);
  } catch (err) { alert(`Could not save gate line: ${err.message}`); }
  cancelDraw();
}

async function deleteCountLine(id) {
  const cam = selectedCam();
  if (!cam) return;
  const lines = (cam.count_lines || []).filter((l) => l.id !== id);
  try {
    const res = await fetch(`/api/cameras/${cam.id}/count-lines`, {
      method: "PUT", headers: { "content-type": "application/json" },
      body: JSON.stringify({ lines }),
    });
    if (!res.ok) throw new Error(await res.text());
    cam.count_lines = lines;
    renderGates();
    drawOverlay();
  } catch (err) { alert(`Could not remove gate line: ${err.message}`); }
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
    // The KPI row, the trend chart and the camera card must all read the same
    // number: the consensus estimate when PET and DM-Count agree, otherwise
    // the detector floor.  `m.count` alone is only the detector's number.
    const shown = headlineCount(m).value;
    const running = cam.status === "running";
    total += shown;
    if (running) {
      tracks += m.active_tracks || 0;
      if (m.infer_fps) { fps += m.infer_fps; fpsN += 1; }
    }
    if (!state.hist.cams[cam.id]) state.hist.cams[cam.id] = [];
    pushMax(state.hist.cams[cam.id], shown, 90);
    for (const z of m.zones || []) {
      if (!state.hist.zones[z.id]) state.hist.zones[z.id] = [];
      pushMax(state.hist.zones[z.id], z.count || 0, 90);
      if (running && LEVEL_ORDER[z.level] >= 2) risk += 1;
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
  renderGates();
  renderValidation();
  renderAlerts();
  renderKpis();
  drawTrend();
  renderEngine();
  renderAnalyticsHealth();
  syncHeatmapCameras();
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

let MODEL_PICKER_READY = false;

/* Detector family picker. The registry decides what is switchable and what is
   merely listed: a family whose optional package is missing is shown disabled
   instead of failing halfway through a load. */
async function initModelPicker(attempt = 0) {
  const sel = document.getElementById("model-select");
  const note = document.getElementById("model-note");
  if (!sel) return;
  let data;
  try {
    const r = await fetch("/api/models");
    if (!r.ok) throw new Error(r.statusText);
    data = await r.json();
  } catch {
    if (note) note.textContent = "model registry unavailable";
    return;
  }
  const models = data.models || [];
  if (!data.active && attempt < 20) {
    // Cold start: the detector is still warming, so there is no active model
    // to preselect yet. Try again rather than showing an empty picker.
    if (note) note.textContent = "waiting for the detector to warm up…";
    setTimeout(() => initModelPicker(attempt + 1), 3000);
    return;
  }
  sel.innerHTML = models.map((m) => {
    const missing = m.available === false;
    const suffix = missing ? ` — needs ${m.requires || "package"}` : "";
    return `<option value="${esc(m.name)}"${m.name === data.active ? " selected" : ""}${missing ? " disabled" : ""}>${esc(m.name + suffix)}</option>`;
  }).join("");
  const active = models.find((m) => m.name === data.active);
  if (note && active) {
    note.textContent = `${active.backend} · ${active.license}`
      + (active.available === false ? " · unavailable" : "")
      + (active.commercial_use === false ? " · non-commercial licence" : "");
  }
  sel.onchange = async () => {
    const name = sel.value;
    sel.disabled = true;
    if (note) note.textContent = `loading ${name}…`;
    try {
      const r = await fetch("/api/models", {
        method: "PUT",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error(body.detail || r.statusText);
      }
      if (note) note.textContent = `active · ${name}`;
    } catch (err) {
      if (note) note.textContent = `switch failed: ${err.message}`;
    } finally {
      sel.disabled = false;
    }
  };
}

function renderEngine() {
  if (!MODEL_PICKER_READY) {
    MODEL_PICKER_READY = true;
    initModelPicker();
  }
  const sys = state.sys || {};
  const rows = [
    ["Model", sys.model || "loading…", !sys.load_error],
    ["Device", sys.device === "gpu" ? "CUDA GPU" : "CPU", sys.device === "gpu"],
    ["CUDA", sys.cuda_available ? "available" : "not available", !!sys.cuda_available],
    ["Sources", `${sys.cameras_running || 0} / ${sys.cameras_total || 0}`, true],
    ["Dashboards", String(sys.ws_clients ?? 0), true],
    ["Uptime", fmtUptime(sys.uptime_s || 0), true],
    ["Inference queue", `${(sys.scheduler || {}).queue_depth ?? 0} queued · ${(sys.scheduler || {}).frame_age_p95_ms ?? 0}ms p95`, true],
    ["Dropped frames", String((sys.scheduler || {}).stale_dropped ?? 0), true],
    ["Alert outbox", sys.outbox && sys.outbox.enabled
      ? `${sys.outbox.pending ?? 0} pending · ${sys.outbox.failed ?? 0} failed`
      : "no webhook configured", !!(sys.outbox && sys.outbox.enabled)],
  ];
  $("sys-kv").innerHTML = rows.map(([k, v, good]) =>
    `<dt>${k}</dt><dd class="${good ? "good" : ""}">${esc(v)}</dd>`).join("");
  const profile = $("detector-profile");
  if (profile && document.activeElement !== profile) profile.value = sys.profile || "balanced";
  const note = $("detector-profile-note");
  if (note) note.textContent = sys.profile === "dense"
    ? `Dense profile active · ${sys.imgsz}px · confidence ${sys.confidence}. Expect more CPU/GPU load and some extra false detections.`
    : "Dense mode uses a larger image and lower confidence threshold. It can improve small-person recall, with slower inference and more false detections.";
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
    const hl = headlineCount(m);
    const sel = cam.id === state.selectedId ? "sel" : "";
    const lvl = m.level || "";
    const health = (m.health && m.health.status) || "unknown";
    return `<div class="cam-item ${sel}" data-id="${cam.id}">
      <i class="dot ${esc(cam.status)}" title="input: ${esc(health)}"></i>
      <div class="cam-body">
        <div class="name">${esc(cam.name)}</div>
        <div class="meta"><b>${hl.prefix}${hl.value}</b> ppl · ${esc(health)}${
          m.degraded ? ` · <span class="tag-degraded">▲ ${isLowerBound(m) ? "undercount" : (m.estimation_mode === "dense_consensus" ? "consensus" : "degraded")}</span>` : ""}</div>
        <div class="cam-foot">
          ${lvl ? `<span class="lvl ${lvl}">${lvl}</span>` : ""}
          <canvas class="spark" width="52" height="18"></canvas>
        </div>
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
  const profileSelect = $("camera-profile");
  if (profileSelect) {
    profileSelect.disabled = !cam;
    if (cam && document.activeElement !== profileSelect) profileSelect.value = cam.profile || "global";
  }
  const has = !!cam;
  if (cam && cam.display && typeof cam.display.detections === "boolean") {
    state.overlays.detections = cam.display.detections;
  }
  // The camera we just switched to may still be baking boxes into its JPEG.
  // Ask once per camera+view so the picture matches the selected chip.
  if (cam) {
    const want = (CAMERA_VIEWS[state.camView] || {}).detections;
    const key = `${cam.id}:${state.camView}`;
    if (typeof want === "boolean" && state.viewSyncKey !== key) {
      state.viewSyncKey = key;
      if (cam.display && cam.display.detections !== want) updateDetectionDisplay(want);
    }
  }

  $("stage-empty").classList.toggle("hidden", has || state.mosaic);
  $("wall-wrap").classList.toggle("hidden", state.view !== "wall" || !has);
  $("stage").classList.toggle("hidden", state.mosaic || (state.view === "wall" && has));
  const mosaicHost = $("mosaic");
  if (mosaicHost) mosaicHost.classList.toggle("hidden", !state.mosaic);

  $("cam-title").textContent = cam ? cam.name : "No camera selected";
  $("cam-sub").textContent = cam
    ? `${String(cam.source_type || "").toUpperCase()} · ${cam.status} · ${
      (m.health && m.health.status) || "health unknown"
    }${cam.detail ? " · " + cam.detail : ""}`
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
      const host = $("mosaic");
      if (host) host.style.setProperty("--feed-ar", `${m.frame_size[0]} / ${m.frame_size[1]}`);
    }
    if (state.mosaic) setFeedRaw(cam);
  } else {
    feed.dataset.cam = "";
    feed.removeAttribute("src");
  }

  const hl = headlineCount(m);
  $("hud-count").textContent = hl.prefix + hl.value;
  renderTrust(m, cam);
  const lvlEl = $("hud-level");
  lvlEl.className = "lvl " + (m.level || "IDLE");
  lvlEl.textContent = m.level || (cam ? String(cam.status).toUpperCase() : "IDLE");
  $("hud-fps").textContent = `${m.infer_fps ?? 0} fps`;
  $("hud-src").textContent = cam ? `${cam.source_type.toUpperCase()} · ${cam.id}` : "—";

  sizeOverlay();
}

/* Trust badge: never colour-only. Icon + words + the source of the number,
   so a colour-blind operator still reads the state correctly. */
const TRUST_STATE = {
  detector: { ico: "◎", label: "DETECTOR", tone: "ok" },
  detector_crosscheck: { ico: "◎", label: "DETECTOR · cross-check", tone: "ok" },
  calibrated_density: { ico: "◍", label: "CALIBRATED DENSITY", tone: "warn" },
  calibrated_head_points: { ico: "•", label: "CALIBRATED HEAD POINTS", tone: "warn" },
  degraded_review: { ico: "▲", label: "ANALYTICS DEGRADED · review", tone: "bad" },
  /* Recovered at higher resolution: a floor, not a count. Marked as the most
     severe trust state because the number only ever understates the crowd. */
  detector_lower_bound: { ico: "≥", label: "DETECTOR · LOWER BOUND", tone: "bad" },
  /* PET and DM-Count agree: a better number than the detector floor, still an
     estimate and never a validated count. */
  dense_consensus: { ico: "≈", label: "MODEL CONSENSUS · ESTIMATE", tone: "warn" },
};

/* The headline number: consensus estimate when the dense models agree,
   detector floor when they do not, plain count otherwise. */
function headlineCount(m) {
  if (!m) return { prefix: "", value: 0 };
  if (m.estimation_mode === "dense_consensus" && m.consensus_count != null) {
    return { prefix: "≈", value: Math.round(m.consensus_count) };
  }
  if (isLowerBound(m)) return { prefix: "≥", value: m.count ?? 0 };
  return { prefix: "", value: m.count ?? 0 };
}

function isLowerBound(m) {
  return m && m.estimation_mode === "detector_lower_bound";
}

function renderTrust(m, cam) {
  const el = $("hud-trust");
  if (!el) return;
  if (!cam || !m.estimation_mode) { el.hidden = true; el.innerHTML = ""; return; }
  const s = TRUST_STATE[m.estimation_mode] || { ico: "◌", label: String(m.estimation_mode).toUpperCase(), tone: "warn" };
  const density = m.density_count == null ? "—" : Math.round(m.density_count);
  const points = m.head_point_count == null ? "—" : Math.round(m.head_point_count);
  let detail;
  if (m.estimation_mode === "dense_consensus") {
    const spread = m.estimator_spread_pct == null ? "?" : Math.round(m.estimator_spread_pct);
    detail = `PET ${points} · DM-Count ${density} · agreement ${spread}% · estimate, not a count`;
  } else if (isLowerBound(m)) {
    const primary = m.primary_count == null ? "?" : m.primary_count;
    // Dense cameras are exactly where PET and DM-Count matter, so their
    // independent estimates stay visible next to the detector floor.
    const est = (points !== "—" || density !== "—")
      ? ` · head points ${points} · density ${density}` : "";
    detail = `full frame ${primary} → detail ${m.count ?? 0}${est} · undercount likely`;
  } else if (m.degraded) {
    detail = `head points ${points} · density ${density} · awaiting operator review`;
  } else {
    detail = `${m.count_source} · head points ${points} · density ${density} · ${m.selected_count ?? 0} selected`;
  }
  el.hidden = false;
  el.className = `trust ${s.tone}`;
  el.innerHTML = `<span class="t-ico">${s.ico}</span>` +
    `<span class="t-label">${esc(s.label)}</span>` +
    `<span class="t-src">${esc(detail)}</span>`;
  el.title = m.degraded_reason || "Count source and estimator state";
}

function renderAnalyticsHealth() {
  const host = $("analytics-health");
  const pill = $("health-state");
  if (!host) return;
  const rows = state.cameras.map((cam) => {
    const h = (cam.metrics || {}).analytics_health;
    return { cam, h };
  }).filter((r) => r.h);
  if (!rows.length) {
    host.innerHTML = `<div class="empty-zone">Waiting for the first inference…</div>`;
    if (pill) { pill.textContent = "…"; pill.className = "pill"; }
    return;
  }
  const degraded = rows.filter((r) => r.h.status === "degraded");
  if (pill) {
    pill.textContent = degraded.length ? `${degraded.length} DEGRADED` : "MEASURING";
    pill.className = "pill " + (degraded.length ? "bad" : "ok");
  }
  host.innerHTML = rows.map(({ cam, h }) => {
    const bad = h.status === "degraded";
    const icon = bad ? "▲" : h.status === "healthy" ? "✓" : "…";
    const reasons = (h.reasons || []).join(", ").replace(/_/g, " ");
    const age = h.last_inference_age_s == null ? "no inference yet" : `${h.last_inference_age_s}s ago`;
    const mode = (cam.metrics || {}).estimation_mode || "";
    return `<div class="health-row ${bad ? "bad" : "ok"}">
      <span class="h-ico">${icon}</span>
      <div class="h-body">
        <div class="h-name">${esc(cam.name)}</div>
        <div class="h-meta">${esc(bad ? reasons || "degraded" : `inference ${age}`)} · ${esc(mode.replace(/_/g, " "))}</div>
      </div>
      <span class="h-count">${(cam.metrics || {}).count ?? 0}</span>
    </div>`;
  }).join("");
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

  if (state.overlays.heatmap && m.heatmap) drawHeat(ctx, m.heatmap, W, H, state.heatOpacity);
  // Area view: push the raw picture back so the areas themselves carry the
  // meaning, instead of competing with a busy scene.
  if (cam && state.camView === "areas" && (cam.zones || []).length) {
    ctx.fillStyle = "rgba(3, 8, 16, 0.62)";
    ctx.fillRect(0, 0, W, H);
  }
  if (cam && state.overlays.zones) drawZones(ctx, cam, m, W, H);
  if (cam && state.overlays.flow) drawFlow(ctx, cam, m, W, H);
  if (cam && state.overlays.countLines) drawCountLines(ctx, m, W, H);
  if (state.draw.active) drawDraft(ctx, W, H);
  if (state.lineDraw.active) drawCountLineDraft(ctx, state.lineDraw.pts, W, H);
  if (state.calibrationDraw.active || state.calibrationDraw.pts.length) {
    drawCalibrationPoints(ctx, state.calibrationDraw.pts, W, H);
  }
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

function drawHeat(ctx, hm, W, H, alpha = 0.84) {
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
  ctx.globalAlpha = alpha;
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
  const baseAlpha = ctx.globalAlpha;
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
  ctx.globalAlpha = baseAlpha * 0.55;
  roundRect(ctx, bx, by, w, h, 7);
  ctx.stroke();
  ctx.globalAlpha = baseAlpha;

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

function drawZones(ctx, cam, m, W, H, opts = {}) {
  const zones = cam.zones || [];
  const mets = m.zones || [];
  const isolated = opts.isolated ?? (state.camView === "areas");
  const compact = !!opts.compact;
  const thin = !!opts.thin;
  ctx.__W = W; ctx.__H = H;
  let lane = 0; // stagger labels that would sit under the HUD

  for (const z of zones) {
    const focused = !state.focusZoneId || state.focusZoneId === z.id;
    ctx.save();
    if (!focused) ctx.globalAlpha = 0.22;
    const met = mets.find((x) => x.id === z.id) || {};
    const level = met.level || "OK";
    const color = LEVEL_COLOR[level] || LEVEL_COLOR.OK;

    polyPath(ctx, z.polygon, W, H);
    ctx.fillStyle = hexA(color, isolated ? (level === "OVERCROWDED" ? 0.42 : 0.26)
      : (level === "OVERCROWDED" ? 0.17 : 0.09));
    ctx.fill();


    ctx.save();
    ctx.strokeStyle = hexA(color, 0.85);
    ctx.lineWidth = thin ? 1 : state.focusZoneId === z.id ? 3 : level === "OVERCROWDED" ? 2 : 1.4;
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
    const floor = compact ? 8 : 78;
    let labelY = minY + 4;
    if (labelY < floor) { labelY = floor + lane * 24; lane += 1; }
    chip(ctx, `${z.name}  ${met.count ?? 0}/${cap}`, minX + 4, labelY, color,
      compact ? "" : `${occ.toFixed(0)}% · ${level}`);
    ctx.restore();
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

function drawCountLines(ctx, metrics, W, H) {
  for (const line of metrics.count_lines || []) {
    const [ax, ay] = line.a || [0, 0], [bx, by] = line.b || [1, 1];
    const x1 = ax * W, y1 = ay * H, x2 = bx * W, y2 = by * H;
    const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy) || 1;
    const nx = -dy / len, ny = dx / len;
    const midX = (x1 + x2) / 2, midY = (y1 + y2) / 2;
    const direction = line.entry_side === "negative" ? -1 : 1;
    const tx = midX + nx * direction * 20, ty = midY + ny * direction * 20;
    ctx.save();
    ctx.strokeStyle = "rgba(56, 189, 248, .95)";
    ctx.fillStyle = "rgba(56, 189, 248, .95)";
    ctx.shadowColor = "rgba(56, 189, 248, .7)";
    ctx.shadowBlur = 10;
    ctx.lineWidth = 3;
    ctx.setLineDash([10, 5]);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    ctx.setLineDash([]);
    const ang = Math.atan2(ty - midY, tx - midX), head = 9;
    ctx.beginPath();
    ctx.moveTo(tx, ty);
    ctx.lineTo(tx - head * Math.cos(ang - .55), ty - head * Math.sin(ang - .55));
    ctx.lineTo(tx - head * Math.cos(ang + .55), ty - head * Math.sin(ang + .55));
    ctx.closePath(); ctx.fill();
    chip(ctx, `${line.name || "Count line"}  ↑${line.entry || 0}  ↓${line.exit || 0}`,
      midX + 8, midY + 8, "#38bdf8", "entry / exit");
    ctx.restore();
  }
}

function drawCountLineDraft(ctx, pts, W, H) {
  if (!pts.length) return;
  ctx.save();
  ctx.strokeStyle = "#38bdf8"; ctx.fillStyle = "#38bdf8";
  ctx.lineWidth = 3; ctx.setLineDash([8, 5]);
  ctx.beginPath(); ctx.moveTo(pts[0][0] * W, pts[0][1] * H);
  if (pts[1]) ctx.lineTo(pts[1][0] * W, pts[1][1] * H);
  ctx.stroke(); ctx.setLineDash([]);
  pts.forEach((p) => { ctx.beginPath(); ctx.arc(p[0] * W, p[1] * H, 5, 0, Math.PI * 2); ctx.fill(); });
  const p = pts[pts.length - 1];
  chip(ctx, `${pts.length}/2 points`, p[0] * W + 8, p[1] * H + 8, "#38bdf8", "count line");
  ctx.restore();
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
  if (state.focusZoneId && !zones.some((z) => z.id === state.focusZoneId)) state.focusZoneId = null;
  $("zone-count").textContent = zones.length ? String(zones.length) : "0";

  if (!zones.length) {
    list.innerHTML = `<div class="empty-zone">No areas on this camera yet.<br>
      Press <b>New area</b> and click the corners on the video, or use
      <b>Auto-grid</b> to split the frame evenly. Areas are saved with the
      camera, so the boundary is still there after a restart.</div>`;
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
    </div>      <div class="zone-main">
      <div class="zone-top"><span class="zone-name"></span><span class="zone-occ"></span></div>
      <div class="zone-sub"><span class="lvl">OK</span><span class="fb"></span></div>
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
    occ: el.querySelector(".zone-occ"),
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
  el.onclick = (event) => {
    if (event.target.closest("input, button")) return;
    setZoneFocus(state.focusZoneId === z.id ? null : z.id);
  };
  return row;
}

function updateZoneRow(row, z, met) {
  const level = met.level || "OK";
  const occ = clamp(met.occupancy_pct ?? 0, 0, 100);
  const color = LEVEL_COLOR[level] || LEVEL_COLOR.OK;
  const cap = met.capacity ?? z.capacity;

  row.el.className = `zone-row lvl-${level}${state.focusZoneId === z.id ? " focused" : ""}`;
  row.fg.style.stroke = color;
  row.fg.style.strokeDashoffset = String(RING_C * (1 - occ / 100));
  row.gtxt.textContent = met.count ?? 0;
  row.gtxt.style.color = color;
  row.name.textContent = z.name;
  row.level.className = "lvl " + level;
  row.level.textContent = level;
  row.occ.innerHTML = `<b>${met.count ?? 0}</b> / ${cap} · ${occ.toFixed(0)}%`;

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
    const data = await r.json();
    // Adopt the server's copy immediately so the list does not blank out until
    // the next snapshot arrives.
    cam.zones = data.zones || payload;
    state.zoneRows.clear();
    $("zone-list").innerHTML = "";
    renderZones();
    drawOverlay();
  } catch (e) {
    alert("Saving areas failed: " + e.message);
  }
}

function renderGates() {
  const host = $("gate-list");
  if (!host) return;
  const cam = selectedCam();
  const lines = (cam && cam.count_lines) || [];
  const met = (cam && cam.metrics && cam.metrics.count_lines) || [];
  $("gate-count").textContent = String(lines.length);
  if (!lines.length) {
    host.innerHTML = `<div class="empty-zone">No gate lines yet. A gate line
      counts people walking past an entrance or exit.</div>`;
    return;
  }
  host.innerHTML = lines.map((l) => {
    const m = met.find((x) => x.id === l.id) || met.find((x) => x.name === l.name) || {};
    return `<div class="gate-row" data-id="${esc(l.id)}">
      <span class="g-name">${esc(l.name || "Gate line")}</span>
      <span class="g-io">↑${m.entry ?? l.entry ?? 0} ↓${m.exit ?? l.exit ?? 0}</span>
      <button class="zone-del" data-gate-del="${esc(l.id)}" title="Remove gate line">✕</button>
    </div>`;
  }).join("");
  host.querySelectorAll("[data-gate-del]").forEach((b) => {
    b.onclick = () => deleteCountLine(b.dataset.gateDel);
  });
}

function renderValidation() {
  const cam = selectedCam();
  const source = $("validation-source")?.value || "head_point";
  const m = (cam && cam.metrics) || {};
  const policy = (cam && cam.estimation_policy) || {};
  const stateEl = $("validation-state");
  const note = $("validation-note");
  const sampleBtn = $("btn-validation-sample");
  const promoteBtn = $("btn-validation-promote");
  const revokeBtn = $("btn-validation-revoke");
  if (!stateEl || !note || !sampleBtn || !promoteBtn || !revokeBtn) return;
  const estimate = source === "head_point" ? m.head_point_count : m.density_count;
  const active = policy.active_source || "none";
  const samples = policy.samples ?? 0;
  const mape = policy.mape == null ? "—" : `${(Number(policy.mape) * 100).toFixed(1)}%`;
  stateEl.textContent = active === source ? "ACTIVE" : policy.approved ? "APPROVED" : "ADVISORY";
  stateEl.className = "pill " + (active === source ? "ok" : policy.approved ? "warn" : "");
  note.textContent = cam
    ? `${source === "head_point" ? "PET head points" : "DM-Count density"}: ${estimate == null ? "no current estimate" : Math.round(estimate)} · ${samples} checks · MAPE ${mape}${policy.expires_at ? ` · valid until ${new Date(policy.expires_at * 1000).toLocaleDateString()}` : ""}`
    : "Select a camera to record validation checks.";
  sampleBtn.disabled = !cam || estimate == null;
  promoteBtn.disabled = !cam || !policy.samples;
  revokeBtn.disabled = !cam || active === "none";
}

async function recordValidationSample() {
  const cam = selectedCam();
  const source = $("validation-source").value;
  const manual = Number($("validation-manual").value);
  const estimate = source === "head_point" ? cam?.metrics?.head_point_count : cam?.metrics?.density_count;
  if (!cam || !Number.isFinite(manual) || manual < 0 || estimate == null) {
    alert("Enter a non-negative hand count while the selected estimate is available.");
    return;
  }
  const btn = $("btn-validation-sample");
  btn.disabled = true;
  try {
    const response = await fetch(`/api/cameras/${encodeURIComponent(cam.id)}/validation-samples`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ source, manual_count: manual, model_count: estimate, by: "operator" }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not record validation check");
    cam.estimation_policy = data;
    $("validation-manual").value = "";
    toastOk("Validation recorded", `${data.samples} checks · MAPE ${(data.mape * 100).toFixed(1)}%.`);
  } catch (err) { toastErr("Validation not recorded", err.message); }
  finally { renderValidation(); }
}

async function promoteValidationSource() {
  const cam = selectedCam();
  if (!cam) return;
  const source = $("validation-source").value;
  const btn = $("btn-validation-promote");
  btn.disabled = true;
  try {
    const response = await fetch(`/api/cameras/${encodeURIComponent(cam.id)}/estimation-policy`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ selected_source: source, by: "operator" }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not promote estimate");
    cam.estimation_policy = data;
    toastOk("Dense estimate promoted", `${source.replace("_", " ")} is now selected for this camera.`);
  } catch (err) { toastErr("Promotion blocked", err.message); }
  finally { renderValidation(); }
}

async function revokeValidationSource() {
  const cam = selectedCam();
  if (!cam) return;
  const btn = $("btn-validation-revoke");
  btn.disabled = true;
  try {
    const response = await fetch(`/api/cameras/${encodeURIComponent(cam.id)}/estimation-policy`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ selected_source: "none", by: "operator", reason: "operator review" }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Could not revoke estimate");
    cam.estimation_policy = data;
    toastOk("Dense estimate revoked", "The camera has returned to advisory dense estimates.");
  } catch (err) { toastErr("Revoke failed", err.message); }
  finally { renderValidation(); }
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
  if ((cam.zones || []).length &&
      !confirm("Auto-grid replaces the areas on this camera. Continue?")) return;
  const cols = clamp(parseInt(prompt("Grid columns", "3"), 10) || 3, 1, 6);
  const rows = clamp(parseInt(prompt("Grid rows", "2"), 10) || 2, 1, 6);
  const capacity = Math.max(1, parseInt(prompt("Capacity per area (people)", "20"), 10) || 20);
  await saveZones(gridZones(cols, rows, capacity));
  toastOk("Areas created", `${cols} × ${rows} grid · capacity ${capacity} per area.`);
}

/* ------------------------------------------------------------ drawing --- */
function startAreaDraw() {
  const cam = selectedCam();
  if (!cam) { alert("Select a camera first — an area belongs to a specific view."); return; }
  if (state.mosaic) setMosaic(false);
  if (state.lineDraw.active) cancelDraw();
  state.draw.active = true;
  state.draw.pts = [];
  $("stage").classList.add("drawing");
  $("draw-banner").classList.remove("hidden");
  $("area-fields").classList.remove("hidden");
  $("gate-fields").classList.add("hidden");
  $("btn-finish").classList.remove("hidden");
  $("btn-undo-pt").classList.remove("hidden");
  $("btn-count-line").disabled = true;
  $("btn-new-area").disabled = true;
  $("zone-name").value = `Area ${(cam.zones || []).length + 1}`;
  switchInspector("areas");
  updateDrawBanner();
  drawOverlay();
}

function updateDrawBanner() {
  if (!state.draw.active) return;
  const n = state.draw.pts.length;
  const left = n < 3 ? `Click corners on the video · ${n} placed, at least 3 needed`
    : `${n} corners placed · name it and press Save area`;
  $("draw-banner-text").textContent = left;
  $("btn-finish").disabled = n < 3;
  $("btn-undo-pt").disabled = n === 0;
}

function undoPoint() {
  if (state.lineDraw.active) {
    state.lineDraw.pts.pop();
  } else if (state.draw.active) {
    state.draw.pts.pop();
    updateDrawBanner();
  }
  drawOverlay();
}

function cancelDraw() {
  state.draw.pts = [];
  state.draw.active = false;
  state.lineDraw = { active: false, pts: [] };
  $("stage").classList.remove("drawing");
  $("draw-banner").classList.add("hidden");
  $("btn-count-line").disabled = false;
  $("btn-new-area").disabled = false;
  drawOverlay();
}

function overlayClick(ev) {
  const rect = $("stage").getBoundingClientRect();
  const x = clamp((ev.clientX - rect.left) / rect.width, 0, 1);
  const y = clamp((ev.clientY - rect.top) / rect.height, 0, 1);
  if (state.calibrationDraw.active) {
    state.calibrationDraw.pts.push([+x.toFixed(5), +y.toFixed(5)]);
    const n = state.calibrationDraw.pts.length;
    $("calibration-note").textContent = n < 4
      ? `Point ${n} saved · click ${4 - n} more ground-plane point${n === 3 ? "" : "s"}`
      : "Four points selected · enter their real-world coordinates";
    drawOverlay();
    if (n >= 4) finishCalibrationDrawing();
    return;
  }
  if (state.lineDraw.active) {
    state.lineDraw.pts.push([+x.toFixed(4), +y.toFixed(4)]);
    if (state.lineDraw.pts.length === 2) {
      const points = state.lineDraw.pts.slice();
      state.lineDraw.active = false;
      saveCountLine(points);
    }
    $("draw-banner-text").textContent = state.lineDraw.pts.length === 1
      ? "First end placed · click the other end of the gate line"
      : "Saving gate line…";
    drawOverlay();
    return;
  }
  if (!state.draw.active) {
    const cam = selectedCam();
    const hit = cam && (cam.zones || []).find((zone) => pointInPolygon([x, y], zone.polygon));
    setZoneFocus(hit ? (state.focusZoneId === hit.id ? null : hit.id) : null);
    return;
  }
  state.draw.pts.push([+x.toFixed(4), +y.toFixed(4)]);
  updateDrawBanner();
  drawOverlay();
}

function drawCalibrationPoints(ctx, points, W, H) {
  points.forEach(([x, y], i) => {
    const px = x * W, py = y * H;
    ctx.beginPath(); ctx.arc(px, py, 8, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(34,211,238,.9)"; ctx.fill();
    ctx.strokeStyle = "#06111d"; ctx.lineWidth = 2; ctx.stroke();
    ctx.fillStyle = "#fff"; ctx.font = "bold 12px sans-serif";
    ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText(String(i + 1), px, py);
  });
}

function startCalibrationDrawing() {
  const cam = selectedCam();
  if (!cam) return;
  const size = cam.metrics?.frame_size;
  if (!size || !size[0] || !size[1]) {
    alert("Wait for this camera's first frame before calibrating.");
    return;
  }
  if (state.draw.active || state.lineDraw.active) cancelDraw();
  state.calibrationDraw = { active: true, pts: [] };
  $("calibration-note").textContent = "Click four visible ground points in order around the area.";
  $("stage").classList.add("drawing");
  drawOverlay();
}

async function finishCalibrationDrawing() {
  const cam = selectedCam();
  const points = state.calibrationDraw.pts.slice(0, 4);
  state.calibrationDraw.active = false;
  $("stage").classList.remove("drawing");
  if (!cam || points.length !== 4) return;
  const world = [];
  for (let i = 0; i < points.length; i++) {
    const x = prompt(`Ground point ${i + 1}: real-world X in metres`);
    if (x === null || !x.trim() || !Number.isFinite(Number(x))) { cancelCalibrationDrawing(); return; }
    const y = prompt(`Ground point ${i + 1}: real-world Y in metres`);
    if (y === null || !y.trim() || !Number.isFinite(Number(y))) { cancelCalibrationDrawing(); return; }
    world.push([Number(x), Number(y)]);
  }
  const [width, height] = cam.metrics.frame_size;
  const image = points.map(([x, y]) => [x * width, y * height]);
  try {
    const response = await fetch(`/api/cameras/${encodeURIComponent(cam.id)}/calibration`, {
      method: "PUT", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_points: image, world_points: world, validated_by: "operator" }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Calibration could not be saved");
    $("calibration-note").textContent = `Saved calibration v${data.calibration.version} · ${data.calibration.inliers} points. Verify against measured ground distances.`;
    toastOk("Calibration saved", "Check that the selected points follow the perimeter and match measured distances.");
  } catch (err) {
    $("calibration-note").textContent = `Calibration failed: ${err.message}`;
  }
  drawOverlay();
}

function cancelCalibrationDrawing() {
  state.calibrationDraw = { active: false, pts: [] };
  $("stage").classList.remove("drawing");
  $("calibration-note").textContent = "Calibration cancelled.";
  drawOverlay();
}

function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i], [xj, yj] = polygon[j];
    const crosses = ((yi > point[1]) !== (yj > point[1])) &&
      (point[0] < (xj - xi) * (point[1] - yi) / ((yj - yi) || 1e-9) + xi);
    if (crosses) inside = !inside;
  }
  return inside;
}

function setZoneFocus(id) {
  state.focusZoneId = id;
  renderZones();
  drawOverlay();
}

async function finishZone() {
  if (!state.draw.active) return;
  if (state.draw.pts.length < 3) {
    alert("Place at least 3 corners on the video first.");
    return;
  }
  const cam = selectedCam();
  if (!cam) return;
  const name = $("zone-name").value.trim() || `Area ${(cam.zones || []).length + 1}`;
  const capacity = Math.max(1, parseInt($("zone-cap").value, 10) || 20);
  const zones = [...(cam.zones || []), {
    name, polygon: state.draw.pts.slice(), capacity, warn_pct: 75,
  }];
  const pts = state.draw.pts.slice();
  cancelDraw();
  await saveZones(zones);
  toastOk("Area saved", `${esc(name)} · capacity ${capacity} · ${pts.length} corners.`);
}

/* ------------------------------------------------------------ cameras --- */
function selectCamera(id) {
  if (state.selectedId === id) return;
  state.selectedId = id;
  state.focusZoneId = null;
  state.zoneRows.clear();
  $("zone-list").innerHTML = "";
  $("feed").dataset.cam = "";
  renderAll();
}

function setView(view) {
  if (state.mosaic) setMosaic(false);
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
    const hl = headlineCount(c.metrics);
    if (node) node.textContent = hl.prefix + hl.value;
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
  if (!confirm(`Remove "${cam.name}" from monitoring? An uploaded video used only by this source will also be deleted.`)) return;
  const button = $("btn-del-cam");
  button.disabled = true;
  try {
    const response = await fetch(`/api/cameras/${encodeURIComponent(cam.id)}`, {
      method: "DELETE",
    });
    if (!response.ok) {
      let detail = `HTTP ${response.status}`;
      try { detail = (await response.json()).detail || detail; } catch {}
      throw new Error(detail);
    }
    state.cameras = state.cameras.filter((item) => item.id !== cam.id);
    state.selectedId = state.cameras[0]?.id || null;
    state.zoneRows.clear();
    $("zone-list").innerHTML = "";
    renderAll();
  } catch (error) {
    alert(`Could not remove source: ${error.message}`);
  } finally {
    button.disabled = false;
  }
}

/* ------------------------------------------------------------- chips --- */
function buildKpis() {
  const defs = [
    { key: "people", label: "People, all sources", unit: "" },
    { key: "tracks", label: "Tracked identities", unit: "" },
    { key: "fps", label: "Inference rate", unit: "fps" },
    { key: "risk", label: "Areas at risk", unit: "" },
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
  // Most severe first, then newest: an operator must never have to scan past a
  // low-priority item to find a capacity breach (HSE control-room practice).
  const list = state.alerts.slice(0, 40).sort((a, b) =>
    (LEVEL_ORDER[b.level] ?? 0) - (LEVEL_ORDER[a.level] ?? 0) || b.ts - a.ts).slice(0, 30);
  const acked = state.acked || (state.acked = new Set());
  const open = state.alerts.filter((a) => a.status !== "closed" && !a.ack_by && !acked.has(a.id)).length;
  const badge = $("alert-badge");
  if (badge) {
    badge.textContent = String(open);
    badge.dataset.zero = open ? "0" : "1";
  }
  const key = list.map((a) => `${a.id}:${a.status || "open"}:${a.ack_by || (acked.has(a.id) ? "ack" : "")}`).join("|");
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
    const isAcked = acked.has(a.id) || !!a.ack_by;
    const isClosed = a.status === "closed";
    const ctx = a.context || {};
    const mode = ctx.estimation_mode && ctx.estimation_mode !== "detector"
      ? `<span class="a-mode">${esc(String(ctx.estimation_mode).replace(/_/g, " "))}</span>` : "";
    const evidence = ctx.evidence
      ? `<a class="ack-btn" href="${esc(ctx.evidence)}" target="_blank" rel="noopener">Evidence</a>` : "";
    return `<div class="alert ${lvl} ${isAcked ? "acked" : ""} ${isClosed ? "closed" : ""}">
      <div class="a-top">
        <span class="lvl ${lvl}">${lvl}</span>
        <span class="a-when">${when}</span>
        <span class="a-zone">${esc(a.cam_name || "")} · ${esc(a.zone || "")}</span>
        ${mode}
        ${evidence}
        <button class="ack-btn" data-ack="${esc(a.id)}" ${isAcked ? "disabled" : ""}>
          ${isAcked ? "✓ acknowledged" : "Acknowledge"}
        </button>
        ${isClosed ? `<span class="closed-label">Closed${a.closed_by ? ` · ${esc(a.closed_by)}` : ""}</span>` :
          `<button class="ack-btn" data-close="${esc(a.id)}" ${isAcked ? "" : "disabled"} title="Acknowledge before closing">Close</button>`}
      </div>
      <div class="a-msg">${esc(a.message || "")}</div>
    </div>`;
  }).join("");

  feed.querySelectorAll(".ack-btn[data-ack]").forEach((btn) => {
    btn.onclick = async (event) => {
      event.stopPropagation();
      const id = btn.dataset.ack;
      btn.disabled = true;
      try {
        const res = await fetch(`/api/alerts/${encodeURIComponent(id)}/ack`, {
          method: "POST", headers: { "content-type": "application/json" },
          body: JSON.stringify({ by: "operator" }),
        });
        if (!res.ok) throw new Error(`ack failed (${res.status})`);
        acked.add(id);
        state.alertsKey = "";
        renderAlerts();
      } catch (err) {
        btn.disabled = false;
        btn.textContent = "ack failed — retry";
      }
    };
  });
  feed.querySelectorAll(".ack-btn[data-close]").forEach((btn) => {
    btn.onclick = async (event) => {
      event.stopPropagation();
      const id = btn.dataset.close;
      btn.disabled = true;
      try {
        const res = await fetch(`/api/alerts/${encodeURIComponent(id)}/close`, {
          method: "POST", headers: { "content-type": "application/json" },
          body: JSON.stringify({ by: "operator" }),
        });
        if (!res.ok) throw new Error(`close failed (${res.status})`);
        state.alertsKey = "";
        await refreshAlerts();
      } catch {
        btn.disabled = false;
        btn.textContent = "close failed — retry";
      }
    };
  });
}

function toast(a) {
  notify("bad", "Capacity alert",
    `<b>${esc(a.zone || "")}</b> — ${esc(a.message || "")}`);
}

function toastOk(title, body) { notify("ok", title, body); }
function toastErr(title, body) { notify("err", title, body); }

function notify(kind, title, body) {
  const host = $("toasts");
  if (!host) return;
  const el = document.createElement("div");
  el.className = `toast ${kind}`;
  const ico = kind === "ok" ? "✓" : "⚠";
  el.innerHTML = `
    <div class="t-ico">${ico}</div>
    <div>
      <div class="t-title">${esc(title)}</div>
      <div class="t-body">${body}</div>
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
