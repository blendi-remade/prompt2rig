// Prompt2Rig — prompt → H3 Max Turbo video → live pose (MediaPipe) → Mixamo rig,
// with an optional GVHMR "HQ" pass on the local GPU that swaps in when ready.
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { FBXLoader } from 'three/addons/loaders/FBXLoader.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { FilesetResolver, PoseLandmarker, DrawingUtils } from '@mediapipe/tasks-vision';
import { createStage } from './stage.js';
import { HAND_LIMITS, Rig, OneEuro, Track, trackToClip, blendSnapshots, smplGlobals, aaToQuat } from './retarget.js';

const $ = (s) => document.querySelector(s);
// Mixamo-rigged characters from the three.js examples (loaded from the CDN, not redistributed here).
const THREE_MODELS = 'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r169/examples/models/gltf/';
const CHARACTERS = {
  xbot: { name: 'X Bot', url: THREE_MODELS + 'Xbot.glb' },
  michelle: { name: 'Michelle', url: THREE_MODELS + 'Michelle.glb' },
  soldier: { name: 'Soldier', url: THREE_MODELS + 'Soldier.glb' },
};
const VIDEO_MODEL_LABEL = { h3: 'H3 Max Turbo', flux3: 'FLUX 3 Max' };
const MP_WASM = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm';
const MP_MODEL = (v) => `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_${v}/float16/latest/pose_landmarker_${v}.task`;

const PERFORMER_PROMPT =
  'Full-body photograph of one young adult standing in a relaxed neutral pose facing the camera, arms relaxed at the sides, ' +
  'wearing a fitted plain light-grey t-shirt, black slim pants and white sneakers. The whole body is visible from head to feet ' +
  'with empty space above the head and below the feet, the person is centered. Plain seamless ' +
  'light-grey studio backdrop and floor, flat even soft lighting, sharp focus, 35mm photo.';

// Presets chosen to retarget cleanly: facing the camera, limbs clearly separated from the body.
// Avoid flips, spins, floor work and arms crossing/hiding behind the torso — pose estimators
// lose track of which limb is which, and the error shows on the rig.
const PRESETS = [
  'does an energetic hip hop dance with wide, open arm movements',
  'throws boxing jabs and hooks toward the camera',
  'does jumping jacks',
  'mimes a full golf swing',
  'shoots an imaginary basketball jump shot',
  'celebrates a victory, jumping and pumping both fists in the air',
  'does a stiff zombie walk in place with arms stretched forward',
  'does a karate kata with straight punches and front kicks',
  'waves hello with both arms, then takes a bow',
  'does a big superhero landing pose, then stands up proudly',
];

const IDLE_PROMPT = 'stands in place in a relaxed idle pose, breathing and gently shifting weight from foot to foot';
const MAX_IDLE_STREAK = 3;
// Stream mode = continue-from-last-frame + play only once HQ is ready + idle between prompts.
const chainOn = () => settings.chain || settings.stream;
const waitOn = () => settings.hq && hqOnline && (settings.waitHq || settings.stream);

// Shared staging that keeps clips easy to read for pose estimation.
const STAGING = 'They stay roughly in place facing the camera, with both arms clearly visible and held away from the torso, no turning around, no flips. ' +
  'Their entire body, including both feet, stays visible in frame the whole time. Static locked-off camera, no camera movement, no zoom, no cuts.';
const videoPrompt = (action, fromImage) => fromImage
  ? `The person ${action}. ${STAGING} Plain studio background.`
  : `Wide full-body shot of one person wearing a plain grey t-shirt and dark pants in an empty light-grey studio. The person ${action}. ${STAGING}`;

// ---- settings ------------------------------------------------------------------
const DEFAULTS = {
  stream: false, source: 'performer', chain: false, duration: 5, resolution: '480P', hq: true, waitHq: false,
  rootMotion: true, poseModel: 'heavy', smoothing: 0.5, showSkel: true, orientation: 'portrait', performerUrls: {},
  character: 'xbot', videoModel: 'h3', swap: false, cinematic: false, hands: true,
};
const settings = (() => { try { return { ...DEFAULTS, ...JSON.parse(localStorage.getItem('p2r') || '{}') }; } catch { return { ...DEFAULTS }; } })();
const save = () => { try { localStorage.setItem('p2r', JSON.stringify(settings)); } catch {} };

function bindSetting(id, kind = 'value', onChange) {
  const el = $('#' + id);
  if (kind === 'checked') el.checked = !!settings[id]; else el.value = settings[id];
  el.addEventListener('input', () => {
    settings[id] = kind === 'checked' ? el.checked : (el.type === 'range' ? Number(el.value) : el.value);
    save(); onChange?.(settings[id]);
  });
  onChange?.(settings[id]);
}

// ---- log / status -------------------------------------------------------------
function log(msg, cls = '') {
  const el = $('#log'); const line = document.createElement('div');
  if (cls) line.className = cls;
  line.textContent = `${new Date().toLocaleTimeString([], { hour12: false })}  ${msg}`;
  el.appendChild(line); el.scrollTop = el.scrollHeight;
}
function pill(id, text, state) { const p = $('#pill-' + id); p.textContent = text; p.className = 'pill ' + (state || ''); }
function banner(text) { const b = $('#banner'); b.textContent = text || ''; b.classList.toggle('on', !!text); }

async function api(path, body) {
  const res = await fetch(path, { method: body ? 'POST' : 'GET', headers: { 'Content-Type': 'application/json' }, body: body && JSON.stringify(body) });
  const json = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(json.error || `${path} → ${res.status}`);
  return json;
}

// ---- three.js stage -------------------------------------------------------------
const stage = createStage($('#stage'));
const { scene } = stage;
// Keep the characters centered in the part of the stage the video panel doesn't cover.
new ResizeObserver(() => { const v = $('#vidwrap'); stage.setInset(v.offsetLeft + v.offsetWidth); }).observe($('#vidwrap'));

// ---- character --------------------------------------------------------------------
let model = null, rig = null;

/** Load a Mixamo character: scaled to meters, facing +Z, centered with feet on y=0. */
async function loadModel(src, name) {
  const isFbx = /\.fbx$/i.test(name);
  const obj = await (isFbx ? new FBXLoader() : new GLTFLoader()).loadAsync(src);
  const root = isFbx ? obj : obj.scene;
  if (isFbx) root.scale.setScalar(0.01);         // Mixamo FBX is in centimeters
  root.updateMatrixWorld(true);
  // Retargeting assumes the character faces +Z (toward the camera); some exports face -Z.
  const armX = (side) => {
    let b; root.traverse((o) => { if (!b && o.isBone && o.name.replace(/^mixamorig\d*[:_]?/i, '') === side + 'Arm') b = o; });
    return b?.getWorldPosition(new THREE.Vector3()).x;
  };
  const lx = armX('Left'), rx = armX('Right');
  if (lx != null && rx != null && lx < rx) { root.rotation.y += Math.PI; root.updateMatrixWorld(true); }
  const box = new THREE.Box3().setFromObject(root);
  root.position.y -= box.min.y;
  root.position.x -= (box.min.x + box.max.x) / 2;
  root.position.z -= (box.min.z + box.max.z) / 2;
  root.traverse((o) => { if (o.isMesh) { o.castShadow = true; o.frustumCulled = false; } });
  root.updateMatrixWorld(true);
  return root;
}

async function loadCharacter(src, name = 'X Bot') {
  pill('rig', 'loading ' + name, 'busy');
  const root = await loadModel(src, name);
  root.name = 'Character';
  const newRig = new Rig(root);
  if (model) { scene.remove(model); stage.removeContactShadow(model); }
  scene.add(root); stage.contactShadow(root); model = root; rig = newRig;
  // Takes store bone-order-specific data; re-bake HQ data for the new skeleton.
  for (const t of takes) { t.live = new Track(rig.order.length); t.liveDone = false; if (t.hqRaw) t.hq = bakeSmpl(t.hqRaw); }
  liveState ||= { hip0: null, baseY: null, mpp: null, scale: null };
  liveSnap = null; switchFrom = null;
  pill('rig', name, 'ok');
  log(`character: ${name} (${rig.order.length} bones, ${rig.height.toFixed(2)} m)`);
  resetCast();
}

// ---- character swap: one character at a time, the next one takes over on every loop -------------
// The selected character stays the hidden "driver" that takes are baked for; whichever character
// is on stage copies its pose each frame, so swapping never needs a re-bake.
const cast = {};        // key → { model, rig } for the non-driver characters, loaded on demand
let onStage = null;     // key of the character currently visible (null = the driver itself)

function resetCast() {
  for (const k of Object.keys(cast)) { scene.remove(cast[k].model); stage.removeContactShadow(cast[k].model); delete cast[k]; }
  onStage = null;
  if (model) model.visible = true;
}

async function swapCharacter() {
  const keys = Object.keys(CHARACTERS);
  const shown = onStage || settings.character;
  const next = keys[(keys.indexOf(shown) + 1) % keys.length];
  if (next === settings.character) { showOnStage(null); return; }
  if (!cast[next]) {
    const m = await loadModel(CHARACTERS[next].url, CHARACTERS[next].name);
    if (cast[next]) return;
    m.visible = false; scene.add(m);
    cast[next] = { model: m, rig: new Rig(m) };
  }
  showOnStage(next);
}

function showOnStage(key) {
  if (onStage && cast[onStage]) { cast[onStage].model.visible = false; stage.removeContactShadow(cast[onStage].model); }
  onStage = key;
  model.visible = !key;
  stage.removeContactShadow(model);
  if (key) { cast[key].model.visible = true; stage.contactShadow(cast[key].model); }
  else stage.contactShadow(model);
  pill('rig', key ? CHARACTERS[key].name : (CHARACTERS[settings.character]?.name || 'character'), 'ok');
}

// ---- MediaPipe ---------------------------------------------------------------------
let landmarker = null, vision = null, mpTs = 0;
async function initPose(variant) {
  pill('pose', 'pose: loading ' + variant, 'busy');
  vision ||= await FilesetResolver.forVisionTasks(MP_WASM);
  const make = (delegate) => PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MP_MODEL(variant), delegate },
    runningMode: 'VIDEO', numPoses: 1,
    minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5, minTrackingConfidence: 0.5,
  });
  const old = landmarker; landmarker = null;
  try { landmarker = await make('GPU'); } catch (e) { log('GPU delegate failed, using CPU: ' + e.message, 'err'); landmarker = await make('CPU'); }
  old?.close();
  pill('pose', 'pose: ' + variant, 'ok');
}

// ---- takes --------------------------------------------------------------------------
let takeSeq = 0;
const takes = [];
class Take {
  constructor(prompt) {
    this.id = ++takeSeq; this.prompt = prompt;
    this.videoUrl = null; this.status = 'generating';
    this.live = null; this.liveDone = false;
    this.hq = null; this.hqRaw = null; this.hqStatus = 'off'; this.hqAt = 0;
    this.rootOffset = null; this.timings = {};
    this.ready = new Promise((r) => (this._resolveReady = r));
  }
  get best() { return this.hq || (this.liveDone ? this.live : null); }
}

function renderTakes() {
  const el = $('#takes'); el.innerHTML = '';
  for (const t of [...takes].reverse()) {
    const card = document.createElement('div');
    card.className = 'take' + (t === current ? ' playing' : '');
    const gen = t.status === 'generating' ? '<span class="pill busy"><span class="spin"></span> generating</span>'
      : t.status === 'error' ? '<span class="pill bad">failed</span>'
      : t.status === 'skipped' ? '<span class="pill">skipped</span>'
      : `<span class="pill">${t.timings.gen?.toFixed(1)}s gen</span>`;
    const live = t.liveDone ? '<span class="pill ok">live ✓</span>' : '';
    const hq = { running: '<span class="pill busy"><span class="spin"></span> HQ</span>', done: `<span class="pill ok">HQ ✓ ${t.timings.hq?.toFixed(1)}s</span>`, error: '<span class="pill bad">HQ ✗</span>' }[t.hqStatus] || '';
    card.innerHTML = `<div class="p" title="${escapeHtml(t.prompt)}">#${t.id} · ${t.auto ? '<i style="color:var(--dim)">idle</i> · ' : ''}${escapeHtml(t.prompt)}</div>
      <div class="meta">${gen}${live}${hq}</div>
      <div class="actions"><button class="btn" data-a="play" ${t.videoUrl ? '' : 'disabled'}>▶ Play</button>
      <button class="btn" data-a="glb" ${t.best ? '' : 'disabled'}>⬇ GLB</button></div>`;
    card.querySelector('[data-a=play]').onclick = () => playTake(t);
    card.querySelector('[data-a=glb]').onclick = () => exportTake(t);
    el.appendChild(card);
  }
}
const escapeHtml = (s) => s.replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

// ---- generation ------------------------------------------------------------------------
let genChain = Promise.resolve();
let lastGenerated = null;

let idleStreak = 0;
function submitPrompt(prompt, { auto = false } = {}) {
  prompt = prompt.trim();
  if (!prompt) return;
  if (!auto) {
    idleStreak = 0;
    // A real prompt shouldn't wait behind an idle filler that hasn't started playing: drop it,
    // and chain from the clip before it instead.
    const idle = lastGenerated;
    if (settings.stream && idle?.auto && !idle.started && idle !== current && idle.status === 'ready' && !genBusy) {
      const qi = queue.indexOf(idle); if (qi >= 0) queue.splice(qi, 1);
      idle.status = 'skipped'; lastGenerated = idle.prevGenerated;
      log(`#${idle.id} idle filler skipped for your prompt`);
    }
  }
  const take = new Take(prompt);
  take.auto = auto;
  takes.push(take); renderTakes();
  // Serialized: "continue from last frame" needs the previous clip's final frame.
  genBusy++;
  genChain = genChain.then(() => generate(take)).catch(() => {}).finally(() => genBusy--);
}
let genBusy = 0;

/** Stream mode: keep one clip in the pipeline so the performer never freezes for long. */
function maybeIdle() {
  if (!settings.stream || idleStreak >= MAX_IDLE_STREAK) return;
  const ahead = queue.length + genBusy + takes.filter((t) => t.status === 'ready' && t.hqStatus === 'running' && !t.started).length;
  if (ahead === 0) { idleStreak++; submitPrompt(IDLE_PROMPT, { auto: true }); }
}

async function ensurePerformer(force = false) {
  const o = settings.orientation;
  if (settings.performerUrls[o] && !force) return settings.performerUrls[o];
  banner('Creating performer…');
  const t0 = performance.now();
  const r = await api('/api/image', { prompt: PERFORMER_PROMPT, image_size: o === 'portrait' ? 'portrait_16_9' : 'landscape_16_9' });
  settings.performerUrls[o] = r.image_url; save();
  showPerformer();
  log(`performer image in ${((performance.now() - t0) / 1000).toFixed(1)}s`, 'ok');
  banner('');
  return r.image_url;
}

async function generate(take) {
  const t0 = performance.now();
  try {
    let image_url = null, from = 'text';
    if (chainOn() && lastGenerated?.videoUrl) {
      image_url = await (lastGenerated.lastFrame ||= lastFrameDataUrl(lastGenerated.videoUrl)); from = `take #${lastGenerated.id} last frame`;
    } else if (settings.source === 'performer') {
      image_url = await ensurePerformer(); from = 'performer';
    }
    banner(`Generating #${take.id}…`);
    log(`#${take.id} → ${VIDEO_MODEL_LABEL[settings.videoModel]} (${from}, ${settings.duration}s ${settings.resolution})`);
    const r = await api('/api/video', {
      prompt: videoPrompt(take.prompt, !!image_url), image_url,
      model: settings.videoModel, duration: settings.duration, resolution: settings.resolution, aspect_ratio: settings.orientation === 'portrait' ? '9:16' : '16:9',
    });
    take.videoUrl = r.video_url; take.status = 'ready';
    take.timings.gen = (performance.now() - t0) / 1000;
    log(`#${take.id} video in ${take.timings.gen.toFixed(1)}s (GPU ${r.timings?.inference?.toFixed(2) ?? '?'}s)`, 'ok');
    take.prevGenerated = lastGenerated;
    lastGenerated = take;
    // Grab the final frame now (in parallel with HQ) so the next chained clip can start immediately.
    if (chainOn()) take.lastFrame = lastFrameDataUrl(take.videoUrl).catch(() => null);
    banner('');
    if (settings.hq && hqOnline) runHQ(take);
    renderTakes();
    if (waitOn() && !(settings.stream && held)) {
      // Normally play only once HQ is ready; if the player starves meanwhile, startStarved()
      // plays it early with the live preview and HQ cross-fades in when it lands.
      take.awaitingHq = true;
      if (!current) banner(`#${take.id}: waiting for HQ motion…`);
      take.hqPromise.finally(() => {
        banner('');
        if (take.awaitingHq && take.status !== 'skipped') { take.awaitingHq = false; enqueue(take); }
      });
    } else enqueue(take);
  } catch (e) {
    take.status = 'error'; renderTakes(); banner('');
    log(`#${take.id} failed: ${e.message}`, 'err');
  }
}

/** Grab a clip's final frame as a JPEG data URI (first frame for the next clip). */
async function lastFrameDataUrl(url) {
  const v = document.createElement('video');
  v.crossOrigin = 'anonymous'; v.muted = true; v.preload = 'auto'; v.src = url;
  await new Promise((res, rej) => { v.onloadeddata = res; v.onerror = () => rej(new Error('cannot load video for last frame')); });
  v.currentTime = Math.max(0, v.duration - 0.05);
  await new Promise((res) => (v.onseeked = res));
  const c = document.createElement('canvas'); c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext('2d').drawImage(v, 0, 0);
  v.removeAttribute('src'); v.load();
  return c.toDataURL('image/jpeg', 0.92);
}

// ---- HQ (GVHMR) --------------------------------------------------------------------------
let hqOnline = false;
async function checkHQ() {
  try { const r = await api('/api/hq/health'); hqOnline = !!r.ok; pill('hq', r.ok ? `HQ: ${r.device || 'ready'}` : 'HQ: ' + (r.error || 'not ready'), r.ok ? 'ok' : 'bad'); }
  catch { hqOnline = false; pill('hq', 'HQ worker offline', 'bad'); }
}

function runHQ(take) {
  const t0 = performance.now();
  take.hqStatus = 'running'; renderTakes();
  take.hqPromise = api('/api/hq', { video_url: take.videoUrl, ...(take.handsMethod ? { hands: take.handsMethod } : {}) })
    .then((r) => {
      take.hqRaw = r; take.hq = bakeSmpl(r); take.hqAt = performance.now();
      take.hqStatus = 'done'; take.timings.hq = (performance.now() - t0) / 1000;
      log(`#${take.id} HQ motion in ${take.timings.hq.toFixed(1)}s (${r.n} frames, worker ${r.elapsed?.toFixed(1)}s)`, 'hq');
    })
    .catch((e) => { take.hqStatus = 'error'; log(`#${take.id} HQ failed: ${e.message}`, 'err'); })
    .finally(renderTakes);
}

// Finger tracking from the HQ worker: per hand, 21 wrist-relative keypoints per frame (GVHMR world axes).
const _kp = Array.from({ length: 21 }, () => new THREE.Vector3());
function applyHandsFrame(hands, f, yawFix) {
  // WiLoR's MANO hands are anatomically valid already; joint limits only help MediaPipe's noisy depth.
  HAND_LIMITS.enabled = hands?.method !== 'wilor';
  for (const [src, side] of [['R', 'Right'], ['L', 'Left']]) {
    const h = hands?.[src];
    if (settings.hands !== false && h?.valid[f]) {
      const o = f * 63;
      for (let j = 0; j < 21; j++) _kp[j].set(h.kp[o + j * 3], h.kp[o + j * 3 + 1], h.kp[o + j * 3 + 2]).applyQuaternion(yawFix);
      rig.applyHand(side, _kp);
    } else rig.relaxHand(side);
  }
}

/** GVHMR SMPL params → Track of Mixamo local rotations + grounded hips path. */
function bakeSmpl(r) {
  const { n, fps, global_orient: GO, body_pose: BP, transl: T, leg_length } = r;
  // Heading: turn GVHMR's world so the character faces the viewer the way the person faces the
  // camera (from the in-camera orientation). Chained clips share the camera, so headings line up.
  const fWorld = new THREE.Vector3(0, 0, 1).applyQuaternion(aaToQuat(GO[0], GO[1], GO[2]));
  let targetYaw = 0;
  if (r.incam_global_orient0) {
    const [a, b, c] = r.incam_global_orient0;
    const fc = new THREE.Vector3(0, 0, 1).applyQuaternion(aaToQuat(a, b, c)); // OpenCV camera axes
    targetYaw = Math.atan2(fc.x, -fc.z);                                        // → three.js (x, -y, -z)
  }
  const yawFix = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), targetYaw - Math.atan2(fWorld.x, fWorld.z));
  const scale = rig.legLength / (leg_length || 0.85);
  const track = new Track(rig.order.length);
  const lows = [];
  const d = new THREE.Vector3();
  for (let f = 0; f < n; f++) {
    rig.reset();
    rig.applySmplGlobal(smplGlobals(GO.subarray ? GO.subarray(f * 3, f * 3 + 3) : GO.slice(f * 3, f * 3 + 3), BP.slice(f * 63, f * 63 + 63), yawFix));
    d.set(T[f * 3] - T[0], T[f * 3 + 1] - T[1], T[f * 3 + 2] - T[2]).applyQuaternion(yawFix).multiplyScalar(scale);
    const p = rig.rest.Hips.wp.clone().add(d);
    rig.setHipsWorld(p);
    applyHandsFrame(r.hands, f, yawFix);
    lows.push(rig.lowestFoot());
    track.push(f / fps, rig.snapshot());
  }
  // Ground: the typical lowest foot sits on the floor; never sink below it.
  const sorted = [...lows].sort((a, b) => a - b);
  const offset = rig.floorY - sorted[Math.floor(sorted.length * 0.08)];
  track.p.forEach((p, f) => { p[1] += offset + Math.max(0, rig.floorY - (lows[f] + offset)); });
  rig.reset();
  return track;
}

// ---- playback -----------------------------------------------------------------------------
const video = $('#video');
const overlay = $('#overlay');
const octx = overlay.getContext('2d');
let current = null;
const queue = [];
let liveSnap = null;
let liveState = null;
const worldFilter = new OneEuro(99);
const hipFilter = new OneEuro(2);
let lastRenderedHips = new THREE.Vector3();
let switchFrom = null, switchAt = 0;

function setSmoothing(s) {
  // s=0 → raw, s=1 → very smooth
  worldFilter.minCutoff = hipFilter.minCutoff = THREE.MathUtils.lerp(6, 0.4, s);
  worldFilter.beta = hipFilter.beta = THREE.MathUtils.lerp(1.2, 0.15, s);
}

let held = false; // stream mode: clip ended, nothing ready yet → hold the last pose
function enqueue(take) {
  queue.push(take);
  if (capturing) return; // the capture script decides what's on screen
  // Switch right away unless the current clip is still on its first (live) pass,
  // or we're chaining (then the next clip continues from this one's last frame).
  if (!current || held || (current.liveDone && !chainOn())) playNext();
}

/** Out of HQ-ready clips: start the next generated one now on the live pass instead of freezing. */
function startStarved() {
  const next = takes.find((t) => t.awaitingHq && t.status === 'ready' && !t.started);
  if (!next) return false;
  next.awaitingHq = false;
  log(`#${next.id} starting on live preview (HQ still running)`);
  enqueue(next);
  return true;
}

function playNext() { const t = queue.shift(); if (t) playTake(t); }

function playTake(take) {
  if (!take.videoUrl) return;
  if (current && rig) { switchFrom = rig.snapshot(); switchAt = performance.now(); }
  held = false; take.started = true;
  if (take.hq) take.hqAt = -1e9; // already HQ: no live→HQ cross-fade
  const prev = current;
  current = take;
  const qi = queue.indexOf(take); if (qi >= 0) queue.splice(qi, 1);
  if (!take.rootOffset) {
    take.rootOffset = new THREE.Vector3();
    if (chainOn() && prev && rig) {
      take.rootOffset.set(lastRenderedHips.x - rig.rest.Hips.wp.x, 0, lastRenderedHips.z - rig.rest.Hips.wp.z);
      take.rootOffset.clampLength(0, 2.5);
    }
  }
  if (!take.liveDone) {
    take.live = new Track(rig.order.length);
    liveState = { hip0: null, baseY: null, mpp: null, scale: null };
    worldFilter.reset(); hipFilter.reset(); liveSnap = null;
  }
  $('#vid-empty').style.display = 'none';
  video.style.display = 'block';
  video.src = take.videoUrl;
  video.onloadedmetadata = () => $('#vidwrap').classList.toggle('portrait', video.videoHeight > video.videoWidth);
  lastDetectT = -1;
  video.play().catch(() => {});
  renderTakes();
  maybeIdle();
}

video.addEventListener('ended', () => {
  if (settings.swap && rig) swapCharacter().catch((e) => log('swap failed: ' + e.message, 'err'));
  if (current && !current.liveDone) {
    current.liveDone = current.live.length > 3;
    log(`#${current.id} live pass: ${current.live.length} frames`);
    renderTakes();
  }
  if (queue.length) playNext();
  else if (settings.stream) { held = true; if (!startStarved()) maybeIdle(); } // freeze on the last frame
  else { video.currentTime = 0; video.play().catch(() => {}); }
});

// Per decoded video frame: run pose, draw overlay, record the live track.
const stats = { frames: 0, detections: 0, world: 0, recorded: 0 };
let lastDetectT = -1;
// Called from the render loop: run pose once per new video frame (requestVideoFrameCallback
// stops firing after the source changes in Chromium, so we poll currentTime instead).
function detectIfNewFrame() {
  if (!landmarker || !rig || !current || video.readyState < 2 || video.seeking) return;
  const mediaTime = video.currentTime;
  if (mediaTime === lastDetectT) return;
  if (mediaTime < lastDetectT) liveLoopReset();
  lastDetectT = mediaTime;
  try {
    stats.frames++;
    const ts = Math.max(mpTs + 1, performance.now()); mpTs = ts;
    const res = landmarker.detectForVideo(video, ts);
    stats.detections += res.landmarks?.length ? 1 : 0;
    stats.world += res.worldLandmarks?.length ? 1 : 0;
    drawOverlay(res);
    if (!current.liveDone && res.worldLandmarks?.[0]) {
      liveSnap = solveLive(res, mediaTime);
      current.live.push(mediaTime, liveSnap); stats.recorded++;
    }
  } catch (e) { console.error(e); }
}
function liveLoopReset() { /* video looped or switched: keep filters from smearing across the jump */ worldFilter.reset(); hipFilter.reset(); }

let lastPose2d = null; // latest MediaPipe pose landmarks (normalized video coords) — used by the capture zoom
function drawOverlay(res) {
  if (res.landmarks?.[0]) lastPose2d = res.landmarks[0];
  overlay.width = video.videoWidth; overlay.height = video.videoHeight;
  octx.clearRect(0, 0, overlay.width, overlay.height);
  if (!settings.showSkel || !res.landmarks?.[0]) return;
  const du = new DrawingUtils(octx);
  du.drawConnectors(res.landmarks[0], PoseLandmarker.POSE_CONNECTIONS, { color: '#7cf29a', lineWidth: 3 });
  du.drawLandmarks(res.landmarks[0], { color: '#5ab8ff', fillColor: '#0e0f11', lineWidth: 2, radius: 3 });
}

const _lm = Array.from({ length: 33 }, () => new THREE.Vector3());
function solveLive(res, t) {
  const w = res.worldLandmarks[0], n2 = res.landmarks[0];
  const raw = new Float32Array(99);
  // MediaPipe world: x = image right (subject's left when facing camera), y down, z away from camera.
  w.forEach((p, i) => { raw[i * 3] = p.x; raw[i * 3 + 1] = -p.y; raw[i * 3 + 2] = -p.z; });
  const f = worldFilter.filter(raw, t);
  for (let i = 0; i < 33; i++) _lm[i].set(f[i * 3], f[i * 3 + 1], f[i * 3 + 2]);
  const vis = w.map((p) => p.visibility ?? 1);

  // Root motion from the 2D hip track, converted to meters via torso length.
  const W = video.videoWidth, H = video.videoHeight;
  const px = (i) => new THREE.Vector2(n2[i].x * W, n2[i].y * H);
  const hipPx = px(23).add(px(24)).multiplyScalar(0.5);
  const shPx = px(11).add(px(12)).multiplyScalar(0.5);
  const hf = hipFilter.filter([hipPx.x, hipPx.y], t);
  const midHip = _lm[23].clone().add(_lm[24]).multiplyScalar(0.5);
  const midSh = _lm[11].clone().add(_lm[12]).multiplyScalar(0.5);
  const mpp = midSh.distanceTo(midHip) / Math.max(1, shPx.distanceTo(hipPx));
  const leg = (_lm[23].distanceTo(_lm[25]) + _lm[25].distanceTo(_lm[27]) + _lm[24].distanceTo(_lm[26]) + _lm[26].distanceTo(_lm[28])) / 2;
  const S = liveState;
  S.mpp = S.mpp == null ? mpp : S.mpp + 0.05 * (mpp - S.mpp);
  S.scale = S.scale == null ? rig.legLength / leg : S.scale + 0.05 * (rig.legLength / leg - S.scale);
  if (!S.hip0) S.hip0 = [hf[0], hf[1]];

  rig.reset();
  rig.applyLandmarks(_lm, vis);
  const p = rig.rest.Hips.wp.clone();
  if (settings.rootMotion) {
    p.x += (hf[0] - S.hip0[0]) * S.mpp * S.scale;
    p.y += -(hf[1] - S.hip0[1]) * S.mpp * S.scale;
  }
  rig.setHipsWorld(p);
  if (S.baseY == null) S.baseY = rig.floorY - rig.lowestFoot();
  p.y += S.baseY; rig.setHipsWorld(p);
  rig.clampToFloor();
  return rig.snapshot();
}

// ---- render loop ----------------------------------------------------------------------------
const bufA = { q: null, p: new Float32Array(3) }, bufB = { q: null, p: new Float32Array(3) }, bufC = { q: null, p: new Float32Array(3) };

let lastRaf = 0;
let capturing = false; // frame-exact capture (tools/capture.mjs) drives tick() itself
function frame() {
  requestAnimationFrame(frame);
  lastRaf = performance.now();
  if (!capturing) tick();
}
// Browsers pause rAF in hidden/occluded views; keep pose + playback running from a timer then.
setInterval(() => { if (!capturing && performance.now() - lastRaf > 100) tick(); }, 33);

function tick() {
  detectIfNewFrame();
  if (rig && current) {
    const n = rig.order.length * 4;
    for (const b of [bufA, bufB, bufC]) if (!b.q || b.q.length !== n) b.q = new Float32Array(n);
    const t = video.currentTime;
    const live = current.liveDone ? current.live.sample(t, bufA) : liveSnap;
    const hq = current.hq ? current.hq.sample(t, bufB) : null;
    let snap = hq || live, src = hq ? 'HQ · GVHMR' : live ? 'LIVE · MediaPipe' : '—';
    if (hq && live) {
      const w = Math.min(1, (performance.now() - current.hqAt) / 600);
      if (w < 1) { snap = blendSnapshots(live.q, live.p, hq.q, hq.p, w, bufC); src = 'LIVE → HQ'; }
    }
    if (snap && switchFrom) {
      const w = Math.min(1, (performance.now() - switchAt) / 250);
      if (w < 1) snap = blendSnapshots(switchFrom.q, switchFrom.p, snap.q, snap.p, w, snap === bufC ? bufA : bufC);
      else switchFrom = null;
    }
    if (snap) {
      rig.applySnapshot(snap);
      if (current.rootOffset?.lengthSq()) { const h = rig.hipsWorld().add(current.rootOffset); rig.setHipsWorld(h); }
      rig.hipsWorld(lastRenderedHips);
    }
    const hud = $('#hud-src'); hud.textContent = src; hud.className = 'tag ' + (hq ? 'src-hq' : 'src-live');
    const buf = settings.stream ? `  ·  ready ${queue.length}  gen ${genBusy}  hq ${takes.filter((x) => x.hqStatus === 'running').length}${held ? '  ·  HOLD' : ''}` : '';
    $('#hud-time').textContent = `#${current.id}  ${t.toFixed(2)}s${buf}`;
  }
  if (rig) {
    const shown = onStage ? cast[onStage].rig : rig;
    if (shown !== rig) shown.copyPoseFrom(rig);
    stage.placeContactShadow(shown.model, shown.hipsWorld(), shown.floorY, shown.height);
  }
  stage.update(rig ? lastRenderedHips : null);
}

// ---- export --------------------------------------------------------------------------------
async function exportTake(take) {
  const track = take.best;
  if (!track) return;
  const name = `take${take.id}_${take.prompt.replace(/[^a-z0-9]+/gi, '_').slice(0, 40)}${take.hq ? '_hq' : '_live'}`;
  const clip = trackToClip(rig, track, name);
  const glb = await new GLTFExporter().parseAsync(model, { binary: true, animations: [clip] });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([glb], { type: 'model/gltf-binary' }));
  a.download = name + '.glb'; a.click();
  // Also keep a copy in ./takes (with the source clip URL) for tools/render_blender.py.
  const q = new URLSearchParams({ name, prompt: take.prompt, ...(take.videoUrl?.startsWith('http') ? { source: take.videoUrl } : {}) });
  fetch('/api/takes?' + q, { method: 'POST', body: glb }).then((r) => r.ok && log(`saved takes/${name}.glb`, 'ok')).catch(() => {});
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  log(`exported ${name}.glb`, 'ok');
}

// ---- UI wiring --------------------------------------------------------------------------------
bindSetting('stream', 'checked', (on) => {
  for (const id of ['chain', 'waitHq']) { const el = $('#' + id); el.disabled = on; el.closest('label').style.opacity = on ? 0.5 : 1; }
});
bindSetting('source'); bindSetting('chain', 'checked'); bindSetting('resolution');
bindSetting('videoModel', 'value', (m) => {
  const d = $('#duration'); d.min = m === 'flux3' ? 5 : 2;  // FLUX 3 Max: 5–20s
  if (settings.duration < d.min) { settings.duration = Number(d.min); d.value = d.min; $('#duration-v').textContent = d.min + 's'; save(); }
});
bindSetting('swap', 'checked', (on) => { if (!on && model) showOnStage(null); });
bindSetting('cinematic', 'checked', (on) => stage.setCinematic(on));
bindSetting('character', 'value', (c) => { if (rig && CHARACTERS[c]) loadCharacter(CHARACTERS[c].url, CHARACTERS[c].name).catch((e) => log('character failed: ' + e.message, 'err')); });
bindSetting('duration', 'value', (v) => ($('#duration-v').textContent = v + 's'));
bindSetting('hq', 'checked');
bindSetting('hands', 'checked', () => { if (rig) for (const t of takes) if (t.hqRaw) t.hq = bakeSmpl(t.hqRaw); }); bindSetting('waitHq', 'checked'); bindSetting('rootMotion', 'checked');
bindSetting('showSkel', 'checked');
bindSetting('smoothing', 'value', setSmoothing);
bindSetting('poseModel', 'value', (v) => { if (vision) initPose(v); });

const chips = $('#chips');
for (const p of PRESETS) {
  const c = document.createElement('span'); c.className = 'chip'; c.textContent = p;
  c.onclick = () => submitPrompt(p);
  chips.appendChild(c);
}
$('#go').onclick = () => submitPrompt($('#prompt').value);
$('#prompt').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); submitPrompt($('#prompt').value); } });
$('#performer-new').onclick = () => ensurePerformer(true).catch((e) => { banner(''); log('performer failed: ' + e.message, 'err'); });
function showPerformer() {
  const img = $('#performer-img'), url = settings.performerUrls[settings.orientation];
  img.classList.toggle('portrait', settings.orientation === 'portrait');
  if (url) img.src = url; else img.removeAttribute('src');
  // Before any clip has loaded, shape the empty video panel like the clips that will come.
  if (!video.videoWidth) $('#vidwrap').classList.toggle('portrait', settings.orientation === 'portrait');
}
if (settings.performerUrl) { settings.performerUrls.landscape ||= settings.performerUrl; delete settings.performerUrl; save(); }
bindSetting('orientation', 'value', showPerformer);

$('#btn-char').onclick = () => $('#file-char').click();
$('#file-char').onchange = (e) => {
  const f = e.target.files[0]; if (!f) return;
  const url = URL.createObjectURL(f);
  loadCharacter(url, f.name).catch((err) => { log('character failed: ' + err.message, 'err'); pill('rig', 'character failed', 'bad'); });
};
$('#btn-vid').onclick = () => $('#file-vid').click();
$('#file-vid').onchange = (e) => {
  const f = e.target.files[0]; if (!f) return;
  const take = new Take(`(file) ${f.name}`);
  take.videoUrl = URL.createObjectURL(f); take.status = 'ready'; take.timings.gen = 0;
  takes.push(take); renderTakes(); enqueue(take);
};

// ---- capture zoom: glide the 3D camera onto a hand and zoom the video panel onto the same hand ----
// k ∈ [0,1] is the zoom amount (the capture script ramps it linearly in time); t is the output time.
// Eased with smootherstep, distance/scale interpolated in log space (constant perceived speed), the
// camera arcs around the hand on the way in and drifts slowly while holding.
let zoomBase = null, zoomTgt = null, zoomOrigin = null;
const smootherstep = (x) => x * x * x * (x * (x * 6 - 15) + 10);
function applyCaptureZoom(k, bone, t = 0) {
  const r = onStage ? cast[onStage].rig : rig;
  if (k <= 0.0005) {
    if (zoomBase) { stage.setFocusLock(null); stage.camera.position.copy(zoomBase.pos); stage.controls.target.copy(zoomBase.tgt); zoomBase = null; }
    for (const el of [video, overlay]) el.style.transform = '';
    overlay.style.opacity = ''; zoomTgt = zoomOrigin = null;
    return;
  }
  if (!zoomBase) {
    const off = stage.camera.position.clone().sub(stage.controls.target);
    zoomBase = { pos: stage.camera.position.clone(), tgt: stage.controls.target.clone(), dir: off.clone().normalize(), dist: off.length() };
  }
  const e = smootherstep(Math.min(1, Math.max(0, k)));
  // Target: palm centre, followed smoothly so the hand stays framed while it moves.
  const hand = r.bones[bone].getWorldPosition(new THREE.Vector3());
  const mid = r.bones[bone.replace('Hand', 'HandMiddle2')];
  const want = mid ? hand.lerp(mid.getWorldPosition(new THREE.Vector3()), 0.6) : hand;
  zoomTgt = zoomTgt ? zoomTgt.lerp(want, 0.3) : want.clone();
  const tgt = zoomBase.tgt.clone().lerp(zoomTgt, e);
  // Direction arcs from the wide shot's to a 3/4 close-up angle, plus a slow drift while held.
  // Nearly straight-on, like the performer's camera, so the close-up reads like the video panel.
  const closeDir = new THREE.Vector3(bone.startsWith('Right') ? 0.06 : -0.06, 0.04, 1).normalize();
  const qArc = new THREE.Quaternion().setFromUnitVectors(zoomBase.dir, closeDir);
  const dir = zoomBase.dir.clone().applyQuaternion(new THREE.Quaternion().slerp(qArc, e));
  dir.applyAxisAngle(new THREE.Vector3(0, 1, 0), 0.09 * Math.sin(t * 0.7) * e);
  const dist = Math.exp(Math.log(zoomBase.dist) + (Math.log(0.52) - Math.log(zoomBase.dist)) * e);
  stage.setFocusLock(tgt);
  stage.camera.position.copy(tgt).addScaledVector(dir, dist);
  stage.update(null);
  // Video panel: same easing, zooming on the performer's matching hand (pose landmarks right 16/18/20, left 15/17/19).
  const ids = bone.startsWith('Right') ? [16, 18, 20] : [15, 17, 19];
  if (lastPose2d) {
    // Wrist + knuckle landmarks, nudged past the knuckles toward the fingertips.
    const w = lastPose2d[ids[0]], k1 = lastPose2d[ids[1]], k2 = lastPose2d[ids[2]];
    const kx = (k1.x + k2.x) / 2, ky = (k1.y + k2.y) / 2;
    const c = [kx + 0.6 * (kx - w.x), ky + 0.6 * (ky - w.y)];
    zoomOrigin = zoomOrigin ? [zoomOrigin[0] + 0.3 * (c[0] - zoomOrigin[0]), zoomOrigin[1] + 0.3 * (c[1] - zoomOrigin[1])] : c;
  }
  const o = zoomOrigin || [0.5, 0.35];
  const scale = Math.exp(Math.log(3.2) * e);
  for (const el of [video, overlay]) { el.style.transformOrigin = `${o[0] * 100}% ${o[1] * 100}%`; el.style.transform = `scale(${scale})`; }
  overlay.style.opacity = String(1 - e); // the stick figure would only clutter the close-up
}

// Debug handle for the console.
window.p2r = {
  get current() { return current; }, get rig() { return rig; }, takes, video, stats,
  stage, HAND_LIMITS, rebake() { for (const t of takes) if (t.hqRaw) t.hq = bakeSmpl(t.hqRaw); },
  /** Debug: frame the camera on a bone (e.g. 'RightHand') at distance d. */
  focusBone(name, d = 0.6, dir = [0.15, 0.1, 1]) {
    const r = onStage ? cast[onStage].rig : rig;
    const p = r.bones[name].getWorldPosition(new THREE.Vector3());
    stage.setCinematic(false); stage.controls.target.copy(p);
    stage.camera.position.copy(p).add(new THREE.Vector3(...dir).normalize().multiplyScalar(d));
    stage.setFocusLock?.(p);
  },
  // ---- frame-exact capture: used by tools/capture.mjs ----
  submit(prompt, { duration } = {}) {
    if (duration) { settings.duration = duration; $('#duration').value = duration; $('#duration-v').textContent = duration + 's'; }
    submitPrompt(prompt); return takes[takes.length - 1].id;
  },
  take(id) { const t = takes.find((x) => x.id === id); return t && { id: t.id, status: t.status, hq: t.hqStatus, url: t.videoUrl, duration: t.hq?.duration ?? null }; },
  async beginCapture(id) {
    const t = takes.find((x) => x.id === id);
    if (!t?.hq) throw new Error('take has no HQ motion');
    capturing = true; video.loop = false;
    if (current !== t) playTake(t);
    video.pause();
    await new Promise((r) => (video.readyState >= 2 ? r() : video.addEventListener('loadeddata', r, { once: true })));
    renderTakes();
    return { duration: video.duration, frames: Math.floor(video.duration * 30) };
  },
  /** Show exactly media time t: seek, pose (GVHMR), overlay, camera, render. */
  async renderAt(t, zoom = null) {
    video.pause();
    if (Math.abs(video.currentTime - t) > 1e-4) {
      const seeked = new Promise((r) => video.addEventListener('seeked', r, { once: true }));
      video.currentTime = t; await seeked;
    }
    lastDetectT = -1; tick();
    if (zoom) applyCaptureZoom(zoom.k, zoom.bone || 'RightHand', zoom.t || 0);
    // Two frames so the compositor has painted (with a timeout: hidden tabs don't run rAF).
    await Promise.race([new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))), new Promise((r) => setTimeout(r, 80))]);
  },
  endCapture() { capturing = false; video.play().catch(() => {}); },
  playUrl(u, hq = true, hands, prompt) { const t = new Take(prompt || 'debug: ' + u.slice(-24)); t.videoUrl = u; t.status = 'ready'; t.handsMethod = hands; takes.push(t); if (hq && hqOnline) runHQ(t); enqueue(t); return { id: t.id }; },
};

// ---- boot --------------------------------------------------------------------------------------
frame();
const startChar = CHARACTERS[settings.character] || CHARACTERS.xbot;
loadCharacter(startChar.url, startChar.name).catch((e) => { log('character failed: ' + e.message, 'err'); pill('rig', 'character failed', 'bad'); });
initPose(settings.poseModel).catch((e) => { log('pose model failed: ' + e.message, 'err'); pill('pose', 'pose failed', 'bad'); });
checkHQ(); setInterval(checkHQ, 10000);
log('ready — type a motion and hit Generate');
