// The 3D stage: studio lighting, reflective floor, contact shadows, cinematic camera.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { Reflector } from 'three/addons/objects/Reflector.js';

const FLOOR_COLOR = 0x15171b;
const FOG_COLOR = 0x111216;

function gradientTexture(top, bottom) {
  const c = document.createElement('canvas'); c.width = 4; c.height = 512;
  const g = c.getContext('2d');
  const grad = g.createLinearGradient(0, 0, 0, 512);
  grad.addColorStop(0, top); grad.addColorStop(0.62, bottom); grad.addColorStop(1, bottom);
  g.fillStyle = grad; g.fillRect(0, 0, 4, 512);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

function blobTexture() {
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d');
  const grad = g.createRadialGradient(64, 64, 0, 64, 64, 64);
  grad.addColorStop(0, 'rgba(0,0,0,0.85)'); grad.addColorStop(0.45, 'rgba(0,0,0,0.45)'); grad.addColorStop(1, 'rgba(0,0,0,0)');
  g.fillStyle = grad; g.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

export function createStage(container) {
  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(2, devicePixelRatio));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  container.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  scene.background = gradientTexture('#262a32', '#111216');
  scene.fog = new THREE.Fog(FOG_COLOR, 9, 26);
  // Soft studio reflections on the characters' materials.
  const pmrem = new THREE.PMREMGenerator(renderer);
  scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
  scene.environmentIntensity = 0.45;

  const camera = new THREE.PerspectiveCamera(32, 1, 0.05, 100);
  camera.position.set(0.4, 1.45, 5.4);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0.95, 0);
  controls.enableDamping = true;
  controls.maxPolarAngle = Math.PI * 0.495; // stay above the floor

  // Lights: warm key with soft shadows, cool rim from behind, low fill.
  scene.add(new THREE.HemisphereLight(0xdfe6ff, 0x16181c, 0.55));
  const key = new THREE.DirectionalLight(0xfff1e0, 2.6);
  key.position.set(-2.8, 5.5, 4.2); key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  key.shadow.bias = -0.0004; key.shadow.normalBias = 0.02;
  Object.assign(key.shadow.camera, { left: -6, right: 6, top: 6, bottom: -6, near: 0.5, far: 20 });
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x8fb8ff, 2.2);
  rim.position.set(3.5, 4, -4.5);
  scene.add(rim);
  const fill = new THREE.DirectionalLight(0xffffff, 0.35);
  fill.position.set(3, 1.5, 4);
  scene.add(fill);

  // Floor: a mirror underneath a mostly-opaque matte surface → a subtle, soft reflection.
  const mirror = new Reflector(new THREE.CircleGeometry(40, 64), {
    color: 0x6f747c,
    textureWidth: Math.round(container.clientWidth * renderer.getPixelRatio() * 0.5) || 1024,
    textureHeight: Math.round(container.clientHeight * renderer.getPixelRatio() * 0.5) || 1024,
  });
  mirror.rotation.x = -Math.PI / 2; mirror.position.y = -0.001;
  scene.add(mirror);
  const floor = new THREE.Mesh(
    new THREE.CircleGeometry(40, 64),
    new THREE.MeshStandardMaterial({ color: FLOOR_COLOR, roughness: 0.9, metalness: 0, transparent: true, opacity: 0.9 }),
  );
  floor.rotation.x = -Math.PI / 2; floor.receiveShadow = true;
  scene.add(floor);
  const grid = new THREE.GridHelper(40, 80, 0x3a3f47, 0x262a30);
  grid.material.transparent = true; grid.material.opacity = 0.35; grid.position.y = 0.002;
  scene.add(grid);

  // Contact shadows: a soft dark blob under each character's hips.
  const blobTex = blobTexture();
  const blobs = new Map();
  function contactShadow(model) {
    const m = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: blobTex, transparent: true, depthWrite: false, opacity: 0.8 }));
    m.rotation.x = -Math.PI / 2; m.position.y = 0.004; m.renderOrder = 1;
    scene.add(m); blobs.set(model, m);
    return m;
  }
  function removeContactShadow(model) { const m = blobs.get(model); if (m) { scene.remove(m); blobs.delete(model); } }
  /** hips: world position; floorY: character's floor; size: character height. */
  function placeContactShadow(model, hips, floorY, size) {
    const m = blobs.get(model); if (!m) return;
    const lift = Math.max(0, hips.y - floorY - size * 0.52);          // > 0 while airborne
    const k = Math.max(0.25, 1 - lift * 1.6);
    m.position.set(hips.x, floorY + 0.004, hips.z);
    m.scale.setScalar(size * 0.55 * (0.8 + 0.2 * k));
    m.material.opacity = 0.8 * k;
  }

  // Camera: follows a focus point; optional slow cinematic arc + push-in.
  const state = { cinematic: false, width: 1.4, inset: 0, pausedUntil: 0, t0: performance.now(), focus: new THREE.Vector3(0, 0.95, 0) };
  controls.addEventListener('start', () => { state.pausedUntil = performance.now() + 6000; });
  /** Distance that fits a ~2.1m-tall group `state.width` meters wide into the visible area. */
  const baseDistance = () => {
    const w = container.clientWidth - state.inset, h = container.clientHeight || 1;
    const vHalf = THREE.MathUtils.degToRad(camera.fov / 2);
    const hHalf = Math.atan(Math.tan(vHalf) * Math.max(0.3, w / h));
    return Math.max(4.6, 1.2 / Math.tan(vHalf), (state.width / 2 + 0.5) / Math.tan(hHalf));
  };
  // Shift the projection so the action is centered in the area not covered by the video panel.
  function applyInset() {
    // filmOffset shifts the frustum by filmOffset / (filmWidth · aspect · 2·tan(fov/2)) of its width.
    const w = container.clientWidth || 1;
    const f = state.inset / 2 / w;
    camera.filmOffset = -f * camera.getFilmWidth() * camera.aspect * 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2));
    camera.updateProjectionMatrix();
  }
  function reframe() {  // one-off move to the right distance (doesn't fight the user afterwards)
    if (state.cinematic) return;
    const dir = camera.position.clone().sub(controls.target).normalize();
    camera.position.copy(controls.target).addScaledVector(dir, baseDistance());
  }
  function fitFog() { const d = baseDistance(); scene.fog.near = d + 3; scene.fog.far = d + 20; }

  function update(focus) {
    if (state.lock) { controls.target.copy(state.lock); controls.update(); renderer.render(scene, camera); return; }
    if (focus) state.focus.lerp(new THREE.Vector3(focus.x, 0.95, focus.z), 0.04);
    const now = performance.now();
    if (state.cinematic && now > state.pausedUntil) {
      const t = (now - state.t0) / 1000;
      const az = 0.5 * Math.sin((t / 26) * Math.PI * 2);                   // ±29° arc around the front
      const dist = baseDistance() * (0.86 + 0.14 * Math.cos((t / 19) * Math.PI * 2)); // slow push-in / pull-out
      const h = 1.15 + 0.3 * Math.sin((t / 31) * Math.PI * 2);
      const want = new THREE.Vector3(state.focus.x + Math.sin(az) * dist, h, state.focus.z + Math.cos(az) * dist);
      camera.position.lerp(want, 0.03);
      controls.target.lerp(state.focus, 0.05);
    } else {
      const delta = state.focus.clone().sub(controls.target).multiplyScalar(0.04);
      controls.target.add(delta); camera.position.add(delta);
    }
    controls.update();
    renderer.render(scene, camera);
  }

  function resize() {
    const { clientWidth: w, clientHeight: h } = container;
    renderer.setSize(w, h); camera.aspect = w / h; applyInset();
    mirror.getRenderTarget().setSize(Math.round(w * renderer.getPixelRatio() * 0.5), Math.round(h * renderer.getPixelRatio() * 0.5));
  }
  new ResizeObserver(resize).observe(container);
  resize();

  return {
    renderer, scene, camera, controls, update, contactShadow, removeContactShadow, placeContactShadow,
    setCinematic(on) { state.cinematic = on; state.t0 = performance.now(); },
    /** Width in meters of what should stay in frame (solo ≈ 1.4, crowd wider). */
    /** Pin the camera target (debug close-ups); null to release. */
    setFocusLock(p) { state.lock = p ? p.clone() : null; },
    setFraming(widthMeters) { state.width = widthMeters; reframe(); fitFog(); },
    /** Pixels on the left covered by the video panel. */
    setInset(px) { if (Math.abs(px - state.inset) > 2) { state.inset = px; applyInset(); reframe(); fitFog(); } },
  };
}
