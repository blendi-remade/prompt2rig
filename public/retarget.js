// Retargeting onto a Mixamo skeleton.
//
// Two sources drive the same Rig:
//   • MediaPipe world landmarks (live, per frame)  → applyLandmarks()
//   • SMPL joint rotations from GVHMR (HQ, baked)   → applySmplGlobal()
// Both end in local bone quaternions + a hips world position, captured with
// snapshot() so takes can be sampled, blended and exported.
import * as THREE from 'three';

const V3 = THREE.Vector3;
const Q = THREE.Quaternion;

export const BONES = [
  'Hips', 'Spine', 'Spine1', 'Spine2', 'Neck', 'Head',
  'LeftShoulder', 'LeftArm', 'LeftForeArm', 'LeftHand',
  'RightShoulder', 'RightArm', 'RightForeArm', 'RightHand',
  'LeftUpLeg', 'LeftLeg', 'LeftFoot', 'LeftToeBase',
  'RightUpLeg', 'RightLeg', 'RightFoot', 'RightToeBase',
];

// SMPL body joint index → Mixamo bone (GVHMR's 22 body joints).
export const SMPL_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19];
export const SMPL_TO_MIXAMO = [
  'Hips', 'LeftUpLeg', 'RightUpLeg', 'Spine', 'LeftLeg', 'RightLeg', 'Spine1', 'LeftFoot', 'RightFoot',
  'Spine2', 'LeftToeBase', 'RightToeBase', 'Neck', 'LeftShoulder', 'RightShoulder', 'Head',
  'LeftArm', 'RightArm', 'LeftForeArm', 'RightForeArm', 'LeftHand', 'RightHand',
];

// MediaPipe Pose landmark indices.
const MP = {
  nose: 0, lEye: 2, rEye: 5, lEar: 7, rEar: 8,
  lShoulder: 11, rShoulder: 12, lElbow: 13, rElbow: 14, lWrist: 15, rWrist: 16,
  lPinky: 17, rPinky: 18, lIndex: 19, rIndex: 20,
  lHip: 23, rHip: 24, lKnee: 25, rKnee: 26, lAnkle: 27, rAnkle: 28,
  lHeel: 29, rHeel: 30, lFoot: 31, rFoot: 32,
};

const FOOT_POINTS = ['LeftToe_End', 'RightToe_End', 'LeftToeBase', 'RightToeBase', 'LeftFoot', 'RightFoot'];

const shortName = (n) => n.replace(/^mixamorig\d*[:_]?/i, '');
const _m4 = new THREE.Matrix4();
const _q1 = new Q(), _q2 = new Q(), _q3 = new Q();
const _v1 = new V3(), _v2 = new V3();

/** Orthonormal frame: X along `x`, Y as close to `yHint` as possible, Z = X × Y. */
export function basisQuat(x, yHint, out = new Q()) {
  const X = x.clone().normalize();
  const Z = new V3().crossVectors(X, yHint).normalize();
  const Y = new V3().crossVectors(Z, X);
  _m4.makeBasis(X, Y, Z);
  return out.setFromRotationMatrix(_m4);
}

const smoothstep = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

export class Rig {
  constructor(model) {
    this.model = model;
    this.bones = {};
    model.traverse((o) => {
      if (o.isBone) { const s = shortName(o.name); if (!(s in this.bones)) this.bones[s] = o; }
    });
    for (const need of ['Hips', 'Spine', 'LeftArm', 'RightArm', 'LeftUpLeg', 'RightUpLeg']) {
      if (!this.bones[need]) throw new Error(`Not a Mixamo rig: bone "${need}" not found`);
    }
    this.order = BONES.filter((n) => this.bones[n]);
    this.hips = this.bones.Hips;

    model.updateMatrixWorld(true);
    this.rest = {};
    for (const [n, b] of Object.entries(this.bones)) {
      this.rest[n] = {
        q: b.quaternion.clone(), p: b.position.clone(),
        wq: b.getWorldQuaternion(new Q()), wp: b.getWorldPosition(new V3()),
      };
    }
    const R = (n) => this.rest[n].wp;
    const chestTop = this.rest.Neck ? R('Neck') : R('Spine2') || R('Spine');
    this.restHipBasis = basisQuat(R('LeftUpLeg').clone().sub(R('RightUpLeg')), chestTop.clone().sub(R('Hips')));
    this.restChestBasis = basisQuat(R('LeftArm').clone().sub(R('RightArm')), chestTop.clone().sub(R('Hips')));
    this.restForward = new V3(0, 0, 1).applyQuaternion(this.restHipBasis);

    // Floor = lowest point of the skinned mesh at rest; per-bone clearance above it.
    const box = new THREE.Box3().setFromObject(model);
    this.floorY = box.min.y;
    this.height = box.max.y - box.min.y;
    this.footClearance = {};
    for (const n of FOOT_POINTS) if (this.rest[n]) this.footClearance[n] = R(n).y - this.floorY;

    this.legLength = R('LeftUpLeg').distanceTo(R('LeftLeg')) + R('LeftLeg').distanceTo(R('LeftFoot'));

    this.child = {
      LeftArm: 'LeftForeArm', LeftForeArm: 'LeftHand', LeftHand: 'LeftHandMiddle1',
      RightArm: 'RightForeArm', RightForeArm: 'RightHand', RightHand: 'RightHandMiddle1',
      LeftUpLeg: 'LeftLeg', LeftLeg: 'LeftFoot', LeftFoot: 'LeftToeBase',
      RightUpLeg: 'RightLeg', RightLeg: 'RightFoot', RightFoot: 'RightToeBase',
    };
  }

  // ---- low-level ------------------------------------------------------------
  reset() {
    for (const [n, b] of Object.entries(this.bones)) { b.quaternion.copy(this.rest[n].q); b.position.copy(this.rest[n].p); }
    this.model.updateMatrixWorld(true);
  }

  setWorldQuat(name, wq) {
    const b = this.bones[name];
    if (!b) return;
    b.parent.getWorldQuaternion(_q3);
    b.quaternion.copy(_q3.invert().multiply(wq));
    b.updateMatrixWorld(true);
  }

  setHipsWorld(pos) {
    this.hips.position.copy(this.hips.parent.worldToLocal(pos.clone()));
    this.hips.updateMatrixWorld(true);
  }

  hipsWorld(out = new V3()) { return this.hips.getWorldPosition(out); }

  /** Lowest world y of the feet, corrected for each joint's rest height above the floor. */
  lowestFoot() {
    let lo = Infinity;
    for (const [n, c] of Object.entries(this.footClearance)) {
      lo = Math.min(lo, this.bones[n].getWorldPosition(_v1).y - c);
    }
    return lo;
  }

  /** Raise the hips so no foot goes below the floor. Returns the lift applied. */
  clampToFloor(extraLift = 0) {
    const lift = Math.max(0, this.floorY - this.lowestFoot()) + extraLift;
    if (lift) { const p = this.hipsWorld(); p.y += lift; this.setHipsWorld(p); }
    return lift;
  }

  // Pose state as plain arrays: local quats in `order`, plus hips world position.
  snapshot(out) {
    const n = this.order.length;
    out = out || { q: new Float32Array(n * 4), p: new Float32Array(3) };
    this.order.forEach((name, i) => this.bones[name].quaternion.toArray(out.q, i * 4));
    this.hipsWorld(_v2).toArray(out.p);
    return out;
  }

  applySnapshot(s) {
    this.order.forEach((name, i) => this.bones[name].quaternion.fromArray(s.q, i * 4));
    this.model.updateMatrixWorld(true);
    this.setHipsWorld(_v2.fromArray(s.p));
  }

  // ---- source A: MediaPipe world landmarks -----------------------------------
  /**
   * `lm` = 33 THREE.Vector3 in three.js space (x = subject's left, y up, z toward camera),
   * hip-centred meters. `vis` = per-landmark visibility.
   */
  applyLandmarks(lm, vis) {
    const P = (i) => lm[i];
    const mid = (a, b) => P(a).clone().add(P(b)).multiplyScalar(0.5);
    const ok = (...ids) => ids.every((i) => (vis?.[i] ?? 1) > 0.35);
    const dir = (a, b) => P(b).clone().sub(P(a));

    const midHip = mid(MP.lHip, MP.rHip);
    const midSh = mid(MP.lShoulder, MP.rShoulder);
    const up = midSh.clone().sub(midHip);

    const qHip = basisQuat(dir(MP.rHip, MP.lHip), up);
    const qChest = basisQuat(dir(MP.rShoulder, MP.lShoulder), up);
    const dHip = qHip.multiply(_q1.copy(this.restHipBasis).invert());
    const dChest = qChest.multiply(_q1.copy(this.restChestBasis).invert());

    // Head frame from ears + eyes (eye→ear line is roughly level).
    const earX = dir(MP.rEar, MP.lEar);
    const fwd = mid(MP.lEye, MP.rEye).sub(mid(MP.lEar, MP.rEar));
    const qHead = basisQuat(earX, new V3().crossVectors(fwd, earX));
    let dHead = qHead.multiply(_q1.copy(this.restChestBasis).invert());
    if (!ok(MP.lEar, MP.rEar)) dHead = dChest.clone();

    const worldFrom = (delta, name) => delta.clone().multiply(this.rest[name].wq);
    this.setWorldQuat('Hips', worldFrom(dHip, 'Hips'));
    this.setWorldQuat('Spine', worldFrom(dHip.clone().slerp(dChest, 0.35), 'Spine'));
    if (this.bones.Spine1) this.setWorldQuat('Spine1', worldFrom(dHip.clone().slerp(dChest, 0.7), 'Spine1'));
    if (this.bones.Spine2) this.setWorldQuat('Spine2', worldFrom(dChest, 'Spine2'));
    if (this.bones.Neck) this.setWorldQuat('Neck', worldFrom(dChest.clone().slerp(dHead, 0.4), 'Neck'));
    if (this.bones.Head) this.setWorldQuat('Head', worldFrom(dHead, 'Head'));

    const fwdChest = new V3(0, 0, 1).applyQuaternion(dChest).applyQuaternion(this.restChestBasis);
    const fwdHip = new V3(0, 0, 1).applyQuaternion(dHip).applyQuaternion(this.restHipBasis);

    for (const side of ['Left', 'Right']) {
      const L = side === 'Left';
      const sh = L ? MP.lShoulder : MP.rShoulder, el = L ? MP.lElbow : MP.rElbow, wr = L ? MP.lWrist : MP.rWrist;
      const hp = L ? MP.lHip : MP.rHip, kn = L ? MP.lKnee : MP.rKnee, an = L ? MP.lAnkle : MP.rAnkle, ft = L ? MP.lFoot : MP.rFoot;
      const ix = L ? MP.lIndex : MP.rIndex, pk = L ? MP.lPinky : MP.rPinky;

      if (ok(sh, el)) this.aimLimb(`${side}Arm`, dir(sh, el), dir(el, wr), this.restForward, fwdChest);
      if (ok(el, wr)) this.aim(`${side}ForeArm`, dir(el, wr));
      if (ok(wr, ix, pk)) this.aimHand(`${side}Hand`, mid(ix, pk).sub(P(wr)), dir(pk, ix), 0.7);
      if (ok(hp, kn)) this.aimLimb(`${side}UpLeg`, dir(hp, kn), dir(kn, an), this.restForward.clone().negate(), fwdHip.clone().negate());
      if (ok(kn, an)) this.aim(`${side}Leg`, dir(kn, an));
      if (ok(an, ft)) this.aim(`${side}Foot`, dir(an, ft));
    }
  }

  /** Swing the bone so its rest bone-axis points along `target` (keeps inherited twist). */
  aim(name, target, weight = 1) {
    const b = this.bones[name], c = this.bones[this.child[name]];
    if (!b || !c || target.lengthSq() < 1e-8) return;
    const cur = c.getWorldPosition(_v1).sub(b.getWorldPosition(_v2)).normalize();
    const delta = new Q().setFromUnitVectors(cur, target.clone().normalize());
    if (weight < 1) delta.slerp(new Q(), 1 - weight);
    this.setWorldQuat(name, delta.multiply(b.getWorldQuaternion(_q2)));
  }

  /**
   * Upper arm / thigh: aim along `upper`, and use the bend plane (direction the
   * lower segment folds toward) to recover twist so elbows and knees hinge the
   * right way. Fades to plain swing when the limb is nearly straight.
   */
  aimLimb(name, upper, lower, restBendDir, fallbackBend) {
    const b = this.bones[name], c = this.bones[this.child[name]];
    if (!b || !c) return;
    const u = upper.clone().normalize();
    const l = lower.clone().normalize();
    const bend = l.clone().sub(u.clone().multiplyScalar(l.dot(u)));
    const bendAmt = bend.length();

    // Swing-only solution.
    this.aim(name, u);
    if (bendAmt < 0.15) return;
    const swingLocal = b.quaternion.clone();

    // Bend-plane solution: absolute, from rest frame (bone axis, rest bend dir).
    const restAxis = this.rest[this.child[name]].wp.clone().sub(this.rest[name].wp).normalize();
    const restBend = restBendDir.clone().sub(restAxis.clone().multiplyScalar(restBendDir.dot(restAxis)));
    if (restBend.lengthSq() < 1e-6) return;
    const qRest = basisQuat(restAxis, restBend);
    const qNow = basisQuat(u, bend.normalize());
    const wq = qNow.multiply(qRest.invert()).multiply(this.rest[name].wq);
    this.setWorldQuat(name, wq);
    const planeLocal = b.quaternion.clone();

    const w = smoothstep(0.15, 0.45, bendAmt);
    b.quaternion.copy(swingLocal.slerp(planeLocal, w));
    b.updateMatrixWorld(true);
    void fallbackBend;
  }

  /** Hand: aim at knuckles, roll from the pinky→index line. */
  aimHand(name, along, across, weight = 1) {
    const b = this.bones[name];
    const side = name.startsWith('Left') ? 'Left' : 'Right';
    const idx = this.rest[`${side}HandIndex1`], pky = this.rest[`${side}HandPinky1`], mid = this.rest[`${side}HandMiddle1`];
    if (!b || !idx || !pky || !mid) { this.aim(name, along, weight); return; }
    const before = b.quaternion.clone();
    const restAlong = mid.wp.clone().sub(this.rest[name].wp);
    const restAcross = idx.wp.clone().sub(pky.wp);
    const qRest = basisQuat(restAlong, new V3().crossVectors(restAlong, restAcross).cross(restAlong));
    const qNow = basisQuat(along, new V3().crossVectors(along, across).cross(along));
    this.setWorldQuat(name, qNow.multiply(qRest.invert()).multiply(this.rest[name].wq));
    if (weight < 1) { b.quaternion.copy(before.slerp(b.quaternion, weight)); b.updateMatrixWorld(true); }
  }

  // ---- source B: SMPL (GVHMR) -------------------------------------------------
  /** `G` = 22 global joint quaternions (three.js frame), relative to the SMPL T-pose. */
  applySmplGlobal(G) {
    for (let j = 0; j < SMPL_TO_MIXAMO.length; j++) {
      const name = SMPL_TO_MIXAMO[j];
      if (!this.bones[name] || !G[j]) continue;
      this.setWorldQuat(name, _q1.copy(G[j]).multiply(this.rest[name].wq));
    }
  }
}

// ---- SMPL forward kinematics ---------------------------------------------------
export function aaToQuat(x, y, z, out = new Q()) {
  const a = Math.hypot(x, y, z);
  if (a < 1e-8) return out.set(0, 0, 0, 1);
  return out.setFromAxisAngle(_v1.set(x / a, y / a, z / a), a);
}

/** One frame: global_orient (3) + body_pose (63) → 22 global quaternions. */
export function smplGlobals(go, bp, yawFix) {
  const G = new Array(22);
  G[0] = yawFix.clone().multiply(aaToQuat(go[0], go[1], go[2]));
  for (let j = 1; j < 22; j++) {
    const k = (j - 1) * 3;
    G[j] = G[SMPL_PARENTS[j]].clone().multiply(aaToQuat(bp[k], bp[k + 1], bp[k + 2]));
  }
  return G;
}

// ---- One Euro filter (for live landmarks) --------------------------------------
export class OneEuro {
  constructor(n, minCutoff = 1.5, beta = 0.6, dCutoff = 1.0) {
    Object.assign(this, { n, minCutoff, beta, dCutoff });
    this.x = null; this.dx = new Float32Array(n); this.t = 0;
  }
  reset() { this.x = null; }
  static alpha(cutoff, dt) { const r = 2 * Math.PI * cutoff * dt; return r / (r + 1); }
  filter(values, t) {
    if (!this.x || t <= this.t) { this.x = Float32Array.from(values); this.t = t; this.dx.fill(0); return this.x; }
    const dt = Math.min(0.2, t - this.t); this.t = t;
    const ad = OneEuro.alpha(this.dCutoff, dt);
    for (let i = 0; i < this.n; i++) {
      const dx = (values[i] - this.x[i]) / dt;
      this.dx[i] += ad * (dx - this.dx[i]);
      const a = OneEuro.alpha(this.minCutoff + this.beta * Math.abs(this.dx[i]), dt);
      this.x[i] += a * (values[i] - this.x[i]);
    }
    return this.x;
  }
}

// ---- Takes: frame storage, sampling, export -------------------------------------
export class Track {
  constructor(boneCount) { this.boneCount = boneCount; this.times = []; this.q = []; this.p = []; }
  push(t, snap) { this.times.push(t); this.q.push(Float32Array.from(snap.q)); this.p.push(Float32Array.from(snap.p)); }
  get length() { return this.times.length; }
  get duration() { return this.times.length ? this.times[this.times.length - 1] : 0; }

  /** Interpolated snapshot at time t (clamped). */
  sample(t, out) {
    const n = this.times.length;
    out = out || { q: new Float32Array(this.boneCount * 4), p: new Float32Array(3) };
    if (!n) return null;
    let i = 0;
    if (t <= this.times[0]) i = 0;
    else if (t >= this.times[n - 1]) i = n - 1;
    else { let lo = 0, hi = n - 1; while (hi - lo > 1) { const m = (lo + hi) >> 1; (this.times[m] <= t ? (lo = m) : (hi = m)); } i = lo; }
    const j = Math.min(n - 1, i + 1);
    const span = this.times[j] - this.times[i];
    const a = span > 1e-6 ? Math.min(1, Math.max(0, (t - this.times[i]) / span)) : 0;
    blendSnapshots(this.q[i], this.p[i], this.q[j], this.p[j], a, out);
    return out;
  }
}

export function blendSnapshots(qa, pa, qb, pb, a, out) {
  for (let k = 0; k < qa.length; k += 4) {
    _q1.fromArray(qa, k); _q2.fromArray(qb, k);
    _q1.slerp(_q2, a).toArray(out.q, k);
  }
  for (let k = 0; k < 3; k++) out.p[k] = pa[k] + (pb[k] - pa[k]) * a;
  return out;
}

/** Bake a Track into a THREE.AnimationClip (local quats + hips local position). */
export function trackToClip(rig, track, name) {
  const times = Float32Array.from(track.times.map((t) => t - track.times[0]));
  const tracks = rig.order.map((bone, bi) => {
    const v = new Float32Array(times.length * 4);
    track.q.forEach((q, f) => v.set(q.subarray(bi * 4, bi * 4 + 4), f * 4));
    return new THREE.QuaternionKeyframeTrack(`${rig.bones[bone].name}.quaternion`, times, v);
  });
  const pos = new Float32Array(times.length * 3);
  const parentInv = new THREE.Matrix4();
  rig.reset();
  parentInv.copy(rig.hips.parent.matrixWorld).invert();
  track.p.forEach((p, f) => _v1.fromArray(p).applyMatrix4(parentInv).toArray(pos, f * 3));
  tracks.push(new THREE.VectorKeyframeTrack(`${rig.hips.name}.position`, times, pos));
  return new THREE.AnimationClip(name, times[times.length - 1] || 0, tracks);
}
