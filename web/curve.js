// Per-box strength curves (pure, no DOM). Same maths as moss_regional_lora/curve.py; both are pinned
// by tests/web/fixtures/curves.json.
//
// A curve is a list of points {x, y, bend?, hold?}: x = progress through the sampling run (0..1),
// y = fraction of the box weight (0..1). The segment that starts at a point takes its shape:
//   bend in [-1, 1]: t ** (4 ** bend); 0 = straight, >0 eases in (slow start), <0 eases out.
//   hold: keep this point's value until the next point, then jump (a hard step).
// A box without a curve uses the node's ramp_start/ramp_end (see legacyRamp).

export const FULL = [{ x: 0, y: 1 }, { x: 1, y: 1 }];
const MIN_EXP = 0.25;
const MAX_EXP = 4;

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

/** Clamp and sort a curve (copy). Throws on malformed input. */
export function normalize(points) {
  if (!Array.isArray(points) || !points.length) throw new Error("curve must be a non-empty list of points");
  return points
    .map((p, i) => {
      if (!p || typeof p !== "object") throw new Error(`curve point ${i} is not an object`);
      for (const k of ["x", "y"]) if (typeof p[k] !== "number" || !Number.isFinite(p[k])) throw new Error(`curve point ${i}: ${k} must be a number`);
      const q = { x: clamp(p.x, 0, 1), y: clamp(p.y, 0, 1) };
      const bend = p.bend === undefined ? 0 : p.bend;
      if (typeof bend !== "number" || !Number.isFinite(bend)) throw new Error(`curve point ${i}: bend must be a number`);
      if (bend) q.bend = clamp(bend, -1, 1);
      if (p.hold !== undefined && typeof p.hold !== "boolean") throw new Error(`curve point ${i}: hold must be true or false`);
      if (p.hold) q.hold = true;
      return { q, i };
    })
    .sort((a, b) => a.q.x - b.q.x || a.i - b.i) // stable, like Python's sorted
    .map((e) => e.q);
}

export function shape(t, bend) {
  return bend ? t ** (4 ** bend) : t;
}

/** Fraction of the box weight at progress p (points must be normalized). */
export function evaluate(points, p) {
  const x = clamp(p, 0, 1);
  if (x <= points[0].x) return points[0].y;
  for (let i = 0; i + 1 < points.length; i++) {
    const a = points[i];
    const b = points[i + 1];
    if (x < b.x) {
      if (a.hold) return a.y;
      const t = (x - a.x) / (b.x - a.x);
      return a.y + (b.y - a.y) * shape(t, a.bend || 0);
    }
  }
  return points[points.length - 1].y;
}

/** The node-level linear ramp as a curve (matches curve.py legacy_ramp). */
export function legacyRamp(rampStart, rampEnd) {
  const rs = clamp(rampStart, 0, 1);
  const re = clamp(rampEnd, 0, 1);
  if (re <= rs) return rs > 0 ? [{ x: 0, y: 0, hold: true }, { x: rs, y: 1 }, { x: 1, y: 1 }] : [{ x: 0, y: 1 }];
  return [{ x: 0, y: 0 }, { x: rs, y: 0 }, { x: re, y: 1 }, { x: 1, y: 1 }];
}

export const PRESETS = {
  Full: () => FULL.map((p) => ({ ...p })),
  "Fade in": () => [{ x: 0, y: 0 }, { x: 1, y: 1 }],
  "Fade out": () => [{ x: 0, y: 1 }, { x: 1, y: 0 }],
  "Stop at 50%": () => [{ x: 0, y: 1, hold: true }, { x: 0.5, y: 0 }, { x: 1, y: 0 }],
};

const round = (v) => Math.round(v * 10000) / 10000;

function tidy(p) {
  const q = { x: round(p.x), y: round(p.y) };
  if (p.bend) q.bend = round(p.bend);
  if (p.hold) q.hold = true;
  return q;
}

/** Insert a point; returns {points, index}. */
export function addPoint(points, x, y) {
  const q = { x: clamp(x, 0, 1), y: clamp(y, 0, 1) };
  let i = points.findIndex((p) => p.x > q.x);
  if (i < 0) i = points.length;
  const out = points.map((p) => ({ ...p }));
  // splitting a held segment keeps it held on both sides
  if (i > 0 && out[i - 1].hold) q.hold = true;
  out.splice(i, 0, q);
  return { points: out.map(tidy), index: i };
}

export function removePoint(points, index) {
  if (points.length <= 1) return points.map((p) => ({ ...p }));
  return points.filter((_, i) => i !== index).map(tidy);
}

/** Move a point, keeping x between its neighbours so the order never changes. */
export function movePoint(points, index, x, y) {
  const out = points.map((p) => ({ ...p }));
  const lo = index > 0 ? out[index - 1].x : 0;
  const hi = index + 1 < out.length ? out[index + 1].x : 1;
  out[index].x = clamp(x, lo, hi);
  out[index].y = clamp(y, 0, 1);
  return out.map(tidy);
}

/** Midpoint handle of segment i (value at t = 0.5), or null for the last point. */
export function segmentHandle(points, i) {
  if (i + 1 >= points.length) return null;
  const a = points[i];
  const b = points[i + 1];
  const y = a.hold ? a.y : a.y + (b.y - a.y) * shape(0.5, a.bend || 0);
  return { x: (a.x + b.x) / 2, y };
}

/** Bend that puts segment i's midpoint value at yMid (dragging the midpoint handle). */
export function bendForMidpoint(points, i, yMid) {
  const a = points[i];
  const b = points[i + 1];
  if (!b || a.y === b.y) return 0;
  const m = clamp((yMid - a.y) / (b.y - a.y), 0.5 ** MAX_EXP, 0.5 ** MIN_EXP);
  const exponent = Math.log(m) / Math.log(0.5);
  const bend = Math.log(exponent) / Math.log(4);
  return Math.abs(bend) < 0.02 ? 0 : clamp(bend, -1, 1);
}

export function setSegment(points, i, patch) {
  const out = points.map((p) => ({ ...p }));
  Object.assign(out[i], patch);
  if (!out[i].bend) delete out[i].bend;
  if (!out[i].hold) delete out[i].hold;
  return out.map(tidy);
}
