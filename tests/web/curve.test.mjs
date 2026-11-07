// Headless tests for web/curve.js. The fixture is generated from moss_regional_lora/curve.py and also
// checked by tests/test_curves.py, so the editor draws exactly what the node applies.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as C from "../../web/curve.js";

const FIXTURE = JSON.parse(readFileSync(fileURLToPath(new URL("./fixtures/curves.json", import.meta.url)), "utf8"));

test("evaluation matches the Python implementation", () => {
  for (const [name, c] of Object.entries(FIXTURE.curves)) {
    const pts = C.normalize(c.curve);
    FIXTURE.samples.forEach((x, k) => {
      assert.ok(Math.abs(C.evaluate(pts, x) - c.expected[k]) < 1e-12, `${name} at ${x}: ${C.evaluate(pts, x)} vs ${c.expected[k]}`);
    });
  }
});

test("legacy ramp reproduces the node ramp", () => {
  const ramp = (p, rs, re) => (p < rs ? 0 : re <= rs || p >= re ? 1 : (p - rs) / (re - rs));
  for (const [rs, re] of [[0.15, 1], [0, 0], [0.3, 0.3], [0.25, 0.75], [0.5, 0.2]]) {
    const c = C.legacyRamp(rs, re);
    for (let k = 0; k <= 20; k++) assert.ok(Math.abs(C.evaluate(c, k / 20) - ramp(k / 20, rs, re)) < 1e-12, `${rs},${re} at ${k / 20}`);
  }
});

test("presets", () => {
  assert.equal(C.evaluate(C.PRESETS.Full(), 0.7), 1);
  assert.equal(C.evaluate(C.PRESETS["Fade in"](), 0.25), 0.25);
  const stop = C.PRESETS["Stop at 50%"]();
  assert.equal(C.evaluate(stop, 0.49), 1);
  assert.equal(C.evaluate(stop, 0.5), 0);
});

test("add, move and remove points", () => {
  let { points, index } = C.addPoint(C.FULL, 0.4, 0.2);
  assert.equal(index, 1);
  assert.deepEqual(points, [{ x: 0, y: 1 }, { x: 0.4, y: 0.2 }, { x: 1, y: 1 }]);
  points = C.movePoint(points, 1, 1.3, -1); // x is kept between its neighbours, y clamped
  assert.deepEqual(points[1], { x: 1, y: 0 });
  points = C.removePoint(points, 1);
  assert.deepEqual(points, [{ x: 0, y: 1 }, { x: 1, y: 1 }]);
  assert.deepEqual(C.removePoint([{ x: 0, y: 1 }], 0), [{ x: 0, y: 1 }]); // never empty
  // splitting a held segment keeps both halves held
  const held = C.addPoint([{ x: 0, y: 1, hold: true }, { x: 1, y: 0 }], 0.5, 0.5).points;
  assert.equal(held[1].hold, true);
});

test("dragging a segment midpoint sets the bend", () => {
  const pts = [{ x: 0, y: 0 }, { x: 1, y: 1 }];
  assert.equal(C.bendForMidpoint(pts, 0, 0.5), 0); // straight
  const easeIn = C.bendForMidpoint(pts, 0, 0.25);
  assert.ok(Math.abs(easeIn - 0.5) < 1e-9); // 0.5 ** (4 ** 0.5) = 0.25
  const bent = C.setSegment(pts, 0, { bend: easeIn });
  assert.ok(Math.abs(C.segmentHandle(bent, 0).y - 0.25) < 1e-4);
  assert.ok(C.bendForMidpoint(pts, 0, 0.8) < 0); // ease out
  assert.equal(C.bendForMidpoint([{ x: 0, y: 1 }, { x: 1, y: 1 }], 0, 0.3), 0); // flat segment: no bend
  assert.equal(C.bendForMidpoint(pts, 0, -5), 1); // clamped
});

test("hold toggle and tidy output", () => {
  const pts = C.setSegment([{ x: 0, y: 1, bend: 0.3 }, { x: 1, y: 0 }], 0, { hold: true, bend: 0 });
  assert.deepEqual(pts, [{ x: 0, y: 1, hold: true }, { x: 1, y: 0 }]);
  assert.deepEqual(C.setSegment(pts, 0, { hold: false }), [{ x: 0, y: 1 }, { x: 1, y: 0 }]);
});

test("normalize clamps, sorts and rejects", () => {
  assert.deepEqual(C.normalize([{ x: 1.4, y: -1 }, { x: 0.5, y: 0.5, bend: 9 }]), [{ x: 0.5, y: 0.5, bend: 1 }, { x: 1, y: 0 }]);
  for (const bad of [[], [{ x: 0 }], [{ x: "a", y: 1 }], [{ x: 0, y: 1, hold: "yes" }]]) {
    assert.throws(() => C.normalize(bad));
  }
});
