// Headless tests for web/paint.js (Character Masks painting). Run: node --test tests/web/
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as P from "../../web/paint.js";

const CASES = JSON.parse(readFileSync(fileURLToPath(new URL("./fixtures/paint_cases.json", import.meta.url)), "utf8")).cases;

test("run-length storage round-trips and matches the shared fixture", () => {
  for (const c of CASES) {
    const layer = P.decode(c.stored);
    assert.deepEqual([...layer.bits], c.bits, JSON.stringify(c.stored));
    assert.deepEqual(P.encode(layer), { w: c.stored.w, h: c.stored.h, runs: c.canonical ?? c.stored.runs });
  }
});

test("a layer stored at another size is scaled (nearest)", () => {
  const small = P.blank(2, 2);
  small.bits.set([1, 0, 0, 1]);
  const big = P.decode(P.encode(small), 4, 4);
  assert.deepEqual([...big.bits], [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1]);
  assert.ok(P.isEmpty(P.decode(null, 3, 3)) && P.decode(null, 3, 3).bits.length === 9);
});

test("brush: stamps and strokes add and remove", () => {
  const l = P.blank(20, 10);
  P.strokeLine(l, 2, 5, 17, 5, 2);
  for (let x = 2; x <= 17; x++) assert.equal(l.bits[5 * 20 + x], 1, `x=${x}`);
  assert.equal(l.bits[0], 0);
  P.stamp(l, 10, 5, 3, 0);                                        // remove
  assert.equal(l.bits[5 * 20 + 10], 0);
  assert.equal(l.bits[5 * 20 + 3], 1);
  P.stamp(l, -5, -5, 2);                                          // off the canvas: nothing, no error
});

test("magic wand: connected fill within tolerance, or every similar pixel in a box", () => {
  // 6 x 3: two red patches separated by a blue column, a dark-red pixel at the far end
  const w = 6, h = 3;
  const px = (r, g, b) => [r, g, b, 255];
  const grid = [
    [px(200, 0, 0), px(205, 0, 0), px(0, 0, 255), px(200, 0, 0), px(200, 0, 0), px(120, 0, 0)],
    [px(200, 0, 0), px(210, 5, 0), px(0, 0, 255), px(200, 0, 0), px(200, 0, 0), px(200, 0, 0)],
    [px(200, 0, 0), px(200, 0, 0), px(0, 0, 255), px(200, 0, 0), px(200, 0, 0), px(200, 0, 0)],
  ];
  const rgba = new Uint8ClampedArray(grid.flat(2));
  const at = (l, x, y) => l.bits[y * w + x];
  const c = P.wand(rgba, w, h, 0, 0, 12, true);
  assert.equal(at(c, 1, 1), 1);                                    // within tolerance, joined
  assert.equal(at(c, 3, 0), 0);                                    // same colour, but across the blue column
  assert.equal(at(c, 2, 0), 0);
  const all = P.wand(rgba, w, h, 0, 0, 12, false);
  assert.equal(at(all, 3, 0), 1);                                  // not connected: everywhere similar
  assert.equal(at(all, 5, 0), 0);                                  // dark red: outside the tolerance
  assert.equal(at(P.wand(rgba, w, h, 0, 0, 90, false), 5, 0), 1);
  const boxed = P.wand(rgba, w, h, 0, 0, 12, false, { x0: 0, y0: 0, x1: 2, y1: 3 });
  assert.equal(at(boxed, 3, 0), 0);                                // limited to the box
  assert.ok(P.isEmpty(P.wand(rgba, w, h, -1, 0, 12)));
});

test("strokeBounds covers every pixel a stroke paints, clipped to the layer", () => {
  for (const [x0, y0, x1, y1, r] of [[10, 10, 40, 25, 3], [0.2, 47.9, 49.5, 0.1, 6], [-5, -5, 3, 3, 4], [25.5, 25.5, 25.5, 25.5, 0.7]]) {
    const m = P.blank(50, 50);
    P.strokeLine(m, x0, y0, x1, y1, r, 1);
    const b = P.strokeBounds(m, x0, y0, x1, y1, r);
    assert.ok(b.x0 >= 0 && b.y0 >= 0 && b.x1 <= 50 && b.y1 <= 50);
    for (let y = 0; y < 50; y++) for (let x = 0; x < 50; x++) {
      if (m.bits[y * 50 + x]) assert.ok(x >= b.x0 && x < b.x1 && y >= b.y0 && y < b.y1, `${x},${y} outside ${JSON.stringify(b)}`);
    }
  }
});

test("wand flood fill on a large flat image is fast and complete", () => {
  const w = 1024, h = 1024;
  const rgba = new Uint8ClampedArray(w * h * 4).fill(200);
  const t = performance.now();
  const m = P.wand(rgba, w, h, 5, 5, 10, true);
  assert.ok(performance.now() - t < 2000);
  assert.equal(m.bits.reduce((a, b) => a + b, 0), w * h);
});
