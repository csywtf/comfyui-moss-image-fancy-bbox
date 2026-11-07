// Headless tests for web/geometry.js (spec 8.10b). Run: node --test tests/web/
import { test } from "node:test";
import assert from "node:assert/strict";
import * as G from "../../web/geometry.js";

const W = 1024;
const H = 768;

test("centred <-> canvas round-trip", () => {
  for (const bbox of [
    [-384, -512, 384, 512],
    [-10, -20, 30, 40],
    [0, 0, 0, 0],
    [100, -700, 900, -600], // partly off-canvas
  ]) {
    const rect = G.centeredToCanvas(bbox, W, H);
    assert.deepEqual(G.canvasToCentered(rect, W, H), bbox);
  }
  for (const rect of [{ x0: 0, y0: 0, x1: 1024, y1: 768 }, { x0: 10, y0: 20, x1: 30, y1: 50 }]) {
    assert.deepEqual(G.centeredToCanvas(G.canvasToCentered(rect, W, H), W, H), rect);
  }
});

test("serialised bbox is integer [y_min, x_min, y_max, x_max] from the centre", () => {
  const b = G.canvasToCentered({ x0: 512.4, y0: 383.6, x1: 600.5, y1: 400.49 }, W, H);
  assert.ok(b.every(Number.isInteger));
  assert.deepEqual(b, [0, 0, 16, 89]);
  // full canvas
  assert.deepEqual(G.canvasToCentered({ x0: 0, y0: 0, x1: W, y1: H }, W, H), [-384, -512, 384, 512]);
});

test("top-left-origin sign cases for each quadrant", () => {
  // a 10x10 box fully inside each quadrant
  const cases = [
    { name: "top-left", rect: { x0: 100, y0: 100, x1: 110, y1: 110 }, sign: [-1, -1, -1, -1] },
    { name: "top-right", rect: { x0: 900, y0: 100, x1: 910, y1: 110 }, sign: [-1, 1, -1, 1] },
    { name: "bottom-left", rect: { x0: 100, y0: 600, x1: 110, y1: 610 }, sign: [1, -1, 1, -1] },
    { name: "bottom-right", rect: { x0: 900, y0: 600, x1: 910, y1: 610 }, sign: [1, 1, 1, 1] },
  ];
  for (const c of cases) {
    const b = G.canvasToCentered(c.rect, W, H);
    assert.deepEqual(b.map(Math.sign), c.sign, c.name);
    assert.ok(b[0] < b[2] && b[1] < b[3], `${c.name}: min < max`);
  }
});

test("centre uses floor(width/2), floor(height/2) like MOSSImageLayoutPrompt", () => {
  assert.deepEqual(G.canvasCenter(33, 17), { cx: 16, cy: 8 });
  assert.deepEqual(G.centeredToCanvas([0, 0, 0, 0], 33, 17), { x0: 16, y0: 8, x1: 16, y1: 8 });
});

test("create-drag produces the expected bbox", () => {
  const rect = G.createRect({ x: 100, y: 50 }, { x: 300, y: 250 });
  assert.deepEqual(G.canvasToCentered(rect, W, H), [50 - 384, 100 - 512, 250 - 384, 300 - 512]);
  // dragging up-left normalises
  assert.deepEqual(G.createRect({ x: 300, y: 250 }, { x: 100, y: 50 }), { x0: 100, y0: 50, x1: 300, y1: 250 });
  // Shift: square, using the larger side, in the drag direction
  assert.deepEqual(G.createRect({ x: 100, y: 100 }, { x: 160, y: 120 }, { square: true }), { x0: 100, y0: 100, x1: 160, y1: 160 });
  assert.deepEqual(G.createRect({ x: 100, y: 100 }, { x: 80, y: 40 }, { square: true }), { x0: 40, y0: 40, x1: 100, y1: 100 });
  // Alt: from centre
  assert.deepEqual(G.createRect({ x: 100, y: 100 }, { x: 130, y: 110 }, { fromCenter: true }), { x0: 70, y0: 90, x1: 130, y1: 110 });
});

test("move", () => {
  const r = { x0: 10, y0: 20, x1: 30, y1: 60 };
  assert.deepEqual(G.moveRect(r, 5, -7), { x0: 15, y0: 13, x1: 35, y1: 53 });
  const b = [-100, -50, 100, 50];
  const moved = G.canvasToCentered(G.moveRect(G.centeredToCanvas(b, W, H), 12, -3), W, H);
  assert.deepEqual(moved, [-103, -38, 97, 62]);
});

test("resize from each of the eight handles", () => {
  const r = { x0: 100, y0: 100, x1: 200, y1: 300 };
  const p = { x: 150, y: 50 };
  const expected = {
    nw: { x0: 150, y0: 50, x1: 200, y1: 300 },
    n: { x0: 100, y0: 50, x1: 200, y1: 300 },
    ne: { x0: 100, y0: 50, x1: 150, y1: 300 },
    e: { x0: 100, y0: 100, x1: 150, y1: 300 },
    se: { x0: 100, y0: 50, x1: 150, y1: 100 },
    s: { x0: 100, y0: 50, x1: 200, y1: 100 },
    sw: { x0: 150, y0: 50, x1: 200, y1: 100 },
    w: { x0: 150, y0: 100, x1: 200, y1: 300 },
  };
  for (const h of G.HANDLES) assert.deepEqual(G.resizeRect(r, h, p), expected[h], h);
  // dragging a corner straight outward
  assert.deepEqual(G.resizeRect(r, "se", { x: 250, y: 350 }), { x0: 100, y0: 100, x1: 250, y1: 350 });
  // dragging past the opposite edge flips (normalised)
  assert.deepEqual(G.resizeRect(r, "e", { x: 40, y: 0 }), { x0: 40, y0: 100, x1: 100, y1: 300 });
});

test("resize with square constraint (Shift)", () => {
  const r = { x0: 100, y0: 100, x1: 200, y1: 300 };
  assert.deepEqual(G.resizeRect(r, "se", { x: 250, y: 180 }, { square: true }), { x0: 100, y0: 100, x1: 250, y1: 250 });
  assert.deepEqual(G.resizeRect(r, "nw", { x: 150, y: 0 }, { square: true }), { x0: -100, y0: 0, x1: 200, y1: 300 });
  const e = G.resizeRect(r, "e", { x: 180, y: 0 }, { square: true });
  assert.equal(e.x1 - e.x0, e.y1 - e.y0);
  assert.equal((e.y0 + e.y1) / 2, 200); // other axis stays centred
});

test("resize from centre (Alt), with and without square", () => {
  const r = { x0: 100, y0: 100, x1: 200, y1: 300 }; // centre (150, 200)
  assert.deepEqual(G.resizeRect(r, "se", { x: 220, y: 260 }, { fromCenter: true }), { x0: 80, y0: 140, x1: 220, y1: 260 });
  assert.deepEqual(G.resizeRect(r, "e", { x: 170, y: 999 }, { fromCenter: true }), { x0: 130, y0: 100, x1: 170, y1: 300 });
  assert.deepEqual(G.resizeRect(r, "se", { x: 160, y: 260 }, { fromCenter: true, square: true }), { x0: 90, y0: 140, x1: 210, y1: 260 });
});

test("snapping", () => {
  const grid = G.gridStep(W, H, 4); // 256 x 192
  assert.deepEqual(grid, { sx: 256, sy: 192 });
  assert.deepEqual(G.snapPoint({ x: 300, y: 100 }, grid), { x: 256, y: 192 });
  assert.deepEqual(G.snapPoint({ x: 400, y: 95 }, grid), { x: 512, y: 0 });
  assert.equal(G.snapValue(17, 0), 17);
  const r = { x0: 10, y0: 10, x1: 110, y1: 60 };
  const m = G.snapMove(r, 240, 170, grid); // top-left -> (250, 180) -> snaps to (256, 192)
  assert.deepEqual(m, { x0: 256, y0: 192, x1: 356, y1: 242 });
  // default grid is a third of the canvas
  assert.deepEqual(G.gridStep(900, 600, 3), { sx: 300, sy: 200 });
});

test("clipping, bounds and clamping", () => {
  assert.equal(G.outOfBounds({ x0: 0, y0: 0, x1: W, y1: H }, W, H), false);
  assert.equal(G.outOfBounds({ x0: -1, y0: 0, x1: 10, y1: 10 }, W, H), true);
  assert.deepEqual(G.clipRect({ x0: -50, y0: 700, x1: 50, y1: 900 }, W, H), { x0: 0, y0: 700, x1: 50, y1: 768 });
  assert.equal(G.clipRect({ x0: 2000, y0: 0, x1: 2100, y1: 10 }, W, H), null);
  assert.deepEqual(G.clampPoint({ x: -5, y: 900 }, W, H), { x: 0, y: 768 });
});

test("hit-testing picks the topmost (last drawn) region and skips hidden ones", () => {
  const a = { x0: 0, y0: 0, x1: 100, y1: 100 };
  const b = { x0: 50, y0: 50, x1: 150, y1: 150 };
  assert.equal(G.hitTest([a, b], { x: 75, y: 75 }), 1);
  assert.equal(G.hitTest([a, b], { x: 25, y: 25 }), 0);
  assert.equal(G.hitTest([a, null], { x: 75, y: 75 }), 0);
  assert.equal(G.hitTest([a, b], { x: 500, y: 500 }), -1);
});

test("handle detection", () => {
  const r = { x0: 100, y0: 100, x1: 200, y1: 300 };
  assert.equal(G.handleAt(r, { x: 101, y: 99 }, 5), "nw");
  assert.equal(G.handleAt(r, { x: 150, y: 302 }, 5), "s");
  assert.equal(G.handleAt(r, { x: 198, y: 200 }, 5), "e");
  assert.equal(G.handleAt(r, { x: 150, y: 200 }, 5), null);
  // tiny box: corners win over edge midpoints
  assert.equal(G.handleAt({ x0: 0, y0: 0, x1: 4, y1: 4 }, { x: 4, y: 4 }, 5), "se");
});

test("fit preserves aspect ratio", () => {
  assert.deepEqual(G.fitCanvas(1024, 512, 400, 400), { scale: 400 / 1024, w: 400, h: 200 });
  assert.deepEqual(G.fitCanvas(512, 1024, 400, 400), { scale: 400 / 1024, w: 200, h: 400 });
});

test("nudge", () => {
  assert.deepEqual(G.nudgeBbox([0, 0, 10, 10], "ArrowRight", false), [0, 1, 10, 11]);
  assert.deepEqual(G.nudgeBbox([0, 0, 10, 10], "ArrowUp", true), [-10, 0, 0, 10]);
});

test("validity", () => {
  assert.ok(G.isValidBbox([0, 0, 1, 1]));
  assert.ok(!G.isValidBbox([1, 0, 0, 1]));
  assert.ok(!G.isValidBbox([0, 0, 1]));
});
