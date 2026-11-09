// Headless tests for web/viewport.js (editor zoom and pan). Run: node --test tests/web/
import { test } from "node:test";
import assert from "node:assert/strict";
import { MAX_ZOOM, MIN_ZOOM, Viewport } from "../../web/viewport.js";

const W = 1024, H = 768, CW = 602, CH = 402;
const toLayout = (v, x, y) => ({ x: x * v.scale + v.ox, y: y * v.scale + v.oy });

test("zoom 1 fits the image, centred", () => {
  const v = new Viewport().view(W, H, CW, CH);
  assert.ok(Math.abs(v.scale - Math.min(600 / W, 400 / H)) < 1e-9);
  assert.ok(Math.abs(v.ox - (CW - W * v.scale) / 2) < 1e-9 && Math.abs(v.oy - (CH - H * v.scale) / 2) < 1e-9);
});

test("zooming keeps the image point under the cursor fixed", () => {
  const vp = new Viewport();
  const v0 = vp.view(W, H, CW, CH);
  const target = toLayout(v0, 700, 200);
  vp.zoomAt(3, target.x, target.y, W, H, CW, CH);
  const v1 = vp.view(W, H, CW, CH);
  const after = toLayout(v1, 700, 200);
  assert.ok(Math.abs(after.x - target.x) < 1e-6 && Math.abs(after.y - target.y) < 1e-6);
  assert.ok(Math.abs(v1.scale / v0.scale - 3) < 1e-9);
});

test("zoom is clamped and reset returns to fit", () => {
  const vp = new Viewport();
  vp.zoomBy(1e6, W, H, CW, CH);
  assert.equal(vp.zoom, MAX_ZOOM);
  vp.zoomBy(1e-9, W, H, CW, CH);
  assert.equal(vp.zoom, MIN_ZOOM);
  vp.panBy(30, 10, W, H, CW, CH);
  vp.reset();
  assert.ok(vp.fitted);
});

test("panning moves the image and can't push it out of sight", () => {
  const vp = new Viewport();
  vp.zoomBy(4, W, H, CW, CH);
  const before = vp.view(W, H, CW, CH);
  vp.panBy(25, -15, W, H, CW, CH);
  const after = vp.view(W, H, CW, CH);
  assert.ok(Math.abs(after.ox - before.ox - 25) < 1e-9 && Math.abs(after.oy - before.oy + 15) < 1e-9);
  vp.panBy(1e6, 1e6, W, H, CW, CH);
  const v = vp.view(W, H, CW, CH);
  // the image still covers at least a quarter of the canvas width and height
  assert.ok(v.ox <= CW - CW / 4 + 1e-6 && v.oy <= CH - CH / 4 + 1e-6);
});
