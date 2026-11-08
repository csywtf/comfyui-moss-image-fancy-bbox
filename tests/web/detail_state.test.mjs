// Headless tests for web/detail_state.js (Character Masks settings and overlay). Run: node --test tests/web/
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as D from "../../web/detail_state.js";

const CASES = JSON.parse(readFileSync(fileURLToPath(new URL("./fixtures/detail_cases.json", import.meta.url)), "utf8")).cases;

test("settings resolve exactly as the node resolves them (shared fixture)", () => {
  for (const c of CASES) {
    const doc = c.stored ? { version: 1, boxes: { x: c.stored } } : D.emptyDoc();
    const d = D.detailFor(doc, { id: "x", label: c.label });
    assert.deepEqual({ ...d, loras: d.loras.map((e) => [e.lora, e.weight]) }, c.expected, JSON.stringify(c.stored));
  }
  assert.deepEqual(D.PART_NAMES, ["face", "eyes", "hands", "hair", "skin"]);
});

test("parse: empty is defaults, bad JSON is an error, unknown fields survive an edit", () => {
  assert.deepEqual(D.parseDetail("").doc, D.emptyDoc());
  for (const bad of ["{oops", "[]", '{"boxes": []}']) assert.equal(D.parseDetail(bad).ok, false, bad);
  const res = D.parseDetail('{"version": 1, "boxes": {"a": {"enabled": false, "x_box": 7}}, "x_top": [1]}');
  assert.ok(res.ok);
  const region = { id: "a", label: "character" };
  const d = D.detailFor(res.doc, region);
  assert.equal(d.enabled, false);
  d.parts.hands.on = true;
  d.loras.push({ lora: "eye.safetensors", weight: 0.5 });
  const next = D.setDetail(res.doc, "a", d);
  const back = JSON.parse(D.serializeDetail(next));
  assert.equal(back.x_top[0], 1);
  assert.equal(back.boxes.a.x_box, 7);
  assert.equal(back.boxes.a.parts.hands.on, true);
  assert.deepEqual(back.boxes.a.loras, [{ lora: "eye.safetensors", weight: 0.5 }]);
  assert.deepEqual(res.doc.boxes.a, { enabled: false, x_box: 7 });            // input untouched
});

test("a box with no entry follows its label; editing one box leaves the others on defaults", () => {
  let doc = D.emptyDoc();
  const alice = { id: "alice", label: "character" };
  const lamp = { id: "lamp", label: "object" };
  const d = D.detailFor(doc, lamp);
  d.enabled = true;
  doc = D.setDetail(doc, "lamp", d);
  assert.deepEqual(Object.keys(doc.boxes), ["lamp"]);
  assert.equal(D.detailFor(doc, lamp).enabled, true);
  assert.deepEqual(D.detailFor(doc, alice), D.defaultDetail("character"));
});

test("overlay: selected box strong, others faint, finest part on top, nothing where no box", () => {
  // 4 pixels: alice face+eyes, alice outline only, bob face, nobody
  const labels = new Uint8ClampedArray([1, 0b11, 255, 255, 1, 0, 255, 255, 2, 0b1, 255, 255, 0, 0, 0, 255]);
  const colors = ["#ff0000", "#00ff00"];
  const all = D.overlayPixels(labels, colors, -1);
  assert.deepEqual([...all.slice(0, 4)], [0x2e, 0xc4, 0xff, 170]);   // eyes colour over face
  assert.deepEqual([...all.slice(4, 8)], [255, 0, 0, 70]);           // alice's outline, her colour
  assert.deepEqual([...all.slice(12, 16)], [0, 0, 0, 0]);
  const bob = D.overlayPixels(labels, colors, 1);
  assert.equal(bob[3], 45);                                           // alice faint while bob is selected
  assert.equal(bob[7], 25);
  assert.deepEqual([...bob.slice(8, 12)], [0xff, 0x9f, 0x1c, 170]);  // bob's face, strong
});
