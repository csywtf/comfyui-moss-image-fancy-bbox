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
    const norm = { ...d, loras: d.loras.map((e) => [e.lora, e.weight]), paint: Object.keys(d.paint).sort() };
    assert.deepEqual(norm, c.expected, JSON.stringify(c.stored));
  }
  assert.deepEqual(D.PART_NAMES, ["face", "eyes", "left_hand", "right_hand", "hair", "skin"]);
});

test("parse: empty is defaults, bad JSON is an error, unknown fields survive an edit", () => {
  assert.deepEqual(D.parseDetail("").doc, D.emptyDoc());
  for (const bad of ["{oops", "[]", '{"boxes": []}']) assert.equal(D.parseDetail(bad).ok, false, bad);
  const res = D.parseDetail('{"version": 1, "boxes": {"a": {"enabled": false, "x_box": 7}}, "x_top": [1]}');
  assert.ok(res.ok);
  const region = { id: "a", label: "character" };
  const d = D.detailFor(res.doc, region);
  assert.equal(d.enabled, false);
  d.parts.left_hand.on = true;
  d.loras.push({ lora: "eye.safetensors", weight: 0.5 });
  d.paint.face = { add: { w: 1, h: 1, runs: [0, 1] }, sub: null };
  const next = D.setDetail(res.doc, "a", d);
  const back = JSON.parse(D.serializeDetail(next));
  assert.equal(back.x_top[0], 1);
  assert.equal(back.boxes.a.x_box, 7);
  assert.equal(back.boxes.a.parts.left_hand.on, true);
  assert.deepEqual(back.boxes.a.loras, [{ lora: "eye.safetensors", weight: 0.5 }]);
  assert.deepEqual(back.boxes.a.paint, { face: { add: { w: 1, h: 1, runs: [0, 1] }, sub: null } });
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

test("settings saved for 'hands' apply to both hands until one is edited", () => {
  const doc = { version: 1, boxes: { a: { parts: { hands: { on: true, denoise: 0.5 } } } } };
  const d = D.detailFor(doc, { id: "a", label: "character" });
  assert.deepEqual(d.parts.left_hand, { on: true, prompt: "", denoise: 0.5, restore: null, whole_crop: false, full_prompt: null });
  assert.deepEqual(d.parts.right_hand, d.parts.left_hand);
  d.parts.right_hand.restore = 0;
  const next = D.detailFor(D.setDetail(doc, "a", d), { id: "a", label: "character" });
  assert.equal(next.parts.right_hand.restore, 0);
  assert.equal(next.parts.left_hand.restore, null);
});

test("paint and the Protect layer are stored; old points are dropped on the next edit", () => {
  let doc = { version: 1, boxes: { a: { points: [{ x: 0.5, y: 0.5, label: 1 }] } } };
  const region = { id: "a", label: "character" };
  const d = D.detailFor(doc, region);
  assert.deepEqual(d.paint, {});
  d.paint.character = { add: { w: 2, h: 1, runs: [1, 1] }, sub: null };
  doc = D.setDetail(doc, "a", d);
  assert.ok(!("points" in doc.boxes.a));
  assert.deepEqual(D.detailFor(doc, region).paint.character.add, { w: 2, h: 1, runs: [1, 1] });
  assert.equal(D.protectOf(doc), null);
  doc = D.setProtect(doc, { w: 1, h: 1, runs: [0, 1] });
  assert.deepEqual(D.protectOf(doc), { w: 1, h: 1, runs: [0, 1] });
  assert.ok(!("protect" in D.setProtect(doc, null)));
  assert.deepEqual(D.PAINT_TARGETS, ["character", "face", "eyes", "left_hand", "right_hand", "hair", "skin"]);
});

test("hands: distinct colours, L/R tags on a pixel of each hand, partAt picks the finest part", () => {
  const color = (n) => D.PARTS.find((p) => p.name === n).color;
  assert.notEqual(color("left_hand"), color("right_hand"));
  // 6x2 label map, box 1: left hand at x 0-1 (a "C" whose centre is empty), right hand at x 4-5; box 2 none
  const w = 6, h = 2;
  const L = 1 << D.PART_NAMES.indexOf("left_hand"), R = 1 << D.PART_NAMES.indexOf("right_hand");
  const labels = new Uint8ClampedArray(w * h * 4);
  const set = (x, y, bits) => labels.set([1, bits, 255, 255], (y * w + x) * 4);
  set(0, 0, L); set(1, 0, L); set(0, 1, L);
  set(4, 0, R); set(5, 0, R); set(4, 1, R); set(5, 1, R);
  set(2, 1, 0);
  const tags = D.partTags(labels, w, h);
  assert.deepEqual(tags.map((t) => t.text), ["L", "R"]);
  for (const t of tags) {
    const i = (Math.floor(t.y) * w + Math.floor(t.x)) * 4;
    assert.equal(D.partAt(labels, i).part, t.part, JSON.stringify(t));
  }
  assert.deepEqual(D.partAt(labels, (1 * w + 2) * 4), { box: 0, part: -1 });       // outline only
  assert.equal(D.partAt(labels, (1 * w + 3) * 4), null);                             // nobody
  const both = new Uint8ClampedArray([1, (1 << D.PART_NAMES.indexOf("face")) | (1 << D.PART_NAMES.indexOf("eyes")), 255, 255]);
  assert.equal(D.PARTS[D.partAt(both, 0).part].name, "eyes");
});

test("part prompts: automatic and edited, as the detailer builds them (shared fixture)", () => {
  const prompts = JSON.parse(readFileSync(fileURLToPath(new URL("./fixtures/detail_cases.json", import.meta.url)), "utf8")).prompts;
  for (const c of prompts) {
    const doc = c.stored ? { version: 1, boxes: { x: c.stored } } : D.emptyDoc();
    const d = D.detailFor(doc, { id: "x", label: "character" });
    assert.equal(D.autoPrompt(c.caption, d, c.part), c.auto, JSON.stringify(c));
    assert.equal(D.partPrompt(c.caption, d, c.part), c.used, JSON.stringify(c));
  }
});
