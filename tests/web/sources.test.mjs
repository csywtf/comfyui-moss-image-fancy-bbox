// The values the region editor reads from nodes wired into its sockets (web/sources.js).
import { test } from "node:test";
import assert from "node:assert/strict";
import { canvasSide, liveNumber, liveText, resolutionSelectorSize } from "../../web/sources.js";

const selector = (aspect_ratio, megapixels, multiple) => ({
  type: "ResolutionSelector",
  outputs: [{ name: "width" }, { name: "height" }],
  widgets: [{ name: "aspect_ratio", value: aspect_ratio }, { name: "megapixels", value: megapixels }, { name: "preview", value: null }, ...(multiple == null ? [] : [{ name: "multiple", value: multiple }])],
});

test("a Resolution Selector is computed exactly as ComfyUI computes it", () => {
  // the values its own preview shows: 21:9, 5 MP, multiple 16 -> 3504 x 1504
  assert.deepEqual(resolutionSelectorSize(selector("21:9 (Ultrawide)", 5, 16)), [3504, 1504]);
  assert.deepEqual(resolutionSelectorSize(selector("1:1 (Square)", 1, 8)), [1024, 1024]);
  assert.deepEqual(resolutionSelectorSize(selector("9:16 (Portrait Widescreen)", 1, 8)), [768, 1368]);
  // Python's round(): 1028 / 8 = 128.5 rounds to the even 128, not 129
  assert.deepEqual(resolutionSelectorSize(selector("1:1 (Square)", 1.0078277587890625, 8)), [1024, 1024]);
  // the advanced "multiple" widget absent: the node's default, 8
  assert.deepEqual(resolutionSelectorSize(selector("1:1 (Square)", 1, null)), [1024, 1024]);
  assert.equal(resolutionSelectorSize(selector("5:4 (Unknown)", 1, 8)), null);
});

test("each Resolution Selector output feeds its own side", () => {
  assert.equal(liveNumber(selector("21:9 (Ultrawide)", 5, 16), 0), 3504);
  assert.equal(liveNumber(selector("21:9 (Ultrawide)", 5, 16), 1), 1504);
});

test("a primitive's number is read live; computed sizes are not", () => {
  assert.equal(liveNumber({ type: "PrimitiveNode", outputs: [{}], widgets: [{ name: "INT", value: 1344 }] }), 1344);
  assert.equal(liveNumber({ type: "PrimitiveInt", outputs: [{}], widgets: [{ name: "value", value: 768 }] }), 768);
  // two outputs (width, height): which one is wired can't be told from a widget
  assert.equal(liveNumber({ type: "SomeResolution", outputs: [{}, {}], widgets: [{ name: "value", value: 5 }] }), null);
  assert.equal(liveNumber({ type: "PrimitiveInt", outputs: [{}], widgets: [{ name: "value", value: "12" }] }), null);
  assert.equal(liveNumber(null), null);
});

test("a text primitive's caption is read live; other text only after a run", () => {
  assert.equal(liveText({ type: "PrimitiveNode", outputs: [{}], widgets: [{ name: "STRING", value: "a scene" }] }), "a scene");
  assert.equal(liveText({ type: "PrimitiveStringMultiline", outputs: [{}], widgets: [{ name: "value", value: "two people" }] }), "two people");
  assert.equal(liveText({ type: "PrimitiveInt", outputs: [{}], widgets: [{ name: "value", value: 3 }] }), null);
  assert.equal(liveText({ type: "SomePromptWriter", outputs: [{}, {}], widgets: [{ name: "value", value: "x" }] }), null);
  assert.equal(liveText(null), null);
});

test("sizes from a socket round down to whole 16 px cells, like the node", () => {
  assert.equal(canvasSide(1000), 992);
  assert.equal(canvasSide(1024), 1024);
  assert.equal(canvasSide(5), 16);
});
