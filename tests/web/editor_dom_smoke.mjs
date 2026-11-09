// Browser smoke test of web/regional_lora_editor.js against a STUB of ComfyUI's frontend API
// (scripts/app.js, scripts/api.js, node.addDOMWidget). It checks the editor's own behaviour —
// tools, selection, the properties panel, LoRA lists, curves, writing the JSON widget, hand-edit
// sync, parse-error safety, undo, keyboard — not ComfyUI integration. Needs Playwright with
// Chromium; skips if unavailable.
//
//   node tests/web/editor_dom_smoke.mjs [final.png] [editor.png]

import { createServer } from "node:http";
import { readFileSync, existsSync } from "node:fs";
import { join, dirname, extname } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import assert from "node:assert/strict";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const require = createRequire(import.meta.url);
let chromium;
try {
  ({ chromium } = require("playwright"));
} catch {
  console.log("SKIP: playwright not available");
  process.exit(0);
}

const SLIDER = "MOSS Image\\\\concept\\\\weight_slider_moss_v5_100.safetensors";
const APP_STUB = `
export const app = { exts: [], graph: { setDirtyCanvas() {} }, registerExtension(e) { this.exts.push(e); } };
window.__app = app;`;
const API_STUB = `
export const api = {
  fetchApi: async (path) => ({
    ok: true,
    json: async () => path === "/models/loras" ? ["alice.safetensors", "bob.safetensors", "${SLIDER}"] : ["ref.png [input]"],
  }),
  apiURL: (p) => "/missing" + p,
};`;
const PAGE = `<!doctype html><html><body style="margin:0;background:#111">
<textarea id="raw" style="width:1180px;height:90px"></textarea>
<div id="host" style="width:1180px;height:700px"></div>
<script type="module">
import "/scripts/app.js";
const app = window.__app;
await import("/extensions/moss/regional_lora_editor.js");
const ext = app.exts[0];
const raw = document.getElementById("raw");
function Node() {
  this.size = [1180, 800];
  this.widgets = [
    { name: "width", value: 1024, options: { min: 16, max: 16384 } },
    { name: "height", value: 768, options: { min: 16, max: 16384 } },
    { name: "regions", get value() { return raw.value; }, set value(v) { raw.value = v; } },
    { name: "ramp_start", value: 0.15, options: { min: 0, max: 1, round: 0.01 } },
    { name: "ramp_end", value: 1.0, options: { min: 0, max: 1, round: 0.01 } },
    { name: "steps", value: 20, options: { min: 1, max: 10000 } },
    { name: "strict", value: true },
    { name: "caption", value: "" },
    { name: "text_position", value: "suffix", options: { values: ["not set", "prefix", "suffix"] } },
    { name: "image_order", value: "raster", options: { values: ["not set", "raster"] } },
    { name: "source_tokens", value: "end", options: { values: ["not set", "none", "start", "between", "end"] } },
  ];
  this.inputs = [];
}
Node.prototype.addDOMWidget = function (name, type, element) {
  element.style.height = "100%";
  document.getElementById("host").appendChild(element);
  return { name, element };
};
Node.prototype.setSize = function (s) { this.size = s; };
Node.prototype.setDirtyCanvas = function () {};
Node.prototype.configure = function (info) { this.configuredWith = info.widgets_values; };   // LGraphNode's
await ext.beforeRegisterNodeDef(Node, { name: "MossRegionalLora" });
window.node = new Node();
window.node.onNodeCreated();
// ComfyUI stand-ins (frontend 1.58): its node container forwards the wheel to the graph unless the pointer
// is over a [data-capture-wheel="true"] element holding the focus; its change tracker runs the workflow undo
// for Ctrl+Z / Ctrl+Y from a window capture listener unless a text field has the focus.
window.__graphWheel = 0;
window.__comfyUndo = 0;
document.getElementById("host").addEventListener("wheel", (e) => {
  const t = e.target.closest?.('[data-capture-wheel="true"]');
  const a = document.activeElement;
  if (t && a && t.contains(a)) return;
  e.preventDefault();
  e.stopPropagation();
  window.__graphWheel++;
}, { capture: true, passive: false });
window.addEventListener("keydown", (e) => {
  const a = document.activeElement;
  if (a && (a.tagName === "INPUT" || a.tagName === "TEXTAREA" || a.isContentEditable)) return;
  if ((e.ctrlKey || e.metaKey) && /^[zy]$/i.test(e.key)) window.__comfyUndo++;
}, true);
window.ready = true;
</script></body></html>`;

const MIME = { ".js": "text/javascript", ".html": "text/html", ".json": "application/json" };
const server = createServer((req, res) => {
  const url = req.url.split("?")[0];
  let body;
  let type = "text/javascript";
  if (url === "/" || url === "/index.html") [body, type] = [PAGE, "text/html"];
  else if (url === "/scripts/app.js") body = APP_STUB;
  else if (url === "/scripts/api.js") body = API_STUB;
  else if (url.startsWith("/extensions/moss/")) {
    const f = join(ROOT, "web", url.slice("/extensions/moss/".length));
    if (existsSync(f)) [body, type] = [readFileSync(f), MIME[extname(f)] || "text/plain"];
  }
  if (body === undefined) {
    res.writeHead(404);
    return res.end();
  }
  res.writeHead(200, { "content-type": type });
  res.end(body);
});
await new Promise((r) => server.listen(0, r));
const base = `http://127.0.0.1:${server.address().port}/`;

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1200, height: 820 }, deviceScaleFactor: 2 });
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
await page.goto(base);
await page.waitForFunction(() => window.ready === true);
// as a user would: click away (graph focused), then move the pointer onto the editor's canvas
async function hoverCanvas() {
  await page.evaluate(() => document.activeElement?.blur?.());
  const vs = page.viewportSize();
  await page.mouse.move(vs.width - 2, vs.height - 2);
  const b = await page.locator(".mrl-canvas-wrap canvas").boundingBox();
  await page.mouse.move(b.x + 6, b.y + b.height - 6);
}
const json = async () => JSON.parse(await page.locator("#raw").inputValue());
const propsTitle = () => page.locator(".mrl-props .mrl-panel-head").innerText();

async function toScreen(x, y) {
  return page.evaluate(
    ([x, y]) => {
      const c = document.querySelector(".mrl-canvas-wrap canvas");
      const r = c.getBoundingClientRect();
      const W = 1024, H = 768;
      const s = Math.min((r.width - 2) / W, (r.height - 2) / H);
      const ox = (r.width - W * s) / 2, oy = (r.height - H * s) / 2;
      return { x: r.left + ox + x * s, y: r.top + oy + y * s };
    },
    [x, y],
  );
}
async function drag(from, to) {
  const a = await toScreen(...from);
  const b = await toScreen(...to);
  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  await page.mouse.move((a.x + b.x) / 2, (a.y + b.y) / 2, { steps: 4 });
  await page.mouse.move(b.x, b.y, { steps: 4 });
  await page.mouse.up();
}
async function click(x, y) {
  const a = await toScreen(x, y);
  await page.mouse.click(a.x, a.y);
}

// 1. the Select tool does not draw: dragging on the background just deselects
assert.match(await propsTitle(), /Canvas/);
await drag([100, 100], [300, 300]);
assert.equal((await json().catch(() => ({ regions: [] }))).regions?.length ?? 0, 0);

// 2. Box tool draws, then hands back to Select with the new box selected
await page.locator(".mrl-tools button").nth(1).click();
await drag([112, 84], [512, 384]);
let s = await json();
assert.equal(s.version, 2);
assert.equal(s.regions.length, 1);
assert.deepEqual(s.regions[0].loras, []);
const [y0, x0, y1, x1] = s.regions[0].bbox;
assert.ok(Math.abs(y0 + 300) <= 2 && Math.abs(x0 + 400) <= 2 && Math.abs(y1) <= 2 && Math.abs(x1) <= 2, `bbox ${s.regions[0].bbox}`);
assert.ok(await page.locator(".mrl-tools button").nth(0).evaluate((b) => b.classList.contains("on")));
assert.match(await propsTitle(), /Region 1/);
console.log("box tool ok", s.regions[0].bbox);

// 3. keyboard nudge, undo/redo
await page.keyboard.press("ArrowRight");
await page.keyboard.press("Shift+ArrowDown");
assert.deepEqual((await json()).regions[0].bbox, [y0 + 10, x0 + 1, y1 + 10, x1 + 1]);
await page.keyboard.press("Control+z");
await page.keyboard.press("Control+z");
assert.deepEqual((await json()).regions[0].bbox, [y0, x0, y1, x1]);
await page.keyboard.press("Control+Shift+z");
assert.deepEqual((await json()).regions[0].bbox, [y0, x0 + 1, y1, x1 + 1]);
console.log("nudge/undo ok");

// 4. move by dragging the body with the Select tool; B / V keys switch tools
await drag([300, 250], [400, 250]);
assert.ok(Math.abs((await json()).regions[0].bbox[1] - (x0 + 1 + 100)) <= 2);
await page.keyboard.press("b");
assert.ok(await page.locator(".mrl-tools button").nth(1).evaluate((b) => b.classList.contains("on")));
await page.keyboard.press("v");
console.log("move ok");

// 5. clicking the background deselects
await click(900, 700);
assert.match(await propsTitle(), /Canvas/);
console.log("background deselect ok");

// 5b. the node's inputs are hidden and edited from the Canvas panel
const wv = (name) => page.evaluate((n) => window.node.widgets.find((w) => w.name === n).value, name);
assert.ok(await page.evaluate(() => window.node.widgets.every((w) => w.hidden === true)));
const sizeIn = page.locator(".mrl-size input[type=number]");
await sizeIn.nth(0).fill("1000");                  // rounded to whole 16 px latent cells
await sizeIn.nth(0).press("Tab");
assert.equal(await wv("width"), 1008);
await page.waitForFunction(() => document.querySelector(".mrl-dims").textContent.startsWith("1008"));
await page.locator('.mrl-size button[title="Swap width and height"]').click();
assert.deepEqual([await wv("width"), await wv("height")], [768, 1008]);
await page.locator('.mrl-size button[title="Swap width and height"]').click();
await sizeIn.nth(0).fill("1024");
await sizeIn.nth(0).press("Tab");
assert.deepEqual([await wv("width"), await wv("height")], [1024, 768]);
await page.locator(".mrl-props textarea.mrl-caption").pressSequentially("a cozy kitchen");
assert.equal(await wv("caption"), "a cozy kitchen");    // written while typing
await page.waitForTimeout(400);                          // a poll while typing keeps the focus
assert.ok(await page.evaluate(() => document.activeElement?.classList.contains("mrl-caption")));
await page.locator(".mrl-props .mrl-field", { hasText: "Strict" }).locator("input").uncheck();
assert.equal(await wv("strict"), false);
assert.equal(await page.locator(".mrl-props .mrl-field", { hasText: "Image order" }).count(), 0); // collapsed
await page.locator(".mrl-props .mrl-adv-head").click();
await page.locator(".mrl-props .mrl-field", { hasText: "Source tok." }).locator("select").selectOption("none");
assert.equal(await wv("source_tokens"), "none");
await page.locator(".mrl-props .mrl-field", { hasText: "Steps" }).locator("input").fill("30");
await page.locator(".mrl-props .mrl-field", { hasText: "Steps" }).locator("input").press("Tab");
assert.equal(await wv("steps"), 30);
// a setting wired to another node shows read-only
await page.evaluate(() => { window.node.inputs = [{ name: "caption", widget: { name: "caption" }, link: 7 }]; });
await page.waitForFunction(() => document.querySelector(".mrl-props textarea.mrl-caption")?.disabled === true);
await page.evaluate(() => { window.node.inputs = []; });
await page.waitForFunction(() => document.querySelector(".mrl-props textarea.mrl-caption")?.disabled === false);
if (process.argv[4]) await page.screenshot({ path: process.argv[4] });
console.log("canvas settings ok");

// 5c. Edit JSON: invalid text is refused with a message, valid text replaces the boxes (undoable)
const before = await page.locator("#raw").inputValue();
await page.locator(".mrl-props button", { hasText: "Edit JSON" }).click();
assert.equal(await page.locator(".mrl-json textarea").inputValue(), before);
if (process.argv[5]) await page.screenshot({ path: process.argv[5] });
await page.locator(".mrl-json textarea").fill('{"regions": [{"bbox": [5, 0, 0, 5]}]}');
await page.locator(".mrl-json button", { hasText: "Apply" }).click();
assert.match(await page.locator(".mrl-json-error").innerText(), /Not applied/);
assert.equal(await page.locator("#raw").inputValue(), before);
const one = JSON.stringify({ version: 2, regions: [{ id: "j", name: "From JSON", bbox: [-100, -100, 100, 100], loras: [] }] });
await page.locator(".mrl-json textarea").fill(one);
await page.locator(".mrl-json button", { hasText: "Apply" }).click();
assert.equal(await page.locator(".mrl-json").count(), 0);
assert.equal(await page.locator("#raw").inputValue(), one);
await page.waitForFunction(() => [...document.querySelectorAll(".mrl-layers .mrl-name")].some((e) => e.textContent === "From JSON"));
await page.locator(".mrl-root").focus();
await page.keyboard.press("Control+z");
assert.equal(await page.locator("#raw").inputValue(), before);
await page.locator(".mrl-props button", { hasText: "Edit JSON" }).click();
await page.keyboard.press("Escape");
assert.equal(await page.locator(".mrl-json").count(), 0);
console.log("edit json ok");

// 6. hand-edited JSON is reflected; a version-1 single-LoRA box is read as a LoRA list
const pasted = {
  version: 1,
  regions: [
    { id: "a", name: "Alice", bbox: [-384, -512, 384, 0], lora: SLIDER.replaceAll("\\\\", "\\"), weight: -2, visible: true, locked: false, color: "#e6194b", group: null, label: "character", caption: "", text: "", palette: [] },
    { id: "b", name: "Bob", bbox: [-384, 0, 384, 600], loras: [], visible: true, locked: true, color: "#4363d8", group: null, label: "object", caption: "", text: "", palette: [] },
  ],
  groups: [],
  editor: { grid_divisions: 3, snap: false, reference_image: "missing.png", reference_opacity: 0.5 },
};
await page.locator("#raw").fill(JSON.stringify(pasted, null, 2));
await page.waitForFunction(() => document.querySelectorAll(".mrl-layers .mrl-row .mrl-name").length === 2, null, { timeout: 2000 });
assert.equal(await page.locator(".mrl-layers .mrl-oob").count(), 1); // Bob extends past x = +512
console.log("hand edit sync ok");

// 7. a locked box does not move
await drag([700, 300], [800, 300]);
assert.deepEqual((await json()).regions[1].bbox, [-384, 0, 384, 600]);
console.log("lock ok");

// 8. selecting a layer row fills the properties panel; the LoRA shows its file name only
await page.locator(".mrl-layers .mrl-row", { hasText: "Alice" }).click();
assert.match(await propsTitle(), /Alice/);
assert.equal((await page.locator(".mrl-lora-name").first().innerText()).trim(), "weight_slider_moss_v5_100");
assert.ok((await page.locator(".mrl-root svg.mrl-icon").count()) > 10); // Lucide icons rendered
// editing a box field converts nothing else; editing the LoRA list converts the version-1 fields
await page.locator(".mrl-props textarea").fill("a tall woman");
await page.locator(".mrl-props textarea").press("Tab");
assert.equal((await json()).regions[0].caption, "a tall woman");
// the caption box can be resized by its corner handle (it sits in a flex column: flex must not pin its height)
const ta = page.locator(".mrl-props textarea").first();
await ta.scrollIntoViewIfNeeded();
const tb = await ta.boundingBox();
await page.mouse.move(tb.x + tb.width - 3, tb.y + tb.height - 3);
await page.mouse.down();
await page.mouse.move(tb.x + tb.width - 3, tb.y + tb.height + 57, { steps: 5 });
await page.mouse.up();
const grown = (await ta.boundingBox()).height;
assert.ok(grown > tb.height + 30, `caption box resized: ${tb.height} -> ${grown}`);
console.log("textarea resize ok");
await page.locator(".mrl-lora-row input[type=number]").first().fill("1.5");
await page.locator(".mrl-lora-row input[type=number]").first().press("Tab");
s = await json();
assert.equal(s.version, 2);
assert.ok(!("lora" in s.regions[0]) && !("weight" in s.regions[0]));
assert.deepEqual(s.regions[0].loras, [{ lora: pasted.regions[0].lora, weight: 1.5 }]); // no curve yet: node ramp
console.log("properties ok");

// 9. add a second LoRA and pick its file from the path dropdown
await page.locator(".mrl-props button", { hasText: "Add LoRA" }).click();
await page.locator(".mrl-lora-name").nth(1).click();
await page.locator(".mrl-picker input").fill("bob");
await page.locator(".mrl-picker-item", { hasText: "bob.safetensors" }).click();
s = await json();
assert.deepEqual(s.regions[0].loras[1], { lora: "bob.safetensors", weight: 1, curve: [{ x: 0, y: 1 }, { x: 1, y: 1 }] });
console.log("lora list ok");

// 10. click a LoRA's mini curve to expand and edit it
await page.locator(".mrl-lora .mrl-spark").nth(1).click();
await page.locator(".mrl-curve-bar button", { hasText: "Stop at 50%" }).click();
s = await json();
assert.deepEqual(s.regions[0].loras[1].curve, [{ x: 0, y: 1, hold: true }, { x: 0.5, y: 0 }, { x: 1, y: 0 }]);
async function graphPx(x, y) {
  return page.evaluate(([x, y]) => {
    const c = document.querySelector(".mrl-curve-wrap canvas");
    const r = c.getBoundingClientRect();
    const PAD = { l: 34, r: 10, t: 8, b: 18 };
    return { x: r.left + PAD.l + x * (r.width - PAD.l - PAD.r), y: r.top + PAD.t + (1 - y) * (r.height - PAD.t - PAD.b) };
  }, [x, y]);
}
let q = await graphPx(0.75, 0.6);
await page.mouse.click(q.x, q.y);
s = await json();
assert.equal(s.regions[0].loras[1].curve.length, 4);
const c = s.regions[0].loras[1].curve;
q = await graphPx((c[1].x + c[2].x) / 2, (c[1].y + c[2].y) / 2);
await page.mouse.move(q.x, q.y);
await page.mouse.down();
const q2 = await graphPx((c[1].x + c[2].x) / 2, 0.1);
await page.mouse.move(q2.x, q2.y, { steps: 4 });
await page.mouse.up();
assert.ok((await json()).regions[0].loras[1].curve[1].bend > 0.2);
await page.locator(".mrl-curve-wrap canvas").focus();
await page.keyboard.press("Delete"); // no point selected after the bend: must not delete the box
assert.equal((await json()).regions.length, 2);
assert.equal((await json()).regions[0].loras.length, 2);
if (process.argv[3]) await page.screenshot({ path: process.argv[3] });
console.log("curve editing ok");

// 11. remove a LoRA; reorder boxes with the layer buttons
await page.locator(".mrl-lora-row .mrl-lora-remove").nth(0).click();
assert.deepEqual((await json()).regions[0].loras.map((e) => e.lora), ["bob.safetensors"]);
await page.locator(".mrl-layers .mrl-row", { hasText: "Bob" }).locator('button[title^="Move up"]').click();
// the eye toggle hides a box (and it is then left out of the output)
await page.locator(".mrl-layers .mrl-row", { hasText: "Alice" }).locator(".mrl-vis").click();
assert.equal((await json()).regions[1].visible, false);
assert.deepEqual((await json()).regions.map((r) => r.name), ["Bob", "Alice"]);
console.log("reorder ok");

// 12. parse error: visible error state, and nothing done in the editor overwrites the text
const broken = '{"version": 2, "regions": [ {"name": "oops"';
await page.locator("#raw").fill(broken);
await page.waitForFunction(() => getComputedStyle(document.querySelector(".mrl-error")).display !== "none", null, { timeout: 2000 });
await page.locator(".mrl-tools button").nth(1).click();
await drag([100, 100], [300, 300]);
await hoverCanvas();
await page.keyboard.press("Delete");
await page.keyboard.press("Control+z");
assert.equal(await page.locator("#raw").inputValue(), broken);
console.log("parse-error safety ok");

// a workflow saved with vae_downscale/patch_size loads with its later values in the right slots
const migrated = await page.evaluate(() => {
  window.node.configure({ widgets_values: [1024, 768, "{}", 16, 1, 0.15, 1.0, 20, true, "cap", "suffix", "raster", "end"] });
  return window.node.configuredWith;
});
assert.deepEqual(migrated, [1024, 768, "{}", 0.15, 1.0, 20, true, "cap", "suffix", "raster", "end"]);
console.log("old widget values migrated ok");


await page.locator("#raw").fill(JSON.stringify(pasted, null, 2));
await page.waitForFunction(() => getComputedStyle(document.querySelector(".mrl-error")).display === "none", null, { timeout: 2000 });
// queueing sends only what affects the image: no names, colours, locks, groups or editor settings
const sent = await page.evaluate(() => JSON.parse(window.node.widgets.find((w) => w.name === "regions").serializeValue()));
assert.ok(sent.regions.length > 0);
for (const r of sent.regions) for (const k of ["name", "color", "locked", "group", "visible"]) assert.ok(!(k in r), k);
assert.ok(!("editor" in sent) && !("groups" in sent));
console.log("backend regions ok");

// zoom and pan: the wheel zooms around the pointer; Hand tool, Space-drag and middle-drag pan without
// editing; the toolbar's fit button resets
const edView = () => page.evaluate(() => { const v = window.node.__mrlEditor.view(); return { scale: v.scale, ox: v.ox, oy: v.oy }; });
async function imgToScreen(x, y) {
  const v = await edView();
  return page.evaluate(([x, y, v]) => {
    const r = document.querySelector(".mrl-canvas-wrap canvas").getBoundingClientRect();
    return { x: r.left + v.ox + x * v.scale, y: r.top + v.oy + y * v.scale };
  }, [x, y, v]);
}
await hoverCanvas();                                                                       // focus: click away, hover
const v0 = await edView();
let q0 = await imgToScreen(700, 300);
await page.mouse.move(q0.x, q0.y);
await page.mouse.wheel(0, -400);
await page.waitForTimeout(50);
const v1 = await edView();
assert.ok(v1.scale > v0.scale * 1.2, `zoomed: ${v0.scale} -> ${v1.scale}`);
const q1 = await imgToScreen(700, 300);
assert.ok(Math.abs(q1.x - q0.x) < 1 && Math.abs(q1.y - q0.y) < 1, "the point under the pointer stays put");
const rawBeforePan = await page.locator("#raw").inputValue();
async function dragBy(dx, dy, opts = {}) {
  const r = await page.locator(".mrl-canvas-wrap canvas").boundingBox();
  const x = r.x + r.width / 2, y = r.y + r.height / 2;
  await page.mouse.move(x, y);
  await page.mouse.down(opts);
  await page.mouse.move(x + dx, y + dy, { steps: 4 });
  await page.mouse.up(opts);
}
await hoverCanvas();
await page.keyboard.press("h");
let a0 = await edView();
await dragBy(60, 25);
let a1 = await edView();
assert.ok(Math.abs(a1.ox - a0.ox - 60) < 1 && Math.abs(a1.oy - a0.oy - 25) < 1, "hand tool pans");
await page.keyboard.press("v");
a0 = await edView();
await page.keyboard.down(" ");
await dragBy(-30, 10);
await page.keyboard.up(" ");
a1 = await edView();
assert.ok(Math.abs(a1.ox - a0.ox + 30) < 1, "space-drag pans");
a0 = a1;
await dragBy(20, 0, { button: "middle" });
a1 = await edView();
assert.ok(Math.abs(a1.ox - a0.ox - 20) < 1, "middle-drag pans");
assert.equal(await page.locator("#raw").inputValue(), rawBeforePan);                     // nothing was edited
await page.locator('.mrl-tools button[title^="Fit"]').click();
const vf = await edView();
assert.ok(Math.abs(vf.scale - v0.scale) < 1e-9 && Math.abs(vf.ox - v0.ox) < 1e-9);
await page.locator('.mrl-tools button[title^="Zoom in"]').click();
assert.ok((await edView()).scale > v0.scale);
await page.keyboard.press("0");
assert.ok(Math.abs((await edView()).scale - v0.scale) < 1e-9);
console.log("zoom/pan ok");
if (process.argv[2]) await page.screenshot({ path: process.argv[2] });
const comfy = await page.evaluate(() => ({ wheel: window.__graphWheel, undo: window.__comfyUndo }));
assert.deepEqual(comfy, { wheel: 0, undo: 0 }, "ComfyUI's own wheel forwarding / undo never ran for editor input");
console.log("comfyui wheel/undo stand-ins untouched ok");
assert.deepEqual(errors, []);
console.log("PASS");
await browser.close();
server.close();
