// Browser smoke test of web/character_masks.js against a STUB of ComfyUI's frontend API and a stub
// "Regional LoRA" node feeding its regions input. Checks the editor's own behaviour: the box list read
// from the linked node, hidden widgets, SAM3 settings, per-box detail settings written to the hidden
// `detail` widget, showing a run's masks, and picking a box on the preview. Not ComfyUI integration.
// Needs Playwright with Chromium; skips if unavailable.
//
//   node tests/web/masks_dom_smoke.mjs [screenshot.png]

import { createServer } from "node:http";
import { readFileSync, existsSync } from "node:fs";
import { join, dirname, extname } from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";
import assert from "node:assert/strict";
import * as P from "../../web/paint.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..", "..");
const require = createRequire(import.meta.url);
let chromium;
try {
  ({ chromium } = require("playwright"));
} catch {
  console.log("SKIP: playwright not available");
  process.exit(0);
}

const APP_STUB = `
export const app = { exts: [], graph: { setDirtyCanvas() {} }, registerExtension(e) { this.exts.push(e); } };
window.__app = app;`;
const API_STUB = `
export const api = {
  fetchApi: async (path) => ({ ok: true, json: async () => path === "/models/loras" ? ["eye_detail.safetensors", "alice.safetensors"] : [] }),
  apiURL: (p) => window.__views[new URLSearchParams(p.split("?")[1]).get("filename")] || "/missing",
};`;

const REGIONS = {
  version: 2,
  regions: [
    { id: "alice", name: "Alice", bbox: [-40, -90, 40, 10], label: "character", caption: "a tall woman", color: "#e6194b", loras: [{ lora: "alice.safetensors", weight: 0.8 }] },
    { id: "bob", name: "Bob", bbox: [-40, -20, 40, 80], label: "character", caption: "", color: "#4363d8", loras: [] },
    { id: "lamp", name: "Lamp", bbox: [-45, 82, -20, 98], label: "object", caption: "a lamp", loras: [] },
    { id: "ghost", name: "Ghost", bbox: [0, 0, 10, 10], label: "character", visible: false, loras: [] },
  ],
  groups: [],
  editor: {},
};

const PAGE = `<!doctype html><html><body style="margin:0;background:#111">
<div id="host" style="width:1080px;height:600px"></div>
<script type="module">
import "/scripts/app.js";
const app = window.__app;
await import("/extensions/moss/character_masks.js");
const ext = app.exts.find((e) => e.name === "moss.CharacterMasksEditor");
const main = { widgets: [
  { name: "width", value: 200 }, { name: "height", value: 100 },
  { name: "regions", value: ${JSON.stringify(JSON.stringify(REGIONS))} },
] };
window.main = main;
function Node() {
  this.size = [1080, 600];
  this.inputs = [{ name: "image", link: 1 }, { name: "regions", link: 2 }, { name: "sam3_model", link: 3 }, { name: "sam3_clip", link: 4 }];
  this.widgets = [
    { name: "threshold", value: 0.5, options: { min: 0, max: 1 } },
    { name: "box_margin", value: 0.15, options: { min: 0, max: 1 } },
    { name: "refine_iterations", value: 2, options: { min: 0, max: 5 } },
    { name: "detail", value: '{"version": 1, "boxes": {}}' },
  ];
}
Node.prototype.getInputNode = function (slot) { return slot === 1 && this.inputs[1].link != null ? main : null; };
Node.prototype.addDOMWidget = function (name, type, element) {
  element.style.height = "100%";
  document.getElementById("host").appendChild(element);
  return { name, element };
};
Node.prototype.setSize = function (s) { this.size = s; };
Node.prototype.setDirtyCanvas = function () {};
await ext.beforeRegisterNodeDef(Node, { name: "MossCharacterMasks" });
window.node = new Node();
window.node.onNodeCreated();

// a fake run result: the image and a label map (R = box number, G = part bits, B = outline)
function png(w, h, paint) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  const g = c.getContext("2d");
  paint(g);
  return c.toDataURL("image/png");
}
const labels = png(200, 100, (g) => {
  const d = g.createImageData(200, 100);
  for (let y = 0; y < 100; y++) for (let x = 0; x < 200; x++) {
    const i = (y * 200 + x) * 4;
    d.data[i + 3] = 255;
    if (x >= 12 && x < 108 && y >= 12 && y < 88) { d.data[i] = 1; d.data[i + 2] = 255; }
    if (x >= 30 && x < 50 && y >= 20 && y < 40) d.data[i + 1] = 1;                 // alice face
    if (x >= 60 && x < 70 && y >= 45 && y < 60) d.data[i + 1] = 4;                 // her hands, touching
    if (x >= 70 && x < 80 && y >= 45 && y < 60) d.data[i + 1] = 8;
    if (x >= 120 && x < 178 && y >= 12 && y < 88) { d.data[i] = 2; d.data[i + 2] = 255; }
    if (x >= 130 && x < 150 && y >= 20 && y < 40) d.data[i + 1] = 1;               // bob face
  }
  g.putImageData(d, 0, 0);
});
const image = png(200, 100, (g) => { const gr = g.createLinearGradient(0, 0, 200, 100); gr.addColorStop(0, "#456"); gr.addColorStop(1, "#cba"); g.fillStyle = gr; g.fillRect(0, 0, 200, 100); });
window.__views = { "img.png": image, "lab.png": labels };
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
// its graph container also takes middle-button presses and drags in its capture phase, hands them to the
// graph and stops them, unless their target is a focused text field
window.__graphPan = 0;
for (const type of ["pointerdown", "pointermove", "pointerup"]) {
  document.getElementById("host").addEventListener(type, (e) => {
    const middle = type === "pointermove" ? (e.buttons & 4) === 4 : e.button === 1 || e.buttons === 4;
    if (!middle) return;
    const t = e.target;
    if ((t.tagName === "TEXTAREA" || t.tagName === "INPUT" || t.isContentEditable) && document.activeElement === t) return;
    e.preventDefault();
    e.stopPropagation();
    window.__graphPan++;
  }, true);
}
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

const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1100, height: 640 }, deviceScaleFactor: 2 });
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
await page.goto(`http://127.0.0.1:${server.address().port}/`);
await page.waitForFunction(() => window.ready === true);
// as a user would: click away (graph focused), then move the pointer onto the editor's canvas
async function hoverCanvas() {
  await page.evaluate(() => document.activeElement?.blur?.());
  const vs = page.viewportSize();
  await page.mouse.move(vs.width - 2, vs.height - 2);
  const b = await page.locator(".mrl-canvas-wrap canvas").boundingBox();
  await page.mouse.move(b.x + 6, b.y + b.height - 6);
}
const detail = async () => JSON.parse(await page.evaluate(() => window.node.widgets.find((w) => w.name === "detail").value));
const wv = (n) => page.evaluate((n) => window.node.widgets.find((w) => w.name === n).value, n);
const row = (name) => page.locator(".mcm-row", { hasText: name });

// 1. boxes from the linked node (hidden ones left out); characters ticked, the object not; widgets hidden
await page.waitForFunction(() => document.querySelectorAll(".mcm-row").length === 3);
assert.deepEqual(await page.locator(".mcm-row .mrl-name").allInnerTexts(), ["Alice", "Bob", "Lamp"]);
assert.deepEqual(await page.locator(".mcm-row input[type=checkbox]").evaluateAll((xs) => xs.map((x) => x.checked)), [true, true, false]);
assert.ok(await page.evaluate(() => window.node.widgets.every((w) => w.hidden === true)));
console.log("box list ok");

// 2. SAM3 settings when no box is selected
assert.match(await page.locator(".mrl-props .mrl-panel-head").innerText(), /Masking/);
await page.locator(".mrl-props .mrl-field", { hasText: "Threshold" }).locator("input").fill("0.35");
await page.locator(".mrl-props .mrl-field", { hasText: "Threshold" }).locator("input").press("Tab");
assert.equal(await wv("threshold"), 0.35);
console.log("settings ok");

// 3. per-box detail settings land in the hidden widget
await row("Alice").click();
assert.match(await page.locator(".mrl-props .mrl-panel-head").innerText(), /Detail: Alice/);
assert.match(await page.locator(".mrl-props").innerText(), /alice \(0\.8\)/);           // the box's own LoRAs
const parts = page.locator(".mcm-parts input.mcm-on");
assert.deepEqual(await parts.evaluateAll((xs) => xs.map((x) => x.checked)), [true, true, false, false, false, false]);
await parts.nth(2).check();                                                              // left hand
let d = await detail();
assert.equal(d.boxes.alice.parts.left_hand.on, true);
assert.equal(d.boxes.alice.parts.right_hand.on, false);
assert.equal(d.boxes.alice.parts.face.on, true);
await page.locator(".mrl-props textarea").first().fill("green eyes, freckles");
await page.locator(".mrl-props textarea").first().press("Tab");
assert.equal((await detail()).boxes.alice.prompt, "green eyes, freckles");
// a part's settings open under its row (one at a time): the prompt its pass will use, denoise, restore, crop
const part = (n) => page.locator(`.mcm-part[data-part="${n}"]`);
const panels = page.locator(".mcm-panel");
const resetBtn = page.locator(".mcm-panel button", { hasText: "Reset prompt" });
assert.equal(await panels.count(), 0);
await part("eyes").locator(".mcm-edit").click();
assert.equal(await panels.count(), 1);
const pp = page.locator(".mcm-panel textarea.mcm-part-prompt");
assert.equal(await pp.inputValue(), "a tall woman, green eyes, freckles, detailed eyes");      // prefilled: what is sent
assert.equal(await resetBtn.isDisabled(), true);
await pp.fill("a tall woman, green eyes, detailed eyes");                                       // no freckles here
await pp.press("Tab");
d = await detail();
assert.equal(d.boxes.alice.parts.eyes.full_prompt, "a tall woman, green eyes, detailed eyes");
assert.equal(await pp.inputValue(), "a tall woman, green eyes, detailed eyes");
assert.equal(await resetBtn.isDisabled(), false);
assert.equal(await part("eyes").locator(".mcm-mod").count(), 1);                               // marked as edited
assert.equal(await part("face").locator(".mcm-mod").count(), 0);
if (process.argv[2]) await page.locator(".mrl-props").screenshot({ path: process.argv[2].replace(/\.png$/, "-part.png") });
await page.locator(".mcm-panel input.mcm-denoise").fill("0.6");
await page.locator(".mcm-panel input.mcm-denoise").press("Tab");
assert.equal((await detail()).boxes.alice.parts.eyes.denoise, 0.6);
await resetBtn.click();                                                                          // back to automatic
assert.equal((await detail()).boxes.alice.parts.eyes.full_prompt, null);
assert.equal(await pp.inputValue(), "a tall woman, green eyes, freckles, detailed eyes");
assert.equal(await resetBtn.isDisabled(), true);
await part("left_hand").locator(".mcm-edit").click();
assert.equal(await panels.count(), 1);                                                          // eyes closed
await page.locator(".mcm-panel input.mcm-restore").fill("0");
await page.locator(".mcm-panel input.mcm-restore").press("Tab");
d = await detail();
assert.equal(d.boxes.alice.parts.left_hand.restore, 0);
assert.equal(d.boxes.alice.parts.right_hand.restore, null);
await part("right_hand").locator(".mcm-part-name").click();                                    // the name opens it too
await page.locator(".mcm-panel input.mcm-crop").check();
d = await detail();
assert.equal(d.boxes.alice.parts.right_hand.whole_crop, true);
assert.equal(d.boxes.alice.parts.left_hand.whole_crop, false);
await part("right_hand").locator(".mcm-edit").click();
assert.equal(await panels.count(), 0);
await page.locator(".mrl-props button", { hasText: "Add detail LoRA" }).click();
await page.locator(".mcm-lora input[type=text]").fill("eye_detail.safetensors");
await page.locator(".mcm-lora input[type=text]").press("Tab");
await page.locator(".mcm-lora input[type=number]").fill("0.5");
await page.locator(".mcm-lora input[type=number]").press("Tab");
assert.deepEqual((await detail()).boxes.alice.loras, [{ lora: "eye_detail.safetensors", weight: 0.5 }]);
await page.locator(".mrl-props .mrl-field", { hasText: "Box LoRAs" }).locator("input").uncheck();
assert.equal((await detail()).boxes.alice.use_box_loras, false);
// ticking a box in the list turns Detail on without selecting it
await row("Lamp").locator("input[type=checkbox]").check();
d = await detail();
assert.equal(d.boxes.lamp.enabled, true);
assert.deepEqual(Object.keys(d.boxes).sort(), ["alice", "lamp"]);                        // bob untouched: defaults
assert.match(await page.locator(".mrl-props .mrl-panel-head").innerText(), /Alice/);
console.log("detail settings ok");

// 4. a run's result: masks on the preview, found / not found per part, click a mask to select its box
await page.evaluate(() => window.node.onExecuted({ moss_character_masks: [{
  image: { filename: "img.png", subfolder: "", type: "temp" }, labels: { filename: "lab.png", subfolder: "", type: "temp" },
  width: 200, height: 100, parts: ["face", "eyes", "left_hand", "right_hand", "hair", "skin"],
  boxes: [{ id: "alice", name: "Alice", outline_found: true, found: { face: true, eyes: true, left_hand: false } },
          { id: "bob", name: "Bob", outline_found: false, found: { face: true, eyes: false } }],
}] }));
await page.waitForFunction(() => /Last run: 2 box/.test(document.querySelector(".mrl-status").textContent));
assert.equal(await row("Alice").locator(".mcm-dot").count(), 3);
assert.equal(await row("Alice").locator(".mcm-dot.missing").count(), 1);
assert.match(await page.locator(".mcm-parts").innerText(), /not found/);
const hit = await page.evaluate(() => {                                                   // a pixel of bob's face
  const c = document.querySelector(".mrl-canvas-wrap canvas").getBoundingClientRect();
  const s = Math.min((c.width - 2) / 200, (c.height - 2) / 100);
  return { x: c.left + (c.width - 200 * s) / 2 + 140 * s, y: c.top + (c.height - 100 * s) / 2 + 30 * s };
});
await page.mouse.click(hit.x, hit.y);
assert.match(await page.locator(".mrl-props .mrl-panel-head").innerText(), /Detail: Bob/);
assert.match(await page.locator(".mrl-props").innerText(), /found no character/);        // outline fallback
assert.match(await page.locator(".mrl-props").innerText(), /no caption/);
console.log("result preview ok");

// hands: an L and an R tag on alice's touching hands; hovering names the part under the pointer
const tags = await page.evaluate(() => window.node.__mcmEditor.tags.map((t) => [t.box, t.text, Math.floor(t.x), Math.floor(t.y)]));
assert.deepEqual(tags.map((t) => t.slice(0, 2)), [[0, "L"], [0, "R"]]);
assert.ok(tags[0][2] >= 60 && tags[0][2] < 70 && tags[1][2] >= 70 && tags[1][2] < 80, JSON.stringify(tags));
const hoverText = async (x, y) => {
  const p = await page.evaluate(([x, y]) => {
    const c = document.querySelector(".mrl-canvas-wrap canvas").getBoundingClientRect();
    const s = Math.min((c.width - 2) / 200, (c.height - 2) / 100);
    return { x: c.left + (c.width - 200 * s) / 2 + x * s, y: c.top + (c.height - 100 * s) / 2 + y * s };
  }, [x, y]);
  await page.mouse.move(p.x, p.y);
  return page.evaluate(() => window.node.__mcmEditor.hoverPart?.text ?? null);
};
assert.equal(await hoverText(75, 52), "Alice · right hand");
if (process.argv[2]) await page.locator(".mrl-canvas-wrap").screenshot({ path: process.argv[2].replace(/\.png$/, "-hands.png") });
assert.equal(await hoverText(64, 52), "Alice · left hand");
assert.equal(await hoverText(40, 30), "Alice · face");
assert.equal(await hoverText(90, 70), null);                                                 // outline only
console.log("hand tags ok");

// 5. painting: brushes add to / remove from the target, undo / redo, painting a part ticks it, Protect, wand
async function imgPx(x, y) {
  return page.evaluate(([x, y]) => {
    const c = document.querySelector(".mrl-canvas-wrap canvas").getBoundingClientRect();
    const s = Math.min((c.width - 2) / 200, (c.height - 2) / 100);
    return { x: c.left + (c.width - 200 * s) / 2 + x * s, y: c.top + (c.height - 100 * s) / 2 + y * s };
  }, [x, y]);
}
const layer = (stored) => P.decode(stored, 200, 100);
const bit = (l, x, y) => l.bits[y * 200 + x];
const count = (l) => l.bits.reduce((a, b) => a + b, 0);
const tool = (title) => page.locator(`.mrl-tools button[title^="${title}"]`).click();
async function strokeImg(pts, at = imgPx) {
  const a = await at(...pts[0]);
  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  for (const p of pts.slice(1)) {
    const b = await at(...p);
    await page.mouse.move(b.x, b.y, { steps: 6 });
  }
  await page.mouse.up();
}
await row("Alice").click();
await hoverCanvas();
await page.keyboard.press("b");
assert.ok(await page.locator(".mcm-opts").isVisible());
assert.equal(await page.locator(".mcm-opts select").inputValue(), "character");
await page.evaluate(() => { window.node.__mcmEditor.brush = 3; });
await strokeImg([[120, 60], [170, 60]]);                                                   // her arm, across bob
d = await detail();
let add = layer(d.boxes.alice.paint.character.add);
assert.ok(bit(add, 120, 60) && bit(add, 145, 60) && bit(add, 170, 60) && !bit(add, 145, 70));
await page.keyboard.press("n");                                                             // remove brush
await strokeImg([[140, 60], [150, 60]]);
d = await detail();
add = layer(d.boxes.alice.paint.character.add);
const sub = layer(d.boxes.alice.paint.character.sub);
assert.ok(!bit(add, 145, 60) && bit(sub, 145, 60) && bit(add, 125, 60));
await page.keyboard.press("Control+z");
assert.equal((await detail()).boxes.alice.paint.character.sub, null);                     // undone
await page.keyboard.press("Control+Shift+z");
assert.ok((await detail()).boxes.alice.paint.character.sub);                               // redone
// the eraser wipes add and remove paint alike, back to SAM3's result; elsewhere both stay
await page.keyboard.press("e");
assert.equal(await page.evaluate(() => window.node.__mcmEditor.tool), "erase");
await strokeImg([[143, 60], [147, 60]]);                                                    // remove paint
await strokeImg([[122, 60], [126, 60]]);                                                    // add paint
d = await detail();
let addE = layer(d.boxes.alice.paint.character.add), subE = layer(d.boxes.alice.paint.character.sub);
assert.ok(!bit(addE, 145, 60) && !bit(subE, 145, 60) && !bit(addE, 124, 60) && !bit(subE, 124, 60));
assert.ok(bit(addE, 165, 60));                                                              // untouched add paint
await page.keyboard.press("Control+z");                                                     // one undo per stroke
d = await detail();
assert.ok(bit(layer(d.boxes.alice.paint.character.add), 124, 60));
await page.keyboard.press("Control+z");
assert.ok(bit(layer((await detail()).boxes.alice.paint.character.sub), 145, 60));
await page.locator(".mcm-opts select").selectOption("right_hand");
await tool("Add brush");
await strokeImg([[30, 80], [40, 80]]);
d = await detail();
assert.equal(d.boxes.alice.parts.right_hand.on, true);                                     // painting a part ticks it
assert.ok(bit(layer(d.boxes.alice.paint.right_hand.add), 35, 80));
await page.locator(".mcm-opts select").selectOption("protect");
await strokeImg([[60, 10], [90, 10]]);
assert.ok(bit(layer((await detail()).protect), 75, 10));                                    // text: never detailed
await tool("Magic wand");
await page.locator(".mcm-opts select").selectOption("hair");
let q = await imgPx(20, 20);
await page.mouse.click(q.x, q.y);
d = await detail();
const picked = layer(d.boxes.alice.paint.hair.add);
assert.ok(bit(picked, 20, 20) && count(picked) > 50 && count(picked) < 200 * 100);
if (process.argv[2]) await page.screenshot({ path: process.argv[2] });
await page.keyboard.down("Alt");
await page.mouse.click(q.x, q.y);
await page.keyboard.up("Alt");
assert.equal((await detail()).boxes.alice.paint.hair.add, null);                           // Alt-click removed them
console.log("paint ok");

// 6. under the graph's zoom (ComfyUI CSS-scales DOM widgets): the canvas keeps its layout size, a brush
// dab lands where it is made, and the cursor is the editor's, not ComfyUI's drag hand
await page.evaluate(() => {
  const host = document.getElementById("host");
  host.style.transform = "scale(0.6)";
  host.style.transformOrigin = "0 0";
});
await row("Bob").click();
await row("Alice").click();                                                                // re-selects and redraws
const geo = await page.evaluate(() => {
  const wrap = document.querySelector(".mrl-canvas-wrap");
  const c = wrap.querySelector("canvas");
  return { canvasW: c.clientWidth, wrapW: wrap.clientWidth, canvasH: c.clientHeight, wrapH: wrap.clientHeight };
});
assert.equal(geo.canvasW, geo.wrapW);
assert.equal(geo.canvasH, geo.wrapH);
async function zoomedPx(x, y) {
  return page.evaluate(([x, y]) => {
    const wrap = document.querySelector(".mrl-canvas-wrap");
    const r = wrap.querySelector("canvas").getBoundingClientRect();
    const z = r.width / wrap.clientWidth;
    const s = Math.min((wrap.clientWidth - 2) / 200, (wrap.clientHeight - 2) / 100);
    return { x: r.left + ((wrap.clientWidth - 200 * s) / 2 + x * s) * z, y: r.top + ((wrap.clientHeight - 100 * s) / 2 + y * s) * z };
  }, [x, y]);
}
await page.locator(".mcm-opts select").selectOption("character");
await tool("Add brush");
await strokeImg([[150, 30], [150, 30]], zoomedPx);
add = layer((await detail()).boxes.alice.paint.character.add);
assert.ok(bit(add, 150, 30) && !bit(add, 158, 30));
const cursor = () => page.evaluate(() => document.querySelector(".mrl-canvas-wrap canvas").style.cursor);
q = await zoomedPx(40, 70);
await page.mouse.move(q.x, q.y);
assert.equal(await cursor(), "none");                                                      // the brush draws its own circle
await tool("Magic wand");
await page.mouse.move(q.x + 2, q.y);
assert.equal(await cursor(), "crosshair");
await tool("Select");
await page.mouse.move(q.x, q.y);
assert.equal(await cursor(), "default");
await page.evaluate(() => { document.getElementById("host").style.transform = ""; });
console.log("zoom ok");

// 7. zoom and pan: a brush dab while zoomed and panned lands on the right image pixel; panning edits nothing
if ((await page.evaluate(() => window.node.__mcmEditor.selected)) !== "alice") await row("Alice").click();
const mv = () => page.evaluate(() => { const v = window.node.__mcmEditor.view(); return { s: v.s, ox: v.ox, oy: v.oy }; });
async function mImg(x, y) {
  const v = await mv();
  return page.evaluate(([x, y, v]) => {
    const r = document.querySelector(".mrl-canvas-wrap canvas").getBoundingClientRect();
    return { x: r.left + v.ox + x * v.s, y: r.top + v.oy + y * v.s };
  }, [x, y, v]);
}
const m0 = await mv();
let mq = await mImg(150, 60);
await page.mouse.move(mq.x, mq.y);
await page.mouse.wheel(0, -500);
await page.waitForTimeout(50);
assert.ok((await mv()).s > m0.s * 1.2);
await hoverCanvas();
await page.keyboard.press("h");
const before7 = JSON.stringify(await detail());
const cb = await page.locator(".mrl-canvas-wrap canvas").boundingBox();
await page.mouse.move(cb.x + cb.width / 2, cb.y + cb.height / 2);
await page.mouse.down();
await page.mouse.move(cb.x + cb.width / 2 - 40, cb.y + cb.height / 2 + 15, { steps: 4 });
await page.mouse.up();
assert.equal(JSON.stringify(await detail()), before7);                                     // panning edits nothing
// the middle button pans with any tool (ComfyUI would otherwise hand it to the graph)
await page.keyboard.press("v");
const mBefore = await mv();
await page.mouse.move(cb.x + cb.width / 2, cb.y + cb.height / 2);
await page.mouse.down({ button: "middle" });
await page.mouse.move(cb.x + cb.width / 2 + 30, cb.y + cb.height / 2, { steps: 4 });
await page.mouse.up({ button: "middle" });
assert.ok(Math.abs((await mv()).ox - mBefore.ox - 30) < 1, "middle-drag pans the masks canvas");
assert.equal(JSON.stringify(await detail()), before7);
assert.match(await page.locator(".mrl-props .mrl-panel-head").innerText(), /Alice/);      // or deselects
await page.keyboard.press("b");
await strokeImg([[100, 40], [100, 40]], mImg);
const dab = layer((await detail()).boxes.alice.paint.character.add);
assert.ok(bit(dab, 100, 40) && !bit(dab, 108, 40));
await page.locator('.mrl-tools button[title^="Fit"]').click();
assert.ok(Math.abs((await mv()).s - m0.s) < 1e-9);
console.log("masks zoom/pan ok");

// 8. disconnected: says what to connect
await page.evaluate(() => { window.node.inputs[1].link = null; });
await page.waitForFunction(() => /Connect the regions output/.test(document.querySelector(".mcm-boxes").textContent));
console.log("disconnected ok");

const comfy = await page.evaluate(() => ({ wheel: window.__graphWheel, undo: window.__comfyUndo, pan: window.__graphPan }));
assert.deepEqual(comfy, { wheel: 0, undo: 0, pan: 0 }, "ComfyUI's own wheel / undo / middle-button forwarding never ran for editor input");
console.log("comfyui wheel/undo/pan stand-ins untouched ok");
assert.deepEqual(errors, []);
console.log("PASS");
await browser.close();
server.close();
