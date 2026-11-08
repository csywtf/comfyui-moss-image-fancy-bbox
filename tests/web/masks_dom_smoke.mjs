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
    if (x >= 120 && x < 178 && y >= 12 && y < 88) { d.data[i] = 2; d.data[i + 2] = 255; }
    if (x >= 130 && x < 150 && y >= 20 && y < 40) d.data[i + 1] = 1;               // bob face
  }
  g.putImageData(d, 0, 0);
});
const image = png(200, 100, (g) => { const gr = g.createLinearGradient(0, 0, 200, 100); gr.addColorStop(0, "#456"); gr.addColorStop(1, "#cba"); g.fillStyle = gr; g.fillRect(0, 0, 200, 100); });
window.__views = { "img.png": image, "lab.png": labels };
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
const parts = page.locator(".mcm-parts input[type=checkbox]");
assert.deepEqual(await parts.evaluateAll((xs) => xs.map((x) => x.checked)), [true, true, false, false, false]);
await parts.nth(2).check();                                                              // hands
let d = await detail();
assert.equal(d.boxes.alice.parts.hands.on, true);
assert.equal(d.boxes.alice.parts.face.on, true);
await page.locator(".mrl-props textarea").fill("green eyes, freckles");
await page.locator(".mrl-props textarea").press("Tab");
assert.equal((await detail()).boxes.alice.prompt, "green eyes, freckles");
const eyesDenoise = page.locator(".mcm-parts input[type=number]").nth(1);
await eyesDenoise.fill("0.6");
await eyesDenoise.press("Tab");
assert.equal((await detail()).boxes.alice.parts.eyes.denoise, 0.6);
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
  width: 200, height: 100, parts: ["face", "eyes", "hands", "hair", "skin"],
  boxes: [{ id: "alice", name: "Alice", outline_found: true, found: { face: true, eyes: true, hands: false } },
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
if (process.argv[2]) await page.screenshot({ path: process.argv[2] });
console.log("result preview ok");

// 5. disconnected: says what to connect
await page.evaluate(() => { window.node.inputs[1].link = null; });
await page.waitForFunction(() => /Connect the regions output/.test(document.querySelector(".mcm-boxes").textContent));
console.log("disconnected ok");

assert.deepEqual(errors, []);
console.log("PASS");
await browser.close();
server.close();
