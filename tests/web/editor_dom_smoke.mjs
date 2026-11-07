// Browser smoke test of web/regional_lora_editor.js against a STUB of ComfyUI's frontend API
// (scripts/app.js, scripts/api.js, node.addDOMWidget). It checks the editor's own behaviour —
// drawing, writing the JSON widget, hand-edit sync, parse-error safety, undo, keyboard — not
// ComfyUI integration. Needs Playwright with Chromium; skips if unavailable.
//
//   node tests/web/editor_dom_smoke.mjs [final.png] [curve-panel.png]

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
export const app = {
  exts: [],
  graph: { setDirtyCanvas() {} },
  registerExtension(e) { this.exts.push(e); },
};
window.__app = app;`;
const API_STUB = `
export const api = {
  fetchApi: async (path) => ({
    ok: true,
    json: async () => path === "/models/loras" ? ["alice.safetensors", "bob.safetensors"] : ["ref.png [input]"],
  }),
  apiURL: (p) => "/missing" + p,
};`;
const PAGE = `<!doctype html><html><body style="margin:0;background:#111">
<textarea id="raw" style="width:700px;height:120px"></textarea>
<div id="host" style="width:700px;height:620px"></div>
<script type="module">
import "/scripts/app.js";
const app = window.__app;
await import("/extensions/moss/regional_lora_editor.js");
const ext = app.exts[0];
const raw = document.getElementById("raw");
function Node() {
  this.size = [700, 700];
  this.widgets = [
    { name: "width", value: 1024 },
    { name: "height", value: 768 },
    { name: "regions", get value() { return raw.value; }, set value(v) { raw.value = v; } },
  ];
}
Node.prototype.addDOMWidget = function (name, type, element, options) {
  element.style.height = "100%";
  document.getElementById("host").appendChild(element);
  return { name, element, options };
};
Node.prototype.setSize = function (s) { this.size = s; };
Node.prototype.setDirtyCanvas = function () {};
await ext.beforeRegisterNodeDef(Node, { name: "MossRegionalLora" });
window.node = new Node();
window.node.onNodeCreated();
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
const page = await browser.newPage({ viewport: { width: 760, height: 820 }, deviceScaleFactor: 2 });
const errors = [];
page.on("pageerror", (e) => errors.push(String(e)));
await page.goto(base);
await page.waitForFunction(() => window.ready === true);
const json = async () => JSON.parse(await page.locator("#raw").inputValue());

// geometry of the drawn canvas: 1024x768 fitted into the wrap
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
async function drag(from, to, mods = []) {
  const a = await toScreen(...from);
  const b = await toScreen(...to);
  for (const m of mods) await page.keyboard.down(m);
  await page.mouse.move(a.x, a.y);
  await page.mouse.down();
  await page.mouse.move((a.x + b.x) / 2, (a.y + b.y) / 2, { steps: 4 });
  await page.mouse.move(b.x, b.y, { steps: 4 });
  await page.mouse.up();
  for (const m of mods) await page.keyboard.up(m);
}

// 1. draw a region: canvas px (112, 84) -> (512, 384) = centred [-300, -400, 0, 0] (rounded)
await drag([112, 84], [512, 384]);
let s = await json();
assert.equal(s.regions.length, 1);
const [y0, x0, y1, x1] = s.regions[0].bbox;
assert.ok(Math.abs(y0 + 300) <= 2 && Math.abs(x0 + 400) <= 2 && Math.abs(y1) <= 2 && Math.abs(x1) <= 2, `bbox ${s.regions[0].bbox}`);
assert.match(await page.locator(".mrl-readout").innerText(), /\[y_min, x_min, y_max, x_max\]/);
console.log("draw ok", s.regions[0].bbox);

// 2. keyboard nudge (canvas focused after the drag)
await page.keyboard.press("ArrowRight");
await page.keyboard.press("Shift+ArrowDown");
s = await json();
assert.deepEqual(s.regions[0].bbox, [y0 + 10, x0 + 1, y1 + 10, x1 + 1]);
console.log("nudge ok");

// 3. undo twice, redo once
await page.keyboard.press("Control+z");
await page.keyboard.press("Control+z");
assert.deepEqual((await json()).regions[0].bbox, [y0, x0, y1, x1]);
await page.keyboard.press("Control+Shift+z");
assert.deepEqual((await json()).regions[0].bbox, [y0, x0 + 1, y1, x1 + 1]);
console.log("undo/redo ok");

// 4. move by dragging the body
await drag([300, 250], [400, 250]);
s = await json();
assert.ok(Math.abs(s.regions[0].bbox[1] - (x0 + 1 + 100)) <= 2, `moved ${s.regions[0].bbox}`);
console.log("move ok");

// 5. hand-edited JSON is reflected without reload
const pasted = {
  version: 1,
  regions: [
    { id: "a", name: "Alice", bbox: [-384, -512, 384, 0], lora: "alice.safetensors", weight: 1, visible: true, locked: false, color: "#e6194b", group: null },
    { id: "b", name: "Bob", bbox: [-384, 0, 384, 600], lora: "bob.safetensors", weight: 0.7, visible: true, locked: true, color: "#4363d8", group: null },
  ],
  groups: [],
  editor: { grid_divisions: 3, snap: false, reference_image: "missing.png", reference_opacity: 0.5 },
};
await page.locator("#raw").fill(JSON.stringify(pasted, null, 2));
await page.waitForFunction(() => document.querySelectorAll(".mrl-row .mrl-bbox").length === 2, null, { timeout: 2000 });
assert.equal(await page.locator(".mrl-oob").count(), 1); // Bob extends past x = +512
console.log("hand edit sync ok");

// 6. a locked region does not move
await drag([700, 300], [800, 300]);
assert.deepEqual((await json()).regions[1].bbox, [-384, 0, 384, 600]);
console.log("lock ok");

// 7. reorder with the explicit buttons
await page.locator(".mrl-row").nth(1).locator("button", { hasText: "▲" }).click();
assert.deepEqual((await json()).regions.map((r) => r.name), ["Bob", "Alice"]);
console.log("reorder ok");

// 7b. layout-prompt fields on the first box
const first = page.locator(".mrl-item").first().locator(".mrl-prompt");
await first.locator("select").selectOption("text");
await page.locator(".mrl-item").first().locator(".mrl-prompt input.mrl-caption").fill("neon sign");
await page.locator(".mrl-item").first().locator(".mrl-prompt input.mrl-caption").press("Tab");
await page.locator(".mrl-item").first().locator(".mrl-prompt input.mrl-render").fill("OPEN");
await page.locator(".mrl-item").first().locator(".mrl-prompt input.mrl-render").press("Tab");
await page.locator(".mrl-item").first().locator(".mrl-prompt button", { hasText: "+ colour" }).click();
s = await json();
assert.equal(s.regions[0].label, "text");
assert.equal(s.regions[0].caption, "neon sign");
assert.equal(s.regions[0].text, "OPEN");
assert.deepEqual(s.regions[0].palette, ["#808080"]);
console.log("prompt fields ok");

// 7c. strength curve panel (edits the selected box: Bob after the reorder above)
await page.locator(".mrl-row").first().click({ position: { x: 300, y: 8 } }).catch(() => {});
async function graphPx(x, y) {
  return page.evaluate(([x, y]) => {
    const c = document.querySelector(".mrl-curve-wrap canvas");
    const r = c.getBoundingClientRect();
    const PAD = { l: 34, r: 10, t: 8, b: 18 };
    return { x: r.left + PAD.l + x * (r.width - PAD.l - PAD.r), y: r.top + PAD.t + (1 - y) * (r.height - PAD.t - PAD.b) };
  }, [x, y]);
}
const sel = (await json()).regions.findIndex((r) => r.name === (s.regions[0].name));
await page.locator(".mrl-curve-bar button", { hasText: "Stop at 50%" }).click();
s = await json();
assert.deepEqual(s.regions[sel].curve, [{ x: 0, y: 1, hold: true }, { x: 0.5, y: 0 }, { x: 1, y: 0 }]);
// add a point at (75%, 60%) on the curve's second segment
let q = await graphPx(0.75, 0.6);
await page.mouse.click(q.x, q.y);
s = await json();
assert.equal(s.regions[sel].curve.length, 4);
assert.ok(Math.abs(s.regions[sel].curve[2].x - 0.75) < 0.02 && Math.abs(s.regions[sel].curve[2].y - 0.6) < 0.03, JSON.stringify(s.regions[sel].curve));
// bend the segment from (50%, 0) to (75%, 60%) by dragging its midpoint handle down
const c = s.regions[sel].curve;
q = await graphPx((c[1].x + c[2].x) / 2, (c[1].y + c[2].y) / 2);
await page.mouse.move(q.x, q.y);
await page.mouse.down();
const q2 = await graphPx((c[1].x + c[2].x) / 2, 0.1);
await page.mouse.move(q2.x, q2.y, { steps: 4 });
await page.mouse.up();
s = await json();
assert.ok(s.regions[sel].curve[1].bend > 0.2, JSON.stringify(s.regions[sel].curve)); // eased in
// double-click the added point to remove it
q = await graphPx(s.regions[sel].curve[2].x, s.regions[sel].curve[2].y);
await page.mouse.dblclick(q.x, q.y);
s = await json();
assert.equal(s.regions[sel].curve.length, 3);
// Delete with the graph focused must not delete the box
await page.locator(".mrl-curve-wrap canvas").focus();
await page.keyboard.press("Delete");
assert.equal((await json()).regions.length, 2);
// undo reaches curve edits too
await page.keyboard.press("Control+z");
assert.equal((await json()).regions[sel].curve.length, 4);
console.log("curve panel ok");
if (process.argv[3]) await page.screenshot({ path: process.argv[3] });

// 8. parse error: visible error state, and nothing the user does on the canvas overwrites the text
const broken = '{"version": 1, "regions": [ {"name": "oops"';
await page.locator("#raw").fill(broken);
await page.waitForFunction(() => getComputedStyle(document.querySelector(".mrl-error")).display !== "none", null, { timeout: 2000 });
await drag([100, 100], [300, 300]);
await page.locator(".mrl-canvas-wrap canvas").focus();
await page.keyboard.press("Delete");
await page.keyboard.press("Control+z");
assert.equal(await page.locator("#raw").inputValue(), broken);
console.log("parse-error safety ok");

// fix it again -> editor recovers
await page.locator("#raw").fill(JSON.stringify(pasted, null, 2));
await page.waitForFunction(() => getComputedStyle(document.querySelector(".mrl-error")).display === "none", null, { timeout: 2000 });
await page.locator(".mrl-row").first().click({ position: { x: 300, y: 8 } }).catch(() => {});
if (process.argv[2]) await page.screenshot({ path: process.argv[2] });
assert.deepEqual(errors, []);
console.log("PASS");
await browser.close();
server.close();
