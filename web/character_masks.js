// Editor for "MOSS-Image Character Masks":
//
//   [boxes] [ preview: last run's image with the masks ] [detail settings of the selected box]
//
// The boxes are read live from the connected "MOSS-Image Regional LoRA + Layout Prompt" node (its regions
// widget, through the `regions` link). The detail settings are stored in this node's hidden `detail`
// widget, never in the main node, so changing them doesn't invalidate the generated image in ComfyUI's
// cache. All of this node's inputs are hidden and edited here.

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import * as G from "./geometry.js";
import * as S from "./state.js";
import * as D from "./detail_state.js";
import { el, hideWidget, injectCss, loraDisplayName, loraList, rememberHeight, zoomButtons, zoomKey } from "./regional_lora_editor.js";
import { Viewport, frameRect, installKeySink, installPanZoom, isKeySink, trackSize } from "./viewport.js";
import * as P from "./paint.js";

// Tools in the floating toolbar: {id, icon (a name in icons.js), key, title}.
const TOOLS = [
  { id: "select", icon: "mouse-pointer-2", key: "v", title: "Select (V): click a character to select its box." },
  { id: "hand", icon: "hand", key: "h", title: "Hand (H): drag to pan. Any tool: hold Space or drag with the middle button to pan; the wheel zooms." },
  { id: "add", icon: "paintbrush", badge: "+", key: "b", title: "Add brush (B): paint pixels into the target (the selected box's character or part, or Protect). [ and ] change the size." },
  { id: "sub", icon: "paintbrush", badge: "\u2212", key: "n", title: "Remove brush (N): paint pixels out of the target (\"this isn't the hand\"). [ and ] change the size." },
  { id: "erase", icon: "eraser", key: "e", title: "Eraser (E): wipe your add and remove paint, so those pixels go back to what SAM3 found. [ and ] change the size." },
  { id: "wand", icon: "wand-sparkles", key: "w", title: "Magic wand (W): click to add similar-coloured pixels to the target, Alt-click to remove them. Tolerance and 'connected' in the tool options." },
];
const BRUSH_TOOLS = ["add", "sub", "erase"];
const PAINT_TOOLS = [...BRUSH_TOOLS, "wand"];
const ADD_RGB = "#ffffff";

const NODE_NAME = "MossCharacterMasks";
const EDITOR_WIDGET = "masks_editor";
const UI_KEY = "moss_character_masks";
const POLL_MS = 250;
const SETTINGS = [
  ["threshold", "Threshold", 0.01, "How confident SAM3 must be to accept a part. Lower finds more."],
  ["box_margin", "Box margin", 0.01, "How far outside each box (as a fraction of its size) to look for its parts."],
  ["refine_iterations", "Refine", 1, "SAM3 mask refinement passes (0 = raw detector masks)."],
];

// The prompt writer's settings (hidden widgets of the node), edited in the Settings panel.
const LLM_SETTINGS = [
  ["llm_max_length", "Max tokens", 1, "Longest reply, in tokens."],
  ["llm_temperature", "Temperature", 0.01, "Randomness (0 = always the most likely words)."],
  ["llm_top_p", "Top p", 0.01, "Only the most likely words that add up to this probability are considered."],
  ["llm_top_k", "Top k", 1, "Only this many of the most likely words are considered (0 = no limit)."],
  ["llm_min_p", "Min p", 0.01, "Words less likely than this fraction of the most likely one are left out."],
  ["llm_repetition_penalty", "Repetition", 0.01, "Penalty for repeating words (1 = none)."],
  ["llm_seed", "Seed", 1, "Random seed of the prompt writer."],
];
// The built-in system prompt: the same file the node reads, so the two never drift apart.
let defaultSystemPrompt = null;
const defaultSystemReady = fetch(new URL("./prompt_writer_system.txt", import.meta.url))
  .then((r) => (r.ok ? r.text() : ""))
  .catch(() => "")
  .then((t) => (defaultSystemPrompt = t.trim()));

const CSS = `
.mcm-boxes{flex:0 0 220px;width:220px;min-width:0}
.mcm-row{display:flex;align-items:center;gap:5px;padding:3px 5px;border-bottom:1px solid rgba(128,128,128,.2);cursor:default;user-select:none}
.mcm-row:hover{background:rgba(128,128,128,.12)}
.mcm-row.sel{background:rgba(80,140,255,.28)}
.mcm-row.off .mrl-name{opacity:.5}
.mcm-dots{display:flex;gap:2px}
.mcm-dot{width:8px;height:8px;border-radius:50%;border:1px solid rgba(0,0,0,.4)}
.mcm-dot.missing{background:transparent!important;border:1px dashed #f88}
.mcm-parts{display:flex;flex-direction:column;gap:1px}
.mcm-part{display:flex;align-items:center;gap:5px;padding:1px 2px;border-radius:3px}
.mcm-part.open{background:rgba(80,140,255,.18)}
.mcm-part .grow,.mcm-part-name{cursor:pointer}
.mcm-part .grow{flex:1;align-self:stretch}
.mcm-part.needs .mcm-part-name{color:#f99}
.mcm-need-text{font-size:10px;color:#f99}
.mcm-need-text.writer{color:#9cf}
.mcm-part.needs.writer .mcm-part-name{color:inherit}
.mcm-gen{font-size:10px;padding:0 4px;border-radius:3px;background:rgba(80,200,140,.3)}
.mcm-llm-on{color:#9cf;opacity:1}
.mcm-system{display:flex;flex-direction:column;gap:3px}
.mrl-props .mcm-system textarea{min-height:120px;resize:vertical;width:100%;box-sizing:border-box;flex:none;font-size:11px}
.mcm-need{font:bold 10px/14px sans-serif;width:14px;height:14px;border-radius:7px;background:#c33;color:#fff;text-align:center;flex:0 0 auto}
.mcm-caption-ref{font-size:11px;line-height:1.35;opacity:.85;padding:4px 6px;border:1px dashed rgba(128,128,128,.5);border-radius:3px;max-height:120px;overflow:auto;white-space:pre-wrap;user-select:text}
.mrl-sec-title .grow{flex:1}
.mcm-mod{font-size:10px;padding:0 4px;border-radius:3px;background:rgba(80,140,255,.35)}
.mcm-panel{margin:1px 0 6px 18px;padding:5px 6px;border:1px solid #6af;border-radius:4px;display:flex;flex-direction:column;gap:4px}
.mcm-panel-head{display:flex;align-items:center;gap:6px;font-size:11px;opacity:.9}
.mcm-panel-head .grow{flex:1}
.mrl-props .mcm-panel textarea{min-height:56px;resize:vertical;width:100%;box-sizing:border-box;flex:none}
.mcm-panel input[type=number]{width:72px}
.mcm-chip{width:10px;height:10px;border-radius:2px}
.mcm-status{font-size:10px;opacity:.8;margin:-1px 0 2px 20px}
.mcm-status.missing{color:#f99;opacity:1}
.mcm-loras{display:flex;flex-direction:column;gap:3px}
.mcm-lora{display:flex;gap:3px;align-items:center}
.mcm-lora input[type=text]{flex:1;min-width:0}
.mcm-lora input[type=number]{width:54px}
.mcm-empty{padding:8px;opacity:.7}
.mrl-tools button{position:relative}
.mcm-badge{position:absolute;right:-2px;bottom:-2px;min-width:11px;height:11px;border-radius:6px;background:#111;border:1px solid #888;font:bold 10px/11px sans-serif;text-align:center;pointer-events:none}
.mcm-opts{position:absolute;left:52px;top:8px;right:8px;z-index:2;display:none;flex-wrap:wrap;align-items:center;gap:6px;padding:4px 6px;background:rgba(30,30,30,.92);border:1px solid var(--border-color,#555);border-radius:6px;box-shadow:0 2px 10px rgba(0,0,0,.45)}
.mcm-opts.on{display:flex}
.mcm-opts label{display:inline-flex;align-items:center;gap:4px}
.mcm-opts input[type=range]{width:90px}
.mcm-opts .mcm-warn{color:#fb4}
`;

function injectMasksCss() {
  injectCss();
  let s = document.getElementById("mcm-css");
  if (!s) {
    s = document.createElement("style");
    s.id = "mcm-css";
    document.head.appendChild(s);
  }
  s.textContent = CSS;
}

/** The node feeding `regions`, following reroutes. */
function regionsSource(node) {
  const graph = node.graph || app.graph;
  let cur = node;
  let slot = (node.inputs || []).findIndex((i) => i.name === "regions");
  for (let hops = 0; hops < 16 && slot >= 0; hops++) {
    let src = cur.getInputNode?.(slot);
    if (src === undefined) {
      const link = graph?.links?.[cur.inputs[slot].link] ?? graph?.links?.get?.(cur.inputs[slot].link);
      src = link ? graph.getNodeById(link.origin_id) : null;
    }
    if (!src) return null;
    if (src.widgets?.some((w) => w.name === "regions")) return src;
    cur = src;
    slot = (src.inputs || []).length ? 0 : -1; // a reroute: follow its single input
  }
  return null;
}

class MasksEditor {
  constructor(node) {
    this.node = node;
    this.selected = null;
    this.tool = "select";
    this.viewport = new Viewport();
    this.history = new S.History(100); // undo / redo of the detail settings (paint strokes included)
    this.paintTarget = "character"; // "character", a part name, or "protect"
    this.brush = 20; // brush radius, image px
    this.tolerance = 32; // magic wand, 0-255 per channel
    this.connected = true;
    this.pixels = null; // ImageData of the last run's image, for the wand
    this.layers = null; // {key, add, sub} bitmaps of the target being painted
    this.stroke = null;
    this.hover = null;
    this.result = null; // the last run's payload
    this.image = null; // HTMLImageElement of the last run's image
    this.labels = null; // ImageData of its label map
    this.overlay = null; // canvas with the overlay for the current selection
    this.loras = [];
    this.sig = "";
    injectMasksCss();
    this.build();
    loraList().then((l) => {
      this.loras = l;
      this.renderProps();
    });
    this.timer = setInterval(() => this.poll(), POLL_MS);
    this.wrapSize = trackSize(this.wrap, () => this.draw());
    this.ro = this.wrapSize.ro;
    this.canvasRect = frameRect(this.canvas);
    this.poll();
    defaultSystemReady.then(() => {
      if (!this.selected) this.renderProps(); // the Settings panel shows the built-in system prompt
    });
  }

  destroy() {
    clearInterval(this.timer);
    this.ro?.disconnect();
    this.panZoom?.dispose();
  }

  widget(name) {
    return this.node.widgets?.find((w) => w.name === name);
  }

  setWidget(name, value) {
    const w = this.widget(name);
    if (!w || w.value === value) return;
    w.value = value;
    w.callback?.(value, app.canvas, this.node);
    this.node.setDirtyCanvas?.(true, true);
    this.node.graph?.change?.();
  }

  // ------------------------------------------------------------------ DOM
  build() {
    this.list = el("div", { class: "mrl-panel-body" });
    const boxes = el("div", { class: "mrl-panel mcm-boxes" }, [el("div", { class: "mrl-panel-head", text: "Boxes" }), this.list]);
    this.canvas = el("canvas");
    this.toolButtons = {};
    const tools = el("div", { class: "mrl-tools" }, [
      ...TOOLS.map((t) => {
        const b = el("button", { icon: t.icon, iconSize: 16, title: t.title, onclick: () => this.setTool(t.id) });
        if (t.badge) b.append(el("span", { class: "mcm-badge", text: t.badge }));
        return (this.toolButtons[t.id] = b);
      }),
      el("div", { class: "mrl-tool-sep" }),
      ...zoomButtons(this),
    ]);
    this.opts = el("div", { class: "mcm-opts" });
    this.wrap = el("div", { class: "mrl-canvas-wrap" }, [this.canvas, tools, this.opts]);
    this.propsTitle = el("span", { class: "grow" });
    this.props = el("div", { class: "mrl-panel-body" });
    const props = el("div", { class: "mrl-panel mrl-props" }, [el("div", { class: "mrl-panel-head" }, [this.propsTitle]), this.props]);
    this.errorBox = el("div", { class: "mrl-error" });
    this.statusLine = el("div", { class: "mrl-status" });
    this.root = el("div", { class: "mrl-root mcm-root", tabindex: "-1" }, [this.errorBox, el("div", { class: "mrl-main" }, [boxes, this.wrap, props]), this.statusLine]);
    this.canvas.style.cursor = "default"; // not ComfyUI's drag hand
    this.canvas.addEventListener("pointerdown", (e) => this.onCanvasDown(e));
    this.canvas.addEventListener("pointermove", (e) => this.onCanvasMove(e));
    this.canvas.addEventListener("pointerup", (e) => this.onCanvasUp(e));
    this.canvas.addEventListener("pointercancel", (e) => this.onCanvasUp(e));
    this.canvas.addEventListener("pointerleave", () => {
      this.hover = null;
      this.hoverPart = null;
      this.requestDraw();
    });
    this.root.addEventListener("pointerdown", (e) => e.stopPropagation());
    this.root.addEventListener("keydown", (e) => {
      e.stopPropagation(); // typing here isn't a graph shortcut
      if (/^(input|textarea|select)$/i.test(e.target?.tagName || "") && !isKeySink(e.target)) return;
      if ((e.ctrlKey || e.metaKey) && /^[zZyY]$/.test(e.key)) {
        e.preventDefault();
        return e.key.toLowerCase() === "y" || e.shiftKey ? this.redo() : this.undo();
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      if (zoomKey(this, e)) return;
      if (e.key === "[" || e.key === "]") {
        this.brush = Math.max(1, Math.min(300, Math.round(this.brush * (e.key === "]" ? 1.25 : 0.8))));
        this.renderOptions();
        return this.draw();
      }
      const tool = TOOLS.find((t) => t.key === e.key.toLowerCase());
      if (tool) this.setTool(tool.id);
    });
    // keyboard focus ComfyUI leaves alone: its Ctrl+Z undo and wheel forwarding stand aside (see viewport.js)
    this.keys = installKeySink(this.root);
    this.panZoom = installPanZoom({
      focus: () => this.keys.focus(),
      canvas: this.canvas,
      root: this.root,
      viewport: this.viewport,
      size: () => {
        const v = this.view();
        return { W: v.W, H: v.H, cw: v.cw, ch: v.ch };
      },
      handTool: () => this.tool === "hand",
      redraw: () => this.draw(),
      cursor: (k) => (this.canvas.style.cursor = k || "default"),
    });
    this.renderTools();
  }

  setTool(id) {
    this.tool = id;
    this.renderTools();
    this.renderOptions();
    this.canvas.style.cursor = this.toolCursor();
    this.draw();
  }

  toolCursor() {
    return { hand: "grab", add: "none", sub: "none", erase: "none", wand: "crosshair" }[this.tool] || "default";
  }

  /** The tool options strip over the canvas: what to paint, brush size, wand tolerance. */
  renderOptions() {
    const paint = PAINT_TOOLS.includes(this.tool);
    this.opts.classList.toggle("on", paint);
    if (!paint) return this.opts.replaceChildren();
    const target = el("select", { title: "What the brushes and the wand paint" });
    const targets = [["character", "Character (whole outline)"], ...D.PARTS.map((p) => [p.name, p.label]), ["protect", "Protect (all boxes: never detailed)"]];
    for (const [v, t] of targets) target.appendChild(el("option", { value: v, text: t }));
    target.value = this.paintTarget;
    target.addEventListener("change", () => {
      this.paintTarget = target.value;
      this.layers = null;
      this.paintCanvas = null;
      this.renderOptions();
      this.draw();
    });
    const items = [el("label", {}, [el("span", { text: "Target" }), target])];
    if (this.tool === "wand") {
      const tol = el("input", { type: "range", min: "0", max: "100", step: "1", value: String(Math.round(this.tolerance / 2.55)), title: "How different a colour may be and still be picked" });
      const tolText = el("span", { text: String(Math.round(this.tolerance / 2.55)) });
      tol.addEventListener("input", () => {
        this.tolerance = Math.round(Number(tol.value) * 2.55);
        tolText.textContent = tol.value;
      });
      const con = el("input", { type: "checkbox", title: "Only pixels joined to the clicked one; off: every similar pixel in the selected box" });
      con.checked = this.connected;
      con.addEventListener("change", () => (this.connected = con.checked));
      items.push(el("label", {}, [el("span", { text: "Tolerance" }), tol, tolText]), el("label", {}, [con, el("span", { text: "Connected" })]), el("span", { class: "mrl-hint", text: "Alt-click removes" }));
      if (!this.pixels) items.push(el("span", { class: "mcm-warn", text: "Run the workflow first: the wand picks colours from the image." }));
    } else {
      const size = el("input", { type: "range", min: "1", max: "300", step: "1", value: String(this.brush), title: "Brush radius (image pixels). [ and ] change it." });
      const sizeText = el("span", { text: String(this.brush) });
      size.addEventListener("input", () => {
        this.brush = Number(size.value);
        sizeText.textContent = size.value;
      });
      items.push(el("label", {}, [el("span", { text: "Size" }), size, sizeText]));
    }
    if (this.paintTarget !== "protect" && !this.selected) items.push(el("span", { class: "mcm-warn", text: "Select a box to paint its character or parts." }));
    this.opts.replaceChildren(...items);
  }

  // ------------------------------------------------------------------ painting
  paintKey() {
    return this.paintTarget === "protect" ? "protect" : this.selected ? `${this.selected}:${this.paintTarget}` : null;
  }

  /** The bitmaps being painted ({add, sub}; Protect has only add), decoded from the settings. */
  paintLayers() {
    const key = this.paintKey();
    const d = this.doc();
    if (!key || !d.ok) return null;
    const { W, H } = this.view();
    if (this.layers?.key === key && this.layers.add.w === W && this.layers.add.h === H) return this.layers;
    if (key === "protect") {
      this.layers = { key, add: P.decode(D.protectOf(d.doc), W, H), sub: null };
    } else {
      const region = this.boxes().find(({ region }) => region.id === this.selected)?.region;
      const t = region ? D.detailFor(d.doc, region).paint[this.paintTarget] : null;
      this.layers = { key, add: P.decode(t?.add, W, H), sub: P.decode(t?.sub, W, H) };
    }
    return this.layers;
  }

  /** Paint ``mask`` (a bitmap) into the target (``add``) or out of it. */
  applyPaint(L, mask, add) {
    if (L.sub) {
      P.apply(add ? L.add : L.sub, mask, 1);
      P.apply(add ? L.sub : L.add, mask, 0);
    } else P.apply(L.add, mask, add ? 1 : 0);
  }

  /** A brush segment straight into the layers (no full-size temporaries), then only that patch of the
   * overlay is redrawn. tool: "add" paints in, "sub" paints out, "erase" wipes both back to unpainted
   * (Protect has one layer: "sub" and "erase" both clear it). */
  stampPaint(L, x0, y0, x1, y1, tool) {
    const r = this.brush;
    P.strokeLine(L.add, x0, y0, x1, y1, r, tool === "add" ? 1 : 0);
    if (L.sub) P.strokeLine(L.sub, x0, y0, x1, y1, r, tool === "sub" ? 1 : 0);
    if (this.paintCanvas && this.paintFor === L) this.updatePaintRect(P.strokeBounds(L.add, x0, y0, x1, y1, r));
    else this.paintCanvas = null;
  }

  requestDraw() {
    if (this._raf) return;
    this._raf = requestAnimationFrame(() => {
      this._raf = 0;
      this.draw();
    });
  }

  /** Store the painted layers in the settings (one undo step). */
  commitPaint() {
    const L = this.layers;
    const d = this.doc();
    if (!L || !d.ok) return;
    const enc = (b) => (b && !P.isEmpty(b) ? P.encode(b) : null);
    if (L.key === "protect") return this.commitDoc(D.setProtect(d.doc, enc(L.add)));
    const region = this.boxes().find(({ region }) => region.id === this.selected)?.region;
    if (!region) return;
    const next = structuredClone(D.detailFor(d.doc, region));
    next.paint[this.paintTarget] = { add: enc(L.add), sub: enc(L.sub) };
    if (L.add && !P.isEmpty(L.add) && next.parts[this.paintTarget]) next.parts[this.paintTarget].on = true; // painting a part ticks it
    this.commitDoc(D.setDetail(d.doc, region.id, next));
  }

  commitDoc(next) {
    this.history.record(this.widget("detail")?.value ?? "");
    this.setWidget("detail", D.serializeDetail(next));
    this.layers = null;
    this.paintCanvas = null;
    this.poll();
    this.draw(); // show the result now, whatever else is going on
  }

  undo() {
    const prev = this.history.undo(this.widget("detail")?.value ?? "");
    if (prev !== null) this.restore(prev);
  }

  redo() {
    const next = this.history.redo(this.widget("detail")?.value ?? "");
    if (next !== null) this.restore(next);
  }

  restore(text) {
    this.setWidget("detail", text);
    this.layers = null;
    this.paintCanvas = null;
    this.sig = "";
    this.poll();
  }

  /** The target's paint as a canvas: added pixels white, removed ones red stripes, Protect grey stripes. */
  buildPaintCanvas() {
    const L = this.paintLayers();
    if (!L) return null;
    const { w, h } = L.add;
    const d = this.doc();
    this.paintProtect = L.key === "protect" || !d.ok ? null : P.decode(D.protectOf(d.doc), w, h);
    const c = document.createElement("canvas");
    c.width = w;
    c.height = h;
    this.paintCtx = c.getContext("2d");
    this.paintImg = this.paintCtx.createImageData(w, h);
    this.paintFor = L;
    this.paintCanvas = c;
    this.updatePaintRect({ x0: 0, y0: 0, x1: w, y1: h });
    return c;
  }

  /** Recolour the overlay inside a rectangle: added pixels white, removed red stripes, Protect grey stripes. */
  updatePaintRect({ x0, y0, x1, y1 }) {
    const L = this.paintFor;
    const prot = this.paintProtect;
    const isProtect = L.key === "protect";
    const w = L.add.w;
    const px = this.paintImg.data;
    for (let y = y0; y < y1; y++) {
      for (let x = x0, i = y * w + x0; x < x1; x++, i++) {
        const o = i * 4;
        const stripe = (x + y) % 8 < 3;
        let r = 0, g = 0, b = 0, a = 0;
        if (L.add.bits[i]) {
          if (isProtect) [r, g, b, a] = [150, 150, 150, stripe ? 230 : 110];
          else [r, g, b, a] = [255, 255, 255, 120];
        } else if (L.sub && L.sub.bits[i]) [r, g, b, a] = [255, 59, 48, stripe ? 220 : 90];
        else if (prot && prot.bits[i] && stripe) [r, g, b, a] = [150, 150, 150, 120];
        px[o] = r;
        px[o + 1] = g;
        px[o + 2] = b;
        px[o + 3] = a;
      }
    }
    if (x1 > x0 && y1 > y0) this.paintCtx.putImageData(this.paintImg, 0, 0, x0, y0, x1 - x0, y1 - y0);
  }

  renderTools() {
    for (const t of TOOLS) this.toolButtons[t.id].classList.toggle("on", this.tool === t.id);
  }

  zoom(factor) {
    const v = this.view();
    if (factor === null) this.viewport.reset();
    else this.viewport.zoomBy(factor, v.W, v.H, v.cw, v.ch);
    this.draw();
  }

  // ------------------------------------------------------------------ data
  source() {
    const src = regionsSource(this.node);
    if (!src) return null;
    const text = src.widgets.find((w) => w.name === "regions")?.value;
    const res = S.parseState(typeof text === "string" ? text : "");
    const num = (n, d) => Math.max(1, Math.round(Number(src.widgets.find((w) => w.name === n)?.value) || d));
    return { ok: res.ok, error: res.error, state: res.ok ? res.state : null, W: num("width", 1024), H: num("height", 1024), text };
  }

  boxes() {
    const src = this.src;
    if (!src?.state) return [];
    return S.emittedRegions(src.state).map((r) => ({ region: r, color: S.regionColor(src.state, src.state.regions.indexOf(r)) }));
  }

  /** The parsed detail settings, re-parsed only when the widget's text changes (it holds the paint too,
   * so it can be large, and this is called many times per redraw). */
  doc() {
    const text = this.widget("detail")?.value;
    if (text !== this._docText || !this._doc) {
      this._docText = text;
      this._doc = D.parseDetail(text);
    }
    return this._doc;
  }

  detailOf(region) {
    const d = this.doc();
    return D.detailFor(d.ok ? d.doc : D.emptyDoc(), region);
  }

  updateDetail(region, change) {
    const d = this.doc();
    if (!d.ok) return;
    const next = structuredClone(D.detailFor(d.doc, region));
    change(next);
    this.commitDoc(D.setDetail(d.doc, region.id, next));
  }

  resultBox(id) {
    const i = this.result?.boxes?.findIndex((b) => b.id === id) ?? -1;
    return i >= 0 ? { index: i, ...this.result.boxes[i] } : null;
  }

  poll() {
    if (!this.root.isConnected) return;
    this.src = this.source();
    const a = document.activeElement;
    const typing = this.props.contains(a) && (a.tagName === "TEXTAREA" || (a.tagName === "INPUT" && /^(text|number)$/.test(a.type)));
    const detailText = this.widget("detail")?.value;
    if (detailText !== this._sigDetail) {
      this._sigDetail = detailText; // compared, never joined: with paint it can be large
      this._detailVersion = (this._detailVersion || 0) + 1;
    }
    const sig = [this.src?.text, this.src?.W, this.src?.H, this._detailVersion, this.llmConnected(), ...[...SETTINGS, ...LLM_SETTINGS, ["llm_use_image"], ["llm_system_prompt"]].map(([n]) => this.widget(n)?.value)].join("\u0001");
    if (sig === this.sig) return;
    this.sig = sig;
    this.renderList();
    if (!typing) this.renderProps(); // a re-render would drop the focus mid-typing
    this.overlay = null;
    this.draw();
  }

  async setResult(payload) {
    if (!payload) return;
    this.result = payload;
    const url = (ref) => api.apiURL(`/view?filename=${encodeURIComponent(ref.filename)}&type=${ref.type}&subfolder=${encodeURIComponent(ref.subfolder || "")}&t=${Date.now()}`);
    const load = (src) =>
      new Promise((ok, fail) => {
        const im = new Image();
        im.onload = () => ok(im);
        im.onerror = fail;
        im.src = src;
      });
    try {
      const [img, lab] = await Promise.all([load(url(payload.image)), load(url(payload.labels))]);
      const c = document.createElement("canvas");
      c.width = lab.naturalWidth;
      c.height = lab.naturalHeight;
      const g = c.getContext("2d", { willReadFrequently: true });
      g.drawImage(lab, 0, 0);
      this.image = img;
      this.labels = g.getImageData(0, 0, c.width, c.height);
      this.tags = D.partTags(this.labels.data, c.width, c.height);
      const ci = document.createElement("canvas");
      ci.width = img.naturalWidth;
      ci.height = img.naturalHeight;
      const gi = ci.getContext("2d", { willReadFrequently: true });
      gi.drawImage(img, 0, 0);
      this.pixels = gi.getImageData(0, 0, ci.width, ci.height);
    } catch {
      this.image = this.labels = this.pixels = null;
      this.tags = [];
    }
    this.layers = null;
    this.paintCanvas = null;
    this.renderOptions();
    // prompts the prompt writer wrote this run: kept as the parts' prompts (one undo step)
    if (payload.generated && Object.keys(payload.generated).length) {
      const d = this.doc();
      if (d.ok) {
        const next = D.storeGenerated(d.doc, payload.generated, this.boxes().map(({ region }) => region));
        if (next !== d.doc) this.commitDoc(next);
      }
    }
    this.sig = "";
    this.poll();
  }

  /** Whether a model is connected to the prompt_llm input. */
  llmConnected() {
    return !!this.node.inputs?.find((i) => i.name === "prompt_llm")?.link;
  }

  // ------------------------------------------------------------------ boxes list
  renderList() {
    const rows = [];
    this.errorBox.style.display = "none";
    if (!this.src) {
      rows.push(el("div", { class: "mcm-empty", text: "Connect the regions output of MOSS-Image Regional LoRA + Layout Prompt." }));
    } else if (!this.src.ok) {
      rows.push(el("div", { class: "mcm-empty", text: "The connected node's regions don't parse; fix them there." }));
    }
    const d = this.doc();
    if (!d.ok) {
      this.errorBox.style.display = "block";
      this.errorBox.textContent = d.error;
    }
    for (const { region, color } of this.boxes()) {
      const det = this.detailOf(region);
      const rb = this.resultBox(region.id);
      const tick = el("input", { type: "checkbox", title: "Detail this box" });
      tick.checked = det.enabled;
      tick.addEventListener("change", () => this.updateDetail(region, (x) => (x.enabled = tick.checked)));
      tick.addEventListener("click", (e) => e.stopPropagation());
      const dots = el("span", { class: "mcm-dots" });
      if (det.enabled && rb) {
        for (const p of D.PARTS) {
          if (!(p.name in (rb.found || {}))) continue;
          const ok = rb.found[p.name];
          dots.appendChild(el("span", { class: `mcm-dot${ok ? "" : " missing"}`, style: `background:${p.color}`, title: `${p.name}: ${ok ? "found" : "not found"}` }));
        }
      }
      const row = el("div", { class: `mcm-row${this.selected === region.id ? " sel" : ""}${det.enabled ? "" : " off"}` }, [
        tick,
        el("span", { class: "mrl-chip", style: `background:${color}` }),
        el("span", { class: "mrl-name", text: region.name || region.id, title: region.name || region.id }),
        el("span", { class: "mrl-badge", text: region.label || "object" }),
        D.missingPrompts(det).length ? el("span", { class: "mcm-need", text: "!", title: `${D.missingPrompts(det).length} part(s) need a prompt` }) : null,
        dots,
      ]);
      row.addEventListener("click", () => this.select(region.id));
      rows.push(row);
    }
    if (this.src?.ok && !rows.length) rows.push(el("div", { class: "mcm-empty", text: "The connected node has no boxes yet." }));
    this.list.replaceChildren(...rows);
    const n = this.boxes().filter(({ region }) => this.detailOf(region).enabled).length;
    const needs = this.boxes().reduce((k, { region }) => k + D.missingPrompts(this.detailOf(region)).length, 0);
    this.statusLine.textContent = (this.result ? `Last run: ${this.result.boxes.length} box(es) masked · ${n} set to detail` : `${n} box(es) set to detail · run the workflow to see the masks`) + (needs ? ` · ${needs} part(s) need a prompt` : "");
  }

  select(id) {
    this.selected = this.selected === id ? null : id;
    this.overlay = null;
    this.layers = null;
    this.paintCanvas = null;
    this.renderOptions();
    this.renderList();
    this.renderProps();
    this.draw();
  }

  // ------------------------------------------------------------------ properties
  field(label, control, title) {
    return el("label", { class: "mrl-field", title }, [el("span", { text: label }), control]);
  }

  renderProps() {
    const entry = this.boxes().find(({ region }) => region.id === this.selected);
    if (!entry) {
      this.propsTitle.textContent = "Masking";
      this.props.replaceChildren(this.settingsSection());
      return;
    }
    const { region } = entry;
    this.propsTitle.textContent = `Detail: ${region.name || region.id}`;
    const ro = !this.doc().ok;
    this.props.replaceChildren(this.boxSection(region), this.lorasSection(region), this.partsSection(region));
    if (ro) for (const x of this.props.querySelectorAll("input,button,textarea")) x.disabled = true;
  }

  settingsSection() {
    const rows = [];
    for (const [name, label, step, title] of SETTINGS) {
      const w = this.widget(name);
      if (!w) continue;
      const c = el("input", { type: "number", step: String(step), value: String(w.value), title });
      c.addEventListener("change", () => {
        let v = Number(c.value);
        if (!Number.isFinite(v)) return (c.value = String(w.value));
        const o = w.options || {};
        if (step === 1) v = Math.round(v);
        if (o.min != null) v = Math.max(o.min, v);
        if (o.max != null) v = Math.min(o.max, v);
        c.value = String(v);
        this.setWidget(name, v);
      });
      rows.push(this.field(label, c, title));
    }
    return el("div", {}, [
      el("div", { class: "mrl-sec" }, [
        el("div", { class: "mrl-sec-title", text: "SAM3" }),
        ...rows,
        el("div", { class: "mrl-hint", text: "Select a box to choose which parts are masked and how they are detailed. Tick a box in the list to detail it." }),
      ]),
      this.writerSection(),
    ]);
  }

  /** The prompt writer's settings: sampling, whether it sees the part, and its system prompt (with Reset). */
  writerSection() {
    const rows = [];
    for (const [name, label, step, title] of LLM_SETTINGS) {
      const w = this.widget(name);
      if (!w) continue;
      const c = el("input", { type: "number", step: String(step), value: String(w.value), title, class: `mcm-${name}` });
      c.addEventListener("change", () => {
        let v = Number(c.value);
        if (!Number.isFinite(v)) return (c.value = String(w.value));
        const o = w.options || {};
        if (step === 1) v = Math.round(v);
        if (o.min != null) v = Math.max(o.min, v);
        if (o.max != null) v = Math.min(o.max, v);
        c.value = String(v);
        this.setWidget(name, v);
      });
      rows.push(this.field(label, c, title));
    }
    const img = this.widget("llm_use_image");
    if (img) {
      const c = el("input", { type: "checkbox", class: "mcm-llm_use_image" });
      c.checked = img.value !== false;
      c.addEventListener("change", () => this.setWidget("llm_use_image", c.checked));
      rows.push(this.field("See the part", c, "Show the model a crop of each part (if it can see images); otherwise it writes from the caption only."));
    }
    const sysW = this.widget("llm_system_prompt");
    let sys = null;
    if (sysW) {
      const def = defaultSystemPrompt ?? "";
      const area = rememberHeight(el("textarea", { class: "mcm-system-prompt", title: "The prompt writer's instructions" }), "systemPrompt");
      area.value = (sysW.value || "").trim() ? sysW.value : def;
      const reset = el("button", { class: "mrl-mini mcm-reset-system", text: "Reset prompt", title: "Back to the built-in instructions" });
      const sync = () => (reset.disabled = area.value.trim() === def.trim());
      sync();
      area.addEventListener("input", sync);
      // stored only when it differs from the built-in one, so later improvements to that reach this node
      area.addEventListener("change", () => this.setWidget("llm_system_prompt", area.value.trim() && area.value.trim() !== def.trim() ? area.value : ""));
      reset.addEventListener("click", () => {
        this.setWidget("llm_system_prompt", "");
        area.value = def;
        sync();
      });
      sys = el("div", { class: "mcm-system" }, [el("div", { class: "mcm-panel-head" }, [el("span", { class: "grow", text: "System prompt" }), reset]), area]);
    }
    const connected = this.llmConnected();
    return el("div", { class: "mrl-sec" }, [
      el("div", { class: "mrl-sec-title", text: "Prompt writer" }),
      el("div", { class: connected ? "mrl-hint mcm-llm-on" : "mrl-hint", text: connected
        ? "A text model is connected to prompt_llm: on each run it writes the prompt of every ticked part that has none, from the box's caption and a crop of the part. You can edit what it writes."
        : "Optional: connect a text model (Load CLIP, e.g. your MOSS-Image text encoder) to prompt_llm and every ticked part without a prompt gets one written on the next run." }),
      ...rows,
      sys,
    ]);
  }

  boxSection(region) {
    const det = this.detailOf(region);
    const on = el("input", { type: "checkbox" });
    on.checked = det.enabled;
    on.addEventListener("change", () => this.updateDetail(region, (x) => (x.enabled = on.checked)));
    const rb = this.resultBox(region.id);
    return el("div", { class: "mrl-sec" }, [
      this.field("Detail", on, "Mask and detail this box"),
      el("div", { class: "mrl-hint", text: region.caption ? `Caption: ${region.caption}` : "This box has no caption." }),
      rb && rb.outline_found === false ? el("div", { class: "mrl-hint", style: "color:#f99", text: "SAM3 found no character in this box; the whole box was used as its outline." }) : null,
    ]);
  }

  lorasSection(region) {
    const det = this.detailOf(region);
    const own = S.lorasOf(region);
    const list = D.editableLoras(own, det);
    const edit = (change) =>
      this.updateDetail(region, (x) => {
        x.lora_list = D.editableLoras(own, x);
        change(x.lora_list);
      });
    const listId = `mcm-loras-${Math.random().toString(36).slice(2)}`;
    const datalist = el("datalist", { id: listId }, this.loras.map((l) => el("option", { value: l })));
    const rows = list.map((e, k) => {
      const name = el("input", { type: "text", list: listId, value: e.lora, placeholder: "LoRA file" });
      name.addEventListener("change", () => edit((l) => (l[k].lora = name.value.trim())));
      const weight = el("input", { type: "number", step: "0.05", value: String(e.weight), title: "Weight (0 turns it off)" });
      weight.addEventListener("change", () => {
        const v = Number(weight.value);
        if (Number.isFinite(v)) edit((l) => (l[k].weight = v));
      });
      const remove = el("button", { icon: "x", class: "mrl-mini", title: "Remove", onclick: () => edit((l) => l.splice(k, 1)) });
      return el("div", { class: "mcm-lora" }, [name, weight, remove]);
    });
    const reset = el("button", { class: "mrl-mini mcm-reset-loras", text: "Reset LoRAs", title: "Back to the box's own LoRAs and weights" });
    reset.disabled = !D.lorasEdited(own, det);
    reset.addEventListener("click", () =>
      this.updateDetail(region, (x) => {
        x.lora_list = null;
        x.use_box_loras = true; // older settings, folded into the list when it is first edited
        x.loras = [];
      }),
    );
    return el("div", { class: "mrl-sec" }, [
      el("div", { class: "mrl-sec-title" }, [el("span", { class: "grow", text: "LoRAs" }), reset]),
      rows.length ? el("div", { class: "mcm-loras" }, rows) : el("div", { class: "mrl-hint", text: "No LoRAs while detailing this box." }),
      datalist,
      el("button", { icon: "plus", text: "Add LoRA", title: "Add a LoRA used while detailing this box", onclick: () => edit((l) => l.push({ lora: "", weight: 1 })) }),
      el("div", { class: "mrl-hint", text: "Starts as the box's own LoRAs: change a weight, remove one or add others for detailing." }),
    ]);
  }

  partsSection(region) {
    const det = this.detailOf(region);
    const rb = this.resultBox(region.id);
    const rows = [];
    for (const p of D.PARTS) {
      const s = det.parts[p.name];
      const on = el("input", { type: "checkbox", class: "mcm-on", title: `Mask and detail the ${p.label}` });
      on.checked = s.on;
      on.addEventListener("change", () => this.updateDetail(region, (x) => (x.parts[p.name].on = on.checked)));
      const known = rb && s.on && det.enabled && p.name in (rb.found || {});
      const open = this.openPart === p.name;
      const edited = !!(s.denoise != null || s.restore != null || s.whole_crop);
      const needsPrompt = det.enabled && s.on && !D.partPrompt(det, p.name);
      const toggle = () => {
        this.openPart = open ? null : p.name;
        this.renderProps();
      };
      rows.push(
        el("div", { class: `mcm-part${open ? " open" : ""}${needsPrompt ? " needs" : ""}${needsPrompt && this.llmConnected() ? " writer" : ""}`, "data-part": p.name }, [
          on,
          el("span", { class: "mcm-chip", style: `background:${p.color}` }),
          el("span", { class: "mcm-part-name", text: p.label + (known && rb.found[p.name] ? " ✓" : ""), onclick: toggle }),
          needsPrompt
            ? this.llmConnected()
              ? el("span", { class: "mcm-need-text writer", text: "written on next run", title: "The connected prompt writer will write this part's prompt on the next run (if the part is found)" })
              : el("span", { class: "mcm-need-text", text: "needs a prompt", title: "The detailer won't run until every ticked part has a prompt: open it to write one, or connect a prompt writer" })
            : null,
          !needsPrompt && s.generated ? el("span", { class: "mcm-gen", text: "generated", title: "Written by the prompt writer; edit it freely" }) : null,
          edited ? el("span", { class: "mcm-mod", text: "edited", title: "This part has its own denoise, restore or crop settings" }) : null,
          el("span", { class: "grow", onclick: toggle }),
          el("button", { class: "mrl-mini mcm-edit", icon: open ? "chevron-down" : "chevron-right", title: `${open ? "Close" : "Edit"} the ${p.label}'s prompt and detail settings`, onclick: toggle }),
        ]),
      );
      if (known && !rb.found[p.name]) rows.push(el("div", { class: "mcm-status missing", text: "not found: try a lower threshold" }));
      if (open) rows.push(this.partPanel(region, det, p));
    }
    return el("div", { class: "mrl-sec" }, [
      el("div", { class: "mrl-sec-title", text: "Parts" }),
      el("div", { class: "mcm-parts" }, rows),
      el("div", { class: "mrl-hint", text: "Every ticked part needs its own prompt: describe just that part, e.g. \"close-up of a detailed realistic hand resting on a hip, short purple fingernails\". Left and right hand as seen in the image. Prompts and settings are used by the detailer only; changing them doesn't re-run SAM3." }),
    ]);
  }

  /** A part's settings, open under its row: the prompt its detail pass will use, denoise, restore, crop. */
  partPanel(region, det, p) {
    const s = det.parts[p.name];
    const prompt = rememberHeight(el("textarea", { class: "mcm-part-prompt", placeholder: `Required: describe just the ${p.label}, e.g. close-up of a detailed ${p.label}...`, title: `The ${p.label}'s detail prompt, sent exactly as written` }), "partPrompt");
    prompt.value = s.full_prompt || "";
    prompt.addEventListener("change", () => {
      const v = prompt.value.trim();
      this.updateDetail(region, (x) => {
        x.parts[p.name].full_prompt = v ? prompt.value : null;
        x.parts[p.name].generated = false; // yours now
      });
    });
    const regen = this.llmConnected() && s.full_prompt
      ? el("button", { class: "mrl-mini mcm-regenerate", text: "Regenerate", title: "Clear this prompt: the prompt writer writes a new one on the next run",
          onclick: () => this.updateDetail(region, (x) => {
            x.parts[p.name].full_prompt = null;
            x.parts[p.name].generated = false;
          }) })
      : null;
    const unit = (key, title) => {
      const input = el("input", { type: "number", min: "0", max: "1", step: "0.05", class: `mcm-${key}`, value: s[key] == null ? "" : String(s[key]), placeholder: "default", title });
      input.addEventListener("change", () => {
        const v = input.value.trim() === "" ? null : Math.min(1, Math.max(0, Number(input.value)));
        if (v === null || Number.isFinite(v)) this.updateDetail(region, (x) => (x.parts[p.name][key] = v));
      });
      return input;
    };
    const crop = el("input", { type: "checkbox", class: "mcm-crop" });
    crop.checked = !!s.whole_crop;
    crop.addEventListener("change", () => this.updateDetail(region, (x) => (x.parts[p.name].whole_crop = crop.checked)));
    return el("div", { class: "mcm-panel" }, [
      el("div", { class: "mcm-panel-head", text: "Character prompt (for reference)" }),
      el("div", { class: "mcm-caption-ref", text: region.caption || "This box has no caption." }),
      el("div", { class: "mcm-panel-head" }, [el("span", { class: "grow", text: "Prompt" }), regen]),
      prompt,
      this.field("Denoise", unit("denoise", ""), `How much the ${p.label} is redrawn (empty: the detailer's default)`),
      this.field("Restore", unit("restore", ""), `Detail restore for the ${p.label}: 1 keeps the original's fine texture, 0 lets the redraw replace it (to change its style). Empty: the detailer's default`),
      this.field("Whole crop", crop, `Redraw the ${p.label} together with its surroundings, then paste back only the ${p.label}. Use it to change the part's style (anime to realistic, say): otherwise the unchanged surroundings pull the redraw back to the old look. Pair it with a high denoise.`),
    ]);
  }

  view() {
    const W = this.result?.width || this.src?.W || 1024;
    const H = this.result?.height || this.src?.H || 1024;
    const v = this.viewport.view(W, H, this.wrapSize.w, this.wrapSize.h);
    return { ...v, s: v.scale };
  }

  /** A pointer event in image pixels. The graph's zoom CSS-scales this editor, so the on-screen offset is
   * normalised back to layout px first. */
  eventPoint(e) {
    const v = this.view();
    const { rect: r, cw, ch } = this.canvasRect();
    const sx = r.width ? cw / r.width : 1;
    const sy = r.height ? ch / r.height : 1;
    return { x: ((e.clientX - r.left) * sx - v.ox) / v.s, y: ((e.clientY - r.top) * sy - v.oy) / v.s, v };
  }

  onCanvasMove(e) {
    const { x, y } = this.eventPoint(e);
    if (this.canvas.style.cursor !== this.toolCursor()) this.canvas.style.cursor = this.toolCursor();
    // which part is under the pointer, named beside it (so touching hands can be told apart)
    const part = this.partUnder(x, y);
    if (part?.text !== this.hoverPart?.text || (part && (Math.abs(part.x - this.hoverPart.x) > 0.5 || Math.abs(part.y - this.hoverPart.y) > 0.5))) {
      this.hoverPart = part;
      this.requestDraw();
    }
    if (BRUSH_TOOLS.includes(this.tool)) {
      this.hover = { x, y };
      if (this.stroke) {
        this.stampPaint(this.stroke.L, this.stroke.x, this.stroke.y, x, y, this.tool);
        this.stroke.x = x;
        this.stroke.y = y;
      }
      this.requestDraw();
    }
  }

  /** {text, x, y} naming the box and part at image point (x, y) of the last run's masks, or null. */
  partUnder(x, y) {
    const lab = this.labels;
    if (!lab || !this.result || x < 0 || y < 0) return null;
    const lx = Math.floor((x * lab.width) / this.view().W), ly = Math.floor((y * lab.height) / this.view().H);
    if (lx >= lab.width || ly >= lab.height) return null;
    const hit = D.partAt(lab.data, (ly * lab.width + lx) * 4);
    if (!hit || hit.part < 0) return null;
    const id = this.result.boxes[hit.box]?.id;
    const region = this.boxes().find(({ region }) => region.id === id)?.region;
    const who = region ? region.name || region.id : `box ${hit.box + 1}`;
    return { text: `${who} · ${D.PARTS[hit.part].label}`, color: D.PARTS[hit.part].color, x, y };
  }

  /** A small label in screen space: dark text on a pill of ``color``. */
  pill(g, text, color, sx, sy, alpha = 1, anchor = "center") {
    g.save();
    g.globalAlpha = alpha;
    g.font = "bold 11px sans-serif";
    const w = Math.ceil(g.measureText(text).width) + 10, h = 16;
    const x = anchor === "center" ? sx - w / 2 : sx;
    const y = anchor === "center" ? sy - h / 2 : sy;
    g.fillStyle = color;
    g.strokeStyle = "rgba(0,0,0,.75)";
    g.lineWidth = 1.5;
    g.beginPath();
    g.roundRect ? g.roundRect(x, y, w, h, 8) : g.rect(x, y, w, h);
    g.fill();
    g.stroke();
    g.fillStyle = "#111";
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillText(text, x + w / 2, y + h / 2 + 0.5);
    g.restore();
  }

  onCanvasUp(e) {
    if (!this.stroke) return;
    try {
      this.canvas.releasePointerCapture(e.pointerId);
    } catch {}
    this.stroke = null;
    this.commitPaint();
  }

  /** Brush strokes and wand clicks; false if the click wasn't a paint action. */
  paintDown(e, x, y) {
    if (!PAINT_TOOLS.includes(this.tool) || e.button !== 0) return false;
    const L = this.paintLayers();
    if (!L) return false; // no box selected yet: the click selects one (the options strip says so)
    if (this.tool === "wand") {
      if (!this.pixels) return true;
      const { W, H } = this.view();
      const sx = this.pixels.width / W, sy = this.pixels.height / H;
      let within = null;
      const region = this.boxes().find(({ region }) => region.id === this.selected)?.region;
      if (region && this.src) {
        const r = G.centeredToCanvas(region.bbox, this.src.W, this.src.H);
        const k = W / this.src.W;
        within = { x0: r.x0 * k * sx, y0: r.y0 * k * sy, x1: r.x1 * k * sx, y1: r.y1 * k * sy };
      }
      let m = P.wand(this.pixels.data, this.pixels.width, this.pixels.height, x * sx, y * sy, this.tolerance, this.connected, within);
      if (this.pixels.width !== W || this.pixels.height !== H) m = P.decode(P.encode(m), W, H);
      this.applyPaint(L, m, !e.altKey);
      this.commitPaint();
      return true;
    }
    this.stroke = { L, x, y };
    this.stampPaint(L, x, y, x, y, this.tool);
    this.canvas.setPointerCapture?.(e.pointerId);
    this.draw();
    return true;
  }

  buildOverlay() {
    if (!this.labels || !this.result) return null;
    const colors = this.result.boxes.map((b) => this.boxes().find(({ region }) => region.id === b.id)?.color || "#cccccc");
    const sel = this.resultBox(this.selected)?.index ?? -1;
    const px = D.overlayPixels(this.labels.data, colors, sel);
    const c = document.createElement("canvas");
    c.width = this.labels.width;
    c.height = this.labels.height;
    c.getContext("2d").putImageData(new ImageData(px, c.width, c.height), 0, 0);
    return c;
  }

  draw() {
    if (!this.root.isConnected) return;
    const v = this.view();
    const dpr = window.devicePixelRatio || 1;
    // resizing a canvas reallocates it: only when the size really changes
    const pw = Math.max(1, Math.round(v.cw * dpr)), ph = Math.max(1, Math.round(v.ch * dpr));
    if (this.canvas.width !== pw) this.canvas.width = pw;
    if (this.canvas.height !== ph) this.canvas.height = ph;
    if (this.canvas.style.width !== `${v.cw}px`) this.canvas.style.width = `${v.cw}px`;
    if (this.canvas.style.height !== `${v.ch}px`) this.canvas.style.height = `${v.ch}px`;
    const g = this.canvas.getContext("2d");
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, v.cw, v.ch);
    g.fillStyle = "#1b1b1b";
    g.fillRect(v.ox, v.oy, v.W * v.s, v.H * v.s);
    if (this.image) {
      g.globalAlpha = this.selected ? 0.55 : 0.8;
      g.drawImage(this.image, v.ox, v.oy, v.W * v.s, v.H * v.s);
      g.globalAlpha = 1;
      this.overlay = this.overlay || this.buildOverlay();
      if (this.overlay) g.drawImage(this.overlay, v.ox, v.oy, v.W * v.s, v.H * v.s);
      // L / R on each hand, so a pair that touches can be told apart
      if (this.labels && this.result) {
        const kx = v.W / this.labels.width, ky = v.H / this.labels.height;
        const sel = this.resultBox(this.selected)?.index ?? -1;
        for (const t of this.tags || []) {
          this.pill(g, t.text, D.PARTS[t.part].color, v.ox + t.x * kx * v.s, v.oy + t.y * ky * v.s, sel < 0 || sel === t.box ? 1 : 0.4);
        }
      }
    } else {
      g.fillStyle = "#888";
      g.font = "12px sans-serif";
      g.textAlign = "center";
      g.fillText("Run the workflow to see the masks", v.cw / 2, v.oy + 18);
    }
    // the boxes, from the main node's canvas scaled onto the image
    const src = this.src;
    if (src?.state) {
      const k = v.W / src.W;
      for (const { region, color } of this.boxes()) {
        const det = this.detailOf(region);
        const r = G.centeredToCanvas(region.bbox, src.W, src.H);
        const sel = region.id === this.selected;
        g.strokeStyle = color;
        g.lineWidth = sel ? 2.5 : 1;
        g.setLineDash(det.enabled ? [] : [4, 3]);
        g.globalAlpha = sel || !this.selected ? 1 : 0.5;
        g.strokeRect(v.ox + r.x0 * k * v.s, v.oy + r.y0 * k * v.s, (r.x1 - r.x0) * k * v.s, (r.y1 - r.y0) * k * v.s);
        g.globalAlpha = 1;
        g.setLineDash([]);
      }
    }
    if (PAINT_TOOLS.includes(this.tool) || this.paintTarget === "protect") {
      if (!this.paintCanvas || this.paintFor !== this.paintLayers()) this.buildPaintCanvas();
      if (this.paintCanvas) {
        g.imageSmoothingEnabled = false;
        g.drawImage(this.paintCanvas, v.ox, v.oy, v.W * v.s, v.H * v.s);
        g.imageSmoothingEnabled = true;
      }
    }
    if (this.hoverPart && !this.stroke) {
      const hx = v.ox + this.hoverPart.x * v.s + 14, hy = v.oy + this.hoverPart.y * v.s + 10;
      this.pill(g, this.hoverPart.text, this.hoverPart.color, Math.min(hx, v.cw - 140), Math.min(hy, v.ch - 18), 0.95, "corner");
    }
    if (this.hover && BRUSH_TOOLS.includes(this.tool)) {
      g.beginPath();
      g.arc(v.ox + this.hover.x * v.s, v.oy + this.hover.y * v.s, Math.max(1, this.brush * v.s), 0, Math.PI * 2);
      g.lineWidth = 1.5;
      g.strokeStyle = { add: ADD_RGB, sub: "#ff3b30", erase: "#bbbbbb" }[this.tool];
      if (this.tool === "erase") g.setLineDash([4, 3]);
      g.stroke();
      g.setLineDash([]);
    }
  }

  onCanvasDown(e) {
    this.keys.focus();
    const { x, y, v } = this.eventPoint(e);
    if (this.paintDown(e, x, y)) return e.preventDefault();
    let hit = null;
    if (this.labels && x >= 0 && y >= 0 && x < this.labels.width && y < this.labels.height) {
      const i = (Math.floor(y) * this.labels.width + Math.floor(x)) * 4;
      const n = this.labels.data[i] - 1;
      if (n >= 0) hit = this.result.boxes[n]?.id ?? null;
    }
    if (!hit && this.src?.state) {
      const k = v.W / this.src.W;
      for (const { region } of [...this.boxes()].reverse()) {
        const r = G.centeredToCanvas(region.bbox, this.src.W, this.src.H);
        if (x >= r.x0 * k && x <= r.x1 * k && y >= r.y0 * k && y <= r.y1 * k) {
          hit = region.id;
          break;
        }
      }
    }
    if (hit) this.select(hit);
    else if (this.selected) this.select(this.selected); // background: deselect
  }
}

app.registerExtension({
  name: "moss.CharacterMasksEditor",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      for (const w of this.widgets || []) hideWidget(w);
      const editor = new MasksEditor(this);
      this.__mcmEditor = editor;
      const widget = this.addDOMWidget(EDITOR_WIDGET, "moss_masks_editor", editor.root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 460,
        getValue: () => undefined,
        setValue: () => {},
      });
      widget.serialize = false;
      this.setSize([Math.max(this.size[0], 1080), Math.max(this.size[1], 640)]);
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      onExecuted?.apply(this, arguments);
      this.__mcmEditor?.setResult(output?.[UI_KEY]?.[0]);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      setTimeout(() => this.__mcmEditor?.poll(), 0);
      return r;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      this.__mcmEditor?.destroy();
      return onRemoved?.apply(this, arguments);
    };
  },
});
