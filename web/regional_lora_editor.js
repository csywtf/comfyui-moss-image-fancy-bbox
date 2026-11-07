// In-node region editor for "MOSS-Image Regional LoRA + Layout Prompt" (spec section 8).
//
//   [layers] [ [tools]      canvas        ] [properties]     (the toolbar floats over the canvas)
//
// The node's `regions` STRING widget holds the JSON and is the only source of truth (8.1c, 8.2a).
// This editor is a view over it: it parses the text, draws it, and writes the serialised result back
// after each edit. It never keeps state the JSON does not have (selection and the active tool are
// view state), and it never writes while the JSON fails to parse (8.9a).

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import * as G from "./geometry.js";
import * as S from "./state.js";
import * as C from "./curve.js";
import { CurvePanel, drawSparkline } from "./curve_panel.js";
import { icon } from "./icons.js";
import { dropRemovedGridValues } from "./migrate.js";

const NODE_NAME = "MossRegionalLora";
const POLL_MS = 250;
const HANDLE_PX = 7; // half-size of a resize handle, in screen px
const MIN_CREATE_PX = 3; // drags shorter than this (image px) are clicks, not creates

// Tools in the floating toolbar. Add new ones here: {id, icon (a name in icons.js), key, title}.
const TOOLS = [
  { id: "select", icon: "mouse-pointer-2", key: "v", title: "Select (V): click to select, drag a box to move it, drag a handle to resize. Click the background to deselect." },
  { id: "box", icon: "square", key: "b", title: "Box (B): drag to draw a new box. Shift: square, Alt: from the centre." },
];

const CSS = `
.mrl-root{display:flex;flex-direction:column;gap:4px;width:100%;height:100%;box-sizing:border-box;font:12px sans-serif;color:var(--input-text,#ddd);outline:none}
.mrl-root{color-scheme:dark;accent-color:#3a6ea5}
.mrl-root input[type=text],.mrl-root input[type=number],.mrl-root select,.mrl-root textarea{background:var(--comfy-input-bg,#222);color:var(--input-text,#ddd);border:1px solid var(--border-color,#555);border-radius:3px;padding:2px 5px;font:12px sans-serif;box-sizing:border-box;outline:none}
.mrl-root input[type=text]:focus,.mrl-root input[type=number]:focus,.mrl-root select:focus,.mrl-root textarea:focus{border-color:#6af}
.mrl-root input::placeholder,.mrl-root textarea::placeholder{color:var(--descrip-text,#888)}
.mrl-root input[type=color]{background:transparent;border:1px solid var(--border-color,#555);border-radius:3px;padding:1px;cursor:pointer}
.mrl-top{display:flex;flex-wrap:wrap;align-items:center;gap:6px}
.mrl-top button,.mrl-root button{background:var(--comfy-input-bg,#222);color:inherit;border:1px solid var(--border-color,#555);border-radius:3px;padding:0 5px;cursor:pointer;line-height:18px}
.mrl-root button:disabled{opacity:.4;cursor:default}
.mrl-root button{display:inline-flex;align-items:center;justify-content:center;gap:3px}
.mrl-icon{display:block;flex:0 0 auto}
.mrl-mini{min-width:18px;height:20px;padding:0 3px!important}
.mrl-layers .mrl-mini:not(.mrl-has-text){background:transparent!important;border-color:transparent!important;padding:0 1px!important;min-width:18px}
.mrl-layers .mrl-mini:not(.mrl-has-text):hover{background:rgba(128,128,128,.25)!important}
.mrl-row,.mrl-group{gap:2px!important}
.mrl-toggle-off{opacity:.45}
.mrl-dims{font-weight:bold}
.mrl-readout{font-family:monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;flex:1;min-width:0}
.mrl-error{display:none;background:#5a1d1d;border:1px solid #c44;padding:4px 6px;border-radius:3px;white-space:pre-wrap}
.mrl-warn{display:none;background:#4a3d12;border:1px solid #a83;padding:2px 6px;border-radius:3px}
.mrl-main{display:flex;flex:1 1 auto;min-height:0;gap:4px}
.mrl-tools{position:absolute;left:8px;top:8px;z-index:2;display:flex;flex-direction:column;gap:3px;padding:4px;background:rgba(30,30,30,.92);border:1px solid var(--border-color,#555);border-radius:6px;box-shadow:0 2px 10px rgba(0,0,0,.45)}
.mrl-tools button{width:30px;height:30px;padding:0}
.mrl-tools button.on{background:#3a6ea5;border-color:#6af}
.mrl-panel{display:flex;flex-direction:column;min-height:0;border:1px solid var(--border-color,#444);border-radius:3px;background:rgba(0,0,0,.15)}
.mrl-panel-head{display:flex;align-items:center;gap:4px;padding:3px 5px;font-weight:bold;border-bottom:1px solid rgba(128,128,128,.3)}
.mrl-panel-head .grow{flex:1}
.mrl-panel-body{overflow:auto;flex:1 1 auto;min-height:0}
.mrl-layers{flex:0 0 260px;width:260px;min-width:0}
.mrl-props{flex:0 0 300px;width:300px;min-width:0}
.mrl-root button{white-space:nowrap}
.mrl-canvas-wrap{position:relative;flex:1 1 auto;min-width:120px;min-height:160px;overflow:hidden}
.mrl-canvas-wrap canvas{position:absolute;left:0;top:0;outline:none}
.mrl-row,.mrl-group{display:flex;align-items:center;gap:4px;padding:2px 4px;border-bottom:1px solid rgba(128,128,128,.2);cursor:default;user-select:none}
.mrl-row:hover{background:rgba(128,128,128,.12)}
.mrl-row.sel{background:rgba(80,140,255,.28)}
.mrl-row.in-group{padding-left:16px}
.mrl-row.hidden-row .mrl-name{opacity:.5}
.mrl-row.drop-before,.mrl-group.drop-before{box-shadow:inset 0 2px 0 #6af}
.mrl-group{background:rgba(128,128,128,.15);font-weight:bold}
.mrl-group input[type=text]{flex:1;min-width:40px;width:60px;background:transparent;color:inherit;border:1px solid transparent;font-weight:bold}
.mrl-group input[type=text]:focus{border-color:#888}
.mrl-chip{width:10px;height:10px;border-radius:2px;flex:0 0 10px}
.mrl-name{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.mrl-badge{font-size:10px;padding:1px 3px;border-radius:3px;background:rgba(128,128,128,.3);display:inline-flex;align-items:center;gap:2px}
.mrl-oob{color:#fb4;display:inline-flex;align-items:center;gap:3px}
.mrl-grip{cursor:grab;opacity:.6;display:inline-flex}
.mrl-mini{padding:0 3px!important;line-height:16px!important;font-size:10px}
.mrl-sec{padding:5px 6px;border-bottom:1px solid rgba(128,128,128,.25);display:flex;flex-direction:column;gap:4px}
.mrl-sec-title{font-weight:bold;opacity:.85;display:flex;align-items:center;gap:6px}
.mrl-sec-title .grow{flex:1}
.mrl-field{display:flex;align-items:center;gap:4px}
.mrl-field>span:first-child{flex:0 0 62px;opacity:.75}
.mrl-field input[type=text],.mrl-field select,.mrl-props textarea{flex:1;min-width:0;background:var(--comfy-input-bg,#222);color:inherit;border:1px solid var(--border-color,#555);border-radius:3px}
.mrl-props textarea{min-height:44px;resize:vertical;font:12px sans-serif}
.mrl-bbox-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:3px}
.mrl-bbox-grid label{display:flex;flex-direction:column;font-size:10px;opacity:.85}
.mrl-bbox-grid input{width:100%;box-sizing:border-box}
.mrl-hint{opacity:.65;font-size:11px}
.mrl-lora{border:1px solid rgba(128,128,128,.3);border-radius:3px;padding:3px;display:flex;flex-direction:column;gap:3px}
.mrl-lora.open{border-color:#6af}
.mrl-lora-row{display:flex;align-items:center;gap:4px}
.mrl-lora-name{flex:1;min-width:0;justify-content:space-between!important}
.mrl-lora-label{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0}
.mrl-lora-row input[type=number]{width:54px}
.mrl-spark{cursor:pointer;flex:0 0 auto;border-radius:2px}
.mrl-spark.open{outline:1px solid #6af}
.mrl-picker{position:fixed;z-index:10000;background:var(--comfy-menu-bg,#1e1e1e);border:1px solid #6af;border-radius:4px;box-shadow:0 4px 16px rgba(0,0,0,.5);display:flex;flex-direction:column;max-height:300px;width:320px;font:12px sans-serif;color:var(--input-text,#ddd)}
.mrl-picker input{margin:4px;background:var(--comfy-input-bg,#222);color:inherit;border:1px solid #555;border-radius:3px;padding:2px 4px}
.mrl-picker-list{overflow:auto}
.mrl-picker-item{padding:2px 8px;cursor:pointer;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mrl-picker-item:hover,.mrl-picker-item.cur{background:rgba(80,140,255,.3)}
.mrl-curve{display:flex;flex-direction:column;gap:2px}
.mrl-curve-bar{display:flex;flex-wrap:wrap;align-items:center;gap:3px}
.mrl-curve-bar button{font-size:11px;padding:0 4px}
.mrl-curve-bar button.on{background:#a63;border-color:#c84}
.mrl-curve-title{display:none}
.mrl-curve-readout{flex-basis:100%;font-family:monospace;opacity:.85;font-size:11px;min-height:14px}
.mrl-sep{width:4px}
.mrl-curve-wrap{position:relative;height:120px}
.mrl-curve-wrap canvas{position:absolute;left:0;top:0;outline:none}
.mrl-status{opacity:.85}
.mrl-props input[type=color]{width:28px;height:20px}
`;

function injectCss() {
  let s = document.getElementById("mrl-css");
  if (!s) {
    s = document.createElement("style");
    s.id = "mrl-css";
    document.head.appendChild(s);
  }
  s.textContent = CSS;
}

// attrs.icon: a Lucide icon name, placed before attrs.text (an icon-only control needs a title)
function el(tag, attrs = {}, children = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "text" || k === "icon") continue;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) e.setAttribute(k, v);
  }
  if (attrs.icon) e.appendChild(icon(attrs.icon, attrs.iconSize || 14));
  if (attrs.text !== undefined && attrs.text !== null) e.appendChild(document.createTextNode(attrs.text));
  if (attrs.icon && attrs.text) e.classList.add("mrl-has-text");
  for (const c of [].concat(children)) if (c) e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  return e;
}

function hexToRgba(hex, a) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex || "");
  if (!m) return `rgba(200,200,200,${a})`;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

/** File name without folders or extension, for display ("MOSS Image\\concept\\x.safetensors" -> "x"). */
export function loraDisplayName(path) {
  if (!path) return "(choose a LoRA)";
  const base = String(path).split(/[\\/]/).pop();
  return base.replace(/\.(safetensors|ckpt|pt|bin|pth)$/i, "");
}

const loraListPromise = { p: null };
function loraList() {
  if (!loraListPromise.p) {
    loraListPromise.p = api
      .fetchApi("/models/loras")
      .then((r) => (r.ok ? r.json() : []))
      .catch(() => []);
  }
  return loraListPromise.p;
}

function inputFileList() {
  return api
    .fetchApi("/internal/files/input")
    .then((r) => (r.ok ? r.json() : []))
    .then((list) => list.map((s) => String(s).replace(/ \[input\]$/, "")))
    .catch(() => []);
}

class RegionEditor {
  constructor(node) {
    this.node = node;
    this.wRegions = node.widgets.find((w) => w.name === "regions");
    this.wWidth = node.widgets.find((w) => w.name === "width");
    this.wHeight = node.widgets.find((w) => w.name === "height");
    this.wRampStart = node.widgets.find((w) => w.name === "ramp_start");
    this.wRampEnd = node.widgets.find((w) => w.name === "ramp_end");
    this.history = new S.History(100);
    this.state = S.emptyState();
    this.parseError = null;
    this.warnings = [];
    this.lastText = null;
    this.lastDims = "";
    this.selected = new Set();
    this.primary = null;
    this.tool = "select";
    this.drag = null;
    this.preview = null; // {id, bbox} during a drag, or {create: rect}
    this.expanded = null; // {id, index}: LoRA entry whose curve is open in the properties panel
    this.refImg = null;
    this.refName = null;
    this.refState = "none"; // none | loading | ok | missing
    this.loras = [];
    this.status = "";
    this.buildDom();
    loraList().then((l) => {
      this.loras = l;
    });
    this.syncFromWidget(true);
    this.timer = setInterval(() => this.poll(), POLL_MS);
    this.ro = new ResizeObserver(() => this.draw());
    this.ro.observe(this.wrap);
  }

  destroy() {
    clearInterval(this.timer);
    this.ro?.disconnect();
    this.curvePanel?.destroy();
    this.closePicker();
  }

  // ------------------------------------------------------------------ DOM
  buildDom() {
    injectCss();
    this.dims = el("span", { class: "mrl-dims" });
    this.undoBtn = el("button", { icon: "undo-2", text: "Undo", title: "Undo (Ctrl/Cmd+Z)", onclick: () => this.undo() });
    this.redoBtn = el("button", { icon: "redo-2", text: "Redo", title: "Redo (Ctrl/Cmd+Shift+Z)", onclick: () => this.redo() });
    this.readout = el("div", { class: "mrl-readout" });
    const top = el("div", { class: "mrl-top" }, [this.dims, this.undoBtn, this.redoBtn, this.readout]);
    this.errorBox = el("div", { class: "mrl-error" });
    this.warnBox = el("div", { class: "mrl-warn" });

    this.toolButtons = {};
    const tools = el(
      "div",
      { class: "mrl-tools" },
      TOOLS.map((t) => (this.toolButtons[t.id] = el("button", { icon: t.icon, iconSize: 16, title: t.title, onclick: () => this.setTool(t.id) }))),
    );

    this.list = el("div", { class: "mrl-panel-body" });
    const layers = el("div", { class: "mrl-panel mrl-layers" }, [
      el("div", { class: "mrl-panel-head" }, [
        el("span", { class: "grow", text: "Layers" }),
        el("button", { icon: "folder-plus", text: "Group", class: "mrl-mini", title: "New group (puts the selected boxes in it)", onclick: () => this.addGroup() }),
      ]),
      this.list,
    ]);

    this.canvas = el("canvas", { tabindex: "0" });
    // the toolbar floats over the left edge of the canvas
    this.wrap = el("div", { class: "mrl-canvas-wrap" }, [this.canvas, tools]);

    this.propsTitle = el("span", { class: "grow" });
    this.props = el("div", { class: "mrl-panel-body" });
    const props = el("div", { class: "mrl-panel mrl-props" }, [el("div", { class: "mrl-panel-head" }, [this.propsTitle]), this.props]);

    this.curvePanel = new CurvePanel({
      target: () => this.expandedEntry(),
      legacy: () => this.legacyCurve(),
      readOnly: () => !!this.parseError,
      snap: () => !!this.state.editor?.snap,
      color: () => (this.primary ? S.regionColor(this.state, this.state.regions.findIndex((r) => r.id === this.primary)) : null),
      commitCurve: (points) => this.updateExpandedEntry({ curve: points }),
    });

    this.statusLine = el("div", { class: "mrl-status" });
    this.root = el("div", { class: "mrl-root", tabindex: "-1" }, [
      top,
      this.errorBox,
      this.warnBox,
      el("div", { class: "mrl-main" }, [layers, this.wrap, props]),
      this.statusLine,
    ]);

    const c = this.canvas;
    c.addEventListener("pointerdown", (e) => this.onDown(e));
    c.addEventListener("pointermove", (e) => this.onMove(e));
    c.addEventListener("pointerup", (e) => this.onUp(e));
    c.addEventListener("pointercancel", () => this.cancelDrag());
    c.addEventListener("dblclick", (e) => e.stopPropagation());
    c.addEventListener("contextmenu", (e) => e.stopPropagation());
    this.root.addEventListener("keydown", (e) => this.onKey(e));
    // keep ComfyUI's canvas from treating clicks inside the editor as node drags
    this.root.addEventListener("pointerdown", (e) => e.stopPropagation());
  }

  // ------------------------------------------------------------------ widget sync
  get text() {
    const v = this.wRegions?.value;
    return typeof v === "string" ? v : "";
  }

  canvasSize() {
    const w = Math.max(1, Math.round(Number(this.wWidth?.value) || 1024));
    const h = Math.max(1, Math.round(Number(this.wHeight?.value) || 1024));
    return { W: w, H: h };
  }

  poll() {
    if (!this.root.isConnected) return;
    const t = this.text;
    if (t !== this.lastText) {
      // hand edit or workflow load: make it undoable from the canvas, then re-render (8.2b)
      if (this.lastText !== null && !this.parseError) this.history.record(this.lastText);
      this.syncFromWidget(false);
    }
    const { W, H } = this.canvasSize();
    const d = `${W}x${H}:${this.wRampStart?.value}:${this.wRampEnd?.value}`;
    if (d !== this.lastDims) {
      this.lastDims = d;
      this.renderAll();
    }
  }

  syncFromWidget() {
    const t = this.text;
    this.lastText = t;
    const res = S.parseState(t);
    if (!res.ok) {
      this.parseError = res.error; // keep the last good state on screen, read-only (8.9a)
    } else {
      this.parseError = null;
      this.state = res.state;
      this.warnings = res.warnings;
      const ids = new Set(this.state.regions.map((r) => r.id));
      for (const id of [...this.selected]) if (!ids.has(id)) this.selected.delete(id);
      if (this.primary && !ids.has(this.primary)) this.primary = null;
    }
    this.renderAll();
  }

  /** Apply a new state: record undo, serialise, write the widget. Refused while the JSON is invalid. */
  commit(newState) {
    if (this.parseError) return false;
    const before = this.text;
    newState.version = Math.max(Number.isInteger(newState.version) ? newState.version : 0, S.SCHEMA_VERSION);
    const text = S.serializeState(newState);
    if (text === before) return false;
    this.history.record(before);
    this.writeText(text);
    this.state = newState;
    this.renderAll();
    return true;
  }

  writeText(text) {
    this.lastText = text;
    if (this.wRegions) this.wRegions.value = text; // the widget's setter updates its textarea and store
    this.node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
  }

  undo() {
    if (this.parseError) return;
    const t = this.history.undo(this.text);
    if (t == null) return;
    this.writeText(t);
    this.syncFromWidget();
  }

  redo() {
    if (this.parseError) return;
    const t = this.history.redo(this.text);
    if (t == null) return;
    this.writeText(t);
    this.syncFromWidget();
  }

  setEditor(patch) {
    const s = S.cloneState(this.state);
    s.editor = { ...s.editor, ...patch };
    this.commit(s);
  }

  // ------------------------------------------------------------------ helpers
  region(id) {
    return this.state.regions.find((r) => r.id === id) || null;
  }

  rectOf(r) {
    const { W, H } = this.canvasSize();
    const bbox = this.preview && this.preview.id === r.id ? this.preview.bbox : r.bbox;
    return G.centeredToCanvas(bbox, W, H);
  }

  view() {
    const { W, H } = this.canvasSize();
    const cw = this.wrap.clientWidth;
    const ch = this.wrap.clientHeight;
    const fit = G.fitCanvas(W, H, Math.max(cw - 2, 1), Math.max(ch - 2, 1));
    return { W, H, cw, ch, scale: fit.scale, ox: (cw - fit.w) / 2, oy: (ch - fit.h) / 2 };
  }

  eventPoint(e) {
    const v = this.view();
    const r = this.canvas.getBoundingClientRect();
    // the canvas element may be CSS-scaled by the graph zoom; normalise to its layout size
    const sx = r.width ? this.canvas.clientWidth / r.width : 1;
    const sy = r.height ? this.canvas.clientHeight / r.height : 1;
    const px = (e.clientX - r.left) * sx - v.ox;
    const py = (e.clientY - r.top) * sy - v.oy;
    return { p: G.screenToImage({ x: px, y: py }, v.scale || 1), v };
  }

  /** The node-level ramp (for LoRA entries without their own curve), from the ramp widgets. */
  legacyCurve() {
    const rs = Number(this.wRampStart?.value ?? 0.15);
    const re = Number(this.wRampEnd?.value ?? 1);
    return C.legacyRamp(Number.isFinite(rs) ? rs : 0.15, Number.isFinite(re) ? re : 1);
  }

  entryCurve(entry) {
    try {
      if (Array.isArray(entry.curve) && entry.curve.length) return { points: C.normalize(entry.curve), own: true };
    } catch {}
    return { points: C.normalize(this.legacyCurve()), own: false };
  }

  select(id, additive) {
    if (id == null) {
      if (!additive) {
        this.selected.clear();
        this.primary = null;
      }
    } else if (additive) {
      if (this.selected.has(id)) {
        this.selected.delete(id);
        if (this.primary === id) this.primary = [...this.selected].pop() ?? null;
      } else {
        this.selected.add(id);
        this.primary = id;
      }
    } else {
      this.selected = new Set([id]);
      this.primary = id;
    }
    if (this.expanded && this.expanded.id !== this.primary) this.expanded = null;
    this.renderAll();
  }

  setTool(id) {
    this.tool = id;
    this.renderTools();
    this.canvas.style.cursor = id === "box" ? "crosshair" : "default";
  }

  // ------------------------------------------------------------------ pointer interaction (8.4a)
  onDown(e) {
    if (e.button !== 0) return;
    e.stopPropagation();
    this.canvas.focus();
    this.closePicker();
    if (this.parseError) return;
    const { p, v } = this.eventPoint(e);
    if (this.tool === "box") {
      this.drag = { kind: "create", start: G.clampPoint(p, v.W, v.H) };
      this.canvas.setPointerCapture(e.pointerId);
      return;
    }
    const additive = e.shiftKey || e.ctrlKey || e.metaKey;
    const tol = HANDLE_PX / (v.scale || 1);
    const prim = this.region(this.primary);
    if (prim && prim.visible !== false && !prim.locked && !additive) {
      const h = G.handleAt(this.rectOf(prim), p, tol);
      if (h) {
        this.drag = { kind: "resize", id: prim.id, handle: h, rect0: this.rectOf(prim) };
        this.canvas.setPointerCapture(e.pointerId);
        return;
      }
    }
    const rects = this.state.regions.map((r) => (r.visible === false ? null : this.rectOf(r)));
    const hit = G.hitTest(rects, p);
    if (hit >= 0) {
      const r = this.state.regions[hit];
      if (additive) {
        this.select(r.id, true);
        return;
      }
      if (!this.selected.has(r.id) || this.primary !== r.id) this.select(r.id, false);
      if (!r.locked) {
        this.drag = { kind: "move", id: r.id, start: p, rect0: this.rectOf(r) };
        this.canvas.setPointerCapture(e.pointerId);
      }
      return;
    }
    // background: deselect
    if (!additive) this.select(null, false);
  }

  snapOn(e) {
    return !!this.state.editor?.snap && !(e.ctrlKey || e.metaKey);
  }

  onMove(e) {
    const { p, v } = this.eventPoint(e);
    if (!this.drag) {
      this.updateCursor(p, v);
      return;
    }
    e.stopPropagation();
    const grid = G.gridStep(v.W, v.H, this.state.editor?.grid_divisions || 3);
    const snap = this.snapOn(e);
    const q = snap ? G.snapPoint(p, grid) : p;
    const mods = { square: e.shiftKey, fromCenter: e.altKey };
    const d = this.drag;
    let rect;
    if (d.kind === "create") {
      const start = snap ? G.snapPoint(d.start, grid) : d.start;
      rect = G.createRect(start, q, mods);
      this.preview = { create: rect };
    } else if (d.kind === "move") {
      rect = snap ? G.snapMove(d.rect0, p.x - d.start.x, p.y - d.start.y, grid) : G.moveRect(d.rect0, p.x - d.start.x, p.y - d.start.y);
      this.preview = { id: d.id, bbox: G.canvasToCentered(rect, v.W, v.H) };
    } else {
      rect = G.resizeRect(d.rect0, d.handle, q, mods);
      this.preview = { id: d.id, bbox: G.canvasToCentered(rect, v.W, v.H) };
    }
    this.draw();
    this.renderReadout();
  }

  onUp(e) {
    if (!this.drag) return;
    e.stopPropagation();
    const d = this.drag;
    const pv = this.preview;
    this.drag = null;
    this.preview = null;
    try {
      this.canvas.releasePointerCapture(e.pointerId);
    } catch {}
    const { W, H } = this.canvasSize();
    if (d.kind === "create" && pv?.create) {
      const { w, h } = G.rectSize(pv.create);
      if (w >= MIN_CREATE_PX && h >= MIN_CREATE_PX) {
        const { state, id } = S.addRegion(this.state, G.canvasToCentered(pv.create, W, H));
        this.selected = new Set([id]);
        this.primary = id;
        this.expanded = null;
        this.setTool("select"); // drawn: back to Select, with the new box selected
        this.commit(state);
        return;
      }
    } else if (pv?.id) {
      this.commit(S.updateRegion(this.state, pv.id, { bbox: pv.bbox }));
      return;
    }
    this.renderAll();
  }

  cancelDrag() {
    this.drag = null;
    this.preview = null;
    this.renderAll();
  }

  updateCursor(p, v) {
    if (this.tool === "box") {
      this.canvas.style.cursor = "crosshair";
      return;
    }
    let cursor = "default";
    const prim = this.region(this.primary);
    if (prim && prim.visible !== false && !prim.locked) {
      const h = G.handleAt(this.rectOf(prim), p, HANDLE_PX / (v.scale || 1));
      if (h) cursor = G.cursorFor(h);
    }
    if (cursor === "default") {
      const rects = this.state.regions.map((r) => (r.visible === false ? null : this.rectOf(r)));
      const hit = G.hitTest(rects, p);
      if (hit >= 0) cursor = this.state.regions[hit].locked ? "not-allowed" : "move";
    }
    this.canvas.style.cursor = cursor;
  }

  // ------------------------------------------------------------------ keyboard (8.4b, 8.4c)
  onKey(e) {
    const tag = (e.target?.tagName || "").toLowerCase();
    const inField = tag === "input" || tag === "select" || tag === "textarea";
    const mod = e.ctrlKey || e.metaKey;
    if (mod && (e.key === "z" || e.key === "Z")) {
      if (inField) return;
      e.preventDefault();
      e.stopPropagation();
      if (e.shiftKey) this.redo();
      else this.undo();
      return;
    }
    if (mod && (e.key === "y" || e.key === "Y") && !inField) {
      e.preventDefault();
      e.stopPropagation();
      this.redo();
      return;
    }
    if (inField) {
      e.stopPropagation(); // typing in a field must not trigger graph shortcuts
      return;
    }
    if (e.key === "Escape") {
      e.stopPropagation();
      if (this.drag) this.cancelDrag();
      else if (this.tool !== "select") this.setTool("select");
      else this.select(null, false);
      return;
    }
    if (!mod && !e.altKey) {
      const tool = TOOLS.find((t) => t.key === e.key.toLowerCase());
      if (tool) {
        e.preventDefault();
        e.stopPropagation();
        this.setTool(tool.id);
        return;
      }
    }
    if (e.key.startsWith("Arrow")) {
      e.preventDefault();
      e.stopPropagation();
      const ids = [...this.selected].filter((id) => !this.region(id)?.locked);
      if (!ids.length) return;
      let s = this.state;
      for (const id of ids) s = S.updateRegion(s, id, { bbox: G.nudgeBbox(this.region(id).bbox, e.key, e.shiftKey) });
      this.commit(s);
      return;
    }
    if (e.key === "Delete" || e.key === "Backspace") {
      e.preventDefault();
      e.stopPropagation();
      const ids = [...this.selected].filter((id) => !this.region(id)?.locked);
      if (!ids.length) return;
      this.selected.clear();
      this.primary = null;
      this.expanded = null;
      this.commit(S.deleteRegions(this.state, ids));
    }
  }

  // ------------------------------------------------------------------ rendering
  renderAll() {
    this.renderTop();
    this.renderTools();
    this.renderReadout();
    this.renderList();
    this.renderProps();
    this.loadReference();
    this.draw();
  }

  renderTop() {
    const { W, H } = this.canvasSize();
    this.dims.textContent = `${W} × ${H} px`;
    this.undoBtn.disabled = !this.history.canUndo || !!this.parseError;
    this.redoBtn.disabled = !this.history.canRedo || !!this.parseError;
    if (this.parseError) {
      this.errorBox.style.display = "block";
      this.errorBox.textContent =
        `Regions JSON cannot be parsed:\n${this.parseError}\n` +
        "The editor is read-only until the text in the \"regions\" field parses again. Nothing has been overwritten.";
    } else {
      this.errorBox.style.display = "none";
    }
    if (this.warnings.length && !this.parseError) {
      this.warnBox.style.display = "block";
      this.warnBox.textContent = this.warnings.join(" · ");
    } else this.warnBox.style.display = "none";
    this.statusLine.textContent = this.status;
  }

  renderTools() {
    for (const t of TOOLS) this.toolButtons[t.id].classList.toggle("on", this.tool === t.id);
  }

  renderReadout() {
    const r = this.region(this.primary);
    if (!r) {
      const n = S.emittedRegions(this.state).length;
      this.readout.textContent = `${n} box${n === 1 ? "" : "es"} · ${this.tool === "box" ? "drag on the canvas to draw a box" : "Box tool (B) to draw, Select (V) to edit"}`;
      return;
    }
    const bbox = this.preview && this.preview.id === r.id ? this.preview.bbox : r.bbox;
    this.readout.textContent = `${r.name || r.id}: bbox [y_min, x_min, y_max, x_max] = [${bbox.join(", ")}]`;
  }

  loadReference() {
    const name = this.state.editor?.reference_image || "";
    if (name === this.refName) return;
    this.refName = name;
    this.refImg = null;
    if (!name) {
      this.refState = "none";
      return;
    }
    this.refState = "loading";
    const img = new Image();
    img.onload = () => {
      if (this.refName !== name) return;
      this.refImg = img;
      this.refState = "ok";
      this.draw();
    };
    img.onerror = () => {
      if (this.refName !== name) return;
      this.refState = "missing";
      this.draw();
    };
    const sub = name.includes("/") ? name.slice(0, name.lastIndexOf("/")) : "";
    const file = name.includes("/") ? name.slice(name.lastIndexOf("/") + 1) : name;
    img.src = api.apiURL(`/view?filename=${encodeURIComponent(file)}&type=input&subfolder=${encodeURIComponent(sub)}`);
  }

  draw() {
    const v = this.view();
    const dpr = window.devicePixelRatio || 1;
    const c = this.canvas;
    c.style.width = `${v.cw}px`;
    c.style.height = `${v.ch}px`;
    const pw = Math.max(1, Math.round(v.cw * dpr));
    const ph = Math.max(1, Math.round(v.ch * dpr));
    if (c.width !== pw) c.width = pw;
    if (c.height !== ph) c.height = ph;
    const ctx = c.getContext("2d");
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, pw, ph);
    if (!v.scale) return;
    const k = v.scale; // image px -> css px
    const px = 1 / k; // one css px in image px
    ctx.setTransform(dpr * k, 0, 0, dpr * k, dpr * v.ox, dpr * v.oy);

    ctx.fillStyle = "#2a2a2a";
    ctx.fillRect(0, 0, v.W, v.H);
    const ed = this.state.editor || {};
    if (this.refState === "ok" && this.refImg) {
      ctx.globalAlpha = this.opacityPreview ?? ed.reference_opacity ?? 0.5;
      ctx.drawImage(this.refImg, 0, 0, v.W, v.H);
      ctx.globalAlpha = 1;
    } else if (this.refState === "missing" || this.refState === "loading") {
      ctx.fillStyle = "rgba(255,255,255,0.35)";
      ctx.font = `${12 * px}px sans-serif`;
      ctx.fillText(this.refState === "missing" ? `reference image unavailable: ${this.refName}` : "loading reference…", 8 * px, v.H - 8 * px);
    }

    // grid (8.3c)
    const div = Math.max(1, ed.grid_divisions || 3);
    ctx.strokeStyle = "rgba(255,255,255,0.18)";
    ctx.lineWidth = px;
    ctx.beginPath();
    for (let i = 1; i < div; i++) {
      const x = (v.W * i) / div;
      const y = (v.H * i) / div;
      ctx.moveTo(x, 0);
      ctx.lineTo(x, v.H);
      ctx.moveTo(0, y);
      ctx.lineTo(v.W, y);
    }
    ctx.stroke();
    ctx.strokeStyle = "rgba(255,255,255,0.35)";
    ctx.setLineDash([4 * px, 4 * px]);
    ctx.beginPath();
    const { cx, cy } = G.canvasCenter(v.W, v.H);
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, v.H);
    ctx.moveTo(0, cy);
    ctx.lineTo(v.W, cy);
    ctx.stroke();
    ctx.setLineDash([]);

    // regions in draw order: index 0 first (8.6a), clipped to the canvas (8.3e)
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, v.W, v.H);
    ctx.clip();
    this.state.regions.forEach((r, i) => {
      if (r.visible === false) return;
      const rect = this.rectOf(r);
      const col = S.regionColor(this.state, i);
      const sel = this.selected.has(r.id);
      const { w, h } = G.rectSize(rect);
      ctx.fillStyle = hexToRgba(col, sel ? 0.3 : 0.18);
      ctx.fillRect(rect.x0, rect.y0, w, h);
      ctx.strokeStyle = col;
      ctx.lineWidth = (sel ? 3 : 2) * px;
      if (sel && r.id !== this.primary) ctx.setLineDash([5 * px, 3 * px]);
      ctx.strokeRect(rect.x0, rect.y0, w, h);
      ctx.setLineDash([]);
      const vis = G.clipRect(rect, v.W, v.H);
      if (vis) {
        const n = S.lorasOf(r).filter((e) => e.lora).length;
        const kind = [n ? `${n} LoRA${n > 1 ? "s" : ""}` : "", S.hasPrompt(r) ? r.label || "object" : ""].filter(Boolean).join(" + ");
        const label = `${r.name || r.id}${kind ? ` · ${kind}` : ""}${r.locked ? " · locked" : ""}`;
        ctx.font = `${12 * px}px sans-serif`;
        const tw = ctx.measureText(label).width;
        ctx.fillStyle = hexToRgba(col, 0.85);
        ctx.fillRect(vis.x0, vis.y0, tw + 8 * px, 16 * px);
        ctx.fillStyle = "#fff";
        ctx.fillText(label, vis.x0 + 4 * px, vis.y0 + 12 * px);
      }
    });
    if (this.preview?.create) {
      const r = this.preview.create;
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = px;
      ctx.setLineDash([5 * px, 3 * px]);
      ctx.strokeRect(r.x0, r.y0, r.x1 - r.x0, r.y1 - r.y0);
      ctx.setLineDash([]);
    }
    ctx.restore();

    // handles on the primary selection (8.3d), Select tool only; unclipped so off-canvas edges stay grabbable
    const prim = this.region(this.primary);
    if (this.tool === "select" && prim && prim.visible !== false && !prim.locked) {
      const hp = G.handlePositions(this.rectOf(prim));
      const s = HANDLE_PX * px;
      ctx.fillStyle = "#fff";
      ctx.strokeStyle = "#000";
      ctx.lineWidth = px;
      for (const h of G.HANDLES) {
        ctx.fillRect(hp[h].x - s / 2, hp[h].y - s / 2, s, s);
        ctx.strokeRect(hp[h].x - s / 2, hp[h].y - s / 2, s, s);
      }
    }
    ctx.strokeStyle = "rgba(255,255,255,0.6)";
    ctx.lineWidth = px;
    ctx.strokeRect(0, 0, v.W, v.H);
    if (this.parseError) {
      ctx.fillStyle = "rgba(120,0,0,0.35)";
      ctx.fillRect(0, 0, v.W, v.H);
    }
  }

  // ------------------------------------------------------------------ layer list (8.6, 8.7)
  renderList() {
    const s = this.state;
    const { W, H } = this.canvasSize();
    const ro = !!this.parseError;
    const rows = [];
    for (const u of S.listUnits(s)) {
      if (u.type === "group") {
        rows.push(this.groupHeader(u, ro));
        if (!u.group.collapsed) for (const i of u.indices) rows.push(this.layerRow(i, true, W, H, ro));
      } else rows.push(this.layerRow(u.index, false, W, H, ro));
    }
    if (!rows.length) rows.push(el("div", { class: "mrl-row mrl-hint", text: "No boxes yet. Pick the Box tool (B) and drag on the canvas." }));
    const tail = el("div", { class: "mrl-row", style: "min-height:16px;border:none" });
    tail.addEventListener("click", () => this.select(null, false));
    this.dropTarget(tail, null, undefined);
    rows.push(tail);
    this.list.replaceChildren(...rows);
  }

  dropTarget(row, beforeId, group) {
    row.addEventListener("dragover", (e) => {
      if (!this.dragItem || this.parseError) return;
      e.preventDefault();
      row.classList.add("drop-before");
    });
    row.addEventListener("dragleave", () => row.classList.remove("drop-before"));
    row.addEventListener("drop", (e) => {
      row.classList.remove("drop-before");
      const item = this.dragItem;
      this.dragItem = null;
      if (!item) return;
      e.preventDefault();
      if (item.type === "region") this.commit(S.dropRegion(this.state, item.id, beforeId, group));
      else this.commit(S.dropGroup(this.state, item.id, beforeId));
    });
  }

  groupHeader(u, ro) {
    const g = u.group;
    const name = el("input", { type: "text", value: g.name, title: "Rename group" });
    name.addEventListener("change", () => this.commit(S.renameGroup(this.state, g.id, name.value)));
    const grip = el("span", { class: "mrl-grip", icon: "grip-vertical", draggable: ro ? "false" : "true", title: "Drag to reorder the group" });
    grip.addEventListener("dragstart", (e) => {
      this.dragItem = { type: "group", id: g.id };
      e.dataTransfer?.setData("text/plain", g.id);
    });
    const row = el("div", { class: "mrl-group" }, [
      grip,
      el("button", { icon: g.collapsed ? "chevron-right" : "chevron-down", class: "mrl-mini", title: "Collapse / expand", onclick: () => this.commit(S.setGroupCollapsed(this.state, g.id, !g.collapsed)) }),
      name,
      el("span", { class: "mrl-badge", text: String(u.indices.length) }),
      el("button", { icon: "eye", class: "mrl-mini", title: "Show all: make every box in the group visible", onclick: () => this.commit(S.setGroupVisible(this.state, g.id, true)) }),
      el("button", { icon: "eye-off", class: "mrl-mini", title: "Hide all: hide every box in the group", onclick: () => this.commit(S.setGroupVisible(this.state, g.id, false)) }),
      el("button", { icon: "chevron-up", class: "mrl-mini", title: "Move group up", onclick: () => this.commit(S.moveGroup(this.state, g.id, -1)) }),
      el("button", { icon: "chevron-down", class: "mrl-mini", title: "Move group down", onclick: () => this.commit(S.moveGroup(this.state, g.id, +1)) }),
      el("button", {
        icon: "trash-2",
        class: "mrl-mini",
        title: "Delete all: the group and every box in it",
        onclick: () => {
          if (!u.indices.length || confirm(`Delete group "${g.name}" and its ${u.indices.length} box(es)?`)) this.commit(S.deleteGroup(this.state, g.id));
        },
      }),
    ]);
    const first = u.indices.length ? this.state.regions[u.indices[0]].id : null;
    this.dropTarget(row, first, g.id);
    if (ro) for (const x of row.querySelectorAll("input,button")) x.disabled = true;
    return row;
  }

  layerRow(i, inGroup, W, H, ro) {
    const s = this.state;
    const r = s.regions[i];
    const id = r.id;
    const oob = G.outOfBounds(G.centeredToCanvas(r.bbox, W, H), W, H);
    const shown = r.visible !== false;
    const vis = el("button", {
      icon: shown ? "eye" : "eye-off",
      class: `mrl-mini mrl-vis${shown ? "" : " mrl-toggle-off"}`,
      title: shown ? "Visible: click to hide (hidden boxes are not emitted)" : "Hidden: click to show",
    });
    vis.addEventListener("click", (e) => {
      e.stopPropagation();
      this.commit(S.updateRegion(this.state, id, { visible: !shown }));
    });
    const grip = el("span", { class: "mrl-grip", icon: "grip-vertical", draggable: ro ? "false" : "true", title: "Drag to reorder" });
    grip.addEventListener("dragstart", (e) => {
      this.dragItem = { type: "region", id };
      e.dataTransfer?.setData("text/plain", id);
    });
    const nLoras = S.lorasOf(r).filter((e) => e.lora).length;
    const stop = (fn) => (e) => {
      e.stopPropagation();
      fn();
    };
    const row = el("div", { class: `mrl-row${this.selected.has(id) ? " sel" : ""}${inGroup ? " in-group" : ""}${r.visible === false ? " hidden-row" : ""}` }, [
      grip,
      el("span", { class: "mrl-chip", style: `background:${S.regionColor(s, i)}` }),
      el("span", { class: "mrl-name", text: r.name || r.id, title: r.name || r.id }),
      oob ? el("span", { class: "mrl-oob", icon: "triangle-alert", title: "Extends beyond the canvas: those parts cover no tokens" }) : null,
      nLoras ? el("span", { class: "mrl-badge", icon: "layers", iconSize: 11, text: String(nLoras), title: `${nLoras} LoRA(s)` }) : null,
      S.hasPrompt(r) ? el("span", { class: "mrl-badge", icon: "type", iconSize: 11, title: "In the layout prompt" }) : null,
      vis,
      el("button", { icon: r.locked ? "lock" : "lock-open", class: `mrl-mini${r.locked ? "" : " mrl-toggle-off"}`, title: r.locked ? "Locked: click to unlock" : "Lock (no moving or resizing on the canvas)", onclick: stop(() => this.commit(S.updateRegion(this.state, id, { locked: !r.locked }))) }),
      el("button", { icon: "chevron-up", class: "mrl-mini", title: "Move up (drawn earlier)", onclick: stop(() => this.commit(S.moveRegion(this.state, id, -1))) }),
      el("button", { icon: "chevron-down", class: "mrl-mini", title: "Move down (drawn later)", onclick: stop(() => this.commit(S.moveRegion(this.state, id, +1))) }),
    ]);
    row.addEventListener("click", (e) => this.select(id, e.shiftKey || e.ctrlKey || e.metaKey));
    this.dropTarget(row, id, inGroup ? r.group : null);
    if (ro) for (const x of row.querySelectorAll("input,button")) x.disabled = true;
    return row;
  }

  addGroup() {
    const { state, id } = S.addGroup(this.state);
    if (this.selected.size) {
      let s = state;
      for (const rid of this.state.regions.map((r) => r.id).filter((x) => this.selected.has(x))) s = S.assignGroup(s, rid, id);
      this.commit(s);
    } else this.commit(state);
  }

  // ------------------------------------------------------------------ properties panel
  renderProps() {
    this.closePicker();
    const ro = !!this.parseError;
    const r = this.region(this.primary);
    let body;
    if (this.selected.size > 1) {
      this.propsTitle.textContent = `${this.selected.size} boxes selected`;
      body = [this.multiSection()];
    } else if (r) {
      this.propsTitle.textContent = r.name || r.id;
      body = [this.boxSection(r), this.promptSection(r), this.lorasSection(r)];
    } else {
      this.propsTitle.textContent = "Canvas";
      body = [this.canvasSection()];
    }
    this.props.replaceChildren(...body);
    // only now is the curve panel (if a LoRA's curve is open) attached and measurable
    if (this.curvePanel.root.isConnected) this.curvePanel.render();
    if (ro) for (const x of this.props.querySelectorAll("input,button,select,textarea")) x.disabled = true;
  }

  field(label, control) {
    return el("label", { class: "mrl-field" }, [el("span", { text: label }), control]);
  }

  canvasSection() {
    const ed = this.state.editor || {};
    const { W, H } = this.canvasSize();
    const grid = el("input", { type: "number", min: "1", max: "32", step: "1", value: String(ed.grid_divisions ?? 3), style: "width:60px" });
    grid.addEventListener("change", () => this.setEditor({ grid_divisions: Math.max(1, parseInt(grid.value) || 3) }));
    const snap = el("input", { type: "checkbox", title: "Snap to grid while dragging (hold Ctrl/Cmd to bypass)" });
    snap.checked = !!ed.snap;
    snap.addEventListener("change", () => this.setEditor({ snap: snap.checked }));
    const refList = el("datalist", { id: `mrl-ref-${Math.random().toString(36).slice(2)}` });
    const ref = el("input", { type: "text", list: refList.id, value: ed.reference_image || "", placeholder: "file in ComfyUI/input" });
    ref.addEventListener("focus", () => inputFileList().then((files) => refList.replaceChildren(...files.map((f) => el("option", { value: f })))));
    ref.addEventListener("change", () => this.setEditor({ reference_image: ref.value.trim() }));
    const op = el("input", { type: "range", min: "0", max: "1", step: "0.05", value: String(ed.reference_opacity ?? 0.5), style: "flex:1" });
    op.addEventListener("input", () => {
      this.opacityPreview = parseFloat(op.value);
      this.draw();
    });
    op.addEventListener("change", () => {
      this.opacityPreview = null;
      this.setEditor({ reference_opacity: parseFloat(op.value) });
    });
    return el("div", { class: "mrl-sec" }, [
      el("div", { class: "mrl-hint", text: `${W} × ${H} px (the node's width/height). Select a box to edit it.` }),
      this.field("Grid ÷", grid),
      this.field("Snap", snap),
      this.field("Reference", ref),
      refList,
      this.field("Opacity", op),
      el("div", { class: "mrl-hint", text: "The reference image is a drawing aid only; it never affects the output." }),
    ]);
  }

  multiSection() {
    const ids = [...this.selected];
    return el("div", { class: "mrl-sec" }, [
      el("div", { class: "mrl-hint", text: "Edit one box at a time; with several selected you can group, move with the arrow keys, or delete them." }),
      el("div", { class: "mrl-field" }, [
        el("button", { icon: "folder-plus", text: "Group them", onclick: () => this.addGroup() }),
        el("button", {
          icon: "trash-2",
          text: "Delete",
          onclick: () => {
            const del = ids.filter((id) => !this.region(id)?.locked);
            this.selected.clear();
            this.primary = null;
            this.commit(S.deleteRegions(this.state, del));
          },
        }),
      ]),
    ]);
  }

  boxSection(r) {
    const id = r.id;
    const upd = (patch) => this.commit(S.updateRegion(this.state, id, patch));
    const i = this.state.regions.indexOf(r);
    const { W, H } = this.canvasSize();
    const name = el("input", { type: "text", value: r.name || "" });
    name.addEventListener("change", () => upd({ name: name.value }));
    const color = el("input", { type: "color", value: S.regionColor(this.state, i), title: "Layer colour" });
    color.addEventListener("change", () => upd({ color: color.value }));
    const labels = ["y_min", "x_min", "y_max", "x_max"];
    const inputs = labels.map((l, k) => el("input", { type: "number", step: "1", value: String(r.bbox[k]), title: `${l} (px from the canvas centre, y down)` }));
    for (const inp of inputs) {
      inp.addEventListener("change", () => {
        const b = inputs.map((x) => Math.round(Number(x.value)));
        if (b.every(Number.isFinite) && b[2] >= b[0] && b[3] >= b[1]) upd({ bbox: b });
        else this.renderProps(); // reject: put the stored values back
      });
    }
    const group = el("select", { title: "Group" });
    group.appendChild(el("option", { value: "", text: "—" }));
    for (const g of this.state.groups) group.appendChild(el("option", { value: g.id, text: g.name }));
    group.value = this.state.groups.some((g) => g.id === r.group) ? r.group : "";
    group.addEventListener("change", () => this.commit(S.assignGroup(this.state, id, group.value || null)));
    const oob = G.outOfBounds(G.centeredToCanvas(r.bbox, W, H), W, H);
    return el("div", { class: "mrl-sec" }, [
      el("div", { class: "mrl-sec-title" }, [
        el("span", { class: "grow", text: "Box" }),
        el("button", {
          icon: "trash-2",
          text: "Delete box",
          class: "mrl-mini",
          onclick: () => {
            this.selected.delete(id);
            this.primary = null;
            this.expanded = null;
            this.commit(S.deleteRegions(this.state, [id]));
          },
        }),
      ]),
      this.field("Name", name),
      el("div", { class: "mrl-field" }, [el("span", { text: "Colour" }), color]),
      el("div", { class: "mrl-hint", text: "bbox [y_min, x_min, y_max, x_max], px from the canvas centre (y down)" }),
      el(
        "div",
        { class: "mrl-bbox-grid" },
        labels.map((l, k) => el("label", {}, [l, inputs[k]])),
      ),
      oob ? el("div", { class: "mrl-oob", icon: "triangle-alert", text: "Extends beyond the canvas: those parts cover no tokens." }) : null,
      this.field("Group", group),
    ]);
  }

  promptSection(r) {
    const id = r.id;
    const upd = (patch) => this.commit(S.updateRegion(this.state, id, patch));
    const label = el("select", { title: "Layout label" });
    for (const l of S.LABELS) label.appendChild(el("option", { value: l, text: l }));
    if (!S.LABELS.includes(r.label)) label.appendChild(el("option", { value: r.label, text: r.label }));
    label.value = r.label || "object";
    label.addEventListener("change", () => upd({ label: label.value }));
    const caption = el("textarea", { placeholder: "caption for the layout prompt (empty = not in the prompt)" });
    caption.value = r.caption || "";
    caption.addEventListener("change", () => upd({ caption: caption.value }));
    const parts = [el("div", { class: "mrl-sec-title", text: "Layout prompt" }), this.field("Label", label), caption];
    if (r.label === "text") {
      const text = el("input", { type: "text", value: r.text || "", placeholder: "text to render" });
      text.addEventListener("change", () => upd({ text: text.value }));
      parts.push(this.field("Text", text));
    }
    const palette = Array.isArray(r.palette) ? r.palette : [];
    const swatches = palette.map((c, k) => {
      const sw = el("input", { type: "color", value: /^#[0-9a-f]{6}$/i.test(c) ? c : "#000000", title: `Palette colour ${k + 1}` });
      sw.addEventListener("change", () => upd({ palette: palette.map((x, j) => (j === k ? sw.value.toUpperCase() : x)) }));
      return el("span", {}, [sw, el("button", { icon: "x", class: "mrl-mini", title: "Remove colour", onclick: () => upd({ palette: palette.filter((_, j) => j !== k) }) })]);
    });
    if (palette.length < S.MAX_PALETTE) swatches.push(el("button", { icon: "plus", text: "colour", class: "mrl-mini", title: "Add a palette colour", onclick: () => upd({ palette: [...palette, "#808080"] }) }));
    parts.push(el("div", { class: "mrl-field" }, [el("span", { text: "Palette" }), ...swatches]));
    return el("div", { class: "mrl-sec" }, parts);
  }

  // ------------------------------------------------------------------ LoRA list
  entries() {
    const r = this.region(this.primary);
    return r ? S.lorasOf(r) : [];
  }

  setEntries(entries) {
    if (!this.primary) return;
    this.commit(S.setLoras(this.state, this.primary, entries));
  }

  expandedEntry() {
    if (!this.expanded || this.expanded.id !== this.primary) return null;
    const e = this.entries()[this.expanded.index];
    return e ? { curve: e.curve, name: loraDisplayName(e.lora) } : null;
  }

  updateExpandedEntry(patch) {
    if (!this.expanded) return;
    const list = this.entries().map((e) => ({ ...e }));
    if (!list[this.expanded.index]) return;
    Object.assign(list[this.expanded.index], patch);
    this.setEntries(list);
  }

  lorasSection(r) {
    const list = S.lorasOf(r);
    const rows = list.map((e, k) => this.loraEntry(r, e, k));
    return el("div", { class: "mrl-sec" }, [
      el("div", { class: "mrl-sec-title" }, [
        el("span", { class: "grow", text: "LoRAs" }),
        el("button", {
          icon: "plus",
          text: "Add LoRA",
          class: "mrl-mini",
          onclick: () => {
            this.expanded = null;
            this.setEntries([...S.lorasOf(this.region(this.primary)).map((x) => ({ ...x })), S.newLoraEntry("")]);
          },
        }),
      ]),
      ...(rows.length ? rows : [el("div", { class: "mrl-hint", text: "No LoRA on this box. Add one to apply it only inside the box." })]),
      rows.length ? el("div", { class: "mrl-hint", text: "Click a LoRA's curve to edit its strength over the run." }) : null,
    ]);
  }

  loraEntry(r, e, k) {
    const open = this.expanded && this.expanded.id === r.id && this.expanded.index === k;
    const i = this.state.regions.indexOf(r);
    const nameBtn = el("button", { class: "mrl-lora-name", title: e.lora || "Choose a LoRA" }, [
      el("span", { class: "mrl-lora-label", text: loraDisplayName(e.lora) }),
      icon("chevron-down", 13),
    ]);
    nameBtn.addEventListener("click", (ev) => {
      ev.stopPropagation();
      this.openPicker(nameBtn, e.lora, (path) => {
        const list = S.lorasOf(this.region(this.primary)).map((x) => ({ ...x }));
        list[k].lora = path;
        this.setEntries(list);
      });
    });
    const weight = el("input", { type: "number", step: "0.05", value: String(e.weight ?? 1), title: "Strength (the curve scales it from 0 to 100%)" });
    weight.addEventListener("change", () => {
      const w = parseFloat(weight.value);
      if (!Number.isFinite(w)) return this.renderProps();
      const list = S.lorasOf(this.region(this.primary)).map((x) => ({ ...x }));
      list[k].weight = w;
      this.setEntries(list);
    });
    const spark = el("canvas", { class: `mrl-spark${open ? " open" : ""}`, title: "Strength over the run: click to edit" });
    const cur = this.entryCurve(e);
    drawSparkline(spark, cur.points, !cur.own, S.regionColor(this.state, i));
    spark.addEventListener("click", () => {
      this.expanded = open ? null : { id: r.id, index: k };
      this.renderProps();
    });
    const remove = el("button", {
      icon: "x",
      class: "mrl-mini mrl-lora-remove",
      title: "Remove this LoRA from the box",
      onclick: () => {
        this.expanded = null;
        this.setEntries(S.lorasOf(this.region(this.primary)).filter((_, j) => j !== k));
      },
    });
    const box = el("div", { class: `mrl-lora${open ? " open" : ""}` }, [el("div", { class: "mrl-lora-row" }, [nameBtn, weight, spark, remove])]);
    if (open) box.appendChild(this.curvePanel.root); // rendered by renderProps once it is in the page
    return box;
  }

  // a filterable dropdown of every LoRA path (clicking the short name opens it)
  openPicker(anchor, current, onPick) {
    this.closePicker();
    const filter = el("input", { type: "text", placeholder: "filter…" });
    const listEl = el("div", { class: "mrl-picker-list" });
    const picker = el("div", { class: "mrl-picker" }, [filter, listEl]);
    const fill = () => {
      const q = filter.value.toLowerCase();
      const items = ["", ...this.loras].filter((p) => !q || p.toLowerCase().includes(q));
      if (current && !this.loras.includes(current) && (!q || current.toLowerCase().includes(q))) items.push(current);
      listEl.replaceChildren(
        ...items.map((p) => {
          const it = el("div", { class: `mrl-picker-item${p === current ? " cur" : ""}`, text: p || "(none)", title: p });
          it.addEventListener("pointerdown", (ev) => {
            ev.preventDefault();
            ev.stopPropagation();
            this.closePicker();
            onPick(p);
          });
          return it;
        }),
      );
    };
    fill();
    filter.addEventListener("input", fill);
    filter.addEventListener("keydown", (ev) => {
      ev.stopPropagation();
      if (ev.key === "Escape") this.closePicker();
      if (ev.key === "Enter") listEl.querySelector('.mrl-picker-item:not([title=""])')?.dispatchEvent(new PointerEvent("pointerdown"));
    });
    picker.addEventListener("pointerdown", (ev) => ev.stopPropagation());
    document.body.appendChild(picker);
    const rect = anchor.getBoundingClientRect();
    const top = Math.min(rect.bottom + 2, window.innerHeight - 310);
    picker.style.left = `${Math.max(4, Math.min(rect.left, window.innerWidth - 330))}px`;
    picker.style.top = `${Math.max(4, top)}px`;
    this.picker = picker;
    this.pickerAway = (ev) => {
      if (!picker.contains(ev.target)) this.closePicker();
    };
    setTimeout(() => document.addEventListener("pointerdown", this.pickerAway, true), 0);
    filter.focus();
    listEl.querySelector(".cur")?.scrollIntoView({ block: "nearest" });
  }

  closePicker() {
    if (this.pickerAway) document.removeEventListener("pointerdown", this.pickerAway, true);
    this.pickerAway = null;
    this.picker?.remove();
    this.picker = null;
  }

  setStatus(summary) {
    if (!summary) return;
    const items = summary.prompt_items != null ? ` · ${summary.prompt_items} layout prompt item(s)` : "";
    if (summary.status === "active") {
      this.status = `Last run: ${summary.regions} LoRA(s) on boxes, ${summary.applied} LoRA layer(s) on ${summary.modules} module(s), ${summary.skipped} key(s) skipped${items}`;
    } else this.status = `Last run: ${summary.message || "inactive"}${items}`;
    this.statusLine.textContent = this.status;
  }
}

app.registerExtension({
  name: "moss.RegionalLoraEditor",
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== NODE_NAME) return;

    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      const r = onNodeCreated?.apply(this, arguments);
      const editor = new RegionEditor(this);
      this.__mrlEditor = editor;
      const widget = this.addDOMWidget("region_editor", "moss_region_editor", editor.root, {
        serialize: false,
        hideOnZoom: false,
        getMinHeight: () => 560,
        getValue: () => undefined,
        setValue: () => {},
      });
      widget.serialize = false; // the JSON lives in the "regions" widget only (8.1c)
      this.setSize([Math.max(this.size[0], 1180), Math.max(this.size[1], 980)]);
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      onExecuted?.apply(this, arguments);
      this.__mrlEditor?.setStatus(output?.moss_regional_lora?.[0]);
    };

    // Rearrange widget values of workflows saved before vae_downscale/patch_size were removed, before
    // ComfyUI assigns them to widgets by position.
    const configure = nodeType.prototype.configure;
    nodeType.prototype.configure = function (info) {
      if (info?.widgets_values) info.widgets_values = dropRemovedGridValues(info.widgets_values);
      return configure.apply(this, arguments);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
      const r = onConfigure?.apply(this, arguments);
      // workflow load: re-read the restored widget value on the next tick
      setTimeout(() => this.__mrlEditor?.poll(), 0);
      return r;
    };

    const onRemoved = nodeType.prototype.onRemoved;
    nodeType.prototype.onRemoved = function () {
      this.__mrlEditor?.destroy();
      return onRemoved?.apply(this, arguments);
    };
  },
});
