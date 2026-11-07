// In-node region editor for "MOSS-Image Regional LoRA" (spec section 8).
//
// The node's `regions` STRING widget holds the JSON and is the only source of truth (8.1c, 8.2a).
// This editor is a view over it: it parses the text, draws it, and writes the serialised result back
// after each edit. It never keeps state the JSON does not have, and it never writes while the JSON
// fails to parse (8.9a).

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import * as G from "./geometry.js";
import * as S from "./state.js";

const NODE_NAME = "MossRegionalLora";
const POLL_MS = 250;
const HANDLE_PX = 7; // half-size of a resize handle, in screen px
const MIN_CREATE_PX = 3; // drags shorter than this (image px) are clicks, not creates

const CSS = `
.mrl-root{display:flex;flex-direction:column;gap:4px;width:100%;height:100%;box-sizing:border-box;font:12px sans-serif;color:var(--input-text,#ddd);outline:none}
.mrl-bar{display:flex;flex-wrap:wrap;align-items:center;gap:6px}
.mrl-bar input[type=number]{width:44px}
.mrl-bar button,.mrl-row button,.mrl-group button{background:var(--comfy-input-bg,#222);color:inherit;border:1px solid var(--border-color,#555);border-radius:3px;padding:0 5px;cursor:pointer;line-height:18px}
.mrl-bar button:disabled{opacity:.4;cursor:default}
.mrl-dims{font-weight:bold}
.mrl-readout{font-family:monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.mrl-error{display:none;background:#5a1d1d;border:1px solid #c44;padding:4px 6px;border-radius:3px;white-space:pre-wrap}
.mrl-warn{display:none;background:#4a3d12;border:1px solid #a83;padding:2px 6px;border-radius:3px}
.mrl-canvas-wrap{position:relative;flex:1 1 auto;min-height:160px;overflow:hidden}
.mrl-canvas-wrap canvas{position:absolute;left:0;top:0;outline:none}
.mrl-list{flex:0 0 auto;max-height:38%;overflow:auto;border:1px solid var(--border-color,#444);border-radius:3px}
.mrl-row,.mrl-group{display:flex;align-items:center;gap:4px;padding:2px 4px;border-bottom:1px solid rgba(128,128,128,.2)}
.mrl-row.sel{background:rgba(80,140,255,.25)}
.mrl-row.in-group{padding-left:18px}
.mrl-row.hidden-row{opacity:.5}
.mrl-row.drop-before,.mrl-group.drop-before{box-shadow:inset 0 2px 0 #6af}
.mrl-group{background:rgba(128,128,128,.15);font-weight:bold}
.mrl-row input[type=text],.mrl-group input[type=text]{width:84px;background:transparent;color:inherit;border:1px solid transparent}
.mrl-row input[type=text]:focus,.mrl-group input[type=text]:focus{border-color:#888}
.mrl-row input[type=number]{width:48px}
.mrl-row select{max-width:150px}
.mrl-row input[type=color]{width:18px;height:16px;padding:0;border:none;background:none}
.mrl-grip{cursor:grab;user-select:none;opacity:.7}
.mrl-bbox{font-family:monospace;font-size:11px;opacity:.85;min-width:130px}
.mrl-oob{color:#fb4}
.mrl-status{opacity:.85}
`;

function injectCss() {
  if (document.getElementById("mrl-css")) return;
  const s = document.createElement("style");
  s.id = "mrl-css";
  s.textContent = CSS;
  document.head.appendChild(s);
}

function el(tag, attrs = {}, children = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else if (v !== undefined && v !== null) e.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c) e.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  return e;
}

function hexToRgba(hex, a) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex || "");
  if (!m) return `rgba(200,200,200,${a})`;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
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
    this.history = new S.History(100);
    this.state = S.emptyState();
    this.parseError = null;
    this.warnings = [];
    this.lastText = null;
    this.lastDims = "";
    this.selected = new Set();
    this.primary = null;
    this.drag = null;
    this.preview = null; // {id, bbox} during a drag, or {create: rect}
    this.refImg = null;
    this.refName = null;
    this.refState = "none"; // none | loading | ok | missing
    this.loras = [];
    this.status = "";
    this.buildDom();
    loraList().then((l) => {
      this.loras = l;
      this.renderList();
    });
    this.syncFromWidget(true);
    this.timer = setInterval(() => this.poll(), POLL_MS);
    this.ro = new ResizeObserver(() => this.draw());
    this.ro.observe(this.wrap);
  }

  destroy() {
    clearInterval(this.timer);
    this.ro?.disconnect();
  }

  // ------------------------------------------------------------------ DOM
  buildDom() {
    injectCss();
    this.dims = el("span", { class: "mrl-dims" });
    this.gridInput = el("input", { type: "number", min: "1", max: "32", step: "1", title: "Grid divisions per side" });
    this.gridInput.addEventListener("change", () => this.setEditor({ grid_divisions: Math.max(1, parseInt(this.gridInput.value) || 3) }));
    this.snapInput = el("input", { type: "checkbox", title: "Snap to grid while dragging (hold Ctrl/Cmd to bypass)" });
    this.snapInput.addEventListener("change", () => this.setEditor({ snap: this.snapInput.checked }));
    this.undoBtn = el("button", { text: "Undo", title: "Ctrl/Cmd+Z", onclick: () => this.undo() });
    this.redoBtn = el("button", { text: "Redo", title: "Ctrl/Cmd+Shift+Z", onclick: () => this.redo() });
    const addGroupBtn = el("button", { text: "+ Group", onclick: () => this.addGroup() });
    this.refInput = el("input", { type: "text", list: "", placeholder: "reference image (input/)", style: "width:130px" });
    this.refList = el("datalist", { id: `mrl-ref-${Math.random().toString(36).slice(2)}` });
    this.refInput.setAttribute("list", this.refList.id);
    this.refInput.addEventListener("focus", () =>
      inputFileList().then((files) => {
        this.refList.replaceChildren(...files.map((f) => el("option", { value: f })));
      }),
    );
    this.refInput.addEventListener("change", () => this.setEditor({ reference_image: this.refInput.value.trim() }));
    this.refOpacity = el("input", { type: "range", min: "0", max: "1", step: "0.05", title: "Reference opacity", style: "width:70px" });
    this.refOpacity.addEventListener("input", () => {
      this.opacityPreview = parseFloat(this.refOpacity.value);
      this.draw();
    });
    this.refOpacity.addEventListener("change", () => {
      this.opacityPreview = null;
      this.setEditor({ reference_opacity: parseFloat(this.refOpacity.value) });
    });

    const bar = el("div", { class: "mrl-bar" }, [
      this.dims,
      el("label", {}, ["Grid ÷ ", this.gridInput]),
      el("label", {}, [this.snapInput, " Snap"]),
      this.undoBtn,
      this.redoBtn,
      addGroupBtn,
      this.refInput,
      this.refList,
      this.refOpacity,
    ]);
    this.readout = el("div", { class: "mrl-readout" });
    this.errorBox = el("div", { class: "mrl-error" });
    this.warnBox = el("div", { class: "mrl-warn" });
    this.canvas = el("canvas", { tabindex: "0" });
    this.wrap = el("div", { class: "mrl-canvas-wrap" }, [this.canvas]);
    this.list = el("div", { class: "mrl-list" });
    this.statusLine = el("div", { class: "mrl-status" });
    this.root = el("div", { class: "mrl-root", tabindex: "-1" }, [
      bar,
      this.readout,
      this.errorBox,
      this.warnBox,
      this.wrap,
      this.list,
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
    const d = `${W}x${H}`;
    if (d !== this.lastDims) {
      this.lastDims = d;
      this.renderAll();
    }
  }

  syncFromWidget(initial) {
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
    this.syncFromWidget(false);
  }

  redo() {
    if (this.parseError) return;
    const t = this.history.redo(this.text);
    if (t == null) return;
    this.writeText(t);
    this.syncFromWidget(false);
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
    this.renderAll();
  }

  // ------------------------------------------------------------------ pointer interaction (8.4a)
  onDown(e) {
    if (e.button !== 0) return;
    e.stopPropagation();
    this.canvas.focus();
    if (this.parseError) return;
    const { p, v } = this.eventPoint(e);
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
    if (!additive) this.select(null, false);
    const start = G.clampPoint(p, v.W, v.H);
    this.drag = { kind: "create", start };
    this.canvas.setPointerCapture(e.pointerId);
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
    let cursor = "crosshair";
    const prim = this.region(this.primary);
    if (prim && prim.visible !== false && !prim.locked) {
      const h = G.handleAt(this.rectOf(prim), p, HANDLE_PX / (v.scale || 1));
      if (h) cursor = G.cursorFor(h);
    }
    if (cursor === "crosshair") {
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
      e.stopPropagation(); // typing in a row field must not trigger graph shortcuts
      return;
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
      this.commit(S.deleteRegions(this.state, ids));
    }
  }

  // ------------------------------------------------------------------ rendering
  renderAll() {
    this.renderBar();
    this.renderReadout();
    this.renderList();
    this.loadReference();
    this.draw();
  }

  renderBar() {
    const { W, H } = this.canvasSize();
    this.dims.textContent = `${W} × ${H} px`;
    const ed = this.state.editor || {};
    if (document.activeElement !== this.gridInput) this.gridInput.value = ed.grid_divisions ?? 3;
    this.snapInput.checked = !!ed.snap;
    if (document.activeElement !== this.refInput) this.refInput.value = ed.reference_image || "";
    if (this.opacityPreview == null) this.refOpacity.value = ed.reference_opacity ?? 0.5;
    this.undoBtn.disabled = !this.history.canUndo || !!this.parseError;
    this.redoBtn.disabled = !this.history.canRedo || !!this.parseError;
    const disabled = !!this.parseError;
    for (const x of [this.gridInput, this.snapInput, this.refInput, this.refOpacity]) x.disabled = disabled;
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

  renderReadout() {
    const r = this.region(this.primary);
    if (!r) {
      const n = S.emittedRegions(this.state).length;
      this.readout.textContent = `No selection · ${n} region${n === 1 ? "" : "s"} emitted · drag on the canvas to add one`;
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

    // canvas area
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
        const label = `${r.name || r.id}${r.locked ? " 🔒" : ""}`;
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

    // handles on the primary selection (8.3d); drawn unclipped so off-canvas edges stay grabbable
    const prim = this.region(this.primary);
    if (prim && prim.visible !== false && !prim.locked) {
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
    // canvas border
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
        if (!u.group.collapsed) for (const i of u.indices) rows.push(this.regionRow(i, true, W, H, ro));
      } else rows.push(this.regionRow(u.index, false, W, H, ro));
    }
    if (!rows.length) rows.push(el("div", { class: "mrl-row", text: "No regions yet. Drag on the canvas to draw one." }));
    const tail = el("div", { class: "mrl-row", style: "min-height:8px;border:none" });
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
    const grip = el("span", { class: "mrl-grip", text: "⠿", draggable: ro ? "false" : "true", title: "Drag to reorder the group" });
    grip.addEventListener("dragstart", (e) => {
      this.dragItem = { type: "group", id: g.id };
      e.dataTransfer?.setData("text/plain", g.id);
    });
    const row = el("div", { class: "mrl-group" }, [
      grip,
      el("button", { text: g.collapsed ? "▸" : "▾", title: "Collapse / expand", onclick: () => this.commit(S.setGroupCollapsed(this.state, g.id, !g.collapsed)) }),
      name,
      el("span", { text: `(${u.indices.length})` }),
      el("button", { text: "Show all", onclick: () => this.commit(S.setGroupVisible(this.state, g.id, true)) }),
      el("button", { text: "Hide all", onclick: () => this.commit(S.setGroupVisible(this.state, g.id, false)) }),
      el("button", { text: "▲", title: "Move group up", onclick: () => this.commit(S.moveGroup(this.state, g.id, -1)) }),
      el("button", { text: "▼", title: "Move group down", onclick: () => this.commit(S.moveGroup(this.state, g.id, +1)) }),
      el("button", {
        text: "Delete all",
        title: "Delete the group and every region in it",
        onclick: () => {
          if (!u.indices.length || confirm(`Delete group "${g.name}" and its ${u.indices.length} region(s)?`))
            this.commit(S.deleteGroup(this.state, g.id));
        },
      }),
    ]);
    const first = u.indices.length ? this.state.regions[u.indices[0]].id : null;
    this.dropTarget(row, first, g.id);
    if (ro) for (const x of row.querySelectorAll("input,button")) x.disabled = true;
    return row;
  }

  regionRow(i, inGroup, W, H, ro) {
    const s = this.state;
    const r = s.regions[i];
    const id = r.id;
    const upd = (patch) => this.commit(S.updateRegion(this.state, id, patch));
    const oob = G.outOfBounds(G.centeredToCanvas(r.bbox, W, H), W, H);

    const vis = el("input", { type: "checkbox", title: "Visible (hidden regions are not emitted)" });
    vis.checked = r.visible !== false;
    vis.addEventListener("change", () => upd({ visible: vis.checked }));
    const color = el("input", { type: "color", value: S.regionColor(s, i), title: "Layer colour" });
    color.addEventListener("change", () => upd({ color: color.value }));
    const name = el("input", { type: "text", value: r.name || "", title: "Name" });
    name.addEventListener("change", () => upd({ name: name.value }));
    const lora = el("select", { title: "LoRA file" });
    const opts = ["", ...this.loras];
    if (r.lora && !opts.includes(r.lora)) opts.push(r.lora);
    for (const o of opts) lora.appendChild(el("option", { value: o, text: o || "(no LoRA)" }));
    lora.value = r.lora || "";
    lora.addEventListener("change", () => upd({ lora: lora.value }));
    const weight = el("input", { type: "number", step: "0.05", value: String(r.weight ?? 1), title: "Weight" });
    weight.addEventListener("change", () => {
      const w = parseFloat(weight.value);
      if (Number.isFinite(w)) upd({ weight: w });
    });
    const lock = el("button", { text: r.locked ? "🔒" : "🔓", title: "Lock (no moving or resizing on the canvas)", onclick: () => upd({ locked: !r.locked }) });
    const group = el("select", { title: "Group" });
    group.appendChild(el("option", { value: "", text: "—" }));
    for (const g of s.groups) group.appendChild(el("option", { value: g.id, text: g.name }));
    group.value = s.groups.some((g) => g.id === r.group) ? r.group : "";
    group.addEventListener("change", () => this.commit(S.assignGroup(this.state, id, group.value || null)));
    const grip = el("span", { class: "mrl-grip", text: "⠿", draggable: ro ? "false" : "true", title: "Drag to reorder" });
    grip.addEventListener("dragstart", (e) => {
      this.dragItem = { type: "region", id };
      e.dataTransfer?.setData("text/plain", id);
    });

    const row = el("div", { class: `mrl-row${this.selected.has(id) ? " sel" : ""}${inGroup ? " in-group" : ""}${r.visible === false ? " hidden-row" : ""}` }, [
      grip,
      vis,
      color,
      name,
      el("span", { class: "mrl-bbox", text: `[${r.bbox.join(", ")}]` }),
      oob ? el("span", { class: "mrl-oob", text: "⚠", title: "Extends beyond the canvas: those parts cover no tokens" }) : null,
      lora,
      weight,
      lock,
      group,
      el("button", { text: "▲", title: "Move up (drawn earlier)", onclick: () => this.commit(S.moveRegion(this.state, id, -1)) }),
      el("button", { text: "▼", title: "Move down (drawn later)", onclick: () => this.commit(S.moveRegion(this.state, id, +1)) }),
      el("button", {
        text: "✕",
        title: "Delete",
        onclick: () => {
          this.selected.delete(id);
          if (this.primary === id) this.primary = null;
          this.commit(S.deleteRegions(this.state, [id]));
        },
      }),
    ]);
    row.addEventListener("click", (e) => {
      if (e.target !== row && e.target.tagName !== "SPAN") return;
      this.select(id, e.shiftKey || e.ctrlKey || e.metaKey);
    });
    this.dropTarget(row, id, inGroup ? r.group : null);
    if (ro) for (const x of row.querySelectorAll("input,button,select")) x.disabled = true;
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

  setStatus(summary) {
    if (!summary) return;
    if (summary.status === "active") {
      this.status = `Last run: ${summary.regions} region(s), ${summary.applied} LoRA layer(s) on ${summary.modules} module(s), ${summary.skipped} key(s) skipped`;
    } else this.status = `Last run: ${summary.message || "inactive"}`;
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
        getMinHeight: () => 460,
        getValue: () => undefined,
        setValue: () => {},
      });
      widget.serialize = false; // the JSON lives in the "regions" widget only (8.1c)
      this.setSize([Math.max(this.size[0], 640), Math.max(this.size[1], 900)]);
      return r;
    };

    const onExecuted = nodeType.prototype.onExecuted;
    nodeType.prototype.onExecuted = function (output) {
      onExecuted?.apply(this, arguments);
      this.__mrlEditor?.setStatus(output?.moss_regional_lora?.[0]);
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
