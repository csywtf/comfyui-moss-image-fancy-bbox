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
import { el, hideWidget, injectCss, loraDisplayName, loraList } from "./regional_lora_editor.js";

const NODE_NAME = "MossCharacterMasks";
const EDITOR_WIDGET = "masks_editor";
const UI_KEY = "moss_character_masks";
const POLL_MS = 250;
const POINT_HIT_PX = 9; // clicking within this many screen px of a point removes it
const SETTINGS = [
  ["threshold", "Threshold", 0.01, "How confident SAM3 must be to accept a part. Lower finds more."],
  ["box_margin", "Box margin", 0.01, "How far outside each box (as a fraction of its size) to look for its parts."],
  ["refine_iterations", "Refine", 1, "SAM3 mask refinement passes (0 = raw detector masks)."],
];

const CSS = `
.mcm-boxes{flex:0 0 220px;width:220px;min-width:0}
.mcm-row{display:flex;align-items:center;gap:5px;padding:3px 5px;border-bottom:1px solid rgba(128,128,128,.2);cursor:default;user-select:none}
.mcm-row:hover{background:rgba(128,128,128,.12)}
.mcm-row.sel{background:rgba(80,140,255,.28)}
.mcm-row.off .mrl-name{opacity:.5}
.mcm-dots{display:flex;gap:2px}
.mcm-dot{width:8px;height:8px;border-radius:50%;border:1px solid rgba(0,0,0,.4)}
.mcm-dot.missing{background:transparent!important;border:1px dashed #f88}
.mcm-parts{display:grid;grid-template-columns:auto 10px 44px 1fr 68px;gap:3px 4px;align-items:center}
.mcm-parts input[type=text]{min-width:0;width:100%}
.mcm-parts input[type=number]{width:68px}
.mcm-chip{width:10px;height:10px;border-radius:2px}
.mcm-status{font-size:10px;opacity:.8;grid-column:2 / -1;margin-top:-2px}
.mcm-status.missing{color:#f99;opacity:1}
.mcm-loras{display:flex;flex-direction:column;gap:3px}
.mcm-lora{display:flex;gap:3px;align-items:center}
.mcm-lora input[type=text]{flex:1;min-width:0}
.mcm-lora input[type=number]{width:54px}
.mcm-empty{padding:8px;opacity:.7}
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
    this.ro = new ResizeObserver(() => this.draw());
    this.ro.observe(this.wrap);
    this.poll();
  }

  destroy() {
    clearInterval(this.timer);
    this.ro?.disconnect();
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
    this.wrap = el("div", { class: "mrl-canvas-wrap" }, [this.canvas]);
    this.propsTitle = el("span", { class: "grow" });
    this.props = el("div", { class: "mrl-panel-body" });
    const props = el("div", { class: "mrl-panel mrl-props" }, [el("div", { class: "mrl-panel-head" }, [this.propsTitle]), this.props]);
    this.errorBox = el("div", { class: "mrl-error" });
    this.statusLine = el("div", { class: "mrl-status" });
    this.root = el("div", { class: "mrl-root mcm-root", tabindex: "-1" }, [this.errorBox, el("div", { class: "mrl-main" }, [boxes, this.wrap, props]), this.statusLine]);
    this.canvas.style.cursor = "default"; // not ComfyUI's drag hand
    this.canvas.addEventListener("pointerdown", (e) => this.onCanvasDown(e));
    this.canvas.addEventListener("pointermove", (e) => this.onCanvasMove(e));
    this.root.addEventListener("pointerdown", (e) => e.stopPropagation());
    this.root.addEventListener("keydown", (e) => e.stopPropagation()); // typing here isn't a graph shortcut
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

  doc() {
    return D.parseDetail(this.widget("detail")?.value);
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
    this.setWidget("detail", D.serializeDetail(D.setDetail(d.doc, region.id, next)));
    this.poll();
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
    const sig = [this.src?.text, this.src?.W, this.src?.H, this.widget("detail")?.value, ...SETTINGS.map(([n]) => this.widget(n)?.value)].join("\u0001");
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
    } catch {
      this.image = this.labels = null;
    }
    this.sig = "";
    this.poll();
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
        dots,
      ]);
      row.addEventListener("click", () => this.select(region.id));
      rows.push(row);
    }
    if (this.src?.ok && !rows.length) rows.push(el("div", { class: "mcm-empty", text: "The connected node has no boxes yet." }));
    this.list.replaceChildren(...rows);
    const n = this.boxes().filter(({ region }) => this.detailOf(region).enabled).length;
    this.statusLine.textContent = this.result ? `Last run: ${this.result.boxes.length} box(es) masked · ${n} set to detail` : `${n} box(es) set to detail · run the workflow to see the masks`;
  }

  select(id) {
    this.selected = this.selected === id ? null : id;
    this.overlay = null;
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
    this.props.replaceChildren(this.boxSection(region), this.pointsSection(region), this.lorasSection(region), this.partsSection(region));
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
    return el("div", { class: "mrl-sec" }, [
      el("div", { class: "mrl-sec-title", text: "SAM3" }),
      ...rows,
      el("div", { class: "mrl-hint", text: "Select a box to choose which parts are masked and how they are detailed. Tick a box in the list to detail it." }),
    ]);
  }

  boxSection(region) {
    const det = this.detailOf(region);
    const on = el("input", { type: "checkbox" });
    on.checked = det.enabled;
    on.addEventListener("change", () => this.updateDetail(region, (x) => (x.enabled = on.checked)));
    const prompt = el("textarea", { placeholder: "Added to every part's prompt, e.g. green eyes, freckles" });
    prompt.value = det.prompt;
    prompt.addEventListener("change", () => this.updateDetail(region, (x) => (x.prompt = prompt.value)));
    const rb = this.resultBox(region.id);
    return el("div", { class: "mrl-sec" }, [
      this.field("Detail", on, "Mask and detail this box"),
      el("div", { class: "mrl-hint", text: region.caption ? `Caption: ${region.caption}` : "This box has no caption; its parts are prompted with the detail prompt and part prompts only." }),
      this.field("Prompt", prompt),
      rb && rb.outline_found === false ? el("div", { class: "mrl-hint", style: "color:#f99", text: "SAM3 found no character in this box; the whole box was used as its outline." }) : null,
    ]);
  }

  pointsSection(region) {
    const pts = this.detailOf(region).points;
    const plus = pts.filter((p) => p.label).length;
    return el("div", { class: "mrl-sec" }, [
      el("div", { class: "mrl-sec-title" }, [
        el("span", { class: "grow", text: "Points" }),
        pts.length ? el("button", { icon: "trash-2", text: "Clear", class: "mrl-mini", title: "Remove this box's points", onclick: () => this.updateDetail(region, (d) => (d.points = [])) }) : null,
      ]),
      el("div", { class: "mrl-hint", text: "Shift-click the preview on this character's arm or hand (+) to make it hers; Alt-click (−) on something that isn't. Click a point to remove it. Masks update on the next run." }),
      pts.length ? el("div", { class: "mrl-hint", text: `${plus} + point(s), ${pts.length - plus} − point(s)` }) : null,
    ]);
  }

  lorasSection(region) {
    const det = this.detailOf(region);
    const use = el("input", { type: "checkbox" });
    use.checked = det.use_box_loras;
    use.addEventListener("change", () => this.updateDetail(region, (x) => (x.use_box_loras = use.checked)));
    const own = S.lorasOf(region).filter((e) => e.lora && e.weight).map((e) => `${loraDisplayName(e.lora)} (${e.weight})`);
    const listId = `mcm-loras-${Math.random().toString(36).slice(2)}`;
    const datalist = el("datalist", { id: listId }, this.loras.map((l) => el("option", { value: l })));
    const rows = det.loras.map((e, k) => {
      const name = el("input", { type: "text", list: listId, value: e.lora, placeholder: "LoRA file" });
      name.addEventListener("change", () => this.updateDetail(region, (x) => (x.loras[k].lora = name.value.trim())));
      const weight = el("input", { type: "number", step: "0.05", value: String(e.weight), title: "Weight" });
      weight.addEventListener("change", () => {
        const v = Number(weight.value);
        if (Number.isFinite(v)) this.updateDetail(region, (x) => (x.loras[k].weight = v));
      });
      const remove = el("button", { icon: "x", class: "mrl-mini", title: "Remove", onclick: () => this.updateDetail(region, (x) => x.loras.splice(k, 1)) });
      return el("div", { class: "mcm-lora" }, [name, weight, remove]);
    });
    return el("div", { class: "mrl-sec" }, [
      el("div", { class: "mrl-sec-title", text: "LoRAs" }),
      this.field("Box LoRAs", use, "Apply the box's own LoRAs (at their weights) while detailing"),
      el("div", { class: "mrl-hint", text: own.length ? own.join(", ") : "The box has no LoRAs." }),
      el("div", { class: "mcm-loras" }, rows),
      datalist,
      el("button", { icon: "plus", text: "Add detail LoRA", title: "A LoRA used only while detailing this box", onclick: () => this.updateDetail(region, (x) => x.loras.push({ lora: "", weight: 1 })) }),
    ]);
  }

  partsSection(region) {
    const det = this.detailOf(region);
    const rb = this.resultBox(region.id);
    const cells = [];
    for (const p of D.PARTS) {
      const s = det.parts[p.name];
      const on = el("input", { type: "checkbox", title: `Mask and detail the ${p.name}` });
      on.checked = s.on;
      on.addEventListener("change", () => this.updateDetail(region, (x) => (x.parts[p.name].on = on.checked)));
      const prompt = el("input", { type: "text", value: s.prompt, placeholder: p.hint, title: `Prompt for the ${p.name} (empty: "${p.hint}")` });
      prompt.addEventListener("change", () => this.updateDetail(region, (x) => (x.parts[p.name].prompt = prompt.value)));
      const dn = el("input", { type: "number", min: "0", max: "1", step: "0.05", value: s.denoise == null ? "" : String(s.denoise), placeholder: "default", title: "Denoise for this part (empty: the detailer's default)" });
      dn.addEventListener("change", () => {
        const v = dn.value.trim() === "" ? null : Math.min(1, Math.max(0, Number(dn.value)));
        if (v === null || Number.isFinite(v)) this.updateDetail(region, (x) => (x.parts[p.name].denoise = v));
      });
      cells.push(on, el("span", { class: "mcm-chip", style: `background:${p.color}` }), el("span", { text: p.name }), prompt, dn);
      if (rb && s.on && det.enabled && p.name in (rb.found || {})) {
        const ok = rb.found[p.name];
        cells.push(el("span"), el("span", { class: `mcm-status${ok ? "" : " missing"}`, text: ok ? "found" : "not found: try a lower threshold" }));
      }
    }
    return el("div", { class: "mrl-sec" }, [
      el("div", { class: "mrl-sec-title", text: "Parts" }),
      el("div", { class: "mcm-parts" }, cells),
      el("div", { class: "mrl-hint", text: "Prompt and denoise are used by the detailer; changing them doesn't re-run SAM3." }),
    ]);
  }

  drawPoints(g, v) {
    for (const { region, color } of this.boxes()) {
      const sel = region.id === this.selected;
      if (this.selected && !sel) continue; // only the selected box's points (or everyone's when none is)
      for (const p of this.detailOf(region).points) {
        const cx = v.ox + p.x * v.W * v.s;
        const cy = v.oy + p.y * v.H * v.s;
        g.beginPath();
        g.arc(cx, cy, 7, 0, Math.PI * 2);
        g.fillStyle = p.label ? color : "#5a1d1d";
        g.fill();
        g.lineWidth = 2;
        g.strokeStyle = p.label ? "#fff" : "#f66";
        g.stroke();
        g.beginPath();
        g.moveTo(cx - 3.5, cy);
        g.lineTo(cx + 3.5, cy);
        if (p.label) {
          g.moveTo(cx, cy - 3.5);
          g.lineTo(cx, cy + 3.5);
        }
        g.strokeStyle = "#fff";
        g.stroke();
      }
    }
  }

  // ------------------------------------------------------------------ preview
  /** Layout geometry (CSS px before the graph's zoom), as the main editor does. */
  view() {
    const cw = this.wrap.clientWidth;
    const ch = this.wrap.clientHeight;
    const W = this.result?.width || this.src?.W || 1024;
    const H = this.result?.height || this.src?.H || 1024;
    const s = Math.max(1e-6, Math.min((cw - 2) / W, (ch - 2) / H));
    return { cw, ch, W, H, s, ox: (cw - W * s) / 2, oy: (ch - H * s) / 2 };
  }

  /** A pointer event in image pixels. The graph's zoom CSS-scales this editor, so the on-screen offset is
   * normalised back to layout px first. */
  eventPoint(e) {
    const v = this.view();
    const r = this.canvas.getBoundingClientRect();
    const sx = r.width ? this.canvas.clientWidth / r.width : 1;
    const sy = r.height ? this.canvas.clientHeight / r.height : 1;
    return { x: ((e.clientX - r.left) * sx - v.ox) / v.s, y: ((e.clientY - r.top) * sy - v.oy) / v.s, v };
  }

  /** Index of the selected box's point under (x, y) image px, or -1. */
  pointAt(region, x, y, v) {
    return this.detailOf(region).points.findIndex((p) => Math.hypot((p.x * v.W - x) * v.s, (p.y * v.H - y) * v.s) <= POINT_HIT_PX);
  }

  onCanvasMove(e) {
    const region = this.boxes().find(({ region }) => region.id === this.selected)?.region;
    let cursor = "default";
    if (region) {
      const { x, y, v } = this.eventPoint(e);
      cursor = this.pointAt(region, x, y, v) >= 0 ? "pointer" : "crosshair";
    }
    if (this.canvas.style.cursor !== cursor) this.canvas.style.cursor = cursor;
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
    this.canvas.width = Math.max(1, Math.round(v.cw * dpr));
    this.canvas.height = Math.max(1, Math.round(v.ch * dpr));
    this.canvas.style.width = `${v.cw}px`;
    this.canvas.style.height = `${v.ch}px`;
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
      this.drawPoints(g, v);
    }
  }

  onCanvasDown(e) {
    const { x, y, v } = this.eventPoint(e);
    const region = this.boxes().find(({ region }) => region.id === this.selected)?.region;
    if (region && this.doc().ok) {
      // click on one of the selected box's points: remove it
      const near = this.pointAt(region, x, y, v);
      if (near >= 0) return this.updateDetail(region, (d) => d.points.splice(near, 1));
      // Shift-click: + (this is hers), Alt-click: - (this isn't)
      if ((e.shiftKey || e.altKey) && x >= 0 && y >= 0 && x <= v.W && y <= v.H) {
        e.preventDefault();
        const label = e.altKey ? 0 : 1;
        const r3 = (n) => Math.round(n * 1000) / 1000;
        return this.updateDetail(region, (d) => d.points.push({ x: r3(x / v.W), y: r3(y / v.H), label }));
      }
    }
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
