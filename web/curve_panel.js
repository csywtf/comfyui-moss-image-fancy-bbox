// Strength-curve panel for the selected box, and the small curve previews in the layer rows.
// The curve maths lives in curve.js; this file is only drawing and pointer handling.
//
//   click empty space        add a point (snaps to 5% steps when the editor's Snap is on; Ctrl/Cmd bypasses)
//   drag a point             move it (it stays between its neighbours)
//   drag a ◆ segment handle  bend that segment (ease in / ease out)
//   double-click / right-click a point, or Delete with the graph focused: remove it
//   Hold / Straight buttons  act on the selected segment

import * as C from "./curve.js";

const PAD = { l: 34, r: 10, t: 8, b: 18 };
const HIT_PX = 8;
const SNAP = 0.05;

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

/** Tiny read-only curve preview for a layer row. `dashed` marks a box that uses the node ramp. */
export function drawSparkline(canvas, points, dashed, color) {
  const w = 44;
  const h = 16;
  const dpr = window.devicePixelRatio || 1;
  canvas.style.width = `${w}px`;
  canvas.style.height = `${h}px`;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "rgba(128,128,128,0.15)";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = color || "#9cf";
  ctx.lineWidth = 1.5;
  if (dashed) ctx.setLineDash([2, 2]);
  ctx.beginPath();
  for (let i = 0; i <= w; i++) {
    const y = C.evaluate(points, i / w);
    const py = 1.5 + (1 - y) * (h - 3);
    if (i === 0) ctx.moveTo(i, py);
    else ctx.lineTo(i, py);
  }
  ctx.stroke();
}

export class CurvePanel {
  /**
   * host: { target(): {curve, name} being edited (curve null/absent = node ramp) or null,
   *         legacy(): node-ramp curve, readOnly(): bool, snap(): bool, commitCurve(points), color(): css colour }
   */
  constructor(host) {
    this.host = host;
    this.sel = null; // {type: "point" | "segment", index}
    this.drag = null;
    this.preview = null;
    this.hover = null;
    this.title = el("span", { class: "mrl-curve-title" });
    this.presets = Object.keys(C.PRESETS).map((name) =>
      el("button", { text: name, title: `Set the curve to "${name}"`, onclick: () => this.commit(C.PRESETS[name]()) }),
    );
    this.holdBtn = el("button", { text: "Hold", title: "Selected segment: keep its start value, then jump (hard step)", onclick: () => this.toggleHold() });
    this.straightBtn = el("button", { text: "Straight", title: "Selected segment: remove its bend", onclick: () => this.straighten() });
    this.readout = el("span", { class: "mrl-curve-readout" });
    this.canvas = el("canvas", { tabindex: "0" });
    this.wrap = el("div", { class: "mrl-curve-wrap" }, [this.canvas]);
    this.root = el("div", { class: "mrl-curve" }, [
      el("div", { class: "mrl-curve-bar" }, [this.title, ...this.presets, el("span", { class: "mrl-sep" }), this.holdBtn, this.straightBtn, this.readout]),
      this.wrap,
    ]);
    const c = this.canvas;
    c.addEventListener("pointerdown", (e) => this.onDown(e));
    c.addEventListener("pointermove", (e) => this.onMove(e));
    c.addEventListener("pointerup", (e) => this.onUp(e));
    c.addEventListener("pointercancel", () => this.cancel());
    c.addEventListener("pointerleave", () => {
      this.hover = null;
      this.renderReadout();
    });
    c.addEventListener("dblclick", (e) => this.onDouble(e));
    c.addEventListener("contextmenu", (e) => this.onContext(e));
    c.addEventListener("keydown", (e) => this.onKey(e));
    this.ro = new ResizeObserver(() => this.draw());
    this.ro.observe(this.wrap);
  }

  destroy() {
    this.ro?.disconnect();
  }

  // ------------------------------------------------------------------ data
  region() {
    return this.host.target();
  }

  /** {points, own}: the curve being shown, and whether the box has its own (false = node ramp). */
  current() {
    const r = this.region();
    if (!r) return null;
    if (this.preview) return { points: this.preview, own: true };
    if (Array.isArray(r.curve) && r.curve.length) {
      try {
        return { points: C.normalize(r.curve), own: true };
      } catch {
        /* parse errors are reported by the editor */
      }
    }
    return { points: C.normalize(this.host.legacy()), own: false };
  }

  commit(points) {
    if (this.host.readOnly() || !this.region()) return;
    this.preview = null;
    this.host.commitCurve(points);
  }

  // ------------------------------------------------------------------ geometry
  box() {
    const w = this.wrap.clientWidth;
    const h = this.wrap.clientHeight;
    return { w, h, pw: Math.max(1, w - PAD.l - PAD.r), ph: Math.max(1, h - PAD.t - PAD.b) };
  }

  toPx(p, b = this.box()) {
    return { x: PAD.l + p.x * b.pw, y: PAD.t + (1 - p.y) * b.ph };
  }

  eventPos(e) {
    const r = this.canvas.getBoundingClientRect();
    const sx = r.width ? this.canvas.clientWidth / r.width : 1;
    const sy = r.height ? this.canvas.clientHeight / r.height : 1;
    const px = (e.clientX - r.left) * sx;
    const py = (e.clientY - r.top) * sy;
    const b = this.box();
    return { px, py, x: (px - PAD.l) / b.pw, y: 1 - (py - PAD.t) / b.ph };
  }

  hit(points, pos) {
    const b = this.box();
    let best = null;
    let bestD = HIT_PX;
    points.forEach((p, i) => {
      const q = this.toPx(p, b);
      const d = Math.hypot(q.x - pos.px, q.y - pos.py);
      if (d <= bestD) {
        best = { type: "point", index: i };
        bestD = d;
      }
    });
    if (best) return best;
    for (let i = 0; i + 1 < points.length; i++) {
      const hnd = C.segmentHandle(points, i);
      if (!hnd) continue;
      const q = this.toPx(hnd, b);
      const d = Math.hypot(q.x - pos.px, q.y - pos.py);
      if (d <= bestD) {
        best = { type: "segment", index: i };
        bestD = d;
      }
    }
    return best;
  }

  snapped(pos, e) {
    let { x, y } = pos;
    if (this.host.snap() && !(e.ctrlKey || e.metaKey)) {
      x = Math.round(x / SNAP) * SNAP;
      y = Math.round(y / SNAP) * SNAP;
    }
    return { x: Math.min(1, Math.max(0, x)), y: Math.min(1, Math.max(0, y)) };
  }

  // ------------------------------------------------------------------ pointer
  onDown(e) {
    e.stopPropagation();
    this.canvas.focus();
    if (e.button !== 0 || this.host.readOnly()) return;
    const cur = this.current();
    if (!cur) return;
    const pos = this.eventPos(e);
    const hit = this.hit(cur.points, pos);
    if (hit) {
      this.sel = hit;
      this.drag = { ...hit, points0: cur.points };
    } else {
      const q = this.snapped(pos, e);
      if (pos.x < -0.02 || pos.x > 1.02 || pos.y < -0.05 || pos.y > 1.05) return;
      const { points, index } = C.addPoint(cur.points, q.x, q.y);
      this.preview = points;
      this.sel = { type: "point", index };
      this.drag = { type: "point", index, points0: points, added: true };
    }
    this.canvas.setPointerCapture(e.pointerId);
    this.render();
  }

  onMove(e) {
    const pos = this.eventPos(e);
    if (!this.drag) {
      const cur = this.current();
      this.hover = cur ? { pos, hit: this.hit(cur.points, pos) } : null;
      this.canvas.style.cursor = this.hover?.hit ? (this.hover.hit.type === "segment" ? "ns-resize" : "move") : "crosshair";
      this.renderReadout();
      return;
    }
    e.stopPropagation();
    const d = this.drag;
    if (d.type === "point") {
      const q = this.snapped(pos, e);
      this.preview = C.movePoint(d.points0, d.index, q.x, q.y);
    } else if (!d.points0[d.index].hold) {
      const bend = C.bendForMidpoint(d.points0, d.index, Math.min(1, Math.max(0, pos.y)));
      this.preview = C.setSegment(d.points0, d.index, { bend });
    }
    this.draw();
    this.renderReadout();
  }

  onUp(e) {
    if (!this.drag) return;
    e.stopPropagation();
    const d = this.drag;
    this.drag = null;
    try {
      this.canvas.releasePointerCapture(e.pointerId);
    } catch {}
    const changed = this.preview && (d.added || JSON.stringify(this.preview) !== JSON.stringify(d.points0));
    if (changed) this.commit(this.preview);
    else {
      // a click on a node-ramp box's handle should still give it its own curve only when edited
      this.preview = null;
      this.render();
    }
  }

  cancel() {
    this.drag = null;
    this.preview = null;
    this.render();
  }

  removeAt(pos) {
    const cur = this.current();
    if (!cur || this.host.readOnly()) return;
    const hit = this.hit(cur.points, pos);
    if (hit?.type === "point" && cur.points.length > 1) {
      this.sel = null;
      this.commit(C.removePoint(cur.points, hit.index));
    }
  }

  onDouble(e) {
    e.stopPropagation();
    this.removeAt(this.eventPos(e));
  }

  onContext(e) {
    e.preventDefault();
    e.stopPropagation();
    this.removeAt(this.eventPos(e));
  }

  onKey(e) {
    if ((e.key === "Delete" || e.key === "Backspace") && !(e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      e.stopPropagation(); // don't let the editor delete the box
      const cur = this.current();
      if (cur && this.sel?.type === "point" && cur.points.length > 1 && !this.host.readOnly()) {
        const idx = this.sel.index;
        this.sel = null;
        this.commit(C.removePoint(cur.points, idx));
      }
    } else if (e.key.startsWith("Arrow")) {
      e.stopPropagation(); // arrows move boxes only when the box canvas has focus
    }
  }

  toggleHold() {
    const cur = this.current();
    if (!cur || this.sel?.type !== "segment") return;
    const seg = cur.points[this.sel.index];
    this.commit(C.setSegment(cur.points, this.sel.index, { hold: !seg.hold, bend: 0 }));
  }

  straighten() {
    const cur = this.current();
    if (!cur || this.sel?.type !== "segment") return;
    this.commit(C.setSegment(cur.points, this.sel.index, { bend: 0 }));
  }

  // ------------------------------------------------------------------ rendering
  render() {
    const r = this.region();
    const cur = this.current();
    const ro = this.host.readOnly();
    if (cur && this.sel && (this.sel.index >= cur.points.length || (this.sel.type === "segment" && this.sel.index + 1 >= cur.points.length))) this.sel = null;
    this.title.textContent = r
      ? `Strength over the run${cur.own ? "" : " (node ramp; edit to give it its own curve)"}`
      : "Strength over the run";
    for (const b of this.presets) b.disabled = !r || ro;
    const segSel = !!(r && this.sel?.type === "segment" && !ro);
    this.holdBtn.disabled = !segSel;
    this.straightBtn.disabled = !segSel || !!cur?.points[this.sel?.index]?.hold || !cur?.points[this.sel?.index]?.bend;
    this.holdBtn.classList.toggle("on", !!(segSel && cur.points[this.sel.index]?.hold));
    this.renderReadout();
    this.draw();
  }

  renderReadout() {
    const cur = this.current();
    if (!cur) {
      this.readout.textContent = "";
      return;
    }
    const pct = (v) => `${Math.round(v * 100)}%`;
    const focus = this.drag ? this.sel : this.hover?.hit || this.sel;
    if (focus?.type === "point") {
      const p = cur.points[focus.index];
      if (p) this.readout.textContent = `at ${pct(p.x)} of the steps: ${pct(p.y)} strength`;
    } else if (focus?.type === "segment") {
      const p = cur.points[focus.index];
      if (p) this.readout.textContent = p.hold ? "segment: hold (hard step)" : `segment: bend ${(p.bend || 0).toFixed(2)}`;
    } else if (this.hover?.pos && this.hover.pos.x >= 0 && this.hover.pos.x <= 1) {
      this.readout.textContent = `at ${pct(this.hover.pos.x)}: ${pct(C.evaluate(cur.points, this.hover.pos.x))}`;
    } else this.readout.textContent = "click to add a point";
  }

  draw() {
    const b = this.box();
    const dpr = window.devicePixelRatio || 1;
    const c = this.canvas;
    c.style.width = `${b.w}px`;
    c.style.height = `${b.h}px`;
    c.width = Math.max(1, Math.round(b.w * dpr));
    c.height = Math.max(1, Math.round(b.h * dpr));
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, b.w, b.h);
    ctx.fillStyle = "#222";
    ctx.fillRect(PAD.l, PAD.t, b.pw, b.ph);
    ctx.font = "10px sans-serif";
    ctx.fillStyle = "rgba(255,255,255,0.6)";
    ctx.strokeStyle = "rgba(255,255,255,0.12)";
    ctx.lineWidth = 1;
    for (let k = 0; k <= 4; k++) {
      const x = PAD.l + (k / 4) * b.pw;
      const y = PAD.t + (k / 4) * b.ph;
      ctx.beginPath();
      ctx.moveTo(x, PAD.t);
      ctx.lineTo(x, PAD.t + b.ph);
      ctx.moveTo(PAD.l, y);
      ctx.lineTo(PAD.l + b.pw, y);
      ctx.stroke();
      ctx.textAlign = "center";
      ctx.fillText(`${k * 25}%`, x, b.h - 5);
      ctx.textAlign = "right";
      if (k % 2 === 0) ctx.fillText(`${100 - k * 25}%`, PAD.l - 4, y + 3);
    }
    const cur = this.current();
    if (!cur) return;
    const r = this.region();
    const color = this.host.color() || "#9cf";
    // curve
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    if (!cur.own) ctx.setLineDash([5, 4]);
    ctx.beginPath();
    const n = Math.max(2, Math.round(b.pw));
    for (let i = 0; i <= n; i++) {
      const x = i / n;
      const q = this.toPx({ x, y: C.evaluate(cur.points, x) }, b);
      if (i === 0) ctx.moveTo(q.x, q.y);
      else ctx.lineTo(q.x, q.y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    // segment handles
    for (let i = 0; i + 1 < cur.points.length; i++) {
      const hnd = C.segmentHandle(cur.points, i);
      if (!hnd) continue;
      const q = this.toPx(hnd, b);
      const selected = this.sel?.type === "segment" && this.sel.index === i;
      ctx.fillStyle = selected ? "#fff" : cur.points[i].hold ? "#c84" : "#888";
      ctx.beginPath();
      ctx.moveTo(q.x, q.y - 5);
      ctx.lineTo(q.x + 5, q.y);
      ctx.lineTo(q.x, q.y + 5);
      ctx.lineTo(q.x - 5, q.y);
      ctx.closePath();
      ctx.fill();
    }
    // points
    cur.points.forEach((p, i) => {
      const q = this.toPx(p, b);
      const selected = this.sel?.type === "point" && this.sel.index === i;
      ctx.fillStyle = selected ? "#fff" : color;
      ctx.strokeStyle = "#000";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(q.x, q.y, selected ? 5.5 : 4.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
    });
    if (this.host.readOnly()) {
      ctx.fillStyle = "rgba(120,0,0,0.35)";
      ctx.fillRect(PAD.l, PAD.t, b.pw, b.ph);
    }
  }
}
