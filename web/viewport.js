// Zoom and pan for the editors' canvases. Pure maths in Viewport (tested headlessly); installPanZoom wires
// the wheel, the hand tool, Space-drag and middle-drag to it.
//
// All positions are layout px (CSS px before the ComfyUI graph's zoom scales the editor). zoom 1 = the
// image fitted to the canvas; the pan offsets the fitted image.

export const MIN_ZOOM = 0.5;
export const MAX_ZOOM = 20;
const WHEEL_STEP = 0.0015; // zoom factor per wheel delta unit: exp(-deltaY * step)
const BUTTON_STEP = 1.25;

export class Viewport {
  constructor() {
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
  }

  /** {W, H, cw, ch, scale, ox, oy}: image px -> layout px is (x * scale + ox, y * scale + oy). */
  view(W, H, cw, ch) {
    const fit = W > 0 && H > 0 ? Math.max(1e-6, Math.min(Math.max(cw - 2, 1) / W, Math.max(ch - 2, 1) / H)) : 1;
    const scale = fit * this.zoom;
    return { W, H, cw, ch, scale, fit, ox: (cw - W * scale) / 2 + this.panX, oy: (ch - H * scale) / 2 + this.panY };
  }

  /** Zoom by ``factor`` keeping the image point under layout (sx, sy) where it is. */
  zoomAt(factor, sx, sy, W, H, cw, ch) {
    const v = this.view(W, H, cw, ch);
    const ix = (sx - v.ox) / v.scale;
    const iy = (sy - v.oy) / v.scale;
    this.zoom = Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, this.zoom * factor));
    const s = v.fit * this.zoom;
    this.panX = sx - ix * s - (cw - W * s) / 2;
    this.panY = sy - iy * s - (ch - H * s) / 2;
    this.clamp(W, H, cw, ch);
  }

  /** Zoom by ``factor`` around the canvas centre (the zoom buttons). */
  zoomBy(factor, W, H, cw, ch) {
    this.zoomAt(factor, cw / 2, ch / 2, W, H, cw, ch);
  }

  panBy(dx, dy, W, H, cw, ch) {
    this.panX += dx;
    this.panY += dy;
    this.clamp(W, H, cw, ch);
  }

  /** Keep at least a quarter of the canvas showing image, so it can't be panned out of sight. */
  clamp(W, H, cw, ch) {
    const v = this.view(W, H, cw, ch);
    const limX = Math.max(0, (W * v.scale + cw) / 2 - cw / 4);
    const limY = Math.max(0, (H * v.scale + ch) / 2 - ch / 4);
    this.panX = Math.min(limX, Math.max(-limX, this.panX));
    this.panY = Math.min(limY, Math.max(-limY, this.panY));
  }

  reset() {
    this.zoom = 1;
    this.panX = 0;
    this.panY = 0;
  }

  get fitted() {
    return this.zoom === 1 && this.panX === 0 && this.panY === 0;
  }
}

export const zoomInFactor = BUTTON_STEP;
export const zoomOutFactor = 1 / BUTTON_STEP;

/**
 * Wire zoom and pan to an editor's canvas.
 * host: {canvas, root, viewport, size() -> {W, H, cw, ch}, handTool() -> bool, redraw(), cursor(c)}
 * The wheel zooms around the pointer; a drag with the hand tool, with Space held, or with the middle
 * button pans. Pan drags are taken before the editor's own pointer handlers (capture phase).
 */
export function installPanZoom(host) {
  const { canvas, root, viewport } = host;
  let space = false;
  let drag = null;

  const layout = (e) => {
    const r = canvas.getBoundingClientRect();
    const sx = r.width ? canvas.clientWidth / r.width : 1;
    const sy = r.height ? canvas.clientHeight / r.height : 1;
    return { x: (e.clientX - r.left) * sx, y: (e.clientY - r.top) * sy };
  };
  const panning = (e) => e.button === 1 || (e.button === 0 && (space || host.handTool()));

  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault(); // zoom the editor, not the ComfyUI graph
      e.stopPropagation();
      const { W, H, cw, ch } = host.size();
      const p = layout(e);
      const delta = e.deltaMode === 1 ? e.deltaY * 16 : e.deltaY;
      viewport.zoomAt(Math.exp(-delta * WHEEL_STEP), p.x, p.y, W, H, cw, ch);
      host.redraw();
    },
    { passive: false },
  );
  canvas.addEventListener(
    "pointerdown",
    (e) => {
      if (!panning(e)) return;
      e.preventDefault();
      e.stopImmediatePropagation(); // the editor never sees a pan as a click
      canvas.focus?.();
      drag = { ...layout(e), id: e.pointerId };
      canvas.setPointerCapture?.(e.pointerId);
      host.cursor("grabbing");
    },
    { capture: true },
  );
  canvas.addEventListener(
    "pointermove",
    (e) => {
      if (!drag) {
        if (space || host.handTool()) {
          e.stopImmediatePropagation();
          host.cursor("grab");
        }
        return;
      }
      e.stopImmediatePropagation();
      const p = layout(e);
      const { W, H, cw, ch } = host.size();
      viewport.panBy(p.x - drag.x, p.y - drag.y, W, H, cw, ch);
      drag.x = p.x;
      drag.y = p.y;
      host.redraw();
    },
    { capture: true },
  );
  const end = (e) => {
    if (!drag) return;
    e.stopImmediatePropagation();
    try {
      canvas.releasePointerCapture?.(drag.id);
    } catch {}
    drag = null;
    host.cursor(space || host.handTool() ? "grab" : "");
  };
  canvas.addEventListener("pointerup", end, { capture: true });
  canvas.addEventListener("pointercancel", end, { capture: true });
  canvas.addEventListener("auxclick", (e) => e.button === 1 && e.preventDefault()); // no middle-click paste/scroll
  const typing = (e) => /^(input|textarea|select)$/i.test(e.target?.tagName || "");
  root.addEventListener("keydown", (e) => {
    if (e.key === " " && !typing(e)) {
      e.preventDefault();
      e.stopPropagation();
      if (!space) {
        space = true;
        if (!drag) host.cursor("grab");
      }
    }
  });
  root.addEventListener("keyup", (e) => {
    if (e.key === " ") {
      space = false;
      if (!drag) host.cursor(host.handTool() ? "grab" : "");
    }
  });
  return { get panning() { return !!drag; } };
}
