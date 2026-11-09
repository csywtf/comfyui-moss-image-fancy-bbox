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

  const canvasRect = frameRect(canvas);
  const layout = (e) => {
    const { rect: r, cw, ch } = canvasRect();
    const sx = r.width ? cw / r.width : 1;
    const sy = r.height ? ch / r.height : 1;
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
      host.focus?.();
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
  const typing = (e) => /^(input|textarea|select)$/i.test(e.target?.tagName || "") && !isKeySink(e.target);
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

// ---------------------------------------------------------------------------------------------------
// Keyboard focus that ComfyUI leaves alone.
//
// ComfyUI's frontend handles some input before an editor sees it: its change tracker runs the workflow
// undo for Ctrl+Z / Ctrl+Y from a window capture listener unless a text field (input, textarea,
// contenteditable) has focus, and its node container forwards the wheel to the graph unless the pointer is
// over a [data-capture-wheel="true"] element that holds the focus. So each editor keeps a hidden,
// read-only textarea as its focus target and marks its root data-capture-wheel: with focus there, the
// editor's own undo and wheel zoom are the only ones that run. It takes the focus when the pointer enters
// or presses inside the editor (except on the editor's own fields) and gives it back when the pointer
// leaves, so graph shortcuts work again outside the editor.

const KEY_SINK = "mrl-keysink";

export function isKeySink(el) {
  return !!el?.classList?.contains(KEY_SINK);
}

function isTextEntry(el) {
  return !!el && !isKeySink(el) && (/^(input|textarea|select)$/i.test(el.tagName || "") || el.isContentEditable);
}

/** Install the focus target in ``root``; returns {el, focus()}. */
export function installKeySink(root) {
  root.setAttribute("data-capture-wheel", "true");
  const sink = document.createElement("textarea");
  sink.className = KEY_SINK;
  sink.readOnly = true;
  sink.tabIndex = -1;
  sink.setAttribute("aria-hidden", "true");
  sink.style.cssText = "position:absolute;left:0;top:0;width:1px;height:1px;opacity:0;pointer-events:none;resize:none;border:0;padding:0;overflow:hidden";
  root.appendChild(sink);
  const focus = () => {
    if (document.activeElement !== sink) sink.focus({ preventScroll: true });
  };
  // the pointer entering takes the focus, unless someone is typing elsewhere
  root.addEventListener("pointerenter", () => {
    const a = document.activeElement;
    if (!a || a === document.body || root.contains(a) || a.tagName === "CANVAS") {
      if (!isTextEntry(a)) focus();
    }
  });
  root.addEventListener("pointerleave", () => {
    if (document.activeElement === sink) sink.blur();
  });
  // a press anywhere in the editor except its own fields (buttons, panels, the canvas) keeps the focus here
  root.addEventListener("pointerdown", (e) => {
    if (!isTextEntry(e.target)) requestAnimationFrame(focus);
  });
  root.addEventListener("focusin", (e) => {
    if (e.target === root) focus();
  });
  return { el: sink, focus };
}

// ---------------------------------------------------------------------------------------------------
// Layout reads without forced layouts.
//
// Reading an element's size or position makes the browser lay out the page first if anything changed
// since the last frame. Other ComfyUI extensions change the page all the time (a status bar that animates
// every second, say), so reading on every pointer move or redraw can mean a full layout each time: the
// brush stutters. The editors read their canvas size only when it is resized, and its on-screen rect at
// most once per frame.

/** The element's client size, kept up to date by a ResizeObserver; ``onResize`` runs after each change. */
export function trackSize(el, onResize) {
  let size = null;
  const read = () => (size = { w: el.clientWidth, h: el.clientHeight });
  const ro = new ResizeObserver(() => {
    read(); // the layout is fresh here: no extra cost
    onResize?.();
  });
  ro.observe(el);
  return {
    ro,
    get w() {
      return (size || read()).w;
    },
    get h() {
      return (size || read()).h;
    },
  };
}

/** A function returning {rect, cw, ch} for ``el`` (screen rect and layout size), read once per frame. */
export function frameRect(el) {
  let cached = null;
  return () => {
    if (!cached) {
      cached = { rect: el.getBoundingClientRect(), cw: el.clientWidth, ch: el.clientHeight };
      requestAnimationFrame(() => (cached = null));
    }
    return cached;
  };
}
