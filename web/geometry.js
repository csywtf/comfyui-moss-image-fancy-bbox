// Pure geometry for the region editor (spec 8.5, 8.10a). No DOM references.
//
// Two coordinate spaces:
//   canvas space   {x0, y0, x1, y1} in image pixels, origin top-left, y down.
//   centred space  [y_min, x_min, y_max, x_max] integers in pixels, origin at the canvas centre
//                  (width // 2, height // 2), y down. This is the serialised bbox (FR-2) and the same
//                  convention as ComfyUI's MOSSImageLayoutPrompt and moss_regional_lora/geometry.py.
//
// centeredToCanvas / canvasToCentered are the ONLY conversion between the two (8.5c).

export const HANDLES = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];

export function canvasCenter(width, height) {
  return { cx: Math.floor(width / 2), cy: Math.floor(height / 2) };
}

/** [y_min, x_min, y_max, x_max] centred -> {x0, y0, x1, y1} top-left canvas pixels. */
export function centeredToCanvas(bbox, width, height) {
  const { cx, cy } = canvasCenter(width, height);
  const [yMin, xMin, yMax, xMax] = bbox;
  return { x0: xMin + cx, y0: yMin + cy, x1: xMax + cx, y1: yMax + cy };
}

/** {x0, y0, x1, y1} top-left canvas pixels -> integer [y_min, x_min, y_max, x_max] centred. */
export function canvasToCentered(rect, width, height) {
  const { cx, cy } = canvasCenter(width, height);
  const r = normalizeRect(rect);
  return [
    Math.round(r.y0 - cy),
    Math.round(r.x0 - cx),
    Math.round(r.y1 - cy),
    Math.round(r.x1 - cx),
  ].map((v) => (Object.is(v, -0) ? 0 : v));
}

export function normalizeRect(r) {
  return {
    x0: Math.min(r.x0, r.x1),
    y0: Math.min(r.y0, r.y1),
    x1: Math.max(r.x0, r.x1),
    y1: Math.max(r.y0, r.y1),
  };
}

export function rectSize(r) {
  return { w: r.x1 - r.x0, h: r.y1 - r.y0 };
}

export function isValidBbox(b) {
  return (
    Array.isArray(b) &&
    b.length === 4 &&
    b.every((v) => typeof v === "number" && Number.isFinite(v)) &&
    b[2] >= b[0] &&
    b[3] >= b[1]
  );
}

// ------------------------------------------------------------------ view fitting
/** Scale a width x height canvas into an available box, preserving aspect ratio (8.3a). */
export function fitCanvas(width, height, availW, availH) {
  if (width <= 0 || height <= 0 || availW <= 0 || availH <= 0) return { scale: 0, w: 0, h: 0 };
  const scale = Math.min(availW / width, availH / height);
  return { scale, w: width * scale, h: height * scale };
}

/** Screen (CSS px relative to the canvas element) -> image pixels. */
export function screenToImage(p, scale) {
  return { x: p.x / scale, y: p.y / scale };
}

export function imageToScreen(p, scale) {
  return { x: p.x * scale, y: p.y * scale };
}

// ------------------------------------------------------------------ grid and snapping
export function gridStep(width, height, divisions) {
  const d = Math.max(1, Math.floor(divisions || 1));
  return { sx: width / d, sy: height / d };
}

export function snapValue(v, step) {
  if (!step || step <= 0) return v;
  return Math.round(v / step) * step;
}

export function snapPoint(p, grid) {
  return { x: snapValue(p.x, grid.sx), y: snapValue(p.y, grid.sy) };
}

// ------------------------------------------------------------------ bounds
export function outOfBounds(rect, width, height) {
  const r = normalizeRect(rect);
  return r.x0 < 0 || r.y0 < 0 || r.x1 > width || r.y1 > height;
}

/** Intersection with the canvas, for drawing only (boxes are clipped, never clamped: 8.3e). */
export function clipRect(rect, width, height) {
  const r = normalizeRect(rect);
  const c = { x0: Math.max(0, r.x0), y0: Math.max(0, r.y0), x1: Math.min(width, r.x1), y1: Math.min(height, r.y1) };
  return c.x1 > c.x0 && c.y1 > c.y0 ? c : null;
}

/** Clamp a point into the canvas (used to keep a creation drag's start on the canvas). */
export function clampPoint(p, width, height) {
  return { x: Math.min(Math.max(p.x, 0), width), y: Math.min(Math.max(p.y, 0), height) };
}

// ------------------------------------------------------------------ hit testing
export function pointInRect(p, rect) {
  const r = normalizeRect(rect);
  return p.x >= r.x0 && p.x <= r.x1 && p.y >= r.y0 && p.y <= r.y1;
}

/**
 * Index of the topmost rect containing p. `rects` is in draw order (index 0 drawn first, so the
 * last one is on top). Entries that are null/undefined (hidden) are skipped. -1 if none.
 */
export function hitTest(rects, p) {
  for (let i = rects.length - 1; i >= 0; i--) {
    if (rects[i] && pointInRect(p, rects[i])) return i;
  }
  return -1;
}

export function handlePositions(rect) {
  const r = normalizeRect(rect);
  const mx = (r.x0 + r.x1) / 2;
  const my = (r.y0 + r.y1) / 2;
  return {
    nw: { x: r.x0, y: r.y0 },
    n: { x: mx, y: r.y0 },
    ne: { x: r.x1, y: r.y0 },
    e: { x: r.x1, y: my },
    se: { x: r.x1, y: r.y1 },
    s: { x: mx, y: r.y1 },
    sw: { x: r.x0, y: r.y1 },
    w: { x: r.x0, y: my },
  };
}

/** Handle under p within `tol` image pixels (corners win over edges), or null. */
export function handleAt(rect, p, tol) {
  const pos = handlePositions(rect);
  let best = null;
  let bestD = Infinity;
  for (const h of ["nw", "ne", "se", "sw", "n", "e", "s", "w"]) {
    const d = Math.max(Math.abs(pos[h].x - p.x), Math.abs(pos[h].y - p.y));
    if (d <= tol && d < bestD - 1e-9) {
      best = h;
      bestD = d;
    }
  }
  return best;
}

// ------------------------------------------------------------------ drag operations
export function moveRect(rect, dx, dy) {
  return { x0: rect.x0 + dx, y0: rect.y0 + dy, x1: rect.x1 + dx, y1: rect.y1 + dy };
}

/** Move so the rect's top-left lands on the grid (snap during a body drag). */
export function snapMove(rect, dx, dy, grid) {
  const moved = moveRect(rect, dx, dy);
  const sx = snapValue(moved.x0, grid.sx) - moved.x0;
  const sy = snapValue(moved.y0, grid.sy) - moved.y0;
  return moveRect(moved, sx, sy);
}

const AFFECTS = {
  nw: { x: "x0", y: "y0" },
  n: { x: null, y: "y0" },
  ne: { x: "x1", y: "y0" },
  e: { x: "x1", y: null },
  se: { x: "x1", y: "y1" },
  s: { x: null, y: "y1" },
  sw: { x: "x0", y: "y1" },
  w: { x: "x0", y: null },
};
const OPPOSITE = { x0: "x1", x1: "x0", y0: "y1", y1: "y0" };

/**
 * Resize `rect` by dragging `handle` to point p.
 *   square:     keep width == height (Shift)
 *   fromCenter: keep the rect's centre fixed (Alt)
 * The result is normalised, so dragging past the opposite edge flips the box.
 */
export function resizeRect(rect, handle, p, { square = false, fromCenter = false } = {}) {
  const r = normalizeRect(rect);
  const a = AFFECTS[handle];
  if (!a) throw new Error(`unknown handle ${handle}`);
  const c = { x: (r.x0 + r.x1) / 2, y: (r.y0 + r.y1) / 2 };
  const out = { ...r };

  // signed extents measured from the anchor (opposite edge, or centre with Alt)
  const anchorX = a.x ? (fromCenter ? c.x : r[OPPOSITE[a.x]]) : null;
  const anchorY = a.y ? (fromCenter ? c.y : r[OPPOSITE[a.y]]) : null;
  let ex = a.x ? p.x - anchorX : null;
  let ey = a.y ? p.y - anchorY : null;

  if (square) {
    if (a.x && a.y) {
      const s = Math.max(Math.abs(ex), Math.abs(ey));
      ex = (ex < 0 ? -1 : 1) * s;
      ey = (ey < 0 ? -1 : 1) * s;
    } else if (a.x) {
      // edge handle: the other axis follows, centred on its current middle
      const s = Math.abs(ex) * (fromCenter ? 2 : 1);
      out.y0 = c.y - s / 2;
      out.y1 = c.y + s / 2;
    } else {
      const s = Math.abs(ey) * (fromCenter ? 2 : 1);
      out.x0 = c.x - s / 2;
      out.x1 = c.x + s / 2;
    }
  }
  if (a.x) {
    if (fromCenter) {
      out.x0 = c.x - Math.abs(ex);
      out.x1 = c.x + Math.abs(ex);
    } else {
      out[a.x] = anchorX + ex;
      out[OPPOSITE[a.x]] = anchorX;
    }
  }
  if (a.y) {
    if (fromCenter) {
      out.y0 = c.y - Math.abs(ey);
      out.y1 = c.y + Math.abs(ey);
    } else {
      out[a.y] = anchorY + ey;
      out[OPPOSITE[a.y]] = anchorY;
    }
  }
  return normalizeRect(out);
}

/** Rect from a creation drag start -> p. Shift: square. Alt: start is the centre. */
export function createRect(start, p, { square = false, fromCenter = false } = {}) {
  let dx = p.x - start.x;
  let dy = p.y - start.y;
  if (square) {
    const s = Math.max(Math.abs(dx), Math.abs(dy));
    dx = (dx < 0 ? -1 : 1) * s;
    dy = (dy < 0 ? -1 : 1) * s;
  }
  if (fromCenter) {
    return normalizeRect({ x0: start.x - dx, y0: start.y - dy, x1: start.x + dx, y1: start.y + dy });
  }
  return normalizeRect({ x0: start.x, y0: start.y, x1: start.x + dx, y1: start.y + dy });
}

/** Arrow-key nudge (8.4b): 1 px, or 10 px with Shift. */
export function nudgeBbox(bbox, key, shift) {
  const step = shift ? 10 : 1;
  const d = { ArrowUp: [-step, 0], ArrowDown: [step, 0], ArrowLeft: [0, -step], ArrowRight: [0, step] }[key];
  if (!d) return bbox;
  const [dy, dx] = d;
  return [bbox[0] + dy, bbox[1] + dx, bbox[2] + dy, bbox[3] + dx];
}

export function cursorFor(handle) {
  return (
    {
      nw: "nwse-resize",
      se: "nwse-resize",
      ne: "nesw-resize",
      sw: "nesw-resize",
      n: "ns-resize",
      s: "ns-resize",
      e: "ew-resize",
      w: "ew-resize",
    }[handle] || "default"
  );
}
