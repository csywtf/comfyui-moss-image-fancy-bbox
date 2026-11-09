// Paint for the Character Masks editor: bitmaps, run-length storage, brush strokes and the magic wand.
// Pure (no DOM), tested headlessly. Mirrored for decoding by moss_regional_lora/character_masks.py.
//
// A paint layer is a w x h bitmap at the image's resolution (1 = painted), stored as
//   {"w": W, "h": H, "runs": [n0, n1, n2, ...]}
// alternating run lengths over the pixels in row-major order, starting with unpainted (n0 may be 0).

/** An empty w x h bitmap. */
export function blank(w, h) {
  return { w, h, bits: new Uint8Array(w * h) };
}

export function encode(layer) {
  const { w, h, bits } = layer;
  const runs = [];
  let cur = 0;
  let n = 0;
  for (let i = 0; i < bits.length; i++) {
    const b = bits[i] ? 1 : 0;
    if (b === cur) n++;
    else {
      runs.push(n);
      cur = b;
      n = 1;
    }
  }
  runs.push(n);
  return { w, h, runs };
}

/** A stored layer as a bitmap of size (w, h) (nearest-neighbour scaled if it was stored at another size). */
export function decode(stored, w, h) {
  const out = blank(w ?? stored?.w ?? 0, h ?? stored?.h ?? 0);
  if (!stored || !Array.isArray(stored.runs) || !(stored.w > 0) || !(stored.h > 0)) return out;
  const src = new Uint8Array(stored.w * stored.h);
  let i = 0;
  let cur = 0;
  for (const n of stored.runs) {
    const end = Math.min(src.length, i + Math.max(0, n | 0));
    if (cur) src.fill(1, i, end);
    i = end;
    cur ^= 1;
  }
  if (stored.w === out.w && stored.h === out.h) {
    out.bits.set(src);
    return out;
  }
  for (let y = 0; y < out.h; y++) {
    const sy = Math.min(stored.h - 1, Math.floor((y * stored.h) / out.h));
    for (let x = 0; x < out.w; x++) out.bits[y * out.w + x] = src[sy * stored.w + Math.min(stored.w - 1, Math.floor((x * stored.w) / out.w))];
  }
  return out;
}

export function isEmpty(layer) {
  return !layer || !layer.bits.some((b) => b);
}

/** Set (value 1) or clear (0) a filled circle of radius r at (cx, cy). */
export function stamp(layer, cx, cy, r, value = 1) {
  const { w, h, bits } = layer;
  const r2 = r * r;
  const y0 = Math.max(0, Math.floor(cy - r)), y1 = Math.min(h - 1, Math.ceil(cy + r));
  const x0 = Math.max(0, Math.floor(cx - r)), x1 = Math.min(w - 1, Math.ceil(cx + r));
  for (let y = y0; y <= y1; y++) {
    const dy = y + 0.5 - cy;
    for (let x = x0; x <= x1; x++) {
      const dx = x + 0.5 - cx;
      if (dx * dx + dy * dy <= r2) bits[y * w + x] = value;
    }
  }
}

/** The pixel rectangle a stroke segment can touch: {x0, y0, x1, y1}, exclusive max, clipped to the layer. */
export function strokeBounds(layer, x0, y0, x1, y1, r) {
  return {
    x0: Math.max(0, Math.floor(Math.min(x0, x1) - r - 1)),
    y0: Math.max(0, Math.floor(Math.min(y0, y1) - r - 1)),
    x1: Math.min(layer.w, Math.ceil(Math.max(x0, x1) + r + 1)),
    y1: Math.min(layer.h, Math.ceil(Math.max(y0, y1) + r + 1)),
  };
}

/** A brush stroke segment: circles every half radius from (x0, y0) to (x1, y1). */
export function strokeLine(layer, x0, y0, x1, y1, r, value = 1) {
  const len = Math.hypot(x1 - x0, y1 - y0);
  const steps = Math.max(1, Math.ceil(len / Math.max(0.5, r / 2)));
  for (let i = 0; i <= steps; i++) stamp(layer, x0 + ((x1 - x0) * i) / steps, y0 + ((y1 - y0) * i) / steps, r, value);
}

/**
 * Magic wand on RGBA pixels (w x h): pixels whose every channel is within `tolerance` (0-255) of the
 * clicked pixel. `connected`: only those joined to the click through such pixels (4-neighbours);
 * otherwise anywhere, limited to `within` ({x0, y0, x1, y1}) if given. Returns a w x h bitmap.
 */
export function wand(rgba, w, h, x, y, tolerance, connected = true, within = null) {
  const out = blank(w, h);
  x = Math.floor(x);
  y = Math.floor(y);
  if (x < 0 || y < 0 || x >= w || y >= h) return out;
  const i0 = (y * w + x) * 4;
  const r0 = rgba[i0], g0 = rgba[i0 + 1], b0 = rgba[i0 + 2];
  const t = Math.max(0, tolerance);
  const near = (p) => {
    const i = p * 4;
    return Math.abs(rgba[i] - r0) <= t && Math.abs(rgba[i + 1] - g0) <= t && Math.abs(rgba[i + 2] - b0) <= t;
  };
  if (!connected) {
    const bx0 = within ? Math.max(0, Math.floor(within.x0)) : 0, by0 = within ? Math.max(0, Math.floor(within.y0)) : 0;
    const bx1 = within ? Math.min(w, Math.ceil(within.x1)) : w, by1 = within ? Math.min(h, Math.ceil(within.y1)) : h;
    for (let yy = by0; yy < by1; yy++) for (let xx = bx0; xx < bx1; xx++) if (near(yy * w + xx)) out.bits[yy * w + xx] = 1;
    return out;
  }
  // flood fill with a preallocated stack (no per-pixel allocation: fast on large images)
  const stack = new Int32Array(w * h);
  let top = 0;
  stack[top++] = y * w + x;
  out.bits[y * w + x] = 1;
  const visit = (q) => {
    if (!out.bits[q] && near(q)) {
      out.bits[q] = 1;
      stack[top++] = q;
    }
  };
  while (top) {
    const p = stack[--top];
    const px = p % w;
    if (px > 0) visit(p - 1);
    if (px < w - 1) visit(p + 1);
    if (p >= w) visit(p - w);
    if (p < w * (h - 1)) visit(p + w);
  }
  return out;
}

/** Combine `src` into `dst` (same size): value 1 sets where src is set, 0 clears there. */
export function apply(dst, src, value = 1) {
  for (let i = 0; i < dst.bits.length; i++) if (src.bits[i]) dst.bits[i] = value;
}
