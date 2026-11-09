// Detail settings of "MOSS-Image Character Masks", and its mask overlay. Pure, so it is tested headlessly.
//
// The node's hidden `detail` widget holds {"version": 1, "boxes": {<box id>: {...}}}. A box with no entry
// uses the defaults for its label; an entry is written in full the first time that box is edited. Unknown
// fields are kept. Mirrors moss_regional_lora/character_masks.py (PARTS, default_detail, resolve_detail).

export const DETAIL_VERSION = 1;

// Same order, prompts' hints and colours as character_masks.py PARTS. Bit i of the label map = PARTS[i].
export const PARTS = [
  { name: "face", label: "face", hint: "detailed face", color: "#ff9f1c" },
  { name: "eyes", label: "eyes", hint: "detailed eyes", color: "#2ec4ff" },
  { name: "left_hand", label: "left hand", hint: "detailed hand", color: "#3ddc84" },   // as seen in the image
  { name: "right_hand", label: "right hand", hint: "detailed hand", color: "#ff5ca8" },
  { name: "hair", label: "hair", hint: "detailed hair", color: "#b77dff" },
  { name: "skin", label: "skin", hint: "detailed skin", color: "#ffe14d" },
];
// settings saved before hands were split into left and right apply to both
const LEGACY_PARTS = { left_hand: "hands", right_hand: "hands" };
export const PART_NAMES = PARTS.map((p) => p.name);
// where parts overlap (eyes inside a face inside skin), the overlay shows the finest
const FINEST_FIRST = ["eyes", "left_hand", "right_hand", "face", "hair", "skin"].map((n) => PART_NAMES.indexOf(n));
const DEFAULT_ON = ["face", "eyes"];

export function defaultDetail(label) {
  const parts = {};
  for (const n of PART_NAMES) parts[n] = { on: DEFAULT_ON.includes(n), prompt: "", denoise: null, restore: null, whole_crop: false };
  return { enabled: label === "character", prompt: "", use_box_loras: true, loras: [], parts, paint: {} };
}

// What a box's paint can target: its whole character (ownership) or one of its parts. Each target holds
// an "add" and a "sub" layer (web/paint.js storage).
export const PAINT_TARGETS = ["character", ...PARTS.map((p) => p.name)];

function readPaint(value) {
  const out = {};
  if (!value || typeof value !== "object") return out;
  for (const t of PAINT_TARGETS) {
    const v = value[t];
    if (!v || typeof v !== "object") continue;
    const layer = (x) => (x && typeof x === "object" && Array.isArray(x.runs) ? x : null);
    if (layer(v.add) || layer(v.sub)) out[t] = { add: layer(v.add), sub: layer(v.sub) };
  }
  return out;
}

/** The Protect layer (no part of any box may cover it), or null. */
export function protectOf(doc) {
  const p = doc?.protect;
  return p && typeof p === "object" && Array.isArray(p.runs) ? p : null;
}

/** A new document with the Protect layer set (null removes it). */
export function setProtect(doc, layer) {
  const next = { ...doc };
  if (layer) next.protect = layer;
  else delete next.protect;
  return next;
}

export function emptyDoc() {
  return { version: DETAIL_VERSION, boxes: {} };
}

/** {ok, doc} or {ok: false, error}. Empty text is an empty document. */
export function parseDetail(text) {
  if (text == null || !String(text).trim()) return { ok: true, doc: emptyDoc() };
  let data;
  try {
    data = JSON.parse(text);
  } catch (e) {
    return { ok: false, error: `detail settings are not valid JSON: ${e.message}` };
  }
  if (!data || typeof data !== "object" || Array.isArray(data)) return { ok: false, error: "detail settings must be a JSON object" };
  if (data.boxes != null && (typeof data.boxes !== "object" || Array.isArray(data.boxes))) {
    return { ok: false, error: "detail settings: 'boxes' must be an object keyed by box id" };
  }
  return { ok: true, doc: { ...data, version: data.version ?? DETAIL_VERSION, boxes: { ...(data.boxes || {}) } } };
}

const isNum = (v) => typeof v === "number" && Number.isFinite(v);

/** A box's settings: what is stored for it over the defaults for its label (bad values are ignored). */
export function detailFor(doc, region) {
  const base = defaultDetail(region?.label || "object");
  const raw = doc?.boxes?.[region?.id];
  if (!raw || typeof raw !== "object") return base;
  const parts = {};
  const unit = (v) => (isNum(v) ? Math.min(1, Math.max(0, v)) : null);
  for (const n of PART_NAMES) {
    let p = raw.parts?.[n];
    if (!p || typeof p !== "object") p = raw.parts?.[LEGACY_PARTS[n]];
    parts[n] = p && typeof p === "object"
      ? {
          on: typeof p.on === "boolean" ? p.on : base.parts[n].on,
          prompt: typeof p.prompt === "string" ? p.prompt : "",
          denoise: unit(p.denoise),
          restore: unit(p.restore),
          whole_crop: typeof p.whole_crop === "boolean" ? p.whole_crop : false,
        }
      : base.parts[n];
  }
  const loras = Array.isArray(raw.loras)
    ? raw.loras.filter((e) => e && typeof e === "object").map((e) => ({ lora: typeof e.lora === "string" ? e.lora : "", weight: isNum(e.weight) ? e.weight : 1 }))
    : [];
  return {
    enabled: typeof raw.enabled === "boolean" ? raw.enabled : base.enabled,
    prompt: typeof raw.prompt === "string" ? raw.prompt : "",
    use_box_loras: typeof raw.use_box_loras === "boolean" ? raw.use_box_loras : true,
    loras,
    parts,
    paint: readPaint(raw.paint),
  };
}

/** A new document with `detail` stored for box `id` (the input is not changed). Unknown fields survive. */
export function setDetail(doc, id, detail) {
  const prev = doc.boxes?.[id] && typeof doc.boxes[id] === "object" ? doc.boxes[id] : {};
  const parts = { ...(prev.parts || {}) };
  for (const n of PART_NAMES) parts[n] = { ...(parts[n] || {}), ...detail.parts[n] };
  return {
    ...doc,
    boxes: {
      ...doc.boxes,
      [id]: (() => {
        const { points: _old, ...rest } = prev; // points were replaced by paint
        const paint = {};
        for (const [t, v] of Object.entries(detail.paint || {})) if (v && (v.add || v.sub)) paint[t] = { add: v.add || null, sub: v.sub || null };
        return {
          ...rest, enabled: detail.enabled, prompt: detail.prompt, use_box_loras: detail.use_box_loras,
          loras: detail.loras.map((e) => ({ ...e })), parts, paint,
        };
      })(),
    },
  };
}

export function serializeDetail(doc) {
  return JSON.stringify(doc);
}

function rgb(hex) {
  const m = /^#?([0-9a-f]{6})$/i.exec(hex || "");
  const n = m ? parseInt(m[1], 16) : 0xcccccc;
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

/**
 * RGBA overlay from the node's label map (R = box number, 1-based; G = part bits; B = 255 inside the
 * outline). The selected box (0-based, -1 for none) is drawn strongly; other boxes faintly.
 */
export function overlayPixels(labels, boxColors, selected) {
  const out = new Uint8ClampedArray(labels.length);
  const partRgb = PARTS.map((p) => rgb(p.color));
  const boxRgb = boxColors.map(rgb);
  for (let i = 0; i < labels.length; i += 4) {
    const box = labels[i] - 1;
    if (box < 0) continue;
    const sel = selected < 0 || box === selected;
    const bits = labels[i + 1];
    let c = null;
    let a = 0;
    if (bits) {
      for (const k of FINEST_FIRST) if (bits & (1 << k)) { c = partRgb[k]; break; }   // eyes over face over skin
      a = sel ? 170 : 45;
    } else if (labels[i + 2]) {
      c = boxRgb[box] || [204, 204, 204];
      a = sel ? 70 : 25;
    }
    if (!c) continue;
    out[i] = c[0];
    out[i + 1] = c[1];
    out[i + 2] = c[2];
    out[i + 3] = a;
  }
  return out;
}

/** The part shown at label-map pixel ``i`` (index into the RGBA data): {box, part} (0-based box, PARTS
 * index; part -1 = outline only), or null outside every character. Same precedence as the overlay. */
export function partAt(labels, i) {
  const box = labels[i] - 1;
  if (box < 0) return null;
  const bits = labels[i + 1];
  for (const k of FINEST_FIRST) if (bits & (1 << k)) return { box, part: k };
  return labels[i + 2] ? { box, part: -1 } : null;
}

// the parts tagged on the overlay, so a pair can be told apart where they touch
export const TAGGED_PARTS = { left_hand: "L", right_hand: "R" };

/**
 * Where to put the tags: one per box and tagged part found in the label map (w x h RGBA), at the part's
 * pixel nearest its centre (the centre itself can fall outside a curled hand). [{box, part, text, x, y}],
 * x and y in label-map pixels (pixel centres).
 */
export function partTags(labels, w, h) {
  const tagged = Object.keys(TAGGED_PARTS).map((n) => PART_NAMES.indexOf(n));
  const acc = new Map(); // "box:part" -> {box, part, sx, sy, n, pts}
  for (let y = 0, i = 0; y < h; y++) {
    for (let x = 0; x < w; x++, i += 4) {
      const box = labels[i] - 1;
      if (box < 0 || !labels[i + 1]) continue;
      for (const k of tagged) {
        if (!(labels[i + 1] & (1 << k))) continue;
        const key = `${box}:${k}`;
        let a = acc.get(key);
        if (!a) acc.set(key, (a = { box, part: k, sx: 0, sy: 0, n: 0, pts: [] }));
        a.sx += x;
        a.sy += y;
        a.n++;
        a.pts.push(x, y);
      }
    }
  }
  const out = [];
  for (const a of acc.values()) {
    const cx = a.sx / a.n, cy = a.sy / a.n;
    let best = 0, bd = Infinity;
    for (let j = 0; j < a.pts.length; j += 2) {
      const d = (a.pts[j] - cx) ** 2 + (a.pts[j + 1] - cy) ** 2;
      if (d < bd) [bd, best] = [d, j];
    }
    out.push({ box: a.box, part: a.part, text: TAGGED_PARTS[PART_NAMES[a.part]], x: a.pts[best] + 0.5, y: a.pts[best + 1] + 0.5 });
  }
  return out.sort((p, q) => p.box - q.box || p.part - q.part);
}
