// Detail settings of "MOSS-Image Character Masks", and its mask overlay. Pure, so it is tested headlessly.
//
// The node's hidden `detail` widget holds {"version": 1, "boxes": {<box id>: {...}}}. A box with no entry
// uses the defaults for its label; an entry is written in full the first time that box is edited. Unknown
// fields are kept. Mirrors moss_regional_lora/character_masks.py (PARTS, default_detail, resolve_detail).

export const DETAIL_VERSION = 1;

// Same order, prompts' hints and colours as character_masks.py PARTS. Bit i of the label map = PARTS[i].
export const PARTS = [
  { name: "face", hint: "detailed face", color: "#ff9f1c" },
  { name: "eyes", hint: "detailed eyes", color: "#2ec4ff" },
  { name: "hands", hint: "detailed hands", color: "#3ddc84" },
  { name: "hair", hint: "detailed hair", color: "#b77dff" },
  { name: "skin", hint: "detailed skin", color: "#ffe14d" },
];
export const PART_NAMES = PARTS.map((p) => p.name);
// where parts overlap (eyes inside a face inside skin), the overlay shows the finest
const FINEST_FIRST = ["eyes", "hands", "face", "hair", "skin"].map((n) => PART_NAMES.indexOf(n));
const DEFAULT_ON = ["face", "eyes"];

export function defaultDetail(label) {
  const parts = {};
  for (const n of PART_NAMES) parts[n] = { on: DEFAULT_ON.includes(n), prompt: "", denoise: null };
  return { enabled: label === "character", prompt: "", use_box_loras: true, loras: [], parts };
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
  for (const n of PART_NAMES) {
    const p = raw.parts?.[n];
    parts[n] = p && typeof p === "object"
      ? {
          on: typeof p.on === "boolean" ? p.on : base.parts[n].on,
          prompt: typeof p.prompt === "string" ? p.prompt : "",
          denoise: isNum(p.denoise) ? Math.min(1, Math.max(0, p.denoise)) : null,
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
      [id]: { ...prev, enabled: detail.enabled, prompt: detail.prompt, use_box_loras: detail.use_box_loras, loras: detail.loras.map((e) => ({ ...e })), parts },
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
