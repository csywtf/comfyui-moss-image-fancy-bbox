// Values the region editor reads from the nodes wired into its width_input / height_input / caption_input
// sockets. Pure (no DOM, no ComfyUI imports), so it is tested headlessly (tests/web/sources.test.mjs).

// ComfyUI's Resolution Selector (comfy_extras/nodes_resolution.py), so the canvas follows its widgets live.
const ASPECT_RATIOS = {
  "1:1 (Square)": [1, 1],
  "2:3 (Portrait Photo)": [2, 3],
  "3:2 (Photo)": [3, 2],
  "3:4 (Portrait Standard)": [3, 4],
  "4:3 (Standard)": [4, 3],
  "9:16 (Portrait Widescreen)": [9, 16],
  "16:9 (Widescreen)": [16, 9],
  "21:9 (Ultrawide)": [21, 9],
};

/** Python's round(): halves go to the even neighbour. */
function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

/** [width, height] a Resolution Selector node outputs, computed exactly as its execute(); null if unreadable. */
export function resolutionSelectorSize(src) {
  const val = (name) => src?.widgets?.find?.((w) => w?.name === name)?.value;
  const ratio = ASPECT_RATIOS[val("aspect_ratio")];
  const mp = val("megapixels");
  const multiple = val("multiple") ?? 8; // the node's default, if its advanced widget isn't there
  if (!ratio || typeof mp !== "number" || !Number.isFinite(mp) || typeof multiple !== "number" || !(multiple > 0)) return null;
  const scale = Math.sqrt((mp * 1024 * 1024) / (ratio[0] * ratio[1]));
  return [roundHalfEven((ratio[0] * scale) / multiple) * multiple, roundHalfEven((ratio[1] * scale) / multiple) * multiple];
}

/** The number output `slot` of a node gives without running: a primitive (PrimitiveNode, or a single-output
 * node with a "value" widget such as Int) or a Resolution Selector; null when only a run knows it. */
export function liveNumber(src, slot = 0) {
  if (!src) return null;
  if (src.type === "ResolutionSelector") return resolutionSelectorSize(src)?.[slot] ?? null;
  const v = primitiveValue(src);
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/** The text a primitive (PrimitiveNode, or a single-output node with a "value" widget such as String) sends;
 * null when only a run knows it. */
export function liveText(src) {
  const v = src ? primitiveValue(src) : null;
  return typeof v === "string" ? v : null;
}

function primitiveValue(src) {
  if (src.type === "PrimitiveNode") return src.widgets?.[0]?.value;
  return (src.outputs || []).length === 1 ? src.widgets?.find?.((x) => x?.name === "value")?.value : undefined;
}

/** A size from width_input / height_input as the node uses it: rounded down to whole 16 px cells, at least one. */
export function canvasSide(v) {
  return Math.max(16, Math.floor(v / 16) * 16);
}
