// Region-list state: parsing, serialisation, list operations, undo history (spec 8.2, 8.6, 8.7).
// Pure: no DOM. The JSON text in the node's `regions` widget is the only source of truth; the editor
// parses it, applies an operation here, and writes the serialised result straight back.
//
// ORDER (8.6a): the `regions` array is in layer-list order. The TOP row of the list is index 0 and is
// DRAWN FIRST (underneath); the bottom row is drawn last (on top). The emitted region list is exactly
// this array order with hidden regions removed. Groups are purely organisational: they only control
// how rows are shown and moved, never what is emitted beyond the order of their members.

export const SCHEMA_VERSION = 1;

export const PALETTE = [
  "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
  "#f032e6", "#bfef45", "#469990", "#9a6324", "#800000", "#000075",
];

export const EDITOR_DEFAULTS = { grid_divisions: 3, snap: false, reference_image: "", reference_opacity: 0.5 };

// A region does one or both jobs: a LoRA inside its box (`lora`, `weight`), and an item of the MOSS-Image
// layout prompt (`label`, `caption`, `text` for text boxes, `palette`) when it has a caption or text.
export const LABELS = ["object", "character", "text"];
export const MAX_PALETTE = 5;

const REGION_DEFAULTS = {
  name: "", bbox: [0, 0, 0, 0], lora: "", weight: 1.0, visible: true, locked: false, color: "", group: null,
  label: "object", caption: "", text: "", palette: [],
};

export function emptyState() {
  return { version: SCHEMA_VERSION, regions: [], groups: [], editor: { ...EDITOR_DEFAULTS } };
}

let idCounter = 0;
export function newId(prefix, existing) {
  const taken = new Set(existing);
  let id;
  do {
    idCounter += 1;
    id = `${prefix}${Date.now().toString(36)}${idCounter.toString(36)}`;
  } while (taken.has(id));
  return id;
}

function isPlainObject(v) {
  return v !== null && typeof v === "object" && !Array.isArray(v);
}

/**
 * Parse widget text. Never throws. Returns {ok: true, state, warnings} or {ok: false, error}.
 * Missing fields get defaults; unknown fields (top-level, per region, per group, in editor) are kept
 * untouched and in their original key order, so a newer or hand-edited file round-trips (8.2c).
 */
export function parseState(text) {
  const warnings = [];
  if (text == null || String(text).trim() === "") return { ok: true, state: emptyState(), warnings };
  let data;
  try {
    data = JSON.parse(text);
  } catch (e) {
    return { ok: false, error: `Invalid JSON: ${e.message}` };
  }
  if (Array.isArray(data)) data = { version: SCHEMA_VERSION, regions: data };
  if (!isPlainObject(data)) return { ok: false, error: "Expected a JSON object with a \"regions\" list" };
  if (data.version === undefined) data = { version: SCHEMA_VERSION, ...data };
  if (!Number.isInteger(data.version)) return { ok: false, error: "\"version\" must be an integer" };
  if (data.version > SCHEMA_VERSION) warnings.push(`Saved by a newer editor (version ${data.version}); unknown fields are kept as-is`);
  if (data.regions === undefined) data.regions = [];
  if (!Array.isArray(data.regions)) return { ok: false, error: "\"regions\" must be a list" };
  if (data.groups === undefined) data.groups = [];
  if (!Array.isArray(data.groups)) return { ok: false, error: "\"groups\" must be a list" };
  if (data.editor === undefined) data.editor = {};
  if (!isPlainObject(data.editor)) return { ok: false, error: "\"editor\" must be an object" };

  const ids = new Set();
  for (let i = 0; i < data.regions.length; i++) {
    const r = data.regions[i];
    if (!isPlainObject(r)) return { ok: false, error: `Region ${i + 1} is not an object` };
    for (const [k, v] of Object.entries(REGION_DEFAULTS)) if (!(k in r)) r[k] = Array.isArray(v) ? [...v] : v;
    if (!("id" in r) || typeof r.id !== "string" || ids.has(r.id)) {
      r.id = newId("r", ids);
    }
    ids.add(r.id);
    const b = r.bbox;
    if (!Array.isArray(b) || b.length !== 4 || !b.every((v) => typeof v === "number" && Number.isFinite(v))) {
      return { ok: false, error: `Region ${i + 1} (${r.name || r.id}): bbox must be [y_min, x_min, y_max, x_max] numbers` };
    }
    if (b[2] < b[0] || b[3] < b[1]) {
      return { ok: false, error: `Region ${i + 1} (${r.name || r.id}): bbox has max < min` };
    }
    if (typeof r.weight !== "number" || !Number.isFinite(r.weight)) {
      return { ok: false, error: `Region ${i + 1} (${r.name || r.id}): weight must be a number` };
    }
    if (!LABELS.includes(r.label)) {
      warnings.push(`Region "${r.name || r.id}" has unknown label "${r.label}"; the node will reject it`);
    }
    if (!Array.isArray(r.palette)) {
      return { ok: false, error: `Region ${i + 1} (${r.name || r.id}): palette must be a list of colours` };
    }
  }
  const gids = new Set();
  for (let i = 0; i < data.groups.length; i++) {
    const g = data.groups[i];
    if (!isPlainObject(g)) return { ok: false, error: `Group ${i + 1} is not an object` };
    if (typeof g.id !== "string" || gids.has(g.id)) g.id = newId("g", gids);
    gids.add(g.id);
    if (!("name" in g)) g.name = `Group ${i + 1}`;
    if (!("collapsed" in g)) g.collapsed = false;
  }
  for (const r of data.regions) {
    if (r.group != null && !gids.has(r.group)) {
      warnings.push(`Region "${r.name || r.id}" refers to unknown group "${r.group}"; shown ungrouped`);
    }
  }
  for (const [k, v] of Object.entries(EDITOR_DEFAULTS)) if (!(k in data.editor)) data.editor[k] = v;
  return { ok: true, state: data, warnings };
}

export function serializeState(state) {
  return JSON.stringify(state, null, 2);
}

export function cloneState(state) {
  return JSON.parse(JSON.stringify(state));
}

/** True if the region becomes a layout-prompt item (it has a caption, or text to render). */
export function hasPrompt(r) {
  return !!(String(r.caption || "").trim() || (r.label === "text" && String(r.text || "").trim()));
}

/** Regions the node will apply, in order (hidden excluded: 8.6d). */
export function emittedRegions(state) {
  return state.regions.filter((r) => r.visible !== false);
}

export function regionColor(state, index) {
  const r = state.regions[index];
  return (r && typeof r.color === "string" && r.color) || PALETTE[index % PALETTE.length];
}

export function groupOf(state, region) {
  return region.group != null ? state.groups.find((g) => g.id === region.group) || null : null;
}

// ------------------------------------------------------------------ display units
/**
 * The layer list as units, in array order: {type: "region", index} for an ungrouped region and
 * {type: "group", group, indices} for a run of consecutive regions in the same group. Groups with no
 * members are appended as empty units so they can still be renamed or deleted.
 */
export function listUnits(state) {
  const units = [];
  const known = new Set(state.groups.map((g) => g.id));
  state.regions.forEach((r, i) => {
    const gid = known.has(r.group) ? r.group : null;
    const last = units[units.length - 1];
    if (gid == null) units.push({ type: "region", index: i });
    else if (last && last.type === "group" && last.group.id === gid) last.indices.push(i);
    else units.push({ type: "group", group: state.groups.find((g) => g.id === gid), indices: [i] });
  });
  const used = new Set(units.filter((u) => u.type === "group").map((u) => u.group.id));
  for (const g of state.groups) if (!used.has(g.id)) units.push({ type: "group", group: g, indices: [] });
  return units;
}

function unitRange(u) {
  if (u.type === "region") return [u.index, u.index];
  return u.indices.length ? [u.indices[0], u.indices[u.indices.length - 1]] : null;
}

function moveBlock(arr, start, end, insertAt) {
  // move arr[start..end] so that it begins at index `insertAt` of the array without the block
  const block = arr.splice(start, end - start + 1);
  arr.splice(insertAt, 0, ...block);
}

// ------------------------------------------------------------------ operations (return new state)
export function addRegion(state, bbox, fields = {}) {
  const s = cloneState(state);
  const id = newId("r", s.regions.map((r) => r.id));
  const n = s.regions.length + 1;
  s.regions.push({
    id,
    name: `Region ${n}`,
    bbox: bbox.map((v) => Math.round(v)),
    lora: "",
    weight: 1.0,
    visible: true,
    locked: false,
    color: PALETTE[(n - 1) % PALETTE.length],
    group: null,
    label: "object",
    caption: "",
    text: "",
    palette: [],
    ...fields,
  });
  return { state: s, id };
}

export function updateRegion(state, id, patch) {
  const s = cloneState(state);
  const r = s.regions.find((x) => x.id === id);
  if (r) Object.assign(r, patch);
  return s;
}

export function deleteRegions(state, ids) {
  const set = new Set(ids);
  const s = cloneState(state);
  s.regions = s.regions.filter((r) => !set.has(r.id));
  return s;
}

/** Move a region one step up (-1) or down (+1). Within a group it stays inside the group's run;
 *  an ungrouped region steps over a whole neighbouring unit (a group moves as one block). */
export function moveRegion(state, id, dir) {
  const s = cloneState(state);
  const i = s.regions.findIndex((r) => r.id === id);
  if (i < 0) return s;
  const known = new Set(s.groups.map((g) => g.id));
  const gid = known.has(s.regions[i].group) ? s.regions[i].group : null;
  if (gid != null) {
    const j = i + dir;
    if (j < 0 || j >= s.regions.length || s.regions[j].group !== gid) return s;
    [s.regions[i], s.regions[j]] = [s.regions[j], s.regions[i]];
    return s;
  }
  const units = listUnits(s).filter((u) => unitRange(u));
  const k = units.findIndex((u) => u.type === "region" && u.index === i);
  const other = units[k + dir];
  if (!other) return s;
  const [a, b] = unitRange(other);
  if (dir < 0) moveBlock(s.regions, i, i, a);
  else moveBlock(s.regions, i, i, b);          // after removal, b is the slot just past the unit
  return s;
}

/** Move a group's run of regions one unit up or down as a block (8.7a). */
export function moveGroup(state, gid, dir) {
  const s = cloneState(state);
  const units = listUnits(s).filter((u) => unitRange(u));
  const k = units.findIndex((u) => u.type === "group" && u.group.id === gid);
  if (k < 0) return s;
  const other = units[k + dir];
  if (!other) return s;
  const [start, end] = unitRange(units[k]);
  const [a, b] = unitRange(other);
  if (dir < 0) moveBlock(s.regions, start, end, a);
  else moveBlock(s.regions, start, end, b - (end - start));
  return s;
}

/** Drag-and-drop: move region `id` to sit just before region `beforeId` (or to the end when null),
 *  adopting the group of the row it is dropped on. */
export function dropRegion(state, id, beforeId, group = undefined) {
  const s = cloneState(state);
  const i = s.regions.findIndex((r) => r.id === id);
  if (i < 0 || id === beforeId) return s;
  const [r] = s.regions.splice(i, 1);
  let j = beforeId == null ? s.regions.length : s.regions.findIndex((x) => x.id === beforeId);
  if (j < 0) j = s.regions.length;
  if (group !== undefined) r.group = group;
  s.regions.splice(j, 0, r);
  return s;
}

/** Drag-and-drop of a whole group before region `beforeId` (or to the end). */
export function dropGroup(state, gid, beforeId) {
  const s = cloneState(state);
  const members = s.regions.filter((r) => r.group === gid);
  if (!members.length) return s;
  if (members.some((r) => r.id === beforeId)) return s;
  const rest = s.regions.filter((r) => r.group !== gid);
  let j = beforeId == null ? rest.length : rest.findIndex((x) => x.id === beforeId);
  if (j < 0) j = rest.length;
  rest.splice(j, 0, ...members);
  s.regions = rest;
  return s;
}

export function addGroup(state, name) {
  const s = cloneState(state);
  const id = newId("g", s.groups.map((g) => g.id));
  s.groups.push({ id, name: name || `Group ${s.groups.length + 1}`, collapsed: false });
  return { state: s, id };
}

export function renameGroup(state, gid, name) {
  const s = cloneState(state);
  const g = s.groups.find((x) => x.id === gid);
  if (g) g.name = name;
  return s;
}

export function setGroupCollapsed(state, gid, collapsed) {
  const s = cloneState(state);
  const g = s.groups.find((x) => x.id === gid);
  if (g) g.collapsed = collapsed;
  return s;
}

export function setGroupVisible(state, gid, visible) {
  const s = cloneState(state);
  for (const r of s.regions) if (r.group === gid) r.visible = visible;
  return s;
}

/** "Delete all": removes the group and every region in it. */
export function deleteGroup(state, gid) {
  const s = cloneState(state);
  s.regions = s.regions.filter((r) => r.group !== gid);
  s.groups = s.groups.filter((g) => g.id !== gid);
  return s;
}

/** Put a region into a group (or null for none). It moves to the end of that group's run so groups
 *  stay contiguous; joining an empty group leaves it where it is. */
export function assignGroup(state, id, gid) {
  let s = cloneState(state);
  const r = s.regions.find((x) => x.id === id);
  if (!r || r.group === gid) return s;
  const members = s.regions.filter((x) => x.group === gid && x.id !== id);
  r.group = gid;
  if (gid != null && members.length) {
    const last = members[members.length - 1];
    const idx = s.regions.findIndex((x) => x.id === last.id);
    const next = s.regions[idx + 1];
    s = dropRegion(s, id, next ? next.id : null);
  }
  return s;
}

// ------------------------------------------------------------------ undo / redo (8.4c)
export class History {
  constructor(limit = 100) {
    this.limit = limit;
    this.past = [];
    this.future = [];
  }

  /** Record the text that was current *before* an edit. */
  record(beforeText) {
    if (this.past[this.past.length - 1] === beforeText) return;
    this.past.push(beforeText);
    if (this.past.length > this.limit) this.past.shift();
    this.future = [];
  }

  undo(currentText) {
    if (!this.past.length) return null;
    this.future.push(currentText);
    return this.past.pop();
  }

  redo(currentText) {
    if (!this.future.length) return null;
    this.past.push(currentText);
    return this.future.pop();
  }

  get canUndo() {
    return this.past.length > 0;
  }

  get canRedo() {
    return this.future.length > 0;
  }
}
