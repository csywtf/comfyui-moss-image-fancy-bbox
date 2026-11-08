// Headless tests for web/state.js: round-trip (8.10c), ordering, groups, undo. Run: node --test tests/web/
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import * as S from "../../web/state.js";

const FIXTURE = readFileSync(fileURLToPath(new URL("./fixtures/regions_full.json", import.meta.url)), "utf8").replace(/\n$/, "");

test("round-trip: parse -> serialise is identical for a fixture with every field, including unknown ones", () => {
  const res = S.parseState(FIXTURE);
  assert.ok(res.ok, res.error);
  assert.equal(S.serializeState(res.state), FIXTURE);
  assert.deepEqual(JSON.parse(S.serializeState(res.state)), JSON.parse(FIXTURE));
  // unknown fields survive an edit too
  const edited = S.updateRegion(res.state, "r-bob", { weight: 0.5 });
  const back = JSON.parse(S.serializeState(edited));
  assert.deepEqual(back.regions[0].x_future, { trigger: "alice", tags: ["a", "b"] });
  assert.deepEqual(back.x_top_level_unknown, { anything: [1, 2, { deep: true }] });
  assert.deepEqual(back.groups[0].x_group_meta, { owner: "me" });
  assert.equal(back.editor.x_zoom, 1.5);
});

test("older or minimal workflows load without crashing", () => {
  for (const text of [
    "",
    "[]",
    '{"regions": []}',
    '[{"bbox": [-10, -10, 10, 10], "lora": "a.safetensors"}]',
    '{"version": 1, "regions": [{"name": "x", "bbox": [0, 0, 5, 5]}]}',
  ]) {
    const res = S.parseState(text);
    assert.ok(res.ok, `${text}: ${res.error}`);
    for (const r of res.state.regions) {
      assert.equal(typeof r.id, "string");
      assert.equal(r.visible, true);
      assert.ok(Array.isArray(S.lorasOf(r)));
    }
  }
});

test("newer version is tolerated with a warning", () => {
  const res = S.parseState('{"version": 7, "regions": [], "brand_new": {"x": 1}}');
  assert.ok(res.ok);
  assert.ok(res.warnings.some((w) => w.includes("newer")));
  assert.deepEqual(JSON.parse(S.serializeState(res.state)).brand_new, { x: 1 });
});

test("parse failures are reported, never thrown", () => {
  for (const text of ["{oops", "42", '{"regions": 3}', '{"regions": [{"bbox": [5, 0, 0, 5]}]}', '{"regions": [{"bbox": "no"}]}']) {
    const res = S.parseState(text);
    assert.equal(res.ok, false, text);
    assert.equal(typeof res.error, "string");
  }
});

function three() {
  let s = S.emptyState();
  for (const name of ["A", "B", "C"]) s = S.addRegion(s, [0, 0, 10, 10], { name }).state;
  return s;
}
const names = (s) => s.regions.map((r) => r.name);

test("emitted order is list order; hidden regions are excluded", () => {
  let s = three();
  s = S.updateRegion(s, s.regions[1].id, { visible: false });
  assert.deepEqual(S.emittedRegions(s).map((r) => r.name), ["A", "C"]);
});

test("move up / move down (ungrouped)", () => {
  let s = three();
  const c = s.regions[2].id;
  s = S.moveRegion(s, c, -1);
  assert.deepEqual(names(s), ["A", "C", "B"]);
  s = S.moveRegion(s, c, -1);
  assert.deepEqual(names(s), ["C", "A", "B"]);
  s = S.moveRegion(s, c, -1); // already first
  assert.deepEqual(names(s), ["C", "A", "B"]);
  s = S.moveRegion(s, c, +1);
  assert.deepEqual(names(s), ["A", "C", "B"]);
});

test("groups: contiguous assignment, units, moving as a block, and no effect on emitted content", () => {
  let s = three();
  s = S.addRegion(s, [0, 0, 10, 10], { name: "D" }).state;
  const { state, id: g } = S.addGroup(s, "G");
  s = state;
  const [a, b, c, d] = s.regions.map((r) => r.id);
  s = S.assignGroup(s, a, g);
  s = S.assignGroup(s, c, g); // C joins after A
  assert.deepEqual(names(s), ["A", "C", "B", "D"]);
  const units = S.listUnits(s);
  assert.equal(units[0].type, "group");
  assert.deepEqual(units[0].indices, [0, 1]);
  // an ungrouped region steps over the whole group
  s = S.moveRegion(s, b, -1);
  assert.deepEqual(names(s), ["B", "A", "C", "D"]);
  // a grouped region moves only inside its group
  s = S.moveRegion(s, a, -1);
  assert.deepEqual(names(s), ["B", "A", "C", "D"]);
  s = S.moveRegion(s, a, +1);
  assert.deepEqual(names(s), ["B", "C", "A", "D"]);
  // the group moves as a unit
  s = S.moveGroup(s, g, +1);
  assert.deepEqual(names(s), ["B", "D", "C", "A"]);
  s = S.moveGroup(s, g, -1);
  s = S.moveGroup(s, g, -1);
  assert.deepEqual(names(s), ["C", "A", "B", "D"]);
  // collapsing/renaming a group changes nothing that is emitted
  const before = JSON.stringify(S.emittedRegions(s));
  s = S.setGroupCollapsed(s, g, true);
  s = S.renameGroup(s, g, "Renamed");
  assert.equal(JSON.stringify(S.emittedRegions(s)), before);
  // hide all / show all
  s = S.setGroupVisible(s, g, false);
  assert.deepEqual(S.emittedRegions(s).map((r) => r.name), ["B", "D"]);
  s = S.setGroupVisible(s, g, true);
  // delete all
  s = S.deleteGroup(s, g);
  assert.deepEqual(names(s), ["B", "D"]);
  assert.equal(s.groups.length, 0);
  void d;
});

test("drag-and-drop reordering", () => {
  let s = three();
  const [a, b, c] = s.regions.map((r) => r.id);
  s = S.dropRegion(s, c, a);
  assert.deepEqual(names(s), ["C", "A", "B"]);
  s = S.dropRegion(s, c, null);
  assert.deepEqual(names(s), ["A", "B", "C"]);
  const { state, id: g } = S.addGroup(s, "G");
  s = S.assignGroup(state, b, g);
  s = S.assignGroup(s, c, g);
  s = S.dropGroup(s, g, a);
  assert.deepEqual(names(s), ["B", "C", "A"]);
});

test("delete", () => {
  let s = three();
  s = S.deleteRegions(s, [s.regions[0].id, s.regions[2].id]);
  assert.deepEqual(names(s), ["B"]);
});

test("operations never mutate their input", () => {
  const s = three();
  const snapshot = S.serializeState(s);
  S.moveRegion(s, s.regions[2].id, -1);
  S.updateRegion(s, s.regions[0].id, { bbox: [1, 2, 3, 4] });
  S.deleteRegions(s, [s.regions[1].id]);
  assert.equal(S.serializeState(s), snapshot);
});

test("undo / redo keeps at least 20 operations", () => {
  const h = new S.History(100);
  let text = "v0";
  for (let i = 1; i <= 25; i++) {
    h.record(text);
    text = `v${i}`;
  }
  for (let i = 24; i >= 0; i--) {
    text = h.undo(text);
    assert.equal(text, `v${i}`);
  }
  assert.equal(h.undo(text), null);
  for (let i = 1; i <= 25; i++) {
    text = h.redo(text);
    assert.equal(text, `v${i}`);
  }
  // a new edit clears the redo stack
  h.undo(text);
  h.record("v24");
  assert.equal(h.canRedo, false);
});

test("prompt fields: defaults for older workflows, and which boxes become layout items", () => {
  const res = S.parseState('{"version": 1, "regions": [{"name": "a", "bbox": [0, 0, 5, 5], "lora": "a.safetensors"}]}');
  assert.ok(res.ok);
  const r = res.state.regions[0];
  assert.equal(r.label, "object");
  assert.equal(r.caption, "");
  assert.deepEqual(r.palette, []);
  assert.equal(S.hasPrompt(r), false);                                   // LoRA only
  assert.equal(S.hasPrompt({ ...r, caption: "a cat" }), true);           // prompt
  assert.equal(S.hasPrompt({ ...r, label: "text", text: "SALE" }), true); // text box with text only
  assert.equal(S.hasPrompt({ ...r, label: "object", text: "SALE" }), false);
  const bad = S.parseState('{"regions": [{"bbox": [0, 0, 1, 1], "label": "person"}]}');
  assert.ok(bad.ok && bad.warnings.some((w) => w.includes("person")));
});

test("LoRA lists: new entries start at full strength, version-1 boxes are read and converted on edit", () => {
  const { state } = S.addRegion(S.emptyState(), [0, 0, 10, 10]);
  assert.deepEqual(state.regions[0].loras, []);
  assert.deepEqual(S.newLoraEntry("a.safetensors"), { lora: "a.safetensors", weight: 1, curve: [{ x: 0, y: 1 }, { x: 1, y: 1 }] });
  const old = S.parseState('{"version": 1, "regions": [{"id": "r", "bbox": [0, 0, 1, 1], "lora": "a", "weight": 0.5}]}');
  assert.ok(old.ok);
  assert.deepEqual(S.lorasOf(old.state.regions[0]), [{ lora: "a", weight: 0.5 }]);   // no curve: node ramp
  const moved = S.setLoras(old.state, "r", [...S.lorasOf(old.state.regions[0]), S.newLoraEntry("b")]);
  const r = moved.regions[0];
  assert.ok(!("lora" in r) && !("weight" in r) && r.loras.length === 2);
  assert.equal(S.parseState('{"regions": [{"bbox": [0, 0, 1, 1], "loras": [{"lora": "a", "curve": [{"x": 0}]}]}]}').ok, false);
  assert.equal(S.parseState('{"regions": [{"bbox": [0, 0, 1, 1], "loras": "a"}]}').ok, false);
  assert.equal(S.parseState('{"regions": [{"bbox": [0, 0, 1, 1], "lora": "a", "curve": [{"x": 0}]}]}').ok, false);
});

test("what the backend sees: view-only edits change nothing, image edits do, hidden boxes are left out", () => {
  const doc = {
    version: 2,
    regions: [
      { id: "a", name: "Alice", bbox: [0, 0, 10, 10], loras: [{ lora: "a.safetensors", weight: 1 }], visible: true,
        locked: false, color: "#e6194b", group: "g1", label: "character", caption: "a woman", text: "", palette: [], x_future: 1 },
      { id: "h", name: "Hidden", bbox: [0, 0, 5, 5], loras: [], visible: false },
    ],
    groups: [{ id: "g1", name: "G", collapsed: false }],
    editor: { grid_divisions: 3, snap: false },
    x_top: "kept",
  };
  const base = S.backendRegions(JSON.stringify(doc, null, 2));
  const sent = JSON.parse(base);
  assert.deepEqual(sent, {
    version: 2,
    regions: [{ id: "a", bbox: [0, 0, 10, 10], loras: [{ lora: "a.safetensors", weight: 1 }], label: "character",
                caption: "a woman", text: "", palette: [], x_future: 1 }],
    x_top: "kept",
  });
  const variant = (fn) => { const d = structuredClone(doc); fn(d); return S.backendRegions(JSON.stringify(d)); };
  // view-only: identical output (same cache key)
  for (const fn of [
    (d) => (d.regions[0].color = "#000000"), (d) => (d.regions[0].locked = true), (d) => (d.regions[0].name = "Renamed"),
    (d) => (d.regions[0].group = null), (d) => (d.groups[0].collapsed = true), (d) => (d.editor.snap = true),
    (d) => (d.regions[1].bbox = [1, 1, 4, 4]),                     // a hidden box moved
  ]) assert.equal(variant(fn), base);
  // what the image depends on: different output
  for (const fn of [
    (d) => (d.regions[0].bbox = [0, 0, 11, 10]), (d) => (d.regions[0].caption = "a man"),
    (d) => (d.regions[0].loras[0].weight = 0.5), (d) => (d.regions[0].label = "object"),
    (d) => (d.regions[1].visible = true),
  ]) assert.notEqual(variant(fn), base);
  // order among visible boxes matters (layer order); relative to a hidden one it doesn't
  assert.equal(variant((d) => d.regions.reverse()), base);
  const twoVisible = (d) => (d.regions[1].visible = true);
  assert.notEqual(variant((d) => { twoVisible(d); d.regions.reverse(); }), variant(twoVisible));
  assert.equal(S.backendRegions("{oops"), "{oops");                 // the node reports bad JSON itself
  assert.deepEqual(JSON.parse(S.backendRegions('[{"bbox": [0, 0, 1, 1], "color": "#fff"}]')), { regions: [{ bbox: [0, 0, 1, 1] }] });
});
