"""MOSS-Image Character Masks: settings, rectangles, ownership, the SAM3 wrapper and the node, with SAM3
replaced by a fake that "finds" known rectangles."""

import json
import sys
import types

import numpy as np
import pytest
import torch

from moss_regional_lora import character_masks as cm
from moss_regional_lora.regions import BoxInfo

W, H = 200, 100            # image (and canvas unless stated)


def box(id, bbox, label="character", name=None, color="#ff0000", loras=()):
    return BoxInfo(id=id, name=name or id, bbox=tuple(bbox), label=label, caption=f"{id} caption",
                   color=color, loras=tuple(loras))


def rect_mask(x0, y0, x1, y1, h=H, w=W):
    m = torch.zeros(h, w, dtype=torch.bool)
    m[y0:y1, x0:x1] = True
    return m


def centred(x0, y0, x1, y1, w=W, h=H):
    """Top-left pixel rect -> the editor's centred [y_min, x_min, y_max, x_max]."""
    return [y0 - h // 2, x0 - w // 2, y1 - h // 2, x1 - w // 2]


class FakeRunner:
    """A SAM3 stand-in over a scene of rectangles, keyed by part name ("face", "eye", "person", ...), each
    list in detection-score order.

    persons / find: like SAM3 with "name:N", the first N detections that are inside the crop and visible
    (not greyed out to NEUTRAL in the image it is given). segment_box: the box shrunk by 2 px, or nothing
    for rects in ``lost``. Records calls."""

    def __init__(self, scene, lost=()):
        self.scene, self.lost, self.calls = scene, set(lost), []

    def _detections(self, image, crop, prompt):
        name, _, n = prompt.partition(":")
        n = int(n or 1)
        x0, y0, x1, y1 = crop
        hits = []
        for r in self.scene.get(name, []):
            m = rect_mask(*r, image.shape[0], image.shape[1])
            inside = torch.zeros_like(m)
            inside[y0:y1, x0:x1] = m[y0:y1, x0:x1]
            visible = inside & ~(image[..., :3] == cm.NEUTRAL).all(dim=-1)
            if visible.any():
                hits.append(visible)
        return hits[:n]

    def persons(self, image, key, crop):
        self.calls.append(("person", crop))
        return self._detections(image, crop, cm.PERSON_PROMPT)

    def segment_box(self, image, key, rect):
        self.calls.append(("box", rect))
        x0, y0, x1, y1 = rect
        if any(rect == r for r in self.lost):
            return torch.zeros(image.shape[0], image.shape[1], dtype=torch.bool)
        return rect_mask(x0 + 2, y0 + 2, x1 - 2, y1 - 2, image.shape[0], image.shape[1])

    def find(self, image, key, crop, prompt):
        self.calls.append(("find", crop, prompt, image.clone()))
        out = torch.zeros(image.shape[0], image.shape[1], dtype=torch.bool)
        for m in self._detections(image, crop, prompt):
            out |= m
        return out


# ------------------------------------------------------------------------- settings
def test_detail_defaults_by_label():
    c = cm.default_detail("character")
    assert c.enabled and c.parts_on() == ["face", "eyes"] and c.use_box_loras and c.prompt == ""
    for label in ("object", "text"):
        d = cm.default_detail(label)
        assert not d.enabled and d.parts_on() == ["face", "eyes"]   # ticking Detail starts from face + eyes
    assert cm.PART_NAMES == ("face", "eyes", "hands", "hair", "skin")
    assert sorted(cm.DETAIL_ORDER) == sorted(cm.PART_NAMES)


def test_stored_settings_override_defaults_and_bad_values_are_ignored():
    b = box("a", [0, 0, 10, 10])
    raw = {"enabled": False, "prompt": "green eyes", "use_box_loras": False,
           "loras": [{"lora": "eye.safetensors", "weight": 0.6}, {"lora": "", "weight": 1}, {"lora": "z", "weight": 0},
                     {"lora": "x", "weight": "a lot"}],
           "parts": {"hands": {"on": True, "prompt": "five fingers", "denoise": 0.5},
                     "eyes": {"on": False}, "face": {"denoise": 7}, "tail": {"on": True}}}
    d = cm.resolve_detail(b, raw)
    assert (d.enabled, d.prompt, d.use_box_loras) == (False, "green eyes", False)
    assert d.loras == (("eye.safetensors", 0.6),)
    assert d.parts_on() == ["face", "hands"]
    assert d.parts["hands"] == cm.PartSettings(True, "five fingers", 0.5)
    assert d.parts["face"].denoise == 1.0                     # clamped
    assert cm.resolve_detail(b, {"enabled": "yes"}).enabled is True   # not a bool: default for a character


def test_parse_detail():
    assert cm.parse_detail("") == {} and cm.parse_detail(None) == {}
    assert cm.parse_detail('{"version": 1, "boxes": {"a": {"enabled": true}, "b": 3}}') == {"a": {"enabled": True}}
    for bad in ("{oops", "[]", '{"boxes": []}'):
        with pytest.raises(cm.DetailError):
            cm.parse_detail(bad)


# ------------------------------------------------------------------------- rectangles
def test_box_rect_scales_from_the_canvas_and_clips():
    # canvas 100x50, image 200x100: everything doubles
    assert cm.box_rect([-25, -50, 0, 0], (100, 50), (200, 100)) == (0, 0, 100, 50)
    assert cm.box_rect(centred(10, 10, 60, 40), (W, H), (W, H)) == (10, 10, 60, 40)
    assert cm.box_rect([-1000, -1000, -20, 1000], (W, H), (W, H)) == (0, 0, W, 30)    # clipped
    assert cm.box_rect([200, 200, 300, 300], (W, H), (W, H)) is None                   # off the image


def test_canvas_of():
    assert cm.canvas_of(json.dumps({"regions": [], "canvas": {"width": 1024, "height": 768}})) == (1024, 768)
    assert cm.canvas_of('{"regions": []}') is None and cm.canvas_of("nope") is None


def test_crop_rect_is_square_inside_the_image_and_contains_the_box():
    for rect in [(10, 10, 60, 40), (0, 0, 30, 90), (150, 60, 200, 100), (90, 45, 110, 55)]:
        c = cm.crop_rect(rect, 0.15, (W, H))
        assert 0 <= c[0] <= rect[0] and 0 <= c[1] <= rect[1] and rect[2] <= c[2] <= W and rect[3] <= c[3] <= H
    c = cm.crop_rect((90, 45, 110, 55), 0.0, (W, H))           # room to grow: square
    assert c[2] - c[0] == c[3] - c[1] == 20
    c = cm.crop_rect((150, 60, 200, 100), 0.0, (W, H))         # at the corner: slides in, keeps its size
    assert (c[2] - c[0], c[3] - c[1]) == (50, 50) and c[2] == W and c[3] == H


def test_dilate_and_mask_rect():
    m = rect_mask(10, 10, 12, 12)
    assert cm.mask_rect(m) == (10, 10, 12, 12) and cm.mask_rect(torch.zeros(4, 4, dtype=torch.bool)) is None
    assert cm.mask_rect(cm.dilate(m, 3)) == (7, 7, 15, 15)


# ------------------------------------------------------------------------- ownership
def test_overlapping_characters_keep_their_own_pixels():
    back = rect_mask(10, 10, 110, 90)       # layer 0 (underneath)
    front = rect_mask(80, 10, 180, 90)      # layer 1 (on top); overlap x 80..110
    # SAM3 run on the back box's crop also sees the front character's face (x 85..100)
    back_face = rect_mask(30, 20, 50, 40) | rect_mask(85, 20, 100, 40)
    front_face = rect_mask(85, 20, 100, 40)
    out = cm.assign_parts([back, front], [{"face": back_face}, {"face": front_face}], [2, 2])
    assert torch.equal(out[0]["face"], rect_mask(30, 20, 50, 40))     # the front face isn't the back's
    assert torch.equal(out[1]["face"], front_face)


def test_unowned_edge_pixels_are_allowed_within_grow_and_contested_ones_go_on_top():
    a = rect_mask(10, 10, 50, 50)
    b = rect_mask(54, 10, 90, 50)           # 4 px gap between the outlines
    hair = rect_mask(45, 12, 60, 14)        # crosses the gap
    out = cm.assign_parts([a, b], [{"hair": hair}, {"hair": hair}], [3, 3])
    # a may use its own pixels and the gap within 3 px (x 50..53); b gets x 51..60; the contested gap -> b
    assert out[0]["hair"][12, 45:51].all() and not out[0]["hair"][12, 51:].any()
    assert out[1]["hair"][12, 51:60].all() and not out[1]["hair"][12, :51].any()
    far = rect_mask(150, 70, 160, 80)       # nowhere near either outline
    assert not cm.assign_parts([a], [{"hands": far}], [3])[0]["hands"].any()


# ------------------------------------------------------------------------- one frame
def frame_setup():
    boxes = [box("alice", centred(10, 10, 110, 90), color="#e6194b"),
             box("bob", centred(80, 10, 180, 90), color="#3cb44b"),
             box("lamp", centred(185, 5, 199, 30), label="object")]
    scene = {"face": [(30, 20, 50, 40), (120, 20, 140, 40)], "eye": [(33, 25, 37, 28), (43, 25, 47, 28)],
             "hand": [(15, 70, 25, 85)]}
    return boxes, scene


def details(boxes, stored=None):
    stored = stored or {}
    return {b.id: cm.resolve_detail(b, stored.get(b.id)) for b in boxes}


def test_mask_frame_finds_ticked_parts_per_character():
    boxes, scene = frame_setup()
    runner = FakeRunner(scene)
    img = torch.rand(H, W, 3)
    f = cm.mask_frame(img, boxes, details(boxes, {"alice": {"parts": {"hands": {"on": True}}}}), None, runner, 0.1)
    assert [b.id for b in f.boxes] == ["alice", "bob"]          # lamp: an object, Detail off by default
    alice, bob = f.boxes
    assert set(alice.parts) == {"face", "eyes", "hands"} and set(bob.parts) == {"face", "eyes"}
    assert alice.found == {"face": True, "eyes": True, "hands": True}
    assert bob.found == {"face": True, "eyes": False}           # no eyes found in bob's crop
    assert torch.equal(alice.parts["face"], rect_mask(30, 20, 50, 40))   # bob's face isn't alice's
    assert torch.equal(bob.parts["face"], rect_mask(120, 20, 140, 40))
    assert alice.caption == "alice caption" and alice.detail.parts_on() == ["face", "eyes", "hands"]
    # SAM3 asked once per box for the outline, then once per ticked part, on a crop around the box
    # no person detected in this scene: each box falls back to SAM3's box prompt
    assert [c[0] for c in runner.calls].count("person") == 2 and [c[0] for c in runner.calls].count("box") == 2
    finds = [c for c in runner.calls if c[0] == "find"]
    assert len(finds) == 5
    for c, r in zip(finds, [alice.rect] * 3 + [bob.rect] * 2):  # each crop holds its box, inside the image
        x0, y0, x1, y1 = c[1]
        assert 0 <= x0 <= r[0] and 0 <= y0 <= r[1] and r[2] <= x1 <= W and r[3] <= y1 <= H


def test_disabled_boxes_are_skipped_and_a_lost_outline_falls_back_to_the_box():
    boxes, scene = frame_setup()
    runner = FakeRunner(scene, lost=[(80, 10, 180, 90)])
    stored = {"alice": {"enabled": False}, "lamp": {"enabled": True, "parts": {"face": {"on": False}}}}
    f = cm.mask_frame(torch.rand(H, W, 3), boxes, details(boxes, stored), None, runner, 0.1)
    assert [b.id for b in f.boxes] == ["bob", "lamp"]
    bob = f.boxes[0]
    assert not bob.outline_found and torch.equal(bob.outline, rect_mask(80, 10, 180, 90))
    assert set(f.boxes[1].parts) == {"eyes"}


def test_compute_parses_regions_and_scales_from_the_canvas():
    regions = json.dumps({"version": 2, "canvas": {"width": 100, "height": 50}, "regions": [
        {"id": "a", "name": "A", "bbox": [-25, -50, 0, 0], "label": "character"},
        {"id": "h", "name": "Hidden", "bbox": [-25, -50, 25, 50], "label": "character", "visible": False}]})
    runner = FakeRunner({"face": [(10, 10, 20, 20)]})
    out = cm.compute(torch.rand(2, H, W, 3), regions, "", runner, 0.0)
    assert len(out.frames) == 2 and [b.id for b in out.frames[0].boxes] == ["a"]
    assert out.frames[0].boxes[0].rect == (0, 0, 100, 50)       # canvas 100x50 on a 200x100 image
    u = out.union()
    assert u.shape == (2, H, W) and u[0, 10:20, 10:20].all() and u[0].sum() == 100


def test_hug_each_character_gets_its_own_face_despite_the_neighbours_in_the_crop():
    """The reported case: two tall boxes side by side, so each crop holds both characters, and SAM3 ranks
    the neighbour's face (and a face-like distractor on the neighbour) above the real one. With the other
    character greyed out, the right character's own face is found."""
    boxes = [box("anime", centred(0, 0, 100, 100)), box("real", centred(100, 0, 200, 100))]
    scene = {"person": [(5, 2, 98, 100), (102, 2, 195, 100)],
             "face": [(80, 20, 98, 38), (60, 60, 75, 75), (110, 15, 135, 40)]}   # anime, distractor, real
    runner = FakeRunner(scene)
    img = torch.rand(H, W, 3)
    f = cm.mask_frame(img, boxes, details(boxes, {b.id: {"parts": {"eyes": {"on": False}}} for b in boxes}),
                      None, runner, 0.15)
    anime, real = f.boxes
    assert real.found == {"face": True} and torch.equal(real.parts["face"], rect_mask(110, 15, 135, 40))
    assert anime.found == {"face": True} and anime.parts["face"][20:38, 80:98].all()
    assert not anime.parts["face"][15:40, 110:135].any()
    # what SAM3 was shown for the right character: the left character's pixels greyed out, its own intact
    view = [c for c in runner.calls if c[0] == "find" and c[1][2] == W][-1][3]
    assert (view[50, 50] == cm.NEUTRAL).all() and torch.equal(view[50, 150], img[50, 150])
    # the outlines are the detected people, not the boxes, so no box-prompt fallback was needed
    assert torch.equal(anime.outline, rect_mask(5, 2, 98, 100)) and anime.outline_found
    assert "box" not in [c[0] for c in runner.calls]


def test_pick_outline_takes_the_person_matching_the_box():
    rect = (100, 0, 200, 100)
    neighbour = rect_mask(5, 2, 120, 100)          # reaches into the box, but mostly outside it
    me = rect_mask(105, 5, 195, 100)
    small = rect_mask(150, 50, 155, 55)            # inside, but a sliver of the box
    assert torch.equal(cm.pick_outline([neighbour, small, me], rect), me)
    assert cm.pick_outline([neighbour, small], rect) is None
    assert cm.pick_outline([], rect) is None


def test_clean_mask_fills_speckle_and_drops_specks():
    speckled = torch.zeros(H, W, dtype=torch.bool)
    speckled[20:60, 20:60] = (torch.arange(40)[:, None] + torch.arange(40)[None, :]) % 2 == 0   # checkerboard
    speckled[80, 150] = True                                                                       # lone speck
    c = cm.clean_mask(speckled, 2)
    assert c[22:58, 22:58].all() and not c[80, 150] and not c[70:, :].any()
    near_edge = rect_mask(5, 2, 98, 100)           # 2 px from the top, touching the bottom: unchanged
    assert torch.equal(cm.clean_mask(near_edge, 2), near_edge)


# ------------------------------------------------------------------------- SAM3 wrapper
class FakeDetect:
    calls = []

    @classmethod
    def execute(cls, model, image, conditioning=None, bboxes=None, threshold=0.5, refine_iterations=2,
                individual_masks=False):
        cls.calls.append({"shape": tuple(image.shape), "cond": conditioning, "bboxes": bboxes,
                          "threshold": threshold, "refine": refine_iterations, "individual": individual_masks})
        m = torch.zeros(1, image.shape[1], image.shape[2])
        m[0, :2, :3] = 1.0                                       # top-left corner of whatever it was given
        if individual_masks:                                     # [N, h, w]: a second detection too
            m = torch.cat([m, torch.zeros(1, image.shape[1], image.shape[2])])
            m[1, -1, -1] = 1.0
        return types.SimpleNamespace(args=(m, []))


class FakeClip:
    def __init__(self):
        self.encoded = []

    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, tokens):
        self.encoded.append(tokens)
        return [[f"cond:{tokens}", {}]]


@pytest.fixture
def fake_sam3(monkeypatch):
    mod = types.ModuleType("comfy_extras.nodes_sam3")
    mod.SAM3_Detect = FakeDetect
    pkg = types.ModuleType("comfy_extras")
    monkeypatch.setitem(sys.modules, "comfy_extras", pkg)
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_sam3", mod)
    FakeDetect.calls = []
    cm._MEMO.clear()
    return FakeDetect


def test_runner_calls_comfyui_sam3_and_memoises(fake_sam3):
    clip = FakeClip()
    model = types.SimpleNamespace(model=object())
    r = cm.Sam3Runner(model, clip, 0.4, 3)
    img = torch.rand(H, W, 3)
    key = cm.image_key(img)
    m = r.segment_box(img, key, (10, 20, 60, 70))
    c = fake_sam3.calls[-1]
    assert c["bboxes"] == [{"x": 10, "y": 20, "width": 50, "height": 50}] and c["cond"] is None
    assert c["shape"] == (1, H, W, 3) and (c["threshold"], c["refine"], c["individual"]) == (0.4, 3, False)
    assert m.dtype == torch.bool and m[:2, :3].all() and m.sum() == 6
    f = r.find(img, key, (100, 30, 160, 90), "eye:2")
    c = fake_sam3.calls[-1]
    assert c["shape"] == (1, 60, 60, 3) and c["cond"] == [["cond:eye:2", {}]] and c["bboxes"] is None
    assert f[30:32, 100:103].all() and f.sum() == 6              # pasted back at the crop's corner
    # same image (by content), same request: no SAM3 call; the prompt is encoded once
    n = len(fake_sam3.calls)
    r.find(img.clone(), cm.image_key(img.clone()), (100, 30, 160, 90), "eye:2")
    r.segment_box(img, key, (10, 20, 60, 70))
    assert len(fake_sam3.calls) == n and clip.encoded == ["eye:2"]
    # people: one mask per detection, pasted back at the crop
    ps = r.persons(img, key, (40, 10, 90, 60))
    c = fake_sam3.calls[-1]
    assert c["cond"] == [[f"cond:{cm.PERSON_PROMPT}", {}]] and c["individual"] is True and c["shape"] == (1, 50, 50, 3)
    assert len(ps) == 2 and ps[0][10:12, 40:43].all() and ps[0].sum() == 6 and ps[1][59, 89] and ps[1].sum() == 1
    n = len(fake_sam3.calls)
    assert len(r.persons(img, key, (40, 10, 90, 60))) == 2 and len(fake_sam3.calls) == n   # memoised
    # a different threshold is a different request
    cm.Sam3Runner(model, clip, 0.6, 3).segment_box(img, key, (10, 20, 60, 70))
    assert len(fake_sam3.calls) == n + 1


def test_part_masks_stay_inside_the_detectors_boxes(fake_sam3, monkeypatch):
    """SAM3's refined face mask runs down the neck; the detector's box stops at the chin."""
    def execute(model, image, conditioning=None, bboxes=None, threshold=0.5, refine_iterations=2,
                individual_masks=False):
        m = torch.zeros(1, image.shape[1], image.shape[2])
        m[0, 10:60, 10:30] = 1.0                                 # face (rows 10-35) + neck (rows 35-60)
        return types.SimpleNamespace(args=(m, [[{"x": 8.0, "y": 8.0, "width": 24.0, "height": 27.0, "score": 0.9}]]))
    monkeypatch.setattr(FakeDetect, "execute", staticmethod(execute))
    r = cm.Sam3Runner(types.SimpleNamespace(model=object()), FakeClip(), 0.5, 2)
    img = torch.rand(H, W, 3)
    f = r.find(img, cm.image_key(img), (100, 0, 200, 100), "face:2")
    # box rows 8..35 grown by 8% of 27 px (2.16) -> rows up to 37; columns all kept; pasted at x+100
    assert f[10:37, 110:130].all() and not f[38:, :].any() and f.sum() == 27 * 20


def test_clip_to_boxes():
    m = rect_mask(0, 0, 50, 50)
    assert torch.equal(cm.clip_to_boxes(m, []), m)                                   # no boxes: unchanged
    out = cm.clip_to_boxes(m, [{"x": 10, "y": 10, "width": 10, "height": 10}, {"x": 40, "y": 0, "width": 5, "height": 5}])
    assert torch.equal(out, rect_mask(8, 8, 22, 22) | rect_mask(38, 0, 47, 7))     # each grown by at least 2 px
    assert torch.equal(cm.clip_to_boxes(m, [{"x": 1}]), m)                           # malformed: unchanged


def test_runner_needs_comfyui_sam3(monkeypatch):
    monkeypatch.setitem(sys.modules, "comfy_extras.nodes_sam3", None)
    with pytest.raises(RuntimeError, match="SAM3"):
        cm.Sam3Runner(None, None, 0.5, 2)


# ------------------------------------------------------------------------- preview and node
def test_label_map_and_preview():
    boxes, scene = frame_setup()
    f = cm.mask_frame(torch.rand(H, W, 3), boxes, details(boxes), None, FakeRunner(scene), 0.1)
    lm = cm.label_map(f)
    assert lm.shape == (H, W, 4) and lm.dtype == np.uint8 and (lm[..., 3] == 255).all()
    assert lm[30, 40, 0] == 1 and lm[30, 40, 1] == 1 and lm[30, 40, 2] == 255     # alice, face
    assert lm[26, 35, 1] == 0b11                                                  # face + eyes bits
    assert lm[30, 130, 0] == 2 and lm[30, 130, 1] == 1                            # bob's face
    assert lm[50, 95, 0] == 2 and lm[50, 95, 1] == 0 and lm[50, 95, 2] == 255     # overlap: bob's outline
    assert lm[95, 195, 0] == 0                                                    # nobody
    img = torch.full((H, W, 3), 0.5)
    p = cm.render_preview(img, f)
    assert p.shape == (H, W, 3) and torch.allclose(p[95, 100], torch.tensor([0.2, 0.2, 0.2]))   # dimmed
    assert not torch.allclose(p[30, 40], img[30, 40])                                           # tinted


def test_node_outputs_and_editor_payload(fake_sam3, monkeypatch, tmp_path):
    fp = types.ModuleType("folder_paths")
    fp.get_temp_directory = lambda: str(tmp_path)
    monkeypatch.setitem(sys.modules, "folder_paths", fp)
    from moss_regional_lora import character_masks_node as cmn
    boxes, scene = frame_setup()
    runner = FakeRunner(scene)
    monkeypatch.setattr(cmn, "_make_runner", lambda *a: runner)
    regions = json.dumps({"version": 2, "canvas": {"width": W, "height": H}, "regions": [
        {"id": b.id, "name": b.name, "bbox": list(b.bbox), "label": b.label} for b in boxes]})
    detail = json.dumps({"version": 1, "boxes": {"bob": {"parts": {"hands": {"on": True}}}}})
    req = cmn.MossCharacterMasks.INPUT_TYPES()["required"]
    assert list(req)[:4] == ["image", "regions", "sam3_model", "sam3_clip"] and list(req)[-1] == "detail"
    out = cmn.MossCharacterMasks().run(torch.rand(1, H, W, 3), regions, None, None, 0.5, 0.1, 2, detail)
    result, preview, mask = out["result"]
    assert isinstance(result, cm.CharacterMasks) and preview.shape == (1, H, W, 3) and mask.shape == (1, H, W)
    ui = out["ui"][cmn.UI_KEY][0]
    assert (ui["width"], ui["height"], ui["parts"]) == (W, H, list(cm.PART_NAMES))
    assert [b["id"] for b in ui["boxes"]] == ["alice", "bob"]
    assert ui["boxes"][1]["found"] == {"face": True, "eyes": False, "hands": False}
    from PIL import Image
    labels = np.array(Image.open(tmp_path / ui["labels"]["filename"]))
    assert labels.shape == (H, W, 4) and labels[30, 40, 0] == 1 and ui["image"]["type"] == "temp"
    assert Image.open(tmp_path / ui["image"]["filename"]).size == (W, H)


def test_node_is_registered():
    from moss_regional_lora import node
    assert node.NODE_CLASS_MAPPINGS["MossCharacterMasks"].__name__ == "MossCharacterMasks"
    assert node.NODE_CLASS_MAPPINGS["MossCharacterMasks"].OUTPUT_NODE is True
    assert node.NODE_DISPLAY_NAME_MAPPINGS["MossCharacterMasks"] == "MOSS-Image Character Masks"


def test_detail_resolution_matches_the_editor():
    """Same fixture as tests/web/detail_state.test.mjs, so the node and the editor agree on every setting."""
    import os
    path = os.path.join(os.path.dirname(__file__), "web", "fixtures", "detail_cases.json")
    for case in json.load(open(path))["cases"]:
        d = cm.resolve_detail(box("x", [0, 0, 1, 1], label=case["label"]), case["stored"])
        got = {"enabled": d.enabled, "prompt": d.prompt, "use_box_loras": d.use_box_loras,
               "loras": [list(e) for e in d.loras],
               "parts": {n: {"on": p.on, "prompt": p.prompt, "denoise": p.denoise} for n, p in d.parts.items()}}
        assert got == case["expected"], case
    assert [p.name for p in cm.PARTS] == ["face", "eyes", "hands", "hair", "skin"]
