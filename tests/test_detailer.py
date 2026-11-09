"""MOSS-Image Regional Detailer: pass planning, crops, blending and the node, with ComfyUI's sampler, VAE
and LoRA loader replaced by stand-ins."""

import sys
import types

import pytest
import torch

from moss_regional_lora import character_masks as cm
from moss_regional_lora import detailer as dt

H, W = 200, 300


def rect_mask(x0, y0, x1, y1):
    m = torch.zeros(H, W, dtype=torch.bool)
    m[y0:y1, x0:x1] = True
    return m


def character(name, parts, caption="", detail=None, loras=()):
    d = cm.resolve_detail(cm.BoxInfo(name, name, (0, 0, 1, 1), "character", caption, "#ff0000", tuple(loras)), detail)
    masks = {n: rect_mask(*r) for n, r in parts.items()}
    return cm.CharacterMask(id=name, name=name, label="character", caption=caption, color="#ff0000",
                            loras=tuple(loras), detail=d, rect=(0, 0, W, H), outline=torch.zeros(H, W, dtype=torch.bool),
                            outline_found=True, parts=masks, found={n: bool(m.any()) for n, m in masks.items()})


def two_characters():
    alice = character("alice", {"eyes": (40, 40, 60, 46), "face": (30, 30, 70, 80), "skin": (20, 20, 90, 190)},
                      caption="a tall woman", loras=[("alice.safetensors", 0.8)],
                      detail={"prompt": "green eyes", "loras": [{"lora": "eye_detail.safetensors", "weight": 0.5}],
                              "parts": {"eyes": {"denoise": 0.6}, "face": {"prompt": "smiling"}, "skin": {"on": True}}})
    bob = character("bob", {"face": (200, 30, 240, 80), "left_hand": (210, 150, 230, 170)},
                    detail={"use_box_loras": False, "parts": {"left_hand": {"on": True}}}, loras=[("bob.safetensors", 1.0)])
    bob.parts["left_hand"] = torch.zeros(H, W, dtype=torch.bool)   # not found: skipped
    return cm.CharacterMasks([cm.FrameMasks(W, H, [alice, bob])])


# ------------------------------------------------------------------------- planning
def test_plan_order_prompts_loras_and_denoise():
    passes = list(dt.plan(two_characters(), 0.35))
    assert [(p.box, p.part) for p in passes] == [("alice", "skin"), ("alice", "face"), ("alice", "eyes"), ("bob", "face")]
    skin, face, eyes, bob_face = passes
    assert face.prompt == "a tall woman, green eyes, smiling"
    assert eyes.prompt == "a tall woman, green eyes, detailed eyes"       # empty part prompt: the hint
    assert bob_face.prompt == "detailed face"                             # no caption, no detail prompt
    assert eyes.loras == (("alice.safetensors", 0.8), ("eye_detail.safetensors", 0.5))
    assert bob_face.loras == ()                                           # box LoRAs turned off
    assert (eyes.denoise, face.denoise, skin.denoise) == (0.6, 0.35, 0.35)
    assert dt.count_passes(two_characters(), 0.35) == 4
    line = dt.describe_pass(eyes)
    assert line.startswith("alice eyes (denoise 0.60;")
    assert "LoRAs: alice.safetensors 0.8, eye_detail.safetensors 0.5" in line
    assert line.endswith("prompt: a tall woman, green eyes, detailed eyes)")
    assert "LoRAs: none" in dt.describe_pass(bob_face)


# ------------------------------------------------------------------------- geometry
def test_crop_region_pads_keeps_a_minimum_and_stays_inside():
    m = rect_mask(100, 100, 120, 110)                     # 20 x 10
    assert dt.crop_region(m, 0.5, 0) == (90, 90, 130, 120)      # half the longer side (10 px) on every side
    x0, y0, x1, y1 = dt.crop_region(m, 0.0, 64)
    assert (x1 - x0, y1 - y0) == (64, 64) and x0 <= 100 and x1 >= 120
    x0, y0, x1, y1 = dt.crop_region(rect_mask(0, 0, 10, 10), 0.0, 64)   # at the corner: slid in
    assert (x0, y0, x1, y1) == (0, 0, 64, 64)
    assert dt.crop_region(rect_mask(0, 0, W, H), 0.5, 64) == (0, 0, W, H)


def test_scale_whole_numbers_capped_and_large_crops_native():
    assert dt.scale_for((0, 0, 100, 50), 1024, 4.0) == 4           # capped by max_upscale
    assert dt.scale_for((0, 0, 330, 330), 1024, 4.0) == 3           # a whole number, long side <= crop_size
    assert dt.scale_for((0, 0, 805, 1000), 1024, 4.0) == 1          # already about crop_size: its own size
    assert dt.scale_for((0, 0, 1400, 900), 1024, 4.0) == 1          # larger, but within 1.5x: still native
    assert dt.scale_for((0, 0, 2048, 1024), 1024, 4.0) == 0.5       # far larger: scaled down to crop_size
    assert dt.scale_for((0, 0, 100, 50), 1024, 2.5) == 2
    assert [dt.padded(n) for n in (1, 16, 17, 805)] == [16, 16, 32, 816]


def test_blend_mask_covers_the_part_and_fades_out():
    m = rect_mask(100, 100, 140, 140)
    soft = dt.blend_mask(m, 4, 12)
    assert (soft[m] == 1).all()                            # the part itself is fully replaced
    assert 0 < soft[120, 146] < 1                          # feathered edge just outside the grown mask
    assert soft[120, 180] == 0 and soft[10, 10] == 0
    assert torch.equal(dt.blend_mask(m, 0, 0), m.float())


# ------------------------------------------------------------------------- detailing
class FakeSampler:
    """Paints each crop a colour per pass and records what it was given."""

    def __init__(self):
        self.calls = []

    def __call__(self, pixels, noise_mask, p, seed):
        self.calls.append({"shape": tuple(pixels.shape), "mask": tuple(noise_mask.shape), "part": p.part,
                           "seed": seed, "prompt": p.prompt, "mask_max": float(noise_mask.max())})
        return torch.full_like(pixels, 0.1 * (len(self.calls))), None


def test_detail_changes_only_the_masked_parts():
    masks = two_characters()
    img = torch.rand(1, H, W, 3)
    s = FakeSampler()
    out = dt.detail(img, masks, s, denoise=0.35, crop_size=256, padding=0.3, max_upscale=4.0, grow=4, feather=8, seed=7)
    assert out.shape == img.shape
    touched = torch.zeros(H, W, dtype=torch.bool)
    for p in dt.plan(masks, 0.35):
        touched |= dt.blend_mask(p.mask, 4, 8) > 0
    assert torch.equal(out[0][~touched], img[0][~touched])            # bit-identical outside every mask
    assert torch.allclose(out[0, 42, 50], torch.full((3,), 0.3))       # alice's eyes: the last pass over her
    assert torch.allclose(out[0, 50, 220], torch.full((3,), 0.4))      # bob's face
    assert [c["part"] for c in s.calls] == ["skin", "face", "eyes", "face"]
    assert [c["seed"] for c in s.calls] == [7, 8, 9, 10]
    for c in s.calls:
        h, w, _ = c["shape"]
        assert h % 16 == 0 and w % 16 == 0 and c["mask"] == (h, w) and c["mask_max"] <= 1.0
    eyes = s.calls[2]
    assert max(eyes["shape"][:2]) <= 256 + 15 and min(eyes["shape"][:2]) >= 2 * 64   # >= 64 px context, upscaled


def test_each_pass_sees_the_image_as_detailed_so_far():
    masks = two_characters()
    seen = []

    def sampler(pixels, noise_mask, p, seed):
        seen.append(float(pixels.mean()))
        return torch.ones_like(pixels), None
    img = torch.zeros(1, H, W, 3)
    dt.detail(img, masks, sampler, denoise=0.35, crop_size=256, padding=0.3, max_upscale=4.0, grow=0, feather=0, seed=0)
    assert seen[0] == 0 and seen[1] > 0                     # the face crop already holds the redrawn skin


def sharp_image():
    torch.manual_seed(0)
    checker = ((torch.arange(W)[None, :, None] // 3 + torch.arange(H)[:, None, None] // 3) % 2).float()
    return (torch.rand(1, H, W, 3) * 0.5 + checker * 0.5).clamp(0, 1)


def test_an_unchanged_redraw_returns_the_image_unchanged():
    """A sampler that changes nothing (like denoise 0): with detail restoration the image comes back as it
    was, even through upscaled crops; without it, only crops sampled at their own size are exact."""
    masks, img = two_characters(), sharp_image()
    identity = lambda pixels, noise_mask, p, seed: (pixels.clone(), pixels.clone())
    out = dt.detail(img, masks, identity, denoise=0.0, crop_size=256, padding=0.3, max_upscale=4.0, grow=4,
                    feather=8, seed=0, restore=1.0)
    assert torch.allclose(out, img, atol=1e-6)
    no_ref = lambda pixels, noise_mask, p, seed: (pixels.clone(), None)
    big = character("big", {"skin": (10, 10, 290, 190)})            # crop ~ the whole image: native size
    one = cm.CharacterMasks([cm.FrameMasks(W, H, [big])])
    out = dt.detail(img, one, no_ref, denoise=0.0, crop_size=256, padding=0.3, max_upscale=4.0, grow=4,
                    feather=8, seed=0, restore=1.0)
    assert torch.allclose(out, img, atol=1e-6)                       # padded, never resampled


def test_restoration_keeps_the_models_change_on_the_original_detail():
    masks, img = two_characters(), sharp_image()

    def bluer(pixels, noise_mask, p, seed):          # the round trip blurs; the model shifts the colour
        ref = dt.gaussian_blur(pixels[..., 0], 1.5)[..., None].repeat(1, 1, 3)
        return ref + torch.tensor([0.0, 0.0, 0.2]), ref
    out = dt.detail(img, masks, bluer, denoise=0.35, crop_size=256, padding=0.3, max_upscale=4.0, grow=0,
                    feather=0, seed=0, restore=1.0)
    inside = masks.frames[0].boxes[1].parts["face"]                  # bob's face: one pass
    expected = (img[0] + torch.tensor([0.0, 0.0, 0.2])).clamp(0, 1)
    assert torch.allclose(out[0][inside], expected[inside], atol=1e-4)   # sharp original + the change


def test_frames_without_passes_pass_through():
    masks = cm.CharacterMasks([cm.FrameMasks(W, H, [])])
    img = torch.rand(1, H, W, 3)
    assert torch.equal(dt.detail(img, masks, FakeSampler(), denoise=0.35, crop_size=256, padding=0.3,
                                 max_upscale=4.0, grow=4, feather=8, seed=0), img)


# ------------------------------------------------------------------------- the node
class FakeVAE:
    def encode(self, pixels):                 # [1, h, w, 3] -> [1, 4, h/16, w/16]
        assert pixels.shape[1] % 16 == 0 and pixels.shape[2] % 16 == 0
        return pixels.permute(0, 3, 1, 2)[:, :, ::16, ::16].repeat(1, 2, 1, 1)[:, :4]

    decodes = 0

    def decode(self, latent):                 # back to pixels, as a flat colour from the latent
        FakeVAE.decodes += 1
        b, c, h, w = latent.shape
        return latent[:, :3].repeat_interleave(16, 2).repeat_interleave(16, 3).permute(0, 2, 3, 1).clamp(0, 1)


class FakeClip:
    def __init__(self):
        self.encoded = []

    def tokenize(self, text):
        return text

    def encode_from_tokens_scheduled(self, tokens):
        self.encoded.append(tokens)
        return [[f"cond:{tokens}", {}]]


class FakeDM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)


class FakeModel:
    """A ModelPatcher stand-in: clone() and add_patches() record the patches applied."""

    def __init__(self, patches=()):
        self.patches = list(patches)
        self.model = types.SimpleNamespace()

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return FakeDM()

    def clone(self):
        return FakeModel(self.patches)

    def add_patches(self, patches, strength_patch=1.0):
        self.patches.append((dict(patches), strength_patch))
        return list(patches)


@pytest.fixture
def fake_comfy(monkeypatch, tmp_path):
    calls = {"ksampler": [], "lora": [], "loaded": []}

    def common_ksampler(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent, denoise=1.0):
        calls["ksampler"].append({"model": model, "seed": seed, "steps": steps, "cfg": cfg, "sampler": sampler_name,
                                  "scheduler": scheduler, "positive": positive, "negative": negative,
                                  "mask": tuple(latent["noise_mask"].shape), "latent": tuple(latent["samples"].shape),
                                  "denoise": denoise})
        return ({"samples": torch.full_like(latent["samples"], 0.9)},)

    def load_lora(sd, key_map):
        """ComfyUI's load_lora: patches for the LoRA keys the key map resolves."""
        calls["lora"].append(sd["name"])
        return {key_map[p]: f"patch:{sd['name']}" for p in key_map
                if any(k.startswith(p + ".") for k in sd if k != "name")}

    for f in ("alice.safetensors", "bob.safetensors", "eye_detail.safetensors", "other_model.safetensors"):
        (tmp_path / f).write_bytes(b"x")

    def raw_lora(path):
        name = path.split("/")[-1]
        layer = "unet_only.layer" if name == "other_model.safetensors" else "transformer.lin"   # diffusers-style names
        return {"name": name, f"{layer}.lora_down.weight": 0, f"{layer}.lora_up.weight": 0}
    mods = {
        "nodes": types.SimpleNamespace(common_ksampler=common_ksampler),
        "comfy": types.ModuleType("comfy"),
        # ComfyUI's own key map doesn't know "transformer." names; the main node's adds them
        "comfy.lora": types.SimpleNamespace(load_lora=load_lora, model_lora_keys_unet=lambda model, km: km),
        "comfy.lora_convert": types.SimpleNamespace(convert_lora=lambda sd: sd),
        "comfy.utils": types.SimpleNamespace(
            load_torch_file=lambda path, safe_load=True: calls["loaded"].append(path) or raw_lora(path),
            ProgressBar=lambda total: types.SimpleNamespace(update=lambda n: None)),
        "folder_paths": types.SimpleNamespace(get_full_path_or_raise=lambda folder, name: str(tmp_path / name)),
        "comfy_execution": types.ModuleType("comfy_execution"),
        "comfy_execution.graph_utils": types.SimpleNamespace(ExecutionBlocker=type("ExecutionBlocker", (), {
            "__init__": lambda self, message: setattr(self, "message", message)})),
    }
    mods["comfy"].lora, mods["comfy"].utils = mods["comfy.lora"], mods["comfy.utils"]
    mods["comfy"].lora_convert = mods["comfy.lora_convert"]
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    from moss_regional_lora import detailer_node
    detailer_node._RAW_LORAS.clear()
    return detailer_node, calls


def run_node(node_mod, masks, **kw):
    args = dict(image=torch.rand(1, H, W, 3), character_masks=masks, model=FakeModel(), clip=FakeClip(), vae=FakeVAE(),
                seed=5, steps=12, cfg=4.5, sampler_name="euler", scheduler="simple", denoise=0.35, crop_size=256,
                padding=0.3, mask_grow=4, feather=8, stop_before_detailing=False)
    args.update(kw)
    return node_mod.MossRegionalDetailer().run(**args), args


def test_node_samples_each_part_with_its_loras_prompt_and_negative(fake_comfy):
    node_mod, calls = fake_comfy
    (out, previews), args = run_node(node_mod, two_characters(), negative="NEG", restore_detail=0.0)
    assert previews.shape[0] == 4 and previews.shape[-1] == 3                # one before/after per pass
    assert out.shape == (1, H, W, 3)
    ks = calls["ksampler"]
    assert len(ks) == 4 and [k["seed"] for k in ks] == [5, 6, 7, 8]
    assert all(k["negative"] == "NEG" and k["steps"] == 12 and k["cfg"] == 4.5 for k in ks)
    assert ks[2]["positive"] == [["cond:a tall woman, green eyes, detailed eyes", {}]] and ks[2]["denoise"] == 0.6
    # alice: box + detail LoRAs, matched through the main node's key map ("transformer.lin" -> the layer)
    assert ks[0]["model"].patches == [({"diffusion_model.lin.weight": "patch:alice.safetensors"}, 0.8),
                                      ({"diffusion_model.lin.weight": "patch:eye_detail.safetensors"}, 0.5)]
    assert ks[0]["model"] is ks[1]["model"] and ks[3]["model"] is args["model"]     # bob: box LoRAs off
    assert calls["lora"] == ["alice.safetensors", "eye_detail.safetensors"]          # patched once, reused
    assert args["model"].patches == []                                                # the input isn't touched
    for k in ks:                                                    # noise mask at pixel size, latent 1/16
        _, _, mh, mw = k["mask"]
        assert k["latent"][2:] == (mh // 16, mw // 16)
    assert torch.allclose(out[0, 42, 50], torch.full((3,), 0.9))    # redrawn inside the part
    assert torch.equal(out[0, 195, 295], args["image"][0, 195, 295])   # untouched far away


def test_node_decodes_a_reference_for_detail_restoration(fake_comfy):
    node_mod, calls = fake_comfy
    FakeVAE.decodes = 0
    run_node(node_mod, two_characters(), restore_detail=0.0)
    assert FakeVAE.decodes == 4                                      # one per pass
    FakeVAE.decodes = 0
    run_node(node_mod, two_characters())                             # default restore_detail = 1
    assert FakeVAE.decodes == 8                                      # plus the unredrawn reference


def test_node_without_negative_encodes_an_empty_one(fake_comfy):
    node_mod, calls = fake_comfy
    clip = FakeClip()
    run_node(node_mod, two_characters(), clip=clip)
    assert clip.encoded[0] == "" and calls["ksampler"][0]["negative"] == [["cond:", {}]]


def test_node_stop_before_detailing_blocks_silently(fake_comfy):
    node_mod, calls = fake_comfy
    (out, previews), _ = run_node(node_mod, two_characters(), stop_before_detailing=True)
    assert type(out).__name__ == "ExecutionBlocker" and out.message is None
    assert type(previews).__name__ == "ExecutionBlocker"
    assert calls["ksampler"] == []


def test_node_with_nothing_to_detail_passes_the_image_through(fake_comfy):
    node_mod, calls = fake_comfy
    img = torch.rand(1, H, W, 3)
    (out, previews), _ = run_node(node_mod, cm.CharacterMasks([cm.FrameMasks(W, H, [])]), image=img)
    assert out is img and calls["ksampler"] == [] and previews.shape == (1, 64, 64, 3)


def test_node_is_registered():
    from moss_regional_lora import node
    cls = node.NODE_CLASS_MAPPINGS["MossRegionalDetailer"]
    assert node.NODE_DISPLAY_NAME_MAPPINGS["MossRegionalDetailer"] == "MOSS-Image Regional Detailer"
    req, opt = cls.INPUT_TYPES()["required"], cls.INPUT_TYPES()["optional"]
    assert req["character_masks"][0] == "MOSS_CHARACTER_MASKS" and "stop_before_detailing" in req
    assert opt["negative"][0] == "CONDITIONING" and cls.RETURN_TYPES == ("IMAGE", "IMAGE")
    assert cls.RETURN_NAMES == ("image", "passes")


# ------------------------------------------------------------------------- pieces far apart
def test_split_mask_far_apart_pieces_and_near_ones():
    hands = rect_mask(10, 150, 30, 170) | rect_mask(260, 20, 280, 40)          # opposite corners
    pieces = dt.split_mask(hands, 0.3, 64, 18)
    assert len(pieces) == 2
    assert torch.equal(pieces[0], rect_mask(260, 20, 280, 40))                   # reading order: top first
    assert torch.equal(pieces[1], rect_mask(10, 150, 30, 170))
    assert torch.equal(pieces[0] | pieces[1], hands)
    eyes = rect_mask(100, 50, 110, 55) | rect_mask(125, 50, 135, 55)              # close: one pass
    assert len(dt.split_mask(eyes, 0.3, 64, 18)) == 1
    fragments = rect_mask(10, 150, 30, 160) | rect_mask(10, 163, 30, 170)         # one hand in two bits
    assert len(dt.split_mask(fragments, 0.3, 0, 0)) == 1
    one = rect_mask(10, 10, 50, 50)
    assert dt.split_mask(one, 0.3, 64, 18)[0] is one                              # a single piece: as it is


def test_hands_far_apart_get_their_own_passes_and_upscaled_crops():
    bob = character("bob", {"left_hand": (10, 150, 30, 170)},
                    detail={"parts": {"face": {"on": False}, "eyes": {"on": False}, "left_hand": {"on": True}}})
    bob.parts["left_hand"] = rect_mask(10, 150, 30, 170) | rect_mask(260, 20, 280, 40)     # (a wide split mask)
    masks = cm.CharacterMasks([cm.FrameMasks(W, H, [bob])])
    passes = list(dt.plan(masks, 0.35, 0.3, 64, 12))
    assert [(p.part, int(p.mask.sum())) for p in passes] == [("left_hand", 400), ("left_hand", 400)]
    assert dt.count_passes(masks, 0.35, 0.3, 64, 12) == 2
    s = FakeSampler()
    img = torch.rand(1, H, W, 3)
    out = dt.detail(img, masks, s, denoise=0.35, crop_size=256, padding=0.3, max_upscale=4.0, grow=4, feather=8, seed=3)
    assert [c["seed"] for c in s.calls] == [3, 4]
    for c in s.calls:                       # each a small crop around one hand, upscaled (was one crop spanning both)
        assert max(c["shape"][:2]) <= 256 + 15 and min(c["shape"][:2]) >= 2 * 64
    assert torch.allclose(out[0, 160, 20], torch.full((3,), 0.2))               # second pass: the lower hand
    assert torch.allclose(out[0, 30, 270], torch.full((3,), 0.1))



def test_node_warns_when_a_lora_matches_no_layer(fake_comfy, caplog):
    node_mod, calls = fake_comfy
    alice = character("alice", {"face": (30, 30, 70, 80)}, loras=[("other_model.safetensors", 1.0)])
    masks = cm.CharacterMasks([cm.FrameMasks(W, H, [alice])])
    run_node(node_mod, masks)
    assert "no layer of the model matched other_model.safetensors" in caplog.text


def test_per_part_restore_overrides_the_node_default():
    alice = character("alice", {"face": (30, 30, 70, 80), "right_hand": (200, 140, 230, 170)},
                      detail={"parts": {"face": {"restore": 0.0}, "right_hand": {"on": True}}})
    masks = cm.CharacterMasks([cm.FrameMasks(W, H, [alice])])
    passes = list(dt.plan(masks, 0.35))
    assert [(p.part, p.restore) for p in passes] == [("face", 0.0), ("right_hand", None)]
    img = sharp_image()

    def bluer(pixels, noise_mask, p, seed):          # the round trip blurs; the model shifts the colour
        ref = dt.gaussian_blur(pixels[..., 0], 1.5)[..., None].repeat(1, 1, 3)
        return ref + torch.tensor([0.0, 0.0, 0.2]), ref
    out = dt.detail(img, masks, bluer, denoise=0.35, crop_size=256, padding=0.3, max_upscale=4.0, grow=0,
                    feather=0, seed=0, restore=1.0)
    face, hand = alice.parts["face"], alice.parts["right_hand"]
    sharp = (img[0] + torch.tensor([0.0, 0.0, 0.2])).clamp(0, 1)
    assert not torch.allclose(out[0][face], sharp[face], atol=1e-3)             # face: restore 0, left blurred
    assert torch.allclose(out[0][hand], sharp[hand], atol=1e-4)                 # hand: node default, restored
    assert dt.needs_reference(masks, 0.35, 0.0) is False
    masks.frames[0].boxes[0] = character("alice", {"face": (30, 30, 70, 80)}, detail={"parts": {"face": {"restore": 0.5}}})
    assert dt.needs_reference(masks, 0.35, 0.0) is True


def test_whole_crop_redraws_the_surroundings_but_pastes_back_only_the_part():
    """A part set to whole crop gets a noise mask covering the whole crop (its context is redrawn too),
    yet the image still changes only under the part's feathered mask."""
    bob = character("bob", {"face": (200, 30, 240, 80), "left_hand": (210, 150, 230, 170)},
                    detail={"parts": {"left_hand": {"on": True, "whole_crop": True}}})
    masks = cm.CharacterMasks([cm.FrameMasks(W, H, [bob])])
    passes = list(dt.plan(masks, 0.5))
    assert [(p.part, p.whole_crop) for p in passes] == [("face", False), ("left_hand", True)]
    assert "whole crop" in dt.describe_pass(passes[1]) and "whole crop" not in dt.describe_pass(passes[0])
    s = FakeSampler()
    masks_seen = []
    orig = s.__call__

    def spy(pixels, noise_mask, p, seed):
        masks_seen.append((p.part, float(noise_mask.min()), float(noise_mask.max())))
        return orig(pixels, noise_mask, p, seed)
    img = torch.rand(1, H, W, 3)
    out = dt.detail(img, masks, spy, denoise=0.5, crop_size=256, padding=0.3, max_upscale=4.0, grow=4, feather=8,
                    seed=0)
    face, hand = masks_seen
    assert face[1] == 0.0                                  # face: only the face is noised
    assert hand[1] == 1.0 and hand[2] == 1.0               # hand: the whole crop is
    touched = torch.zeros(H, W, dtype=torch.bool)
    for p in passes:
        touched |= dt.blend_mask(p.mask, 4, 8) > 0
    assert torch.equal(out[0][~touched], img[0][~touched])  # still nothing changes outside the masks


def test_grown_feathered_edge_stops_at_other_characters():
    """A thin part right next to another character: growing and feathering its mask would spread onto the
    neighbour; the pass must leave every pixel the neighbour owns exactly as it was."""
    real = character("real", {"left_hand": (100, 50, 106, 120)}, detail={"parts": {"left_hand": {"on": True}}})
    anime = character("anime", {"left_hand": (107, 50, 160, 120)}, detail={"parts": {"left_hand": {"on": False}}})
    anime.parts.pop("left_hand")                           # not detailed: only its ownership matters here
    owner = torch.full((H, W), -1, dtype=torch.int64)
    owner[:, :106] = 0                                     # real owns the left, anime the right
    owner[:, 107:] = 1
    masks = cm.CharacterMasks([cm.FrameMasks(W, H, [real, anime], owner)])
    (p,) = list(dt.plan(masks, 0.5))
    assert p.keep_out is not None and bool(p.keep_out[60, 130]) and not bool(p.keep_out[60, 103])
    img = torch.rand(1, H, W, 3)
    seen = []

    def sampler(pixels, noise_mask, p, seed):
        seen.append(noise_mask)
        return torch.ones_like(pixels), None
    out = dt.detail(img, masks, sampler, denoise=0.5, crop_size=256, padding=0.3, max_upscale=4.0, grow=6,
                    feather=12, seed=0)
    assert torch.equal(out[0][owner == 1], img[0][owner == 1])        # the neighbour is untouched
    assert torch.allclose(out[0, 80, 103], torch.ones(3))             # the part itself is redrawn
    assert bool((out[0, 80, 106] != img[0, 80, 106]).any())           # unowned edge pixels may still blend
    # without an ownership map (older results) the edge spreads as before
    masks.frames[0].owner = None
    out2 = dt.detail(img, masks, sampler, denoise=0.5, crop_size=256, padding=0.3, max_upscale=4.0, grow=6,
                     feather=12, seed=0)
    assert not torch.equal(out2[0][owner == 1], img[0][owner == 1])


def test_pass_previews_show_each_crop_before_and_after():
    masks = two_characters()
    keep = []
    img = torch.rand(1, H, W, 3)
    dt.detail(img, masks, FakeSampler(), denoise=0.35, crop_size=256, padding=0.3, max_upscale=4.0, grow=4,
              feather=8, seed=0, keep=keep)
    assert len(keep) == 4                                               # one per pass, in log order
    for k, t in enumerate(keep):
        h, w2, c = t.shape
        assert c == 3 and w2 % 2 == 0
        assert torch.allclose(t[:, w2 // 2:], torch.full((h, w2 // 2, 3), 0.1 * (k + 1)))   # right: the redraw
    batch = dt.stack_previews(keep)
    assert batch.shape[0] == 4 and batch.shape[1] == max(t.shape[0] for t in keep)
    assert dt.stack_previews([]).shape == (1, 64, 64, 3)
