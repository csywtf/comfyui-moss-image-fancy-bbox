"""The node against a minimal stand-in for ComfyUI's ModelPatcher API (FR-1, FR-10, FR-11, NFR-4).

The stand-in reproduces only the parts of comfy.model_patcher / comfy.patcher_extension the node
touches (clone, set_injections, add/remove_wrappers_with_key, inject_model/eject_model). It is a
contract test of how the node uses that API, not a test of ComfyUI itself.
"""

import functools
import json
import math
import sys
import types

import pytest
import torch

import toy
from toy import FULL, ToyDiT, bit_identical, context, latent, topts
from moss_regional_lora import layout_config
from moss_regional_lora.layout import Layout, LayoutNotEstablished

LORA_NAME = "char.safetensors"


class PatcherInjection:
    def __init__(self, inject, eject):
        self.inject, self.eject = inject, eject


class WrappersMP:
    APPLY_MODEL = "apply_model"


class FakePatcher:
    def __init__(self, dm):
        self.dm = dm
        self.model = types.SimpleNamespace(state_dict=dm.state_dict)
        self.injections = {}
        self.wrappers = {}
        self.is_injected = False

    def clone(self):
        n = FakePatcher(self.dm)
        n.model = self.model                      # like ModelPatcher.clone: the BaseModel is shared
        n.injections = {k: list(v) for k, v in self.injections.items()}
        n.wrappers = {t: {k: list(v) for k, v in d.items()} for t, d in self.wrappers.items()}
        n.is_injected = self.is_injected
        return n

    def get_model_object(self, name):
        assert name == "diffusion_model"
        return self.dm

    def set_injections(self, key, injections):
        self.injections[key] = injections

    def remove_injections(self, key):
        self.injections.pop(key, None)

    def add_wrapper_with_key(self, wtype, key, fn):
        self.wrappers.setdefault(wtype, {}).setdefault(key, []).append(fn)

    def remove_wrappers_with_key(self, wtype, key):
        self.wrappers.get(wtype, {}).pop(key, None)

    def inject_model(self):
        if self.is_injected:
            return
        for injs in self.injections.values():
            for i in injs:
                i.inject(self)
                self.is_injected = True

    def eject_model(self):
        if not self.is_injected:
            return
        for injs in self.injections.values():
            for i in injs:
                i.eject(self)
        self.is_injected = False

    def apply_model(self, x, ctx, transformer_options):
        """Mimics BaseModel.apply_model running the APPLY_MODEL wrappers around the diffusion model."""
        fns = [f for d in self.wrappers.get(WrappersMP.APPLY_MODEL, {}).values() for f in d]

        def inner(x, t, c_concat, c_crossattn, control, transformer_options, **kw):
            return self.dm(x, c_crossattn)

        call = inner
        for f in reversed(fns):
            call = functools.partial(f, _Exec(call))
        return call(x, None, None, ctx, None, transformer_options)


class _Exec:
    class_obj = None

    def __init__(self, fn):
        self.fn = fn

    def __call__(self, *a, **k):
        return self.fn(*a, **k)


@pytest.fixture
def fake_comfy(monkeypatch, tmp_path):
    torch.manual_seed(0)
    dm = ToyDiT(Layout("suffix", "raster", "none")).double()
    lora_file = tmp_path / LORA_NAME
    lora_file.write_bytes(b"x")
    m = dm.blocks[0].qkv
    g = torch.Generator().manual_seed(5)
    sd = {"lora_unet_blocks_0_qkv.lora_down.weight": torch.randn(2, m.in_features, generator=g, dtype=torch.float64),
          "lora_unet_blocks_0_qkv.lora_up.weight": torch.randn(m.out_features, 2, generator=g, dtype=torch.float64),
          "lora_unet_blocks_0_qkv.alpha": torch.tensor(2.0),
          "lora_unet_nonexistent.lora_down.weight": torch.zeros(2, 2),
          "lora_unet_nonexistent.lora_up.weight": torch.zeros(2, 2)}

    comfy = types.ModuleType("comfy")
    pe = types.ModuleType("comfy.patcher_extension")
    pe.PatcherInjection, pe.WrappersMP = PatcherInjection, WrappersMP
    utils = types.ModuleType("comfy.utils")
    utils.load_torch_file = lambda path, safe_load=True: dict(sd)
    fp = types.ModuleType("folder_paths")
    fp.get_full_path_or_raise = lambda folder, name: str(tmp_path / name)
    fp.get_full_path = lambda folder, name: str(tmp_path / name) if (tmp_path / name).exists() else None
    comfy.patcher_extension, comfy.utils = pe, utils
    for name, mod in (("comfy", comfy), ("comfy.patcher_extension", pe), ("comfy.utils", utils), ("folder_paths", fp)):
        monkeypatch.setitem(sys.modules, name, mod)
    from moss_regional_lora import node
    node._LORA_CACHE.clear()
    return FakePatcher(dm), node


def regions_json(*regions):
    return json.dumps({"version": 1, "regions": [
        {"id": f"r{i}", "name": f"r{i}", "bbox": list(b), "lora": LORA_NAME, "weight": w, "visible": True}
        for i, (b, w) in enumerate(regions)]})


# All three layout options at "not set": exercises the refusal path now that the node has measured defaults.
UNSET = dict(text_position="not set", image_order="not set", source_tokens="not set")


def apply(node, patcher, regions, strict=True, **kw):
    args = dict(width=toy.W, height=toy.H, regions=regions,
                ramp_start=0.0, ramp_end=0.0, steps=10, strict=strict)
    args.update(kw)
    return node.MossRegionalLora().apply(patcher, **args)


def set_layout(monkeypatch, t="suffix", o="raster", s="none"):
    monkeypatch.setattr(layout_config, "TEXT_POSITION", t)
    monkeypatch.setattr(layout_config, "IMAGE_ORDER", o)
    monkeypatch.setattr(layout_config, "SOURCE_TOKENS", s)


def test_cache_fingerprint_follows_the_lora_files(fake_comfy, tmp_path):
    """Same inputs and unchanged LoRA files: same fingerprint, so ComfyUI reuses the cached result and
    nothing after the node re-runs. Overwriting a used LoRA file changes it (NFR-4)."""
    _, node = fake_comfy
    fp = node.MossRegionalLora.IS_CHANGED
    regions = regions_json((FULL, 1.0))
    first = fp(regions=regions, width=toy.W)
    assert isinstance(first, str) and LORA_NAME in first
    assert fp(regions=regions, width=toy.W) == first
    lora = tmp_path / LORA_NAME
    lora.write_bytes(b"a different, longer file")
    assert fp(regions=regions) != first
    # hidden boxes don't load their file, so they don't count; no LoRA at all is a stable empty fingerprint
    hidden = json.dumps({"version": 2, "regions": [{"id": "h", "bbox": list(FULL), "visible": False,
                                                    "loras": [{"lora": LORA_NAME, "weight": 1}]}]})
    assert fp(regions=hidden) == fp(regions=json.dumps({"version": 2, "regions": []})) == ""
    # a LoRA that isn't on disk is named as missing (the run itself raises)
    assert "missing" in fp(regions=regions_json((FULL, 1.0)).replace(LORA_NAME, "gone.safetensors"))
    # regions wired from another node can't be fingerprinted here: always re-run, as before
    assert math.isnan(fp(width=1))
    # unparsable text: re-run whenever it changes (the run reports the error)
    assert fp(regions="{oops") == "{oops"


def test_cached_result_is_reusable(fake_comfy, monkeypatch):
    """Reusing one execution's model for a second run (what a cache hit does) gates exactly as before."""
    patcher, node = fake_comfy
    set_layout(monkeypatch)
    base = patcher.dm(latent(), context())
    m = apply(node, patcher, regions_json((FULL, 1.0)))["result"][0]
    outs = []
    for _ in range(2):                       # two queue runs on the same cached patcher
        m.inject_model()
        outs.append(m.apply_model(latent(), context(), topts()))
        m.eject_model()
        assert bit_identical(patcher.dm(latent(), context()), base)   # nothing left behind between runs
    assert bit_identical(outs[0], outs[1]) and not bit_identical(outs[0], base)


def test_strict_refuses_with_unset_layout(fake_comfy):
    patcher, node = fake_comfy
    with pytest.raises(LayoutNotEstablished, match="discover_moss_layout"):
        apply(node, patcher, regions_json((FULL, 1.0)), **UNSET)


def test_non_strict_unset_layout_passes_model_through_unpatched(fake_comfy):
    patcher, node = fake_comfy
    out = apply(node, patcher, regions_json((FULL, 1.0)), strict=False, **UNSET)
    m = out["result"][0]
    assert m is not patcher and not m.injections and not any(m.wrappers.values())
    assert out["ui"]["moss_regional_lora"][0]["status"] == "inactive"


def test_applies_reports_and_does_not_stack(fake_comfy, monkeypatch):
    patcher, node = fake_comfy
    set_layout(monkeypatch)
    out = apply(node, patcher, regions_json((FULL, 1.0)))
    m1 = out["result"][0]
    summary = out["ui"]["moss_regional_lora"][0]
    assert summary["applied"] == 1 and summary["skipped"] == 2   # the two 'nonexistent' keys
    assert list(m1.injections) == ["moss_regional_lora"]
    assert len(m1.wrappers["apply_model"]["moss_regional_lora"]) == 1
    # chaining the node on its own output replaces instead of stacking
    m2 = apply(node, m1, regions_json((FULL, 1.0)))["result"][0]
    assert len(m2.injections["moss_regional_lora"]) == 1
    assert len(m2.wrappers["apply_model"]["moss_regional_lora"]) == 1
    # the original model object is untouched
    assert not patcher.injections and not patcher.wrappers


def test_end_to_end_through_wrapper_and_injection(fake_comfy, monkeypatch):
    patcher, node = fake_comfy
    set_layout(monkeypatch)
    lat, ctx = latent(2), context(2)
    opts = topts((0, 1), step=5)
    base = patcher.dm(lat, ctx)
    m = apply(node, patcher, regions_json((FULL, 1.0)))["result"][0]
    m.inject_model()
    out = m.apply_model(lat, ctx, opts)
    assert bit_identical(out[1], base[1])           # uncond row clean
    assert not torch.equal(out[0], base[0])         # cond row changed
    # re-execution: second node run, ComfyUI ejects the old patcher and injects the new one
    m.eject_model()
    m_again = apply(node, patcher, regions_json((FULL, 1.0)))["result"][0]
    m_again.inject_model()
    assert bit_identical(m_again.apply_model(lat, ctx, opts), out)
    m_again.eject_model()
    assert bit_identical(patcher.dm(lat, ctx), base)  # hooks fully removed


def test_reexecution_without_eject_still_single_application(fake_comfy, monkeypatch):
    """Even if the old patcher is never ejected, a new run must not double-apply."""
    patcher, node = fake_comfy
    set_layout(monkeypatch)
    lat, ctx = latent(1), context(1)
    opts = topts((0,), step=5)
    m1 = apply(node, patcher, regions_json((FULL, 1.0)))["result"][0]
    m1.inject_model()
    once = m1.apply_model(lat, ctx, opts)
    m2 = apply(node, patcher, regions_json((FULL, 1.0)))["result"][0]
    m2.inject_model()
    assert bit_identical(m2.apply_model(lat, ctx, opts), once)
    hooks = [h for h in patcher.dm.blocks[0].qkv._forward_hooks.values() if type(h).__name__ == "_ModuleGate"]
    assert len(hooks) == 1


def test_no_regions_strips_previous(fake_comfy, monkeypatch):
    patcher, node = fake_comfy
    set_layout(monkeypatch)
    m1 = apply(node, patcher, regions_json((FULL, 1.0)))["result"][0]
    m2 = apply(node, m1, json.dumps({"version": 1, "regions": []}))["result"][0]
    assert not m2.injections and not m2.wrappers["apply_model"]


# ------------------------------------------------------------------ layout prompt output
def prompt_box(bbox=(-10, -10, 10, 10), **kw):
    r = {"bbox": list(bbox), "caption": "a cat", "visible": True}
    r.update(kw)
    return r


def test_prompt_only_boxes_work_in_strict_mode_without_discovery(fake_comfy):
    patcher, node = fake_comfy
    out = apply(node, patcher, json.dumps({"version": 1, "regions": [prompt_box()]}), caption="a garden")
    m, prompt = out["result"][:2]
    layout = json.loads(prompt)
    assert layout["summary"] == {"caption": "a garden"}
    assert layout["items"] == [{"label": "object", "caption": "a cat", "bbox": [-10, -10, 10, 10]}]
    assert not m.injections and out["ui"]["moss_regional_lora"][0]["status"] == "prompt-only"


def test_lora_box_still_refused_in_strict_mode_without_discovery(fake_comfy):
    patcher, node = fake_comfy
    with pytest.raises(LayoutNotEstablished):
        apply(node, patcher, json.dumps({"version": 1, "regions": [prompt_box(lora=LORA_NAME)]}), caption="x",
              **UNSET)


def test_non_strict_unset_layout_still_outputs_prompt(fake_comfy):
    patcher, node = fake_comfy
    out = apply(node, patcher, json.dumps({"version": 1, "regions": [prompt_box(lora=LORA_NAME)]}),
                strict=False, caption="x", **UNSET)
    m, prompt = out["result"][:2]
    assert not m.injections and json.loads(prompt)["items"][0]["caption"] == "a cat"


def test_prompt_and_lora_together(fake_comfy, monkeypatch):
    patcher, node = fake_comfy
    set_layout(monkeypatch)
    regions = json.dumps({"version": 1, "regions": [
        prompt_box(FULL, label="character", caption="Alice", lora=LORA_NAME),
        prompt_box(caption="", lora=LORA_NAME),            # LoRA only
        prompt_box(caption="a red ball"),                  # prompt only
    ]})
    out = apply(node, patcher, regions, caption="scene")
    m, prompt = out["result"][:2]
    assert out["ui"]["moss_regional_lora"][0]["regions"] == 2
    assert [i["caption"] for i in json.loads(prompt)["items"]] == ["Alice", "a red ball"]
    assert list(m.injections) == ["moss_regional_lora"]


def test_outputs_and_caption_input_declared(fake_comfy):
    _, node = fake_comfy
    cls = node.MossRegionalLora
    assert cls.RETURN_TYPES[:2] == ("MODEL", "STRING")
    assert list(cls.INPUT_TYPES()["required"])[-1] == "caption"   # appended: old workflows keep widget order


# ------------------------------------------------------------------ layout as advanced options
def test_advanced_options_supply_the_layout(fake_comfy):
    """With the three options set on the node, LoRA boxes run even though layout_config.py is unset."""
    patcher, node = fake_comfy
    out = apply(node, patcher, regions_json((FULL, 1.0)),
                text_position="suffix", image_order="raster", source_tokens="none")
    assert out["ui"]["moss_regional_lora"][0]["status"] == "active"
    assert list(out["result"][0].injections) == ["moss_regional_lora"]


def test_partly_set_options_name_only_the_missing_ones(fake_comfy):
    patcher, node = fake_comfy
    with pytest.raises(LayoutNotEstablished) as e:
        apply(node, patcher, regions_json((FULL, 1.0)), text_position="suffix", image_order="raster",
              source_tokens="not set")
    msg = str(e.value)
    assert "source_tokens" in msg and "text_position" not in msg
    assert "MOSS-Image Layout Discovery" in msg


def test_not_set_options_fall_back_to_layout_config(fake_comfy, monkeypatch):
    patcher, node = fake_comfy
    set_layout(monkeypatch, s="none")
    out = apply(node, patcher, regions_json((FULL, 1.0)), source_tokens="not set")
    assert out["ui"]["moss_regional_lora"][0]["status"] == "active"


def test_advanced_options_declared(fake_comfy):
    _, node = fake_comfy
    opt = node.MossRegionalLora.INPUT_TYPES()["optional"]
    for name, default in (("text_position", "suffix"), ("image_order", "raster"), ("source_tokens", "end")):
        choices, spec = opt[name]
        assert choices[0] == "not set" and spec["default"] == default and spec["advanced"] is True
    assert "MossLayoutDiscovery" in node.NODE_CLASS_MAPPINGS


def test_measured_layout_defaults_apply_without_options(fake_comfy):
    """Without the options (an API prompt) the node uses the measured defaults and LoRA boxes run."""
    patcher, node = fake_comfy
    out = apply(node, patcher, regions_json((FULL, 1.0)))
    assert out["ui"]["moss_regional_lora"][0]["status"] == "active"
    assert list(out["result"][0].injections) == ["moss_regional_lora"]


def test_grid_inputs_removed_and_read_from_the_model(fake_comfy, monkeypatch):
    """vae_downscale/patch_size are no longer inputs: the patch size comes from the diffusion model and
    the px-per-latent-cell from its latent format (used only to check the latent matches the canvas)."""
    patcher, node = fake_comfy
    req = node.MossRegionalLora.INPUT_TYPES()["required"]
    assert "vae_downscale" not in req and "patch_size" not in req
    set_layout(monkeypatch)
    patcher.model.latent_format = types.SimpleNamespace(spacial_downscale_ratio=toy.VAE)
    m = apply(node, patcher, regions_json((FULL, 1.0)))["result"][0]
    ctrl = next(f for d in m.wrappers.values() for fs in d.values() for f in fs).__self__
    assert (ctrl.patch_size, ctrl.latent_downscale) == (toy.PATCH, toy.VAE)
    m.inject_model()
    m.apply_model(latent(), context(), topts())             # the toy latent is the canvas size: runs
    big = torch.zeros(1, toy.LATENT_C, 16, 16, dtype=torch.float64)
    with pytest.raises(Exception, match="node's canvas is 64x64"):
        m.apply_model(big, context(), topts())
    m.eject_model()


def test_width_and_height_outputs(fake_comfy, monkeypatch):
    """The canvas size is passed on so the empty latent node can take it and stay in sync."""
    patcher, node = fake_comfy
    assert node.MossRegionalLora.RETURN_TYPES[:4] == ("MODEL", "STRING", "INT", "INT")
    assert node.MossRegionalLora.RETURN_NAMES[2:4] == ("width", "height")
    set_layout(monkeypatch)
    for regions in (regions_json((FULL, 1.0)), json.dumps({"version": 2, "regions": []})):   # active / prompt-only
        assert apply(node, patcher, regions, width=toy.W, height=toy.H)["result"][2:4] == (toy.W, toy.H)


def test_regions_output_carries_the_boxes_and_canvas(fake_comfy, monkeypatch):
    """The regions output is the editor JSON (unknown fields kept) plus the canvas size, appended last."""
    patcher, node = fake_comfy
    assert node.MossRegionalLora.RETURN_NAMES[-1] == "regions"
    set_layout(monkeypatch)
    regions = json.dumps({"version": 2, "regions": [{"id": "a", "bbox": list(FULL), "x_extra": 1,
                                                     "loras": [{"lora": LORA_NAME, "weight": 1}]}]})
    for text in (regions, json.dumps({"version": 2, "regions": []})):      # active / prompt-only
        out = json.loads(apply(node, patcher, text, width=toy.W, height=toy.H)["result"][4])
        assert out["canvas"] == {"width": toy.W, "height": toy.H}
        assert out["regions"] == json.loads(text)["regions"]
