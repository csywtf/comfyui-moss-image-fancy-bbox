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


def apply(node, patcher, regions, strict=True, **kw):
    args = dict(width=toy.W, height=toy.H, regions=regions, vae_downscale=toy.VAE, patch_size=toy.PATCH,
                ramp_start=0.0, ramp_end=0.0, steps=10, strict=strict)
    args.update(kw)
    return node.MossRegionalLora().apply(patcher, **args)


def set_layout(monkeypatch, t="suffix", o="raster", s="none"):
    monkeypatch.setattr(layout_config, "TEXT_POSITION", t)
    monkeypatch.setattr(layout_config, "IMAGE_ORDER", o)
    monkeypatch.setattr(layout_config, "SOURCE_TOKENS", s)


def test_is_changed_always(fake_comfy):
    _, node = fake_comfy
    v = node.MossRegionalLora.IS_CHANGED(width=1)
    assert isinstance(v, float) and math.isnan(v)


def test_strict_refuses_with_unset_layout(fake_comfy):
    patcher, node = fake_comfy
    with pytest.raises(LayoutNotEstablished, match="discover_moss_layout"):
        apply(node, patcher, regions_json((FULL, 1.0)))


def test_non_strict_unset_layout_passes_model_through_unpatched(fake_comfy):
    patcher, node = fake_comfy
    out = apply(node, patcher, regions_json((FULL, 1.0)), strict=False)
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
