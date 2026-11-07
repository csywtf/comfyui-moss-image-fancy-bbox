"""FR-8 (equivalence with LoraLoader), FR-9 (formats), FR-11 (strict refusal), region JSON parsing."""

import os
import sys
import types

import pytest
import torch

import toy
from toy import FULL, N_IMG, ToyDiT, bit_identical, context, controller, latent, topts
from moss_regional_lora.layout import Layout, LayoutNotEstablished, LayoutInvalid, load_layout
from moss_regional_lora.lora_file import generic_key_map, parse_lora
from moss_regional_lora.gating import iter_linear_modules
from moss_regional_lora.regions import RegionsError, parse_regions

IMAGE_ONLY = Layout("suffix", "raster", "none")   # with zero text tokens the whole sequence is image tokens


def toy_key_map(model):
    return generic_key_map(n for n, _ in iter_linear_modules(model))


def lora_sd(model, style, rank=4, alpha=8.0, seed=0, paths=("blocks.0.qkv", "blocks.1.proj")):
    g = torch.Generator().manual_seed(seed)
    mods = dict(model.named_modules())
    sd = {}
    for p in paths:
        m = mods[p]
        down = torch.randn(rank, m.in_features, generator=g, dtype=torch.float64) * 0.1
        up = torch.randn(m.out_features, rank, generator=g, dtype=torch.float64) * 0.1
        if style == "kohya":
            k = "lora_unet_" + p.replace(".", "_")
            sd[k + ".lora_down.weight"], sd[k + ".lora_up.weight"] = down, up
            if alpha is not None:
                sd[k + ".alpha"] = torch.tensor(alpha)
        else:  # peft / diffusers
            k = "transformer." + p
            sd[k + ".lora_A.weight"], sd[k + ".lora_B.weight"] = down, up
            if alpha is not None:
                sd[k + ".alpha"] = torch.tensor(alpha)
    return sd


def comfy_reference_merge(model, sd, key_map, strength):
    """ComfyUI's LoraLoader math, transcribed from comfy/weight_adapter/lora.py LoRAAdapter.calculate_weight:
        alpha = v[2] / mat2.shape[0] if v[2] is not None else 1.0
        weight += ((strength * alpha) * (mat1 @ mat2)).type(weight.dtype)      # mat1 = up, mat2 = down
    computed in float32 intermediate in ComfyUI; float64 here."""
    merged = ToyDiT(model.layout).double()
    merged.load_state_dict(model.state_dict())
    mods = dict(merged.named_modules())
    for prefix, to in key_map.items():
        for up_s, down_s in ((".lora_up.weight", ".lora_down.weight"), (".lora_B.weight", ".lora_A.weight")):
            if prefix + up_s in sd:
                up, down = sd[prefix + up_s], sd[prefix + down_s]
                alpha = sd[prefix + ".alpha"].item() / down.shape[0] if prefix + ".alpha" in sd else 1.0
                w = mods[to[len("diffusion_model."):-len(".weight")]].weight
                with torch.no_grad():
                    w += ((strength * alpha) * torch.mm(up, down)).type(w.dtype)
    return merged


# ------------------------------------------------------------------ FR-8
@pytest.mark.parametrize("style,alpha", [("kohya", 8.0), ("kohya", None), ("peft", None), ("peft", 2.0)])
@pytest.mark.parametrize("strength", [1.0, 0.65])
def test_full_coverage_matches_loraloader_merge(style, alpha, strength):
    torch.manual_seed(0)
    model = ToyDiT(IMAGE_ONLY).double()
    sd = lora_sd(model, style, alpha=alpha)
    lat = torch.randn(1, toy.LATENT_C, toy.H // toy.VAE, toy.W // toy.VAE, dtype=torch.float64)
    ctx = context(n=0)
    merged = comfy_reference_merge(model, sd, toy_key_map(model), strength)
    expected = merged(lat, ctx)
    base = model(lat, ctx)
    parsed = parse_lora(sd, toy_key_map(model))
    ctrl = controller(model, [(FULL, strength)], [parsed.targets], IMAGE_ONLY)
    ctrl.inject()
    ctrl.begin_call(lat, topts(), c_crossattn=ctx)
    got = model(lat, ctx)
    ctrl.end_call()
    torch.testing.assert_close(got, expected, rtol=1e-10, atol=1e-12)
    assert not torch.allclose(got, base), "the LoRA must actually change the output"


def test_kohya_and_peft_scale_differently_for_same_matrices():
    torch.manual_seed(0)
    model = ToyDiT(IMAGE_ONLY).double()
    k = parse_lora(lora_sd(model, "kohya", rank=4, alpha=8.0), toy_key_map(model))
    p = parse_lora(lora_sd(model, "peft", rank=4, alpha=None), toy_key_map(model))
    assert {t.alpha_scale for t in k.targets} == {2.0}
    assert {t.alpha_scale for t in p.targets} == {1.0}


@pytest.mark.skipif(not os.environ.get("COMFYUI_PATH"), reason="set COMFYUI_PATH to compare against real comfy.lora")
def test_full_coverage_matches_real_comfy_lora():
    """Same as above but the merge is done by ComfyUI's own load_lora + calculate_weight."""
    sys.path.insert(0, os.environ["COMFYUI_PATH"])
    import comfy.lora
    torch.manual_seed(0)
    model = ToyDiT(IMAGE_ONLY).double()
    for style, alpha in (("kohya", 8.0), ("peft", None)):
        sd = lora_sd(model, style, alpha=alpha)
        km = toy_key_map(model)
        patches = comfy.lora.load_lora(sd, km)
        merged = ToyDiT(IMAGE_ONLY).double()
        merged.load_state_dict(model.state_dict())
        msd = merged.state_dict()
        for key, adapter in patches.items():
            w = msd[key[len("diffusion_model."):]]
            new = adapter.calculate_weight(w.clone(), key, 0.65, 1.0, None, lambda a: a, intermediate_dtype=torch.float64)
            with torch.no_grad():
                w.copy_(new)
        lat = torch.randn(1, toy.LATENT_C, toy.H // toy.VAE, toy.W // toy.VAE, dtype=torch.float64)
        ctx = context(n=0)
        parsed = parse_lora(sd, km)
        ctrl = controller(model, [(FULL, 0.65)], [parsed.targets], IMAGE_ONLY)
        ctrl.inject()
        ctrl.begin_call(lat, topts(), c_crossattn=ctx)
        got = model(lat, ctx)
        ctrl.end_call()
        ctrl.eject()
        torch.testing.assert_close(got, merged(lat, ctx), rtol=1e-10, atol=1e-12)


# ------------------------------------------------------------------ FR-9
def test_formats_resolve_and_unknown_keys_are_counted_not_raised():
    torch.manual_seed(0)
    model = ToyDiT(IMAGE_ONLY).double()
    sd = lora_sd(model, "kohya")
    sd.update(lora_sd(model, "peft", paths=("blocks.1.qkv",)))
    sd["lora_te_text_model_encoder_layers_0_mlp_fc1.lora_down.weight"] = torch.zeros(4, 8)   # text encoder key
    sd["lora_te_text_model_encoder_layers_0_mlp_fc1.lora_up.weight"] = torch.zeros(8, 4)
    sd["lora_unet_blocks_0_qkv.dora_scale"] = torch.ones(1)                                 # unsupported component
    sd["something.random"] = torch.ones(1)
    parsed = parse_lora(sd, toy_key_map(model))
    paths = sorted(t.module_path for t in parsed.targets)
    assert paths == ["blocks.1.proj", "blocks.1.qkv"]
    reasons = " ".join(r for _, r in parsed.skipped)
    assert "unsupported adapter component" in reasons and "did not resolve" in reasons
    assert parsed.skipped_count == 4   # 2 te keys, the dora'd qkv prefix, the random key


def test_kohya_key_on_non_linear_module_is_skipped():
    torch.manual_seed(0)
    model = ToyDiT(IMAGE_ONLY).double()
    km = toy_key_map(model)
    km["lora_unet_blocks_0"] = "diffusion_model.blocks.0.weight"   # resolves to a Block, not a Linear
    sd = {"lora_unet_blocks_0.lora_down.weight": torch.zeros(2, toy.DIM),
          "lora_unet_blocks_0.lora_up.weight": torch.zeros(toy.DIM, 2)}
    parsed = parse_lora(sd, km)
    ctrl = controller(model, [(FULL, 1.0)], [parsed.targets], IMAGE_ONLY)
    assert ctrl.applied_count == 0 and len(ctrl.skipped) == 1


def test_sliced_key_map_entry():
    torch.manual_seed(0)
    model = ToyDiT(IMAGE_ONLY).double()
    km = {"blocks.0.q": ("diffusion_model.blocks.0.qkv.weight", (0, 0, toy.DIM))}
    sd = {"blocks.0.q.lora_A.weight": torch.zeros(2, toy.DIM), "blocks.0.q.lora_B.weight": torch.zeros(toy.DIM, 2)}
    (t,) = parse_lora(sd, km).targets
    assert (t.module_path, t.out_offset, t.out_size) == ("blocks.0.qkv", 0, toy.DIM)


# ------------------------------------------------------------------ FR-11
def test_strict_refuses_when_layout_unset_and_names_tool_and_constants():
    cfg = types.SimpleNamespace(TEXT_POSITION=None, IMAGE_ORDER="raster", SOURCE_TOKENS=None)
    with pytest.raises(LayoutNotEstablished) as e:
        load_layout(cfg, strict=True)
    msg = str(e.value)
    assert "tools/discover_moss_layout.py" in msg
    assert "TEXT_POSITION" in msg and "SOURCE_TOKENS" in msg and "IMAGE_ORDER" not in msg


def test_shipped_config_is_unset():
    """The repository must not ship guessed constants."""
    with pytest.raises(LayoutNotEstablished):
        load_layout(strict=True)
    assert load_layout(strict=False) is None


def test_invalid_value_always_raises():
    cfg = types.SimpleNamespace(TEXT_POSITION="middle", IMAGE_ORDER="raster", SOURCE_TOKENS="none")
    with pytest.raises(LayoutInvalid):
        load_layout(cfg, strict=False)


def test_layout_positions():
    for t in ("prefix", "suffix"):
        for s in ("none", "start", "between", "end"):
            lay = Layout(t, "raster", s)
            mask = lay.sequence_mask([1.0] * 4, lay.sequence_length(4, 3, 2), 3, 2)
            start = lay.image_start(3, 2)
            assert mask[start:start + 4] == [1.0] * 4 and sum(mask) == 4
    assert Layout("prefix", "raster", "none").sequence_mask([1.0] * 4, 99, 3, 0) is None


# ------------------------------------------------------------------ region JSON
def test_regions_parse_order_hidden_and_skips():
    text = """{"version": 1, "regions": [
        {"name": "a", "bbox": [-10, -10, 10, 10], "lora": "a.safetensors", "weight": 0.5, "group": "g"},
        {"name": "hidden", "bbox": [0, 0, 1, 1], "lora": "h.safetensors", "visible": false},
        {"name": "nolora", "bbox": [0, 0, 1, 1], "lora": ""},
        {"name": "b", "bbox": [0, 0, 5, 5], "lora": "b.safetensors", "future_field": {"x": 1}}],
        "groups": [{"id": "g", "name": "G"}], "something_new": true}"""
    regions, notes = parse_regions(text)
    assert [r.name for r in regions] == ["a", "b"]
    assert [r.index for r in regions] == [0, 1]
    assert regions[0].weight == 0.5 and regions[1].weight == 1.0
    assert any("nolora" in n for n in notes)


@pytest.mark.parametrize("text", ["{not json", '{"regions": [{"bbox": [1, 2, 3]}]}', '{"regions": 5}',
                                  '{"regions": [{"bbox": [5, 0, 0, 5], "lora": "x"}]}',
                                  '{"regions": [{"bbox": [0, 0, 5, 5], "lora": "x", "weight": "a"}]}'])
def test_regions_malformed_raise(text):
    with pytest.raises(RegionsError):
        parse_regions(text)


def test_newer_version_is_tolerated():
    regions, notes = parse_regions('{"version": 99, "regions": [{"bbox": [0,0,1,1], "lora": "x"}]}')
    assert len(regions) == 1 and any("newer" in n for n in notes)
