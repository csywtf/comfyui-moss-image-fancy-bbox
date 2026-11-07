"""Several LoRAs per box (schema version 2), and reading version-1 single-LoRA boxes."""

import json

import pytest
import torch

import toy
from toy import ToyDiT, bit_identical, context, latent, make_lora, target, topts
from moss_regional_lora.curve import CurvePoint
from moss_regional_lora.gating import RegionalLoraController
from moss_regional_lora.layout import Layout
from moss_regional_lora.regions import RegionsError, parse_document


def doc(*regions, version=2):
    return json.dumps({"version": version, "regions": list(regions)})


def test_each_entry_becomes_a_spec_on_the_same_box():
    d = parse_document(doc({"name": "A", "bbox": [-10, -10, 10, 10], "loras": [
        {"lora": "dir/style.safetensors", "weight": 0.5},
        {"lora": "char.safetensors", "weight": -1.5, "curve": [{"x": 0, "y": 1, "hold": True}, {"x": 0.4, "y": 0}]},
        {"lora": "", "weight": 1.0},                 # no file chosen yet
        {"lora": "off.safetensors", "weight": 0},     # weight 0
    ]}))
    assert [(s.lora, s.weight) for s in d.loras] == [("dir/style.safetensors", 0.5), ("char.safetensors", -1.5)]
    assert all(s.bbox == (-10.0, -10.0, 10.0, 10.0) for s in d.loras)
    assert [s.index for s in d.loras] == [0, 1]
    assert d.loras[0].curve is None and d.loras[1].curve[0] == CurvePoint(0.0, 1.0, 0.0, True)
    assert any("no file selected" in n for n in d.notes) and any("weight 0" in n for n in d.notes)


def test_version_1_single_lora_fields_still_read():
    d = parse_document(doc({"bbox": [0, 0, 5, 5], "lora": "old.safetensors", "weight": 0.7,
                            "curve": [{"x": 0, "y": 1}]}, version=1))
    assert [(s.lora, s.weight, s.curve) for s in d.loras] == [("old.safetensors", 0.7, (CurvePoint(0.0, 1.0),))]


def test_loras_list_wins_over_old_fields():
    d = parse_document(doc({"bbox": [0, 0, 5, 5], "lora": "old.safetensors",
                            "loras": [{"lora": "new.safetensors", "weight": 1}]}))
    assert [s.lora for s in d.loras] == ["new.safetensors"]


@pytest.mark.parametrize("bad", [{"loras": "x"}, {"loras": [5]}, {"loras": [{"lora": 3}]},
                                 {"loras": [{"lora": "a", "weight": "b"}]}, {"loras": [{"lora": "a", "curve": []}]}])
def test_malformed_entries_raise(bad):
    with pytest.raises(RegionsError):
        parse_document(doc({"bbox": [0, 0, 1, 1], **bad}))


def test_two_loras_in_one_box_sum_inside_and_stay_out_of_the_rest():
    layout = Layout("suffix", "raster", "none")
    torch.manual_seed(0)
    model = ToyDiT(layout).double()
    lin = model.blocks[0].qkv
    d = parse_document(doc({"bbox": [-32, -32, 32, 0], "loras": [{"lora": "a", "weight": 1.0}, {"lora": "b", "weight": 0.5}]}))
    (d0, u0, _), (d1, u1, _) = make_lora(lin, seed=1), make_lora(lin, seed=2)
    ctrl = RegionalLoraController(d.loras, [[target("blocks.0.qkv", d0, u0)], [target("blocks.0.qkv", d1, u1)]], layout,
                                  toy.W, toy.H, toy.VAE, toy.PATCH, 0.0, 0.0, 10, True).bind(model)
    seen = {}
    def capture_base(m, a, o):
        seen.setdefault("x", a[0].detach().clone())
        seen.setdefault("base", o.detach().clone())

    lin.register_forward_hook(capture_base)
    ctrl.inject()
    lin.register_forward_hook(lambda m, a, o: seen.setdefault("final", o.detach().clone()))
    ctrl.begin_call(latent(), topts(), c_crossattn=context())
    model(latent(), context())
    ctrl.end_call()
    left = torch.zeros(seen["base"].shape[1], dtype=torch.bool)
    left[:toy.N_IMG] = (torch.arange(toy.N_IMG) % toy.COLS) < toy.COLS // 2
    a0, a1 = ctrl.gates["blocks.0.qkv"].adapters
    expected = seen["base"] + (a0.delta(seen["x"]) + a1.delta(seen["x"]))    # same summation order as the hook
    assert torch.equal(seen["final"][:, left], expected[:, left])
    assert bit_identical(seen["final"][:, ~left], seen["base"][:, ~left])
