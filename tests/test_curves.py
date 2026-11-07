"""Per-box strength curves (curve.py) and their use by the gating controller."""

import json
import os

import pytest
import torch

import toy
from toy import FULL, ToyDiT, bit_identical, context, latent, make_lora, target, topts
from moss_regional_lora.curve import CurveError, CurvePoint, evaluate, legacy_ramp, parse_curve
from moss_regional_lora.gating import RegionalLoraController, ramp_factor
from moss_regional_lora.layout import Layout
from moss_regional_lora.regions import RegionSpec, RegionsError, parse_document

FIXTURE = os.path.join(os.path.dirname(__file__), "web", "fixtures", "curves.json")
SUFFIX = Layout("suffix", "raster", "none")
PATH = "blocks.0.qkv"


def test_fixture_matches_python_evaluation():
    """The same fixture is checked by tests/web/curve.test.mjs, so Python and the editor agree."""
    data = json.load(open(FIXTURE))
    for name, case in data["curves"].items():
        pts = parse_curve(case["curve"])
        for x, want in zip(data["samples"], case["expected"]):
            assert evaluate(pts, x) == pytest.approx(want, abs=1e-12), (name, x)


def test_shapes():
    full = parse_curve([{"x": 0, "y": 1}, {"x": 1, "y": 1}])
    assert all(evaluate(full, x) == 1.0 for x in (0, 0.3, 1))
    stop = parse_curve([{"x": 0, "y": 1, "hold": True}, {"x": 0.4, "y": 0}, {"x": 1, "y": 0}])
    assert evaluate(stop, 0.39) == 1.0 and evaluate(stop, 0.4) == 0.0 and evaluate(stop, 0.9) == 0.0
    fade = parse_curve([{"x": 0, "y": 0}, {"x": 1, "y": 1}])
    assert evaluate(fade, 0.25) == 0.25
    ease_in = parse_curve([{"x": 0, "y": 0, "bend": 0.5}, {"x": 1, "y": 1}])
    ease_out = parse_curve([{"x": 0, "y": 0, "bend": -0.5}, {"x": 1, "y": 1}])
    assert evaluate(ease_in, 0.5) < 0.5 < evaluate(ease_out, 0.5)
    assert evaluate(ease_in, 0.5) == pytest.approx(0.25)          # t ** 2


def test_parse_clamps_sorts_and_rejects():
    pts = parse_curve([{"x": 1.4, "y": -1}, {"x": 0.5, "y": 0.5, "bend": 9}])
    assert pts == (CurvePoint(0.5, 0.5, 1.0, False), CurvePoint(1.0, 0.0))
    for bad in ([], [{"x": 0}], [{"x": "a", "y": 1}], [{"x": 0, "y": 1, "hold": "yes"}], "full"):
        with pytest.raises(CurveError):
            parse_curve(bad)


def test_legacy_ramp_equals_node_ramp():
    for rs, re in ((0.15, 1.0), (0.0, 0.0), (0.3, 0.3), (0.0, 0.5), (0.5, 0.2), (0.25, 0.75)):
        c = legacy_ramp(rs, re)
        for steps in (2, 7, 10, 20):
            for step in range(steps):
                assert evaluate(c, step / max(steps - 1, 1)) == pytest.approx(ramp_factor(step, steps, rs, re)), (rs, re, step)


def test_region_json_curve_parsing():
    doc = parse_document(json.dumps({"regions": [
        {"bbox": [0, 0, 1, 1], "lora": "a", "curve": [{"x": 0, "y": 1}]},
        {"bbox": [0, 0, 1, 1], "lora": "b"}]}))
    assert doc.loras[0].curve == (CurvePoint(0.0, 1.0),) and doc.loras[1].curve is None
    with pytest.raises(RegionsError):
        parse_document(json.dumps({"regions": [{"bbox": [0, 0, 1, 1], "lora": "a", "curve": [{"x": 0}]}]}))


def _two_box_run(curve_left, curve_right, step, steps_in_schedule, node_steps=20, ramp=(0.0, 0.0)):
    torch.manual_seed(0)
    model = ToyDiT(SUFFIX).double()
    lin = dict(model.named_modules())[PATH]
    d0, u0, _ = make_lora(lin, seed=1)
    d1, u1, _ = make_lora(lin, seed=2)
    left, right = (-32, -32, 32, 0), (-32, 0, 32, 32)
    specs = [RegionSpec(0, "L", left, "a", 1.0, curve_left), RegionSpec(1, "R", right, "b", 1.0, curve_right)]
    ctrl = RegionalLoraController(specs, [[target(PATH, d0, u0)], [target(PATH, d1, u1)]], SUFFIX,
                                  toy.W, toy.H, toy.VAE, toy.PATCH, ramp[0], ramp[1], node_steps, True).bind(model)
    seen = {}
    lin.register_forward_hook(lambda m, a, o: seen.setdefault("base", o.detach().clone()))
    ctrl.inject()
    lin.register_forward_hook(lambda m, a, o: seen.setdefault("final", o.detach().clone()))
    ctrl.begin_call(latent(), topts((0,), step=step, steps=steps_in_schedule), c_crossattn=context())
    model(latent(), context())
    ctrl.end_call()
    cols = torch.arange(toy.N_IMG) % toy.COLS
    return seen["base"][:, :toy.N_IMG], seen["final"][:, :toy.N_IMG], cols < toy.COLS // 2


def test_each_box_follows_its_own_curve():
    """At 60% of the run, the left box's hard stop (at 40%) has ended while the right box is at full strength."""
    stop = parse_curve([{"x": 0, "y": 1, "hold": True}, {"x": 0.4, "y": 0}, {"x": 1, "y": 0}])
    full = parse_curve([{"x": 0, "y": 1}, {"x": 1, "y": 1}])
    base, final, is_left = _two_box_run(stop, full, step=6, steps_in_schedule=11)     # progress 0.6
    assert bit_identical(final[:, is_left], base[:, is_left])
    assert not torch.equal(final[:, ~is_left], base[:, ~is_left])
    base, final, is_left = _two_box_run(stop, full, step=2, steps_in_schedule=11)     # progress 0.2
    assert not torch.equal(final[:, is_left], base[:, is_left])


def test_step_count_comes_from_the_sampler_schedule():
    """Node 'steps' says 20, the sampler runs 10: the last sampler step must count as the end of the run."""
    late = parse_curve([{"x": 0, "y": 0, "hold": True}, {"x": 0.95, "y": 1}, {"x": 1, "y": 1}])
    base, final, is_left = _two_box_run(late, late, step=9, steps_in_schedule=10, node_steps=20)
    assert not torch.equal(final, base)          # 9/9 = 1.0 -> on; with the node's 20 it would be 9/19 -> off


def test_boxes_without_curve_use_node_ramp():
    base, final, _ = _two_box_run(None, None, step=0, steps_in_schedule=10, ramp=(0.15, 1.0))
    assert bit_identical(final, base)            # node ramp: 0 at the first step
