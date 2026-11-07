"""NFR-3: the hooks compile with fullgraph=True (no graph breaks) and do not recompile per step."""

import os

import pytest
import torch

from toy import TOP_LEFT_QUADRANT, ToyDiT, context, controller, latent, make_lora, target, topts
from moss_regional_lora.layout import Layout

BACKENDS = ["eager", "aot_eager"] + (["inductor"] if os.environ.get("RUN_INDUCTOR") else [])


@pytest.mark.parametrize("backend", BACKENDS)
def test_fullgraph_compile_one_graph_across_steps(backend):
    torch._dynamo.reset()
    from torch._dynamo.utils import counters
    counters.clear()
    layout = Layout("suffix", "raster", "none")
    torch.manual_seed(0)
    model = ToyDiT(layout).double()
    mods = dict(model.named_modules())
    d0, u0, _ = make_lora(mods["blocks.0.qkv"])
    d1, u1, _ = make_lora(mods["blocks.1.proj"], seed=3)
    ctrl = controller(model, [(TOP_LEFT_QUADRANT, 1.0)],
                      [[target("blocks.0.qkv", d0, u0), target("blocks.1.proj", d1, u1)]],
                      layout, ramp=(0.15, 1.0), steps=10)
    ctrl.inject()
    compiled = torch.compile(model, backend=backend, fullgraph=True)
    for step in range(10):
        lat, ctx = latent(2), context(2)
        ctrl.begin_call(lat, topts((0, 1), step, 10), c_crossattn=ctx)
        got = compiled(lat, ctx)
        ctrl.end_call()          # strict: would raise if the hit counter did not survive compilation
        ctrl.begin_call(lat, topts((0, 1), step, 10), c_crossattn=ctx)
        ref = model(lat, ctx)
        ctrl.end_call()
        torch.testing.assert_close(got, ref, rtol=1e-12, atol=1e-12)
    assert counters["stats"]["unique_graphs"] == 1, dict(counters["stats"])
