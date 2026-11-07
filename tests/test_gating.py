"""Acceptance criteria 1-6 (spec section 6) plus FR-5, FR-6, FR-9, FR-11 behaviour, on a toy model."""

import pytest
import torch

import toy
from toy import (ALL_LAYOUTS, FULL, N_IMG, N_TEXT, OFF_CANVAS, TOP_LEFT_QUADRANT, ToyDiT, bit_identical,
                 context, controller, latent, make_lora, target, topts)
from moss_regional_lora.gating import HOOK_ATTR, RegionalLoraError, ramp_factor, step_index
from moss_regional_lora.geometry import coverage_grid, raster_flatten
from moss_regional_lora.layout import Layout

SUFFIX = Layout("suffix", "raster", "none")
PREFIX = Layout("prefix", "raster", "none")
PATH = "blocks.0.qkv"
# A box whose edges cut through patches, so the mask has fractional values (FR-4):
# x from -24 to +8 px, y from -32 to -4 px on a 64x64 canvas with 16 px patches.
SOFT_BOX = (-32, -24, -4, 8)


def setup(layout, regions, seed=0, paths=(PATH,), alpha=None, ramp=(0.0, 0.0), steps=10, strict=True):
    torch.manual_seed(seed)
    model = ToyDiT(layout).double()
    mods = dict(model.named_modules())
    targets = []
    for i, _ in enumerate(regions):
        ts = []
        for p in paths:
            down, up, _ = make_lora(mods[p], seed=seed + 10 * i + hash(p) % 7)
            ts.append(target(p, down, up, alpha))
        targets.append(ts)
    ctrl = controller(model, regions, targets, layout, ramp=ramp, steps=steps, strict=strict)
    return model, ctrl


class Capture:
    """Records a module's input, base output (before our hook) and final output (after it)."""

    def __init__(self, module):
        self.module = module
        self.h1 = module.register_forward_hook(self._base)

    def _base(self, m, args, out):
        self.x, self.base = args[0].detach().clone(), out.detach().clone()

    def after_inject(self):
        self.h2 = self.module.register_forward_hook(self._final)

    def _final(self, m, args, out):
        self.final = out.detach().clone()


def run(model, ctrl, lat, ctx, opts, refs=()):
    ctrl.begin_call(lat, opts, c_crossattn=ctx, ref_latents=list(refs) or None)
    try:
        return model(lat, ctx, refs)
    finally:
        ctrl.end_call()


def seq_mask(layout, box, n_text=N_TEXT, n_src=0):
    img = raster_flatten(coverage_grid(box, toy.W, toy.H, toy.VAE, toy.PATCH))
    return torch.tensor(layout.sequence_mask(img, layout.sequence_length(N_IMG, n_text, n_src), n_text, n_src),
                        dtype=torch.float64)


def run_captured(layout, regions, opts=None, batch=1, **kw):
    model, ctrl = setup(layout, regions, **kw)
    cap = Capture(dict(model.named_modules())[PATH])
    ctrl.inject()
    cap.after_inject()
    run(model, ctrl, latent(batch), context(batch), opts or topts())
    return model, ctrl, cap


# ---------------------------------------------------------------- criteria 1 & 2
@pytest.mark.parametrize("layout", [SUFFIX, PREFIX], ids=["suffix", "prefix"])
def test_masked_in_equals_base_plus_delta_times_mask_exactly(layout):
    model, ctrl, cap = run_captured(layout, [(SOFT_BOX, 0.8)])
    mask = seq_mask(layout, SOFT_BOX)
    assert ((mask > 0) & (mask < 1)).any(), "fixture must exercise fractional coverage"
    delta = ctrl.gates[PATH].adapters[0].delta(cap.x)
    expected = cap.base + delta * mask[None, :, None]
    inside = mask > 0
    assert torch.equal(cap.final[:, inside], expected[:, inside])
    assert not torch.equal(cap.final[:, inside], cap.base[:, inside]), "delta must actually change masked-in tokens"


@pytest.mark.parametrize("layout", ALL_LAYOUTS, ids=lambda l: f"{l.text_position}-{l.source_tokens}")
def test_masked_out_positions_are_bit_identical_to_base(layout):
    model, ctrl, cap = run_captured(layout, [(SOFT_BOX, 1.5)])
    outside = seq_mask(layout, SOFT_BOX) == 0
    assert outside.any() and (~outside).any()
    assert bit_identical(cap.final[:, outside], cap.base[:, outside])


def test_masked_out_survives_nonfinite_delta():
    """where-select, not 0 * delta: an inf/NaN delta must not leak into masked-out tokens."""
    model, ctrl, cap = run_captured(SUFFIX, [(SOFT_BOX, 1.0)])
    ad = ctrl.gates[PATH].adapters[0]
    ad.up_master = ad.up_master.clone().fill_(float("inf"))
    ad.down = ad.up = None
    run(model, ctrl, latent(), context(), topts())
    outside = seq_mask(SUFFIX, SOFT_BOX) == 0
    assert bit_identical(cap.final[:, outside], cap.base[:, outside])


def test_negative_zero_base_values_are_preserved_outside():
    """base + 0.0 would turn -0.0 into +0.0; the where-select must keep the base's exact bits."""

    class NegZeroLinear(torch.nn.Linear):
        def forward(self, x):
            return torch.full(x.shape[:-1] + (self.out_features,), -0.0, dtype=x.dtype)

    torch.manual_seed(0)
    model, ctrl = setup(SUFFIX, [(TOP_LEFT_QUADRANT, 1.0)])
    model.blocks[0].qkv.__class__ = NegZeroLinear
    lin = model.blocks[0].qkv
    cap = Capture(lin)
    ctrl.inject()
    cap.after_inject()
    run(model, ctrl, latent(), context(), topts())
    outside = seq_mask(SUFFIX, TOP_LEFT_QUADRANT) == 0
    assert torch.signbit(cap.base[:, outside]).all()
    assert torch.signbit(cap.final[:, outside]).all()


# ---------------------------------------------------------------- criterion 3
def test_all_zero_mask_is_a_noop_on_the_whole_tensor():
    model, ctrl, cap = run_captured(SUFFIX, [(OFF_CANVAS, 1.0)])
    assert seq_mask(SUFFIX, OFF_CANVAS).abs().sum() == 0
    assert bit_identical(cap.final, cap.base)


# ---------------------------------------------------------------- criterion 4
def test_ramp_start_step_zero_contributes_exactly_zero():
    model, ctrl, cap = run_captured(SUFFIX, [(FULL, 1.0)], opts=topts(step=0, steps=10), ramp=(0.15, 1.0))
    assert bit_identical(cap.final, cap.base)


def test_ramp_shape():
    assert ramp_factor(0, 10, 0.15, 1.0) == 0.0
    assert ramp_factor(1, 10, 0.15, 1.0) == 0.0            # 1/9 = 0.111 < 0.15
    assert ramp_factor(9, 10, 0.15, 1.0) == 1.0            # last step is full strength
    assert 0.0 < ramp_factor(5, 10, 0.15, 1.0) < 1.0
    assert ramp_factor(5, 10, 0.15, 1.0) == pytest.approx((5 / 9 - 0.15) / 0.85)
    assert ramp_factor(0, 10, 0.0, 0.0) == 1.0             # ramp disabled
    assert ramp_factor(3, 10, 0.5, 0.5) == 0.0 and ramp_factor(5, 10, 0.5, 0.5) == 1.0


def test_ramp_scales_contribution_linearly():
    """The delta at a mid-ramp step is ramp * (full-strength delta), on masked-in tokens."""
    full = run_captured(SUFFIX, [(FULL, 1.0)], opts=topts(step=5, steps=10), ramp=(0.0, 0.0))[2]
    half = run_captured(SUFFIX, [(FULL, 1.0)], opts=topts(step=5, steps=10), ramp=(0.15, 1.0))[2]
    r = ramp_factor(5, 10, 0.15, 1.0)
    img = seq_mask(SUFFIX, FULL) > 0
    torch.testing.assert_close((half.final - half.base)[:, img], r * (full.final - full.base)[:, img])


def test_step_index_from_schedule():
    ss = [1.0, 0.75, 0.5, 0.25, 0.0]
    assert [step_index(s, ss) for s in ss[:-1]] == [0, 1, 2, 3]
    assert step_index(0.6, ss) == 1          # Heun-style intermediate evaluation between steps 1 and 2


# ---------------------------------------------------------------- criterion 5 (FR-6)
def test_unconditional_row_is_bit_identical_to_base():
    model, ctrl, cap = run_captured(SUFFIX, [(FULL, 1.0)], opts=topts(cond_or_uncond=(0, 1)), batch=2)
    assert bit_identical(cap.final[1], cap.base[1])
    assert not torch.equal(cap.final[0], cap.base[0])


def test_unconditional_first_in_batch():
    model, ctrl, cap = run_captured(SUFFIX, [(FULL, 1.0)], opts=topts(cond_or_uncond=(1, 0)), batch=2)
    assert bit_identical(cap.final[0], cap.base[0])
    assert not torch.equal(cap.final[1], cap.base[1])


# ---------------------------------------------------------------- criterion 6 (FR-10)
def _n_our_hooks(module):
    return sum(1 for h in module._forward_hooks.values() if type(h).__name__ == "_ModuleGate")


def test_install_twice_is_idempotent():
    model, ctrl = setup(SUFFIX, [(FULL, 1.0)])
    lin = dict(model.named_modules())[PATH]
    ctrl.inject()
    out_once = run(model, ctrl, latent(), context(), topts())
    ctrl.inject()
    ctrl.inject()
    assert _n_our_hooks(lin) == 1
    out_twice = run(model, ctrl, latent(), context(), topts())
    assert bit_identical(out_once, out_twice)


def test_reexecution_replaces_instead_of_stacking():
    """A second controller (the node run again) replaces the first one's hooks."""
    model, first = setup(SUFFIX, [(FULL, 1.0)])
    mods = dict(model.named_modules())
    first.inject()
    second = controller(model, [(FULL, 1.0)], [[target(PATH, first.gates[PATH].adapters[0].down_master,
                                                       first.gates[PATH].adapters[0].up_master)]], SUFFIX)
    second.inject()
    assert _n_our_hooks(mods[PATH]) == 1
    assert getattr(mods[PATH], HOOK_ATTR) is second.gates[PATH]
    out = run(model, second, latent(), context(), topts())
    first.eject()                       # stale controller's eject must not remove the new hook
    assert _n_our_hooks(mods[PATH]) == 1
    out_again = run(model, second, latent(), context(), topts())
    assert bit_identical(out, out_again)
    # and equals a single fresh application
    model2, fresh = setup(SUFFIX, [(FULL, 1.0)])
    fresh.inject()
    assert bit_identical(out, run(model2, fresh, latent(), context(), topts()))


def test_eject_restores_module_exactly():
    model, ctrl = setup(SUFFIX, [(FULL, 1.0)])
    lin = dict(model.named_modules())[PATH]
    before = {k: v.clone() for k, v in model.state_dict().items()}
    ref = model(latent(), context())
    ctrl.inject()
    run(model, ctrl, latent(), context(), topts())
    ctrl.eject()
    assert _n_our_hooks(lin) == 0 and not hasattr(lin, HOOK_ATTR) and "comfy_force_forward" not in lin.__dict__
    assert bit_identical(model(latent(), context()), ref)
    for k, v in model.state_dict().items():
        assert bit_identical(v, before[k]), f"parameter {k} was modified (NFR-1)"


def test_parameters_never_modified_while_active():
    model, ctrl = setup(SUFFIX, [(FULL, 1.0)])
    ptrs = {k: (v.data_ptr(), v.clone()) for k, v in model.state_dict().items()}
    ctrl.inject()
    run(model, ctrl, latent(), context(), topts())
    for k, v in model.state_dict().items():
        assert v.data_ptr() == ptrs[k][0] and bit_identical(v, ptrs[k][1])


def test_inactive_outside_a_wrapped_call():
    """A branch of the graph that uses the same base model without the wrapper sees the clean model."""
    model, ctrl = setup(SUFFIX, [(FULL, 1.0)])
    ref = model(latent(), context())
    ctrl.inject()
    assert bit_identical(model(latent(), context()), ref)


# ---------------------------------------------------------------- FR-5: text and source tokens are 0
@pytest.mark.parametrize("layout", ALL_LAYOUTS, ids=lambda l: f"{l.text_position}-{l.source_tokens}")
def test_text_and_source_positions_untouched(layout):
    model, ctrl = setup(layout, [(FULL, 1.0)])
    cap = Capture(dict(model.named_modules())[PATH])
    ctrl.inject()
    cap.after_inject()
    ref = torch.randn(1, toy.LATENT_C, 4, 8, dtype=torch.float64)      # 2x4 = 8 source tokens
    run(model, ctrl, latent(), context(), topts(), refs=[ref])
    n_src = 8 if layout.source_tokens != "none" else 0
    if layout.source_tokens == "none":
        return  # this toy layout carries no source tokens; nothing further to check
    start = layout.image_start(N_TEXT, n_src)
    is_img = torch.zeros(cap.base.shape[1], dtype=torch.bool)
    is_img[start:start + N_IMG] = True
    assert bit_identical(cap.final[:, ~is_img], cap.base[:, ~is_img])
    assert not torch.equal(cap.final[:, is_img], cap.base[:, is_img])


def test_text_only_and_image_only_sequences_get_no_delta():
    """Modules that never see the joint sequence (e.g. a text embedder) are left exactly alone."""
    model, ctrl = setup(SUFFIX, [(FULL, 1.0)], paths=("img_in", PATH))
    lin = dict(model.named_modules())["img_in"]
    cap = Capture(lin)
    ctrl.inject()
    cap.after_inject()
    run(model, ctrl, latent(), context(), topts())
    assert bit_identical(cap.final, cap.base)
    assert any(p == "img_in" for p, _ in ctrl.state.unmatched)


# ---------------------------------------------------------------- multiple regions, sliced targets
def test_two_regions_sum_and_stay_separate():
    left = (-32, -32, 32, 0)
    right = (-32, 0, 32, 32)
    model, ctrl, cap = run_captured(SUFFIX, [(left, 1.0), (right, 0.5)])
    d0 = ctrl.gates[PATH].adapters[0].delta(cap.x)
    d1 = ctrl.gates[PATH].adapters[1].delta(cap.x)
    m0, m1 = seq_mask(SUFFIX, left), seq_mask(SUFFIX, right)
    expected = cap.base + (d0 * m0[None, :, None] + d1 * m1[None, :, None])
    img = (m0 + m1) > 0
    assert torch.equal(cap.final[:, img], expected[:, img])
    assert bit_identical(cap.final[:, ~img], cap.base[:, ~img])


def test_sliced_target_writes_only_its_output_features():
    """A LoRA addressing q of a fused qkv weight (as ComfyUI's MOSS key map produces)."""
    torch.manual_seed(0)
    model = ToyDiT(SUFFIX).double()
    lin = dict(model.named_modules())[PATH]
    g = torch.Generator().manual_seed(3)
    down = torch.randn(2, toy.DIM, generator=g, dtype=torch.float64)
    up = torch.randn(toy.DIM, 2, generator=g, dtype=torch.float64)
    ctrl = controller(model, [(FULL, 1.0)], [[target(PATH, down, up, offset=toy.DIM, size=toy.DIM)]], SUFFIX)
    cap = Capture(lin)
    ctrl.inject()
    cap.after_inject()
    run(model, ctrl, latent(), context(), topts())
    img = seq_mask(SUFFIX, FULL) > 0
    changed = (cap.final != cap.base)[:, img]
    assert not changed[..., :toy.DIM].any() and not changed[..., 2 * toy.DIM:].any()
    assert changed[..., toy.DIM:2 * toy.DIM].any()


# ---------------------------------------------------------------- fail loudly
def test_wrong_layout_raises_in_strict_mode():
    """The real model appends source tokens, but the node is configured with SOURCE_TOKENS="none":
    the joint length differs from every length the node expects, so nothing is gated and strict raises.
    (A prefix/suffix mix-up keeps the length identical and is caught only by discovery, which is why
    FR-11 refuses to run until discovery has been done.)"""
    model, ctrl = setup(Layout("suffix", "raster", "none"), [(FULL, 1.0)])
    model.layout = Layout("suffix", "raster", "end")
    ctrl.inject()
    ref = torch.zeros(1, toy.LATENT_C, 4, 8, dtype=torch.float64)
    with pytest.raises(RegionalLoraError, match="discover_moss_layout"):
        run(model, ctrl, latent(), context(), topts(), refs=[ref])


def test_source_tokens_in_or_out_of_sequence_both_accepted():
    """With SOURCE_TOKENS set, a pass whose source images are not in the sequence (precomputed K/V)
    is still gated correctly."""
    layout = Layout("prefix", "raster", "between")
    model, ctrl = setup(layout, [(FULL, 1.0)])
    model.layout = Layout("prefix", "raster", "none")     # this pass keeps the refs out of the sequence
    cap = Capture(dict(model.named_modules())[PATH])
    ctrl.inject()
    cap.after_inject()
    ref = torch.zeros(1, toy.LATENT_C, 4, 8, dtype=torch.float64)
    ctrl.begin_call(latent(), topts(), c_crossattn=context(), ref_latents=[ref])
    model(latent(), context())
    ctrl.end_call()
    mask = seq_mask(layout, FULL, n_src=0)
    assert bit_identical(cap.final[:, mask == 0], cap.base[:, mask == 0])
    assert not torch.equal(cap.final[:, mask > 0], cap.base[:, mask > 0])


def test_canvas_latent_mismatch_raises():
    model, ctrl = setup(SUFFIX, [(FULL, 1.0)])
    ctrl.inject()
    wrong = torch.zeros(1, toy.LATENT_C, 16, 16, dtype=torch.float64)
    with pytest.raises(RegionalLoraError, match="canvas"):
        ctrl.begin_call(wrong, topts(), c_crossattn=context())


def test_non_strict_warns_and_stays_inert(caplog):
    model, ctrl = setup(SUFFIX, [(FULL, 1.0)], strict=False)
    ctrl.inject()
    wrong = torch.zeros(1, toy.LATENT_C, 16, 16, dtype=torch.float64)
    assert ctrl.begin_call(wrong, topts(), c_crossattn=context()) is False
    assert not ctrl.state.active


def test_unresolvable_targets_are_skipped_and_counted():
    torch.manual_seed(0)
    model = ToyDiT(SUFFIX).double()
    good = target(PATH, *make_lora(dict(model.named_modules())[PATH])[:2])
    missing = target("blocks.9.qkv", good.down, good.up)
    not_linear = target("blocks", good.down, good.up)
    bad_shape = target("blocks.0.proj", good.down, good.up)       # qkv-shaped LoRA on proj
    ctrl = controller(model, [(FULL, 1.0)], [[good, missing, not_linear, bad_shape]], SUFFIX)
    assert ctrl.applied_count == 1
    assert len(ctrl.skipped) == 3


def test_linear_like_module_that_is_not_nn_linear():
    """ComfyUI's quantised Linear subclasses nn.Module only; it must still be hooked."""

    class QuantLinear(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.in_features, self.out_features = inner.in_features, inner.out_features
            self.inner = inner

        def forward(self, x):
            return self.inner(x)

    torch.manual_seed(0)
    model = ToyDiT(SUFFIX).double()
    model.blocks[0].qkv = QuantLinear(model.blocks[0].qkv)
    down, up, _ = make_lora(model.blocks[0].qkv.inner)
    ctrl = controller(model, [(FULL, 1.0)], [[target(PATH, down, up)]], SUFFIX)
    assert ctrl.applied_count == 1 and not ctrl.skipped
    cap = Capture(model.blocks[0].qkv)
    ctrl.inject()
    cap.after_inject()
    run(model, ctrl, latent(), context(), topts())
    img = seq_mask(SUFFIX, FULL) > 0
    assert not torch.equal(cap.final[:, img], cap.base[:, img])
    assert bit_identical(cap.final[:, ~img], cap.base[:, ~img])
