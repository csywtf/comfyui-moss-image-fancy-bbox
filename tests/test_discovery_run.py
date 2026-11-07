"""End-to-end run of the discovery passes on the toy transformer.

ComfyUI's sampler, text encoder, VAE and conditioning helpers are replaced by minimal stand-ins, so
this checks the discovery's own logic (which block call is the main pass, the comparisons, the
single-patch probe and its image analysis, the edit pass, the report) against a model whose layout is
known. It does not check ComfyUI integration or anything about MOSS-Image itself.
"""

import sys
import types
import zlib

import pytest
import torch

import toy
from toy import ToyDiT
from moss_regional_lora import discovery as disc
from moss_regional_lora.layout import Layout

VAE = toy.VAE        # 8 px per latent cell
PATCH = toy.PATCH    # 2 latent cells per token -> 16 px patches


def untoken(tokens, B, C, h, w, p):
    return tokens.reshape(B, h // p, w // p, C, p, p).permute(0, 3, 1, 4, 2, 5).reshape(B, C, h, w)


class FakeClip:
    def tokenize(self, prompt):
        return prompt.split() or [""]

    def encode_from_tokens_scheduled(self, words):
        rows = [torch.randn(toy.DIM, generator=torch.Generator().manual_seed(zlib.crc32(w.encode()))) for w in words]
        return [[torch.stack(rows)[None].double(), {}]]


class FakeVae:
    def decode(self, latent):
        img = latent[:, :3].clamp(-3, 3) / 6 + 0.5
        return img.repeat_interleave(VAE, -2).repeat_interleave(VAE, -1).movedim(1, -1)

    def encode(self, image):
        lat = image.movedim(-1, 1)[:, :, ::VAE, ::VAE]
        return lat.double()


class FakePatcher:
    def __init__(self, dm):
        self.dm = dm

    def get_model_object(self, name):
        if name == "diffusion_model":
            return self.dm
        assert name == "latent_format"
        return types.SimpleNamespace(spacial_downscale_ratio=VAE, latent_channels=toy.LATENT_C)


@pytest.fixture
def fake_comfy(monkeypatch):
    torch.manual_seed(0)
    dm = ToyDiT(Layout("suffix", "raster", "none")).double()

    def sample(model, noise, steps, cfg, sampler, scheduler, pos, neg, latent, denoise=1.0, seed=None):
        ctx, extra = pos[0]
        refs = extra.get("reference_latents") or []
        in_seq = extra.get("reference_latents_method") == "index_timestep_zero"
        if refs and not in_seq:
            model.dm.blocks[0](torch.zeros(1, 4, toy.DIM, dtype=torch.float64))   # a precompute pass, outside any model call
        if refs:
            ctx = ctx + 0.01      # like MOSS-Image: a source image shifts the text tokens' position encoding
        old = model.dm.layout
        model.dm.layout = Layout("suffix", "raster", "end" if (refs and in_seq) else "none")
        try:
            x = noise.double()
            B, C, h, w = x.shape
            for _ in range(steps):
                out = model.dm(x, context=ctx, refs=refs if in_seq else ())
                x = x + 0.5 * untoken(out, B, C, h, w, PATCH)
        finally:
            model.dm.layout = old
        return x

    sample_mod = types.ModuleType("comfy.sample")
    sample_mod.sample = sample
    sample_mod.fix_empty_latent_channels = lambda model, lat: lat
    sample_mod.prepare_noise = lambda lat, seed: torch.randn(lat.shape, generator=torch.Generator().manual_seed(seed))
    utils_mod = types.ModuleType("comfy.utils")
    helpers = types.ModuleType("node_helpers")
    helpers.conditioning_set_values = lambda cond, values: [[c, {**d, **values}] for c, d in cond]
    comfy = types.ModuleType("comfy")
    comfy.sample, comfy.utils = sample_mod, utils_mod
    for name, mod in (("comfy", comfy), ("comfy.sample", sample_mod), ("comfy.utils", utils_mod), ("node_helpers", helpers)):
        monkeypatch.setitem(sys.modules, name, mod)
    return FakePatcher(dm)


def settings(**kw):
    s = dict(width=toy.W, height=toy.H, edit_methods=[None, "index_timestep_zero"], probe_strength=8.0)
    s.update(kw)
    return disc.Settings(**s)


def edit_image():
    return torch.rand(1, toy.H, toy.W, 3, generator=torch.Generator().manual_seed(7))


def test_full_run_recovers_the_toy_layout(fake_comfy):
    lines = []
    d = disc.Discovery(fake_comfy, FakeClip(), FakeVae(), settings(), lines.append)
    f = d.run_all(edit_image())
    report = "\n".join(lines)
    assert f.problems == [], report
    assert (f.vae_downscale, f.patch_size) == (VAE, PATCH)
    assert f.text_position == "suffix"
    assert f.image_order == "raster"
    assert f.source_tokens == "end"          # out of the sequence for 'default', appended for index_timestep_zero
    assert [n for n, _ in d.images] == ["prediction", "probe_r0_c0", "diff_r0_c0", "probe_r0_c1", "diff_r0_c1"]
    assert "text_position = suffix" in report and "NOT ESTABLISHED" not in report


def test_run_without_edit_image_leaves_source_tokens_unset(fake_comfy):
    lines = []
    f = disc.Discovery(fake_comfy, FakeClip(), FakeVae(), settings(), lines.append).run_all(None)
    report = "\n".join(lines)
    assert f.source_tokens is None and f.text_position == "suffix" and f.image_order == "raster"
    assert "source_tokens = NOT ESTABLISHED" in report and "'none' is exact" in report


def test_discovery_node_outputs_report_and_images(fake_comfy):
    from moss_regional_lora.discovery_node import MossLayoutDiscovery
    out = MossLayoutDiscovery().run(fake_comfy, FakeClip(), FakeVae(), toy.W, toy.H, 42, "a lake",
                                    edit_image=edit_image())
    report, images = out["result"]
    assert "text_position = suffix" in report and "source_tokens = end" in report
    assert images.shape == (5, toy.H, toy.W, 3)
    assert out["ui"]["moss_layout_report"] == [report]


def test_discovery_node_reports_errors_instead_of_crashing(fake_comfy):
    from moss_regional_lora.discovery_node import MossLayoutDiscovery
    out = MossLayoutDiscovery().run(fake_comfy, FakeClip(), FakeVae(), toy.W, toy.H, 42, "a lake",
                                    blocks="does_not_exist_but_is_a_path.x")
    report = out["result"][0]
    assert "ERROR" in report and "Block lists found: blocks" in report
