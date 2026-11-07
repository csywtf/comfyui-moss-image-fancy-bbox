"""A toy single-stream transformer and helpers. Nothing here knows MOSS-Image's real layout:
tests are parametrised over every layout the node supports."""

import torch
import torch.nn as nn

from moss_regional_lora.gating import RegionalLoraController
from moss_regional_lora.layout import Layout
from moss_regional_lora.lora_file import LoraTarget
from moss_regional_lora.regions import RegionSpec

# 64x64 canvas, vae 8, patch 2 -> 4x4 patch grid of 16 px cells -> 16 image tokens
W = H = 64
VAE = 8
PATCH = 2
ROWS = COLS = 4
N_IMG = ROWS * COLS
N_TEXT = 5
DIM = 12
LATENT_C = 3


class Block(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        a = torch.softmax(q @ k.transpose(-1, -2) / q.shape[-1] ** 0.5, dim=-1) @ v
        return x + self.proj(a)


class ToyDiT(nn.Module):
    """Builds the joint sequence according to ``layout`` and runs two blocks."""

    def __init__(self, layout: Layout, dim=DIM, patch=PATCH):
        super().__init__()
        self.layout = layout
        self.patch = patch
        self.img_in = nn.Linear(LATENT_C * patch * patch, dim)
        self.blocks = nn.ModuleList([Block(dim), Block(dim)])
        self.out = nn.Linear(dim, LATENT_C * patch * patch)

    def tokens(self, latent):
        B, C, h, w = latent.shape
        p = self.patch
        t = latent.reshape(B, C, h // p, p, w // p, p).permute(0, 2, 4, 1, 3, 5).reshape(B, (h // p) * (w // p), C * p * p)
        return t  # raster order

    def forward(self, latent, context, refs=()):
        img = self.img_in(self.tokens(latent))
        src = [self.img_in(self.tokens(r)) for r in refs]
        lay = self.layout
        parts = []
        if lay.source_tokens == "start":
            parts += src
        if lay.text_position == "prefix":
            parts += [context]
            if lay.source_tokens == "between":
                parts += src
            parts += [img]
        else:
            parts += [img]
            if lay.source_tokens == "between":
                parts += src
            parts += [context]
        if lay.source_tokens == "end":
            parts += src
        x = torch.cat(parts, dim=1)
        for b in self.blocks:
            x = b(x)
        start = lay.image_start(context.shape[1], sum(s.shape[1] for s in src))
        return self.out(x[:, start:start + img.shape[1]])


def make_lora(module: nn.Linear, rank=2, seed=0, alpha=None, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    down = torch.randn(rank, module.in_features, generator=g, dtype=dtype)
    up = torch.randn(module.out_features, rank, generator=g, dtype=dtype)
    return down, up, alpha


def target(path, down, up, alpha=None, offset=0, size=None):
    return LoraTarget(source_prefix=path, weight_key=f"diffusion_model.{path}.weight", module_path=path,
                      down=down, up=up, alpha_scale=(alpha / down.shape[0]) if alpha is not None else 1.0,
                      out_offset=offset, out_size=size)


def controller(model, regions, targets, layout, ramp=(0.0, 0.0), steps=10, strict=True):
    specs = [RegionSpec(index=i, name=f"r{i}", bbox=tuple(b), lora=f"l{i}", weight=w)
             for i, (b, w) in enumerate(regions)]
    return RegionalLoraController(specs, targets, layout, W, H, VAE, PATCH, ramp[0], ramp[1], steps, strict).bind(model)


def latent(batch=1, dtype=torch.float64):
    return torch.zeros(batch, LATENT_C, H // VAE, W // VAE, dtype=dtype)


def context(batch=1, n=N_TEXT, dtype=torch.float64, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(batch, n, DIM, generator=g, dtype=dtype)


def topts(cond_or_uncond=(0,), step=0, steps=10):
    sample_sigmas = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64)
    return {"cond_or_uncond": list(cond_or_uncond), "sigmas": sample_sigmas[step:step + 1].clone(),
            "sample_sigmas": sample_sigmas}


def bits(t: torch.Tensor) -> torch.Tensor:
    """Bit pattern view so -0.0 vs +0.0 and NaN payloads count as differences."""
    return t.contiguous().view({torch.float64: torch.int64, torch.float32: torch.int32,
                                torch.float16: torch.int16, torch.bfloat16: torch.int16}[t.dtype])


def bit_identical(a, b) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(bits(a), bits(b))


ALL_LAYOUTS = [Layout(t, "raster", s) for t in ("prefix", "suffix") for s in ("none", "start", "between", "end")]

FULL = (-H // 2, -W // 2, H // 2, W // 2)           # whole canvas
TOP_LEFT_QUADRANT = (-H // 2, -W // 2, 0, 0)
OFF_CANVAS = (H, W, 2 * H, 2 * W)
