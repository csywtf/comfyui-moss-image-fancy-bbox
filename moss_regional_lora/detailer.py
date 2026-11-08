"""MOSS-Image Regional Detailer: redraw each character's parts with that character's prompt and LoRAs.

Input: the image and MOSS-Image Character Masks' result. For each image, characters are detailed back to
front (layer order) and each character's parts coarse to fine (skin, hair, face, hands, eyes), each part
as its own pass on the image so far:

1. **Crop** the part's mask (grown and feathered) plus padding, at least ``crop_size / max_upscale`` px
   on a side so small parts keep some context, and resize it to about ``crop_size`` on its long side
   (multiples of 16, MOSS-Image's latent cell).
2. **Re-sample** it with a noise mask, so only the part changes: the prompt is the box's caption, the
   box's detail prompt and the part's prompt (or its hint, "detailed eyes"); the LoRAs are the box's own
   (unless turned off) plus its detail-only LoRAs; the denoise is the part's own or the node default.
3. **Blend** the result back with the feathered mask. Pixels outside it are never touched.

The sampling itself is a callable handed in by the node (ComfyUI's VAE and sampler); everything here is
plain torch, tested without ComfyUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from .character_masks import DETAIL_ORDER, PART_BY_NAME, CharacterMask, CharacterMasks, Rect, dilate, mask_rect

LATENT_CELL = 16


@dataclass(frozen=True)
class Pass:
    frame: int
    box: str                                   # box name, for logs
    part: str
    mask: torch.Tensor                         # bool [H, W]
    prompt: str
    loras: Tuple[Tuple[str, float], ...]       # (file, weight), applied to the model only
    denoise: float


def part_prompt(b: CharacterMask, part: str) -> str:
    """Caption, then the box's detail prompt, then the part's prompt (or its hint)."""
    text = b.detail.parts[part].prompt.strip() or PART_BY_NAME[part].hint
    return ", ".join(t for t in (b.caption.strip(), b.detail.prompt.strip(), text) if t)


def pass_loras(b: CharacterMask) -> Tuple[Tuple[str, float], ...]:
    return (tuple(b.loras) if b.detail.use_box_loras else ()) + tuple(b.detail.loras)


def plan(masks: CharacterMasks, default_denoise: float) -> Iterator[Pass]:
    """Every pass, in order: per image, characters back to front, parts coarse to fine. Parts not found
    (empty masks) are skipped."""
    for f, frame in enumerate(masks.frames):
        for b in frame.boxes:
            for part in DETAIL_ORDER:
                m = b.parts.get(part)
                if m is None or not bool(m.any()):
                    continue
                s = b.detail.parts.get(part)
                denoise = s.denoise if s is not None and s.denoise is not None else default_denoise
                yield Pass(f, b.name, part, m, part_prompt(b, part), pass_loras(b), float(denoise))


def crop_region(mask: torch.Tensor, padding: float, min_side: float) -> Rect:
    """The mask's bounding box grown by ``padding`` of its longer side on every side, at least
    ``min_side`` wide and tall, kept inside the image (slid in from an edge rather than cut, where it fits)."""
    H, W = mask.shape
    x0, y0, x1, y1 = mask_rect(mask)
    pad = padding * max(x1 - x0, y1 - y0)
    w = min(W, max(x1 - x0 + 2 * pad, min_side))
    h = min(H, max(y1 - y0 + 2 * pad, min_side))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    nx0 = min(max(0.0, cx - w / 2), W - w)
    ny0 = min(max(0.0, cy - h / 2), H - h)
    return (int(nx0), int(ny0), min(W, int(round(nx0 + w))), min(H, int(round(ny0 + h))))


def target_size(rect: Rect, crop_size: int, max_upscale: float) -> Tuple[int, int]:
    """(width, height) to sample a crop at: about ``crop_size`` on the long side, upscaled at most
    ``max_upscale`` times, in whole latent cells."""
    w, h = rect[2] - rect[0], rect[3] - rect[1]
    scale = min(crop_size / max(w, h), max_upscale)
    snap = lambda v: max(LATENT_CELL, int(round(v * scale / LATENT_CELL)) * LATENT_CELL)
    return snap(w), snap(h)


def gaussian_blur(m: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a float [H, W] map (zero outside the image)."""
    if sigma <= 0:
        return m
    r = max(1, int(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=torch.float32)
    k = torch.exp(-(x * x) / (2 * sigma * sigma))
    k = (k / k.sum()).to(m.dtype)
    t = m[None, None]
    t = F.conv2d(F.pad(t, (r, r, 0, 0)), k.view(1, 1, 1, -1))
    t = F.conv2d(F.pad(t, (0, 0, r, r)), k.view(1, 1, -1, 1))
    return t[0, 0]


def blend_mask(mask: torch.Tensor, grow: int, feather: float) -> torch.Tensor:
    """Float [H, W] in [0, 1]: the part grown by ``grow`` px, its edge feathered over about ``feather`` px.
    The part itself is always 1, so it is fully replaced."""
    m = dilate(mask, int(grow)).float()
    soft = gaussian_blur(m, feather / 3.0) if feather > 0 else m
    return torch.maximum(soft, mask.float()).clamp(0, 1)


def resize(img: torch.Tensor, w: int, h: int) -> torch.Tensor:
    """[H, W, C] (or [H, W]) to [h, w, ...], bicubic with antialiasing, clamped to [0, 1]."""
    flat = img.dim() == 2
    t = (img[..., None] if flat else img).permute(2, 0, 1)[None].float()
    if t.shape[-2:] != (h, w):
        t = F.interpolate(t, size=(h, w), mode="bicubic", align_corners=False, antialias=True)
    out = t[0].permute(1, 2, 0).clamp(0, 1)
    return out[..., 0] if flat else out


Sampler = Callable[[torch.Tensor, torch.Tensor, Pass, int], torch.Tensor]


def detail_frame(image: torch.Tensor, passes: Sequence[Pass], sample: Sampler, crop_size: int, padding: float,
                 max_upscale: float, grow: int, feather: float, seed: int, on_pass=None) -> torch.Tensor:
    """Run ``passes`` on one image [H, W, C]. ``sample(pixels [h, w, 3], noise_mask [h, w], pass, seed)``
    returns the re-sampled crop [h, w, 3]. Each pass works on the image as detailed so far."""
    out = image.detach().cpu().float().clone()
    for k, p in enumerate(passes):
        soft = blend_mask(p.mask, grow, feather)
        x0, y0, x1, y1 = crop_region(soft > 0, padding, crop_size / max_upscale)
        tw, th = target_size((x0, y0, x1, y1), crop_size, max_upscale)
        crop = out[y0:y1, x0:x1, :3]
        a = soft[y0:y1, x0:x1]
        result = sample(resize(crop, tw, th), resize(a, tw, th), p, seed + k)
        result = resize(result[..., :3].detach().cpu().float(), x1 - x0, y1 - y0)
        a3 = a[..., None]
        blended = crop * (1 - a3) + result * a3
        # outside the feathered mask the original pixels are kept exactly (no resampling round trip)
        out[y0:y1, x0:x1, :3] = torch.where(a3 > 0, blended, crop)   # an alpha channel, if any, is kept
        if on_pass is not None:
            on_pass(k, p)
    return out


def detail(images: torch.Tensor, masks: CharacterMasks, sample: Sampler, *, denoise: float, crop_size: int,
           padding: float, max_upscale: float, grow: int, feather: float, seed: int, on_pass=None) -> torch.Tensor:
    passes = list(plan(masks, denoise))
    out = []
    for f in range(images.shape[0]):
        mine = [p for p in passes if p.frame == f] if f < len(masks.frames) else []
        out.append(detail_frame(images[f], mine, sample, crop_size, padding, max_upscale, grow, feather,
                                seed + 1000 * f, on_pass) if mine else images[f].detach().cpu().float())
    return torch.stack(out)


def count_passes(masks: CharacterMasks, denoise: float) -> int:
    return sum(1 for _ in plan(masks, denoise))
