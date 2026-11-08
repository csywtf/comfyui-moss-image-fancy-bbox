"""MOSS-Image Regional Detailer: redraw each character's parts with that character's prompt and LoRAs.

Input: the image and MOSS-Image Character Masks' result. For each image, characters are detailed back to
front (layer order) and each character's parts coarse to fine (skin, hair, face, hands, eyes), each part
as its own pass on the image so far:

1. **Crop** the part's mask (grown and feathered) plus padding, at least ``crop_size / max_upscale`` px
   on a side so small parts keep some context. A crop smaller than ``crop_size`` is upscaled by a whole
   number (Lanczos), up to ``crop_size`` and ``max_upscale``; a larger one is sampled at its own size.
   Either way it is padded (not resized) to whole 16 px latent cells.
   A part in pieces far apart (hands on either side of the body) is detailed as one pass per group of
   pieces whose crops would overlap, so each gets its own upscaled crop; two eyes stay one pass.
2. **Re-sample** it with a noise mask, so only the part changes: the prompt is the box's caption, the
   box's detail prompt and the part's prompt (or its hint, "detailed eyes"); the LoRAs are the box's own
   (unless turned off) plus its detail-only LoRAs; the denoise is the part's own or the node default.
3. **Scale back** by area averaging (the exact inverse of a whole-number upscale's size) and **restore
   detail**: the crop is also decoded without being redrawn, and what that round trip lost (resizing and
   the VAE) is added back, so only the model's changes are kept. At denoise 0 the image comes back
   unchanged.
4. **Blend** the result back with the feathered mask. Pixels outside it are never touched.

The sampling itself is a callable handed in by the node (ComfyUI's VAE and sampler); everything here is
plain torch, tested without ComfyUI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

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


def plan(masks: CharacterMasks, default_denoise: float, padding: float = 0.3, min_side: float = 0.0,
         margin: float = 0.0) -> Iterator[Pass]:
    """Every pass, in order: per image, characters back to front, parts coarse to fine. Parts not found
    (empty masks) are skipped. A part in pieces far apart (two hands on opposite sides) becomes one pass
    per group of pieces whose crops would overlap (see :func:`split_mask`), each with its own crop."""
    for f, frame in enumerate(masks.frames):
        for b in frame.boxes:
            for part in DETAIL_ORDER:
                m = b.parts.get(part)
                if m is None or not bool(m.any()):
                    continue
                s = b.detail.parts.get(part)
                denoise = s.denoise if s is not None and s.denoise is not None else default_denoise
                for piece in split_mask(m, padding, min_side, margin):
                    yield Pass(f, b.name, part, piece, part_prompt(b, part), pass_loras(b), float(denoise))


GRID = 256    # pieces are found on a grid this many cells across, so it stays quick on large images


def _components(grid) -> "list":
    """8-connected components of a 2-D bool numpy array: a list of (ys, xs) index arrays."""
    import numpy as np
    from collections import deque
    seen = np.zeros(grid.shape, dtype=bool)
    h, w = grid.shape
    out = []
    for y0, x0 in zip(*np.nonzero(grid)):
        if seen[y0, x0]:
            continue
        seen[y0, x0] = True
        q, ys, xs = deque([(y0, x0)]), [], []
        while q:
            y, x = q.popleft()
            ys.append(y)
            xs.append(x)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and grid[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
        out.append((np.array(ys), np.array(xs)))
    return out


def _grown(rect, padding: float, min_side: float, margin: float):
    x0, y0, x1, y1 = rect
    pad = margin + padding * max(x1 - x0, y1 - y0)
    w, h = max(x1 - x0 + 2 * pad, min_side), max(y1 - y0 + 2 * pad, min_side)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def split_mask(mask: torch.Tensor, padding: float, min_side: float = 0.0, margin: float = 0.0) -> List[torch.Tensor]:
    """A part's mask as groups of pieces to detail separately: pieces (connected areas) whose padded crops
    (``padding`` of their size plus ``margin`` px, at least ``min_side``) overlap stay together, so two
    eyes are one pass and hands on either side of the body are two. In reading order (top to bottom, then
    left to right). A mask in one group comes back as it is."""
    import numpy as np
    H, W = mask.shape
    cell = max(1, -(-max(H, W) // GRID))
    grid = (F.max_pool2d(mask[None, None].float(), cell, stride=cell, ceil_mode=True)[0, 0] > 0).numpy()
    comps = _components(grid)
    if len(comps) <= 1:
        return [mask]
    groups = []   # [cell indices of members, rect in pixels]
    for ys, xs in comps:
        groups.append([[(ys, xs)], (int(xs.min()) * cell, int(ys.min()) * cell,
                                    min(W, (int(xs.max()) + 1) * cell), min(H, (int(ys.max()) + 1) * cell))])
    merged = True
    while merged:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i][1], groups[j][1]
                if _overlap(_grown(a, padding, min_side, margin), _grown(b, padding, min_side, margin)):
                    groups[i] = [groups[i][0] + groups[j][0],
                                 (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))]
                    del groups[j]
                    merged = True
                    break
            if merged:
                break
    if len(groups) == 1:
        return [mask]
    groups.sort(key=lambda g: (g[1][1], g[1][0]))
    out = []
    for members, _ in groups:
        sel = np.zeros(grid.shape, dtype=bool)
        for ys, xs in members:
            sel[ys, xs] = True
        full = torch.from_numpy(sel).repeat_interleave(cell, 0).repeat_interleave(cell, 1)[:H, :W]
        out.append(mask & full)
    return out


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


def scale_for(rect: Rect, crop_size: int, max_upscale: float) -> float:
    """How much to scale a crop before sampling: a whole number from 1 up to ``max_upscale`` that keeps
    its long side within ``crop_size`` (whole numbers scale back exactly by area averaging); a crop far
    larger than ``crop_size`` (over 1.5x) is scaled down to it instead, to bound the cost."""
    long_side = max(rect[2] - rect[0], rect[3] - rect[1])
    if long_side > 1.5 * crop_size:
        return crop_size / long_side
    return float(max(1, min(int(max_upscale), int(crop_size // long_side))))


def padded(n: int) -> int:
    return -(-n // LATENT_CELL) * LATENT_CELL


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


def resize(img: torch.Tensor, w: int, h: int, method: str = "lanczos") -> torch.Tensor:
    """[H, W, C] (or [H, W]) to [h, w, ...] in float, unclamped. ``lanczos``: per channel in float with
    Pillow (ComfyUI ships it; bicubic otherwise); ``area``: box averaging, the exact inverse of a
    whole-number upscale's size."""
    flat = img.dim() == 2
    x = (img[..., None] if flat else img).detach().cpu().float()
    if tuple(x.shape[:2]) == (h, w):
        out = x
    elif method == "area":
        out = F.interpolate(x.permute(2, 0, 1)[None], size=(h, w), mode="area")[0].permute(1, 2, 0)
    else:
        try:
            import numpy as np
            from PIL import Image
            chans = [np.asarray(Image.fromarray(np.ascontiguousarray(x[..., c].numpy()), mode="F")
                                .resize((w, h), Image.LANCZOS)) for c in range(x.shape[-1])]
            out = torch.from_numpy(np.stack(chans, -1).copy())
        except ImportError:
            out = F.interpolate(x.permute(2, 0, 1)[None], size=(h, w), mode="bicubic", align_corners=False)[0].permute(1, 2, 0)
    return out[..., 0] if flat else out


def pad_to_cells(img: torch.Tensor, value=None) -> torch.Tensor:
    """Pad [h, w, C] (or [h, w]) on the right and bottom to whole latent cells: edge pixels repeated, or
    ``value``."""
    flat = img.dim() == 2
    x = img[..., None] if flat else img
    h, w = x.shape[:2]
    ph, pw = padded(h) - h, padded(w) - w
    if ph or pw:
        t = x.permute(2, 0, 1)[None]
        t = F.pad(t, (0, pw, 0, ph), mode="constant", value=value) if value is not None else F.pad(t, (0, pw, 0, ph), mode="replicate")
        x = t[0].permute(1, 2, 0)
    return x[..., 0] if flat else x


# sample(pixels [h, w, 3], noise_mask [h, w], pass, seed) -> (redrawn [h, w, 3], reference [h, w, 3] or None):
# ``reference`` is the same crop decoded without redrawing (the round trip alone), for detail restoration.
Sampler = Callable[[torch.Tensor, torch.Tensor, Pass, int], Tuple[torch.Tensor, Optional[torch.Tensor]]]


def detail_frame(image: torch.Tensor, passes: Sequence[Pass], sample: Sampler, crop_size: int, padding: float,
                 max_upscale: float, grow: int, feather: float, seed: int, on_pass=None,
                 restore: float = 1.0) -> torch.Tensor:
    """Run ``passes`` on one image [H, W, C]; each works on the image as detailed so far."""
    out = image.detach().cpu().float().clone()
    for k, p in enumerate(passes):
        soft = blend_mask(p.mask, grow, feather)
        x0, y0, x1, y1 = crop_region(soft > 0, padding, crop_size / max_upscale)
        w, h = x1 - x0, y1 - y0
        s = scale_for((x0, y0, x1, y1), crop_size, max_upscale)
        tw, th = max(1, round(w * s)), max(1, round(h * s))
        crop = out[y0:y1, x0:x1, :3]
        a = soft[y0:y1, x0:x1]
        big = pad_to_cells(resize(crop, tw, th).clamp(0, 1))
        big_mask = pad_to_cells(resize(a, tw, th, "area" if s < 1 else "lanczos").clamp(0, 1), value=0.0)
        result, reference = sample(big, big_mask, p, seed + k)
        back = "area" if s > 1 else "lanczos"          # area exactly undoes a whole-number upscale's size
        res = resize(result[:th, :tw, :3], w, h, back)
        if reference is not None and restore > 0:
            # keep only what the model changed: what the round trip (resizing, VAE) lost is put back
            delta = res - resize(reference[:th, :tw, :3], w, h, back)
            res = restore * (crop + delta) + (1 - restore) * res
        res = res.clamp(0, 1)
        a3 = a[..., None]
        blended = crop * (1 - a3) + res * a3
        out[y0:y1, x0:x1, :3] = torch.where(a3 > 0, blended, crop)   # outside the mask: exactly as it was
        if on_pass is not None:
            on_pass(k, p)
    return out


def detail(images: torch.Tensor, masks: CharacterMasks, sample: Sampler, *, denoise: float, crop_size: int,
           padding: float, max_upscale: float, grow: int, feather: float, seed: int, on_pass=None,
           restore: float = 1.0) -> torch.Tensor:
    passes = list(plan(masks, denoise, padding, crop_size / max_upscale, grow + feather))
    out = []
    for f in range(images.shape[0]):
        mine = [p for p in passes if p.frame == f] if f < len(masks.frames) else []
        out.append(detail_frame(images[f], mine, sample, crop_size, padding, max_upscale, grow, feather,
                                seed + 1000 * f, on_pass, restore) if mine else images[f].detach().cpu().float())
    return torch.stack(out)


def count_passes(masks: CharacterMasks, denoise: float, padding: float = 0.3, min_side: float = 0.0,
                 margin: float = 0.0) -> int:
    return sum(1 for _ in plan(masks, denoise, padding, min_side, margin))
