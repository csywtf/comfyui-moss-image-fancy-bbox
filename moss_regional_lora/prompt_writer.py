"""Write missing part prompts with a local text model (MOSS-Image Character Masks' optional prompt_llm input).

The model is any ComfyUI text encoder that can generate text (Qwen3, Qwen3-VL, Qwen3.5 such as the MOSS-Image
text encoder, Gemma 3/4), loaded with Load CLIP and called the way ComfyUI's Generate Text node calls it. For
each ticked, found part without a prompt it gets the box's caption and, if it can see images, a crop around
the part, and writes one short prompt for just that part. Plain torch apart from the clip calls, so it is
tested with a stand-in.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, replace
from typing import Dict, Optional

import torch
import torch.nn.functional as F

log = logging.getLogger("moss_regional_lora")

_SYSTEM_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "web",
                            "prompt_writer_system.txt")


def default_system_prompt() -> str:
    """The built-in instructions (web/prompt_writer_system.txt, also the editor's default)."""
    try:
        with open(_SYSTEM_FILE, encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ("Write one short image prompt describing only the given part of the character, in the "
                "character's art style. Reply with the prompt only.")


@dataclass(frozen=True)
class WriterSettings:
    max_length: int = 160
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 64
    min_p: float = 0.05
    repetition_penalty: float = 1.05
    seed: int = 0
    use_image: bool = True
    system_prompt: str = ""          # "" = the built-in one

    @property
    def system(self) -> str:
        return self.system_prompt.strip() or default_system_prompt()


PART_WORDS = {"face": "face", "eyes": "eyes", "left_hand": "hand (the one on the left side of the image)",
              "right_hand": "hand (the one on the right side of the image)", "hair": "hair", "skin": "skin"}


def user_message(caption: str, part: str, with_image: bool) -> str:
    who = caption.strip() or "(no description given)"
    seen = "\nThe image shows the area around this part." if with_image else ""
    return f"Character description: {who}\nPart to write a prompt for: {PART_WORDS.get(part, part)}{seen}"


def part_crop(image: torch.Tensor, mask: torch.Tensor, pad: float = 0.6, min_side: int = 192,
              max_side: int = 768) -> torch.Tensor:
    """The image [H, W, C] around ``mask`` (padded by ``pad`` of its size, at least ``min_side`` px), scaled
    down to at most ``max_side`` px on its long side: [h, w, 3]."""
    H, W = mask.shape
    ys, xs = torch.nonzero(mask, as_tuple=True)
    x0, x1, y0, y1 = int(xs.min()), int(xs.max()) + 1, int(ys.min()), int(ys.max()) + 1
    w = max(x1 - x0, 1)
    h = max(y1 - y0, 1)
    side_w = max(int(w * (1 + 2 * pad)), min(min_side, W))
    side_h = max(int(h * (1 + 2 * pad)), min(min_side, H))
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0 = int(max(0, min(W - side_w, cx - side_w / 2)))
    y0 = int(max(0, min(H - side_h, cy - side_h / 2)))
    crop = image[y0:y0 + min(side_h, H), x0:x0 + min(side_w, W), :3].float()
    long = max(crop.shape[0], crop.shape[1])
    if long > max_side:
        s = max_side / long
        t = crop.permute(2, 0, 1)[None]
        t = F.interpolate(t, size=(max(1, round(crop.shape[0] * s)), max(1, round(crop.shape[1] * s))),
                          mode="area")
        crop = t[0].permute(1, 2, 0)
    return crop.clamp(0, 1)


def clean(text: str) -> str:
    """The prompt from a reply: reasoning dropped, first non-empty line, no label, quotes or trailing dot."""
    _, sep, after = text.partition("</think>")
    text = after if sep else text
    for line in text.splitlines():
        line = line.strip().strip("`").strip()
        if not line or line.lower().startswith("<think>"):
            continue
        for label in ("prompt:", "final prompt:", "answer:"):
            if line.lower().startswith(label):
                line = line[len(label):].strip()
        line = line.strip('"“”\'')
        return line.rstrip(".").strip()
    return ""


def generate(clip, settings: WriterSettings, caption: str, part: str, crop: Optional[torch.Tensor]) -> str:
    """One prompt from the model; if it can't take the image, asked again without it."""
    attempts = [crop, None] if crop is not None else [None]
    last_error = None
    for img in attempts:
        try:
            extra = {"images": [img[None].to(torch.float32)]} if img is not None else {}
            tokens = clip.tokenize(user_message(caption, part, img is not None), min_length=1, thinking=False,
                                   system_prompt=settings.system, **extra)
            kw = dict(do_sample=settings.temperature > 0, max_length=settings.max_length,
                      temperature=max(settings.temperature, 0.01), top_k=settings.top_k, top_p=settings.top_p,
                      min_p=settings.min_p, repetition_penalty=settings.repetition_penalty, seed=settings.seed)
            try:
                ids = clip.generate(tokens, presence_penalty=0.0, **kw)
            except TypeError:           # older ComfyUI: no presence_penalty
                ids = clip.generate(tokens, **kw)
            text = clean(clip.decode(ids))
            if text:
                return text
        except Exception as e:  # noqa: BLE001 - a model that can't see images raises here; retry text-only
            last_error = e
            if img is not None:
                log.info("[MOSS Character Masks] prompt_llm could not use the image (%s); writing from the "
                         "caption only", e)
    if last_error is not None:
        raise RuntimeError(f"prompt_llm could not write a prompt: {last_error}") from last_error
    return ""


def fill_missing(masks, image: torch.Tensor, clip, settings: WriterSettings) -> Dict[str, Dict[str, str]]:
    """Write a prompt for every part of a detailed box that was found but has none yet (first image of the
    batch), store it in the boxes' settings of every image, and return {box id: {part: prompt}}."""
    if clip is None or not masks.frames:
        return {}
    if not callable(getattr(clip, "generate", None)):
        raise ValueError("prompt_llm: this CLIP can't generate text. Load a text model that can (Qwen3, "
                         "Qwen3-VL, Qwen3.5 such as the MOSS-Image text encoder, or Gemma) with Load CLIP.")
    written: Dict[str, Dict[str, str]] = {}
    frame = masks.frames[0]
    for b in frame.boxes:
        if not b.detail.enabled:
            continue
        for part, m in b.parts.items():
            s = b.detail.parts.get(part)
            if s is None or not s.on or (s.full_prompt or "").strip() or m is None or not bool(m.any()):
                continue
            crop = part_crop(image[0], m) if settings.use_image else None
            text = generate(clip, settings, b.caption, part, crop)
            if not text:
                log.warning("[MOSS Character Masks] prompt_llm wrote nothing for %s %s", b.name, part)
                continue
            written.setdefault(b.id, {})[part] = text
            log.info("[MOSS Character Masks] prompt_llm: %s %s -> %s", b.name, part, text)
    for f in masks.frames:                      # the same settings apply to every image of the batch
        for b in f.boxes:
            for part, text in written.get(b.id, {}).items():
                b.detail.parts[part] = replace(b.detail.parts[part], full_prompt=text)
    return written
