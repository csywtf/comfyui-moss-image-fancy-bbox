"""MOSS-Image Character Masks: each box's character outline and part masks, found with SAM3.

For every visible box with Detail on:

1. **Outline**: SAM3 segments the main object inside the box (a box prompt), giving the character's
   silhouette. If it finds nothing, the box itself stands in for the outline.
2. **Parts**: on a crop around the box and outline (more pixels for small faces), SAM3 finds each
   ticked part by text ("face", "eye:2", ...), and the results are pasted back.
3. **Ownership**: every pixel inside some outline belongs to the topmost box (last in the layer list)
   whose outline covers it. A box's parts keep only pixels it owns, plus unowned pixels just outside its
   outline (hair strands, edges); a pixel two boxes could both claim goes to the one on top.

The detail settings (which parts, prompts, denoise, detail LoRAs) live on this node, not in the main
node's regions, so changing them never invalidates the generated image in ComfyUI's cache.

Pure parts (settings, rectangles, ownership, preview) have no ComfyUI imports and are tested headlessly;
SAM3 is reached through :class:`Sam3Runner`, which tests replace.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from .geometry import centered_to_canvas
from .regions import BoxInfo, parse_document

log = logging.getLogger("moss_regional_lora")

Rect = Tuple[int, int, int, int]   # x0, y0, x1, y1 in image pixels, exclusive max


@dataclass(frozen=True)
class Part:
    name: str
    sam3_prompt: str     # SAM3 text prompt; ":N" = up to N detections (comfy/text_encoders/sam3_clip.py)
    hint: str            # default detail prompt for this part
    color: str           # preview colour


# Listed in the editor's order. The detailer works coarse to fine (DETAIL_ORDER).
PARTS: Tuple[Part, ...] = (
    Part("face", "face:1", "detailed face", "#ff9f1c"),
    Part("eyes", "eye:2", "detailed eyes", "#2ec4ff"),
    Part("hands", "hand:2", "detailed hands", "#3ddc84"),
    Part("hair", "hair:1", "detailed hair", "#b77dff"),
    Part("skin", "skin:4", "detailed skin", "#ffe14d"),
)
PART_NAMES = tuple(p.name for p in PARTS)
PART_BY_NAME = {p.name: p for p in PARTS}
DETAIL_ORDER = ("skin", "hair", "face", "hands", "eyes")
DEFAULT_PARTS_ON = ("face", "eyes")
DETAIL_VERSION = 1


class DetailError(ValueError):
    pass


# ----------------------------------------------------------------------------- detail settings
@dataclass(frozen=True)
class PartSettings:
    on: bool
    prompt: str = ""                     # "" = the part's hint
    denoise: Optional[float] = None      # None = the detailer's default


@dataclass(frozen=True)
class BoxDetail:
    enabled: bool
    prompt: str = ""
    use_box_loras: bool = True
    loras: Tuple[Tuple[str, float], ...] = ()
    parts: Dict[str, PartSettings] = field(default_factory=dict)

    def parts_on(self) -> List[str]:
        return [n for n in PART_NAMES if self.parts.get(n, PartSettings(False)).on]


def default_detail(label: str) -> BoxDetail:
    """Characters are detailed (face and eyes) unless turned off; other boxes start off."""
    return BoxDetail(enabled=(label == "character"),
                     parts={n: PartSettings(on=n in DEFAULT_PARTS_ON) for n in PART_NAMES})


def parse_detail(text) -> Dict[str, dict]:
    """The node's ``detail`` widget: ``{"version": 1, "boxes": {box_id: {...}}}``. Empty means defaults."""
    if text is None or not str(text).strip():
        return {}
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise DetailError(f"detail settings are not valid JSON: {e}") from None
    if not isinstance(data, dict):
        raise DetailError("detail settings must be a JSON object")
    boxes = data.get("boxes", {})
    if not isinstance(boxes, dict):
        raise DetailError("detail settings: 'boxes' must be an object keyed by box id")
    return {str(k): v for k, v in boxes.items() if isinstance(v, dict)}


def _bool(v, default):
    return v if isinstance(v, bool) else default


def _str(v):
    return v if isinstance(v, str) else ""


def _denoise(v) -> Optional[float]:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return min(1.0, max(0.0, float(v)))


def resolve_detail(box: BoxInfo, raw: Optional[dict]) -> BoxDetail:
    """A box's settings: what the editor stored for it, over the defaults for its label."""
    base = default_detail(box.label)
    if not raw:
        return base
    loras = []
    for e in raw.get("loras") or []:
        if isinstance(e, dict) and isinstance(e.get("lora"), str) and e["lora"]:
            w = e.get("weight", 1.0)
            if not isinstance(w, bool) and isinstance(w, (int, float)) and w != 0:
                loras.append((e["lora"], float(w)))
    parts = dict(base.parts)
    raw_parts = raw.get("parts") if isinstance(raw.get("parts"), dict) else {}
    for n in PART_NAMES:
        p = raw_parts.get(n)
        if isinstance(p, dict):
            parts[n] = PartSettings(on=_bool(p.get("on"), parts[n].on), prompt=_str(p.get("prompt")),
                                    denoise=_denoise(p.get("denoise")))
    return BoxDetail(enabled=_bool(raw.get("enabled"), base.enabled), prompt=_str(raw.get("prompt")),
                     use_box_loras=_bool(raw.get("use_box_loras"), True), loras=tuple(loras), parts=parts)


# ----------------------------------------------------------------------------- rectangles
def canvas_of(regions_text: str) -> Optional[Tuple[int, int]]:
    """The canvas size the main node adds to its ``regions`` output, if present."""
    try:
        c = json.loads(regions_text).get("canvas")
        w, h = int(c["width"]), int(c["height"])
        return (w, h) if w > 0 and h > 0 else None
    except (ValueError, TypeError, KeyError, AttributeError):
        return None


def box_rect(bbox, canvas: Tuple[int, int], image: Tuple[int, int]) -> Optional[Rect]:
    """A centred canvas box as an image rectangle (scaled if the image isn't the canvas size), clipped."""
    cw, ch = canvas
    iw, ih = image
    x0, y0, x1, y1 = centered_to_canvas(bbox, cw, ch)
    sx, sy = iw / cw, ih / ch
    r = (max(0, int(x0 * sx)), max(0, int(y0 * sy)), min(iw, int(round(x1 * sx))), min(ih, int(round(y1 * sy))))
    return r if r[2] > r[0] and r[3] > r[1] else None


def union_rect(a: Rect, b: Optional[Rect]) -> Rect:
    if b is None:
        return a
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def mask_rect(mask: torch.Tensor) -> Optional[Rect]:
    ys = torch.nonzero(mask.any(dim=1)).flatten()
    xs = torch.nonzero(mask.any(dim=0)).flatten()
    if len(ys) == 0:
        return None
    return (int(xs[0]), int(ys[0]), int(xs[-1]) + 1, int(ys[-1]) + 1)


def crop_rect(rect: Rect, margin: float, image: Tuple[int, int]) -> Rect:
    """``rect`` grown by ``margin`` of its size on each side, then towards a square (SAM3 resizes its input
    to a square, so a square crop isn't stretched), clipped to the image."""
    iw, ih = image
    x0, y0, x1, y1 = rect
    w, h = x1 - x0, y1 - y0
    x0, x1 = x0 - w * margin, x1 + w * margin
    y0, y1 = y0 - h * margin, y1 + h * margin
    side = max(x1 - x0, y1 - y0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    x0, x1 = cx - side / 2, cx + side / 2
    y0, y1 = cy - side / 2, cy + side / 2
    # slide back inside the image before clipping, so a box at the edge keeps its size where it can
    if x0 < 0:
        x0, x1 = 0, x1 - x0
    if x1 > iw:
        x0, x1 = x0 - (x1 - iw), iw
    if y0 < 0:
        y0, y1 = 0, y1 - y0
    if y1 > ih:
        y0, y1 = y0 - (y1 - ih), ih
    return (max(0, int(x0)), max(0, int(y0)), min(iw, int(round(x1))), min(ih, int(round(y1))))


def dilate(mask: torch.Tensor, r: int) -> torch.Tensor:
    if r <= 0:
        return mask
    m = mask[None, None].float()
    return F.max_pool2d(m, 2 * r + 1, stride=1, padding=r)[0, 0] > 0


def grow_radius(rect: Rect) -> int:
    return max(2, round(0.02 * max(rect[2] - rect[0], rect[3] - rect[1])))


# ----------------------------------------------------------------------------- ownership
def assign_parts(outlines: Sequence[torch.Tensor], raw_parts: Sequence[Dict[str, torch.Tensor]],
                 grow: Sequence[int]) -> List[Dict[str, torch.Tensor]]:
    """Keep each box's part masks to what it owns. ``outlines`` are in layer order (last = on top).

    A pixel inside outlines belongs to the topmost box covering it. A box's part may use pixels it owns
    and unowned pixels within ``grow`` of its outline; where two boxes' parts still meet, the top one wins.
    """
    if not outlines:
        return []
    shape = outlines[0].shape
    owner = torch.full(shape, -1, dtype=torch.int64)
    for i, o in enumerate(outlines):            # later boxes overwrite: topmost wins
        owner[o] = i
    out: List[Dict[str, torch.Tensor]] = []
    for i, (o, parts) in enumerate(zip(outlines, raw_parts)):
        allowed = dilate(o, grow[i]) & ((owner == i) | (owner == -1))
        out.append({n: m & allowed for n, m in parts.items()})
    taken = torch.zeros(shape, dtype=torch.bool)
    for i in range(len(out) - 1, -1, -1):        # top box first; lower boxes lose contested pixels
        claimed = torch.zeros(shape, dtype=torch.bool)
        for n in out[i]:
            out[i][n] = out[i][n] & ~taken
            claimed |= out[i][n]
        taken |= claimed
    return out


# ----------------------------------------------------------------------------- results
@dataclass
class CharacterMask:
    id: str
    name: str
    label: str
    caption: str
    color: str
    loras: Tuple[Tuple[str, float], ...]        # the box's own LoRAs (file, weight)
    detail: BoxDetail
    rect: Rect                                  # the box on the image
    outline: torch.Tensor                       # bool [H, W]
    outline_found: bool                         # False: SAM3 found nothing, the box stands in
    parts: Dict[str, torch.Tensor]              # bool [H, W] per ticked part
    found: Dict[str, bool]                      # per ticked part: any pixels after ownership


@dataclass
class FrameMasks:
    width: int
    height: int
    boxes: List[CharacterMask]


@dataclass
class CharacterMasks:
    """What MOSS-Image Character Masks hands to the detailer: one entry per image in the batch."""
    frames: List[FrameMasks]

    def union(self) -> torch.Tensor:
        """All parts of all boxes as a ComfyUI MASK [B, H, W]."""
        out = []
        for f in self.frames:
            m = torch.zeros(f.height, f.width, dtype=torch.bool)
            for b in f.boxes:
                for pm in b.parts.values():
                    m |= pm
            out.append(m.float())
        return torch.stack(out)


# ----------------------------------------------------------------------------- SAM3
_MEMO: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_MEMO_SIZE = 96


def image_key(image: torch.Tensor) -> str:
    """Content hash of an image, so the same image re-run (e.g. from ComfyUI's cache) reuses SAM3 results."""
    return hashlib.blake2b(image.detach().contiguous().cpu().float().numpy().tobytes(), digest_size=16).hexdigest()


class Sam3Runner:
    """SAM3 through ComfyUI's own SAM3 Detect node, with results memoised by image content."""

    def __init__(self, model, clip, threshold: float, refine_iterations: int):
        try:
            from comfy_extras.nodes_sam3 import SAM3_Detect
        except ImportError as e:
            raise RuntimeError("MOSS-Image Character Masks needs ComfyUI's SAM3 nodes (ComfyUI 0.44 or newer)") from e
        self._detect = SAM3_Detect
        self.model, self.clip = model, clip
        self.threshold, self.refine = float(threshold), int(refine_iterations)
        self._conds: Dict[str, object] = {}
        self._model_id = id(getattr(model, "model", model))

    def _run(self, image, **kw) -> torch.Tensor:
        out = self._detect.execute(self.model, image[None], threshold=self.threshold,
                                   refine_iterations=self.refine, individual_masks=False, **kw)
        masks = out.args[0] if hasattr(out, "args") else out[0]
        return masks[0].detach().cpu() > 0.5

    def _memo(self, key, fn):
        key = (self._model_id, self.threshold, self.refine) + key
        hit = _MEMO.get(key)
        if hit is not None:
            _MEMO.move_to_end(key)
            return hit.clone()
        value = fn()
        _MEMO[key] = value
        while len(_MEMO) > _MEMO_SIZE:
            _MEMO.popitem(last=False)
        return value.clone()

    def segment_box(self, image: torch.Tensor, img_key: str, rect: Rect) -> torch.Tensor:
        """The main object inside ``rect`` of ``image`` [H, W, C]: bool [H, W]."""
        x0, y0, x1, y1 = rect
        box = {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}
        return self._memo((img_key, "box", rect), lambda: self._run(image, bboxes=[box]))

    def find(self, image: torch.Tensor, img_key: str, crop: Rect, prompt: str) -> torch.Tensor:
        """``prompt`` found inside ``crop`` of ``image``: bool [H, W] (zero outside the crop)."""
        def run():
            x0, y0, x1, y1 = crop
            if prompt not in self._conds:
                self._conds[prompt] = self.clip.encode_from_tokens_scheduled(self.clip.tokenize(prompt))
            m = self._run(image[y0:y1, x0:x1], conditioning=self._conds[prompt])
            full = torch.zeros(image.shape[0], image.shape[1], dtype=torch.bool)
            full[y0:y1, x0:x1] = m
            return full
        return self._memo((img_key, "find", crop, prompt), run)


# ----------------------------------------------------------------------------- one image
def mask_frame(image: torch.Tensor, boxes: Sequence[BoxInfo], details: Dict[str, BoxDetail],
               canvas: Optional[Tuple[int, int]], runner, margin: float) -> FrameMasks:
    """Outline and part masks for each detailed box of one image [H, W, C]."""
    H, W = int(image.shape[0]), int(image.shape[1])
    canvas = canvas or (W, H)
    key = image_key(image)
    todo = []
    for b in boxes:
        d = details[b.id]
        if not d.enabled:
            continue
        rect = box_rect(b.bbox, canvas, (W, H))
        if rect is None:
            log.info("[MOSS Character Masks] box '%s' is outside the image; skipped", b.name)
            continue
        outline = runner.segment_box(image, key, rect)
        found = bool(outline.any())
        if not found:
            outline = torch.zeros(H, W, dtype=torch.bool)
            outline[rect[1]:rect[3], rect[0]:rect[2]] = True
        crop = crop_rect(union_rect(rect, mask_rect(outline)), margin, (W, H))
        raw = {n: runner.find(image, key, crop, PART_BY_NAME[n].sam3_prompt) for n in d.parts_on()}
        todo.append((b, d, rect, outline, found, raw))
    owned = assign_parts([t[3] for t in todo], [t[5] for t in todo], [grow_radius(t[2]) for t in todo])
    out = []
    for (b, d, rect, outline, found, _), parts in zip(todo, owned):
        out.append(CharacterMask(id=b.id, name=b.name, label=b.label, caption=b.caption, color=b.color,
                                 loras=b.loras, detail=d, rect=rect, outline=outline, outline_found=found,
                                 parts=parts, found={n: bool(m.any()) for n, m in parts.items()}))
    return FrameMasks(W, H, out)


def compute(images: torch.Tensor, regions: str, detail: str, runner, margin: float) -> CharacterMasks:
    doc = parse_document(regions)
    stored = parse_detail(detail)
    details = {b.id: resolve_detail(b, stored.get(b.id)) for b in doc.boxes}
    canvas = canvas_of(regions)
    return CharacterMasks([mask_frame(images[i], doc.boxes, details, canvas, runner, margin)
                           for i in range(images.shape[0])])


# ----------------------------------------------------------------------------- preview
def _rgb(hex_color: str) -> Tuple[float, float, float]:
    h = hex_color.lstrip("#")
    try:
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except (ValueError, IndexError):
        return (0.8, 0.8, 0.8)


def render_preview(image: torch.Tensor, frame: FrameMasks) -> torch.Tensor:
    """The image dimmed outside the characters, each part tinted in its colour, boxes outlined: [H, W, 3]."""
    img = image[..., :3].detach().cpu().float().clone()
    inside = torch.zeros(frame.height, frame.width, dtype=torch.bool)
    for b in frame.boxes:
        inside |= b.outline
    img[~inside] *= 0.4
    for b in frame.boxes:
        c = torch.tensor(_rgb(b.color))
        img[b.outline] = img[b.outline] * 0.8 + c * 0.2
        for n in PART_NAMES:
            m = b.parts.get(n)
            if m is not None and m.any():
                img[m] = img[m] * 0.45 + torch.tensor(_rgb(PART_BY_NAME[n].color)) * 0.55
        x0, y0, x1, y1 = b.rect
        t = max(1, round(max(frame.width, frame.height) / 400))
        img[y0:min(y0 + t, y1), x0:x1] = c
        img[max(y1 - t, y0):y1, x0:x1] = c
        img[y0:y1, x0:min(x0 + t, x1)] = c
        img[y0:y1, max(x1 - t, x0):x1] = c
    return img.clamp(0, 1)


def label_map(frame: FrameMasks):
    """RGBA uint8 [H, W, 4] for the node's editor: R = box number (1-based, 0 = none), G = part bits
    (bit i = PARTS[i]), B = 255 inside that box's outline, A = 255."""
    import numpy as np
    H, W = frame.height, frame.width
    out = np.zeros((H, W, 4), dtype=np.uint8)
    out[..., 3] = 255
    for i, b in enumerate(frame.boxes):
        o = b.outline.numpy()                       # later boxes overwrite: topmost wins, as in ownership
        out[o, 0] = i + 1
        out[o, 2] = 255
    for i, b in enumerate(frame.boxes):
        bits = np.zeros((H, W), dtype=np.uint8)
        for k, n in enumerate(PART_NAMES):
            m = b.parts.get(n)
            if m is not None:
                bits[m.numpy()] |= (1 << k)
        sel = bits > 0
        out[sel, 0] = i + 1
        out[sel, 1] = bits[sel]
        out[sel, 2] = np.where(b.outline.numpy()[sel], 255, 0)
    return out
