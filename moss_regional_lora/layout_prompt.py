"""MOSS-Image layout prompt, in the exact format of ComfyUI's ``MOSSImageLayoutPrompt`` node.

That node (comfy_extras/nodes_moss_image.py) writes, compactly and with ``ensure_ascii=False``::

    {"summary": {"caption": <prompt, stripped>},
     "items": [{"label": "object" | "character" | "text", "caption": <desc>,
                ["text": <text to render>,]  ["palette": [<hex>, ...],]
                "bbox": [y0, x0, y1, x1]}, ...],
     "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}

with integer pixel boxes measured from the canvas centre, y down: the same convention as this
node's regions, so boxes are copied straight across (non-integer hand-typed values are widened to the
enclosing integers, as that node does). With no items it returns the caption unchanged.
"""

from __future__ import annotations

import json
import math
from typing import Sequence

from .regions import PromptItem


def _enclosing(bbox):
    y0, x0, y1, x1 = bbox
    return [math.floor(y0), math.floor(x0), math.ceil(y1), math.ceil(x1)]


def build_layout_prompt(caption: str, items: Sequence[PromptItem]) -> str:
    if not items:
        return caption
    out = []
    for it in items:
        item = {"label": it.label, "caption": it.caption}
        if it.label == "text":
            item["text"] = it.text
        if it.palette:
            item["palette"] = list(it.palette)
        item["bbox"] = _enclosing(it.bbox)
        out.append(item)
    layout = {"summary": {"caption": caption.strip()}, "items": out,
              "bbox_coord_space": "abs_centralized", "bbox_format": "yxyx"}
    return json.dumps(layout, ensure_ascii=False, separators=(",", ":"))
