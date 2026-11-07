"""Parse the editor's region JSON into the list the node applies.

The JSON is authored by the in-node editor (web/state.js) and may be hand-edited.
Schema (version 1)::

    {
      "version": 1,
      "regions": [
        {"id": "...", "name": "...", "bbox": [y_min, x_min, y_max, x_max],
         "lora": "file.safetensors", "weight": 1.0,
         "visible": true, "locked": false, "color": "#rrggbb", "group": null | "<group id>",
         "label": "object" | "character" | "text", "caption": "...", "text": "...",
         "palette": ["#RRGGBB", ...],
         ...unknown fields are preserved by the editor and ignored here...}
      ],
      "groups": [{"id": "...", "name": "...", "collapsed": false}],
      "editor": {...view settings, never affect output...}
    }

Array order is list order: index 0 is the top row of the layer list and is drawn
first (underneath); the last element is drawn last (on top). Hidden regions are
excluded from the output entirely (8.6d). Groups have no effect on output.

A region does one or both of two jobs:
  * LoRA: it has a ``lora`` and a non-zero ``weight`` -> the LoRA is applied inside its box.
  * Prompt: it has a ``caption`` (or, for a ``"text"`` label, text to render) -> it becomes an item of
    the MOSS-Image layout prompt, exactly as ComfyUI's MOSS-Image Layout Prompt node would write it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import List, Tuple

from .geometry import GeometryError, validate_bbox

SCHEMA_VERSION = 1
LABELS = ("object", "character", "text")
MAX_PALETTE = 5   # Create Bounding Boxes keeps at most five colours per box


class RegionsError(ValueError):
    pass


@dataclass(frozen=True)
class RegionSpec:
    index: int          # position in the emitted list
    name: str
    bbox: Tuple[float, float, float, float]
    lora: str
    weight: float


@dataclass(frozen=True)
class PromptItem:
    name: str
    bbox: Tuple[float, float, float, float]
    label: str                     # "object" | "character" | "text"
    caption: str
    text: str                      # only used for label "text"
    palette: Tuple[str, ...]


@dataclass
class ParsedRegions:
    loras: List[RegionSpec]
    prompt_items: List[PromptItem]
    notes: List[str]


def parse_regions(text: str) -> Tuple[List[RegionSpec], List[str]]:
    """Return (regions whose LoRA to apply, notes). Raises RegionsError on malformed input."""
    doc = parse_document(text)
    return doc.loras, doc.notes


def _palette(value, where) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, dict):          # same leniency as ComfyUI's normalize_palette
        value = list(value.values())
    if not isinstance(value, list):
        raise RegionsError(f"{where}: 'palette' must be a list of colours")
    return tuple(c.upper() for c in value if isinstance(c, str) and c)[:MAX_PALETTE]


def _opt_str(r, key, where) -> str:
    v = r.get(key)
    if v is None:
        return ""
    if not isinstance(v, str):
        raise RegionsError(f"{where}: '{key}' must be a string")
    return v


def parse_document(text: str) -> ParsedRegions:
    """Parse the region JSON into LoRA regions and layout-prompt items (both in list order)."""
    notes: List[str] = []
    if text is None or not str(text).strip():
        return ParsedRegions([], [], ["regions JSON is empty"])
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise RegionsError(f"regions is not valid JSON: {e}") from None
    if isinstance(data, list):  # bare list of regions, accepted for hand-editing convenience
        data = {"version": SCHEMA_VERSION, "regions": data}
    if not isinstance(data, dict):
        raise RegionsError("regions JSON must be an object with a 'regions' list")
    version = data.get("version", SCHEMA_VERSION)
    if not isinstance(version, int) or isinstance(version, bool):
        raise RegionsError(f"'version' must be an integer, got {version!r}")
    if version > SCHEMA_VERSION:
        notes.append(f"regions JSON version {version} is newer than this node ({SCHEMA_VERSION}); "
                     "unknown fields are ignored")
    raw = data.get("regions", [])
    if not isinstance(raw, list):
        raise RegionsError("'regions' must be a list")

    out: List[RegionSpec] = []
    items: List[PromptItem] = []
    for i, r in enumerate(raw):
        if not isinstance(r, dict):
            raise RegionsError(f"region {i} is not an object")
        name = str(r.get("name") or f"region {i + 1}")
        if r.get("visible", True) is False:
            continue
        try:
            bbox = validate_bbox(r.get("bbox"))
        except GeometryError as e:
            raise RegionsError(f"region {i} ({name}): {e}") from None
        lora = r.get("lora") or ""
        if not isinstance(lora, str):
            raise RegionsError(f"region {i} ({name}): 'lora' must be a filename string")
        weight = r.get("weight", 1.0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
            raise RegionsError(f"region {i} ({name}): 'weight' must be a finite number, got {weight!r}")
        where = f"region {i} ({name})"
        label = r.get("label") or "object"
        if label not in LABELS:
            raise RegionsError(f"{where}: 'label' must be one of {LABELS}, got {label!r}")
        caption = _opt_str(r, "caption", where)
        render = _opt_str(r, "text", where) if label == "text" else ""
        palette = _palette(r.get("palette"), where)
        has_prompt = bool(caption.strip() or render.strip())
        if has_prompt:
            items.append(PromptItem(name=name, bbox=bbox, label=label, caption=caption, text=render, palette=palette))

        if not lora:
            if not has_prompt:
                notes.append(f"region '{name}' has neither a LoRA nor a caption; it does nothing")
            continue
        if weight == 0:
            notes.append(f"region '{name}' has LoRA weight 0; LoRA skipped")
            continue
        out.append(RegionSpec(index=len(out), name=name, bbox=bbox, lora=lora, weight=float(weight)))
    return ParsedRegions(out, items, notes)
