"""Parse the editor's region JSON into the list the node applies.

The JSON is authored by the in-node editor (web/state.js) and may be hand-edited.
Schema (version 2)::

    {
      "version": 2,
      "regions": [
        {"id": "...", "name": "...", "bbox": [y_min, x_min, y_max, x_max],
         "loras": [{"lora": "dir/file.safetensors", "weight": 1.0,
                    "curve": [{"x": 0, "y": 1}, ...]}, ...],   (curve: see curve.py; absent = node ramp)
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

Version 1 regions carried a single LoRA as top-level ``lora`` / ``weight`` / ``curve`` fields; they are
still read (as a one-entry ``loras`` list) when a region has no ``loras`` list.

A region does one or both of two jobs:
  * LoRA: each ``loras`` entry with a file and a non-zero ``weight`` is applied inside the box, as its
    own gated delta with its own strength curve.
  * Prompt: it has a ``caption`` (or, for a ``"text"`` label, text to render) -> it becomes an item of
    the MOSS-Image layout prompt, exactly as ComfyUI's MOSS-Image Layout Prompt node would write it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .curve import CurveError, CurvePoint, parse_curve
from .geometry import GeometryError, validate_bbox

SCHEMA_VERSION = 2
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
    curve: Optional[Tuple[CurvePoint, ...]] = None   # None: use the node's ramp_start/ramp_end


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
        where = f"region {i} ({name})"
        entries = _lora_entries(r, where)
        label = r.get("label") or "object"
        if label not in LABELS:
            raise RegionsError(f"{where}: 'label' must be one of {LABELS}, got {label!r}")
        caption = _opt_str(r, "caption", where)
        render = _opt_str(r, "text", where) if label == "text" else ""
        palette = _palette(r.get("palette"), where)
        has_prompt = bool(caption.strip() or render.strip())
        if has_prompt:
            items.append(PromptItem(name=name, bbox=bbox, label=label, caption=caption, text=render, palette=palette))

        active = [e for e in entries if e[0] and e[1] != 0]
        if not active and not has_prompt:
            notes.append(f"region '{name}' has neither a LoRA nor a caption; it does nothing")
        for lora, weight, curve in entries:
            if not lora:
                if weight:
                    notes.append(f"region '{name}' has a LoRA entry with no file selected; skipped")
                continue
            if weight == 0:
                notes.append(f"region '{name}': {lora} has weight 0; skipped")
                continue
            # one spec per (box, LoRA): each gets its own mask copy and its own curve
            out.append(RegionSpec(index=len(out), name=name, bbox=bbox, lora=lora, weight=weight, curve=curve))
    return ParsedRegions(out, items, notes)


def _lora_entry(e, where) -> Tuple[str, float, Optional[Tuple[CurvePoint, ...]]]:
    if not isinstance(e, dict):
        raise RegionsError(f"{where}: each 'loras' entry must be an object")
    lora = e.get("lora") or ""
    if not isinstance(lora, str):
        raise RegionsError(f"{where}: 'lora' must be a filename string")
    weight = e.get("weight", 1.0)
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight):
        raise RegionsError(f"{where}: 'weight' must be a finite number, got {weight!r}")
    curve = None
    if e.get("curve") is not None:
        try:
            curve = parse_curve(e["curve"], where)
        except CurveError as err:
            raise RegionsError(str(err)) from None
    return lora, float(weight), curve


def _lora_entries(r, where) -> List[Tuple[str, float, Optional[Tuple[CurvePoint, ...]]]]:
    """The region's LoRAs: its ``loras`` list, or (version 1) its top-level lora/weight/curve."""
    if r.get("loras") is not None:
        if not isinstance(r["loras"], list):
            raise RegionsError(f"{where}: 'loras' must be a list")
        return [_lora_entry(e, where) for e in r["loras"]]
    if r.get("lora"):
        return [_lora_entry({k: r.get(k) for k in ("lora", "weight", "curve") if k in r}, where)]
    if "lora" in r and r["lora"] is not None and not isinstance(r["lora"], str):
        raise RegionsError(f"{where}: 'lora' must be a filename string")
    return []
