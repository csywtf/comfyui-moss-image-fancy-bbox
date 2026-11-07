"""Parse the editor's region JSON into the list the node applies.

The JSON is authored by the in-node editor (web/state.js) and may be hand-edited.
Schema (version 1)::

    {
      "version": 1,
      "regions": [
        {"id": "...", "name": "...", "bbox": [y_min, x_min, y_max, x_max],
         "lora": "file.safetensors", "weight": 1.0,
         "visible": true, "locked": false, "color": "#rrggbb", "group": null | "<group id>",
         ...unknown fields are preserved by the editor and ignored here...}
      ],
      "groups": [{"id": "...", "name": "...", "collapsed": false}],
      "editor": {...view settings, never affect output...}
    }

Array order is list order: index 0 is the top row of the layer list and is drawn
first (underneath); the last element is drawn last (on top). Hidden regions are
excluded from the output entirely (8.6d). Groups have no effect on output.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import List, Tuple

from .geometry import GeometryError, validate_bbox

SCHEMA_VERSION = 1


class RegionsError(ValueError):
    pass


@dataclass(frozen=True)
class RegionSpec:
    index: int          # position in the emitted list
    name: str
    bbox: Tuple[float, float, float, float]
    lora: str
    weight: float


def parse_regions(text: str) -> Tuple[List[RegionSpec], List[str]]:
    """Return (regions to apply, notes). Raises RegionsError on malformed input."""
    notes: List[str] = []
    if text is None or not str(text).strip():
        return [], ["regions JSON is empty"]
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
        if not lora:
            notes.append(f"region '{name}' has no LoRA selected; skipped")
            continue
        if weight == 0:
            notes.append(f"region '{name}' has weight 0; skipped")
            continue
        out.append(RegionSpec(index=len(out), name=name, bbox=bbox, lora=lora, weight=float(weight)))
    return out, notes
