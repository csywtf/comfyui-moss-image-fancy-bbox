"""Per-box strength curves: how much of a box's LoRA weight applies at each point of the sampling run.

A curve is a list of points, stored in the region JSON as::

    "curve": [{"x": 0.0, "y": 1.0}, {"x": 0.4, "y": 1.0, "hold": true}, {"x": 1.0, "y": 0.0}]

``x`` is the progress through the sampling run (0 = first step, 1 = last step) and ``y`` the fraction
of the box's weight applied (0..1). Between two points the value is interpolated; the segment that
*starts* at a point takes that point's shape fields:

  ``bend``  in [-1, 1], default 0: shapes the segment as ``t ** (4 ** bend)``. 0 is a straight line;
            positive bends ease in (slow start), negative bends ease out (fast start).
  ``hold``  default false: keep the start point's value until the next point, then jump (a hard step).

Before the first point the value is the first point's y, after the last point the last point's y.
A box with no curve uses the node's ramp_start/ramp_end instead (workflows saved before curves).

web/curve.js implements the same function for drawing; tests/web/fixtures/curves.json pins both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


class CurveError(ValueError):
    pass


@dataclass(frozen=True)
class CurvePoint:
    x: float
    y: float
    bend: float = 0.0
    hold: bool = False


def _num(v, name, where):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v or v in (float("inf"), float("-inf")):
        raise CurveError(f"{where}: curve point '{name}' must be a finite number, got {v!r}")
    return float(v)


def parse_curve(raw, where="curve") -> Tuple[CurvePoint, ...]:
    """Validate a curve from JSON. x and y are clamped to [0, 1], bend to [-1, 1]; points sorted by x."""
    if not isinstance(raw, list) or not raw:
        raise CurveError(f"{where}: 'curve' must be a non-empty list of points")
    pts: List[CurvePoint] = []
    for i, p in enumerate(raw):
        if not isinstance(p, dict):
            raise CurveError(f"{where}: curve point {i} is not an object")
        x = min(1.0, max(0.0, _num(p.get("x"), "x", where)))
        y = min(1.0, max(0.0, _num(p.get("y"), "y", where)))
        bend = min(1.0, max(-1.0, _num(p.get("bend", 0.0), "bend", where)))
        hold = p.get("hold", False)
        if not isinstance(hold, bool):
            raise CurveError(f"{where}: curve point 'hold' must be true or false")
        pts.append(CurvePoint(x, y, bend, hold))
    return tuple(sorted(pts, key=lambda q: q.x))   # stable: equal x keep their order (a vertical jump)


def shape(t: float, bend: float) -> float:
    if bend == 0.0:
        return t
    return t ** (4.0 ** bend)


def evaluate(points: Sequence[CurvePoint], progress: float) -> float:
    """Fraction of the box weight at ``progress`` in [0, 1]."""
    p = min(1.0, max(0.0, float(progress)))
    if p <= points[0].x:
        return points[0].y
    for a, b in zip(points, points[1:]):
        if p < b.x:
            if a.hold:
                return a.y
            t = (p - a.x) / (b.x - a.x)        # b.x > a.x here, since a.x < p < b.x
            return a.y + (b.y - a.y) * shape(t, a.bend)
    return points[-1].y


def legacy_ramp(ramp_start: float, ramp_end: float) -> Tuple[CurvePoint, ...]:
    """The node-level linear ramp (0 until ramp_start, rising to 1 at ramp_end) as a curve."""
    rs, re = min(1.0, max(0.0, ramp_start)), min(1.0, max(0.0, ramp_end))
    if re <= rs:  # a step at ramp_start
        return (CurvePoint(0.0, 0.0, hold=True), CurvePoint(rs, 1.0), CurvePoint(1.0, 1.0)) if rs > 0 else (CurvePoint(0.0, 1.0),)
    return (CurvePoint(0.0, 0.0), CurvePoint(rs, 0.0), CurvePoint(re, 1.0), CurvePoint(1.0, 1.0))
