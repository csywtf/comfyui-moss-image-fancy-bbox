"""Bounding-box geometry: centred pixel boxes -> latent grid -> patch grid -> coverage masks.

Pure Python, no torch, so the maths can be tested anywhere.

Box convention (FR-2), shared with ComfyUI's ``MOSSImageLayoutPrompt`` node:
``[y_min, x_min, y_max, x_max]`` in pixels, origin at the canvas centre, y pointing
down, x pointing right. The centre is ``(width // 2, height // 2)`` in top-left
canvas pixels, which is the same integer centre ``MOSSImageLayoutPrompt`` uses.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple


class GeometryError(ValueError):
    pass


def centered_to_canvas(bbox: Sequence[float], width: int, height: int) -> Tuple[float, float, float, float]:
    """``[y_min, x_min, y_max, x_max]`` centred -> ``(x0, y0, x1, y1)`` with a top-left origin."""
    y_min, x_min, y_max, x_max = validate_bbox(bbox)
    cx, cy = width // 2, height // 2
    return (x_min + cx, y_min + cy, x_max + cx, y_max + cy)


def canvas_to_centered(x0: float, y0: float, x1: float, y1: float, width: int, height: int) -> List[float]:
    """Inverse of :func:`centered_to_canvas`."""
    cx, cy = width // 2, height // 2
    return [y0 - cy, x0 - cx, y1 - cy, x1 - cx]


def validate_bbox(bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise GeometryError(f"bbox must be [y_min, x_min, y_max, x_max], got {bbox!r}")
    try:
        y_min, x_min, y_max, x_max = (float(v) for v in bbox)
    except (TypeError, ValueError):
        raise GeometryError(f"bbox values must be numbers, got {bbox!r}") from None
    if y_max < y_min or x_max < x_min:
        raise GeometryError(f"bbox {list(bbox)!r} has max < min; expected [y_min, x_min, y_max, x_max]")
    return y_min, x_min, y_max, x_max


def patch_grid_size(width: int, height: int, vae_downscale: int, patch_size: int) -> Tuple[int, int]:
    """(rows, cols) of the patch grid. Fails if the canvas does not tile exactly."""
    if vae_downscale < 1 or patch_size < 1:
        raise GeometryError("vae_downscale and patch_size must be >= 1")
    cell = vae_downscale * patch_size
    if width % cell or height % cell:
        raise GeometryError(
            f"canvas {width}x{height} is not a multiple of vae_downscale*patch_size = {cell}; "
            "the patch grid would not tile the canvas")
    return height // cell, width // cell


def _axis_coverage(lo: float, hi: float, n: int) -> List[float]:
    """Fraction of each unit cell ``[i, i+1)``, i in [0, n), covered by ``[lo, hi)``."""
    out = []
    for i in range(n):
        overlap = min(hi, i + 1.0) - max(lo, float(i))
        out.append(overlap if overlap > 0.0 else 0.0)
    return out


def coverage_grid(bbox: Sequence[float], width: int, height: int, vae_downscale: int, patch_size: int) -> List[List[float]]:
    """Coverage-weighted mask on the patch grid, ``rows x cols``, values in [0, 1].

    A patch fully inside the box is 1.0, fully outside 0.0, and a patch straddling an
    edge gets the fraction of its area inside the box (FR-4). Out-of-canvas parts of a
    box are simply not covered by any patch (clipped, not clamped).
    """
    rows, cols = patch_grid_size(width, height, vae_downscale, patch_size)
    x0, y0, x1, y1 = centered_to_canvas(bbox, width, height)
    # pixels -> latent cells -> patch cells
    lx0, ly0, lx1, ly1 = (v / vae_downscale for v in (x0, y0, x1, y1))
    px0, py0, px1, py1 = (v / patch_size for v in (lx0, ly0, lx1, ly1))
    cov_x = _axis_coverage(px0, px1, cols)
    cov_y = _axis_coverage(py0, py1, rows)
    return [[cy * cx for cx in cov_x] for cy in cov_y]


def raster_flatten(grid: List[List[float]]) -> List[float]:
    """Image-token order ``index = row * cols + col`` (IMAGE_ORDER = "raster")."""
    return [v for row in grid for v in row]
