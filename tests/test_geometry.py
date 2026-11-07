"""Acceptance criterion 7: mask geometry. Pure Python, no torch needed."""

import pytest

from moss_regional_lora.geometry import (GeometryError, canvas_to_centered, centered_to_canvas, coverage_grid,
                                         grid_coverage, patch_grid_size, raster_flatten)


def mass(grid):
    return sum(sum(r) for r in grid)


def n_tokens(grid):
    return len(grid) * len(grid[0])


@pytest.mark.parametrize("w,h,vae,patch", [(64, 64, 8, 2), (1024, 768, 16, 1), (512, 512, 8, 1), (96, 160, 16, 2)])
def test_full_canvas_box_is_all_ones(w, h, vae, patch):
    g = coverage_grid([-(h // 2), -(w // 2), h - h // 2, w - w // 2], w, h, vae, patch)
    assert all(v == 1.0 for row in g for v in row)


@pytest.mark.parametrize("box", [[1000, 1000, 2000, 2000], [-900, -900, -600, -600], [-10, 600, 10, 700]])
def test_off_canvas_box_is_all_zeros(box):
    g = coverage_grid(box, 1024, 1024, 16, 1)
    assert all(v == 0.0 for row in g for v in row)


@pytest.mark.parametrize("quadrant", [[-512, -512, 0, 0], [-512, 0, 0, 512], [0, -512, 512, 0], [0, 0, 512, 512]])
def test_quadrant_box_covers_a_quarter_by_mass(quadrant):
    g = coverage_grid(quadrant, 1024, 1024, 16, 1)
    assert mass(g) == n_tokens(g) / 4


def test_quadrant_with_soft_edges_still_a_quarter():
    # odd patch grid (5x5 at 80px/16): a quadrant boundary cuts the middle row/column of patches in half
    g = coverage_grid([-40, -40, 0, 0], 80, 80, 16, 1)
    assert mass(g) == pytest.approx(n_tokens(g) / 4)
    assert g[2][2] == 0.25 and g[2][0] == 0.5 and g[0][2] == 0.5


def test_token_zero_is_top_left_patch():
    # a box exactly covering the top-left 16x16 px patch of a 1024x1024 canvas
    g = coverage_grid([-512, -512, -496, -496], 1024, 1024, 16, 1)
    flat = raster_flatten(g)
    assert flat[0] == 1.0 and sum(flat) == 1.0
    # and the patch right of it is token 1, the patch below it is token `cols`
    right = raster_flatten(coverage_grid([-512, -496, -496, -480], 1024, 1024, 16, 1))
    below = raster_flatten(coverage_grid([-496, -512, -480, -496], 1024, 1024, 16, 1))
    assert right.index(1.0) == 1 and below.index(1.0) == 64


@pytest.mark.parametrize("box", [[-100, -300, 50, 300], [-512, -77, 512, 77], [0, -1, 1, 1]])
def test_centre_aligned_box_is_horizontally_symmetric(box):
    g = coverage_grid(box, 1024, 1024, 16, 1)
    for row in g:
        assert row == row[::-1]


@pytest.mark.parametrize("box", [[-512, -512, 0, 0], [-300, -123, 77, 401], [-37, -5, 211, 19], [500, 500, 900, 900]])
def test_mask_mass_is_invariant_to_patch_size(box):
    w = h = 1024
    fractions = []
    for vae, patch in [(8, 1), (8, 2), (16, 1), (16, 2), (8, 4)]:
        g = coverage_grid(box, w, h, vae, patch)
        fractions.append(mass(g) / n_tokens(g))
    assert all(f == pytest.approx(fractions[0], abs=1e-12) for f in fractions)
    # and equals the box area inside the canvas over the canvas area
    x0, y0, x1, y1 = centered_to_canvas(box, w, h)
    inside = max(0, min(x1, w) - max(x0, 0)) * max(0, min(y1, h) - max(y0, 0))
    assert fractions[0] == pytest.approx(inside / (w * h))


def test_partial_patch_gets_fractional_value():
    g = coverage_grid([-512, -512, -504, -508], 1024, 1024, 16, 1)    # 4 x 8 px of a 16 x 16 patch
    assert g[0][0] == pytest.approx(4 * 8 / 256)
    assert 0 < g[0][0] < 1


def test_y_points_down():
    top = coverage_grid([-512, -512, -256, 512], 1024, 1024, 16, 1)
    assert all(v == 1.0 for v in top[0]) and all(v == 0.0 for v in top[-1])


def test_centre_matches_moss_layout_prompt_for_odd_sizes():
    # ComfyUI's MOSSImageLayoutPrompt uses width // 2, height // 2 as the origin
    assert centered_to_canvas([0, 0, 0, 0], 33, 17) == (16, 8, 16, 8)
    assert canvas_to_centered(16, 8, 20, 9, 33, 17) == [0, 0, 1, 4]


def test_round_trip():
    for box in ([-10, -20, 30, 40], [0, 0, 0, 0], [-512, -512, 512, 512]):
        x0, y0, x1, y1 = centered_to_canvas(box, 1024, 768)
        assert canvas_to_centered(x0, y0, x1, y1, 1024, 768) == box


def test_invalid_inputs():
    with pytest.raises(GeometryError):
        coverage_grid([10, 0, 0, 10], 64, 64, 8, 2)       # y_max < y_min
    with pytest.raises(GeometryError):
        coverage_grid([0, 0, 10], 64, 64, 8, 2)
    with pytest.raises(GeometryError):
        patch_grid_size(100, 64, 8, 2)                    # 100 not a multiple of 16


@pytest.mark.parametrize("box", [[-512, -512, 0, 0], [-300, -17, 211, 400], [-10, 600, 10, 700]])
def test_grid_coverage_matches_the_patch_grid_and_scales_proportionally(box):
    assert grid_coverage(box, 1024, 768, 48, 64) == coverage_grid(box, 1024, 768, 16, 1)
    # the same canvas on a half-resolution grid covers the same fraction of the image
    half = grid_coverage(box, 1024, 768, 24, 32)
    assert abs(mass(half) / n_tokens(half) - mass(coverage_grid(box, 1024, 768, 16, 1)) / (48 * 64)) < 1e-12
    with pytest.raises(GeometryError):
        grid_coverage(box, 1024, 768, 0, 64)
