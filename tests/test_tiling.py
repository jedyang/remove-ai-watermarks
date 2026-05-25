"""Unit tests for the pure tiling helpers (no GPU/model required).

``tiling.py`` imports torch at module top, so skip cleanly when torch is
absent. The helpers themselves are pure numpy/PIL/math -- they decide how a
large image is split into overlapping tiles and blended back, so a regression
here would seam or crop the CtrlRegen output wrongly.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from PIL import Image

from remove_ai_watermarks.noai.ctrlregen.tiling import (
    make_blend_weight,
    resize_center_crop,
    tile_positions,
)


class TestTilePositions:
    def test_image_smaller_than_tile_single_position(self):
        assert tile_positions(500, 512, 64) == [0]

    def test_image_equal_to_tile_single_position(self):
        assert tile_positions(512, 512, 64) == [0]

    def test_first_is_zero_last_is_total_minus_tile(self):
        # The tiles must fully cover the span: first starts at 0, last ends at
        # the far edge (start == total - tile), or the image's edge is missed.
        pos = tile_positions(2000, 512, 64)
        assert pos[0] == 0
        assert pos[-1] == 2000 - 512

    def test_overlap_positions_are_monotonic_and_exact(self):
        assert tile_positions(1000, 512, 64) == [0, 244, 488]

    def test_zero_overlap_tiles_are_contiguous(self):
        # 1024 wide, 512 tile, no overlap -> two tiles butting at 512.
        assert tile_positions(1024, 512, 0) == [0, 512]


class TestMakeBlendWeight:
    def test_zero_overlap_is_all_ones(self):
        w = make_blend_weight(8, 8, 0)
        assert w.shape == (8, 8)
        assert w.dtype == np.float64
        assert np.all(w == 1.0)

    def test_overlap_ramps_corners_to_zero_center_to_one(self):
        w = make_blend_weight(16, 16, 4)
        assert w[0, 0] == 0.0  # cosine ramp starts at 0
        assert w[8, 8] == 1.0  # center is unweighted
        assert w.max() == 1.0
        assert w.min() == 0.0

    def test_weight_is_point_symmetric(self):
        # Symmetric ramps on both edges -> mask equals its 180-degree rotation,
        # so opposite tile seams blend identically.
        w = make_blend_weight(16, 16, 4)
        assert np.allclose(w, w[::-1, ::-1])


class TestResizeCenterCrop:
    @pytest.mark.parametrize(("width", "height"), [(400, 800), (800, 400), (300, 300), (1000, 1001)])
    def test_output_is_always_square_of_requested_size(self, width: int, height: int):
        out = resize_center_crop(Image.new("RGB", (width, height)), 256)
        assert out.size == (256, 256)

    def test_default_size_is_512(self):
        out = resize_center_crop(Image.new("RGB", (640, 480)))
        assert out.size == (512, 512)
