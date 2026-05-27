"""Tests for the Doubao visible-watermark engine."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from remove_ai_watermarks.doubao_engine import DoubaoEngine, load_image_bgr

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "doubao-1.png"


# ── Locate ──────────────────────────────────────────────────────────


class TestLocate:
    def test_box_anchored_bottom_right(self):
        eng = DoubaoEngine()
        img = np.zeros((2048, 2048, 3), np.uint8)
        loc = eng.locate(img)
        # right and bottom edges sit close to the image corner (within margins)
        assert 2048 - (loc.x + loc.w) < int(2048 * 0.03)
        assert 2048 - (loc.y + loc.h) < int(2048 * 0.03)
        assert loc.is_fallback  # geometry anchor, no bundled template yet

    def test_box_scales_with_width(self):
        eng = DoubaoEngine()
        small = eng.locate(np.zeros((1024, 1024, 3), np.uint8))
        large = eng.locate(np.zeros((2048, 2048, 3), np.uint8))
        # width-relative geometry: 2x wider image -> ~2x wider box
        assert large.w == pytest.approx(small.w * 2, rel=0.1)


# ── Detect + remove on the real sample ──────────────────────────────


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample image not present")
class TestRealSample:
    def test_detects_watermark(self):
        eng = DoubaoEngine()
        det = eng.detect(load_image_bgr(SAMPLE))
        assert det.detected
        assert det.confidence > 0.0
        assert det.coverage > 0.04

    def test_remove_reduces_glyph_coverage(self):
        eng = DoubaoEngine()
        img = load_image_bgr(SAMPLE)
        before = eng.detect(img).coverage
        out = eng.remove_watermark(img)
        after = eng.detect(out).coverage
        # the inpaint should clear most glyph pixels from the corner box
        assert after < before * 0.5

    def test_pixels_outside_box_untouched(self):
        eng = DoubaoEngine()
        img = load_image_bgr(SAMPLE)
        out = eng.remove_watermark(img)
        # top-left quadrant is far from the bottom-right mark: must be identical
        h, w = img.shape[:2]
        assert np.array_equal(img[: h // 2, : w // 2], out[: h // 2, : w // 2])


# ── Negative + safety guard ─────────────────────────────────────────


class TestNegativeAndGuard:
    def test_clean_image_not_detected(self):
        eng = DoubaoEngine()
        # smooth gradient, no watermark
        ramp = np.tile(np.linspace(0, 255, 1024, dtype=np.uint8), (1024, 1))
        img = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
        det = eng.detect(img)
        assert not det.detected

    def test_clean_image_returned_unchanged(self):
        eng = DoubaoEngine()
        ramp = np.tile(np.linspace(0, 255, 1024, dtype=np.uint8), (1024, 1))
        img = cv2.cvtColor(ramp, cv2.COLOR_GRAY2BGR)
        out = eng.remove_watermark(img)
        assert np.array_equal(img, out)

    def test_document_background_guard(self):
        """A dense high-frequency corner (document-like) trips the coverage
        guard, so the image is left untouched rather than smeared."""
        eng = DoubaoEngine()
        rng = np.random.default_rng(0)
        img = np.full((1024, 1024, 3), 255, np.uint8)
        # fill the bottom-right box area with random grayish text-like noise
        loc = eng.locate(img)
        x, y, bw, bh = loc.bbox
        noise = rng.integers(150, 246, size=(bh, bw), dtype=np.uint8)
        img[y : y + bh, x : x + bw] = noise[:, :, None]
        out = eng.remove_watermark(img)
        assert np.array_equal(img, out)
