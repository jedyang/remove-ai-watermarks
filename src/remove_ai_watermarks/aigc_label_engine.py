"""AIGC label watermark engine for "AI生成" style label marks.

Many Chinese AI image platforms (e.g., Tongyi Wanxiang, Zhipu, etc.) stamp a
semi-transparent "AI生成" (AI Generated) text label in the **top-left** corner,
often inside a rounded-rectangle border. Unlike the deterministic reverse-alpha
engines (Doubao, Jimeng, Gemini), this mark has no fixed alpha template --
it varies in font, size, border style and opacity across platforms. Detection
relies on adaptive bright-region extraction; removal uses inpainting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ── Geometry ────────────────────────────────────────────────────────────────
# The "AI生成" label sits in the top-left corner.  Box size scales with image
# width (like the other engines).  These are generous bounds -- the actual mark
# is smaller, but the box must contain it for the mask extraction to work.

LABEL_WIDTH_FRAC = 0.18   # box width as fraction of image width
LABEL_HEIGHT_FRAC = 0.07  # box height
MARGIN_LEFT_FRAC = 0.005  # small margin from left edge
MARGIN_TOP_FRAC = 0.005   # small margin from top edge

# ── Detection thresholds ───────────────────────────────────────────────────
# The label is typically low-contrast on dark backgrounds (luma ~60-120 vs
# background ~10-30), so we use much lower luma thresholds than Doubao/Jimeng.

DETECT_MIN_COVERAGE = 0.03    # minimum fraction of box that must be masked
MAX_SATURATION = 70           # grayish threshold (slightly relaxed)
TOPHAT_DELTA = 3              # brighter-than-local-bg threshold (very sensitive)
LOGO_MIN_LUMA = 50            # absolute luma floor (low for dark-background labels)

# Dark-background fallback: when the whole corner is very dark (< 40 mean luma),
# relax further.
DARK_MAX_ROI_MEAN_LUMA = 40
DARK_LOGO_MIN_LUMA = 35
DARK_LOGO_LUMA_DELTA = 12

# NCC-like confidence: we don't have a fixed glyph template, so confidence is
# derived from coverage density and contrast.
MIN_CONFIDENCE = 0.15         # below this -> not considered a real detection


@dataclass(frozen=True)
class AIGCDetection:
    """Result of AIGC label detection."""

    detected: bool = False
    confidence: float = 0.0
    region: tuple[int, int, int, int] = (0, 0, 0, 0)
    coverage: float = 0.0


@dataclass
class AIGCLocation:
    """Located label box in absolute pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    is_fallback: bool = True

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


class AIGCLabelEngine:
    """Detect and remove "AI生成"-style AIGC label watermarks.

    Scans the **top-left** corner (and optionally all corners) for semi-transparent
    light-gray text labels.  Removal is via inpainting (no alpha template).
    """

    def __init__(
        self,
        *,
        width_frac: float = LABEL_WIDTH_FRAC,
        height_frac: float = LABEL_HEIGHT_FRAC,
        margin_left_frac: float = MARGIN_LEFT_FRAC,
        margin_top_frac: float = MARGIN_TOP_FRAC,
    ) -> None:
        self.width_frac = width_frac
        self.height_frac = height_frac
        self.margin_left_frac = margin_left_frac
        self.margin_top_frac = margin_top_frac
        self._last_detected_corner: str = "top-left"

    # ── Locate ─────────────────────────────────────────────────────────────

    def _locate_corner(self, image: NDArray[Any], corner: str) -> AIGCLocation:
        """Anchor the label box in the given corner by geometry."""
        h, w = image.shape[:2]
        wm_w = max(40, int(w * self.width_frac))
        wm_h = max(16, int(w * self.height_frac))
        margin_x = max(4, int(w * self.margin_left_frac))
        margin_y = max(4, int(w * self.margin_top_frac))

        if corner == "top-left":
            x = min(margin_x, max(0, w - wm_w))
            y = min(margin_y, max(0, h - wm_h))
        elif corner == "top-right":
            x = max(0, w - margin_x - wm_w)
            y = min(margin_y, max(0, h - wm_h))
        elif corner == "bottom-left":
            x = min(margin_x, max(0, w - wm_w))
            y = max(0, h - margin_y - wm_h)
        elif corner == "bottom-right":
            x = max(0, w - margin_x - wm_w)
            y = max(0, h - margin_y - wm_h)
        else:
            raise ValueError(f"Unknown corner: {corner}")

        wm_w = min(wm_w, w - x)
        wm_h = min(wm_h, h - y)
        return AIGCLocation(x=x, y=y, w=wm_w, h=wm_h)

    def _all_corners(self) -> list[str]:
        """Return corners to scan (top-left first -- most common location)."""
        return ["top-left", "top-right", "bottom-left", "bottom-right"]

    def _infer_corner_from_region(
        self, image: NDArray[Any], region: tuple[int, int, int, int]
    ) -> None:
        """Set _last_detected_corner from region center."""
        h, w = image.shape[:2]
        rx, ry, rw, rh = region
        cx = rx + rw / 2.0
        cy = ry + rh / 2.0
        in_right = cx > w / 2.0
        in_bottom = cy > h / 2.0
        if in_bottom:
            self._last_detected_corner = "bottom-right" if in_right else "bottom-left"
        else:
            self._last_detected_corner = "top-right" if in_right else "top-left"

    # ── Mask extraction ────────────────────────────────────────────────────

    def extract_mask(self, image: NDArray[Any], loc: AIGCLocation) -> NDArray[Any]:
        """Build a uint8 mask (255 = label pixels) for the box.

        Adapted from doubao_engine.extract_mask with lower thresholds suitable
        for faint labels on very dark backgrounds.
        """
        h, w = image.shape[:2]
        x, y, bw, bh = loc.bbox
        if bh < 16 or bw < 16:
            return np.zeros((h, w), np.uint8)

        roi = image[y : y + bh, x : x + bw]
        if roi.ndim == 2:
            roi = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        elif roi.shape[2] == 4:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR)
        roi = roi.astype(np.float32)

        luma = roi.mean(axis=2)
        sat = roi.max(axis=2) - roi.min(axis=2)
        grayish = sat < MAX_SATURATION

        sigma = max(4.0, bh * 0.4)
        local_bg = cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma, sigmaY=sigma)
        tophat = luma - local_bg

        cand = grayish & (tophat > TOPHAT_DELTA) & (luma > LOGO_MIN_LUMA)
        glyph = cand.astype(np.uint8) * 255
        glyph = cv2.morphologyEx(glyph, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        glyph = cv2.morphologyEx(glyph, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

        primary_coverage = float((glyph > 0).sum()) / float(max(1, bw * bh))

        # Dark-background fallback
        if primary_coverage < DETECT_MIN_COVERAGE and luma.mean() <= DARK_MAX_ROI_MEAN_LUMA:
            median_luma = float(np.median(luma))
            dark_cand = (
                grayish & (luma > DARK_LOGO_MIN_LUMA) & (luma > median_luma + DARK_LOGO_LUMA_DELTA)
            )
            dark_glyph = dark_cand.astype(np.uint8) * 255
            dark_glyph = cv2.morphologyEx(dark_glyph, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            dark_glyph = cv2.morphologyEx(dark_glyph, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            dark_cov = float((dark_glyph > 0).sum()) / float(max(1, bw * bh))
            logger.debug(
                "AIGC label dark fallback: roi_mean=%.1f primary_cov=%.3f dark_cov=%.3f",
                luma.mean(), primary_coverage, dark_cov,
            )
            if dark_cov > primary_coverage:
                glyph = dark_glyph

        mask = np.zeros((h, w), np.uint8)
        mask[y : y + bh, x : x + bw] = glyph
        return mask

    # ── Detect ─────────────────────────────────────────────────────────────

    def detect(self, image: NDArray[Any]) -> AIGCDetection:
        """Detect the AIGC label mark. Returns the best corner match."""
        all_dets = self.detect_all(image)
        if not all_dets:
            return AIGCDetection()
        return max(all_dets, key=lambda d: d.confidence)

    def detect_all(self, image: NDArray[Any]) -> list[AIGCDetection]:
        """Detect AIGC label marks in ALL corners that exceed coverage/confidence thresholds."""
        dets: list[AIGCDetection] = []
        if image is None or image.size == 0:
            return dets

        for corner in self._all_corners():
            loc = self._locate_corner(image, corner)
            mask = self.extract_mask(image, loc)
            x, y, bw, bh = loc.bbox
            box = mask[y : y + bh, x : x + bw]
            coverage = float((box > 0).sum()) / float(max(1, bw * bh))

            if coverage >= DETECT_MIN_COVERAGE:
                # Confidence = coverage-based score (we have no NCC template)
                confidence = min(coverage * 5.0, 1.0)  # scale coverage to ~0-1 range
                if confidence >= MIN_CONFIDENCE:
                    det = AIGCDetection(
                        detected=True,
                        confidence=confidence,
                        region=loc.bbox,
                        coverage=coverage,
                    )
                    dets.append(det)
                    logger.debug(
                        "AIGC label detect %s: coverage=%.3f conf=%.2f",
                        corner, coverage, confidence,
                    )

        dets.sort(key=lambda d: d.confidence, reverse=True)
        return dets

    # ── Remove (inpaint-based) ─────────────────────────────────────────────

    def remove_watermark(
        self, image: NDArray[Any], *, inpaint_radius: int = 5
    ) -> NDArray[Any]:
        """Remove the AIGC label by extracting its mask and inpainting.

        Uses the detection mask directly (no alpha template).  Placement follows
        ``_last_detected_corner`` set by ``_infer_corner_from_region`` or the
        default ``detect()`` best corner.
        """
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        h, w = image.shape[:2]
        if h < 32 or w < 64:
            return image.copy()

        loc = self._locate_corner(image, self._last_detected_corner)
        mask = self.extract_mask(image, loc)

        # Dilate mask slightly to cover edges
        if mask.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(mask, kernel, iterations=2)
            result = cv2.inpaint(image, dilated, inpaint_radius, cv2.INPAINT_TELEA)
            return result

        return image.copy()
