"""Samsung Galaxy AI visible watermark removal engine.

Samsung's on-device Generative AI photo edits (Generative Edit / Sketch to Image /
Portrait Studio on Galaxy phones) stamp a visible localized wordmark -- a sparkle
icon followed by a "generated with AI" string -- in the **bottom-left** corner: a
light, low-opacity semi-transparent white overlay. The string is locale-specific;
this engine is calibrated for the Italian "Contenuti generati dall'AI" variant
(issue #37, captures from @f-liva). Other locales need their own captured alpha
template, but the geometry and removal recipe are shared.

Like the Gemini sparkle and the Doubao / Jimeng marks it is a fixed overlay, so
removal starts from **reverse-alpha blending** against a captured alpha map
(``remove_watermark_reverse_alpha``): ``original = (wm - a*logo)/(1-a)``. The logo
is pure white (255,255,255); the alpha map was solved from the GRAY Samsung capture
(see ``data/samsung_capture/``), bundled as ``assets/samsung_alpha.png`` -- the same
careful build as Jimeng/Doubao (cubic-background fit, mean over channels, full halo
extent, unblurred). The Samsung mark is faint (peak alpha ~0.38), so the glyph reads
as a soft light-gray strip.

The mark is anchored bottom-LEFT (Doubao/Jimeng are bottom-right) and scales with
image WIDTH (~0.32 of width). The flat calibration captures arrive at the phone's
flat-edit size (~1086 wide) while real photos are ~3000 wide, so a single alpha map
cannot pixel-cancel the upscaled, per-image re-rasterized mark; removal therefore
NCC-aligns the alpha to the actual mark (always), reverse-alphas, then clears the
residual with a deliberately THIN inpaint over the glyph footprint -- the exact
recipe Jimeng uses. Verified on the flat captures and a real ~2958-wide download.

Detection (``detect``) matches the bundled glyph silhouette against the corner
candidate via normalized correlation, keying on the actual mark shape rather than
coverage heuristics. Samsung edits also carry C2PA + the Galaxy ``genAIType``
marker (see ``metadata``/``identify``), so the visible path is the stripped-metadata
fallback / the *removal* path, not a new ``identify`` signal.

``locate`` (geometry box) and ``extract_mask`` (the candidate glyph mask the
detector correlates) mirror the Doubao/Jimeng engines. Fast, offline, no GPU.
Arbitrary-region inpainting still lives in ``region_eraser`` / the ``erase`` command.
"""

# cv2/numpy boundary: third-party libs ship no usable element types; relax the
# unknown-type rules for this file only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# Geometry as a fraction of image WIDTH. The Samsung mark scales with width and is
# anchored bottom-LEFT. The box is intentionally generous (the glyph mask tightens
# it and the alignment search refines position); values cover the 1086 flat captures
# and the ~2958 real photos (both measured at width_frac ~0.31).
WM_WIDTH_FRAC = 0.40
WM_HEIGHT_FRAC = 0.060
MARGIN_LEFT_FRAC = 0.004
MARGIN_BOTTOM_FRAC = 0.002

# Glyph appearance: a low-saturation light gray rendered brighter than the
# surrounding content (white top-hat), same polarity logic as Doubao/Jimeng so a
# white-paper document is left untouched. LOGO_MIN_LUMA is lower than Jimeng's
# because the Samsung mark is fainter (peak alpha ~0.38), so on a mid/dark
# background the glyph luma is lower; the top-hat + NCC shape gate keep precision.
MAX_SATURATION = 55  # max channel spread to count a pixel as "grayish"
LOGO_MIN_LUMA = 110  # glyphs are at least this bright in absolute terms
TOPHAT_DELTA = 8  # glyph must exceed the local background by this many levels

# Detection matches the bundled alpha-template glyph silhouette
# (assets/samsung_alpha.png) against the candidate via zero-mean normalized
# correlation (cv2 TM_CCOEFF_NORMED). A small coverage floor skips the template
# match on a near-empty candidate box. The threshold is validated against the real
# capture set and the other visible marks (Doubao/Jimeng/Gemini must not cross-fire).
DETECT_MIN_COVERAGE = 0.01
DETECT_NCC_THRESHOLD = 0.40

# ── Reverse-alpha (recovery, Gemini/Doubao/Jimeng-style) ─────────────
# The Samsung mark is a fixed semi-transparent white overlay; given its alpha map
# the original pixels are recovered by inverting the blend. The logo is pure white
# (the white capture confirms it). The alpha map was solved from the GRAY capture by
# scripts/visible_alpha_solve.py (cubic-background fit, mean over channels, full halo,
# unblurred); the bundled asset (assets/samsung_alpha.png) is that template (a*255)
# at the captured width. The mark scales with image WIDTH, and the flat captures are
# ~2.7x smaller than real photos, so a pure width-scale is only approximate; removal
# also registers the template to the actual mark via a TM_CCOEFF_NORMED scale+position
# search (`_aligned_alpha_map`).
_ALPHA_NATIVE_WIDTH = 1086
_ALPHA_LOGO_BGR: tuple[float, float, float] = (255.0, 255.0, 255.0)
# Geometry below is emitted by scripts/visible_alpha_solve.py for the bundled
# asset -- keep them in sync when the asset is rebuilt.
_ALPHA_WIDTH_FRAC = 0.3195  # asset width / image width -- the alignment scale seed
_ALPHA_HEIGHT_FRAC = 0.0378
# Margins (of image WIDTH) of the captured mark -- the geometry record / where to
# seed; alignment refines the actual position, so these are not load-bearing.
_ALPHA_MARGIN_LEFT_FRAC = 0.0110
_ALPHA_MARGIN_BOTTOM_FRAC = 0.0064
# Alignment scale search (np.linspace args) around the width-scaled glyph size --
# wider than Jimeng's because the flat captures are far off the real-photo width, so
# the per-image scale can drift more from the width-scaled seed.
_ALPHA_ALIGN_SEARCH = (0.85, 1.18, 23)
# Residual inpaint footprint: a single capture upscaled to the real-photo width
# cannot pixel-cancel the re-rasterized mark, so the glyph footprint (alpha above
# this) is always inpainted after reverse-alpha (dilated by this kernel, INPAINT_NS).
# Kept deliberately THIN -- reverse-alpha already recovers the true background under
# the semi-transparent mark, so the inpaint only finishes the residual edges.
_RESIDUAL_ALPHA_FLOOR = 0.05
_RESIDUAL_DILATE = 5
_RESIDUAL_INPAINT_RADIUS = 2
_alpha_template_cache: NDArray[Any] | None = None


def _alpha_template() -> NDArray[Any] | None:
    """Lazily load the bundled Samsung alpha template (float [0,1]), or None."""
    global _alpha_template_cache
    if _alpha_template_cache is None:
        from pathlib import Path

        from remove_ai_watermarks import image_io

        path = Path(__file__).parent / "assets" / "samsung_alpha.png"
        img = image_io.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        _alpha_template_cache = img.astype(np.float32) / 255.0
    return _alpha_template_cache


@dataclass(frozen=True)
class SamsungLocation:
    """Located watermark box (bottom-left), in absolute pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    is_fallback: bool = True  # geometry anchor (no template match) -> always True for now

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass
class SamsungDetection:
    """Result of visible Samsung Galaxy AI watermark detection."""

    detected: bool = False
    confidence: float = 0.0
    region: tuple[int, int, int, int] = (0, 0, 0, 0)
    coverage: float = 0.0  # fraction of the box occupied by glyph pixels


_silhouette_cache: NDArray[Any] | None = None


def _glyph_silhouette() -> NDArray[Any] | None:
    """Binary glyph silhouette (255 = glyph) from the bundled alpha map, used as the
    detection template. None if the alpha asset is missing. The threshold is a
    fraction of the (faint) peak alpha so the thin strokes survive."""
    global _silhouette_cache
    if _silhouette_cache is None:
        at = _alpha_template()
        if at is None:
            return None
        _silhouette_cache = (at > 0.10).astype(np.uint8) * 255
    return _silhouette_cache


def _template_match_score(box_mask: NDArray[Any], image_width: int) -> float:
    """Zero-mean normalized correlation of the alpha-template glyph silhouette
    (scaled to the mark's expected size) against the candidate ``box_mask``."""
    sil = _glyph_silhouette()
    if sil is None or box_mask.size == 0:
        return 0.0
    gw = min(box_mask.shape[1] - 1, max(16, int(_ALPHA_WIDTH_FRAC * image_width)))
    gh = min(box_mask.shape[0] - 1, max(4, int(_ALPHA_HEIGHT_FRAC * image_width)))
    if gw < 16 or gh < 4:
        return 0.0
    template = cv2.resize(sil, (gw, gh), interpolation=cv2.INTER_NEAREST)
    return float(cv2.matchTemplate(box_mask, template, cv2.TM_CCOEFF_NORMED).max())


class SamsungEngine:
    """Remove the visible Samsung Galaxy AI watermark (locate -> mask -> reverse-alpha)."""

    def __init__(
        self,
        *,
        width_frac: float = WM_WIDTH_FRAC,
        height_frac: float = WM_HEIGHT_FRAC,
        margin_left_frac: float = MARGIN_LEFT_FRAC,
        margin_bottom_frac: float = MARGIN_BOTTOM_FRAC,
    ) -> None:
        self.width_frac = width_frac
        self.height_frac = height_frac
        self.margin_left_frac = margin_left_frac
        self.margin_bottom_frac = margin_bottom_frac
        # Track which corner the last detection found, so removal targets the same corner.
        self._last_detected_corner: str = "bottom-left"

    # ── Locate ────────────────────────────────────────────────────────

    def locate(self, image: NDArray[Any]) -> SamsungLocation:
        """Anchor the watermark box in the bottom-left corner by geometry."""
        h, w = image.shape[:2]
        return self._locate_corner(image, "bottom-left")

    def _locate_corner(self, image: NDArray[Any], corner: str) -> SamsungLocation:
        """Anchor the watermark box in the given corner by geometry.

        Args:
            corner: One of 'bottom-right', 'bottom-left', 'top-right', 'top-left'.
        """
        h, w = image.shape[:2]
        wm_w = max(40, int(w * self.width_frac))
        wm_h = max(16, int(w * self.height_frac))
        margin_x = max(2, int(w * self.margin_left_frac))
        margin_y = max(2, int(w * self.margin_bottom_frac))

        if corner == "bottom-right":
            x = max(0, w - margin_x - wm_w)
            y = max(0, h - margin_y - wm_h)
        elif corner == "bottom-left":
            x = min(margin_x, max(0, w - wm_w))
            y = max(0, h - margin_y - wm_h)
        elif corner == "top-right":
            x = max(0, w - margin_x - wm_w)
            y = min(margin_y, max(0, h - wm_h))
        elif corner == "top-left":
            x = min(margin_x, max(0, w - wm_w))
            y = min(margin_y, max(0, h - wm_h))
        else:
            raise ValueError(f"Unknown corner: {corner}")

        wm_w = min(wm_w, w - x)
        wm_h = min(wm_h, h - y)
        return SamsungLocation(x=x, y=y, w=wm_w, h=wm_h, is_fallback=True)

    def _all_corners(self) -> list[str]:
        """Return the four corners to scan, in priority order."""
        return ["bottom-right", "bottom-left", "top-right", "top-left"]

    def _infer_corner_from_region(self, image: NDArray[Any], region: tuple[int, int, int, int]) -> None:
        """Set _last_detected_corner based on which corner the region is closest to."""
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

    # ── Mask ──────────────────────────────────────────────────────────

    def extract_mask(self, image: NDArray[Any], loc: SamsungLocation) -> NDArray[Any]:
        """Build a full-image uint8 mask (255 = watermark glyph) for the box.

        Polarity-aware: the mark is a light, low-saturation gray rendered brighter
        than the local background (white top-hat), so a white-paper document is left
        untouched (nothing brighter than its surroundings is masked there).
        """
        h, w = image.shape[:2]
        x, y, bw, bh = loc.bbox
        # A degenerate ROI (a sliver from an extremely wide/short image) cannot hold
        # the mark and would feed cv2's GaussianBlur/morphology a ~1-px-tall array,
        # which can fault the native code on some platforms (mirrors the Doubao/Jimeng
        # guard). Skip the cv2 pipeline and return an empty mask there.
        if bh < 16 or bw < 16:
            return np.zeros((h, w), np.uint8)
        # Normalize the ROI to 3-channel BGR: a 2D grayscale or 4-channel BGRA input
        # would otherwise break the axis=2 channel reductions below.
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
        glyph = cv2.morphologyEx(glyph, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        mask = np.zeros((h, w), np.uint8)
        mask[y : y + bh, x : x + bw] = glyph
        return mask

    # ── Detect ────────────────────────────────────────────────────────

    def detect(self, image: NDArray[Any]) -> SamsungDetection:
        """Detect the visible Samsung mark by matching the alpha-template glyph
        silhouette against the corner candidate (TM_CCOEFF_NORMED).

        Scans all four corners and returns the best match."""
        all_dets = self.detect_all(image)
        if not all_dets:
            return SamsungDetection()
        return max(all_dets, key=lambda d: d.confidence)

    def detect_all(self, image: NDArray[Any]) -> list[SamsungDetection]:
        """Detect visible Samsung marks in ALL corners that exceed the NCC threshold.

        Returns a list of detections (one per corner that fired), ordered by confidence
        descending. Used for removing multiple watermarks from different corners.
        """
        dets: list[SamsungDetection] = []
        if image is None or image.size == 0:
            return dets
        for corner in self._all_corners():
            loc = self._locate_corner(image, corner)
            mask = self.extract_mask(image, loc)
            x, y, bw, bh = loc.bbox
            box = mask[y : y + bh, x : x + bw]
            coverage = float((box > 0).sum()) / float(max(1, bw * bh))
            if coverage >= DETECT_MIN_COVERAGE:
                score = _template_match_score(box, image.shape[1])
                logger.debug("Samsung detect %s: coverage=%.3f ncc=%.2f", corner, coverage, score)
                if score >= DETECT_NCC_THRESHOLD:
                    det = SamsungDetection(
                        detected=True,
                        confidence=score,
                        region=loc.bbox,
                        coverage=coverage,
                    )
                    dets.append(det)
        dets.sort(key=lambda d: d.confidence, reverse=True)
        return dets

    # ── Reverse-alpha (recovery + residual inpaint) ───────────────────

    def reverse_alpha_available(self, image: NDArray[Any]) -> bool:
        """True if the bundled alpha map is loadable (NCC alignment places it at any
        resolution; the caller still gates on ``detect``)."""
        return image is not None and image.size > 0 and _alpha_template() is not None

    def _fixed_alpha_map(self, image: NDArray[Any]) -> tuple[NDArray[Any], tuple[int, int, int, int]] | None:
        """Place the template by fixed width-relative geometry.
        Placement respects the corner detected by detect()."""
        at = _alpha_template()
        if at is None:
            return None
        h, w = image.shape[:2]
        gw = min(w, max(1, int(_ALPHA_WIDTH_FRAC * w)))
        gh = min(h, max(1, int(_ALPHA_HEIGHT_FRAC * w)))
        corner = self._last_detected_corner
        margin_x = int(_ALPHA_MARGIN_LEFT_FRAC * w)
        margin_y = int(_ALPHA_MARGIN_BOTTOM_FRAC * w)
        if corner == "bottom-right":
            ax = max(0, w - margin_x - gw)
            ay = max(0, h - margin_y - gh)
        elif corner == "bottom-left":
            ax = min(margin_x, max(0, w - gw))
            ay = max(0, h - margin_y - gh)
        elif corner == "top-right":
            ax = max(0, w - margin_x - gw)
            ay = min(margin_y, max(0, h - gh))
        elif corner == "top-left":
            ax = min(margin_x, max(0, w - gw))
            ay = min(margin_y, max(0, h - gh))
        else:
            ax = min(margin_x, max(0, w - gw))
            ay = max(0, h - margin_y - gh)
        amap = np.zeros((h, w), np.float32)
        amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh), interpolation=cv2.INTER_LINEAR)
        return amap, (ax, ay, gw, gh)

    def _aligned_alpha_map(self, image: NDArray[Any]) -> tuple[NDArray[Any], tuple[int, int, int, int]] | None:
        """Register the captured template to the actual mark via a TM_CCOEFF_NORMED
        scale + position search -- so the single capture works off the captured
        width. Returns ``(alpha_map, glyph_bbox)`` or None."""
        at = _alpha_template()
        sil = _glyph_silhouette()
        if at is None or sil is None:
            return None
        h, w = image.shape[:2]
        loc = self._locate_corner(image, self._last_detected_corner)
        bx, by, bw, bh = loc.bbox
        box_mask = self.extract_mask(image, loc)[by : by + bh, bx : bx + bw]
        expected = _ALPHA_WIDTH_FRAC * w
        best: tuple[float, int, int, int, int] | None = None
        for scale in np.linspace(*_ALPHA_ALIGN_SEARCH):
            gw, gh = int(expected * scale), int(_ALPHA_HEIGHT_FRAC * w * scale)
            if gw < 16 or gh < 4 or gw >= bw or gh >= bh:
                continue
            t = cv2.resize(sil, (gw, gh), interpolation=cv2.INTER_NEAREST)
            _, score, _, top_left = cv2.minMaxLoc(cv2.matchTemplate(box_mask, t, cv2.TM_CCOEFF_NORMED))
            if best is None or score > best[0]:
                best = (score, gw, gh, top_left[0], top_left[1])
        if best is None:
            return None
        _, gw, gh, ox, oy = best
        ax, ay = bx + ox, by + oy
        amap = np.zeros((h, w), np.float32)
        amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh), interpolation=cv2.INTER_LINEAR)
        return amap, (ax, ay, gw, gh)

    def _apply_reverse_alpha(self, image: NDArray[Any], amap: NDArray[Any]) -> NDArray[Any]:
        """Invert the alpha blend with ``amap``: ``original = (wm - a*logo)/(1-a)``."""
        a3 = np.clip(amap, 0.0, 1.0)[:, :, None]
        logo = np.array(_ALPHA_LOGO_BGR, np.float32)
        return np.clip((image.astype(np.float32) - a3 * logo) / np.clip(1.0 - a3, 0.25, 1.0), 0, 255).astype(np.uint8)

    def remove_watermark_reverse_alpha(self, image: NDArray[Any], *, residual_inpaint: bool = True) -> NDArray[Any]:
        """Recover the original pixels by inverting the alpha blend, then clear the
        residual outline with a thin inpaint over the glyph footprint.

        Placement: fixed geometry AND the NCC-aligned placement are always tried and
        the one leaving the least residual mark (lowest re-``detect`` confidence) is
        kept -- the flat capture is far off the real-photo width and the mark
        re-rasterizes per image, so fixed geometry alone is not reliable. A single
        capture cannot pixel-cancel the upscaled mark, so a deliberately THIN residual
        inpaint (``_RESIDUAL_*``) follows. Call only when
        :meth:`reverse_alpha_available` and the mark is detected.
        """
        # Normalize to 3-channel BGR so a 2D grayscale or 4-channel BGRA input does
        # not break the reverse-alpha math (which assumes a 3-channel logo).
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        # An image too small to hold the mark would make the geometry boxes degenerate
        # and feed cv2.resize a ~1-px-tall target; skip cv2 entirely (mirrors Jimeng).
        h, w = image.shape[:2]
        if h < 32 or w < 64:
            return image.copy()
        maps = [c for c in (self._fixed_alpha_map(image), self._aligned_alpha_map(image)) if c is not None]
        if not maps:
            return image.copy()
        best_out: NDArray[Any] | None = None
        best_amap: NDArray[Any] | None = None
        best_residual = float("inf")
        for amap, _region in maps:
            out = self._apply_reverse_alpha(image, amap)
            residual = self.detect(out).confidence
            if residual < best_residual:
                best_residual, best_out, best_amap = residual, out, amap
        if best_out is None or best_amap is None:  # pragma: no cover - maps is non-empty
            return image.copy()
        if residual_inpaint:
            kernel = np.ones((_RESIDUAL_DILATE, _RESIDUAL_DILATE), np.uint8)
            rm = cv2.dilate((best_amap > _RESIDUAL_ALPHA_FLOOR).astype(np.uint8) * 255, kernel)
            best_out = cv2.inpaint(best_out, rm, _RESIDUAL_INPAINT_RADIUS, cv2.INPAINT_NS)
        return best_out


def load_image_bgr(path: str | Path) -> NDArray[Any]:
    """Read an image as BGR ndarray (helper for scripts/tests)."""
    from remove_ai_watermarks import image_io

    img = image_io.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    return img
