"""Shared base for the reverse-alpha visible text-mark engines.

The Doubao "豆包AI生成", Jimeng "★ 即梦AI", and Samsung "✦ Contenuti generati
dall'AI" marks are the SAME algorithm: anchor a bottom-corner box by width-relative
geometry, extract the light low-saturation glyph candidate, detect by matching the
bundled alpha-glyph silhouette via ``TM_CCOEFF_NORMED``, and remove by inverting the
alpha blend ``original = (wm - a*logo)/(1-a)`` (always trying fixed AND NCC-aligned
placement, keeping the lower-residual one) plus a thin footprint inpaint.

They differ ONLY in a bounded set of tuned values captured by :class:`TextMarkConfig`:
the constants, the bundled asset, the corner (Doubao/Jimeng bottom-right, Samsung
bottom-left), and a few structural knobs (the morphology-open kernel size and the
minimum glyph width used by the alignment / template-match). Each engine module is a
thin :class:`TextMarkEngine` subclass plus the test-facing module constants/helpers.

Gemini stays a SEPARATE engine (``gemini_engine``): its multi-size fixed-slot sparkle
model is genuinely different, not a tuned variant of this one.
"""

# cv2/numpy boundary: third-party libs ship no usable element types; relax the
# unknown-type rules for this file only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import cv2
import numpy as np

from remove_ai_watermarks import image_io

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TextMarkConfig:
    """All per-mark tuning for a reverse-alpha text-mark engine."""

    name: str  # short label for log lines (e.g. "Doubao")
    asset_name: str  # bundled alpha PNG under assets/ (e.g. "doubao_alpha.png")
    corner: Literal["br", "bl"]  # bottom-right (Doubao/Jimeng) or bottom-left (Samsung)
    margin_floor: int  # min margin in px for locate (4 for br marks, 2 for Samsung)
    # locate geometry (fraction of image WIDTH)
    width_frac: float
    height_frac: float
    margin_x_frac: float  # right margin (br) or left margin (bl)
    margin_bottom_frac: float
    # glyph appearance
    max_saturation: float
    logo_min_luma: float
    tophat_delta: float
    morph_open_size: int  # MORPH_OPEN kernel side (5 for br marks, 3 for Samsung)
    # detection
    detect_min_coverage: float
    detect_ncc_threshold: float
    # alpha-map geometry (fraction of WIDTH) emitted by scripts/visible_alpha_solve.py
    alpha_width_frac: float
    alpha_height_frac: float
    alpha_margin_x_frac: float
    alpha_margin_bottom_frac: float
    alpha_align_search: tuple[float, float, int]  # np.linspace(start, stop, num) scale search
    min_gw: int  # minimum glyph width for the template match / align search (8 br, 16 Samsung)
    alpha_logo_bgr: tuple[float, float, float] = (255.0, 255.0, 255.0)
    # residual inpaint over the glyph footprint (thin)
    residual_alpha_floor: float = 0.05
    residual_dilate: int = 5
    residual_inpaint_radius: int = 2


@dataclass
class TextMarkLocation:
    """Located watermark box, in absolute pixel coordinates."""

    x: int
    y: int
    w: int
    h: int
    is_fallback: bool = True  # geometry anchor (no template match) -> always True for now

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


@dataclass
class TextMarkDetection:
    """Result of visible text-mark detection."""

    detected: bool = False
    confidence: float = 0.0
    region: tuple[int, int, int, int] = (0, 0, 0, 0)
    coverage: float = 0.0  # fraction of the box occupied by glyph pixels


# Alpha / silhouette templates, cached per asset name (the originals cached per
# module global; this keys by asset so the three engines share the loader without
# re-reading). Only SUCCESSFUL loads are cached, so a missing asset is retried.
_alpha_cache: dict[str, NDArray[Any]] = {}
_silhouette_cache: dict[str, NDArray[Any]] = {}


def load_alpha_template(asset_name: str) -> NDArray[Any] | None:
    """Lazily load the bundled alpha template (float [0,1]) for ``asset_name``, or None."""
    cached = _alpha_cache.get(asset_name)
    if cached is not None:
        return cached
    path = Path(__file__).parent / "assets" / asset_name
    img = image_io.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    _alpha_cache[asset_name] = img.astype(np.float32) / 255.0
    return _alpha_cache[asset_name]


def glyph_silhouette(asset_name: str) -> NDArray[Any] | None:
    """Binary glyph silhouette (255 = glyph) from the bundled alpha map, or None."""
    cached = _silhouette_cache.get(asset_name)
    if cached is not None:
        return cached
    at = load_alpha_template(asset_name)
    if at is None:
        return None
    _silhouette_cache[asset_name] = (at > 0.15).astype(np.uint8) * 255
    return _silhouette_cache[asset_name]


def template_match_score(box_mask: NDArray[Any], image_width: int, config: TextMarkConfig) -> float:
    """Zero-mean normalized correlation of the alpha-template glyph silhouette
    (scaled to the mark's expected size) against the candidate ``box_mask``.

    ``TM_CCOEFF_NORMED`` keys on glyph SHAPE, not coverage, so a dense textured
    corner does not score highly -- only the actual glyph shape does.
    """
    sil = glyph_silhouette(config.asset_name)
    if sil is None or box_mask.size == 0:
        return 0.0
    gw = min(box_mask.shape[1] - 1, max(config.min_gw, int(config.alpha_width_frac * image_width)))
    gh = min(box_mask.shape[0] - 1, max(4, int(config.alpha_height_frac * image_width)))
    if gw < config.min_gw or gh < 4:
        return 0.0
    template = cv2.resize(sil, (gw, gh), interpolation=cv2.INTER_NEAREST)
    return float(cv2.matchTemplate(box_mask, template, cv2.TM_CCOEFF_NORMED).max())


class TextMarkEngine:
    """Reverse-alpha visible text-mark remover (locate -> mask -> detect -> reverse-alpha)."""

    def __init__(self, config: TextMarkConfig) -> None:
        self.config = config

    # ── Templates (delegate to the asset-keyed module cache) ────────────

    def _alpha_template(self) -> NDArray[Any] | None:
        return load_alpha_template(self.config.asset_name)

    def _glyph_silhouette(self) -> NDArray[Any] | None:
        return glyph_silhouette(self.config.asset_name)

    def _template_match_score(self, box_mask: NDArray[Any], image_width: int) -> float:
        return template_match_score(box_mask, image_width, self.config)

    # ── Locate ──────────────────────────────────────────────────────────

    def locate(self, image: NDArray[Any]) -> TextMarkLocation:
        """Anchor the watermark box in the configured bottom corner by geometry."""
        c = self.config
        h, w = image.shape[:2]
        wm_w = max(40, int(w * c.width_frac))
        wm_h = max(16, int(w * c.height_frac))
        margin_x = max(c.margin_floor, int(w * c.margin_x_frac))
        margin_b = max(c.margin_floor, int(w * c.margin_bottom_frac))
        x = max(0, w - margin_x - wm_w) if c.corner == "br" else min(margin_x, max(0, w - wm_w))
        y = max(0, h - margin_b - wm_h)
        wm_w = min(wm_w, w - x)
        wm_h = min(wm_h, h - y)
        return TextMarkLocation(x=x, y=y, w=wm_w, h=wm_h, is_fallback=True)

    # ── Mask ────────────────────────────────────────────────────────────

    def extract_mask(self, image: NDArray[Any], loc: TextMarkLocation) -> NDArray[Any]:
        """Build a full-image uint8 mask (255 = watermark glyph) for the box.

        Polarity-aware: the mark is a light, low-saturation gray rendered brighter
        than the local background (white top-hat), so a white-paper document is left
        untouched (nothing brighter than its surroundings is masked there).
        """
        c = self.config
        h, w = image.shape[:2]
        x, y, bw, bh = loc.bbox
        # A degenerate ROI (a sliver from an extremely wide/short image) cannot hold
        # the mark and would feed cv2's GaussianBlur/morphology a ~1-px-tall array,
        # which can fault native code on some platforms. Skip the cv2 pipeline.
        if bh < 16 or bw < 16:
            return np.zeros((h, w), np.uint8)
        # Normalize the ROI to 3-channel BGR (grayscale / BGRA would break axis=2).
        roi = image_io.to_bgr(image[y : y + bh, x : x + bw]).astype(np.float32)

        luma = roi.mean(axis=2)
        sat = roi.max(axis=2) - roi.min(axis=2)
        grayish = sat < c.max_saturation

        # Local background model: a strong Gaussian blur (sigma ~ box height); the
        # white top-hat (luma - local_bg) lights up bright thin strokes regardless
        # of the absolute background level.
        sigma = max(4.0, bh * 0.4)
        local_bg = cv2.GaussianBlur(luma, (0, 0), sigmaX=sigma, sigmaY=sigma)
        tophat = luma - local_bg

        cand = grayish & (tophat > c.tophat_delta) & (luma > c.logo_min_luma)
        glyph = cand.astype(np.uint8) * 255
        glyph = cv2.morphologyEx(glyph, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        k = c.morph_open_size
        glyph = cv2.morphologyEx(glyph, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))

        mask = np.zeros((h, w), np.uint8)
        mask[y : y + bh, x : x + bw] = glyph
        return mask

    # ── Detect ──────────────────────────────────────────────────────────

    def detect(self, image: NDArray[Any]) -> TextMarkDetection:
        """Detect the mark by matching the alpha-template glyph silhouette against
        the corner candidate (``TM_CCOEFF_NORMED``); keys on glyph SHAPE, not coverage."""
        c = self.config
        det = TextMarkDetection()
        if image is None or image.size == 0:
            return det
        loc = self.locate(image)
        mask = self.extract_mask(image, loc)
        x, y, bw, bh = loc.bbox
        box = mask[y : y + bh, x : x + bw]
        coverage = float((box > 0).sum()) / float(max(1, bw * bh))
        det.region = loc.bbox
        det.coverage = coverage
        if coverage >= c.detect_min_coverage:
            score = self._template_match_score(box, image.shape[1])
            det.confidence = score
            det.detected = score >= c.detect_ncc_threshold
            logger.debug("%s detect: coverage=%.3f ncc=%.2f detected=%s", c.name, coverage, score, det.detected)
        return det

    # ── Reverse-alpha (recovery + thin residual inpaint) ────────────────

    def reverse_alpha_available(self, image: NDArray[Any]) -> bool:
        """True if the bundled alpha map is loadable (NCC alignment places it at any
        resolution; the caller still gates on ``detect`` so a clean corner is untouched)."""
        return image is not None and image.size > 0 and self._alpha_template() is not None

    def _fixed_alpha_map(self, image: NDArray[Any]) -> tuple[NDArray[Any], tuple[int, int, int, int]] | None:
        """Place the template by fixed width-relative geometry (pixel-exact at the
        captured width)."""
        c = self.config
        at = self._alpha_template()
        if at is None:
            return None
        h, w = image.shape[:2]
        # Clamp both dims so a wide/short image cannot overflow the slice assignment.
        gw = min(w, max(1, int(c.alpha_width_frac * w)))
        gh = min(h, max(1, int(c.alpha_height_frac * w)))
        if c.corner == "br":
            ax = max(0, w - int(c.alpha_margin_x_frac * w) - gw)
        else:  # bottom-left
            ax = min(max(0, int(c.alpha_margin_x_frac * w)), max(0, w - gw))
        ay = max(0, h - int(c.alpha_margin_bottom_frac * w) - gh)
        amap = np.zeros((h, w), np.float32)
        amap[ay : ay + gh, ax : ax + gw] = cv2.resize(at, (gw, gh), interpolation=cv2.INTER_LINEAR)
        return amap, (ax, ay, gw, gh)

    def _aligned_alpha_map(self, image: NDArray[Any]) -> tuple[NDArray[Any], tuple[int, int, int, int]] | None:
        """Register the captured template to the actual mark via a TM_CCOEFF_NORMED
        scale + position search. Returns ``(alpha_map, glyph_bbox)`` or None."""
        c = self.config
        at = self._alpha_template()
        sil = self._glyph_silhouette()
        if at is None or sil is None:
            return None
        h, w = image.shape[:2]
        loc = self.locate(image)
        bx, by, bw, bh = loc.bbox
        box_mask = self.extract_mask(image, loc)[by : by + bh, bx : bx + bw]
        expected = c.alpha_width_frac * w
        best: tuple[float, int, int, int, int] | None = None
        for scale in np.linspace(*c.alpha_align_search):
            gw, gh = int(expected * scale), int(c.alpha_height_frac * w * scale)
            if gw < c.min_gw or gh < 4 or gw >= bw or gh >= bh:
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
        logo = np.array(self.config.alpha_logo_bgr, np.float32)
        return np.clip((image.astype(np.float32) - a3 * logo) / np.clip(1.0 - a3, 0.25, 1.0), 0, 255).astype(np.uint8)

    def remove_watermark_reverse_alpha(self, image: NDArray[Any], *, residual_inpaint: bool = True) -> NDArray[Any]:
        """Recover the original pixels by inverting the alpha blend, then clear the
        residual outline with a thin inpaint over the glyph footprint.

        Placement: fixed geometry AND the NCC-aligned placement are always tried and
        the one leaving the least residual mark (lowest re-``detect`` confidence) is
        kept -- the mark re-rasterizes a few px per image, so fixed geometry alone is
        not reliable. A single capture cannot pixel-cancel the mark on every image, so
        a deliberately THIN residual inpaint (``residual_*``) follows: reverse-alpha
        has already recovered the true background under the mark, so the inpaint only
        finishes the residual edges instead of smearing the whole footprint. Call only
        when :meth:`reverse_alpha_available` and the mark is detected.
        """
        c = self.config
        # Normalize to 3-channel BGR (the reverse-alpha math assumes a 3-channel logo).
        image = image_io.to_bgr(image)
        # An image too small to hold the mark would make the geometry boxes degenerate
        # and feed cv2.resize a ~1-px-tall target; skip cv2 entirely.
        h, w = image.shape[:2]
        if h < 32 or w < 64:
            return image.copy()
        maps = [m for m in (self._fixed_alpha_map(image), self._aligned_alpha_map(image)) if m is not None]
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
            kernel = np.ones((c.residual_dilate, c.residual_dilate), np.uint8)
            rm = cv2.dilate((best_amap > c.residual_alpha_floor).astype(np.uint8) * 255, kernel)
            best_out = cv2.inpaint(best_out, rm, c.residual_inpaint_radius, cv2.INPAINT_NS)
        return best_out
