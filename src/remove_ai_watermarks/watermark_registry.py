"""Registry of known visible watermarks.

A single catalog that ties each known visible mark to (a) where it usually sits,
(b) how to recognize it there, and (c) how to remove it. One pass over the
registry detects every known mark in its usual place and removes the ones
present.

**Reverse-alpha based.** A known mark is a fixed semi-transparent overlay, so it
is removed by inverting the alpha blend against a captured alpha map
(``original = (wm - a*logo)/(1-a)``) -- recovering the true pixels rather than
inpainting a guess. Gemini and Doubao recover exactly with no inpaint at native on
bright/flat backgrounds (Gemini falls back to inpainting the sparkle footprint when
reverse-alpha would over-subtract on a dark background -- issue #30, see gemini_engine);
Jimeng adds a thin residual inpaint over the glyph footprint to clear the outline
its per-image render variation leaves behind (still seeded by the reverse-alpha
recovery, not a blind inpaint). Detection is consistent with that: each mark is
recognized by matching its known shape/template (the thing we invert), not by
heuristics. A mark is therefore listed here only once a real alpha map has been
captured for it; everything else (arbitrary logos/objects) is the user-directed
``erase --region`` tool, not this catalog.

Entries:
  - ``gemini`` -- Google Gemini / Nano Banana sparkle, bottom-right.
  - ``doubao`` -- ByteDance Doubao "豆包AI生成" text strip, bottom-right.
  - ``jimeng`` -- ByteDance Jimeng / Dreamina "★ 即梦AI" wordmark, bottom-right.
  - ``samsung`` -- Samsung Galaxy AI "Contenuti generati dall'AI" strip, bottom-left.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

# cv2 method for the Gemini reverse-alpha edge-residual cleanup (not a standalone
# remover): "ns" / "telea".
InpaintMethod = Literal["telea", "ns"]
Region = tuple[int, int, int, int]


@dataclass(frozen=True)
class MarkDetection:
    """Uniform detection result for a known mark (across heterogeneous engines)."""

    key: str
    label: str
    location: str
    detected: bool
    confidence: float
    region: Region


@dataclass(frozen=True)
class KnownMark:
    """A known visible watermark: where it lives, how to find and remove it."""

    key: str
    label: str
    location: str  # usual place, human-readable ("bottom-right")
    in_auto: bool  # participate in `--mark auto` scanning
    recovery: str  # removal strategy (all reverse-alpha today)
    _detect: Callable[[NDArray[Any]], MarkDetection]
    _remove: Callable[..., tuple[NDArray[Any], Region | None]]

    def detect(self, image: NDArray[Any]) -> MarkDetection:
        return self._detect(image)

    def remove(
        self,
        image: NDArray[Any],
        *,
        inpaint_method: InpaintMethod = "ns",
        inpaint: bool = True,
        inpaint_strength: float = 0.85,
        force: bool = False,
        target_region: Region | None = None,
    ) -> tuple[NDArray[Any], Region | None]:
        """Remove this mark by reverse-alpha; returns ``(result, region)`` where
        ``region`` is the removed mark's bbox (for residual-inpaint positioning),
        or None if nothing was removed. NB: the CLI does NOT use ``region`` to
        clear alpha on save -- that zeroing caused the issue-#30 white box.

        ``inpaint`` / ``inpaint_strength`` / ``inpaint_method`` tune the Gemini
        reverse-alpha edge-residual cleanup only. ``force`` removes at the mark's
        usual location even without a positive detection (the ``--no-detect`` path).
        ``target_region`` (internal) forces removal at a specific bbox instead of
        the engine's global-best corner -- used by the multi-mark removal path.
        """
        return self._remove(image, inpaint_method, inpaint, inpaint_strength, force, target_region)


# Gemini-sparkle confidence above which the registry treats it as a confident
# detection for arbitration. Matches identify's corpus-validated sparkle
# threshold (0.5): the gemini engine's own detect flag uses a looser internal
# threshold and weakly fires (~0.36) on unrelated bottom-right text (e.g. the
# Doubao mark), which would otherwise let it hijack `--mark auto`. 0.5 gives 0
# false positives on the corpus.
_GEMINI_AUTO_MIN_CONF = 0.5

# ── Engine adapters (lazy singletons; engines are cv2-only, no model load) ──

_engines: dict[str, Any] = {}


def _engine(key: str) -> Any:
    if key not in _engines:
        if key == "gemini":
            from remove_ai_watermarks.gemini_engine import GeminiEngine

            _engines[key] = GeminiEngine()
        elif key == "doubao":
            from remove_ai_watermarks.doubao_engine import DoubaoEngine

            _engines[key] = DoubaoEngine()
        elif key == "jimeng":
            from remove_ai_watermarks.jimeng_engine import JimengEngine

            _engines[key] = JimengEngine()
        elif key == "samsung":
            from remove_ai_watermarks.samsung_engine import SamsungEngine

            _engines[key] = SamsungEngine()
        elif key == "aigc_label":
            from remove_ai_watermarks.aigc_label_engine import AIGCLabelEngine

            _engines[key] = AIGCLabelEngine()
        else:  # pragma: no cover - guarded by the registry keys
            raise KeyError(key)
    return _engines[key]


def _gemini_detect(image: NDArray[Any]) -> MarkDetection:
    d = _engine("gemini").detect_watermark(image)
    detected = bool(d.detected) and d.confidence >= _GEMINI_AUTO_MIN_CONF
    return MarkDetection("gemini", "Google Gemini sparkle", "bottom-right", detected, d.confidence, d.region)


def _gemini_remove(
    image: NDArray[Any], inpaint_method: InpaintMethod, inpaint: bool, strength: float, force: bool,
    target_region: Region | None = None,
) -> tuple[NDArray[Any], Region | None]:
    """Remove a Gemini sparkle.  When *target_region* is set (from detect_all),
    the engine's internal corner pointer is nudged so removal targets that corner."""
    engine = _engine("gemini")
    det = engine.detect_watermark(image)
    if not det.detected:
        if not force:
            return image.copy(), None
        # Forced (--no-detect): remove at the default sparkle slot for the size.
        from remove_ai_watermarks.gemini_engine import get_watermark_config

        h, w = image.shape[:2]
        cfg = get_watermark_config(w, h)
        px, py = cfg.get_position(w, h)
        region = (px, py, cfg.logo_size, cfg.logo_size)
        result = engine.remove_watermark_custom(image, region)
        if inpaint:
            result = engine.inpaint_residual(result, region, strength=strength, method=inpaint_method)
        return result, region
    # If a specific region is given (multi-mark path), nudge the corner pointer.
    if target_region is not None:
        engine._infer_corner_from_region(image, target_region)
    result = engine.remove_watermark(image)
    # Reverse-alpha leaves a faint residual at the sparkle edge; the engine's
    # own residual inpaint cleans that seam (part of its reverse-alpha pipeline).
    if inpaint:
        result = engine.inpaint_residual(result, det.region, strength=strength, method=inpaint_method)
    return result, det.region


def _doubao_detect(image: NDArray[Any]) -> MarkDetection:
    d = _engine("doubao").detect(image)
    return MarkDetection("doubao", "Doubao 豆包AI生成 text", "bottom-right", d.detected, d.confidence, d.region)


def _doubao_remove(
    image: NDArray[Any], _inpaint_method: InpaintMethod, _inpaint: bool, _strength: float, force: bool,
    target_region: Region | None = None,
) -> tuple[NDArray[Any], Region | None]:
    """Remove a Doubao watermark.  When *target_region* is set (from detect_all),
    the engine's internal corner pointer is nudged so reverse-alpha targets that
    corner instead of the global best."""
    engine = _engine("doubao")
    det = engine.detect(image)
    if (det.detected or force) and engine.reverse_alpha_available(image):
        if target_region is not None:
            engine._infer_corner_from_region(image, target_region)
        result = engine.remove_watermark_reverse_alpha(image)
        if force and not det.detected:
            result = engine.inpaint_force_fallback(result, det.region)
        return result, (det.region if det.detected else target_region)
    return image.copy(), None


def _jimeng_detect(image: NDArray[Any]) -> MarkDetection:
    d = _engine("jimeng").detect(image)
    return MarkDetection("jimeng", "Jimeng 即梦AI wordmark", "bottom-right", d.detected, d.confidence, d.region)


def _jimeng_remove(
    image: NDArray[Any], _inpaint_method: InpaintMethod, _inpaint: bool, _strength: float, force: bool,
    target_region: Region | None = None,
) -> tuple[NDArray[Any], Region | None]:
    """Remove a Jimeng watermark.  When *target_region* is set (from detect_all),
    the engine's internal corner pointer is nudged so reverse-alpha targets that
    corner instead of the global best."""
    engine = _engine("jimeng")
    det = engine.detect(image)
    if (det.detected or force) and engine.reverse_alpha_available(image):
        if target_region is not None:
            engine._infer_corner_from_region(image, target_region)
        return engine.remove_watermark_reverse_alpha(image), (det.region if det.detected else target_region)
    return image.copy(), None


def _samsung_detect(image: NDArray[Any]) -> MarkDetection:
    d = _engine("samsung").detect(image)
    return MarkDetection("samsung", "Samsung Galaxy AI text", "bottom-left", d.detected, d.confidence, d.region)


def _samsung_remove(
    image: NDArray[Any], _inpaint_method: InpaintMethod, _inpaint: bool, _strength: float, force: bool,
    target_region: Region | None = None,
) -> tuple[NDArray[Any], Region | None]:
    """Remove a Samsung watermark.  When *target_region* is set (from detect_all),
    the engine's internal corner pointer is nudged so reverse-alpha targets that
    corner instead of the global best."""
    engine = _engine("samsung")
    det = engine.detect(image)
    if (det.detected or force) and engine.reverse_alpha_available(image):
        if target_region is not None:
            engine._infer_corner_from_region(image, target_region)
        return engine.remove_watermark_reverse_alpha(image), (det.region if det.detected else target_region)
    return image.copy(), None


def _aigc_label_detect(image: NDArray[Any]) -> MarkDetection:
    d = _engine("aigc_label").detect(image)
    return MarkDetection("aigc_label", "AIGC AI生成 label", "top-left", d.detected, d.confidence, d.region)


def _aigc_label_remove(
    image: NDArray[Any], _inpaint_method: InpaintMethod, _inpaint: bool, _strength: float, force: bool,
    target_region: Region | None = None,
) -> tuple[NDArray[Any], Region | None]:
    """Remove an AIGC label watermark via inpainting.
    When *target_region* is set (from detect_all), the engine's internal corner
    pointer is nudged so removal targets that corner."""
    engine = _engine("aigc_label")
    det = engine.detect(image)
    if det.detected or force:
        if target_region is not None:
            engine._infer_corner_from_region(image, target_region)
        result = engine.remove_watermark(image)
        return result, (det.region if det.detected else target_region)
    return image.copy(), None


_REGISTRY: tuple[KnownMark, ...] = (
    KnownMark("gemini", "Google Gemini sparkle", "bottom-right", True, "reverse-alpha", _gemini_detect, _gemini_remove),
    KnownMark(
        "doubao", "Doubao 豆包AI生成 text", "bottom-right", True, "reverse-alpha", _doubao_detect, _doubao_remove
    ),
    KnownMark(
        "jimeng", "Jimeng 即梦AI wordmark", "bottom-right", True, "reverse-alpha", _jimeng_detect, _jimeng_remove
    ),
    KnownMark(
        "samsung", "Samsung Galaxy AI text", "bottom-left", True, "reverse-alpha", _samsung_detect, _samsung_remove
    ),
    KnownMark(
        "aigc_label", "AIGC AI生成 label", "top-left", True, "inpaint", _aigc_label_detect, _aigc_label_remove
    ),
)


def known_marks() -> tuple[KnownMark, ...]:
    """All registered known visible watermarks."""
    return _REGISTRY


def mark_keys() -> list[str]:
    """Keys of all registered marks (for CLI choices)."""
    return [m.key for m in _REGISTRY]


def get_mark(key: str) -> KnownMark:
    """Look up a known mark by key (raises KeyError if unknown)."""
    for m in _REGISTRY:
        if m.key == key:
            return m
    raise KeyError(key)


def detect_marks(image: NDArray[Any], *, include_explicit: bool = True) -> list[MarkDetection]:
    """Detect every known mark in its usual place.

    Returns one MarkDetection per scanned mark (``detected`` flags which fired).
    ``include_explicit=False`` scans only the ``in_auto`` marks -- the set used
    by ``--mark auto``.
    """
    return [m.detect(image) for m in _REGISTRY if include_explicit or m.in_auto]


def best_auto_mark(image: NDArray[Any]) -> MarkDetection | None:
    """The highest-confidence detected ``in_auto`` mark, or None if none fired."""
    fired = [d for d in detect_marks(image, include_explicit=False) if d.detected]
    return max(fired, key=lambda d: d.confidence) if fired else None


def _regions_overlap(
    r1: tuple[int, int, int, int], r2: tuple[int, int, int, int], *, iou_threshold: float = 0.1
) -> bool:
    """Return True if two (x,y,w,h) regions have IoU above *iou_threshold*."""
    x1, y1, w1, h1 = r1
    x2, y2, w2, h2 = r2
    # Convert to (x1,y1,x2,y2) for overlap calculation
    ax1, ay1, ax2, ay2 = x1, y1, x1 + w1, y1 + h1
    bx1, by1, bx2, by2 = x2, y2, x2 + w2, y2 + h2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area1, area2 = w1 * h1, w2 * h2
    union = area1 + area2 - inter
    if union <= 0:
        return False
    return inter / union >= iou_threshold


def detect_all_auto_marks(image: NDArray[Any]) -> list[MarkDetection]:
    """Detect ALL watermarks that exceed their thresholds across all corners.

    Unlike :func:`detect_marks` (which returns at most one detection per engine,
    the best corner), this returns **every** corner detection from every
    ``in_auto`` engine whose confidence exceeds its threshold.  Used to remove
    multiple marks that may appear in different corners of a single image.

    Deduplication: if an ``aigc_label`` detection overlaps significantly with
    another engine's detection (e.g. jimeng in the same corner), the aigc_label
    one is dropped to avoid double-processing.
    """
    all_dets: list[MarkDetection] = []
    for m in _REGISTRY:
        if not m.in_auto:
            continue
        # Use each engine's detect_all() to get multi-corner results
        key = m.key
        eng = _engine(key)
        if key == "gemini":
            raw_dets = eng.detect_all_watermarks(image)
            for rd in raw_dets:
                detected = bool(rd.detected) and rd.confidence >= _GEMINI_AUTO_MIN_CONF
                all_dets.append(
                    MarkDetection("gemini", "Google Gemini sparkle", "bottom-right",
                                  detected, rd.confidence, rd.region)
                )
        elif key == "doubao":
            raw_dets = eng.detect_all(image)
            for rd in raw_dets:
                all_dets.append(
                    MarkDetection("doubao", "Doubao 豆包AI生成 text", "bottom-right",
                                  True, rd.confidence, rd.region)
                )
        elif key == "jimeng":
            raw_dets = eng.detect_all(image)
            for rd in raw_dets:
                all_dets.append(
                    MarkDetection("jimeng", "Jimeng 即梦AI wordmark", "bottom-right",
                                  True, rd.confidence, rd.region)
                )
        elif key == "samsung":
            raw_dets = eng.detect_all(image)
            for rd in raw_dets:
                all_dets.append(
                    MarkDetection("samsung", "Samsung Galaxy AI text", "bottom-left",
                                  True, rd.confidence, rd.region)
                )
        elif key == "aigc_label":
            raw_dets = eng.detect_all(image)
            for rd in raw_dets:
                all_dets.append(
                    MarkDetection("aigc_label", "AIGC AI生成 label", "top-left",
                                  True, rd.confidence, rd.region)
                )
    # Dedup: remove aigc_label detections that overlap with other (more specific) engines
    non_aigc = [d for d in all_dets if d.key != "aigc_label"]
    aigc_dets = [d for d in all_dets if d.key == "aigc_label"]
    filtered_aigc = []
    for ad in aigc_dets:
        if not any(_regions_overlap(ad.region, nd.region) for nd in non_aigc):
            filtered_aigc.append(ad)
    all_dets = non_aigc + filtered_aigc
    all_dets.sort(key=lambda d: d.confidence, reverse=True)
    return all_dets
