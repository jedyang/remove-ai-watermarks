"""Automatic pipeline planning for the ``--auto`` quality mode.

``plan(image_path)`` inspects the INPUT image (before the diffusion model loads)
and returns the quality modes to use, so the pipeline can adapt to content. It is
meant to run as the FIRST step of the invisible/all pipeline, wherever that pipeline
runs (locally, or the raiw.cc Modal GPU worker) -- never on a memory-constrained web
host (image work there OOM-crashes the container).

Routing is **quality-priority**: ControlNet (text/face-structure preservation) is the
default; it is only skipped for a clearly structure-less image (no face, no text,
near-zero edges), where plain SDXL is cheaper and just as good. GFPGAN face
restoration is enabled when a face is present. A mild sharpen + grain polish is added
when a smoothing pass (controlnet or face restore) ran, to counter the over-smoothed
"AI look".

Detection is **cv2-only and torch-free**: OpenCV YuNet (``cv2.FaceDetectorYN``) for
faces -- a 232 KB MIT-licensed model bundled in ``assets/`` -- plus a Canny
edge-density + MSER region heuristic for text/structure. The whole planner peaks
~100 MB RSS in a few ms, so it adds nothing meaningful to a GPU run and runs anywhere
the pipeline runs. (Phase 1 applies a fixed mild polish; an adaptive Laplacian-variance
polish that measures the OUTPUT is a later phase.)

The text heuristic is a deliberately rough Phase-1 placeholder (DBNet via cv2.dnn is
the planned precision upgrade); it only ever ADDS controlnet, so a miss is backstopped
by the edge-density route and a false positive only costs a controlnet run.
"""

# cv2/numpy boundary: cv2 ships no usable element types; relax the unknown-type rules
# for this file only.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingTypeArgument=false, reportMissingTypeStubs=false, reportMissingImports=false, reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportCallIssue=false, reportIndexIssue=false, reportOperatorIssue=false, reportOptionalMemberAccess=false, reportOptionalCall=false, reportOptionalSubscript=false, reportOptionalOperand=false, reportAttributeAccessIssue=false, reportPrivateImportUsage=false, reportPrivateUsage=false, reportInvalidTypeForm=false, reportConstantRedefinition=false, reportUnnecessaryComparison=false
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# ── Routing thresholds (tunable; quality-priority -> controlnet unless clearly flat) ──
# Canny edge-density below this, AND no face AND no text -> plain SDXL (nothing to
# preserve). The headshot measures ~0.022, a busy photo higher; only a near-flat
# gradient/solid image falls under 0.008.
_STRUCTURELESS_EDGE_MAX = 0.008
# MSER regions per megapixel above this -> likely text. Rough Phase-1 heuristic: a
# no-text portrait measures a few hundred/MP, dense text far more. Set high so it
# rarely false-fires; it only ever ADDS controlnet so miscalibration is low-harm.
_TEXT_MSER_PER_MP = 1500.0
_FACE_SCORE = 0.6  # YuNet confidence for a face to count
# Downscale the long side to this for DETECTION only (faces stay detectable down to
# ~10px, and this bounds YuNet/MSER cost on huge inputs). Removal runs at full res.
_DETECT_MAX_SIDE = 1024

# Auto polish applied only when a smoothing pass ran (controlnet or face restore),
# to counter the soft "AI look". Conservative defaults; the user can override.
_AUTO_UNSHARP = 0.5
_AUTO_HUMANIZE = 2.0
_UPSCALE_FLOOR = 1024

_YUNET_ASSET = "face_detection_yunet_2023mar.onnx"  # MIT (Shiqi Yu), OpenCV Zoo
_yunet: Any = None  # lazy singleton


@dataclass(frozen=True)
class AutoConfig:
    """Resolved quality modes from content analysis (the ``--auto`` plan)."""

    pipeline: str  # "default" | "controlnet"
    restore_faces: bool
    unsharp: float
    humanize: float
    min_resolution: int
    # signals retained for logging / debugging a bad pick
    has_face: bool
    has_text: bool
    edge_density: float
    width: int
    height: int

    @property
    def reason(self) -> str:
        """One-line human-readable summary of the plan (logged per image)."""
        bits = ["face" if self.has_face else "no-face"]
        if self.has_text:
            bits.append("text")
        bits.append(f"edges={self.edge_density:.3f}")
        rf = ", face-restore on" if self.restore_faces else ""
        polish = f", unsharp {self.unsharp}/grain {self.humanize}" if (self.unsharp or self.humanize) else ""
        return f"{'+'.join(bits)} -> {self.pipeline} pipeline{rf}{polish}"


def _to_bgr(image: NDArray[Any]) -> NDArray[Any]:
    """Normalize a 2D grayscale or 4-channel BGRA array to 3-channel BGR."""
    import cv2

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def _to_gray(image: NDArray[Any]) -> NDArray[Any]:
    """Single-channel grayscale; passes a 2D (already-gray) input through unchanged."""
    import cv2

    if image.ndim == 3 and image.shape[2] >= 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _downscale_for_detection(image: NDArray[Any]) -> NDArray[Any]:
    """Shrink the long side to ``_DETECT_MAX_SIDE`` for cheap, bounded detection."""
    import cv2

    h, w = image.shape[:2]
    long_side = max(h, w)
    if long_side <= _DETECT_MAX_SIDE:
        return image
    scale = _DETECT_MAX_SIDE / long_side
    return cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


def detect_face(image: NDArray[Any]) -> bool:
    """True if OpenCV YuNet finds at least one face. cv2-only, torch-free."""
    import cv2

    global _yunet
    img = _to_bgr(image)
    h, w = img.shape[:2]
    if h < 1 or w < 1:
        return False
    try:
        if _yunet is None:
            model = Path(__file__).parent / "assets" / _YUNET_ASSET
            _yunet = cv2.FaceDetectorYN.create(str(model), "", (w, h), _FACE_SCORE, 0.3, 5000)
        _yunet.setInputSize((w, h))
        _, faces = _yunet.detect(img)
    except cv2.error as e:  # malformed input / model
        logger.debug("YuNet face detect failed (%s); assuming no face", e)
        return False
    return faces is not None and len(faces) > 0


def detect_text(image: NDArray[Any]) -> bool:
    """Rough MSER-based text-presence heuristic (Phase-1 placeholder for DBNet)."""
    import cv2

    gray = _to_gray(image)
    h, w = gray.shape[:2]
    try:
        regions, _ = cv2.MSER_create().detectRegions(gray)
    except cv2.error:
        return False
    per_mp = len(regions) / max(1e-6, (h * w) / 1e6)
    return per_mp > _TEXT_MSER_PER_MP


def edge_density(image: NDArray[Any]) -> float:
    """Fraction of Canny edge pixels -- a cheap 'has structure' proxy in [0, 1]."""
    import cv2

    gray = _to_gray(image)
    edges = cv2.Canny(gray, 100, 200)
    return float((edges > 0).mean())


def plan(image_path: Path) -> AutoConfig | None:
    """Inspect the input image and return the quality modes, or None if unreadable.

    Pure analysis: loads the image, runs the cv2 detectors on a downscaled copy, and
    applies the quality-priority routing rules. Safe to call wherever the pipeline
    runs; no diffusion model is loaded.
    """
    from remove_ai_watermarks import image_io

    image = image_io.imread(image_path)
    if image is None:
        return None

    h, w = image.shape[:2]
    small = _downscale_for_detection(image)
    gray = _to_gray(small)  # convert once; the text/edge detectors pass a gray input through
    has_face = detect_face(small)  # YuNet needs the 3-channel image
    has_text = detect_text(gray)
    edges = edge_density(gray)

    structureless = (not has_face) and (not has_text) and edges < _STRUCTURELESS_EDGE_MAX
    pipeline = "default" if structureless else "controlnet"
    restore_faces = has_face
    smoothing = pipeline == "controlnet" or restore_faces

    cfg = AutoConfig(
        pipeline=pipeline,
        restore_faces=restore_faces,
        unsharp=_AUTO_UNSHARP if smoothing else 0.0,
        humanize=_AUTO_HUMANIZE if smoothing else 0.0,
        min_resolution=_UPSCALE_FLOOR,
        has_face=has_face,
        has_text=has_text,
        edge_density=edges,
        width=w,
        height=h,
    )
    logger.debug("auto plan for %s: %s", image_path, cfg.reason)
    return cfg
