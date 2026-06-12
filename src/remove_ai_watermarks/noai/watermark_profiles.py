"""Watermark removal model profiles and the default strength.

Pure configuration and lookup functions with no ML dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"

# Canonical pipeline-profile names + the back-compat alias. The plain SDXL img2img
# profile is ``sdxl``; ``default`` is kept as an accepted alias (it was the profile's
# name before ``controlnet`` became the default-selected pipeline, 2026-06-09).
SDXL_PROFILE = "sdxl"
_PROFILE_ALIASES = {"default": SDXL_PROFILE}


def normalize_profile(profile: str) -> str:
    """Canonicalize a pipeline-profile name, resolving the ``default`` -> ``sdxl`` alias."""
    normalized = profile.strip().lower()
    return _PROFILE_ALIASES.get(normalized, normalized)


# The SDXL-native canny ControlNet used by the ``controlnet`` pipeline. The
# ControlNet is an add-on to the SDXL base checkpoint (DEFAULT_MODEL_ID), not a
# separate base model, so both the ``sdxl`` and ``controlnet`` profiles load the
# same base weights and share the same vendor-adaptive strength ladder (see below).
CONTROLNET_CANNY_MODEL = "xinsir/controlnet-canny-sdxl-1.0"

# Vendor-adaptive default denoising strength for the SDXL img2img scrub, overridable
# from the CLI (`--strength`). The right strength depends on which vendor's SynthID is
# present (detected from the C2PA issuer, metadata.synthid_source). The SAME ladder
# applies to BOTH pipelines (`sdxl` plain img2img and `controlnet`) -- see "why one
# ladder" below.
#
# Data basis (see docs/synthid.md sections 2.2 / 5.5): the values are the ORACLE-
# CERTIFIED controlnet floors (2026-06-04, isolated Modal cert app, each vendor on its
# own verifier): OpenAI 0.20 (2 photoreal x 3 seeds = 6/6 clean, resolution-independent),
# Google 0.30 (clean on 2/2 seeds, validated ONLY at <= 1536 -- Gemini is resolution-
# sensitive, native ~2816 likely needs ~0.35+). Unknown vendor gets the Google (more
# robust watermark) value: safe-by-default.
#
# Why ONE ladder for both pipelines (2026-06-09): the certification was run on
# controlnet, and it does NOT transfer to `sdxl` by symmetry -- the two pipelines have
# OPPOSITE hard cases (controlnet leaves SynthID on photoreal, `sdxl` leaves it on flat
# graphics; the content-x-pipeline table in docs/synthid.md §5.1). BUT on its OWN hard
# case (flat fills) `sdxl` is the WEAKER remover -- plain img2img at low strength barely
# perturbs a flat region -- so it needs AT LEAST as much strength as controlnet, not
# less. Hence the certified controlnet floor is the right floor for `sdxl` too. The
# higher strength costs little quality where it matters: `controlnet` is now the default
# pipeline, so `sdxl` is reached only for structure-less inputs (via `--auto`) or an
# explicit `--pipeline sdxl`, where over-regeneration has no faces/text to damage. NOTE:
# this is a MARGIN argument for `sdxl`, not a fresh certification -- there is no local
# SynthID detector, so if an oracle still reads SynthID on a flat `sdxl` output, raise
# `--strength`.
OPENAI_STRENGTH = 0.20
GEMINI_STRENGTH = 0.30
UNKNOWN_STRENGTH = 0.30
# Backwards-compatible alias: the vendor-unknown value (what a caller gets without a
# detected vendor). Kept as DEFAULT_STRENGTH for existing references.
DEFAULT_STRENGTH = UNKNOWN_STRENGTH

# Detected-vendor -> default strength. Vendor strings come from `vendor_for_strength`.
_VENDOR_STRENGTH = {"openai": OPENAI_STRENGTH, "google": GEMINI_STRENGTH}


def strength_default_help() -> str:
    """One-line description of the vendor-adaptive default, derived from the constants.

    Single source of truth for the CLI ``--strength`` help so the numbers can never
    drift from the actual ladder (they did once when the per-pipeline split was unified).
    """
    return (
        f"vendor-adaptive (OpenAI {OPENAI_STRENGTH} / Google {GEMINI_STRENGTH} / "
        f"unknown {UNKNOWN_STRENGTH}, from the C2PA issuer; same ladder for both pipelines)"
    )


def resolve_strength(strength: float | None, vendor: str | None = None) -> float:
    """Resolve the denoising strength, applying the vendor default when unset.

    ``None`` means "the user did not pass ``--strength``", which resolves
    **vendor-adaptively**: ``vendor`` (``"openai"`` / ``"google"`` / None, from
    ``vendor_for_strength``) selects ``OPENAI_STRENGTH`` / ``GEMINI_STRENGTH`` /
    ``UNKNOWN_STRENGTH``. The same ladder applies to both pipelines (see the module
    comment for why one ladder is correct). An explicit value always wins (including
    ``0.0`` -- the check is ``is None``, not falsiness). Shared by the CLI (for display)
    and the engine (for execution) so the two never disagree -- both must pass the SAME
    ``vendor``.
    """
    if strength is not None:
        return strength
    return _VENDOR_STRENGTH.get(vendor or "", UNKNOWN_STRENGTH)


def vendor_for_strength(image_path: Path) -> Literal["openai", "google"] | None:
    """Detect the SynthID vendor for strength selection: ``"openai"`` / ``"google"`` / None.

    Reads the C2PA SynthID proxy (``metadata.synthid_source``) on the ORIGINAL input,
    so it must run before any pass that strips metadata. When both issuers appear (a
    rare multi-sign anomaly) Google wins -- the more-robust watermark -> safer (higher)
    strength. Returns None when metadata is stripped or the issuer is neither vendor,
    which maps to ``UNKNOWN_STRENGTH``. Lazy-imports ``metadata`` to keep this module
    dependency-light.
    """
    try:
        from remove_ai_watermarks.metadata import synthid_source

        src = (synthid_source(image_path) or "").lower()
    except Exception:  # metadata unreadable -> treat as unknown vendor
        return None
    if "google" in src:
        return "google"
    if "openai" in src:
        return "openai"
    return None
