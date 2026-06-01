"""Vendored noai-watermark code for invisible watermark removal.

Original: https://github.com/mertizci/noai-watermark (MIT License)

The public API (``WatermarkRemover`` / ``remove_watermark`` / ``remove_ai_metadata``)
is exposed **lazily** via PEP 562 ``__getattr__``: importing a light submodule
(e.g. ``noai.c2pa`` / ``noai.constants`` from ``identify``) must NOT eagerly pull
``watermark_remover``, which imports torch + diffusers at module top. Keeping this
lazy is what lets ``import remove_ai_watermarks.identify`` stay cheap (~36 MB, no
torch) even in a full install where the ``gpu``/``detect`` extras are present --
otherwise the mere presence of torch in the env inflated identify to ~420 MB and
risked OOM on a 512 MB host.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from remove_ai_watermarks.noai.cleaner import remove_ai_metadata
    from remove_ai_watermarks.noai.watermark_remover import WatermarkRemover, remove_watermark

__all__ = ["WatermarkRemover", "remove_ai_metadata", "remove_watermark"]


def __getattr__(name: str) -> object:
    """Resolve the public API on first access (PEP 562), not at package import."""
    if name == "remove_ai_metadata":
        from remove_ai_watermarks.noai.cleaner import remove_ai_metadata

        return remove_ai_metadata
    if name in ("WatermarkRemover", "remove_watermark"):
        from remove_ai_watermarks.noai import watermark_remover

        return getattr(watermark_remover, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
