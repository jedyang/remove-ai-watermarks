"""Tests for the provenance identifier (identify.py).

Pure attribution logic is unit-tested directly; end-to-end verdicts assert
against the real committed C2PA / IPTC fixtures in data/samples/.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from remove_ai_watermarks.identify import (
    ProvenanceReport,
    _ai_tools_in,
    _attribute_platform,
    _issuers_in,
    identify,
)

SAMPLES_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"


# ── Pure attribution logic (no file IO) ─────────────────────────────


class TestAttributePlatform:
    def test_openai(self):
        assert "OpenAI" in (_attribute_platform(["OpenAI"]) or "")

    def test_designer_wins_over_openai_backend(self):
        # Microsoft Designer signs as "OpenAI, Microsoft"; name the product.
        platform = _attribute_platform(["OpenAI", "Microsoft"])
        assert platform
        assert "Designer" in platform

    def test_adobe(self):
        assert _attribute_platform(["Adobe"]) == "Adobe Firefly"

    def test_google(self):
        assert "Google" in (_attribute_platform(["Google LLC"]) or "")

    def test_truepic_is_signer_not_generator(self):
        platform = _attribute_platform(["Truepic"])
        assert platform
        assert "signer" in platform.lower()

    def test_empty_is_none(self):
        assert _attribute_platform([]) is None


class TestIssuersIn:
    def test_finds_openai(self):
        assert _issuers_in(b"...OpenAI...trainedAlgorithmicMedia") == ["OpenAI"]

    def test_finds_multiple_sorted(self):
        assert _issuers_in(b"Microsoft and OpenAI") == ["Microsoft", "OpenAI"]

    def test_none_present(self):
        assert _issuers_in(b"just some bytes") == []


class TestAiToolsIn:
    def test_finds_generator(self):
        assert _ai_tools_in(b"...claim_generator Imagen 3...") == ["Imagen"]

    def test_none_present(self):
        assert _ai_tools_in(b"a regular photo, no tools") == []


class TestIdentifyNonPng:
    """Non-PNG containers (JPEG/WebP/AVIF) carry C2PA where the caBX parser can't
    reach; identify recovers issuer + generator via the binary scan. Synthetic
    byte blobs mirror tests/test_metadata.py::TestSynthIDSourceNonPng.
    """

    def _c2pa_jpeg(self, tmp_path: Path, blob: bytes) -> Path:
        path = tmp_path / "img.jpg"
        path.write_bytes(b"\xff\xd8\xff\xe1jumbc2pa" + blob + b"\xff\xd9")
        return path

    def test_google_imagen_jpeg(self, tmp_path: Path):
        path = self._c2pa_jpeg(tmp_path, b"Google Imagen ... trainedAlgorithmicMedia")
        r = identify(path, check_visible=False)
        assert r.is_ai_generated is True
        assert r.platform is not None
        assert "Google" in r.platform
        # Generator recovered from the non-PNG blob shows up in the c2pa signal.
        c2pa_signal = next(s for s in r.signals if s.name == "c2pa")
        assert "Imagen" in c2pa_signal.detail

    def test_openai_jpeg_has_synthid(self, tmp_path: Path):
        path = self._c2pa_jpeg(tmp_path, b"OpenAI DALL-E ... trainedAlgorithmicMedia")
        r = identify(path, check_visible=False)
        assert any("SynthID" in w for w in r.watermarks)


# ── End-to-end verdicts on real fixtures ────────────────────────────


@pytest.mark.skipif(not SAMPLES_DIR.exists(), reason="data/samples not present")
class TestIdentifyRealSamples:
    def test_openai_chatgpt(self):
        r = identify(SAMPLES_DIR / "chatgpt-1.png", check_visible=False)
        assert r.is_ai_generated is True
        assert r.confidence == "high"
        assert r.platform
        assert "OpenAI" in r.platform
        assert any("C2PA" in w for w in r.watermarks)
        assert any("SynthID" in w for w in r.watermarks)

    def test_adobe_firefly_has_no_synthid(self):
        r = identify(SAMPLES_DIR / "firefly-1.png", check_visible=False)
        assert r.is_ai_generated is True
        assert r.platform == "Adobe Firefly"
        assert not any("SynthID" in w for w in r.watermarks)

    def test_iptc_made_with_ai(self):
        # mj-1.png carries the IPTC digitalSourceType "Made with AI" marker.
        r = identify(SAMPLES_DIR / "mj-1.png", check_visible=False)
        assert r.is_ai_generated is True
        assert any("IPTC" in w for w in r.watermarks)

    def test_clean_photo_is_unknown_not_clean(self):
        r = identify(SAMPLES_DIR / "not-ai-1.jpeg", check_visible=False)
        assert r.is_ai_generated is None  # never asserted False
        assert r.platform is None
        assert r.confidence == "none"
        assert r.watermarks == []

    def test_strip_caveat_always_present(self):
        r = identify(SAMPLES_DIR / "not-ai-1.jpeg", check_visible=False)
        assert any("not proof" in c for c in r.caveats)

    def test_returns_report_dataclass(self):
        assert isinstance(identify(SAMPLES_DIR / "firefly-1.png", check_visible=False), ProvenanceReport)
