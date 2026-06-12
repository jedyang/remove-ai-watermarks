"""Tests for Unicode-safe cv2 image IO (issue #17)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from remove_ai_watermarks import image_io

if TYPE_CHECKING:
    from pathlib import Path

# Non-ASCII filenames that break cv2.imread/imwrite on Windows (issue #17).
_UNICODE_NAMES = [
    "jimeng-2026-05-27-一面白色的墙.png",  # Chinese
    "тест-изображение.png",  # Cyrillic
    "café-señor.png",  # accented Latin
]


def _make_bgr() -> np.ndarray:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[2:6, 2:6] = (10, 120, 240)  # a BGR block so the round-trip is checkable
    return img


class TestUnicodeRoundTrip:
    def test_write_then_read_preserves_pixels(self, tmp_path: Path) -> None:
        for name in _UNICODE_NAMES:
            path = tmp_path / name
            src = _make_bgr()
            assert image_io.imwrite(path, src) is True
            assert path.exists()
            out = image_io.imread(path)
            assert out is not None
            # PNG is lossless: pixels must match exactly.
            assert np.array_equal(out, src)

    def test_alpha_round_trip_with_unchanged_flag(self, tmp_path: Path) -> None:
        path = tmp_path / "豆包-alpha.png"
        bgra = np.zeros((8, 8, 4), dtype=np.uint8)
        bgra[..., 3] = 128
        assert image_io.imwrite(path, bgra) is True
        out = image_io.imread(path, cv2.IMREAD_UNCHANGED)
        assert out is not None
        assert out.shape[2] == 4
        assert np.array_equal(out, bgra)

    def test_reads_file_written_by_raw_cv2(self, tmp_path: Path) -> None:
        # An ASCII file written by plain cv2 must read back identically through
        # the wrapper (decode path is byte-compatible with cv2.imread).
        path = tmp_path / "ascii.png"
        src = _make_bgr()
        cv2.imwrite(str(path), src)
        out = image_io.imread(path)
        assert out is not None
        assert np.array_equal(out, src)


class TestToBgr:
    def test_grayscale_2d_promoted_to_bgr(self) -> None:
        gray = np.full((4, 5), 120, dtype=np.uint8)
        out = image_io.to_bgr(gray)
        assert out.shape == (4, 5, 3)
        # GRAY2BGR replicates the channel, so all three match the source.
        assert np.array_equal(out[..., 0], gray)
        assert np.array_equal(out[..., 0], out[..., 2])

    def test_single_channel_3d_promoted(self) -> None:
        gray = np.full((4, 5, 1), 7, dtype=np.uint8)
        assert image_io.to_bgr(gray).shape == (4, 5, 3)

    def test_bgra_dropped_to_bgr(self) -> None:
        bgra = np.zeros((4, 5, 4), dtype=np.uint8)
        bgra[..., :3] = (10, 120, 240)
        out = image_io.to_bgr(bgra)
        assert out.shape == (4, 5, 3)
        assert np.array_equal(out, bgra[..., :3])

    def test_bgr_returned_unchanged(self) -> None:
        bgr = _make_bgr()
        out = image_io.to_bgr(bgr)
        assert out is bgr  # 3-channel: no copy


class TestFailureSemantics:
    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert image_io.imread(tmp_path / "does-not-exist-不存在.png") is None

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.png"
        path.write_bytes(b"")
        assert image_io.imread(path) is None

    def test_undecodable_file_returns_none(self, tmp_path: Path) -> None:
        path = tmp_path / "garbage.png"
        path.write_bytes(b"not an image")
        assert image_io.imread(path) is None

    def test_imwrite_to_missing_directory_returns_false(self, tmp_path: Path) -> None:
        # An unwritable path must return False (cv2.imwrite contract), not raise.
        path = tmp_path / "no-such-dir" / "out.png"
        assert image_io.imwrite(path, _make_bgr()) is False
