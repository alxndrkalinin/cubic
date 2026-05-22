"""Tests for ``cubic.metrics.ms_ssim`` (torchmetrics-faithful MS-SSIM)."""

from __future__ import annotations

import numpy as np
import pytest

from cubic.metrics.ms_ssim import DEFAULT_BETAS, ms_ssim, _avgpool2


def _data_range(img: np.ndarray) -> float:
    """Compute a non-zero ``data_range`` (handles constant inputs).

    ``ndarray.ptp`` was removed in NumPy 2.0, so use ``max - min`` plus a
    tiny floor to avoid div-by-zero on constant images.
    """
    return float(img.max() - img.min()) + 1e-12


def _edge_heavy_chessboard(shape: tuple[int, int], block: int = 16) -> np.ndarray:
    """Build a chessboard pattern with sharp edges."""
    h, w = shape
    yy, xx = np.indices((h, w))
    pat = ((yy // block) + (xx // block)) % 2
    return pat.astype(np.float32)


# ---------------------------------------------------------------------------
# Identity / basic behavior
# ---------------------------------------------------------------------------


def test_identity_returns_one():
    """``ms_ssim(img, img)`` is 1.0 within float64 round-off."""
    rng = np.random.default_rng(0)
    img = rng.random((256, 256)).astype(np.float64)
    score = ms_ssim(img, img, data_range=_data_range(img))
    assert score > 1.0 - 1e-6, f"identity should give 1.0, got {score}"


def test_identity_returns_one_float32():
    """Identity holds for float32 too, within slightly looser tolerance."""
    rng = np.random.default_rng(1)
    img = rng.random((256, 256)).astype(np.float32)
    score = ms_ssim(img, img, data_range=_data_range(img))
    # float32 round-off is larger but still well under 1e-4.
    assert score > 1.0 - 1e-4, f"identity (float32) should give 1.0, got {score}"


def test_default_betas_finest_to_coarsest():
    """Confirm canonical torchmetrics weight order (used by parity test)."""
    assert DEFAULT_BETAS == (0.0448, 0.2856, 0.3001, 0.2363, 0.1333)


# ---------------------------------------------------------------------------
# Shape / dimensionality
# ---------------------------------------------------------------------------


def test_accepts_2d_input():
    """2-D ``(H, W)`` inputs are valid."""
    rng = np.random.default_rng(2)
    img = rng.random((256, 256)).astype(np.float32)
    pred = img + 0.05 * rng.standard_normal(img.shape).astype(np.float32)
    out = ms_ssim(pred, img, data_range=_data_range(img))
    assert isinstance(out, float)
    assert 0.0 <= out <= 1.0


def test_accepts_3d_input():
    """3-D ``(N, H, W)`` inputs are accepted."""
    rng = np.random.default_rng(3)
    img = rng.random((2, 256, 256)).astype(np.float32)
    pred = img + 0.05 * rng.standard_normal(img.shape).astype(np.float32)
    out = ms_ssim(pred, img, data_range=_data_range(img))
    assert isinstance(out, float)


def test_rejects_1d_input():
    """``ndim=1`` raises ``ValueError``."""
    a = np.zeros(256, dtype=np.float32)
    with pytest.raises(ValueError, match="ndim"):
        ms_ssim(a, a, data_range=1.0)


def test_shape_mismatch_raises():
    """Mismatched shapes raise ``ValueError``."""
    a = np.zeros((256, 256), dtype=np.float32)
    b = np.zeros((256, 128), dtype=np.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        ms_ssim(a, b, data_range=1.0)


# ---------------------------------------------------------------------------
# Boundary / validation cases
# ---------------------------------------------------------------------------


def test_kernel_size_zero_raises():
    """``kernel_size=0`` raises ``ValueError``."""
    img = np.zeros((256, 256), dtype=np.float32)
    with pytest.raises(ValueError, match="kernel_size"):
        ms_ssim(img, img, data_range=1.0, kernel_size=0)


def test_kernel_size_even_raises():
    """Even ``kernel_size`` (e.g. 2) raises ``ValueError``."""
    img = np.zeros((256, 256), dtype=np.float32)
    with pytest.raises(ValueError, match="kernel_size"):
        ms_ssim(img, img, data_range=1.0, kernel_size=2)


def test_data_range_zero_raises():
    """``data_range=0`` raises ``ValueError``."""
    img = np.zeros((256, 256), dtype=np.float32)
    with pytest.raises(ValueError, match="data_range"):
        ms_ssim(img, img, data_range=0.0)


def test_data_range_non_finite_raises():
    """Non-finite ``data_range`` raises ``ValueError``."""
    img = np.zeros((256, 256), dtype=np.float32)
    with pytest.raises(ValueError, match="data_range"):
        ms_ssim(img, img, data_range=float("nan"))
    with pytest.raises(ValueError, match="data_range"):
        ms_ssim(img, img, data_range=float("inf"))


def test_spatial_below_min_raises():
    """Spatial dim 175 (below the 176 minimum) raises ``ValueError``."""
    img = np.zeros((175, 175), dtype=np.float32)
    with pytest.raises(ValueError, match=">="):
        ms_ssim(img, img, data_range=1.0)


def test_min_spatial_176_accepted():
    """Spatial dim 176 (exact minimum) is accepted."""
    rng = np.random.default_rng(4)
    img = rng.random((176, 176)).astype(np.float32)
    # Should not raise.
    out = ms_ssim(img, img, data_range=_data_range(img))
    assert out > 1.0 - 1e-4


# ---------------------------------------------------------------------------
# _avgpool2 helper
# ---------------------------------------------------------------------------


def test_avgpool2_even_shape():
    """``_avgpool2`` halves even spatial dims."""
    rng = np.random.default_rng(5)
    x = rng.random((10, 10)).astype(np.float32)
    out = _avgpool2(x)
    assert out.shape == (5, 5)
    # Match a manual 2x2 mean for the (0,0) cell.
    expected = x[:2, :2].mean()
    assert abs(float(out[0, 0]) - float(expected)) < 1e-6


def test_avgpool2_odd_shape_trims_to_even():
    """``_avgpool2`` trims to the largest even extent before pooling."""
    rng = np.random.default_rng(6)
    x = rng.random((11, 13)).astype(np.float32)
    out = _avgpool2(x)
    # Trimmed to 10x12 then halved.
    assert out.shape == (5, 6)


def test_avgpool2_batched():
    """``_avgpool2`` handles leading batch axes via ``...`` indexing."""
    rng = np.random.default_rng(7)
    x = rng.random((3, 12, 14)).astype(np.float32)
    out = _avgpool2(x)
    assert out.shape == (3, 6, 7)


# ---------------------------------------------------------------------------
# Direct torchmetrics parity (skipped if torchmetrics is unavailable)
# ---------------------------------------------------------------------------


def test_ms_ssim_matches_torchmetrics():
    """Match torchmetrics' MultiScaleStructuralSimilarityIndexMeasure within 1e-3."""
    pytest.importorskip("torchmetrics")
    torch = pytest.importorskip("torch")
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

    rng = np.random.default_rng(0)
    regimes: list[tuple[str, np.ndarray]] = [
        ("uniform_random_256", rng.random((256, 256)).astype(np.float32)),
        ("chessboard_edges", _edge_heavy_chessboard((256, 256))),
        ("identity_constant", np.ones((256, 256), dtype=np.float32)),
        (
            "near_zero",
            rng.random((256, 256)).astype(np.float32) * 1e-3,
        ),
    ]

    for name, img in regimes:
        pred = img + 0.05 * rng.standard_normal(img.shape).astype(np.float32)
        dr = _data_range(img)
        ours = ms_ssim(pred, img, data_range=dr)

        m = MultiScaleStructuralSimilarityIndexMeasure(data_range=dr)
        theirs = float(
            m(
                torch.from_numpy(pred[None, None]),
                torch.from_numpy(img[None, None]),
            )
        )
        assert abs(ours - theirs) < 1e-3, (
            f"[{name}] ours={ours:.6f} theirs={theirs:.6f} "
            f"diff={abs(ours - theirs):.6e}"
        )
