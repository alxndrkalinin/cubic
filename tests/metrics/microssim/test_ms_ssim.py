"""Tests for ``cubic.metrics.ms_ssim`` (torchmetrics-faithful MS-SSIM)."""

from __future__ import annotations

import numpy as np
import pytest

from cubic.metrics.ms_ssim import (
    DEFAULT_BETAS,
    ms_ssim,
    _avgpool2,
    _ssim_scale_component,
)


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
    """Identity holds for float32 input (promoted to float64 internally)."""
    rng = np.random.default_rng(1)
    img = rng.random((256, 256)).astype(np.float32)
    score = ms_ssim(img, img, data_range=_data_range(img))
    assert score > 1.0 - 1e-4, f"identity (float32) should give 1.0, got {score}"


# ---------------------------------------------------------------------------
# Integer input (dtype promotion)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32])
def test_integer_input_matches_float64_reference(dtype):
    """Integer input is promoted to float64, not wrapped and not rescaled.

    Pre-fix ``np.pad`` preserved the integer dtype, so ``i1p * i1p`` wrapped
    mod ``2**n``, and skimage's ``gaussian`` (``preserve_range=False``)
    additionally divided by the dtype max while ``c1`` / ``c2`` stayed in the
    caller's ``data_range`` units. ``ms_ssim(u8, u8 + 5, data_range=255)``
    returned 0.7446 against a float64 reference of 0.9998.
    """
    rng = np.random.default_rng(0)
    img = (rng.random((256, 256)) * 200).astype(dtype)
    pred = (img + 5).astype(dtype)

    ours = ms_ssim(img, pred, data_range=255)
    ref = ms_ssim(img.astype(np.float64), pred.astype(np.float64), data_range=255)
    assert ours == ref, f"{np.dtype(dtype).name}: {ours} != float64 reference {ref}"
    assert ours > 0.999, f"{np.dtype(dtype).name}: {ours}"


def test_integer_identity_returns_one():
    """``ms_ssim(x, x)`` on ``uint8`` is 1.0, not 0.9999993.

    Even the identity case lost six digits pre-fix, because skimage rescaled
    the ``uint8`` input by 1/255 while ``c1`` / ``c2`` were built from the
    caller's ``data_range=255``.
    """
    rng = np.random.default_rng(1)
    img = (rng.random((256, 256)) * 200).astype(np.uint8)
    score = ms_ssim(img, img, data_range=255)
    assert score > 1.0 - 1e-9, f"integer identity should give 1.0, got {score}"


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


def test_batch_equals_mean_of_per_image():
    """``(N, H, W)`` MS-SSIM is the mean of per-image scores, not a batch pool.

    The slices are deliberately dissimilar so that pooling the maps across the
    batch (the old behavior) would diverge from the per-image mean.
    """
    rng = np.random.default_rng(11)
    gt = rng.random((3, 256, 256)).astype(np.float64)
    pred = gt.copy()
    pred[0] += 0.02 * rng.standard_normal(gt[0].shape)  # near-identical
    pred[1] = rng.random((256, 256))  # unrelated
    pred[2] += 0.2 * rng.standard_normal(gt[2].shape)  # noisy
    dr = float(gt.max() - gt.min())

    per_image = [float(ms_ssim(pred[i], gt[i], data_range=dr)) for i in range(3)]
    batched = ms_ssim(pred, gt, data_range=dr)
    assert abs(batched - float(np.mean(per_image))) < 1e-9


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
# _ssim_scale_component helper
# ---------------------------------------------------------------------------


def test_ssim_scale_component_returns_only_requested_map():
    """The helper reduces spatial axes only and honours ``full_ssim``.

    It used to always build both the full SSIM map and the cropped CS map,
    wasting a full-size divide at every scale (5 with the default betas)
    since MS-SSIM never consumes both at the same scale.
    """
    rng = np.random.default_rng(8)
    img = rng.random((3, 64, 64))
    pred = img + 0.05 * rng.standard_normal(img.shape)
    kwargs = {"c1": 1e-4, "c2": 9e-4, "kernel_size": 11, "sigma": 1.5}

    cs = _ssim_scale_component(pred, img, full_ssim=False, **kwargs)
    full = _ssim_scale_component(pred, img, full_ssim=True, **kwargs)

    # Batch axis preserved, spatial axes reduced.
    assert cs.shape == (3,)
    assert full.shape == (3,)
    # The two maps are genuinely different quantities.
    assert not np.allclose(cs, full)

    # 2-D input reduces to a scalar.
    assert _ssim_scale_component(pred[0], img[0], full_ssim=False, **kwargs).ndim == 0


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


@pytest.mark.parametrize("kernel_size", [7, 9])
def test_ms_ssim_matches_torchmetrics_non_default_kernel(kernel_size: int):
    """Non-default ``kernel_size`` matches torchmetrics within 1e-3.

    Guards the desync where the reflect-pad/crop tracked ``kernel_size`` while
    the Gaussian radius was fixed by ``sigma``/``truncate``: for a smaller
    kernel the Gaussian leaked ``cval=0`` into the valid region.
    """
    pytest.importorskip("torchmetrics")
    torch = pytest.importorskip("torch")
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

    rng = np.random.default_rng(0)
    img = rng.random((256, 256)).astype(np.float32)
    pred = img + 0.05 * rng.standard_normal(img.shape).astype(np.float32)
    dr = _data_range(img)

    ours = ms_ssim(pred, img, data_range=dr, kernel_size=kernel_size)
    m = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=dr, kernel_size=kernel_size
    )
    theirs = float(
        m(torch.from_numpy(pred[None, None]), torch.from_numpy(img[None, None]))
    )
    assert abs(ours - theirs) < 1e-3, (
        f"kernel_size={kernel_size} ours={ours:.6f} theirs={theirs:.6f}"
    )


def test_kernel_size_governs_gaussian_width():
    """``kernel_size`` changes the score at fixed ``sigma``.

    Documents the deliberate divergence from torchmetrics: torchmetrics sizes
    its Gaussian from ``sigma`` alone (``int(3.5*sigma+0.5)*2+1``,
    ``ssim.py:126``) and ignores ``kernel_size`` on the Gaussian path, whereas
    here ``kernel_size`` sets both the reflect-pad width and the Gaussian
    radius. The docstrings state this; the test pins it.
    """
    rng = np.random.default_rng(9)
    img = rng.random((256, 256))
    pred = img + 0.05 * rng.standard_normal(img.shape)
    dr = _data_range(img)
    wide = ms_ssim(pred, img, data_range=dr, sigma=1.5, kernel_size=11)
    narrow = ms_ssim(pred, img, data_range=dr, sigma=1.5, kernel_size=7)
    assert wide != narrow, "kernel_size must affect the Gaussian width"


@pytest.mark.parametrize(
    ("sigma", "kernel_size"), [(1.5, 11), (1.5, 7), (0.8, 11), (3.0, 7)]
)
def test_gaussian_width_divergence_stays_within_tolerance(
    sigma: float, kernel_size: int
):
    """The documented ``sigma`` / ``kernel_size`` divergence stays under 1e-3.

    Only ``sigma=1.5, kernel_size=11`` is an exact agreement (torchmetrics'
    ``gauss_kernel_size`` is then 11 too). The other combinations diverge
    because torchmetrics ignores ``kernel_size`` here; pin the magnitude so a
    future numerics change cannot widen the gap silently.
    """
    pytest.importorskip("torchmetrics")
    torch = pytest.importorskip("torch")
    from torchmetrics.image import MultiScaleStructuralSimilarityIndexMeasure

    rng = np.random.default_rng(0)
    img = rng.random((256, 256)).astype(np.float32)
    pred = img + 0.05 * rng.standard_normal(img.shape).astype(np.float32)
    dr = _data_range(img)

    ours = ms_ssim(pred, img, data_range=dr, sigma=sigma, kernel_size=kernel_size)
    m = MultiScaleStructuralSimilarityIndexMeasure(
        data_range=dr, sigma=sigma, kernel_size=kernel_size
    )
    theirs = float(
        m(torch.from_numpy(pred[None, None]), torch.from_numpy(img[None, None]))
    )
    assert abs(ours - theirs) < 1e-3, (
        f"sigma={sigma} kernel_size={kernel_size} "
        f"ours={ours:.6f} theirs={theirs:.6f} diff={abs(ours - theirs):.3e}"
    )
