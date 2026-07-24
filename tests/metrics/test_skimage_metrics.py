"""Tests for image quality metrics."""

import warnings

import numpy as np
import pytest
from skimage import metrics as skimage_metrics

from cubic.metrics.skimage_metrics import psnr, ssim, nrmse


@pytest.fixture
def test_images() -> tuple[np.ndarray, np.ndarray]:
    """Create test images for metric comparison."""
    rng = np.random.default_rng(42)
    img1 = rng.random((8, 8)).astype(float)
    img2 = img1 + 0.1 * rng.random((8, 8)).astype(float)
    return img1, img2


@pytest.fixture
def test_mask() -> np.ndarray:
    """Create a test mask."""
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    return mask


def test_nrmse(
    test_images: tuple[np.ndarray, np.ndarray], test_mask: np.ndarray
) -> None:
    """Test NRMSE: matches skimage, scale-invariant, and masked versions."""
    img1, img2 = test_images

    # (i) Matches scikit-image implementation
    cubic_result = nrmse(img1, img2, scale_invariant=False)
    skimage_result = skimage_metrics.normalized_root_mse(img1, img2)
    assert np.isclose(cubic_result, skimage_result)

    # (ii) Scale-invariant version
    img2_scaled = 2 * img1
    err_scale_inv = nrmse(img1, img2_scaled, scale_invariant=True)
    err_non_scale_inv = nrmse(img1, img2_scaled, scale_invariant=False)
    assert np.isclose(err_scale_inv, 0.0)
    assert not np.isclose(err_non_scale_inv, 0.0)

    # (iii) Masked version
    img2_masked = img2.copy()
    img2_masked[test_mask] = img1[test_mask]
    err_masked = nrmse(img1, img2_masked, mask=test_mask)
    assert np.isclose(err_masked, 0.0)

    # Combination: scale-invariant with mask
    err_scale_inv_masked = nrmse(
        img1, img2_scaled, mask=test_mask, scale_invariant=True
    )
    assert np.isclose(err_scale_inv_masked, 0.0)


def test_psnr(
    test_images: tuple[np.ndarray, np.ndarray], test_mask: np.ndarray
) -> None:
    """Test PSNR: matches skimage, scale-invariant, and masked versions."""
    img1, img2 = test_images
    data_range = float(img1.max() - img1.min())

    # (i) Matches scikit-image implementation
    cubic_result = psnr(img1, img2, data_range=data_range, scale_invariant=False)
    skimage_result = skimage_metrics.peak_signal_noise_ratio(
        img1, img2, data_range=data_range
    )
    assert np.isclose(cubic_result, skimage_result)

    # (ii) Scale-invariant version
    img2_scaled = 2 * img1
    result_scale_inv = psnr(
        img1, img2_scaled, data_range=data_range, scale_invariant=True
    )
    # After scale-invariant normalization, arrays are algebraically identical but
    # float64 rounding produces MSE ≈ 7.7e-34, giving PSNR ≈ 341 dB instead of inf.
    # Threshold: PSNR must exceed what you'd get if MSE were at machine epsilon.
    min_psnr = -10 * np.log10(np.finfo(img1.dtype).eps)
    assert result_scale_inv > min_psnr

    # (iii) Masked version
    img2_masked = img2.copy()
    img2_masked[test_mask] = img1[test_mask]
    result_masked = psnr(img1, img2_masked, mask=test_mask, data_range=data_range)
    assert result_masked == float("inf")

    # Combination: scale-invariant with mask
    result_scale_inv_masked = psnr(
        img1, img2_scaled, mask=test_mask, scale_invariant=True, data_range=data_range
    )
    assert result_scale_inv_masked > min_psnr


def _torch_min_max(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Per-input min-max normalize to [0, 1] — reference for ``normalize='min_max'``."""
    x = x.astype(np.float64)
    rng = x.max() - x.min()
    if rng < eps:
        rng = eps
    return (x - x.min()) / rng


def test_nrmse_normalize_min_max_matches_torch_reference(
    test_images: tuple[np.ndarray, np.ndarray],
) -> None:
    """``normalize='min_max'`` applies per-input normalization before NRMSE."""
    img1, img2 = test_images
    a = _torch_min_max(img1)
    b = _torch_min_max(img2)
    expected = float(np.sqrt(np.mean((a - b) ** 2)))
    result = nrmse(img1, img2, normalize="min_max")
    assert np.isclose(result, expected, rtol=1e-10)


def test_psnr_normalize_min_max_matches_torch_reference(
    test_images: tuple[np.ndarray, np.ndarray],
) -> None:
    """``normalize='min_max'`` applies per-input normalization before PSNR."""
    img1, img2 = test_images
    a = _torch_min_max(img1)
    b = _torch_min_max(img2)
    mse = float(np.mean((a - b) ** 2))
    expected = -10.0 * np.log10(mse)
    result = psnr(img1, img2, normalize="min_max")
    assert np.isclose(result, expected, rtol=1e-10)


def test_nrmse_normalize_rejects_unknown_value() -> None:
    """Unknown normalize values raise ValueError."""
    a = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="not supported"):
        nrmse(a, a, normalize="zscore")


def test_psnr_normalize_rejects_unknown_value() -> None:
    """Unknown normalize values raise ValueError."""
    a = np.ones((4, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="not supported"):
        psnr(a, a, normalize="zscore")


def test_ssim_spatial_dims_2_4d_matches_2d_loop() -> None:
    """4-D ``[N,C,H,W]`` dispatch averages SSIM across the N*C slabs."""
    rng = np.random.default_rng(0)
    n, c, h, w = 2, 3, 16, 16
    a = rng.random((n, c, h, w)).astype(np.float32)
    b = a + 0.05 * rng.random((n, c, h, w)).astype(np.float32)
    batched = ssim(
        a, b, spatial_dims=2, win_size=3, gaussian_weights=False, data_range=1.0
    )
    manual = float(
        np.mean(
            [
                ssim(
                    a[i, j],
                    b[i, j],
                    win_size=3,
                    gaussian_weights=False,
                    data_range=1.0,
                )
                for i in range(n)
                for j in range(c)
            ]
        )
    )
    assert np.isclose(batched, manual, rtol=1e-10)


def test_ssim_spatial_dims_3_5d_matches_3d_loop() -> None:
    """5-D ``[N,C,D,H,W]`` dispatch averages SSIM across the N*C slabs."""
    rng = np.random.default_rng(1)
    n, c, d, h, w = 1, 1, 4, 16, 16
    a = rng.random((n, c, d, h, w)).astype(np.float32)
    b = a + 0.05 * rng.random((n, c, d, h, w)).astype(np.float32)
    batched = ssim(
        a, b, spatial_dims=3, win_size=3, gaussian_weights=False, data_range=1.0
    )
    manual = ssim(
        a[0, 0],
        b[0, 0],
        win_size=3,
        gaussian_weights=False,
        data_range=1.0,
    )
    assert np.isclose(batched, manual, rtol=1e-10)


def test_ssim_spatial_dims_rejects_bad_ndim() -> None:
    """Wrong ndim for the requested ``spatial_dims`` raises."""
    a = np.zeros((1, 1, 8, 8), dtype=np.float32)
    with pytest.raises(ValueError, match="expects ndim=5"):
        ssim(a, a, spatial_dims=3)


def test_ssim_spatial_dims_rejects_mask() -> None:
    """``mask`` is unsupported in the batched path."""
    a = np.zeros((1, 1, 8, 8), dtype=np.float32)
    m = np.ones((8, 8), dtype=bool)
    with pytest.raises(ValueError, match="mask is not supported"):
        ssim(a, a, spatial_dims=2, mask=m)


def test_torch_cuda_tensor_routes_to_cupy_not_numpy(gpu_available) -> None:
    """Torch CUDA tensor is converted to cupy (not numpy) by _canonicalize_torch.

    Previously the decorator used ``asnumpy`` unconditionally, causing a
    pointless GPU→CPU transfer for metrics that are cupy/cucim-capable.
    With CuPy available, a CUDA tensor should stay on GPU as a cupy view.
    """
    torch = pytest.importorskip("torch")
    if not gpu_available or not torch.cuda.is_available():
        pytest.skip("GPU not available")
    import cupy as cp

    from cubic.metrics.skimage_metrics import _canonicalize_torch

    t = torch.ones(4, 4, dtype=torch.float32).cuda()
    (out,) = _canonicalize_torch(t)
    assert isinstance(out, cp.ndarray), f"expected cupy.ndarray, got {type(out)}"
    assert out.data.ptr == t.data_ptr(), "expected zero-copy view (same GPU pointer)"


def test_ssim_accepts_cpu_torch_tensor() -> None:
    """A CPU torch.Tensor is auto-canonicalized to NumPy by the decorator."""
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    a_np = rng.random((1, 1, 4, 16, 16)).astype(np.float32)
    b_np = a_np + 0.05 * rng.random((1, 1, 4, 16, 16)).astype(np.float32)
    expected = ssim(
        a_np,
        b_np,
        spatial_dims=3,
        win_size=3,
        gaussian_weights=False,
        data_range=1.0,
    )
    a_t = torch.from_numpy(a_np)
    b_t = torch.from_numpy(b_np)
    actual = ssim(
        a_t,
        b_t,
        spatial_dims=3,
        win_size=3,
        gaussian_weights=False,
        data_range=1.0,
    )
    assert isinstance(actual, float)
    assert np.isclose(actual, expected, rtol=1e-10)


def test_ssim_torch_tensor_close_to_torch_ssim_reference() -> None:
    """Cubic 5-D SSIM is within 1e-3 of the reference torch SSIM kernel.

    The reference kernel uses a 3-D Gaussian with sigma=1.5, kernel=11,
    and replicate padding (a standard torch SSIM implementation). cubic
    uses skimage's gaussian filter with reflect-mode + cropping, which
    produces slightly different boundary handling — drift up to ~1e-3 is
    expected.
    """
    torch = pytest.importorskip("torch")
    import torch.nn.functional as F

    def _ref_torch_ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0):
        """Inline 3-D torch SSIM with Gaussian kernel and replicate padding."""
        k = 11
        sigma = 1.5
        coords = torch.arange(k, dtype=torch.float32) - k // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        kernel = (g[:, None, None] * g[None, :, None] * g[None, None, :]).reshape(
            1, 1, k, k, k
        )
        pad = k // 2
        pad_tuple = (pad,) * 6
        xp = F.pad(x, pad_tuple, mode="replicate")
        yp = F.pad(y, pad_tuple, mode="replicate")
        mu_x = F.conv3d(xp, kernel)
        mu_y = F.conv3d(yp, kernel)
        mu_xy = mu_x * mu_y
        sigma_x_sq = F.relu(F.conv3d(xp * xp, kernel) - mu_x * mu_x)
        sigma_y_sq = F.relu(F.conv3d(yp * yp, kernel) - mu_y * mu_y)
        sigma_xy = F.conv3d(xp * yp, kernel) - mu_xy
        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2
        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x * mu_x + mu_y * mu_y + c1) * (sigma_x_sq + sigma_y_sq + c2)
        )
        return float(ssim_map.mean())

    rng = np.random.default_rng(0)
    a = rng.random((1, 1, 16, 64, 64)).astype(np.float32)
    b = (a + 0.05 * rng.random((1, 1, 16, 64, 64))).astype(np.float32)
    expected = _ref_torch_ssim(torch.from_numpy(a), torch.from_numpy(b), data_range=1.0)
    actual = ssim(
        torch.from_numpy(a),
        torch.from_numpy(b),
        spatial_dims=3,
        data_range=1.0,
        gaussian_weights=True,
    )
    assert abs(actual - expected) < 1e-3, f"|actual={actual} - ref={expected}| ≥ 1e-3"


def test_ssim(
    test_images: tuple[np.ndarray, np.ndarray], test_mask: np.ndarray
) -> None:
    """Test SSIM: matches skimage, scale-invariant, and masked versions."""
    img1, img2 = test_images
    data_range = float(img1.max() - img1.min())

    # (i) Matches scikit-image implementation
    cubic_result = ssim(
        img1, img2, data_range=data_range, win_size=3, scale_invariant=False
    )
    skimage_result = skimage_metrics.structural_similarity(
        img1, img2, data_range=data_range, win_size=3
    )
    assert np.isclose(cubic_result, skimage_result)

    # (ii) Scale-invariant version
    img2_scaled = 2 * img1
    result_scale_inv = ssim(
        img1, img2_scaled, scale_invariant=True, data_range=data_range, win_size=3
    )
    assert np.isclose(result_scale_inv, 1.0)

    # (iii) Masked version
    img2_masked = img2.copy()
    img2_masked[test_mask] = img1[test_mask]
    result_masked = ssim(
        img1, img2_masked, mask=test_mask, data_range=data_range, win_size=3
    )
    assert np.isclose(result_masked, 1.0)

    # Masked version with full=True
    result_masked_full, ssim_map = ssim(
        img1, img2_masked, mask=test_mask, data_range=data_range, win_size=3, full=True
    )
    assert np.isclose(result_masked_full, 1.0)
    assert ssim_map.shape == img1.shape

    # Combination: scale-invariant with mask
    result_scale_inv_masked = ssim(
        img1,
        img2_scaled,
        mask=test_mask,
        scale_invariant=True,
        data_range=data_range,
        win_size=3,
    )
    assert np.isclose(result_scale_inv_masked, 1.0)


def test_masked_ssim_erosion_matches_gaussian_default_window() -> None:
    """Masked SSIM erodes by the window skimage actually used.

    With ``gaussian_weights=True`` and no explicit ``win_size``, skimage's
    default is ``2 * int(3.5 * 1.5 + 0.5) + 1 == 11``, but the erosion
    footprint was hard-coded to 7. Windows straddling the mask boundary
    therefore survived the erosion and pulled out-of-mask pixels into the
    "masked" mean.
    """
    rng = np.random.default_rng(0)
    a = rng.random((40, 40))
    b = a.copy()
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    b[~mask] = 999.0  # everything outside the mask is corrupted

    got = ssim(a, b, data_range=1.0, mask=mask, gaussian_weights=True)
    assert got == pytest.approx(1.0, abs=1e-9)

    # The valid-centre count must match the win=11 footprint (20-10 per axis).
    from cubic.skimage import morphology

    valid = morphology.erosion(mask, morphology.footprint_rectangle((11, 11)))
    assert int(valid.sum()) == 100


def test_masked_ssim_uniform_default_window_still_seven() -> None:
    """Without ``gaussian_weights``, skimage's default window stays 7."""
    rng = np.random.default_rng(1)
    a = rng.random((40, 40))
    b = a.copy()
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    b[~mask] = 999.0

    got = ssim(a, b, data_range=1.0, mask=mask, gaussian_weights=False)
    assert got == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("ndim", [2, 3])
def test_masked_ssim_footprint_is_not_deprecated(ndim: int) -> None:
    """The validity footprint uses ``footprint_rectangle``, not ``square``/``cube``.

    ``morphology.square``/``cube`` are deprecated in scikit-image 0.25 and
    removed in 0.27, so the masked path emitted a live ``FutureWarning``.
    ``footprint_rectangle`` produces a byte-identical footprint for these
    symmetric widths, which is what keeps the win_size=11 erosion correct.
    """
    rng = np.random.default_rng(10)
    shape = (24, 24) if ndim == 2 else (12, 12, 12)
    a = rng.random(shape)
    b = a + 0.05 * rng.standard_normal(shape)
    mask = np.zeros(shape, dtype=bool)
    mask[(slice(4, shape[0] - 4),) * ndim] = True

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        result = ssim(a, b, data_range=1.0, mask=mask, win_size=3)
    assert np.isfinite(result)


@pytest.mark.parametrize("width", [3, 7, 11])
def test_footprint_rectangle_matches_square_and_cube(width: int) -> None:
    """Pin the equivalence the migration relies on, for 2-D and 3-D."""
    from cubic.skimage import morphology

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        np.testing.assert_array_equal(
            morphology.footprint_rectangle((width, width)), morphology.square(width)
        )
        np.testing.assert_array_equal(
            morphology.footprint_rectangle((width,) * 3), morphology.cube(width)
        )


@pytest.mark.parametrize("sigma", [0.8, 1.5, 3.0])
def test_masked_ssim_erosion_follows_sigma(sigma: float) -> None:
    """The erosion window tracks ``sigma``, not a hard-coded 11.

    skimage derives ``win_size = 2 * int(3.5 * sigma + 0.5) + 1``, and ``sigma``
    reaches it through ``**kwargs``. Pinning 11 (the ``sigma=1.5`` value)
    under-erodes for any larger sigma: at ``sigma=3.0`` skimage's window is 23,
    so out-of-mask pixels leaked back into the "masked" mean and this fixture
    returned 0.2067 instead of the correct value.
    """
    from skimage.metrics import structural_similarity as sk_ssim

    from cubic.skimage import morphology

    rng = np.random.default_rng(1)
    a = rng.random((64, 64))
    b = a + 0.02 * rng.random((64, 64))
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True
    b[~mask] = 999.0  # corrupt everything outside the mask

    win = 2 * int(3.5 * sigma + 0.5) + 1
    _, ssim_map = sk_ssim(
        a, b, data_range=1.0, gaussian_weights=True, sigma=sigma, full=True
    )
    valid = morphology.erosion(mask, morphology.footprint_rectangle((win, win)))
    expected = float(ssim_map[valid].mean())

    got = ssim(a, b, data_range=1.0, mask=mask, gaussian_weights=True, sigma=sigma)
    assert got == pytest.approx(expected, abs=1e-9)


def test_masked_ssim_window_larger_than_mask_is_nan_not_a_number() -> None:
    """No valid centre must report NaN rather than averaging invalid windows."""
    rng = np.random.default_rng(1)
    a = rng.random((64, 64))
    b = a + 0.02 * rng.random((64, 64))
    mask = np.zeros((64, 64), dtype=bool)
    mask[16:48, 16:48] = True  # 32 px, smaller than sigma=5.0's 37 px window

    got = ssim(a, b, data_range=1.0, mask=mask, gaussian_weights=True, sigma=5.0)
    assert np.isnan(got)


def test_masked_ssim_rejects_gradient() -> None:
    """The masked path forces ``full=True``, so ``gradient`` cannot be honoured.

    Previously this reached skimage and raised
    ``ValueError: too many values to unpack (expected 2)`` from the 3-tuple.
    """
    rng = np.random.default_rng(0)
    a = rng.random((32, 32))
    b = a + 0.01
    mask = np.zeros((32, 32), dtype=bool)
    mask[8:24, 8:24] = True

    with pytest.raises(ValueError, match="gradient=True is not supported with mask"):
        ssim(a, b, data_range=1.0, mask=mask, gradient=True)


def test_masked_ssim_explicit_win_size_takes_precedence() -> None:
    """An explicit ``win_size`` still drives the erosion footprint."""
    rng = np.random.default_rng(2)
    a = rng.random((40, 40))
    b = a.copy()
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    b[~mask] = 999.0

    got = ssim(a, b, data_range=1.0, mask=mask, win_size=3)
    assert got == pytest.approx(1.0, abs=1e-9)


@pytest.mark.parametrize("flag", ["full", "gradient"])
def test_ssim_spatial_dims_rejects_tuple_returning_flags(flag: str) -> None:
    """``full``/``gradient`` are rejected in the batched path, not crashed on.

    ``structural_similarity`` returns a tuple for either flag, and the
    batched branch wrapped its result in ``float()``, so both raised an
    opaque ``TypeError: float() argument must be ... not 'tuple'``.
    """
    rng = np.random.default_rng(3)
    a = rng.random((1, 1, 8, 8))
    b = rng.random((1, 1, 8, 8))
    with pytest.raises(ValueError, match=f"{flag}=True is not supported"):
        ssim(a, b, spatial_dims=2, data_range=1.0, **{flag: True})


def test_ssim_spatial_dims_allows_falsy_flags() -> None:
    """Explicitly-false ``full``/``gradient`` are still accepted."""
    rng = np.random.default_rng(4)
    a = rng.random((1, 1, 8, 8))
    b = a + 0.01 * rng.random((1, 1, 8, 8))
    result = ssim(
        a, b, spatial_dims=2, data_range=1.0, win_size=3, full=False, gradient=False
    )
    assert isinstance(result, float)


@pytest.mark.parametrize("metric", [nrmse, psnr, ssim])
def test_mask_is_keyword_only(metric) -> None:
    """``mask`` cannot be supplied positionally.

    ``scale_invariant`` reads the mask from ``kwargs``, so a positionally
    supplied mask landed in ``*args``, the unmasked normalization branch
    ran, and ``alpha``/``range_param`` were computed over the whole image.
    """
    rng = np.random.default_rng(5)
    # Large enough that the default 7-wide SSIM window still leaves valid
    # centres after the mask is eroded.
    a = rng.random((16, 16))
    b = a + 0.1 * rng.random((16, 16))
    m = np.zeros((16, 16), dtype=bool)
    m[2:14, 2:14] = True
    # One positional per pre-mask parameter, then the mask.
    n_positional = {nrmse: 3, psnr: 2, ssim: 6}[metric]
    args = (None,) * n_positional
    with pytest.raises(TypeError, match="positional argument"):
        metric(a, b, *args, m)
    # The keyword form is what callers must use, and it works.
    assert np.isfinite(float(metric(a, b, mask=m, data_range=1.0)))


def test_pcc_mask_is_keyword_only() -> None:
    """``pcc`` also refuses a positional mask."""
    from cubic.metrics.pcc import pcc

    rng = np.random.default_rng(6)
    a = rng.random((8, 8))
    b = a + 0.1 * rng.random((8, 8))
    m = np.zeros((8, 8), dtype=bool)
    m[2:6, 2:6] = True
    with pytest.raises(TypeError, match="positional argument"):
        pcc(a, b, m)
    assert np.isfinite(pcc(a, b, mask=m))


@pytest.mark.parametrize("conflicting", [{"data_range": 1.0}, {"normalize": "min_max"}])
def test_nrmse_rejects_denominator_conflicts(
    test_images: tuple[np.ndarray, np.ndarray], conflicting: dict
) -> None:
    """``data_range`` used to silently win over ``normalization``.

    ``normalize="min_max"`` is the same hole one step removed: it sets
    ``data_range=1.0`` internally, which then short-circuits past
    ``normalization``.
    """
    img1, img2 = test_images
    with pytest.raises(ValueError, match="mutually exclusive"):
        nrmse(img1, img2, normalization="min-max", **conflicting)


def test_nrmse_rejects_normalization_with_scale_invariant(
    test_images: tuple[np.ndarray, np.ndarray],
) -> None:
    """``scale_invariant`` injects ``data_range``, so ``normalization`` conflicts.

    ``nrmse(a, b, normalization="min-max", scale_invariant=True)`` used to
    return the plain scale-invariant result, silently ignoring the
    requested normalization.
    """
    img1, img2 = test_images
    with pytest.raises(ValueError, match="incompatible with normalization"):
        nrmse(img1, img2, normalization="min-max", scale_invariant=True)


def test_nrmse_normalization_alone_still_works(
    test_images: tuple[np.ndarray, np.ndarray],
) -> None:
    """``normalization`` on its own is forwarded to skimage."""
    img1, img2 = test_images
    assert nrmse(img1, img2, normalization="min-max") == pytest.approx(
        skimage_metrics.normalized_root_mse(img1, img2, normalization="min-max")
    )


@pytest.mark.parametrize("metric", [nrmse, psnr, ssim])
def test_scale_invariant_rejects_constant_image_true(metric) -> None:
    """A constant ``image_true`` divides by a zero std instead of returning nan."""
    rng = np.random.default_rng(7)
    const = np.full((8, 8), 2.0)
    other = rng.random((8, 8))
    with pytest.raises(ValueError, match="image_true std"):
        metric(const, other, scale_invariant=True)


@pytest.mark.parametrize("metric", [nrmse, psnr, ssim])
def test_scale_invariant_rejects_constant_image_test(metric) -> None:
    """A constant ``image_test`` gives a zero-energy ``alpha`` denominator."""
    rng = np.random.default_rng(8)
    const = np.full((8, 8), 2.0)
    other = rng.random((8, 8))
    with pytest.raises(ValueError, match="image_test variance"):
        metric(other, const, scale_invariant=True)


def test_scale_invariant_rejects_constant_masked_region() -> None:
    """A mask selecting a constant region is rejected too."""
    rng = np.random.default_rng(9)
    a = rng.random((8, 8))
    b = rng.random((8, 8))
    a[2:6, 2:6] = 3.0  # constant inside the mask, varying outside
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    with pytest.raises(ValueError, match="masked image_true std"):
        psnr(a, b, mask=mask, scale_invariant=True)


def test_normalize_min_max_on_constant_input() -> None:
    """A constant input falls back to ``eps`` instead of dividing by zero.

    ``_min_max_to_unit`` maps a constant image to all zeros, so the two
    normalized inputs are identical: NRMSE is 0 and PSNR is infinite.
    """
    const = np.full((8, 8), 3.0)
    assert float(nrmse(const, const, normalize="min_max")) == 0.0
    assert float(psnr(const, const, normalize="min_max")) == float("inf")


@pytest.mark.parametrize("metric", [nrmse, psnr])
def test_normalize_min_max_constant_cupy_input(metric, gpu_available: bool) -> None:
    """``_min_max_to_unit`` must not rebuild the range as a CuPy 0-d array.

    ``type(rng)(eps)`` evaluates ``cupy.ndarray(1e-8)`` for a CuPy input,
    whose first constructor argument is a *shape*, so a constant GPU array
    raised ``TypeError: 'float' object cannot be interpreted as an integer``.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    from cubic.cuda import ascupy

    const = np.full((8, 8), 3.0)
    gpu = metric(ascupy(const), ascupy(const), normalize="min_max")
    cpu = metric(const, const, normalize="min_max")
    assert float(gpu) == pytest.approx(float(cpu))


@pytest.mark.parametrize("ndim", [2, 3])
def test_ssim_masked_gpu_matches_cpu(ndim: int, gpu_available: bool) -> None:
    """Masked SSIM runs on GPU and matches the CPU result.

    Regression for the masked path: the footprint builder receives only a
    shape (an int for the former ``square``/``cube``, a tuple for today's
    ``footprint_rectangle``), so the proxy sees no array argument and
    returns a host footprint; cuCIM's ``erosion`` rejected a NumPy
    footprint paired with a GPU mask (``ValueError: footprint must be
    either an ndarray or Sequence``). The footprint is now moved onto the
    mask's device first.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    from cubic.cuda import ascupy

    rng = np.random.default_rng(0)
    shape = (64, 64) if ndim == 2 else (32, 32, 32)
    img1 = rng.random(shape).astype(np.float32)
    img2 = img1 + 0.05 * rng.standard_normal(shape).astype(np.float32)
    mask = np.zeros(shape, dtype=bool)
    sl = (slice(8, shape[0] - 8),) * ndim
    mask[sl] = True

    cpu = ssim(img1, img2, data_range=1.0, mask=mask)
    gpu = float(ssim(ascupy(img1), ascupy(img2), data_range=1.0, mask=ascupy(mask)))
    assert np.isclose(cpu, gpu, atol=1e-4)
