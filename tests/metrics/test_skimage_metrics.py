"""Tests for image quality metrics."""

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


@pytest.mark.parametrize("ndim", [2, 3])
def test_ssim_masked_gpu_matches_cpu(ndim: int, gpu_available: bool) -> None:
    """Masked SSIM runs on GPU and matches the CPU result.

    Regression for the masked path: ``morphology.square``/``cube`` receive
    only an int, so the proxy returns a host footprint; cuCIM's ``erosion``
    rejected a NumPy footprint paired with a GPU mask
    (``ValueError: footprint must be either an ndarray or Sequence``).
    The footprint is now moved onto the mask's device first.
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
