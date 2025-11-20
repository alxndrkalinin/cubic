"""Tests for image quality metrics."""

import numpy as np
import pytest
from skimage import metrics as skimage_metrics

from cubic.metrics.skimage_metrics import psnr, ssim, nrmse


@pytest.fixture
def test_images() -> tuple[np.ndarray, np.ndarray]:
    """Create test images for metric comparison."""
    np.random.seed(42)
    img1 = np.random.rand(8, 8).astype(float)
    img2 = img1 + 0.1 * np.random.rand(8, 8).astype(float)
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
    assert result_scale_inv == float("inf")

    # (iii) Masked version
    img2_masked = img2.copy()
    img2_masked[test_mask] = img1[test_mask]
    result_masked = psnr(img1, img2_masked, mask=test_mask, data_range=data_range)
    assert result_masked == float("inf")

    # Combination: scale-invariant with mask
    result_scale_inv_masked = psnr(
        img1, img2_scaled, mask=test_mask, scale_invariant=True, data_range=data_range
    )
    assert result_scale_inv_masked == float("inf")


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
