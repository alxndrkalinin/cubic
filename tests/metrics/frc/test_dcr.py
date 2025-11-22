"""Tests for DCR (Decorrelation Analysis) resolution calculations."""

import numpy as np
import pytest

from cubic.metrics.frc import dcr_resolution


def make_test_image_2d(
    shape: tuple[int, int] = (64, 64),
    blob_sigma: float = 4.0,
    noise_sigma: float = 0.05,
    random_seed: int = 42,
) -> np.ndarray:
    """Generate a simple 2D test image with blobs and noise."""
    y, x = shape
    yy, xx = np.meshgrid(np.arange(y), np.arange(x), indexing="ij")

    # Create a few Gaussian blobs
    image = np.zeros(shape, dtype=float)
    centers = [(y // 3, x // 3), (2 * y // 3, 2 * x // 3)]
    for cy, cx in centers:
        dist2 = (yy - cy) ** 2 + (xx - cx) ** 2
        image += np.exp(-dist2 / (2 * blob_sigma**2))

    # Add noise
    rng = np.random.default_rng(seed=random_seed)
    image += rng.normal(scale=noise_sigma, size=shape)

    # Normalize to [0, 1]
    image -= image.min()
    if image.max() > 0:
        image /= image.max()

    return image


def make_test_image_3d(
    shape: tuple[int, int, int] = (32, 64, 64),
    blob_sigma: float = 4.0,
    noise_sigma: float = 0.05,
    random_seed: int = 42,
) -> np.ndarray:
    """Generate a simple 3D test image with blobs and noise."""
    z, y, x = shape
    zz, yy, xx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing="ij")

    # Create a few Gaussian blobs
    volume = np.zeros(shape, dtype=float)
    centers = [(z // 3, y // 2, x // 2), (2 * z // 3, y // 2, x // 2)]
    for cz, cy, cx in centers:
        dist2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        volume += np.exp(-dist2 / (2 * blob_sigma**2))

    # Add noise
    rng = np.random.default_rng(seed=random_seed)
    volume += rng.normal(scale=noise_sigma, size=shape)

    # Normalize to [0, 1]
    volume -= volume.min()
    if volume.max() > 0:
        volume /= volume.max()

    return volume


def test_dcr_resolution_2d_returns_positive():
    """Test that DCR returns positive resolution for 2D images."""
    image = make_test_image_2d(shape=(64, 64))
    res = dcr_resolution(image, bin_delta=3, spacing=0.1)
    assert res > 0, "DCR resolution should be positive for 2D image"
    assert np.isfinite(res), "DCR resolution should be finite"


def test_dcr_resolution_3d_returns_positive():
    """Test that DCR returns positive resolution for 3D images."""
    volume = make_test_image_3d(shape=(32, 64, 64))
    res = dcr_resolution(volume, bin_delta=3, spacing=0.1)
    assert res > 0, "DCR resolution should be positive for 3D image"
    assert np.isfinite(res), "DCR resolution should be finite"


def test_dcr_2d_and_3d():
    """Test that DCR works for both 2D and 3D images."""
    image_2d = make_test_image_2d(shape=(64, 64))
    image_3d = make_test_image_3d(shape=(32, 64, 64))

    res_2d = dcr_resolution(image_2d, bin_delta=3, spacing=0.1)
    res_3d = dcr_resolution(image_3d, bin_delta=3, spacing=0.1)

    assert res_2d > 0, "2D DCR resolution should be positive"
    assert res_3d > 0, "3D DCR resolution should be positive"
    assert np.isfinite(res_2d), "2D DCR resolution should be finite"
    assert np.isfinite(res_3d), "3D DCR resolution should be finite"


def test_dcr_with_different_spacing():
    """Test that DCR handles different spacing values correctly."""
    image = make_test_image_2d(shape=(64, 64))

    # Test with different spacing values
    res_index = dcr_resolution(image, bin_delta=3, spacing=None)  # Index units
    res_small = dcr_resolution(image, bin_delta=3, spacing=0.1)   # 0.1 units/pixel
    res_large = dcr_resolution(image, bin_delta=3, spacing=10.0)  # 10 units/pixel

    # All should be positive
    assert res_index > 0 and res_small > 0 and res_large > 0

    # Resolution scales with spacing (larger spacing -> coarser image -> larger resolution value)
    # Note: res_index is in index units, so not directly comparable to physical units
    assert res_large > res_small, "Resolution should scale with spacing in physical units"


def test_dcr_with_anisotropic_spacing():
    """Test that DCR handles anisotropic spacing correctly."""
    volume = make_test_image_3d(shape=(32, 64, 64))

    # Anisotropic spacing (different z vs xy)
    spacing_aniso = [0.3, 0.1, 0.1]  # z, y, x
    res = dcr_resolution(volume, bin_delta=3, spacing=spacing_aniso)

    assert res > 0, "DCR should work with anisotropic spacing"
    assert np.isfinite(res), "DCR resolution should be finite with anisotropic spacing"


def test_dcr_invalid_dimensions():
    """Test that DCR raises error for invalid dimensions."""
    # 1D image should fail
    image_1d = np.random.randn(64)
    with pytest.raises(ValueError, match="DCR requires 2D or 3D images"):
        dcr_resolution(image_1d)

    # 4D image should fail
    image_4d = np.random.randn(8, 16, 16, 16)
    with pytest.raises(ValueError, match="DCR requires 2D or 3D images"):
        dcr_resolution(image_4d)


def test_dcr_backend_consistency_2d():
    """Test that mask and hist backends produce similar results for 2D."""
    image = make_test_image_2d(shape=(64, 64))

    res_mask = dcr_resolution(image, bin_delta=3, spacing=0.1, backend="mask")
    res_hist = dcr_resolution(image, bin_delta=3, spacing=0.1, backend="hist")

    # Backends should produce similar results (within tolerance)
    assert np.isclose(res_mask, res_hist, rtol=0.05), (
        f"Backend mismatch: mask={res_mask:.3f}, hist={res_hist:.3f}"
    )


def test_dcr_backend_consistency_3d():
    """Test that mask and hist backends produce similar results for 3D.

    Both backends now use unified binning architecture:
    - radial_edges() computes bin boundaries in physical units when spacing provided
    - Iterators use fftfreq(n, d=spacing) to match histogram backend
    - Overflow bins (beyond Nyquist) excluded in both backends
    """
    volume = make_test_image_3d(shape=(32, 64, 64))

    res_mask = dcr_resolution(volume, bin_delta=3, spacing=0.1, backend="mask")
    res_hist = dcr_resolution(volume, bin_delta=3, spacing=0.1, backend="hist")

    # Backends should produce similar results (within tolerance)
    assert np.isclose(res_mask, res_hist, rtol=0.05), (
        f"Backend mismatch: mask={res_mask:.3f}, hist={res_hist:.3f}, "
        f"diff={(abs(res_mask - res_hist) / res_hist * 100):.1f}%"
    )


def test_dcr_calculate_function():
    """Test that calculate_dcr returns proper data structure."""
    from cubic.metrics.frc import calculate_dcr

    image = make_test_image_2d(shape=(64, 64))
    dcr_data = calculate_dcr(image, bin_delta=3, spacing=0.1, backend="mask")

    # Check that data structure is valid
    assert "correlation" in dir(dcr_data)
    assert "resolution" in dir(dcr_data)

    # Check correlation data
    corr = dcr_data.correlation["correlation"]
    freq = dcr_data.correlation["frequency"]
    points = dcr_data.correlation["points-x-bin"]

    assert len(corr) > 0, "Correlation array should not be empty"
    assert len(freq) > 0, "Frequency array should not be empty"
    assert len(corr) == len(freq), "Correlation and frequency arrays should match"
    assert len(points) > 0, "Points array should not be empty"

    # Check resolution data
    res = dcr_data.resolution["resolution"]
    spacing_val = dcr_data.resolution["spacing"]

    assert res > 0, "Resolution should be positive"
    assert spacing_val > 0, "Spacing should be positive"


def test_dcr_gpu_support():
    """Test that histogram backend works with GPU arrays."""
    pytest.importorskip("cupy")
    from cubic.cuda import CUDAManager, ascupy

    if CUDAManager().get_num_gpus() == 0:
        pytest.skip("No GPU available")

    image_cpu = make_test_image_2d(shape=(64, 64))
    image_gpu = ascupy(image_cpu)

    # Compute on GPU with histogram backend
    res_gpu = dcr_resolution(image_gpu, bin_delta=3, spacing=0.1, backend="hist")

    # Compute on CPU for comparison
    res_cpu = dcr_resolution(image_cpu, bin_delta=3, spacing=0.1, backend="hist")

    # Results should be very close
    assert np.isclose(res_gpu, res_cpu, rtol=0.01), (
        f"GPU/CPU mismatch: gpu={res_gpu:.3f}, cpu={res_cpu:.3f}"
    )


# ============================================================================
# DCR vs FRC Comparison Tests
# ============================================================================


def test_dcr_vs_frc_same_image_2d():
    """Compare DCR and FRC resolution estimates on same 2D image."""
    from cubic.metrics.frc import frc_resolution

    image = make_test_image_2d(shape=(64, 64))

    # DCR analyzes single image
    res_dcr = dcr_resolution(image, bin_delta=3, spacing=0.1)

    # FRC uses checkerboard splitting (single-image mode)
    res_frc = frc_resolution(image, bin_delta=3, spacing=0.1, backend="mask")

    # Both should return positive, finite values
    assert res_dcr > 0 and np.isfinite(res_dcr), "DCR resolution should be valid"
    assert res_frc > 0 and np.isfinite(res_frc), "FRC resolution should be valid"

    # Log comparison for analysis (don't assert equality - methods are different)
    print(f"\nDCR vs FRC comparison (2D):")
    print(f"  DCR: {res_dcr:.3f}")
    print(f"  FRC: {res_frc:.3f}")
    print(f"  Ratio (DCR/FRC): {res_dcr / res_frc:.3f}")


def test_dcr_vs_frc_curves_2d():
    """Compare DCR and FRC curve characteristics."""
    from cubic.metrics.frc import calculate_dcr, calculate_frc

    image = make_test_image_2d(shape=(64, 64))

    # Get full curves from both methods
    dcr_data = calculate_dcr(image, bin_delta=3, spacing=0.1, backend="mask")
    frc_data = calculate_frc(image, bin_delta=3, spacing=0.1, backend="mask")

    # Both should have correlation data
    dcr_curve = dcr_data.correlation["correlation"]
    frc_curve = frc_data.correlation["correlation"]

    # Check curve shapes
    assert len(dcr_curve) > 0, "DCR curve should not be empty"
    assert len(frc_curve) > 0, "FRC curve should not be empty"

    # DCR curve characteristics (after smoothing, may not be strictly monotonic)
    # Check that curve has expected behavior: starts high, generally trends upward
    assert dcr_curve[0] > 0, "DCR should start positive"
    assert np.max(dcr_curve) > np.min(dcr_curve), "DCR curve should have variation"

    # FRC should be monotonically decreasing (correlation decay)
    frc_diffs = np.diff(frc_curve[: len(frc_curve) // 2])  # Check first half

    # Most early FRC differences should be negative (decreasing)
    assert np.sum(frc_diffs < 0) > len(frc_diffs) * 0.5, (
        "FRC curve should be mostly decreasing in first half"
    )


def test_dcr_noise_sensitivity():
    """Test DCR behavior with different noise levels."""
    base_image = make_test_image_2d(shape=(64, 64), blob_sigma=4.0, noise_sigma=0.0)

    resolutions = []
    noise_levels = [0.0, 0.05, 0.10, 0.20]

    for noise in noise_levels:
        noisy = base_image + np.random.RandomState(42).normal(
            0, noise, base_image.shape
        )
        res = dcr_resolution(noisy, bin_delta=3, spacing=0.1)
        resolutions.append(res)
        print(f"Noise={noise:.2f}: DCR resolution={res:.3f}")

    # Resolution should generally degrade with noise
    # (though not strictly monotonic due to peak finding)
    assert all(r > 0 and np.isfinite(r) for r in resolutions), (
        "All resolutions should be valid"
    )


def test_dcr_empty_frequency_bins():
    """Test DCR handles images with limited frequency content."""
    # Create very low-frequency image (large blobs)
    image = make_test_image_2d(shape=(64, 64), blob_sigma=20.0)

    res = dcr_resolution(image, bin_delta=3, spacing=0.1)

    # Should still return valid resolution (even if poor)
    assert res > 0 and np.isfinite(res), "Should handle low-frequency images"


def test_dcr_curve_values_valid():
    """Test that DCR curve values are in valid range."""
    from cubic.metrics.frc import calculate_dcr

    image = make_test_image_2d(shape=(64, 64))
    dcr_data = calculate_dcr(image, bin_delta=3, spacing=0.1)

    curve = dcr_data.correlation["correlation"]

    # DCR values should be positive and less than 1
    assert np.all(curve >= 0), "DCR curve should be non-negative"
    assert np.all(curve <= 1.5), "DCR curve should not exceed ~1 (allow small margin)"
    assert np.all(np.isfinite(curve)), "DCR curve should have finite values"


def test_dcr_bin_delta_effect():
    """Test that bin_delta affects results consistently."""
    image = make_test_image_2d(shape=(64, 64))

    # Test with different bin widths
    res_fine = dcr_resolution(image, bin_delta=1, spacing=0.1)
    res_coarse = dcr_resolution(image, bin_delta=5, spacing=0.1)

    # Both should be valid
    assert res_fine > 0 and np.isfinite(res_fine)
    assert res_coarse > 0 and np.isfinite(res_coarse)

    # Results may differ significantly due to binning affecting peak finding
    # Just check that both are reasonable (within order of magnitude)
    ratio = max(res_fine, res_coarse) / min(res_fine, res_coarse)
    assert ratio < 10.0, (
        f"Resolution estimates differ too much (>10x): "
        f"fine={res_fine:.3f}, coarse={res_coarse:.3f}"
    )
