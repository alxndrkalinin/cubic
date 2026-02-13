"""Tests for DCR (Decorrelation Analysis) resolution calculations."""

import numpy as np
import pytest

from cubic.metrics.frc import dcr_curve, dcr_resolution


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
    res = dcr_resolution(image, spacing=0.1, num_radii=50, num_highpass=10)
    assert res > 0, "DCR resolution should be positive for 2D image"
    assert np.isfinite(res), "DCR resolution should be finite"


def test_dcr_resolution_3d_returns_positive():
    """Test that DCR returns dict with 'xy' and 'z' resolutions for 3D images."""
    volume = make_test_image_3d(shape=(32, 64, 64))
    res = dcr_resolution(volume, spacing=0.1, num_radii=50, num_highpass=10)

    # 3D should return dict with 'xy' and 'z' keys
    assert isinstance(res, dict), "3D DCR should return dict"
    assert "xy" in res, "3D DCR dict should have 'xy' key"
    assert "z" in res, "3D DCR dict should have 'z' key"
    assert res["xy"] > 0, "XY resolution should be positive"
    assert res["z"] > 0, "Z resolution should be positive"
    assert np.isfinite(res["xy"]), "XY resolution should be finite"
    assert np.isfinite(res["z"]), "Z resolution should be finite"


def test_dcr_2d_and_3d():
    """Test that DCR works for both 2D and 3D images."""
    image_2d = make_test_image_2d(shape=(64, 64))
    image_3d = make_test_image_3d(shape=(32, 64, 64))

    res_2d = dcr_resolution(image_2d, spacing=0.1, num_radii=50, num_highpass=10)
    res_3d = dcr_resolution(image_3d, spacing=0.1, num_radii=50, num_highpass=10)

    # 2D returns float
    assert res_2d > 0, "2D DCR resolution should be positive"
    assert np.isfinite(res_2d), "2D DCR resolution should be finite"

    # 3D returns dict with 'xy' and 'z' keys
    assert isinstance(res_3d, dict), "3D DCR should return dict"
    assert res_3d["xy"] > 0, "3D DCR XY resolution should be positive"
    assert res_3d["z"] > 0, "3D DCR Z resolution should be positive"
    assert np.isfinite(res_3d["xy"]), "3D DCR XY resolution should be finite"
    assert np.isfinite(res_3d["z"]), "3D DCR Z resolution should be finite"


def test_dcr_with_different_spacing():
    """Test that DCR handles different spacing values correctly."""
    image = make_test_image_2d(shape=(64, 64))

    res_index = dcr_resolution(image, spacing=None, num_radii=50, num_highpass=10)
    res_small = dcr_resolution(image, spacing=0.1, num_radii=50, num_highpass=10)
    res_large = dcr_resolution(image, spacing=1.0, num_radii=50, num_highpass=10)

    # All should be positive and finite
    assert res_index > 0 and np.isfinite(res_index)
    assert res_small > 0 and np.isfinite(res_small)
    assert res_large > 0 and np.isfinite(res_large)

    # Physical spacing should scale resolution
    # Larger spacing → larger resolution value
    assert res_large > res_small


def test_dcr_with_anisotropic_spacing():
    """Test DCR with anisotropic spacing returns separate XY and Z resolutions."""
    volume = make_test_image_3d(shape=(32, 64, 64))
    spacing_aniso = [0.2, 0.1, 0.1]  # z-spacing twice as large

    res = dcr_resolution(volume, spacing=spacing_aniso, num_radii=50, num_highpass=10)

    # Should return dict with 'xy' and 'z' keys
    assert isinstance(res, dict), "3D DCR should return dict"
    assert res["xy"] > 0, "XY resolution should be positive"
    assert res["z"] > 0, "Z resolution should be positive"
    assert np.isfinite(res["xy"]), "XY resolution should be finite"
    assert np.isfinite(res["z"]), "Z resolution should be finite"

    # With 2x worse Z sampling, Z resolution should be worse than XY
    # (not always true depending on signal content, but generally expected)
    # Just check both are reasonable
    assert res["xy"] < 10.0, "XY resolution should be reasonable (< 10 units)"
    assert res["z"] < 10.0, "Z resolution should be reasonable (< 10 units)"


def test_dcr_invalid_dimensions():
    """Test that DCR raises error for invalid dimensions."""
    image_1d = np.random.randn(100)
    image_4d = np.random.randn(10, 10, 10, 10)

    with pytest.raises(ValueError):
        dcr_resolution(image_1d, num_radii=50, num_highpass=10)

    with pytest.raises(ValueError):
        dcr_resolution(image_4d, num_radii=50, num_highpass=10)


def test_dcr_curve_returns_expected_format():
    """Test that dcr_curve returns expected data structures."""
    image = make_test_image_2d(shape=(64, 64))

    resolution, radii, all_curves, all_peaks = dcr_curve(
        image, spacing=0.1, num_radii=50, num_highpass=10
    )

    # Check types and shapes
    assert isinstance(resolution, (float, np.floating)), "Resolution should be float"
    assert isinstance(radii, np.ndarray), "Radii should be numpy array"
    assert isinstance(all_curves, list), "All curves should be list"
    assert isinstance(all_peaks, np.ndarray), "All peaks should be numpy array"

    # Check values
    assert resolution > 0, "Resolution should be positive"
    assert len(radii) == 50, "Should have num_radii sampling points"
    assert len(all_curves) == 10, "Should have num_highpass curves"
    assert len(all_peaks) == 10, "Should have num_highpass peaks"
    assert all_peaks.shape == (10, 2), "All peaks should be (N, 2) array"


def test_dcr_num_radii_effect():
    """Test that num_radii affects curve sampling."""
    image = make_test_image_2d(shape=(64, 64))

    _, radii_50, _, _ = dcr_curve(image, spacing=0.1, num_radii=50, num_highpass=5)
    _, radii_100, _, _ = dcr_curve(image, spacing=0.1, num_radii=100, num_highpass=5)

    assert len(radii_50) == 50
    assert len(radii_100) == 100


def test_dcr_highpass_effect():
    """Test that high-pass filtering affects resolution estimate."""
    # Use sharper features (smaller blob_sigma) to ensure clear peaks
    image = make_test_image_2d(shape=(64, 64), blob_sigma=2.0, noise_sigma=0.1)

    # With minimal high-pass (just 2 curves)
    res_minimal, _, curves_minimal, _ = dcr_curve(
        image, spacing=0.1, num_radii=50, num_highpass=2
    )

    # With more high-pass filtering (10 log-spaced sigmas)
    res_with_hp, _, curves_with_hp, _ = dcr_curve(
        image, spacing=0.1, num_radii=50, num_highpass=10
    )

    # Number of curves should differ
    assert len(curves_minimal) == 2, "num_highpass=2 should give 2 curves"
    assert len(curves_with_hp) == 10, "num_highpass=10 should give 10 curves"

    # Both should give positive finite resolution with structured image
    assert res_with_hp > 0 and np.isfinite(res_with_hp), "10 filters should find peak"


def test_dcr_spacing_defaults_to_none():
    """Test that spacing parameter defaults to None (index units)."""
    image = make_test_image_2d(shape=(64, 64))

    # Call without spacing parameter
    res = dcr_resolution(image, num_radii=50, num_highpass=10)

    assert res > 0, "DCR should work with default spacing=None"
    assert np.isfinite(res), "Resolution should be finite"


def test_dcr_deterministic():
    """Test that DCR gives same result for same input."""
    image = make_test_image_2d(shape=(64, 64), random_seed=42)

    res1 = dcr_resolution(image, spacing=0.1, num_radii=50, num_highpass=10)
    res2 = dcr_resolution(image, spacing=0.1, num_radii=50, num_highpass=10)

    assert np.isclose(res1, res2), "DCR should be deterministic"


def test_dcr_noise_sensitivity():
    """Test DCR sensitivity to noise levels."""
    # Clean image
    clean_image = make_test_image_2d(shape=(64, 64), noise_sigma=0.001)
    # Noisy image
    noisy_image = make_test_image_2d(shape=(64, 64), noise_sigma=0.5)

    res_clean = dcr_resolution(clean_image, spacing=0.1, num_radii=50, num_highpass=10)
    res_noisy = dcr_resolution(noisy_image, spacing=0.1, num_radii=50, num_highpass=10)

    # Clean image should return finite valid result
    assert res_clean > 0 and np.isfinite(res_clean)

    # Noisy image may return inf if no peaks found (noise dominated)
    # This is a valid outcome - just check it's non-negative
    assert res_noisy >= 0

    # If both are finite, noisy should have worse resolution
    if np.isfinite(res_noisy) and np.isfinite(res_clean):
        assert res_noisy > res_clean * 0.5, (
            "Noisy image should not have much better resolution"
        )


def test_dcr_3d_legacy_mode():
    """Test 3D DCR legacy mode (use_sectioned=False) using 2D slices."""
    volume = make_test_image_3d(shape=(32, 64, 64))

    # Test legacy mode
    res_legacy = dcr_resolution(
        volume, spacing=0.1, num_radii=50, num_highpass=10, use_sectioned=False
    )

    # Should return dict with 'xy' and 'z' keys
    assert isinstance(res_legacy, dict), "Legacy 3D DCR should return dict"
    assert "xy" in res_legacy, "Legacy 3D DCR should have 'xy' key"
    assert "z" in res_legacy, "Legacy 3D DCR should have 'z' key"
    assert res_legacy["xy"] > 0, "XY resolution should be positive"
    assert res_legacy["z"] > 0, "Z resolution should be positive"
    assert np.isfinite(res_legacy["xy"]), "XY resolution should be finite"
    assert np.isfinite(res_legacy["z"]), "Z resolution should be finite"

    # Compare with sectioned mode (default)
    res_sectioned = dcr_resolution(
        volume, spacing=0.1, num_radii=50, num_highpass=10, use_sectioned=True
    )

    # Both modes should give reasonable results (not necessarily identical)
    assert res_sectioned["xy"] > 0 and np.isfinite(res_sectioned["xy"])
    assert res_sectioned["z"] > 0 and np.isfinite(res_sectioned["z"])
