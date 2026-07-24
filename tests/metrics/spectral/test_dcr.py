"""Tests for DCR (Decorrelation Analysis) resolution calculations."""

import numpy as np
import pytest

from cubic.metrics.spectral import dcr_curve, dcr_resolution, dcr_curve_3d_sectioned
from cubic.metrics.spectral.dcr import (
    _smooth_curve,
    _kc_to_resolution,
    _refinement_ranges,
)


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
    # With refine=True (default), curves include coarse + refined passes
    assert len(all_curves) >= 10, "Should have at least num_highpass curves"
    assert len(all_peaks) >= 10, "Should have at least num_highpass peaks"
    assert all_peaks.shape[1] == 2, "All peaks should be (N, 2) array"


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

    # More HP filters should produce more curves (with refinement, 2x)
    assert len(curves_minimal) >= 2, "num_highpass=2 should give at least 2 curves"
    assert len(curves_with_hp) >= 10, "num_highpass=10 should give at least 10 curves"
    assert len(curves_with_hp) > len(curves_minimal), "More HP filters = more curves"

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

    # A noise-dominated image may have no decorrelation peak at all, which is
    # reported as NaN ("no measurement"), never as inf or a negative number.
    assert np.isnan(res_noisy) or res_noisy > 0

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


def test_dcr_refine_returns_expected_format():
    """Refinement returns coarse + refined curves and peaks."""
    image = make_test_image_2d(shape=(64, 64))
    resolution, radii, all_curves, all_peaks = dcr_curve(
        image,
        spacing=0.1,
        num_radii=50,
        num_highpass=5,
        refine=True,
    )
    assert resolution > 0
    # Should have 2 * num_highpass curves (coarse + refined)
    assert len(all_curves) == 10
    assert all_peaks.shape == (10, 2)


def test_dcr_refine_deterministic():
    """Refined DCR gives same result for same input."""
    image = make_test_image_2d(shape=(64, 64), random_seed=42)
    res1, _, _, _ = dcr_curve(
        image,
        spacing=0.1,
        num_radii=50,
        num_highpass=5,
        refine=True,
    )
    res2, _, _, _ = dcr_curve(
        image,
        spacing=0.1,
        num_radii=50,
        num_highpass=5,
        refine=True,
    )
    assert np.isclose(res1, res2)


def test_dcr_refine_changes_resolution():
    """Refined result may differ from single-pass."""
    image = make_test_image_2d(shape=(128, 128), blob_sigma=3.0, noise_sigma=0.05)
    res_single, _, _, _ = dcr_curve(
        image,
        spacing=0.1,
        num_radii=50,
        num_highpass=10,
    )
    res_refined, _, _, _ = dcr_curve(
        image,
        spacing=0.1,
        num_radii=50,
        num_highpass=10,
        refine=True,
    )
    # Both should be positive and finite
    assert res_single > 0 and np.isfinite(res_single)
    assert res_refined > 0 and np.isfinite(res_refined)


# ---------- Regression tests ----------


def _band_limited_volume(
    shape: tuple[int, int, int] = (64, 64, 64),
    sigma: float = 2.0,
    seed: int = 3,
) -> np.ndarray:
    """Gaussian-smoothed noise: broadband content with a clear DCR cutoff."""
    from cubic.skimage import filters

    rng = np.random.default_rng(seed)
    volume = filters.gaussian(
        rng.normal(size=shape).astype(np.float32), sigma=sigma, preserve_range=True
    )
    return volume + 0.002 * rng.normal(size=shape).astype(np.float32)


@pytest.mark.parametrize("spacing", [None, 1.0, [0.2, 0.1, 0.1]])
def test_dcr_sectioned_and_legacy_use_the_same_units(spacing) -> None:
    """Sectioned and slice-based 3D DCR must agree to within a factor of 2.

    Regression guard for the k_max convention mismatch: the sectioned path used
    cycles per *image* (``min/max(n // 2)``, matching ``radial_edges``) while the
    2D path used cycles per *pixel* (0.5, from ``radial_k_grid``). Dividing by
    incommensurate units made the sectioned result off by ~n — on a 64³ volume
    sectioned reported 0.096/0.153 against 4.77/5.24 for the slice-based path.

    The tolerance is a factor of 2 rather than something tighter because the two
    paths measure genuinely different data (a 3D angular cone versus a single
    2D mid-slice) and quantize k_c to ``num_radii`` steps.
    """
    volume = _band_limited_volume()

    sectioned = dcr_resolution(volume, spacing=spacing, use_sectioned=True)
    legacy = dcr_resolution(volume, spacing=spacing, use_sectioned=False)

    for key in ("xy", "z"):
        assert np.isfinite(sectioned[key]), f"sectioned {key} is not finite"
        assert np.isfinite(legacy[key]), f"legacy {key} is not finite"
        ratio = sectioned[key] / legacy[key]
        assert 0.5 < ratio < 2.0, (
            f"{key}: sectioned={sectioned[key]:.4f} vs legacy={legacy[key]:.4f} "
            f"(ratio {ratio:.3f}) — the two paths disagree on units"
        )


def test_dcr_3d_spacing_one_matches_index_units() -> None:
    """``spacing=1.0`` must behave exactly like ``spacing=None``.

    ``radial_bin_id`` treated an all-ones spacing as index units while
    ``radial_edges`` stayed physical, so every non-DC voxel was clipped into the
    last bin and 3D DCR returned ``{'xy': inf, 'z': inf}``.
    """
    volume = _band_limited_volume()

    res_one = dcr_resolution(volume, spacing=1.0)
    res_none = dcr_resolution(volume, spacing=None)

    for key in ("xy", "z"):
        assert np.isfinite(res_one[key]), f"spacing=1.0 gave {res_one[key]} for {key}"
        assert res_one[key] == pytest.approx(res_none[key], rel=1e-9)


def test_dcr_2d_spacing_one_matches_index_units() -> None:
    """The 2D path must agree between ``spacing=1.0`` and ``spacing=None``."""
    image = make_test_image_2d(shape=(64, 64))
    res_one = dcr_resolution(image, spacing=1.0, num_radii=50, num_highpass=5)
    res_none = dcr_resolution(image, spacing=None, num_radii=50, num_highpass=5)
    assert np.isfinite(res_one)
    assert res_one == pytest.approx(res_none, rel=1e-9)


@pytest.mark.parametrize("angle_delta", [45, 30, 15])
def test_dcr_curve_3d_sectioned_supports_any_divisor_of_90(angle_delta: int) -> None:
    """Every valid angle_delta must return both sectors.

    Regression guard: all sectors were mapped onto the two keys ``"z"``/``"xy"``
    by ``"z" if center < 45 else "xy"``, so they overwrote each other and
    ``angle_delta=90`` raised ``KeyError: 'z'`` in step 5.
    """
    volume = _band_limited_volume(shape=(32, 32, 32))
    out = dcr_curve_3d_sectioned(
        volume,
        spacing=[0.2, 0.1, 0.1],
        angle_delta=angle_delta,
        num_radii=50,
        num_highpass=3,
    )

    assert set(out) == {"xy", "z"}
    for sector in ("xy", "z"):
        assert out[sector]["resolution"] > 0
        assert np.asarray(out[sector]["peaks"]).shape[1] == 2
        assert len(out[sector]["curves"]) > 0


@pytest.mark.parametrize("angle_delta", [90, 20, 0, -45])
def test_dcr_sectioned_rejects_invalid_angle_delta(angle_delta: int) -> None:
    """angle_delta must divide 90 and leave at least two polar sectors."""
    volume = _band_limited_volume(shape=(16, 16, 16))
    with pytest.raises(ValueError, match="angle_delta"):
        dcr_curve_3d_sectioned(
            volume, angle_delta=angle_delta, num_radii=20, num_highpass=2
        )


@pytest.mark.parametrize("window", [2, 3, 4, 5, 11])
def test_smooth_curve_accepts_small_windows(window: int) -> None:
    """Savitzky-Golay smoothing must not raise for windows of 3 or 4.

    ``win`` floors to 3 for both, and scipy requires ``polyorder <
    window_length``, so the hardcoded ``polyorder=3`` raised ValueError.
    """
    curve = np.linspace(0.0, 1.0, 30) ** 2
    smoothed = _smooth_curve(curve, window)
    assert smoothed.shape == curve.shape
    assert np.all(np.isfinite(smoothed))


def test_refinement_ranges_returns_ascending_sigma_range() -> None:
    """The refined sigma range must not come out inverted.

    ``_refinement_ranges`` indexes ``sigmas`` by peak index, so peaks and sigmas
    have to be aligned one-to-one and the sigmas ascending. The 3D sectioned
    path used to pass 1 + len(union) peaks alongside num_highpass + 1
    non-monotonic sigmas, which fell into the length fallback and returned
    ``(sigmas[0], sigmas[-1])`` from a descending array.
    """
    peaks = np.array([[0.0, 0.0], [0.2, 0.5], [0.6, 0.9], [0.3, 0.4]])
    sigmas = np.array([1.0, 2.0, 4.0, 8.0])

    result = _refinement_ranges(peaks, sigmas)
    assert result is not None
    r_min, r_max, sigma_min, sigma_max = result
    assert r_min < r_max
    assert sigma_min <= sigma_max


def test_refinement_ranges_rejects_misaligned_sigmas() -> None:
    """A peaks/sigmas length mismatch is a bug, not something to fall back on."""
    peaks = np.array([[0.2, 0.5], [0.6, 0.9], [0.3, 0.4]])
    with pytest.raises(ValueError, match="aligned"):
        _refinement_ranges(peaks, np.array([1.0, 2.0]))


def test_dcr_legacy_3d_forwards_refine(monkeypatch: pytest.MonkeyPatch) -> None:
    """``refine`` must reach the 2D curves in the slice-based 3D path.

    It was the only keyword not forwarded, so ``use_sectioned=False,
    refine=False`` still ran the refinement pass.
    """
    from cubic.metrics.spectral import dcr as dcr_mod

    seen: list[bool] = []
    real_dcr_curve = dcr_mod.dcr_curve

    def recording_dcr_curve(image, **kwargs):
        seen.append(kwargs["refine"])
        return real_dcr_curve(image, **kwargs)

    monkeypatch.setattr(dcr_mod, "dcr_curve", recording_dcr_curve)

    volume = make_test_image_3d(shape=(16, 32, 32))
    dcr_resolution(
        volume,
        spacing=0.1,
        num_radii=20,
        num_highpass=2,
        use_sectioned=False,
        refine=False,
    )

    assert seen == [False, False]


def _gpu_available() -> bool:
    from cubic.cuda import CUDAManager

    return CUDAManager().get_num_gpus() > 0


@pytest.mark.skipif(not _gpu_available(), reason="requires a CUDA GPU")
@pytest.mark.parametrize("use_sectioned", [True, False])
def test_dcr_3d_matches_between_devices(use_sectioned: bool) -> None:
    """3D DCR must give identical results on CPU and GPU."""
    from cubic.cuda import ascupy

    volume = _band_limited_volume(shape=(32, 32, 32))
    spacing = [0.2, 0.1, 0.1]

    res_cpu = dcr_resolution(volume, spacing=spacing, use_sectioned=use_sectioned)
    res_gpu = dcr_resolution(
        ascupy(volume), spacing=spacing, use_sectioned=use_sectioned
    )

    for key in ("xy", "z"):
        assert float(res_cpu[key]) == pytest.approx(float(res_gpu[key]), rel=1e-9)


def _axially_band_limited_volume(
    shape: tuple[int, int, int] = (32, 128, 128), seed: int = 11
) -> np.ndarray:
    """Noise smoothed more along Z than XY, like a real anisotropic PSF.

    Both cutoffs then sit inside the measured range, which is what makes the
    decorrelation curves peak instead of rising to the edge. Note DCR needs
    residual independent noise above the cutoff: smooth the volume much harder
    and both directions legitimately stop being measurable.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    vol = gaussian_filter(rng.random(shape).astype(np.float32), (2.0, 1.5, 1.5))
    return vol + 0.02 * rng.normal(size=shape).astype(np.float32)


def test_dcr_3d_axial_resolution_is_measurable_and_respects_floor() -> None:
    """3D DCR must report a finite Z above the ``2 * spacing_z`` floor.

    Guard against the axial sector silently losing its cutoff: ``z`` used to
    come back as ``inf`` for every spacing, which reads as a divide-by-zero
    escaping rather than a measurement.
    """
    volume = _axially_band_limited_volume()
    spacing = [0.2, 0.065, 0.065]

    res = dcr_resolution(volume, spacing=spacing)

    for key, floor in (("xy", 2 * spacing[2]), ("z", 2 * spacing[0])):
        assert np.isfinite(res[key]), f"{key} resolution is {res[key]}"
        assert res[key] >= floor, (
            f"{key} resolution {res[key]:.4f} is below the {floor} µm floor"
        )
    # Smoothing is 2x stronger along Z, so the axial estimate must be coarser.
    assert res["z"] > res["xy"]


def test_dcr_never_reports_inf_resolution() -> None:
    """A direction with no decorrelation peak must yield NaN, never inf.

    A volume barely smoothed along Z stays correlated towards the axial
    Nyquist, so the Z curve can rise to the edge of the measured range with no
    cutoff to find. The honest report is "no measurement" (NaN), never
    infinitely poor resolution from a divide-by-zero.

    Whether this particular curve has an interior peak depends on the
    scikit-image version (0.25 finds one here, 0.26 does not), so this asserts
    only the version-independent invariant. The NaN conversion itself is
    pinned deterministically by
    :func:`test_kc_to_resolution_returns_nan_without_a_cutoff`.
    """
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(0)
    volume = gaussian_filter(
        rng.random((16, 64, 64)).astype(np.float32), (0.5, 1.0, 1.0)
    )

    for spacing in (None, 1.0, 2.0):
        res = dcr_resolution(volume, spacing=spacing)
        for key in ("xy", "z"):
            assert not np.isinf(res[key]), f"{key} is inf at spacing={spacing}"
            # Either an honest NaN, or a physically plausible positive number.
            if not np.isnan(res[key]):
                assert res[key] > 0.0, (
                    f"{key} resolution {res[key]} at spacing={spacing} must be positive"
                )
        assert np.isfinite(res["xy"])


def test_kc_to_resolution_returns_nan_without_a_cutoff() -> None:
    """The conversion itself must not turn a missing cutoff into inf."""
    assert np.isnan(_kc_to_resolution(0.0, 0.5))
    assert np.isnan(_kc_to_resolution(0.5, 0.0))
    assert _kc_to_resolution(0.5, 0.5) == pytest.approx(4.0)
