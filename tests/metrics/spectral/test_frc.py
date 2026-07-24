"""Tests for FRC and FSC resolution calculations."""

from typing import Any
from collections.abc import Sequence

import numpy as np
import pytest
from skimage import data

from cubic.cuda import CUDAManager, ascupy
from cubic.skimage import filters
from cubic.metrics.spectral import (
    calculate_frc,
    frc_resolution,
    fsc_resolution,
    five_crop_resolution,
    grid_crop_resolution,
)
from cubic.metrics.spectral.frc import (
    _fsc_hist_compute,
    preprocess_images,
    _calibration_factor,
    _normalization_spacing,
    _fsc_extract_resolution,
)
from cubic.metrics.spectral.radial import (
    _kmax_phys,
    radial_edges,
    radial_bin_id,
    radial_k_grid,
)
from cubic.metrics.spectral.analysis import (
    FourierCorrelationData,
    FourierCorrelationAnalysis,
    FourierCorrelationDataCollection,
    calculate_resolution_threshold_curve,
)


def _fractional_to_absolute(
    shape: tuple[int, int, int],
    frac_centers: Sequence[tuple[float, float, float]],
) -> list[tuple[int, int, int]]:
    """Convert fractional blob centres to absolute (z, y, x) voxel indices."""
    z, y, x = shape
    abs_centres: list[tuple[int, int, int]] = []
    for fz, fy, fx in frac_centers:
        fz_c, fy_c, fx_c = np.clip((fz, fy, fx), 0.0, 1.0)
        abs_centres.append(
            (
                round(fz_c * (z - 1)),
                round(fy_c * (y - 1)),
                round(fx_c * (x - 1)),
            )
        )
    return abs_centres


def make_fake_cells3d(
    shape: tuple[int, int, int] = (32, 64, 64),
    centres_frac: Sequence[tuple[float, float, float]] = (
        (0.33, 0.5, 0.5),
        (0.66, 0.5, 0.5),
    ),
    blob_sigma: float = 4.0,
    noise_sigma: float | None = 0.01,
    random_seed: int = 42,
) -> np.ndarray:
    """Generate a simple 3-D "cells" volume for testing."""
    z, y, x = shape
    zz, yy, xx = np.meshgrid(
        np.arange(z), np.arange(y), np.arange(x), indexing="ij", copy=False
    )
    volume = np.zeros(shape, dtype=float)

    for cz, cy, cx in _fractional_to_absolute(shape, centres_frac):
        dist2 = (zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2
        volume += np.exp(-dist2 / (2 * blob_sigma**2))

    # smooth the blobs a bit
    volume = filters.gaussian(volume, sigma=1.0, preserve_range=True)

    # optional Gaussian noise
    if noise_sigma:
        rng = np.random.default_rng(seed=random_seed)
        volume += rng.normal(scale=noise_sigma, size=shape)

    # rescale to [0, 1]
    volume -= volume.min()
    if volume.max() > 0:
        volume /= volume.max()

    return volume


@pytest.fixture(scope="module")
def cells_volume() -> tuple[np.ndarray, list[float]]:
    """Return single-channel cells3d volume and spacing or a synthetic fallback."""
    try:
        volume = data.cells3d()[:, 1]
        spacing = [0.29, 0.26, 0.26]
    except Exception:
        volume = make_fake_cells3d(shape=(32, 64, 64), random_seed=42)
        spacing = [1.0, 1.0, 1.0]
    return volume, spacing


def _gpu_available() -> bool:
    if not hasattr(_gpu_available, "_cached"):
        _gpu_available._cached = CUDAManager().get_num_gpus() > 0  # type: ignore[attr-defined]
    return _gpu_available._cached  # type: ignore[attr-defined]


def _middle_slice(volume: np.ndarray) -> np.ndarray:
    return volume[volume.shape[0] // 2]


def _assert_positive(result: Any) -> None:
    """Recursively assert that result contains positive values."""
    if isinstance(result, dict):
        for val in result.values():
            _assert_positive(val)
    else:
        assert float(result) > 0


def test_frc_all_backends_devices(
    cells_volume: tuple[np.ndarray, list[float]],
) -> None:
    """Test all backend-device combinations in a single test to minimize redundant calculations."""
    volume, spacing = cells_volume
    slice_cpu = _middle_slice(volume)
    xy_spacing = spacing[1:]  # [y, x] spacing

    # Prepare GPU image if available
    has_gpu = _gpu_available()
    slice_gpu = ascupy(slice_cpu) if has_gpu else None

    # Compute all combinations exactly once
    results = {}
    for backend in ["mask", "hist"]:
        # CPU
        results[(backend, "cpu")] = calculate_frc(
            slice_cpu,
            bin_delta=1,
            spacing=xy_spacing,
            backend=backend,
            disable_hamming=False,
        )

        # GPU
        if has_gpu:
            results[(backend, "gpu")] = calculate_frc(
                slice_gpu,
                bin_delta=1,
                spacing=xy_spacing,
                backend=backend,
                disable_hamming=False,
            )

    # Test 1: Each result should have valid structure and positive resolution
    for (backend, device), result in results.items():
        _assert_positive(result.resolution["resolution"])
        corr = result.correlation["correlation"]
        freq = result.correlation["frequency"]
        assert len(corr) > 0, f"Empty correlation for {backend}-{device}"
        assert len(freq) > 0, f"Empty frequency for {backend}-{device}"
        assert len(corr) == len(freq), f"Length mismatch for {backend}-{device}"

    # Test 2: Backend consistency on CPU (mask vs hist)
    mask_cpu = results[("mask", "cpu")]
    hist_cpu = results[("hist", "cpu")]
    corr_mask = mask_cpu.correlation["correlation"]
    corr_hist = hist_cpu.correlation["correlation"]
    freq_mask = mask_cpu.correlation["frequency"]
    freq_hist = hist_cpu.correlation["frequency"]
    min_len = min(len(corr_mask), len(corr_hist)) - 1

    assert np.allclose(
        corr_mask[:min_len],
        corr_hist[:min_len],
        atol=0.015,
        rtol=0.03,
    ), "FRC correlation should match between backends on CPU"

    assert np.allclose(
        freq_mask[:min_len],
        freq_hist[:min_len],
        atol=0.001,
        rtol=0.01,
    ), "FRC frequencies should match between backends on CPU"

    if not has_gpu:
        return  # Skip GPU tests if not available

    # Test 3: Backend consistency on GPU (mask vs hist)
    mask_gpu = results[("mask", "gpu")]
    hist_gpu = results[("hist", "gpu")]
    corr_mask_gpu = mask_gpu.correlation["correlation"]
    corr_hist_gpu = hist_gpu.correlation["correlation"]
    freq_mask_gpu = mask_gpu.correlation["frequency"]
    freq_hist_gpu = hist_gpu.correlation["frequency"]
    min_len_gpu = min(len(corr_mask_gpu), len(corr_hist_gpu)) - 1

    assert np.allclose(
        corr_mask_gpu[:min_len_gpu],
        corr_hist_gpu[:min_len_gpu],
        atol=0.015,
        rtol=0.03,
    ), "FRC correlation should match between backends on GPU"

    assert np.allclose(
        freq_mask_gpu[:min_len_gpu],
        freq_hist_gpu[:min_len_gpu],
        atol=0.001,
        rtol=0.01,
    ), "FRC frequencies should match between backends on GPU"

    # Test 4: Device consistency (CPU vs GPU for each backend)
    for backend in ["mask", "hist"]:
        cpu_result = results[(backend, "cpu")]
        gpu_result = results[(backend, "gpu")]

        corr_cpu = cpu_result.correlation["correlation"]
        corr_gpu = gpu_result.correlation["correlation"]
        freq_cpu = cpu_result.correlation["frequency"]
        freq_gpu = gpu_result.correlation["frequency"]

        min_len = min(len(corr_cpu), len(corr_gpu)) - 1

        assert np.allclose(
            corr_cpu[:min_len],
            corr_gpu[:min_len],
            atol=1e-5,
            rtol=1e-5,
        ), f"FRC correlation should match CPU/GPU for {backend} backend"

        assert np.allclose(
            freq_cpu[:min_len],
            freq_gpu[:min_len],
            atol=1e-6,
            rtol=1e-6,
        ), f"FRC frequencies should match CPU/GPU for {backend} backend"

    # Test 5: Cross-consistency (all combinations vs reference)
    ref = mask_cpu
    ref_corr = ref.correlation["correlation"]
    ref_freq = ref.correlation["frequency"]

    for (backend, device), result in results.items():
        if backend == "mask" and device == "cpu":
            continue  # Skip reference

        corr = result.correlation["correlation"]
        freq = result.correlation["frequency"]
        min_len = min(len(ref_corr), len(corr)) - 1

        # Same backend should have tighter tolerance
        if backend == "mask":
            atol_corr, rtol_corr = 1e-5, 1e-5
            atol_freq, rtol_freq = 1e-6, 1e-6
        else:
            # Different backend (hist vs mask) has relaxed tolerance
            atol_corr, rtol_corr = 0.015, 0.03
            atol_freq, rtol_freq = 0.001, 0.01

        assert np.allclose(
            ref_corr[:min_len],
            corr[:min_len],
            atol=atol_corr,
            rtol=rtol_corr,
        ), f"Correlation mismatch: mask-cpu vs {backend}-{device}"

        assert np.allclose(
            ref_freq[:min_len],
            freq[:min_len],
            atol=atol_freq,
            rtol=rtol_freq,
        ), f"Frequency mismatch: mask-cpu vs {backend}-{device}"


def test_calibration_factor() -> None:
    """Test the one-image FRC/FSC calibration factor function."""
    # The calibration factor should be > 1 for frequencies in the typical range
    # At the 1/7 threshold, typical crossing frequencies are 0.1-0.5
    for freq in [0.1, 0.2, 0.3, 0.4, 0.5]:
        factor = _calibration_factor(freq)
        # Correction factor should be roughly between 0.5 and 1.0
        # (dividing by it increases the resolution value)
        assert 0.4 < factor < 1.1, (
            f"Unexpected calibration factor {factor} at freq {freq}"
        )

    # At very low frequencies, the exponential term dominates
    factor_low = _calibration_factor(0.05)
    assert factor_low > 0, "Calibration factor should be positive"

    # The calibration curve is monotonically increasing
    factors = [_calibration_factor(f) for f in [0.1, 0.2, 0.3, 0.4]]
    assert all(factors[i] <= factors[i + 1] for i in range(len(factors) - 1)), (
        "Calibration factor should increase with frequency"
    )


def test_fsc_resolution_single_image(
    cells_volume: tuple[np.ndarray, list[float]],
) -> None:
    """Test FSC resolution with single-image mode (checkerboard split)."""
    volume, spacing = cells_volume

    # Single-image FSC should return positive resolution values
    result = fsc_resolution(
        volume,
        bin_delta=1,  # Match miplib paper methodology
        angle_delta=45,
        spacing=spacing,
        backend="hist",
    )

    assert "xy" in result, "FSC result should have 'xy' key"
    assert "z" in result, "FSC result should have 'z' key"

    # XY resolution should be positive and finite
    assert result["xy"] > 0, "XY resolution should be positive"
    assert np.isfinite(result["xy"]), "XY resolution should be finite"

    # Z resolution may be NaN if the analyzer cannot find a valid threshold
    # crossing (this is expected behavior - honest NaN is better than a
    # fallback value from an inconsistent threshold)
    if np.isfinite(result["z"]):
        assert result["z"] > 0, "Z resolution should be positive when finite"

    # Resolution should be in a reasonable range (in microns for cells3d)
    # cells3d has ~0.26 um XY spacing, typical XY resolution 0.3-1.0 um
    if spacing != [1.0, 1.0, 1.0]:  # Skip if using synthetic fallback
        assert 0.1 < result["xy"] < 5.0, (
            f"XY resolution {result['xy']} out of expected range"
        )
        if np.isfinite(result["z"]):
            assert 0.1 < result["z"] < 20.0, (
                f"Z resolution {result['z']} out of expected range"
            )


def test_fsc_all_sectors_processed(
    cells_volume: tuple[np.ndarray, list[float]],
) -> None:
    """Test that FSC reports both directions from the sectioned data.

    XY is read from the most XY-dominated sector as measured; Z from the highest
    sector below 45 degrees that crosses the threshold, projected onto the Z
    axis (see :func:`_fsc_extract_resolution`).
    """
    volume, spacing = cells_volume

    result = fsc_resolution(
        volume,
        bin_delta=1,
        angle_delta=15,
        spacing=spacing,
        backend="hist",
    )

    assert "xy" in result and "z" in result

    # XY should always be finite for structured test data
    assert np.isfinite(result["xy"]), "XY resolution should be finite"
    assert result["xy"] > 0, "XY resolution should be positive"

    # Z may or may not be finite depending on data, but if finite must be positive
    if np.isfinite(result["z"]):
        assert result["z"] > 0, "Z resolution should be positive when finite"


def _axially_band_limited_pair(
    shape: tuple[int, int, int] = (64, 64, 64),
    spacing: tuple[float, float, float] = (0.5, 0.19, 0.19),
    kz_cut: float = 0.5,
    kxy_cut: float = 1.5,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return two noisy views of a volume with an exact axial frequency cutoff.

    The object's Fourier support is an ideal box: zero beyond ``kz_cut`` along Z
    and beyond ``kxy_cut`` in XY, both in physical cycles per unit. Two
    independent noise realizations therefore correlate only inside that box, so
    the true axial resolution is exactly the period at the cutoff, ``1 / kz_cut``.
    """
    rng = np.random.default_rng(seed)
    kz = np.fft.fftfreq(shape[0], d=spacing[0])
    ky = np.fft.fftfreq(shape[1], d=spacing[1])
    kx = np.fft.fftfreq(shape[2], d=spacing[2])
    KZ, KY, KX = np.meshgrid(kz, ky, kx, indexing="ij")
    band = (np.abs(KZ) <= kz_cut) & (np.sqrt(KY**2 + KX**2) <= kxy_cut)

    obj = np.real(np.fft.ifftn(np.fft.fftn(rng.normal(size=shape)) * band))
    obj = (obj - obj.mean()) / obj.std()
    image1 = (obj + rng.normal(0, 1.0, shape)).astype(np.float32)
    image2 = (obj + rng.normal(0, 1.0, shape)).astype(np.float32)
    return image1, image2


def test_sectioned_fsc_projects_z_onto_the_axis() -> None:
    """The reported z is the sector's shell radius divided by cos(theta).

    A sector centred on theta measures ``|k| = k_z / cos(theta)``, not ``k_z``,
    and the cascade normally reports the 38-degree sector -- so the raw sector
    period understates the axial period by ``cos(theta)``. Against a volume with
    an exact axial cutoff at 0.5 cycles/um (true axial period 2.0 um), the
    unprojected number read 1.369 um (1.46x too fine); the old Koho eq. (5)
    multiplier ``1 + (spacing_z/spacing_xy - 1)|cos(theta)|`` read 3.13 um
    (1.56x too coarse).
    """
    spacing = (0.5, 0.19, 0.19)
    kz_cut = 0.5
    image1, image2 = _axially_band_limited_pair(spacing=spacing, kz_cut=kz_cut)

    result = fsc_resolution(
        image1, image2, spacing=list(spacing), angle_delta=15, use_max_nyquist=True
    )

    # Per-sector crossings, uncorrected, so the projection can be pinned exactly.
    fsc_data, max_freq = _fsc_hist_compute(
        image1,
        image2,
        bin_delta=1,
        angle_delta=15,
        spacing_list=list(spacing),
        exclude_axis_angle=0.0,
        use_max_nyquist=True,
        zero_padding=False,
        average=False,
    )
    spacing_eff = _normalization_spacing(max_freq)
    per_sector: dict[int, float] = {}
    for angle in sorted(fsc_data):
        coll = FourierCorrelationDataCollection()
        coll[angle] = fsc_data[angle]
        analyzed = FourierCorrelationAnalysis(
            coll,
            spacing_eff,
            resolution_threshold="fixed",
            threshold_value=0.143,
            curve_fit_type="smooth-spline",
        ).execute()
        res = analyzed[angle].resolution["resolution"]
        if np.isfinite(res) and res > 0:
            per_sector[angle] = float(res)

    # The cascade takes the highest sector below 45 degrees that crossed.
    reporting = max(a for a in per_sector if a < 45)
    expected = per_sector[reporting] / np.cos(np.deg2rad(reporting))
    assert result["z"] == pytest.approx(expected, rel=1e-6)
    # Guard against the projection silently becoming a no-op.
    assert expected > per_sector[reporting] * 1.05

    # Physical check: within the threshold-crossing bias of the true 2.0 um.
    # Unprojected this ratio is 0.685, and with the old z_factor it is 1.56.
    assert 0.78 <= result["z"] / (1.0 / kz_cut) <= 1.15


def _single_bin_sector(crosses: bool) -> FourierCorrelationData:
    """Return one sector's curve, either decaying through 0.143 or staying above."""
    freq = np.linspace(0, 1, 50)
    ds = FourierCorrelationData()
    ds.correlation["correlation"] = (
        np.maximum(1.0 - 2.0 * freq, -0.1) if crosses else np.full_like(freq, 0.9)
    )
    ds.correlation["frequency"] = freq
    ds.correlation["points-x-bin"] = np.ones(50)
    return ds


def test_sectioned_fsc_never_reports_z_from_an_xy_sector() -> None:
    """When no sector below 45 degrees crosses, z is nan rather than an XY number.

    Sectors at or above 45 degrees are XY-limited: their band edge is set by the
    in-plane cutoff, so it carries no axial information, and dividing by
    ``cos(82 degrees)`` would inflate it 7x. The cascade used to fall back to
    them, silently reporting an in-plane number as the axial resolution.
    """
    fsc_data = {
        angle: _single_bin_sector(crosses=angle >= 45)
        for angle in (8, 22, 38, 52, 68, 82)
    }

    with pytest.warns(RuntimeWarning, match="No FSC threshold crossing"):
        result = _fsc_extract_resolution(
            fsc_data,
            spacing_list=[0.5, 0.19, 0.19],
            max_freq=2.6316,
            single_image=False,
            resolution_threshold="fixed",
            threshold_value=0.143,
        )

    assert np.isnan(result["z"])
    # The XY sectors did cross, so xy is still reported.
    assert np.isfinite(result["xy"]) and result["xy"] > 0


def test_sectioned_fsc_rejects_a_single_sector() -> None:
    """angle_delta=90 gives one sector, which cannot separate Z from XY.

    Previously this produced a number: the lone 0-90 degree sector was reported
    as both xy and z.
    """
    image1, image2 = _axially_band_limited_pair(shape=(32, 32, 32))
    with pytest.raises(ValueError, match="required to separate XY from Z"):
        fsc_resolution(image1, image2, angle_delta=90)


def test_z_correction_at_boundary_angles() -> None:
    """Verify z_correction multiplier at theta=0 (Z) and theta=90 (XY).

    The z_multiplier formula (analysis.py) is:
        z_multiplier = 1 + (z_correction - 1) * |cos(angle)|

    At polar angle 0 (Z axis): cos(0)=1 -> multiplier = z_correction
    At polar angle 90 (XY plane): cos(90)=0 -> multiplier = 1.0
    """
    from cubic.metrics.spectral.analysis import FourierCorrelationAnalysis

    z_corr = 2.5

    # Build minimal single-bin data for two sectors: 0 (Z) and 90 (XY)
    for angle_deg, expected_mult in [(0, z_corr), (90, 1.0)]:
        coll = FourierCorrelationDataCollection()
        ds = FourierCorrelationData()
        # Monotonically decreasing correlation that crosses any threshold
        freq = np.linspace(0, 1, 50)
        corr = np.maximum(1.0 - 2.0 * freq, -0.1)
        ds.correlation["correlation"] = corr
        ds.correlation["frequency"] = freq
        ds.correlation["points-x-bin"] = np.ones(50)
        coll[angle_deg] = ds

        spacing = 0.1  # arbitrary
        analyzer = FourierCorrelationAnalysis(
            coll,
            spacing,
            resolution_threshold="fixed",
            threshold_value=1 / 7,
        )
        result = analyzer.execute(z_correction=z_corr)
        analyzed = result[angle_deg]

        # Extract resolution-point crossing frequency
        cross_freq = analyzed.resolution["resolution-point"][1]
        resolution = analyzed.resolution["resolution"]

        # Resolution = z_multiplier * (2 * spacing / cross_freq)
        # So z_multiplier = resolution * cross_freq / (2 * spacing)
        actual_mult = resolution * cross_freq / (2 * spacing)
        assert actual_mult == pytest.approx(expected_mult, rel=1e-3), (
            f"At angle={angle_deg}, expected multiplier={expected_mult}, "
            f"got {actual_mult}"
        )


# ---------- Binomial split FRC/FSC tests ----------


def _make_poisson_image_2d(
    shape: tuple[int, int] = (128, 128),
    signal_peak: float = 200.0,
    blob_sigma: float = 10.0,
    seed: int = 0,
) -> np.ndarray:
    """Create a 2D Poisson image with Gaussian blobs for FRC tests."""
    rng = np.random.default_rng(seed)
    yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), indexing="ij", copy=False
    )
    cy, cx = shape[0] // 2, shape[1] // 2
    rate = signal_peak * np.exp(
        -((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * blob_sigma**2)
    )
    rate += 5.0  # background
    return rng.poisson(rate).astype(np.float32)


def test_frc_binomial_basic() -> None:
    """Binomial split produces valid FRC with positive resolution."""
    img = _make_poisson_image_2d(seed=42)
    result = calculate_frc(
        img,
        split_type="binomial",
        backend="hist",
        rng=0,
    )
    assert result.resolution["resolution"] > 0
    assert np.isfinite(result.resolution["resolution"])
    # Should NOT have calibration correction applied
    # (We don't test the exact value, just that it's valid)


def test_frc_binomial_no_calibration() -> None:
    """Binomial split skips cutoff correction (differs from checkerboard)."""
    img = _make_poisson_image_2d(seed=1)
    result_binom = calculate_frc(img, split_type="binomial", backend="hist", rng=10)
    result_checker = calculate_frc(img, split_type="checkerboard", backend="hist")
    # Both should produce positive resolution
    assert result_binom.resolution["resolution"] > 0
    assert result_checker.resolution["resolution"] > 0
    # They will differ because checkerboard applies calibration and halves dims
    assert (
        result_binom.resolution["resolution"] != result_checker.resolution["resolution"]
    )


def test_frc_binomial_n_repeats() -> None:
    """n_repeats>1 produces correlation-std and resolution-std."""
    img = _make_poisson_image_2d(seed=2, signal_peak=500)
    result = calculate_frc(
        img,
        split_type="binomial",
        backend="hist",
        n_repeats=5,
        rng=42,
    )
    assert result.correlation["correlation-std"] is not None
    assert len(result.correlation["correlation-std"]) == len(
        result.correlation["correlation"]
    )
    assert result.resolution["resolution-std"] is not None
    assert result.resolution["resolution-std"] >= 0
    assert result.resolution["resolution"] > 0


def test_frc_binomial_1frc_vs_2frc() -> None:
    """1FRC(binomial) ≈ 2FRC(two independent Poisson draws) within tolerance.

    This is a statistical test: we generate two independent Poisson images
    from the same rate map, compute 2-image FRC, then compare with 1FRC
    from a single Poisson draw using binomial split.
    """
    rng = np.random.default_rng(100)
    shape = (128, 128)
    yy, xx = np.meshgrid(
        np.arange(shape[0]), np.arange(shape[1]), indexing="ij", copy=False
    )
    cy, cx = shape[0] // 2, shape[1] // 2
    rate = 300.0 * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * 15.0**2)) + 10.0

    # Two independent draws for gold standard 2FRC
    img_a = rng.poisson(rate).astype(np.float32)
    img_b = rng.poisson(rate).astype(np.float32)

    res_2frc = calculate_frc(img_a, img_b, backend="hist").resolution["resolution"]

    # Single image 1FRC with binomial split (averaged over repeats)
    img_single = rng.poisson(rate).astype(np.float32)
    res_1frc = calculate_frc(
        img_single,
        split_type="binomial",
        backend="hist",
        n_repeats=10,
        rng=42,
    ).resolution["resolution"]

    # They should be in the same ballpark (within 40% relative tolerance)
    # This is a statistical test, not exact
    assert abs(res_1frc - res_2frc) / res_2frc < 0.4, (
        f"1FRC={res_1frc:.4f} vs 2FRC={res_2frc:.4f}"
    )


def test_frc_binomial_readout_noise_plateau() -> None:
    """Readout noise correction reduces high-frequency FRC plateau bias.

    Flat field + Poisson + Gaussian read noise → without correction the 1FRC
    has a bias plateau at high-k. With correct readout_noise_rms the plateau
    should be closer to 0.
    """
    rng = np.random.default_rng(200)
    shape = (128, 128)
    # Flat field with moderate signal
    signal = 100.0
    readout_sigma = 5.0
    img = rng.poisson(signal, size=shape).astype(np.float64)
    img += rng.normal(0, readout_sigma, size=shape)
    img = np.clip(img, 0, None).astype(np.float32)

    # FRC without readout correction
    result_no_corr = calculate_frc(
        img,
        split_type="binomial",
        backend="hist",
        n_repeats=5,
        rng=10,
    )
    curve_no_corr = result_no_corr.correlation["correlation"]

    # FRC with readout correction
    result_with_corr = calculate_frc(
        img,
        split_type="binomial",
        backend="hist",
        readout_noise_rms=readout_sigma,
        n_repeats=5,
        rng=10,
    )
    curve_with_corr = result_with_corr.correlation["correlation"]

    # High-frequency tail (last 25% of bins): corrected should be lower
    n = len(curve_no_corr)
    tail_no = np.mean(np.abs(curve_no_corr[3 * n // 4 :]))
    tail_with = np.mean(np.abs(curve_with_corr[3 * n // 4 :]))
    assert tail_with <= tail_no, (
        f"Corrected high-freq plateau {tail_with:.3f} should be <= "
        f"uncorrected {tail_no:.3f}"
    )


def test_fsc_binomial_basic() -> None:
    """FSC with binomial split returns valid XY/Z resolution."""
    rng = np.random.default_rng(42)
    shape = (64, 64, 64)
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
        copy=False,
    )
    rate = 300.0 * np.exp(
        -(
            (zz - 32) ** 2 / (2 * 8.0**2)
            + (yy - 32) ** 2 / (2 * 8.0**2)
            + (xx - 32) ** 2 / (2 * 8.0**2)
        )
    )
    rate += 20.0
    vol = rng.poisson(rate).astype(np.float32)

    result = fsc_resolution(
        vol,
        spacing=[0.5, 0.5, 0.5],
        split_type="binomial",
        counts_mode="counts",
        backend="hist",
        angle_delta=45,
        rng=0,
    )
    assert "xy" in result
    assert "z" in result
    # At least XY should produce a valid resolution
    assert result["xy"] > 0
    assert np.isfinite(result["xy"])


def test_frc_binomial_camera_calibration() -> None:
    """Camera calibration params (gain, offset, readout_noise_rms) flow through frc_resolution."""
    rng = np.random.default_rng(77)
    shape = (128, 128)
    gain = 2.0
    offset = 100.0
    readout_sigma = 3.0

    # Simulate raw camera data: electrons → ADU with gain/offset
    rate = (
        150.0
        * np.exp(
            -(
                (np.arange(shape[0])[:, None] - 64) ** 2
                + (np.arange(shape[1])[None, :] - 64) ** 2
            )
            / (2 * 12.0**2)
        )
        + 10.0
    )
    electrons = rng.poisson(rate)
    electrons += rng.normal(0, readout_sigma, size=shape).astype(int)
    img_adu = (electrons * gain + offset).astype(np.float32)

    result = frc_resolution(
        img_adu,
        split_type="binomial",
        counts_mode="counts",
        gain=gain,
        offset=offset,
        readout_noise_rms=readout_sigma,
        backend="hist",
        rng=42,
    )
    assert result > 0
    assert np.isfinite(result)


def test_counts_mode_warns_without_binomial() -> None:
    """counts_mode='poisson_thinning' warns when split_type='checkerboard'."""
    img = _make_poisson_image_2d(seed=5)
    with pytest.warns(UserWarning, match="counts_mode"):
        calculate_frc(img, split_type="checkerboard", counts_mode="poisson_thinning")


def test_checkerboard_default_unchanged(
    cells_volume: tuple[np.ndarray, list[float]],
) -> None:
    """Default checkerboard behavior is unchanged by the new parameters."""
    volume, spacing = cells_volume
    slice_2d = _middle_slice(volume)
    xy_spacing = spacing[1:]

    # Default call (no new params)
    result_default = calculate_frc(
        slice_2d, bin_delta=1, spacing=xy_spacing, backend="hist"
    )
    # Explicit checkerboard call
    result_checker = calculate_frc(
        slice_2d,
        bin_delta=1,
        spacing=xy_spacing,
        backend="hist",
        split_type="checkerboard",
    )
    np.testing.assert_allclose(
        result_default.correlation["correlation"],
        result_checker.correlation["correlation"],
        rtol=1e-10,
    )
    assert result_default.resolution["resolution"] == pytest.approx(
        result_checker.resolution["resolution"], rel=1e-10
    )


def _low_freq_power(image: np.ndarray, max_k: float = 10.0) -> float:
    """Sum |FFT|^2 over the low-frequency annulus 0.5 < k < max_k."""
    f = np.fft.fftn(image - image.mean())
    h, w = f.shape
    yy, xx = np.meshgrid(np.fft.fftfreq(h) * h, np.fft.fftfreq(w) * w, indexing="ij")
    r = np.sqrt(yy**2 + xx**2)
    mask = (r > 0.5) & (r < max_k)
    return float(np.sum(np.abs(f[mask]) ** 2))


def test_preprocess_images_centers_before_windowing_high_dc_offset() -> None:
    """High-DC-offset input must not bleed into low-frequency bins via the taper.

    Regression guard: if mean-subtraction is moved back to after Hamming
    windowing, the DC offset μ multiplied by the non-zero-mean taper
    creates a μ·(w(x) − mean(w)) low-frequency artifact that survives the
    later image − image.mean() DC removal. This test pins the ordering by
    comparing the low-frequency FFT power against a centered-input baseline.
    """
    rng = np.random.default_rng(0)
    h, w = 256, 256
    structure = rng.gamma(2.0, 850.0, (h, w)).astype(np.float32)
    offset_image = (structure + 2500.0).astype(np.float32)
    centered_image = structure.astype(np.float32)

    p_offset, _ = preprocess_images(offset_image.copy(), zero_padding=False)
    p_centered, _ = preprocess_images(centered_image.copy(), zero_padding=False)

    # The two preprocessing outputs must have indistinguishable low-frequency
    # power, since they differ only by a global constant — i.e. centering
    # must happen before the taper is applied.
    ratio = _low_freq_power(p_offset) / _low_freq_power(p_centered)
    assert 0.5 < ratio < 2.0, (
        f"DC-offset input has {ratio:.1f}x the low-freq power of centered "
        f"input — preprocess_images must mean-center before Hamming windowing"
    )


def test_preprocess_images_centers_before_padding_non_cubic_input() -> None:
    """Non-cubic zero-padded input must not leak the DC offset through the pad ring.

    Regression guard for the second ordering invariant: mean-subtraction
    happens before pad_image_to_cube so the zero ring stays at zero
    instead of carrying -diluted_mean into the Hamming taper edges.
    """
    rng = np.random.default_rng(0)
    structure = rng.gamma(2.0, 850.0, (200, 256)).astype(np.float32)
    offset_image = (structure + 2500.0).astype(np.float32)

    # Pad to (256, 256) cube via zero_padding=True.
    p_offset, _ = preprocess_images(offset_image.copy(), zero_padding=True)
    # A square (256, 256) image of the centered structure is the cleanest
    # reachable reference (no pad ring at all).
    structure_square = rng.gamma(2.0, 850.0, (256, 256)).astype(np.float32)
    p_ref, _ = preprocess_images(structure_square.copy(), zero_padding=False)

    ratio = _low_freq_power(p_offset) / _low_freq_power(p_ref)
    assert 0.5 < ratio < 2.0, (
        f"Padded DC-offset input has {ratio:.1f}x the low-freq power of "
        f"centered reference — preprocess_images must mean-center before "
        f"pad_image_to_cube"
    )


def test_resolution_returns_nan_when_curve_below_threshold() -> None:
    """FSC/FRC curves that never start above the threshold must yield NaN.

    Regression guard: predictions with near-zero correlation to GT
    produce curves that sit below 0.143 from the lowest measured
    frequency. ``first_guess`` previously returned ``x[0]`` in that
    case, feeding an unbounded ``fmin`` that wandered into extrapolated
    territory and produced absurd roots (negative or near-zero),
    yielding resolutions of millions of µm via ``2 * spacing / root``.
    The fix returns ``None`` (→ NaN) when the curve starts already
    below the threshold and also rejects fmin roots outside the data's
    frequency range.
    """
    freqs = np.linspace(0.05, 1.0, 50)

    # Case 1: below-threshold-everywhere curve (zero-correlation prediction).
    # Deterministic constant well below 0.143 so the scenario is invariant
    # across NumPy/RNG versions.
    below = FourierCorrelationData()
    below.correlation["frequency"] = freqs
    below.correlation["correlation"] = np.full(50, 0.02)
    below.correlation["points-x-bin"] = np.full(50, 100.0)
    coll = FourierCorrelationDataCollection()
    coll[0] = below
    res_below = (
        FourierCorrelationAnalysis(
            coll,
            spacing=0.108,
            resolution_threshold="fixed",
            threshold_value=0.143,
            curve_fit_type="smooth-spline",
        )
        .execute(z_correction=1.0)[0]
        .resolution["resolution"]
    )
    assert np.isnan(res_below), (
        f"Below-threshold curve must return NaN, got {res_below} "
        "(unbounded fmin previously produced millions-of-µm resolutions)"
    )

    # Case 2: above-threshold-everywhere curve (perfect prediction).
    above = FourierCorrelationData()
    above.correlation["frequency"] = freqs
    above.correlation["correlation"] = np.full(50, 0.9)
    above.correlation["points-x-bin"] = np.full(50, 100.0)
    coll2 = FourierCorrelationDataCollection()
    coll2[0] = above
    res_above = (
        FourierCorrelationAnalysis(
            coll2,
            spacing=0.108,
            resolution_threshold="fixed",
            threshold_value=0.143,
            curve_fit_type="smooth-spline",
        )
        .execute(z_correction=1.0)[0]
        .resolution["resolution"]
    )
    assert np.isnan(res_above), (
        f"Above-threshold curve must return NaN, got {res_above}"
    )

    # Case 3: legitimate crossing — must still produce a finite resolution.
    legit = FourierCorrelationData()
    legit.correlation["frequency"] = freqs
    legit.correlation["correlation"] = 0.9 * np.exp(-3 * freqs) + 0.02
    legit.correlation["points-x-bin"] = np.full(50, 100.0)
    coll3 = FourierCorrelationDataCollection()
    coll3[0] = legit
    res_legit = (
        FourierCorrelationAnalysis(
            coll3,
            spacing=0.108,
            resolution_threshold="fixed",
            threshold_value=0.143,
            curve_fit_type="smooth-spline",
        )
        .execute(z_correction=1.0)[0]
        .resolution["resolution"]
    )
    assert np.isfinite(res_legit) and 0 < res_legit < 100, (
        f"Legitimate crossing must yield finite reasonable resolution, got {res_legit}"
    )


# ---------- Regression tests ----------


def _band_limited_volume(
    shape: tuple[int, int, int] = (16, 300, 300),
    sigma: tuple[float, float, float] = (2.0, 4.0, 4.0),
    seed: int = 5,
) -> np.ndarray:
    """Smoothed Poisson noise: broadband structure with a clear FRC crossing."""
    rng = np.random.default_rng(seed)
    vol = filters.gaussian(
        rng.normal(size=shape).astype(np.float32), sigma=sigma, preserve_range=True
    )
    vol = (vol - vol.min()) / (vol.max() - vol.min()) * 200.0
    return rng.poisson(vol).astype(np.float32)


@pytest.mark.parametrize("crop_fn", [five_crop_resolution, grid_crop_resolution])
def test_crop_resolution_returns_per_slice_floats(crop_fn: Any) -> None:
    """Tiled resolution helpers must return scalars, not crash on 2D slices.

    Regression guard: both helpers fed 2D slices (``loc_image.max(0)`` and
    ``loc_image[i]``) to the 3D-only ``fsc_resolution``, which raised
    ``IndexError: tuple index out of range`` while unpacking a third axis, and
    would then have handed dicts to ``np.median(..., axis=0)``.
    """
    volume = _band_limited_volume()
    result = crop_fn(volume, spacing=(0.2, 0.065, 0.065), crop_size=128)

    assert set(result) == {"max_projection", "xy", "xz"}
    # One aggregated value for the projection, one per Z plane for the slices.
    assert np.ndim(result["max_projection"]) == 0
    assert np.asarray(result["xy"]).shape == (volume.shape[0],)
    assert np.asarray(result["xz"]).shape == (volume.shape[0],)

    assert np.isfinite(result["max_projection"]) and result["max_projection"] > 0
    for key, floor in (("xy", 2 * 0.065), ("xz", 2 * 0.2)):
        values = np.asarray(result[key], dtype=float)
        assert np.all(np.isfinite(values)), f"{key} has non-finite entries: {values}"
        # A raw crossing gives 1 / (f_c * kmax) >= 2 * spacing, but the
        # single-image checkerboard estimate is then divided by the empirical
        # calibration factor, which exceeds 1 for crossings at the band edge.
        assert np.all(values >= floor / _calibration_factor(1.0)), (
            f"{key} values implausibly below the {floor} um floor: {values}"
        )
        assert np.median(values) >= floor, f"{key} median below {floor} um: {values}"


@pytest.mark.parametrize("crop_fn", [five_crop_resolution, grid_crop_resolution])
def test_crop_resolution_measures_xz_slices_unpadded(
    crop_fn: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """XZ slices must reach the FRC at their native rectangular shape.

    They used to be reflect-padded along Z up to ``crop_size``, replicating the
    real data 2-16x; both checkerboard halves then stayed near-identical and the
    FRC curve never descended through the threshold, so ``xz`` came back NaN for
    most slices and inflated ~4x for the rest.
    """
    from cubic.metrics.spectral import frc as frc_mod

    seen: list[tuple[int, ...]] = []
    real_frc_resolution = frc_mod.frc_resolution

    def recording_frc_resolution(image: np.ndarray, *args: Any, **kwargs: Any) -> float:
        seen.append(image.shape)
        return real_frc_resolution(image, *args, **kwargs)

    monkeypatch.setattr(frc_mod, "frc_resolution", recording_frc_resolution)

    n_z = 32
    volume = _band_limited_volume(shape=(n_z, 300, 300))
    crop_fn(volume, spacing=(0.2, 0.065, 0.065), crop_size=128)

    # XY planes and the max projection are square; XZ slices keep (n_z, 128).
    assert (n_z, 128) in seen, f"no XZ slice was measured unpadded, saw {set(seen)}"
    assert all(shape in {(128, 128), (n_z, 128)} for shape in seen), (
        f"unexpected slice shapes: {sorted(set(seen))}"
    )


@pytest.mark.parametrize("n_z", [8, 32])
@pytest.mark.parametrize("crop_fn", [five_crop_resolution, grid_crop_resolution])
def test_crop_resolution_xz_is_measurable(crop_fn: Any, n_z: int) -> None:
    """Most XZ slices must yield a finite resolution above the 2*spacing_z floor.

    Guards the padding removal at several Z depths: with reflect padding the
    replication factor was crop_size / n_z, so the shallower the stack the more
    thoroughly the FRC curve was flattened — per-slice XZ came back NaN for
    almost every slice and the plain-median aggregate was NaN too. A shape-only
    check cannot catch that; this asserts the values themselves.
    """
    spacing = (0.2, 0.065, 0.065)
    volume = _band_limited_volume(shape=(n_z, 300, 300))

    per_slice = np.asarray(
        crop_fn(volume, spacing=spacing, crop_size=128, aggregate=None)["xz"],
        dtype=float,
    )
    finite = np.isfinite(per_slice)
    assert finite.mean() > 0.5, (
        f"only {finite.sum()}/{finite.size} XZ slices are measurable at Z={n_z}"
    )
    floor = 2 * spacing[0]
    assert np.median(per_slice[finite]) >= floor

    # The default nan-aware aggregate must not be poisoned by the NaN slices.
    aggregated = np.asarray(
        crop_fn(volume, spacing=spacing, crop_size=128)["xz"], float
    )
    assert np.all(np.isfinite(aggregated))
    assert np.all(aggregated >= floor / _calibration_factor(1.0))
    assert np.median(aggregated) >= floor


def _anisotropic_volume() -> tuple[np.ndarray, list[float]]:
    """Anisotropic blob volume with spacing whose Z Nyquist is the limit."""
    rng = np.random.default_rng(0)
    zz, yy, xx = np.meshgrid(
        np.arange(32), np.arange(128), np.arange(128), indexing="ij", copy=False
    )
    vol = np.exp(
        -(
            (zz - 16) ** 2 / (2 * 4.0**2)
            + (yy - 64) ** 2 / (2 * 8.0**2)
            + (xx - 64) ** 2 / (2 * 8.0**2)
        )
    )
    vol += 0.01 * rng.normal(size=vol.shape)
    return vol.astype(np.float32), [0.2, 0.065, 0.065]


@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"use_max_nyquist": True},
        {"resample_isotropic": True},
    ],
)
def test_fsc_resolution_respects_two_pixel_floor(kwargs: dict[str, Any]) -> None:
    """Sectioned FSC must never report a resolution below the sampling limit.

    Regression guard for the Nyquist mismatch: ``_calculate_fsc_sectioned_hist``
    normalizes the frequency axis by the *minimum* Nyquist (Z), but the analyzer
    was handed the XY spacing, which it inverts as ``2 * spacing / root``. On
    (32, 128, 128) at [0.2, 0.065, 0.065] that reported XY = 0.1225 µm — below
    the 2 * 0.065 µm pixel floor, so physically impossible.
    """
    volume, spacing = _anisotropic_volume()
    result = fsc_resolution(volume, spacing=spacing, angle_delta=15, **kwargs)

    floor_xy = 2 * spacing[2]
    assert np.isfinite(result["xy"]), "XY resolution should be measurable"
    assert result["xy"] >= floor_xy, (
        f"XY resolution {result['xy']:.4f} is below the {floor_xy} µm pixel floor"
    )
    if np.isfinite(result["z"]):
        assert result["z"] >= floor_xy


def test_fsc_resolution_inverts_its_own_frequency_axis() -> None:
    """Reported resolution must equal ``1 / (f_c * max_freq)``.

    Two-image mode skips the checkerboard calibration factor, so the conversion
    from the crossing frequency is exact and the Nyquist used for normalization
    must be the one used for inversion.
    """
    volume, spacing = _anisotropic_volume()
    rng = np.random.default_rng(7)
    # Independent noise well above the signal's high-frequency content, so the
    # FSC actually falls through the threshold within the measured range.
    vol_a = volume + 0.3 * rng.normal(size=volume.shape)
    vol_b = volume + 0.3 * rng.normal(size=volume.shape)

    fsc_data, max_freq = _fsc_hist_compute(
        vol_a,
        vol_b,
        bin_delta=1,
        angle_delta=45,
        spacing_list=spacing,
        exclude_axis_angle=0.0,
        use_max_nyquist=False,
        zero_padding=False,
        average=True,
    )
    assert max_freq == pytest.approx(_kmax_phys(vol_a.shape, spacing))

    result = _fsc_extract_resolution(
        fsc_data,
        spacing_list=spacing,
        max_freq=max_freq,
        single_image=False,
        resolution_threshold="fixed",
        threshold_value=0.143,
    )

    # Re-derive the XY crossing frequency independently.
    xy_angle = max(fsc_data)
    coll = FourierCorrelationDataCollection()
    coll[xy_angle] = fsc_data[xy_angle]
    analyzed = FourierCorrelationAnalysis(
        coll,
        1.0 / (2.0 * max_freq),
        resolution_threshold="fixed",
        threshold_value=0.143,
        curve_fit_type="smooth-spline",
    ).execute(z_correction=1)[xy_angle]
    f_c = analyzed.resolution["resolution-point"][1]

    assert np.isfinite(f_c)
    assert result["xy"] == pytest.approx(1.0 / (f_c * max_freq), rel=1e-6)


@pytest.mark.parametrize("backend", ["mask", "hist"])
def test_frc_spacing_one_matches_index_units(backend: str) -> None:
    """``spacing=1.0`` must behave exactly like ``spacing=None``.

    Regression guard: ``radial_bin_id`` treated an all-ones spacing as None and
    switched to index units while ``radial_edges`` stayed in physical units, so
    ``np.digitize`` clipped every non-DC voxel into the last bin — the hist
    backend returned NaN for ``spacing=1.0`` and a valid number for None.
    """
    volume, _ = _anisotropic_volume()
    image = volume[16]

    res_one = frc_resolution(image, spacing=1.0, backend=backend)
    res_none = frc_resolution(image, spacing=None, backend=backend)

    assert np.isfinite(res_one), "spacing=1.0 must yield a finite resolution"
    assert res_one == pytest.approx(res_none, rel=1e-9)


@pytest.mark.parametrize("shape", [(64, 128), (32, 64, 64)])
def test_index_units_bin_identically_to_unit_spacing(shape: tuple[int, ...]) -> None:
    """``spacing=None`` must bin exactly like ``spacing=1.0`` on non-square input.

    The old index-unit branch scaled each axis by its own length
    (``fftfreq(n) * n``), which turns a constant-radius ring into an ellipse in
    physical frequency as soon as the axes differ in length. A (64, 128) array
    then produced 32 bins for ``None`` against 64 for ``1.0``, with different
    per-voxel assignments, so the two spellings of "no physical units" disagreed.
    Square input happened to agree, which is why it went unnoticed.
    """
    unit = [1.0] * len(shape)

    edges_none, radii_none = radial_edges(shape, 1.0, spacing=None)
    edges_one, radii_one = radial_edges(shape, 1.0, spacing=unit)

    np.testing.assert_allclose(edges_none, edges_one)
    np.testing.assert_allclose(radii_none, radii_one)

    bid_none = radial_bin_id(shape, edges_none, spacing=None)
    bid_one = radial_bin_id(shape, edges_one, spacing=unit)
    np.testing.assert_array_equal(bid_none, bid_one)


def test_radial_k_grid_kmax_is_derived_not_hardcoded() -> None:
    """An odd axis tops out below Nyquist, so k_max cannot be a constant 0.5.

    ``radial_k_grid`` returned 0.5 for every index-unit grid; the highest
    frequency an odd axis of length n actually carries is ``(n // 2) / n``.
    """
    k_radius, k_max = radial_k_grid((63, 63))

    assert k_max == pytest.approx(31 / 63)
    assert k_max < 0.5
    # Nothing on the grid exceeds the reported maximum along an axis.
    assert float(np.abs(np.fft.fftfreq(63)).max()) == pytest.approx(k_max)
    # Even axes are unchanged at exactly Nyquist.
    assert radial_k_grid((64, 64))[1] == pytest.approx(0.5)


@pytest.mark.parametrize("backend", ["mask", "hist"])
def test_frc_frequency_axis_reaches_nyquist_on_non_square_input(backend: str) -> None:
    """The FRC frequency axis must span [0, 1] regardless of aspect ratio.

    Normalizing by ``shape[0] // 2`` compressed the axis for non-square input
    (shape (128, 64) reached only 0.48), halving every crossing frequency and so
    doubling the reported resolution.
    """
    volume, _ = _anisotropic_volume()
    image = volume[16][:, :64]

    result = calculate_frc(
        image, spacing=1.0, backend=backend, zero_padding=False, bin_delta=1
    )
    freq = np.asarray(result.correlation["frequency"])
    assert freq.max() > 0.95, f"frequency axis stops at {freq.max():.3f}"
    assert freq.max() <= 1.0


def test_frc_backends_agree_on_non_square_input() -> None:
    """Both backends must land on the same axis for non-square input."""
    volume, _ = _anisotropic_volume()
    image = volume[16][:, :64]
    kwargs: dict[str, Any] = dict(spacing=1.0, zero_padding=False, bin_delta=1)

    res_mask = frc_resolution(image, backend="mask", **kwargs)
    res_hist = frc_resolution(image, backend="hist", **kwargs)
    assert res_mask == pytest.approx(res_hist, rel=0.02)


@pytest.mark.parametrize("angle_delta", [20, 100, 0, -15])
@pytest.mark.parametrize("backend", ["mask", "hist"])
def test_fsc_rejects_angle_delta_not_dividing_90(
    angle_delta: int, backend: str
) -> None:
    """``n_angle = 90 // angle_delta`` truncated non-divisors silently.

    An angle_delta of 20 folded the leftover 80-90 degree wedge into the last
    sector; anything above 90 produced zero sectors and NaN resolutions.
    """
    volume, spacing = _anisotropic_volume()
    with pytest.raises(ValueError, match="angle_delta"):
        fsc_resolution(
            volume, spacing=spacing, angle_delta=angle_delta, backend=backend
        )


def test_threshold_curve_does_not_mutate_stored_points() -> None:
    """Computing a threshold curve must not patch the stored FRC data."""
    data_set = FourierCorrelationData()
    points = np.array([10.0, 20.0, 30.0, 0.0])
    data_set.correlation["frequency"] = np.linspace(0.1, 1.0, 4)
    data_set.correlation["correlation"] = np.linspace(1.0, 0.0, 4)
    data_set.correlation["points-x-bin"] = points

    calculate_resolution_threshold_curve(data_set, "one-bit", 0.143, 7.0)

    np.testing.assert_array_equal(
        data_set.correlation["points-x-bin"], [10.0, 20.0, 30.0, 0.0]
    )
    np.testing.assert_array_equal(points, [10.0, 20.0, 30.0, 0.0])


def test_data_collection_iteration_is_restartable() -> None:
    """A broken-out-of iteration must not leave the collection half-consumed.

    The shared ``iter_index`` was only reset on ``StopIteration``, so any
    ``break`` made the next loop start mid-collection.
    """
    coll = FourierCorrelationDataCollection()
    for key in (0, 45, 90):
        coll[key] = FourierCorrelationData()

    for _ in coll:
        break

    assert [key for key, _ in coll] == ["0", "45", "90"]
    # Nested iteration must be independent too.
    pairs = [(a, b) for a, _ in coll for b, _ in coll]
    assert len(pairs) == 9


@pytest.mark.skipif(not _gpu_available(), reason="requires a CUDA GPU")
def test_sectioned_fsc_matches_between_devices() -> None:
    """The sectioned hist backend must give identical results on CPU and GPU.

    Covers the device handling in ``_calculate_fsc_sectioned_hist``, where the
    radial and angular edges are built on the host and moved to the input's
    device.
    """
    volume, spacing = _anisotropic_volume()

    res_cpu = fsc_resolution(volume, spacing=spacing, angle_delta=15)
    res_gpu = fsc_resolution(ascupy(volume), spacing=spacing, angle_delta=15)

    for key in ("xy", "z"):
        assert res_cpu[key] == pytest.approx(res_gpu[key], rel=1e-9, nan_ok=True)
