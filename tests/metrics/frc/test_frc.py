"""Tests for FRC and FSC resolution calculations."""

from typing import Any
from collections.abc import Sequence

import numpy as np
import pytest
from skimage import data

from cubic.cuda import CUDAManager, ascupy
from cubic.skimage import filters
from cubic.metrics.frc import calculate_frc, fsc_resolution
from cubic.metrics.frc.frc import _calibration_factor


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
        assert 0.4 < factor < 1.1, f"Unexpected calibration factor {factor} at freq {freq}"

    # At very low frequencies, the exponential term dominates
    factor_low = _calibration_factor(0.05)
    assert factor_low > 0, "Calibration factor should be positive"

    # The calibration curve is monotonically increasing
    factors = [_calibration_factor(f) for f in [0.1, 0.2, 0.3, 0.4]]
    assert all(
        factors[i] <= factors[i + 1] for i in range(len(factors) - 1)
    ), "Calibration factor should increase with frequency"


def test_fsc_resolution_single_image(
    cells_volume: tuple[np.ndarray, list[float]],
) -> None:
    """Test FSC resolution with single-image mode (checkerboard split)."""
    volume, spacing = cells_volume

    # Single-image FSC should return positive resolution values
    result = fsc_resolution(
        volume,
        bin_delta=10,
        angle_delta=45,
        spacing=spacing,
        backend="hist",
    )

    assert "xy" in result, "FSC result should have 'xy' key"
    assert "z" in result, "FSC result should have 'z' key"

    # Resolution values should be positive and finite
    assert result["xy"] > 0, "XY resolution should be positive"
    assert result["z"] > 0, "Z resolution should be positive"
    assert np.isfinite(result["xy"]), "XY resolution should be finite"
    assert np.isfinite(result["z"]), "Z resolution should be finite"

    # Resolution should be in a reasonable range (in microns for cells3d)
    # cells3d has ~0.26 um XY spacing, typical XY resolution 0.3-1.0 um
    # cells3d has ~0.29 um Z spacing, typical Z resolution varies depending on
    # threshold and analysis method. With 1/7 fixed threshold and calibration
    # correction, values can be quite small for well-sampled data.
    if spacing != [1.0, 1.0, 1.0]:  # Skip if using synthetic fallback
        assert 0.1 < result["xy"] < 5.0, f"XY resolution {result['xy']} out of expected range"
        assert 0.1 < result["z"] < 20.0, f"Z resolution {result['z']} out of expected range"
