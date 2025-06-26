from typing import Sequence, Any

import pytest
import numpy as np
from skimage import data

from morphocell.skimage import filters
from morphocell.cuda import ascupy, CUDAManager
from morphocell.metrics.frc import frc_resolution, fsc_resolution


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
def cells_volume() -> tuple[np.ndarray, list[float], bool]:
    """Return single-channel cells3d volume and spacing or indicate fallback."""
    try:
        volume = data.cells3d()[:, 1]
        spacing = [0.29, 0.26, 0.26]
        return volume, spacing, True
    except Exception:
        volume = make_fake_cells3d(shape=(32, 64, 64), random_seed=42)
        spacing = [1.0, 1.0, 1.0]
        return volume, spacing, False


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


def test_calculate_frc_cpu_vs_gpu(
    cells_volume: tuple[np.ndarray, list[float], bool],
) -> None:
    volume, spacing, real_data = cells_volume
    if not real_data:
        pytest.skip("cells3d dataset unavailable")
    slice_image = _middle_slice(volume)

    # Use 2D spacing (xy only)
    xy_spacing = spacing[1:]  # [y, x] spacing

    cpu_res = frc_resolution(slice_image, spacing=xy_spacing)
    _assert_positive(cpu_res)

    if _gpu_available():
        gpu_res = frc_resolution(ascupy(slice_image), spacing=xy_spacing)
        assert np.isclose(cpu_res, gpu_res, atol=1e-5)


def test_calculate_fsc_cpu_vs_gpu(
    cells_volume: tuple[np.ndarray, list[float], bool],
) -> None:
    volume, spacing, real_data = cells_volume
    if not real_data:
        pytest.skip("cells3d dataset unavailable")

    rng = np.random.default_rng(42)
    noisy_volume = volume.astype(np.float32) + rng.normal(0, 0.1, volume.shape).astype(
        np.float32
    )

    cpu_res = fsc_resolution(noisy_volume, spacing=spacing, bin_delta=1)
    _assert_positive(cpu_res["xy"])
    _assert_positive(cpu_res["z"])

    if _gpu_available():
        gpu_res = fsc_resolution(ascupy(noisy_volume), spacing=spacing, bin_delta=1)
        assert np.allclose(
            [cpu_res["xy"], cpu_res["z"]],
            [gpu_res["xy"], gpu_res["z"]],
            atol=1e-3,
        )


# def test_calculate_frc_single_vs_two_image() -> None:
#     """Test that single image FRC works correctly."""
#     # Create a test image with known properties
#     rng = np.random.default_rng(42)
#     test_image = rng.random((64, 64)).astype(np.float32)

#     # Single image FRC
#     single_res = frc_resolution(test_image)
#     _assert_positive(single_res)

#     # Two image FRC (should give different result)
#     test_image2 = test_image + rng.normal(0, 0.1, test_image.shape).astype(np.float32)
#     two_res = frc_resolution(test_image, test_image2)
#     _assert_positive(two_res)

#     # Results should be different
#     assert not np.isclose(single_res, two_res, rtol=0.1)


# def test_calculate_fsc_single_vs_two_image() -> None:
#     """Test that single image FSC works correctly."""
#     # Create a test volume with known properties
#     rng = np.random.default_rng(42)
#     test_volume = rng.random((32, 32, 32)).astype(np.float32)

#     # Single image FSC
#     single_res = fsc_resolution(test_volume, bin_delta=5)
#     _assert_positive(single_res["xy"])
#     _assert_positive(single_res["z"])

#     # Two image FSC (should give different result)
#     test_volume2 = test_volume + rng.normal(0, 0.2, test_volume.shape).astype(np.float32)
#     two_res = fsc_resolution(test_volume, test_volume2, bin_delta=5)
#     _assert_positive(two_res["xy"])
#     _assert_positive(two_res["z"])

#     # Results should be different
#     assert not np.allclose([single_res["xy"], single_res["z"]],
#                           [two_res["xy"], two_res["z"]], rtol=0.1)
