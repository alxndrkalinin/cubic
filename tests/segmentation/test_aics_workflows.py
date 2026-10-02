"""Tests for the device-agnostic Allen Cell Structure Segmenter workflows."""

from itertools import combinations_with_replacement

import numpy as np
import pytest
from scipy import ndimage as ndi
from scipy.stats import norm
from skimage.morphology import remove_small_objects

from cubic.cuda import ascupy, asnumpy, get_device
from cubic.segmentation import aics_workflows, workflow_sec61b, workflow_tomm20
from cubic.segmentation.segment_utils import _SKIMAGE_USES_MAX_SIZE
from cubic.segmentation.aics_workflows import (
    _gaussian_nearest,
    _hessian_2d_eigen_max,
    _itk_boundary_regions,
    _near_float32_midpoint,
    intensity_normalization,
    remove_small_objects_aics,
    vesselness_slice_by_slice,
    gradient_anisotropic_diffusion,
)


def _to_device(array: np.ndarray, use_gpu: bool, gpu_available: bool) -> np.ndarray:
    """Return ``array`` on the GPU, skipping the test when there is none."""
    if not use_gpu:
        return array
    if not gpu_available:
        pytest.skip("GPU not available")
    return ascupy(array)


def _filaments(shape: tuple[int, int, int], seed: int = 0) -> np.ndarray:
    """Noisy volume of bright random line segments, a stand-in for ER/mito."""
    rng = np.random.default_rng(seed)
    img = np.zeros(shape, dtype=np.float32)
    nz, ny, nx = shape
    for _ in range(40):
        z = rng.integers(0, nz)
        y0, x0 = rng.integers(0, ny), rng.integers(0, nx)
        angle = rng.uniform(0, np.pi)
        for s in range(rng.integers(8, 30)):
            y = int(y0 + s * np.sin(angle))
            x = int(x0 + s * np.cos(angle))
            if 0 <= y < ny and 0 <= x < nx:
                img[z, y, x] = 1.0
    img = ndi.gaussian_filter(img, sigma=(0.8, 1.0, 1.0))
    img += rng.normal(0, 0.02, size=shape).astype(np.float32)
    return (img * 1000 + 100).astype(np.float32)


def _reference_vesselness(image: np.ndarray, sigma: float) -> np.ndarray:
    """``aicssegmentation.core.vessel.vesselnessSliceBySlice`` (tau=1, one sigma)."""
    mip = image.max(axis=0)
    out = np.zeros(image.shape)
    width = image.shape[2]
    for zz in range(image.shape[0]):
        plane = np.concatenate((image[zz], mip), axis=1)
        smoothed = ndi.gaussian_filter(plane, sigma=sigma, mode="nearest", truncate=3.0)
        grads = np.gradient(smoothed)
        elems = [
            (sigma**2) * np.gradient(grads[a], axis=b)
            for a, b in combinations_with_replacement(range(2), 2)
        ]
        hessian = np.stack(
            [np.stack([elems[0], elems[1]], -1), np.stack([elems[1], elems[2]], -1)],
            -2,
        )
        eig = np.linalg.eigvalsh(hessian)
        order = np.abs(eig).argsort(axis=-1)
        eigen2 = np.take_along_axis(eig, order, axis=-1)[..., 1]
        lam3 = eigen2.copy()
        lam3[(lam3 < 0) & (lam3 >= lam3.min())] = lam3.min()
        resp = np.square(eigen2) * np.abs(lam3 - eigen2)
        den = np.power(2 * np.abs(eigen2) + np.abs(lam3 - eigen2), 3)
        den[den == 0] = 1e-10
        resp = 27 * resp / den
        resp[eigen2 < 0.5 * lam3] = 1
        resp[eigen2 >= 0] = 0
        resp[np.isinf(resp)] = 0
        out[zz, :, : width - 3] = resp[:, : width - 3]
    return out


@pytest.mark.parametrize("use_gpu", [False, True])
@pytest.mark.parametrize("inclusive", [False, True])
@pytest.mark.parametrize("per_slice", [False, True])
def test_remove_small_objects_matches_skimage(
    use_gpu: bool, inclusive: bool, per_slice: bool, gpu_available: bool
) -> None:
    """Sizes strictly below (or, inclusive, at most) ``min_size`` are dropped."""
    rng = np.random.default_rng(1)
    mask = ndi.binary_opening(rng.random((6, 64, 64)) > 0.6)
    min_size = 4
    # Largest removed size; scikit-image < 0.26 only has the exclusive min_size.
    largest = min_size if inclusive else min_size - 1
    if _SKIMAGE_USES_MAX_SIZE:
        size_kwargs = {"max_size": largest}
    else:
        size_kwargs = {"min_size": largest + 1}

    def reference(m: np.ndarray) -> np.ndarray:
        return remove_small_objects(m, connectivity=1, **size_kwargs)

    if per_slice:
        expected = np.stack([reference(m) for m in mask])
    else:
        expected = reference(mask)
    out = remove_small_objects_aics(
        _to_device(mask, use_gpu, gpu_available),
        min_size,
        per_slice=per_slice,
        inclusive=inclusive,
    )
    assert get_device(out) == ("GPU" if use_gpu else "CPU")
    np.testing.assert_array_equal(asnumpy(out), expected)


@pytest.mark.parametrize("use_gpu", [False, True])
def test_gaussian_matches_scipy_bitwise(use_gpu: bool, gpu_available: bool) -> None:
    """SciPy's float64-per-pass rounding is reproduced on both devices."""
    img = _filaments((4, 48, 64))
    expected = ndi.gaussian_filter(img, sigma=(0, 1.5, 1.5), mode="nearest", truncate=3)
    out = _gaussian_nearest(_to_device(img, use_gpu, gpu_available), [0.0, 1.5, 1.5])
    assert out.dtype == np.float32
    np.testing.assert_array_equal(asnumpy(out), expected)


@pytest.mark.parametrize("use_gpu", [False, True])
def test_vesselness_matches_reference(use_gpu: bool, gpu_available: bool) -> None:
    """Batched vesselness equals the per-slice LAPACK reference."""
    img = intensity_normalization(_filaments((5, 40, 56)), (2.5, 7.5))
    expected = _reference_vesselness(img, 1.0)
    out = vesselness_slice_by_slice(_to_device(img, use_gpu, gpu_available), [1.0])
    assert out.dtype == np.float64
    np.testing.assert_allclose(asnumpy(out), expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(asnumpy(out) > 0.15, expected > 0.15)


def test_itk_boundary_regions_partition_the_volume() -> None:
    """Interior first, then x/y/z faces; every voxel exactly once."""
    for shape in [(4, 5, 6), (1, 3, 2), (2, 2, 2), (3, 7)]:
        regions = _itk_boundary_regions(shape)
        count = np.zeros(shape, dtype=int)
        for r in regions:
            count[r] += 1
        assert (count == 1).all(), shape
        assert all(sl == slice(1, n - 1) for sl, n in zip(regions[0], shape) if n > 2)


def test_near_float32_midpoint() -> None:
    """Only values within the margin of a float32 rounding midpoint are flagged."""
    one = 1.0
    midpoint = (one + float(np.nextafter(np.float32(1), np.float32(2)))) / 2
    assert _near_float32_midpoint(midpoint, 1e-9)
    assert _near_float32_midpoint(midpoint * (1 + 1e-11), 1e-9)
    assert not _near_float32_midpoint(one, 1e-9)
    assert not _near_float32_midpoint(0.0, 1e-9)


@pytest.mark.parametrize("use_gpu", [False, True])
def test_diffusion_serial_k_fallback_matches_fast_path(
    use_gpu: bool, gpu_available: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Forcing ITK's serial K accumulation every iteration changes nothing here."""
    img = intensity_normalization(_filaments((6, 40, 52)), (2.5, 7.5))
    dev = _to_device(img, use_gpu, gpu_available)
    fast = asnumpy(gradient_anisotropic_diffusion(dev))
    monkeypatch.setattr(aics_workflows, "_K_SERIAL_MARGIN", np.inf)
    serial = asnumpy(gradient_anisotropic_diffusion(dev))
    np.testing.assert_array_equal(serial, fast)


def test_hessian_eigen_gpu_matches_cpu(gpu_available: bool) -> None:
    """The closed-form eigenvalue is computed identically on both devices."""
    if not gpu_available:
        pytest.skip("GPU not available")
    img = _filaments((3, 32, 40))
    cpu = _hessian_2d_eigen_max(img, 1.0)
    gpu = _hessian_2d_eigen_max(ascupy(img), 1.0)
    np.testing.assert_array_equal(asnumpy(gpu), cpu)


@pytest.mark.parametrize("use_gpu", [False, True])
def test_intensity_normalization_matches_reference(
    use_gpu: bool, gpu_available: bool
) -> None:
    """Bitwise equal to aicssegmentation's ``norm.fit`` clip-and-rescale."""
    img = _filaments((4, 32, 32))
    original = img.copy()
    out = intensity_normalization(_to_device(img, use_gpu, gpu_available), (2.5, 7.5))
    np.testing.assert_array_equal(img, original)
    ref = img.copy()
    m, s = norm.fit(ref.flat)
    lo, hi = max(m - 2.5 * s, ref.min()), min(m + 7.5 * s, ref.max())
    ref[ref > hi] = hi
    ref[ref < lo] = lo
    expected = (ref - lo + 1e-8) / (hi - lo + 1e-8)
    assert out.dtype == np.float32
    np.testing.assert_array_equal(asnumpy(out), expected)


def test_intensity_normalization_validates_input() -> None:
    """Integer, non-finite, and malformed-parameter inputs raise."""
    with pytest.raises(TypeError):
        intensity_normalization(np.ones((2, 2), dtype=np.uint16), (1, 1))
    with pytest.raises(ValueError, match="non-finite"):
        intensity_normalization(np.array([1.0, np.nan], dtype=np.float32), (1, 1))
    with pytest.raises(ValueError, match="scaling_param"):
        intensity_normalization(np.ones((2, 2), dtype=np.float32), (1, 1, 1, 1))  # type: ignore[arg-type]


def test_diffusion_constant_image_is_unchanged() -> None:
    """Zero gradient everywhere gives ``K == 0`` and, as in ITK, no update."""
    img = np.full((4, 8, 8), 3.0, dtype=np.float32)
    np.testing.assert_array_equal(gradient_anisotropic_diffusion(img), img)


def test_diffusion_smooths_noise_and_keeps_mean() -> None:
    """Diffusion lowers the variance of noise; Neumann boundaries conserve the mean."""
    rng = np.random.default_rng(3)
    img = rng.normal(size=(6, 24, 24)).astype(np.float32)
    out = gradient_anisotropic_diffusion(img)
    assert out.dtype == np.float32
    assert out.std() < 0.5 * img.std()
    np.testing.assert_allclose(out.mean(), img.mean(), atol=1e-5)


def test_diffusion_kernel_matches_array_path(gpu_available: bool) -> None:
    """The fused CUDA kernel reproduces the array-operation path bit for bit."""
    if not gpu_available:
        pytest.skip("GPU not available")
    img = intensity_normalization(_filaments((6, 40, 52)), (2.5, 7.5))
    cpu = gradient_anisotropic_diffusion(img, spacing=(2.0, 1.0, 1.0))
    gpu = gradient_anisotropic_diffusion(ascupy(img), spacing=(2.0, 1.0, 1.0))
    assert get_device(gpu) == "GPU"
    np.testing.assert_array_equal(asnumpy(gpu), cpu)


def test_diffusion_2d_gpu_matches_cpu(gpu_available: bool) -> None:
    """2D CuPy input runs the array path on the device and matches NumPy."""
    if not gpu_available:
        pytest.skip("GPU not available")
    img = intensity_normalization(_filaments((1, 40, 52))[0], (2.5, 7.5))
    gpu = gradient_anisotropic_diffusion(ascupy(img))
    assert get_device(gpu) == "GPU"
    np.testing.assert_array_equal(asnumpy(gpu), gradient_anisotropic_diffusion(img))


def test_diffusion_kernel_handles_non_contiguous_input(gpu_available: bool) -> None:
    """A strided view is copied to C order before the kernel indexes it."""
    if not gpu_available:
        pytest.skip("GPU not available")
    img = intensity_normalization(_filaments((40, 24, 6)), (2.5, 7.5))
    view = img.transpose(2, 1, 0)
    cpu = gradient_anisotropic_diffusion(np.ascontiguousarray(view))
    gpu = gradient_anisotropic_diffusion(ascupy(img).transpose(2, 1, 0))
    np.testing.assert_array_equal(asnumpy(gpu), cpu)


def test_vesselness_rejects_too_narrow_image() -> None:
    """The reference zeroes the last three columns, so it needs more than three."""
    with pytest.raises(ValueError, match="wider than 3"):
        vesselness_slice_by_slice(np.ones((2, 8, 3), dtype=np.float32), [1.0])


def test_diffusion_matches_itk() -> None:
    """Bitwise agreement with ``itk.GradientAnisotropicDiffusionImageFilter``."""
    itk = pytest.importorskip("itk")
    img = intensity_normalization(_filaments((6, 40, 52)), (2.5, 7.5))
    image = itk.GetImageFromArray(img)
    image.SetSpacing([1.0, 1.0, 1.0])
    filt = itk.GradientAnisotropicDiffusionImageFilter.New(image)
    filt.SetNumberOfIterations(10)
    filt.SetTimeStep(0.0625)
    filt.SetConductanceParameter(1.2)
    filt.Update()
    expected = itk.GetArrayFromImage(filt.GetOutput())
    np.testing.assert_array_equal(gradient_anisotropic_diffusion(img), expected)


def test_diffusion_serial_k_matches_itk(monkeypatch: pytest.MonkeyPatch) -> None:
    """The replayed ITK summation order itself is bitwise ITK."""
    itk = pytest.importorskip("itk")
    img = intensity_normalization(_filaments((6, 40, 52)), (2.5, 7.5))
    filt = itk.GradientAnisotropicDiffusionImageFilter.New(itk.GetImageFromArray(img))
    filt.SetNumberOfIterations(10)
    filt.SetTimeStep(0.0625)
    filt.SetConductanceParameter(1.2)
    filt.Update()
    monkeypatch.setattr(aics_workflows, "_K_SERIAL_MARGIN", np.inf)
    np.testing.assert_array_equal(
        gradient_anisotropic_diffusion(img), itk.GetArrayFromImage(filt.GetOutput())
    )


@pytest.mark.parametrize("workflow", [workflow_sec61b, workflow_tomm20])
def test_workflow_gpu_matches_cpu(workflow, gpu_available: bool) -> None:
    """Each workflow returns the same mask on both devices."""
    if not gpu_available:
        pytest.skip("GPU not available")
    img = _filaments((6, 48, 64), seed=4)
    cpu = workflow(img)
    gpu = workflow(ascupy(img))
    assert cpu.dtype == bool and cpu.any()
    assert get_device(gpu) == "GPU"
    np.testing.assert_array_equal(asnumpy(gpu), cpu)


@pytest.mark.parametrize(
    ("workflow", "module_name", "func_name"),
    [
        (workflow_sec61b, "seg_sec61b", "Workflow_sec61b"),
        (workflow_tomm20, "seg_tomm20", "Workflow_tomm20"),
    ],
)
def test_workflow_matches_aicssegmentation(
    workflow, module_name: str, func_name: str
) -> None:
    """Masks equal ``aicssegmentation``'s under the installed scikit-image."""
    pytest.importorskip("itk")
    module = pytest.importorskip(f"aicssegmentation.structure_wrapper.{module_name}")
    img = _filaments((6, 48, 64), seed=5)
    expected = getattr(module, func_name)(img.copy(), output_type="array") > 0
    out = workflow(img, size_filter_inclusive=_SKIMAGE_USES_MAX_SIZE)
    np.testing.assert_array_equal(out, expected)
