"""Tests for the skimage proxy module."""

import warnings
from typing import Any

import numpy as np
import pytest

import cubic.cuda as mc_cuda
import cubic.skimage as mc_skimage
from cubic.cuda import CUDAManager, ascupy, asnumpy, get_device


@pytest.mark.parametrize("use_gpu", [False, True])
def test_dispatch_on_array_passed_by_keyword(
    use_gpu: bool, gpu_available: bool
) -> None:
    """A GPU array under a non-``image`` keyword routes to the cuCIM backend.

    ``measure.label(label_image=...)`` passes the array under ``label_image``,
    a keyword the old first-arg/``image``-only detection never inspected — so a
    GPU array there routed to host scikit-image and crashed. Detection now
    scans every argument.
    """
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[5:7, 5:7] = 1
    cpu_res = mc_skimage.measure.label(label_image=mask)

    if use_gpu:
        if not gpu_available:
            pytest.skip("GPU not available")
        gpu_res = mc_skimage.measure.label(label_image=ascupy(mask))
        assert get_device(gpu_res) == "GPU"
        assert np.array_equal(asnumpy(gpu_res), cpu_res)
    else:
        assert np.array_equal(mc_skimage.measure.label(label_image=mask), cpu_res)


@pytest.mark.parametrize("use_gpu", [False, True])
def test_filters_gaussian_cpu_vs_gpu(use_gpu: bool, gpu_available: bool) -> None:
    """Compare Gaussian filtering on CPU and GPU."""
    img = np.random.random((5, 5)).astype(np.float32)
    cpu_res = mc_skimage.filters.gaussian(img, sigma=1.0, preserve_range=True)

    if use_gpu:
        if not gpu_available:
            pytest.skip("GPU not available")
        cp = CUDAManager().get_cp()
        gpu_res = mc_skimage.filters.gaussian(
            cp.asarray(img), sigma=1.0, preserve_range=True
        )
        assert np.allclose(asnumpy(gpu_res), cpu_res, atol=1e-6)
    else:
        gpu_res = mc_skimage.filters.gaussian(img, sigma=1.0, preserve_range=True)
        assert np.allclose(gpu_res, cpu_res)


def test_missing_cucim_function_falls_back_to_cpu(gpu_available: bool) -> None:
    """A function absent from cuCIM runs on the host instead of raising.

    The skimage proxy had no CPU fallback (unlike the SciPy proxy), so GPU input
    to any cuCIM-less function raised ``AttributeError``. cuCIM has no
    ``skimage.morphology.skeletonize``.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    mask = np.zeros((16, 16), dtype=bool)
    mask[4:12, 6:10] = True
    cpu_res = mc_skimage.morphology.skeletonize(mask)

    with pytest.warns(UserWarning, match="falling back to CPU"):
        gpu_res = mc_skimage.morphology.skeletonize(ascupy(mask))
    assert get_device(gpu_res) == "GPU"
    np.testing.assert_array_equal(asnumpy(gpu_res), cpu_res)


def test_cpu_fallback_returns_tuple_results_to_gpu(gpu_available: bool) -> None:
    """Every array in a tuple result comes back on the GPU.

    ``measure.marching_cubes`` is absent from cuCIM and returns
    ``(verts, faces, normals, values)``.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    volume = np.zeros((16, 16, 16), dtype=np.float32)
    volume[4:12, 4:12, 4:12] = 1.0

    with pytest.warns(UserWarning, match="falling back to CPU"):
        result = mc_skimage.measure.marching_cubes(ascupy(volume), level=0.5)
    assert isinstance(result, tuple) and len(result) == 4
    assert all(get_device(item) == "GPU" for item in result)

    expected = mc_skimage.measure.marching_cubes(volume, level=0.5)
    for got, want in zip(result, expected):
        np.testing.assert_allclose(asnumpy(got), want)


def test_cpu_fallback_leaves_scalar_results_on_the_host(gpu_available: bool) -> None:
    """A scalar returned by the host fallback must not become a 0-d GPU array.

    The return path moved anything with a ``dtype`` to the GPU, and NumPy scalars
    have one, so a count or measurement returned alongside arrays — the
    ``return_num=True`` shape — came back as a 0-d device array and ``range()``
    over it raised ``TypeError``. ``measure.mesh_surface_area`` is absent from
    cuCIM and returns a single float.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    volume = np.zeros((16, 16, 16), dtype=np.float32)
    volume[4:12, 4:12, 4:12] = 1.0

    with pytest.warns(UserWarning, match="falling back to CPU"):
        verts, faces, _, _ = mc_skimage.measure.marching_cubes(
            ascupy(volume), level=0.5
        )
    assert get_device(verts) == "GPU"

    with pytest.warns(UserWarning, match="falling back to CPU"):
        area = mc_skimage.measure.mesh_surface_area(verts, faces)

    assert getattr(area, "ndim", 0) == 0, f"expected a scalar, got ndim={area.ndim}"
    # The failure this guards: an int-like 0-d device array cannot index or
    # drive a range, so exercise the plain-scalar contract directly.
    assert float(area) > 0.0
    assert range(int(area)) is not None

    expected = mc_skimage.measure.mesh_surface_area(asnumpy(verts), asnumpy(faces))
    np.testing.assert_allclose(float(area), float(expected))


def test_cpu_fallback_warning_points_at_the_caller(gpu_available: bool) -> None:
    """The fallback warning must name the caller's line, not cubic's internals.

    Without ``stacklevel`` the warning is attributed to ``cuda.py``, which tells
    a user nothing about which of their calls triggered it.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    volume = np.zeros((16, 16, 16), dtype=np.float32)
    volume[4:12, 4:12, 4:12] = 1.0

    with pytest.warns(UserWarning, match="falling back to CPU") as record:
        mc_skimage.measure.marching_cubes(ascupy(volume), level=0.5)

    assert record[0].filename == __file__, (
        f"warning attributed to {record[0].filename}, expected this test file"
    )


def test_missing_cucim_module_falls_back_to_cpu(
    monkeypatch: pytest.MonkeyPatch, gpu_available: bool
) -> None:
    """A missing ``cucim.skimage`` submodule also routes to the host."""
    if not gpu_available:
        pytest.skip("GPU not available")

    real_import_module = mc_cuda.import_module

    def fake_import_module(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("cucim"):
            raise ModuleNotFoundError(f"simulated missing module: {name}")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(mc_cuda, "import_module", fake_import_module)

    img = np.random.random((5, 5)).astype(np.float32)
    cpu_res = mc_skimage.filters.gaussian(img, sigma=1.0, preserve_range=True)

    with pytest.warns(UserWarning, match="falling back to CPU"):
        gpu_res = mc_skimage.filters.gaussian(
            ascupy(img), sigma=1.0, preserve_range=True
        )
    assert get_device(gpu_res) == "GPU"
    assert np.allclose(asnumpy(gpu_res), cpu_res, atol=1e-6)


def test_cpu_route_unknown_function_raises_without_warning() -> None:
    """An unknown function on the host route raises without a bogus GPU warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning here fails the test
        with pytest.raises(AttributeError):
            mc_skimage.filters.definitely_not_a_skimage_function(np.zeros((4, 4)))
