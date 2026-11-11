"""GPU mesh features: CUDA marching cubes and ``ArrayMesh`` against the CPU path."""

import numpy as np
import pytest
from scipy import ndimage
from skimage.measure import marching_cubes

trimesh = pytest.importorskip("trimesh")

from cubic.cuda import ascupy, asnumpy  # noqa: E402
from cubic.feature import mesh  # noqa: E402
from cubic.feature._array_mesh import ArrayMesh  # noqa: E402
from cubic.feature._marching_cubes_cuda import (  # noqa: E402
    marching_cubes as marching_cubes_gpu,
)


def _canonical(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique vertices, and faces as sorted rows of vertex ranks.

    Each face is rotated to start at its lowest-ranked vertex, which keeps its
    orientation, so two meshes compare equal exactly when they have the same
    vertices and the same oriented triangles in any order.
    """
    unique, rank = np.unique(verts, axis=0, return_inverse=True)
    ranked = rank.reshape(-1)[faces]
    start = np.argmin(ranked, axis=1)[:, None]
    rotated = np.take_along_axis(ranked, (start + np.arange(3)) % 3, axis=1)
    return unique, np.unique(rotated, axis=0)


def _assert_same_mesh(v_gpu, f_gpu, v_cpu: np.ndarray, f_cpu: np.ndarray) -> None:
    """Assert a GPU mesh equals a CPU one bit for bit, up to vertex and face order."""
    gpu = _canonical(asnumpy(v_gpu), asnumpy(f_gpu))
    cpu = _canonical(v_cpu, f_cpu)
    np.testing.assert_array_equal(gpu[0], cpu[0])
    np.testing.assert_array_equal(gpu[1], cpu[1])


def _volumes() -> dict[str, tuple[np.ndarray, float, tuple, bool]]:
    rng = np.random.default_rng(0)
    smooth = ndimage.gaussian_filter(rng.normal(size=(28, 28, 28)), 2).astype(
        np.float32
    )
    z, y, x = np.meshgrid(
        *[np.arange(n) - (n - 1) / 2 for n in (22, 26, 24)], indexing="ij"
    )
    ball = np.pad(z**2 + y**2 + x**2 <= 9**2, 1).astype(np.float32)
    return {
        # dense with every Lewiner case, including the ambiguous ones
        "binary noise": (
            (rng.random((20, 20, 20)) > 0.5).astype(np.float32),
            0.5,
            (1.0, 1.0, 1.0),
            True,
        ),
        "continuous noise": (
            rng.normal(size=(20, 20, 20)).astype(np.float32),
            0.0,
            (1.0, 1.0, 1.0),
            True,
        ),
        "anisotropic ball": (ball, 0.5, (0.325, 0.088, 0.088), True),
        # values equal to the level create coincident vertices
        "level hits, merged": (
            (np.round(smooth * 4) / 4).astype(np.float32),
            0.0,
            (1.0, 1.0, 1.0),
            False,
        ),
    }


@pytest.mark.parametrize("name", list(_volumes()))
def test_gpu_marching_cubes_matches_skimage(name: str, gpu_available: bool) -> None:
    """The CUDA port yields skimage's mesh bit for bit, up to vertex and face order."""
    if not gpu_available:
        pytest.skip("GPU not available")
    volume, level, spacing, allow_degenerate = _volumes()[name]
    v_cpu, f_cpu, _, _ = marching_cubes(
        volume, level, spacing=spacing, allow_degenerate=allow_degenerate
    )
    v_gpu, f_gpu = marching_cubes_gpu(ascupy(volume), level, spacing, allow_degenerate)

    assert asnumpy(v_gpu).dtype == v_cpu.dtype
    _assert_same_mesh(v_gpu, f_gpu, v_cpu, f_cpu)


def test_array_mesh_matches_trimesh() -> None:
    """On NumPy arrays, ArrayMesh reproduces trimesh's features to rounding.

    It recomputes the trimesh quantities the features read (mass properties,
    adjacency, dihedral angles, angle defects) with trimesh's formulas, and
    hands hull-based features to trimesh itself.
    """
    z, y, x = np.meshgrid(
        *[np.arange(n) - (n - 1) / 2 for n in (30, 30, 30)], indexing="ij"
    )
    r2 = z**2 + y**2 + x**2
    hollow = np.pad((r2 <= 12**2) & ~(r2 <= 5**2), 1)
    verts, faces, _, _ = marching_cubes(hollow, spacing=(2.0, 1.0, 1.0))
    reference = trimesh.Trimesh(verts, faces)
    trimesh.repair.fix_normals(reference)

    expected = mesh.extract_mesh_features(reference)
    got = mesh.extract_mesh_features(ArrayMesh(verts, faces))

    assert got == pytest.approx(expected, rel=1e-12)


def _level_hit_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Mesh of noise rounded to quarters, whose 0.5 level is hit exactly.

    Removing the degenerate faces leaves coincident and unreferenced vertices.
    """
    rng = np.random.default_rng(1)
    volume = np.pad(np.round(rng.random((16, 16, 16)) * 4) / 4, 1).astype(np.float32)
    verts, faces, _, _ = marching_cubes(volume, 0.5, allow_degenerate=False)
    return verts, faces


def test_array_mesh_merges_vertices_like_trimesh(gpu_available: bool) -> None:
    """Equal and unreferenced vertices are merged as trimesh's processing does.

    Unmerged, each unreferenced vertex added a 2 pi angle defect and each
    duplicate split the defects and edges at a seam: the sum of |defect| / 4 pi
    (total |K|) came out 480.9 against trimesh's 496.1 on this mesh. The mesh
    is not watertight, so the topology-dependent quantities are compared
    directly instead of through extract_mesh_features.

    Defects are compared away from zero-area (collinear) faces: whether trimesh
    or ArrayMesh zeroes such a face's angles depends on the last bit of a dot
    product near 1, which arccos amplifies to its 1e-8 tolerance, on any device.
    """
    verts, faces = _level_hit_mesh()
    assert len(np.unique(faces)) < len(verts)
    assert len(np.unique(verts, axis=0)) < len(verts)
    reference = trimesh.Trimesh(verts, faces)  # process=True, as on the CPU path
    key = {tuple(p): i for i, p in enumerate(np.round(reference.vertices * 1e8))}

    arrays = [(verts, faces)]
    if gpu_available:
        arrays.append((ascupy(verts), ascupy(faces)))
    for v, f in arrays:
        merged = ArrayMesh(v, f)
        m_verts, m_faces = asnumpy(merged.vertices), asnumpy(merged.faces)
        assert len(m_verts) == len(reference.vertices)
        assert len(np.unique(m_faces)) == len(m_verts)
        assert merged.euler_number == reference.euler_number
        assert merged.area == pytest.approx(reference.area, rel=1e-12)
        match = np.array([key[tuple(p)] for p in np.round(m_verts * 1e8)])
        flat = np.zeros(len(m_verts), dtype=bool)
        flat[m_faces[asnumpy(merged.area_faces) == 0].ravel()] = True
        np.testing.assert_allclose(
            asnumpy(merged.vertex_defects)[~flat],
            reference.vertex_defects[match[~flat]],
            atol=1e-9,
        )


def test_array_mesh_refuses_in_place_changes() -> None:
    """A delegated method that would change the mesh raises; reads still work.

    Changing only the host copy would leave the arrays and the cached volume
    and area describing a different mesh from the delegated attributes.
    """
    z, y, x = np.meshgrid(*[np.arange(n) - 9.5 for n in (20, 20, 20)], indexing="ij")
    verts, faces, _, _ = marching_cubes(np.pad(z**2 + y**2 + x**2 <= 8**2, 1))
    ball = ArrayMesh(verts, faces)

    with pytest.raises(TypeError, match="in place"):
        ball.apply_scale(2.0)

    assert ball.to_trimesh().volume == pytest.approx(ball.volume, rel=1e-12)
    assert ball.copy().volume == pytest.approx(ball.volume, rel=1e-12)


def test_gpu_marching_cubes_runs_on_the_input_device(gpu_available: bool) -> None:
    """A volume on GPU 1 is meshed on GPU 1 while GPU 0 is current."""
    if not gpu_available:
        pytest.skip("GPU not available")
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("needs two GPUs")
    volume, level, spacing, allow_degenerate = _volumes()["anisotropic ball"]
    v_cpu, f_cpu, _, _ = marching_cubes(volume, level, spacing=spacing)
    # float32 is used as is; float64 makes the call copy it to contiguous float32,
    # which the unfixed function did on GPU 0, returning the mesh there
    for dtype in (cp.float32, cp.float64):
        with cp.cuda.Device(1):
            on_1 = cp.asarray(volume, dtype=dtype)

        with cp.cuda.Device(0):
            v_gpu, f_gpu = marching_cubes_gpu(on_1, level, spacing, allow_degenerate)

        assert v_gpu.device.id == 1 and f_gpu.device.id == 1
        _assert_same_mesh(v_gpu, f_gpu, v_cpu, f_cpu)


def _labels() -> np.ndarray:
    z, y, x = np.meshgrid(*[np.arange(n) for n in (30, 64, 64)], indexing="ij")
    labels = np.zeros((30, 64, 64), dtype=np.int32)
    labels[(z - 14) ** 2 + (y - 18) ** 2 / 2 + (x - 20) ** 2 <= 9**2] = 1
    torus = (np.sqrt((y - 44) ** 2 + (x - 42) ** 2) - 12) ** 2 + (z - 14) ** 2 <= 5**2
    labels[torus] = 2
    labels[0:6, 50:64, 0:14] = 3  # touches the border
    return labels


def test_extract_features_on_gpu_matches_cpu(gpu_available: bool) -> None:
    """A CuPy label image gives the CPU features, to rounding.

    Both meshes are identical: the marching cubes port matches skimage, and the
    curvature blur rounds like SciPy on the GPU. Only summation order differs.
    The third label is too thin for the default blur, so its four
    curvature-distribution features are NaN on both devices.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    labels = _labels()
    spacing = (0.3, 0.1, 0.1)

    with pytest.warns(UserWarning, match="vanishes"):
        cpu_labels, cpu, _ = mesh.extract_features(labels, spacing=spacing)
    with pytest.warns(UserWarning, match="vanishes"):
        gpu_labels, gpu, _ = mesh.extract_features(ascupy(labels), spacing=spacing)

    np.testing.assert_array_equal(gpu_labels, cpu_labels)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-9, equal_nan=True)


def test_fractal_dimension_stays_on_the_gpu(gpu_available: bool, monkeypatch) -> None:
    """Fractal Dimension of a CuPy mask never builds a host trimesh."""
    if not gpu_available:
        pytest.skip("GPU not available")

    def fail(self):
        raise AssertionError("fell back to a host trimesh")

    monkeypatch.setattr(ArrayMesh, "to_trimesh", fail)
    mask = ascupy(_labels() == 1)
    name = "Fractal Dimension"

    on_gpu = mesh.extract_surface_features(
        mask, spacing=(0.3, 0.1, 0.1), features=[name]
    )[name]
    on_cpu = mesh.extract_surface_features(
        asnumpy(mask), spacing=(0.3, 0.1, 0.1), features=[name]
    )[name]

    assert on_gpu == pytest.approx(on_cpu, rel=1e-9)


def test_mask2mesh_on_gpu(gpu_available: bool) -> None:
    """A CuPy mask is meshed on the GPU; only ``spacing`` is accepted as a kwarg."""
    if not gpu_available:
        pytest.skip("GPU not available")
    mask = ascupy(_labels() == 1)

    on_gpu = mesh.mask2mesh(mask, spacing=(0.3, 0.1, 0.1))
    on_cpu = mesh.mask2mesh(asnumpy(mask), spacing=(0.3, 0.1, 0.1))

    assert isinstance(on_gpu, ArrayMesh)
    assert on_gpu.volume == pytest.approx(on_cpu.volume, rel=1e-12)
    with pytest.raises(ValueError, match="does not support"):
        mesh.mask2mesh(mask, marching_cubes_kwargs={"step_size": 2})
