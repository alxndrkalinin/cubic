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
    cpu, gpu = _canonical(v_cpu, f_cpu), _canonical(asnumpy(v_gpu), asnumpy(f_gpu))
    np.testing.assert_array_equal(gpu[0], cpu[0])
    np.testing.assert_array_equal(gpu[1], cpu[1])


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
    names = [n for n in mesh.mesh_feature_list() if n != "Fractal Dimension"]

    expected = mesh.extract_mesh_features(reference, names)
    got = mesh.extract_mesh_features(ArrayMesh(verts, faces), names)

    assert got == pytest.approx(expected, rel=1e-12)
