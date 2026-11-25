"""Fixtures shared by the feature tests."""

import warnings
from typing import Any
from collections.abc import Callable

import numpy as np
import pytest
from scipy import ndimage

from cubic.cuda import asnumpy


def _largest_piece(mask: np.ndarray) -> np.ndarray:
    """Return the largest 26-connected piece of ``mask``."""
    pieces = ndimage.label(mask, structure=np.ones((3, 3, 3)))[0]
    return pieces == np.argmax(np.bincount(pieces.ravel())[1:]) + 1


def _ragged_ball(fraction: float) -> np.ndarray:
    """Return a 41^3 mask of a radius-15 ball with ``fraction`` of its shell flipped.

    The shell is the voxels within one voxel of the radius; only the largest
    26-connected piece is kept. ``fraction=0`` is the clean ball.
    """
    rng = np.random.default_rng(0)
    z, y, x = np.meshgrid(*[np.arange(41) - 20.0] * 3, indexing="ij")
    r = np.sqrt(z**2 + y**2 + x**2)
    shell = (np.abs(r - 15) <= 1.0) & (rng.random(r.shape) < fraction)
    return _largest_piece((r <= 15) ^ shell)


def _ball(radius: float, shape: int) -> np.ndarray:
    """Return a ``shape**3`` boolean mask of a centred ball of ``radius`` voxels."""
    grid = np.meshgrid(*[np.arange(shape) - (shape - 1) / 2] * 3, indexing="ij")
    return sum(axis**2 for axis in grid) <= radius**2


@pytest.fixture
def ball_of() -> Callable[[float, int], np.ndarray]:
    """Return ``_ball``: a centred ball mask of a given radius in a cube of a given side."""
    return _ball


def _hinge(angle: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the vertices and faces of two triangles at ``angle`` to each other.

    The faces ``(0, 1, 2)`` and ``(1, 0, 3)`` share the edge from vertex 0 to
    vertex 1; their dihedral angle is ``angle``.
    """
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, -np.cos(angle), np.sin(angle)]]
    )
    faces = np.array([[0, 1, 2], [1, 0, 3]])
    return vertices, faces


@pytest.fixture
def hinge_of() -> Callable[[float], tuple[np.ndarray, np.ndarray]]:
    """Return ``_hinge``: the vertices and faces of two triangles at a given angle."""
    return _hinge


@pytest.fixture
def ragged_ball() -> np.ndarray:
    """Return a 41^3 mask of a radius-15 ball with 30% of its boundary shell flipped.

    Only the largest 26-connected piece is kept. Its mesh has 33 edges shared
    by four faces, so trimesh does not call it watertight.
    """
    return _ragged_ball(0.3)


@pytest.fixture
def ragged_ball_of() -> Callable[[float], np.ndarray]:
    """Return ``_ragged_ball``: the ball with a given fraction of its shell flipped.

    Its mesh has 0, 3 and 33 edges shared by four faces at 2%, 10% and 30%.
    """
    return _ragged_ball


def _noise_blob(seed: int) -> np.ndarray:
    """Return a lobed mask: blurred noise cut at its top 30%, as one thick piece.

    32^3 normal noise from ``default_rng(seed)`` blurred at sigma 3,
    thresholded at its 70th percentile, the largest 26-connected piece cropped
    to its bounding box.
    """
    noise = ndimage.gaussian_filter(
        np.random.default_rng(seed).normal(size=(32, 32, 32)), 3.0
    )
    blob = _largest_piece(noise > np.quantile(noise, 0.7))
    return blob[ndimage.find_objects(blob.astype(int))[0]]


@pytest.fixture
def noise_blob_of() -> Callable[[int], np.ndarray]:
    """Return ``_noise_blob``: a lobed mask cut from blurred noise of a given seed."""
    return _noise_blob


def _lumpy_nucleus(seed: int) -> np.ndarray:
    """Return a 40 x 64 x 64 mask of an ellipsoid with lumps, the size of a nucleus.

    The ellipsoid has semi-axes (14, 26, 20) voxels; its normalized radius is
    perturbed by 0.25 times normal noise from ``default_rng(seed)`` blurred at
    sigma 4 and scaled to unit deviation.
    """
    shape, semi_axes = (40, 64, 64), (14, 26, 20)
    grid = np.meshgrid(*[np.arange(n) - (n - 1) / 2 for n in shape], indexing="ij")
    radius = np.sqrt(sum((g / a) ** 2 for g, a in zip(grid, semi_axes)))
    noise = ndimage.gaussian_filter(np.random.default_rng(seed).normal(size=shape), 4.0)
    return radius + 0.25 * noise / noise.std() <= 1


@pytest.fixture
def lumpy_nucleus_of() -> Callable[[int], np.ndarray]:
    """Return ``_lumpy_nucleus``: a lumpy ellipsoid mask of a given seed."""
    return _lumpy_nucleus


def _assert_far_object_like_near(
    nucleus: np.ndarray, to_device: Callable[[np.ndarray], Any]
) -> None:
    """Check that ``nucleus`` 2030 voxels from the origin is measured as near it.

    ``nucleus`` is 40 x 64 x 64; ``to_device`` moves each mask to the device
    to measure on. Every feature of ``extract_surface_features`` of the far
    copy, and of ``extract_features`` of it as a label, must match the near
    copy's to 1e-12, and ``mask2mesh(..., sigma=...)`` of the far copy must be
    the near copy's mesh moved by the offset.
    """
    # Local import: the voxel and texture tests share this conftest and need no trimesh.
    from cubic.feature import mesh

    names = mesh.mesh_feature_list(optional=True)
    spacing = (2.685, 1.0, 1.0)
    near = np.zeros((44, 68, 68), dtype=bool)
    near[2:42, 2:66, 2:66] = nucleus
    far = np.zeros((44, 68, 2100), dtype=bool)
    far[2:42, 2:66, 2030:2094] = nucleus

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = [
            mesh.extract_surface_features(
                to_device(mask), spacing=spacing, features=names
            )
            for mask in (near, far)
        ]
        _, as_label, _ = mesh.extract_features(
            to_device(far.astype(np.int32)), names, spacing=spacing
        )
    meshes = [
        mesh.mask2mesh(to_device(mask), spacing=spacing, sigma=5.0)
        for mask in (near, far)
    ]

    for name, labelled in zip(names, as_label[0]):
        near_value = float(values[0][name])
        assert np.isfinite(near_value), name
        assert float(values[1][name]) == pytest.approx(near_value, rel=1e-12), name
        assert labelled == pytest.approx(near_value, rel=1e-12), name
    offset = np.array([0.0, 0.0, 2028.0])
    np.testing.assert_array_equal(asnumpy(meshes[0].faces), asnumpy(meshes[1].faces))
    np.testing.assert_allclose(
        asnumpy(meshes[1].vertices) - offset,
        asnumpy(meshes[0].vertices),
        rtol=0,
        atol=1e-9,
    )


@pytest.fixture
def assert_far_object_like_near() -> Callable[..., None]:
    """Return ``_assert_far_object_like_near``, for either device."""
    return _assert_far_object_like_near


def _flat_vertices(surface: object) -> np.ndarray:
    """Return which vertices of ``surface`` have all their face normals equal to 1e-9.

    Computed on the host from ``face_normals`` and ``faces``, of a
    ``trimesh.Trimesh`` or an ``ArrayMesh`` on either device.
    """
    normals = np.asarray(asnumpy(surface.face_normals))
    faces = np.asarray(asnumpy(surface.faces))
    low = np.full((len(surface.vertices), 3), np.inf)
    high = -low
    for corner in range(3):
        np.minimum.at(low, faces[:, corner], normals)
        np.maximum.at(high, faces[:, corner], normals)
    return (high - low).max(axis=1) <= 1e-9


@pytest.fixture
def flat_vertices_of() -> Callable[[object], np.ndarray]:
    """Return ``_flat_vertices``: which vertices of a mesh lie on a flat patch."""
    return _flat_vertices


@pytest.fixture
def shared_edge_masks() -> dict[str, np.ndarray]:
    """Masks whose meshes have edges shared by four faces, with that count.

    The ragged balls at 10% and 30%, a random 8^3 mask at 35% fill, and the
    largest 26-connected piece of noise blurred at sigma 0.7 and thresholded at
    its top 10%, cropped to its bounding box. The last two vanish under the
    default curvature blur.
    """
    rng = np.random.default_rng(0)
    noise = ndimage.gaussian_filter(rng.normal(size=(32, 32, 32)), 0.7)
    blob = _largest_piece(noise > np.quantile(noise, 0.9))
    return {
        "ragged ball 10%": _ragged_ball(0.1),
        "ragged ball 30%": _ragged_ball(0.3),
        "random 8^3": np.random.default_rng(0).random((8, 8, 8)) < 0.35,
        "blurred noise": blob[ndimage.find_objects(blob.astype(int))[0]],
    }
