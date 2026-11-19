"""Fixtures shared by the feature tests."""

from collections.abc import Callable

import numpy as np
import pytest
from scipy import ndimage


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


@pytest.fixture
def noise_blob() -> np.ndarray:
    """Return a lobed mask: blurred noise cut at its top 30%, as one thick piece.

    32^3 normal noise (seed 8) blurred at sigma 4, thresholded at its 70th
    percentile, the largest 26-connected piece cropped to its bounding box:
    about 8800 voxels and 4400 mesh vertices, thick enough to survive the
    default curvature blur at the spacing (0.29, 0.108, 0.108). At that spacing
    the sum of its face areas differs by an ulp between NumPy and CuPy.
    """
    noise = ndimage.gaussian_filter(
        np.random.default_rng(8).normal(size=(32, 32, 32)), 4.0
    )
    blob = _largest_piece(noise > np.quantile(noise, 0.7))
    return blob[ndimage.find_objects(blob.astype(int))[0]]


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
