"""Fixtures shared by the feature tests."""

import numpy as np
import pytest
from scipy import ndimage


@pytest.fixture
def ragged_ball() -> np.ndarray:
    """Return a 41^3 mask of a radius-15 ball with 30% of its boundary shell flipped.

    Only the largest 26-connected piece is kept. Its mesh has 33 edges shared
    by four faces, so trimesh does not call it watertight.
    """
    rng = np.random.default_rng(0)
    z, y, x = np.meshgrid(*[np.arange(41) - 20.0] * 3, indexing="ij")
    r = np.sqrt(z**2 + y**2 + x**2)
    shell = (np.abs(r - 15) <= 1.0) & (rng.random(r.shape) < 0.3)
    pieces = ndimage.label((r <= 15) ^ shell, structure=np.ones((3, 3, 3)))[0]
    return pieces == np.argmax(np.bincount(pieces.ravel())[1:]) + 1
