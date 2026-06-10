"""Tests for Richardson-Lucy deconvolution wrappers."""

import numpy as np
import pytest

from cubic.preprocessing.deconvolution import (
    richardson_lucy_iter,
    richardson_lucy_skimage,
)
from cubic.preprocessing.richardson_lucy_xp import richardson_lucy_xp


def test_richardson_lucy_skimage_observer_matches_single_call() -> None:
    """The observer path returns the same result as the no-observer call.

    Previously it looped ``num_iter=1`` on the previous output, which re-clips
    every iteration and feeds the deconvolution back as the image, diverging
    from a single ``num_iter=n`` call.
    """
    rng = np.random.default_rng(0)
    image = rng.random((16, 16)).astype(np.float64)
    psf = np.zeros((3, 3), dtype=np.float64)
    psf[1, 1] = 1.0

    snapshots: list[np.ndarray] = []
    out_obs = richardson_lucy_skimage(
        image, psf, n_iter=4, observer_fn=lambda est, i: snapshots.append(est.copy())
    )
    out_single = richardson_lucy_skimage(image, psf, n_iter=4)

    assert len(snapshots) == 4
    assert np.allclose(out_obs, out_single)
    assert np.allclose(snapshots[-1], out_single)


@pytest.mark.parametrize("implementation", ["xp", "skimage"])
def test_decon_iter_2d_input(implementation: str) -> None:
    """2-D images run through the deconvolution wrappers (rank-agnostic slice)."""
    image = np.zeros((8, 8), dtype=np.float32)
    image[4, 4] = 1.0
    psf = np.ones((3, 3), dtype=np.float32) / 9.0
    out = richardson_lucy_iter(
        image, psf, n_iter=2, implementation=implementation, pad_size_z=0
    )
    assert out.shape == image.shape
    assert np.all(np.isfinite(out))


def test_richardson_lucy_iter() -> None:
    """Ensure both implementations run and preserve shape."""
    image = np.zeros((3, 3, 3), dtype=np.float32)
    image[1, 1, 1] = 1.0
    psf = np.ones((3, 3, 3), dtype=np.float32) / 27.0
    res_xp = richardson_lucy_iter(image, psf, n_iter=1, implementation="xp")
    res_sk = richardson_lucy_iter(image, psf, n_iter=1, implementation="skimage")
    assert res_xp.shape == image.shape
    assert res_sk.shape == image.shape


@pytest.mark.parametrize("noncirc", [False, True])
def test_richardson_lucy_xp_psf_smaller_than_image_odd_diff(noncirc: bool) -> None:
    """A smaller PSF with odd size differences pads to the image without error.

    The previous ``pad_image_to_shape`` padded symmetrically by ``diff // 2`` on
    both sides, falling one element short for odd diffs and tripping its assert.
    """
    rng = np.random.default_rng(0)
    image = rng.random((10, 32, 32)).astype(np.float64)
    psf = np.zeros((5, 7, 7), dtype=np.float64)
    psf[2, 3, 3] = 1.0  # centered delta -> output approximates the input

    out = richardson_lucy_xp(image, psf, n_iter=3, noncirc=noncirc)
    assert out.shape == image.shape
    assert np.all(np.isfinite(out))
