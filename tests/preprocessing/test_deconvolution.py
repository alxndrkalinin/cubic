"""Tests for Richardson-Lucy deconvolution wrappers."""

import numpy as np
import pytest

from cubic.cuda import asnumpy, to_device
from cubic.image_utils import pad_image_to_shape
from cubic.preprocessing.deconvolution import (
    decon_xpy,
    richardson_lucy_iter,
    deconv_iter_num_finder,
    richardson_lucy_skimage,
)
from cubic.preprocessing.richardson_lucy_xp import richardson_lucy_xp


def _gaussian_psf(
    shape: tuple[int, ...], sigmas: tuple[float, ...], dtype: type = np.float32
) -> np.ndarray:
    """Build a centered, anisotropic Gaussian PSF normalized to sum == 1."""
    coords = [np.arange(s) - (s - 1) / 2.0 for s in shape]
    grids = np.meshgrid(*coords, indexing="ij")
    d2 = sum((g / sig) ** 2 for g, sig in zip(grids, sigmas))
    psf = np.exp(-0.5 * d2).astype(dtype)
    return psf / psf.sum()


def _fft_convolve(image: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Circular FFT convolution of ``image`` with a centered ``psf``."""
    if psf.shape != image.shape:
        psf = pad_image_to_shape(psf, image.shape, mode="constant")
    otf = np.fft.fftn(np.fft.ifftshift(psf))
    return np.real(np.fft.ifftn(np.fft.fftn(image) * otf))


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


@pytest.mark.parametrize("device", ["CPU", "GPU"])
def test_richardson_lucy_xp_backprojector_runs(
    device: str, gpu_available: bool
) -> None:
    """An unmatched back projector runs, preserves shape, and stays finite."""
    if device == "GPU" and not gpu_available:
        pytest.skip("GPU not available")

    rng = np.random.default_rng(0)
    image = (rng.random((12, 18, 16)).astype(np.float32) * 100 + 5).astype(np.float32)
    psf = _gaussian_psf((7, 7, 7), sigmas=(1.5, 1.2, 1.2))
    bp = _gaussian_psf((7, 7, 7), sigmas=(2.0, 1.6, 1.6))

    image = to_device(image, device)
    out = richardson_lucy_xp(
        image, to_device(psf, device), n_iter=2, backprojector=to_device(bp, device)
    )
    assert out.shape == image.shape
    assert np.all(np.isfinite(asnumpy(out)))


def test_richardson_lucy_xp_backprojector_cpu_gpu_parity(gpu_available: bool) -> None:
    """The unmatched path matches between CPU and GPU."""
    if not gpu_available:
        pytest.skip("GPU not available")

    rng = np.random.default_rng(1)
    image = (rng.random((12, 18, 16)).astype(np.float32) * 100 + 5).astype(np.float32)
    psf = _gaussian_psf((7, 7, 7), sigmas=(1.5, 1.2, 1.2))
    bp = _gaussian_psf((7, 7, 7), sigmas=(2.0, 1.6, 1.6))

    cpu_out = richardson_lucy_xp(image, psf, n_iter=3, backprojector=bp)
    gpu_out = richardson_lucy_xp(
        to_device(image, "GPU"),
        to_device(psf, "GPU"),
        n_iter=3,
        backprojector=to_device(bp, "GPU"),
    )
    assert np.allclose(asnumpy(gpu_out), cpu_out, atol=1e-4)


def test_richardson_lucy_xp_unmatched_faster_convergence() -> None:
    """An unmatched WB back projector beats the matched path at equal iterations.

    A sharp phantom is blurred with the forward PSF and given mild Poisson
    noise. At a small, equal iteration count the unmatched (Wiener-Butterworth)
    update should restore the phantom more accurately than the matched RL path.
    """
    # create_backprojector belongs to the WB workstream; used only for this test.
    from cubic.preprocessing import create_backprojector

    rng = np.random.default_rng(7)
    phantom = np.zeros((16, 32, 32), dtype=np.float32)
    for z, y, x in [(8, 10, 12), (6, 22, 20), (10, 16, 24), (8, 24, 8)]:
        phantom[z, y, x] = 1000.0

    psf = _gaussian_psf((9, 11, 11), sigmas=(2.0, 1.6, 1.6))
    blurred = _fft_convolve(phantom, psf)
    blurred = np.maximum(blurred, 0.0)
    noisy = rng.poisson(blurred).astype(np.float32)

    bp_wb = create_backprojector(psf, "wiener-butterworth")

    n_iter = 2
    matched = richardson_lucy_xp(noisy, psf, n_iter=n_iter)
    unmatched = richardson_lucy_xp(noisy, psf, n_iter=n_iter, backprojector=bp_wb)

    mse_matched = float(np.mean((asnumpy(matched) - phantom) ** 2))
    mse_unmatched = float(np.mean((asnumpy(unmatched) - phantom) ** 2))

    assert np.all(np.isfinite(asnumpy(unmatched)))
    assert mse_unmatched < mse_matched


def test_richardson_lucy_xp_backprojector_noncirc_raises() -> None:
    """Non-circulant mode with a back projector is not implemented."""
    image = np.ones((6, 6, 6), dtype=np.float32)
    psf = _gaussian_psf((3, 3, 3), sigmas=(1.0, 1.0, 1.0))
    bp = _gaussian_psf((3, 3, 3), sigmas=(1.2, 1.2, 1.2))
    with pytest.raises(NotImplementedError):
        richardson_lucy_xp(image, psf, n_iter=1, backprojector=bp, noncirc=True)


def test_richardson_lucy_xp_backprojector_mask_raises() -> None:
    """A mask with a back projector is not implemented."""
    image = np.ones((6, 6, 6), dtype=np.float32)
    psf = _gaussian_psf((3, 3, 3), sigmas=(1.0, 1.0, 1.0))
    bp = _gaussian_psf((3, 3, 3), sigmas=(1.2, 1.2, 1.2))
    mask = (image > 0).astype(np.float32)
    with pytest.raises(NotImplementedError):
        richardson_lucy_xp(image, psf, n_iter=1, backprojector=bp, mask=mask)


def test_richardson_lucy_xp_backprojector_shape_mismatch_raises() -> None:
    """A back projector whose shape differs from the PSF raises ValueError."""
    image = np.ones((6, 6, 6), dtype=np.float32)
    psf = _gaussian_psf((3, 3, 3), sigmas=(1.0, 1.0, 1.0))
    bp = _gaussian_psf((5, 5, 5), sigmas=(1.2, 1.2, 1.2))
    with pytest.raises(ValueError, match="must match psf shape"):
        richardson_lucy_xp(image, psf, n_iter=1, backprojector=bp)


def test_richardson_lucy_iter_skimage_backprojector_raises() -> None:
    """The skimage implementation rejects a back projector."""
    image = np.ones((6, 6, 6), dtype=np.float32)
    psf = _gaussian_psf((3, 3, 3), sigmas=(1.0, 1.0, 1.0))
    bp = _gaussian_psf((3, 3, 3), sigmas=(1.2, 1.2, 1.2))
    with pytest.raises(ValueError, match="only supported with implementation='xp'"):
        richardson_lucy_iter(image, psf, implementation="skimage", backprojector=bp)


def test_deconv_iter_num_finder_skimage_backprojector_raises() -> None:
    """deconv_iter_num_finder rejects a back projector with skimage."""
    image = np.ones((6, 6, 6), dtype=np.float32)
    psf = _gaussian_psf((3, 3, 3), sigmas=(1.0, 1.0, 1.0))
    bp = _gaussian_psf((3, 3, 3), sigmas=(1.2, 1.2, 1.2))

    def metric_fn(prev, cur):
        return 0.0

    with pytest.raises(ValueError, match="only supported with implementation='xpy'"):
        deconv_iter_num_finder(
            image,
            psf,
            metric_fn,
            0.5,
            implementation="skimage",
            backprojector=bp,
        )


def test_decon_xpy_backprojector_pad_psf_raises() -> None:
    """A back projector with pad_psf=True is rejected with a clear error."""
    image = np.ones((6, 6, 6), dtype=np.float32)
    psf = _gaussian_psf((3, 3, 3), sigmas=(1.0, 1.0, 1.0))
    bp = _gaussian_psf((3, 3, 3), sigmas=(1.2, 1.2, 1.2))
    with pytest.raises(ValueError, match="pad_psf=True is not supported"):
        decon_xpy(image, psf, n_iter=1, pad_psf=True, backprojector=bp)
