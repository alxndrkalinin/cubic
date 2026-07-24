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
    with pytest.raises(ValueError, match="only supported with the NumPy/CuPy"):
        richardson_lucy_iter(image, psf, implementation="skimage", backprojector=bp)


def test_deconv_iter_num_finder_skimage_backprojector_raises() -> None:
    """deconv_iter_num_finder rejects a back projector with skimage."""
    image = np.ones((6, 6, 6), dtype=np.float32)
    psf = _gaussian_psf((3, 3, 3), sigmas=(1.0, 1.0, 1.0))
    bp = _gaussian_psf((3, 3, 3), sigmas=(1.2, 1.2, 1.2))

    def metric_fn(prev, cur):
        return 0.0

    with pytest.raises(ValueError, match="only supported with the NumPy/CuPy"):
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


# --- Caller-array safety -----------------------------------------------------


def test_richardson_lucy_xp_mask_does_not_mutate_caller() -> None:
    """The masked path must not zero the caller's image in place.

    ``util.img_as_float`` returns the *same object* for float input, so
    ``image *= mask`` wrote straight into the caller's array.
    """
    image = np.ones((4, 6, 6), dtype=np.float64)
    psf = np.zeros((3, 3, 3), dtype=np.float64)
    psf[1, 1, 1] = 1.0
    mask = np.ones_like(image)
    mask[0] = 0.0
    original = image.copy()

    richardson_lucy_xp(image, psf, n_iter=2, mask=mask)
    assert np.array_equal(image, original)


# --- Shape validation --------------------------------------------------------


@pytest.mark.parametrize("unmatched", [False, True], ids=["matched", "unmatched"])
def test_richardson_lucy_xp_psf_larger_than_image_raises(unmatched: bool) -> None:
    """An oversized PSF raises a ValueError naming both shapes.

    ``pad_image_to_shape`` only pads, so this used to trip its bare assert.
    """
    image = np.ones((4, 4, 4), dtype=np.float64)
    psf = _gaussian_psf((7, 7, 7), sigmas=(1.0, 1.0, 1.0)).astype(np.float64)
    kwargs = {"backprojector": psf.copy()} if unmatched else {}
    with pytest.raises(ValueError, match=r"\(7, 7, 7\).*\(4, 4, 4\)"):
        richardson_lucy_xp(image, psf, n_iter=1, **kwargs)


def test_richardson_lucy_xp_psf_ndim_mismatch_raises() -> None:
    """A PSF with the wrong number of axes is rejected explicitly."""
    image = np.ones((8, 8, 8), dtype=np.float64)
    psf = np.ones((3, 3), dtype=np.float64) / 9.0
    with pytest.raises(ValueError, match="same number of axes"):
        richardson_lucy_xp(image, psf, n_iter=1)


def test_richardson_lucy_xp_mask_shape_mismatch_raises() -> None:
    """A mask that is not image-shaped is rejected before broadcasting fails."""
    image = np.ones((6, 6, 6), dtype=np.float64)
    psf = _gaussian_psf((3, 3, 3), sigmas=(1.0, 1.0, 1.0)).astype(np.float64)
    with pytest.raises(ValueError, match="must equal"):
        richardson_lucy_xp(image, psf, n_iter=1, mask=np.ones((4, 6, 6)))


def test_decon_xpy_mask_with_pad_size_z() -> None:
    """``mask`` is padded alongside the image, so pad_size_z > 0 works.

    The mask used to be forwarded unpadded, which broke broadcasting inside
    :func:`richardson_lucy_xp`.
    """
    rng = np.random.default_rng(3)
    image = rng.random((6, 8, 8)) + 0.5
    psf = _gaussian_psf((3, 3, 3), sigmas=(1.0, 1.0, 1.0)).astype(np.float64)
    mask = np.ones_like(image)
    mask[:, :2] = 0.0

    out = decon_xpy(image, psf, n_iter=2, pad_size_z=2, mask=mask)
    assert out.shape == image.shape
    assert np.all(np.isfinite(out))
    # Masked-out voxels are restored from the original image.
    assert np.allclose(out[:, :2], image[:, :2])
    # An all-ones mask reduces to the unmasked result.
    out_trivial = decon_xpy(
        image, psf, n_iter=2, pad_size_z=2, mask=np.ones_like(image)
    )
    out_nomask = decon_xpy(image, psf, n_iter=2, pad_size_z=2)
    assert np.allclose(out_trivial, out_nomask)


# --- small_value floor on the unmatched path ---------------------------------


def test_unmatched_small_value_scales_with_data() -> None:
    """The default floor scales with the data instead of being an absolute 1e-3.

    ``img_as_float`` maps uint16 into [0, 1], where the reference's absolute
    ``1e-3`` equals 65.5 counts and clamps a large fraction of the image.
    """
    rng = np.random.default_rng(0)
    image = rng.integers(0, 201, size=(8, 12, 12)).astype(np.uint16)
    psf = _gaussian_psf((5, 5, 5), sigmas=(1.2, 1.2, 1.2))
    bp = _gaussian_psf((5, 5, 5), sigmas=(1.6, 1.6, 1.6))

    scaled = richardson_lucy_xp(image, psf, n_iter=1, backprojector=bp)
    absolute = richardson_lucy_xp(
        image, psf, n_iter=1, backprojector=bp, small_value=1e-3
    )

    expected_floor = 1e-6 * (200.0 / 65535.0)
    assert float(np.asarray(scaled).min()) == pytest.approx(expected_floor, rel=1e-6)
    # The absolute floor pins a large share of the output; the scaled one does not.
    assert float((np.asarray(absolute) == 1e-3).mean()) > 0.2
    assert float((np.asarray(scaled) == expected_floor).mean()) < 0.01


def test_small_value_without_backprojector_raises() -> None:
    """``small_value`` is meaningless on the matched path, so it is rejected."""
    image = np.ones((6, 6, 6), dtype=np.float64)
    psf = _gaussian_psf((3, 3, 3), sigmas=(1.0, 1.0, 1.0)).astype(np.float64)
    with pytest.raises(ValueError, match="unmatched back-projector path only"):
        richardson_lucy_xp(image, psf, n_iter=1, small_value=1e-6)


@pytest.mark.parametrize("bad", ["zeros", "inf"])
def test_unmatched_degenerate_image_raises(bad: str) -> None:
    """A zero or non-finite floor cannot drive the epsilon-free ratio.

    ``small_value`` defaults to ``1e-6 * image.max()``. For an all-zero image
    that is 0; for an image containing ``inf`` it is ``inf``, which passed a
    bare ``> 0`` check and turned the whole volume into ``inf`` and then NaN
    with no exception.
    """
    image = np.zeros((6, 6, 6), dtype=np.float64)
    if bad == "inf":
        image = np.ones((6, 6, 6), dtype=np.float64)
        image[0, 0, 0] = np.inf
    psf = _gaussian_psf((3, 3, 3), sigmas=(1.0, 1.0, 1.0)).astype(np.float64)
    with pytest.raises(ValueError, match="small_value must be finite and > 0"):
        richardson_lucy_xp(image, psf, n_iter=1, backprojector=psf.copy())


# --- skimage observer parity --------------------------------------------------


@pytest.mark.parametrize("n_iter", [0, 1, 3])
def test_richardson_lucy_skimage_observer_matches_no_observer(n_iter: int) -> None:
    """The observer path returns the no-observer result, including at n_iter=0.

    With ``n_iter=0`` the observer path returned ``image`` untouched while the
    no-observer path returned skimage's constant initial estimate.
    """
    rng = np.random.default_rng(5)
    image = rng.random((12, 12))
    psf = np.zeros((3, 3))
    psf[1, 1] = 1.0

    out_obs = richardson_lucy_skimage(
        image, psf, n_iter=n_iter, observer_fn=lambda est, i: None
    )
    out_plain = richardson_lucy_skimage(image, psf, n_iter=n_iter)
    assert np.allclose(out_obs, out_plain)


# --- Implementation aliases ---------------------------------------------------


@pytest.mark.parametrize("implementation", ["xp", "xpy"])
def test_implementation_aliases_accepted(implementation: str) -> None:
    """Both entry points accept ``"xp"`` and ``"xpy"`` for the NumPy/CuPy backend."""
    image = np.ones((6, 6, 6), dtype=np.float64) * 0.5
    psf = _gaussian_psf((3, 3, 3), sigmas=(1.0, 1.0, 1.0)).astype(np.float64)

    out = richardson_lucy_iter(image, psf, n_iter=1, implementation=implementation)
    assert out.shape == image.shape

    thresh_iter, results = deconv_iter_num_finder(
        image,
        psf,
        lambda prev, cur: 0.0,
        1.0,
        max_iter=2,
        pad_size_z=0,
        implementation=implementation,
    )
    assert thresh_iter == 0
    assert len(results) == 3


@pytest.mark.parametrize(
    "call",
    [
        lambda image, psf: richardson_lucy_iter(image, psf, implementation="bogus"),
        lambda image, psf: deconv_iter_num_finder(
            image, psf, lambda a, b: 0.0, 1.0, implementation="bogus"
        ),
    ],
    ids=["richardson_lucy_iter", "deconv_iter_num_finder"],
)
def test_unknown_implementation_raises(call) -> None:
    """An unknown implementation name is rejected by both entry points."""
    image = np.ones((6, 6, 6), dtype=np.float64)
    psf = _gaussian_psf((3, 3, 3), sigmas=(1.0, 1.0, 1.0)).astype(np.float64)
    with pytest.raises(ValueError, match="Unknown implementation"):
        call(image, psf)


# --- deconv_iter_num_finder contract -----------------------------------------


def test_deconv_iter_num_finder_aborts_at_threshold(monkeypatch) -> None:
    """Hitting the threshold stops the deconvolution instead of the observer only.

    The observer used to return early while the RL loop kept running to
    ``max_iter``, computing iterations that were then discarded.
    """
    import cubic.preprocessing.deconvolution as decon_module

    real_make = decon_module._make_unpad_observer
    observed: list[int] = []

    def counting_make(observer_fn, pad_size_z, orig_size_z):
        wrapped = real_make(observer_fn, pad_size_z, orig_size_z)

        def counting(restored_image, i, *args):
            observed.append(i)
            return wrapped(restored_image, i, *args)

        return counting

    monkeypatch.setattr(decon_module, "_make_unpad_observer", counting_make)

    image = np.ones((6, 8, 8), dtype=np.float64) * 0.5
    psf = _gaussian_psf((3, 3, 3), sigmas=(1.0, 1.0, 1.0)).astype(np.float64)

    thresh_iter, results = deconv_iter_num_finder(
        image, psf, lambda prev, cur: 1.0, 0.5, max_iter=25, pad_size_z=0
    )

    assert thresh_iter == 2
    assert observed == [1, 2]  # iterations 3..25 never ran
    assert len(results) == 3


def test_deconv_iter_num_finder_results_schema() -> None:
    """``results`` entries follow the documented schema."""
    image = np.ones((6, 8, 8), dtype=np.float64) * 0.5
    psf = _gaussian_psf((3, 3, 3), sigmas=(1.0, 1.0, 1.0)).astype(np.float64)

    thresh_iter, results = deconv_iter_num_finder(
        image,
        psf,
        lambda prev, cur: (0.25, "extra"),
        0.5,
        max_iter=3,
        pad_size_z=1,
    )

    assert thresh_iter == 0  # 0.25 never exceeds the 0.5 threshold
    assert len(results) == 4
    # Entry 0 is the input image: sentinel gain, no metric_result.
    assert results[0]["metric_gain"] == 0.5
    assert "metric_result" not in results[0]
    for entry in results:
        assert isinstance(entry["iter_image"], np.ndarray)
        assert np.asarray(entry["iter_image"]).shape == image.shape
    for entry in results[1:]:
        assert entry["metric_gain"] == 0.25
        assert entry["metric_result"] == (0.25, "extra")


def test_richardson_lucy_iter_exported_from_package() -> None:
    """``richardson_lucy_iter`` is the documented entry point, so it is exported."""
    import cubic.preprocessing as preprocessing

    assert "richardson_lucy_iter" in preprocessing.__all__
    assert preprocessing.richardson_lucy_iter is richardson_lucy_iter


# --- Even-shaped PSF back projector ------------------------------------------


def test_even_psf_backprojector_is_not_shifted() -> None:
    """An even-shaped PSF's back projector must stay registered to the forward PSF.

    A plain ``psf[::-1]`` shifts the flipped PSF by one voxel per even-sized
    axis. Reconstruction with such a shifted back projector is measurably worse
    than with the correctly centered one.
    """
    from cubic.preprocessing import create_backprojector

    shape = (8, 8, 8)
    coords = [np.arange(s) - s // 2 for s in shape]
    grids = np.meshgrid(*coords, indexing="ij")
    d2 = sum((g / sig) ** 2 for g, sig in zip(grids, (1.6, 1.2, 1.0)))
    offs = [np.arange(s) - s // 2 - o for s, o in zip(shape, (1, 1, 2))]
    grids_off = np.meshgrid(*offs, indexing="ij")
    d2_off = sum((g / sig) ** 2 for g, sig in zip(grids_off, (1.6, 1.2, 1.0)))
    psf = np.exp(-0.5 * d2) + 0.6 * np.exp(-0.5 * d2_off)
    psf = psf / psf.sum()

    phantom = np.zeros((16, 16, 16))
    for z, y, x in [(8, 7, 6), (5, 11, 10), (11, 4, 12)]:
        phantom[z, y, x] = 1000.0
    blurred = np.maximum(_fft_convolve(phantom, psf), 0.0) + 1.0

    bp = create_backprojector(psf, "traditional")
    bp_shifted = np.roll(bp, -1, axis=(0, 1, 2))

    mse_correct = float(
        np.mean(
            (asnumpy(richardson_lucy_xp(blurred, psf, 30, backprojector=bp)) - phantom)
            ** 2
        )
    )
    mse_shifted = float(
        np.mean(
            (
                asnumpy(richardson_lucy_xp(blurred, psf, 30, backprojector=bp_shifted))
                - phantom
            )
            ** 2
        )
    )
    assert mse_correct < 0.9 * mse_shifted


# --- Matched path CPU/GPU parity ---------------------------------------------


@pytest.mark.parametrize("noncirc", [False, True])
@pytest.mark.parametrize("use_mask", [False, True], ids=["nomask", "mask"])
def test_richardson_lucy_xp_matched_cpu_gpu_parity(
    noncirc: bool, use_mask: bool, gpu_available: bool
) -> None:
    """The matched path matches between CPU and GPU (NumPy dispatch to CuPy)."""
    if not gpu_available:
        pytest.skip("GPU not available")
    if noncirc and use_mask:
        pytest.skip("noncirc with a mask is not a supported combination")

    rng = np.random.default_rng(11)
    image = rng.random((10, 16, 16)) + 0.25
    psf = _gaussian_psf((5, 5, 5), sigmas=(1.4, 1.1, 1.1)).astype(np.float64)
    mask = np.ones_like(image)
    mask[:, :3] = 0.0

    kwargs = {"mask": mask} if use_mask else {}
    cpu_out = richardson_lucy_xp(image, psf, n_iter=3, noncirc=noncirc, **kwargs)
    gpu_kwargs = {"mask": to_device(mask, "GPU")} if use_mask else {}
    gpu_out = richardson_lucy_xp(
        to_device(image, "GPU"),
        to_device(psf, "GPU"),
        n_iter=3,
        noncirc=noncirc,
        **gpu_kwargs,
    )
    assert np.allclose(asnumpy(gpu_out), cpu_out, atol=1e-8)
