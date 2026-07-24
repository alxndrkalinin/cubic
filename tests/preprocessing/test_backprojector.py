"""Tests for the Wiener-Butterworth back-projector PSF generator."""

import numpy as np
import pytest

from cubic.cuda import asnumpy, to_device
from cubic.preprocessing.backprojector import (
    _fwhm_psf,
    _centered_grid,
    _butterworth_mask,
    _endpoint_indices,
    create_backprojector,
)

_BP_TYPES = ("traditional", "gaussian", "butterworth", "wiener", "wiener-butterworth")


def _gaussian_psf(
    shape: tuple[int, ...], sigmas: tuple[float, ...], dtype: type = np.float32
) -> np.ndarray:
    """Build a centered, anisotropic Gaussian PSF normalized to sum == 1."""
    coords = [np.arange(s) - (s - 1) / 2.0 for s in shape]
    grids = np.meshgrid(*coords, indexing="ij")
    d2 = sum((g / sig) ** 2 for g, sig in zip(grids, sigmas))
    psf = np.exp(-0.5 * d2).astype(dtype)
    return psf / psf.sum()


def _fftshift_centered_psf(
    shape: tuple[int, ...], sigmas: tuple[float, ...], dtype: type = np.float64
) -> np.ndarray:
    """Gaussian PSF centered on index ``shape[d] // 2`` (the fftshift origin).

    Such a PSF is even-symmetric in the wrapped sense (``x[k] == x[N - k]``) for
    both odd and even sizes, so ``fftn(ifftshift(psf))`` is purely real.
    """
    coords = [np.arange(s) - s // 2 for s in shape]
    grids = np.meshgrid(*coords, indexing="ij")
    d2 = sum((g / sig) ** 2 for g, sig in zip(grids, sigmas))
    psf = np.exp(-0.5 * d2).astype(dtype)
    return psf / psf.sum()


def _asymmetric_psf(shape: tuple[int, ...], dtype: type = np.float64) -> np.ndarray:
    """Two offset Gaussians, so flips and centering conventions are observable."""
    psf = _fftshift_centered_psf(shape, (2.5, 1.5, 1.2), dtype=dtype)
    coords = [np.arange(s) - s // 2 - off for s, off in zip(shape, (1, 2, 1))]
    grids = np.meshgrid(*coords, indexing="ij")
    d2 = sum((g / sig) ** 2 for g, sig in zip(grids, (2.5, 1.5, 1.2)))
    psf = psf + 0.3 * np.exp(-0.5 * d2).astype(dtype)
    return psf / psf.sum()


def _psf_3d() -> np.ndarray:
    """Anisotropic 3-D (ZYX) Gaussian PSF: broader along Z, tighter in XY."""
    return _gaussian_psf((15, 21, 21), sigmas=(2.5, 1.5, 1.2))


def _psf_2d() -> np.ndarray:
    """Anisotropic 2-D (YX) Gaussian PSF."""
    return _gaussian_psf((21, 25), sigmas=(1.6, 1.2))


def _passband_cv(psf: np.ndarray, bp: np.ndarray) -> float:
    """Coefficient of variation of the combined |OTF_f * OTF_bp| over passband.

    The passband is where the forward OTF magnitude exceeds 10% of its max.
    A flatter combined transfer function (the WB design goal) yields a lower CV.
    """
    psf_n = psf / psf.sum()
    otf_f = np.fft.fftn(np.fft.ifftshift(psf_n))
    otf_bp = np.fft.fftn(np.fft.ifftshift(bp))
    mag_f = np.fft.fftshift(np.abs(otf_f))
    combined = np.fft.fftshift(np.abs(otf_f * otf_bp))
    mask = mag_f > 0.1 * mag_f.max()
    vals = combined[mask]
    return float(vals.std() / vals.mean())


@pytest.mark.parametrize("bp_type", _BP_TYPES)
@pytest.mark.parametrize("psf_fn", [_psf_2d, _psf_3d], ids=["2d", "3d"])
def test_shape_finite_normalized(bp_type: str, psf_fn) -> None:
    """Every bp_type yields a finite, same-shape PSF normalized to sum 1."""
    psf = psf_fn()
    bp = create_backprojector(psf, bp_type)
    assert bp.shape == psf.shape
    assert bp.dtype == psf.dtype
    assert np.all(np.isfinite(bp))
    assert bp.sum() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("psf_fn", [_psf_2d, _psf_3d], ids=["2d", "3d"])
def test_traditional_is_normalized_flip(psf_fn) -> None:
    """The traditional back projector equals the normalized flipped PSF."""
    psf = psf_fn()
    bp = create_backprojector(psf, "traditional")
    ndim = psf.ndim
    flipped = psf[(slice(None, None, -1),) * ndim]
    expected = flipped / flipped.sum()
    assert np.allclose(bp, expected, atol=1e-6)


@pytest.mark.parametrize("psf_fn", [_psf_2d, _psf_3d], ids=["2d", "3d"])
def test_wb_flatter_than_matched(psf_fn) -> None:
    """WB combined transfer function is flatter than the matched back projector.

    Matched back projector: OTF_bp = conj(OTF_f), so the product is |OTF_f|^2,
    which is sharply peaked. The WB design intentionally flattens the passband.
    """
    psf = psf_fn()
    psf_n = psf / psf.sum()

    wb = create_backprojector(psf, "wiener-butterworth")
    cv_wb = _passband_cv(psf, wb)

    # Matched back projector PSF whose OTF is conj(OTF_f).
    otf_f = np.fft.fftn(np.fft.ifftshift(psf_n))
    matched_bp = np.fft.fftshift(np.real(np.fft.ifftn(np.conj(otf_f))))
    cv_matched = _passband_cv(psf, matched_bp)

    assert cv_wb < cv_matched


@pytest.mark.parametrize("bp_type", _BP_TYPES)
@pytest.mark.parametrize("shape", [(15, 21, 21), (16, 20, 20)], ids=["odd", "even"])
def test_cpu_gpu_parity(
    bp_type: str, shape: tuple[int, ...], gpu_available: bool
) -> None:
    """GPU result matches the CPU result for each back-projector type."""
    if not gpu_available:
        pytest.skip("GPU not available")
    psf = _asymmetric_psf(shape, dtype=np.float32)
    cpu_out = create_backprojector(psf, bp_type)
    gpu_out = create_backprojector(to_device(psf, "GPU"), bp_type)
    assert np.allclose(asnumpy(gpu_out), cpu_out, atol=1e-4)


def test_unknown_bp_type_raises() -> None:
    """An unrecognized bp_type raises ValueError."""
    with pytest.raises(ValueError, match="Unknown bp_type"):
        create_backprojector(_psf_2d(), "nonexistent")


@pytest.mark.parametrize("ndim", [1, 4])
def test_bad_ndim_raises(ndim: int) -> None:
    """PSFs that are not 2-D or 3-D raise ValueError."""
    psf = np.ones((5,) * ndim, dtype=np.float32)
    psf = psf / psf.sum()
    with pytest.raises(ValueError, match="2-D or 3-D"):
        create_backprojector(psf, "wiener-butterworth")


def test_res_flag2_requires_i_res() -> None:
    """res_flag=2 without i_res raises ValueError."""
    with pytest.raises(ValueError, match="res_flag=2 requires i_res"):
        create_backprojector(_psf_3d(), "butterworth", res_flag=2, i_res=None)


def test_fwhm_1d_no_trailing_crossing_returns_nan() -> None:
    """A profile with no second half-max crossing yields NaN, not IndexError."""
    from cubic.preprocessing.backprojector import _fwhm_1d

    assert np.isnan(_fwhm_1d(np.arange(10.0)))  # monotonic rising
    assert np.isnan(_fwhm_1d(np.linspace(1.0, 0.0, 10)))  # monotonic falling


@pytest.mark.parametrize(
    "psf",
    [
        np.zeros((7, 7, 7), dtype=np.float32),  # zero sum
        np.full((7, 7, 7), np.nan, dtype=np.float32),  # non-finite sum
    ],
    ids=["zero_sum", "nan_sum"],
)
def test_create_backprojector_unnormalizable_psf_raises(psf: np.ndarray) -> None:
    """A PSF that cannot be sum-normalized raises before producing NaNs."""
    with pytest.raises(ValueError, match="finite, non-zero sum"):
        create_backprojector(psf, "wiener-butterworth")


def test_create_backprojector_unresolvable_psf_raises() -> None:
    """A corner-peaked PSF (no FWHM crossing through the peak) raises clearly."""
    z, y, x = np.mgrid[0:8, 0:8, 0:8]
    psf = np.exp(-(z + y + x).astype(np.float32))
    psf /= psf.sum()
    with pytest.raises(ValueError, match="FWHM"):
        create_backprojector(psf, "wiener-butterworth")


def _corner_psf() -> np.ndarray:
    """Build a corner-peaked PSF whose FWHM cannot be measured (even-shaped)."""
    z, y, x = np.mgrid[0:8, 0:8, 0:8]
    psf = np.exp(-(z + y + x).astype(np.float32))
    return psf / psf.sum()


# --- Centering convention (fftshift origin at shape // 2) --------------------


@pytest.mark.parametrize("size", [15, 16], ids=["odd", "even"])
def test_butterworth_mask_ifft_is_real(size: int) -> None:
    """The DC-centered Butterworth mask is even in frequency, so its ifft is real.

    With the reference's ``(S - 1) / 2`` origin the mask is asymmetric about DC
    for even sizes, ``ifftn`` picks up a large imaginary part, and
    ``np.real(...)`` silently discards it.
    """
    shape = (size, size, size)
    reference = np.zeros(shape, dtype=np.float64)
    mask = _butterworth_mask(shape, (5.0, 5.0, 5.0), 3.0, 10, reference)
    out = np.fft.ifftn(np.fft.ifftshift(mask))
    assert np.abs(out.imag).max() / np.abs(out.real).max() < 1e-12


@pytest.mark.parametrize("size", [15, 16], ids=["odd", "even"])
def test_traditional_bp_otf_is_conjugate_of_forward_otf(size: int) -> None:
    """``traditional`` must satisfy ``OTF_bp == conj(OTF_f)`` exactly.

    A plain ``psf[::-1]`` maps centered index ``j -> -j - 1`` on even-sized axes,
    which shifts the back projector by one voxel per even axis.
    """
    shape = (size, size, size)
    psf = _asymmetric_psf(shape)
    bp = create_backprojector(psf, "traditional")

    otf_f = np.fft.fftn(np.fft.ifftshift(psf))
    otf_bp = np.fft.fftn(np.fft.ifftshift(bp))
    rel = float(np.abs(otf_bp - np.conj(otf_f)).max() / np.abs(otf_f).max())
    assert rel < 1e-12


@pytest.mark.parametrize("bp_type", _BP_TYPES)
@pytest.mark.parametrize("size", [15, 16], ids=["odd", "even"])
def test_symmetric_psf_yields_real_bp_otf(bp_type: str, size: int) -> None:
    """A PSF symmetric about ``shape // 2`` gives a back projector with a real OTF.

    Any residual imaginary part is a linear phase ramp, i.e. a sub-voxel or
    one-voxel misregistration of the back projector against the forward PSF.
    """
    shape = (size, size, size)
    psf = _fftshift_centered_psf(shape, (2.5, 1.5, 1.2))
    bp = create_backprojector(psf, bp_type)
    otf_bp = np.fft.fftn(np.fft.ifftshift(bp))
    assert np.abs(otf_bp.imag).max() <= 1e-10 * np.abs(otf_bp).max()


@pytest.mark.parametrize("size", [15, 21])
def test_odd_shapes_match_reference_conventions(size: int) -> None:
    """For odd sizes the fftshift origin coincides with the reference's (S-1)/2.

    Pins the three sites changed for even shapes -- the centered grid, the PSF
    flip, and the cutoff endpoint indices -- as bit-identical to the MATLAB
    reference whenever every axis is odd.
    """
    shape = (size, size, size)
    psf = _asymmetric_psf(shape)

    for d, grid in enumerate(_centered_grid(shape, psf)):
        axis_shape = [1, 1, 1]
        axis_shape[d] = size
        reference = (np.arange(size) - (size - 1) / 2.0).reshape(axis_shape)
        assert np.array_equal(grid, reference)

    # The reference flip is a plain reverse -- no roll on any axis.
    flipped = (psf / psf.sum())[::-1, ::-1, ::-1]
    assert np.array_equal(
        create_backprojector(psf, "traditional"), flipped / flipped.sum()
    )

    for t in (1.0, 2.5, 3.7, size / 2.0):
        center = (size - 1) / 2.0
        reference_pair = (
            min(max(int(np.floor(center - t + 0.5)), 0), size - 1),
            min(max(int(np.floor(center + t + 0.5)), 0), size - 1),
        )
        assert _endpoint_indices(size, t) == reference_pair


@pytest.mark.parametrize(
    "size,t,expected",
    [
        (15, 2.5, (5, 10)),  # odd: fftshift origin == the reference's (S-1)/2
        (16, 0.0, (8, 8)),
        (16, 2.5, (6, 11)),  # even: (S-1)/2 would give (5, 10)
        (20, 3.7, (6, 14)),  # even: (S-1)/2 would give (6, 13)
    ],
)
def test_endpoint_indices_use_the_fftshift_origin(
    size: int, t: float, expected: tuple[int, int]
) -> None:
    """Cutoff endpoints are measured from the fftshift DC index, ``size // 2``.

    The odd-size assertions in ``test_odd_shapes_match_reference_conventions``
    cannot catch a regression here, because ``size // 2`` and ``(size - 1) / 2``
    round to the same index for odd sizes. These even sizes separate them, and
    the endpoints set ``beta_fp`` and ``beta_wiener``, so a wrong origin
    silently reshapes the Wiener and Butterworth filters for every even PSF.
    """
    assert _endpoint_indices(size, t) == expected

    # The t=0 endpoint must land exactly on the fftshift DC bin.
    dc = np.zeros(size)
    dc[0] = 1.0
    assert _endpoint_indices(size, 0.0)[0] == int(np.argmax(np.fft.fftshift(dc)))


# --- res_flag / i_res drive the gaussian projector ---------------------------


def test_gaussian_bp_width_follows_res_flag() -> None:
    """The gaussian projector's FWHM tracks ``res``, not always the PSF FWHM.

    ``sigma`` was hard-wired to ``fwhm / 2.3548``, so ``res_flag`` and ``i_res``
    had no effect at all and the iSIM mode (``res_flag=0``) was sqrt(2) too wide.
    """
    # Generously sampled PSF, so the discrete FWHM measurement stays accurate.
    shape = (33, 33, 33)
    psf = _fftshift_centered_psf(shape, (4.0, 3.0, 2.5))
    fwhm = _fwhm_psf(psf)

    bp_fwhm = create_backprojector(psf, "gaussian", res_flag=1)
    bp_isim = create_backprojector(psf, "gaussian", res_flag=0)
    bp_explicit = create_backprojector(
        psf, "gaussian", res_flag=2, i_res=(6.0, 8.0, 10.0)
    )

    # The three modes must actually differ (they were bit-identical before).
    assert not np.array_equal(bp_fwhm, bp_isim)
    assert not np.array_equal(bp_fwhm, bp_explicit)

    # res_flag=1: sigma = FWHM / 2.3548, so the projector's own FWHM is the PSF's.
    # rtol accommodates the discrete FWHM measurement, not the sigma itself.
    assert np.allclose(_fwhm_psf(bp_fwhm), fwhm, rtol=2e-2)
    # res_flag=0 (iSIM): FWHM / sqrt(2); the measurement bias cancels in the ratio.
    assert np.allclose(
        np.asarray(_fwhm_psf(bp_isim)) / np.asarray(_fwhm_psf(bp_fwhm)),
        1.0 / np.sqrt(2.0),
        rtol=2e-2,
    )
    # res_flag=2: the requested resolution limits.
    assert np.allclose(_fwhm_psf(bp_explicit), (6.0, 8.0, 10.0), rtol=2e-2)


# --- Parameter validation ---------------------------------------------------


@pytest.mark.parametrize("bp_type", ["butterworth", "wiener-butterworth"])
@pytest.mark.parametrize("beta", [0.0, -0.5, 1.5, 2.0])
def test_beta_out_of_range_raises(bp_type: str, beta: float) -> None:
    """``beta`` outside ``(0, 1]`` made ``ee`` negative and returned all-NaN."""
    with pytest.raises(ValueError, match="beta must satisfy"):
        create_backprojector(_psf_3d(), bp_type, beta=beta)


@pytest.mark.parametrize("bp_type", ["wiener", "wiener-butterworth"])
@pytest.mark.parametrize("alpha", [0.0, -0.1])
def test_alpha_non_positive_raises(bp_type: str, alpha: float) -> None:
    """``alpha == 0`` divided by zero at exact OTF nulls."""
    with pytest.raises(ValueError, match="alpha must be"):
        create_backprojector(_psf_3d(), bp_type, alpha=alpha)


def _box_psf() -> np.ndarray:
    """Build a box PSF, whose OTF has exactly-zero bins at multiples of 16 / 4."""
    psf = np.zeros((16, 16, 16), dtype=np.float64)
    psf[6:10, 6:10, 6:10] = 1.0
    return psf / psf.sum()


def test_box_psf_with_otf_nulls_is_finite() -> None:
    """A box PSF has exact OTF nulls; the default alpha keeps the result finite."""
    psf = _box_psf()
    otf = np.fft.fftn(np.fft.ifftshift(psf))
    assert int((np.abs(otf) == 0).sum()) > 0  # the null bins really are exact
    for bp_type in ("wiener", "wiener-butterworth"):
        bp = create_backprojector(psf, bp_type, res_flag=2, i_res=(5.0, 5.0, 5.0))
        assert np.all(np.isfinite(bp))


def test_zero_forward_gain_at_cutoff_raises() -> None:
    """'auto' alpha/beta need a non-zero forward gain at the cutoff.

    ``i_res=4`` lands the cutoff of a 4-voxel box PSF exactly on an OTF null, so
    ``beta_fp`` is 0 and the auto-substitution would divide by zero.
    """
    with pytest.raises(ValueError, match="no gain"):
        create_backprojector(
            _box_psf(), "wiener-butterworth", res_flag=2, i_res=(4.0, 4.0, 4.0)
        )


@pytest.mark.parametrize("res_flag", [-1, 3, 10])
def test_invalid_res_flag_raises(res_flag: int) -> None:
    """An unknown res_flag is rejected for every bp_type, including traditional."""
    with pytest.raises(ValueError, match="res_flag must be 0, 1, or 2"):
        create_backprojector(_psf_3d(), "traditional", res_flag=res_flag)


@pytest.mark.parametrize(
    "i_res", [(0.0, 5.0, 5.0), (5.0, -1.0, 5.0), (5.0, 5.0, np.nan)]
)
def test_i_res_non_positive_raises(i_res: tuple[float, ...]) -> None:
    """Non-positive or non-finite i_res entries divide by zero downstream."""
    with pytest.raises(ValueError, match="i_res entries must be finite"):
        create_backprojector(_psf_3d(), "butterworth", res_flag=2, i_res=i_res)


# --- Lazy FWHM / resolution computation -------------------------------------


def test_traditional_skips_fwhm_estimation() -> None:
    """``traditional`` needs no FWHM, so an unresolvable PSF is still accepted."""
    psf = _corner_psf()
    bp = create_backprojector(psf, "traditional")
    otf_f = np.fft.fftn(np.fft.ifftshift(psf / psf.sum()))
    otf_bp = np.fft.fftn(np.fft.ifftshift(bp))
    assert np.abs(otf_bp - np.conj(otf_f)).max() < 1e-6 * np.abs(otf_f).max()


@pytest.mark.parametrize("bp_type", ["gaussian", "butterworth", "wiener-butterworth"])
def test_res_flag2_skips_fwhm_estimation(bp_type: str) -> None:
    """``res_flag=2`` supplies the cutoffs directly, so no FWHM is needed."""
    bp = create_backprojector(_corner_psf(), bp_type, res_flag=2, i_res=(3.0, 3.0, 3.0))
    assert bp.shape == (8, 8, 8)
    assert np.all(np.isfinite(bp))
