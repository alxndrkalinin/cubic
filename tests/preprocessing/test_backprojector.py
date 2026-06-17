"""Tests for the Wiener-Butterworth back-projector PSF generator."""

import numpy as np
import pytest

from cubic.cuda import asnumpy, to_device
from cubic.preprocessing.backprojector import create_backprojector

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
def test_cpu_gpu_parity(bp_type: str, gpu_available: bool) -> None:
    """GPU result matches the CPU result for each back-projector type."""
    if not gpu_available:
        pytest.skip("GPU not available")
    psf = _psf_3d()
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
