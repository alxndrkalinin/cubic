"""Tests for cubic.metrics.bandlimited — band-limited similarity metrics."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from cubic.metrics.pcc import pcc
from cubic.metrics.bandlimited import (
    otf_cutoff,
    spectral_pcc,
    nyquist_cutoff,
    estimate_cutoff,
    band_limited_pcc,
    spectral_weights,
    band_limited_ssim,
    butterworth_lowpass,
    estimate_noise_floor,
    radial_power_spectrum,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gpu_available() -> bool:
    """Return True if CuPy + GPU are usable."""
    try:
        import cupy as cp

        cp.zeros(1)
        return True
    except Exception:
        return False


def _make_synthetic_pair(
    shape: tuple[int, ...] = (128, 128),
    signal_freq: float = 2.0,
    noise_sigma: float = 0.5,
    spacing: float = 0.065,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Create a synthetic (prediction, target, true_cutoff) triplet.

    *prediction* = sum of low-frequency Gaussian blobs.
    *target* = prediction + high-frequency noise.
    *true_cutoff* ~ ``signal_freq`` (cycles / um).
    """
    rng = np.random.default_rng(seed)

    coords = [np.arange(n) * spacing for n in shape]
    grids = np.meshgrid(*coords, indexing="ij")
    centre = [n * spacing / 2 for n in shape]
    sigma = 1.0 / signal_freq

    prediction = np.zeros(shape, dtype=np.float32)
    for offset in np.linspace(-0.3, 0.3, 3):
        for dim in range(len(shape)):
            shifted_centre = list(centre)
            shifted_centre[dim] += offset
            r2 = sum((g - c) ** 2 for g, c in zip(grids, shifted_centre))
            prediction += np.exp(-r2 / (2 * sigma**2)).astype(np.float32)

    target = prediction + noise_sigma * rng.standard_normal(shape).astype(np.float32)
    return prediction, target, signal_freq


# ===================================================================
# Unit tests — Butterworth low-pass
# ===================================================================


def test_butterworth_shape() -> None:
    """Butterworth filter has the same shape as input."""
    H = butterworth_lowpass((64, 64), cutoff=5.0)
    assert H.shape == (64, 64)


def test_butterworth_dc_equals_one() -> None:
    """DC component of Butterworth filter is 1."""
    H = butterworth_lowpass((64, 64), cutoff=5.0)
    assert H[0, 0] == pytest.approx(1.0, abs=1e-6)


def test_butterworth_values_in_unit_interval() -> None:
    """Butterworth filter values are in [0, 1]."""
    H = butterworth_lowpass((64, 64), cutoff=5.0, spacing=[0.065, 0.065])
    assert float(np.min(H)) >= 0.0
    assert float(np.max(H)) <= 1.0 + 1e-6


def test_butterworth_monotonic_radial_decay() -> None:
    """Butterworth values decrease along a radial line from DC."""
    H = butterworth_lowpass((128, 128), cutoff=5.0, spacing=[0.065, 0.065])
    row = H[0, : H.shape[1] // 2 + 1]
    diffs = np.diff(row)
    assert np.all(diffs <= 1e-6)


def test_butterworth_higher_order_steeper() -> None:
    """Higher order gives sharper transition."""
    H2 = butterworth_lowpass((64, 64), cutoff=5.0, spacing=[0.065, 0.065], order=2)
    H5 = butterworth_lowpass((64, 64), cutoff=5.0, spacing=[0.065, 0.065], order=5)
    assert float(np.sum(H5 < 0.5)) >= float(np.sum(H2 < 0.5))


def test_butterworth_3d() -> None:
    """Butterworth filter works for 3-D inputs."""
    H = butterworth_lowpass((16, 32, 32), cutoff=3.0, spacing=[0.2, 0.065, 0.065])
    assert H.shape == (16, 32, 32)
    assert H[0, 0, 0] == pytest.approx(1.0, abs=1e-6)


def test_butterworth_invalid_cutoff_raises() -> None:
    """Zero cutoff raises ValueError."""
    with pytest.raises(ValueError, match="cutoff"):
        butterworth_lowpass((64, 64), cutoff=0.0)


def test_butterworth_invalid_order_raises() -> None:
    """Order < 1 raises ValueError."""
    with pytest.raises(ValueError, match="order"):
        butterworth_lowpass((64, 64), cutoff=5.0, order=0)


# ===================================================================
# Unit tests — OTF cutoff
# ===================================================================


def test_otf_widefield_known_value() -> None:
    """Widefield OTF cutoff matches analytical value."""
    f = otf_cutoff(1.4, 0.52, modality="widefield")
    assert f == pytest.approx(2 * 1.4 / 0.52, rel=1e-3)


def test_otf_confocal_doubles_widefield() -> None:
    """Confocal OTF cutoff is 2x widefield."""
    f_wf = otf_cutoff(1.4, 0.52, modality="widefield")
    f_cf = otf_cutoff(1.4, 0.52, modality="confocal")
    assert f_cf == pytest.approx(2 * f_wf, rel=1e-6)


def test_otf_lightsheet_equals_widefield() -> None:
    """Light-sheet lateral OTF cutoff equals widefield."""
    f_wf = otf_cutoff(1.4, 0.52, modality="widefield")
    f_ls = otf_cutoff(1.4, 0.52, modality="lightsheet")
    assert f_ls == pytest.approx(f_wf, rel=1e-6)


def test_otf_higher_na_higher_cutoff() -> None:
    """Higher NA gives higher OTF cutoff."""
    assert otf_cutoff(1.4, 0.52) > otf_cutoff(0.8, 0.52)


def test_otf_unknown_modality_raises() -> None:
    """Unknown modality raises ValueError."""
    with pytest.raises(ValueError, match="modality"):
        otf_cutoff(1.4, 0.52, modality="unknown")


# ===================================================================
# Unit tests — Nyquist cutoff
# ===================================================================


def test_nyquist_isotropic() -> None:
    """Nyquist cutoff for isotropic spacing."""
    f = nyquist_cutoff(0.065)
    assert f == pytest.approx(0.5 / 0.065, rel=1e-6)


def test_nyquist_anisotropic_uses_coarsest() -> None:
    """Anisotropic spacing uses the coarsest axis."""
    f = nyquist_cutoff([0.2, 0.065, 0.065])
    assert f == pytest.approx(0.5 / 0.2, rel=1e-6)


def test_nyquist_scalar_matches_list() -> None:
    """Scalar spacing gives same result as equal-element list."""
    assert nyquist_cutoff(0.1) == pytest.approx(nyquist_cutoff([0.1, 0.1]), rel=1e-6)


# ===================================================================
# Unit tests — estimate_cutoff
# ===================================================================


def test_estimate_cutoff_returns_finite() -> None:
    """Estimate cutoff returns a finite positive value."""
    rng = np.random.default_rng(0)
    img = rng.standard_normal((64, 64)).astype(np.float32)
    c = estimate_cutoff(img, spacing=0.065)
    assert np.isfinite(c) and c > 0


def test_estimate_cutoff_skips_unavailable_bounds() -> None:
    """Cutoff estimation works without OTF parameters."""
    rng = np.random.default_rng(0)
    img = rng.standard_normal((64, 64)).astype(np.float32)
    c = estimate_cutoff(img, spacing=0.065)
    assert c > 0


def test_estimate_cutoff_returns_min_of_bounds() -> None:
    """Adding a tight OTF bound should pull cutoff down."""
    rng = np.random.default_rng(0)
    img = rng.standard_normal((64, 64)).astype(np.float32)
    c_no_otf = estimate_cutoff(img, spacing=0.065)
    c_with_otf = estimate_cutoff(
        img, spacing=0.065, numerical_aperture=0.3, wavelength_emission=0.52
    )
    assert c_with_otf <= c_no_otf + 1e-6


def test_estimate_cutoff_deterministic() -> None:
    """Same input gives same cutoff."""
    rng = np.random.default_rng(7)
    img = rng.standard_normal((64, 64)).astype(np.float32)
    c1 = estimate_cutoff(img, spacing=0.065)
    c2 = estimate_cutoff(img, spacing=0.065)
    assert c1 == c2


# ===================================================================
# Unit tests — radial power spectrum
# ===================================================================


def test_radial_power_spectrum_lengths() -> None:
    """Radii and power arrays have matching lengths."""
    rng = np.random.default_rng(0)
    img = rng.standard_normal((64, 64)).astype(np.float32)
    radii, power = radial_power_spectrum(img, spacing=0.065)
    assert len(radii) == len(power)
    assert len(radii) > 0


def test_radial_power_spectrum_white_noise_flat() -> None:
    """White-noise power spectrum is approximately flat."""
    rng = np.random.default_rng(42)
    img = rng.standard_normal((128, 128)).astype(np.float32)
    _, power = radial_power_spectrum(img)
    cv = np.std(power) / np.mean(power)
    assert cv < 1.0


def test_radial_power_spectrum_sinusoid_peak() -> None:
    """Power spectrum peaks near the true frequency of a sinusoid."""
    x = np.arange(128) * 0.065
    freq = 3.0
    img = np.sin(2 * np.pi * freq * x)[None, :] * np.ones((128, 1))
    img = img.astype(np.float32)
    radii, power = radial_power_spectrum(img, spacing=[0.065, 0.065])
    peak_freq = radii[np.argmax(power)]
    assert abs(peak_freq - freq) < 1.0


# ===================================================================
# Unit tests — noise floor and spectral weights
# ===================================================================


def test_noise_floor_positive() -> None:
    """Noise floor is positive for non-zero power."""
    power = np.array([100, 50, 20, 10, 5, 2, 1], dtype=np.float32)
    nf = estimate_noise_floor(power, tail_fraction=0.3)
    assert nf > 0


def test_noise_floor_tail_fraction_effect() -> None:
    """Larger tail fraction yields higher noise floor estimate."""
    power = np.array([100, 50, 20, 10, 5, 2, 1], dtype=np.float32)
    nf_small = estimate_noise_floor(power, tail_fraction=0.15)
    nf_large = estimate_noise_floor(power, tail_fraction=0.5)
    assert nf_large >= nf_small


def test_noise_floor_uses_only_the_tail() -> None:
    """Only the tail bins matter — the parameter list no longer takes radii.

    ``estimate_noise_floor`` used to accept a ``radii`` argument it never
    read, so a caller could pass mismatched radii (or ``None``) and get a
    value back regardless.
    """
    power = np.array([100, 50, 20, 10, 5, 2, 1], dtype=np.float32)
    # tail_fraction=0.3 of 7 bins -> ceil(2.1) = 3 bins: [5, 2, 1]
    assert estimate_noise_floor(power, tail_fraction=0.3) == pytest.approx(
        float(np.mean(power[-3:]))
    )
    # Changing only the leading (non-tail) bins cannot change the estimate.
    altered = power.copy()
    altered[:4] *= 1000.0
    assert estimate_noise_floor(altered, tail_fraction=0.3) == pytest.approx(
        estimate_noise_floor(power, tail_fraction=0.3)
    )


def test_spectral_weights_range() -> None:
    """Spectral weights are in [0, 1] with max == 1."""
    power = np.array([100, 50, 20, 10, 5, 2, 1], dtype=np.float32)
    radii = np.linspace(0, 1, len(power))
    nf = estimate_noise_floor(power, tail_fraction=0.3)
    w = spectral_weights(radii, power, nf)
    assert float(np.min(w)) >= 0.0
    assert float(np.max(w)) == pytest.approx(1.0, abs=1e-6)


def test_spectral_weights_zero_below_noise() -> None:
    """Bins with power at or below noise have zero weight."""
    power = np.array([100, 50, 2, 1, 1, 1, 1], dtype=np.float32)
    radii = np.linspace(0, 1, len(power))
    w = spectral_weights(radii, power, noise_floor=3.0)
    assert np.all(w[2:] == 0.0)


def test_spectral_weights_all_below_noise_are_all_zero() -> None:
    """No bin above the noise floor yields all-zero weights, not max == 1.

    The docstring used to promise ``max(w) == 1`` unconditionally, but the
    rescaling is skipped when nothing exceeds the floor.
    """
    power = np.array([1, 1, 1, 1], dtype=np.float32)
    radii = np.linspace(0, 1, len(power))
    w = spectral_weights(radii, power, noise_floor=5.0)
    assert np.all(w == 0.0)


def test_spectral_weights_cutoff_zeroes() -> None:
    """Bins above cutoff have zero weight."""
    power = np.array([100, 80, 60, 40, 20, 10, 5], dtype=np.float32)
    radii = np.linspace(0, 1, len(power))
    w = spectral_weights(radii, power, noise_floor=0.0, cutoff=0.5)
    assert np.all(w[radii > 0.5] == 0.0)


# ===================================================================
# Integration tests — band-limited PCC
# ===================================================================


def test_bl_pcc_identical() -> None:
    """Band-limited PCC of identical images is 1."""
    rng = np.random.default_rng(0)
    img = rng.standard_normal((64, 64)).astype(np.float32)
    r = band_limited_pcc(img, img, spacing=0.065, cutoff=5.0)
    assert r == pytest.approx(1.0, abs=1e-4)


def test_bl_pcc_range() -> None:
    """Band-limited PCC is in [-1, 1]."""
    pred, tgt, _ = _make_synthetic_pair(seed=0)
    r = band_limited_pcc(pred, tgt, spacing=0.065, cutoff=5.0)
    assert -1.0 <= r <= 1.0


def test_bl_pcc_noisy_improvement() -> None:
    """Band-limited PCC >= standard PCC on noisy data."""
    pred, tgt, _ = _make_synthetic_pair(noise_sigma=1.0, seed=1)
    p = pred.ravel() - pred.ravel().mean()
    t = tgt.ravel() - tgt.ravel().mean()
    std_pcc = float(np.sum(p * t) / np.sqrt(np.sum(p**2) * np.sum(t**2)))
    bl_pcc = band_limited_pcc(pred, tgt, spacing=0.065, cutoff=3.0)
    assert bl_pcc >= std_pcc - 0.02


def test_bl_pcc_2d_and_3d() -> None:
    """Band-limited PCC works for both 2-D and 3-D inputs."""
    rng = np.random.default_rng(5)
    img2d = rng.standard_normal((64, 64)).astype(np.float32)
    r2 = band_limited_pcc(img2d, img2d, spacing=0.065, cutoff=5.0)
    assert r2 == pytest.approx(1.0, abs=1e-4)

    img3d = rng.standard_normal((16, 32, 32)).astype(np.float32)
    r3 = band_limited_pcc(img3d, img3d, spacing=[0.2, 0.065, 0.065], cutoff=3.0)
    assert r3 == pytest.approx(1.0, abs=1e-4)


def test_bl_pcc_shape_mismatch_raises() -> None:
    """Mismatched shapes raise ValueError."""
    a = np.zeros((64, 64), dtype=np.float32)
    b = np.zeros((64, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        band_limited_pcc(a, b, spacing=0.065, cutoff=5.0)


# ===================================================================
# Integration tests — band-limited SSIM
# ===================================================================


def test_bl_ssim_identical() -> None:
    """Band-limited SSIM of identical images is ~1."""
    rng = np.random.default_rng(0)
    img = rng.standard_normal((64, 64)).astype(np.float32)
    s = band_limited_ssim(img, img, spacing=0.065, cutoff=5.0)
    assert s == pytest.approx(1.0, abs=1e-3)


def test_bl_ssim_filtered_improves_noisy() -> None:
    """Band-limited SSIM >= raw SSIM on noisy data."""
    from cubic.metrics.skimage_metrics import ssim as raw_ssim

    pred, tgt, _ = _make_synthetic_pair(noise_sigma=1.0, seed=3)
    data_range = float(tgt.max() - tgt.min())
    raw = float(raw_ssim(pred, tgt, data_range=data_range))
    bl = band_limited_ssim(pred, tgt, spacing=0.065, cutoff=3.0)
    assert bl >= raw - 0.05


# ===================================================================
# Integration tests — spectral PCC
# ===================================================================


def test_spectral_pcc_identical() -> None:
    """Spectral PCC of identical images is ~1."""
    rng = np.random.default_rng(0)
    img = rng.standard_normal((64, 64)).astype(np.float32)
    r = spectral_pcc(img, img, spacing=0.065)
    assert r == pytest.approx(1.0, abs=1e-3)


def test_spectral_pcc_noisy_improvement() -> None:
    """Spectral PCC >= standard PCC on noisy data."""
    pred, tgt, _ = _make_synthetic_pair(noise_sigma=1.0, seed=2)
    p = pred.ravel() - pred.ravel().mean()
    t = tgt.ravel() - tgt.ravel().mean()
    std_pcc = float(np.sum(p * t) / np.sqrt(np.sum(p**2) * np.sum(t**2)))
    sp = spectral_pcc(pred, tgt, spacing=0.065)
    assert sp >= std_pcc - 0.05


def test_spectral_pcc_range() -> None:
    """Spectral PCC is in [-1, 1]."""
    pred, tgt, _ = _make_synthetic_pair(seed=10)
    r = spectral_pcc(pred, tgt, spacing=0.065)
    assert -1.0 <= r <= 1.0


def test_spectral_pcc_shape_mismatch_raises() -> None:
    """Mismatched shapes raise ValueError."""
    a = np.zeros((64, 64), dtype=np.float32)
    b = np.zeros((64, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="Shape mismatch"):
        spectral_pcc(a, b, spacing=0.065)


def test_spectral_pcc_nbins_low() -> None:
    """nbins_low excludes low-frequency bins without crashing."""
    pred, tgt, _ = _make_synthetic_pair(noise_sigma=0.5, seed=8)
    r0 = spectral_pcc(pred, tgt, spacing=0.065, nbins_low=0)
    r3 = spectral_pcc(pred, tgt, spacing=0.065, nbins_low=3)
    assert -1.0 <= r0 <= 1.0
    assert -1.0 <= r3 <= 1.0


def test_spectral_pcc_taper_low() -> None:
    """taper_low applies soft cosine ramp to low-frequency bins."""
    pred, tgt, _ = _make_synthetic_pair(noise_sigma=0.5, seed=31)
    r = spectral_pcc(pred, tgt, spacing=0.065, taper_low=3)
    assert -1.0 <= r <= 1.0


def test_spectral_pcc_negative_nbins_low() -> None:
    """Negative nbins_low raises ValueError."""
    a = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="nbins_low"):
        spectral_pcc(a, a, spacing=0.065, nbins_low=-1)


def test_spectral_pcc_negative_taper_low() -> None:
    """Negative taper_low raises ValueError."""
    a = np.zeros((64, 64), dtype=np.float32)
    with pytest.raises(ValueError, match="taper_low"):
        spectral_pcc(a, a, spacing=0.065, taper_low=-1)


# ===================================================================
# GPU / CPU parity tests
# ===================================================================


@pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
def test_device_parity_bl_pcc() -> None:
    """Band-limited PCC matches between CPU and GPU."""
    import cupy as cp

    rng = np.random.default_rng(99)
    pred = rng.standard_normal((64, 64)).astype(np.float32)
    tgt = pred + 0.3 * rng.standard_normal((64, 64)).astype(np.float32)

    r_cpu = band_limited_pcc(pred, tgt, spacing=0.065, cutoff=5.0)
    r_gpu = band_limited_pcc(
        cp.asarray(pred), cp.asarray(tgt), spacing=0.065, cutoff=5.0
    )
    assert abs(r_cpu - r_gpu) < 1e-4


@pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
def test_device_parity_spectral_pcc() -> None:
    """Spectral PCC matches between CPU and GPU."""
    import cupy as cp

    rng = np.random.default_rng(99)
    pred = rng.standard_normal((64, 64)).astype(np.float32)
    tgt = pred + 0.3 * rng.standard_normal((64, 64)).astype(np.float32)

    r_cpu = spectral_pcc(pred, tgt, spacing=0.065)
    r_gpu = spectral_pcc(cp.asarray(pred), cp.asarray(tgt), spacing=0.065)
    assert abs(r_cpu - r_gpu) < 1e-4


@pytest.mark.skipif(not _gpu_available(), reason="GPU not available")
def test_device_parity_bl_ssim() -> None:
    """Band-limited SSIM matches between CPU and GPU."""
    import cupy as cp

    rng = np.random.default_rng(99)
    pred = rng.standard_normal((64, 64)).astype(np.float32)
    tgt = pred + 0.3 * rng.standard_normal((64, 64)).astype(np.float32)

    s_cpu = band_limited_ssim(pred, tgt, spacing=0.065, cutoff=5.0)
    s_gpu = band_limited_ssim(
        cp.asarray(pred), cp.asarray(tgt), spacing=0.065, cutoff=5.0
    )
    assert abs(s_cpu - s_gpu) < 1e-3


# ===================================================================
# Stability tests
# ===================================================================


def test_stability_cutoff_perturbation() -> None:
    """BL-PCC changes < 5% for +/-10% cutoff perturbation."""
    pred, tgt, _ = _make_synthetic_pair(seed=50)
    cutoff = 3.0
    r0 = band_limited_pcc(pred, tgt, spacing=0.065, cutoff=cutoff)
    r_lo = band_limited_pcc(pred, tgt, spacing=0.065, cutoff=cutoff * 0.9)
    r_hi = band_limited_pcc(pred, tgt, spacing=0.065, cutoff=cutoff * 1.1)
    assert abs(r_lo - r0) < 0.05
    assert abs(r_hi - r0) < 0.05


def test_stability_order() -> None:
    """BL-PCC is similar for filter order 2 vs 3."""
    pred, tgt, _ = _make_synthetic_pair(seed=51)
    r2 = band_limited_pcc(pred, tgt, spacing=0.065, cutoff=3.0, filter_order=2)
    r3 = band_limited_pcc(pred, tgt, spacing=0.065, cutoff=3.0, filter_order=3)
    assert abs(r2 - r3) < 0.05


def test_stability_tail_fraction() -> None:
    """Spectral PCC is stable for tail_fraction 0.15-0.25."""
    pred, tgt, _ = _make_synthetic_pair(seed=52)
    r15 = spectral_pcc(pred, tgt, spacing=0.065, tail_fraction=0.15)
    r20 = spectral_pcc(pred, tgt, spacing=0.065, tail_fraction=0.20)
    r25 = spectral_pcc(pred, tgt, spacing=0.065, tail_fraction=0.25)
    assert abs(r15 - r20) < 0.05
    assert abs(r20 - r25) < 0.05


def test_stability_estimate_cutoff_deterministic() -> None:
    """Same input gives same cutoff estimate."""
    rng = np.random.default_rng(53)
    img = rng.standard_normal((64, 64)).astype(np.float32)
    c1 = estimate_cutoff(img, spacing=0.065)
    c2 = estimate_cutoff(img, spacing=0.065)
    assert c1 == c2


# ===================================================================
# Tests — method parameter (FRC / FSC cutoff estimation)
# ===================================================================


def test_estimate_cutoff_method_dcr_default() -> None:
    """method='dcr' matches the default (no method arg) behaviour."""
    rng = np.random.default_rng(60)
    img = rng.standard_normal((64, 64)).astype(np.float32)
    c_default = estimate_cutoff(img, spacing=0.065)
    c_dcr = estimate_cutoff(img, spacing=0.065, method="dcr")
    assert c_default == c_dcr


def test_estimate_cutoff_method_frc_2d() -> None:
    """method='frc' returns a finite positive cutoff for 2-D images."""
    rng = np.random.default_rng(61)
    img = rng.standard_normal((128, 128)).astype(np.float32)
    c = estimate_cutoff(img, spacing=0.065, method="frc")
    assert np.isfinite(c) and c > 0


def test_estimate_cutoff_method_frc_3d() -> None:
    """method='frc' uses FSC for 3-D images and returns finite positive."""
    rng = np.random.default_rng(62)
    img = rng.standard_normal((16, 64, 64)).astype(np.float32)
    c = estimate_cutoff(img, spacing=[0.2, 0.065, 0.065], method="frc")
    assert np.isfinite(c) and c > 0


def test_estimate_cutoff_method_both() -> None:
    """method='both' cutoff <= each individual method's cutoff."""
    rng = np.random.default_rng(63)
    img = rng.standard_normal((128, 128)).astype(np.float32)
    c_dcr = estimate_cutoff(img, spacing=0.065, method="dcr")
    c_frc = estimate_cutoff(img, spacing=0.065, method="frc")
    c_both = estimate_cutoff(img, spacing=0.065, method="both")
    # "both" takes the minimum of all bounds, so it should be <= each
    assert c_both <= c_dcr + 1e-6
    assert c_both <= c_frc + 1e-6


def test_bl_pcc_method_frc() -> None:
    """band_limited_pcc with method='frc' gives identity = 1.0."""
    rng = np.random.default_rng(64)
    img = rng.standard_normal((128, 128)).astype(np.float32)
    r = band_limited_pcc(img, img, spacing=0.065, method="frc")
    assert r == pytest.approx(1.0, abs=1e-4)


def test_bl_ssim_method_frc() -> None:
    """band_limited_ssim with method='frc' gives identity ~ 1.0."""
    rng = np.random.default_rng(65)
    img = rng.standard_normal((128, 128)).astype(np.float32)
    s = band_limited_ssim(img, img, spacing=0.065, method="frc")
    assert s == pytest.approx(1.0, abs=1e-3)


# ===================================================================
# Tests — FSC default preprocessing in estimate_cutoff
# ===================================================================


def test_estimate_cutoff_fsc_defaults_anisotropic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Anisotropic 3-D: FSC gets zero_padding=True, resample_isotropic=True."""
    captured: dict = {}

    def fake_fsc(image, **kwargs):
        captured.update(kwargs)
        return {"xy": 0.5, "z": 1.0}

    import cubic.metrics.bandlimited as _bl_mod

    monkeypatch.setattr(_bl_mod, "fsc_resolution", fake_fsc)

    rng = np.random.default_rng(70)
    img = rng.standard_normal((16, 64, 64)).astype(np.float32)
    estimate_cutoff(img, spacing=[0.3, 0.065, 0.065], method="frc")

    assert captured["zero_padding"] is True
    assert captured["resample_isotropic"] is True


def test_estimate_cutoff_fsc_user_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """User frc_kwargs override the FSC defaults."""
    captured: dict = {}

    def fake_fsc(image, **kwargs):
        captured.update(kwargs)
        return {"xy": 0.5, "z": 1.0}

    import cubic.metrics.bandlimited as _bl_mod

    monkeypatch.setattr(_bl_mod, "fsc_resolution", fake_fsc)

    rng = np.random.default_rng(71)
    img = rng.standard_normal((16, 64, 64)).astype(np.float32)
    estimate_cutoff(
        img,
        spacing=[0.3, 0.065, 0.065],
        method="frc",
        frc_kwargs={"zero_padding": False, "resample_isotropic": False},
    )

    assert captured["zero_padding"] is False
    assert captured["resample_isotropic"] is False


def test_estimate_cutoff_fsc_isotropic_no_resample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Isotropic 3-D: FSC gets resample_isotropic=False."""
    captured: dict = {}

    def fake_fsc(image, **kwargs):
        captured.update(kwargs)
        return {"xy": 0.5, "z": 1.0}

    import cubic.metrics.bandlimited as _bl_mod

    monkeypatch.setattr(_bl_mod, "fsc_resolution", fake_fsc)

    rng = np.random.default_rng(72)
    img = rng.standard_normal((32, 32, 32)).astype(np.float32)
    estimate_cutoff(img, spacing=[0.065, 0.065, 0.065], method="frc")

    assert captured["zero_padding"] is True
    assert captured["resample_isotropic"] is False


def test_estimate_cutoff_fsc_defaults_with_both(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """method='both' still applies FSC defaults for the FSC branch."""
    captured: dict = {}

    def fake_fsc(image, **kwargs):
        captured.update(kwargs)
        return {"xy": 0.5, "z": 1.0}

    import cubic.metrics.bandlimited as _bl_mod

    monkeypatch.setattr(_bl_mod, "fsc_resolution", fake_fsc)

    rng = np.random.default_rng(73)
    img = rng.standard_normal((16, 64, 64)).astype(np.float32)
    estimate_cutoff(img, spacing=[0.3, 0.065, 0.065], method="both")

    assert captured["zero_padding"] is True
    assert captured["resample_isotropic"] is True


# ===================================================================
# Regression tests — method validation
# ===================================================================


@pytest.mark.parametrize("bad_method", ["DCR", "FRC", "Both", "banana", ""])
def test_estimate_cutoff_rejects_unknown_method(bad_method: str) -> None:
    """An unrecognised or miscased method raises instead of silently falling back.

    Both data-driven branches tested ``method in (...)`` with no ``else``,
    so a typo skipped every data-driven bound and returned the pure
    Nyquist bound — a silent downgrade of the metric.
    """
    rng = np.random.default_rng(80)
    img = rng.standard_normal((64, 64)).astype(np.float32)
    with pytest.raises(ValueError, match="Unknown method"):
        estimate_cutoff(img, spacing=0.065, method=bad_method)


def test_estimate_cutoff_bad_method_is_not_nyquist() -> None:
    """The miscased method used to return exactly the Nyquist-only bound."""
    rng = np.random.default_rng(81)
    img = rng.standard_normal((64, 64)).astype(np.float32)
    nyquist_only = 0.9 * nyquist_cutoff(0.065)
    with pytest.raises(ValueError):
        estimate_cutoff(img, spacing=0.065, method="DCR")
    # The genuine dcr path must be at or below the Nyquist bound, i.e. the
    # data-driven bound really does participate.
    assert estimate_cutoff(img, spacing=0.065, method="dcr") <= nyquist_only + 1e-9


def test_estimate_cutoff_safety_factors_apply() -> None:
    """The ``safety`` dict scales the corresponding bound."""
    rng = np.random.default_rng(82)
    img = rng.standard_normal((64, 64)).astype(np.float32)
    default = estimate_cutoff(img, spacing=0.065)
    tightened = estimate_cutoff(img, spacing=0.065, safety={"dcr": 0.5})
    assert tightened <= default + 1e-9
    # A pure Nyquist scaling is directly predictable.
    assert estimate_cutoff(
        img, spacing=0.065, safety={"nyquist": 0.1, "dcr": 100.0}
    ) == pytest.approx(0.1 * nyquist_cutoff(0.065))


def test_estimate_cutoff_rejects_unknown_safety_key() -> None:
    """A misspelled safety key raises rather than being ignored."""
    rng = np.random.default_rng(83)
    img = rng.standard_normal((64, 64)).astype(np.float32)
    with pytest.raises(ValueError, match="Unknown safety keys"):
        estimate_cutoff(img, spacing=0.065, safety={"nyqist": 0.5})


def test_otf_cutoff_no_longer_takes_refractive_index() -> None:
    """The unused ``medium_refractive_index`` parameter is gone."""
    with pytest.raises(TypeError):
        otf_cutoff(1.4, 0.52, medium_refractive_index=1.33)  # type: ignore[call-arg]


# ===================================================================
# Regression tests — spectral_pcc weights come from the correlated spectrum
# ===================================================================


def _reference_spectral_pcc(
    prediction: np.ndarray,
    target: np.ndarray,
    spacing: float,
    *,
    raw_power: bool,
    bin_delta: float = 1.0,
    tail_fraction: float = 0.2,
) -> tuple[float, np.ndarray]:
    """Independent spectral-PCC reference, returning ``(r, per_bin_weights)``.

    With ``raw_power=False`` the per-bin weights are derived from the same
    mean-subtracted, apodised target spectrum that is correlated (correct).
    With ``raw_power=True`` they come from a raw FFT of the target, which
    is what the implementation used to do.
    """
    from cubic.image_utils import tukey_window
    from cubic.metrics.spectral.radial import (
        radial_edges,
        reduce_power,
        radial_bin_id,
    )

    sp = [float(spacing)] * prediction.ndim
    pred_w = tukey_window(prediction.astype(np.float32) - prediction.mean())
    targ_w = tukey_window(target.astype(np.float32) - target.mean())
    F_pred = np.fft.fftn(pred_w)
    F_targ = np.fft.fftn(targ_w)

    edges, _ = radial_edges(prediction.shape, bin_delta=bin_delta, spacing=sp)
    bid = radial_bin_id(prediction.shape, edges, spacing=sp)

    F_for_power = np.fft.fftn(target.astype(np.float32)) if raw_power else F_targ
    S2, N = reduce_power(F_for_power, bid)
    power = (S2 / np.maximum(N.astype(np.float64), 1.0)).astype(np.float32)

    n_tail = max(1, int(np.ceil(len(power) * tail_fraction)))
    noise = float(np.mean(power[-n_tail:]))
    w = np.maximum(power - noise, 0.0)
    w = w / float(np.max(w))

    W = np.zeros_like(bid, dtype=np.float32)
    valid = bid >= 0
    W[valid] = w[bid[valid]]

    num = float(np.sum(W * np.real(F_pred.ravel() * np.conj(F_targ.ravel()))))
    den = float(
        np.sqrt(
            np.sum(W * np.abs(F_pred.ravel()) ** 2)
            * np.sum(W * np.abs(F_targ.ravel()) ** 2)
        )
    )
    return float(num / den), w


def test_spectral_pcc_weights_from_apodized_spectrum() -> None:
    """Weights are derived from the spectrum the metric actually correlates.

    ``spectral_pcc`` mean-subtracts and apodises before transforming, but
    it used to call ``radial_power_spectrum(target)``, which FFTs the raw
    target with neither step. The DC pedestal and the edge discontinuity
    then leaked into the low-frequency bins and inflated their power, so
    the noise-floor comparison was made against a spectrum the metric
    never used.
    """
    # Heavy noise makes the correlation strongly frequency-dependent, so the
    # choice of weighting spectrum has a large effect on the result.
    pred, tgt, _ = _make_synthetic_pair(noise_sigma=4.0, seed=201)

    got = spectral_pcc(pred, tgt, spacing=0.065)
    correct, w_correct = _reference_spectral_pcc(pred, tgt, 0.065, raw_power=False)
    stale, w_stale = _reference_spectral_pcc(pred, tgt, 0.065, raw_power=True)

    # Guard the test itself: the two weightings must genuinely disagree,
    # otherwise this would pass for the wrong implementation too.
    assert float(np.max(np.abs(w_correct - w_stale))) > 1e-3

    # The assertion is relative, not a tuned magnitude: whatever the radial
    # binning, the implementation must sit far closer to the windowed-spectrum
    # reference than to the raw-spectrum one. The match tolerance covers the
    # float32 weight cast inside ``spectral_weights``.
    match_err = abs(got - correct)
    stale_err = abs(got - stale)
    assert match_err < 1e-6
    assert stale_err > 1e-4
    assert stale_err > 100 * match_err


def test_spectral_pcc_spacing_none_raises_clearly() -> None:
    """``spacing=None`` reports the missing argument, not a TypeError deep inside.

    It used to fail with ``TypeError: 'NoneType' object is not iterable``
    from inside the shared spacing helper.
    """
    a = np.zeros((32, 32), dtype=np.float32)
    with pytest.raises(ValueError, match="spacing is required"):
        spectral_pcc(a, a, spacing=None)  # type: ignore[arg-type]


def test_band_limited_metrics_spacing_none_raise_clearly() -> None:
    """Both hard-cutoff metrics reject ``spacing=None`` the same way."""
    rng = np.random.default_rng(202)
    a = rng.standard_normal((32, 32)).astype(np.float32)
    with pytest.raises(ValueError, match="spacing is required"):
        band_limited_pcc(a, a, spacing=None, cutoff=5.0)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="spacing is required"):
        band_limited_ssim(a, a, spacing=None, cutoff=5.0)  # type: ignore[arg-type]


@pytest.mark.parametrize("shape", [(128, 128), (24, 32, 32)])
def test_spectral_pcc_spacing_one_is_not_a_special_case(
    shape: tuple[int, ...],
) -> None:
    """``spacing=1.0`` must behave like any other isotropic spacing.

    The radial bin edges and the per-voxel frequency magnitudes both scale
    as ``1 / (n * spacing)``, so an isotropic rescale cannot change the
    binning — ``spectral_pcc`` is invariant to the spacing *value*.
    ``radial_bin_id`` used to treat an all-1.0 spacing as ``None`` (index
    units) while ``radial_edges`` stayed in physical units, which
    collapsed every non-DC voxel into the last bin, flattened the weights
    and silently degraded the metric to an unweighted correlation. The fix
    lives in ``cubic/metrics/spectral/radial.py``; this pins the
    observable consequence here.
    """
    pred, tgt, _ = _make_synthetic_pair(shape=shape, noise_sigma=3.0, seed=300)

    r_at_one = spectral_pcc(pred, tgt, spacing=1.0)
    for other in (0.5, 2.0):
        assert r_at_one == pytest.approx(
            spectral_pcc(pred, tgt, spacing=other), abs=1e-6
        )

    # A flattened weighting would drop the metric onto the unweighted PCC.
    p = pred.ravel() - pred.ravel().mean()
    t = tgt.ravel() - tgt.ravel().mean()
    unweighted = float(np.sum(p * t) / np.sqrt(np.sum(p**2) * np.sum(t**2)))
    assert r_at_one > unweighted + 0.1


# ===================================================================
# Regression tests — degenerate (constant) inputs
# ===================================================================


def test_band_limited_ssim_constant_input_is_nan_without_warning() -> None:
    """A constant input gives nan explicitly, not 0/0 inside skimage.

    Mean subtraction turns a constant image into all zeros, so the
    filtered data range is 0, which zeroes SSIM's c1/c2 stabilisers.
    """
    const = np.full((64, 64), 3.0, dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        s = band_limited_ssim(const, const, spacing=0.065, cutoff=5.0)
    assert math.isnan(s)


def test_band_limited_ssim_constant_input_honours_explicit_data_range() -> None:
    """An explicit data_range keeps SSIM computable on a constant input."""
    const = np.full((64, 64), 3.0, dtype=np.float32)
    s = band_limited_ssim(const, const, spacing=0.065, cutoff=5.0, data_range=1.0)
    assert s == pytest.approx(1.0, abs=1e-6)


def test_degenerate_convention_is_nan_across_metrics() -> None:
    """All three band-limited metrics agree on the degenerate-input result.

    ``band_limited_pcc`` used to return 0.0 (its own ``_pearson`` helper)
    while ``band_limited_ssim`` returned nan and ``cubic.metrics.pcc``
    returned nan.
    """
    const = np.full((64, 64), 3.0, dtype=np.float32)
    assert math.isnan(band_limited_pcc(const, const, spacing=0.065, cutoff=5.0))
    assert math.isnan(band_limited_ssim(const, const, spacing=0.065, cutoff=5.0))
    assert math.isnan(spectral_pcc(const, const, spacing=0.065))
    assert math.isnan(pcc(const, const))


def test_band_limited_pcc_matches_pcc_on_filtered_images() -> None:
    """band_limited_pcc delegates to ``cubic.metrics.pcc``, not a private copy."""
    from cubic.metrics.bandlimited import _apply_lowpass, _require_spacing

    pred, tgt, _ = _make_synthetic_pair(noise_sigma=0.6, seed=204)
    spacing_seq = _require_spacing(0.065, pred.ndim)
    pred_f = _apply_lowpass(pred, 3.0, spacing=spacing_seq)
    targ_f = _apply_lowpass(tgt, 3.0, spacing=spacing_seq)
    expected = pcc(pred_f, targ_f)
    assert band_limited_pcc(pred, tgt, spacing=0.065, cutoff=3.0) == pytest.approx(
        expected, abs=1e-12
    )


def test_spectral_pcc_all_bins_excluded_is_nan() -> None:
    """Excluding every bin leaves nothing to correlate → nan."""
    pred, tgt, _ = _make_synthetic_pair(noise_sigma=0.5, seed=205)
    r = spectral_pcc(pred, tgt, spacing=0.065, nbins_low=10_000)
    assert math.isnan(r)
