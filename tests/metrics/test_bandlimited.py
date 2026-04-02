"""Tests for cubic.metrics.bandlimited — band-limited similarity metrics."""

from __future__ import annotations

import numpy as np
import pytest

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
    radii = np.linspace(0, 1, len(power))
    nf = estimate_noise_floor(radii, power, tail_fraction=0.3)
    assert nf > 0


def test_noise_floor_tail_fraction_effect() -> None:
    """Larger tail fraction yields higher noise floor estimate."""
    power = np.array([100, 50, 20, 10, 5, 2, 1], dtype=np.float32)
    radii = np.linspace(0, 1, len(power))
    nf_small = estimate_noise_floor(radii, power, tail_fraction=0.15)
    nf_large = estimate_noise_floor(radii, power, tail_fraction=0.5)
    assert nf_large >= nf_small


def test_spectral_weights_range() -> None:
    """Spectral weights are in [0, 1] with max == 1."""
    power = np.array([100, 50, 20, 10, 5, 2, 1], dtype=np.float32)
    radii = np.linspace(0, 1, len(power))
    nf = estimate_noise_floor(radii, power, tail_fraction=0.3)
    w = spectral_weights(radii, power, nf)
    assert float(np.min(w)) >= 0.0
    assert float(np.max(w)) == pytest.approx(1.0, abs=1e-6)


def test_spectral_weights_zero_below_noise() -> None:
    """Bins with power at or below noise have zero weight."""
    power = np.array([100, 50, 2, 1, 1, 1, 1], dtype=np.float32)
    radii = np.linspace(0, 1, len(power))
    w = spectral_weights(radii, power, noise_floor=3.0)
    assert np.all(w[2:] == 0.0)


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
