# Band-Limited Similarity Metrics

## Motivation

Virtual staining models predict fluorescence from label-free inputs. The fluorescence ground truth contains measurement noise and deconvolution artefacts at high spatial frequencies. Standard pixel-wise metrics (PCC, SSIM) penalise models for *not* reproducing this noise, leading to artificially low scores. Band-limited metrics solve this by comparing images only over spatial frequencies that carry reliable biological signal.

## How It Works

1. **Estimate cutoff frequency** — determine the highest spatial frequency that contains trustworthy signal.
2. **Butterworth low-pass filter** — apply a smooth frequency-domain filter to both prediction and target, suppressing everything above the cutoff.
3. **Compute metric** — calculate PCC or SSIM on the filtered images.

The Butterworth filter has the form:

```
H(k) = 1 / (1 + (k / k_c)^(2n))
```

where `k` is the radial spatial frequency, `k_c` is the cutoff, and `n` is the filter order (default 2). This provides a smooth roll-off without ringing artefacts.

## Cutoff Estimation Methods

`estimate_cutoff()` computes up to four independent bounds and returns their minimum.

### DCR (Decorrelation Analysis) — `method="dcr"` (default)

Analyses the image's own frequency content via decorrelation analysis (Descloux et al., 2019). Returns a resolution estimate in physical units; the cutoff is `dcr_safety / resolution`.

**Pros**: Fast, single-image, no splitting required.
**Cons**: May overestimate resolution on very clean (denoised) images.

### FRC/FSC (Fourier Ring/Shell Correlation) — `method="frc"`

Splits the image using checkerboard sampling, then computes FRC (2-D) or FSC (3-D) correlation between the half-datasets. The resolution where the correlation drops below a threshold (typically 1/7) gives the cutoff: `frc_safety / resolution`.

**Pros**: Detects low-power high-frequency signal (useful for super-resolution). Widely used in cryo-EM and super-resolution microscopy.
**Cons**: Slightly slower than DCR (requires splitting + correlation). May return NaN on very high-SNR images where all frequencies remain correlated.

### OTF (Optical Transfer Function) — physics-based

Uses the objective's numerical aperture and emission wavelength to compute the theoretical diffraction limit: `otf_safety * 2*NA/lambda` (widefield) or `otf_safety * 4*NA/lambda` (confocal).

**Pros**: Independent of image content — always available if optical parameters are known.
**Cons**: Does not account for sample-dependent degradation or super-resolution techniques.

### Nyquist — always available

The sampling-limited frequency: `nyquist_safety * 0.5 / max(spacing)`. Acts as a hard upper bound — you can never resolve frequencies above Nyquist.

## DCR vs FRC/FSC: When to Use Which

| Scenario | Recommended | Rationale |
|---|---|---|
| Standard confocal/widefield | `"dcr"` | Fast, reliable, no splitting artefacts |
| Super-resolution (STED, PALM, SIM) | `"frc"` | Detects low-power high-frequency signal that DCR may miss |
| High-SNR denoised images | `"dcr"` | FRC may return NaN when all frequencies are correlated |
| Conservative estimate | `"both"` | Takes the minimum of both — never overestimates |
| 3-D volumes | `"dcr"` or `"both"` | Both support 3-D; FSC provides anisotropic (XY/Z) estimates |
| Quick benchmarking | `"dcr"` | Fastest single-image method |

## API Reference

### `estimate_cutoff(image, *, spacing, method="dcr", ...)`

Returns the recommended low-pass cutoff frequency (cycles/length).

Key parameters:
- `method`: `"dcr"` (default), `"frc"`, or `"both"`
- `dcr_safety`, `frc_safety`: safety factors for each data-driven bound (default 1.0)
- `otf_safety`, `nyquist_safety`: safety factors for physics bounds
- `dcr_kwargs`, `frc_kwargs`: extra arguments forwarded to the underlying resolution functions

### `band_limited_pcc(prediction, target, *, spacing, method="dcr", ...)`

Band-limited Pearson correlation coefficient. Applies Butterworth low-pass filtering before computing PCC.

### `band_limited_ssim(prediction, target, *, spacing, method="dcr", ...)`

Band-limited SSIM. Applies Butterworth low-pass filtering before computing structural similarity.

### `spectral_pcc(prediction, target, *, spacing, ...)`

Spectrally-weighted PCC. Instead of a hard cutoff, applies soft per-frequency weights derived from the target's radial power spectrum.

## Usage Examples

### Basic 2-D with DCR (default)

```python
from cubic.metrics.bandlimited import band_limited_pcc, band_limited_ssim

pcc = band_limited_pcc(prediction, target, spacing=0.065)
ssim = band_limited_ssim(prediction, target, spacing=0.065)
```

### FRC-based cutoff for super-resolution

```python
pcc = band_limited_pcc(prediction, target, spacing=0.065, method="frc")
```

### Conservative estimate with both methods

```python
pcc = band_limited_pcc(prediction, target, spacing=0.065, method="both")
```

### 3-D volume with anisotropic spacing

```python
pcc = band_limited_pcc(
    prediction, target,
    spacing=[0.2, 0.065, 0.065],  # Z, Y, X
    method="dcr",
)
```

### Comparing cutoff methods

```python
from cubic.metrics.bandlimited import estimate_cutoff

cutoff_dcr = estimate_cutoff(image, spacing=0.065, method="dcr")
cutoff_frc = estimate_cutoff(image, spacing=0.065, method="frc")
cutoff_both = estimate_cutoff(image, spacing=0.065, method="both")

print(f"DCR cutoff:  {cutoff_dcr:.2f} cycles/um")
print(f"FRC cutoff:  {cutoff_frc:.2f} cycles/um")
print(f"Both cutoff: {cutoff_both:.2f} cycles/um (min of all bounds)")
```

## Preprocessing Differences by Method

`estimate_cutoff()` applies method-appropriate preprocessing defaults when calling the underlying resolution estimators:

| Step | DCR | FRC (2-D) | FSC (3-D) |
|---|---|---|---|
| Apodization | Tukey (alpha=0.1) | Hamming | Hamming |
| Zero-padding | No | Yes (to cube) | Yes (forced by `estimate_cutoff`) |
| Image splitting | No | Checkerboard | Checkerboard + Z-summing |
| Isotropic resampling | No | N/A | Auto (if anisotropic spacing) |
| Mean subtraction | Yes | Yes (before FFT) | Yes (before FFT) |

For 3-D inputs, `estimate_cutoff` sets `zero_padding=True` and `resample_isotropic=True` (when `max(spacing)/min(spacing) > 1.5`) by default. These can be overridden via `frc_kwargs`.

## Preprocessing Differences by Method

`estimate_cutoff()` applies method-appropriate preprocessing defaults when calling the underlying resolution estimators:

| Step | DCR | FRC (2-D) | FSC (3-D) |
|---|---|---|---|
| Apodization | Tukey (alpha=0.1) | Hamming | Hamming |
| Zero-padding | No | Yes (to cube) | Yes (forced by `estimate_cutoff`) |
| Image splitting | No | Checkerboard | Checkerboard + Z-summing |
| Isotropic resampling | No | N/A | Auto (if anisotropic spacing) |
| Mean subtraction | Yes | Yes (before FFT) | Yes (before FFT) |

For 3-D inputs, `estimate_cutoff` sets `zero_padding=True` and `resample_isotropic=True` (when `max(spacing)/min(spacing) > 1.5`) by default. These can be overridden via `frc_kwargs`.

## Math Background

### Cutoff-resolution conversion

Both DCR and FRC/FSC return resolution in physical units where `resolution = 1 / k_c_physical`. The cutoff conversion is:

```
cutoff = safety_factor / resolution
```

This is identical for both methods since both express resolution as the inverse of the cutoff frequency.

### Butterworth filter

The Butterworth low-pass filter of order `n` at cutoff `k_c`:

```
H(k) = 1 / (1 + (k / k_c)^(2n))
```

- At `k = 0` (DC): `H = 1` (full pass)
- At `k = k_c`: `H = 0.5` (-3 dB point)
- At `k >> k_c`: `H -> 0` (full stop)
- Higher order `n` gives a steeper transition

### Spectral PCC

The spectrally-weighted PCC assigns per-frequency weights based on the target's power spectrum above the noise floor:

```
r = sum_k W(k) * Re{F_pred(k) * F_target*(k)} /
    sqrt(sum_k W(k) * |F_pred(k)|^2 * sum_k W(k) * |F_target(k)|^2)
```

where `W(k) = max(0, P_target(k) - noise_floor)` normalised to [0, 1].

## References

- Descloux, A., et al. (2019). Parameter-free image resolution estimation based on decorrelation analysis. *Nature Methods* 16, 918-924.
- Banterle, N., et al. (2013). Fourier ring correlation as a resolution criterion for super-resolution microscopy. *J. Struct. Biol.* 183, 363-367.
- Nieuwenhuizen, R.P.J., et al. (2013). Measuring image resolution in optical nanoscopy. *Nature Methods* 10, 557-562.
- Koho, S., et al. (2019). Fourier ring correlation simplifies image restoration in fluorescence microscopy. *Nat. Commun.* 10, 3103.
