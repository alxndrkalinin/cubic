# Band-Limited Similarity Metrics

## Motivation

Virtual staining models predict fluorescence from label-free inputs. The ground truth contains measurement noise at high spatial frequencies. Standard metrics (PCC, SSIM) penalize models for *not* reproducing this noise. Band-limited metrics solve this by comparing images only over frequencies that carry reliable signal.

## How It Works

1. **Estimate cutoff frequency** — determine the highest frequency with trustworthy signal
2. **Butterworth low-pass filter** — suppress everything above the cutoff
3. **Compute metric** — calculate PCC or SSIM on the filtered images

## Quick Start

```python
from cubic.metrics.bandlimited import band_limited_pcc, band_limited_ssim, spectral_pcc

# Band-limited PCC (DCR-based cutoff, default)
pcc = band_limited_pcc(prediction, target, spacing=0.065)

# Band-limited SSIM
ssim = band_limited_ssim(prediction, target, spacing=0.065)

# FRC-based cutoff (better for super-resolution)
pcc = band_limited_pcc(prediction, target, spacing=0.065, method="frc")

# Conservative: minimum of DCR and FRC cutoffs
pcc = band_limited_pcc(prediction, target, spacing=0.065, method="both")

# 3D volume
pcc = band_limited_pcc(prediction, target, spacing=[0.2, 0.065, 0.065])

# Spectrally-weighted PCC (soft weighting, no hard cutoff)
pcc = spectral_pcc(prediction, target, spacing=0.065)
```

## Cutoff Estimation

`estimate_cutoff()` computes up to four bounds and returns their minimum:

| Method | Source | Best for |
|--------|--------|----------|
| DCR (default) | Image frequency content | Standard microscopy |
| FRC/FSC | Checkerboard split correlation | Super-resolution |
| OTF | NA and wavelength | Always available if optics known |
| Nyquist | Pixel spacing | Hard upper bound |

```python
from cubic.metrics.bandlimited import estimate_cutoff

cutoff = estimate_cutoff(image, spacing=0.065, method="dcr")
print(f"Cutoff: {cutoff:.2f} cycles/µm")
```

## API

| Function | Description |
|----------|-------------|
| `band_limited_pcc(pred, target, spacing=...)` | Band-limited Pearson correlation |
| `band_limited_ssim(pred, target, spacing=...)` | Band-limited SSIM |
| `spectral_pcc(pred, target, spacing=...)` | Spectrally-weighted PCC |
| `estimate_cutoff(image, spacing=...)` | Data-driven cutoff frequency |
| `butterworth_lowpass(shape, cutoff, ...)` | Frequency-domain Butterworth filter |

## References

- Descloux et al. (2019) "Parameter-free image resolution estimation based on decorrelation analysis", *Nature Methods* 16:918-924.
- Koho et al. (2019) "Fourier ring correlation simplifies image restoration in fluorescence microscopy", *Nature Communications* 10:3103.
