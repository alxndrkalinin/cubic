# FRC/FSC/DCR Resolution Metrics

This module provides resolution estimation methods for fluorescence microscopy images:

- **FRC** (Fourier Ring Correlation) — 2D resolution estimation
- **FSC** (Fourier Shell Correlation) — 3D resolution estimation with XY/Z separation
- **DCR** (Decorrelation Analysis) — Parameter-free, single-image resolution estimation

## Quick Start

```python
from cubic.metrics.spectral import frc_resolution, fsc_resolution, dcr_resolution

# 2D FRC (single-image checkerboard split)
res = frc_resolution(image_2d, spacing=0.065)

# 3D FSC with isotropic resampling (recommended for anisotropic data)
res = fsc_resolution(image_3d, spacing=[0.2, 0.065, 0.065], resample_isotropic=True)
# Returns: {'xy': float, 'z': float}

# 3D FSC with two images (gold standard)
res = fsc_resolution(image_a, image_b, spacing=[0.2, 0.065, 0.065])

# 3D DCR (single image, no threshold needed)
res = dcr_resolution(image_3d, spacing=[0.2, 0.065, 0.065])
# Returns: {'xy': float, 'z': float}
```

## When to Use Each Method

| Scenario | Recommended | Rationale |
|----------|-------------|-----------|
| Two acquisitions available | FSC (two-image) | Gold standard, independent noise |
| Single 3D image | FSC (single-image) | Checkerboard split approximation |
| Quick quality check | DCR | No threshold selection needed |
| Super-resolution (STED, PALM) | FSC | DCR may underestimate due to power weighting |
| Deconvolution stopping criterion | FSC | Smooth convergence tracking (Koho et al. 2019, Fig. 3) |

## Key Parameters

### FSC

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spacing` | None | Physical spacing [Z, Y, X] in microns |
| `resample_isotropic` | False | Resample to isotropic voxels before FSC |
| `exclude_axis_angle` | 0.0 | Exclude frequencies near Z axis (degrees) |
| `backend` | `"hist"` | GPU-accelerated histogram-based backend |

### DCR

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spacing` | None | Physical spacing [Z, Y, X] in microns |
| `num_radii` | 100 | Number of radial sampling points |
| `num_highpass` | 10 | Number of high-pass filter levels |
| `use_sectioned` | True | Full 3D angular sectoring for XY/Z |

## Module Structure

- `frc.py` — FRC (2D) and FSC (3D) implementations
- `dcr.py` — DCR following Descloux et al. 2019
- `radial.py` — Shared radial binning utilities
- `analysis.py` — Curve fitting and resolution extraction
- `plot.py` — Plotting utilities (requires matplotlib)

## References

1. Descloux et al. (2019) "Parameter-free image resolution estimation based on decorrelation analysis", *Nature Methods* 16:918-924.
2. Nieuwenhuizen et al. (2013) "Measuring image resolution in optical nanoscopy", *Nature Methods* 10:557-562.
3. Koho et al. (2019) "Fourier ring correlation simplifies image restoration in fluorescence microscopy", *Nature Communications* 10:3103.
