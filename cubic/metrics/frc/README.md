# FRC/FSC/DCR Resolution Metrics

This module provides implementations of resolution estimation methods for fluorescence microscopy images:

- **FRC** (Fourier Ring Correlation) - 2D resolution estimation
- **FSC** (Fourier Shell Correlation) - 3D resolution estimation
- **DCR** (Decorrelation Analysis) - Parameter-free resolution estimation from a single image

## API

```python
from cubic.metrics.frc import frc_resolution, fsc_resolution, dcr_resolution

# 2D FRC
res = frc_resolution(image_2d, spacing=0.065, backend='hist')

# 3D FSC - Single-image mode (uses checkerboard splitting)
res = fsc_resolution(image_3d, spacing=[0.2, 0.065, 0.065], backend='hist')
# Returns: {'xy': float, 'z': float}

# 3D FSC - Two-image mode (gold standard, requires paired acquisitions)
res = fsc_resolution(image_a, image_b, spacing=[0.2, 0.065, 0.065], backend='hist')

# 3D FSC with isotropic resampling (recommended for anisotropic data)
res = fsc_resolution(image_3d, spacing=[0.5, 0.126, 0.126], resample_isotropic=True)

# 3D FSC with axial exclusion (avoids piezo/interpolation artifacts)
res = fsc_resolution(image_3d, spacing=[0.2, 0.065, 0.065],
                     resample_isotropic=True, exclude_axis_angle=10.0)

# 3D FSC without split averaging (faster, slightly higher variance)
res = fsc_resolution(image_3d, spacing=[0.2, 0.065, 0.065], average=False)

# 2D DCR
res = dcr_resolution(image_2d, spacing=0.065)  # Returns float

# 3D DCR - Full angular sectioning (recommended)
res = dcr_resolution(image_3d, spacing=[0.2, 0.065, 0.065],
                     resample_isotropic=True, exclude_axis_angle=5.0)
# Returns: {'xy': float, 'z': float}

# 3D DCR - Legacy (2D slice-based, faster but less accurate)
res = dcr_resolution(image_3d, spacing=[0.2, 0.065, 0.065], use_sectioned=False)
```

## Backends

Two backends are available for FRC/FSC:

| Backend | Precision | Algorithm | GPU Support | Use Case |
|---------|-----------|-----------|-------------|----------|
| `hist` | float32 | Vectorized (bincount) | 10-16x speedup | **Recommended** for production |
| `mask` | float64 | Iterator-based (Python loops) | No speedup | Legacy, deprecated |

**Backend Differences:**

| Feature | `hist` (recommended) | `mask` (deprecated) |
|---------|---------------------|---------------------|
| 2D FRC | Consistent (~1-2% diff) | Consistent (~1-2% diff) |
| 3D FSC angles | 5 sectors (0° to 90°) | 24 sectors (0° to 360°) |
| 3D FSC approach | Koho et al. 2019 convention | Full rotation SFSC |
| Axial exclusion | Supported | Not supported |
| GPU acceleration | Yes (10-70x) | No |

**Important**: For 3D FSC, the backends use **different angular sectoring approaches** and results differ significantly. The `hist` backend uses angular binning from Z axis (theta=0°) to XY plane (theta=90°). The `mask` backend (from miplib) uses full 360° rotation SFSC. **Use `hist` backend for new projects**; `mask` is deprecated.

## Key Files

- `frc.py` – FRC (2D) and FSC (3D) implementations with both backends
- `dcr.py` – DCR (Decorrelation Analysis) following Descloux et al. 2019
- `radial.py` – Shared radial binning utilities (`radial_edges()`, `radial_bin_id()`, `radial_k_grid()`, `sectioned_bin_id()`)
- `iterators.py` – Mask backend iterators (`FourierRingIterator`, `FourierShellIterator`)
- `analysis.py` – Curve fitting and resolution extraction from FRC/FSC curves

### Frequency Unit Conventions

The codebase uses two frequency conventions depending on context:

| Function | Convention | Range | Nyquist (k_max) |
|----------|------------|-------|-----------------|
| `radial_k_grid(spacing=None)` | Normalized (cycles/sample) | [0, 0.5] | 0.5 |
| `radial_bin_id()` | Integer bin index | [0, n//2] | min(n//2) |
| `radial_k_grid(spacing=sp)` | Physical (cycles/unit) | [0, 1/(2·sp)] | min Nyquist |

When `spacing=None`, resolution is returned in **pixels** (index units). When spacing is provided, resolution is in the same physical units as spacing.

## 3D FSC Implementation

### Single-Image vs Two-Image FSC

| Mode | Usage | Noise Independence | Recommended For |
|------|-------|-------------------|-----------------|
| **Two-image** | `fsc_resolution(img_a, img_b, ...)` | Truly independent | Gold standard, high-SNR data |
| **Single-image** | `fsc_resolution(img, ...)` | Partially correlated (checkerboard split) | When paired acquisitions unavailable |

**Two-image FSC** requires two separate acquisitions of the same region with independent noise realizations. This is the gold standard method used in Koho et al. 2019.

**Single-image FSC** uses the paper's 3D splitting strategy: (1) sum consecutive Z pairs, (2) apply diagonal XY checkerboard split. This creates a 1-pixel XY shift between splits while preserving Z proportions.

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spacing` | None | Physical spacing [Z, Y, X] in microns. Required for isotropic resampling. |
| `resample_isotropic` | False | Resample to isotropic voxels before FSC. Recommended for anisotropic data. |
| `exclude_axis_angle` | 0.0 | Exclude frequencies within this angle (degrees) from Z axis. Typical: 5-10°. |
| `use_max_nyquist` | False | Extend frequency range to max Nyquist (XY) for better XY resolution on anisotropic data. |
| `average` | True | Average both diagonal splits (single-image mode only). |
| `backend` | 'hist' | 'hist' (fast, GPU) or 'mask' (reference). |

### Angle Convention

Following Koho et al. 2019:
- **angle = 0°**: XY plane (lateral resolution)
- **angle = 90°**: Z axis (axial resolution)

### Threshold

Both backends use the **one-bit threshold** (SNRe=0.5) as recommended by Koho et al. 2019 for 3D FSC. This threshold varies with the number of voxels per bin and is more conservative than the fixed 0.143 used for 2D FRC.

## Validation Against Koho et al. 2019

Tested with STED tubulin images from the paper's data repository (paired acquisitions, 30×972×1024, spacing Z=100nm, XY=30nm). Reference values from Supplementary Figure 10:

| Method | XY | Z | Source |
|--------|-----|-----|--------|
| Paper FRC | 134 nm | 611 nm | Suppl. Fig. 10 |
| Paper SFSC | **143 nm** | **638 nm** | Suppl. Fig. 10 |

Our implementation results:

| Configuration | XY | Z | Notes |
|---------------|-----|-----|-------|
| Single-image, iso=True, excl=10° | **141 nm** | 91 nm | **XY matches paper!** |
| Two-image, maxNyq=True | 189 nm | 478 nm | Z closer to paper |
| Two-image, iso=True, excl=10° | 75 nm | 142 nm | |

**XY resolution validation**: Single-image FSC with isotropic resampling gives **141 nm**, matching the paper's SFSC value of **143 nm** within 2%.

**Z resolution discrepancy**: Our Z measurements (91-478 nm) consistently underestimate the paper's ~638 nm. This is due to:
1. **Different angular decomposition**: Our hist backend uses conical polar sectors (0-90°), while the paper's SFSC uses wedge-shaped sectors rotated around the Y-axis (0-360° azimuthal). These sample different parts of Fourier space.
2. **Test data limitations**: The paired acquisitions (_a and _b) show 98% correlation, suggesting they may not have fully independent noise.
3. **Limited Z extent**: Only 30 Z slices limits the frequency resolution in the Z direction.

## Limitations and Caveats

### When FSC Results May Be Unreliable

1. **Limited Z extent**: Volumes with <50 Z slices (after isotropic resampling) have insufficient voxels in the Z-sector for reliable measurements.

2. **High-SNR single-image data**: Checkerboard splits remain highly correlated, and the correlation may never cross the threshold → NaN results.

3. **Noisy Z-sector correlation**: For STED and other super-resolution data with extreme XY/Z anisotropy, the Z-sector correlation can oscillate above/below threshold, leading to unreliable curve fitting.

4. **Single-image vs two-image**: Single-image FSC may give different (often more optimistic) resolution values than two-image FSC due to partial noise correlation.

### Recommendations

| Scenario | Recommendation |
|----------|----------------|
| High-SNR data | Use two-image FSC with paired acquisitions |
| Anisotropic data (Z >> XY spacing) | Enable `resample_isotropic=True` |
| STED/super-resolution | Use `exclude_axis_angle=5.0` to 10.0 |
| NaN results for XY | Try `use_max_nyquist=True` to extend frequency range |
| NaN results for Z | Z measurement is likely unreliable; trust XY only |
| Match paper's XY values | Use single-image + `resample_isotropic=True` + `exclude_axis_angle=10` |
| Quantitative comparison | Use two-image FSC for fair comparisons |

### Known Differences from Paper

Our implementation uses **angular binning** (sectors from XY plane to Z axis), while the paper uses a **dual-wedge SFSC** structure rotated around the Y-axis. This may cause:
- Different sampling of Z-direction frequencies
- Different sensitivity to anisotropic data
- XY measurements are consistent; Z measurements may differ

## 3D DCR Implementation

DCR (Decorrelation Analysis) from Descloux et al. 2019 has been extended to support full 3D analysis with angular sectoring, similar to FSC.

### DCR Algorithm Overview

1. Compute FFT and phase-normalize: I_n(k) = I(k) / |I(k)|
2. For each radius r, compute Pearson correlation d(r) between I and I_n·M(r)
3. Apply N_g high-pass Gaussian filters and repeat
4. Resolution = max(r_0, r_1, ..., r_Ng) peak frequencies (Equation 2 in paper)

### 3D DCR Modes

| Mode | Usage | Speed | Accuracy |
|------|-------|-------|----------|
| **Sectioned** (default) | `use_sectioned=True` | Slower | More accurate 3D analysis |
| **Legacy** | `use_sectioned=False` | Fast | Uses middle XY and XZ slices |

### DCR Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spacing` | None | Physical spacing [Z, Y, X] in microns |
| `resample_isotropic` | False | Resample to isotropic voxels before analysis |
| `exclude_axis_angle` | 0.0 | Exclude frequencies near Z axis (degrees) |
| `use_sectioned` | True | Use full 3D angular sectoring (vs. 2D slices) |
| `num_radii` | 100 | Number of radial sampling points |
| `num_highpass` | 10 | Number of high-pass filter levels |

### Logarithmic Sigma Spacing

Following the NanoPyx/ImageJ DecorrAnalysis convention from Descloux et al. 2019, high-pass filter sigmas are **logarithmically spaced** from 0.5 pixels to min(shape)/2 pixels:

```python
# Example: for a 512×512 image with num_highpass=10
sigmas = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0]  # pixels
```

**Why logarithmic spacing?**

| Spacing | Consistency | Rationale |
|---------|-------------|-----------|
| Linear | Variable (10-15% CV) | Non-uniform frequency sampling |
| **Logarithmic** | **Perfect (0% CV)** | Uniform sampling in log-frequency space |

The logarithmic spacing ensures that resolution estimates are **consistent regardless of `num_highpass` value** (5, 10, 15, 20, or 30 all give identical results). This also provides a 3× speedup by reducing the default from 30 to 10 iterations without any loss in accuracy.

### DCR vs FSC Comparison

| Feature | DCR | FSC |
|---------|-----|-----|
| Input required | Single image | Single or paired images |
| Parameter-free | Yes | No (threshold selection) |
| High-SNR handling | Robust | May give NaN (correlation doesn't cross threshold) |
| Speed (GPU) | Moderate | Fast |
| XY/Z resolution | Both | Both |

**When to use DCR**: DCR is useful when FSC returns NaN for high-SNR single-image data, as it doesn't rely on a correlation threshold crossing.

### DCR Validation

Tested on astrocyte confocal data (30×2160×2560, spacing Z=0.27μm, XY=0.054μm):

| Method | XY Resolution | Notes |
|--------|---------------|-------|
| 2D DCR (middle XY slice) | 282 nm | Reference |
| 3D DCR sectioned (iso + excl) | 275 nm | Matches 2D within 3% |
| 3D DCR legacy (2D slices) | 282 nm | Same as 2D |

## References

1. **DCR**: Descloux et al. (2019) "Parameter-free image resolution estimation based on decorrelation analysis", *Nature Methods* 16:918-924. (PDF available in `s41592-019-0515-7.pdf`)
2. **FRC**: Nieuwenhuizen et al. (2013) "Measuring image resolution in optical nanoscopy", *Nature Methods* 10:557-562.
3. **FSC**: Koho et al. (2019) "Fourier ring correlation simplifies image restoration in fluorescence microscopy", *Nature Communications* 10:3103. (PDF available in `s41467-019-11024-z.pdf`)
