# FRC/FSC/DCR Resolution Metrics

This module provides implementations of resolution estimation methods for fluorescence microscopy images:

- **FRC** (Fourier Ring Correlation) - 2D resolution estimation
- **FSC** (Fourier Shell Correlation) - 3D resolution estimation
- **DCR** (Decorrelation Analysis) - Parameter-free resolution estimation from a single image

## Quick Start

```python
from cubic.metrics.frc import frc_resolution, fsc_resolution, dcr_resolution

# 2D FRC (requires two images or uses checkerboard split)
res = frc_resolution(image_2d, spacing=0.065)

# 3D FSC - Single-image mode (uses checkerboard splitting)
res = fsc_resolution(image_3d, spacing=[0.2, 0.065, 0.065])
# Returns: {'xy': float, 'z': float}

# 3D FSC - Two-image mode (gold standard)
res = fsc_resolution(image_a, image_b, spacing=[0.2, 0.065, 0.065])

# 3D DCR - Single image, no threshold needed
res = dcr_resolution(image_3d, spacing=[0.2, 0.065, 0.065])
# Returns: {'xy': float, 'z': float}
```

## Method Comparison: FRC/FSC vs DCR

FRC/FSC and DCR measure fundamentally different properties, which can lead to different resolution estimates on the same data.

### What Each Method Measures

| Method | Measures | Formula | Resolution Criterion |
|--------|----------|---------|---------------------|
| **FRC/FSC** | Signal consistency between two images | `corr(FFT₁, FFT₂)` at each frequency | Where correlation drops below threshold (0.143) |
| **DCR** | Cumulative power distribution | `Σ\|F(k)\| / √(power × count)` | Where decorrelation curve peaks |

### Key Differences

**FRC/FSC (Correlation-based):**
- Compares two independent measurements (or checkerboard splits)
- Asks: "Is there reproducible signal at this frequency?"
- Insensitive to power distribution - only cares about consistency
- High-frequency signal with low power can still show high correlation

**DCR (Decorrelation-based):**
- Uses a single image
- Correlates original FFT `I(k)` with normalized FFT `I_n(k) = I(k)/|I(k)|`
- Result is still weighted by `|I(k)|` (power spectrum)
- Peaks where: `|F(r)| ≈ average |F| up to r`
- Dominated by where most spectral power is concentrated

### Why Results Can Differ Significantly

Consider STED microscopy data with ~100nm XY resolution:

| Frequency Region | Spectral Power | FRC/FSC Response | DCR Response |
|------------------|----------------|------------------|--------------|
| Low (r~0.2) | HIGH | High correlation | **Peak here** (most power) |
| High (r~0.8) | LOW | High correlation (signal consistent) | Already declining |

**Example: STED Tubulin Data**

| Method | XY Resolution | Z Resolution | Notes |
|--------|---------------|--------------|-------|
| FSC (two-image) | ~140 nm | ~600 nm | Gold standard |
| FSC (single-image) | ~100 nm | ~300 nm | Checkerboard split |
| DCR | ~370 nm | ~450 nm | Power-weighted |

DCR gives ~3x worse XY resolution because:
1. STED creates high-frequency signal (fine structures)
2. This signal has **low amplitude** compared to low-frequency components
3. FSC detects it because it's **consistent** between measurements
4. DCR misses it because it has **low power**

### When to Use Each Method

| Scenario | Recommended Method | Rationale |
|----------|-------------------|-----------|
| Two acquisitions available | FSC (two-image) | Gold standard, truly independent noise |
| Single image, standard microscopy | FSC (single-image) | Good approximation via checkerboard split |
| Single image, high SNR | DCR | FSC may give NaN (correlation doesn't cross threshold) |
| Super-resolution (STED, PALM) | FSC | DCR underestimates due to power distribution |
| Quick quality check | DCR | No threshold selection needed |
| Confocal microscopy | Either | Both work well for typical confocal data |

### Mathematical Detail: Why DCR is Power-Weighted

DCR normalizes one side of the correlation but not both:

```
d(r) = corr(I, I_n · M_r)
     = Σ I(k) · conj(I(k)/|I(k)|) / √(Σ|I|² · count)
     = Σ |I(k)| / √(total_power × count)
```

The `|I(k)|` term in the numerator means d(r) is weighted by spectral magnitude. The normalization `I_n = I/|I|` removes magnitude from one operand, but the original `I(k)` still contributes its magnitude to the product.

## API Reference

### FRC (2D)

```python
res = frc_resolution(image_2d, spacing=0.065)

# With two images
res = frc_resolution(image_a, image_b, spacing=0.065)
```

### FSC (3D)

```python
# Single-image with checkerboard split
res = fsc_resolution(image_3d, spacing=[0.2, 0.065, 0.065])

# Two-image (gold standard)
res = fsc_resolution(image_a, image_b, spacing=[0.2, 0.065, 0.065])

# With isotropic resampling (recommended for anisotropic data)
res = fsc_resolution(image_3d, spacing=[0.5, 0.126, 0.126], resample_isotropic=True)

# With axial exclusion (avoids piezo artifacts)
res = fsc_resolution(image_3d, spacing=[0.2, 0.065, 0.065], exclude_axis_angle=10.0)
```

### DCR (2D/3D)

```python
# 2D DCR
res = dcr_resolution(image_2d, spacing=0.065)  # Returns float

# 3D DCR with angular sectoring (default)
res = dcr_resolution(image_3d, spacing=[0.2, 0.065, 0.065])
# Returns: {'xy': float, 'z': float}

# 3D DCR legacy mode (2D slices, faster)
res = dcr_resolution(image_3d, spacing=[0.2, 0.065, 0.065], use_sectioned=False)
```

## Key Parameters

### FSC Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spacing` | None | Physical spacing [Z, Y, X] in microns |
| `resample_isotropic` | False | Resample to isotropic voxels before FSC |
| `exclude_axis_angle` | 0.0 | Exclude frequencies within this angle from Z axis |
| `use_max_nyquist` | False | Extend frequency range to XY Nyquist |
| `backend` | 'hist' | 'hist' (fast, GPU-accelerated) |

### DCR Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spacing` | None | Physical spacing [Z, Y, X] in microns |
| `num_radii` | 100 | Number of radial sampling points |
| `num_highpass` | 10 | Number of high-pass filter levels |
| `smoothing` | 11 | Savitzky-Golay filter window for curve smoothing (None to disable) |
| `use_sectioned` | True | Use full 3D angular sectoring |

## 3D Angular Sectoring

Both FSC and DCR use azimuthal angular sectoring in the Y-Z plane following Koho et al. 2019:

- **phi ≈ 0° or 180°**: Z-dominated frequencies → Z resolution
- **phi ≈ 90° or 270°**: Y-dominated frequencies → XY resolution

With `angle_delta=45°`, 8 sectors are created and grouped:
- Z sectors: 0-45°, 135-180°, 180-225°, 315-360° (near Z axis)
- XY sectors: 45-90°, 90-135°, 225-270°, 270-315° (near XY plane)

### Anisotropic Data Handling

For anisotropic voxels (e.g., Z spacing >> XY spacing):

- **FSC**: Uses `use_max_nyquist=True` to extend radial bins to XY Nyquist
- **DCR**: Automatically uses sector-specific radial binning (XY Nyquist for XY sectors, Z Nyquist for Z sectors)

## Threshold Conventions

| Method | Threshold | Description |
|--------|-----------|-------------|
| FRC (2D) | 0.143 (1/7) | Fixed threshold |
| FSC (3D) | One-bit (SNRe=0.5) | Varies with voxels per bin |
| DCR | N/A | Peak-finding, no threshold |

## Key Files

- `frc.py` – FRC (2D) and FSC (3D) implementations
- `dcr.py` – DCR (Decorrelation Analysis) following Descloux et al. 2019
- `radial.py` – Shared radial binning utilities
- `analysis.py` – Curve fitting and resolution extraction

## Limitations

### FSC Limitations

1. **Limited Z extent**: <50 Z slices may give unreliable Z resolution
2. **High-SNR single-image**: Correlation may not cross threshold → NaN
3. **Requires noise independence**: Single-image splits have correlated noise

### DCR Limitations

1. **Power-weighted**: Underestimates resolution when high-frequency signal has low power
2. **Not ideal for super-resolution**: STED, PALM, STORM data often gives worse estimates than FSC
3. **Monotonic curves**: Low-SNR data may show no peak → returns infinity

## References

1. **DCR**: Descloux et al. (2019) "Parameter-free image resolution estimation based on decorrelation analysis", *Nature Methods* 16:918-924.
2. **FRC**: Nieuwenhuizen et al. (2013) "Measuring image resolution in optical nanoscopy", *Nature Methods* 10:557-562.
3. **FSC**: Koho et al. (2019) "Fourier ring correlation simplifies image restoration in fluorescence microscopy", *Nature Communications* 10:3103.
