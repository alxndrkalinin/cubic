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

### Image Splitting Methods

For single-image FRC/FSC, cubic supports two splitting strategies:

| Property | Checkerboard (default) | Binomial (`split_type="binomial"`) |
|----------|----------------------|-----------------------------------|
| Output size | Stride-2 subsampled (half per dim) | Same as input |
| Calibration correction | Yes (diagonal shift) | No |
| Reverse-split averaging | Yes | N/A (use `n_repeats` instead) |
| Input requirement | Any image | Photon counts or Poisson rates |
| Sampling requirement | Adequate oversampling | Any |
| Uncertainty quantification | Fwd/rev averaging | Repeat M times → curve std |

**When to use which:**
- **Binomial + counts mode**: Raw camera data with known gain/offset/readout noise. Recommended for data where the checkerboard subsampling may bias high-frequency statistics.
- **Checkerboard** (default): No calibration available, or compatibility with existing workflows (Koho et al. 2019).
- **Poisson thinning**: Fallback heuristic for float/deconvolved images — measures self-consistency of a noise model, not physical resolution.

```python
# Binomial split (raw camera data)
res = frc_resolution(image_2d, spacing=0.065, split_type="binomial",
                     gain=2.0, offset=100.0)

# Binomial split with uncertainty (multiple repeats)
result = calculate_frc(image_2d, spacing=0.065, split_type="binomial",
                       n_repeats=5, rng=42)
# result.correlation["correlation-std"]  # per-ring std
# result.resolution["resolution-std"]    # resolution uncertainty

# 3D FSC with binomial split
res = fsc_resolution(image_3d, spacing=[0.2, 0.065, 0.065],
                     split_type="binomial", gain=2.0, offset=100.0)
```

Note: For deconvolved images, `counts_mode="poisson_thinning"` treats pixel values as
Poisson rates. The resulting 1FRC measures the effective reproducible bandwidth of the
processing pipeline, not ground-truth resolution.

**Reference:** Rieger, Droste, Gerritsma, ten Brink, Stallinga. "Single image Fourier
ring correlation." *Optics Express* 32(12):21767, 2024.

## Key Parameters

### FSC

| Parameter | Default | Description |
|-----------|---------|-------------|
| `spacing` | None | Physical spacing [Z, Y, X] in microns; None means pixels |
| `resample_isotropic` | False | Resample to isotropic voxels before FSC |
| `angle_delta` | 15 | Angular sector width in degrees; must divide 90 |
| `exclude_axis_angle` | 0.0 | Exclude frequencies near Z axis (degrees) |
| `axial_floor_factor` | 2.0 | Report `z` as `nan` below this multiple of the pre-resampling Z spacing; 0 disables |
| `backend` | `"hist"` | GPU-accelerated histogram-based backend |

`fsc_resolution` returns `{"xy": ..., "z": ...}`. A sector centred on polar angle
theta measures the shell radius `|k| = k_z / cos(theta)`, not `k_z`, so the two
keys are extracted differently:

- `xy` comes from the most XY-dominated sector, reported as measured.
- `z` comes from the highest sector *below* 45 degrees that crosses the
  threshold, with its period divided by `cos(theta)` to project onto the Z axis.
  Sectors at or above 45 degrees are XY-limited and are never used for `z`; when
  none below 45 degrees crosses, `z` is `nan` with a warning rather than an
  in-plane number relabelled as axial.

With `resample_isotropic=True` the axial correction changes: interpolating Z up
to isotropic voxels adds no information, so the real axial band limit stays at
the original Z Nyquist while the grid now runs to the XY one. The Koho et al.
(2019) eq. (5) factor `1 + (z_spacing/xy_spacing - 1)·|cos(theta)|` converts back,
and is used in place of the geometric projection (the two address different
errors and are not combined). This is the path that reproduces the paper: on
their Fig. 4b pollen stack it gives XY 0.586 µm / Z 4.38 µm against a published
0.59 / 3.91.

`spacing=None` is the same frequency grid as `spacing=1.0` — cycles per pixel —
so resolutions come back in pixels.

#### Axial Nyquist floor

A flat FSC crossing cannot land below the sampling limit, because no Fourier
shell exists past Nyquist. Both axial corrections above break that guarantee:
they rescale a crossing that *was* on the grid, so the product is unbounded.
`axial_floor_factor` (default `2.0`) re-imposes the limit, reporting `nan` with a
warning when `z` comes out finer than `2 × the original Z spacing`. The factor
multiplies the spacing captured *before* resampling — interpolating Z refines the
grid but adds no information, so the band limit does not move.

This restores a property the other methods already have. Descloux's DCR searches
over `linspace(0, 1) × k_max` and inverts as `2 · spacing / k_c`, so it is
structurally incapable of returning less than `2 · spacing`; ours additionally
caps the peak search at `r_max=0.9`. Koho et al. (2019) require
`d_pixel ≤ d_min / (2√2)` for the single-image checkerboard split — a stricter
`2√2 ≈ 2.83` floor, available as `axial_floor_factor=2*np.sqrt(2)`. Rieger et al.
(2024) likewise note the method needs the derived resolution to sit "well above
the sampling distance", and Diebolder et al. (2015) caution that conical FSC
"should be seen as a qualitative tool for comparison of resolution isotropies
rather than a quantitative method that yields absolute resolution values", since
measurements "might rather reflect the spatial sampling".

Pass `axial_floor_factor=0.0` to disable the check. Only `z` needs it: `xy` is
reported as measured, so the frequency grid already bounds it at `2 × spacing_xy`.
The deprecated `backend="mask"` path applies no axial correction and no floor.

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
- `radial.py` — Shared radial binning utilities and per-bin reducers
- `iterators.py` — Fourier ring/shell mask iterators (`backend="mask"`)
- `analysis.py` — Curve fitting and resolution extraction
- `plot.py` — Plotting utilities (requires matplotlib)

## References

1. Descloux et al. (2019) "Parameter-free image resolution estimation based on decorrelation analysis", *Nature Methods* 16:918-924.
2. Nieuwenhuizen et al. (2013) "Measuring image resolution in optical nanoscopy", *Nature Methods* 10:557-562.
3. Koho et al. (2019) "Fourier ring correlation simplifies image restoration in fluorescence microscopy", *Nature Communications* 10:3103.
