# Plan: Fix 3D Sectored DCR to Match Original Algorithm

## Context

The 3D sectored DCR in cubic produces XY = 1.42 µm on pollen data (expected ~0.59 µm from Koho et al. 2019 / FSC gives 0.58 µm). Root cause analysis comparing cubic's implementation against the original MATLAB ImDecorr (Descloux et al. 2019), the paper supplement, and NanoPyx identified multiple deviations from the original algorithm that compound to produce incorrect results.

## Difference Analysis: Impact on Pollen XY Failure

### 1. k_max normalization (CRITICAL — root cause of XY failure)

| | ImDecorr | cubic 3D sectored |
|--|----------|-------------------|
| **k_max** | 2D only — single Nyquist | `_kmax_phys(shape, spacing)` = **min-Nyquist** (Z axis) for ALL sectors |

**Impact**: For pollen (spacing Z=0.25, XY=0.0777 µm), Z-Nyquist = 2.0 µm⁻¹, XY-Nyquist = 6.43 µm⁻¹. The XY signal cutoff at ~1.7 µm⁻¹ normalizes to r ≈ 0.85 (near boundary) when using Z-Nyquist, but r ≈ 0.26 when using XY-Nyquist. Normalizing to min-Nyquist pushes the XY signal to the boundary where peak finding fails.

**Fix**: Use **per-sector k_max** — Z-Nyquist for Z sector, XY-Nyquist for XY sector. This is analogous to how `fsc_resolution` uses `use_max_nyquist` for XY sectors.

**File**: `cubic/metrics/spectral/dcr.py`, `_compute_decorrelation_curve_sectioned` (line 416) and `_dcr_curve_3d_sectioned` (line 517).

### 2. No d₀ (unfiltered) curve (HIGH — affects sigma range)

| | ImDecorr | cubic |
|--|----------|-------|
| **d₀** | Computed first, peak r₀ used to set `gMax = 2/r₀` | Not computed |

**Impact**: The unfiltered curve's peak position anchors the HP sigma range. Without it, cubic's fixed sigma range [0.5, min(shape)/2] wastes levels in irrelevant territory. The supplement (Supplementary Note 1.1, Section III) explicitly states sigmas are distributed between `2/r₀` and `0.15`.

**Fix**: Compute d₀ first (no HP filtering). Use r₀ to set adaptive `sigma_max = 2/r₀`. Also add r₀ as a bonus peak candidate (ImDecorr line 196: `kc(end+1) = k0`).

**File**: `cubic/metrics/spectral/dcr.py`, `_dcr_curve_3d_sectioned` (line 471+).

### 3. HP sigma range (HIGH — wrong concentration of levels)

| | ImDecorr | NanoPyx | cubic |
|--|----------|---------|-------|
| **Weakest HP** | `size(im)/4` (~128) + adaptive `gMax = 2/r₀` | g_min = 0.14 | σ_max = min(shape)/2 |
| **Strongest HP** | g = 0.15 (σ ≈ 0.096) | auto | σ_min = 0.5 |
| **Range focus** | Concentrated around signal region via adaptive gMax | Adaptive | Fixed, spread over 3 orders of magnitude |

**Impact**: ImDecorr concentrates its 10 HP levels in the narrow band [2/r₀, 0.15] (in Fourier-domain g units), centered on the signal. cubic spreads 10 levels from σ=0.5 to σ=90.5 — most are in irrelevant very-weak-HP territory where curves are monotonically increasing (no useful peak).

**Fix**: After computing d₀, set `sigma_max ≈ 2/r₀` (converted from ImDecorr's Fourier-domain g). Keep `sigma_min = 0.15` (matching ImDecorr's strongest HP) or scale by the σ↔g conversion. The key point: distribute levels between where the signal is and where it isn't.

**Conversion between ImDecorr g and cubic σ**: ImDecorr applies HP in Fourier domain as `H(R) = 1 - exp(-2g²R²)`, while cubic applies spatial Gaussian subtraction giving `H(f) = 1 - exp(-2π²σ²f²)`. With R = 2f (Nyquist at R=1, f=0.5), the equivalence is **σ_cubic ≈ 2g/π**.

**File**: `cubic/metrics/spectral/dcr.py`, `_generate_highpass_sigmas` (line 543+) and callers.

### 4. Refinement off by default (MEDIUM — loses precision)

| | ImDecorr | NanoPyx | cubic |
|--|----------|---------|-------|
| **Refinement** | Always 2-pass | Always 2-pass | `refine=False` (off) |

**Impact**: Supplement shows the 2-pass refinement narrows both sigma and frequency ranges around the coarse peaks. Without it, resolution is limited by coarse radial sampling. The supplement (Note 1.1, Section III) describes this as part of the core algorithm, not an optional enhancement.

**Fix**: Change default to `refine=True`. The existing refinement code (lines 222-285) already implements the NanoPyx convention.

**File**: `cubic/metrics/spectral/dcr.py`, `dcr_curve` (line 105), `dcr_resolution` (line 564+), `_dcr_curve_3d_sectioned` (needs refinement added).

### 5. Smoothing (MEDIUM — not in original, can mask real peaks)

| | ImDecorr | NanoPyx | cubic |
|--|----------|---------|-------|
| **Smoothing** | None | None | Savitzky-Golay (window=11) |

**Impact**: The supplement (Supplementary Note 1) explicitly states d(r) is "intrinsically smooth and noiseless" because "two neighboring values d(r) and d(r+Δr) share a large amount of information" (cumulative integration acts as natural smoothing). External smoothing is unnecessary and can shift peak positions, especially for sharp peaks in narrow sectors.

**Fix**: Change default to `smoothing=None`. Keep the parameter available for edge cases but don't apply by default.

**File**: `cubic/metrics/spectral/dcr.py`, `_compute_decorrelation_curve` (line 305), `_compute_decorrelation_curve_sectioned` (line 383).

### 6. floor(1000*cc)/1000 rounding (MEDIUM — prevents spurious peaks)

| | ImDecorr | NanoPyx | cubic |
|--|----------|---------|-------|
| **Rounding** | `floor(1000*cc)/1000` | None | None |

**Impact**: ImDecorr floors each d(r) value to 3 decimal places. This digitizes the curve to 0.001 steps, which means monotonically increasing curves with increments < 0.001/step are flattened — their "peaks" have zero prominence and are automatically rejected. This is an elegant implicit solution to the boundary artifact problem we hit in the XY sector.

**Fix**: Not needed if the k_max normalization (fix #1) is correct — curves should have genuine peaks. But could add as a robustness measure. Lower priority than fixes 1-5.

### 7. SNR gate (LOW — secondary protection)

| | ImDecorr | cubic |
|--|----------|-------|
| **SNR gate** | `kc(SNR < 0.05) = 0` — reject peaks with amplitude < 0.05 | None |

**Impact**: Rejects peaks from noise-only curves. The supplement says SNR < 0.1 produces no reliable peaks.

**Fix**: Add `if a_peak < 0.05: continue` in `_find_peak_in_curve` or as a post-filter on peak collection.

**File**: `cubic/metrics/spectral/dcr.py`, `_find_peak_in_curve` or `_dcr_curve_3d_sectioned`.

### 8. Nr = 100 vs 50 (NEGLIGIBLE)

The supplement's sensitivity analysis shows N_r ∈ [30, 100] gives ±1.3 nm variation. Nr=100 is fine — marginally better than 50.

### 9. Local maximum check in peak finder (ALREADY FIXED)

The local descent check we added earlier correctly rejects points on monotonically increasing slopes. This aligns with ImDecorr's `getDcorrLocalMax` which iteratively trims from the boundary and checks prominence against subsequent minimum.

## Implementation Plan

### Step 1: Per-sector k_max in `_compute_decorrelation_curve_sectioned`
- Compute `k_max_xy = _kmax_phys_max(shape, spacing)` (XY Nyquist) and `k_max_z = _kmax_phys(shape, spacing)` (min/Z Nyquist)
- Normalize radii to the sector's own k_max
- Return sector-specific k_max in the results dict

### Step 2: Compute d₀ and adaptive sigma range in `_dcr_curve_3d_sectioned`
- Before the HP loop, compute d₀ (unfiltered) for each sector
- Find r₀ (peak of d₀) per sector
- Set `sigma_max = 2/r₀` (converted from ImDecorr's g convention: σ = 2g/π, g=2/r₀ → σ ≈ 4/(πr₀))
- Add r₀ as a candidate peak

### Step 3: Enable refinement by default
- Change `refine=False` → `refine=True` in function signatures
- Ensure `_dcr_curve_3d_sectioned` has refinement logic (currently only `dcr_curve` has it)

### Step 4: Disable smoothing by default
- Change `smoothing=11` → `smoothing=None` in `_compute_decorrelation_curve` and `_compute_decorrelation_curve_sectioned`

### Step 5: Add SNR gate
- Reject peaks with amplitude < 0.05 (matching ImDecorr)

### Step 6: Update notebook and verify
- Rerun `examples/notebooks/resolution_estimation_3d.ipynb`
- Verify XY resolution ≈ 0.5–0.7 µm (close to FSC's 0.58 µm and paper's 0.59 µm)
- Verify Z resolution ≈ 2.5–4.0 µm (reasonable range)
- Verify vertical lines in plots align with visible curve peaks

## Key Files

- `cubic/metrics/spectral/dcr.py` — all DCR changes (primary)
- `cubic/metrics/spectral/radial.py` — `_kmax_phys_max` for per-sector Nyquist
- `examples/plot_utils.py` — already updated with vertical lines
- `examples/notebooks/resolution_estimation_3d.ipynb` — verification

## Verification

```bash
# 1. Run existing tests
pytest tests/metrics/frc/ -v

# 2. Rerun notebook
jupyter nbconvert --execute examples/notebooks/resolution_estimation_3d.ipynb

# 3. Verify pollen results:
#    - DCR XY ≈ 0.5-0.7 µm (was 1.42, should be near FSC's 0.58)
#    - DCR Z ≈ 2.5-4.0 µm
#    - Vertical lines at genuine peaks
#    - FSC results unchanged

# 4. Run astrocyte benchmark (256,512,1024 crops) on GPU
#    to verify stability across crop sizes
```

## References

- **Original ImDecorr**: `/hpc/mydata/alex.kalinin/ImDecorr/` — Descloux et al. 2019 MATLAB implementation
- **NanoPyx**: `/hpc/mydata/alex.kalinin/nanopyx/` — Python re-implementation
- **DCR paper**: Descloux et al. (2019) "Parameter-free image resolution estimation based on decorrelation analysis", *Nature Methods* 16:918-924
- **DCR supplement**: `cubic/metrics/spectral/41592_2019_515_MOESM1_ESM.pdf`
- **SFSC paper**: Koho et al. (2019) "Fourier ring correlation simplifies image restoration in fluorescence microscopy", *Nature Communications* 10:3103
