# Examples

## Notebooks

| Notebook | Description | Data |
|----------|-------------|------|
| [`resolution_estimation_2d.ipynb`](../notebooks/resolution_estimation_2d.ipynb) | 2D FRC + DCR on STED datasets (Tubulin, Vimentin, COS7) | STED TIF files |
| [`resolution_estimation_3d.ipynb`](../notebooks/resolution_estimation_3d.ipynb) | 3D FSC + DCR on pollen confocal data (Koho et al. 2019) | Pollen ND2 |
| [`deconvolution_iterations_3d.ipynb`](../notebooks/deconvolution_iterations_3d.ipynb) | Track RL deconvolution with PSNR, SSIM, FSC, and DCR | Astrocyte TIF + PSF |

## Benchmark Scripts

| Script | Purpose | Data | Run |
|--------|---------|------|-----|
| `benchmark_resolution_methods.py` | Compare FRC/FSC/DCR across crop sizes | STED tubulin + astrocyte | `python benchmark_resolution_methods.py` |

## Plotting Utilities

Plotting functions for resolution curve visualization are in the main package:

```python
from cubic.metrics.spectral.plot import plot_fsc_sectors, plot_dcr_sectors, plot_frc_curve, plot_dcr_curves
```

- `plot_fsc_sectors(fsc_data, spacing, ...)` — FSC correlation curves for XY/Z sectors
- `plot_dcr_sectors(sector_data, ...)` — DCR decorrelation curves for XY/Z sectors
- `plot_frc_curve(frc_data, ax)` — 2D FRC correlation + threshold + crossing
- `plot_dcr_curves(radii, curves, peaks, ax)` — 2D DCR decorrelation curves

Requires matplotlib: `pip install cubic[plot]`

## Output

All scripts save plots/CSVs to `benchmark_plots/` subdirectory (not repo root).

## Configuration

Most scripts use Hydra. Config files are in `conf/`. Override via CLI:

```bash
python sfsc_paper_comparison.py angle_delta=30 bin_delta=5
python benchmark_resolution_methods.py crop.size=1024
```
