### Data for running examples

Most example notebooks auto-download their datasets on first run using [pooch](https://www.fatiando.org/pooch/) into this directory (`examples/data`). The sources, filenames, and (where helpful) direct download links are listed below.

#### 1. Estimating the number of deconvolution iterations

Used in `examples/notebooks/deconvolution_iterations_3d.ipynb`. Auto-downloaded on first run using pooch.

- **`astr_vpa_hoechst.tif`** — Hoechst-stained astrocyte nuclei (3D stack, 16-bit). Yokogawa CQ1 confocal, voxel 0.1625 × 0.1625 × 0.3 µm.
- **`astr_vpa_hoechst_psf_na095_cropped.tif`** — Theoretical 3D PSF (NA = 0.95) modeled with the Richards & Wolf algorithm from the [PSFGenerator](https://bigwww.epfl.ch/algorithms/psfgenerator/) ImageJ/Fiji plugin. Center-cropped to 30 × 210 × 210 (≥ 99.5% of energy).

Source: [Zenodo record](https://doi.org/10.5281/zenodo.20514102) (DOI [10.5281/zenodo.20514102](https://doi.org/10.5281/zenodo.20514102)).

#### 2. 2D resolution estimation (FRC and DCR)

Used in `examples/notebooks/resolution_estimation_2d.ipynb`.

- **`Tubulin_057nm.tif`** — Tubulin STED (56.5 nm pixels). Abberior Expert Line STED, 633 nm excitation, 100x/1.4 objective.
  [download from figshare](https://ndownloader.figshare.com/files/15202592)
  Source: Koho et al. (2019) *Nat. Commun.* 10:3103, [figshare collection](https://doi.org/10.6084/m9.figshare.c.4511663.v1).

- **`Vimentin_029nm.tif`** — Vimentin STED (29 nm pixels). Same microscope as tubulin.
  [download from figshare](https://ndownloader.figshare.com/files/15202565)
  Source: Koho et al. (2019) *Nat. Commun.* 10:3103, [figshare collection](https://doi.org/10.6084/m9.figshare.c.4511663.v1).

- **`demo_COS7_a-tub_abberior_star635_confocal_STED.tif`** — COS7 alpha-tubulin STED (15 nm pixels). Leica TCS SP8 STED 3X, 634 nm excitation, 775 nm depletion, 100x/1.40 objective.
  [download from GitHub](https://github.com/Ades91/ImDecorr/raw/refs/heads/master/test_image.tif) (saved as `test_image.tif` in source repo)
  Source: Descloux et al. (2019) *Nat. Methods* 16:918-924, [ImDecorr](https://github.com/Ades91/ImDecorr).

#### 3. 3D resolution estimation (FSC and DCR)

Used in `examples/notebooks/resolution_estimation_3d.ipynb`.

- **`40x_TAGoff_z_galvo.nd2`** — Pollen confocal (512x512x181, voxel 78x78x250 nm). Nikon A1 confocal, 40x/1.2 water, 488 nm excitation, GaAsP detector.
  [download from figshare](https://ndownloader.figshare.com/files/15203144)
  Source: Koho et al. (2019) *Nat. Commun.* 10:3103, [figshare dataset](https://doi.org/10.6084/m9.figshare.8159165.v1).

#### 4. 3D cell monolayer segmentation

Used in `examples/notebooks/segmentation_3d_monolayer.ipynb`. Auto-downloaded on first run using pooch.

- **`3d_monolayer_xy1_ch0.tif`** — Membrane channel (60x256x256, uint16).
- **`3d_monolayer_xy1_ch1.tif`** — Mitochondria channel (60x256x256, uint16).
- **`3d_monolayer_xy1_ch2.tif`** — DNA channel (60x256x256, uint16).
- **`3d_monolayer_xy1_ch2_NucleiLabels.tiff`** — CellProfiler nuclei reference labels.
- **`3d_monolayer_xy1_ch0_CellsLabels.tiff`** — CellProfiler cell reference labels.

Source: hiPSC data from the Allen Institute for Cell Science, provided with the
[CellProfiler 3D monolayer tutorial](https://github.com/CellProfiler/tutorials/tree/master/3d_monolayer).
[GitHub release assets](https://github.com/alxndrkalinin/cubic/releases/tag/v0.7.0a1).
