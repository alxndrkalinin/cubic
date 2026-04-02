# cubic

`cubic` is a Python library that accelerates processing and analysis of
multidimensional (2D/3D+) bioimages using CUDA.
By leveraging GPU-enabled operations where possible, it offers substantial
speed ups over purely CPU-based approaches.
`cubic`'s device-agnostic API wraps scipy/scikit-image and cupy/cuCIM,
allowing users to add GPU acceleration to existing codebases by simply replacing import
statements and transferring input arrays to the target device.
It also provides custom GPU-accelerated implementations of additional
features, including Fourier Ring and Shell Correlation for image resolution,
faster Richardson-Lucy deconvolution, average precision (AP) for segmentation
quality assessment, and other features.

## Getting started

### Dependencies
* Python >=3.10
* numpy/scipy/scikit-image
* [optional] CUDA>=11.x, [CuPy](https://docs.cupy.dev/en/stable/install.html), [cuCIM](https://github.com/rapidsai/cucim?tab=readme-ov-file#install-cucim)
* [optional] Cellpose for segmentation

### Installation

Install optional CUDA dependencies if GPU support is needed.

Install from PyPI:

```bash
pip install cubic
```

Or install from source:

```bash
git clone https://github.com/alxndrkalinin/cubic.git
cd cubic
pip install .
```

Optional extras from `pyproject.toml` enable additional functionality:

```bash
# mesh feature extraction
pip install '.[mesh]'
# segmentation via Cellpose
pip install '.[cellpose]'
# developer tools (pre-commit, pytest)
pip install -e '.[dev]'
# install everything
pip install -e '.[all]'
```

### Testing
Run style checks and tests using `pre-commit` and `pytest`:

```bash
pre-commit run --all-files
pytest
```

### Contributing
Contributions and bug reports are welcome. Install development dependencies and
set up pre-commit hooks:

```bash
pip install -e '.[dev]'
pre-commit install
```

Pre-commit will then run style checks automatically. Please open an issue or
pull request on GitHub.

## Usage

### Example Notebooks

| Notebook | Description |
|----------|-------------|
| [Resolution Estimation (2D)](examples/notebooks/resolution_estimation_2d.ipynb) | FRC and DCR on STED microscopy data |
| [Resolution Estimation (3D)](examples/notebooks/resolution_estimation_3d.ipynb) | FSC and DCR on 3D confocal pollen data |
| [Split Comparison (FRC/FSC)](examples/notebooks/split_comparison_frc_fsc.ipynb) | Checkerboard vs binomial splitting for single-image FRC/FSC |
| [Deconvolution Iterations (3D)](examples/notebooks/deconvolution_iterations_3d.ipynb) | RL deconvolution stopping criteria via PSNR, SSIM, FSC, DCR |
| [3D Monolayer Segmentation](examples/notebooks/segmentation_3d_monolayer.ipynb) | 3D nuclei and cell segmentation of hiPSC monolayer |
| [3D Feature Extraction](examples/notebooks/feature_extraction_3d.ipynb) | GPU-accelerated regionprops on 3D fluorescence data |

## Citation
If you use `cubic` in your research, please cite it:

```bibtex
@inproceedings{kalinin2025cubic,
  title={cubic: CUDA-accelerated 3D BioImage Computing},
  author={Kalinin, Alexandr A and Carpenter, Anne E and Singh, Shantanu and O’Meara, Matthew J},
  booktitle={International Conference on Computer Vision Workshops (ICCVW)},
  year={2025},
  organization={IEEE}
}
```
