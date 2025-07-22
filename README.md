# cubic

`cubic` is a Python library that accelerates processing and analysis of
multidimensional (2D/3D+) bioimages using CUDA.
By leveraging GPU-enabled operations where possible, it offers substantial
speed ups over purely CPU-based approaches.
`cubic`'s device-agnostic API wraps scipy/scikit-image and cupy/cuCIM,
allowing to add GPU acceleration to existing codebases by simply replacing import
statement and and transferring input arrays to the target device.
It also provides custom GPU-accelerated implementations of additional
features, including Forier Ring and Shell Correlation for image resolution,
faster Richardson-Lucy deconvolution, average precision (AP) for segmentation
quality assesement, and other.

## Getting started

### Dependencies
* Python >=3.10
* numpy/scipy/scikit-image
* [optional] CUDA 11.x-12.x, cupy, cuCIM
* [optional] Cellpose for segmentation

### Installation
Clone the repository and install the base library:

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
See the example notebooks in `examples/notebooks/`.

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
