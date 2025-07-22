# cubic

cubic is a Python library for morphometric analysis of multidimensional
bioimages with CUDA acceleration. It was created to streamline common
morphology workflows on large 3D microscopy data. The library provides tools
for deconvolution, segmentation (via Cellpose integration) and feature
extraction, all accessible from a simple Python API. By leveraging GPU enabled
operations where possible, cubic offers substantial speed ups over purely
CPU based approaches.

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
See the example notebooks in `examples/notebooks/` for demonstrations of
deconvolution and segmentation workflows.

## Citation
If you use `cubic` in your research, please cite it:

```bibtex
@inproceedings{kalinin2025cubic,
  title={cubic: CUDA-accelerated 3D BioImage Computing},
  author={Kalinin, Alexandr A and Carpenter, Anne E and Singh, Shantanu and O’Meara, Matthew J},
  booktitle={International Conference on Computer Vision Workshop (ICCVW)},
  year={2025},
  organization={IEEE}
}
```
