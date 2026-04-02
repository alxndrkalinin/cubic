# Project Guide for AI Agents

This file provides guidance for AI agents working with the **cubic** repository.

## Project Overview

*cubic* is a Python library for morphometric analysis of multidimensional bioimages with optional CUDA acceleration. It provides a device-agnostic API that wraps SciPy/scikit-image and CuPy/cuCIM, enabling GPU acceleration of existing codebases by simply replacing import statements and transferring input arrays to the target device. Source code lives under the `cubic/` package and tests are located in `tests/`.

**Key Design Principle**: **All functions automatically support both CPU and GPU without any code changes** - they work with NumPy arrays (CPU) or CuPy arrays (GPU) based solely on the input array's device location. The same function call works on both devices; just transfer the input array to the desired device using `cubic.cuda` functions.

**Citation**: Kalinin et al. (2025) "cubic: CUDA-accelerated 3D BioImage Computing", ICCV Workshop.

## Directory Structure

- `cubic/` – Python package containing all library modules
- `tests/` – pytest test suite
- `examples/` – example notebooks and scripts (read-only)
- `build/` – build artifacts (should not be modified)
- `.github/workflows/` – CI configuration

## Key Modules

### Device Management (`cubic/cuda.py`)

Core utilities for device-agnostic computation:
- `CUDAManager` – Singleton managing CuPy/cuCIM resources
- `get_array_module(array)` – Returns `np` or `cp` based on array location (**use sparingly** - prefer `np.` directly)
- `asnumpy(array)` / `ascupy(array)` – Transfer arrays between CPU/GPU (**preferred** over direct CuPy calls)
- `to_device(array, device)` – Move array to specific device ("CPU" or "GPU")
- `to_same_device(source, reference)` – Move source array to same device as reference
- `check_same_device(*arrays)` – Verify all arrays are on the same device
- `get_device(array)` – Returns "CPU" or "GPU"

**Important**:
- `get_array_module()` should only be used when creating new arrays that must be on a specific device. For most operations, use `np.` directly - NumPy functions work on both NumPy and CuPy arrays through duck typing.
- **Always use `cubic.cuda` functions** for device operations (moving arrays, checking devices) rather than directly calling CuPy functions. This maintains the abstraction layer and ensures consistent behavior.

### Resolution Metrics (`cubic/metrics/spectral/`)

FRC (Fourier Ring Correlation), FSC (Fourier Shell Correlation), and DCR (Decorrelation Analysis) for image resolution estimation. See `cubic/metrics/spectral/README.md` for detailed documentation and API usage.

### Preprocessing (`cubic/preprocessing/`)

- `deconvolution.py` – Richardson-Lucy deconvolution with observer callbacks
- `richardson_lucy_xp.py` – Device-agnostic RL implementation
- `thresholding.py` – Thresholding utilities

### Segmentation (`cubic/segmentation/`)

- `cellpose.py` – Cellpose integration
- `segment_utils.py` – Segmentation utilities
- `_clear_border.py` – Border clearing operations

### Device-Agnostic Wrappers

- `cubic/scipy.py` – Proxy module for device-agnostic SciPy/cupyx.scipy access
- `cubic/skimage.py` – Proxy module for device-agnostic scikit-image/cuCIM access
- `cubic/cucim/` – CuCIM integration for GPU-accelerated image I/O

These modules automatically route function calls to CPU (NumPy/SciPy/scikit-image) or GPU (CuPy/cuCIM) implementations based on input array device.

### Image Utilities (`cubic/image_utils.py`, `cubic/image.py`)

- `Image` class – Simple container for image data and metadata with device management
- `checkerboard_split()` / `reverse_checkerboard_split()` – Split images for FRC/FSC calculation
- Window functions (Hamming, Tukey) and other image processing utilities

### Feature Extraction (`cubic/feature/`)

- `mesh.py` – Mesh-based feature extraction
- `voxel.py` – Voxel-based feature extraction

### Other Metrics (`cubic/metrics/`)

- `average_precision.py` – AP for segmentation quality assessment
- `skimage_metrics.py` – PSNR, SSIM (device-agnostic)
- `feature.py` – Morphology correlation metrics

## Coding Conventions

- Target Python version is **3.10+** and type annotations are required.
- Follow the existing style: snake_case for functions, PascalCase for classes, and triple-quoted docstrings.
- Ruff is used for linting and formatting; run with automatic fixes when possible.
- Keep functions and classes concise with descriptive names and inline comments for complex logic.

### ⚠️ **CRITICAL: Device-Agnostic Code Pattern**

**All functions in `cubic` automatically support both CPU and GPU without any code changes** - they work with NumPy arrays (CPU) or CuPy arrays (GPU) based solely on the input array's device location. The same function call works on both devices; just transfer the input array to the desired device using `cubic.cuda` functions.

**Avoid using `xp` (array module) interface as much as possible.** Prefer `np.` or array methods (`.func()`) to maximize code portability between NumPy and CuPy without modifications.

**Preferred approach** (use `np.` directly):
```python
import numpy as np

# NumPy functions work on both NumPy and CuPy arrays
result = np.fft.fftn(image)           # ✅ Works on both CPU/GPU arrays
result = np.abs(array)                # ✅ Works on both CPU/GPU arrays
result = np.bincount(bin_id, weights) # ✅ Works on both CPU/GPU arrays
result = np.sqrt(k0 * k0 + k1 * k1)   # ✅ Works on both CPU/GPU arrays
result = array.ravel()                 # ✅ Array methods work on both
result = array.astype(np.float32)     # ✅ Array methods work on both
```

**Avoid when possible** (using `xp` interface):
```python
from cubic.cuda import get_array_module

xp = get_array_module(array)
result = xp.fft.fftn(image)  # ⚠️ Only use when necessary
result = xp.asarray(data)     # ⚠️ Only use when creating new arrays on specific device
```

**When to use `xp`** (limited cases):
- Creating new arrays that must be on the same device as existing arrays: `xp.asarray()`, `xp.zeros()`, `xp.ones()`
- Device-specific functions not available in NumPy: `xp.fft.fftfreq()` for device placement
- Functions that don't work with NumPy's duck-typing: rare, prefer `np.` when possible

**Device operations** (use `cubic.cuda` functions):
When you need to move arrays between devices or check device placement, **always use functions from `cubic.cuda`** rather than directly calling CuPy functions:

```python
from cubic.cuda import asnumpy, ascupy, to_device, to_same_device, check_same_device, get_device

# ✅ Preferred: Use cubic.cuda functions
cpu_array = asnumpy(gpu_array)              # Move to CPU
gpu_array = ascupy(cpu_array)                # Move to GPU
target_array = to_device(source_array, "GPU")  # Move to specific device
aligned_array = to_same_device(array1, array2)  # Move to same device as reference
check_same_device(array1, array2)           # Verify same device
device = get_device(array)                   # Check current device

# ❌ Avoid: Direct CuPy calls
import cupy as cp
cpu_array = cp.asnumpy(gpu_array)  # Don't do this - breaks abstraction
```

**Rationale**: Using `np.` directly allows code to work seamlessly with both NumPy and CuPy arrays through duck typing. This maximizes portability and allows users to port NumPy code in/out with minimal modifications. The `xp` interface should only be used when absolutely necessary for device placement or when NumPy functions don't support CuPy arrays (rare). For device operations, always use `cubic.cuda` functions to maintain the abstraction layer and ensure consistent behavior.

## Commit Workflow

Before running `git commit`, follow these steps every time:

1. **List** all pending changes (`git diff --stat`).
2. **Group** them by concept — e.g., "device-abstraction cleanup", "input validation", "docstring fixes", "new tests". Each group becomes one commit.
3. **Plan** the N commits (N ≥ 1) before executing any.
4. **Stage and commit** each group separately with a focused message.

Never group by workflow step ("plan fixes", "review fixes"). Always group by what the change *is*.

## Programmatic Checks

Before committing, ensure the following commands succeed from the repository root:

```bash
ruff check .
ruff format --check .
mypy --ignore-missing-imports cubic/
pytest
```

Tests may skip automatically when GPU hardware is not available but should still be executed.

## Pull Request Guidelines

- Provide a clear description of the change and its rationale.
- Ensure all programmatic checks pass.
- Keep PRs focused on a single objective and reference related issues when relevant.

## References

Key paper for understanding the library:

- **cubic**: Kalinin et al. (2025) "cubic: CUDA-accelerated 3D BioImage Computing", ICCV Workshop.

For resolution metrics (FRC/FSC/DCR), see `cubic/metrics/spectral/README.md` for detailed documentation and references.
