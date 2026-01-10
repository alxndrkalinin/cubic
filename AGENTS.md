# Project Guide for AI Agents

This file provides guidance for AI agents working with the **cubic** repository.

## Project Overview

*cubic* is a Python library for morphometric analysis of multidimensional bioimages with optional CUDA acceleration.  Source code lives under the `cubic/` package and tests are located in `tests/`.

## Directory Structure

- `cubic/` – Python package containing all library modules
- `tests/` – pytest test suite
- `examples/` – example notebooks and data (read‑only)
- `build/` – build artefacts (should not be modified)
- `.github/workflows/` – CI configuration

## Key Implementation Details

### FRC/FSC (Fourier Ring/Shell Correlation)

Located in `cubic/metrics/frc/`. Two backends with identical binning but different algorithms:

- **Mask backend** (default): Iterator-based, float64 precision, CPU-only. Uses `FourierRingIterator` (2D) and `FourierShellIterator` (3D) classes in `iterators.py`.
- **Hist backend**: Histogram-based, float32 precision, CPU/GPU support. Uses `radial_bin_id()` + `np.bincount()` in `radial.py`.

Both backends use `radial_edges()` from `radial.py` for consistent bin edge definitions (divides Nyquist range evenly). Results differ by ~1-2% due to precision/order-of-operations. When modifying FRC/FSC code, ensure both backends remain consistent and test with `backend` parameter.

## Coding Conventions

- Target Python version is **3.10+** and type annotations are required.
- Follow the existing style: snake_case for functions, PascalCase for classes, and triple‑quoted docstrings.
- Ruff is used for linting and formatting; run with automatic fixes when possible.
- Keep functions and classes concise with descriptive names and inline comments for complex logic.

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

