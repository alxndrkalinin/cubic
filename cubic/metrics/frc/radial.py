"""Radial binning utilities for histogram-based FRC/FSC."""

from collections.abc import Sequence

import numpy as np


def radial_edges(
    shape: tuple[int, ...],
    bin_delta: float = 1.0,
    spacing: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build uniform radial bin edges and centers for 2D/3D unshifted FFT grids.

    bin_delta is always in index bins. When spacing is provided, it converts
    to physical frequency units internally.

    Args:
        shape: Image shape (2D or 3D)
        bin_delta: Bin width in index bins (default: 1.0)
        spacing: Physical spacing per axis. None uses index units,
                 given uses physical frequency (cycles per length).

    Returns
    -------
        edges: (M+1,) radial bin edges from 0 to kmax
        radii: (M,) radial bin centers (midpoints)
    """
    if bin_delta <= 0:
        raise ValueError("bin_delta must be > 0")

    ndim = len(shape)
    if spacing is not None:
        if len(spacing) != ndim:
            raise ValueError(f"spacing length {len(spacing)} must match dims {ndim}")
        # Physical frequency step per index bin along each axis: Δk_i = 1/(n_i·spacing_i)
        dk_axes = [1.0 / (n * sp) for n, sp in zip(shape, spacing)]
        step = bin_delta * min(dk_axes)
        axes = [np.fft.fftfreq(n, d=sp) for n, sp in zip(shape, spacing)]
    else:
        # Index units: fftfreq(n) * n gives [0, 1, 2, ..., n//2, -(n//2), ..., -1]
        step = bin_delta
        axes = [np.fft.fftfreq(n) * n for n in shape]

    # Maximum radial frequency
    kmax = min(np.max(np.abs(ax)) for ax in axes)

    # Build edges from 0 to kmax with step size
    nb = max(1, int(np.ceil(kmax / step)))
    edges = np.linspace(0.0, kmax, nb + 1, dtype=np.float64)
    radii = 0.5 * (edges[:-1] + edges[1:])
    return edges, radii


def radial_bin_id(
    shape: tuple[int, ...],
    edges: np.ndarray,
    spacing: Sequence[float] | None = None,
) -> np.ndarray:
    """
    Compute radial bin ID for each voxel in unshifted FFT grid.

    Args:
        shape: Image shape (2D or 3D)
        edges: Radial bin edges from radial_edges()
        spacing: Physical spacing per axis (None for index units)

    Returns
    -------
        Flattened int32 array with bin_id ∈ [0, nbins-1].
        DC term (K≈0) is excluded with bin_id = -1.
    """
    ndim = len(shape)
    if spacing is not None:
        if len(spacing) != ndim:
            raise ValueError(f"spacing length {len(spacing)} must match dims {ndim}")
        axes = [np.fft.fftfreq(n, d=sp) for n, sp in zip(shape, spacing)]
    else:
        axes = [np.fft.fftfreq(n) * n for n in shape]

    # Build radial coordinate grid
    grids = np.meshgrid(*axes, indexing="ij")
    K = np.sqrt(sum(g**2 for g in grids)).ravel()

    # Assign bin IDs
    bid = np.digitize(K, edges) - 1
    bid = np.clip(bid, 0, len(edges) - 2).astype(np.int32, copy=False)

    # Exclude DC (K ≈ 0)
    bid[K < 1e-10] = -1
    return bid


def reduce_power(F: np.ndarray, bin_id: np.ndarray):
    """Sum per-bin power Σ|F|² and counts (DC excluded)."""
    valid = bin_id >= 0
    nbins = int(bin_id[valid].max()) + 1 if valid.any() else 0
    a = np.abs(F).ravel()[valid]
    S2 = np.bincount(bin_id[valid], weights=a * a, minlength=nbins)
    N = np.bincount(bin_id[valid], minlength=nbins)
    return S2, N


def reduce_abs(F: np.ndarray, bin_id: np.ndarray):
    """Sum per-bin magnitude Σ|F| and counts (DC excluded)."""
    valid = bin_id >= 0
    nbins = int(bin_id[valid].max()) + 1 if valid.any() else 0
    a = np.abs(F).ravel()[valid]
    S1 = np.bincount(bin_id[valid], weights=a, minlength=nbins)
    N = np.bincount(bin_id[valid], minlength=nbins)
    return S1, N


def reduce_cross(
    FX: np.ndarray,
    FY: np.ndarray,
    bin_id: np.ndarray,
    numerator: str = "real",
):
    """
    Sum per-bin cross-spectrum (DC excluded).

    numerator='real': Σ Re{X·conj(Y)} (classic FRC/FSC, can be negative).
    numerator='mag': Σ |X|·|Y|.
    """
    valid = bin_id >= 0
    nbins = int(bin_id[valid].max()) + 1 if valid.any() else 0
    X = FX.ravel()[valid]
    Y = FY.ravel()[valid]
    Sxy_re = np.bincount(
        bin_id[valid], weights=X.real * Y.real + X.imag * Y.imag, minlength=nbins
    )
    if numerator == "real":
        return Sxy_re, None
    Sxy_mag = np.bincount(
        bin_id[valid],
        weights=np.hypot(X.real, X.imag) * np.hypot(Y.real, Y.imag),
        minlength=nbins,
    )
    return Sxy_re, Sxy_mag


def frc_from_sums(
    Sx2: np.ndarray,
    Sy2: np.ndarray,
    Sxy: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    """Compute FRC/FSC curve from per-bin sums: Sxy / sqrt(Sx2·Sy2)."""
    denom = np.sqrt(np.maximum(Sx2, 0.0) * np.maximum(Sy2, 0.0)) + eps
    return np.clip(Sxy / denom, -1.0, 1.0)
