"""Radial binning utilities for histogram-based FRC/FSC."""

from functools import lru_cache
from collections.abc import Sequence

import numpy as np


def _kmax_index(shape: tuple[int, ...]) -> float:
    """Compute minimum Nyquist frequency in index units (unshifted FFT)."""
    return float(min(n // 2 for n in shape))


def _kmax_phys(shape: tuple[int, ...], spacing: Sequence[float]) -> float:
    """Compute minimum Nyquist frequency in physical units."""
    return float(min((n // 2) / (n * float(sp)) for n, sp in zip(shape, spacing)))


@lru_cache(maxsize=256)
def _radial_edges_cached(
    shape: tuple[int, ...],
    bin_delta: float,
    spacing_key: tuple[float, ...] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute radial bin edges and centers (cached).

    Args:
        shape: Image shape (tuple of ints)
        bin_delta: Bin width in index bins
        spacing_key: Physical spacing tuple (hashable) or None

    Returns
    -------
        edges: (M+1,) radial bin edges from 0 to kmax
        radii: (M,) radial bin centers (midpoints)
    """
    if bin_delta <= 0:
        raise ValueError("bin_delta must be > 0")

    if spacing_key is None:
        # Index units: step = bin_delta, kmax = floor(n/2)
        step = float(bin_delta)
        kmax = _kmax_index(shape)
    else:
        # Physical units: one index bin in physical units along axis i: Δk_i = 1/(n_i·spacing_i)
        dk_min = min(1.0 / (n * sp) for n, sp in zip(shape, spacing_key))
        step = float(bin_delta) * dk_min
        kmax = _kmax_phys(shape, spacing_key)

    # Build edges from 0 to kmax with step size
    nb = max(1, int(np.ceil(kmax / step)))
    edges = np.linspace(0.0, kmax, nb + 1, dtype=np.float64)
    radii = 0.5 * (edges[:-1] + edges[1:])

    # Make arrays read-only to avoid accidental mutation of cached data
    edges.setflags(write=False)
    radii.setflags(write=False)

    return edges, radii


def radial_edges(
    shape: tuple[int, ...],
    bin_delta: float = 1.0,
    spacing: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build uniform radial bin edges and centers for 2D/3D unshifted FFT grids.

    bin_delta is always in index bins. When spacing is provided, it converts
    to physical frequency units internally. Results are cached for performance.

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
    # Validate spacing dimensions
    if spacing is not None:
        ndim = len(shape)
        if len(spacing) != ndim:
            raise ValueError(f"spacing length {len(spacing)} must match dims {ndim}")
        spacing_key = tuple(float(s) for s in spacing)
    else:
        spacing_key = None

    # Convert to hashable types and call cached implementation
    return _radial_edges_cached(
        tuple(int(n) for n in shape), float(bin_delta), spacing_key
    )


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
