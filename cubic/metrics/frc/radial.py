import numpy as np
from typing import Tuple, Optional
from math import floor


def radial_bins(
    shape: Tuple[int, ...],
    bin_delta: int = 1,
    nbins: Optional[int] = None,
    unit: str = "index",
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Build uniform radial bin edges and centers for a given grid shape.
    Frequencies in 'index' units (fftfreq(n)*n) so it's scale-agnostic.
    Returns (edges, radii, nbins).
    
    Args:
        shape: Image shape (2D or 3D)
        bin_delta: Bin width (step size between bins). Used to calculate nbins
                   if nbins is None. Default: 1.
        nbins: Optional explicit number of bins. If provided, overrides bin_delta.
        unit: Unit type (currently only "index" is supported)
    
    Returns:
        Tuple of (edges, radii, nbins)
    """
    axes = [np.fft.fftfreq(n) * n for n in shape]
    kmax = min(float(np.max(np.abs(ax))) for ax in axes)
    if nbins is None:
        # Calculate nbins from bin_delta to match iterator backend behavior
        # For 2D: nbins = floor(shape[0] / (2 * bin_delta))
        # For consistency with radial_bins default, use min(shape)
        nbins = int(floor(min(shape) / (2 * bin_delta)))
    edges = np.linspace(0.0, kmax, nbins + 1, dtype=np.float64)
    radii = 0.5 * (edges[:-1] + edges[1:])
    return edges, radii, nbins


def radial_bin_id(shape: Tuple[int, ...], edges: np.ndarray) -> np.ndarray:
    """
    Compute a single flat bin-id per voxel; no boolean masks.
    Returns int array of size np.prod(shape) with values in [0, nbins-1].
    """
    axes = [np.fft.fftfreq(n) * n for n in shape]
    grids = np.meshgrid(*axes, indexing="ij")
    K = np.sqrt(sum(g**2 for g in grids))
    bid = np.digitize(K.ravel(), edges) - 1
    nbins = len(edges) - 1
    return np.clip(bid, 0, nbins - 1).astype(np.int32, copy=False)


def reduce_power(F: np.ndarray, bin_id: np.ndarray, nbins: int):
    """
    Per-bin Σ|F|^2 and counts. Works for 2D/3D; NumPy API only (CuPy-aware).
    """
    a = np.abs(F).ravel()
    S2 = np.bincount(bin_id, weights=a * a, minlength=nbins)
    N = np.bincount(bin_id, minlength=nbins)
    return S2, N


def reduce_cross(
    FX: np.ndarray,
    FY: np.ndarray,
    bin_id: np.ndarray,
    nbins: int,
    numerator: str = "real",
):
    """
    Per-bin cross-spectrum sums. 'numerator' in {'real','mag'}.
    - 'real': sum Re{X * conj(Y)} (classic FRC numerator)
    - 'mag' : sum |X * conj(Y)| (optional)
    """
    X = FX.ravel()
    Y = FY.ravel()
    # Re{X conj Y} = Xr*Yr + Xi*Yi
    Sxy_re = np.bincount(
        bin_id, weights=(X.real * Y.real + X.imag * Y.imag), minlength=nbins
    )
    if numerator == "real":
        return Sxy_re, None
    # |X conj Y| = |X| * |Y|
    aX = np.hypot(X.real, X.imag)
    aY = np.hypot(Y.real, Y.imag)
    Sxy_mag = np.bincount(bin_id, weights=(aX * aY), minlength=nbins)
    return Sxy_re, Sxy_mag


def frc_from_sums(
    Sx2: np.ndarray,
    Sy2: np.ndarray,
    Sxy_re: np.ndarray,
    Sxy_mag: Optional[np.ndarray] = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Compute FRC curve from per-bin sums. Default: real-numerator FRC.
    """
    denom = np.sqrt(np.maximum(Sx2, 0) * np.maximum(Sy2, 0)) + eps
    num = Sxy_re if Sxy_mag is None else Sxy_mag
    frc = np.clip(num / denom, -1.0, 1.0)  # real-numerator can be negative
    return frc
