"""Radial binning utilities for histogram-based FRC/FSC."""

from functools import lru_cache
from collections.abc import Sequence

import numpy as np

from cubic.cuda import get_array_module


def _kmax_index(shape: tuple[int, ...]) -> float:
    """Compute minimum Nyquist frequency in index units (unshifted FFT)."""
    return float(min(n // 2 for n in shape))


def _kmax_phys(shape: tuple[int, ...], spacing: Sequence[float]) -> float:
    """Compute minimum Nyquist frequency in physical units."""
    return float(min((n // 2) / (n * float(sp)) for n, sp in zip(shape, spacing)))


def _kmax_phys_max(shape: tuple[int, ...], spacing: Sequence[float]) -> float:
    """Compute maximum Nyquist frequency in physical units.

    Used for sectioned FSC where XY-dominant sectors need to extend
    to the XY Nyquist, not the minimum (typically Z) Nyquist.
    """
    return float(max((n // 2) / (n * float(sp)) for n, sp in zip(shape, spacing)))


@lru_cache(maxsize=256)
def _radial_edges_cached(
    shape: tuple[int, ...],
    bin_delta: float,
    spacing_key: tuple[float, ...] | None,
    use_max_nyquist: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute radial bin edges and centers (cached).

    Args:
        shape: Image shape (tuple of ints)
        bin_delta: Bin width in index bins
        spacing_key: Physical spacing tuple (hashable) or None
        use_max_nyquist: If True, use maximum Nyquist (for SFSC with
            anisotropic data). Default False uses minimum Nyquist.

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
        if use_max_nyquist:
            kmax = _kmax_phys_max(shape, spacing_key)
        else:
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
    use_max_nyquist: bool = False,
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
        use_max_nyquist: If True, use maximum Nyquist frequency across all axes
                         instead of minimum. Useful for sectioned FSC where
                         XY-dominant sectors need to extend to XY Nyquist.
                         Default False (use minimum Nyquist for 2D FRC/3D FSC).

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
        tuple(int(n) for n in shape), float(bin_delta), spacing_key, use_max_nyquist
    )


def radial_bin_id(
    shape: tuple[int, ...],
    edges: np.ndarray,
    spacing: Sequence[float] | None = None,
) -> np.ndarray:
    """
    Compute radial bin ID for each voxel in unshifted FFT grid.

    Device-aware: returns array on same device as edges (CPU/GPU).
    Uses broadcasting instead of meshgrid to reduce memory usage.

    Args:
        shape: Image shape (2D or 3D)
        edges: Radial bin edges from radial_edges()
        spacing: Physical spacing per axis (None for index units)

    Returns
    -------
        Flattened int32 array with bin_id ∈ [0, nbins-1].
        DC term (K≈0) is excluded with bin_id = -1.
    """
    # Get array module to create fftfreq on correct device
    xp = get_array_module(edges)

    ndim = len(shape)
    if ndim not in (2, 3):
        raise ValueError("Only 2D and 3D images are supported")

    # Check if spacing is effectively "no scaling" (all 1.0)
    if spacing is not None:
        if len(spacing) != ndim:
            raise ValueError(f"spacing length {len(spacing)} must match dims {ndim}")
        if all(abs(sp - 1.0) < 1e-10 for sp in spacing):
            spacing = None  # Treat spacing=1.0 as no spacing

    if spacing is not None:
        # xp.fft.fftfreq needed to create arrays on correct device
        axes = [
            xp.fft.fftfreq(n, d=sp).astype(np.float32) for n, sp in zip(shape, spacing)
        ]
    else:
        # xp.fft.fftfreq needed to create arrays on correct device
        axes = [xp.fft.fftfreq(n).astype(np.float32) * n for n in shape]

    # Build K with broadcasting
    if ndim == 2:
        k0 = axes[0][:, None]
        k1 = axes[1][None, :]
        K = np.sqrt(k0 * k0 + k1 * k1).ravel()
    else:  # ndim == 3
        k0 = axes[0][:, None, None]
        k1 = axes[1][None, :, None]
        k2 = axes[2][None, None, :]
        K = np.sqrt(k0 * k0 + k1 * k1 + k2 * k2).ravel()

    # Bin on same device
    bid = np.digitize(K, edges) - 1
    nbins = int(edges.size) - 1

    # Clip overflow bins to last bin (match mask backend behavior)
    bid = np.clip(bid, 0, nbins - 1).astype(np.int32, copy=False)

    # Exclude DC robustly using dtype-specific threshold
    tiny = np.finfo(K.dtype).tiny if hasattr(K, "dtype") else 1e-12
    bid[K < tiny] = -1

    return bid


def radial_k_grid(
    shape: tuple[int, ...],
    spacing: Sequence[float] | None = None,
) -> tuple[np.ndarray, float]:
    """
    Compute radial frequency magnitude for each point in unshifted FFT grid.

    This is the shared infrastructure for computing physical frequency coordinates,
    used by both FRC/FSC (via radial_bin_id) and DCR.

    Parameters
    ----------
    shape : tuple
        Image shape (2D or 3D)
    spacing : sequence of float, optional
        Physical spacing per axis. If None, uses index units.

    Returns
    -------
    k_radius : ndarray
        Frequency magnitude at each grid point (same shape as FFT output)
    k_max : float
        Maximum frequency (Nyquist limit)
    """
    ndim = len(shape)
    if ndim not in (2, 3):
        raise ValueError("Only 2D and 3D images are supported")

    if spacing is not None:
        if len(spacing) != ndim:
            raise ValueError(f"spacing length {len(spacing)} must match dims {ndim}")
        # Physical frequency coordinates: fftfreq(n, d=sp) gives cycles per unit
        axes = [
            np.fft.fftfreq(n, d=float(sp)).astype(np.float32)
            for n, sp in zip(shape, spacing)
        ]
        k_max = _kmax_phys(shape, spacing)
    else:
        # Index frequency coordinates
        axes = [np.fft.fftfreq(n).astype(np.float32) for n in shape]
        k_max = 0.5  # Nyquist in index units

    # Build k_radius using broadcasting for efficiency
    if ndim == 2:
        k0 = axes[0][:, None]
        k1 = axes[1][None, :]
        k_radius = np.sqrt(k0 * k0 + k1 * k1)
    else:  # ndim == 3
        k0 = axes[0][:, None, None]
        k1 = axes[1][None, :, None]
        k2 = axes[2][None, None, :]
        k_radius = np.sqrt(k0 * k0 + k1 * k1 + k2 * k2)

    return k_radius, float(k_max)


def reduce_power(F: np.ndarray, bin_id: np.ndarray):
    """
    Sum per-bin power Σ|F|² and counts (DC excluded).

    Device-aware: preserves device of input arrays.
    """
    valid = bin_id >= 0
    nbins = int(bin_id[valid].max()) + 1 if valid.any() else 0
    # np.abs, np.bincount all preserve device
    a = np.abs(F).ravel()[valid]
    S2 = np.bincount(bin_id[valid], weights=a * a, minlength=nbins)
    N = np.bincount(bin_id[valid], minlength=nbins)
    return S2, N


def reduce_abs(F: np.ndarray, bin_id: np.ndarray):
    """
    Sum per-bin magnitude Σ|F| and counts (DC excluded).

    Device-aware: preserves device of input arrays.
    """
    valid = bin_id >= 0
    nbins = int(bin_id[valid].max()) + 1 if valid.any() else 0
    # np.abs, np.bincount all preserve device
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

    Device-aware: preserves device of input arrays.

    numerator='real': Σ Re{X·conj(Y)} (classic FRC/FSC, can be negative).
    numerator='mag': Σ |X|·|Y|.
    """
    valid = bin_id >= 0
    nbins = int(bin_id[valid].max()) + 1 if valid.any() else 0
    X = FX.ravel()[valid]
    Y = FY.ravel()[valid]
    # np.bincount, np.hypot all preserve device
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
    """
    Compute FRC/FSC curve from per-bin sums: Sxy / sqrt(Sx2·Sy2).

    Device-aware: all operations preserve device naturally.
    """
    # np.sqrt, np.maximum, np.clip all preserve device
    denom = np.sqrt(np.maximum(Sx2, 0.0) * np.maximum(Sy2, 0.0)) + eps
    return np.clip(Sxy / denom, -1.0, 1.0)


# --- Angular sectioning for 3D FSC ---


def sectioned_bin_id(
    shape: tuple[int, int, int],
    radial_edges: np.ndarray,
    angle_edges: np.ndarray,
    spacing: Sequence[float] | None = None,
    exclude_axis_angle: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute combined radial+angular bin IDs for 3D sectioned FSC.

    Bins are indexed as: combined_id = angle_id * n_radial_bins + radial_id

    Parameters
    ----------
    shape : tuple
        3D image shape (Z, Y, X)
    radial_edges : ndarray
        Radial frequency bin edges
    angle_edges : ndarray
        Angular bin edges in degrees (0° = Z axis, 90° = XY plane)
    spacing : sequence of float, optional
        Physical spacing per axis [z, y, x]. If None, uses index units.
    exclude_axis_angle : float, optional
        Exclude frequencies within this angle (in degrees) from the Z axis.
        This follows Koho et al. 2019 to avoid piezo/interpolation artifacts
        near the optical axis. Default: 0.0 (no exclusion).

    Returns
    -------
    radial_id : ndarray
        Flattened radial bin IDs
    angle_id : ndarray
        Flattened angular bin IDs (angle from Z axis)
    """
    xp = get_array_module(radial_edges)

    if len(shape) != 3:
        raise ValueError("sectioned_bin_id requires 3D shape")

    # Check if spacing is effectively "no scaling" (all 1.0)
    if spacing is not None:
        if len(spacing) != 3:
            raise ValueError("spacing must have 3 elements for 3D")
        if all(abs(sp - 1.0) < 1e-10 for sp in spacing):
            spacing = None

    if spacing is not None:
        axes = [
            xp.fft.fftfreq(n, d=sp).astype(np.float32) for n, sp in zip(shape, spacing)
        ]
    else:
        axes = [xp.fft.fftfreq(n).astype(np.float32) * n for n in shape]

    # Build frequency grids with broadcasting
    kz = axes[0][:, None, None]
    ky = axes[1][None, :, None]
    kx = axes[2][None, None, :]

    # Radial frequency magnitude
    k_xy = np.sqrt(ky * ky + kx * kx)
    k_radius = np.sqrt(kz * kz + k_xy * k_xy).ravel()

    # Angle from Z axis (0° = along Z, 90° = in XY plane)
    # arctan2(k_xy, |kz|) gives angle from Z axis
    theta = np.degrees(np.arctan2(k_xy, np.abs(kz))).ravel()

    # Radial binning
    n_radial = int(radial_edges.size) - 1
    radial_id = np.digitize(k_radius, radial_edges) - 1
    radial_id = np.clip(radial_id, 0, n_radial - 1).astype(np.int32)

    # Angular binning
    n_angle = int(angle_edges.size) - 1
    angle_id = np.digitize(theta, angle_edges) - 1
    angle_id = np.clip(angle_id, 0, n_angle - 1).astype(np.int32)

    # Exclude DC
    tiny = np.finfo(k_radius.dtype).tiny if hasattr(k_radius, "dtype") else 1e-12
    radial_id[k_radius < tiny] = -1
    angle_id[k_radius < tiny] = -1

    # Exclude frequencies near Z axis (theta < exclude_axis_angle)
    # Following Koho et al. 2019 to avoid piezo/interpolation artifacts
    if exclude_axis_angle > 0:
        axis_mask = theta < exclude_axis_angle
        radial_id[axis_mask] = -1
        angle_id[axis_mask] = -1

    return radial_id, angle_id


def reduce_power_sectioned(
    F: np.ndarray,
    radial_id: np.ndarray,
    angle_id: np.ndarray,
    n_radial: int,
    n_angle: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sum per-bin power for sectioned FSC: Σ|F|² per (angle, radius) bin.

    Returns
    -------
    S2 : ndarray, shape (n_angle, n_radial)
        Power sum per bin
    N : ndarray, shape (n_angle, n_radial)
        Count per bin
    """
    valid = (radial_id >= 0) & (angle_id >= 0)
    rid = radial_id[valid]
    aid = angle_id[valid]
    a = np.abs(F).ravel()[valid]

    # Combined bin index
    combined_id = aid * n_radial + rid
    n_combined = n_angle * n_radial

    S2_flat = np.bincount(combined_id, weights=a * a, minlength=n_combined)
    N_flat = np.bincount(combined_id, minlength=n_combined)

    return S2_flat.reshape(n_angle, n_radial), N_flat.reshape(n_angle, n_radial)


def reduce_cross_sectioned(
    FX: np.ndarray,
    FY: np.ndarray,
    radial_id: np.ndarray,
    angle_id: np.ndarray,
    n_radial: int,
    n_angle: int,
) -> np.ndarray:
    """
    Sum per-bin cross-spectrum for sectioned FSC: Σ Re{X·conj(Y)}.

    Returns
    -------
    Sxy : ndarray, shape (n_angle, n_radial)
        Cross-spectrum sum per bin
    """
    valid = (radial_id >= 0) & (angle_id >= 0)
    rid = radial_id[valid]
    aid = angle_id[valid]
    X = FX.ravel()[valid]
    Y = FY.ravel()[valid]

    combined_id = aid * n_radial + rid
    n_combined = n_angle * n_radial

    Sxy_flat = np.bincount(
        combined_id, weights=X.real * Y.real + X.imag * Y.imag, minlength=n_combined
    )

    return Sxy_flat.reshape(n_angle, n_radial)
