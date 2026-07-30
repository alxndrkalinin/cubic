"""Radial binning utilities for histogram-based FRC/FSC."""

from functools import lru_cache
from collections.abc import Sequence

import numpy as np

from cubic.cuda import get_array_module


def _normalize_spacing(
    spacing: float | Sequence[float] | None,
    ndim: int,
) -> list[float] | None:
    """Normalize spacing to a list of floats or None.

    Consolidates the repeated pattern of converting scalar/sequence/None spacing
    into a uniform list used across FRC, FSC, and DCR functions.
    """
    if spacing is None:
        return None
    if isinstance(spacing, (int, float)):
        return [float(spacing)] * ndim
    return [float(s) for s in spacing]


def _spacing_or_unit(spacing: Sequence[float] | None, ndim: int) -> tuple[float, ...]:
    """Return *spacing* as a tuple, treating ``None`` as one unit per axis.

    ``None`` means "index units", which is the same frequency grid as a spacing
    of 1: cycles per pixel. Keeping them as one code path matters for non-square
    input. Scaling each axis by its own length instead (the old index-unit
    branch, ``fftfreq(n) * n``) makes a constant-radius ring an *ellipse* in
    physical frequency once the axes differ in length, so a single bin averaged
    unlike frequencies together, and ``None`` and ``1.0`` disagreed: a (64, 128)
    array gave 32 bins for ``None`` against 64 for ``1.0``, with different
    per-voxel bin assignments. Square input was unaffected, which is why this
    survived.
    """
    if spacing is None:
        return (1.0,) * ndim
    if len(spacing) != ndim:
        raise ValueError(f"spacing length {len(spacing)} must match dims {ndim}")
    return tuple(float(s) for s in spacing)


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
    spacing_key: tuple[float, ...],
    use_max_nyquist: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute radial bin edges and centers (cached).

    Args:
        shape: Image shape (tuple of ints)
        bin_delta: Bin width in index bins
        spacing_key: Physical spacing tuple (hashable); all ones for index units
        use_max_nyquist: If True, use maximum Nyquist (for SFSC with
            anisotropic data). Default False uses minimum Nyquist.

    Returns
    -------
        edges: (M+1,) radial bin edges from 0 to kmax
        radii: (M,) radial bin centers (midpoints)
    """
    if bin_delta <= 0:
        raise ValueError("bin_delta must be > 0")

    # One index bin in physical units along axis i: Δk_i = 1/(n_i·spacing_i)
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
        spacing: Physical spacing per axis. None means one unit per axis, i.e.
                 cycles per pixel (see :func:`_spacing_or_unit`).
        use_max_nyquist: If True, use maximum Nyquist frequency across all axes
                         instead of minimum. Useful for sectioned FSC where
                         XY-dominant sectors need to extend to XY Nyquist.
                         Default False (use minimum Nyquist for 2D FRC/3D FSC).

    Returns
    -------
        edges: (M+1,) radial bin edges from 0 to kmax
        radii: (M,) radial bin centers (midpoints)
    """
    # Convert to hashable types and call cached implementation
    return _radial_edges_cached(
        tuple(int(n) for n in shape),
        float(bin_delta),
        _spacing_or_unit(spacing, len(shape)),
        use_max_nyquist,
    )


def _binning_edges(edges: np.ndarray) -> np.ndarray:
    """Shrink bin edges by a few float32 epsilons for use with ``np.digitize``.

    The frequency grid is built in float32 while the edges are float64, so a
    frequency that sits exactly on a bin edge — the innermost non-DC ring always
    does, at ``k = 1 / (n * spacing)`` — can round just below it and bin one
    step too low. That shifts the whole low-frequency end of the curve relative
    to the mask backend, whose rings are half-open ``[start, stop)`` in float64.
    Nudging the edges down puts boundary frequencies in the upper bin, as
    ``[start, stop)`` requires.
    """
    return edges * (1.0 - 4.0 * float(np.finfo(np.float32).eps))


def radial_bin_id(
    shape: tuple[int, ...],
    edges: np.ndarray,
    spacing: Sequence[float] | None = None,
    exclude_overflow: bool = False,
) -> np.ndarray:
    """
    Compute radial bin ID for each voxel in unshifted FFT grid.

    Device-aware: returns array on same device as edges (CPU/GPU).
    Uses broadcasting instead of meshgrid to reduce memory usage.

    Args:
        shape: Image shape (2D or 3D)
        edges: Radial bin edges from radial_edges()
        spacing: Physical spacing per axis (None for index units).
                 Must use the same units as ``edges`` — pass the same value
                 that was given to :func:`radial_edges`.
        exclude_overflow: If True, drop frequencies above ``edges[-1]``
                 (bin_id = -1) instead of folding them into the last bin.
                 See the note below. Default False (miplib behaviour).

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

    # xp.fft.fftfreq needed to create arrays on correct device
    axes = [
        xp.fft.fftfreq(n, d=sp).astype(np.float32)
        for n, sp in zip(shape, _spacing_or_unit(spacing, ndim))
    ]

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
    bid = np.digitize(K, _binning_edges(edges)) - 1
    nbins = int(edges.size) - 1

    # Frequencies between kmax and the FFT corners (|K| > edges[-1]) have no
    # bin of their own. By default they are folded into the last bin to match
    # the mask backend, so the final correlation value is not a true ring/shell
    # average — in 3D roughly half of all voxels land there. exclude_overflow
    # drops them instead, at the cost of a noisier last bin.
    overflow = bid >= nbins if exclude_overflow else None
    bid = np.clip(bid, 0, nbins - 1).astype(np.int32, copy=False)

    # Exclude DC robustly using dtype-specific threshold
    tiny = np.finfo(K.dtype).tiny
    bid[K < tiny] = -1
    if overflow is not None:
        bid[overflow] = -1

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
        Physical spacing per axis. None means one unit per axis, i.e. cycles per
        pixel (see :func:`_spacing_or_unit`).

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

    spacing = _spacing_or_unit(spacing, ndim)
    # Physical frequency coordinates: fftfreq(n, d=sp) gives cycles per unit.
    # k_max is derived, not hardcoded to 0.5: an odd axis tops out at
    # (n // 2) / n, which is below Nyquist.
    axes = [
        np.fft.fftfreq(n, d=float(sp)).astype(np.float32)
        for n, sp in zip(shape, spacing)
    ]
    k_max = _kmax_phys(shape, spacing)

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


def reduce_power(F: np.ndarray, bin_id: np.ndarray, nbins: int | None = None):
    """
    Sum per-bin power Σ|F|² and counts (DC excluded).

    Device-aware: preserves device of input arrays.

    Args:
        nbins: Number of bins, i.e. ``edges.size - 1``. Pass it explicitly:
            deriving it from the data (the ``None`` fallback) silently returns
            a curve shorter than ``radii`` whenever a trailing bin is empty,
            which callers then pair positionally with the bin centers.
    """
    valid = bin_id >= 0
    bins = bin_id[valid]
    if nbins is None:
        nbins = int(bins.max()) + 1 if valid.any() else 0
    # np.abs, np.bincount all preserve device
    a = np.abs(F).ravel()[valid]
    S2 = np.bincount(bins, weights=a * a, minlength=nbins)
    N = np.bincount(bins, minlength=nbins)
    return S2, N


def reduce_frc_sums(
    FX: np.ndarray,
    FY: np.ndarray,
    bin_id: np.ndarray,
    nbins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sum the per-bin power and cross-spectrum needed for an FRC/FSC curve.

    Returns ``(Sx2, Sy2, Sxy, N)`` with ``Sxy = Σ Re{X·conj(Y)}`` (signed, as
    in miplib). All four reductions share a single boolean-index pass over
    ``bin_id``, avoiding the repeated ``bin_id[valid]`` gathers and the
    duplicate count that separate power/cross reducers would each redo.

    Device-aware: preserves device of input arrays.
    """
    valid = bin_id >= 0
    bins = bin_id[valid]
    X = FX.ravel()[valid]
    Y = FY.ravel()[valid]
    Sx2 = np.bincount(bins, weights=X.real * X.real + X.imag * X.imag, minlength=nbins)
    Sy2 = np.bincount(bins, weights=Y.real * Y.real + Y.imag * Y.imag, minlength=nbins)
    Sxy = np.bincount(bins, weights=X.real * Y.real + X.imag * Y.imag, minlength=nbins)
    N = np.bincount(bins, minlength=nbins)
    return Sx2, Sy2, Sxy, N


def frc_from_sums(
    Sx2: np.ndarray,
    Sy2: np.ndarray,
    Sxy: np.ndarray,
    *,
    signed: bool = True,
) -> np.ndarray:
    """
    Compute FRC/FSC curve from per-bin sums: Sxy / sqrt(Sx2·Sy2).

    The quotient is evaluated in the log domain because ``Sx2 * Sy2`` overflows
    float32 for realistic image sizes. Bins with no power in either input
    return 0.

    Device-aware: all operations preserve device naturally.

    Args:
        signed: Keep the sign of ``Sxy`` (default, matches miplib). The
            cross-spectrum sum is negative wherever the two inputs are
            anticorrelated — the high-frequency noise floor of a binomial
            counts split, for instance. Passing False returns the magnitude,
            which biases that noise floor positive.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        eps = np.finfo(np.float32).tiny
        mag = np.exp(
            np.log(np.clip(np.abs(Sxy), eps, None))
            - 0.5 * (np.log(np.clip(Sx2, eps, None)) + np.log(np.clip(Sy2, eps, None)))
        )
        # |Sxy| <= sqrt(Sx2·Sy2) by Cauchy-Schwarz, so values above 1 are
        # float error; empty bins (all sums 0) collapse to exp(0) = 1.
        mag = np.nan_to_num(np.clip(mag, 0.0, 1.0))
    mag = np.where((Sx2 > 0) & (Sy2 > 0), mag, 0.0)
    return np.where(Sxy < 0, -mag, mag) if signed else mag


def _validate_angle_delta(angle_delta: int, min_sectors: int = 1) -> int:
    """Validate ``angle_delta`` and return the number of polar sectors.

    ``n_angle = 90 // angle_delta`` truncates silently for non-divisors of 90:
    an ``angle_delta`` of 20 folds the leftover 80-90 degree wedge into the last
    sector, and any value above 90 yields zero sectors (empty results and NaN
    resolutions with no error).
    """
    if angle_delta <= 0 or 90 % angle_delta != 0:
        raise ValueError(
            f"angle_delta must be a positive divisor of 90 degrees, got {angle_delta}"
        )
    n_angle = 90 // angle_delta
    if n_angle < min_sectors:
        raise ValueError(
            f"angle_delta={angle_delta} gives {n_angle} polar sector(s), but "
            f"{min_sectors} are required to separate XY from Z"
        )
    return n_angle


def _sector_edges(angle_delta: int, min_sectors: int = 1) -> tuple[int, np.ndarray]:
    """Validate ``angle_delta`` and build the polar sector edges it implies.

    Returns the sector count together with the ``(n_angle + 1,)`` edge array
    spanning 0-90 degrees. Both sectioned backends (FSC and DCR) derive their
    geometry here so they cannot drift on the edges while agreeing on the count.
    """
    n_angle = _validate_angle_delta(angle_delta, min_sectors=min_sectors)
    edges = np.array(
        [float(i * angle_delta) for i in range(n_angle + 1)], dtype=np.float32
    )
    return n_angle, edges


# --- Angular sectioning for 3D FSC ---


def sectioned_bin_id(
    shape: tuple[int, int, int],
    radial_edges: np.ndarray,
    angle_edges: np.ndarray,
    spacing: Sequence[float] | None = None,
    exclude_axis_angle: float = 0.0,
    exclude_overflow: bool = False,
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
        Angular bin edges in degrees (polar angle from Z axis, 0-90°).
        - 0° = Z-dominated frequencies (|kz| >> k_xy) → Z resolution
        - 90° = XY-dominated frequencies (k_xy >> |kz|) → XY resolution
        where k_xy = sqrt(kx² + ky²). This captures all lateral frequencies.
    spacing : sequence of float, optional
        Physical spacing per axis [z, y, x]. None means one unit per axis, i.e.
        cycles per pixel (see :func:`_spacing_or_unit`).
    exclude_axis_angle : float, optional
        Exclude frequencies within this angle (in degrees) from the Z axis.
        This follows Koho et al. 2019 to avoid piezo/interpolation artifacts
        near the optical axis. Default: 0.0 (no exclusion).
    exclude_overflow : bool, optional
        If True, drop frequencies above ``radial_edges[-1]`` instead of folding
        them into the last shell. See :func:`radial_bin_id`. Default: False.

    Returns
    -------
    radial_id : ndarray
        Flattened radial bin IDs
    angle_id : ndarray
        Flattened angular bin IDs (polar angle from Z axis, 0-90°)
    """
    xp = get_array_module(radial_edges)

    if len(shape) != 3:
        raise ValueError("sectioned_bin_id requires 3D shape")

    axes = [
        xp.fft.fftfreq(n, d=sp).astype(np.float32)
        for n, sp in zip(shape, _spacing_or_unit(spacing, 3))
    ]

    # Build frequency grids with broadcasting
    kz = axes[0][:, None, None]
    ky = axes[1][None, :, None]
    kx = axes[2][None, None, :]

    # Radial frequency magnitude
    k_xy = np.sqrt(ky * ky + kx * kx)
    k_radius = np.sqrt(kz * kz + k_xy * k_xy).ravel()

    # Polar angle from Z axis:
    # - theta ≈ 0° when |kz| >> k_xy → Z-dominated → Z resolution
    # - theta ≈ 90° when k_xy >> |kz| → XY-dominated → XY resolution
    # Range: [0°, 90°] (always positive since we use |kz|)
    # arctan2 broadcasts k_xy (1,Y,X) against |kz| (Z,1,1) to (Z,Y,X) on its own.
    theta = np.degrees(np.arctan2(k_xy, np.abs(kz))).ravel()

    # Radial binning
    n_radial = int(radial_edges.size) - 1
    radial_id = np.digitize(k_radius, _binning_edges(radial_edges)) - 1
    # See radial_bin_id for what folding the overflow into the last shell means.
    overflow = radial_id >= n_radial if exclude_overflow else None
    radial_id = np.clip(radial_id, 0, n_radial - 1).astype(np.int32)

    # Angular binning
    n_angle = int(angle_edges.size) - 1
    angle_id = np.digitize(theta, angle_edges) - 1
    angle_id = np.clip(angle_id, 0, n_angle - 1).astype(np.int32)

    # Exclude DC
    tiny = np.finfo(k_radius.dtype).tiny
    radial_id[k_radius < tiny] = -1
    angle_id[k_radius < tiny] = -1

    if overflow is not None:
        radial_id[overflow] = -1
        angle_id[overflow] = -1

    # Exclude frequencies near Z axis (theta near 0°)
    # Following Koho et al. 2019 to avoid piezo/interpolation artifacts
    if exclude_axis_angle > 0:
        axis_mask = theta < exclude_axis_angle
        radial_id[axis_mask] = -1
        angle_id[axis_mask] = -1

    return radial_id, angle_id


def reduce_frc_sums_sectioned(
    FX: np.ndarray,
    FY: np.ndarray,
    radial_id: np.ndarray,
    angle_id: np.ndarray,
    n_radial: int,
    n_angle: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sum the per-(angle, radius) power and cross-spectrum for a sectioned FSC.

    The sectioned counterpart of :func:`reduce_frc_sums`: one masked pass over
    the flattened bin IDs feeds all four reductions.

    Returns
    -------
    Sx2, Sy2, Sxy, N : ndarray, each shape (n_angle, n_radial)
        Power sums, cross-spectrum sum ``Σ Re{X·conj(Y)}`` and voxel count.
    """
    valid = (radial_id >= 0) & (angle_id >= 0)
    combined_id = angle_id[valid] * n_radial + radial_id[valid]
    n_combined = n_angle * n_radial
    X = FX.ravel()[valid]
    Y = FY.ravel()[valid]

    def _bin(weights: np.ndarray | None) -> np.ndarray:
        return np.bincount(combined_id, weights=weights, minlength=n_combined).reshape(
            n_angle, n_radial
        )

    return (
        _bin(X.real * X.real + X.imag * X.imag),
        _bin(Y.real * Y.real + Y.imag * Y.imag),
        _bin(X.real * Y.real + X.imag * Y.imag),
        _bin(None),
    )
