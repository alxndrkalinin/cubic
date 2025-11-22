"""Decorrelation Analysis (DCR) for single-image resolution estimation."""

from collections.abc import Sequence

import numpy as np
from numpy.fft import fftn
from scipy.signal import find_peaks, savgol_filter

from cubic.cuda import asnumpy, get_array_module


def dcr_curve(
    vol: np.ndarray,
    iterator,
    *,
    spacing: Sequence[float] | None = None,
    cap_k: float | None = None,
    smooth: int = 5,
    drop_bins: int = 2,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute decorrelation curve and estimate resolution.

    Uses unshifted FFT convention consistent with FRC/FSC implementation.

    Parameters
    ----------
    vol : np.ndarray
        2D or 3D image (already windowed/padded if desired)
    iterator : FourierRingIterator or FourierShellIterator
        Iterator providing radial bins in unshifted FFT coordinates
    spacing : sequence of floats, optional
        Physical spacing per axis. If provided, resolution is in physical units.
    cap_k : float, optional
        Maximum frequency to consider (in index or physical units)
    smooth : int
        Savitzky-Golay smoothing window size (default: 5)
    drop_bins : int
        Number of low-frequency bins to ignore for peak finding (default: 2)

    Returns
    -------
    resolution : float
        Estimated resolution (in index units if spacing=None, physical units otherwise)
    K : np.ndarray
        Frequency values for each bin
    r : np.ndarray
        Decorrelation curve values

    Notes
    -----
    DCR curve is defined as:
        r(k) = Σ_{k'≤k} |F(k')| / √(E²_total × N(k))

    Resolution is estimated as the first local maximum of r(k).
    """
    vol = vol.astype(np.float32, copy=False)
    vol = vol - np.mean(vol)
    # Use unshifted FFT (consistent with FRC/FSC)
    F = fftn(vol)
    absF = np.abs(F)
    E2tot = np.sum(absF**2)

    # Accumulate Σ|F| and counts per shell
    # Iterator yields (ind_ring, idx) where idx is the bin index
    radii = iterator.radii
    S1 = np.zeros(len(radii), dtype=np.float64)
    N = np.zeros(len(radii), dtype=np.float64)
    K = radii.copy()

    for ind_ring, idx in iterator:
        if cap_k is not None and K[idx] > cap_k:
            break
        # Extract Fourier coefficients at this ring
        subset = absF[ind_ring]
        S1[idx] = np.sum(subset)
        N[idx] = len(subset)

    S1 = np.asarray(S1, dtype=np.float64)
    N = np.asarray(N, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    # Note: K values come from iterator.radii, which are already in physical units
    # if spacing was passed to the iterator. No manual conversion needed here.

    # DCR curve (cumulative decorrelation)
    cumN = np.cumsum(N)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.cumsum(S1) / np.sqrt(E2tot * cumN)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    # Find first local maximum beyond low-frequency bins
    if smooth and smooth > 1:
        win = smooth if (smooth % 2 == 1) else smooth + 1
        win = max(5, min(win, (len(r) // 2) * 2 + 1))
        r = savgol_filter(r, window_length=win, polyorder=2, mode="interp")

    start = max(1, int(drop_bins))
    peaks, _ = find_peaks(r, plateau_size=(1, None))
    peaks = peaks[peaks >= start]
    idx = int(peaks[0]) if peaks.size else int(np.argmax(r[start:]) + start)

    # Resolution is 1/frequency
    resolution = 1.0 / K[idx] if K[idx] > 0 else float("inf")

    return resolution, K[: len(r)], r


def dcr_curve_hist(
    vol: np.ndarray,
    bin_delta: int,
    *,
    spacing: Sequence[float] | None = None,
    cap_k: float | None = None,
    smooth: int = 5,
    drop_bins: int = 2,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Compute decorrelation curve using histogram backend (GPU-compatible).

    Parameters
    ----------
    vol : np.ndarray
        2D or 3D image
    bin_delta : int
        Bin width in frequency space
    spacing : sequence of floats, optional
        Physical spacing per axis. If provided, resolution is in physical units.
    cap_k : float, optional
        Maximum frequency to consider (in index or physical units)
    smooth : int
        Savitzky-Golay smoothing window size (default: 5)
    drop_bins : int
        Number of low-frequency bins to ignore for peak finding (default: 2)

    Returns
    -------
    resolution : float
        Estimated resolution (in index units if spacing=None, physical units otherwise)
    K : np.ndarray
        Frequency values for each bin
    r : np.ndarray
        Decorrelation curve values
    """
    from .radial import reduce_abs, radial_edges, radial_bin_id

    vol = vol.astype(np.float32, copy=False)
    vol = vol - np.mean(vol)

    # Compute unshifted FFT
    F = fftn(vol)
    xp = get_array_module(F)

    # Build radial bins
    shape = vol.shape
    edges, radii = radial_edges(shape, bin_delta, spacing=spacing)
    edges = xp.asarray(edges)  # Transfer edges to device

    # Compute bin IDs
    bin_id = radial_bin_id(shape, edges, spacing=spacing)

    # Sum |F| and counts per bin
    S1, N = reduce_abs(F, bin_id)

    # Total power (on device)
    absF = xp.abs(F)
    E2tot = xp.sum(absF**2)

    # Apply cap_k if specified
    if cap_k is not None:
        if spacing is None:
            valid = radii <= cap_k
        else:
            valid = radii <= cap_k
        S1 = S1[valid]
        N = N[valid]
        radii = radii[valid]

    # Transfer to CPU for cumsum and analysis
    S1 = asnumpy(S1).astype(np.float64)
    N = asnumpy(N).astype(np.float64)
    E2tot = float(asnumpy(E2tot))
    K = radii  # Already on CPU

    # DCR curve (cumulative decorrelation)
    cumN = np.cumsum(N)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.cumsum(S1) / np.sqrt(E2tot * cumN)
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)

    # Find first local maximum beyond low-frequency bins
    if smooth and smooth > 1:
        win = smooth if (smooth % 2 == 1) else smooth + 1
        win = max(5, min(win, (len(r) // 2) * 2 + 1))
        r = savgol_filter(r, window_length=win, polyorder=2, mode="interp")

    start = max(1, int(drop_bins))
    peaks, _ = find_peaks(r, plateau_size=(1, None))
    peaks = peaks[peaks >= start]
    idx = int(peaks[0]) if peaks.size else int(np.argmax(r[start:]) + start)

    # Resolution is 1/frequency
    resolution = 1.0 / K[idx] if K[idx] > 0 else float("inf")

    return resolution, K[: len(r)], r


def dcr_resolution(
    image: np.ndarray,
    *,
    bin_delta: int = 1,
    spacing: float | Sequence[float] | None = None,
    smooth: int = 5,
    drop_bins: int = 2,
    backend: str = "mask",
) -> float:
    """
    Calculate resolution using Decorrelation Analysis (DCR).

    DCR analyzes the decorrelation of Fourier components as a function
    of spatial frequency. Unlike FRC which compares two images, DCR
    analyzes a single image's frequency content.

    Parameters
    ----------
    image : np.ndarray
        2D or 3D input image
    bin_delta : int
        Bin width in frequency space (default: 1)
    spacing : float or sequence of floats
        Physical spacing per axis in nm/μm (default: 1.0).
        If float, same spacing is used for all axes.
    smooth : int
        Savitzky-Golay smoothing window size (default: 5)
    drop_bins : int
        Number of low-frequency bins to ignore for peak finding (default: 2)
    backend : str
        Backend to use: "mask" (iterator-based, CPU-only) or
        "hist" (histogram-based, GPU-compatible). Default: "mask"

    Returns
    -------
    resolution : float
        Estimated resolution in physical units (based on spacing)

    Notes
    -----
    DCR curve is defined as:
        r(k) = Σ_{k'≤k} |F(k')| / √(E²_total × N(k))

    where:
    - F(k) = Fourier transform of the image
    - N(k) = number of Fourier components up to frequency k
    - E²_total = total power in the image

    Resolution is estimated as 1/k_max where k_max is the first local
    maximum of r(k), indicating where high-frequency content begins
    to decorrelate.

    Comparison with FRC
    -------------------
    - **FRC**: Compares two images, measures correlation between them
    - **DCR**: Analyzes single image, measures decorrelation in frequency

    Use DCR when you have only one image and want to assess frequency
    content without checkerboard splitting or other image-pair generation.

    Backend Selection
    -----------------
    - **mask**: Float64 precision, iterator-based, CPU-only
    - **hist**: Float32 precision, histogram-based, GPU-compatible, faster

    Examples
    --------
    >>> from cubic.metrics.frc import dcr_resolution
    >>> import numpy as np
    >>> image = np.random.randn(64, 64)  # Test image
    >>> res = dcr_resolution(image, spacing=0.1)  # 0.1 μm/pixel
    >>> print(f"Resolution: {res:.2f} μm")
    """
    # Validate dimensions
    if image.ndim not in (2, 3):
        raise ValueError(f"DCR requires 2D or 3D images, got {image.ndim}D")

    # Normalize spacing to sequence
    if spacing is None:
        spacing_seq = None
    elif isinstance(spacing, (int, float)):
        spacing_seq = [float(spacing)] * image.ndim
    else:
        spacing_seq = list(spacing)

    # Select backend
    if backend == "hist":
        # Histogram backend (GPU-compatible)
        resolution, _, _ = dcr_curve_hist(
            image,
            bin_delta,
            spacing=spacing_seq,
            smooth=smooth,
            drop_bins=drop_bins,
        )
    elif backend == "mask":
        # Iterator backend (CPU-only)
        if image.ndim == 2:
            from .iterators import FourierRingIterator

            iterator = FourierRingIterator(image.shape, bin_delta, spacing=spacing_seq)
        else:  # image.ndim == 3
            from .iterators import FourierShellIterator

            iterator = FourierShellIterator(image.shape, bin_delta, spacing=spacing_seq)

        resolution, _, _ = dcr_curve(
            image,
            iterator,
            spacing=spacing_seq,
            smooth=smooth,
            drop_bins=drop_bins,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'mask' or 'hist'")

    return resolution


def calculate_dcr(
    image: np.ndarray,
    *,
    bin_delta: int = 1,
    spacing: float | Sequence[float] | None = None,
    smooth: int = 5,
    drop_bins: int = 2,
    backend: str = "mask",
):
    """
    Calculate DCR curve and resolution, returning full analysis data.

    Parameters
    ----------
    image : np.ndarray
        2D or 3D input image
    bin_delta : int
        Bin width in frequency space (default: 1)
    spacing : float or sequence of floats
        Physical spacing per axis in nm/μm (default: 1.0).
        If float, same spacing is used for all axes.
    smooth : int
        Savitzky-Golay smoothing window size (default: 5)
    drop_bins : int
        Number of low-frequency bins to ignore for peak finding (default: 2)
    backend : str
        Backend to use: "mask" (iterator-based, CPU-only) or
        "hist" (histogram-based, GPU-compatible). Default: "mask"

    Returns
    -------
    FourierCorrelationData
        Object containing DCR curve, frequency values, and resolution estimate

    Examples
    --------
    >>> from cubic.metrics.frc import calculate_dcr
    >>> import numpy as np
    >>> image = np.random.randn(64, 64)
    >>> dcr_data = calculate_dcr(image, spacing=0.1)
    >>> print(f"Resolution: {dcr_data.resolution['resolution']:.2f} μm")
    >>> # Plot DCR curve
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(dcr_data.correlation['frequency'], dcr_data.correlation['correlation'])
    >>> plt.xlabel('Spatial Frequency')
    >>> plt.ylabel('Decorrelation')
    """
    from .analysis import FourierCorrelationData

    # Validate dimensions
    if image.ndim not in (2, 3):
        raise ValueError(f"DCR requires 2D or 3D images, got {image.ndim}D")

    # Normalize spacing to sequence
    if spacing is None:
        spacing_seq = None
    elif isinstance(spacing, (int, float)):
        spacing_seq = [float(spacing)] * image.ndim
    else:
        spacing_seq = list(spacing)

    # Select backend and compute DCR curve
    if backend == "hist":
        resolution, K, r = dcr_curve_hist(
            image,
            bin_delta,
            spacing=spacing_seq,
            smooth=smooth,
            drop_bins=drop_bins,
        )
        # For histogram backend, get N from reduce_abs
        from .radial import reduce_abs, radial_edges, radial_bin_id

        F = fftn(image.astype(np.float32, copy=False) - np.mean(image))
        xp = get_array_module(F)
        edges, _ = radial_edges(image.shape, bin_delta, spacing=spacing_seq)
        edges = xp.asarray(edges)
        bin_id = radial_bin_id(image.shape, edges, spacing=spacing_seq)
        _, N = reduce_abs(F, bin_id)
        N = asnumpy(N).astype(np.float32)
    elif backend == "mask":
        # Iterator backend
        if image.ndim == 2:
            from .iterators import FourierRingIterator

            iterator = FourierRingIterator(image.shape, bin_delta, spacing=spacing_seq)
        else:  # image.ndim == 3
            from .iterators import FourierShellIterator

            iterator = FourierShellIterator(image.shape, bin_delta, spacing=spacing_seq)

        resolution, K, r = dcr_curve(
            image,
            iterator,
            spacing=spacing_seq,
            smooth=smooth,
            drop_bins=drop_bins,
        )
        # Get N by creating fresh iterator (iterators can only be used once)
        if image.ndim == 2:
            from .iterators import FourierRingIterator

            iterator_N = FourierRingIterator(image.shape, bin_delta, spacing=spacing_seq)
        else:  # image.ndim == 3
            from .iterators import FourierShellIterator

            iterator_N = FourierShellIterator(image.shape, bin_delta, spacing=spacing_seq)

        vol = image.astype(np.float32, copy=False) - np.mean(image)
        F = fftn(vol)
        absF = np.abs(F)
        radii = iterator_N.radii
        N = np.zeros(len(radii), dtype=np.float32)
        for ind_ring, idx in iterator_N:
            N[idx] = len(absF[ind_ring])
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'mask' or 'hist'")

    # Normalize frequency to [0, 1] range for compatibility with FRC
    freq_normalized = K / np.max(K) if np.max(K) > 0 else K

    # Create FourierCorrelationData object
    data = FourierCorrelationData()
    data.correlation["correlation"] = r.astype(np.float32)
    data.correlation["frequency"] = freq_normalized.astype(np.float32)
    data.correlation["points-x-bin"] = N
    data.resolution["resolution"] = float(resolution)
    data.resolution["spacing"] = np.mean(spacing_seq) if spacing_seq else 1.0

    return data


def _shells_and_radii_from_iterator(iterator):
    """
    Extract shells and radii from FRC/FSC iterator.

    Works with iterators using unshifted FFT coordinates (post-fix).
    Supports common iterator patterns:
      - iterable of shells + attribute .radii
      - iterator that yields (shell_mask, index)

    Parameters
    ----------
    iterator : FourierRingIterator or FourierShellIterator
        Iterator providing radial bins

    Returns
    -------
    shells : list
        List of shell masks (boolean arrays or index tuples)
    radii : np.ndarray
        Radial frequency values for each shell (bin midpoints)
    """
    # Case 1: iterator has .radii attribute and is iterable
    # This is the standard pattern for FourierRingIterator and FourierShellIterator
    if hasattr(iterator, "radii") and hasattr(iterator, "__iter__"):
        shells = []
        for item in iterator:
            # Iterator yields (mask_indices, shell_index)
            if isinstance(item, tuple) and len(item) >= 1:
                shells.append(item[0])  # Take mask indices
            else:
                shells.append(item)
        radii = np.asarray(iterator.radii)
        return shells, radii

    # Case 2: explicit .shells() method + radii getter
    if hasattr(iterator, "shells") and callable(iterator.shells):
        shells = list(iterator.shells())
        radii = getattr(iterator, "radii", None)
        if radii is None and hasattr(iterator, "get_radii"):
            radii = iterator.get_radii()
        return shells, np.asarray(radii)

    # Case 3: iterator yields (shell, radius) pairs
    shells, radii = [], []
    for item in iterator:
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            shells.append(item[0])
            # If item has explicit radius, use it; otherwise accumulate indices
            if len(item) > 1 and isinstance(item[1], (int, float)):
                radii.append(float(item[1]))
        else:
            raise TypeError(
                "Iterator must yield (shell, index) or have .radii attribute"
            )

    # If no radii were collected, try to get from iterator
    if not radii and hasattr(iterator, "radii"):
        radii = np.asarray(iterator.radii)
    else:
        radii = np.asarray(radii)

    return shells, radii
