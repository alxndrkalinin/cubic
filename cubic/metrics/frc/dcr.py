import numpy as np
from numpy.fft import fftn, fftshift
from scipy.signal import find_peaks, savgol_filter


def dcr_curve(vol, iterator, cap_k=None, smooth=5, drop_bins=2):
    """vol: 2D/3D array (already windowed/padded); iterator: your FRC ring/shell iterator."""
    vol = vol.astype(np.float32, copy=False)
    vol = vol - np.mean(vol)
    F = fftshift(fftn(vol))
    absF = np.abs(F)
    E2tot = np.sum(absF**2)

    # --- pull shells + radii from your iterator (supports common patterns) ---
    shells, radii = _shells_and_radii_from_iterator(iterator)

    # --- accumulate Σ|F| and counts per shell ---
    S1, N, K = [], [], []
    for sh, kr in zip(shells, radii):
        if cap_k is not None and kr > cap_k:
            break
        if isinstance(sh, (tuple, list)) and len(sh) == 2:  # (idx, weights)
            idx, w = sh
            S1.append(np.sum(absF[idx] * w))
            N.append(np.sum(w))
        else:  # boolean mask or integer indices
            idx = sh
            S1.append(np.sum(absF[idx]))
            N.append(np.count_nonzero(idx))
        K.append(kr)

    S1 = np.asarray(S1, dtype=np.float64)
    N = np.asarray(N, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)

    # DCR curve (phase-normalized cumulative correlation)
    r = np.cumsum(S1) / np.sqrt(E2tot * np.cumsum(N))

    # first local max beyond low-k bins
    if smooth and smooth > 1:
        win = smooth if (smooth % 2 == 1) else smooth + 1
        win = max(5, min(win, (len(r) // 2) * 2 + 1))
        r = savgol_filter(r, window_length=win, polyorder=2, mode="interp")

    start = max(1, int(drop_bins))
    peaks, _ = find_peaks(r, plateau_size=(1, None))
    peaks = peaks[peaks >= start]
    idx = int(peaks[0]) if peaks.size else int(np.argmax(r[start:]) + start)

    return K[idx], K[: len(r)], r


def _shells_and_radii_from_iterator(iterator):
    """
    Accepts the same iterator you use for FRC.
    Supports these common shapes:
      - iterable of shells + attribute .radii
      - iterator.shells() + iterator.radii (or .get_radii())
      - iterator that yields (shell, radius)
    Returns (shells_list, radii_array).
    """
    # case 1: iterator itself is iterable and has .radii
    if hasattr(iterator, "radii") and hasattr(iterator, "__iter__"):
        shells = list(iter(iterator))
        radii = np.asarray(getattr(iterator, "radii"))
        return shells, radii

    # case 2: explicit .shells() + radii getter
    if hasattr(iterator, "shells") and callable(iterator.shells):
        shells = list(iterator.shells())
        radii = getattr(iterator, "radii", None)
        if radii is None and hasattr(iterator, "get_radii"):
            radii = iterator.get_radii()
        return shells, np.asarray(radii)

    # case 3: iterator yields (shell, radius)
    shells, radii = [], []
    for item in iterator:
        if isinstance(item, (tuple, list)) and len(item) == 2:
            shells.append(item[0])
            radii.append(float(item[1]))
        else:
            raise TypeError(
                "Iterator must yield (shell, radius) or expose shells()+radii"
            )
    return shells, np.asarray(radii)
