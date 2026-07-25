"""Implements 2D/3D Fourier Ring/Shell Correlation."""

import logging
import warnings
from typing import Literal
from collections.abc import Callable, Sequence

import numpy as np

from cubic.cuda import asnumpy, to_same_device
from cubic.image_utils import (
    crop_bl,
    crop_br,
    crop_tl,
    crop_tr,
    crop_center,
    binomial_split,
    hamming_window,
    pad_image_to_cube,
    rescale_isotropic,
    checkerboard_split,
    get_xy_block_coords,
    reverse_checkerboard_split,
)

from .radial import (
    _kmax_phys,
    radial_edges,
    _sector_edges,
    frc_from_sums,
    radial_bin_id,
    _kmax_phys_max,
    reduce_frc_sums,
    sectioned_bin_id,
    _normalize_spacing,
    _validate_angle_delta,
    reduce_frc_sums_sectioned,
)
from .analysis import (
    FourierCorrelationData,
    FourierCorrelationAnalysis,
    FourierCorrelationDataCollection,
)
from .iterators import FourierRingIterator, AxialExcludeSectionedFourierShellIterator

logger = logging.getLogger(__name__)

_BINOMIAL_SINGLE_REPEAT_MSG = (
    "Binomial split with n_repeats=1; consider n_repeats>=3 for stability."
)


def _make_repeat_rngs(
    rng: np.random.Generator | int | None, n_repeats: int
) -> list[np.random.Generator]:
    """Create deterministic independent RNG states for each repeat.

    Note: passing a ``np.random.Generator`` advances its state by drawing
    *n_repeats* seed integers.  Use an integer seed for fully reproducible
    results across multiple calls.
    """
    if isinstance(rng, int):
        ss = np.random.SeedSequence(rng)
        return [np.random.default_rng(s) for s in ss.spawn(n_repeats)]
    if isinstance(rng, np.random.Generator):
        seeds = rng.integers(0, 2**32 - 1, size=n_repeats, dtype=np.uint32)
        return [np.random.default_rng(int(s)) for s in seeds]
    # None → fresh unseeded sequence
    ss = np.random.SeedSequence()
    return [np.random.default_rng(s) for s in ss.spawn(n_repeats)]


def _normalization_spacing(max_freq: float) -> float:
    """Return the spacing implied by a normalized frequency axis.

    :class:`FourierCorrelationAnalysis` inverts a threshold crossing as
    ``2 * spacing / root``, which only equals the physical ``1 / (root * kmax)``
    when the spacing it is given matches the Nyquist frequency the axis was
    normalized by. Passing a raw axis spacing instead silently rescales every
    resolution by the anisotropy ratio.

    ``spacing=None`` needs no special case: the grid is then in cycles per pixel,
    where ``max_freq`` is 0.5 for even axes and the implied spacing is 1 pixel.
    Returning a hardcoded 1.0 was wrong for odd axes, which top out at
    ``(n // 2) / n`` rather than 0.5.
    """
    return 1.0 / (2.0 * max_freq)


def _frc_dataset(
    frc: np.ndarray, spatial_freq: np.ndarray, n_points: np.ndarray
) -> FourierCorrelationData:
    """Package an FRC/FSC curve, dropping bins that contain no frequencies.

    With ``bin_delta=1`` the innermost bin spans only the DC term, which is
    always excluded, so it holds no measurement at all. Keeping it would feed
    the curve fit a fabricated correlation (the quotient of empty sums), and
    that point decides whether the analyzer sees the curve as starting above
    the threshold.
    """
    valid = n_points > 0
    data_set = FourierCorrelationData()
    data_set.correlation["correlation"] = frc[valid]
    data_set.correlation["frequency"] = spatial_freq[valid]
    data_set.correlation["points-x-bin"] = n_points[valid]
    return data_set


def frc_checkerboard_split(
    image: np.ndarray,
    reverse: bool = False,
    disable_3d_sum: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Split image into two by checkerboard pattern."""
    if reverse:
        return reverse_checkerboard_split(image, disable_3d_sum=disable_3d_sum)
    return checkerboard_split(image, disable_3d_sum=disable_3d_sum)


def preprocess_images(
    image1: np.ndarray,
    image2: np.ndarray | None = None,
    *,
    zero_padding: bool = True,
    pad_mode: str = "constant",
    reverse_split: bool = False,
    disable_hamming: bool = False,
    disable_3d_sum: bool = False,
    split_type: Literal["checkerboard", "binomial"] = "checkerboard",
    counts_mode: Literal["counts", "poisson_thinning"] = "counts",
    gain: float = 1.0,
    offset: float = 0.0,
    readout_noise_rms: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess input images with all modifications (padding, windowing, splitting)."""
    single_image = image2 is None
    # Mean-subtract before padding so a large DC offset isn't multiplied by
    # the non-zero-mean Hamming taper (which would otherwise create a
    # low-frequency artifact ~ mean * (w(x) - mean(w)) that survives later
    # DC removal) and so the zero-padded ring stays at zero rather than at
    # -diluted_mean. Binomial counts splitting requires raw photon counts
    # (binomial_split clips negatives and rounds to integers), so its halves
    # are centered after splitting instead.
    pre_subtract = not disable_hamming and not (
        single_image and split_type == "binomial"
    )

    if pre_subtract:
        image1 = image1 - image1.mean()

    # Apply padding to first image
    if len(set(image1.shape)) > 1 and zero_padding:
        image1 = pad_image_to_cube(image1, mode=pad_mode)

    if image2 is None:
        if split_type == "binomial":
            image1, image2 = binomial_split(
                image1,
                p=0.5,
                counts_mode=counts_mode,
                gain=gain,
                offset=offset,
                readout_noise_rms=readout_noise_rms,
                rng=rng,
            )
        else:
            # Split single image using checkerboard pattern
            image1, image2 = frc_checkerboard_split(
                image1, reverse=reverse_split, disable_3d_sum=disable_3d_sum
            )
    else:
        # Apply padding to second image
        if pre_subtract:
            image2 = image2 - image2.mean()
        if len(set(image2.shape)) > 1 and zero_padding:
            image2 = pad_image_to_cube(image2, mode=pad_mode)

    # Apply Hamming windowing to both images independently. Binomial halves
    # were not centered above (the split needed raw counts), so center them
    # here.
    if not disable_hamming:
        if single_image and split_type == "binomial":
            image1 = hamming_window(image1 - image1.mean())
            image2 = hamming_window(image2 - image2.mean())
        else:
            image1 = hamming_window(image1)
            image2 = hamming_window(image2)

    return image1, image2


class FRC(object):
    """A class for calculating 2D Fourier ring correlation (unshifted FFT)."""

    def __init__(self, image1: np.ndarray, image2: np.ndarray, iterator):
        """Create new FRC executable object and perform FFT on input images.

        The frequency units come from *iterator*, which was built with the
        spacing, so no separate spacing argument is needed.
        """
        if image1.shape != image2.shape:
            raise ValueError("The image dimensions do not match")
        if image1.ndim != 2:
            raise ValueError("Fourier ring correlation requires 2D images.")

        self.iterator = iterator
        # Compute unshifted FFT (mean-subtracted, no fftshift)
        self.fft_image1 = np.fft.fftn(image1 - image1.mean())
        self.fft_image2 = np.fft.fftn(image2 - image2.mean())

    def execute(self):
        """Calculate the FRC."""
        radii = self.iterator.radii
        c1 = np.zeros(radii.shape, dtype=np.float32)
        c2 = np.zeros(radii.shape, dtype=np.float32)
        c3 = np.zeros(radii.shape, dtype=np.float32)
        points = np.zeros(radii.shape, dtype=np.float32)

        for ind_ring, idx in self.iterator:
            subset1 = self.fft_image1[ind_ring]
            subset2 = self.fft_image2[ind_ring]
            c1[idx] = np.sum(subset1 * np.conjugate(subset2)).real
            c2[idx] = np.sum(np.abs(subset1) ** 2)
            c3[idx] = np.sum(np.abs(subset2) ** 2)

            points[idx] = len(subset1)

        # Normalize the frequency axis by the Nyquist the bins were built from
        # (edges[-1]), not by shape[0] // 2 or the last bin centre: both make
        # identical data land on different axes, and on non-square input
        # shape[0] // 2 compresses the axis so every crossing sits at a
        # fraction of its true normalized frequency.
        spatial_freq = asnumpy(
            radii.astype(np.float32) / float(self.iterator.edges[-1])
        )
        frc = frc_from_sums(asnumpy(c2), asnumpy(c3), asnumpy(c1))
        return _frc_dataset(frc, spatial_freq, asnumpy(points))


def _calculate_frc_core(
    image1: np.ndarray,
    image2: np.ndarray,
    bin_delta: int,
    *,
    backend: str = "mask",
    spacing: Sequence[float] | None = None,
) -> FourierCorrelationDataCollection:
    """
    Core FRC calculation logic.

    Args:
        image1: First input image
        image2: Second input image
        bin_delta: Bin width (step size between bins). Controls binning resolution
                   for both backends.
        backend: "mask" (existing iterators) or "hist" (radial histogram)
        spacing: Physical spacing per axis. If None, uses index units.
    """
    assert image1.shape == image2.shape
    frc_data = FourierCorrelationDataCollection()

    if backend == "hist":
        # Histogram backend using radial binning
        # Compute unshifted FFT
        fft_image1 = np.fft.fftn(image1 - image1.mean())
        fft_image2 = np.fft.fftn(image2 - image2.mean())

        # Build radial bins
        shape = fft_image1.shape
        edges, radii = radial_edges(shape, bin_delta, spacing=spacing)
        nbins = len(radii)

        bin_id = radial_bin_id(
            shape, to_same_device(edges, fft_image1), spacing=spacing
        )
        Sx2, Sy2, Sxy, N = reduce_frc_sums(fft_image1, fft_image2, bin_id, nbins)
        frc = asnumpy(frc_from_sums(Sx2, Sy2, Sxy))

        # Normalize the frequency axis by the Nyquist the bins were built from
        # (see FRC.execute).
        spatial_freq = radii.astype(np.float32) / float(edges[-1])
        frc_data[0] = _frc_dataset(frc, spatial_freq, asnumpy(N.astype(np.float32)))
    else:
        # Default mask/iterator backend
        iterator = FourierRingIterator(image1.shape, bin_delta, spacing=spacing)
        frc_task = FRC(image1, image2, iterator)
        frc_data[0] = frc_task.execute()

    return frc_data


def _calibration_factor(freq_at_crossing: float) -> float:
    """Calculate calibration factor for one-image FRC/FSC.

    The checkerboard split creates a diagonal shift between subimage pairs that
    causes frequency compression in the FRC/FSC curve due to the Fourier shift
    theorem (Supplementary Note 1, Koho et al. 2019).

    This correction was empirically calibrated against two-image FRC at the 1/7
    threshold by imaging the same field of view at different pixel sizes
    (Supplementary Note 2, Supplementary Fig. 3, Koho et al. 2019).

    Parameters
    ----------
    freq_at_crossing : float
        Normalized frequency (0-1) at threshold crossing.

    Returns
    -------
    float
        Calibration factor. Divide raw resolution by this to get corrected value.
    """

    def calibration_func(x: float, a: float, b: float, c: float, d: float) -> float:
        return a * np.exp(c * (x - b)) + d

    # Parameters from miplib calibration (Koho et al. 2019)
    params = [0.95988146, 0.97979108, 13.90441896, 0.55146136]
    return calibration_func(freq_at_crossing, *params)


def _apply_cutoff_correction(result: FourierCorrelationData) -> None:
    """Apply cut-off correction for single image FRC."""
    point = result.resolution["resolution-point"][1]
    cut_off_correction = _calibration_factor(point)
    result.resolution["spacing"] /= cut_off_correction
    result.resolution["resolution"] /= cut_off_correction


def _calculate_frc_single_pass(
    image1: np.ndarray,
    image2: np.ndarray | None,
    *,
    bin_delta: int,
    spacing: list[float] | None,
    zero_padding: bool,
    pad_mode: str,
    disable_hamming: bool,
    average: bool,
    backend: str,
    split_type: Literal["checkerboard", "binomial"],
    counts_mode: Literal["counts", "poisson_thinning"],
    gain: float,
    offset: float,
    readout_noise_rms: float,
    rng: np.random.Generator | int | None,
) -> tuple[FourierCorrelationDataCollection, float]:
    """Run a single FRC pass (split + FFT + radial binning).

    For checkerboard splits this includes the forward+reverse averaging.
    For binomial splits this is a single random split.

    Returns the FRC data and the spacing implied by its frequency-axis
    normalization, which the analyzer must use to invert a crossing (see
    :func:`_normalization_spacing`). That is *not* ``spacing[0]``: the axis is
    normalized by the minimum Nyquist across axes, which belongs to the axis
    with the *coarsest* spacing, whichever one that is.
    """
    single_image = image2 is None
    use_checkerboard = split_type == "checkerboard"
    reverse = average and single_image and use_checkerboard
    # No copy needed: preprocess_images rebinds at every step, it never
    # mutates its input.
    original_image1 = image1 if reverse else None

    image1_proc, image2_proc = preprocess_images(
        image1,
        image2,
        zero_padding=zero_padding,
        pad_mode=pad_mode,
        disable_hamming=disable_hamming,
        split_type=split_type,
        counts_mode=counts_mode,
        gain=gain,
        offset=offset,
        readout_noise_rms=readout_noise_rms,
        rng=rng,
    )

    # Adjust spacing to match preprocessed image shape (padding may have changed dims)
    spacing_adj = spacing
    if spacing_adj is not None and len(spacing_adj) != image1_proc.ndim:
        if len(spacing_adj) < image1_proc.ndim:
            spacing_adj = list(spacing_adj) + [spacing_adj[0]] * (
                image1_proc.ndim - len(spacing_adj)
            )
        else:
            spacing_adj = list(spacing_adj[: image1_proc.ndim])

    frc_data = _calculate_frc_core(
        image1_proc,
        image2_proc,
        bin_delta,
        backend=backend,
        spacing=spacing_adj,
    )

    # Average with reverse checkerboard pattern (only for checkerboard single-image)
    if reverse:
        if original_image1 is None:
            raise RuntimeError("original_image1 must be set when reverse=True")
        image1_rev, image2_rev = preprocess_images(
            original_image1,
            None,
            reverse_split=True,
            zero_padding=zero_padding,
            pad_mode=pad_mode,
            disable_hamming=disable_hamming,
        )

        frc_data_rev = _calculate_frc_core(
            image1_rev,
            image2_rev,
            bin_delta,
            backend=backend,
            spacing=spacing_adj,
        )

        frc_data[0].correlation["correlation"] = (
            0.5 * frc_data[0].correlation["correlation"]
            + 0.5 * frc_data_rev[0].correlation["correlation"]
        )

    edges, _ = radial_edges(image1_proc.shape, bin_delta, spacing=spacing_adj)
    return frc_data, _normalization_spacing(float(edges[-1]))


def calculate_frc(
    image1: np.ndarray,
    image2: np.ndarray | None = None,
    *,
    bin_delta: int = 1,
    resolution_threshold: str = "fixed",
    threshold_value: float = 0.143,
    snr_value: float = 7.0,
    curve_fit_type: str = "spline",
    curve_fit_degree: int = 3,
    smoothing_factor: float = 0.05,
    disable_hamming: bool = False,
    average: bool = True,
    z_correction: float = 1.0,
    spacing: float | Sequence[float] | None = None,
    zero_padding: bool = True,
    pad_mode: str = "constant",
    backend: str = "mask",
    split_type: Literal["checkerboard", "binomial"] = "checkerboard",
    counts_mode: Literal["counts", "poisson_thinning"] = "counts",
    gain: float = 1.0,
    offset: float = 0.0,
    readout_noise_rms: float = 0.0,
    n_repeats: int = 1,
    rng: np.random.Generator | int | None = None,
) -> FourierCorrelationData:
    """
    Calculate a regular FRC with single or two image inputs.

    Args:
        bin_delta: Bin width (step size between bins). Controls binning resolution
                   for both backends. Default: 1.
        backend: "mask" (existing iterators) or "hist" (radial histogram)
        split_type: "checkerboard" (default, Koho et al. 2019) or "binomial"
                    (Rieger et al. 2024). Binomial splitting preserves image size
                    and needs no calibration correction.
        counts_mode: For binomial split: "counts" (raw photon counts) or
                     "poisson_thinning" (float/deconvolved fallback).
        gain: Camera gain (ADU/electron) for binomial counts mode.
        offset: Camera offset (ADU) for binomial counts mode.
        readout_noise_rms: Read-noise std in electrons for binomial counts mode.
        n_repeats: Number of independent binomial splits to average. Only used
                   when split_type="binomial" and single-image mode (image2 is
                   None). Ignored otherwise. Produces correlation-std and
                   resolution-std in the result. Default: 1.
        rng: Random number generator, seed, or None for binomial split.
    """
    single_image = image2 is None
    spacing = _normalize_spacing(spacing, image1.ndim)

    use_binomial = split_type == "binomial" and single_image

    if n_repeats > 1 and not use_binomial:
        warnings.warn(
            f"n_repeats={n_repeats} ignored: only used with "
            f"split_type='binomial' and single-image mode.",
            UserWarning,
            stacklevel=2,
        )

    if counts_mode != "counts" and not use_binomial:
        warnings.warn(
            f"counts_mode={counts_mode!r} ignored: only applies to "
            f"split_type='binomial' in single-image mode.",
            UserWarning,
            stacklevel=2,
        )

    if use_binomial and n_repeats == 1:
        logger.info(_BINOMIAL_SINGLE_REPEAT_MSG)

    if use_binomial and n_repeats > 1:
        # --- Multi-repeat binomial averaging ---
        rngs = _make_repeat_rngs(rng, n_repeats)
        all_curves: list[np.ndarray] = []
        all_resolutions: list[float] = []

        for rep_rng in rngs:
            frc_data, spacing_eff = _calculate_frc_single_pass(
                image1,
                None,
                bin_delta=bin_delta,
                spacing=spacing,
                zero_padding=zero_padding,
                pad_mode=pad_mode,
                disable_hamming=disable_hamming,
                average=average,
                backend=backend,
                split_type="binomial",
                counts_mode=counts_mode,
                gain=gain,
                offset=offset,
                readout_noise_rms=readout_noise_rms,
                rng=rep_rng,
            )

            # Analyze this repeat
            analyzer = FourierCorrelationAnalysis(
                frc_data,
                spacing_eff,
                resolution_threshold=resolution_threshold,
                threshold_value=threshold_value,
                snr_value=snr_value,
                curve_fit_type=curve_fit_type,
                curve_fit_degree=curve_fit_degree,
                smoothing_factor=smoothing_factor,
            )
            rep_result = analyzer.execute(z_correction=z_correction)[0]
            all_curves.append(rep_result.correlation["correlation"])
            all_resolutions.append(rep_result.resolution["resolution"])

        # Note: resolution-std is computed from per-repeat resolutions, while the
        # final resolution comes from analyzing the mean curve. These are different
        # statistical procedures — std reflects inter-split variability, not
        # uncertainty in the mean-curve resolution estimate.
        curves_stack = np.array(all_curves)
        mean_curve = np.nanmean(curves_stack, axis=0)
        std_curve = np.nanstd(curves_stack, axis=0)
        res_std = float(np.nanstd(all_resolutions))

        # Build final result from averaged curve
        # Re-use frequency/points from last repeat (identical across repeats)
        final_data = FourierCorrelationDataCollection()
        final_ds = FourierCorrelationData()
        final_ds.correlation["correlation"] = mean_curve
        final_ds.correlation["frequency"] = frc_data[0].correlation["frequency"]
        final_ds.correlation["points-x-bin"] = frc_data[0].correlation["points-x-bin"]
        final_ds.correlation["correlation-std"] = std_curve
        final_data[0] = final_ds

        analyzer = FourierCorrelationAnalysis(
            final_data,
            spacing_eff,
            resolution_threshold=resolution_threshold,
            threshold_value=threshold_value,
            snr_value=snr_value,
            curve_fit_type=curve_fit_type,
            curve_fit_degree=curve_fit_degree,
            smoothing_factor=smoothing_factor,
        )
        result = analyzer.execute(z_correction=z_correction)[0]
        result.correlation["correlation-std"] = std_curve
        result.resolution["resolution-std"] = res_std

        # Fallback: if the mean-curve resolution is NaN (e.g. averaged curve
        # never crosses the threshold), use the mean of per-repeat resolutions.
        if np.isnan(result.resolution["resolution"]):
            valid = [r for r in all_resolutions if not np.isnan(r)]
            if valid:
                result.resolution["resolution"] = float(np.mean(valid))

        return result

    # --- Single pass (checkerboard or single binomial) ---
    frc_data, spacing_eff = _calculate_frc_single_pass(
        image1,
        image2,
        bin_delta=bin_delta,
        spacing=spacing,
        zero_padding=zero_padding,
        pad_mode=pad_mode,
        disable_hamming=disable_hamming,
        average=average,
        backend=backend,
        split_type=split_type,
        counts_mode=counts_mode,
        gain=gain,
        offset=offset,
        readout_noise_rms=readout_noise_rms,
        rng=rng,
    )

    # Analyze results
    analyzer = FourierCorrelationAnalysis(
        frc_data,
        spacing_eff,
        resolution_threshold=resolution_threshold,
        threshold_value=threshold_value,
        snr_value=snr_value,
        curve_fit_type=curve_fit_type,
        curve_fit_degree=curve_fit_degree,
        smoothing_factor=smoothing_factor,
    )
    result = analyzer.execute(z_correction=z_correction)[0]

    # Apply cut-off correction (only for checkerboard single-image case)
    if single_image and split_type == "checkerboard":
        _apply_cutoff_correction(result)

    return result


def frc_resolution(
    image1: np.ndarray,
    image2: np.ndarray | None = None,
    *,
    bin_delta: int = 1,
    spacing: float | Sequence[float] | None = None,
    zero_padding: bool = True,
    pad_mode: str = "constant",
    curve_fit_type: str = "smooth-spline",
    smoothing_factor: float = 0.05,
    backend: str = "mask",
    split_type: Literal["checkerboard", "binomial"] = "checkerboard",
    counts_mode: Literal["counts", "poisson_thinning"] = "counts",
    gain: float = 1.0,
    offset: float = 0.0,
    readout_noise_rms: float = 0.0,
    n_repeats: int = 1,
    rng: np.random.Generator | int | None = None,
) -> float:
    """Calculate either single- or two-image FRC-based 2D image resolution."""
    frc_result = calculate_frc(
        image1,
        image2,
        bin_delta=bin_delta,
        curve_fit_type=curve_fit_type,
        smoothing_factor=smoothing_factor,
        spacing=spacing,
        zero_padding=zero_padding,
        pad_mode=pad_mode,
        backend=backend,
        split_type=split_type,
        counts_mode=counts_mode,
        gain=gain,
        offset=offset,
        readout_noise_rms=readout_noise_rms,
        n_repeats=n_repeats,
        rng=rng,
    )

    return frc_result.resolution["resolution"]


class DirectionalFSC(object):
    """Calculate the directional FSC between two images (unshifted FFT)."""

    def __init__(self, image1: np.ndarray, image2: np.ndarray, iterator):
        """Initialize the directional FSC."""
        if image1.ndim != 3 or image1.shape[0] <= 1:
            raise ValueError("Image must be 3D")

        if image1.shape != image2.shape:
            raise ValueError("Image dimensions do not match")

        self.iterator = iterator
        # Compute unshifted FFT (mean-subtracted, no fftshift)
        self.fft_image1 = np.fft.fftn(image1 - image1.mean())
        self.fft_image2 = np.fft.fftn(image2 - image2.mean())

    def execute(self):
        """Calculate the FSC."""
        data_structure = FourierCorrelationDataCollection()
        radii, angles = self.iterator.steps
        freq_nyq = self.iterator.nyquist
        shape = (angles.shape[0], radii.shape[0])
        c1 = np.zeros(shape, dtype=np.float32)
        c2 = np.zeros(shape, dtype=np.float32)
        c3 = np.zeros(shape, dtype=np.float32)
        points = np.zeros(shape, dtype=np.float32)

        # iterate through the sphere and calculate initial values
        for ind_ring, shell_idx, rotation_idx in self.iterator:
            subset1 = self.fft_image1[ind_ring]
            subset2 = self.fft_image2[ind_ring]

            c1[rotation_idx, shell_idx] = np.sum(subset1 * np.conjugate(subset2)).real
            c2[rotation_idx, shell_idx] = np.sum(np.abs(subset1) ** 2)
            c3[rotation_idx, shell_idx] = np.sum(np.abs(subset2) ** 2)

            points[rotation_idx, shell_idx] = len(subset1)

        # calculate FRC for every orientation
        spatial_freq = asnumpy(radii.astype(np.float32) / freq_nyq)
        for i in range(angles.size):
            frc = frc_from_sums(asnumpy(c2[i]), asnumpy(c3[i]), asnumpy(c1[i]))
            data_structure[angles[i]] = _frc_dataset(
                frc, spatial_freq, asnumpy(points[i])
            )

        return data_structure


def calculate_sectioned_fsc(
    image1: np.ndarray,
    image2: np.ndarray | None = None,
    *,
    bin_delta: int = 1,
    angle_delta: int = 15,
    extract_angle_delta: float = 0.1,
    resolution_threshold: str = "fixed",
    threshold_value: float = 0.143,
    snr_value: float = 7.0,
    curve_fit_type: str = "spline",
    curve_fit_degree: int = 3,
    smoothing_factor: float = 0.05,
    disable_hamming: bool = False,
    z_correction: float = 1.0,
    disable_3d_sum: bool = False,
    spacing: float | Sequence[float] | None = None,
    zero_padding: bool = True,
    pad_mode: str = "constant",
    split_type: Literal["checkerboard", "binomial"] = "checkerboard",
    counts_mode: Literal["counts", "poisson_thinning"] = "counts",
    gain: float = 1.0,
    offset: float = 0.0,
    readout_noise_rms: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> FourierCorrelationDataCollection:
    """Calculate sectioned FSC for one or two images."""
    single_image = image2 is None
    _validate_angle_delta(angle_delta)

    spacing = _normalize_spacing(spacing, image1.ndim)

    image1, image2 = preprocess_images(
        image1,
        image2,
        zero_padding=zero_padding,
        pad_mode=pad_mode,
        disable_hamming=disable_hamming,
        disable_3d_sum=disable_3d_sum,
        split_type=split_type,
        counts_mode=counts_mode,
        gain=gain,
        offset=offset,
        readout_noise_rms=readout_noise_rms,
        rng=rng,
    )

    iterator = AxialExcludeSectionedFourierShellIterator(
        image1.shape,
        bin_delta,
        angle_delta,
        extract_angle_delta,  # type: ignore[arg-type]
        spacing=spacing,
    )
    fsc_task = DirectionalFSC(image1, image2, iterator)
    data = fsc_task.execute()

    analyzer = FourierCorrelationAnalysis(
        data,
        # DirectionalFSC normalizes by the iterator's Nyquist, so the analyzer
        # must invert with the spacing that Nyquist implies.
        _normalization_spacing(iterator.nyquist),
        resolution_threshold=resolution_threshold,
        threshold_value=threshold_value,
        snr_value=snr_value,
        curve_fit_type=curve_fit_type,
        curve_fit_degree=curve_fit_degree,
        smoothing_factor=smoothing_factor,
    )
    result = analyzer.execute(z_correction=z_correction)

    # Apply cut-off correction (only for checkerboard single-image case)
    if single_image and split_type == "checkerboard":
        for angle, dataset in result:
            _apply_cutoff_correction(dataset)

    return result


def _calculate_fsc_sectioned_hist(
    image1: np.ndarray,
    image2: np.ndarray,
    *,
    bin_delta: int = 1,
    angle_delta: int = 45,
    spacing: Sequence[float] | None = None,
    exclude_axis_angle: float = 0.0,
    use_max_nyquist: bool = False,
) -> tuple[dict[int, FourierCorrelationData], float]:
    """
    Calculate sectioned FSC using vectorized histogram approach.

    Uses polar angle from Z axis (0-90°):
    - theta ≈ 0° = Z-dominated frequencies → Z resolution
    - theta ≈ 90° = XY-dominated frequencies → XY resolution

    Parameters
    ----------
    angle_delta : int
        Angular bin width in degrees. Must divide 90. Default 45 gives 2 bins.
        Use 15 to match mask backend's angular resolution.
    exclude_axis_angle : float
        Exclude frequencies within this angle (in degrees) from the Z axis.
        Following Koho et al. 2019 to avoid piezo/interpolation artifacts.
        Default: 0.0 (no exclusion).
    use_max_nyquist : bool
        If True, extend radial frequency range to maximum Nyquist (typically XY)
        instead of minimum Nyquist (typically Z). This allows XY-dominant sectors
        to measure higher frequencies for better XY resolution estimates.
        Default: False.

    Returns
    -------
    results : dict[int, FourierCorrelationData]
        Per-sector FSC data keyed by polar angle (degrees).
    max_freq : float
        Nyquist frequency the ``"frequency"`` axis was normalized by. Callers
        must convert crossings back to physical units with this value.
    """
    # Two sectors minimum: a single sector spanning 0-90° cannot separate the
    # axial from the in-plane cutoff, so it has no Z resolution to report.
    #
    # Angular edges: polar angle from Z axis (0-90°)
    # - theta ≈ 0° = Z-dominated frequencies → Z resolution
    # - theta ≈ 90° = XY-dominated frequencies → XY resolution
    n_angle, angle_edges = _sector_edges(angle_delta, min_sectors=2)

    # Compute FFT
    fft_image1 = np.fft.fftn(image1 - image1.mean())
    fft_image2 = np.fft.fftn(image2 - image2.mean())

    shape = image1.shape

    # Build radial edges
    r_edges, radii = radial_edges(
        shape, bin_delta, spacing=spacing, use_max_nyquist=use_max_nyquist
    )
    n_radial = len(radii)

    # Get bin IDs
    shape3d = (shape[0], shape[1], shape[2])
    radial_id, angle_id = sectioned_bin_id(
        shape3d,
        to_same_device(r_edges, image1),
        to_same_device(angle_edges, image1),
        spacing=spacing,
        exclude_axis_angle=exclude_axis_angle,
    )

    # Compute per-bin sums
    Sx2, Sy2, Sxy, N = reduce_frc_sums_sectioned(
        fft_image1, fft_image2, radial_id, angle_id, n_radial, n_angle
    )

    del fft_image1, fft_image2, radial_id, angle_id

    # Compute FSC for each angle
    results = {}

    # Nyquist for normalization
    if spacing is not None:
        max_freq = (
            _kmax_phys_max(shape, spacing)
            if use_max_nyquist
            else _kmax_phys(shape, spacing)
        )
    else:
        max_freq = float(
            max(n // 2 for n in shape)
            if use_max_nyquist
            else min(n // 2 for n in shape)
        )

    spatial_freq = asnumpy(radii.astype(np.float32) / max_freq)

    # Map bin index to output angle (polar angle from Z axis):
    # - theta ≈ 0° = Z-dominated frequencies → Z resolution
    # - theta ≈ 90° = XY-dominated frequencies → XY resolution
    #
    # Output angle is the bin center in polar coordinates (0-90°)
    for aid in range(n_angle):
        # Bin center in polar coords (midpoint of actual edge values)
        output_angle = int(
            round(0.5 * (float(angle_edges[aid]) + float(angle_edges[aid + 1])))
        )

        fsc = frc_from_sums(
            asnumpy(Sx2[aid]),
            asnumpy(Sy2[aid]),
            asnumpy(Sxy[aid]),
        )
        # Bins with no points are dropped: at low frequencies an angular sector
        # may hold no data at all (DC is purely Z-like, for instance).
        results[output_angle] = _frc_dataset(
            fsc, spatial_freq, asnumpy(N[aid].astype(np.float32))
        )

    return results, max_freq


def _resample_isotropic_for_fsc(
    image1: np.ndarray,
    image2: np.ndarray | None,
    spacing: list[float],
    resample_order: int = 1,
) -> tuple[np.ndarray, np.ndarray | None, list[float], float]:
    """Resample images to isotropic voxel size for FSC calculation.

    Handles target_z_size calculation, rescale_isotropic calls for
    image1/image2, even-dim cropping, and the anisotropy ratio that the axial
    correction needs.

    Parameters
    ----------
    image1 : np.ndarray
        First 3D image.
    image2 : np.ndarray or None
        Second 3D image (optional).
    spacing : list[float]
        Physical spacing per axis [z, y, x].
    resample_order : int
        Interpolation order (0=nearest, 1=linear, 3=cubic).

    Returns
    -------
    image1 : np.ndarray
        Resampled first image.
    image2 : np.ndarray or None
        Resampled second image (if provided).
    spacing_iso : list[float]
        Isotropic spacing (XY spacing for all axes).
    anisotropy : float
        Original ``z_spacing / xy_spacing``, for the Koho eq. (5) axial
        correction; see :func:`_fsc_extract_resolution`.
    """
    spacing_tuple = tuple(spacing)
    iso_spacing = spacing_tuple[1]  # Y spacing (assumes Y == X)
    if not np.isclose(spacing_tuple[1], spacing_tuple[2], rtol=1e-3):
        raise ValueError(
            f"Isotropic resampling requires equal XY spacing, "
            f"got Y={spacing_tuple[1]}, X={spacing_tuple[2]}"
        )

    target_z_size = int(round(image1.shape[0] * spacing_tuple[0] / iso_spacing))
    if target_z_size % 2 != 0:
        target_z_size -= 1  # Make even for checkerboard split

    image1 = rescale_isotropic(
        image1,
        spacing_tuple,
        downscale_xy=False,
        order=resample_order,
        preserve_range=True,
        target_z_size=target_z_size,
    ).astype(image1.dtype)

    if image2 is not None:
        image2 = rescale_isotropic(
            image2,
            spacing_tuple,
            downscale_xy=False,
            order=resample_order,
            preserve_range=True,
            target_z_size=target_z_size,
        ).astype(image2.dtype)

    # Crop to even dimensions (required for checkerboard split)
    even_shape = tuple(s - (s % 2) for s in image1.shape)
    if image1.shape != even_shape:
        slices = tuple(slice(0, es) for es in even_shape)
        image1 = image1[slices]
        if image2 is not None:
            image2 = image2[slices]

    spacing_iso = [iso_spacing] * image1.ndim
    return image1, image2, spacing_iso, spacing_tuple[0] / iso_spacing


def _fsc_hist_compute(
    image1: np.ndarray,
    image2: np.ndarray | None,
    *,
    bin_delta: int,
    angle_delta: int,
    spacing_list: list[float] | None,
    exclude_axis_angle: float,
    use_max_nyquist: bool,
    zero_padding: bool,
    pad_mode: str = "constant",
    average: bool,
    split_type: Literal["checkerboard", "binomial"] = "checkerboard",
    counts_mode: Literal["counts", "poisson_thinning"] = "counts",
    gain: float = 1.0,
    offset: float = 0.0,
    readout_noise_rms: float = 0.0,
    rng: np.random.Generator | int | None = None,
) -> tuple[dict[int, FourierCorrelationData], float]:
    """Compute sectioned FSC data using the hist backend.

    Handles single-image detection, forward checkerboard/binomial split, and
    optional reverse-split averaging (checkerboard only).

    Returns
    -------
    fsc_data : dict[int, FourierCorrelationData]
        Per-sector FSC data keyed by polar angle (degrees).
    max_freq : float
        Nyquist frequency the frequency axis was normalized by.
    """
    single_image = image2 is None
    use_checkerboard = split_type == "checkerboard"
    do_average = average and single_image and use_checkerboard

    # Reverse split needs the unprocessed input; preprocess_images never
    # mutates it, so no copy is required.
    original_image1 = image1 if do_average else None

    # Preprocess images (forward split)
    image1_proc, image2_proc = preprocess_images(
        image1,
        image2,
        zero_padding=zero_padding,
        pad_mode=pad_mode,
        disable_hamming=False,
        disable_3d_sum=False,
        split_type=split_type,
        counts_mode=counts_mode,
        gain=gain,
        offset=offset,
        readout_noise_rms=readout_noise_rms,
        rng=rng,
    )

    # Calculate sectioned FSC (forward split)
    fsc_data, max_freq = _calculate_fsc_sectioned_hist(
        image1_proc,
        image2_proc,
        bin_delta=bin_delta,
        angle_delta=angle_delta,
        spacing=spacing_list,
        exclude_axis_angle=exclude_axis_angle,
        use_max_nyquist=use_max_nyquist,
    )

    # Average with reverse split if enabled (checkerboard only)
    if do_average:
        assert original_image1 is not None  # set above when do_average=True
        image1_rev, image2_rev = preprocess_images(
            original_image1,  # type: ignore[arg-type]
            None,
            zero_padding=zero_padding,
            pad_mode=pad_mode,
            disable_hamming=False,
            disable_3d_sum=False,
            reverse_split=True,
        )

        fsc_data_rev, _ = _calculate_fsc_sectioned_hist(
            image1_rev,
            image2_rev,
            bin_delta=bin_delta,
            angle_delta=angle_delta,
            spacing=spacing_list,
            exclude_axis_angle=exclude_axis_angle,
            use_max_nyquist=use_max_nyquist,
        )

        for angle in fsc_data.keys():
            if angle in fsc_data_rev:
                corr_fwd = np.asarray(fsc_data[angle].correlation["correlation"])
                corr_rev = np.asarray(fsc_data_rev[angle].correlation["correlation"])
                fsc_data[angle].correlation["correlation"] = (
                    0.5 * corr_fwd + 0.5 * corr_rev
                )

    return fsc_data, max_freq


def _fsc_extract_resolution(
    fsc_data: dict[int, FourierCorrelationData],
    *,
    spacing_list: list[float] | None,
    max_freq: float,
    single_image: bool,
    resolution_threshold: str,
    threshold_value: float,
    resampled_anisotropy: float | None = None,
    xy_curve_fit_type: str = "smooth-spline",
    z_curve_fit_type: str = "smooth-spline",
    apply_cutoff: bool = True,
) -> dict[str, float]:
    """Extract XY and Z resolution from sectioned FSC data.

    Both directions invert their threshold crossing with the spacing implied by
    the frequency-axis normalization (see :func:`_normalization_spacing`), so a
    crossing at ``f_c`` maps to ``1 / (f_c * max_freq)``.

    A sector centered on the polar angle theta measures the *shell radius*
    ``|k| = k_z / cos(theta)`` at which correlation dies, not ``k_z`` itself, so
    XY and Z are extracted differently:

    - **XY**: reported as measured, from the highest-angle (most XY-dominated)
      sector. For theta near 90° the shell radius already is the in-plane
      frequency.
    - **Z**: the sector's period is divided by ``cos(theta)`` to project the
      shell radius onto the Z axis. The cascade starts from the highest sector
      below 45° (best statistics) and moves toward the axis. Sectors at or above
      45° are never used: they are XY-limited, so their band edge says nothing
      about the axial cutoff, and ``1 / cos(theta)`` would amplify it by up to
      7x. When no sector below 45° yields a crossing the result is ``nan``.

    That projection is purely geometric and carries no spacing term. It applies
    to a grid whose axes already carry their true physical Nyquists. When the
    volume was interpolated up to isotropic voxels instead, pass
    *resampled_anisotropy* and the Koho et al. (2019) eq. (5) multiplier
    ``1 + (anisotropy - 1) * |cos(theta)|`` is used in its place: resampling adds
    no information, so the real axial band limit stays at the original Z Nyquist
    while the grid now runs to the XY one, and that factor is what converts back.
    The two corrections address different errors and are not combined.

    ``FourierCorrelationAnalysis`` also still exposes ``z_correction`` directly,
    for callers reproducing miplib numbers.

    Parameters
    ----------
    fsc_data : dict
        Per-sector FSC data from _fsc_hist_compute.
    spacing_list : list[float] or None
        Physical spacing [z, y, x]. None for index units.
    max_freq : float
        Nyquist frequency the FSC frequency axis was normalized by.
    single_image : bool
        Whether single-image mode (for cutoff correction).
    resampled_anisotropy : float, optional
        ``z_spacing / xy_spacing`` of the *original* volume, when the input was
        interpolated up to isotropic voxels. None (default) means the frequency
        grid carries true per-axis Nyquists, so Z is projected geometrically.
    resolution_threshold : str
        Threshold criterion for resolution calculation.
    threshold_value : float
        Fixed threshold value.
    xy_curve_fit_type, z_curve_fit_type : str
        Curve fitting method for the XY and Z sectors.
    apply_cutoff : bool
        Whether to apply the checkerboard cutoff correction for single-image
        mode. Set to False for binomial splits. Default True.

    Returns
    -------
    dict[str, float]
        Resolution values with 'xy' and 'z' keys.
    """
    spacing_eff = _normalization_spacing(max_freq)

    angles = sorted(fsc_data.keys())

    def _resolution(cascade: list[int], fit_type: str) -> tuple[float, int | None]:
        """Return the first finite resolution along a cascade, and its sector."""
        for angle in cascade:
            coll = FourierCorrelationDataCollection()
            coll[angle] = fsc_data[angle]
            analyzer = FourierCorrelationAnalysis(
                coll,
                spacing_eff,
                resolution_threshold=resolution_threshold,
                threshold_value=threshold_value,
                curve_fit_type=fit_type,
            )
            analyzed = analyzer.execute()
            if single_image and apply_cutoff:
                _apply_cutoff_correction(analyzed[angle])
            res = analyzed[angle].resolution["resolution"]
            if np.isfinite(res) and res > 0:
                return float(res), angle
        return float("nan"), None

    # --- XY: highest-angle (most XY-like) sector first, reported as measured
    xy_resolution, _ = _resolution(list(reversed(angles)), xy_curve_fit_type)

    # --- Z: only sectors below 45°, from the highest (best statistics) inward
    z_cascade = [a for a in reversed(angles) if a < 45]
    z_measured, z_angle = _resolution(z_cascade, z_curve_fit_type)
    if z_angle is None:
        z_resolution = float("nan")
        if z_cascade:
            warnings.warn(
                f"No FSC threshold crossing in any Z-dominated sector "
                f"{z_cascade}; axial resolution is nan",
                RuntimeWarning,
                stacklevel=3,
            )
        else:
            warnings.warn(
                f"angle_delta leaves no sector below 45° (centers {angles}), so "
                "axial resolution cannot be separated from XY; z is nan",
                RuntimeWarning,
                stacklevel=3,
            )
    elif resampled_anisotropy is None:
        # Project the sector's shell radius onto Z: the measured period is
        # cos(theta) / k_z, and the axial period is 1 / k_z.
        z_resolution = z_measured / float(np.cos(np.deg2rad(z_angle)))
    else:
        # Koho et al. (2019) eq. (5). Interpolating Z up to isotropic voxels adds
        # no information, so the volume's real axial band limit stays at the
        # *original* Z Nyquist while the grid now runs to the XY one; this factor
        # converts back. That is a different job from the geometric projection
        # above, and on the pollen stack of Koho et al. Fig. 4b it is the one
        # that reproduces the published 3.91 um (projecting instead gave 2.02).
        z_resolution = z_measured * (
            1.0 + (resampled_anisotropy - 1.0) * abs(float(np.cos(np.deg2rad(z_angle))))
        )

    return {"xy": xy_resolution, "z": z_resolution}


def fsc_resolution(
    image1: np.ndarray,
    image2: np.ndarray | None = None,
    *,
    bin_delta: int = 1,
    angle_delta: int = 15,
    zero_padding: bool | None = None,
    pad_mode: str = "constant",
    spacing: float | Sequence[float] | None = None,
    resample_isotropic: bool = False,
    resample_order: int = 1,
    average: bool = True,
    exclude_axis_angle: float = 0.0,
    use_max_nyquist: bool = False,
    resolution_threshold: str = "fixed",
    threshold_value: float = 0.143,
    backend: str = "hist",
    xy_curve_fit_type: str = "smooth-spline",
    z_curve_fit_type: str = "smooth-spline",
    split_type: Literal["checkerboard", "binomial"] = "checkerboard",
    counts_mode: Literal["counts", "poisson_thinning"] = "counts",
    gain: float = 1.0,
    offset: float = 0.0,
    readout_noise_rms: float = 0.0,
    n_repeats: int = 1,
    rng: np.random.Generator | int | None = None,
) -> dict[str, float]:
    """
    Calculate either single- or two-image FSC-based 3D image resolution.

    Args:
        image1: First 3D input image
        image2: Second 3D input image (optional, uses checkerboard split if None)
        bin_delta: Bin width for radial binning. Default 1 matches the miplib
                      paper methodology (Koho et al. 2019).
        angle_delta: Angular bin width in degrees (default 15, same as mask backend)
        zero_padding: Whether to pad image to cube. Default: True for mask backend,
                      False for hist backend. The hist backend handles anisotropic
                      volumes correctly using physical frequency coordinates.
        spacing: Physical spacing per axis [z, y, x]. If None, uses index units.
        resample_isotropic: If True, resample images to isotropic voxel size before
                            FSC calculation. This matches the methodology in Koho et al.
                            2019 and is recommended for anisotropic volumes with limited
                            Z extent. Requires spacing to be provided. Default: False.
        resample_order: Interpolation order for isotropic resampling (0=nearest neighbor,
                        1=linear, 3=cubic). The miplib paper uses order=0 (nearest neighbor).
                        Default: 1 (linear interpolation).
        average: If True and single-image mode, average results from both diagonal
                 checkerboard splits (forward and reverse) to reduce variance.
                 Following Koho et al. 2019 methodology. Default: True.
        exclude_axis_angle: Exclude frequencies within this angle (in degrees) from
                            the Z axis. Following Koho et al. 2019 to avoid artifacts
                            from piezo stage motion and interpolation near the optical
                            axis. Default: 0.0 (no exclusion). Typical value: 5.0.
        use_max_nyquist: If True (hist backend only), extend frequency range to maximum
                         Nyquist (typically XY) instead of minimum (typically Z). This
                         allows XY-dominant sectors to measure higher frequencies for
                         better XY resolution estimates on anisotropic data.
                         Default: False.
        resolution_threshold: Threshold criterion for resolution calculation.
                              Options: "fixed", "one-bit", "half-bit", "three-sigma".
                              Default: "fixed" (uses threshold_value).
        threshold_value: Fixed threshold value when resolution_threshold="fixed".
                         Default: 0.143 (1/7 threshold). Note: The one-image calibration
                         correction was empirically calibrated for the 1/7 threshold
                         (Koho et al. 2019), so using other values may affect accuracy.
        backend: "hist" (vectorized, GPU-accelerated) or "mask" (deprecated).
        xy_curve_fit_type: Curve fit for the XY sector ("spline",
                           "smooth-spline" or "polynomial").
        z_curve_fit_type: Curve fit for the Z sector.
        split_type: "checkerboard" (default) or "binomial" (Rieger et al. 2024).
        counts_mode: For binomial split: "counts" or "poisson_thinning".
        gain: Camera gain (ADU/electron) for binomial counts mode.
        offset: Camera offset (ADU) for binomial counts mode.
        readout_noise_rms: Read-noise std in electrons for binomial counts mode.
        n_repeats: Number of independent binomial splits to average. Only used
                   when split_type="binomial" and single-image mode (image2 is
                   None). Ignored otherwise. Default: 1.
        rng: Random number generator, seed, or None for binomial split.

    Returns
    -------
    dict[str, float]
        Resolution values with keys:

        - ``"xy"``: XY resolution in physical units (or index units).
        - ``"z"``: Z resolution in physical units (or index units).
        - ``"xy_std"``: Std of XY resolution across repeats (binomial only,
          0.0 when n_repeats=1).
        - ``"z_std"``: Std of Z resolution across repeats (binomial only,
          0.0 when n_repeats=1).
    """
    # Set default zero_padding based on backend
    if zero_padding is None:
        zero_padding = backend == "mask"

    _validate_angle_delta(angle_delta)
    single_image = image2 is None
    use_binomial = split_type == "binomial" and single_image

    if n_repeats > 1 and not use_binomial:
        warnings.warn(
            f"n_repeats={n_repeats} ignored: only used with "
            f"split_type='binomial' and single-image mode.",
            UserWarning,
            stacklevel=2,
        )

    if counts_mode != "counts" and not use_binomial:
        warnings.warn(
            f"counts_mode={counts_mode!r} ignored: only applies to "
            f"split_type='binomial' in single-image mode.",
            UserWarning,
            stacklevel=2,
        )

    # --- Isotropic resampling (optional) ---
    # None keeps the geometric axial projection; resampling swaps in the Koho
    # eq. (5) correction instead (see _fsc_extract_resolution).
    resampled_anisotropy: float | None = None

    if resample_isotropic:
        if spacing is None:
            raise ValueError("resample_isotropic=True requires spacing to be provided")
        spacing_list = _normalize_spacing(spacing, image1.ndim)
        if spacing_list is None:
            raise RuntimeError("_normalize_spacing returned None with non-None spacing")
        image1, image2, spacing_list, resampled_anisotropy = (
            _resample_isotropic_for_fsc(
                image1,
                image2,
                spacing_list,  # type: ignore[arg-type]
                resample_order,
            )
        )
        spacing = spacing_list

    # --- Mask backend (deprecated) ---
    if backend == "mask":
        warnings.warn(
            "backend='mask' is deprecated and will be removed in a future version. "
            "Use backend='hist' (default) for faster GPU-accelerated computation.",
            DeprecationWarning,
            stacklevel=2,
        )
        if use_binomial and n_repeats > 1:
            warnings.warn(
                f"n_repeats={n_repeats} ignored: mask backend does not "
                f"support multi-repeat binomial splitting.",
                UserWarning,
                stacklevel=2,
            )
        fsc_result = calculate_sectioned_fsc(
            image1,
            image2,
            bin_delta=bin_delta,
            angle_delta=angle_delta,
            resolution_threshold=resolution_threshold,
            spacing=spacing,
            zero_padding=zero_padding,
            pad_mode=pad_mode,
            split_type=split_type,
            counts_mode=counts_mode,
            gain=gain,
            offset=offset,
            readout_noise_rms=readout_noise_rms,
            rng=rng,
        )

        angle_to_resolution = {
            int(angle): dataset.resolution["resolution"]
            for angle, dataset in fsc_result
        }

        z_res = 0.5 * (
            angle_to_resolution.get(0, np.nan) + angle_to_resolution.get(180, np.nan)
        )
        xy_res = 0.5 * (
            angle_to_resolution.get(90, np.nan) + angle_to_resolution.get(270, np.nan)
        )
        result = {"xy": xy_res, "z": z_res}
        if use_binomial:
            result["xy_std"] = 0.0
            result["z_std"] = 0.0
        return result

    # --- Hist backend ---
    spacing_list = _normalize_spacing(spacing, image1.ndim)

    if use_binomial and n_repeats > 1:
        # --- Multi-repeat binomial FSC ---
        rngs = _make_repeat_rngs(rng, n_repeats)
        all_results: list[dict[str, float]] = []

        for rep_rng in rngs:
            fsc_data, max_freq = _fsc_hist_compute(
                image1,
                None,
                bin_delta=bin_delta,
                angle_delta=angle_delta,
                spacing_list=spacing_list,
                exclude_axis_angle=exclude_axis_angle,
                use_max_nyquist=use_max_nyquist,
                zero_padding=zero_padding,
                pad_mode=pad_mode,
                average=average,
                split_type="binomial",
                counts_mode=counts_mode,
                gain=gain,
                offset=offset,
                readout_noise_rms=readout_noise_rms,
                rng=rep_rng,
            )

            rep_res = _fsc_extract_resolution(
                fsc_data,
                spacing_list=spacing_list,
                max_freq=max_freq,
                single_image=True,
                resampled_anisotropy=resampled_anisotropy,
                resolution_threshold=resolution_threshold,
                threshold_value=threshold_value,
                xy_curve_fit_type=xy_curve_fit_type,
                z_curve_fit_type=z_curve_fit_type,
                apply_cutoff=False,
            )
            all_results.append(rep_res)

        xy_vals = [r["xy"] for r in all_results]
        z_vals = [r["z"] for r in all_results]
        return {
            "xy": float(np.nanmean(xy_vals)),
            "z": float(np.nanmean(z_vals)),
            "xy_std": float(np.nanstd(xy_vals)),
            "z_std": float(np.nanstd(z_vals)),
        }

    # --- Single pass ---
    if use_binomial and n_repeats == 1:
        logger.info(_BINOMIAL_SINGLE_REPEAT_MSG)

    fsc_data, max_freq = _fsc_hist_compute(
        image1,
        image2,
        bin_delta=bin_delta,
        angle_delta=angle_delta,
        spacing_list=spacing_list,
        exclude_axis_angle=exclude_axis_angle,
        use_max_nyquist=use_max_nyquist,
        zero_padding=zero_padding,
        pad_mode=pad_mode,
        average=average,
        split_type=split_type,
        counts_mode=counts_mode,
        gain=gain,
        offset=offset,
        readout_noise_rms=readout_noise_rms,
        rng=rng,
    )

    result = _fsc_extract_resolution(
        fsc_data,
        spacing_list=spacing_list,
        max_freq=max_freq,
        single_image=single_image,
        resampled_anisotropy=resampled_anisotropy,
        resolution_threshold=resolution_threshold,
        threshold_value=threshold_value,
        xy_curve_fit_type=xy_curve_fit_type,
        z_curve_fit_type=z_curve_fit_type,
        apply_cutoff=split_type == "checkerboard",
    )
    if use_binomial:
        result["xy_std"] = 0.0
        result["z_std"] = 0.0
    return result


def _crop_slice_resolutions(
    crops: list[np.ndarray],
    *,
    bin_delta: int,
    spacing_list: list[float],
    crop_size: int,
    aggregate: Callable | None,
) -> dict[str, np.ndarray]:
    """Measure 2D FRC resolution on XY slices, XZ slices and the max projection.

    Shared by :func:`grid_crop_resolution` and :func:`five_crop_resolution`,
    which differ only in how the crop locations are chosen. Every slice is 2D,
    so resolution comes from :func:`frc_resolution` (a float per slice) rather
    than the 3D-only :func:`fsc_resolution`.

    XZ slices are measured on their native rectangular shape. They used to be
    reflect-padded along Z up to square, which replicated the real data 2-16x
    and left both checkerboard halves near-identical, so the FRC curve never
    descended through the threshold: at crop_size 128 the padded XZ resolution
    came back NaN for all but a handful of slices, and the few finite values
    were inflated ~4x. Measuring the unpadded slice needs the frequency axis to
    be normalized by the true minimum Nyquist, which it now is.
    """
    spacing_xy = (spacing_list[1], spacing_list[2])
    spacing_xz = (spacing_list[0], spacing_list[2])

    max_projection_resolutions = []
    xy_resolutions = []
    xz_resolutions = []
    for loc_image in crops:
        max_projection_resolutions.append(
            frc_resolution(loc_image.max(0), bin_delta=bin_delta, spacing=spacing_xy)
        )

        xy_slice_resolutions = []
        xz_slice_resolutions = []
        # One XZ slice per Z plane, spread evenly across the crop's Y extent.
        xz_rows = np.linspace(0, crop_size - 1, num=loc_image.shape[0], dtype=int)

        for slice_idx in range(loc_image.shape[0]):
            xy_slice_resolutions.append(
                frc_resolution(
                    loc_image[slice_idx], bin_delta=bin_delta, spacing=spacing_xy
                )
            )

            # zero_padding=False keeps the slice rectangular: padding it to a
            # cube would reintroduce the correlated-halves problem the reflect
            # padding caused, just with zeros.
            xz_slice_resolutions.append(
                frc_resolution(
                    loc_image[:, xz_rows[slice_idx], :],
                    bin_delta=bin_delta,
                    spacing=spacing_xz,
                    zero_padding=False,
                )
            )

        xy_resolutions.append(xy_slice_resolutions)
        xz_resolutions.append(xz_slice_resolutions)

    measured = {
        "max_projection": max_projection_resolutions,
        "xy": xy_resolutions,
        "xz": xz_resolutions,
    }
    if aggregate is None:
        return {key: np.asarray(values) for key, values in measured.items()}

    # The default ``np.nanmedian`` warns only when *every* input to an output
    # position is NaN, so a tile that measured nothing at any plane vanishes
    # from the aggregate silently -- the caller cannot tell what fraction of the
    # sampled field produced no measurement.
    for key, values in measured.items():
        arr = np.asarray(values, dtype=float)
        n_nan = int(np.isnan(arr).sum())
        if n_nan:
            warnings.warn(
                f"{n_nan} of {arr.size} {key} tile measurement(s) had no "
                "threshold crossing and are excluded from the aggregate",
                RuntimeWarning,
                stacklevel=3,
            )
    return {key: aggregate(values, axis=0) for key, values in measured.items()}


def _validate_crop_inputs(
    image: np.ndarray,
    spacing: float | Sequence[float] | None,
    crop_size: int,
    caller: str,
) -> list[float]:
    """Validate a tiled-resolution call and return the spacing as a list."""
    if spacing is None:
        raise ValueError(f"spacing is required for {caller}")
    spacing_list = _normalize_spacing(spacing, image.ndim)
    if image.ndim != 3 or spacing_list is None or len(spacing_list) != 3:
        raise ValueError(
            f"Expected 3D image and 3-element spacing, got shape {image.shape} "
            f"and spacing {spacing!r}"
        )
    if image.shape[0] >= image.shape[1] or image.shape[0] >= image.shape[2]:
        raise ValueError(f"Z dimension must be smallest, got shape {image.shape}")
    if image.shape[1] <= crop_size or image.shape[2] <= crop_size:
        raise ValueError(
            f"XY dimensions must exceed crop_size={crop_size}, got shape {image.shape}"
        )
    return spacing_list


def grid_crop_resolution(
    image: np.ndarray,
    *,
    bin_delta: int = 1,
    spacing: float | Sequence[float] | None = None,
    crop_size: int = 512,
    aggregate: Callable | None = np.nanmedian,
) -> dict[str, np.ndarray]:
    """Calculate FRC-based 3D image resolution by tiling and taking 2D slices along XY and XZ.

    Args:
        image: 3D image with Z as the smallest axis.
        bin_delta: Bin width for radial binning.
        spacing: Physical spacing per axis [z, y, x] (required).
        crop_size: Side length of the non-overlapping XY tiles.
        aggregate: Reduction applied across tiles (axis 0). Defaults to
            ``np.nanmedian`` because a slice with no threshold crossing is NaN,
            and plain ``np.median`` would propagate that single NaN to the whole
            aggregate. Pass None to get the raw per-tile resolutions.

    Returns
    -------
    dict[str, np.ndarray]
        ``"max_projection"``, ``"xy"`` and ``"xz"`` resolutions in the units of
        *spacing*.
    """
    spacing_list = _validate_crop_inputs(
        image, spacing, crop_size, "grid_crop_resolution"
    )
    crops = [
        image[:, y1:y2, x1:x2]
        for y1, y2, x1, x2 in get_xy_block_coords(image.shape, crop_size)
    ]
    return _crop_slice_resolutions(
        crops,
        bin_delta=bin_delta,
        spacing_list=spacing_list,
        crop_size=crop_size,
        aggregate=aggregate,
    )


def five_crop_resolution(
    image: np.ndarray,
    *,
    bin_delta: int = 1,
    spacing: float | Sequence[float] | None = None,
    crop_size: int = 512,
    aggregate: Callable | None = np.nanmedian,
) -> dict[str, np.ndarray]:
    """Calculate FRC-based 3D image resolution by taking 2D slices along XY and XZ at 4 corners and the center.

    Takes the same arguments as :func:`grid_crop_resolution`, but samples five
    fixed crops (four corners plus the centre) instead of a full tiling.
    """
    spacing_list = _validate_crop_inputs(
        image, spacing, crop_size, "five_crop_resolution"
    )
    crops = [
        loc(image, crop_size)
        for loc in (crop_tl, crop_bl, crop_tr, crop_br, crop_center)
    ]
    return _crop_slice_resolutions(
        crops,
        bin_delta=bin_delta,
        spacing_list=spacing_list,
        crop_size=crop_size,
        aggregate=aggregate,
    )


def frc_resolution_difference(
    image1: np.ndarray,
    image2: np.ndarray,
    *,
    bin_delta: int = 3,
    spacing: float | tuple[float, float] | None = None,
    backend: str = "mask",
) -> float:
    """Calculate difference between FRC-based resulutions of two images."""
    if isinstance(spacing, (int, float)):
        spacing = (spacing, spacing)

    image1_res = frc_resolution(
        image1, bin_delta=bin_delta, spacing=spacing, backend=backend
    )
    image2_res = frc_resolution(
        image2, bin_delta=bin_delta, spacing=spacing, backend=backend
    )
    return (image2_res - image1_res) * 1000  # return diff in nm
