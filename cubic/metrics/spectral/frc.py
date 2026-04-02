"""Implements 2D/3D Fourier Ring/Shell Correlation."""

import logging
import warnings
from typing import Any, Literal
from collections.abc import Callable, Sequence

import numpy as np

from cubic.cuda import asnumpy, get_array_module
from cubic.image_utils import (
    crop_bl,
    crop_br,
    crop_tl,
    crop_tr,
    pad_image,
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
    reduce_cross,
    reduce_power,
    frc_from_sums,
    radial_bin_id,
    _kmax_phys_max,
    sectioned_bin_id,
    _normalize_spacing,
    reduce_cross_sectioned,
    reduce_power_sectioned,
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


def _empty_aggregate(*args: Any, **kwargs: Any) -> Any:
    """Return unchanged first argument."""
    return args[0]


def frc_checkerboard_split(
    image: np.ndarray,
    reverse: bool = False,
    disable_3d_sum: bool = False,
    preserve_range: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Split image into two by checkerboard pattern."""
    if reverse:
        return reverse_checkerboard_split(
            image,
            disable_3d_sum=disable_3d_sum,
            preserve_range=preserve_range,
        )
    else:
        return checkerboard_split(
            image,
            disable_3d_sum=disable_3d_sum,
            preserve_range=preserve_range,
        )


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

    # Apply padding to first image
    if len(set(image1.shape)) > 1 and zero_padding:
        image1 = pad_image_to_cube(image1, mode=pad_mode)

    if single_image:
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
        if image2 is None:
            raise RuntimeError("image2 must not be None when single_image is False")
        if len(set(image2.shape)) > 1 and zero_padding:
            image2 = pad_image_to_cube(image2, mode=pad_mode)

    # Apply Hamming windowing to both images independently
    if not disable_hamming:
        image1 = hamming_window(image1)
        if image2 is None:
            raise RuntimeError("image2 must be set after splitting")
        image2 = hamming_window(image2)

    if image2 is None:
        raise RuntimeError("image2 must be set after preprocessing")
    return image1, image2


class FRC(object):
    """A class for calculating 2D Fourier ring correlation (unshifted FFT)."""

    def __init__(self, image1: np.ndarray, image2: np.ndarray, iterator, spacing=None):
        """Create new FRC executable object and perform FFT on input images."""
        if image1.shape != image2.shape:
            raise ValueError("The image dimensions do not match")
        if image1.ndim != 2:
            raise ValueError("Fourier ring correlation requires 2D images.")

        self.iterator = iterator
        self.spacing = spacing
        # Compute unshifted FFT (mean-subtracted, no fftshift)
        self.fft_image1 = np.fft.fftn(image1 - image1.mean())
        self.fft_image2 = np.fft.fftn(image2 - image2.mean())
        self.freq_nyq = int(np.floor(image1.shape[0] / 2.0))
        self.shape = image1.shape

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

        # Calculate FRC
        # If spacing was provided, radii are in physical units; normalize to [0,1]
        # If spacing was None, radii are in index units; normalize by Nyquist
        if self.spacing is None:
            spatial_freq = asnumpy(radii.astype(np.float32) / self.freq_nyq)
        else:
            # radii are in physical units; normalize to max frequency
            max_freq = float(np.max(radii))
            spatial_freq = asnumpy(radii.astype(np.float32) / max_freq)
        c1 = asnumpy(c1)
        c2 = asnumpy(c2)
        c3 = asnumpy(c3)
        n_points = asnumpy(points)

        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            eps = np.finfo(c1.dtype).tiny
            c1_safe = np.clip(np.abs(c1), eps, None)
            c2_safe = np.clip(c2, eps, None)
            c3_safe = np.clip(c3, eps, None)

            frc = np.exp(np.log(c1_safe) - 0.5 * (np.log(c2_safe) + np.log(c3_safe)))
            frc[frc == np.inf] = 0.0
            frc = np.nan_to_num(frc)

        data_set = FourierCorrelationData()
        data_set.correlation["correlation"] = frc
        data_set.correlation["frequency"] = spatial_freq
        data_set.correlation["points-x-bin"] = n_points

        return data_set


def _calculate_frc_core(
    image1: np.ndarray,
    image2: np.ndarray,
    bin_delta: int,
    *,
    backend: str = "mask",
    spacing: Sequence[float] | None = None,
    signed: bool = False,
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
        xp = get_array_module(fft_image1)
        edges = xp.asarray(edges)

        bin_id = radial_bin_id(shape, edges, spacing=spacing)

        Sx2, Nx = reduce_power(fft_image1, bin_id)
        Sy2, Ny = reduce_power(fft_image2, bin_id)
        Sxy_re, _ = reduce_cross(fft_image1, fft_image2, bin_id, numerator="real")

        # Compute FRC in log domain: exp(log|Sxy| - 0.5*(log(Sx2) + log(Sy2)))
        # Log domain avoids float32 overflow in Sx2 * Sy2.
        # When signed=True (binomial counts split), preserve the sign of Sxy so
        # anticorrelated noise at high frequencies is not flipped positive by abs().
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            eps = np.finfo(np.float32).tiny
            Sx2_safe = np.clip(Sx2, eps, None)
            Sy2_safe = np.clip(Sy2, eps, None)
            Sxy_abs = np.clip(np.abs(Sxy_re), eps, None)

            frc = np.exp(np.log(Sxy_abs) - 0.5 * (np.log(Sx2_safe) + np.log(Sy2_safe)))
            if signed:
                frc = np.where(Sxy_re < 0, -frc, frc)
            frc[frc == np.inf] = 0.0
            frc = np.nan_to_num(frc)

        # Convert radii to normalized spatial frequency
        # If spacing was provided, radii are in physical units; normalize to [0,1]
        # If spacing was None, radii are in index units; normalize by Nyquist
        freq_nyq = int(np.floor(shape[0] / 2.0))
        if spacing is None:
            spatial_freq = radii.astype(np.float32) / freq_nyq
        else:
            # radii are in physical units; normalize to max frequency
            max_freq = float(np.max(radii))
            spatial_freq = radii.astype(np.float32) / max_freq
        frc = asnumpy(frc)
        n_points = asnumpy(Nx.astype(np.float32))

        data_set = FourierCorrelationData()
        data_set.correlation["correlation"] = frc
        data_set.correlation["frequency"] = spatial_freq
        data_set.correlation["points-x-bin"] = n_points
        frc_data[0] = data_set
    else:
        # Default mask/iterator backend
        iterator = FourierRingIterator(image1.shape, bin_delta, spacing=spacing)
        frc_task = FRC(image1, image2, iterator, spacing=spacing)
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
    signed: bool = False,
) -> FourierCorrelationDataCollection:
    """Run a single FRC pass (split + FFT + radial binning).

    For checkerboard splits this includes the forward+reverse averaging.
    For binomial splits this is a single random split.
    """
    single_image = image2 is None
    use_checkerboard = split_type == "checkerboard"
    reverse = average and single_image and use_checkerboard
    original_image1 = image1.copy() if reverse else None

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
        signed=signed,
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

    return frc_data


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
    # Binomial counts split produces anticorrelated noise (n1 + n2 = n).
    # Use signed FRC so the negative noise correlation is not flipped positive by abs().
    use_signed = use_binomial and counts_mode == "counts"

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
            frc_data = _calculate_frc_single_pass(
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
                signed=use_signed,
            )

            # Analyze this repeat
            analyzer = FourierCorrelationAnalysis(
                frc_data,
                spacing[0] if spacing is not None else 1.0,
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
            spacing[0] if spacing is not None else 1.0,
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
    frc_data = _calculate_frc_single_pass(
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
        signed=use_signed,
    )

    # Analyze results
    analyzer = FourierCorrelationAnalysis(
        frc_data,
        spacing[0] if spacing is not None else 1.0,
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

    def __init__(
        self,
        image1: np.ndarray,
        image2: np.ndarray,
        iterator,
        normalize_power: bool = False,
    ):
        """Initialize the directional FSC."""
        if image1.ndim != 3 or image1.shape[0] <= 1:
            raise ValueError("Image must be 3D")

        if image1.shape != image2.shape:
            raise ValueError("Image dimensions do not match")

        self.iterator = iterator
        # Compute unshifted FFT (mean-subtracted, no fftshift)
        self.fft_image1 = np.fft.fftn(image1 - image1.mean())
        self.fft_image2 = np.fft.fftn(image2 - image2.mean())
        if normalize_power:
            pixels = image1.shape[0] ** 3
            self.fft_image1 /= np.array(pixels * np.mean(image1))
            self.fft_image2 /= np.array(pixels * np.mean(image2))

        self._result = None

    @property
    def result(self):
        """Return the FRC results."""
        if self._result is None:
            return self.execute()
        else:
            return self._result

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
        for i in range(angles.size):
            spatial_freq = asnumpy(radii.astype(np.float32) / freq_nyq)
            c1_i = asnumpy(c1[i])
            c2_i = asnumpy(c2[i])
            c3_i = asnumpy(c3[i])
            n_points = asnumpy(points[i])

            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                eps = np.finfo(c2_i.dtype).tiny
                c1_safe = np.clip(np.abs(c1_i), eps, None)
                c2_safe = np.clip(c2_i, eps, None)
                c3_safe = np.clip(c3_i, eps, None)

                frc = np.exp(
                    np.log(c1_safe) - 0.5 * (np.log(c2_safe) + np.log(c3_safe))
                )
                frc[frc == np.inf] = 0.0
                frc = np.nan_to_num(frc)

            result = FourierCorrelationData()
            result.correlation["correlation"] = frc
            result.correlation["frequency"] = spatial_freq
            result.correlation["points-x-bin"] = n_points

            data_structure[angles[i]] = result

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

    eff_spacing = spacing if spacing is not None else [1.0] * image1.ndim
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
        eff_spacing[0],
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
) -> dict[int, FourierCorrelationData]:
    """
    Calculate sectioned FSC using vectorized histogram approach.

    Uses polar angle from Z axis (0-90°):
    - theta ≈ 0° = Z-dominated frequencies → Z resolution
    - theta ≈ 90° = XY-dominated frequencies → XY resolution

    Parameters
    ----------
    angle_delta : int
        Angular bin width in degrees. Default 45 gives 2 bins.
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
    """
    xp = get_array_module(image1)

    # Compute FFT
    fft_image1 = np.fft.fftn(image1 - image1.mean())
    fft_image2 = np.fft.fftn(image2 - image2.mean())

    shape = image1.shape

    # Build radial edges
    r_edges, radii = radial_edges(
        shape, bin_delta, spacing=spacing, use_max_nyquist=use_max_nyquist
    )
    r_edges = xp.asarray(r_edges)
    n_radial = len(radii)

    # Angular edges: polar angle from Z axis (0-90°)
    # - theta ≈ 0° = Z-dominated frequencies → Z resolution
    # - theta ≈ 90° = XY-dominated frequencies → XY resolution
    n_angle = 90 // angle_delta
    angle_edges = xp.asarray(
        [float(i * angle_delta) for i in range(n_angle + 1)], dtype=np.float32
    )

    # Get bin IDs
    shape3d = (shape[0], shape[1], shape[2])
    radial_id, angle_id = sectioned_bin_id(
        shape3d,
        r_edges,
        angle_edges,
        spacing=spacing,
        exclude_axis_angle=exclude_axis_angle,
    )

    # Compute per-bin sums
    Sx2, Nx = reduce_power_sectioned(fft_image1, radial_id, angle_id, n_radial, n_angle)
    Sy2, Ny = reduce_power_sectioned(fft_image2, radial_id, angle_id, n_radial, n_angle)
    Sxy = reduce_cross_sectioned(
        fft_image1, fft_image2, radial_id, angle_id, n_radial, n_angle
    )

    angle_edges_cpu = asnumpy(angle_edges).copy()
    del fft_image1, fft_image2
    del radial_id, angle_id, r_edges, angle_edges

    # Compute FSC for each angle
    results = {}

    # Nyquist for normalization
    if spacing is not None:
        if use_max_nyquist:
            max_freq = _kmax_phys_max(shape, spacing)
        else:
            max_freq = _kmax_phys(shape, spacing)
    else:
        if use_max_nyquist:
            max_freq = max(n // 2 for n in shape)
        else:
            max_freq = min(n // 2 for n in shape)

    spatial_freq = asnumpy(radii.astype(np.float32) / max_freq)

    # Map bin index to output angle (polar angle from Z axis):
    # - theta ≈ 0° = Z-dominated frequencies → Z resolution
    # - theta ≈ 90° = XY-dominated frequencies → XY resolution
    #
    # Output angle is the bin center in polar coordinates (0-90°)
    for aid in range(n_angle):
        # Bin center in polar coords (midpoint of actual edge values)
        output_angle = int(
            round(0.5 * (float(angle_edges_cpu[aid]) + float(angle_edges_cpu[aid + 1])))
        )

        fsc = frc_from_sums(
            asnumpy(Sx2[aid]),
            asnumpy(Sy2[aid]),
            asnumpy(Sxy[aid]),
        )
        n_points = asnumpy(Nx[aid].astype(np.float32))

        # Filter out bins with no points (can happen at low frequencies
        # where angular sectors may have no data, e.g., DC is purely Z-like)
        valid_mask = n_points > 0
        fsc_valid = fsc[valid_mask]
        freq_valid = spatial_freq[valid_mask]
        n_points_valid = n_points[valid_mask]

        data_set = FourierCorrelationData()
        data_set.correlation["correlation"] = fsc_valid
        data_set.correlation["frequency"] = freq_valid
        data_set.correlation["points-x-bin"] = n_points_valid

        results[output_angle] = data_set

    return results


def _resample_isotropic_for_fsc(
    image1: np.ndarray,
    image2: np.ndarray | None,
    spacing: list[float],
    resample_order: int = 1,
) -> tuple[np.ndarray, np.ndarray | None, list[float], float, float]:
    """Resample images to isotropic voxel size for FSC calculation.

    Extracts the isotropic resampling block from fsc_resolution. Handles
    target_z_size calculation, rescale_isotropic calls for image1/image2,
    even-dim cropping, and z_factor computation.

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
    z_factor : float
        Anisotropy factor for k(theta) correction.
    original_spacing_z : float
        Original Z spacing before resampling.
    """
    spacing_tuple = tuple(spacing)
    original_spacing_z = spacing_tuple[0]
    iso_spacing = spacing_tuple[1]  # Y spacing (assumes Y == X)
    if not np.isclose(spacing_tuple[1], spacing_tuple[2], rtol=1e-3):
        raise ValueError(
            f"Isotropic resampling requires equal XY spacing, "
            f"got Y={spacing_tuple[1]}, X={spacing_tuple[2]}"
        )

    # z_factor from ORIGINAL spacing for k(theta) correction
    z_factor = original_spacing_z / iso_spacing
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
    return image1, image2, spacing_iso, z_factor, original_spacing_z


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
) -> dict[int, FourierCorrelationData]:
    """Compute sectioned FSC data using the hist backend.

    Handles single-image detection, forward checkerboard/binomial split, and
    optional reverse-split averaging (checkerboard only).

    Returns
    -------
    dict[int, FourierCorrelationData]
        Per-sector FSC data keyed by polar angle (degrees).
    """
    single_image = image2 is None
    use_checkerboard = split_type == "checkerboard"
    do_average = average and single_image and use_checkerboard

    # Save original for reverse split if averaging
    original_image1 = image1.copy() if do_average else None

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
    fsc_data = _calculate_fsc_sectioned_hist(
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

        fsc_data_rev = _calculate_fsc_sectioned_hist(
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

    return fsc_data


def _fsc_extract_resolution(
    fsc_data: dict[int, FourierCorrelationData],
    *,
    spacing_list: list[float] | None,
    single_image: bool,
    z_factor: float,
    original_spacing_z: float | None,
    resolution_threshold: str,
    threshold_value: float,
    z_curve_fit_type: str = "smooth-spline",
    apply_cutoff: bool = True,
) -> dict[str, float]:
    """Extract XY and Z resolution from sectioned FSC data.

    XY and Z are processed separately to handle k(theta) anisotropy correction
    correctly:

    - **XY**: The highest-angle sector (most XY-dominated) is processed with
      ``z_correction=1`` (no correction).  At ~82-90° polar angle, k(theta)≈1
      so no correction is needed; applying one inflates the result.
    - **Z**: Sectors are processed with full ``z_correction=z_factor``.  The
      cascade starts from the highest angle below 45° and moves downward (where
      k(theta) is large and statistics are reasonable), then falls back to
      angles above 45° if no crossing is found below.

    This matches the miplib approach where the mask backend has sectors at
    exactly 90° (k=1 for XY) and 0° (k=z for Z).  Our hist backend doesn't
    reach 90°/0°, so we emulate by skipping correction for XY and applying it
    for Z sectors.

    Parameters
    ----------
    fsc_data : dict
        Per-sector FSC data from _fsc_hist_compute.
    spacing_list : list[float] or None
        Physical spacing [z, y, x]. None for index units.
    single_image : bool
        Whether single-image mode (for cutoff correction).
    z_factor : float
        Anisotropy factor for k(theta) correction (z_spacing / xy_spacing).
    original_spacing_z : float or None
        Original Z spacing (set when isotropic resampling was used).
    resolution_threshold : str
        Threshold criterion for resolution calculation.
    threshold_value : float
        Fixed threshold value.
    apply_cutoff : bool
        Whether to apply the checkerboard cutoff correction for single-image
        mode. Set to False for binomial splits. Default True.

    Returns
    -------
    dict[str, float]
        Resolution values with 'xy' and 'z' keys.
    """
    if spacing_list is not None:
        spacing_xy = spacing_list[1]  # Y spacing (assumes Y==X)
    else:
        spacing_xy = 1.0

    angles = sorted(fsc_data.keys())

    # --- XY: no k(theta) correction ---
    # Process highest-angle sector (most XY-like) with z_correction=1.
    # At ~82° polar angle k(theta)≈1.3, so applying the full correction
    # inflates XY by ~30%.  Using z_correction=1 gives the raw (correct) XY.
    xy_resolution = float("nan")
    for angle in reversed(angles):  # highest angle first
        coll = FourierCorrelationDataCollection()
        coll[angle] = fsc_data[angle]
        analyzer = FourierCorrelationAnalysis(
            coll,
            spacing_xy,
            resolution_threshold=resolution_threshold,
            threshold_value=threshold_value,
            curve_fit_type="smooth-spline",
        )
        analyzed = analyzer.execute(z_correction=1)  # no k(theta)
        if single_image and apply_cutoff:
            _apply_cutoff_correction(analyzed[angle])
        res = analyzed[angle].resolution["resolution"]
        if np.isfinite(res) and res > 0:
            xy_resolution = res
            break

    # --- Z: with k(theta) correction ---
    # Cascade from highest angle below 45° downward (best statistics with
    # significant k(theta)), then fall back to angles above 45°.
    z_below_45 = [a for a in reversed(angles) if a < 45]
    z_above_45 = [a for a in angles if a >= 45]
    z_cascade = z_below_45 + z_above_45

    z_resolution = float("nan")
    for angle in z_cascade:
        coll = FourierCorrelationDataCollection()
        coll[angle] = fsc_data[angle]
        analyzer = FourierCorrelationAnalysis(
            coll,
            spacing_xy,
            resolution_threshold=resolution_threshold,
            threshold_value=threshold_value,
            curve_fit_type=z_curve_fit_type,
        )
        analyzed = analyzer.execute(z_correction=z_factor)
        if single_image and apply_cutoff:
            _apply_cutoff_correction(analyzed[angle])
        res = analyzed[angle].resolution["resolution"]
        if np.isfinite(res) and res > 0:
            z_resolution = res
            break

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
        backend: "hist" (vectorized, GPU-accelerated) or "mask" (deprecated)
                       (matches mask backend Z wedge coverage of ~8% of Fourier space).
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
    original_spacing_z = None
    z_factor = 1.0

    if resample_isotropic:
        if spacing is None:
            raise ValueError("resample_isotropic=True requires spacing to be provided")
        spacing_list = _normalize_spacing(spacing, image1.ndim)
        if spacing_list is None:
            raise RuntimeError("_normalize_spacing returned None with non-None spacing")
        image1, image2, spacing_list, z_factor, original_spacing_z = (
            _resample_isotropic_for_fsc(image1, image2, spacing_list, resample_order)  # type: ignore[arg-type]
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

    # Calculate z_factor from spacing if not already set by isotropic resampling
    if not resample_isotropic and spacing_list is not None:
        z_factor = spacing_list[0] / spacing_list[1]

    if use_binomial and n_repeats > 1:
        # --- Multi-repeat binomial FSC ---
        rngs = _make_repeat_rngs(rng, n_repeats)
        all_results: list[dict[str, float]] = []

        for rep_rng in rngs:
            fsc_data = _fsc_hist_compute(
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
                single_image=True,
                z_factor=z_factor,
                original_spacing_z=original_spacing_z,
                resolution_threshold=resolution_threshold,
                threshold_value=threshold_value,
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

    fsc_data = _fsc_hist_compute(
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
        single_image=single_image,
        z_factor=z_factor,
        original_spacing_z=original_spacing_z,
        resolution_threshold=resolution_threshold,
        threshold_value=threshold_value,
        z_curve_fit_type=z_curve_fit_type,
        apply_cutoff=split_type == "checkerboard",
    )
    if use_binomial:
        result["xy_std"] = 0.0
        result["z_std"] = 0.0
    return result


def grid_crop_resolution(
    image: np.ndarray,
    *,
    bin_delta: int = 1,
    spacing: float | Sequence[float] | None = None,
    crop_size: int = 512,
    pad_mode: str = "reflect",
    return_resolution: bool = True,
    aggregate: Callable | None = np.median,
) -> dict[str, np.ndarray]:
    """Calculate FRC-based 3D image resolution by tiling and taking 2D slices along XY and XZ."""
    aggregate_fn: Callable
    if not return_resolution or aggregate is None:
        aggregate_fn = _empty_aggregate
    else:
        aggregate_fn = aggregate

    if isinstance(spacing, (int, float)):
        spacing = [spacing, spacing, spacing]
    if spacing is None:
        raise ValueError("spacing is required for grid_crop_resolution")
    spacing_list: list[float] = list(spacing)

    if len(image.shape) != 3 or len(spacing_list) != 3:
        raise ValueError(
            f"Expected 3D image and 3-element spacing, got shape {image.shape}"
        )
    if image.shape[0] >= image.shape[1] or image.shape[0] >= image.shape[2]:
        raise ValueError(f"Z dimension must be smallest, got shape {image.shape}")
    if image.shape[1] <= crop_size or image.shape[2] <= crop_size:
        raise ValueError(
            f"XY dimensions must exceed crop_size={crop_size}, got shape {image.shape}"
        )

    spacing_xy = (spacing_list[1], spacing_list[2])
    spacing_xz = (spacing_list[0], spacing_list[2])

    locations = get_xy_block_coords(image.shape, crop_size)

    max_projection_resolutions = []
    xy_resolutions = []
    xz_resolutions = []
    for y1, y2, x1, x2 in locations:
        loc_image = image[:, y1:y2, x1:x2]
        max_projection_resolution = fsc_resolution(
            loc_image.max(0),
            bin_delta=bin_delta,
            spacing=spacing_xy,
        )
        max_projection_resolutions.append(max_projection_resolution)

        xy_slice_resolutions = []
        xz_slice_resolutions = []
        xz_slices = np.linspace(0, crop_size - 1, num=loc_image.shape[0], dtype=int)

        for slice_idx in range(loc_image.shape[0]):
            xy_slice_resolutions.append(
                fsc_resolution(
                    loc_image[slice_idx, :, :],
                    bin_delta=bin_delta,
                    spacing=spacing_xy,
                )
            )

            xz_slice = loc_image[:, xz_slices[slice_idx], :]

            half = (xz_slice.shape[1] - xz_slice.shape[0]) // 2
            if (xz_slice.shape[1] - xz_slice.shape[0]) % 2 != 0:
                pad_arg: int | tuple[int, int] = (half + 1, half)
            else:
                pad_arg = half
            padded_xz_slice = pad_image(xz_slice, pad_arg, 0, pad_mode)

            xz_slice_resolutions.append(
                fsc_resolution(
                    padded_xz_slice,
                    bin_delta=bin_delta,
                    spacing=spacing_xz,
                )
            )

        xy_resolutions.append(xy_slice_resolutions)
        xz_resolutions.append(xz_slice_resolutions)

    return {
        "max_projection": aggregate_fn(max_projection_resolutions, axis=0),  # type: ignore[arg-type]
        "xy": aggregate_fn(xy_resolutions, axis=0),  # type: ignore[arg-type]
        "xz": aggregate_fn(xz_resolutions, axis=0),  # type: ignore[arg-type]
    }


def five_crop_resolution(
    image: np.ndarray,
    *,
    bin_delta: int = 1,
    spacing: float | Sequence[float] | None = None,
    crop_size: int = 512,
    pad_mode: str = "reflect",
    return_resolution: bool = True,
    aggregate: Callable = np.median,
) -> dict[str, np.ndarray]:
    """Calculate FRC-based 3D image resolution by taking 2D slices along XY and XZ at 4 corners and the center."""
    aggregate_fn: Callable
    if not return_resolution or aggregate is None:
        aggregate_fn = _empty_aggregate
    else:
        aggregate_fn = aggregate

    if isinstance(spacing, (int, float)):
        spacing = [spacing, spacing, spacing]
    if spacing is None:
        raise ValueError("spacing is required for five_crop_resolution")
    spacing_list: list[float] = list(spacing)

    if len(image.shape) != 3 or len(spacing_list) != 3:
        raise ValueError(
            f"Expected 3D image and 3-element spacing, got shape {image.shape}"
        )
    if image.shape[0] >= image.shape[1] or image.shape[0] >= image.shape[2]:
        raise ValueError(f"Z dimension must be smallest, got shape {image.shape}")
    if image.shape[1] <= crop_size or image.shape[2] <= crop_size:
        raise ValueError(
            f"XY dimensions must exceed crop_size={crop_size}, got shape {image.shape}"
        )

    spacing_xy = (spacing_list[1], spacing_list[2])
    spacing_xz = (spacing_list[0], spacing_list[2])

    locations = [crop_tl, crop_bl, crop_tr, crop_br, crop_center]
    max_projection_resolutions = []
    xy_resolutions = []
    xz_resolutions = []
    for loc in locations:
        loc_image = loc(image, crop_size)  # type: ignore

        max_projection_resolution = fsc_resolution(
            loc_image.max(0),
            bin_delta=bin_delta,
            spacing=spacing_xy,
        )
        max_projection_resolutions.append(max_projection_resolution)

        xy_slice_resolutions = []
        xz_slice_resolutions = []
        xz_slices = np.linspace(0, crop_size - 1, num=loc_image.shape[0], dtype=int)
        for slice_idx in range(loc_image.shape[0]):
            xy_slice_resolutions.append(
                fsc_resolution(
                    loc_image[slice_idx, :, :],
                    bin_delta=bin_delta,
                    spacing=spacing_xy,
                )
            )

            xz_slice = loc_image[:, xz_slices[slice_idx], :]

            padded_xz_slice = pad_image(
                xz_slice, (xz_slice.shape[1] - xz_slice.shape[0]) // 2, 0, pad_mode
            )

            xz_slice_resolutions.append(
                fsc_resolution(
                    padded_xz_slice,
                    bin_delta=bin_delta,
                    spacing=spacing_xz,
                )
            )

        xy_resolutions.append(xy_slice_resolutions)
        xz_resolutions.append(xz_slice_resolutions)

    return {
        "max_projection": aggregate_fn(max_projection_resolutions, axis=0),  # type: ignore[arg-type]
        "xy": aggregate_fn(xy_resolutions, axis=0),  # type: ignore[arg-type]
        "xz": aggregate_fn(xz_resolutions, axis=0),  # type: ignore[arg-type]
    }


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
