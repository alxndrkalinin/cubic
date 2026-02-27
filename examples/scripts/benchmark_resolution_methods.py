#!/usr/bin/env python3
"""
Benchmark Resolution Estimation Methods: FRC, FSC, and DCR.

This script compares different resolution estimation methods on STED microscopy data:
- 2D FRC (Fourier Ring Correlation) on XY and XZ slices
- 2D/3D DCR (Decorrelation Analysis)
- 3D FSC (Fourier Shell Correlation) with hist backend

Uses published preprocessing conventions:
- FRC/FSC: Hamming window, zero padding
- DCR: Tukey window (alpha=0.1), no padding

Requires GPU (CuPy) for execution.

Usage:
    python benchmark_resolution_methods.py                  # default settings
    python benchmark_resolution_methods.py crop.size=1024   # different crop size
    python benchmark_resolution_methods.py data_name=astrocyte image_path=path/to/image.tif
"""

import gc
import csv
import sys
from pathlib import Path

import hydra
import numpy as np
import matplotlib
from omegaconf import OmegaConf, DictConfig

matplotlib.use("Agg")  # Non-interactive backend for faster plot generation
import time
import warnings

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cubic.image_utils import hamming_window, rescale_isotropic, checkerboard_split
from cubic.metrics.spectral import dcr_resolution, frc_resolution, fsc_resolution
from cubic.metrics.spectral.dcr import dcr_curve
from cubic.metrics.spectral.frc import (
    calculate_frc,
    pad_image_to_cube,
    _calculate_fsc_sectioned_hist,
)

# Require GPU
try:
    import cupy as cp

    from cubic.cuda import ascupy, asnumpy
except ImportError:
    raise RuntimeError("GPU (CuPy) is required for benchmarking. Please install CuPy.")


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB using CUDA runtime."""
    mem_info = cp.cuda.runtime.memGetInfo()
    # mem_info returns (free, total)
    used = (mem_info[1] - mem_info[0]) / (1024 * 1024)
    return used


def reset_gpu_memory():
    """Reset GPU memory pool and return baseline usage."""
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()
    cp.cuda.Stream.null.synchronize()
    return get_gpu_memory_mb()


def load_sted_images(data_path=None):
    """Load STED tubulin A/B images."""
    from skimage import io

    if data_path is None:
        # Default path relative to cubic package
        data_path = Path(__file__).parent.parent / "data"

    img_a = io.imread(data_path / "Tubulin_STED_8bit_0_a.tif")
    img_b = io.imread(data_path / "Tubulin_STED_8bit_0_b.tif")
    return img_a.astype(np.float32), img_b.astype(np.float32)


def load_single_image(image_path):
    """Load a single image for single-image-only evaluation."""
    from skimage import io

    img = io.imread(image_path)
    return img.astype(np.float32)


def save_results_csv(results, output_path, data_name="", crop_size=None):
    """Save benchmark results to CSV file.

    Parameters
    ----------
    results : list of tuples
        List of (method_name, xy_resolution, z_resolution, time_ms, peak_mem_mb) tuples
    output_path : Path
        Path to save CSV file
    data_name : str
        Name of the input data
    crop_size : int, optional
        Crop half-size used for the benchmark
    """
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["method", "xy_nm", "z_nm", "time_ms", "peak_mem_mb", "data", "crop_size"]
        )
        for name, xy, z, t, mem in results:
            xy_val = (
                xy
                if xy is not None and not (isinstance(xy, float) and np.isnan(xy))
                else ""
            )
            z_val = (
                z
                if z is not None and not (isinstance(z, float) and np.isnan(z))
                else ""
            )
            writer.writerow(
                [
                    name,
                    xy_val,
                    z_val,
                    f"{t:.1f}",
                    f"{mem:.1f}",
                    data_name,
                    crop_size or "",
                ]
            )


def time_function(func, *args, n_runs=3, **kwargs):
    """Time a function and track peak GPU memory usage.

    Returns: (result, avg_time_ms, peak_memory_mb)
    """
    # Warm-up run and get memory baseline
    baseline_mem = reset_gpu_memory()
    result = func(*args, **kwargs)

    cp.cuda.Stream.null.synchronize()
    first_run_mem = get_gpu_memory_mb()

    times = []
    peak_memories = []

    for _ in range(n_runs):
        # Reset memory before each run
        baseline_mem = reset_gpu_memory()

        start = time.perf_counter()
        result = func(*args, **kwargs)
        cp.cuda.Stream.null.synchronize()
        end = time.perf_counter()

        # Measure memory delta
        current_mem = get_gpu_memory_mb()
        peak_mem = max(0, current_mem - baseline_mem)

        times.append((end - start) * 1000)
        peak_memories.append(peak_mem)

    # Use the first run memory as a better estimate (includes all allocations)
    peak_memories.append(max(0, first_run_mem - baseline_mem))

    return result, np.mean(times), np.max(peak_memories) if peak_memories else 0.0


def plot_frc_curves(img_a, img_b, spacing, title, ax):
    """Plot FRC correlation curve."""
    # Preprocess
    if len(set(img_a.shape)) > 1:
        img_a = pad_image_to_cube(img_a)
        img_b = pad_image_to_cube(img_b)

    img_a = hamming_window(img_a)
    img_b = hamming_window(img_b)

    # Compute FRC
    frc_result = calculate_frc(img_a, img_b, spacing=spacing)

    freq = np.asarray(frc_result.correlation["frequency"])
    corr = np.asarray(frc_result.correlation["correlation"])

    ax.plot(freq, corr, "b-", linewidth=1.5, label="FRC")
    ax.axhline(y=1 / 7, color="r", linestyle="--", label="1/7 threshold")
    ax.axhline(y=0.5, color="orange", linestyle=":", label="0.5 threshold")

    # Find resolution at 1/7 threshold
    below = corr < 1 / 7
    if np.any(below):
        idx = np.argmax(below)
        res_freq = freq[idx]
        if isinstance(spacing, (list, tuple)):
            sp = spacing[0] if len(spacing) == 1 else spacing[-1]
        else:
            sp = spacing
        resolution = sp / res_freq if res_freq > 0 else np.nan
        ax.axvline(x=res_freq, color="g", alpha=0.5, linestyle="-")
        ax.set_title(f"{title}\nResolution: {resolution:.1f} nm")
    else:
        ax.set_title(f"{title}\nResolution: N/A")

    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("Correlation")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_dcr_curve(img, spacing, title, ax):
    """Plot DCR decorrelation curve."""
    resolution_nm, radii, all_curves, all_peaks = dcr_curve(img, spacing=spacing)

    radii = np.asarray(radii)
    all_peaks = np.asarray(all_peaks)

    # Plot all curves with transparency
    for i, curve in enumerate(all_curves):
        alpha = 0.3 + 0.7 * (i / len(all_curves))
        ax.plot(radii, np.asarray(curve), "b-", linewidth=1, alpha=alpha)

    # Find maximum peak position
    if len(all_peaks) > 0:
        peak_freqs = all_peaks[:, 0]
        max_peak_freq = np.max(peak_freqs)
        if max_peak_freq > 0:
            ax.axvline(
                x=max_peak_freq,
                color="r",
                linestyle="--",
                alpha=0.7,
                label=f"Peak at r={max_peak_freq:.3f}",
            )

    ax.set_title(f"{title}\nResolution: {resolution_nm:.1f} nm")
    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("Decorrelation")
    ax.set_xlim([0, 1])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_fsc_sectors(fsc_data, spacing_xy, spacing_z_orig, title, axes):
    """Plot FSC curves for different angular sectors."""
    angles = sorted(fsc_data.keys())

    def dist_to_xy(a):
        return min(abs(a - 90), abs(a - 270))

    def dist_to_z(a):
        return min(a, abs(a - 180), abs(a - 360))

    angle_xy = min(angles, key=dist_to_xy)
    angle_z = min(angles, key=dist_to_z)

    z_freq_limit = spacing_xy / spacing_z_orig

    # Plot XY sector
    ax = axes[0]
    data = fsc_data[angle_xy]
    freq = np.asarray(data.correlation["frequency"])
    corr = np.asarray(data.correlation["correlation"])

    ax.plot(freq, corr, "b-", linewidth=1.5)
    ax.axhline(y=0.5, color="r", linestyle="--", label="0.5 threshold")

    below = corr < 0.5
    if np.any(below):
        idx = np.argmax(below)
        res_freq = freq[idx]
        resolution = spacing_xy / res_freq if res_freq > 0 else np.nan
        ax.axvline(x=res_freq, color="g", alpha=0.5)
        ax.set_title(f"XY Sector ({angle_xy}°)\nRes: {resolution:.1f} nm")
    else:
        ax.set_title(f"XY Sector ({angle_xy}°)\nRes: N/A")

    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("Correlation")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Plot Z sector
    ax = axes[1]
    data = fsc_data[angle_z]
    freq = np.asarray(data.correlation["frequency"])
    corr = np.asarray(data.correlation["correlation"])

    ax.plot(freq, corr, "b-", linewidth=1.5)
    ax.axhline(y=0.5, color="r", linestyle="--", label="0.5 threshold")
    ax.axvline(x=z_freq_limit, color="orange", linestyle=":", label=f"Z Nyquist")

    below = corr < 0.5
    if np.any(below):
        idx = np.argmax(below)
        res_freq = freq[idx]
        resolution = spacing_z_orig / res_freq if res_freq > 0 else np.nan
        ax.axvline(x=res_freq, color="g", alpha=0.5)
        ax.set_title(f"Z Sector ({angle_z}°)\nRes: {resolution:.1f} nm")
    else:
        ax.set_title(f"Z Sector ({angle_z}°)\nRes: N/A")

    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("Correlation")
    ax.set_xlim([0, 0.6])
    ax.set_ylim([0, 1.05])
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_fsc_polar(fsc_data, spacing_xy, ax):
    """Plot FSC resolution as polar/radial plot by angle."""
    angles = sorted(fsc_data.keys())
    resolutions = []

    for angle in angles:
        data = fsc_data[angle]
        freq = np.asarray(data.correlation["frequency"])
        corr = np.asarray(data.correlation["correlation"])

        below = corr < 0.5
        if np.any(below):
            idx = np.argmax(below)
            res_freq = freq[idx]
            resolution = spacing_xy / res_freq if res_freq > 0 else np.nan
        else:
            resolution = np.nan
        resolutions.append(resolution)

    angles_rad = np.deg2rad(angles)
    angles_rad = np.append(angles_rad, angles_rad[0])
    resolutions = np.append(resolutions, resolutions[0])

    ax.plot(angles_rad, resolutions, "b-o", markersize=4, linewidth=1.5)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_title("Resolution by Angle\n(nm)", pad=20)

    ax.annotate("Z (0°/180°)", xy=(0, ax.get_ylim()[1] * 0.8), fontsize=8, ha="center")
    ax.annotate(
        "XY (90°/270°)", xy=(np.pi / 2, ax.get_ylim()[1] * 0.8), fontsize=8, ha="center"
    )


def plot_fsc_diagnostic(fsc_data, spacing_xy, spacing_z_orig, output_path, dpi=150):
    """Plot per-sector FSC curves with both threshold types for diagnostics.

    Visualizes each angular sector's FSC curve with fixed (0.143) and one-bit
    threshold overlaid, showing exactly where/why threshold crossings fail.

    Parameters
    ----------
    fsc_data : dict[int, FourierCorrelationData]
        Per-angle FSC data from ``_calculate_fsc_sectioned_hist``.
    spacing_xy : float
        XY pixel spacing in nm.
    spacing_z_orig : float
        Original Z spacing in nm (before isotropic resampling).
    output_path : Path
        Path to save the diagnostic plot.
    dpi : int
        Plot resolution.
    """
    angles = sorted(fsc_data.keys())
    n_sectors = len(angles)
    if n_sectors == 0:
        return

    # Classify sectors: 0-45° = Z-dominated, 45-90° = XY-dominated
    z_angles = [a for a in angles if a <= 45]
    xy_angles = [a for a in angles if a > 45]

    ncols = min(4, n_sectors)
    nrows = (n_sectors + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3.5 * nrows), squeeze=False
    )

    fixed_thr = 1.0 / 7.0  # 0.143

    for idx, angle in enumerate(angles):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]

        data = fsc_data[angle]
        freq = np.asarray(data.correlation["frequency"])
        corr = np.asarray(data.correlation["correlation"])
        points_per_bin = np.asarray(data.correlation["points-x-bin"])

        # One-bit threshold: (0.5 + 2.4142/sqrt(N)) / (1.5 + 1.4142/sqrt(N))
        sqrt_n = np.sqrt(np.maximum(points_per_bin, 1))
        one_bit = (0.5 + 2.4142 / sqrt_n) / (1.5 + 1.4142 / sqrt_n)

        # Plot FSC curve
        ax.plot(freq, corr, "b-", linewidth=1.5, label="FSC")

        # Plot thresholds
        ax.axhline(
            y=fixed_thr,
            color="red",
            linestyle="--",
            linewidth=1,
            alpha=0.7,
            label=f"fixed ({fixed_thr:.3f})",
        )
        ax.plot(
            freq,
            one_bit,
            "orange",
            linestyle="-.",
            linewidth=1,
            alpha=0.8,
            label="one-bit",
        )

        # Mark crossing points
        for thr_vals, color, name in [
            (np.full_like(corr, fixed_thr), "red", "fixed"),
            (one_bit, "orange", "one-bit"),
        ]:
            diff = corr - thr_vals
            crossings = np.where(diff <= 0)[0]
            if len(crossings) > 0:
                ci = crossings[0]
                ax.axvline(x=freq[ci], color=color, alpha=0.4, linewidth=1)
                # Convert to resolution
                spacing = spacing_xy if angle > 45 else spacing_z_orig
                res_nm = spacing / freq[ci] if freq[ci] > 0 else np.nan
                ax.annotate(
                    f"{name}: {res_nm:.0f} nm",
                    xy=(freq[ci], corr[ci]),
                    xytext=(5, 10),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                )

        # Sector label
        sector_type = "XY" if angle > 45 else "Z"
        ax.set_title(f"{angle}° ({sector_type})", fontsize=10)
        ax.set_xlabel("Normalized frequency", fontsize=8)
        ax.set_ylabel("Correlation", fontsize=8)
        ax.set_xlim([0, 1])
        ax.set_ylim([-0.05, 1.05])
        ax.legend(fontsize=6, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for idx in range(n_sectors, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    fig.suptitle(
        f"FSC Diagnostic: Per-Sector Curves with Threshold Comparison\n"
        f"spacing_xy={spacing_xy} nm, spacing_z={spacing_z_orig} nm, "
        f"{n_sectors} sectors",
        fontsize=12,
    )
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def plot_comparison_bars(results, ax):
    """Plot comparison bar chart."""
    methods = []
    xy_vals = []
    z_vals = []

    for item in results:
        name, xy, z = item[0], item[1], item[2]
        methods.append(
            name.replace(" (two-image)", "\n(two-img)").replace(
                " (single-image)", "\n(single)"
            )
        )
        xy_vals.append(xy if xy and not np.isnan(xy) and not np.isinf(xy) else 0)
        z_vals.append(z if z and not np.isnan(z) and not np.isinf(z) else 0)

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(
        x - width / 2, xy_vals, width, label="XY", color="steelblue", alpha=0.8
    )
    bars2 = ax.bar(x + width / 2, z_vals, width, label="Z", color="coral", alpha=0.8)

    ax.axhline(
        y=140, color="steelblue", linestyle="--", alpha=0.5, label="Paper XY (~140nm)"
    )
    ax.axhline(
        y=638, color="coral", linestyle="--", alpha=0.5, label="Paper Z (~638nm)"
    )

    ax.set_ylabel("Resolution (nm)")
    ax.set_title("Resolution Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=8)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val in zip(bars1, xy_vals):
        if val > 0:
            ax.annotate(
                f"{val:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )
    for bar, val in zip(bars2, z_vals):
        if val > 0:
            ax.annotate(
                f"{val:.0f}",
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
            )


def plot_timing_bars(results, ax):
    """Plot timing comparison bar chart."""
    methods = []
    times = []

    for item in results:
        name, t = item[0], item[3]
        methods.append(
            name.replace(" (two-image)", "\n(two)").replace(
                " (single-image)", "\n(single)"
            )
        )
        times.append(t)

    colors = [
        "steelblue" if "mask" not in name.lower() else "lightcoral" for name in methods
    ]
    bars = ax.bar(methods, times, color=colors, alpha=0.8)

    ax.set_ylabel("Time (ms)")
    ax.set_title("Execution Time Comparison")
    ax.set_yscale("log")
    ax.tick_params(axis="x", labelsize=7, rotation=45)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, t in zip(bars, times):
        label = f"{t:.0f}ms" if t < 1000 else f"{t / 1000:.1f}s"
        ax.annotate(
            label,
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )


def run_benchmark(
    output_dir,
    data_dir,
    spacing_cfg,
    crop_cfg,
    method_cfg,
    methods_cfg,
    timing_cfg,
    plots_cfg,
    image_path=None,
    single_image_only=False,
    data_name="",
):
    """Run the benchmark with given configuration.

    Uses published preprocessing conventions:
    - FRC/FSC: Hamming window, zero padding
    - DCR: Tukey window (alpha=0.1), no padding

    Parameters
    ----------
    image_path : str, optional
        Path to a single image for single-image-only evaluation.
        If provided, only single-image methods will be run.
    single_image_only : bool
        If True, only run single-image methods (skip two-image FRC/FSC).
    data_name : str
        Name of the input data for organizing outputs.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("RESOLUTION ESTIMATION BENCHMARK")
    print("=" * 80)
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")

    # Load data
    if image_path:
        print(f"\nLoading single image: {image_path}")
        img_a = load_single_image(image_path)
        img_b = None
        single_image_only = True  # Force single-image mode
        print(f"Full image shape: {img_a.shape}")
    else:
        print("\nLoading STED tubulin images...")
        data_path = Path(data_dir) if data_dir else None
        img_a, img_b = load_sted_images(data_path)
        print(f"Full image shape: {img_a.shape}")

    if single_image_only:
        print("Mode: single-image only")

    # Crop
    if crop_cfg.get("enabled", True):
        z, y, x = img_a.shape
        cy, cx = y // 2, x // 2
        size = crop_cfg.get("size", 512)
        half = size // 2
        img_a_crop = img_a[:, cy - half : cy + half, cx - half : cx + half]
        img_b_crop = (
            img_b[:, cy - half : cy + half, cx - half : cx + half]
            if img_b is not None
            else None
        )
        print(f"Cropped shape: {img_a_crop.shape}")
    else:
        img_a_crop = img_a
        img_b_crop = img_b

    # Spacing
    spacing_3d = (spacing_cfg["z"], spacing_cfg["y"], spacing_cfg["x"])
    spacing_xy = (spacing_cfg["y"], spacing_cfg["x"])
    spacing_xz = (spacing_cfg["z"], spacing_cfg["x"])

    # 2D slices
    mid_z = img_a_crop.shape[0] // 2
    mid_y = img_a_crop.shape[1] // 2
    xy_a = img_a_crop[mid_z]
    xz_a = img_a_crop[:, mid_y, :]
    xy_b = img_b_crop[mid_z] if img_b_crop is not None else None
    xz_b = img_b_crop[:, mid_y, :] if img_b_crop is not None else None

    # Move to GPU
    img_a_gpu = ascupy(img_a_crop)
    img_b_gpu = ascupy(img_b_crop) if img_b_crop is not None else None
    xy_a_gpu = ascupy(xy_a)
    xy_b_gpu = ascupy(xy_b) if xy_b is not None else None
    xz_a_gpu = ascupy(xz_a)
    xz_b_gpu = ascupy(xz_b) if xz_b is not None else None

    # Get method configs
    frc_method = method_cfg.get("frc", {})
    fsc_method = method_cfg.get("fsc", {})
    dcr_method = method_cfg.get("dcr", {})

    n_runs = timing_cfg.get("n_runs", 3)

    # ==================== COMPUTE ALL RESULTS ====================
    # Using published preprocessing conventions:
    #   FRC/FSC: Hamming window, zero padding
    #   DCR: Tukey window (alpha=0.1), no padding
    print("\nComputing resolution estimates...")
    print("  FRC: Hamming window, zero padding")
    print("  FSC: Hamming window, isotropic resampling")
    print("  DCR: Tukey window, no padding")
    results = []

    # 2D FRC XY (two-image)
    if methods_cfg.get("frc_2d_xy", True) and not single_image_only:
        res, t, mem = time_function(
            frc_resolution,
            xy_a_gpu,
            xy_b_gpu,
            spacing=spacing_xy,
            zero_padding=True,
            bin_delta=frc_method.get("bin_delta", 1),
            backend=frc_method.get("backend", "hist"),
            n_runs=n_runs,
        )
        results.append(("2D FRC XY (two-image)", res, None, t, mem))

    # 2D FRC XY (single-image)
    if methods_cfg.get("frc_2d_single", True):
        sub1, sub2 = checkerboard_split(xy_a_gpu)
        res, t, mem = time_function(
            frc_resolution,
            sub1,
            sub2,
            spacing=spacing_xy,
            zero_padding=True,
            bin_delta=frc_method.get("bin_delta", 1),
            backend=frc_method.get("backend", "hist"),
            n_runs=n_runs,
        )
        results.append(("2D FRC XY (single-image)", res, None, t, mem))

    # 2D FRC XZ (two-image)
    if methods_cfg.get("frc_2d_xz", True) and not single_image_only:
        res, t, mem = time_function(
            frc_resolution,
            xz_a_gpu,
            xz_b_gpu,
            spacing=spacing_xz,
            zero_padding=True,
            bin_delta=frc_method.get("bin_delta", 1),
            backend=frc_method.get("backend", "hist"),
            n_runs=n_runs,
        )
        results.append(("2D FRC XZ (two-image)", None, res, t, mem))

    # 2D DCR XY (uses internal Tukey window, no padding)
    if methods_cfg.get("dcr_2d_xy", True):
        res, t, mem = time_function(
            dcr_resolution,
            xy_a_gpu,
            spacing=spacing_xy,
            num_radii=dcr_method.get("num_radii", 100),
            num_highpass=dcr_method.get("num_highpass", 10),
            windowing=True,
            n_runs=n_runs,
        )
        results.append(("2D DCR XY", res, None, t, mem))

    # 2D DCR XY refined (two-pass refinement)
    if methods_cfg.get("dcr_2d_xy_refined", True):
        res, t, mem = time_function(
            dcr_resolution,
            xy_a_gpu,
            spacing=spacing_xy,
            num_radii=dcr_method.get("num_radii", 100),
            num_highpass=dcr_method.get("num_highpass", 10),
            windowing=True,
            refine=True,
            n_runs=n_runs,
        )
        results.append(("2D DCR XY (refined)", res, None, t, mem))

    # 2D DCR XZ (uses internal Tukey window, no padding)
    if methods_cfg.get("dcr_2d_xz", True):
        res, t, mem = time_function(
            dcr_resolution,
            xz_a_gpu,
            spacing=spacing_xz,
            num_radii=dcr_method.get("num_radii", 100),
            num_highpass=dcr_method.get("num_highpass", 10),
            windowing=True,
            n_runs=n_runs,
        )
        results.append(("2D DCR XZ", None, res, t, mem))

    # 2D DCR XZ refined (two-pass refinement)
    if methods_cfg.get("dcr_2d_xz_refined", True):
        res, t, mem = time_function(
            dcr_resolution,
            xz_a_gpu,
            spacing=spacing_xz,
            num_radii=dcr_method.get("num_radii", 100),
            num_highpass=dcr_method.get("num_highpass", 10),
            windowing=True,
            refine=True,
            n_runs=n_runs,
        )
        results.append(("2D DCR XZ (refined)", None, res, t, mem))

    # 3D DCR (uses internal Tukey window, no padding, no isotropic resampling)
    if methods_cfg.get("dcr_3d", True):
        res, t, mem = time_function(
            dcr_resolution,
            img_a_gpu,
            spacing=spacing_3d,
            num_radii=dcr_method.get("num_radii", 100),
            num_highpass=dcr_method.get("num_highpass", 10),
            windowing=True,
            n_runs=n_runs,
        )
        if isinstance(res, dict):
            results.append(("3D DCR", res.get("xy"), res.get("z"), t, mem))
        else:
            results.append(("3D DCR", res, res, t, mem))

    # 3D DCR refined (two-pass refinement)
    if methods_cfg.get("dcr_3d_refined", True):
        res, t, mem = time_function(
            dcr_resolution,
            img_a_gpu,
            spacing=spacing_3d,
            num_radii=dcr_method.get("num_radii", 100),
            num_highpass=dcr_method.get("num_highpass", 10),
            windowing=True,
            refine=True,
            n_runs=n_runs,
        )
        if isinstance(res, dict):
            results.append(("3D DCR (refined)", res.get("xy"), res.get("z"), t, mem))
        else:
            results.append(("3D DCR (refined)", res, res, t, mem))

    # 3D FSC hist (two-image, isotropic resampling, zero padding)
    if methods_cfg.get("fsc_3d_hist", True) and not single_image_only:
        res, t, mem = time_function(
            fsc_resolution,
            img_a_gpu,
            img_b_gpu,
            spacing=spacing_3d,
            resample_isotropic=True,
            zero_padding=True,
            backend="hist",
            resolution_threshold="one-bit",
            bin_delta=fsc_method.get("bin_delta", 10),
            angle_delta=fsc_method.get("angle_delta", 15),
            n_runs=n_runs,
        )
        results.append(("3D FSC hist (two-image)", res.get("xy"), res.get("z"), t, mem))

    # 3D FSC hist (single-image, isotropic resampling, zero padding)
    if methods_cfg.get("fsc_3d_hist_single", True):
        res, t, mem = time_function(
            fsc_resolution,
            img_a_gpu,
            spacing=spacing_3d,
            resample_isotropic=True,
            zero_padding=True,
            average=True,
            backend="hist",
            resolution_threshold="one-bit",
            bin_delta=fsc_method.get("bin_delta", 10),
            angle_delta=fsc_method.get("angle_delta", 15),
            n_runs=n_runs,
        )
        results.append(
            ("3D FSC hist (single-image)", res.get("xy"), res.get("z"), t, mem)
        )

    # 3D FSC mask
    if methods_cfg.get("fsc_3d_mask", False) and not single_image_only:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            res, t, mem = time_function(
                fsc_resolution,
                img_a_gpu,
                img_b_gpu,
                spacing=spacing_3d,
                resample_isotropic=True,
                zero_padding=True,
                backend="mask",
                bin_delta=fsc_method.get("bin_delta", 10),
                angle_delta=fsc_method.get("angle_delta", 15),
                n_runs=n_runs,
            )
            results.append(
                ("3D FSC mask (two-image)", res.get("xy"), res.get("z"), t, mem)
            )

    # 3D FSC mask single
    if methods_cfg.get("fsc_3d_mask_single", False):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            res, t, mem = time_function(
                fsc_resolution,
                img_a_gpu,
                spacing=spacing_3d,
                resample_isotropic=True,
                zero_padding=True,
                average=True,
                backend="mask",
                bin_delta=fsc_method.get("bin_delta", 10),
                angle_delta=fsc_method.get("angle_delta", 15),
                n_runs=n_runs,
            )
            results.append(
                ("3D FSC mask (single-image)", res.get("xy"), res.get("z"), t, mem)
            )

    # ==================== PRINT SUMMARY TABLE ====================
    def fmt(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "—"
        if isinstance(val, float) and np.isinf(val):
            return "inf"
        return f"{val:.1f}"

    def fmt_time(t):
        return f"{t:.0f} ms" if t < 1000 else f"{t / 1000:.1f} s"

    def fmt_mem(m):
        return f"{m:.0f} MB" if m > 0 else "—"

    print("\n" + "=" * 105)
    print("SUMMARY TABLE")
    print("=" * 105)
    print(f"{'Method':<35} {'XY (nm)':<12} {'Z (nm)':<12} {'Time':<12} {'GPU Mem':<12}")
    print("-" * 105)

    for name, xy, z, t, mem in results:
        if "XY" in name and "3D" not in name:
            print(
                f"{name:<35} {fmt(xy):<12} {'—':<12} {fmt_time(t):<12} {fmt_mem(mem):<12}"
            )
        elif "XZ" in name:
            print(
                f"{name:<35} {'—':<12} {fmt(z):<12} {fmt_time(t):<12} {fmt_mem(mem):<12}"
            )
        else:
            print(
                f"{name:<35} {fmt(xy):<12} {fmt(z):<12} {fmt_time(t):<12} {fmt_mem(mem):<12}"
            )

    print("-" * 105)
    print(
        f"{'Paper (Koho et al. 2019)':<35} {'~140':<12} {'~638':<12} {'—':<12} {'—':<12}"
    )
    print("=" * 105)

    # ==================== SAVE RESULTS TO CSV ====================
    csv_path = output_dir / "results.csv"
    crop_size = crop_cfg.get("size") if crop_cfg.get("enabled", True) else None
    save_results_csv(results, csv_path, data_name=data_name, crop_size=crop_size)
    print(f"\nResults saved to: {csv_path}")

    # ==================== GENERATE PLOTS ====================
    if plots_cfg.get("save", True):
        print("\nGenerating plots...")
        dpi = plots_cfg.get("dpi", 150)

        # Figure 1: FRC Curves
        if not single_image_only:
            fig1, axes = plt.subplots(1, 3, figsize=(14, 4))
            plot_frc_curves(xy_a, xy_b, spacing_xy, "2D FRC XY (two-image)", axes[0])
            plot_frc_curves(xz_a, xz_b, spacing_xz, "2D FRC XZ (two-image)", axes[1])
            sub1_np, sub2_np = checkerboard_split(xy_a)
            plot_frc_curves(
                sub1_np, sub2_np, spacing_xy, "2D FRC XY (single-image)", axes[2]
            )
            plt.tight_layout()
            fig1.savefig(output_dir / "benchmark_frc_curves.png", dpi=dpi)
            print(f"  Saved: {output_dir / 'benchmark_frc_curves.png'}")
        else:
            # Single-image only: show checkerboard FRC
            fig1, ax = plt.subplots(1, 1, figsize=(6, 4))
            sub1_np, sub2_np = checkerboard_split(xy_a)
            plot_frc_curves(
                sub1_np, sub2_np, spacing_xy, "2D FRC XY (single-image)", ax
            )
            plt.tight_layout()
            fig1.savefig(output_dir / "benchmark_frc_curves.png", dpi=dpi)
            print(f"  Saved: {output_dir / 'benchmark_frc_curves.png'}")

        # Figure 2: DCR Curves
        fig2, axes = plt.subplots(1, 3, figsize=(14, 4))
        plot_dcr_curve(xy_a, spacing_xy, "2D DCR XY", axes[0])
        plot_dcr_curve(xz_a, spacing_xz, "2D DCR XZ", axes[1])
        plot_dcr_curve(img_a_crop, spacing_3d, "3D DCR", axes[2])
        plt.tight_layout()
        fig2.savefig(output_dir / "benchmark_dcr_curves.png", dpi=dpi)
        print(f"  Saved: {output_dir / 'benchmark_dcr_curves.png'}")

        # Figure 3: 3D FSC Sectors
        spacing_iso = [spacing_3d[1]] * 3
        target_z = int(round(img_a_crop.shape[0] * spacing_3d[0] / spacing_3d[1]))
        if target_z % 2 != 0:
            target_z -= 1

        img_a_iso = rescale_isotropic(
            img_a_crop,
            spacing_3d,
            downscale_xy=False,
            order=1,
            preserve_range=True,
            target_z_size=target_z,
        )
        img_a_proc = pad_image_to_cube(img_a_iso)

        fsc_bin_delta = fsc_method.get("bin_delta", 1)
        fsc_angle_delta = fsc_method.get("angle_delta", 15)

        if not single_image_only:
            img_b_iso = rescale_isotropic(
                img_b_crop,
                spacing_3d,
                downscale_xy=False,
                order=1,
                preserve_range=True,
                target_z_size=target_z,
            )
            img_b_proc = pad_image_to_cube(img_b_iso)

            fsc_data = _calculate_fsc_sectioned_hist(
                img_a_proc,
                img_b_proc,
                bin_delta=fsc_bin_delta,
                angle_delta=fsc_angle_delta,
                spacing=spacing_iso,
                exclude_axis_angle=0.0,
                use_max_nyquist=False,
            )
        else:
            # Single-image mode: use checkerboard split
            sub1, sub2 = checkerboard_split(img_a_proc)
            fsc_data = _calculate_fsc_sectioned_hist(
                sub1,
                sub2,
                bin_delta=fsc_bin_delta,
                angle_delta=fsc_angle_delta,
                spacing=spacing_iso,
                exclude_axis_angle=0.0,
                use_max_nyquist=False,
            )

        fig3 = plt.figure(figsize=(14, 5))
        ax1 = fig3.add_subplot(131)
        ax2 = fig3.add_subplot(132)
        ax3 = fig3.add_subplot(133, projection="polar")

        fsc_title = (
            "3D FSC (single-image)" if single_image_only else "3D FSC hist (two-image)"
        )
        plot_fsc_sectors(fsc_data, spacing_3d[1], spacing_3d[0], fsc_title, [ax1, ax2])
        plot_fsc_polar(fsc_data, spacing_3d[1], ax3)

        plt.suptitle(
            f"3D FSC Sectored Analysis ({'single-image' if single_image_only else 'two-image'})",
            fontsize=12,
        )
        plt.tight_layout()
        fig3.savefig(output_dir / "benchmark_fsc_sectors.png", dpi=dpi)
        print(f"  Saved: {output_dir / 'benchmark_fsc_sectors.png'}")

        # Figure 3b: FSC Diagnostic (per-sector curves with both thresholds)
        diag_path = output_dir / "benchmark_fsc_diagnostic.png"
        plot_fsc_diagnostic(fsc_data, spacing_3d[1], spacing_3d[0], diag_path, dpi=dpi)
        print(f"  Saved: {diag_path}")

        # Figure 4: Comparison Summary
        fig4, axes = plt.subplots(1, 2, figsize=(14, 5))
        plot_comparison_bars(results, axes[0])
        plot_timing_bars(results, axes[1])
        plt.tight_layout()
        fig4.savefig(output_dir / "benchmark_comparison.png", dpi=dpi)
        print(f"  Saved: {output_dir / 'benchmark_comparison.png'}")

        # Figure 5: All-in-one summary
        fig5 = plt.figure(figsize=(16, 12))

        if not single_image_only:
            ax = fig5.add_subplot(3, 4, 1)
            plot_frc_curves(xy_a, xy_b, spacing_xy, "FRC XY", ax)
            ax = fig5.add_subplot(3, 4, 2)
            plot_frc_curves(xz_a, xz_b, spacing_xz, "FRC XZ", ax)
        else:
            # Single-image: show checkerboard FRC
            ax = fig5.add_subplot(3, 4, 1)
            sub1_np, sub2_np = checkerboard_split(xy_a)
            plot_frc_curves(sub1_np, sub2_np, spacing_xy, "FRC XY (single)", ax)
            ax = fig5.add_subplot(3, 4, 2)
            sub1_xz, sub2_xz = checkerboard_split(xz_a)
            plot_frc_curves(sub1_xz, sub2_xz, spacing_xz, "FRC XZ (single)", ax)

        ax = fig5.add_subplot(3, 4, 3)
        plot_dcr_curve(xy_a, spacing_xy, "DCR XY", ax)
        ax = fig5.add_subplot(3, 4, 4)
        plot_dcr_curve(img_a_crop, spacing_3d, "DCR 3D", ax)

        ax1 = fig5.add_subplot(3, 4, 5)
        ax2 = fig5.add_subplot(3, 4, 6)
        plot_fsc_sectors(fsc_data, spacing_3d[1], spacing_3d[0], "FSC 3D", [ax1, ax2])

        ax = fig5.add_subplot(3, 4, 7, projection="polar")
        plot_fsc_polar(fsc_data, spacing_3d[1], ax)

        ax = fig5.add_subplot(3, 2, 5)
        plot_comparison_bars(results, ax)
        ax = fig5.add_subplot(3, 2, 6)
        plot_timing_bars(results, ax)

        mode_str = " (single-image)" if single_image_only else ""
        plt.suptitle(
            f"Resolution Estimation Methods Benchmark{mode_str}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()
        fig5.savefig(output_dir / "benchmark_summary.png", dpi=dpi)
        print(f"  Saved: {output_dir / 'benchmark_summary.png'}")

        plt.close("all")

    print("\nDone!")
    return results


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Hydra entry point."""
    run_benchmark(
        output_dir=cfg.output_dir,
        data_dir=cfg.get("data_dir"),
        spacing_cfg=OmegaConf.to_container(cfg.spacing),
        crop_cfg=OmegaConf.to_container(cfg.crop),
        method_cfg=OmegaConf.to_container(cfg.method),
        methods_cfg=OmegaConf.to_container(cfg.methods),
        timing_cfg=OmegaConf.to_container(cfg.timing),
        plots_cfg=OmegaConf.to_container(cfg.plots),
        image_path=cfg.get("image_path"),
        single_image_only=cfg.get("single_image_only", False),
        data_name=cfg.get("data_name", ""),
    )


if __name__ == "__main__":
    main()
