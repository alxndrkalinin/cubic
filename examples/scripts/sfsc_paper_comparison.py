#!/usr/bin/env python3
"""
SFSC Paper Comparison: Validate our implementation against Koho et al. 2019.

This script systematically compares our sectored Fourier Shell Correlation (SFSC)
implementation with the methodology described in:

    Koho et al. (2019) "Fourier ring correlation simplifies image restoration
    in fluorescence microscopy", Nature Communications 10:3103.

Key investigation points:
1. Extract directional resolution at all angles (0°-90° in 15° steps)
2. Compare with/without k(θ) correction (paper's equation 5)
3. Visualize correlation curves at each angle
4. Generate polar plots matching Figure 4b format

Paper's k(θ) correction (equation 5):
    "If no such correction is made, all the numerical resolution values
    calculated with FPC/FRC/SFSC at orientations θ ≠ 0 + nπ will be
    unrealistically high [i.e., better-looking smaller values]"

    k(θ) = 1 + (z-1) × |sin(θ)|

    where z = spacing_z / spacing_xy (anisotropy factor)

Usage:
    python sfsc_paper_comparison.py                           # Default settings
    python sfsc_paper_comparison.py image_path=/path/to.tif   # Custom image
    python sfsc_paper_comparison.py spacing=[0.250,0.0777,0.0777]  # Custom spacing

Author: Claude Code / Alex Kalinin
"""

import sys
from typing import Sequence
from pathlib import Path

import hydra
import numpy as np
import matplotlib
from omegaconf import OmegaConf, DictConfig

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes

# Add parent directory to path for cubic imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cubic.cuda import ascupy, asnumpy, get_device
from cubic.image_utils import hamming_window, rescale_isotropic, checkerboard_split
from cubic.metrics.spectral import dcr_resolution, fsc_resolution
from cubic.metrics.spectral.frc import preprocess_images, _calculate_fsc_sectioned_hist
from cubic.metrics.spectral.analysis import (
    FourierCorrelationData,
    FourierCorrelationAnalysis,
    FourierCorrelationDataCollection,
    fit_frc_curve,
    calculate_resolution_threshold_curve,
)


def apply_tukey_window(image: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Apply Tukey (tapered cosine) window to image."""
    from scipy.signal import windows

    result = image.copy().astype(np.float32)
    for axis, axis_size in enumerate(image.shape):
        filter_shape = [1] * image.ndim
        filter_shape[axis] = axis_size
        window = windows.tukey(axis_size, alpha=alpha).reshape(filter_shape)
        np.power(window, 1.0 / image.ndim, out=window)
        result *= window
    return result


def load_image(image_path: str | Path) -> np.ndarray:
    """Load 3D image from various formats."""
    from skimage import io

    path = Path(image_path)

    if path.suffix.lower() == ".nd2":
        try:
            import nd2

            with nd2.ND2File(path) as f:
                img = f.asarray()
                sizes = f.sizes
                print(f"  ND2 dimensions: {sizes}")

                # Handle based on actual dimension order
                # Common orders: (Z, C, Y, X), (T, Z, C, Y, X), (C, Z, Y, X)
                if img.ndim == 4 and "C" in sizes and "Z" in sizes:
                    # Find channel axis and take first channel
                    dim_order = list(sizes.keys())
                    c_axis = dim_order.index("C")
                    img = np.take(img, 0, axis=c_axis)
                    print(f"  Selected first channel, shape: {img.shape}")
                elif img.ndim == 4:
                    # Fallback: assume first axis is extra, take first
                    img = img[0]
                    print(f"  Took first slice of first axis, shape: {img.shape}")

                return img.astype(np.float32)
        except ImportError:
            print("nd2 package not available, trying skimage.io")

    img = io.imread(path)
    return img.astype(np.float32)


def extract_directional_resolutions(
    fsc_data: dict[int, FourierCorrelationData],
    spacing: float,
    z_factor: float,
    apply_k_correction: bool = False,
    angle_delta: int = 15,
) -> dict[int, dict]:
    """
    Extract resolution at each angular sector.

    Parameters
    ----------
    fsc_data : dict
        FSC data from _calculate_fsc_sectioned_hist(), keyed by polar angle
    spacing : float
        Physical spacing (isotropic after resampling)
    z_factor : float
        Original anisotropy factor (spacing_z / spacing_xy)
    apply_k_correction : bool
        If True, apply paper's k(θ) correction
    angle_delta : int
        Angular bin width in degrees. Used to determine bin edges for k(θ)
        correction at boundary sectors (Z and XY).

    Returns
    -------
    dict mapping angle -> {resolution, freq_at_crossing, correlation_curve, ...}
    """
    results = {}

    for angle, data in fsc_data.items():
        freq = np.asarray(data.correlation["frequency"])
        corr = np.asarray(data.correlation["correlation"])
        n_points = np.asarray(data.correlation["points-x-bin"])

        # Create data collection for analysis
        data_collection = FourierCorrelationDataCollection()
        data_set = FourierCorrelationData()
        data_set.correlation["correlation"] = corr
        data_set.correlation["frequency"] = freq
        data_set.correlation["points-x-bin"] = n_points
        data_collection[0] = data_set

        # Run analysis
        analyzer = FourierCorrelationAnalysis(
            data_collection,
            spacing,
            resolution_threshold="one-bit",
            curve_fit_type="spline",
        )

        try:
            analyzed = analyzer.execute()[0]
            raw_resolution = analyzed.resolution["resolution"]
            freq_at_crossing = analyzed.resolution["resolution-point"][1]
            threshold = analyzed.resolution["threshold"]
        except Exception as e:
            raw_resolution = np.nan
            freq_at_crossing = np.nan
            threshold = None

        # Apply k(θ) correction if requested
        # Paper's equation (5): k(θ) = 1 + (z-1) × |sin(θ)|
        # Paper defines θ from XY plane, but our convention is polar angle from Z axis
        # So we use cos(θ) instead of sin(θ):
        #   - θ=0° (Z): cos(0)=1 → k=z_factor (maximum correction)
        #   - θ=90° (XY): cos(90)=0 → k=1 (no correction)
        #
        # Use bin edges for boundary sectors to get exact correction:
        #   - Z sector (lowest angle bin): use 0° for maximum correction
        #   - XY sector (highest angle bin): use 90° for no correction
        #   - Intermediate sectors: use bin center (reasonable approximation)
        if apply_k_correction and not np.isnan(raw_resolution):
            all_angles = sorted(fsc_data.keys())

            if angle == min(all_angles):
                # Z sector: use lower bin edge (0°) for maximum correction
                angle_for_correction = 0.0
            elif angle == max(all_angles):
                # XY sector: use upper bin edge (90°) for no correction
                angle_for_correction = 90.0
            else:
                # Intermediate: use bin center
                angle_for_correction = float(angle)

            k_theta = 1 + (z_factor - 1) * np.abs(
                np.cos(np.radians(angle_for_correction))
            )
            corrected_resolution = raw_resolution * k_theta
        else:
            k_theta = 1.0
            corrected_resolution = raw_resolution

        results[angle] = {
            "angle": angle,
            "raw_resolution": raw_resolution,
            "k_theta": k_theta,
            "corrected_resolution": corrected_resolution,
            "freq_at_crossing": freq_at_crossing,
            "frequency": freq,
            "correlation": corr,
            "n_points": n_points,
            "threshold": threshold,
        }

    return results


def plot_correlation_curves(
    results: dict[int, dict],
    output_path: Path,
    title: str = "SFSC Correlation Curves by Polar Angle",
):
    """
    Plot correlation curves for all angular sectors.

    Similar to Figure 4a in Koho et al. 2019.
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Sort angles for consistent plotting
    angles = sorted(results.keys())

    # Color map from Z (blue) to XY (red)
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(0, 90)

    for i, angle in enumerate(angles[:6]):  # Plot up to 6 angles
        ax = axes[i]
        data = results[angle]

        freq = data["frequency"]
        corr = data["correlation"]
        threshold = data["threshold"]

        color = cmap(norm(angle))

        # Plot correlation curve
        ax.plot(freq, corr, "-", color=color, linewidth=2, label="FSC")

        # Plot threshold
        if threshold is not None:
            ax.plot(freq, threshold, "--", color="gray", linewidth=1, label="One-bit")

        # Mark crossing point
        if not np.isnan(data["freq_at_crossing"]):
            ax.axvline(
                data["freq_at_crossing"],
                color="red",
                linestyle=":",
                alpha=0.7,
                label=f"Res={data['raw_resolution']:.3f}μm",
            )

        ax.set_xlabel("Spatial Frequency (normalized)")
        ax.set_ylabel("FSC")
        ax.set_title(f"θ={angle}° (k={data['k_theta']:.2f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(angles), 6):
        axes[i].set_visible(False)

    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation curves to {output_path}")


def plot_polar_resolution(
    results_no_correction: dict[int, dict],
    results_with_correction: dict[int, dict] | None,
    output_path: Path,
    title: str = "SFSC Directional Resolution",
):
    """
    Create polar plot of resolution vs angle.

    Similar to Figure 4b in Koho et al. 2019.
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="polar")

    angles = sorted(results_no_correction.keys())

    # Convert to radians and extend to full circle (mirror about 90°)
    theta_half = np.array([np.radians(a) for a in angles])
    theta_full = np.concatenate([theta_half, np.pi - theta_half[::-1]])

    # Extract resolutions (no correction)
    res_no_corr = np.array([results_no_correction[a]["raw_resolution"] for a in angles])
    res_no_corr_full = np.concatenate([res_no_corr, res_no_corr[::-1]])

    # Plot without correction
    ax.plot(
        theta_full,
        res_no_corr_full,
        "b-o",
        linewidth=2,
        markersize=6,
        label="No k(θ) correction",
    )

    # Plot with correction if available
    if results_with_correction is not None:
        res_with_corr = np.array(
            [results_with_correction[a]["corrected_resolution"] for a in angles]
        )
        res_with_corr_full = np.concatenate([res_with_corr, res_with_corr[::-1]])
        ax.plot(
            theta_full,
            res_with_corr_full,
            "r-s",
            linewidth=2,
            markersize=6,
            label="With k(θ) correction",
        )

    # Customize polar plot
    ax.set_theta_zero_location("N")  # 0° at top (Z axis)
    ax.set_theta_direction(-1)  # Clockwise
    ax.set_thetamin(0)
    ax.set_thetamax(180)

    # Add angle labels
    ax.set_xticks(np.radians([0, 30, 60, 90, 120, 150, 180]))
    ax.set_xticklabels(["0° (Z)", "30°", "60°", "90° (XY)", "120°", "150°", "180°"])

    ax.set_ylabel("Resolution (μm)", labelpad=30)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0))
    ax.set_title(title, pad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved polar plot to {output_path}")


def plot_comparison_table(
    results_no_correction: dict[int, dict],
    results_with_correction: dict[int, dict],
    output_path: Path,
    paper_values: dict | None = None,
):
    """Create comparison table as figure."""
    angles = sorted(results_no_correction.keys())

    # Prepare data
    data = []
    for angle in angles:
        row = {
            "Angle (°)": angle,
            "Raw Res (μm)": results_no_correction[angle]["raw_resolution"],
            "k(θ)": results_with_correction[angle]["k_theta"],
            "Corrected Res (μm)": results_with_correction[angle][
                "corrected_resolution"
            ],
        }
        if paper_values and angle in paper_values:
            row["Paper Res (μm)"] = paper_values[angle]
        data.append(row)

    # Create figure with table
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    columns = list(data[0].keys())
    cell_text = [
        [
            f"{row[col]:.3f}" if isinstance(row[col], float) else str(row[col])
            for col in columns
        ]
        for row in data
    ]

    table = ax.table(
        cellText=cell_text,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color header
    for i, col in enumerate(columns):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(color="white", weight="bold")

    plt.title(
        "SFSC Resolution Comparison: With/Without k(θ) Correction", fontsize=12, pad=20
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison table to {output_path}")


def debug_resolution_calculation(
    results: dict[int, dict],
    spacing: float,
    paper_xy: float = 0.5987,
    paper_z: float = 3.8876,
):
    """
    Print detailed debug output to identify source of 2x discrepancy with paper.

    This function prints intermediate values for XY and Z sectors to help
    identify exactly where the 2x factor enters the resolution calculation.

    Paper reference values (Koho et al. 2019, 40x_TAGoff_z_galvo.nd2):
    - XY resolution: 0.5987 μm
    - Z resolution: 3.8876 μm
    """
    print("\n" + "=" * 80)
    print("DEBUG: RESOLUTION CALCULATION INVESTIGATION")
    print("=" * 80)
    print(f"Isotropic spacing: {spacing} μm")
    print(f"Paper reference - XY: {paper_xy} μm, Z: {paper_z} μm")
    print()

    # Find XY (highest angle, ~82°) and Z (lowest angle, ~8°) sectors
    angles = sorted(results.keys())
    angle_xy = max(angles)  # ~82° for angle_delta=15
    angle_z = min(angles)  # ~8° for angle_delta=15

    for label, angle in [("XY", angle_xy), ("Z", angle_z)]:
        data = results[angle]
        paper_val = paper_xy if label == "XY" else paper_z

        print(f"\n{'=' * 40}")
        print(f"{label} SECTOR (θ={angle}°)")
        print(f"{'=' * 40}")

        freq = np.asarray(data["frequency"])
        corr = np.asarray(data["correlation"])
        n_pts = np.asarray(data["n_points"])
        threshold = data["threshold"]
        raw_resolution = data["raw_resolution"]
        freq_at_crossing = data["freq_at_crossing"]

        print(f"\n1. FSC CURVE DATA (first 10 bins):")
        print(
            f"   {'Bin':<5} {'Freq':<10} {'FSC':<10} {'Threshold':<10} {'Points':<10}"
        )
        print(f"   {'-' * 45}")
        thr_vals = threshold if threshold is not None else [np.nan] * len(freq)
        for i in range(min(10, len(freq))):
            print(
                f"   {i:<5} {freq[i]:<10.4f} {corr[i]:<10.4f} {thr_vals[i]:<10.4f} {n_pts[i]:<10.0f}"
            )

        print(f"\n2. THRESHOLD CROSSING:")
        print(f"   Raw resolution (no k(θ)): {raw_resolution:.4f} μm")
        print(f"   Frequency at crossing: {freq_at_crossing:.4f} (normalized)")
        print(f"   k(θ) factor: {data['k_theta']:.4f}")
        print(f"   Corrected resolution: {data['corrected_resolution']:.4f} μm")

        print(f"\n3. RESOLUTION FORMULA CHECK:")
        print(f"   resolution = 2 × spacing / freq_at_crossing")
        if not np.isnan(freq_at_crossing) and freq_at_crossing > 0:
            calculated_res = 2 * spacing / freq_at_crossing
            print(f"   = 2 × {spacing} / {freq_at_crossing:.4f}")
            print(f"   = {calculated_res:.4f} μm")
        else:
            print(f"   (Cannot calculate - invalid freq_at_crossing)")

        print(f"\n4. COMPARISON WITH PAPER:")
        print(f"   Our result: {raw_resolution:.4f} μm")
        print(f"   Paper result: {paper_val:.4f} μm")
        if not np.isnan(raw_resolution) and raw_resolution > 0:
            ratio = paper_val / raw_resolution
            print(f"   Ratio (paper/ours): {ratio:.2f}x")

            # Compute what frequency would give paper's resolution
            expected_freq = 2 * spacing / paper_val
            print(f"\n5. EXPECTED FREQUENCY FOR PAPER RESULT:")
            print(f"   If resolution = {paper_val} μm, then:")
            print(f"   freq = 2 × spacing / resolution")
            print(f"       = 2 × {spacing} / {paper_val}")
            print(f"       = {expected_freq:.4f} (normalized)")
            print(f"   Our freq: {freq_at_crossing:.4f}")
            print(f"   Freq ratio: {freq_at_crossing / expected_freq:.2f}x")

    # Print the full correlation curves for detailed inspection
    print("\n" + "=" * 80)
    print("FULL CORRELATION CURVES (for XY and Z sectors)")
    print("=" * 80)

    for label, angle in [("XY", angle_xy), ("Z", angle_z)]:
        data = results[angle]
        freq = np.asarray(data["frequency"])
        corr = np.asarray(data["correlation"])
        threshold = data["threshold"]

        print(f"\n{label} Sector (θ={angle}°):")
        print(f"{'Freq':<10} {'FSC':<10} {'Threshold':<10} {'FSC-Thr':<10}")
        print("-" * 40)
        thr_vals = threshold if threshold is not None else [np.nan] * len(freq)
        for i in range(len(freq)):
            diff = corr[i] - thr_vals[i] if threshold is not None else np.nan
            # Mark the crossing point
            marker = (
                " <-- CROSSING"
                if i > 0 and diff < 0 and corr[i - 1] >= thr_vals[i - 1]
                else ""
            )
            print(
                f"{freq[i]:<10.4f} {corr[i]:<10.4f} {thr_vals[i]:<10.4f} {diff:<10.4f}{marker}"
            )


def diagnose_correlation_curves(results: dict[int, dict]):
    """
    Diagnose why correlation curves might not cross the threshold.

    Helps understand issues like Z resolution returning NaN.
    """
    print("\n" + "-" * 60)
    print("CORRELATION CURVE DIAGNOSTICS")
    print("-" * 60)

    for angle in sorted(results.keys()):
        data = results[angle]
        corr = data["correlation"]
        freq = data["frequency"]
        n_points = data["n_points"]
        threshold = data["threshold"]

        # Basic stats
        min_corr = np.nanmin(corr)
        max_corr = np.nanmax(corr)
        mean_corr = np.nanmean(corr)

        # Check if crosses typical thresholds
        crosses_half = np.any(corr < 0.5)
        crosses_143 = np.any(corr < 0.143)

        # Find approximate threshold if available
        if threshold is not None:
            mean_threshold = np.nanmean(threshold)
            crosses_threshold = np.any(corr < threshold)
        else:
            mean_threshold = np.nan
            crosses_threshold = False

        # Total points in this sector
        total_points = np.sum(n_points)

        status = "OK" if not np.isnan(data["raw_resolution"]) else "NO CROSSING"

        print(f"\nAngle θ={angle}° ({status}):")
        print(
            f"  Correlation range: [{min_corr:.3f}, {max_corr:.3f}], mean={mean_corr:.3f}"
        )
        print(f"  Total frequency bins: {len(corr)}, total points: {total_points:.0f}")
        print(f"  Mean one-bit threshold: {mean_threshold:.3f}")
        print(
            f"  Crosses 0.5: {crosses_half}, Crosses 0.143: {crosses_143}, Crosses threshold: {crosses_threshold}"
        )

        if not crosses_threshold and threshold is not None:
            # Find where correlation is closest to threshold
            diff = corr - threshold
            closest_idx = np.argmin(np.abs(diff))
            print(
                f"  Closest approach: corr={corr[closest_idx]:.3f} vs threshold={threshold[closest_idx]:.3f} at freq={freq[closest_idx]:.3f}"
            )


def export_fsc_curves_to_csv(
    results: dict[int, dict],
    output_dir: Path,
    prefix: str = "fsc_curve",
):
    """
    Export FSC curves to CSV files for detailed external analysis.

    Parameters
    ----------
    results : dict
        Results from extract_directional_resolutions
    output_dir : Path
        Directory to save CSV files
    prefix : str
        Prefix for CSV filenames
    """
    import csv

    for angle, data in results.items():
        freq = np.asarray(data["frequency"])
        corr = np.asarray(data["correlation"])
        n_points = np.asarray(data["n_points"])
        threshold = data["threshold"]

        csv_path = output_dir / f"{prefix}_{angle}deg.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["frequency", "correlation", "threshold", "n_points"])
            thr_vals = threshold if threshold is not None else [np.nan] * len(freq)
            for i in range(len(freq)):
                writer.writerow([freq[i], corr[i], thr_vals[i], n_points[i]])

        print(f"  Exported FSC curve for θ={angle}° to {csv_path}")

    # Also export a summary file with resolutions
    summary_path = output_dir / f"{prefix}_summary.csv"
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "angle",
                "raw_resolution",
                "k_theta",
                "corrected_resolution",
                "freq_at_crossing",
            ]
        )
        for angle in sorted(results.keys()):
            data = results[angle]
            writer.writerow(
                [
                    angle,
                    data["raw_resolution"],
                    data["k_theta"],
                    data["corrected_resolution"],
                    data["freq_at_crossing"],
                ]
            )
    print(f"  Exported resolution summary to {summary_path}")


def print_summary(
    results_no_correction: dict[int, dict],
    results_with_correction: dict[int, dict],
    z_factor: float,
):
    """Print summary of directional resolutions."""
    print("\n" + "=" * 80)
    print("SFSC DIRECTIONAL RESOLUTION SUMMARY")
    print("=" * 80)
    print(f"Anisotropy factor (z = spacing_z/spacing_xy): {z_factor:.3f}")
    print()

    print(
        f"{'Angle (°)':<12} {'Raw Res (μm)':<15} {'k(θ)':<10} {'Corrected Res (μm)':<20}"
    )
    print("-" * 60)

    angles = sorted(results_no_correction.keys())
    for angle in angles:
        raw = results_no_correction[angle]["raw_resolution"]
        k_theta = results_with_correction[angle]["k_theta"]
        corrected = results_with_correction[angle]["corrected_resolution"]
        print(f"{angle:<12} {raw:<15.4f} {k_theta:<10.3f} {corrected:<20.4f}")

    print("-" * 60)

    # Extract XY and Z resolutions
    angle_z = min(angles)  # Closest to 0° = Z
    angle_xy = max(angles)  # Closest to 90° = XY

    print(f"\nKey results:")
    print(
        f"  XY resolution (θ={angle_xy}°): {results_with_correction[angle_xy]['corrected_resolution']:.4f} μm"
    )
    print(
        f"  Z resolution (θ={angle_z}°):  {results_with_correction[angle_z]['corrected_resolution']:.4f} μm"
    )
    print(
        f"  Z/XY ratio: {results_with_correction[angle_z]['corrected_resolution'] / results_with_correction[angle_xy]['corrected_resolution']:.2f}"
    )
    print()

    # Note about paper
    print("Paper reference (Koho et al. 2019, Figure 4b, pollen data):")
    print("  XY resolution (θ=90°): 0.6 μm")
    print("  Z resolution (θ=0°):   3.9 μm")
    print("  Z/XY ratio: 6.5")
    print("=" * 80)


@hydra.main(version_base=None, config_path=".", config_name="sfsc_paper_comparison")
def main(cfg: DictConfig):
    """Run SFSC paper comparison analysis."""
    print(OmegaConf.to_yaml(cfg))

    # Extract config
    image_path = cfg.get("image_path", None)
    spacing = list(cfg.get("spacing", [0.250, 0.0777, 0.0777]))  # [Z, Y, X] in μm
    angle_delta = cfg.get("angle_delta", 15)
    bin_delta = cfg.get("bin_delta", 1)  # miplib uses d_bin=1
    window_type = cfg.get("window_type", "hamming")  # "hamming" or "tukey"
    resample_order = cfg.get(
        "resample_order", 1
    )  # 0=nearest (paper), 1=linear (default)
    crop_to_cube = cfg.get("crop_to_cube", False)  # miplib crops to cube
    crop_size = cfg.get("crop_size", 500)  # miplib uses 500^3
    export_curves = cfg.get("export_curves", True)  # Export FSC curves to CSV
    output_dir = Path(cfg.get("output_dir", "."))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    if image_path is None:
        # Default: use example data
        default_path = Path(__file__).parent.parent / "data" / "40x_TAGoff_z_galvo.nd2"
        if default_path.exists():
            image_path = default_path
        else:
            raise ValueError("No image_path provided and no default data found")

    print(f"Loading image from: {image_path}")
    image = load_image(image_path)
    print(f"Image shape: {image.shape}")
    print(f"Spacing (Z, Y, X): {spacing} μm")

    # Transfer to GPU for acceleration
    print("\nTransferring image to GPU...")
    image = ascupy(image)
    print(f"Image device: {get_device(image)}")

    # Calculate anisotropy factor
    spacing_z = spacing[0]
    spacing_xy = spacing[1]  # Assume Y == X
    z_factor = spacing_z / spacing_xy
    print(f"Anisotropy factor (z): {z_factor:.3f}")

    # Resample to isotropic voxels (following Koho et al. 2019)
    # miplib paper uses order=0 (nearest neighbor), our default is order=1 (linear)
    print(f"\nResampling to isotropic voxels (order={resample_order})...")
    iso_spacing = spacing_xy
    target_z_size = int(round(image.shape[0] * spacing_z / iso_spacing))
    if target_z_size % 2 != 0:
        target_z_size -= 1  # Ensure even for checkerboard split

    image_iso = rescale_isotropic(
        image,
        spacing,
        downscale_xy=False,
        order=resample_order,
        preserve_range=True,
        target_z_size=target_z_size,
    ).astype(np.float32)
    print(f"Isotropic shape: {image_iso.shape}")
    print(f"Isotropic spacing: {iso_spacing} μm")

    # Ensure even dimensions for checkerboard split
    even_shape = tuple(s - (s % 2) for s in image_iso.shape)
    if image_iso.shape != even_shape:
        slices = tuple(slice(0, es) for es in even_shape)
        image_iso = image_iso[slices]
        print(f"Cropped to even shape: {image_iso.shape}")

    # Crop to cube if requested (miplib crops to 500^3 after resampling)
    if crop_to_cube:
        min_dim = min(image_iso.shape)
        actual_crop_size = min(min_dim, crop_size)
        # Ensure even size
        actual_crop_size = actual_crop_size - (actual_crop_size % 2)
        center = [s // 2 for s in image_iso.shape]
        half = actual_crop_size // 2
        slices = tuple(slice(c - half, c + half) for c in center)
        image_iso = image_iso[slices]
        print(f"Cropped to cube: {image_iso.shape} (requested: {crop_size}^3)")

    # Preprocess: apply window and split
    # Paper uses Hamming window; we support both Hamming and Tukey for comparison
    print(f"\nPreprocessing images ({window_type} window, checkerboard split)...")

    if window_type.lower() == "hamming":
        # Use preprocess_images with Hamming window (matching paper)
        image1, image2 = preprocess_images(
            image_iso,
            None,  # Single image mode
            zero_padding=False,
            disable_hamming=False,  # Use Hamming window
            disable_3d_sum=False,
        )
    elif window_type.lower() == "tukey":
        # Apply Tukey window manually, then split
        image_windowed = apply_tukey_window(image_iso, alpha=0.1)
        image1, image2 = checkerboard_split(
            image_windowed,
            disable_3d_sum=False,
            preserve_range=False,
        )
    else:
        raise ValueError(
            f"Unknown window_type: {window_type}. Use 'hamming' or 'tukey'"
        )

    # Calculate sectioned FSC at all angles
    print(f"\nCalculating SFSC with angle_delta={angle_delta}°...")
    spacing_iso = [iso_spacing] * 3  # Isotropic spacing

    fsc_data = _calculate_fsc_sectioned_hist(
        image1,
        image2,
        bin_delta=bin_delta,
        angle_delta=angle_delta,
        spacing=spacing_iso,
        exclude_axis_angle=0.0,  # No axis exclusion for full comparison
        use_max_nyquist=False,
    )

    print(f"Got FSC data for angles: {sorted(fsc_data.keys())}")

    # Extract directional resolutions WITHOUT k(θ) correction
    print("\nExtracting resolutions without k(θ) correction...")
    results_no_correction = extract_directional_resolutions(
        fsc_data,
        spacing=iso_spacing,
        z_factor=z_factor,
        apply_k_correction=False,
        angle_delta=angle_delta,
    )

    # Extract directional resolutions WITH k(θ) correction
    print("Extracting resolutions with k(θ) correction...")
    results_with_correction = extract_directional_resolutions(
        fsc_data,
        spacing=iso_spacing,
        z_factor=z_factor,
        apply_k_correction=True,
        angle_delta=angle_delta,
    )

    # Print diagnostics to understand correlation behavior
    diagnose_correlation_curves(results_no_correction)

    # Print detailed debug output for 2x discrepancy investigation
    debug_resolution_calculation(
        results_with_correction,
        spacing=iso_spacing,
        paper_xy=0.5987,  # Paper's XY resolution for this image
        paper_z=3.8876,  # Paper's Z resolution for this image
    )

    # Export FSC curves to CSV for detailed external analysis
    if export_curves:
        print("\nExporting FSC curves to CSV...")
        export_fsc_curves_to_csv(
            results_with_correction,
            output_dir,
            prefix="fsc_curve_with_correction",
        )
        export_fsc_curves_to_csv(
            results_no_correction,
            output_dir,
            prefix="fsc_curve_no_correction",
        )

    # Print summary
    print_summary(results_no_correction, results_with_correction, z_factor)

    # Generate plots
    print("\nGenerating plots...")

    # 1. Correlation curves at each angle
    plot_correlation_curves(
        results_no_correction,
        output_dir / "sfsc_correlation_curves.png",
        title="SFSC Correlation Curves by Polar Angle",
    )

    # 2. Polar plot of resolution vs angle
    plot_polar_resolution(
        results_no_correction,
        results_with_correction,
        output_dir / "sfsc_polar_resolution.png",
        title="SFSC Directional Resolution (Polar Plot)",
    )

    # 3. Comparison table
    plot_comparison_table(
        results_no_correction,
        results_with_correction,
        output_dir / "sfsc_comparison_table.png",
    )

    # Also run the standard fsc_resolution for comparison
    # Test both resample orders to identify impact on resolution
    print("\nRunning standard fsc_resolution for comparison...")
    print(f"(Paper reference: XY=0.5987μm, Z=3.8876μm, using bin_delta={bin_delta})")

    # With current config's resample_order
    try:
        fsc_result = fsc_resolution(
            image,
            spacing=spacing,
            resample_isotropic=True,
            resample_order=resample_order,
            angle_delta=angle_delta,
            bin_delta=bin_delta,
        )
        print(
            f"  order={resample_order}, bin_delta={bin_delta}: XY={fsc_result['xy']:.4f}μm, Z={fsc_result['z']:.4f}μm"
        )
    except Exception as e:
        print(f"  order={resample_order} failed: {e}")

    # With order=0 (nearest neighbor, matching paper's miplib)
    if resample_order != 0:
        try:
            fsc_result_nn = fsc_resolution(
                image,
                spacing=spacing,
                resample_isotropic=True,
                resample_order=0,  # Nearest neighbor (paper's method)
                angle_delta=angle_delta,
                bin_delta=bin_delta,
            )
            print(
                f"  order=0 (paper), bin_delta={bin_delta}: XY={fsc_result_nn['xy']:.4f}μm, Z={fsc_result_nn['z']:.4f}μm"
            )
        except Exception as e:
            print(f"  order=0 failed: {e}")

    # With order=1 (linear)
    if resample_order != 1:
        try:
            fsc_result_lin = fsc_resolution(
                image,
                spacing=spacing,
                resample_isotropic=True,
                resample_order=1,  # Linear interpolation
                angle_delta=angle_delta,
                bin_delta=bin_delta,
            )
            print(
                f"  order=1 (linear), bin_delta={bin_delta}: XY={fsc_result_lin['xy']:.4f}μm, Z={fsc_result_lin['z']:.4f}μm"
            )
        except Exception as e:
            print(f"  order=1 failed: {e}")

    # Compare with DCR
    print("\nRunning DCR for comparison...")
    try:
        dcr_result = dcr_resolution(
            image,
            spacing=spacing,
            use_sectioned=True,
        )
        print(f"  DCR: XY={dcr_result['xy']:.4f}μm, Z={dcr_result['z']:.4f}μm")
    except Exception as e:
        print(f"  DCR failed: {e}")

    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    # Create default config if it doesn't exist
    config_path = Path(__file__).parent / "sfsc_paper_comparison.yaml"
    if not config_path.exists():
        default_config = """# SFSC Paper Comparison Configuration
# Koho et al. 2019 comparison settings

# Image settings
image_path: null  # Path to 3D image (null = use default example data)
spacing: [0.250, 0.0777, 0.0777]  # [Z, Y, X] in micrometers

# SFSC parameters
angle_delta: 15  # Angular bin width in degrees (paper uses 15°)
bin_delta: 10    # Radial bin width

# Resampling parameters
# 0 = nearest neighbor (paper's miplib method)
# 1 = linear interpolation (our default)
resample_order: 1

# Output
output_dir: ./sfsc_comparison_output
"""
        with open(config_path, "w") as f:
            f.write(default_config)
        print(f"Created default config at {config_path}")

    main()
