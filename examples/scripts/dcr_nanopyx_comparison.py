#!/usr/bin/env python3
"""DCR Implementation Comparison: Cubic vs NanoPyx.

This script creates an informative comparison plot showing how cubic and NanoPyx
DCR implementations arrive at their resolution estimates. It reveals the algorithmic
differences that explain the ~39% discrepancy between the methods.

The comparison includes:
1. Image preview (input data)
2. All d(r) curves overlaid from both implementations
3. Peak detection scatter plot (r_peak vs amplitude)
4. Resolution and timing comparison bars
5. Summary table with parameter differences

Usage:
    python dcr_nanopyx_comparison.py
    python dcr_nanopyx_comparison.py crop.size=512

References
----------
Descloux et al. (2019) "Parameter-free image resolution estimation based on
    decorrelation analysis" Nature Methods
NanoPyx: https://github.com/HenriquesLab/NanoPyx
"""

import sys
import time
from pathlib import Path

import hydra
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig
from skimage.io import imread

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cubic.cuda import ascupy, asnumpy, get_device
from cubic.metrics.spectral.dcr import dcr_curve, _generate_highpass_sigmas


def astrocyte_cell_region_crop(
    shape_zyx: tuple[int, ...], size_xy: int
) -> tuple[slice, slice, slice]:
    """Return (z_slice, y_slice, x_slice) for a crop of size_xy that captures cells.

    For the astrocyte image (astr_vpa_hoechst.tif):
    - Cells are located roughly at Y=1000:end, X=1300:end
    - Top-left and bottom-right ~512x512 corners are empty
    - This function returns a smart crop that maximizes cell content

    Parameters
    ----------
    shape_zyx : tuple
        Shape of the image (Z, Y, X) or (Y, X)
    size_xy : int
        Crop size in pixels (e.g., 256, 512, 1024)

    Returns
    -------
    tuple of slices
        (z_slice, y_slice, x_slice) for indexing the image
    """
    # Handle 2D or 3D shapes
    if len(shape_zyx) == 2:
        z, y, x = 1, shape_zyx[0], shape_zyx[1]
    else:
        z, y, x = shape_zyx[0], shape_zyx[1], shape_zyx[2]

    # Cell region boundaries (avoiding empty corners)
    y_lo, y_hi = 1000, y - 512
    x_lo, x_hi = 1300, x - 512

    if size_xy <= (y_hi - y_lo) and size_xy <= (x_hi - x_lo):
        # Crop fits within cell region - center it
        cy, cx = (y_lo + y_hi) / 2, (x_lo + x_hi) / 2
        y0 = int(cy - size_xy / 2)
        x0 = int(cx - size_xy / 2)

        # Shift right to avoid empty left edge for smaller crops
        x_shift = 50 if size_xy == 256 else (300 if size_xy == 512 else 0)
        x0 += x_shift

        # Clamp to valid range
        y0 = max(y_lo, min(y0, y_hi - size_xy))
        x0 = max(x_lo, min(x0, x - size_xy))
    else:
        # Large crop (1024+): anchor at cell-region start to maximize cell content
        y0, x0 = y_lo, x_lo

    z_slice = slice(0, z)
    y_slice = slice(y0, y0 + size_xy)
    x_slice = slice(x0, x0 + size_xy)

    # Verify crop produces the expected size
    crop_y = y_slice.stop - y_slice.start
    crop_x = x_slice.stop - x_slice.start
    if crop_y != size_xy or crop_x != size_xy:
        raise ValueError(
            f"Crop size mismatch: expected {size_xy}x{size_xy}, "
            f"got {crop_y}x{crop_x} (y={y_slice}, x={x_slice})"
        )

    # Verify crop is within image bounds
    if y_slice.stop > y or x_slice.stop > x:
        raise ValueError(
            f"Crop exceeds image bounds: image is {y}x{x}, "
            f"crop ends at y={y_slice.stop}, x={x_slice.stop}"
        )

    return (z_slice, y_slice, x_slice)


def load_image(
    image_path: Path, crop_size: int = 1024, smart_crop: bool = True
) -> tuple[np.ndarray, dict]:
    """Load and crop image, returning metadata.

    Parameters
    ----------
    image_path : Path
        Path to the image file
    crop_size : int
        Size of the crop (e.g., 256, 512, 1024)
    smart_crop : bool
        If True and image is astrocyte, use smart crop to capture cell region.
        If False, use center crop.
    """
    img = imread(image_path).astype(np.float32)
    is_astrocyte = "astr" in image_path.name.lower()

    metadata = {
        "path": str(image_path),
        "original_shape": img.shape,
        "dtype": str(img.dtype),
    }

    # Determine if we should use smart crop for astrocyte images
    use_smart_crop = smart_crop and is_astrocyte and img.ndim == 3

    if use_smart_crop and crop_size:
        # Use smart crop that captures cell region
        z_slice, y_slice, x_slice = astrocyte_cell_region_crop(img.shape, crop_size)
        # Take middle Z slice from the cropped region
        mid_z = img.shape[0] // 2
        img = img[mid_z, y_slice, x_slice]
        metadata["slice"] = f"z={mid_z}"
        metadata["crop"] = (
            f"{crop_size}x{crop_size} smart (y={y_slice.start}, x={x_slice.start})"
        )
    else:
        # Standard processing: take middle Z slice first, then center crop
        if img.ndim == 3:
            mid_z = img.shape[0] // 2
            img = img[mid_z]
            metadata["slice"] = f"z={mid_z}"

        # Center crop
        if crop_size and crop_size < min(img.shape):
            cy, cx = img.shape[0] // 2, img.shape[1] // 2
            half = crop_size // 2
            img = img[cy - half : cy + half, cx - half : cx + half]
            metadata["crop"] = f"{crop_size}x{crop_size} center"

    metadata["final_shape"] = img.shape
    return img, metadata


def run_cubic_dcr(
    image: np.ndarray,
    pixel_size: float,
    num_radii: int = 50,
    num_highpass: int = 10,
    refine: bool = False,
) -> dict:
    """Run cubic DCR and extract all intermediate data."""
    start = time.perf_counter()
    resolution, radii, all_curves, all_peaks = dcr_curve(
        image,
        spacing=pixel_size,
        num_radii=num_radii,
        num_highpass=num_highpass,
        smoothing=11,
        windowing=True,
        refine=refine,
    )
    elapsed = (time.perf_counter() - start) * 1000

    # Get highpass sigmas for documentation
    sigmas = _generate_highpass_sigmas(image.shape, num_highpass)

    return {
        "resolution": resolution * 1000,  # Convert to nm
        "time_ms": elapsed,
        "radii": np.array(radii),
        "all_curves": [np.array(c) for c in all_curves],
        "all_peaks": np.array(all_peaks),
        "num_radii": num_radii,
        "num_highpass": num_highpass,
        "sigmas": sigmas,
        "smoothing": 11,
        "windowing": "Tukey (alpha=0.1)",
        "refine": refine,
    }


def run_nanopyx_dcr(
    image: np.ndarray, pixel_size: float, num_radii: int = 50, num_highpass: int = 10
) -> dict:
    """Run NanoPyx DCR and return results including native plot.

    NanoPyx's DecorrAnalysis uses Cython cdef attributes for internal data,
    which are not accessible from Python. We use plot_results() to get their
    native visualization of the decorrelation curves.
    """
    from nanopyx.core.analysis.decorr import DecorrAnalysis

    start = time.perf_counter()
    da = DecorrAnalysis(
        n_r=num_radii, n_g=num_highpass, pixel_size=pixel_size * 1000, units="nm"
    )
    da.run_analysis(image.astype(np.float32))
    elapsed = (time.perf_counter() - start) * 1000

    # Get native plot from NanoPyx (includes all internal curves)
    plot_image = da.plot_results()

    # Infer k_c from resolution: resolution = 2 * pixel_size / k_c
    kc_from_resolution = (
        2 * pixel_size * 1000 / da.resolution
        if da.resolution > 0 and not np.isinf(da.resolution)
        else 0
    )

    return {
        "resolution": da.resolution,
        "time_ms": elapsed,
        "plot_image": plot_image,
        "kc": kc_from_resolution,
        "num_radii": num_radii,
        "num_highpass": num_highpass,
        "windowing": "Cosine edge apodization",
    }


def plot_comparison(
    image: np.ndarray,
    cubic_data: dict,
    cubic_refined_data: dict,
    nanopyx_data: dict,
    metadata: dict,
    pixel_size: float,
    output_path: Path,
):
    """Create comprehensive comparison plot with 3 methods."""
    # Use a clean style
    plt.style.use("seaborn-v0_8-whitegrid")

    fig = plt.figure(figsize=(16, 16))

    # Grid layout: 3 rows x 2 cols
    # Row 1: Image preview | Cubic single-pass curves
    # Row 2: Cubic refined curves | NanoPyx native plot
    # Row 3: Resolution & timing bars | Summary table
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.7], hspace=0.35, wspace=0.25)

    # ==================== Panel 1: Image Preview ====================
    ax1 = fig.add_subplot(gs[0, 0])
    vmin, vmax = np.percentile(image, [1, 99])
    ax1.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    ax1.set_title("Input Image", fontsize=12, fontweight="bold")

    # Add metadata annotation
    info_text = (
        f"Shape: {metadata['final_shape']}\nPixel size: {pixel_size * 1000:.1f} nm"
    )
    if "crop" in metadata:
        info_text += f"\n{metadata['crop']}"
    ax1.text(
        0.02,
        0.98,
        info_text,
        transform=ax1.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )
    ax1.axis("off")

    # ==================== Panel 2: Cubic Single-Pass Curves ====================
    ax2 = fig.add_subplot(gs[0, 1])

    n_coarse = cubic_data["num_highpass"]
    for i, curve in enumerate(cubic_data["all_curves"]):
        alpha = 0.3 + 0.6 * (i / n_coarse)
        ax2.plot(
            cubic_data["radii"],
            curve,
            "-",
            color="steelblue",
            alpha=alpha,
            linewidth=1.5,
        )

    # Mark final k_c position
    cubic_kc = cubic_data["all_peaks"][:, 0].max()
    ax2.axvline(
        cubic_kc,
        color="navy",
        linestyle="--",
        linewidth=2,
        label=f"k_c = {cubic_kc:.3f}",
    )

    ax2.set_xlabel("Normalized Frequency (r)", fontsize=11)
    ax2.set_ylabel("Decorrelation d(r)", fontsize=11)
    ax2.set_title(
        f"Cubic Single-Pass ({cubic_data['resolution']:.1f} nm)",
        fontsize=12,
        fontweight="bold",
    )
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(True, alpha=0.3)

    # ==================== Panel 3: Cubic Refined Curves ====================
    ax3 = fig.add_subplot(gs[1, 0])

    n_coarse_ref = cubic_refined_data["num_highpass"]
    n_total = len(cubic_refined_data["all_curves"])
    n_refined = n_total - n_coarse_ref

    # Plot coarse curves (light blue, dashed)
    for i in range(n_coarse_ref):
        curve = cubic_refined_data["all_curves"][i]
        # Coarse curves may have different radii than refined; use index-based x-axis
        # or the original radii. Since coarse uses [0,1] and refined uses narrowed range,
        # plot coarse on [0,1] radii
        coarse_radii = np.linspace(0, 1, len(curve), dtype=np.float32)
        alpha = 0.2 + 0.4 * (i / n_coarse_ref)
        ax3.plot(
            coarse_radii,
            curve,
            "--",
            color="steelblue",
            alpha=alpha,
            linewidth=1.0,
        )

    # Plot refined curves (teal, solid)
    for i in range(n_coarse_ref, n_total):
        curve = cubic_refined_data["all_curves"][i]
        alpha = 0.3 + 0.6 * ((i - n_coarse_ref) / max(n_refined, 1))
        ax3.plot(
            cubic_refined_data["radii"],
            curve,
            "-",
            color="teal",
            alpha=alpha,
            linewidth=1.5,
        )

    # Mark coarse k_c
    coarse_peaks = cubic_refined_data["all_peaks"][:n_coarse_ref, 0]
    coarse_kc = coarse_peaks.max() if len(coarse_peaks) > 0 else 0
    if coarse_kc > 0:
        ax3.axvline(
            coarse_kc,
            color="steelblue",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label=f"coarse k_c = {coarse_kc:.3f}",
        )

    # Mark refined k_c
    refined_kc = cubic_refined_data["all_peaks"][:, 0].max()
    ax3.axvline(
        refined_kc,
        color="darkgreen",
        linestyle="--",
        linewidth=2,
        label=f"refined k_c = {refined_kc:.3f}",
    )

    ax3.set_xlabel("Normalized Frequency (r)", fontsize=11)
    ax3.set_ylabel("Decorrelation d(r)", fontsize=11)
    ax3.set_title(
        f"Cubic Refined ({cubic_refined_data['resolution']:.1f} nm)",
        fontsize=12,
        fontweight="bold",
    )
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.legend(fontsize=9, loc="lower right")
    ax3.grid(True, alpha=0.3)

    # ==================== Panel 4: NanoPyx Native Plot ====================
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.imshow(nanopyx_data["plot_image"])
    ax4.set_title(
        f"NanoPyx Native ({nanopyx_data['resolution']:.1f} nm)",
        fontsize=12,
        fontweight="bold",
    )
    ax4.axis("off")

    # ==================== Panel 5: Resolution & Time Comparison ====================
    ax5 = fig.add_subplot(gs[2, 0])

    methods = ["Cubic\n(single)", "Cubic\n(refined)", "NanoPyx"]
    resolutions = [
        cubic_data["resolution"],
        cubic_refined_data["resolution"],
        nanopyx_data["resolution"],
    ]
    times = [
        cubic_data["time_ms"],
        cubic_refined_data["time_ms"],
        nanopyx_data["time_ms"],
    ]
    colors = ["steelblue", "teal", "darkred"]

    # Create twin axis for time
    ax5_twin = ax5.twinx()

    # Bar chart for resolution
    x = np.arange(len(methods))
    width = 0.35
    bars_res = ax5.bar(
        x - width / 2, resolutions, width, color=colors, alpha=0.8, label="Resolution"
    )

    # Bar chart for time
    bars_time = ax5_twin.bar(
        x + width / 2, times, width, color=colors, alpha=0.4, hatch="//", label="Time"
    )

    # Add value labels
    for bar, val in zip(bars_res, resolutions):
        ax5.annotate(
            f"{val:.1f} nm",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )
    for bar, val in zip(bars_time, times):
        ax5_twin.annotate(
            f"{val:.1f} ms",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            fontsize=9,
        )

    ax5.set_ylabel("Resolution (nm)", fontsize=11)
    ax5_twin.set_ylabel("Time (ms)", fontsize=11)
    ax5.set_xticks(x)
    ax5.set_xticklabels(methods, fontsize=10, fontweight="bold")

    # Calculate differences vs NanoPyx
    diff_single = (
        (cubic_data["resolution"] - nanopyx_data["resolution"])
        / nanopyx_data["resolution"]
        * 100
    )
    diff_refined = (
        (cubic_refined_data["resolution"] - nanopyx_data["resolution"])
        / nanopyx_data["resolution"]
        * 100
    )

    ax5.set_title(
        f"Resolution & Performance\n"
        f"vs NanoPyx: single {diff_single:+.1f}%, refined {diff_refined:+.1f}%",
        fontsize=12,
        fontweight="bold",
    )
    ax5.grid(True, alpha=0.3, axis="y")

    # ==================== Panel 6: Summary Table ====================
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis("off")

    # Build comparison table
    cubic_kc = cubic_data["all_peaks"][:, 0].max()
    refined_kc = cubic_refined_data["all_peaks"][:, 0].max()
    nanopyx_kc = nanopyx_data["kc"]
    table_data = [
        ["Parameter", "Cubic (single)", "Cubic (refined)", "NanoPyx"],
        [
            "Resolution",
            f"{cubic_data['resolution']:.1f} nm",
            f"{cubic_refined_data['resolution']:.1f} nm",
            f"{nanopyx_data['resolution']:.1f} nm",
        ],
        [
            "k_c",
            f"{cubic_kc:.4f}",
            f"{refined_kc:.4f}",
            f"{nanopyx_kc:.4f}",
        ],
        [
            "Time",
            f"{cubic_data['time_ms']:.0f} ms",
            f"{cubic_refined_data['time_ms']:.0f} ms",
            f"{nanopyx_data['time_ms']:.0f} ms",
        ],
        [
            "Refinement",
            "None",
            "Two-pass",
            "Two-pass",
        ],
        [
            "vs NanoPyx",
            f"{diff_single:+.1f}%",
            f"{diff_refined:+.1f}%",
            "reference",
        ],
    ]

    table = ax6.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        loc="center",
        cellLoc="center",
        colColours=["lightgray"] * 4,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    # Style header row
    for j in range(4):
        table[(0, j)].set_text_props(fontweight="bold")

    # Highlight resolution and vs-NanoPyx rows
    for i in [1, 5]:
        for j in range(4):
            table[(i, j)].set_facecolor("#fff3cd")

    ax6.set_title("Summary Comparison", fontsize=12, fontweight="bold", pad=20)

    # Main title
    fig.suptitle(
        "DCR Algorithm Comparison: Cubic (single) vs Cubic (refined) vs NanoPyx",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    # Adjust layout
    fig.subplots_adjust(
        top=0.94, bottom=0.02, left=0.05, right=0.95, hspace=0.35, wspace=0.25
    )
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved: {output_path}")
    plt.close()
    # Reset style
    plt.style.use("default")


def print_diagnostic_summary(
    cubic_data: dict, cubic_refined_data: dict, nanopyx_data: dict
):
    """Print detailed diagnostic summary to console."""
    print("\n" + "=" * 90)
    print("DCR COMPARISON DIAGNOSTIC SUMMARY")
    print("=" * 90)

    nanopyx_kc = nanopyx_data["kc"]
    cubic_kc = cubic_data["all_peaks"][:, 0].max()
    refined_kc = cubic_refined_data["all_peaks"][:, 0].max()

    print(
        f"\n{'METRIC':<25} {'CUBIC (single)':<18} {'CUBIC (refined)':<18} {'NANOPYX':<18}"
    )
    print("-" * 80)
    print(
        f"{'Resolution (nm)':<25} {cubic_data['resolution']:<18.2f} "
        f"{cubic_refined_data['resolution']:<18.2f} {nanopyx_data['resolution']:<18.2f}"
    )
    print(
        f"{'Cut-off freq (k_c)':<25} {cubic_kc:<18.4f} "
        f"{refined_kc:<18.4f} {nanopyx_kc:<18.4f}"
    )
    print(
        f"{'Execution time (ms)':<25} {cubic_data['time_ms']:<18.2f} "
        f"{cubic_refined_data['time_ms']:<18.2f} {nanopyx_data['time_ms']:<18.2f}"
    )

    # Cubic single-pass peaks
    print(f"\n{'CUBIC SINGLE-PASS PEAKS':<30}")
    print("-" * 80)
    for i, (r, a) in enumerate(cubic_data["all_peaks"]):
        if r > 0:
            marker = " <<< SELECTED" if r == cubic_kc else ""
            print(f"  HP {i:2d}: r={r:.4f}, A={a:.4f}{marker}")

    # Cubic refined peaks
    n_coarse = cubic_refined_data["num_highpass"]
    print(f"\n{'CUBIC REFINED PEAKS':<30}")
    print("-" * 80)
    print("  Coarse pass:")
    for i in range(n_coarse):
        r, a = cubic_refined_data["all_peaks"][i]
        if r > 0:
            print(f"    HP {i:2d}: r={r:.4f}, A={a:.4f}")
    print("  Refined pass:")
    for i in range(n_coarse, len(cubic_refined_data["all_peaks"])):
        r, a = cubic_refined_data["all_peaks"][i]
        if r > 0:
            marker = " <<< SELECTED" if r == refined_kc else ""
            print(f"    HP {i - n_coarse:2d}: r={r:.4f}, A={a:.4f}{marker}")

    print(f"\n{'KEY INSIGHTS':<30}")
    print("-" * 80)

    diff_single = (
        (cubic_data["resolution"] - nanopyx_data["resolution"])
        / nanopyx_data["resolution"]
        * 100
    )
    diff_refined = (
        (cubic_refined_data["resolution"] - nanopyx_data["resolution"])
        / nanopyx_data["resolution"]
        * 100
    )
    print(f"\n1. Resolution vs NanoPyx:")
    print(f"   - Single-pass: {diff_single:+.1f}%")
    print(f"   - Refined:     {diff_refined:+.1f}%")
    improvement = abs(diff_single) - abs(diff_refined)
    if improvement > 0:
        print(f"   => Refinement closes gap by {improvement:.1f} percentage points")
    else:
        print(f"   => Refinement does not improve agreement")

    print(f"\n2. k_c comparison:")
    print(f"   - Single vs NanoPyx: {cubic_kc - nanopyx_kc:+.4f}")
    print(f"   - Refined vs NanoPyx: {refined_kc - nanopyx_kc:+.4f}")

    print("\n3. Algorithm differences:")
    print("   - Cubic (single): One-pass, Savgol smoothing, max(r_i) selection")
    print(
        "   - Cubic (refined): Two-pass (coarse + narrowed range), same as NanoPyx strategy"
    )
    print("   - NanoPyx: Two-phase refinement (Cython, native implementation)")

    speedup_single = nanopyx_data["time_ms"] / cubic_data["time_ms"]
    speedup_refined = nanopyx_data["time_ms"] / cubic_refined_data["time_ms"]
    print(f"\n4. Performance vs NanoPyx:")
    if speedup_single > 1:
        print(f"   - Single-pass: {speedup_single:.1f}x faster")
    else:
        print(f"   - Single-pass: {1 / speedup_single:.1f}x slower")
    if speedup_refined > 1:
        print(f"   - Refined:     {speedup_refined:.1f}x faster")
    else:
        print(f"   - Refined:     {1 / speedup_refined:.1f}x slower")

    print("\n" + "=" * 90)


@hydra.main(
    version_base=None,
    config_path="conf",
    config_name="dcr_comparison",
)
def main(cfg: DictConfig) -> None:
    """Run DCR comparison between Cubic and NanoPyx implementations."""
    # Resolve paths
    script_dir = Path(__file__).parent
    default_image = script_dir.parent / "data" / "astr_vpa_hoechst.tif"
    image_path = Path(cfg.image_path) if cfg.image_path else default_image

    # Get parameters from config
    crop_size = cfg.crop.size if cfg.crop.enabled else None
    pixel_size = cfg.spacing.xy
    num_radii = cfg.dcr.num_radii
    num_highpass = cfg.dcr.num_highpass

    # Output path with crop size in filename
    output_dir = Path(cfg.output_dir)
    output_path = output_dir / f"dcr_comparison_{crop_size}.png"

    print(f"Loading image: {image_path}")
    image, metadata = load_image(image_path, crop_size, smart_crop=True)
    print(f"  Original shape: {metadata['original_shape']}")
    print(f"  Final shape: {metadata['final_shape']}")
    if "crop" in metadata:
        print(f"  Crop: {metadata['crop']}")

    # Move to GPU for cubic runs
    image_gpu = ascupy(image)
    print(f"  Device: {get_device(image_gpu)}")

    # Warm up GPU (first CuPy call includes JIT compilation overhead)
    _ = np.fft.fftn(image_gpu)

    print("\nRunning Cubic DCR (single-pass, GPU)...")
    cubic_data = run_cubic_dcr(
        image_gpu, pixel_size, num_radii, num_highpass, refine=False
    )
    print(f"  Resolution: {cubic_data['resolution']:.2f} nm")
    print(f"  Time: {cubic_data['time_ms']:.2f} ms")

    print("\nRunning Cubic DCR (refined, GPU)...")
    cubic_refined_data = run_cubic_dcr(
        image_gpu, pixel_size, num_radii, num_highpass, refine=True
    )
    print(f"  Resolution: {cubic_refined_data['resolution']:.2f} nm")
    print(f"  Time: {cubic_refined_data['time_ms']:.2f} ms")

    print("\nRunning NanoPyx DCR...")
    nanopyx_data = run_nanopyx_dcr(image, pixel_size, num_radii, num_highpass)
    print(f"  Resolution: {nanopyx_data['resolution']:.2f} nm")
    print(f"  Time: {nanopyx_data['time_ms']:.2f} ms")

    # Print diagnostic summary
    print_diagnostic_summary(cubic_data, cubic_refined_data, nanopyx_data)

    # Generate comparison plot
    print("\nGenerating comparison plot...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(
        image,
        cubic_data,
        cubic_refined_data,
        nanopyx_data,
        metadata,
        pixel_size,
        output_path,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
