#!/usr/bin/env python3
"""
Debug script to investigate XY/Z resolution swap in FSC and DCR.

This script investigates why FSC and DCR return Z resolution that appears
better (smaller value) than XY resolution, which is physically impossible
for confocal microscopy.

Problem:
- FSC (bin_delta=1): XY=1123.5nm, Z=873.1nm -> Z < XY (wrong!)
- DCR: XY=inf, Z=408.1nm -> Z < XY (wrong!)

For confocal microscopy, Z resolution should always be WORSE (larger value)
than XY resolution due to the elongated PSF along the optical axis.

Investigation goals:
1. Debug FSC resolution mapping - verify angle assignment
2. Debug DCR resolution mapping - verify angle assignment
3. Check if k(θ) correction is being applied correctly
4. Compare raw vs corrected resolution values

Author: Claude Code
"""

import sys
from pathlib import Path

import numpy as np

# Add parent directory to path for cubic imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from cubic.cuda import ascupy, asnumpy, get_device
from cubic.image_utils import rescale_isotropic
from cubic.metrics.spectral import dcr_resolution, fsc_resolution
from cubic.metrics.spectral.dcr import (
    _dcr_curve_3d_sectioned,
    _compute_decorrelation_curve_sectioned,
)
from cubic.metrics.spectral.frc import preprocess_images, _calculate_fsc_sectioned_hist
from cubic.metrics.spectral.analysis import (
    FourierCorrelationData,
    FourierCorrelationAnalysis,
    FourierCorrelationDataCollection,
)


def load_test_image(image_path: str | Path) -> np.ndarray:
    """Load 3D test image."""
    from skimage import io

    path = Path(image_path)

    if path.suffix.lower() == ".nd2":
        try:
            import nd2

            with nd2.ND2File(path) as f:
                img = f.asarray()
                sizes = f.sizes
                print(f"  ND2 dimensions: {sizes}")

                if img.ndim == 4 and "C" in sizes and "Z" in sizes:
                    dim_order = list(sizes.keys())
                    c_axis = dim_order.index("C")
                    img = np.take(img, 0, axis=c_axis)
                    print(f"  Selected first channel, shape: {img.shape}")
                elif img.ndim == 4:
                    img = img[0]
                    print(f"  Took first slice, shape: {img.shape}")

                return img.astype(np.float32)
        except ImportError:
            print("nd2 package not available, trying skimage.io")

    img = io.imread(path)
    return img.astype(np.float32)


def debug_fsc_sectors(
    image: np.ndarray,
    spacing: list[float],
    bin_delta: int = 1,
    angle_delta: int = 15,
) -> None:
    """
    Debug FSC sector assignment and resolution calculation.

    This function prints detailed information about:
    1. All sector angles and their resolutions
    2. The angle selection for XY and Z
    3. Raw vs corrected resolution values
    4. z_factor calculation
    """
    print("\n" + "=" * 80)
    print("DEBUG: FSC SECTOR ASSIGNMENT AND RESOLUTION")
    print("=" * 80)

    spacing_z = spacing[0]
    spacing_xy = spacing[1]
    z_factor = spacing_z / spacing_xy

    print(f"\nSpacing: Z={spacing_z}, XY={spacing_xy}")
    print(f"z_factor (anisotropy): {z_factor:.3f}")
    print(f"bin_delta: {bin_delta}, angle_delta: {angle_delta}")

    # Resample to isotropic
    iso_spacing = spacing_xy
    target_z_size = int(round(image.shape[0] * spacing_z / iso_spacing))
    if target_z_size % 2 != 0:
        target_z_size -= 1

    image_iso = rescale_isotropic(
        image,
        tuple(spacing),
        downscale_xy=False,
        order=1,
        preserve_range=True,
        target_z_size=target_z_size,
    ).astype(np.float32)

    # Ensure even dimensions
    even_shape = tuple(s - (s % 2) for s in image_iso.shape)
    if image_iso.shape != even_shape:
        slices = tuple(slice(0, es) for es in even_shape)
        image_iso = image_iso[slices]

    print(f"\nOriginal shape: {image.shape}")
    print(f"Isotropic shape: {image_iso.shape}")

    # Preprocess images
    image1, image2 = preprocess_images(
        image_iso,
        None,
        zero_padding=False,
        disable_hamming=False,
        disable_3d_sum=False,
    )

    # Calculate sectioned FSC
    spacing_iso = [iso_spacing] * 3
    fsc_data = _calculate_fsc_sectioned_hist(
        image1,
        image2,
        bin_delta=bin_delta,
        angle_delta=angle_delta,
        spacing=spacing_iso,
        exclude_axis_angle=0.0,
        use_max_nyquist=False,
    )

    angles = sorted(fsc_data.keys())
    print(f"\nSector angles (polar angle from Z axis): {angles}")
    print(f"  - θ ≈ 0° means Z-dominated (|kz| >> k_xy) → Z resolution")
    print(f"  - θ ≈ 90° means XY-dominated (k_xy >> |kz|) → XY resolution")

    # Analyze each sector
    print(
        f"\n{'Angle':<8} {'Raw Res (μm)':<15} {'k(θ)':<8} {'Corrected (μm)':<15} {'Sector':<10}"
    )
    print("-" * 60)

    results = {}
    for angle in angles:
        data = fsc_data[angle]

        # Create data collection for analysis
        data_collection = FourierCorrelationDataCollection()
        data_set = FourierCorrelationData()
        data_set.correlation["correlation"] = np.asarray(
            data.correlation["correlation"]
        )
        data_set.correlation["frequency"] = np.asarray(data.correlation["frequency"])
        data_set.correlation["points-x-bin"] = np.asarray(
            data.correlation["points-x-bin"]
        )
        data_collection[0] = data_set

        # Run analysis
        analyzer = FourierCorrelationAnalysis(
            data_collection,
            iso_spacing,
            resolution_threshold="one-bit",
            curve_fit_type="spline",
        )

        try:
            analyzed = analyzer.execute()[0]
            raw_resolution = analyzed.resolution["resolution"]
        except Exception:
            raw_resolution = np.nan

        # Determine which edge to use for k(θ) correction
        if angle == min(angles):
            # Z sector: use lower bin edge (0°) for maximum correction
            angle_for_correction = 0.0
            sector_type = "Z"
        elif angle == max(angles):
            # XY sector: use upper bin edge (90°) for no correction
            angle_for_correction = 90.0
            sector_type = "XY"
        else:
            # Intermediate: use bin center
            angle_for_correction = float(angle)
            sector_type = "intermediate"

        # Calculate k(θ) correction: k(θ) = 1 + (z_factor - 1) × |cos(θ)|
        k_theta = 1 + (z_factor - 1) * np.abs(np.cos(np.radians(angle_for_correction)))
        corrected_resolution = (
            raw_resolution * k_theta if not np.isnan(raw_resolution) else np.nan
        )

        results[angle] = {
            "raw": raw_resolution,
            "k_theta": k_theta,
            "corrected": corrected_resolution,
            "sector_type": sector_type,
        }

        print(
            f"{angle}°{'':<5} {raw_resolution:<15.4f} {k_theta:<8.3f} {corrected_resolution:<15.4f} {sector_type:<10}"
        )

    # Summary of XY and Z resolution
    angle_xy = max(angles)
    angle_z = min(angles)

    print(f"\n{'=' * 60}")
    print("SUMMARY:")
    print(f"  XY sector (θ={angle_xy}°, XY-dominated):")
    print(f"    Raw resolution: {results[angle_xy]['raw']:.4f} μm")
    print(f"    k(θ) correction: {results[angle_xy]['k_theta']:.4f}")
    print(f"    Corrected resolution: {results[angle_xy]['corrected']:.4f} μm")
    print(f"  Z sector (θ={angle_z}°, Z-dominated):")
    print(f"    Raw resolution: {results[angle_z]['raw']:.4f} μm")
    print(f"    k(θ) correction: {results[angle_z]['k_theta']:.4f}")
    print(f"    Corrected resolution: {results[angle_z]['corrected']:.4f} μm")

    z_res = results[angle_z]["corrected"]
    xy_res = results[angle_xy]["corrected"]

    print(f"\n  Z/XY ratio: {z_res / xy_res:.2f} (should be > 1 for confocal)")

    if z_res > xy_res:
        print("  ✓ Z > XY: Physically correct for confocal microscopy")
    else:
        print("  ✗ Z < XY: WRONG! Z should be worse (larger) than XY")

    return results


def debug_dcr_sectors(
    image: np.ndarray,
    spacing: list[float],
    angle_delta: int = 45,
) -> None:
    """
    Debug DCR sector assignment and resolution calculation.

    This function investigates why DCR returns Z resolution better than XY.
    """
    print("\n" + "=" * 80)
    print("DEBUG: DCR SECTOR ASSIGNMENT AND RESOLUTION")
    print("=" * 80)

    spacing_z = spacing[0]
    spacing_xy = spacing[1]
    z_factor = spacing_z / spacing_xy

    print(f"\nSpacing: Z={spacing_z}, XY={spacing_xy}")
    print(f"z_factor (anisotropy): {z_factor:.3f}")
    print(f"angle_delta: {angle_delta}")

    # Resample to isotropic
    iso_spacing = spacing_xy
    target_z_size = int(round(image.shape[0] * spacing_z / iso_spacing))
    if target_z_size % 2 != 0:
        target_z_size -= 1

    image_iso = rescale_isotropic(
        image,
        tuple(spacing),
        downscale_xy=False,
        order=1,
        preserve_range=True,
        target_z_size=target_z_size,
    ).astype(np.float32)

    # Ensure even dimensions
    even_shape = tuple(s - (s % 2) for s in image_iso.shape)
    if image_iso.shape != even_shape:
        slices = tuple(slice(0, es) for es in even_shape)
        image_iso = image_iso[slices]

    print(f"\nOriginal shape: {image.shape}")
    print(f"Isotropic shape: {image_iso.shape}")

    # Get the DCR curves for each sector
    # Using internal function to see raw values
    spacing_arr = np.array([iso_spacing] * 3, dtype=np.float32)

    # Prepare image for DCR
    image_prep = image_iso.copy().astype(np.float32)
    image_prep -= np.mean(image_prep)

    # Get decorrelation curves
    sector_results = _compute_decorrelation_curve_sectioned(
        image_prep,
        num_radii=100,
        angle_delta=angle_delta,
        bin_delta=1,
        spacing=spacing_arr,
        exclude_axis_angle=0.0,
        smoothing=11,
    )

    print(f"\nDCR sector assignment (center < 45° → Z, >= 45° → XY):")
    print(f"  Sectors found: {list(sector_results.keys())}")

    # Analyze each sector
    print(
        f"\n{'Sector':<8} {'k_max':<12} {'Peak (norm)':<12} {'k_c_phys':<12} {'Raw Res (μm)':<15}"
    )
    print("-" * 60)

    results = {}
    for sector_name in ["xy", "z"]:
        radii_norm, d_curve, k_max = sector_results[sector_name]

        # Find peak in curve
        from cubic.metrics.spectral.dcr import _find_peak_in_curve

        r_peak, _ = _find_peak_in_curve(radii_norm, d_curve)

        if r_peak > 0 and k_max > 0:
            k_c_physical = r_peak * k_max
            raw_resolution = 1.0 / k_c_physical
        else:
            k_c_physical = 0
            raw_resolution = float("inf")

        results[sector_name] = {
            "k_max": k_max,
            "r_peak": r_peak,
            "k_c_physical": k_c_physical,
            "raw": raw_resolution,
        }

        print(
            f"{sector_name:<8} {k_max:<12.4f} {r_peak:<12.4f} {k_c_physical:<12.4f} {raw_resolution:<15.4f}"
        )

    # Check if k(θ) correction should be applied
    print(f"\n{'=' * 60}")
    print("k(θ) CORRECTION ANALYSIS:")
    print(f"  z_factor = {z_factor:.3f}")
    print(f"  For Z sector (θ=0°): k(0°) = z_factor = {z_factor:.3f}")
    print(f"  For XY sector (θ=90°): k(90°) = 1.0")

    # Calculate what corrected values should be
    z_raw = results["z"]["raw"]
    xy_raw = results["xy"]["raw"]

    if not np.isinf(z_raw):
        z_corrected = z_raw * z_factor
    else:
        z_corrected = float("inf")
    xy_corrected = xy_raw  # No correction needed for XY (k(90°) = 1)

    print(f"\nDCR RESULTS:")
    print(f"  XY sector:")
    print(f"    Raw resolution: {xy_raw:.4f} μm")
    print(f"    Corrected (should apply k(90°)=1): {xy_corrected:.4f} μm")
    print(f"  Z sector:")
    print(f"    Raw resolution: {z_raw:.4f} μm")
    print(f"    Corrected (should apply k(0°)={z_factor:.3f}): {z_corrected:.4f} μm")

    if not np.isinf(z_corrected) and not np.isinf(xy_corrected):
        print(f"\n  Z/XY ratio (corrected): {z_corrected / xy_corrected:.2f}")

        if z_corrected > xy_corrected:
            print("  ✓ Z > XY: Physically correct for confocal microscopy")
        else:
            print("  ✗ Z < XY: WRONG! Z should be worse (larger) than XY")

    # Note: Check if DCR code applies this correction
    print(f"\n⚠️  IMPORTANT: Check if _dcr_curve_3d_sectioned applies k(θ) correction!")
    print(f"    FSC has this at frc.py:1041-1042, DCR may be missing it.")


def run_standard_resolution_functions(
    image: np.ndarray,
    spacing: list[float],
) -> None:
    """Run the standard fsc_resolution and dcr_resolution functions and compare results."""
    print("\n" + "=" * 80)
    print("STANDARD RESOLUTION FUNCTION COMPARISON")
    print("=" * 80)

    print("\nRunning fsc_resolution()...")
    try:
        fsc_result = fsc_resolution(
            image,
            spacing=spacing,
            resample_isotropic=True,
            bin_delta=1,
            angle_delta=15,
        )
        print(f"  FSC XY resolution: {fsc_result['xy']:.4f} μm")
        print(f"  FSC Z resolution:  {fsc_result['z']:.4f} μm")
        if fsc_result["z"] > fsc_result["xy"]:
            print("  ✓ Z > XY: Correct")
        else:
            print("  ✗ Z < XY: WRONG!")
    except Exception as e:
        print(f"  FSC failed: {e}")

    print("\nRunning dcr_resolution()...")
    try:
        dcr_result = dcr_resolution(
            image,
            spacing=spacing,
        )
        print(f"  DCR XY resolution: {dcr_result['xy']:.4f} μm")
        print(f"  DCR Z resolution:  {dcr_result['z']:.4f} μm")
        if dcr_result["z"] > dcr_result["xy"]:
            print("  ✓ Z > XY: Correct")
        else:
            print("  ✗ Z < XY: WRONG!")
    except Exception as e:
        print(f"  DCR failed: {e}")


def main():
    """Run debug investigation."""
    print("=" * 80)
    print("DEBUG: XY/Z RESOLUTION SWAP INVESTIGATION")
    print("=" * 80)

    # Use pollen data or astrocyte data
    data_dir = Path(__file__).parent.parent / "data"

    # Try pollen data first (40x_TAGoff_z_galvo.nd2)
    pollen_path = data_dir / "40x_TAGoff_z_galvo.nd2"
    astrocyte_path = data_dir / "astr_vpa_hoechst.tif"

    if pollen_path.exists():
        image_path = pollen_path
        # Pollen data spacing from Koho et al. 2019
        spacing = [0.250, 0.0777, 0.0777]  # [Z, Y, X] in μm
        print(f"Using pollen data: {image_path}")
    elif astrocyte_path.exists():
        image_path = astrocyte_path
        # Astrocyte data spacing
        spacing = [0.3, 0.1625, 0.1625]  # [Z, Y, X] in μm
        print(f"Using astrocyte data: {image_path}")
    else:
        print("No test data found. Creating synthetic test data...")
        # Create synthetic 3D data
        np.random.seed(42)
        image = np.random.randn(30, 128, 128).astype(np.float32)
        spacing = [0.3, 0.1, 0.1]  # Typical confocal spacing

        # Run debug functions
        print(f"Spacing: {spacing} μm")
        debug_fsc_sectors(image, spacing)
        debug_dcr_sectors(image, spacing)
        run_standard_resolution_functions(image, spacing)
        return

    # Load image
    print(f"\nLoading image from: {image_path}")
    image = load_test_image(image_path)
    print(f"Image shape: {image.shape}")
    print(f"Spacing (Z, Y, X): {spacing} μm")

    # Transfer to GPU if available
    try:
        image = ascupy(image)
        print(f"Image device: {get_device(image)}")
    except Exception:
        print("GPU not available, using CPU")

    # Take a center crop for faster processing
    crop_size = 128
    center = [s // 2 for s in image.shape]
    half_z = min(15, image.shape[0] // 2)
    half_xy = crop_size // 2

    image_crop = image[
        center[0] - half_z : center[0] + half_z,
        center[1] - half_xy : center[1] + half_xy,
        center[2] - half_xy : center[2] + half_xy,
    ]

    print(f"\nUsing center crop: {image_crop.shape}")

    # Run debug functions
    debug_fsc_sectors(asnumpy(image_crop), spacing)
    debug_dcr_sectors(asnumpy(image_crop), spacing)
    run_standard_resolution_functions(asnumpy(image_crop), spacing)


if __name__ == "__main__":
    main()
