"""Compare FRC and DCR resolution metrics during deconvolution.

This script runs Richardson-Lucy deconvolution and computes both FRC and DCR
resolution metrics at each iteration, allowing direct comparison of the two
methods on identical intermediate images.

Usage:
    python compare_frc_dcr_deconv.py --image path/to/image.tif --psf path/to/psf.tif
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

from cubic.cuda import ascupy, asnumpy
from cubic.skimage import util
from cubic.metrics.spectral import dcr_resolution, frc_resolution
from cubic.preprocessing.deconvolution import richardson_lucy_iter


def compute_both_resolutions(
    image: np.ndarray,
    *,
    spacing: float | None = None,
    frc_kwargs: dict | None = None,
    dcr_kwargs: dict | None = None,
) -> dict:
    """Compute both FRC and DCR resolution for a 2D image.

    Parameters
    ----------
    image : np.ndarray
        2D image (or middle slice of 3D)
    spacing : float, optional
        Pixel spacing in physical units
    frc_kwargs : dict, optional
        Additional arguments for frc_resolution
    dcr_kwargs : dict, optional
        Additional arguments for dcr_resolution

    Returns
    -------
    dict with 'frc' and 'dcr' resolution values

    Notes
    -----
    Single-image FRC uses checkerboard splitting which can be noisy.
    Use curve_fit_type="spline" (not "smooth-spline") for more stable results.
    """
    frc_kwargs = frc_kwargs or {}
    dcr_kwargs = dcr_kwargs or {}

    # Compute FRC (single-image mode uses checkerboard split)
    # Note: curve_fit_type="spline" is more stable than "smooth-spline"
    frc_res = frc_resolution(image, spacing=spacing, **frc_kwargs)

    # Compute DCR (single-image, no split needed)
    dcr_res = dcr_resolution(image, spacing=spacing, **dcr_kwargs)

    return {"frc": frc_res, "dcr": dcr_res}


def run_deconv_with_dual_metrics(
    image: np.ndarray,
    psf: np.ndarray,
    *,
    n_iter: int = 25,
    spacing: float | None = None,
    use_gpu: bool = True,
    frc_kwargs: dict | None = None,
    dcr_kwargs: dict | None = None,
    verbose: bool = True,
) -> dict:
    """Run deconvolution tracking both FRC and DCR at each iteration.

    Parameters
    ----------
    image : np.ndarray
        Input 3D image (ZYX)
    psf : np.ndarray
        Point spread function
    n_iter : int
        Number of iterations
    spacing : float, optional
        XY pixel spacing in physical units (e.g., microns)
    use_gpu : bool
        Whether to use GPU acceleration
    frc_kwargs : dict, optional
        Additional arguments for FRC calculation
    dcr_kwargs : dict, optional
        Additional arguments for DCR calculation
    verbose : bool
        Print progress

    Returns
    -------
    dict with iteration results including both metrics
    """
    # Default FRC kwargs: use "spline" curve_fit_type for stability (not "smooth-spline")
    frc_kwargs = frc_kwargs or {
        "bin_delta": 3,
        "backend": "hist",
        "curve_fit_type": "spline",
    }
    # Default DCR kwargs
    dcr_kwargs = dcr_kwargs or {"num_radii": 50, "num_highpass": 10}

    # Prepare image
    image = util.img_as_float(image)
    psf = util.img_as_float(psf)

    if use_gpu:
        image = ascupy(image)
        psf = ascupy(psf)

    # Storage for results
    results = []
    mid_z = image.shape[0] // 2

    # Compute initial resolution
    init_slice = asnumpy(image[mid_z])
    init_res = compute_both_resolutions(
        init_slice, spacing=spacing, frc_kwargs=frc_kwargs, dcr_kwargs=dcr_kwargs
    )
    results.append(
        {
            "iteration": 0,
            "frc_resolution": init_res["frc"],
            "dcr_resolution": init_res["dcr"],
            "frc_improvement": 0.0,
            "dcr_improvement": 0.0,
        }
    )

    if verbose:
        unit = "µm" if spacing else "px"
        print(
            f"Initial: FRC={init_res['frc']:.4f} {unit}, DCR={init_res['dcr']:.4f} {unit}"
        )

    # Observer function
    def dual_metric_observer(restored_image, iteration):
        mid_slice = asnumpy(restored_image[mid_z])

        res = compute_both_resolutions(
            mid_slice, spacing=spacing, frc_kwargs=frc_kwargs, dcr_kwargs=dcr_kwargs
        )

        # Compute improvement (negative = better resolution)
        frc_imp = res["frc"] - results[-1]["frc_resolution"]
        dcr_imp = res["dcr"] - results[-1]["dcr_resolution"]

        results.append(
            {
                "iteration": iteration,
                "frc_resolution": res["frc"],
                "dcr_resolution": res["dcr"],
                "frc_improvement": frc_imp,
                "dcr_improvement": dcr_imp,
            }
        )

        if verbose:
            # Convert to nm if spacing provided
            scale = 1000 if spacing else 1
            unit = "nm" if spacing else "px"
            print(
                f"Iter {iteration:2d}: "
                f"FRC={res['frc']:.4f} (Δ={frc_imp * scale:+.1f} {unit}), "
                f"DCR={res['dcr']:.4f} (Δ={dcr_imp * scale:+.1f} {unit})"
            )

    # Run deconvolution
    _ = richardson_lucy_iter(
        image,
        psf,
        n_iter=n_iter,
        implementation="xp",
        observer_fn=dual_metric_observer,
        pad_size_z=1,
    )

    return {
        "results": results,
        "spacing": spacing,
        "frc_kwargs": frc_kwargs,
        "dcr_kwargs": dcr_kwargs,
    }


def plot_comparison(data: dict, output_path: Path | None = None):
    """Plot FRC vs DCR comparison."""
    results = data["results"]
    spacing = data["spacing"]

    iterations = [r["iteration"] for r in results]
    frc_res = [r["frc_resolution"] for r in results]
    dcr_res = [r["dcr_resolution"] for r in results]
    frc_imp = [r["frc_improvement"] for r in results[1:]]
    dcr_imp = [r["dcr_improvement"] for r in results[1:]]

    # Convert to nm if spacing provided
    scale = 1000 if spacing else 1
    unit = "nm" if spacing else "pixels"

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top left: Absolute resolution
    ax = axes[0, 0]
    ax.plot(iterations, [r * scale for r in frc_res], "o-", label="FRC", linewidth=2)
    ax.plot(iterations, [r * scale for r in dcr_res], "s-", label="DCR", linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(f"Resolution ({unit})")
    ax.set_title("Resolution vs Iteration")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Lower is better

    # Top right: Per-iteration improvement
    ax = axes[0, 1]
    ax.plot(
        iterations[1:], [i * scale for i in frc_imp], "o-", label="FRC", linewidth=2
    )
    ax.plot(
        iterations[1:], [i * scale for i in dcr_imp], "s-", label="DCR", linewidth=2
    )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Iteration")
    ax.set_ylabel(f"Resolution change ({unit})")
    ax.set_title("Per-iteration Improvement (negative = better)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom left: FRC vs DCR scatter
    ax = axes[1, 0]
    ax.scatter(
        [r * scale for r in frc_res],
        [r * scale for r in dcr_res],
        c=iterations,
        cmap="viridis",
        s=100,
    )
    min_val = min(min(frc_res), min(dcr_res)) * scale
    max_val = max(max(frc_res), max(dcr_res)) * scale
    ax.plot([min_val, max_val], [min_val, max_val], "k--", alpha=0.5, label="y=x")
    ax.set_xlabel(f"FRC Resolution ({unit})")
    ax.set_ylabel(f"DCR Resolution ({unit})")
    ax.set_title("FRC vs DCR Resolution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Iteration")

    # Bottom right: Correlation of improvements
    ax = axes[1, 1]
    ax.scatter(
        [i * scale for i in frc_imp],
        [i * scale for i in dcr_imp],
        c=iterations[1:],
        cmap="viridis",
        s=100,
    )
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3)
    ax.set_xlabel(f"FRC Improvement ({unit})")
    ax.set_ylabel(f"DCR Improvement ({unit})")
    ax.set_title("FRC vs DCR Improvement Correlation")
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label("Iteration")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
    else:
        plt.show()


def print_summary(data: dict):
    """Print summary statistics."""
    results = data["results"]
    spacing = data["spacing"]

    scale = 1000 if spacing else 1
    unit = "nm" if spacing else "pixels"

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    # Initial and final resolutions
    init = results[0]
    final = results[-1]

    print(f"\nInitial resolution:")
    print(f"  FRC: {init['frc_resolution'] * scale:.1f} {unit}")
    print(f"  DCR: {init['dcr_resolution'] * scale:.1f} {unit}")

    print(f"\nFinal resolution (iter {final['iteration']}):")
    print(f"  FRC: {final['frc_resolution'] * scale:.1f} {unit}")
    print(f"  DCR: {final['dcr_resolution'] * scale:.1f} {unit}")

    # Total improvement
    frc_total = (final["frc_resolution"] - init["frc_resolution"]) * scale
    dcr_total = (final["dcr_resolution"] - init["dcr_resolution"]) * scale

    print(f"\nTotal improvement:")
    print(
        f"  FRC: {frc_total:+.1f} {unit} ({frc_total / init['frc_resolution'] / scale * 100:+.1f}%)"
    )
    print(
        f"  DCR: {dcr_total:+.1f} {unit} ({dcr_total / init['dcr_resolution'] / scale * 100:+.1f}%)"
    )

    # Correlation between metrics
    frc_res = np.array([r["frc_resolution"] for r in results])
    dcr_res = np.array([r["dcr_resolution"] for r in results])
    corr = np.corrcoef(frc_res, dcr_res)[0, 1]

    print(f"\nCorrelation between FRC and DCR: {corr:.4f}")


def main():  # noqa: D103
    parser = argparse.ArgumentParser(
        description="Compare FRC and DCR during deconvolution"
    )
    parser.add_argument(
        "--image", type=Path, required=True, help="Path to input image (TIFF)"
    )
    parser.add_argument("--psf", type=Path, required=True, help="Path to PSF (TIFF)")
    parser.add_argument("--n-iter", type=int, default=25, help="Number of iterations")
    parser.add_argument(
        "--spacing", type=float, default=None, help="XY pixel spacing (e.g., 0.1625 µm)"
    )
    parser.add_argument("--output", type=Path, default=None, help="Output plot path")
    parser.add_argument(
        "--no-gpu", action="store_true", help="Disable GPU acceleration"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    # Load data
    print(f"Loading image: {args.image}")
    image = imread(args.image)
    print(f"  Shape: {image.shape}, dtype: {image.dtype}")

    print(f"Loading PSF: {args.psf}")
    psf = imread(args.psf)
    print(f"  Shape: {psf.shape}, dtype: {psf.dtype}")

    # Run comparison
    print(f"\nRunning {args.n_iter} iterations of Richardson-Lucy deconvolution...")
    print("Computing both FRC and DCR at each iteration\n")

    data = run_deconv_with_dual_metrics(
        image,
        psf,
        n_iter=args.n_iter,
        spacing=args.spacing,
        use_gpu=not args.no_gpu,
        verbose=not args.quiet,
    )

    # Print summary
    print_summary(data)

    # Plot
    plot_comparison(data, args.output)


if __name__ == "__main__":
    main()
