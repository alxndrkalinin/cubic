"""Plotting utilities for FRC/FSC/DCR resolution curves.

Requires matplotlib (install with ``pip install cubic[plot]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .analysis import (
    FourierCorrelationData,
    FourierCorrelationAnalysis,
    FourierCorrelationDataCollection,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def _import_matplotlib():
    """Import matplotlib, raising a clear error if unavailable."""
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting functions. "
            "Install it with: pip install cubic[plot]"
        ) from None


def plot_fsc_sectors(
    fsc_data: dict[int, FourierCorrelationData],
    spacing_iso: float,
    *,
    threshold: str = "one-bit",
    curve_fit_type: str = "smooth-spline",
    smoothing_factor: float = 0.2,
    axes: tuple[Axes, Axes] | None = None,
) -> Figure | None:
    """Plot FSC correlation curves for XY and Z sectors.

    Parameters
    ----------
    fsc_data : dict
        Per-sector FSC data from ``_calculate_fsc_sectioned_hist``.
    spacing_iso : float
        Isotropic spacing in physical units (used for resolution calculation).
    threshold : str
        Threshold type for resolution curve ("one-bit", "half-bit", "fixed").
    curve_fit_type : str
        Curve fitting method ("spline", "smooth-spline", "polynomial").
    smoothing_factor : float
        Smoothing factor for smooth-spline fitting.
    axes : tuple of two Axes, optional
        If provided, plot onto these axes. Otherwise create a new figure.

    Returns
    -------
    Figure or None
        The created figure, or None if axes were provided.
    """
    plt = _import_matplotlib()

    angles = sorted(fsc_data.keys())
    angle_xy = max(angles)
    angle_z = min(angles)

    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    for ax, angle, label in [
        (axes[0], angle_xy, "XY"),
        (axes[1], angle_z, "Z"),
    ]:
        data = fsc_data[angle]
        freq = np.asarray(data.correlation["frequency"])
        corr = np.asarray(data.correlation["correlation"])
        n_points = np.asarray(data.correlation["points-x-bin"])

        # Run analysis to get threshold curve and resolution
        coll = FourierCorrelationDataCollection()
        ds = FourierCorrelationData()
        ds.correlation["correlation"] = corr
        ds.correlation["frequency"] = freq
        ds.correlation["points-x-bin"] = n_points
        coll[0] = ds

        analyzer = FourierCorrelationAnalysis(
            coll,
            spacing_iso,
            resolution_threshold=threshold,
            curve_fit_type=curve_fit_type,
            smoothing_factor=smoothing_factor,
        )
        analyzed = None
        try:
            analyzed = analyzer.execute()[0]
            res_val = analyzed.resolution["resolution"]
            thr_curve = analyzed.resolution["threshold"]
        except Exception:
            res_val = np.nan
            thr_curve = None

        ax.plot(freq, corr, "-", color="steelblue", linewidth=0.75, label="FSC")
        if analyzed is not None and "curve-fit" in analyzed.correlation.keys:
            curve_fit = np.asarray(analyzed.correlation["curve-fit"])
            ax.plot(
                freq,
                curve_fit,
                ":",
                color="steelblue",
                linewidth=0.75,
                alpha=0.7,
                label="fit",
            )
        if thr_curve is not None:
            ax.plot(freq, thr_curve, "--", color="gray", linewidth=1, label=threshold)
        if np.isfinite(res_val) and analyzed is not None:
            cross_freq = analyzed.resolution["resolution-point"][1]
            ax.axvline(
                x=cross_freq,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"f_c = {cross_freq:.3f}",
            )
        ax.set_title(f"{label} sector ({angle}\u00b0)")

        # Overlay intermediate sectors as light background curves
        cmap = plt.cm.coolwarm
        norm = plt.Normalize(0, 90)
        for other_angle in angles:
            if other_angle in (angle_xy, angle_z):
                continue
            other = fsc_data[other_angle]
            ax.plot(
                np.asarray(other.correlation["frequency"]),
                np.asarray(other.correlation["correlation"]),
                "-",
                color=cmap(norm(other_angle)),
                linewidth=0.5,
                alpha=0.3,
            )

        ax.set_xlabel("Normalized frequency")
        ax.set_ylabel("FSC")
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    return fig


def plot_dcr_sectors(
    sector_data: dict[str, dict],
    *,
    axes: tuple[Axes, Axes] | None = None,
) -> Figure | None:
    """Plot DCR decorrelation curves for XY and Z sectors.

    Parameters
    ----------
    sector_data : dict
        Output of :func:`dcr_curve_3d_sectioned`. Keys ``"xy"`` / ``"z"``,
        each containing ``"radii"``, ``"curves"``, ``"peaks"``, ``"k_max"``,
        ``"resolution"``.
    axes : tuple of two Axes, optional
        If provided, plot onto these axes. Otherwise create a new figure.

    Returns
    -------
    Figure or None
        The created figure, or None if axes were provided.
    """
    plt = _import_matplotlib()

    fig = None
    if axes is None:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    colors = {"xy": "steelblue", "z": "darkorange"}
    titles = {"xy": "XY sector", "z": "Z sector"}

    for ax, sector in zip(axes, ["xy", "z"]):
        sd = sector_data[sector]
        radii = np.asarray(sd["radii"])
        curves = sd["curves"]
        peaks = np.asarray(sd["peaks"])
        resolution = sd["resolution"]
        n = len(curves)

        for i, d_curve in enumerate(curves):
            alpha = 0.2 + 0.8 * (i / max(n - 1, 1))
            ax.plot(radii, d_curve, "-", color=colors[sector], linewidth=1, alpha=alpha)

        # Mark k_c from pre-computed peaks
        valid_peaks = peaks[peaks[:, 0] > 0]
        if len(valid_peaks) > 0:
            k_c_norm = float(np.max(valid_peaks[:, 0]))
            ax.axvline(
                x=k_c_norm,
                color="red",
                linestyle="--",
                alpha=0.7,
                label=f"k_c = {k_c_norm:.3f}",
            )
            ax.set_title(f"{titles[sector]} \u2014 {resolution:.4f} \u00b5m")
        else:
            ax.set_title(f"{titles[sector]} \u2014 no peak")

        ax.set_xlabel("Normalized frequency")
        ax.set_ylabel("Decorrelation")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    return fig


def plot_frc_curve(
    frc_data: FourierCorrelationData,
    ax: Axes,
) -> None:
    """Plot FRC correlation curve with threshold and resolution crossing.

    Parameters
    ----------
    frc_data : FourierCorrelationData
        Analyzed FRC result from :func:`calculate_frc`.
    ax : Axes
        Matplotlib axes to plot on.
    """
    _import_matplotlib()

    freq = np.asarray(frc_data.correlation["frequency"])
    corr = np.asarray(frc_data.correlation["correlation"])
    thr = np.asarray(frc_data.resolution["threshold"])
    res = frc_data.resolution["resolution"]

    ax.plot(freq, corr, "-", color="steelblue", linewidth=0.75, label="FRC")
    if "curve-fit" in frc_data.correlation.keys:
        curve_fit = np.asarray(frc_data.correlation["curve-fit"])
        ax.plot(
            freq,
            curve_fit,
            ":",
            color="steelblue",
            linewidth=0.75,
            alpha=0.7,
            label="fit",
        )
    ax.plot(freq, thr, "--", color="gray", linewidth=1, label="fixed (1/7)")

    if np.isfinite(res):
        cross_freq = frc_data.resolution["resolution-point"][1]
        ax.axvline(
            x=cross_freq,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"f_c = {cross_freq:.3f}",
        )
        ax.set_title(f"FRC \u2014 {res * 1000:.1f} nm")
    else:
        ax.set_title("FRC \u2014 no crossing")

    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("FRC")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_dcr_curves(
    radii_refined: np.ndarray,
    all_curves: list[np.ndarray],
    all_peaks: np.ndarray,
    ax: Axes,
    num_highpass: int = 10,
) -> None:
    """Plot DCR decorrelation curves with correct x-axes.

    ``dcr_curve(refine=True)`` returns coarse curves (x in 0..1) followed by
    refined curves (x in the narrowed range given by *radii_refined*).

    Parameters
    ----------
    radii_refined : ndarray
        Radii array returned by :func:`dcr_curve`.
    all_curves : list of ndarray
        All decorrelation curves (coarse + refined).
    all_peaks : ndarray
        Peak positions array from :func:`dcr_curve`.
    ax : Axes
        Matplotlib axes to plot on.
    num_highpass : int
        Number of coarse (high-pass) curves before refined curves begin.
    """
    _import_matplotlib()

    n_coarse = num_highpass
    n_total = len(all_curves)
    has_refined = n_total > n_coarse
    num_points = len(all_curves[0])

    # Coarse curves span full 0..1 range
    radii_coarse = np.linspace(0, 1, num_points, dtype=np.float32)

    for i, d_curve in enumerate(all_curves):
        is_refined = has_refined and i >= n_coarse
        r = radii_refined if is_refined else radii_coarse
        alpha = 0.2 + 0.8 * (i / max(n_total - 1, 1))
        clr = "darkorange" if is_refined else "steelblue"
        ax.plot(r, d_curve, "-", color=clr, linewidth=1, alpha=alpha)

    # Find the overall peak (max r among all peaks)
    valid_peaks = all_peaks[all_peaks[:, 0] > 0]
    if len(valid_peaks) > 0:
        k_c_norm = float(np.max(valid_peaks[:, 0]))
        ax.axvline(
            x=k_c_norm,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"k_c = {k_c_norm:.3f}",
        )

    ax.set_xlabel("Normalized frequency")
    ax.set_ylabel("Decorrelation")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
