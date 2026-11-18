"""Extract features from a label image using ``trimesh``.

Each label is converted into a surface mesh with marching cubes and measured
with ``trimesh``. A CuPy label image or mask stays on the GPU: it is meshed by a
CUDA port of the same marching cubes and measured as an ``ArrayMesh``, which
computes the trimesh quantities on the device and hands the convex hull and
bounding primitives to trimesh on the host.
"""

from __future__ import annotations

import sys
import numbers
import warnings
from typing import Any, TypeVar, Callable, SupportsFloat
from functools import wraps
from collections.abc import Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import QhullError
from skimage.measure import marching_cubes

try:  # pragma: no cover - optional dependency
    import trimesh
except ImportError:  # pragma: no cover - import failure path
    trimesh = None  # type: ignore[assignment]
    warnings.warn(
        "trimesh is not installed. Mesh features are unavailable.",
        stacklevel=2,
    )

from ..cuda import asnumpy, get_device, to_same_device
from ._array_mesh import ArrayMesh, face_areas
from ..image_utils import gaussian_filter_like_scipy
from ._marching_cubes_cuda import marching_cubes as _marching_cubes_gpu

F = TypeVar("F", bound=Callable[..., Any])

# ``Trimesh.integral_mean_curvature`` signs its dihedral angles only from this
# release on; older ones overstate the mean curvature of any concave surface.
_MIN_TRIMESH = (4, 6, 12)


def _check_trimesh_version(version: str) -> None:
    """Raise ``ImportError`` for a trimesh release older than ``_MIN_TRIMESH``."""
    parts = version.split(".")[:3]
    # A part that is not a plain number (a pre-release tag) cannot be compared.
    if all(part.isdigit() for part in parts) and tuple(map(int, parts)) < _MIN_TRIMESH:
        raise ImportError(
            f"cubic.feature.mesh requires trimesh>={'.'.join(map(str, _MIN_TRIMESH))}"
            f"; found {version}. Install with 'cubic[mesh]'"
        )


# trimesh reports no version when its package metadata is missing.
if trimesh is not None and trimesh.__version__ is not None:
    _check_trimesh_version(trimesh.__version__)


def _require_trimesh(func: F) -> F:
    """Ensure ``trimesh`` is available before calling ``func``."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if trimesh is None:
            raise RuntimeError(
                "trimesh is required for mesh operations. Install with 'cubic[mesh]'"
            )
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def mesh_feature_list(optional: bool = False) -> list[str]:
    """Return the names of the surface features of a label, in column order.

    ``extract_features`` and ``extract_surface_features`` return the default
    features. With ``optional=True`` the optional ones follow them; those are
    computed only when requested by name (``features=[...]``). All but
    ``Fractal Dimension`` and ``Outline Fractal Dimension``, which are measured
    on the mask at several blur scales, are measured on a mesh of it, so
    ``extract_mesh_features`` cannot compute those two.

    No default feature is defined on the image axes. The axis lengths are the
    full axes of the ellipsoid with the same inertia, and ``Extent`` is the
    volume over the oriented bounding box volume. Voxelization still moves
    every feature a little with orientation, and the oriented box and bounding
    cylinder come from trimesh's approximate minimum searches, so they move
    more: ``Extent`` spans 0.50-0.55 for one ellipsoid turned by 0, 30 and 45
    degrees. The mean curvature is averaged over the surface, in inverse units
    of ``spacing``. ``Euler Characteristic`` is twice the number of separate
    pieces plus cavities, minus twice the number of handles: 2 for a ball, 4
    for a hollow ball, 0 for a torus.

    Three of the last four features need curvature at each vertex, which on a
    voxel staircase measures the staircase, so they are computed on a mesh of
    the mask blurred at ``curvature_sigma`` (see ``extract_surface_features``).
    ``Total Absolute Gaussian Curvature`` is the integral of ``|K|`` over the
    surface divided by ``4 pi``: 1 for any convex body and more for every saddle
    region (grooves, waists, the necks of lobes), unitless. ``Avg Shape Index``
    (Koenderink and van Doorn, 1992) runs from -1 (cup) through 0 (saddle) to 1
    (cap), and ``Avg Curvedness`` is ``sqrt((k1^2 + k2^2) / 2)``, in inverse
    units of ``spacing``; both are averaged over the surface by area.
    ``Euler Characteristic``, ``Avg Mean Curvature`` and these three are
    approximate on a mesh with edges shared by more than two faces, which
    marching cubes makes where voxels touch only along an edge, and a warning
    says so when one is requested.
    ``Fractal Dimension`` describes the surface detail finer than
    ``curvature_sigma``, which those three leave out: ``2 - slope`` of log
    surface area against log blur scale, relative to the mask's smooth twin
    (its blur at the top scale), over the window ``fractal_scales`` of
    ``extract_surface_features``. It is about 2 for a surface without such
    detail, whatever its size and coarse shape, and rises with roughness, as
    long as the top scale is at most a quarter of the object's thickness
    (``Min Axis Length``). Above that it reads low, and a warning says so.

    ``Outline Fractal Dimension``, the one optional feature, is the same
    construction on the outline of the mask's projection along the first axis:
    ``1 - slope`` of log contour length of the blurred footprint against log
    blur scale, relative to the footprint's smooth twin, with the blur isotropic
    in the in-plane spacing, over the window ``outline_fractal_scales`` of
    ``extract_surface_features``. It is about 1 for a smooth outline and rises
    with outline roughness, and it does not read low for flat objects. It is
    defined on the image axes, and meant for adherent, flat objects imaged
    along the first axis: turning an object about that axis leaves it nearly
    unchanged, tilting it does not. Its footprint must be wide against the top
    scale: a disc narrower than about 3.2 times the top scale gives NaN, with a
    warning, because its smooth twin vanishes, and a smooth disc reads 0.95 at
    4 and 0.99 at 5 times the top scale across.
    """
    if optional:
        return [*_MESH_FEATURES, *_MASK_FEATURES]
    return [
        *_MESH_FEATURES,
        *(name for name in _MASK_FEATURES if name not in _OPTIONAL_FEATURES),
    ]


def _resolve_features(features: Sequence[str] | None) -> list[str]:
    """Return ``features``, or the default ones for ``None``; raise for unknown names.

    An explicit list may name default and optional features.
    """
    if features is None:
        return mesh_feature_list()
    # A string is a sequence too, and would be checked letter by letter.
    if isinstance(features, str):
        raise TypeError(
            f"features must be a sequence of names, not the string {features!r}"
        )
    # Listed first: checking the names would exhaust a one-shot iterable.
    names = list(features)
    known = mesh_feature_list(optional=True)
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(f"Unknown mesh features {unknown}; available: {known}")
    return names


@_require_trimesh
def extract_features(
    label_image: np.ndarray,
    features: list[str] | None = None,
    spacing: Sequence[float] | None = None,
    curvature_sigma: float | None = None,
    fractal_scales: Sequence[float] | None = None,
    outline_fractal_scales: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract mesh features for every label of a label image.

    Each label is meshed independently over its own bounding box, which is
    orders of magnitude cheaper than running marching cubes over the full volume
    once per label.

    Parameters
    ----------
    label_image : np.ndarray
        3D integer label image, NumPy or CuPy. Background (label 0) is
        excluded. A CuPy image is meshed and measured on the GPU.
    features : list of str, optional
        Feature names to extract, defaults to ``mesh_feature_list()``; the
        columns follow this order. Only these features are computed. Optional
        features (``mesh_feature_list(optional=True)``, e.g. ``Outline Fractal
        Dimension``) are computed only when named here.
    spacing : sequence of float, optional
        Voxel spacing ``(z, y, x)``, finite and positive, in any length unit;
        see ``extract_surface_features`` for how exactly the features convert
        with it. Without it an anisotropic stack is meshed as if isotropic,
        scaling volumes and areas by the spacing product.
    curvature_sigma : float, optional
        Blur scale, in units of ``spacing``, of the mesh behind the
        curvature-distribution features; see ``extract_surface_features``.
    fractal_scales : sequence of float, optional
        ``(lowest, highest)`` blur scale, in units of ``spacing``, of the
        ``Fractal Dimension`` fit; see ``extract_surface_features``.
    outline_fractal_scales : sequence of float, optional
        ``(lowest, highest)`` blur scale, in units of ``spacing``, of the
        ``Outline Fractal Dimension`` fit; see ``extract_surface_features``.

    Returns
    -------
    labels : np.ndarray
        Label of each row, background excluded.
    feature_values : np.ndarray
        ``(n_labels, n_features)`` matrix, ``(0, n_features)`` when there are no
        labels, so that it always stacks with results from other images.
    column_names : list of str
        Name of each column, in column order, as returned by
        ``voxel.extract_features``.
    """
    # Labels and bounding boxes come from a host copy; the crops are taken from
    # the input itself, so a CuPy image is meshed on the GPU.
    host_labels = asnumpy(label_image)
    if not np.issubdtype(host_labels.dtype, np.integer):
        raise ValueError(
            f"label_image must have an integer dtype; got {host_labels.dtype}"
        )
    # Keep only positive labels, matching ``voxel.regionprops``, which ignores
    # both the background and negative values.
    labels = np.unique(host_labels)
    labels = labels[labels > 0]

    # Resolved before any meshing, so an unknown name or a bad window fails
    # even on an image without labels.
    features = _resolve_features(features)
    _fractal_window(fractal_scales, "fractal_scales")
    _fractal_window(outline_fractal_scales, "outline_fractal_scales")
    _unit(spacing)
    _curvature_sigma(spacing, curvature_sigma)
    if not features:
        return labels, np.empty((len(labels), 0)), features
    # ``find_objects`` returns the bounding box slices indexed by ``label - 1``,
    # so each label is meshed over its own crop instead of the whole volume.
    bboxes = ndimage.find_objects(host_labels)
    feature_values = []
    for label in labels:
        mask = label_image[bboxes[label - 1]] == label
        mesh_features = extract_surface_features(
            mask,
            spacing=spacing,
            features=features,
            curvature_sigma=curvature_sigma,
            fractal_scales=fractal_scales,
            outline_fractal_scales=outline_fractal_scales,
        )
        feature_values.append([mesh_features[feat] for feat in features])

    feature_matrix = np.asarray(feature_values, dtype=float).reshape(
        len(labels), len(features)
    )
    return labels, feature_matrix, features


@_require_trimesh
def extract_surface_features(
    mask: np.ndarray,
    spacing: Sequence[float] | None = None,
    features: Sequence[str] | None = None,
    curvature_sigma: float | None = None,
    fractal_scales: Sequence[float] | None = None,
    outline_fractal_scales: Sequence[float] | None = None,
) -> dict[str, float]:
    """Generate mesh from a single label mask and extract its surface features.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask of a single object.
    spacing : sequence of float, optional
        Voxel spacing ``(z, y, x)``, finite and positive. The features are
        computed in units of the finest spacing and converted back, so they are
        in units of ``spacing``, in metres or nanometres alike. Given in
        another unit, they convert with it exactly (to the rounding of the
        conversion) when the ratios of the spacings round alike in both units,
        and otherwise to about 1e-8 relative; ``Bounding Cylinder Volume``
        comes from an approximate search and can then move by a few 1e-4.
    features : sequence of str, optional
        Feature names to compute, defaults to ``mesh_feature_list()``. Optional
        features (``mesh_feature_list(optional=True)``, e.g. ``Outline Fractal
        Dimension``) are computed only when named here.
    curvature_sigma : float, optional
        Gaussian blur scale, in units of ``spacing``, of the mask that the
        curvature-distribution features are measured on. Defaults to twice the
        coarsest voxel spacing, which removes the voxel staircase on every axis.
        It sets the scale of the shape being described: larger values keep only
        coarser folds. It is also the top of the default ``fractal_scales``.
        An object too small to survive the blur gets NaN for the features that
        depend on it, with a warning.
    fractal_scales : sequence of float, optional
        ``(lowest, highest)`` blur scale, in units of ``spacing``, of the
        ``Fractal Dimension`` fit, which uses 5 geometric scales between them
        and a smooth twin blurred at the highest. Both must be finite, with
        ``0 < lowest < highest``. Defaults to the finest voxel spacing up to
        ``curvature_sigma``, which gives NaN, with a warning, unless
        ``curvature_sigma`` exceeds the finest spacing. When the highest scale
        exceeds a quarter of the
        object's ``Min Axis Length``, the value is still returned, with a
        warning that it is biased low. Warnings are attributed to the line
        that called this function, or ``extract_features``, which is one
        line for every label. The thin-object text is the same for every
        object, and Python's default filter keys a text by the calling line,
        so it shows the text once per calling line: once for a label image,
        and once for a loop over images that calls ``extract_features`` on
        one line.
    outline_fractal_scales : sequence of float, optional
        ``(lowest, highest)`` blur scale, in units of ``spacing``, of the
        ``Outline Fractal Dimension`` fit, in the same form as
        ``fractal_scales``. Defaults to 4 to 16 times the coarser in-plane
        spacing. An outline that vanishes at these scales gives NaN, with a
        warning.

    Raises
    ------
    TypeError
        If ``fractal_scales`` or ``outline_fractal_scales`` is not a sequence
        of real numbers.
    ValueError
        If ``fractal_scales`` or ``outline_fractal_scales`` is not two finite
        numbers with ``0 < lowest < highest``, ``spacing`` is not three
        finite, positive numbers ``(z, y, x)``, or ``curvature_sigma`` is not
        finite and positive, whichever features are requested.
    """
    features = _resolve_features(features)
    window = _fractal_window(fractal_scales, "fractal_scales")
    outline_window = _fractal_window(outline_fractal_scales, "outline_fractal_scales")
    # Measured in units of the finest spacing, where trimesh's absolute
    # tolerances see only the spacing ratios, and converted back at the end.
    unit = _unit(spacing)
    step = _step(spacing) / unit
    # Validated in the caller's units, so that an error names their number, and
    # before any meshing, then normalised; the default, twice the coarsest
    # spacing, normalises to twice the coarsest normalised spacing.
    sigma = _curvature_sigma(spacing, curvature_sigma) / unit
    mesh = mask2mesh(mask, spacing=None if spacing is None else tuple(step))
    blurred = [n for n in features if n in _SMOOTHED_FEATURES or n in _MASK_FEATURES]
    on_mesh = [name for name in features if name not in _MASK_FEATURES]
    # Each fractal feature's (lowest, highest) blur scale, resolved once.
    fd_low, fd_high = (
        (float(step.min()), sigma)
        if window is None
        else (window[0] / unit, window[1] / unit)
    )
    in_plane = float(step[1:].max())
    outline_low, outline_high = (
        (4 * in_plane, 16 * in_plane)
        if outline_window is None
        else (outline_window[0] / unit, outline_window[1] / unit)
    )
    values: dict[str, float] = {}
    # An explicit window is validated, so only the default can be empty; it is
    # decided before any blur.
    if "Fractal Dimension" in features and fd_low >= fd_high:
        warnings.warn(
            f"Fractal Dimension needs curvature_sigma above the finest voxel "
            f"spacing {step.min() * unit:g}; got {sigma * unit:g}. Returning NaN.",
            stacklevel=_stacklevel_outside_package(),
        )
        values["Fractal Dimension"] = float("nan")

    # The mask's blur, padding and 0.5 level at a scale, each computed at most
    # once per call: the curvature features and Fractal Dimension's top scale
    # share it when both are curvature_sigma.
    blurs: dict[float, tuple[np.ndarray, np.ndarray, Any]] = {}

    def blur_at(scale: float) -> tuple[np.ndarray, np.ndarray, Any]:
        if scale not in blurs:
            volume, pad = _padded_blur(mask, step, scale)
            blurs[scale] = (volume, pad, _level_set(volume, step))
        return blurs[scale]

    # Features that need the blur at curvature_sigma, NaN together if the mask
    # vanishes there.
    at_sigma = [
        name
        for name in blurred
        if name not in values
        and (
            name in _SMOOTHED_FEATURES
            or (name == "Fractal Dimension" and fd_high == sigma)
        )
    ]
    smooth_mesh = None
    if at_sigma:
        _, pad, level_set = blur_at(sigma)
        if level_set is None:
            warnings.warn(
                f"Mask of {int(mask.sum())} voxels vanishes when blurred at "
                f"curvature_sigma={sigma * unit:g}; returning NaN for {at_sigma}.",
                stacklevel=_stacklevel_outside_package(),
            )
            values.update(dict.fromkeys(at_sigma, float("nan")))
        elif any(name in _SMOOTHED_FEATURES for name in features):
            smooth_mesh = _level_set_mesh(level_set, pad, step)
    values.update(
        _measure_mesh(
            mesh,
            [name for name in on_mesh if name not in values],
            smooth_mesh=smooth_mesh,
            unit=unit,
        )
    )
    if "Outline Fractal Dimension" in features:
        values["Outline Fractal Dimension"] = _outline_fractal_dimension(
            mask, step, outline_low, outline_high, unit=unit
        )
    if "Fractal Dimension" in features and "Fractal Dimension" not in values:
        values["Fractal Dimension"] = _fractal_dimension(
            mask, step, fd_low, fd_high, *blur_at(fd_high), unit=unit
        )
        # Half the shortest axis of the same-inertia ellipsoid is the smallest
        # half-extent; the estimate is unbiased only while the top scale stays
        # below half of it.
        if np.isfinite(values["Fractal Dimension"]) and fd_high > (
            float(_equivalent_ellipsoid_axes(mesh)[0]) / _THIN_OBJECT_RATIO
        ):
            # One text for every object. The default filter keys a text by the
            # calling line, so it reports a label image of many thin objects
            # once, and a loop over images that calls extract_features on one
            # line once.
            warnings.warn(
                _THIN_OBJECT_WARNING, stacklevel=_stacklevel_outside_package()
            )
    return _to_caller_units({name: values[name] for name in features}, unit)


# Fractal Dimension reads low once the object's thickness (Min Axis Length) is
# below this many times the top blur scale.
_THIN_OBJECT_RATIO = 4
_THIN_OBJECT_WARNING = (
    "Fractal Dimension: the 3D estimate is biased low for objects thinner than "
    f"{_THIN_OBJECT_RATIO} times the top blur scale of its fit (curvature_sigma, "
    "or the highest of fractal_scales); at least one object is that thin. For "
    "flat objects imaged along the first axis, see Outline Fractal Dimension."
)


def _fractal_window(
    scales: Sequence[float] | None, keyword: str
) -> tuple[float, float] | None:
    """Return ``(lowest, highest)`` of the window ``keyword``, or None for None.

    Raises ``TypeError``, naming ``keyword``, unless it is a sequence (or 1D
    array) of real numbers, and ``ValueError`` unless those are two finite
    numbers with ``0 < lowest < highest``.
    """
    if scales is None:
        return None
    # str and bytes are sequences, of characters and of byte values.
    is_array = isinstance(scales, np.ndarray) and scales.ndim == 1
    if isinstance(scales, (str, bytes)) or not (
        is_array or isinstance(scales, Sequence)
    ):
        raise TypeError(
            f"{keyword} must be a sequence of two numbers (lowest, highest); "
            f"got {scales!r}"
        )
    if not all(isinstance(s, numbers.Real) for s in scales):
        raise TypeError(f"{keyword} must hold real numbers; got {scales!r}")
    message = (
        f"{keyword} must be two finite numbers (lowest, highest) with "
        f"0 < lowest < highest; got {scales!r}"
    )
    if len(scales) != 2:
        raise ValueError(message)
    low, high = (float(s) for s in scales)
    # False for NaN and for either infinity, too.
    if not 0 < low < high < float("inf"):
        raise ValueError(message)
    return low, high


def _curvature_sigma(spacing: Sequence[float] | None, sigma: float | None) -> float:
    """Return ``sigma``, or twice the coarsest voxel spacing for ``None``."""
    if sigma is None:
        return 2 * (1.0 if spacing is None else float(max(spacing)))
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"curvature_sigma must be finite and positive; got {sigma}")
    return sigma


def _marching_cubes(
    volume: np.ndarray,
    level: float | None = None,
    *,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    allow_degenerate: bool = True,
    **kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """``(verts, faces)`` from skimage's Lewiner marching cubes, or its CUDA port.

    Takes the arguments of ``skimage.measure.marching_cubes``; ``level=None``
    is halfway between the volume's minimum and maximum. A CuPy volume is
    meshed on the GPU, which takes only ``level``, ``spacing`` and
    ``allow_degenerate``; other keywords go to skimage for a NumPy volume.

    The faces wind outward from the region above ``level``, so a closed
    surface encloses a positive volume. Both implementations wind them the
    other way for skimage's default ``gradient_direction="descent"``, so their
    columns are reversed; ``gradient_direction="ascent"`` (CPU only) already
    winds them outward and is returned as is. All faces turn together, so the
    shell of a cavity faces the material.
    """
    if get_device(volume) == "GPU":
        if kwargs:
            raise ValueError(f"GPU marching cubes does not support {sorted(kwargs)}.")
        verts, faces = _marching_cubes_gpu(volume, level, spacing, allow_degenerate)
    else:
        verts, faces, _, _ = marching_cubes(
            volume, level, spacing=spacing, allow_degenerate=allow_degenerate, **kwargs
        )
        if kwargs.get("gradient_direction") == "ascent":
            return verts, faces
    return verts, faces[:, ::-1].copy()


def _closed_mesh(verts: np.ndarray, faces: np.ndarray) -> Any:
    """Wrap outward-wound marching-cubes output as a mesh on its device."""
    if get_device(verts) == "GPU":
        return ArrayMesh(verts, faces)
    return trimesh.Trimesh(vertices=verts, faces=faces)


def _smooth_mask2mesh(
    mask_3d: np.ndarray, spacing: Sequence[float] | None, sigma: float
) -> Any:
    """Mesh the 0.5 level of the mask blurred at ``sigma``; None if it vanishes."""
    step = _step(spacing)
    volume, pad = _padded_blur(mask_3d, step, sigma)
    level_set = _level_set(volume, step)
    return None if level_set is None else _level_set_mesh(level_set, pad, step)


def _level_set_mesh(
    level_set: tuple[np.ndarray, np.ndarray], pad: np.ndarray, step: np.ndarray
) -> Any:
    """Return the closed mesh of a padded blur's level set, in the mask's frame."""
    verts, faces = level_set
    return _closed_mesh(verts - to_same_device(pad * step, verts), faces)


def _step(spacing: Sequence[float] | None) -> np.ndarray:
    return np.ones(3) if spacing is None else np.asarray(spacing, dtype=float)


def _unit(spacing: Sequence[float] | None) -> float:
    """Return the finest voxel spacing, the unit the features are measured in.

    1 for ``None``. Raises ``ValueError`` unless ``spacing`` is exactly three
    numbers ``(z, y, x)``, each finite and positive.
    """
    if spacing is None:
        return 1.0
    step = _step(spacing)
    if step.shape != (3,) or not np.all(np.isfinite(step) & (step > 0)):
        raise ValueError(
            f"spacing must be three finite, positive numbers (z, y, x); got {spacing!r}"
        )
    return float(step.min())


def _to_caller_units(values: dict[str, float], unit: float) -> dict[str, float]:
    """Convert features measured in units of ``unit`` to the caller's units."""
    return {name: v * unit ** _FEATURE_DIMENSIONS[name] for name, v in values.items()}


def _padded_blur(
    mask: np.ndarray, step: np.ndarray, sigma: float, tight: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mask blurred at ``sigma`` (units of ``step``) and its padding.

    The mask may have any number of axes, one ``step`` per axis. The blur is
    isotropic in physical units, so its width in voxels differs per axis on an
    anisotropic grid.

    ``tight`` pads by one voxel and blurs with zeros beyond it (``constant``
    mode) instead of padding past the blur's reach: the values on the mask's
    frame and that ring are bitwise the same, and every ring value is below
    0.5 (see ``_smooth_twin``), so the 0.5 level still closes inside the
    array, at a fraction of the cost of a wide blur.
    """
    sigma_voxels = sigma / step
    if tight:
        pad = np.ones(mask.ndim, dtype=int)
        mode = "constant"
    else:
        # Pad past the blur's reach so it neither clips at the crop nor touches
        # the border, which would leave the surface open.
        pad = np.ceil(3 * sigma_voxels).astype(int) + 1
        mode = "reflect"
    # Rounds like SciPy on the GPU too, so both devices mesh the same volume.
    volume = gaussian_filter_like_scipy(
        np.pad(mask.astype(np.float32), [(p, p) for p in pad]), sigma_voxels, mode
    )
    return volume, pad


def _level_set(
    volume: np.ndarray, step: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(verts, faces)`` of the 0.5 level of ``volume``, on its device.

    None if the volume never exceeds 0.5. Vertices are in the volume's frame,
    padding included.
    """
    if float(volume.max()) <= 0.5:
        return None
    return _marching_cubes(volume, 0.5, spacing=tuple(step), allow_degenerate=False)


def _level_set_area(level_set: tuple[np.ndarray, np.ndarray] | None) -> float:
    """Return the surface area of a level set from ``_level_set``; NaN for None."""
    if level_set is None:
        return float("nan")
    verts, faces = level_set
    return float(face_areas(verts.astype(np.float64), faces).sum())


@_require_trimesh
def mask2mesh(
    mask_3d: np.ndarray,
    spacing: Sequence[float] | None = None,
    marching_cubes_kwargs: dict[str, Any] | None = None,
    sigma: float | None = None,
) -> Any:
    """Convert 3D mask to a mesh using marching cubes.

    The mask is padded with one voxel of background on every side, which closes
    the surface of a label touching the volume border.

    Parameters
    ----------
    mask_3d : np.ndarray
        Boolean mask of a single object. A CuPy mask is meshed on the GPU and
        returned as an ``ArrayMesh``; a NumPy one as a ``trimesh.Trimesh``.
    spacing : sequence of float, optional
        Voxel spacing ``(z, y, x)`` passed to marching cubes, so that lengths,
        areas and volumes are in physical units. trimesh compares lengths with
        absolute tolerances of about 1e-8, so measure the mesh only in units
        where its edges are far above that (micrometres, not metres);
        ``extract_surface_features`` and ``extract_features`` do not have this
        limit.
    marching_cubes_kwargs : dict, optional
        Extra keyword arguments for ``skimage.measure.marching_cubes``; on the
        GPU only ``level``, ``spacing`` and ``allow_degenerate`` are supported.
    sigma : float, optional
        If given, mesh the mask blurred by a Gaussian of this width, in units of
        ``spacing``, instead of the voxel staircase. ``marching_cubes_kwargs``
        is then not used.

    Raises
    ------
    ValueError
        If ``sigma`` is given and the blurred mask never reaches 0.5, i.e. the
        object is too small for that blur, or if the voxel spacing, given as
        ``spacing`` or in ``marching_cubes_kwargs``, is not three finite,
        positive numbers ``(z, y, x)``.
    """
    # A negative spacing would turn the mesh inside out, a zero one flatten it.
    _unit(spacing)
    if sigma is not None:
        mesh = _smooth_mask2mesh(mask_3d, spacing, _curvature_sigma(spacing, sigma))
        if mesh is None:
            raise ValueError(f"The mask vanishes when blurred at sigma={sigma}.")
        return mesh
    marching_cubes_kwargs = dict(marching_cubes_kwargs or {})
    if spacing is not None:
        if "spacing" in marching_cubes_kwargs:
            raise ValueError(
                "Voxel spacing given both as 'spacing' and in "
                "'marching_cubes_kwargs'; pass it only once."
            )
        marching_cubes_kwargs["spacing"] = tuple(float(s) for s in spacing)
    else:
        _unit(marching_cubes_kwargs.get("spacing"))
    verts, faces = _marching_cubes(np.pad(mask_3d, 1), **marching_cubes_kwargs)
    # Undo the 1-voxel pad so vertices stay in the input's coordinate frame.
    # Without this every vertex, and therefore ``bounds`` and ``centroid``, sits
    # one voxel high on each axis -- scaled by ``spacing``, so the offset is in
    # physical units. Every feature in ``mesh_feature_list`` is
    # translation-invariant, but ``mask2mesh`` is public and its mesh is
    # expected to overlay the image.
    one_voxel = np.asarray(marching_cubes_kwargs.get("spacing", 1.0), dtype=float)
    verts = verts - to_same_device(one_voxel, verts)
    return _closed_mesh(verts, faces)


def sphericity(v: float, sa: float) -> float:
    """Return the Wadell sphericity of a body of volume ``v`` and surface area ``sa``.

    This is the surface area of the sphere of equal volume divided by ``sa``: 1
    for a sphere and smaller for any other shape.
    """
    return np.power(np.pi, 1 / 3) * np.power(6 * v, 2 / 3) / sa


@_require_trimesh
def extract_mesh_features(
    mesh: trimesh.Trimesh,
    features: Sequence[str] | None = None,
    smooth_mesh: trimesh.Trimesh | None = None,
) -> dict[str, float]:
    """Extract surface features from a ``trimesh`` object.

    ``Euler Characteristic`` and the four curvature features are approximate
    on a mesh with edges shared by more than two faces, and a warning is
    raised when one of them is requested and the mesh it is measured on has
    such edges.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Closed surface of a single object. trimesh compares lengths with
        absolute tolerances of about 1e-8, so the mesh should be in units where
        its edges are far above that (micrometres, not metres);
        ``extract_surface_features`` and ``extract_features`` do not have this
        limit.
    features : sequence of str, optional
        Feature names to compute, defaults to ``mesh_feature_list()`` without
        ``Fractal Dimension`` and ``Outline Fractal Dimension``, which are
        measured on the mask. Only these are computed.
    smooth_mesh : trimesh.Trimesh, optional
        Smoothed surface of the same object (e.g. ``mask2mesh(mask, sigma=...)``)
        for the curvature-distribution features. Without it they are measured on
        ``mesh``, which on a raw voxel mesh mostly measures the staircase.

    Raises
    ------
    ValueError
        If a feature name is unknown or needs the mask (``Fractal Dimension``,
        ``Outline Fractal Dimension``),
        or if the mesh volume or surface area is not finite and positive. A
        non-positive volume is what an inside-out mesh reports; an open surface
        is not detected in general, since its signed volume depends on the
        origin.
    """
    features = list(_MESH_FEATURES) if features is None else _resolve_features(features)
    from_mask = [name for name in features if name in _MASK_FEATURES]
    if from_mask:
        raise ValueError(
            f"{from_mask} is measured on the mask at several blur scales, not on "
            "one mesh; use extract_surface_features or extract_features."
        )
    return _measure_mesh(mesh, features, smooth_mesh=smooth_mesh)


def _measure_mesh(
    mesh: trimesh.Trimesh,
    features: Sequence[str],
    smooth_mesh: trimesh.Trimesh | None = None,
    unit: float = 1.0,
) -> dict[str, float]:
    """Return ``features`` of a mesh whose lengths are in units of ``unit``.

    The values stay in the mesh's units; ``unit`` only converts the volume or
    area printed by the errors to the caller's units.
    """
    # Validate the mesh before any feature: an inside-out mesh reports a
    # negative volume, which would silently propagate into a NaN sphericity and
    # negative solidity and extent instead of failing. An open surface is not
    # detected in general: the sign of its volume depends on the origin.
    volume = float(mesh.volume)
    surface_area = float(mesh.area)
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError(
            f"Mesh volume must be finite and positive; got {volume * unit**3}. "
            "The mesh is inside out or its surface is open "
            f"(is_watertight={mesh.is_watertight})."
        )
    if not np.isfinite(surface_area) or surface_area <= 0:
        raise ValueError(
            "Mesh surface area must be finite and positive; got "
            f"{surface_area * unit**2}."
        )

    curvature_mesh = mesh if smooth_mesh is None else smooth_mesh

    def measured_on(name: str) -> Any:
        return curvature_mesh if name in _SMOOTHED_FEATURES else mesh

    # The per-vertex curvatures behind both curvature averages, computed once
    # when either is requested. Kept by this call only, not on the mesh, which
    # may be the caller's own.
    curvatures = (
        _principal_curvatures(curvature_mesh)
        if any(name in _CURVATURE_AVERAGES for name in features)
        else None
    )

    def measure(name: str) -> SupportsFloat:
        if curvatures is not None and name in _CURVATURE_AVERAGES:
            return _CURVATURE_AVERAGES[name](*curvatures)
        return _MESH_FEATURES[name](measured_on(name))

    # trimesh caches volume, convex hull and bounding primitives on the mesh, so
    # features sharing them do not recompute them.
    values = {name: float(measure(name)) for name in features}
    # Only the meshes that a requested approximate feature was measured on are
    # checked, each once: keyed by identity, since a Trimesh hashes its content.
    checked = {
        id(measured_on(name)): measured_on(name)
        for name in features
        if name in _SHARED_EDGE_FEATURES
    }
    # The meshes whose face adjacency a requested feature has already built.
    with_adjacency = {
        id(measured_on(name)) for name in features if name in _ADJACENCY_FEATURES
    }
    if any(
        _overshared_edge_count(m, adjacency_measured=key in with_adjacency)
        for key, m in checked.items()
    ):
        warnings.warn(_SHARED_EDGE_WARNING, stacklevel=_stacklevel_outside_package())
    return values


def _overshared_edge_count(mesh: Any, adjacency_measured: bool = False) -> int:
    """Return the number of edges of ``mesh`` shared by more than two faces.

    An ``ArrayMesh`` counts them from its own edge grouping, on its device. A
    ``trimesh.Trimesh`` counts them from ``edges_unique_inverse``, which its
    ``euler_number`` caches. With ``adjacency_measured``, set when a feature
    has built the mesh's face adjacency, that adjacency is tested first: it
    pairs the two faces of each edge shared by exactly two, so it has 3 / 2
    pairs per face exactly when every edge has two faces, and the count is
    then 0.
    """
    if isinstance(mesh, ArrayMesh):
        return mesh.overshared_edge_count
    if adjacency_measured and 2 * len(mesh.face_adjacency) == 3 * len(mesh.faces):
        return 0
    return int((np.bincount(mesh.edges_unique_inverse) > 2).sum())


def _stacklevel_outside_package() -> int:
    """Return the ``stacklevel`` that names the first caller outside ``cubic``.

    For a ``warnings.warn`` in the function that calls this one: it skips that
    function and every frame of a ``cubic`` module above it (helpers, the
    feature registry, ``_require_trimesh``'s wrapper, ``extract_features``'
    per-label loop, another ``cubic`` module calling this one), so every
    warning of this module reaches the user's line through every public entry
    point. Through ``extract_features`` that is one line for every label. The
    default filter keys a text by the calling line, so it shows a constant text
    once per calling line: once for a label image, and once for a loop over
    images that calls ``extract_features`` on one line. When every frame to the
    top of the stack is a ``cubic`` frame, it is the level of the outermost one.
    Not for the import-time warning: ``warnings`` skips importlib's frames there
    and this count does not.
    """
    package = __name__.partition(".")[0]
    frame = sys._getframe(1)
    level = 1
    # "cubic" itself or "cubic.<anything>", not a module that merely starts so.
    while (
        frame is not None
        and frame.f_globals.get("__name__", "").partition(".")[0] == package
    ):
        frame = frame.f_back  # type: ignore[assignment]
        level += 1
    # Off the top of the stack every frame is cubic's, and one level past the
    # outermost would be attributed to "sys": name the outermost instead.
    return level if frame is not None else level - 1


def _bounding_cylinder_volume(mesh: trimesh.Trimesh) -> float:
    """Return the minimum bounding cylinder volume, 0 for degenerate meshes."""
    try:
        return mesh.bounding_cylinder.volume
    except (ValueError, QhullError) as e:
        # Degenerate meshes fail in the convex hull (QhullError) or in the
        # eigendecomposition of the inertia tensor (LinAlgError, a ValueError).
        warnings.warn(
            f"Unable extract bounding cylinder volume. Returning 0. Exception: {e}",
            stacklevel=_stacklevel_outside_package(),
        )
        return 0.0


def _equivalent_ellipsoid_axes(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return the full axis lengths, ascending, of the same-inertia ellipsoid.

    A solid ellipsoid of mass ``m`` and semi-axes ``a, b, c`` has principal
    moments ``m (b^2 + c^2) / 5`` and so on, so
    ``a^2 = 5 (I_b + I_c - I_a) / (2 m)``, the convention of regionprops'
    ``axis_major_length``. Unlike bounding box extents, they do not depend on
    the object's orientation.
    """
    moments = mesh.principal_inertia_components
    # (I_b + I_c - I_a) / 2 is the second moment along axis a, so it is never
    # negative; the clip only absorbs rounding for flat objects.
    second_moments = np.maximum(moments.sum() - 2 * moments, 0) / 2
    return np.sort(2 * np.sqrt(5 * second_moments / mesh.mass))


def _avg_mean_curvature(mesh: trimesh.Trimesh) -> float:
    """Return the surface average of the mean curvature, ``1 / R`` on a sphere.

    trimesh integrates it as half the sum of edge length times signed dihedral
    angle over the edges (Cohen-Steiner and Morvan, 2003).
    """
    return mesh.integral_mean_curvature / mesh.area


def _mixed_area(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return the mixed Voronoi area of each vertex (Meyer et al., 2003).

    Each triangle gives a vertex its Voronoi share, or half or a quarter of its
    area when the triangle is obtuse at or away from that vertex, so the areas
    sum to the mesh area.
    """
    corners = mesh.vertices[mesh.faces]
    angles = mesh.face_angles
    # Squared length of the edge opposite each corner.
    opposite = np.stack(
        [
            ((corners[:, (i + 1) % 3] - corners[:, (i + 2) % 3]) ** 2).sum(axis=1)
            for i in range(3)
        ],
        axis=1,
    )
    cot = 1 / np.tan(angles)
    nxt, prv = [1, 2, 0], [2, 0, 1]
    voronoi = (opposite[:, nxt] * cot[:, nxt] + opposite[:, prv] * cot[:, prv]) / 8
    face_area = mesh.area_faces[:, None]
    obtuse = angles > np.pi / 2
    area = np.where(
        obtuse,
        face_area / 2,
        np.where(obtuse.any(axis=1, keepdims=True), face_area / 4, voronoi),
    )
    return np.bincount(
        mesh.faces.ravel(), weights=area.ravel(), minlength=len(mesh.vertices)
    )


def _principal_curvatures(
    mesh: trimesh.Trimesh,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the per-vertex principal curvatures ``k1 <= k2`` and vertex areas.

    Gaussian curvature is the angle defect and mean curvature the edge sum of
    length times signed dihedral angle over four times the vertex area, as in
    LONI ShapeTools; ``k1, k2 = H -+ sqrt(H^2 - K)``, clamped to ``H`` where
    discretisation makes ``H^2 < K``.
    """
    area = _mixed_area(mesh)
    gaussian = mesh.vertex_defects / area
    edges = mesh.face_adjacency_edges
    lengths = np.linalg.norm(
        mesh.vertices[edges[:, 1]] - mesh.vertices[edges[:, 0]], axis=1
    )
    signs = np.where(mesh.face_adjacency_convex, 1.0, -1.0)
    edge_sum = np.bincount(
        edges.ravel(),
        weights=np.repeat(lengths * mesh.face_adjacency_angles * signs, 2),
        minlength=len(mesh.vertices),
    )
    mean = edge_sum / (4 * area)
    spread = np.sqrt(np.maximum(mean**2 - gaussian, 0))
    return mean - spread, mean + spread, area


def _area_average(values: np.ndarray, area: np.ndarray) -> float:
    """Average per-vertex ``values`` over the surface, weighting by vertex area."""
    valid = (area > 0) & np.isfinite(values)
    return float((values[valid] * area[valid]).sum() / area[valid].sum())


def _shape_index(k1: np.ndarray, k2: np.ndarray, area: np.ndarray) -> float:
    """Return the area-weighted shape index, 1 on a sphere and 0 on a saddle."""
    return _area_average((2 / np.pi) * np.arctan2(k1 + k2, k2 - k1), area)


def _curvedness(k1: np.ndarray, k2: np.ndarray, area: np.ndarray) -> float:
    """Return the area-weighted curvedness, ``1 / R`` on a sphere of radius R."""
    return _area_average(np.sqrt((k1**2 + k2**2) / 2), area)


def _smooth_twin(blurred: np.ndarray, pad: np.ndarray) -> np.ndarray:
    """Return the mask's blur thresholded at 0.5, cropped to the mask's own frame.

    Outside that frame every mask voxel lies to one side along some axis, so
    the separable blur there is at most the kernel's weight beyond its centre,
    ``(1 - w0) / 2 < 0.5``: the threshold loses nothing to the crop.
    """
    return (blurred > 0.5)[tuple(slice(p, -p) for p in pad)]


def _blur_ladder_dimension(
    name: str,
    mask: np.ndarray,
    step: np.ndarray,
    low: float,
    high: float,
    measure: Callable[[np.ndarray], float],
    top_blur: np.ndarray,
    top_pad: np.ndarray,
    top_measure: float,
    *,
    unit: float,
    n_levels: int = 5,
) -> float:
    """Return ``mask.ndim - 1 - slope`` of log measure ratio against log blur scale.

    The boundary measure (area in 3D, length in 2D) of a boundary of fractal
    dimension ``D`` taken at resolution ``s`` scales as ``s^(ndim - 1 - D)``.
    The mask is blurred at ``n_levels`` geometric scales from ``low`` up to
    ``high``, and ``measure`` of each blur's 0.5 level is divided by the same
    for the mask's smooth twin, its blur at ``high`` thresholded at 0.5. The
    twin shares the object's coarse shape and voxel grid, so the shrinking of
    curved boundaries under blur and the voxel staircase cancel in the ratio,
    which leaves the detail finer than ``high``. Without the twin, smooth
    ellipsoids of nucleus size measure 2.05-2.20 in 3D. Every step runs on the
    mask's device; NaN, with a warning naming ``name``, if any level vanishes.

    ``top_blur`` is the mask's blur at ``high`` with its padding ``top_pad``,
    and ``top_measure`` the measure of its 0.5 level, which the caller has
    already computed: the twin is thresholded from the first and cropped to
    the mask's frame, and the second is the mask's value at the last scale.
    ``step``, ``low`` and ``high`` are multiples of ``unit``, a length in the
    caller's units, which only converts the scale the warning prints.
    The other blurs of the ladder (the mask's below ``high``, the twin's at
    every scale) use the one-voxel padding of ``_padded_blur``: their field
    equals the wide blur's on the frame bit for bit, and only the float32
    vertices of each level set move, by rounding, with the frame's offset in
    the array.
    """
    twin = _smooth_twin(top_blur, top_pad)

    def at(image: np.ndarray, scale: float) -> float:
        return measure(_padded_blur(image, step, scale, tight=True)[0])

    # geomspace returns ``high`` itself as the last scale, where the mask's
    # measure is the one already computed
    scales = np.geomspace(low, high, n_levels)
    mask_values = [at(mask, s) for s in scales[:-1]]
    mask_values.append(top_measure)
    ratios = [v / at(twin, s) for v, s in zip(mask_values, scales)]
    if not np.all(np.isfinite(ratios)):
        warnings.warn(
            f"{name}: the mask or its smooth twin vanishes when blurred at a "
            f"scale up to {high * unit:g} (an object narrower than the blur); "
            f"returning NaN.",
            stacklevel=_stacklevel_outside_package(),
        )
        return float("nan")
    return mask.ndim - 1 - float(np.polyfit(np.log(scales), np.log(ratios), 1)[0])


def _fractal_dimension(
    mask_3d: np.ndarray,
    step: np.ndarray,
    low: float,
    high: float,
    blurred: np.ndarray,
    pad: np.ndarray,
    level_set: tuple[np.ndarray, np.ndarray] | None,
    unit: float,
) -> float:
    """Return ``2 - slope`` of log surface-area ratio against log blur scale.

    ``_blur_ladder_dimension`` on the 3D mask, with the area of each level set.
    ``blurred`` is the mask's blur at ``high`` with its padding ``pad``, and
    ``level_set`` its 0.5 level from ``_level_set``; that top blur keeps the
    wide padding, since it is shared with the curvature features' mesh, while
    the ladder's other blurs are tight.
    """

    def area(volume: np.ndarray) -> float:
        return _level_set_area(_level_set(volume, step))

    return _blur_ladder_dimension(
        "Fractal Dimension",
        mask_3d,
        step,
        low,
        high,
        area,
        blurred,
        pad,
        _level_set_area(level_set),
        unit=unit,
    )


def _contour_length(field: np.ndarray, plane_step: np.ndarray) -> float:
    """Return the length of the 0.5 contour of a 2D ``field``; NaN if it has none.

    Marching cubes on the field stacked twice along a new first axis, one unit
    apart, gives a ribbon whose area is the contour length times that unit, on
    the field's device. It matches skimage's ``find_contours`` to 1e-7 relative
    (float32 vertices), except in saddle cells, which marching cubes resolves
    by the bilinear interpolant, like ``find_contours(fully_connected="high")``
    when the saddle lies above 0.5.
    """
    ribbon = np.stack([field, field])
    return _level_set_area(_level_set(ribbon, np.array([1.0, *plane_step])))


def _outline_fractal_dimension(
    mask_3d: np.ndarray, step: np.ndarray, low: float, high: float, unit: float
) -> float:
    """Return ``1 - slope`` of log contour-length ratio against log blur scale.

    ``_blur_ladder_dimension`` on the mask's footprint along the first axis,
    with the length of each blur's 0.5 contour and the in-plane spacing. All
    its 2D blurs, the top one included, use the one-voxel padding.
    """
    footprint = mask_3d.any(axis=0)
    plane_step = step[1:]
    top, pad = _padded_blur(footprint, plane_step, high, tight=True)

    def length(field: np.ndarray) -> float:
        return _contour_length(field, plane_step)

    return _blur_ladder_dimension(
        "Outline Fractal Dimension",
        footprint,
        plane_step,
        low,
        high,
        length,
        top,
        pad,
        length(top),
        unit=unit,
    )


# Measured on the smoothed mesh when one is available; see mesh_feature_list.
_SMOOTHED_FEATURES = (
    "Total Absolute Gaussian Curvature",
    "Avg Shape Index",
    "Avg Curvedness",
)

# Approximate on a mesh with edges shared by more than two faces, with a
# warning: the Euler number counts such an edge once, the face adjacency behind
# the mean curvature (trimesh's ``group_rows(require_count=2)``, ArrayMesh's
# runs of two) leaves it out, and the angle defect at each of its ends sums the
# corners of both sheets that meet there.
_SHARED_EDGE_FEATURES = (
    "Euler Characteristic",
    "Avg Mean Curvature",
    *_SMOOTHED_FEATURES,
)
_SHARED_EDGE_WARNING = (
    "At least one mesh has edges shared by more than two faces, which marching "
    "cubes makes where voxels touch only along an edge. "
    f"{', '.join(_SHARED_EDGE_FEATURES[:-1])} and {_SHARED_EDGE_FEATURES[-1]} "
    "are approximate on such a mesh; their values are returned as computed."
)

# Measured on the mask itself, at several blur scales; see mesh_feature_list.
_MASK_FEATURES = ("Fractal Dimension", "Outline Fractal Dimension")

# Computed only when requested by name; see mesh_feature_list.
_OPTIONAL_FEATURES = ("Outline Fractal Dimension",)

# Each feature is computed only when requested; the dict order is the default
# column order of ``mesh_feature_list``.
_MESH_FEATURES: dict[str, Callable[[trimesh.Trimesh], SupportsFloat]] = {
    "Area": lambda m: m.area,
    "Volume": lambda m: m.volume,
    "Min Axis Length": lambda m: _equivalent_ellipsoid_axes(m)[0],
    "Med Axis Length": lambda m: _equivalent_ellipsoid_axes(m)[1],
    "Max Axis Length": lambda m: _equivalent_ellipsoid_axes(m)[2],
    "Inertia PC1": lambda m: m.principal_inertia_components[0],
    "Inertia PC2": lambda m: m.principal_inertia_components[1],
    "Inertia PC3": lambda m: m.principal_inertia_components[2],
    "Oriented Bounding Box Volume": lambda m: m.bounding_box_oriented.volume,
    "Bounding Cylinder Volume": _bounding_cylinder_volume,
    "Bounding Sphere Volume": lambda m: m.bounding_sphere.volume,
    "Convex Hull Volume": lambda m: m.convex_hull.volume,
    "Convex Hull Area": lambda m: m.convex_hull.area,
    "Sphericity": lambda m: sphericity(float(m.volume), float(m.area)),
    "Extent": lambda m: m.volume / m.bounding_box_oriented.volume,
    "Solidity": lambda m: m.volume / m.convex_hull.volume,
    # The surface average of the Gaussian curvature would be exactly
    # 2 pi * euler_number / area (Gauss-Bonnet), so it adds nothing to Area
    # beyond this integer.
    "Euler Characteristic": lambda m: m.euler_number,
    "Avg Mean Curvature": _avg_mean_curvature,
    "Total Absolute Gaussian Curvature": lambda m: (
        np.abs(m.vertex_defects).sum() / (4 * np.pi)
    ),
    "Avg Shape Index": lambda m: _shape_index(*_principal_curvatures(m)),
    "Avg Curvedness": lambda m: _curvedness(*_principal_curvatures(m)),
}

# Features that average the per-vertex principal curvatures ``(k1, k2, area)``;
# _measure_mesh computes those once for all of them. The registry entries above
# compute them for one feature.
_CURVATURE_AVERAGES: dict[
    str, Callable[[np.ndarray, np.ndarray, np.ndarray], SupportsFloat]
] = {
    "Avg Shape Index": _shape_index,
    "Avg Curvedness": _curvedness,
}
# Build trimesh's face adjacency on the mesh they are measured on.
_ADJACENCY_FEATURES = ("Avg Mean Curvature", *_CURVATURE_AVERAGES)

# Power of the length unit in each feature: a feature measured with ``spacing``
# in units u is ``u ** dimension`` times the same feature measured in units 1.
# extract_surface_features measures in units of the finest spacing and converts
# back with these.
_FEATURE_DIMENSIONS: dict[str, int] = {
    "Area": 2,
    "Volume": 3,
    "Min Axis Length": 1,
    "Med Axis Length": 1,
    "Max Axis Length": 1,
    "Inertia PC1": 5,
    "Inertia PC2": 5,
    "Inertia PC3": 5,
    "Oriented Bounding Box Volume": 3,
    "Bounding Cylinder Volume": 3,
    "Bounding Sphere Volume": 3,
    "Convex Hull Volume": 3,
    "Convex Hull Area": 2,
    "Sphericity": 0,
    "Extent": 0,
    "Solidity": 0,
    "Euler Characteristic": 0,
    "Avg Mean Curvature": -1,
    "Total Absolute Gaussian Curvature": 0,
    "Avg Shape Index": 0,
    "Avg Curvedness": -1,
    "Fractal Dimension": 0,
    "Outline Fractal Dimension": 0,
}
