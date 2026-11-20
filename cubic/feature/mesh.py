"""Extract features from a label image using ``trimesh``.

Each label is converted into a surface mesh with marching cubes and measured
with ``trimesh``. A CuPy label image or mask stays on the GPU: it is meshed by a
CUDA port of the same marching cubes and measured as an ``ArrayMesh``, which
computes the trimesh quantities on the device and hands the convex hull and
bounding sphere to trimesh on the host; the bounding box and cylinder are
measured on the host from that hull.

Units and trimesh tolerances. trimesh compares lengths, areas and cross
products with absolute tolerances, so whether a result is right can depend on
the length unit of ``spacing``. There are three levels:

1. ``extract_features`` and ``extract_surface_features`` measure in units of
   the finest voxel spacing, on meshes centred on the origin, and convert the
   values back, so they work in any unit. Both mesh and blur each object
   over its own bounding box, so their values do not depend on where the
   object lies in the image: ``extract_surface_features`` gives a mask the
   values ``extract_features`` gives the same object as a label.
2. ``mask2mesh`` builds its mesh in those units and returns it in the
   caller's units, and ``extract_mesh_features`` measures a mesh at the scale
   of its own edges, centred on the origin, and converts back. Both work in
   any unit, for meshes from ``mask2mesh``, and the second at any position
   but for a mesh with a centre of mass set on it, which is not centred.
3. Attributes read straight off a returned mesh, and a mesh the caller builds
   themselves, are subject to trimesh's absolute tolerances. trimesh sets a
   face normal to zero when the norm of the face's cross product is below
   1e-13, which happens once edges are shorter than about 3e-7, so face
   normals and everything derived from them (adjacency angles, convexity,
   ``integral_mean_curvature``) are then wrong. Separately, trimesh merges
   vertices closer than 1e-8 when it builds a mesh. ``volume`` and ``area``
   are right in any unit.
"""

from __future__ import annotations

import sys
import math
import inspect
import numbers
import warnings
from typing import Any, TypeVar, Callable, TypeAlias, SupportsFloat
from functools import cache, wraps
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

from ..cuda import (
    asnumpy,
    get_device,
    on_device_of,
    to_same_device,
    ordered_bincount,
)
from ..scipy import ndimage as device_ndimage
from ._array_mesh import ArrayMesh, face_areas
from ..image_utils import gaussian_filter_like_scipy
from ._marching_cubes_cuda import marching_cubes as _marching_cubes_gpu

F = TypeVar("F", bound=Callable[..., Any])

# pyproject's mesh floor: cubic itself no longer needs 4.6.12; the floor is kept.
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


def _on_input_device(func: F) -> F:
    """Run ``func`` with the GPU that holds its first argument current.

    The first argument is an array, or a mesh, whose vertices decide. CuPy
    allocates every new array on the current device, so without this the
    pad, blur and mesh of an input on another GPU land on the current one.
    """
    name = next(iter(inspect.signature(func).parameters))

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        first = args[0] if args else kwargs.get(name)
        held = first.vertices if isinstance(first, ArrayMesh) else first
        with on_device_of(held):
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
    every feature a little with orientation, and the box more than most:
    ``Extent`` spans 0.50-0.55 for one ellipsoid turned by 0, 30 and 45
    degrees. ``Oriented Bounding Box Volume`` is the smallest volume of the
    boxes that enclose the mesh with a face on a face of its convex hull,
    each with the exact minimum-area rectangle across that face's normal; it
    turns, scales and reorders with the mesh to rounding. It is an upper bound
    of the minimum enclosing box, which can instead have two adjacent faces
    each on a hull edge: it is twice the minimum for a regular tetrahedron.
    ``Bounding Cylinder Volume`` is the smallest volume of the cylinders that
    enclose the mesh about the normals of its convex hull faces and its three
    principal axes of inertia, each with the exact minimum enclosing circle
    across its axis; it turns and scales with the mesh to rounding. It is an
    upper bound of the minimum enclosing cylinder.
    The mean curvature is averaged over the surface, in inverse units
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
    units of ``spacing``. Curvedness is averaged over the surface by area, and
    the shape index by area times curvedness: the shape index is the angle of
    ``(k1, k2)`` and curvedness its length, so where the surface is nearly
    planar the angle is set by rounding, and it counts in proportion to the
    curvature it describes. It is NaN, with a warning, on a mesh without a
    curved vertex.
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
@_on_input_device
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
@_on_input_device
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
        and otherwise to about 1e-14 relative.
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
    # Measured on the object's bounding box, as extract_features crops each
    # label, so that no value depends on where the object lies in the mask: a
    # level-set vertex that rounds to a grid point or a cube centre in float32
    # keeps that float32 position, and float32 rounds coarser far from the
    # frame's origin. The crop makes the frame the object's own. Every blur and
    # mesh pads the box as it needs.
    box = _object_box(mask)
    if box is not None:
        mask = mask[box]
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

    @cache  # the mask's mass, counted once for every scale
    def mass() -> float:
        return _mask_mass(mask)

    def blur_at(scale: float) -> tuple[np.ndarray, np.ndarray, Any]:
        if scale not in blurs:
            volume, pad = _padded_blur(mask, step, scale, mass=mass())
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


def _curvature_sigma(
    spacing: Sequence[float] | None, sigma: float | None, name: str = "curvature_sigma"
) -> float:
    """Return ``sigma``, or twice the coarsest voxel spacing for ``None``.

    ``name`` is the caller's parameter, for the error message.
    """
    if sigma is None:
        return 2 * (1.0 if spacing is None else float(max(spacing)))
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"{name} must be finite and positive; got {sigma}")
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


def _scaled_copy(
    surface: Any,
    factor: float,
    center: bool = False,
    bounds: tuple[list[float], list[float]] | None = None,
) -> Any:
    """Return a copy of ``surface`` with every vertex times ``factor``, unmerged.

    See "Units and trimesh tolerances" in the module docstring. With
    ``center``, the middle of the box of the vertices of its faces
    (``_vertex_bounds``, or ``bounds`` when already computed) is subtracted
    from every vertex first, so the copy is centred on the origin; the same
    vertices give the same copy on either device. A mesh with a centre of
    mass set on it is not centred: trimesh
    shifts its inertia from the origin to that point as if it were the
    centroid, which depends on where the mesh lies. ``surface`` is not
    modified. An ``ArrayMesh`` stays one; any other mesh, a trimesh primitive
    included, becomes a plain ``trimesh.Trimesh`` of its vertices and faces
    that keeps a density or centre of mass set on it.
    """
    # trimesh keeps a density or centre of mass set on a mesh apart from its
    # cache, where reading them computes nothing on ``surface``.
    overrides = {} if isinstance(surface, ArrayMesh) else surface._data.data
    if not center or "center_mass" in overrides:
        bounds = None
    elif bounds is None:
        bounds = _vertex_bounds(surface)
    origin = None
    if bounds is not None:
        middle = np.array([(low + high) / 2 for low, high in zip(*bounds)])
        origin = middle if np.isfinite(middle).all() else None
    if isinstance(surface, ArrayMesh):
        return surface.scaled(factor, origin=origin)
    # A new mesh, since a primitive refuses new vertices, and not
    # ``apply_scale``, which leaves the vertices as they are for any factor
    # within 1e-8 of 1.
    shift = 0.0 if origin is None else origin
    copied = trimesh.Trimesh(
        vertices=(np.asarray(surface.vertices) - shift) * factor,
        faces=np.array(surface.faces),
        process=False,
    )
    if "density" in overrides:
        copied.density = overrides["density"]
    if "center_mass" in overrides:
        copied.center_mass = np.asarray(overrides["center_mass"]) * factor
    return copied


def _smooth_mask2mesh(
    mask_3d: np.ndarray, spacing: Sequence[float] | None, sigma: float
) -> Any:
    """Mesh the 0.5 level of the mask blurred at ``sigma``; None if it vanishes.

    The mask is blurred and meshed over the bounding box of the object, which
    makes the frame the object's own: a vertex that rounds to a grid point or a
    cube centre in float32 keeps that float32 position, which depends on where
    the object lies in the array. The vertices are then moved back to the
    mask's frame in float64.
    """
    step = _step(spacing)
    box = _object_box(mask_3d)
    if box is None:
        return None
    volume, pad = _padded_blur(mask_3d[box], step, sigma)
    level_set = _level_set(volume, step)
    if level_set is None:
        return None
    corner = np.array([s.start for s in box])
    return _level_set_mesh(level_set, pad - corner, step)


def _object_box(mask: np.ndarray) -> tuple[slice, ...] | None:
    """Return the slices of the smallest box holding every nonzero voxel of ``mask``.

    None for a mask without one.
    """
    occupied = mask if mask.dtype == bool else mask != 0
    return device_ndimage.find_objects(occupied.view(np.uint8), 1)[0]


def _level_set_mesh(
    level_set: tuple[np.ndarray, np.ndarray], pad: np.ndarray, step: np.ndarray
) -> Any:
    """Return the closed mesh of a padded blur's level set, in the mask's frame.

    ``pad`` is the number of voxels, per axis, by which the origin of the
    blurred volume lies before the origin of the mask's frame.
    """
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
    mask: np.ndarray,
    step: np.ndarray,
    sigma: float,
    tight: bool = False,
    mass: float | None = None,
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

    When ``_blur_bound`` shows that the blur stays below 0.5 everywhere, in
    either mode, no padded array is built and no blur runs: a float32 array
    of zeros, shaped like the mask padded by one voxel and on the mask's
    device, stands in for the blur, with a padding of ones. Every caller only
    takes the 0.5 level of the blur or thresholds it at 0.5, and both are
    empty for the stand-in as for the real blur. ``mass`` is the mask's
    ``_mask_mass``, for a caller that blurs one mask at several scales; it is
    computed here when not given.
    """
    sigma_voxels = sigma / step
    if _blur_bound(mask, sigma_voxels, mass) < _VANISHING_BOUND:
        shape = tuple(n + 2 for n in mask.shape)
        return np.zeros_like(mask, dtype=np.float32, shape=shape), np.ones(
            mask.ndim, dtype=int
        )
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


# ``_padded_blur`` skips a blur whose ``_blur_bound`` is below this. The margin
# under 0.5 absorbs the float32 rounding of the real blur's passes and the
# float64 rounding of the bound, both below 1e-6 relative.
_VANISHING_BOUND = 0.49
# Kernel taps on each side of the centre that ``_largest_kernel_weight`` sums
# exactly; the taps beyond them are bounded by an integral.
_EXACT_TAPS = 16


def _largest_kernel_weight(sigma: float) -> float:
    """Return an upper bound on the largest weight of a 1D Gaussian blur kernel.

    The kernel is the one ``gaussian_filter_like_scipy`` applies (SciPy's, and
    cupyx's, which builds the same one), at its default ``truncate`` of 4: the
    samples of ``g(x) = exp(-x^2 / (2 sigma^2))`` at the integers ``|x| <= r``,
    ``r = int(4 * sigma + 0.5)``, divided by their sum ``S``. The
    largest weight is the centre one, ``1 / S``, so a lower bound on ``S``
    gives the upper bound. The taps with ``|x| <= m = min(r, _EXACT_TAPS)``
    are summed as they are. ``g`` decreases for ``x >= 0``, so each tap
    beyond, ``m < x <= r``, is at least the integral of ``g`` over
    ``[x, x + 1]``: ``S`` is at least the exact sum plus twice the integral of
    ``g`` from ``m + 1`` to ``r + 1``, a difference of error functions. The
    cost does not grow with ``sigma``, and the bound is the weight itself, to
    rounding, while ``r <= _EXACT_TAPS``. A ``sigma`` with ``r = 0`` (SciPy
    also skips the axes of ``sigma <= 1e-15``) leaves the axis as it is:
    weight 1.
    """
    # 4 is the default ``truncate`` of ``gaussian_filter_like_scipy``.
    radius = int(4.0 * sigma + 0.5)
    taps = min(radius, _EXACT_TAPS)
    total = 1.0 + 2.0 * sum(
        math.exp(-0.5 * (x / sigma) ** 2) for x in range(1, taps + 1)
    )
    if radius > taps:
        scale = sigma * math.sqrt(2.0)
        total += (
            sigma
            * math.sqrt(2.0 * math.pi)
            * (math.erf((radius + 1) / scale) - math.erf((taps + 1) / scale))
        )
    return 1.0 / total


def _mask_mass(mask: np.ndarray) -> float:
    """Return the sum of the positive values of ``mask``, its voxel count if boolean."""
    if mask.dtype == bool:
        return float(np.count_nonzero(mask))
    return float(np.maximum(mask, 0).sum(dtype=np.float64))


def _blur_bound(
    mask: np.ndarray, sigma_voxels: np.ndarray, mass: float | None = None
) -> float:
    """Return an upper bound on every value of ``_padded_blur`` of ``mask``.

    ``sigma_voxels`` is the blur width per axis, in voxels, and ``mass`` the
    mask's ``_mask_mass``, computed here when not given. The separable blur
    of a voxel is the product of one weight of each axis's kernel, so a
    blurred value is at most the sum of the mask's positive values times the
    product over axes of ``_largest_kernel_weight``.

    That holds in both padding modes. ``tight`` blurs with zeros beyond the
    array. With the wide padding, in ``reflect`` mode, a kernel also reaches
    mirror images of voxels. Along one axis, with ``s = sigma_voxels``, the
    kernel's radius ``r <= 4 s + 0.5`` and the array's length
    ``L >= 6 s + 3 > r`` (the padding is ``ceil(3 s) + 1`` per side): for an
    output at ``x`` and a voxel at ``v``, the voxel's two nearest mirror
    images are ``x + v + 1`` and ``2 L - 1 - v - x`` away, which add up to
    ``2 L``, and images further out are more than ``L`` away, so the output
    reaches at most one image of the voxel. A voxel lies ``d >= 3 s + 1``
    voxels inside a border, and an output ``e`` voxels inside it reaches the
    image across that border, ``e + d + 1`` away, only if ``e <= s - 1.5``;
    the voxel itself is then ``d - e >= 2 s + 2.5`` away. Both weights are
    then at most ``exp(-2) / S``, and their sum is below the centre weight
    ``1 / S``.
    """
    weight = _mask_mass(mask) if mass is None else mass
    for s in sigma_voxels:
        weight *= _largest_kernel_weight(float(s))
    return weight


def _level_set(
    volume: np.ndarray, step: np.ndarray
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return ``(verts, faces)`` of the 0.5 level of ``volume``, on its device.

    None if the volume never exceeds 0.5. Vertices are in the volume's frame,
    padding included.
    """
    if float(volume.max()) <= 0.5:
        return None
    verts, faces = _marching_cubes(volume, 0.5, allow_degenerate=False)
    verts = _float64_vertices(volume, verts, 0.5)
    # Each column times a Python float, in place on the fresh float64 array: the
    # same products as ``verts * step``, without copying ``step`` to the GPU, and
    # none for a step of 1, which leaves every double as it is.
    for axis, factor in enumerate(np.asarray(step, dtype=np.float64).tolist()):
        if factor != 1.0:
            verts[:, axis] *= factor
    return verts, faces


# skimage's FLT_EPSILON in the vertex interpolation is the double epsilon.
_MC_EPS = float(np.spacing(1.0))


def _float64_vertices(
    volume: np.ndarray, verts: np.ndarray, level: float
) -> np.ndarray:
    """Return marching-cubes vertices (index units) in float64, on their device.

    Both marching cubes compute each edge vertex in float64, as ``floor + t_far
    / (t_near + t_far)`` along the edge with ``t = 1 / (eps + |value - level|)``,
    and round it to float32; the rounding step depends on where the object
    lies in the array and is amplified by the curvature estimates. A float32
    vertex with exactly one non-integer coordinate lies on the grid edge from
    its floor along that axis, or is a cube-centre vertex on that line: it is
    recomputed in float64 when that edge straddles ``level`` and the result
    rounds back to it. Every other vertex, rounded onto a grid point or a
    centre vertex, keeps its float32 position. The result depends only on the
    float32 vertex, so both devices give the same mesh, and vertices coincide
    in float64 exactly when they do in float32.
    """
    v32 = verts.astype(np.float32, copy=False)
    corner = np.floor(v32)
    on_edge = v32 != corner
    single = on_edge.sum(axis=1) == 1
    # Every row is computed and the rows that qualify are selected, since picking
    # them by index makes the host wait for the device. A row with several
    # non-integer coordinates reads the field at its floor, inside the volume,
    # and is discarded.
    along = on_edge & single[:, None]
    near = corner.astype(np.int64)
    # the float32 field both marching cubes mesh
    values = [
        volume[tuple(index.T)].astype(np.float32, copy=False).astype(np.float64) - level
        for index in (near, near + along)
    ]
    t_near, t_far = (1.0 / (_MC_EPS + np.abs(value)) for value in values)
    pos = np.where(along, near + (t_far / (t_near + t_far))[:, None], near)
    keep = (
        single
        & ((values[0] > 0) != (values[1] > 0))
        & (pos.astype(np.float32) == v32).all(axis=1)
    )
    return np.where(keep[:, None], pos, v32)


def _level_set_area(level_set: tuple[np.ndarray, np.ndarray] | None) -> float:
    """Return the surface area of a level set from ``_level_set``; NaN for None."""
    if level_set is None:
        return float("nan")
    verts, faces = level_set
    return float(face_areas(verts.astype(np.float64, copy=False), faces).sum())


@_require_trimesh
@_on_input_device
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
        areas and volumes are in physical units, in any length unit. The mesh
        is built in units of the finest spacing and returned in these; see
        "Units and trimesh tolerances" in the module docstring for which of
        its attributes are right in tiny units.
    marching_cubes_kwargs : dict, optional
        Extra keyword arguments for ``skimage.measure.marching_cubes``; on the
        GPU only ``level``, ``spacing`` and ``allow_degenerate`` are supported.
    sigma : float, optional
        If given, mesh the mask blurred by a Gaussian of this width, in units of
        ``spacing``, instead of the voxel staircase. ``marching_cubes_kwargs``
        is not used then and cannot be combined with it.

    Raises
    ------
    ValueError
        If ``sigma`` is given and the blurred mask never reaches 0.5, i.e. the
        object is too small for that blur; if ``sigma`` is not finite and
        positive, or is given together with ``marching_cubes_kwargs``; or if
        the voxel spacing, given as ``spacing`` or in
        ``marching_cubes_kwargs``, is not three finite, positive numbers
        ``(z, y, x)``.
    """
    # A negative spacing would turn the mesh inside out, a zero one flatten it.
    # The mesh is built in units of the finest spacing, where trimesh's
    # absolute merge tolerance sees only the spacing ratios, and then scaled
    # back to the caller's units without merging again.
    unit = _unit(spacing)
    if sigma is not None:
        if marching_cubes_kwargs:
            raise ValueError(
                "marching_cubes_kwargs cannot be combined with sigma: the blurred "
                "mask is meshed without them. Pass the voxel spacing as 'spacing'."
            )
        # Validated in the caller's units, so that an error names their number.
        mesh = _smooth_mask2mesh(
            mask_3d,
            None if spacing is None else tuple(_step(spacing) / unit),
            _curvature_sigma(spacing, sigma, "sigma") / unit,
        )
        if mesh is None:
            raise ValueError(f"The mask vanishes when blurred at sigma={sigma}.")
    else:
        marching_cubes_kwargs = dict(marching_cubes_kwargs or {})
        if spacing is None:
            spacing = marching_cubes_kwargs.get("spacing")
            unit = _unit(spacing)
        elif "spacing" in marching_cubes_kwargs:
            raise ValueError(
                "Voxel spacing given both as 'spacing' and in "
                "'marching_cubes_kwargs'; pass it only once."
            )
        if spacing is not None:
            marching_cubes_kwargs["spacing"] = tuple(_step(spacing) / unit)
        verts, faces = _marching_cubes(np.pad(mask_3d, 1), **marching_cubes_kwargs)
        # Undo the 1-voxel pad so vertices stay in the input's coordinate
        # frame. Without this every vertex, and therefore ``bounds`` and
        # ``centroid``, sits one voxel high on each axis -- scaled by
        # ``spacing``, so the offset is in physical units. Every feature in
        # ``mesh_feature_list`` is translation-invariant, but ``mask2mesh`` is
        # public and its mesh is expected to overlay the image.
        one_voxel = np.asarray(marching_cubes_kwargs.get("spacing", 1.0), dtype=float)
        verts = verts - to_same_device(one_voxel, verts)
        mesh = _closed_mesh(verts, faces)
    return mesh if unit == 1 else _scaled_copy(mesh, unit)


def sphericity(v: float, sa: float) -> float:
    """Return the Wadell sphericity of a body of volume ``v`` and surface area ``sa``.

    This is the surface area of the sphere of equal volume divided by ``sa``: 1
    for a sphere and smaller for any other shape.
    """
    return np.power(np.pi, 1 / 3) * np.power(6 * v, 2 / 3) / sa


@_require_trimesh
@_on_input_device
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
        Closed surface of a single object, in any length unit; the features
        are in that unit. It is measured at the scale of its own edges, so a
        mesh from ``mask2mesh`` gives its features in any unit; see "Units and
        trimesh tolerances" in the module docstring. A trimesh primitive
        (``Sphere``, ``Box``) is measured as its triangles. ``mesh`` and
        ``smooth_mesh`` are not modified.
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
    # Measured on copies scaled by one factor so that the edges of ``mesh`` are
    # of order 1, where trimesh's absolute tolerances see the shape and not the
    # unit, and converted back.
    bounds = _vertex_bounds(mesh)
    unit = _edge_unit(mesh, bounds)
    # The same object passed as both is measured once, as with no smooth mesh.
    values = _measure_mesh(
        mesh,
        features,
        smooth_mesh=None if smooth_mesh is mesh else smooth_mesh,
        unit=unit,
        scale=1 / unit,
        bounds=bounds,
    )
    return _to_caller_units(values, unit)


def _vertex_bounds(surface: Any) -> tuple[list[float], list[float]] | None:
    """Return the per-axis minima and maxima of the vertices of the faces of ``surface``.

    Exact on any device and in any order of the faces; None for a mesh
    without faces.
    """
    faces = surface.faces
    if len(faces) == 0:
        return None
    vertices = surface.vertices
    used = np.zeros_like(vertices[:, 0], dtype=bool)
    used[faces.ravel()] = True
    corners = vertices[used]
    # One axis at a time: NumPy reduces a (V, 3) array along axis 0 ten times
    # slower than its columns.
    return (
        [float(corners[:, i].min()) for i in range(3)],
        [float(corners[:, i].max()) for i in range(3)],
    )


def _edge_unit(
    surface: Any, bounds: tuple[list[float], list[float]] | None = None
) -> float:
    """Return a length of the order of the edges of ``surface``.

    The square root of the area of the axis-aligned bounding box of the
    vertices of its faces per face: it scales with the unit of the vertices,
    and on 144 voxel meshes it was 0.95 to 1.4 times the square root of the
    mean face area.

    It uses only the per-axis extremes of those vertices, which are exact on
    any device and in any order, and the face count, so a NumPy and a CuPy mesh
    with the same vertices get the same unit however their faces are listed.
    1 for a mesh without faces or whose box has no finite, positive area, which
    the volume and area check of ``_measure_mesh`` then reports. ``bounds``
    are ``_vertex_bounds(surface)`` when already computed.
    """
    if bounds is None:
        bounds = _vertex_bounds(surface)
    if bounds is None:
        return 1.0
    a, b, c = (high - low for low, high in zip(*bounds))
    area = 2 * (a * b + b * c + c * a)
    if not math.isfinite(area) or area <= 0:
        return 1.0
    return math.sqrt(area / len(surface.faces))


def _measure_mesh(
    mesh: trimesh.Trimesh,
    features: Sequence[str],
    smooth_mesh: trimesh.Trimesh | None = None,
    unit: float = 1.0,
    scale: float = 1.0,
    bounds: tuple[list[float], list[float]] | None = None,
) -> dict[str, float]:
    """Return ``features`` of ``mesh`` times ``scale``, a length in units of ``unit``.

    They are measured on copies of ``mesh`` and ``smooth_mesh`` times
    ``scale`` and centred on the middle of their boxes (``_scaled_copy``):
    every feature is translation-invariant, and coordinates of the order of
    the object's size round less than those of a far-off mesh. ``bounds`` are
    ``_vertex_bounds(mesh)`` when already computed. The values stay in the
    units of the copies; ``unit`` only converts the volume or area printed by
    the errors to the caller's units. The caller's meshes are left as they
    are.
    """
    mesh = _scaled_copy(mesh, scale, center=True, bounds=bounds)
    if smooth_mesh is not None:
        smooth_mesh = _scaled_copy(smooth_mesh, scale, center=True)
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

    # What several features share is computed at most once per call: the edge
    # terms of each mesh, behind Avg Mean Curvature and the per-vertex
    # curvatures, which are one mesh without a smooth mesh; the per-vertex
    # curvatures behind both curvature averages; the convex hull's distinct
    # face normals behind the box and the cylinder; and the oriented box behind
    # its volume and Extent. trimesh caches the volume, convex hull and
    # bounding primitives on the mesh itself.
    edge_terms: dict[int, tuple[Any, Any]] = {}

    def edge_curvatures_of(surface: Any) -> tuple[Any, Any]:
        if id(surface) not in edge_terms:
            edge_terms[id(surface)] = _edge_curvatures(surface)
        return edge_terms[id(surface)]

    curvatures = (
        _principal_curvatures(curvature_mesh, edge_curvatures_of(curvature_mesh))
        if any(name in _CURVATURE_AVERAGES for name in features)
        else None
    )

    @cache
    def hull_axes() -> _HullAxes:
        return _hull_axes(mesh)

    @cache
    def box() -> float:
        return _oriented_bounding_box_volume(mesh, hull_axes())

    def measure(name: str) -> SupportsFloat:
        if curvatures is not None and name in _CURVATURE_AVERAGES:
            return _CURVATURE_AVERAGES[name](*curvatures)
        if name == "Avg Mean Curvature":
            return _avg_mean_curvature(mesh, edge_curvatures_of(mesh))
        if name == "Oriented Bounding Box Volume":
            return box()
        if name == "Extent":
            return _extent(mesh, box())
        if name == "Bounding Cylinder Volume":
            return _bounding_cylinder_volume(mesh, hull_axes())
        return _MESH_FEATURES[name](measured_on(name))

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


# The convex hull of a mesh and the distinct axes of its face normals, or the
# error building the hull raised (``_hull_axes``).
_HullAxes: TypeAlias = tuple[Any, np.ndarray] | ValueError | QhullError


def _bounding_cylinder_volume(
    mesh: trimesh.Trimesh, found: _HullAxes | None = None
) -> float:
    """Return the smallest enclosing cylinder volume about a hull normal or inertia axis.

    The candidate axes are the normals of the faces of the convex hull and the
    three principal axes of inertia. About each axis the cylinder is exact:
    its height is the extent of the hull vertices along the axis and its
    radius that of the minimum enclosing circle of their projection. The
    volume is the smallest of these, so it turns and scales with the mesh, and
    an upper bound of the minimum enclosing cylinder. An elongated body with
    rounded or pointed ends can have no hull face normal to its long axis,
    which the inertia axes supply. Measured on the host from the hull that
    trimesh caches for ``Convex Hull Volume``; 0, with a warning, for a mesh
    whose hull cannot be built. ``found`` is ``_hull_axes(mesh)`` when
    already computed.
    """
    usable = _usable_hull(
        _hull_axes(mesh) if found is None else found, "bounding cylinder volume"
    )
    if usable is None:
        return 0.0
    hull, normals = usable
    inertia_axes = np.linalg.eigh(np.asarray(mesh.moment_inertia, dtype=np.float64))[1]
    return _min_cylinder_volume(
        np.asarray(hull.vertices, dtype=np.float64),
        np.vstack([normals, inertia_axes.T]),
    )


def _hull_axes(mesh: trimesh.Trimesh) -> _HullAxes:
    """Return the convex hull of ``mesh`` and the distinct axes of its face normals.

    The hull is the one trimesh caches for ``Convex Hull Volume``, on the host.
    For a mesh whose hull cannot be built, the error trimesh raised instead,
    which ``_usable_hull`` reports for each feature that needs the hull.
    """
    try:
        hull = mesh.convex_hull
    except (ValueError, QhullError) as e:
        return e
    normals = np.asarray(hull.face_normals, dtype=np.float64)
    # trimesh gives a face too small for a normal a zero one.
    normals = normals[(normals * normals).sum(axis=1) > 0.5]
    # An axis and its reverse give the same cylinder or box, and the faces of
    # one flat side of the hull share a normal to rounding: keep one of each.
    largest = np.abs(normals).argmax(axis=1)
    normals *= np.sign(normals[np.arange(len(normals)), largest])[:, None]
    _, first = np.unique(normals.round(9), axis=0, return_index=True)
    return hull, normals[np.sort(first)]


def _usable_hull(found: _HullAxes, name: str) -> tuple[Any, np.ndarray] | None:
    """Return the hull and axes ``found``, or None, warning, for a failed hull.

    The warning names the feature ``name``, which is then 0.
    """
    if isinstance(found, Exception):
        warnings.warn(
            f"Unable extract {name}. Returning 0. Exception: {found}",
            stacklevel=_stacklevel_outside_package(),
        )
        return None
    return found


def _oriented_bounding_box_volume(
    mesh: trimesh.Trimesh, found: _HullAxes | None = None
) -> float:
    """Return the smallest volume of the enclosing boxes with a face on a hull face.

    The candidate axes are the normals of the faces of the convex hull. About
    each axis the box is exact: its height is the extent of the hull vertices
    along the axis and its base the minimum-area rectangle of their
    projection, which has a side on an edge of the projection's convex hull
    (Freeman and Shapira, 1975). The volume is the smallest of these, so it
    turns and scales with the mesh to rounding. A minimum-volume box has a
    face on a hull face or two adjacent faces each on a hull edge
    (O'Rourke, 1985); only the first kind is searched, so this is an upper
    bound of the minimum, exact for a box and twice it for a regular
    tetrahedron. Measured on the host from the hull that trimesh caches for
    ``Convex Hull Volume``; 0, with a warning, for a mesh whose hull cannot be
    built or has no face with a normal. ``found`` is ``_hull_axes(mesh)``
    when already computed.
    """
    name = "oriented bounding box volume"
    usable = _usable_hull(_hull_axes(mesh) if found is None else found, name)
    if usable is None:
        return 0.0
    if len(usable[1]) == 0:
        warnings.warn(
            f"Unable extract {name}. Returning 0. No face of the convex hull has "
            "a normal.",
            stacklevel=_stacklevel_outside_package(),
        )
        return 0.0
    return _min_box_volume(*usable)


def _extent(mesh: trimesh.Trimesh, box: float | None = None) -> float:
    """Return the volume over the oriented bounding box volume; 0 without a box.

    ``box`` is that volume when already computed.
    """
    if box is None:
        box = _oriented_bounding_box_volume(mesh)
    return float(mesh.volume) / box if box > 0 else 0.0


def _min_box_volume(hull: trimesh.Trimesh, axes: np.ndarray) -> float:
    """Return the smallest volume of the boxes about unit ``axes`` enclosing ``hull``.

    Each box spans the extent of the hull vertices along its axis, and its base
    is the minimum-area rectangle of their projection across it
    (``_box_volumes``). Exact: an axis is skipped only when a lower bound of
    its volume exceeds a volume already found (``_smallest_over_axes``). The
    bound is the area of the projection times its height. For a hull of many
    vertices it is that of the hull of the few extreme ones
    (``_bounding_subset``), and then the box of that hull about the axis,
    which every box enclosing the hull encloses.
    """
    points = np.asarray(hull.vertices, dtype=np.float64)
    frames = _axis_frames(axes)
    few = _bounding_subset(points)
    inner = hull
    if len(few) < len(points):
        # Should Qhull fail on the few, the hull bounds itself.
        try:
            inner = trimesh.convex.convex_hull(few)
        except (ValueError, QhullError):
            pass
    normals = np.asarray(inner.face_normals, dtype=np.float64)
    areas = np.asarray(inner.area_faces, dtype=np.float64)
    corners = np.asarray(inner.vertices, dtype=np.float64)
    # The hull covers its projection twice, so this is the projection's area.
    lower = np.concatenate(
        [
            areas
            @ np.abs(normals @ axes[c].T)
            / 2
            * np.ptp(corners @ axes[c].T, axis=0)
            for c in _axis_chunks(len(axes))
        ]
    )
    refine = None if inner is hull else _box_volumes(inner, axes, frames)
    return _smallest_over_axes(lower, _box_volumes(hull, axes, frames), refine)


def _box_volumes(
    hull: trimesh.Trimesh, axes: np.ndarray, frames: np.ndarray
) -> Callable[[np.ndarray], np.ndarray]:
    """Return the volume of the smallest box about each of a batch of ``axes``.

    Each box spans the extent of the hull vertices along its axis, and its base
    is the minimum-area rectangle of their projection across it. Its sides run
    along the projection of a silhouette edge, one between a hull face that
    faces along the axis and one that does not; every edge of the projection's
    hull is one, and any other direction gives a larger rectangle. The
    projection's hull is that of the silhouette edges' ends, and its widths
    across each side are found by rotating calipers (``_projection_support``),
    O(S log S) for S silhouette edges, or for at most ``_DENSE_SIDES`` sides
    from every end, O(S^2) but fewer operations. ``frames`` are
    ``_axis_frames(axes)``.
    """
    points = np.asarray(hull.vertices, dtype=np.float64)
    normals = np.asarray(hull.face_normals, dtype=np.float64)
    pairs = np.asarray(hull.face_adjacency)
    edges = np.asarray(hull.face_adjacency_edges)

    def volumes(batch: np.ndarray) -> np.ndarray:
        facing = normals @ axes[batch].T
        ahead, behind = facing > 0, facing < 0
        # A face across the axis, of either sign, counts on both sides.
        silhouette = ~(
            (ahead[pairs[:, 0]] & ahead[pairs[:, 1]])
            | (behind[pairs[:, 0]] & behind[pairs[:, 1]])
        )
        # (axes, sides): each axis's silhouette edges, padded with repeats of
        # its first, which change no extent.
        sides_of = _padded_rows(np.ascontiguousarray(silhouette.T))
        across = frames[batch, 1:].transpose(0, 2, 1)
        # The projection, in two unit vectors across each axis, so that each
        # rectangle is exactly a rectangle even for an edge along the axis,
        # whose projection points anywhere: (axes, sides, ends, 2).
        flat = (points[edges[sides_of].reshape(len(batch), -1)] @ across).reshape(
            *sides_of.shape, 2, 2
        )
        sides = flat[:, :, 1] - flat[:, :, 0]
        length = np.sqrt((sides * sides).sum(axis=2))
        sides /= np.where(length > 0, length, 1.0)[..., None]
        both = np.concatenate([sides, sides[..., ::-1] * np.array([-1.0, 1.0])], 1)
        count = sides.shape[1]
        if count <= _DENSE_SIDES:
            # Every end against every side, which takes fewer operations than
            # the calipers for few sides.
            reach = both @ flat.reshape(len(batch), -1, 2).transpose(0, 2, 1)
            width = reach.max(axis=2) - reach.min(axis=2)
        else:
            # The outward normal of each side in the projection: the
            # combination of its two faces' normals that is across the axis,
            # or the first normal when both faces are across it.
            faces = pairs[sides_of]
            on = facing.T[np.arange(len(batch))[:, None, None], faces]
            outward = (
                np.abs(on[..., 1:]) * normals[faces[..., 0]]
                + np.abs(on[..., :1]) * normals[faces[..., 1]]
            )
            outward = np.where(
                (on == 0).all(axis=2)[..., None], normals[faces[..., 0]], outward
            )
            support = _projection_support(
                flat, outward @ across, np.concatenate([both, -both], 1)
            )
            width = support[:, : 2 * count] + support[:, 2 * count :]
        base = np.where(length > 0, width[:, :count] * width[:, count:], np.inf)
        height = np.ptp(points @ axes[batch].T, axis=0)
        return base.min(axis=1) * height

    return volumes


# Up to this many silhouette edges, the widths of a projection are found from
# every end, in fewer operations than the calipers for few edges; its
# (32, 2 S, 2 S) products then stay below 70 MB.
_DENSE_SIDES = 256


def _projection_support(
    flat: np.ndarray, outward: np.ndarray, directions: np.ndarray
) -> np.ndarray:
    """Return the largest projection onto each of ``directions`` of each convex polygon.

    ``flat`` holds the ends ``(k, s, 2, 2)`` of the ``s`` boundary segments
    of each of ``k`` convex polygons, in any order and with repeats, and
    ``outward`` ``(k, s, 2)`` the outward normal of each, of any length;
    ``directions`` are ``(k, q, 2)``. The result, ``(k, q)``, is the largest
    product of a segment end with each direction, from rotating calipers: the
    segments are put in order around the boundary by the angle of their
    middles about the mean middle, where their normals turn once around; the
    ends of the segments whose normals' angles bracket a direction's, and of
    their neighbours, hold the largest product. A polygon whose normals do not
    turn once around, to 1e-6, is searched in full.
    """
    k, s = outward.shape[:2]
    rows = np.arange(k)[:, None]
    middle = flat.mean(axis=2)
    middle -= middle.mean(axis=1, keepdims=True)
    order = np.argsort(np.arctan2(middle[..., 1], middle[..., 0]), axis=1)
    ends = flat[rows, order]
    angle = np.arctan2(outward[..., 1], outward[..., 0])[rows, order]
    turns = (np.diff(angle, axis=1, append=angle[:, :1]) + np.pi) % (2 * np.pi) - np.pi
    # The normals' angles from the first, turned on to rounding: nondecreasing
    # and below 2 pi, so that they bracket every direction once.
    turned = np.zeros((k, s))
    turned[:, 1:] = np.cumsum(turns[:, :-1], axis=1)
    turned = np.maximum.accumulate(turned, axis=1)
    wanted = (np.arctan2(directions[..., 1], directions[..., 0]) - angle[:, :1]) % (
        2 * np.pi
    )
    # One sorted search for every polygon: each row's angles shifted above
    # the previous row's.
    shift = 8.0 * np.arange(k)[:, None]
    after = np.searchsorted((turned + shift).ravel(), (wanted + shift).ravel())
    after = after.reshape(wanted.shape) - s * rows
    x, y = ends[..., 0], ends[..., 1]
    across, along = directions[..., 0], directions[..., 1]
    support = np.full(wanted.shape, -np.inf)
    for offset in (-2, -1, 0, 1):
        near = (after + offset) % s
        for end in (0, 1):
            products = x[rows, near, end] * across + y[rows, near, end] * along
            np.maximum(support, products, out=support)
    # Rounding cannot turn the normals by much less or more than once around;
    # a polygon they do not turn once around is searched among all its ends.
    total = turns.sum(axis=1)
    for row in np.flatnonzero(np.abs(total - 2 * np.pi) > 1e-6):
        support[row] = (directions[row] @ flat[row].reshape(-1, 2).T).max(axis=1)
    return support


def _padded_rows(mask: np.ndarray) -> np.ndarray:
    """Return the column indices of the True entries of each row of ``mask``.

    As a ``(rows, most per row)`` array, each row padded with its first index;
    every row needs one True entry.
    """
    rows, columns = np.nonzero(mask)
    counts = np.bincount(rows, minlength=len(mask))
    starts = np.cumsum(counts) - counts
    padded = np.repeat(columns[starts][:, None], int(counts.max()), axis=1)
    padded[rows, np.arange(len(columns)) - starts[rows]] = columns
    return padded


def _axis_chunks(count: int) -> list[np.ndarray]:
    """Split ``range(count)`` into chunks of at most 256 axes, to bound the memory.

    One empty chunk for no axes.
    """
    return np.array_split(np.arange(count), max(1, -(-count // 256)))


def _smallest_over_axes(
    lower: np.ndarray,
    volumes: Callable[[np.ndarray], np.ndarray],
    refine: Callable[[np.ndarray], np.ndarray] | None = None,
) -> float:
    """Return the smallest of ``volumes(batch)`` over all axes, exactly.

    The axes are tried in batches of 32 in increasing order of ``lower``, a
    lower bound of each axis's volume, and those whose bound exceeds the
    smallest volume found are skipped. ``refine(batch)``, when given, is a
    tighter lower bound, cheaper than ``volumes``, tested on the axes left.
    """
    order = np.argsort(lower, kind="stable")
    best = math.inf
    for start in range(0, len(order), 32):
        batch = order[start : start + 32]
        batch = batch[lower[batch] * (1 - _ENCLOSING_SLACK) <= best]
        if len(batch) == 0:
            break
        if refine is not None:
            batch = batch[refine(batch) * (1 - _ENCLOSING_SLACK) <= best]
            if len(batch) == 0:
                continue
        best = min(best, float(volumes(batch).min()))
    return best


# Hulls of more vertices than this are bounded by a subset of them first.
_SUBSET_ABOVE = 4096


def _bounding_subset(points: np.ndarray) -> np.ndarray:
    """Return ``points``, or for more than ``_SUBSET_ABOVE`` the few extreme ones.

    Those extreme along either way of each of 512 directions spread over the
    sphere: a shape that encloses ``points`` encloses them, so their bounding
    shapes bound those of ``points`` from below, closely for a hull of many
    small faces.
    """
    if len(points) <= _SUBSET_ABOVE:
        return points
    i = np.arange(512) + 0.5
    z = 1 - i / 256
    turn = np.pi * (1 + math.sqrt(5)) * i
    ring = np.sqrt(1 - z * z)
    directions = np.stack([ring * np.cos(turn), ring * np.sin(turn), z])
    # 32 directions at a time, to bound the memory.
    extremes = []
    for start in range(0, 512, 32):
        along = points @ directions[:, start : start + 32]
        extremes += [along.argmax(axis=0), along.argmin(axis=0)]
    return points[np.unique(np.concatenate(extremes))]


# Relative slack of the enclosing tests, against the rounding of distances, and
# a bound on the growth steps of a minimum enclosing circle, far above the at
# most 8 measured for 71,556 hull face normals of 204 meshes.
_ENCLOSING_SLACK = 1e-12
_ENCLOSING_STEPS = 1000


def _axis_frames(axes: np.ndarray) -> np.ndarray:
    """Return ``(k, 3, 3)`` orthonormal frames whose first row is each unit axis.

    The two unit vectors across each axis come from its cross product with the
    coordinate axis it is least aligned with.
    """
    helper = np.eye(3)[np.abs(axes).argmin(axis=1)]
    across = np.cross(axes, helper)
    across /= np.sqrt((across * across).sum(axis=1))[:, None]
    return np.stack([axes, across, np.cross(axes, across)], axis=1)


def _min_cylinder_volume(points: np.ndarray, axes: np.ndarray) -> float:
    """Return the smallest volume of the cylinders about unit ``axes`` enclosing ``points``.

    Each cylinder spans the extent of the points along its axis and has the
    radius of their minimum enclosing circle across it. Exact: an axis is
    skipped only when a lower bound of its volume, from the radius of at
    least half the largest width of the projection, exceeds a volume already
    found (``_smallest_over_axes``): first that of the points, or for many
    points the few extreme ones (``_bounding_subset``), which every cylinder
    enclosing ``points`` encloses, then that of all of them in more
    directions.
    """
    frames = _axis_frames(axes)

    def widths(batch: np.ndarray, of: np.ndarray, turns: int = 2) -> np.ndarray:
        # (axes, 3, points): each point along and across each axis.
        along = frames[batch] @ of.T
        extent = np.ptp(along, axis=2)
        # The largest width across the axis in ``turns`` directions spread
        # evenly over a half turn, the frame's two among them.
        width = extent[:, 1:].max(axis=1)
        for k in range(1, turns):
            if 2 * k != turns:
                turn = k * math.pi / turns
                spread = math.cos(turn) * along[:, 1] + math.sin(turn) * along[:, 2]
                width = np.maximum(width, np.ptp(spread, axis=1))
        return np.pi * (width / 2) ** 2 * extent[:, 0]

    def volumes(batch: np.ndarray) -> np.ndarray:
        along = frames[batch] @ points.T
        height = np.ptp(along[:, 0], axis=1)
        return np.pi * _enclosing_radii(along[:, 1:].transpose(0, 2, 1)) ** 2 * height

    few = _bounding_subset(points)
    # Widths in the 2 directions of each frame, or of few points in 8, on
    # every axis; then of every point in 8 on the axes left. Two directions
    # can miss a diameter by an eighth of a turn, 8 by a thirty-second.
    turns = 2 if len(few) == len(points) else 8
    lower = np.concatenate([widths(c, few, turns) for c in _axis_chunks(len(axes))])
    return _smallest_over_axes(lower, volumes, lambda batch: widths(batch, points, 8))


def _enclosing_radii(points: np.ndarray) -> np.ndarray:
    """Return the radius of the minimum enclosing circle of each set ``(k, n, 2)``.

    The radius of ``_enclosing_circles``.
    """
    return _enclosing_circles(points)[1]


def _enclosing_circles(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the centre and radius of the minimum enclosing circle of each set ``(k, n, 2)``.

    The circle of a support set of at most three points grows by the point
    farthest outside it: the new support set is the smallest circle through
    two or three of the four that encloses all four. The radius grows at
    every step, so no support set returns, and the circle reached encloses
    every point, which makes it the minimum enclosing circle, unique for any
    point set. Rounding can end the growth early on nearly cocircular points:
    the circle through another three of them comes out an ulp smaller while
    the farthest point is still up to 3e-11 outside. Far from the origin, or
    on nearly collinear points, rounding can also leave no circle that
    encloses the four within the slack; the growth of that set then stops
    there, as it does after ``_ENCLOSING_STEPS`` steps. The centre returned is
    the last one and the radius its distance to the farthest point, so the
    circle encloses every point, and the cylinder about it stays an upper
    bound.
    """
    rows = np.arange(len(points))
    first = points[rows, ((points - points[:, :1]) ** 2).sum(axis=2).argmax(axis=1)]
    second = points[rows, ((points - first[:, None]) ** 2).sum(axis=2).argmax(axis=1)]
    support = np.stack([first, second, second], axis=1)
    center = (first + second) / 2
    r2 = ((first - center) ** 2).sum(axis=1)
    slack = (1 + _ENCLOSING_SLACK) ** 2
    # Pairs (a third index repeating the second) and triples of the four points.
    subsets = [(0, 1, 1), (0, 2, 2), (0, 3, 3), (1, 2, 2), (1, 3, 3), (2, 3, 3)]
    subsets += [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    growing = np.ones(len(points), dtype=bool)
    for _ in range(_ENCLOSING_STEPS):
        d2 = ((points - center[:, None]) ** 2).sum(axis=2)
        farthest = d2.argmax(axis=1)
        growing &= d2[rows, farthest] > r2 * slack
        if not growing.any():
            break
        four = np.concatenate([support, points[rows, farthest][:, None]], axis=1)
        new_r2 = np.full_like(r2, np.inf)
        new_center, new_support = center, support
        for subset in subsets:
            trio = four[:, list(subset)]
            c, ok = _circle_center(trio)
            c_r2 = ((trio[:, 0] - c) ** 2).sum(axis=1)
            encloses = (
                ((four - c[:, None]) ** 2).sum(axis=2) <= c_r2[:, None] * slack
            ).all(axis=1)
            take = growing & ok & encloses & (c_r2 < new_r2)
            new_r2 = np.where(take, c_r2, new_r2)
            new_center = np.where(take[:, None], c, new_center)
            new_support = np.where(take[:, None, None], trio, new_support)
        growing &= np.isfinite(new_r2) & (new_r2 > r2)
        r2 = np.where(growing, new_r2, r2)
        center = np.where(growing[:, None], new_center, center)
        support = np.where(growing[:, None, None], new_support, support)
    return center, np.sqrt(((points - center[:, None]) ** 2).sum(axis=2).max(axis=1))


def _circle_center(trio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the center of the circle through each ``(k, 3, 2)`` trio, and where it exists.

    A trio whose last two points coincide stands for the pair of its first
    two, whose circle has them as a diameter. Three collinear points have no
    circle through them.
    """
    a, b, c = trio[:, 0], trio[:, 1], trio[:, 2]
    pair = (b == c).all(axis=1)
    ab, ac = b - a, c - a
    det = 2 * (ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])
    ab2, ac2 = (ab * ab).sum(axis=1), (ac * ac).sum(axis=1)
    ok = pair | (np.abs(det) > _ENCLOSING_SLACK * np.sqrt(ab2 * ac2))
    det = np.where(pair | ~ok, 1.0, det)
    offset = np.stack(
        [ac[:, 1] * ab2 - ab[:, 1] * ac2, ab[:, 0] * ac2 - ac[:, 0] * ab2], axis=1
    )
    center = np.where(pair[:, None], (a + b) / 2, a + offset / det[:, None])
    return center, ok


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


# Dihedral angles of at most this many radians are set to 0: they are the
# rounding residues of coplanar faces. In units of the edges they reached
# 1.7e-13 rad on 60 raw nuclei, and the smallest other angle on 240 blurred
# nuclei was 1.6e-9 rad; this is about the geometric middle.
_FLAT_ANGLE = 1e-11
# Angle defects of at most this many radians are set to 0: at a flat vertex
# the defect is 2 pi minus a sum of face angles, a few ulps of 2 pi.
_FLAT_DEFECT = 1e-14


def _dihedral_angles(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return the angle between the normals of each pair of adjacent faces.

    ``arctan2(|n0 x n1|, n0 . n1)``, set to 0 at or below ``_FLAT_ANGLE``. It
    is accurate to rounding at every angle, where trimesh's
    ``face_adjacency_angles``, ``arccos(n0 . n1)``, cannot resolve angles below
    1.5e-8 and so reads coplanar faces differently at different scales and
    orientations.
    """
    normals = mesh.face_normals[mesh.face_adjacency]
    n0, n1 = normals[:, 0], normals[:, 1]
    angles = np.arctan2(np.linalg.norm(np.cross(n0, n1), axis=1), (n0 * n1).sum(axis=1))
    return np.where(angles <= _FLAT_ANGLE, 0.0, angles)


def _edge_curvatures(mesh: trimesh.Trimesh) -> tuple[np.ndarray, np.ndarray]:
    """Return the edges shared by two faces and their length times signed angle.

    The angle is ``_dihedral_angles``, positive on convex edges. Half the sum
    over the edges is the integral of the mean curvature (Cohen-Steiner and
    Morvan, 2003). Every curvature feature reads its dihedral angles from
    here, on a ``trimesh.Trimesh`` and an ``ArrayMesh`` alike.
    """
    edges = mesh.face_adjacency_edges
    lengths = np.linalg.norm(
        mesh.vertices[edges[:, 1]] - mesh.vertices[edges[:, 0]], axis=1
    )
    signs = np.where(mesh.face_adjacency_convex, 1.0, -1.0)
    return edges, lengths * _dihedral_angles(mesh) * signs


def _avg_mean_curvature(
    mesh: trimesh.Trimesh, edge_curvatures: tuple[Any, Any] | None = None
) -> float:
    """Return the surface average of the mean curvature, ``1 / R`` on a sphere.

    ``edge_curvatures`` is ``_edge_curvatures(mesh)`` when already computed.
    """
    _, terms = _edge_curvatures(mesh) if edge_curvatures is None else edge_curvatures
    return float(terms.sum()) / (2 * float(mesh.area))


def _mixed_area(mesh: trimesh.Trimesh) -> np.ndarray:
    """Return the mixed Voronoi area of each vertex (Meyer et al., 2003).

    Each triangle gives a vertex its Voronoi share, or half or a quarter of its
    area when the triangle is obtuse at or away from that vertex, so the areas
    sum to the mesh area.
    """
    corners = mesh.vertices[mesh.faces]
    angles = mesh.face_angles
    nxt, prv = [1, 2, 0], [2, 0, 1]
    # Squared length of the edge opposite each corner.
    opposite = ((corners[:, nxt] - corners[:, prv]) ** 2).sum(axis=-1)
    # cos / sin, not 1 / tan: with glibc 2.28, np.tan took 2.4e-4 s per angle
    # at pi / 2, the right angles of every voxel mesh, and 1e-8 s elsewhere.
    cot = np.cos(angles) / np.sin(angles)
    voronoi = (opposite[:, nxt] * cot[:, nxt] + opposite[:, prv] * cot[:, prv]) / 8
    face_area = mesh.area_faces[:, None]
    obtuse = angles > np.pi / 2
    area = np.where(
        obtuse,
        face_area / 2,
        np.where(obtuse.any(axis=1, keepdims=True), face_area / 4, voronoi),
    )
    return ordered_bincount(mesh.faces.ravel(), area.ravel(), len(mesh.vertices))


def _principal_curvatures(
    mesh: trimesh.Trimesh, edge_curvatures: tuple[Any, Any] | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the per-vertex principal curvatures ``k1 <= k2`` and vertex areas.

    Gaussian curvature is the angle defect and mean curvature the edge sum of
    ``_edge_curvatures`` over four times the vertex area, as in LONI
    ShapeTools; ``k1, k2 = H -+ sqrt(H^2 - K)``, clamped to ``H`` where
    discretisation makes ``H^2 < K``.

    At a flat vertex both sums are rounding residues. Its dihedral angles are
    0 (``_FLAT_ANGLE``), so its edge sum is exactly 0, and its angle defect is
    set to 0 where its absolute value is at most ``_FLAT_DEFECT`` radians, so
    ``k1 = k2 = 0`` there on either device and at any scale and orientation.
    The defect test matters on raw voxel meshes, where thousands of vertices
    are flat and carry a rounding-residue defect of at most 1e-14 radians:
    left as it is, the defect gives the vertex a Gaussian curvature that the
    surface does not have, and a ``k1, k2`` that Avg Curvedness counts.

    ``edge_curvatures`` is ``_edge_curvatures(mesh)`` when already computed.
    """
    area = _mixed_area(mesh)
    defects = mesh.vertex_defects
    defects = np.where(np.abs(defects) <= _FLAT_DEFECT, 0.0, defects)
    gaussian = defects / area
    edges, terms = (
        _edge_curvatures(mesh) if edge_curvatures is None else edge_curvatures
    )
    edge_sum = ordered_bincount(edges.ravel(), np.repeat(terms, 2), len(mesh.vertices))
    mean = edge_sum / (4 * area)
    spread = np.sqrt(np.maximum(mean**2 - gaussian, 0))
    return mean - spread, mean + spread, area


def _area_average(values: np.ndarray, area: np.ndarray) -> float:
    """Average per-vertex ``values`` over the surface, weighting by vertex area."""
    valid = (area > 0) & np.isfinite(values)
    return float((values[valid] * area[valid]).sum() / area[valid].sum())


def _vertex_curvedness(k1: np.ndarray, k2: np.ndarray) -> np.ndarray:
    """Return the curvedness ``sqrt((k1^2 + k2^2) / 2)`` of each vertex."""
    return np.sqrt((k1**2 + k2**2) / 2)


def _shape_index(k1: np.ndarray, k2: np.ndarray, area: np.ndarray) -> float:
    """Return the shape index averaged by area times curvedness.

    1 on a sphere, 0.5 on a cylinder and 0 on a saddle, as the area average.
    The shape index ``(2 / pi) atan2(k1 + k2, k2 - k1)`` is the angle of
    ``(k1, k2)``, undefined at a planar point and set by rounding near one; its
    weight ``sqrt((k1^2 + k2^2) / 2)``, the curvedness, goes to 0 there, so
    each vertex's term is continuous in its curvatures. NaN,
    with a warning, when no vertex is curved.
    """
    weight = area * _vertex_curvedness(k1, k2)
    valid = (weight > 0) & np.isfinite(weight)
    if not bool(valid.any()):
        warnings.warn(
            "Avg Shape Index is undefined on a mesh whose vertices are all "
            "planar (k1 = k2 = 0); returning NaN.",
            stacklevel=_stacklevel_outside_package(),
        )
        return float("nan")
    shape_index = (2 / np.pi) * np.arctan2(k1 + k2, k2 - k1)
    return _area_average(shape_index, np.where(valid, weight, 0.0))


def _curvedness(k1: np.ndarray, k2: np.ndarray, area: np.ndarray) -> float:
    """Return the area-weighted curvedness, ``1 / R`` on a sphere of radius R."""
    return _area_average(_vertex_curvedness(k1, k2), area)


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

    def at(image: np.ndarray, scale: float, mass: float) -> float:
        return measure(_padded_blur(image, step, scale, tight=True, mass=mass)[0])

    # geomspace returns ``high`` itself as the last scale, where the mask's
    # measure is the one already computed; each image's mass is counted once
    scales = np.geomspace(low, high, n_levels)
    mask_mass, twin_mass = _mask_mass(mask), _mask_mass(twin)
    mask_values = [at(mask, s, mask_mass) for s in scales[:-1]]
    mask_values.append(top_measure)
    ratios = [v / at(twin, s, twin_mass) for v, s in zip(mask_values, scales)]
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
    the field's device. It matches skimage's ``find_contours`` to 1e-15 relative,
    except in saddle cells, which marching cubes resolves
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
    "Oriented Bounding Box Volume": _oriented_bounding_box_volume,
    "Bounding Cylinder Volume": _bounding_cylinder_volume,
    "Bounding Sphere Volume": lambda m: m.bounding_sphere.volume,
    "Convex Hull Volume": lambda m: m.convex_hull.volume,
    "Convex Hull Area": lambda m: m.convex_hull.area,
    "Sphericity": lambda m: sphericity(float(m.volume), float(m.area)),
    "Extent": _extent,
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
