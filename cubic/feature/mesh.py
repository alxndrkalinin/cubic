"""Extract features from a label image using ``trimesh``.

Each label is converted into a surface mesh with marching cubes and measured
with ``trimesh``. A CuPy label image or mask stays on the GPU: it is meshed by a
CUDA port of the same marching cubes and measured as an ``ArrayMesh``, which
computes the trimesh quantities on the device and hands the convex hull and
bounding primitives to trimesh on the host.
"""

from __future__ import annotations

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
    on the mask, are measured on a mesh of it.

    No default feature is defined on the image axes. The axis lengths are the
    full axes
    of the ellipsoid with the same inertia, and ``Extent`` is the volume over the
    oriented bounding box volume. Voxelization still moves every feature a little
    with orientation, and the oriented box and bounding cylinder come from
    trimesh's approximate minimum searches, so they move more: ``Extent`` spans
    0.50-0.55 for one ellipsoid turned by 0, 30 and 45 degrees. The mean
    curvature is averaged over the surface, in inverse units of ``spacing``.
    ``Euler Characteristic`` is twice the number of separate pieces plus
    cavities, minus twice the number of handles: 2 for a ball, 4 for a hollow
    ball, 0 for a torus.

    Three of the last four features need curvature at each vertex, which on a
    voxel staircase measures the staircase, so they are computed on a mesh of
    the mask blurred at ``curvature_sigma`` (see ``extract_surface_features``).
    ``Total Absolute Gaussian Curvature`` is the integral of ``|K|`` over the
    surface divided by ``4 pi``: 1 for any convex body and more for every saddle
    region (grooves, waists, the necks of lobes), unitless. ``Avg Shape Index``
    (Koenderink and van Doorn, 1992) runs from -1 (cup) through 0 (saddle) to 1
    (cap), and ``Avg Curvedness`` is ``sqrt((k1^2 + k2^2) / 2)``, in inverse
    units of ``spacing``; both are averaged over the surface by area.
    ``Fractal Dimension`` describes the surface detail finer than
    ``curvature_sigma``, which those three leave out: ``2 - slope`` of log
    surface area against log blur scale, relative to the mask's smooth twin
    (its blur at the top scale). The scales run from the finest voxel spacing
    to ``curvature_sigma``, or over ``fractal_scales`` when it is given (see
    ``extract_surface_features``). It is about 2 for a surface without such
    detail, whatever its size and coarse shape, and rises with roughness, as
    long as the top scale is at most a quarter of the object's thickness
    (``Min Axis Length``). Above that it reads low, and a warning says so.

    ``Outline Fractal Dimension``, the one optional feature, is the same
    construction on the outline of the mask's projection along the first axis:
    ``1 - slope`` of log contour length of the blurred footprint against log
    blur scale, relative to the footprint's smooth twin, with the blur isotropic
    in the in-plane spacing. Its scales run from 4 to 16 times the coarser
    in-plane spacing, or over ``outline_fractal_scales``. It is about 1 for a smooth
    outline and rises with outline roughness, and it does not read low for
    flat objects. It is defined on the image axes, and meant for adherent, flat
    objects imaged along the first axis: turning an object about that axis
    leaves it nearly unchanged, tilting it does not. Its footprint must be
    wide against the top scale: a disc narrower than about 3.2 times the top
    scale (51 voxels across at the default window) gives NaN, with a warning,
    because its smooth twin vanishes, and a smooth disc reads 0.95 at 4 and
    0.99 at 5 times the top scale across. Both fractal dimensions are measured
    on the mask, so ``extract_mesh_features`` cannot compute them.
    """
    if optional:
        return [*_MESH_FEATURES, *_MASK_FEATURES]
    return [*_MESH_FEATURES, *_DEFAULT_MASK_FEATURES]


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
    known = mesh_feature_list(optional=True)
    unknown = [name for name in features if name not in known]
    if unknown:
        raise ValueError(f"Unknown mesh features {unknown}; available: {known}")
    return list(features)


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
        Voxel spacing ``(z, y, x)`` forwarded to marching cubes. Without it an
        anisotropic stack is meshed as if isotropic, scaling volumes and areas
        by the spacing product.
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
        Voxel spacing ``(z, y, x)``.
    features : sequence of str, optional
        Feature names to compute, defaults to ``mesh_feature_list()``. Optional
        features (``mesh_feature_list(optional=True)``, e.g. ``Outline Fractal
        Dimension``) are computed only when named here.
    curvature_sigma : float, optional
        Gaussian blur scale, in units of ``spacing``, of the mask that the
        curvature-distribution features are measured on. Defaults to twice the
        coarsest voxel spacing, which removes the voxel staircase on every axis.
        It sets the scale of the shape being described: larger values keep only
        coarser folds. Without ``fractal_scales`` it is also the top of the
        scale range of ``Fractal Dimension``, which then starts at the finest
        voxel spacing: that feature is NaN, with a warning, unless
        ``curvature_sigma`` exceeds the finest spacing. An object too small to
        survive the blur gets NaN for the features that depend on it, with a
        warning.
    fractal_scales : sequence of float, optional
        ``(lowest, highest)`` blur scale, in units of ``spacing``, of the
        ``Fractal Dimension`` fit, which uses 5 geometric scales between them
        and a smooth twin blurred at the highest. Both must be finite, with
        ``0 < lowest < highest``. Defaults to the finest voxel spacing up to
        ``curvature_sigma``. When the highest scale exceeds a quarter of the
        object's ``Min Axis Length``, the value is still returned, with a
        warning that it is biased low; the warning text is the same for every
        object, so Python's default filter shows it once.
    outline_fractal_scales : sequence of float, optional
        ``(lowest, highest)`` blur scale, in units of ``spacing``, of the
        ``Outline Fractal Dimension`` fit, in the same form as
        ``fractal_scales``. Defaults to 4 to 16 times the coarser in-plane
        spacing. An outline that vanishes at these scales gives NaN, with a
        warning.

    Raises
    ------
    ValueError
        If ``fractal_scales`` or ``outline_fractal_scales`` is not two finite
        numbers with ``0 < lowest < highest``.
    """
    features = _resolve_features(features)
    window = _fractal_window(fractal_scales, "fractal_scales")
    outline_window = _fractal_window(outline_fractal_scales, "outline_fractal_scales")
    mesh = mask2mesh(mask, spacing=spacing)
    blurred = [n for n in features if n in _SMOOTHED_FEATURES or n in _MASK_FEATURES]
    if not blurred:
        return extract_mesh_features(mesh, features=features)
    on_mesh = [name for name in features if name not in _MASK_FEATURES]
    sigma = _curvature_sigma(spacing, curvature_sigma)
    step = _step(spacing)
    # Features that need the mask's blur at curvature_sigma, which is blurred
    # and meshed once for all of them. A Fractal Dimension over its own window
    # does not.
    at_sigma = [
        name
        for name in blurred
        if name in _SMOOTHED_FEATURES
        or (name == "Fractal Dimension" and window is None)
    ]
    values: dict[str, float] = {}
    smooth_mesh = None
    if at_sigma:
        smooth_volume, pad = _padded_blur(mask, step, sigma)
        level_set = _level_set(smooth_volume, step)
        if level_set is None:
            warnings.warn(
                f"Mask of {int(mask.sum())} voxels vanishes when blurred at "
                f"curvature_sigma={sigma:g}; returning NaN for {at_sigma}.",
                stacklevel=2,
            )
            values = dict.fromkeys(at_sigma, float("nan"))
        elif any(name in _SMOOTHED_FEATURES for name in features):
            smooth_mesh = _level_set_mesh(level_set, pad, step)
    values.update(
        extract_mesh_features(
            mesh,
            features=[name for name in on_mesh if name not in values],
            smooth_mesh=smooth_mesh,
        )
    )
    if "Outline Fractal Dimension" in features:
        values["Outline Fractal Dimension"] = _outline_fractal_dimension(
            mask, step, outline_window
        )
    if "Fractal Dimension" in features and "Fractal Dimension" not in values:
        if window is not None:
            low, high = window
            top_volume, top_pad = _padded_blur(mask, step, high)
            top_level_set = _level_set(top_volume, step)
            values["Fractal Dimension"] = _fractal_dimension(
                mask, step, low, high, top_volume, top_pad, top_level_set
            )
        elif sigma <= step.min():
            warnings.warn(
                f"Fractal Dimension needs curvature_sigma above the finest voxel "
                f"spacing {step.min():g}; got {sigma:g}. Returning NaN.",
                stacklevel=2,
            )
            values["Fractal Dimension"] = float("nan")
        else:
            values["Fractal Dimension"] = _fractal_dimension(
                mask, step, float(step.min()), sigma, smooth_volume, pad, level_set
            )
        top = sigma if window is None else window[1]
        # Half the shortest axis of the same-inertia ellipsoid is the smallest
        # half-extent; the estimate is unbiased only while the top scale stays
        # below half of it.
        if np.isfinite(values["Fractal Dimension"]) and top > (
            float(_equivalent_ellipsoid_axes(mesh)[0]) / 4
        ):
            # One text for every object, so that the default filter reports a
            # label image of many thin objects once.
            warnings.warn(_THIN_OBJECT_WARNING, stacklevel=2)
    return {name: values[name] for name in features}


_THIN_OBJECT_WARNING = (
    "Fractal Dimension: the 3D estimate is biased low for objects thinner than "
    "four times the top blur scale of its fit (curvature_sigma, or the highest "
    "of fractal_scales); at least one object is that thin. For flat objects "
    "imaged along the first axis, see Outline Fractal Dimension."
)


def _fractal_window(
    scales: Sequence[float] | None, keyword: str
) -> tuple[float, float] | None:
    """Return ``(lowest, highest)`` of the window ``keyword``, or None for None.

    Raises ``ValueError``, naming ``keyword``, unless they are two finite
    numbers with ``0 < lowest < highest``.
    """
    if scales is None:
        return None
    message = (
        f"{keyword} must be two finite numbers (lowest, highest) with "
        f"0 < lowest < highest; got {scales!r}"
    )
    if isinstance(scales, str):
        raise ValueError(message)
    try:
        low, high = (float(s) for s in scales)
    except (TypeError, ValueError) as e:
        raise ValueError(message) from e
    if not (np.isfinite(low) and np.isfinite(high) and 0 < low < high):
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
    """
    if get_device(volume) == "GPU":
        if kwargs:
            raise ValueError(f"GPU marching cubes does not support {sorted(kwargs)}.")
        return _marching_cubes_gpu(volume, level, spacing, allow_degenerate)
    verts, faces, _, _ = marching_cubes(
        volume, level, spacing=spacing, allow_degenerate=allow_degenerate, **kwargs
    )
    return verts, faces


def _closed_mesh(verts: np.ndarray, faces: np.ndarray) -> Any:
    """Wrap marching-cubes output as an outward-oriented mesh on its device."""
    if get_device(verts) == "GPU":
        return ArrayMesh(verts, faces)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    trimesh.repair.fix_normals(mesh)
    return mesh


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


def _padded_blur(
    mask: np.ndarray, step: np.ndarray, sigma: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return the mask blurred at ``sigma`` (units of ``step``) and its padding.

    The mask may have any number of axes, one ``step`` per axis. The blur is
    isotropic in physical units, so its width in voxels differs per axis on an
    anisotropic grid.
    """
    sigma_voxels = sigma / step
    # Pad past the blur's reach so it neither clips at the crop nor touches the
    # border, which would leave the surface open.
    pad = np.ceil(3 * sigma_voxels).astype(int) + 1
    # Rounds like SciPy on the GPU too, so both devices mesh the same volume.
    volume = gaussian_filter_like_scipy(
        np.pad(mask.astype(np.float32), [(p, p) for p in pad]), sigma_voxels
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

    The mask is padded with one voxel of background on every side. Without it a
    label touching the volume border yields an open (non-watertight) surface,
    whose ``volume`` is meaningless and often negative, which in turn makes
    sphericity NaN and solidity and extent negative.

    Parameters
    ----------
    mask_3d : np.ndarray
        Boolean mask of a single object. A CuPy mask is meshed on the GPU and
        returned as an ``ArrayMesh``; a NumPy one as a ``trimesh.Trimesh``.
    spacing : sequence of float, optional
        Voxel spacing ``(z, y, x)`` passed to marching cubes, so that lengths,
        areas and volumes are in physical units.
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
        object is too small for that blur.
    """
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

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Closed surface of a single object.
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
        or if the mesh volume or surface area is not finite and positive, which
        is what an open (non-watertight) surface reports.
    """
    features = list(_MESH_FEATURES) if features is None else _resolve_features(features)
    from_mask = [name for name in features if name in _MASK_FEATURES]
    if from_mask:
        raise ValueError(
            f"{from_mask} is measured on the mask at several blur scales, not on "
            "one mesh; use extract_surface_features or extract_features."
        )
    # Validate the mesh before any feature: an open surface (e.g. a mask meshed
    # without a background border) reports a negative volume, which silently
    # propagates into a NaN sphericity and negative solidity and extent instead
    # of failing.
    volume = float(mesh.volume)
    surface_area = float(mesh.area)
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError(
            f"Mesh volume must be finite and positive; got {volume}. "
            f"The mesh is likely not watertight (is_watertight={mesh.is_watertight})."
        )
    if not np.isfinite(surface_area) or surface_area <= 0:
        raise ValueError(
            f"Mesh surface area must be finite and positive; got {surface_area}."
        )

    curvature_mesh = mesh if smooth_mesh is None else smooth_mesh
    # trimesh caches volume, convex hull and bounding primitives on the mesh, so
    # features sharing them do not recompute them.
    return {
        name: float(
            _MESH_FEATURES[name](curvature_mesh if name in _SMOOTHED_FEATURES else mesh)
        )
        for name in features
    }


def _bounding_cylinder_volume(mesh: trimesh.Trimesh) -> float:
    """Return the minimum bounding cylinder volume, 0 for degenerate meshes."""
    try:
        return mesh.bounding_cylinder.volume
    except (ValueError, QhullError) as e:
        # Degenerate meshes fail in the convex hull (QhullError) or in the
        # eigendecomposition of the inertia tensor (LinAlgError, a ValueError).
        warnings.warn(
            f"Unable extract bounding cylinder volume. Returning 0. Exception: {e}",
            stacklevel=2,
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
    angle over the edges (Cohen-Steiner and Morvan, 2003). Before trimesh 4.6.12
    the angles were unsigned, which overstated it on any concave surface.
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


def _avg_shape_index(mesh: trimesh.Trimesh) -> float:
    """Return the area-weighted shape index, 1 on a sphere and 0 on a saddle."""
    k1, k2, area = _principal_curvatures(mesh)
    return _area_average((2 / np.pi) * np.arctan2(k1 + k2, k2 - k1), area)


def _avg_curvedness(mesh: trimesh.Trimesh) -> float:
    """Return the area-weighted curvedness, ``1 / R`` on a sphere of radius R."""
    k1, k2, area = _principal_curvatures(mesh)
    return _area_average(np.sqrt((k1**2 + k2**2) / 2), area)


def _smooth_twin(blurred: np.ndarray, pad: np.ndarray) -> np.ndarray:
    """Return the mask's blur thresholded at 0.5, cropped to the mask's own frame.

    Outside that frame every mask voxel lies to one side along some axis, so
    the separable blur there is at most the kernel's weight beyond its centre,
    ``(1 - w0) / 2 < 0.5``: the threshold loses nothing to the crop.
    """
    return (blurred > 0.5)[tuple(slice(p, -p) for p in pad)]


def _fractal_dimension(
    mask_3d: np.ndarray,
    step: np.ndarray,
    low: float,
    high: float,
    blurred: np.ndarray,
    pad: np.ndarray,
    level_set: tuple[np.ndarray, np.ndarray] | None,
    n_levels: int = 5,
) -> float:
    """Return ``2 - slope`` of log area ratio against log blur scale.

    The area of a surface of fractal dimension ``D`` measured at resolution
    ``s`` scales as ``s^(2 - D)``. The mask is blurred at ``n_levels`` scales
    from ``low`` up to ``high``, and the area of each blur's 0.5 level is
    divided by the same area for the mask's smooth twin, its blur at ``high``
    thresholded at 0.5. The twin shares the object's coarse shape and voxel
    grid, so the shrinking of curved surfaces under blur and the voxel
    staircase cancel in the ratio, which leaves the detail finer than ``high``.
    Without the twin, smooth ellipsoids of nucleus size measure 2.05-2.20.
    Every step runs on the mask's device; NaN, with a warning, if any level
    vanishes. ``_outline_fractal_dimension`` is the 2D form on the footprint.

    ``blurred`` is the mask's blur at ``high`` with its padding ``pad``, and
    ``level_set`` its 0.5 level from ``_level_set``, which the caller has
    already computed: the twin is thresholded from the first and cropped to the
    mask's frame, and the second is the mask's level at the last scale.
    """
    twin = _smooth_twin(blurred, pad)

    def area(volume: np.ndarray, scale: float) -> float:
        return _level_set_area(_level_set(_padded_blur(volume, step, scale)[0], step))

    # geomspace returns ``high`` itself as the last scale, where the mask's
    # level set is the one already computed
    scales = np.geomspace(low, high, n_levels)
    mask_areas = [area(mask_3d, s) for s in scales[:-1]]
    mask_areas.append(_level_set_area(level_set))
    ratios = [a / area(twin, s) for a, s in zip(mask_areas, scales)]
    if not np.all(np.isfinite(ratios)):
        warnings.warn(
            f"Fractal Dimension: the mask or its smooth twin vanishes when blurred "
            f"at a scale up to {high:g} (an object thinner than the blur); "
            f"returning NaN.",
            stacklevel=3,
        )
        return float("nan")
    return 2 - float(np.polyfit(np.log(scales), np.log(ratios), 1)[0])


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
    mask_3d: np.ndarray,
    step: np.ndarray,
    window: tuple[float, float] | None,
    n_levels: int = 5,
) -> float:
    """Return ``1 - slope`` of log contour-length ratio against log blur scale.

    The 2D form of ``_fractal_dimension`` on the mask's footprint along the
    first axis: the footprint is blurred at ``n_levels`` geometric scales over
    ``window`` (4 to 16 times the coarser in-plane spacing by default), and the
    length of each blur's 0.5 contour is divided by the same length for the
    footprint's smooth twin, its blur at the top scale thresholded at 0.5.
    Every step runs on the mask's device; NaN, with a warning, if any contour
    vanishes.
    """
    footprint = mask_3d.any(axis=0)
    plane_step = step[1:]
    if window is None:
        low, high = 4 * float(plane_step.max()), 16 * float(plane_step.max())
    else:
        low, high = window
    top, pad = _padded_blur(footprint, plane_step, high)
    twin = _smooth_twin(top, pad)

    def length(image: np.ndarray, scale: float) -> float:
        return _contour_length(_padded_blur(image, plane_step, scale)[0], plane_step)

    # geomspace returns ``high`` itself as the last scale, where the footprint's
    # blur is the one already computed
    scales = np.geomspace(low, high, n_levels)
    lengths = [length(footprint, s) for s in scales[:-1]]
    lengths.append(_contour_length(top, plane_step))
    ratios = [a / length(twin, s) for a, s in zip(lengths, scales)]
    if not np.all(np.isfinite(ratios)):
        warnings.warn(
            f"Outline Fractal Dimension: the footprint or its smooth twin vanishes "
            f"when blurred at a scale up to {high:g} (an outline narrower than the "
            f"blur); returning NaN.",
            stacklevel=3,
        )
        return float("nan")
    return 1 - float(np.polyfit(np.log(scales), np.log(ratios), 1)[0])


# Measured on the smoothed mesh when one is available; see mesh_feature_list.
_SMOOTHED_FEATURES = (
    "Total Absolute Gaussian Curvature",
    "Avg Shape Index",
    "Avg Curvedness",
)

# Measured on the mask itself, at several blur scales; see mesh_feature_list.
# The optional ones are computed only when requested by name.
_DEFAULT_MASK_FEATURES = ("Fractal Dimension",)
_OPTIONAL_FEATURES = ("Outline Fractal Dimension",)
_MASK_FEATURES = (*_DEFAULT_MASK_FEATURES, *_OPTIONAL_FEATURES)

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
    "Avg Shape Index": _avg_shape_index,
    "Avg Curvedness": _avg_curvedness,
}
