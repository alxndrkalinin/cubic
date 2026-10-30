"""Extract features from a label image using ``trimesh``.

Each label is converted into a surface mesh with marching cubes and measured
with ``trimesh``.
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

from ..cuda import asnumpy

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


def mesh_feature_list() -> list[str]:
    """Return the names of the features extracted from a trimesh object.

    No feature is defined on the image axes. The axis lengths are the full axes
    of the ellipsoid with the same inertia, and ``Extent`` is the volume over the
    oriented bounding box volume. Voxelization still moves every feature a little
    with orientation, and the oriented box and bounding cylinder come from
    trimesh's approximate minimum searches, so they move more: ``Extent`` spans
    0.50-0.55 for one ellipsoid turned by 0, 30 and 45 degrees. The mean
    curvature is averaged over the surface, in inverse units of ``spacing``.
    ``Euler Characteristic`` is twice the number of separate pieces plus
    cavities, minus twice the number of handles: 2 for a ball, 4 for a hollow
    ball, 0 for a torus.

    The last three features need curvature at each vertex, which on a voxel
    staircase measures the staircase, so they are computed on a mesh of the mask
    blurred at ``curvature_sigma`` (see ``extract_surface_features``).
    ``Total Absolute Gaussian Curvature`` is the integral of ``|K|`` over the
    surface divided by ``4 pi``: 1 for any convex body and more for every saddle
    region (grooves, waists, the necks of lobes), unitless. ``Avg Shape Index``
    (Koenderink and van Doorn, 1992) runs from -1 (cup) through 0 (saddle) to 1
    (cap), and ``Avg Curvedness`` is ``sqrt((k1^2 + k2^2) / 2)``, in inverse
    units of ``spacing``; both are averaged over the surface by area.
    """
    return list(_MESH_FEATURES)


def _resolve_features(features: Sequence[str] | None) -> list[str]:
    """Return ``features``, or all of them for ``None``; raise for unknown names."""
    if features is None:
        return mesh_feature_list()
    # A string is a sequence too, and would be checked letter by letter.
    if isinstance(features, str):
        raise TypeError(
            f"features must be a sequence of names, not the string {features!r}"
        )
    unknown = [name for name in features if name not in _MESH_FEATURES]
    if unknown:
        raise ValueError(
            f"Unknown mesh features {unknown}; available: {mesh_feature_list()}"
        )
    return list(features)


@_require_trimesh
def extract_features(
    label_image: np.ndarray,
    features: list[str] | None = None,
    spacing: Sequence[float] | None = None,
    curvature_sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract mesh features for every label of a label image.

    Each label is meshed independently over its own bounding box, which is
    orders of magnitude cheaper than running marching cubes over the full volume
    once per label.

    Parameters
    ----------
    label_image : np.ndarray
        3D integer label image. Background (label 0) is excluded.
    features : list of str, optional
        Feature names to extract, defaults to ``mesh_feature_list()``; the
        columns follow this order. Only these features are computed.
    spacing : sequence of float, optional
        Voxel spacing ``(z, y, x)`` forwarded to marching cubes. Without it an
        anisotropic stack is meshed as if isotropic, scaling volumes and areas
        by the spacing product.
    curvature_sigma : float, optional
        Blur scale, in units of ``spacing``, of the mesh behind the
        curvature-distribution features; see ``extract_surface_features``.

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
    label_image = asnumpy(label_image)
    if not np.issubdtype(label_image.dtype, np.integer):
        raise ValueError(
            f"label_image must have an integer dtype; got {label_image.dtype}"
        )
    # Keep only positive labels, matching ``voxel.regionprops``, which ignores
    # both the background and negative values.
    labels = np.unique(label_image)
    labels = labels[labels > 0]

    # Resolved before any meshing, so an unknown name fails even on an image
    # without labels.
    features = _resolve_features(features)
    if not features:
        return labels, np.empty((len(labels), 0)), features
    # ``find_objects`` returns the bounding box slices indexed by ``label - 1``,
    # so each label is meshed over its own crop instead of the whole volume.
    bboxes = ndimage.find_objects(label_image)
    feature_values = []
    for label in labels:
        mask = label_image[bboxes[label - 1]] == label
        mesh_features = extract_surface_features(
            mask, spacing=spacing, features=features, curvature_sigma=curvature_sigma
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
) -> dict[str, float]:
    """Generate mesh from a single label mask and extract its surface features.

    Parameters
    ----------
    mask : np.ndarray
        Boolean mask of a single object.
    spacing : sequence of float, optional
        Voxel spacing ``(z, y, x)``.
    features : sequence of str, optional
        Feature names to compute, defaults to ``mesh_feature_list()``.
    curvature_sigma : float, optional
        Gaussian blur scale, in units of ``spacing``, of the mask that the
        curvature-distribution features are measured on. Defaults to twice the
        coarsest voxel spacing, which removes the voxel staircase on every axis.
        It sets the scale of the shape being described: larger values keep only
        coarser folds. An object too small to survive the blur gets NaN for those
        features, with a warning.
    """
    features = _resolve_features(features)
    mesh = mask2mesh(mask, spacing=spacing)
    smoothed = [name for name in features if name in _SMOOTHED_FEATURES]
    if not smoothed:
        return extract_mesh_features(mesh, features=features)
    sigma = _curvature_sigma(spacing, curvature_sigma)
    smooth_mesh = _smooth_mask2mesh(mask, spacing, sigma)
    if smooth_mesh is not None:
        return extract_mesh_features(mesh, features=features, smooth_mesh=smooth_mesh)
    warnings.warn(
        f"Mask of {int(mask.sum())} voxels vanishes when blurred at "
        f"curvature_sigma={sigma:g}; returning NaN for {smoothed}.",
        stacklevel=2,
    )
    rest = extract_mesh_features(
        mesh, features=[name for name in features if name not in smoothed]
    )
    return {name: rest.get(name, float("nan")) for name in features}


def _curvature_sigma(spacing: Sequence[float] | None, sigma: float | None) -> float:
    """Return ``sigma``, or twice the coarsest voxel spacing for ``None``."""
    if sigma is None:
        return 2 * (1.0 if spacing is None else float(max(spacing)))
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"curvature_sigma must be finite and positive; got {sigma}")
    return sigma


def _smooth_mask2mesh(
    mask_3d: np.ndarray, spacing: Sequence[float] | None, sigma: float
) -> trimesh.Trimesh | None:
    """Mesh the 0.5 level of the mask blurred at ``sigma``; None if it vanishes.

    The blur is isotropic in physical units, so its width in voxels differs per
    axis on an anisotropic stack.
    """
    step = np.ones(3) if spacing is None else np.asarray(spacing, dtype=float)
    sigma_voxels = sigma / step
    # Pad past the blur's reach so it neither clips at the crop nor touches the
    # border, which would leave the surface open.
    pad = np.ceil(3 * sigma_voxels).astype(int) + 1
    volume = ndimage.gaussian_filter(
        np.pad(mask_3d.astype(np.float32), [(p, p) for p in pad]), sigma_voxels
    )
    if volume.max() <= 0.5:
        return None
    verts, faces, _, _ = marching_cubes(
        volume, 0.5, spacing=tuple(step), allow_degenerate=False
    )
    mesh = trimesh.Trimesh(vertices=verts - pad * step, faces=faces)
    trimesh.repair.fix_normals(mesh)
    return mesh


@_require_trimesh
def mask2mesh(
    mask_3d: np.ndarray,
    spacing: Sequence[float] | None = None,
    marching_cubes_kwargs: dict[str, Any] | None = None,
    sigma: float | None = None,
) -> trimesh.Trimesh:
    """Convert 3D mask to a mesh using marching cubes.

    The mask is padded with one voxel of background on every side. Without it a
    label touching the volume border yields an open (non-watertight) surface,
    whose ``volume`` is meaningless and often negative, which in turn makes
    sphericity NaN and solidity and extent negative.

    Parameters
    ----------
    mask_3d : np.ndarray
        Boolean mask of a single object.
    spacing : sequence of float, optional
        Voxel spacing ``(z, y, x)`` passed to marching cubes, so that lengths,
        areas and volumes are in physical units.
    marching_cubes_kwargs : dict, optional
        Extra keyword arguments for ``skimage.measure.marching_cubes``.
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
    verts, faces, _, _ = marching_cubes(np.pad(mask_3d, 1), **marching_cubes_kwargs)
    # Undo the 1-voxel pad so vertices stay in the input's coordinate frame.
    # Without this every vertex, and therefore ``bounds`` and ``centroid``, sits
    # one voxel high on each axis -- scaled by ``spacing``, so the offset is in
    # physical units. Every feature in ``mesh_feature_list`` is
    # translation-invariant, but ``mask2mesh`` is public and its mesh is
    # expected to overlay the image.
    verts = verts - np.asarray(marching_cubes_kwargs.get("spacing", 1.0), dtype=float)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    trimesh.repair.fix_normals(mesh)
    return mesh


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
        Feature names to compute, defaults to ``mesh_feature_list()``. Only
        these are computed.
    smooth_mesh : trimesh.Trimesh, optional
        Smoothed surface of the same object (e.g. ``mask2mesh(mask, sigma=...)``)
        for the curvature-distribution features. Without it they are measured on
        ``mesh``, which on a raw voxel mesh mostly measures the staircase.

    Raises
    ------
    ValueError
        If a feature name is unknown, or if the mesh volume or surface area is
        not finite and positive, which is what an open (non-watertight) surface
        reports.
    """
    features = _resolve_features(features)
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


# Measured on the smoothed mesh when one is available; see mesh_feature_list.
_SMOOTHED_FEATURES = (
    "Total Absolute Gaussian Curvature",
    "Avg Shape Index",
    "Avg Curvedness",
)

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
