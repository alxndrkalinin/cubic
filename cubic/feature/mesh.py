"""Extract features from a label image using ``trimesh``.

Each label is converted into a surface mesh with marching cubes and measured
with ``trimesh``.
"""

from __future__ import annotations

import warnings
from typing import Any, TypeVar, Callable
from functools import wraps
from collections.abc import Sequence

import numpy as np
from scipy import ndimage
from scipy.spatial import QhullError
from skimage.measure import marching_cubes

_TRIMESH_AVAILABLE = False
try:  # pragma: no cover - optional dependency
    import trimesh
except ImportError:  # pragma: no cover - import failure path
    trimesh = None  # type: ignore[assignment]
    warnings.warn(
        "trimesh is not installed. Mesh features are unavailable.",
        stacklevel=2,
    )
else:
    _TRIMESH_AVAILABLE = True

from ..cuda import asnumpy

F = TypeVar("F", bound=Callable[..., Any])


def _require_trimesh(func: F) -> F:
    """Ensure ``trimesh`` is available before calling ``func``."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if not _TRIMESH_AVAILABLE:
            raise RuntimeError(
                "trimesh is required for mesh operations. Install with 'cubic[mesh]'"
            )
        return func(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def mesh_feature_list() -> list[str]:
    """Return the names of the features extracted from a trimesh object.

    Every feature is independent of the object's orientation. The axis lengths
    are the full axes of the ellipsoid with the same inertia, and ``Extent`` is
    the volume over the oriented bounding box volume. The curvatures are
    averages over the surface, in inverse units of ``spacing`` (squared for the
    Gaussian curvature).
    """
    return list(_MESH_FEATURES)


def _check_feature_names(features: Sequence[str]) -> None:
    """Raise for names that are not in ``mesh_feature_list()``.

    Checked before any meshing: an unknown name used to raise ``KeyError`` only
    after every feature of the first label had been computed.
    """
    unknown = [name for name in features if name not in _MESH_FEATURES]
    if unknown:
        raise ValueError(
            f"Unknown mesh features {unknown}; available: {mesh_feature_list()}"
        )


@_require_trimesh
def extract_features(
    label_image: np.ndarray,
    features: list[str] | None = None,
    spacing: Sequence[float] | None = None,
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

    if features is None:
        features = mesh_feature_list()
    _check_feature_names(features)
    # ``find_objects`` returns the bounding box slices indexed by ``label - 1``,
    # so each label is meshed over its own crop instead of the whole volume.
    bboxes = ndimage.find_objects(label_image)
    feature_values = []
    for label in labels:
        mask = label_image[bboxes[label - 1]] == label
        mesh_features = extract_surface_features(
            mask, spacing=spacing, features=features
        )
        feature_values.append([mesh_features[feat] for feat in features])

    feature_matrix = np.asarray(feature_values, dtype=float).reshape(
        len(labels), len(features)
    )
    return labels, feature_matrix, list(features)


@_require_trimesh
def extract_surface_features(
    mask: np.ndarray,
    spacing: Sequence[float] | None = None,
    features: Sequence[str] | None = None,
) -> dict[str, float]:
    """Generate mesh from a single label mask and extract its surface features."""
    mesh = mask2mesh(mask, spacing=spacing)
    return extract_mesh_features(mesh, features=features)


@_require_trimesh
def mask2mesh(
    mask_3d: np.ndarray,
    spacing: Sequence[float] | None = None,
    marching_cubes_kwargs: dict[str, Any] | None = None,
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
    """
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
    mesh: trimesh.Trimesh, features: Sequence[str] | None = None
) -> dict[str, float]:
    """Extract surface features from a ``trimesh`` object.

    Parameters
    ----------
    mesh : trimesh.Trimesh
        Closed surface of a single object.
    features : sequence of str, optional
        Feature names to compute, defaults to ``mesh_feature_list()``. Only
        these are computed.

    Raises
    ------
    ValueError
        If a feature name is unknown, or if the mesh volume or surface area is
        not finite and positive, which is what an open (non-watertight) surface
        reports.
    """
    # Validate before anything else: an open surface (e.g. a mask meshed without
    # a background border) reports a negative volume, which silently propagates
    # into a NaN sphericity and negative solidity and extent instead of failing.
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

    if features is None:
        features = mesh_feature_list()
    _check_feature_names(features)
    # trimesh caches volume, convex hull and bounding primitives on the mesh, so
    # features sharing them do not recompute them.
    return {name: float(_MESH_FEATURES[name](mesh)) for name in features}


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
    ``axis_major_length``. Unlike the axis-aligned bounding box extents used
    before, it does not change when the object rotates: those moved from
    (12, 16, 48) to (12, 36, 36) for an ellipsoid turned by 45 degrees.
    """
    moments = mesh.principal_inertia_components
    # (I_b + I_c - I_a) / 2 is the second moment along axis a, so it is never
    # negative; the clip only absorbs rounding for flat objects.
    second_moments = np.maximum(moments.sum() - 2 * moments, 0) / 2
    return np.sort(2 * np.sqrt(5 * second_moments / mesh.mass))


def _avg_gaussian_curvature(mesh: trimesh.Trimesh) -> float:
    """Return the surface average of the Gaussian curvature, ``1 / R^2`` on a sphere.

    The angle defects of a closed surface sum to ``2 pi`` times its Euler
    characteristic (Gauss-Bonnet), so this is ``4 pi / area`` for an object
    without handles or cavities and 0 for a torus.
    """
    return mesh.vertex_defects.sum() / mesh.area


def _avg_mean_curvature(mesh: trimesh.Trimesh) -> float:
    """Return the surface average of the mean curvature, ``1 / R`` on a sphere.

    The integral of the mean curvature over a polyhedral surface is half the sum
    of edge length times signed dihedral angle over its edges (Cohen-Steiner and
    Morvan, 2003). Both curvatures used to be vertex averages of trimesh's
    curvature measures over balls of radius 1 in the units of ``spacing``, so
    they did not convert with the unit: giving the spacing in tenths changed
    them 78x and 11x instead of 100x and 10x. The mean curvature measure also
    looped over vertices in Python, 15 s for a 13k-vertex nucleus.
    """
    edges = mesh.vertices[mesh.face_adjacency_edges]
    lengths = np.linalg.norm(edges[:, 1] - edges[:, 0], axis=1)
    signs = np.where(mesh.face_adjacency_convex, 1.0, -1.0)
    return 0.5 * np.sum(lengths * mesh.face_adjacency_angles * signs) / mesh.area


# Each feature is computed only when requested; the dict order is the default
# column order of ``mesh_feature_list``.
_MESH_FEATURES: dict[str, Callable[[trimesh.Trimesh], float]] = {
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
    "Sphericity": lambda m: sphericity(m.volume, m.area),
    "Extent": lambda m: m.volume / m.bounding_box_oriented.volume,
    "Solidity": lambda m: m.volume / m.convex_hull.volume,
    "Avg Gaussian Curvature": _avg_gaussian_curvature,
    "Avg Mean Curvature": _avg_mean_curvature,
}
