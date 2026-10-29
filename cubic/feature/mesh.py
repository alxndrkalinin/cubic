"""Extract features from a label image using ``trimesh``.

Each label is converted into a surface mesh with marching cubes and measured
with ``trimesh``. ``rtree`` is additionally required for the curvature features.
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

_MISSING_MESH_DEPS: list[str] = []
try:  # pragma: no cover - optional dependency
    import trimesh
    from trimesh.curvature import (
        discrete_mean_curvature_measure,
        discrete_gaussian_curvature_measure,
    )
except ImportError:  # pragma: no cover - import failure path
    trimesh = None  # type: ignore[assignment]
    _MISSING_MESH_DEPS.append("trimesh")
try:  # pragma: no cover - optional dependency
    # ``discrete_mean_curvature_measure`` queries ``mesh.face_adjacency_tree``,
    # which trimesh builds with ``rtree``; without it the curvature features
    # raise ``ModuleNotFoundError`` deep inside trimesh instead of here.
    import rtree  # noqa: F401
except ImportError:  # pragma: no cover - import failure path
    _MISSING_MESH_DEPS.append("rtree")
if "trimesh" in _MISSING_MESH_DEPS:  # pragma: no cover - import failure path
    warnings.warn(
        "trimesh is not installed. Mesh features are unavailable.",
        stacklevel=2,
    )
elif "rtree" in _MISSING_MESH_DEPS:  # pragma: no cover - import failure path
    warnings.warn(
        "rtree is not installed. Mesh curvature features are unavailable.",
        stacklevel=2,
    )

from ..cuda import asnumpy

F = TypeVar("F", bound=Callable[..., Any])


def _require_mesh_deps(*deps: str) -> Callable[[F], F]:
    """Ensure the optional mesh dependencies ``deps`` are importable.

    Only the mean curvature feature needs ``rtree``, so it is required per
    function rather than module-wide: mesh construction and every other feature
    work with ``trimesh`` alone.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            missing = [dep for dep in deps if dep in _MISSING_MESH_DEPS]
            if missing:
                raise RuntimeError(
                    f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} "
                    "required for mesh operations. Install with 'cubic[mesh]'"
                )
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


def mesh_feature_list() -> list[str]:
    """Return the names of the features extracted from a trimesh object."""
    return list(_MESH_FEATURES)


@_require_mesh_deps("trimesh")
def extract_features(
    label_image: np.ndarray,
    features: list[str] | None = None,
    spacing: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
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

    features = features or mesh_feature_list()
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

    return labels, np.asarray(feature_values, dtype=float).reshape(
        len(labels), len(features)
    )


@_require_mesh_deps("trimesh")
def extract_surface_features(
    mask: np.ndarray,
    spacing: Sequence[float] | None = None,
    features: Sequence[str] | None = None,
) -> dict[str, float]:
    """Generate mesh from a single label mask and extract its surface features."""
    mesh = mask2mesh(mask, spacing=spacing)
    return extract_mesh_features(mesh, features=features)


@_require_mesh_deps("trimesh")
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


def ellipsoid_sphericity(v: float, sa: float) -> float:
    """Calculate the sphericity of an ellipsoid given its volume and surface area."""
    return np.power(np.pi, 1 / 3) * np.power(6 * v, 2 / 3) / sa


@_require_mesh_deps("trimesh")
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
        these are computed, so leaving out the curvature features skips their
        cost, which dominates the default set.

    Raises
    ------
    ValueError
        If the mesh volume or surface area is not finite and positive, which is
        what an open (non-watertight) surface reports.
    RuntimeError
        If the mean curvature is requested without ``rtree`` installed.
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


def _avg_gaussian_curvature(mesh: trimesh.Trimesh) -> float:
    """Return the vertex average of the discrete Gaussian curvature measure."""
    return discrete_gaussian_curvature_measure(mesh, mesh.vertices, 1).mean()


@_require_mesh_deps("rtree")
def _avg_mean_curvature(mesh: trimesh.Trimesh) -> float:
    """Return the vertex average of the discrete mean curvature measure.

    ``discrete_mean_curvature_measure`` queries ``mesh.face_adjacency_tree``,
    which trimesh builds with ``rtree``, hence the extra dependency.
    """
    return discrete_mean_curvature_measure(mesh, mesh.vertices, 1).mean()


# Each feature is computed only when requested; the dict order is the default
# column order of ``mesh_feature_list``.
_MESH_FEATURES: dict[str, Callable[[trimesh.Trimesh], float]] = {
    "Area": lambda m: m.area,
    "Volume": lambda m: m.volume,
    "Min Axis Length": lambda m: np.sort(m.extents)[0],
    "Med Axis Length": lambda m: np.sort(m.extents)[1],
    "Max Axis Length": lambda m: np.sort(m.extents)[2],
    "Scale": lambda m: m.scale,
    "Inertia PC1": lambda m: m.principal_inertia_components[0],
    "Inertia PC2": lambda m: m.principal_inertia_components[1],
    "Inertia PC3": lambda m: m.principal_inertia_components[2],
    "Bounding Box Volume": lambda m: m.bounding_box.volume,
    "Oriented Bounding Box Volume": lambda m: m.bounding_box_oriented.volume,
    "Bounding Cylinder Volume": _bounding_cylinder_volume,
    "Bounding Sphere Volume": lambda m: m.bounding_sphere.volume,
    "Convex Hull Volume": lambda m: m.convex_hull.volume,
    "Convex Hull Area": lambda m: m.convex_hull.area,
    "Sphericity": lambda m: ellipsoid_sphericity(m.volume, m.area),
    "Extent": lambda m: m.volume / m.bounding_box.volume,
    "Solidity": lambda m: m.volume / m.convex_hull.volume,
    "Avg Gaussian Curvature": _avg_gaussian_curvature,
    "Avg Mean Curvature": _avg_mean_curvature,
}
