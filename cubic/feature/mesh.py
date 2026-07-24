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

    Only the curvature features need ``rtree``, so it is required per function
    rather than module-wide: mesh construction and every other feature work with
    ``trimesh`` alone.
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
    """Return a list of features extracted from a trimesh object."""
    return [
        "Area",
        "Volume",
        "Min Axis Length",
        "Med Axis Length",
        "Max Axis Length",
        "Scale",
        "Inertia PC1",
        "Inertia PC2",
        "Inertia PC3",
        "Bounding Box Volume",
        "Oriented Bounding Box Volume",
        "Bounding Cylinder Volume",
        "Bounding Sphere Volume",
        "Convex Hull Volume",
        "Convex Hull Area",
        "Sphericity",
        "Extent",
        "Solidity",
        "Avg Gaussian Curvature",
        "Avg Mean Curvature",
    ]


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
        columns follow this order.
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
        mesh_features = extract_surface_features(mask, spacing=spacing)
        feature_values.append([mesh_features[feat] for feat in features])

    return labels, np.asarray(feature_values, dtype=float).reshape(
        len(labels), len(features)
    )


@_require_mesh_deps("trimesh")
def extract_surface_features(
    mask: np.ndarray, spacing: Sequence[float] | None = None
) -> dict[str, float]:
    """Generate mesh from a single label mask and extract its surface features."""
    mesh = mask2mesh(mask, spacing=spacing)
    return extract_mesh_features(mesh)


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
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    trimesh.repair.fix_normals(mesh)
    return mesh


def ellipsoid_sphericity(v: float, sa: float) -> float:
    """Calculate the sphericity of an ellipsoid given its volume and surface area."""
    return np.power(np.pi, 1 / 3) * np.power(6 * v, 2 / 3) / sa


@_require_mesh_deps("trimesh")
def extract_mesh_features(mesh: trimesh.Trimesh) -> dict[str, float]:
    """Extract surface features from a ``trimesh`` object.

    Raises
    ------
    ValueError
        If the mesh volume or surface area is not finite and positive, which is
        what an open (non-watertight) surface reports.
    RuntimeError
        If ``rtree`` is missing, as the curvature features need it.
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

    axis_len = sorted(mesh.extents)
    inertia_pcs = mesh.principal_inertia_components

    try:
        bounding_volume = mesh.bounding_cylinder.volume
    except (ValueError, QhullError) as e:
        # Degenerate meshes fail in the convex hull (QhullError) or in the
        # eigendecomposition of the inertia tensor (LinAlgError, a ValueError).
        bounding_volume = 0.0
        warnings.warn(
            f"Unable extract bounding cylinder volume. Returning 0. Exception: {e}",
            stacklevel=2,
        )

    gaussian_curvature, mean_curvature = _average_curvatures(mesh)

    return {
        "Area": surface_area,
        "Volume": volume,
        "Min Axis Length": axis_len[0],
        "Med Axis Length": axis_len[1],
        "Max Axis Length": axis_len[2],
        "Scale": mesh.scale,
        "Inertia PC1": inertia_pcs[0],
        "Inertia PC2": inertia_pcs[1],
        "Inertia PC3": inertia_pcs[2],
        "Bounding Box Volume": mesh.bounding_box.volume,
        "Oriented Bounding Box Volume": mesh.bounding_box_oriented.volume,
        "Bounding Cylinder Volume": bounding_volume,
        "Bounding Sphere Volume": mesh.bounding_sphere.volume,
        "Convex Hull Volume": mesh.convex_hull.volume,
        "Convex Hull Area": mesh.convex_hull.area,
        "Sphericity": ellipsoid_sphericity(volume, surface_area),
        "Extent": volume / mesh.bounding_box.volume,
        "Solidity": volume / mesh.convex_hull.volume,
        "Avg Gaussian Curvature": gaussian_curvature,
        "Avg Mean Curvature": mean_curvature,
    }


@_require_mesh_deps("trimesh", "rtree")
def _average_curvatures(mesh: trimesh.Trimesh) -> tuple[float, float]:
    """Return the vertex-averaged (Gaussian, mean) discrete curvatures.

    ``discrete_mean_curvature_measure`` queries ``mesh.face_adjacency_tree``,
    which trimesh builds with ``rtree``, hence the extra dependency.
    """
    gaussian = discrete_gaussian_curvature_measure(mesh, mesh.vertices, 1).mean()
    mean = discrete_mean_curvature_measure(mesh, mesh.vertices, 1).mean()
    return float(gaussian), float(mean)
