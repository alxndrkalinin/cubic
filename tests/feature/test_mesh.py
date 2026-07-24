"""Tests for mesh-based feature extraction."""

import numpy as np
import pytest
from skimage.measure import marching_cubes

trimesh = pytest.importorskip("trimesh")

from cubic.feature import mesh  # noqa: E402

_CUBE_SIDE = 5


def _cube_volume(origin: int, shape: int = 14) -> np.ndarray:
    """Return a ``shape**3`` volume holding one 5^3 cube of label 1 at ``origin``."""
    volume = np.zeros((shape,) * 3, dtype=np.int32)
    end = origin + _CUBE_SIDE
    volume[origin:end, origin:end, origin:end] = 1
    return volume


def test_extract_features_excludes_background_label(monkeypatch) -> None:
    """Background (label 0) produces no row, matching voxel.regionprops.

    Previously ``np.unique`` kept 0, so a meaningless background mesh and a
    spurious ``label == 0`` row were emitted alongside the real objects. The
    per-label surface computation is stubbed so the test targets the label
    filtering without depending on trimesh's optional ``rtree`` backend.
    """
    names = mesh.mesh_feature_list()
    seen_masks: list[int] = []

    def fake_surface_features(mask: np.ndarray, spacing=None) -> dict[str, float]:
        # Record which label each call corresponds to (mask sum is unique here).
        seen_masks.append(int(mask.sum()))
        return dict.fromkeys(names, 0.0)

    monkeypatch.setattr(mesh, "extract_surface_features", fake_surface_features)

    volume = np.zeros((12, 12, 12), dtype=np.int32)
    volume[2:6, 2:6, 2:6] = 1  # 64 voxels
    volume[7:11, 7:11, 7:11] = 2  # 64 voxels

    labels, feature_values = mesh.extract_features(volume)

    assert sorted(labels.tolist()) == [1, 2]
    assert 0 not in labels.tolist()
    assert feature_values.shape == (2, len(names))
    # Background (label 0) mask sum would be 12**3 - 128 = 1600; never visited.
    assert 1600 not in seen_masks


def test_extract_features_meshes_only_the_label_bbox(monkeypatch) -> None:
    """Each label is meshed over its own bounding box, not the whole volume."""
    seen_shapes: list[tuple[int, ...]] = []

    def fake_surface_features(mask: np.ndarray, spacing=None) -> dict[str, float]:
        seen_shapes.append(mask.shape)
        return dict.fromkeys(mesh.mesh_feature_list(), 0.0)

    monkeypatch.setattr(mesh, "extract_surface_features", fake_surface_features)

    volume = np.zeros((64, 64, 64), dtype=np.int32)
    volume[2:6, 2:6, 2:6] = 1
    volume[10:13, 20:24, 30:35] = 2

    mesh.extract_features(volume)

    assert seen_shapes == [(4, 4, 4), (3, 4, 5)]


def test_extract_features_empty_label_image() -> None:
    """An empty label image gives a (0, n_features) matrix, not shape (0,)."""
    features = ["Volume", "Area"]
    labels, feature_values = mesh.extract_features(
        np.zeros((8, 8, 8), dtype=np.int32), features
    )

    assert labels.shape == (0,)
    # (0,) would break ``feature_values[:, k]`` and vstack across FOVs.
    assert feature_values.shape == (0, 2)
    assert np.vstack([feature_values, np.zeros((1, 2))]).shape == (1, 2)


def test_extract_features_rejects_non_integer_labels() -> None:
    """A float label image raises instead of failing inside find_objects."""
    with pytest.raises(ValueError, match="integer dtype"):
        mesh.extract_features(np.zeros((4, 4, 4), dtype=np.float32))


def test_mask2mesh_pads_border_touching_mask() -> None:
    """A label touching the volume border still yields a closed, positive mesh.

    Without a background border, marching cubes leaves the surface open: the
    5^3 cube at the volume corner gave ``is_watertight=False`` and
    ``volume=-89.52``, which then made sphericity NaN and solidity and extent
    negative without tripping the finiteness guard.
    """
    border = mesh.mask2mesh(_cube_volume(0) == 1)
    interior = mesh.mask2mesh(_cube_volume(4) == 1)

    assert border.is_watertight
    assert border.volume > 0
    # Translation-invariant: the corner cube measures like the interior one.
    assert border.volume == pytest.approx(interior.volume)
    assert border.area == pytest.approx(interior.area)


def test_mask2mesh_spacing_scales_geometry() -> None:
    """Voxel spacing reaches marching cubes, so anisotropy is not ignored."""
    mask = _cube_volume(4) == 1
    isotropic = mesh.mask2mesh(mask)
    anisotropic = mesh.mask2mesh(mask, spacing=(4.0, 1.0, 1.0))

    # A 4x larger z step scales the volume by 4 (it was silently ignored before).
    assert anisotropic.volume == pytest.approx(4 * isotropic.volume)
    assert anisotropic.volume > isotropic.volume

    with pytest.raises(ValueError, match="only once"):
        mesh.mask2mesh(
            mask, spacing=(4.0, 1.0, 1.0), marching_cubes_kwargs={"spacing": (1, 1, 1)}
        )


def test_extract_features_forwards_spacing(monkeypatch) -> None:
    """``extract_features`` threads ``spacing`` down to the per-label mesh."""
    seen: list[object] = []

    def fake_surface_features(mask: np.ndarray, spacing=None) -> dict[str, float]:
        seen.append(spacing)
        return dict.fromkeys(mesh.mesh_feature_list(), 0.0)

    monkeypatch.setattr(mesh, "extract_surface_features", fake_surface_features)
    mesh.extract_features(_cube_volume(4), spacing=(4.0, 1.0, 1.0))

    assert seen == [(4.0, 1.0, 1.0)]


def test_extract_mesh_features_rejects_open_mesh() -> None:
    """A non-watertight mesh raises instead of reporting a negative volume."""
    # Reproduce the pre-padding behaviour: mesh a border-touching mask directly.
    verts, faces, _, _ = marching_cubes(_cube_volume(0) == 1)
    open_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    trimesh.repair.fix_normals(open_mesh)

    assert not open_mesh.is_watertight
    with pytest.raises(ValueError, match="volume must be finite and positive"):
        mesh.extract_mesh_features(open_mesh)


def test_extract_features_border_label_geometry(monkeypatch) -> None:
    """A border-touching label yields the same valid geometry as an interior one.

    Runs the real mesh pipeline end to end with only the curvature step stubbed,
    so it covers the padding fix without the optional ``rtree`` backend. Before
    the fix the corner cube gave ``Volume=-89.52``, ``Sphericity=nan``,
    ``Solidity=-1.20`` and ``Extent=-0.98``.
    """
    monkeypatch.setattr(mesh, "_average_curvatures", lambda _mesh: (0.0, 0.0))
    features = ["Volume", "Area", "Sphericity", "Solidity", "Extent"]

    _, border = mesh.extract_features(_cube_volume(0), features)
    _, interior = mesh.extract_features(_cube_volume(4), features)

    np.testing.assert_allclose(border, interior)
    volume, area, sphericity, solidity, extent = border[0]
    assert volume > 0 and area > 0
    assert 0 < sphericity <= 1
    assert 0 < solidity <= 1.001
    assert 0 < extent <= 1


def test_extract_surface_features_full_pipeline() -> None:
    """The full per-label pipeline returns finite, physically sensible features."""
    pytest.importorskip("rtree", reason="trimesh curvature requires rtree")

    features = mesh.extract_surface_features(_cube_volume(0) == 1)

    assert set(features) == set(mesh.mesh_feature_list())
    assert all(np.isfinite(value) for value in features.values())
    assert features["Volume"] > 0
    assert 0 < features["Sphericity"] <= 1
    assert 0 < features["Solidity"] <= 1.001
    assert 0 < features["Extent"] <= 1


def test_curvature_requires_rtree() -> None:
    """The missing-dependency error names ``rtree``, not just ``trimesh``.

    ``discrete_mean_curvature_measure`` imports ``rtree`` deep inside trimesh,
    so without this guard the failure was a bare ``ModuleNotFoundError`` while
    the guard advertised ``trimesh`` as the only requirement.
    """
    mesh_obj = mesh.mask2mesh(_cube_volume(4) == 1)

    if "rtree" in mesh._MISSING_MESH_DEPS:
        with pytest.raises(RuntimeError, match=r"rtree.*cubic\[mesh\]"):
            mesh._average_curvatures(mesh_obj)
    else:
        gaussian, mean = mesh._average_curvatures(mesh_obj)
        assert np.isfinite(gaussian) and np.isfinite(mean)
