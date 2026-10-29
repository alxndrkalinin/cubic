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


def _rotated_ellipsoid(rot_deg: float) -> np.ndarray:
    """Return a label image of an ellipsoid with full axes (12, 48, 16).

    The ellipsoid is rotated by ``rot_deg`` about axis 0, i.e. within the
    plane of axes 1 and 2.
    """
    z, y, x = np.meshgrid(
        *[np.arange(n) - (n - 1) / 2 for n in (20, 60, 60)], indexing="ij"
    )
    t = np.deg2rad(rot_deg)
    y, x = np.cos(t) * y + np.sin(t) * x, -np.sin(t) * y + np.cos(t) * x
    return ((z / 6) ** 2 + (y / 24) ** 2 + (x / 8) ** 2 <= 1).astype(np.int32)


def test_extract_features_excludes_background_label(monkeypatch) -> None:
    """Background (label 0) produces no row, matching voxel.regionprops.

    Previously ``np.unique`` kept 0, so a meaningless background mesh and a
    spurious ``label == 0`` row were emitted alongside the real objects. The
    per-label surface computation is stubbed so the test targets the label
    filtering alone.
    """
    names = mesh.mesh_feature_list()
    seen_masks: list[int] = []

    def fake_surface_features(
        mask: np.ndarray, spacing=None, features=None
    ) -> dict[str, float]:
        # Record which label each call corresponds to (mask sum is unique here).
        seen_masks.append(int(mask.sum()))
        return dict.fromkeys(names, 0.0)

    monkeypatch.setattr(mesh, "extract_surface_features", fake_surface_features)

    volume = np.zeros((12, 12, 12), dtype=np.int32)
    volume[2:6, 2:6, 2:6] = 1  # 64 voxels
    volume[7:11, 7:11, 7:11] = 2  # 64 voxels

    labels, feature_values, columns = mesh.extract_features(volume)

    assert sorted(labels.tolist()) == [1, 2]
    assert 0 not in labels.tolist()
    assert feature_values.shape == (2, len(names))
    assert columns == names
    # Background (label 0) mask sum would be 12**3 - 128 = 1600; never visited.
    assert 1600 not in seen_masks


def test_extract_features_meshes_only_the_label_bbox(monkeypatch) -> None:
    """Each label is meshed over its own bounding box, not the whole volume."""
    seen_shapes: list[tuple[int, ...]] = []

    def fake_surface_features(
        mask: np.ndarray, spacing=None, features=None
    ) -> dict[str, float]:
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
    labels, feature_values, columns = mesh.extract_features(
        np.zeros((8, 8, 8), dtype=np.int32), features
    )

    assert labels.shape == (0,)
    # (0,) would break ``feature_values[:, k]`` and vstack across FOVs.
    assert feature_values.shape == (0, 2)
    assert columns == features
    assert np.vstack([feature_values, np.zeros((1, 2))]).shape == (1, 2)


@pytest.mark.parametrize("label", [0, 1])
def test_extract_features_rejects_unknown_feature_names(label: int) -> None:
    """An unknown name raises before any label is meshed, even with no labels."""
    with pytest.raises(ValueError, match=r"Unknown mesh features \['Scale'\]"):
        mesh.extract_features(_cube_volume(4) * label, ["Volume", "Scale"])


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


@pytest.mark.parametrize("spacing", [None, (4.0, 1.0, 1.0)])
def test_mask2mesh_vertices_stay_in_the_input_frame(
    spacing: tuple[float, float, float] | None,
) -> None:
    """The 1-voxel pad must not shift the mesh out of the image frame.

    ``mask2mesh`` pads to close border-touching surfaces, so the pad has to be
    subtracted again. Left in, every vertex sat one voxel high on each axis —
    scaled by ``spacing``, so a z-spacing of 4 displaced the mesh by 4.0
    physical units, and ``bounds``/``centroid`` no longer overlaid the image.
    """
    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[2:4, 2:4, 2:4] = True  # voxel centres 2 and 3 on every axis

    kwargs = {"spacing": spacing} if spacing is not None else {}
    got = mesh.mask2mesh(mask, marching_cubes_kwargs=kwargs)

    step = np.asarray(spacing if spacing is not None else (1.0, 1.0, 1.0))
    # Marching cubes puts the surface midway between the on and off voxels.
    np.testing.assert_allclose(got.centroid, 2.5 * step)
    np.testing.assert_allclose(got.bounds[0], 1.5 * step)
    np.testing.assert_allclose(got.bounds[1], 3.5 * step)


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

    def fake_surface_features(
        mask: np.ndarray, spacing=None, features=None
    ) -> dict[str, float]:
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


def test_extract_features_border_label_geometry() -> None:
    """A border-touching label yields the same valid geometry as an interior one.

    Runs the real mesh pipeline end to end. Before the padding fix the corner
    cube gave ``Volume=-89.52``, ``Sphericity=nan``, ``Solidity=-1.20`` and
    ``Extent=-0.98``.
    """
    features = ["Volume", "Area", "Sphericity", "Solidity", "Extent"]

    _, border, _ = mesh.extract_features(_cube_volume(0), features)
    _, interior, _ = mesh.extract_features(_cube_volume(4), features)

    np.testing.assert_allclose(border, interior)
    volume, area, sphericity, solidity, extent = border[0]
    assert volume > 0 and area > 0
    assert 0 < sphericity <= 1
    assert 0 < solidity <= 1.001
    assert 0 < extent <= 1


@pytest.mark.parametrize("rot_deg", [0, 30, 45])
def test_shape_features_do_not_depend_on_orientation(rot_deg: float) -> None:
    """Axis lengths and extent follow the object's axes, not the image axes.

    They used to come from the axis-aligned bounding box: turning this
    ellipsoid by 45 degrees moved the axis lengths from (12, 16, 48) to
    (12, 36, 36) and the extent from 0.52 to 0.31.
    """
    features = ["Min Axis Length", "Med Axis Length", "Max Axis Length", "Extent"]
    _, values, _ = mesh.extract_features(_rotated_ellipsoid(rot_deg), features)

    np.testing.assert_allclose(values[0, :3], [12, 16, 48], rtol=0.03)
    # An ellipsoid fills pi / 6 of the box around its axes.
    assert values[0, 3] == pytest.approx(np.pi / 6, rel=0.1)


def test_extract_features_computes_only_requested(monkeypatch) -> None:
    """Features that were not requested are never computed.

    Mean curvature took 15.4 s of a 15.8 s per-label total on a 13k-vertex
    nucleus, and a ``["Volume"]`` request used to pay for it anyway.
    """

    def fail(_mesh) -> float:
        raise AssertionError("an unrequested feature was computed")

    for name in mesh.mesh_feature_list():
        if name not in ("Volume", "Area"):
            monkeypatch.setitem(mesh._MESH_FEATURES, name, fail)

    _, values, _ = mesh.extract_features(_cube_volume(4), ["Area", "Volume"])

    cube = mesh.mask2mesh(_cube_volume(4) == 1)
    np.testing.assert_allclose(values, [[cube.area, cube.volume]])


def test_extract_surface_features_full_pipeline() -> None:
    """The full per-label pipeline returns finite, physically sensible features."""
    features = mesh.extract_surface_features(_cube_volume(0) == 1)

    assert set(features) == set(mesh.mesh_feature_list())
    assert all(np.isfinite(value) for value in features.values())
    assert features["Volume"] > 0
    assert 0 < features["Sphericity"] <= 1
    assert 0 < features["Solidity"] <= 1.001
    assert 0 < features["Extent"] <= 1


def _ball(radius: float, shape: int) -> np.ndarray:
    """Return a ``shape**3`` boolean mask of a centred ball of ``radius`` voxels."""
    grid = np.meshgrid(*[np.arange(shape) - (shape - 1) / 2] * 3, indexing="ij")
    return sum(axis**2 for axis in grid) <= radius**2


def test_curvature_of_a_sphere() -> None:
    """The curvatures are surface averages: 1 / R and 1 / R^2 on a sphere."""
    radius = 8
    features = mesh.extract_surface_features(
        _ball(radius, 22),
        features=["Avg Mean Curvature", "Avg Gaussian Curvature", "Area"],
    )

    # The voxel staircase leaves the mean curvature 1.6% low at this radius.
    assert features["Avg Mean Curvature"] == pytest.approx(1 / radius, rel=0.03)
    # Gauss-Bonnet: the Gaussian curvature integrates to 4 pi exactly.
    integral = features["Avg Gaussian Curvature"] * features["Area"]
    assert integral == pytest.approx(4 * np.pi, rel=1e-9)


def test_curvature_converts_with_the_spacing_unit() -> None:
    """Giving the same spacing in other units rescales curvature as 1 / length.

    The vertex-averaged ball measures used before integrated over a radius of 1
    in the units of ``spacing``: spacing 0.1 instead of 1 changed them 78x and
    11x rather than 100x and 10x.
    """
    names = ["Avg Mean Curvature", "Avg Gaussian Curvature"]
    mask = _ball(8, 22)
    coarse = mesh.extract_surface_features(mask, (1.0, 1.0, 1.0), names)
    fine = mesh.extract_surface_features(mask, (0.1, 0.1, 0.1), names)

    assert fine[names[0]] / coarse[names[0]] == pytest.approx(10, rel=1e-6)
    assert fine[names[1]] / coarse[names[1]] == pytest.approx(100, rel=1e-6)


def test_gaussian_curvature_follows_topology() -> None:
    """A torus has zero total Gaussian curvature, a hollow ball twice a ball's."""
    z, y, x = np.meshgrid(*[np.arange(50) - 24.5] * 3, indexing="ij")
    torus = (np.sqrt(y**2 + x**2) - 14) ** 2 + z**2 <= 6**2
    hollow = _ball(12, 32) & ~_ball(6, 32)
    names = ["Avg Gaussian Curvature", "Area"]

    torus_k, _ = mesh.extract_surface_features(torus, features=names).values()
    hollow_k, hollow_area = mesh.extract_surface_features(
        hollow, features=names
    ).values()

    assert torus_k == pytest.approx(0, abs=1e-12)
    assert hollow_k * hollow_area == pytest.approx(8 * np.pi, rel=1e-9)


def test_sphericity_is_one_for_a_sphere_only() -> None:
    """Wadell sphericity is 1 for a sphere and below 1 for a cube."""
    assert mesh.sphericity(4 / 3 * np.pi, 4 * np.pi) == pytest.approx(1)
    assert mesh.sphericity(1.0, 6.0) == pytest.approx((np.pi / 6) ** (1 / 3))
