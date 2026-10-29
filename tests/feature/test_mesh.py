"""Tests for mesh-based feature extraction."""

import sys
import subprocess

import numpy as np
import pytest
from skimage.measure import marching_cubes

trimesh = pytest.importorskip("trimesh")

import cubic.feature  # noqa: E402
from cubic.feature import mesh  # noqa: E402

_CUBE_SIDE = 5


def _cube_volume(origin: int, shape: int = 14) -> np.ndarray:
    """Return a ``shape**3`` volume holding one 5^3 cube of label 1 at ``origin``."""
    volume = np.zeros((shape,) * 3, dtype=np.int32)
    end = origin + _CUBE_SIDE
    volume[origin:end, origin:end, origin:end] = 1
    return volume


def _centred_grid(*shape: int) -> list[np.ndarray]:
    """Return ``ij`` voxel coordinates of ``shape``, centred on its middle."""
    return np.meshgrid(*[np.arange(n) - (n - 1) / 2 for n in shape], indexing="ij")


def _ball(radius: float, shape: int) -> np.ndarray:
    """Return a ``shape**3`` boolean mask of a centred ball of ``radius`` voxels."""
    return sum(axis**2 for axis in _centred_grid(shape, shape, shape)) <= radius**2


def _rotated_ellipsoid(rot_deg: float) -> np.ndarray:
    """Return a label image of an ellipsoid with full axes (12, 48, 16).

    The ellipsoid is rotated by ``rot_deg`` about axis 0, i.e. within the
    plane of axes 1 and 2.
    """
    z, y, x = _centred_grid(20, 60, 60)
    t = np.deg2rad(rot_deg)
    y, x = np.cos(t) * y + np.sin(t) * x, -np.sin(t) * y + np.cos(t) * x
    return ((z / 6) ** 2 + (y / 24) ** 2 + (x / 8) ** 2 <= 1).astype(np.int32)


@pytest.fixture
def surface_calls(monkeypatch) -> list[dict]:
    """Stub per-label meshing and record the mask and spacing of each call."""
    calls: list[dict] = []

    def fake_surface_features(mask, spacing=None, features=None) -> dict[str, float]:
        calls.append({"mask": mask, "spacing": spacing})
        return dict.fromkeys(features or mesh.mesh_feature_list(), 0.0)

    monkeypatch.setattr(mesh, "extract_surface_features", fake_surface_features)
    return calls


def test_extract_features_excludes_background_label(surface_calls) -> None:
    """Background (label 0) produces no row, matching voxel.regionprops.

    Previously ``np.unique`` kept 0, so a meaningless background mesh and a
    spurious ``label == 0`` row were emitted alongside the real objects. The
    per-label surface computation is stubbed so the test targets the label
    filtering alone.
    """
    names = mesh.mesh_feature_list()
    volume = np.zeros((12, 12, 12), dtype=np.int32)
    volume[2:6, 2:6, 2:6] = 1  # 64 voxels
    volume[7:11, 7:11, 7:11] = 2  # 64 voxels

    labels, feature_values, columns = mesh.extract_features(volume)

    assert sorted(labels.tolist()) == [1, 2]
    assert 0 not in labels.tolist()
    assert feature_values.shape == (2, len(names))
    assert columns == names
    # Background (label 0) mask sum would be 12**3 - 128 = 1600; never visited.
    assert [int(call["mask"].sum()) for call in surface_calls] == [64, 64]


def test_extract_features_meshes_only_the_label_bbox(surface_calls) -> None:
    """Each label is meshed over its own bounding box, not the whole volume."""
    volume = np.zeros((64, 64, 64), dtype=np.int32)
    volume[2:6, 2:6, 2:6] = 1
    volume[10:13, 20:24, 30:35] = 2

    mesh.extract_features(volume)

    assert [call["mask"].shape for call in surface_calls] == [(4, 4, 4), (3, 4, 5)]


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


def test_extract_features_with_no_features_meshes_nothing(surface_calls) -> None:
    """``features=[]`` gives one empty row per label without meshing any."""
    labels, feature_values, columns = mesh.extract_features(_cube_volume(4), [])

    assert labels.tolist() == [1]
    assert feature_values.shape == (1, 0)
    assert columns == []
    assert surface_calls == []


@pytest.mark.parametrize("label", [0, 1])
def test_extract_features_rejects_unknown_feature_names(label: int) -> None:
    """An unknown name raises before any label is meshed, even with no labels."""
    with pytest.raises(ValueError, match=r"Unknown mesh features \['Scale'\]"):
        mesh.extract_features(_cube_volume(4) * label, ["Volume", "Scale"])


def test_extract_features_rejects_a_bare_feature_name() -> None:
    """A single name passed as a string is rejected, not split into letters.

    It used to report ``Unknown mesh features ['V', 'o', 'l', 'u', 'm', 'e']``.
    """
    with pytest.raises(TypeError, match="not the string 'Volume'"):
        mesh.extract_features(_cube_volume(4), "Volume")


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


def test_extract_features_forwards_spacing(surface_calls) -> None:
    """``extract_features`` threads ``spacing`` down to the per-label mesh."""
    mesh.extract_features(_cube_volume(4), spacing=(4.0, 1.0, 1.0))

    assert [call["spacing"] for call in surface_calls] == [(4.0, 1.0, 1.0)]


def _open_mesh() -> trimesh.Trimesh:
    """Return the open surface of a border-touching mask meshed without padding."""
    verts, faces, _, _ = marching_cubes(_cube_volume(0) == 1)
    open_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    trimesh.repair.fix_normals(open_mesh)
    return open_mesh


def test_unknown_feature_names_are_reported_before_any_meshing(monkeypatch) -> None:
    """Every entry point names the unknown feature before meshing or validating.

    ``extract_surface_features`` used to mesh first, and ``extract_mesh_features``
    rejected an open mesh before looking at the names.
    """

    def fail(*args, **kwargs):
        raise AssertionError("meshed before checking the feature names")

    with pytest.raises(ValueError, match="Unknown mesh features"):
        mesh.extract_mesh_features(_open_mesh(), ["Scale"])
    monkeypatch.setattr(mesh, "mask2mesh", fail)
    with pytest.raises(ValueError, match="Unknown mesh features"):
        mesh.extract_surface_features(_cube_volume(4) == 1, features=["Scale"])


def test_extract_mesh_features_rejects_open_mesh() -> None:
    """A non-watertight mesh raises instead of reporting a negative volume."""
    # Reproduce the pre-padding behaviour: mesh a border-touching mask directly.
    open_mesh = _open_mesh()

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


def test_curvature_of_a_sphere() -> None:
    """Mean curvature is a surface average, 1 / R on a sphere of radius R."""
    radius = 8
    features = mesh.extract_surface_features(
        _ball(radius, 22), features=["Avg Mean Curvature"]
    )

    # The voxel staircase leaves the mean curvature 1.6% low at this radius.
    assert features["Avg Mean Curvature"] == pytest.approx(1 / radius, rel=0.03)


def test_curvature_converts_with_the_spacing_unit() -> None:
    """Giving the same spacing in other units rescales curvature as 1 / length.

    The vertex-averaged ball measure used before integrated over a radius of 1
    in the units of ``spacing``: spacing 0.1 instead of 1 changed it 11x rather
    than 10x.
    """
    names = ["Avg Mean Curvature", "Euler Characteristic"]
    mask = _ball(8, 22)
    coarse = mesh.extract_surface_features(mask, (1.0, 1.0, 1.0), names)
    fine = mesh.extract_surface_features(mask, (0.1, 0.1, 0.1), names)

    assert fine[names[0]] / coarse[names[0]] == pytest.approx(10, rel=1e-6)
    assert fine[names[1]] == coarse[names[1]] == 2


def test_cavity_counts_as_concave_in_the_mean_curvature() -> None:
    """A cavity's shell faces the material, so its curvature counts negative.

    The hollow ball integrates to the outer ball minus the inner one (82.1).
    Orienting each shell outward on its own would add them instead (244.2).
    """
    outer, inner = _ball(12, 32), _ball(6, 32)
    names = ["Avg Mean Curvature", "Area"]

    def integral(mask: np.ndarray) -> float:
        h, area = mesh.extract_surface_features(mask, features=names).values()
        return h * area

    # The two sides agree to 1.5e-7 relative, not to the last bit.
    assert integral(outer & ~inner) == pytest.approx(
        integral(outer) - integral(inner), rel=1e-6
    )


def test_euler_characteristic_counts_topology() -> None:
    """Twice the pieces plus cavities, minus twice the handles."""
    z, y, x = _centred_grid(50, 50, 50)
    torus = (np.sqrt(y**2 + x**2) - 14) ** 2 + z**2 <= 6**2
    two_balls = np.concatenate([_ball(6, 16), _ball(6, 16)])
    shapes = {
        "ball": (_ball(8, 22), 2),
        "hollow ball": (_ball(12, 32) & ~_ball(6, 32), 4),
        "two balls": (two_balls, 4),
        "torus": (torus, 0),
    }
    got = {
        name: mesh.extract_surface_features(mask, features=["Euler Characteristic"])
        for name, (mask, _) in shapes.items()
    }

    assert {name: f["Euler Characteristic"] for name, f in got.items()} == {
        name: expected for name, (_, expected) in shapes.items()
    }


def test_sphericity_is_one_for_a_sphere_only() -> None:
    """Wadell sphericity is 1 for a sphere and below 1 for a cube."""
    assert mesh.sphericity(4 / 3 * np.pi, 4 * np.pi) == pytest.approx(1)
    assert mesh.sphericity(1.0, 6.0) == pytest.approx((np.pi / 6) ** (1 / 3))


def test_mesh_is_reachable_from_cubic_feature_without_eager_import() -> None:
    """``cubic.feature.mesh`` resolves on access, not on ``import cubic.feature``.

    It used to raise ``AttributeError`` unless the submodule had been imported
    explicitly; importing it eagerly would add trimesh's 0.75-3 s import to
    every user of the voxel and texture features.
    """
    assert cubic.feature.__getattr__("mesh") is mesh
    # Only a fresh interpreter can show that the package import, star import
    # included, skips trimesh.
    code = (
        "import sys\nfrom cubic.feature import *\nassert 'trimesh' not in sys.modules\n"
    )
    # A timeout turns a hung import (e.g. a CUDA probe) into a failure.
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
