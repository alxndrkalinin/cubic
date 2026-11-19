"""Tests for mesh-based feature extraction."""

import gc
import sys
import _thread
import weakref
import warnings
import subprocess
from typing import Any, NoReturn
from collections.abc import Callable

import numpy as np
import pytest
from scipy import ndimage
from skimage.measure import marching_cubes

trimesh = pytest.importorskip("trimesh")

import cubic.feature  # noqa: E402
from cubic.feature import mesh  # noqa: E402
from cubic.feature._array_mesh import face_areas  # noqa: E402

_CUBE_SIDE = 5
_SPACING_ERROR = r"spacing must be three finite, positive numbers \(z, y, x\); got "


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


def _torus() -> np.ndarray:
    """Return a 50^3 boolean mask of a centred torus, ring radius 14 and tube radius 6."""
    z, y, x = _centred_grid(50, 50, 50)
    return (np.sqrt(y**2 + x**2) - 14) ** 2 + z**2 <= 6**2


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

    def fake_surface_features(
        mask,
        spacing=None,
        features=None,
        curvature_sigma=None,
        fractal_scales=None,
        outline_fractal_scales=None,
    ) -> dict[str, float]:
        calls.append({"mask": mask, "spacing": spacing})
        return dict.fromkeys(features, 0.0)

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
    """An unknown name raises before any label is meshed, even with no labels.

    The message lists the optional features too.
    """
    with pytest.raises(
        ValueError,
        match=r"Unknown mesh features \['Scale'\].*'Outline Fractal Dimension'",
    ):
        mesh.extract_features(_cube_volume(4) * label, ["Volume", "Scale"])


def test_outline_fractal_dimension_is_optional() -> None:
    """The optional list is the 22 default features followed by the outline.

    The defaults are the orientation-invariant features, ending in Fractal
    Dimension. Every entry point computes the outline when it is named.
    """
    default = mesh.mesh_feature_list()

    assert len(default) == 22 and default[-1] == "Fractal Dimension"
    assert mesh.mesh_feature_list(optional=True) == [
        *default,
        "Outline Fractal Dimension",
    ]

    labels = np.zeros((3, 80, 80), dtype=np.int32)
    labels[1, 2:78, 2:78] = 1
    _, values, columns = mesh.extract_features(labels, ["Outline Fractal Dimension"])
    assert columns == ["Outline Fractal Dimension"]
    assert np.isfinite(values).all()
    crop = labels[ndimage.find_objects(labels)[0]] == 1
    surface = mesh.extract_surface_features(
        crop, features=["Volume", "Outline Fractal Dimension"]
    )
    assert list(surface) == ["Volume", "Outline Fractal Dimension"]
    assert surface["Outline Fractal Dimension"] == values[0, 0]


def test_default_features_skip_the_outline(monkeypatch) -> None:
    """features=None returns the default list and never computes the outline."""

    def fail(*args, **kwargs):
        raise AssertionError("computed the outline")

    monkeypatch.setattr(mesh, "_outline_fractal_dimension", fail)
    labels = _ball(8, 22).astype(np.int32)
    names = mesh.mesh_feature_list()

    _, values, columns = mesh.extract_features(labels)
    features = mesh.extract_surface_features(labels == 1)

    assert values.shape == (1, len(names)) and columns == names
    assert list(features) == names


def test_extract_features_rejects_a_bare_feature_name() -> None:
    """A single name passed as a string is rejected, not split into letters.

    It used to report ``Unknown mesh features ['V', 'o', 'l', 'u', 'm', 'e']``.
    """
    with pytest.raises(TypeError, match="not the string 'Volume'"):
        mesh.extract_features(_cube_volume(4), "Volume")


def test_extract_features_accepts_a_one_shot_iterable_of_names() -> None:
    """A generator of names gives the same columns as the list of them."""
    names = ["Volume", "Area"]
    _, from_list, _ = mesh.extract_features(_cube_volume(4), names)
    _, from_generator, columns = mesh.extract_features(
        _cube_volume(4), (name for name in names)
    )

    assert columns == names
    np.testing.assert_array_equal(from_generator, from_list)


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


def test_extract_features_orients_a_ragged_label_outward(
    ragged_ball: np.ndarray,
) -> None:
    """A label whose mesh is not watertight is measured, not rejected.

    ``fix_normals`` left such a mesh with marching cubes' inward winding, and
    the volume check raised ``Mesh volume must be finite and positive; got
    -14291.58`` for the whole label image.
    """
    assert not mesh.mask2mesh(ragged_ball).is_watertight
    image = np.zeros((41, 41, 90), dtype=np.int32)
    image[:, :, :41][ragged_ball] = 1
    image[10:20, 10:20, 60:70] = 2  # a clean cube next to it

    labels, values, _ = mesh.extract_features(image, ["Volume"])

    np.testing.assert_array_equal(labels, [1, 2])
    assert values.shape == (2, 1)
    # Close to the voxel counts, 14224 and 1000 (14291.6 and 985.7: marching
    # cubes cuts the cube's edges and corners).
    voxels = [ragged_ball.sum(), 1000]
    np.testing.assert_allclose(values[:, 0], voxels, rtol=0.02)


def test_mask2mesh_orients_random_masks_outward() -> None:
    """Random masks, whose meshes are mostly not watertight, all mesh outward.

    With ``fix_normals``, the 44 of these 50 meshes that are not watertight
    came out inside out.
    """
    rng = np.random.default_rng(0)
    surfaces = [mesh.mask2mesh(rng.random((8, 8, 8)) < 0.35) for _ in range(50)]

    assert sum(not s.is_watertight for s in surfaces) > 25
    assert all(s.volume > 0 for s in surfaces)


def test_mask2mesh_orients_an_ascent_mesh_outward(ragged_ball: np.ndarray) -> None:
    """With ``gradient_direction="ascent"`` the faces already wind outward.

    They must then be kept, so the volume equals the default direction's.
    """
    ascent = mesh.mask2mesh(
        ragged_ball, marching_cubes_kwargs={"gradient_direction": "ascent"}
    )

    assert ascent.volume > 0
    assert ascent.volume == pytest.approx(mesh.mask2mesh(ragged_ball).volume, rel=1e-12)


@pytest.mark.parametrize("spacing", [None, (4.0, 1.0, 1.0), (4e-9, 1e-9, 1e-9)])
def test_mask2mesh_vertices_stay_in_the_input_frame(
    spacing: tuple[float, float, float] | None,
) -> None:
    """The 1-voxel pad must not shift the mesh out of the image frame.

    ``mask2mesh`` pads to close border-touching surfaces, so the pad has to be
    subtracted again. Left in, every vertex sat one voxel high on each axis —
    scaled by ``spacing``, so a z-spacing of 4 displaced the mesh by 4.0
    physical units, and ``bounds``/``centroid`` no longer overlaid the image.
    The nanometre spacing is built in units of its finest value and scaled
    back; built in metres, trimesh merged vertices closer than 1e-8.
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


@pytest.mark.parametrize(
    "spacing",
    [(-1.0, -1.0, -1.0), (0.0, 1.0, 1.0)]
    + [0.5, (1.0,), (1.0, 1.0), (1.0, 1.0, 1.0, 1.0)],
)
@pytest.mark.parametrize("source", ["spacing", "marching_cubes_kwargs", "sigma"])
def test_mask2mesh_rejects_a_spacing_that_is_not_three_positive_numbers(
    source: str, spacing: float | tuple[float, ...]
) -> None:
    """mask2mesh rejects such a spacing from either source, as extract_features does.

    An all-negative spacing used to return an inside-out mesh and a zero one a
    flat mesh; with ``sigma`` it failed deep in the blur.
    """
    kwargs = {
        "spacing": {"spacing": spacing},
        "marching_cubes_kwargs": {"marching_cubes_kwargs": {"spacing": spacing}},
        "sigma": {"spacing": spacing, "sigma": 2.0},
    }[source]

    with pytest.raises(ValueError, match=_SPACING_ERROR):
        mesh.mask2mesh(_ball(8, 22), **kwargs)


def test_extract_features_forwards_spacing(surface_calls) -> None:
    """``extract_features`` threads ``spacing`` down to the per-label mesh."""
    mesh.extract_features(_cube_volume(4), spacing=(4.0, 1.0, 1.0))

    assert [call["spacing"] for call in surface_calls] == [(4.0, 1.0, 1.0)]


def _inside_out_mesh() -> trimesh.Trimesh:
    """Return the closed mesh of a cube with every face wound inward."""
    return mesh.mask2mesh(_cube_volume(4) == 1).invert()


def test_unknown_feature_names_are_reported_before_any_meshing(monkeypatch) -> None:
    """Every entry point names the unknown feature before meshing or validating.

    ``extract_surface_features`` used to mesh first, and ``extract_mesh_features``
    rejected a mesh that fails its volume check before looking at the names.
    """

    def fail(*args, **kwargs):
        raise AssertionError("meshed before checking the feature names")

    with pytest.raises(ValueError, match="volume must be finite and positive"):
        mesh.extract_mesh_features(_inside_out_mesh())
    with pytest.raises(ValueError, match="Unknown mesh features"):
        mesh.extract_mesh_features(_inside_out_mesh(), ["Scale"])
    monkeypatch.setattr(mesh, "mask2mesh", fail)
    with pytest.raises(ValueError, match="Unknown mesh features"):
        mesh.extract_surface_features(_cube_volume(4) == 1, features=["Scale"])


def test_extract_mesh_features_rejects_an_inside_out_mesh() -> None:
    """A closed mesh wound inward raises instead of reporting a negative volume."""
    inside_out = _inside_out_mesh()

    assert inside_out.is_watertight
    assert inside_out.volume < 0
    with pytest.raises(ValueError, match="positive; got -.*inside out"):
        mesh.extract_mesh_features(inside_out)


def test_edge_unit_does_not_depend_on_how_the_mesh_is_listed(
    noise_blob_of: Callable[[int], np.ndarray],
) -> None:
    """A mesh is measured at the same unit however its vertices and faces are listed.

    The vertices are permuted, the faces permuted, and each face rotated to
    start at its next vertex, which keeps its orientation. The rotation changes
    the operand order of the cross product, so some face areas change by an
    ulp, which the test asserts; for this mesh the square root of the exactly
    summed mean face area then changes too. The unit must be bitwise equal.
    """
    surface = mesh.mask2mesh(noise_blob_of(2), spacing=(0.29, 0.108, 0.108))
    vertices, faces = np.asarray(surface.vertices), np.asarray(surface.faces)
    perm = np.random.default_rng(0).permutation(len(vertices))
    new_index = np.empty_like(perm)
    new_index[perm] = np.arange(len(perm))
    order = np.random.default_rng(1).permutation(len(faces))
    rotated = np.roll(new_index[faces], -1, axis=1)
    relisted = trimesh.Trimesh(vertices[perm], rotated[order], process=False)
    areas = face_areas(vertices, faces)
    relisted_areas = face_areas(vertices[perm], rotated)  # in the order of ``faces``

    assert (relisted_areas != areas).any()
    assert mesh._edge_unit(relisted) == mesh._edge_unit(surface)


def test_edge_unit_ignores_vertices_no_face_uses() -> None:
    """A vertex that no face uses does not change the unit a mesh is measured at."""
    box = trimesh.creation.box()
    stray = np.vstack([box.vertices, [1e6, 1e6, 1e6]])
    with_stray = trimesh.Trimesh(stray, box.faces, process=False)

    assert mesh._edge_unit(with_stray) == mesh._edge_unit(box)


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

    A ``["Volume"]`` request used to compute every feature, mean curvature
    included. The curvature averages are computed apart from the registry, from
    per-vertex curvatures shared between them, so those fail too, and with one
    average requested the other fails.
    """

    def fail(*_args: Any) -> float:
        raise AssertionError("an unrequested feature was computed")

    def fail_unrequested(patch: pytest.MonkeyPatch, requested: list[str]) -> None:
        for name in mesh.mesh_feature_list():
            if name not in requested:
                patch.setitem(mesh._MESH_FEATURES, name, fail)
        for name in mesh._CURVATURE_AVERAGES:
            if name not in requested:
                patch.setitem(mesh._CURVATURE_AVERAGES, name, fail)

    with monkeypatch.context() as patch:
        fail_unrequested(patch, ["Area", "Volume"])
        patch.setattr(mesh, "_principal_curvatures", fail)
        _, values, _ = mesh.extract_features(_cube_volume(4), ["Area", "Volume"])

    cube = mesh.mask2mesh(_cube_volume(4) == 1)
    np.testing.assert_allclose(values, [[cube.area, cube.volume]])

    # A ball, which survives the curvature blur that the cube does not.
    for name in mesh._CURVATURE_AVERAGES:
        with monkeypatch.context() as patch:
            fail_unrequested(patch, [name])
            _, values, _ = mesh.extract_features(_ball(8, 22).astype(np.int32), [name])
        assert np.isfinite(values).all(), name


def test_extract_surface_features_full_pipeline() -> None:
    """The full per-label pipeline returns finite, physically sensible features.

    The cube touches the volume border and is large enough to survive the
    default curvature blur.
    """
    cube = np.zeros((14, 14, 14), dtype=bool)
    cube[:10, :10, :10] = True
    features = mesh.extract_surface_features(cube)

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
    than 10x. The default blur follows the spacing, so the unitless curvature
    features stay put and curvedness scales as 1 / length like the mean.
    """
    names = ["Avg Mean Curvature", "Euler Characteristic", *_CURVATURE_FEATURES]
    mask = _ball(8, 22)
    coarse = mesh.extract_surface_features(mask, (1.0, 1.0, 1.0), names)
    fine = mesh.extract_surface_features(mask, (0.1, 0.1, 0.1), names)

    assert fine[names[0]] / coarse[names[0]] == pytest.approx(10, rel=1e-6)
    assert fine[names[1]] == coarse[names[1]] == 2
    for name in ("Total Absolute Gaussian Curvature", "Avg Shape Index"):
        assert fine[name] == pytest.approx(coarse[name], rel=1e-6)
    assert fine["Avg Curvedness"] == pytest.approx(10 * coarse["Avg Curvedness"])


def test_default_curvature_blur_is_twice_the_coarsest_spacing() -> None:
    """The default blur is 2 * max(spacing), not twice the finest or the mean.

    Only an anisotropic stack tells the three apart: here 8, 2 and 4.
    """
    z, y, x = _centred_grid(16, 64, 64)
    mask = (z / 6) ** 2 + (y / 24) ** 2 + (x / 16) ** 2 <= 1
    spacing = (4.0, 1.0, 1.0)

    default = mesh.extract_surface_features(mask, spacing, _CURVATURE_FEATURES)
    coarsest = mesh.extract_surface_features(
        mask, spacing, _CURVATURE_FEATURES, curvature_sigma=8.0
    )
    finest = mesh.extract_surface_features(
        mask, spacing, _CURVATURE_FEATURES, curvature_sigma=2.0
    )

    assert all(np.isfinite(value) for value in default.values())
    assert default == coarsest
    assert default["Avg Curvedness"] != finest["Avg Curvedness"]


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
    two_balls = np.concatenate([_ball(6, 16), _ball(6, 16)])
    shapes = {
        "ball": (_ball(8, 22), 2),
        "hollow ball": (_ball(12, 32) & ~_ball(6, 32), 4),
        "two balls": (two_balls, 4),
        "torus": (_torus(), 0),
    }
    got = {
        name: mesh.extract_surface_features(mask, features=["Euler Characteristic"])
        for name, (mask, _) in shapes.items()
    }

    assert {name: f["Euler Characteristic"] for name, f in got.items()} == {
        name: expected for name, (_, expected) in shapes.items()
    }


_CURVATURE_FEATURES = [
    "Total Absolute Gaussian Curvature",
    "Avg Shape Index",
    "Avg Curvedness",
]


def test_curvature_distribution_of_a_ball() -> None:
    """A convex ball: total |K| / 4 pi of 1, shape index near 1, curvedness 1 / R."""
    features = mesh.extract_surface_features(
        _ball(10, 26), features=_CURVATURE_FEATURES
    )

    # Every angle defect of a convex surface is positive, so |K| integrates to
    # exactly the 4 pi of Gauss-Bonnet.
    assert features["Total Absolute Gaussian Curvature"] == pytest.approx(1, rel=1e-6)
    # Per-vertex principal curvatures are noisy, which pulls the area-average
    # below 1 (0.92 here).
    assert features["Avg Shape Index"] > 0.85
    # The blur shrinks a convex body by about sigma^2 / R, raising curvedness.
    assert features["Avg Curvedness"] == pytest.approx(1 / 10, rel=0.15)


def test_total_absolute_gaussian_curvature_grows_with_saddles() -> None:
    """Saddle regions push total |K| / 4 pi above the convex value of 1.

    A torus has as much negative as positive curvature, so 2; two overlapping
    balls have a saddle-shaped waist where they join.
    """
    zz, yy, xx = _centred_grid(24, 24, 36)
    peanut = (zz**2 + yy**2 + (xx - 6) ** 2 <= 8**2) | (
        zz**2 + yy**2 + (xx + 6) ** 2 <= 8**2
    )
    name = "Total Absolute Gaussian Curvature"

    torus_k = mesh.extract_surface_features(_torus(), features=[name])[name]
    peanut_k = mesh.extract_surface_features(peanut, features=[name])[name]

    assert torus_k == pytest.approx(2, rel=0.05)
    assert peanut_k > 1.1


def test_shape_index_tells_a_torus_from_a_ball() -> None:
    """Saddles lower the area-averaged shape index: a torus reads well below a ball.

    The ball is all cap (0.92 here); the inner half of the torus is saddle (0.50 here).
    """
    name = "Avg Shape Index"
    ball = mesh.extract_surface_features(_ball(10, 26), features=[name])[name]
    torus = mesh.extract_surface_features(_torus(), features=[name])[name]

    assert torus < ball - 0.2


def test_flat_residues_leave_real_curvature_as_it_is() -> None:
    """On a blurred flat ellipsoid only rounding residues are set to 0.

    The terraces that the blur leaves on its shallow slopes have angle defects
    down to 4.5e-13 radians and edge sums down to 1.3e-13 of their terms, and
    no angle defect lies between 1e-14 and 1e-13. What the threshold
    sets to 0 on the copy that ``extract_mesh_features`` measures are residues
    of at most four ulps: at the unit of ``_edge_unit``, 260 angle defects of
    8.9e-16 (one ulp of 2 pi), which moves Avg Shape Index by 3.6e-9.
    """
    z, y, x = _centred_grid(20, 132, 132)
    flat = (z / 6) ** 2 + (y / 60) ** 2 + (x / 60) ** 2 <= 1
    smooth = mesh.mask2mesh(flat, sigma=2.0)
    measured = mesh._scaled_copy(smooth, 1 / mesh._edge_unit(smooth))
    defects = np.abs(measured.vertex_defects)
    edges = measured.face_adjacency_edges
    lengths = np.linalg.norm(
        measured.vertices[edges[:, 1]] - measured.vertices[edges[:, 0]], axis=1
    )
    terms = np.repeat(lengths * measured.face_adjacency_angles, 2)
    terms *= np.repeat(np.where(measured.face_adjacency_convex, 1.0, -1.0), 2)
    n = len(measured.vertices)
    edge_sum = np.abs(np.bincount(edges.ravel(), weights=terms, minlength=n))
    term_sum = np.bincount(edges.ravel(), weights=np.abs(terms), minlength=n)
    ratio = edge_sum / np.where(term_sum > 0, term_sum, np.inf)

    zeroed_defects = defects[(defects > 0) & (defects <= mesh._FLAT_RESIDUE)]
    zeroed_ratios = ratio[(edge_sum > 0) & (edge_sum <= mesh._FLAT_RESIDUE * term_sum)]
    assert (zeroed_defects <= 4 * np.spacing(2 * np.pi)).all(), zeroed_defects.max()
    assert (zeroed_ratios <= 4 * np.finfo(float).eps).all(), zeroed_ratios.max()
    assert not ((defects > 1e-14) & (defects <= 1e-13)).any()


@pytest.mark.parametrize("rot_deg", [30, 45])
def test_curvature_distribution_does_not_depend_on_orientation(rot_deg: float) -> None:
    """Rotating an ellipsoid barely moves the curvature features.

    Shape index and curvedness agree within 1%. Total |K| of this thin convex
    ellipsoid (minor semi-axis 6 voxels) reads 1.02-1.05 rather than 1: the
    default blur leaves faint terraces on its shallow slopes, and their saddles
    depend on how the slopes meet the voxel grid.
    """
    upright = mesh.extract_surface_features(
        _rotated_ellipsoid(0) == 1, features=_CURVATURE_FEATURES
    )
    turned = mesh.extract_surface_features(
        _rotated_ellipsoid(rot_deg) == 1, features=_CURVATURE_FEATURES
    )

    for name in ("Avg Shape Index", "Avg Curvedness"):
        assert turned[name] == pytest.approx(upright[name], rel=0.01)
    for features in (upright, turned):
        assert 1 <= features["Total Absolute Gaussian Curvature"] < 1.06


_NUCLEUS_SPACING = (0.3, 0.13, 0.13)


def _rough_ellipsoid(amplitude: float, axes=(2.5, 5.0, 6.0)) -> np.ndarray:
    """Return an ellipsoid with semi-axes ``axes`` (um) at ``_NUCLEUS_SPACING``.

    Its radius is perturbed by band-limited noise (0.4 um wavelength) of
    relative standard deviation ``amplitude``; 0 gives the smooth ellipsoid.
    """
    shape = [
        2 * int(np.ceil((1.3 * a + 1) / s)) + 1 for a, s in zip(axes, _NUCLEUS_SPACING)
    ]
    grid = _centred_grid(*shape)
    radius = np.sqrt(
        sum((g * s / a) ** 2 for g, s, a in zip(grid, _NUCLEUS_SPACING, axes))
    )
    if amplitude:
        noise = ndimage.gaussian_filter(
            np.random.default_rng(0).standard_normal(radius.shape),
            [0.4 / s for s in _NUCLEUS_SPACING],
        )
        radius = radius + amplitude * noise / noise.std()
    return radius < 1


def _fractal_dimension(mask: np.ndarray, fractal_scales=None) -> float:
    name = "Fractal Dimension"
    return mesh.extract_surface_features(
        mask, spacing=_NUCLEUS_SPACING, features=[name], fractal_scales=fractal_scales
    )[name]


def test_fractal_dimension_of_a_smooth_surface_is_two() -> None:
    """A surface without detail finer than curvature_sigma has D = 2, whatever its size.

    Its area ratio to the smooth twin stays constant across blur scales. The
    smallest curvature radius of these ellipsoids, c^2 / a, is 1.0-1.6 um, above
    curvature_sigma (0.6 um). Measuring the area of the blurred mask alone gives
    2.05-2.20 for them, because blur shrinks small, curved surfaces most. A flat
    (1.5, 6, 8) um ellipsoid, whose rim radius is 0.28 um, measures 2.01: its rim
    is detail finer than curvature_sigma.
    """
    for axes in [(2.5, 5.0, 6.0), (3.0, 6.0, 8.0), (4.0, 8.0, 10.0)]:
        assert _fractal_dimension(_rough_ellipsoid(0, axes)) == pytest.approx(
            2, abs=0.005
        )


def test_fractal_dimension_rises_with_surface_roughness() -> None:
    """Roughness finer than curvature_sigma raises D, monotonically."""
    smooth, rough, rougher = (
        _fractal_dimension(_rough_ellipsoid(amplitude)) for amplitude in (0, 0.04, 0.08)
    )

    assert smooth + 0.01 < rough < rougher
    assert rougher > 2.05


def test_default_fractal_window_is_the_finest_spacing_to_curvature_sigma() -> None:
    """Without fractal_scales the fit spans the finest spacing to curvature_sigma.

    The explicit window blurs the mask at its top scale itself instead of
    reusing the curvature blur, and gets the same value bit for bit.
    """
    mask = _rough_ellipsoid(0.04)
    sigma = mesh._curvature_sigma(_NUCLEUS_SPACING, None)

    default = _fractal_dimension(mask)
    explicit = _fractal_dimension(mask, (min(_NUCLEUS_SPACING), sigma))

    assert np.isfinite(default)
    assert explicit == default


def test_fractal_window_changes_only_fractal_dimension() -> None:
    """A custom window moves Fractal Dimension and leaves the curvature features."""
    mask = _rough_ellipsoid(0.04)
    names = ["Fractal Dimension", *_CURVATURE_FEATURES]
    default = mesh.extract_surface_features(mask, _NUCLEUS_SPACING, names)
    custom = mesh.extract_surface_features(
        mask, _NUCLEUS_SPACING, names, fractal_scales=(0.3, 1.2)
    )

    assert np.isfinite(custom["Fractal Dimension"])
    assert abs(custom["Fractal Dimension"] - default["Fractal Dimension"]) > 1e-3
    for name in _CURVATURE_FEATURES:
        assert custom[name] == default[name]


def test_fractal_window_does_not_need_curvature_sigma() -> None:
    """With its own window, D no longer needs curvature_sigma above the spacing.

    The default window of this call would be empty and give NaN.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        features = mesh.extract_surface_features(
            _ball(10, 26),
            features=["Fractal Dimension"],
            curvature_sigma=1.0,
            fractal_scales=(1.0, 3.0),
        )

    assert not [w for w in caught if "Fractal Dimension" in str(w.message)]
    # 1.96: a top scale of 0.3 radii already biases a smooth ball low.
    assert features["Fractal Dimension"] == pytest.approx(2, abs=0.05)


@pytest.mark.parametrize(
    ("scales", "error"),
    [
        ((1.0,), ValueError),
        ((1.0, 2.0, 3.0), ValueError),
        ((2.0, 1.0), ValueError),
        ((1.0, 1.0), ValueError),
        ((0.0, 1.0), ValueError),
        ((-1.0, 1.0), ValueError),
        ((1.0, float("inf")), ValueError),
        ((float("nan"), 1.0), ValueError),
        ((1.0, float("nan")), ValueError),
        ((float("-inf"), 1.0), ValueError),
        (("a", "b"), TypeError),
        ("12", TypeError),
        (b"12", TypeError),
        ({1.0: None, 2.0: None}, TypeError),
        (2.0, TypeError),
    ],
)
@pytest.mark.parametrize("keyword", ["fractal_scales", "outline_fractal_scales"])
def test_fractal_window_is_validated(scales, error: type, keyword: str) -> None:
    """A window that is not two finite numbers 0 < lowest < highest raises.

    TypeError when it is not a sequence of numbers, ValueError when the values
    or their count are wrong. The error names the keyword. It raises before
    any meshing, so also on a label image without labels.
    """
    match = f"^{keyword} must"
    with pytest.raises(error, match=match):
        mesh.extract_surface_features(
            _ball(8, 22), features=["Fractal Dimension"], **{keyword: scales}
        )
    with pytest.raises(error, match=match):
        mesh.extract_features(np.zeros((4, 4, 4), np.int32), **{keyword: scales})


def test_each_fractal_window_moves_only_its_own_feature() -> None:
    """fractal_scales leaves the outline bitwise as is, and the other way round."""
    mask = _rough_cylinder(0.04)
    names = ["Fractal Dimension", _OUTLINE]

    def measure(**windows) -> dict[str, float]:
        return mesh.extract_surface_features(mask, _FLAT_SPACING, names, **windows)

    default = measure()
    by_3d = measure(fractal_scales=(0.15, 0.4))
    by_outline = measure(outline_fractal_scales=(0.3, 0.9))

    assert by_3d[_OUTLINE] == default[_OUTLINE]
    assert by_3d["Fractal Dimension"] != default["Fractal Dimension"]
    assert by_outline["Fractal Dimension"] == default["Fractal Dimension"]
    assert by_outline[_OUTLINE] != default[_OUTLINE]
    assert all(np.isfinite(v) for f in (by_3d, by_outline) for v in f.values())


def test_extract_features_forwards_the_fractal_window() -> None:
    """extract_features passes fractal_scales to every label's crop."""
    name = ["Fractal Dimension"]
    labels = _ball(10, 26).astype(np.int32)
    crop = labels[ndimage.find_objects(labels)[0]] == 1

    _, values, _ = mesh.extract_features(labels, name, fractal_scales=(1.0, 3.0))
    expected = mesh.extract_surface_features(
        crop, features=name, fractal_scales=(1.0, 3.0)
    )
    default = mesh.extract_surface_features(crop, features=name)

    assert values[0, 0] == expected[name[0]] != default[name[0]]


def _slab(planes: int, shape=(12, 60, 60)) -> np.ndarray:
    """Return a square slab ``planes`` voxels thick, 40 voxels wide."""
    slab = np.zeros(shape, dtype=bool)
    z0 = (shape[0] - planes) // 2
    slab[z0 : z0 + planes, 10:50, 10:50] = True
    return slab


def _thin_warnings(caught) -> list:
    return [w for w in caught if "biased low for objects thinner" in str(w.message)]


def test_fractal_dimension_warns_for_an_object_thinner_than_four_top_scales() -> None:
    """A slab under 4 x curvature_sigma thick warns once and keeps its value.

    The 6-plane slab is 1.8 um thick, its Min Axis Length 2.3 um, below 4 x 0.6
    um; the ellipsoid's is 5 um. A window topping out at 0.5 um clears the slab.
    The warning points to Outline Fractal Dimension.
    """
    names = ["Fractal Dimension", "Min Axis Length"]
    results = {}
    for name, mask, scales in [
        ("slab", _slab(6), None),
        ("ball", _rough_ellipsoid(0), None),
        ("slab, low window", _slab(6), (0.13, 0.5)),
    ]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            features = mesh.extract_surface_features(
                mask, _NUCLEUS_SPACING, names, fractal_scales=scales
            )
        results[name] = (_thin_warnings(caught), features)

    thin, slab = results["slab"]
    assert len(thin) == 1
    assert "see Outline Fractal Dimension" in str(thin[0].message)
    assert slab["Min Axis Length"] < 4 * 0.6
    assert np.isfinite(slab["Fractal Dimension"])
    for name in ("ball", "slab, low window"):
        thin, features = results[name]
        assert thin == []
        assert np.isfinite(features["Fractal Dimension"])


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("thin slab", "biased low for objects thinner"),
        ("outline narrower than the blur", "Outline Fractal Dimension: the mask"),
        ("3D twin vanishes", "Fractal Dimension: the mask or its smooth twin"),
        ("mask vanishes at curvature_sigma", "vanishes when blurred at curvature"),
        ("empty default window", "needs curvature_sigma above"),
    ],
)
@pytest.mark.parametrize("entry", ["extract_surface_features", "extract_features"])
def test_warnings_point_at_the_caller(case: str, match: str, entry: str) -> None:
    """Warnings are attributed to the calling line, not to cubic, from either entry.

    The _require_trimesh wrapper adds a frame; with stacklevel=2 every warning
    landed on that wrapper in mesh.py, and through extract_features a fixed
    stacklevel landed on its per-label loop. The outline and 3D-twin cases are
    raised inside the shared blur-ladder estimator.
    """
    thin_sheet = np.zeros((9, 60, 60), dtype=bool)
    thin_sheet[3:6, 10:50, 10:50] = True
    calls = {
        "thin slab": (_slab(6), _NUCLEUS_SPACING, ["Fractal Dimension"], {}),
        "outline narrower than the blur": (_ball(8, 22), None, [_OUTLINE], {}),
        "3D twin vanishes": (thin_sheet, _NUCLEUS_SPACING, ["Fractal Dimension"], {}),
        "mask vanishes at curvature_sigma": (
            np.ones((1, 1, 1), dtype=bool),
            None,
            ["Avg Curvedness"],
            {},
        ),
        "empty default window": (
            _ball(8, 22),
            None,
            ["Fractal Dimension"],
            {"curvature_sigma": 1.0},
        ),
    }
    mask, spacing, names, kwargs = calls[case]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        if entry == "extract_features":
            mesh.extract_features(mask.astype(np.int32), names, spacing, **kwargs)
        else:
            mesh.extract_surface_features(mask, spacing, names, **kwargs)

    matching = [w for w in caught if match in str(w.message)]
    assert len(matching) == 1
    assert matching[0].filename == __file__


@pytest.mark.parametrize(
    ("module", "skipped"), [("cubic.fake_caller", True), ("cubical", False)]
)
def test_warnings_skip_every_cubic_module(module: str, skipped: bool) -> None:
    """A caller in another cubic module is skipped too, up to the user's line.

    Only frames of mesh.py used to be skipped, so such a caller was named. A
    module whose name merely starts with "cubic" is the user's own.
    """
    filename = f"<{module}>"
    source = (
        "def call(mask):\n"
        "    return mesh.extract_surface_features(mask, None, ['Avg Curvedness'])\n"
    )
    namespace = {"__name__": module, "mesh": mesh}
    exec(compile(source, filename, "exec"), namespace)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        namespace["call"](np.ones((1, 1, 1), dtype=bool))

    matching = [w for w in caught if "vanishes when blurred" in str(w.message)]
    assert [w.filename for w in matching] == [__file__ if skipped else filename]


def test_warnings_name_the_outermost_frame_when_every_frame_is_cubic() -> None:
    """With only cubic frames on the stack, the warning names the outermost.

    One level past it ``warnings`` attributes the warning to "sys". A thread
    started with ``_thread.start_new_thread`` has no frame below its target, so
    the code compiled here under a cubic module name is the whole stack.
    """
    filename = "<cubic.fake_entry>"
    source = (
        "def warn_here():\n"
        "    level = mesh._stacklevel_outside_package()\n"
        "    warnings.warn('probe', stacklevel=level)\n"
        "    return level\n"
        "def entry(result, done):\n"
        "    try:\n"
        "        with warnings.catch_warnings(record=True) as caught:\n"
        "            warnings.simplefilter('always')\n"
        "            result.append(warn_here())\n"
        "        result.append(caught[0].filename)\n"
        "    finally:\n"
        "        done.release()\n"
    )
    namespace = {"__name__": "cubic.fake_entry", "mesh": mesh, "warnings": warnings}
    exec(compile(source, filename, "exec"), namespace)
    result: list[Any] = []
    done = _thread.allocate_lock()
    done.acquire()

    _thread.start_new_thread(namespace["entry"], (result, done))

    assert done.acquire(timeout=60)
    # warn_here is level 1 and entry, the outermost frame, level 2.
    assert result == [2, filename]


def _failing_bounding_cylinder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make every trimesh bounding cylinder fail as a degenerate mesh's does."""

    def fail(self: trimesh.Trimesh) -> NoReturn:
        raise ValueError("degenerate")

    monkeypatch.setattr(trimesh.Trimesh, "bounding_cylinder", property(fail))


@pytest.mark.parametrize(
    "entry", ["extract_features", "extract_surface_features", "extract_mesh_features"]
)
def test_bounding_cylinder_warning_points_at_the_caller(
    entry: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bounding-cylinder fallback warns at the calling line and returns 0.

    With stacklevel=2 it pointed at the feature registry in mesh.py.
    """
    _failing_bounding_cylinder(monkeypatch)
    mask = _ball(6, 16)
    names = ["Bounding Cylinder Volume"]
    calls = {
        "extract_features": lambda: mesh.extract_features(mask.astype(np.int32), names)[
            1
        ][0, 0],
        "extract_surface_features": lambda: mesh.extract_surface_features(
            mask, features=names
        )[names[0]],
        "extract_mesh_features": lambda: mesh.extract_mesh_features(
            mesh.mask2mesh(mask), names
        )[names[0]],
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        value = calls[entry]()

    matching = [w for w in caught if "bounding cylinder" in str(w.message)]
    assert value == 0.0
    assert [w.filename for w in matching] == [__file__]


def test_thin_object_warning_is_shown_once_for_a_label_image() -> None:
    """Many thin labels give one warning under Python's default filter.

    The filter keys a text by the calling line, so the warning is shown once
    per calling line: once for this label image. The slabs differ in thickness
    and width, so a text naming each object's size would be shown once per
    label.
    """
    labels = np.zeros((12, 60, 180), dtype=np.int32)
    for i, (planes, width) in enumerate([(6, 40), (5, 34), (6, 28)]):
        labels[3 : 3 + planes, 10 : 10 + width, 60 * i : 60 * i + width] = i + 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("default")
        _, values, _ = mesh.extract_features(
            labels, ["Fractal Dimension"], spacing=_NUCLEUS_SPACING
        )

    assert len(_thin_warnings(caught)) == 1
    assert np.isfinite(values).all()


def test_thin_object_check_runs_only_for_fractal_dimension(monkeypatch) -> None:
    """Without Fractal Dimension the ellipsoid axes are never computed."""

    def fail(*args, **kwargs):
        raise AssertionError("computed the ellipsoid axes")

    monkeypatch.setattr(mesh, "_equivalent_ellipsoid_axes", fail)
    features = mesh.extract_surface_features(
        _slab(6), _NUCLEUS_SPACING, ["Volume", "Avg Curvedness"]
    )

    assert features["Volume"] > 0


_SHARED_EDGE_TEXT = "edges shared by more than two faces"


def _shared_edge_warnings(
    call: Callable[[], object], simplefilter: str = "always"
) -> list:
    """Run ``call`` and return the shared-edge warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(simplefilter)
        call()
    return [w for w in caught if _SHARED_EDGE_TEXT in str(w.message)]


@pytest.mark.parametrize(
    "names",
    [
        ["Euler Characteristic"],
        ["Avg Mean Curvature"],
        *([name] for name in _CURVATURE_FEATURES),
    ],
)
def test_shared_edge_warning_fires_on_a_ragged_ball_only(
    names: list[str], ragged_ball_of: Callable[[float], np.ndarray]
) -> None:
    """Each approximate feature warns on the ragged ball's mesh, not a clean ball's.

    The ragged ball's mesh has 33 edges shared by four faces. Measured on the
    raw mesh, with no smooth mesh, so the three curvature-distribution features
    see those edges too. The values are returned all the same.
    """
    for fraction, expected in [(0.3, 1), (0.0, 0)]:
        surface = mesh.mask2mesh(ragged_ball_of(fraction))
        values = {}

        def call(
            surface: trimesh.Trimesh = surface, values: dict[str, float] = values
        ) -> None:
            values.update(mesh.extract_mesh_features(surface, names))

        assert len(_shared_edge_warnings(call)) == expected, fraction
        assert np.isfinite(values[names[0]])


def test_shared_edge_warning_comes_from_the_raw_mesh_features(
    ragged_ball: np.ndarray,
) -> None:
    """The default pipeline warns for Euler and mean curvature, not the blurred mesh.

    The blurred mesh behind the curvature-distribution features has no edge on
    more than two faces, so requesting only those does not warn, nor do
    features that the shared edges do not affect.
    """
    assert mesh._overshared_edge_count(mesh.mask2mesh(ragged_ball)) == 33
    assert mesh._overshared_edge_count(mesh.mask2mesh(ragged_ball, sigma=2.0)) == 0
    assert mesh.mask2mesh(ragged_ball).euler_number == -477  # odd: not a surface

    for names, expected in [
        (None, 1),
        (["Euler Characteristic", "Avg Mean Curvature"], 1),
        (_CURVATURE_FEATURES, 0),
        (["Volume", "Area"], 0),
    ]:
        caught = _shared_edge_warnings(
            lambda names=names: mesh.extract_surface_features(
                ragged_ball, features=names
            )
        )
        assert len(caught) == expected, names


def test_shared_edge_warning_follows_the_mesh_each_feature_is_measured_on(
    ragged_ball_of: Callable[[float], np.ndarray],
) -> None:
    """The curvature-distribution features check the smooth mesh; the others ``mesh``.

    The ragged mesh is passed as either one, with a clean mesh as the other.
    """
    ragged = mesh.mask2mesh(ragged_ball_of(0.3))
    clean = mesh.mask2mesh(ragged_ball_of(0.0))

    for surface, smooth, names, expected in [
        (clean, ragged, ["Avg Curvedness"], 1),
        (ragged, clean, ["Avg Curvedness"], 0),
        (ragged, clean, ["Euler Characteristic", "Avg Curvedness"], 1),
        (clean, ragged, ["Euler Characteristic", "Volume"], 0),
    ]:
        caught = _shared_edge_warnings(
            lambda s=surface, m=smooth, n=names: mesh.extract_mesh_features(
                s, n, smooth_mesh=m
            )
        )
        assert len(caught) == expected, (names, surface is ragged)


def test_shared_edges_are_counted_only_for_an_approximate_feature(
    ragged_ball: np.ndarray, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without an approximate feature the shared edges are never counted."""

    def fail(*args: Any, **kwargs: Any) -> NoReturn:
        raise AssertionError("counted the shared edges")

    monkeypatch.setattr(mesh, "_overshared_edge_count", fail)
    names = [
        name
        for name in mesh.mesh_feature_list()
        if name not in mesh._SHARED_EDGE_FEATURES
    ]
    features = mesh.extract_surface_features(ragged_ball, features=names)

    assert list(features) == names


def test_shared_edge_count_tested_by_the_face_adjacency(
    shared_edge_masks: dict[str, np.ndarray],
    ragged_ball_of: Callable[[float], np.ndarray],
) -> None:
    """Testing the face adjacency first gives the full count on every mesh.

    When a feature has built the face adjacency, the count starts from it: it
    has 3 / 2 pairs per face exactly when every edge has two faces. An open
    surface fails that test without a shared edge, and falls back to the count.
    """
    open_mask = np.zeros((6, 6, 6), dtype=bool)
    open_mask[:3, :3, :3] = True
    verts, faces, _, _ = marching_cubes(open_mask, 0.5)
    builders = {
        **{
            name: lambda m=mask: mesh.mask2mesh(m)
            for name, mask in shared_edge_masks.items()
        },
        "clean ball": lambda: mesh.mask2mesh(ragged_ball_of(0.0)),
        "blurred ragged ball": lambda: mesh.mask2mesh(ragged_ball_of(0.3), sigma=2.0),
        "open surface": lambda: trimesh.Trimesh(vertices=verts, faces=faces),
    }
    counts = {}
    for name, build in builders.items():
        full = mesh._overshared_edge_count(build())
        assert mesh._overshared_edge_count(build(), adjacency_measured=True) == full
        counts[name] = full

    assert counts["ragged ball 30%"] == 33
    assert counts["clean ball"] == counts["open surface"] == 0
    # The mean curvature builds the adjacency, so the warning takes that path.
    for name in ("ragged ball 30%", "clean ball"):
        caught = _shared_edge_warnings(
            lambda s=builders[name](): mesh.extract_mesh_features(
                s, ["Avg Mean Curvature"]
            )
        )
        assert len(caught) == (counts[name] > 0), name


@pytest.mark.parametrize(
    "entry", ["extract_features", "extract_surface_features", "extract_mesh_features"]
)
def test_shared_edge_warning_points_at_the_caller(
    entry: str, ragged_ball: np.ndarray
) -> None:
    """The warning is attributed to the calling line through every public entry."""
    names = ["Euler Characteristic"]
    calls = {
        "extract_features": lambda: mesh.extract_features(
            ragged_ball.astype(np.int32), names
        ),
        "extract_surface_features": lambda: mesh.extract_surface_features(
            ragged_ball, features=names
        ),
        "extract_mesh_features": lambda: mesh.extract_mesh_features(
            mesh.mask2mesh(ragged_ball), names
        ),
    }

    caught = _shared_edge_warnings(calls[entry])

    assert len(caught) == 1
    assert caught[0].filename == __file__


def test_shared_edge_warning_is_shown_once_for_a_label_image(
    ragged_ball_of: Callable[[float], np.ndarray],
) -> None:
    """Two labels with shared edges give one warning under the default filter.

    The filter keys a text by the calling line, so the warning is shown once
    per calling line: once for this label image.
    """
    labels = np.zeros((41, 41, 82), dtype=np.int32)
    labels[:, :, :41][ragged_ball_of(0.1)] = 1
    labels[:, :, 41:][ragged_ball_of(0.3)] = 2

    caught = _shared_edge_warnings(
        lambda: mesh.extract_features(labels, ["Euler Characteristic"]), "default"
    )

    assert len(caught) == 1


@pytest.mark.parametrize("name", ["Fractal Dimension", "Outline Fractal Dimension"])
def test_fractal_dimension_needs_the_mask(name: str) -> None:
    """extract_mesh_features refuses either D and leaves it out of its defaults.

    A single mesh cannot be re-blurred.
    """
    ball = mesh.mask2mesh(_ball(8, 22))

    with pytest.raises(ValueError, match=f"{name}.*mask"):
        mesh.extract_mesh_features(ball, [name])
    assert name not in mesh.extract_mesh_features(ball)


_OUTLINE = "Outline Fractal Dimension"


def _outline_fd(mask: np.ndarray, spacing=None, scales=None) -> float:
    return mesh.extract_surface_features(
        mask, spacing=spacing, features=[_OUTLINE], outline_fractal_scales=scales
    )[_OUTLINE]


def _flat_ellipsoid(axes, spacing=None, rot_deg: float = 0.0) -> np.ndarray:
    """Ellipsoid with semi-axes ``axes`` in units of ``spacing``, turned about z."""
    step = (1.0, 1.0, 1.0) if spacing is None else spacing
    shape = [2 * int(np.ceil((1.2 * a + 1) / s)) + 1 for a, s in zip(axes, step)]
    z, y, x = (g * s for g, s in zip(_centred_grid(*shape), step))
    t = np.deg2rad(rot_deg)
    y, x = np.cos(t) * y + np.sin(t) * x, -np.sin(t) * y + np.cos(t) * x
    return (z / axes[0]) ** 2 + (y / axes[1]) ** 2 + (x / axes[2]) ** 2 <= 1


_FLAT_SPACING = (0.29, 0.108, 0.108)


def _rough_cylinder(amplitude: float, radius: float = 6.0) -> np.ndarray:
    """Return a 4-plane cylinder (radius in um) at ``_FLAT_SPACING``.

    Its outline radius carries band-limited noise (0.5 um wavelength) of
    standard deviation ``amplitude`` radii; 0 gives the smooth disc.
    """
    plane = _FLAT_SPACING[1:]
    n = 2 * int(np.ceil(1.4 * radius / plane[0])) + 1
    y, x = (g * s for g, s in zip(_centred_grid(n, n), plane))
    r = np.sqrt(y**2 + x**2)
    noise = ndimage.gaussian_filter(
        np.random.default_rng(0).standard_normal(r.shape), [0.5 / s for s in plane]
    )
    footprint = np.pad(r + amplitude * radius * noise / noise.std() < radius, 2)
    return np.repeat(footprint[None], 4, axis=0)


@pytest.mark.parametrize(
    ("spacing", "axes"),
    [
        (None, (6, 60, 80)),
        (None, (10, 50, 50)),
        (_NUCLEUS_SPACING, (2.0, 8.0, 10.0)),
        (_FLAT_SPACING, (2.0, 6.0, 8.0)),
        ((1.0, 0.2, 0.1), (3.0, 8.0, 10.0)),
    ],
)
def test_outline_fractal_dimension_of_a_smooth_outline_is_one(spacing, axes) -> None:
    """A smooth elliptical footprint gives 1, also on an anisotropic plane.

    Measured: 0.992 for the (0.2, 0.1) plane, whose 16-voxel top scale is 0.4
    of the minor semi-axis, and 0.995-0.999 for the others.
    """
    assert _outline_fd(_flat_ellipsoid(axes, spacing), spacing) == pytest.approx(
        1, abs=0.01
    )


def test_outline_fractal_dimension_rises_with_outline_roughness() -> None:
    """Outline roughness inside the window raises D, monotonically.

    Measured 0.997, 1.008 and 1.054 for 0, 4% and 8% of the radius.
    """
    smooth, rough, rougher = (
        _outline_fd(_rough_cylinder(amplitude), _FLAT_SPACING)
        for amplitude in (0, 0.04, 0.08)
    )

    assert smooth == pytest.approx(1, abs=0.01)
    assert smooth < rough < rougher
    assert rougher > smooth + 0.03


def test_outline_fractal_dimension_follows_turns_about_the_first_axis() -> None:
    """Turning the object about the projection axis leaves D (nearly) as is.

    A quarter turn maps the voxel grid onto itself, which leaves only the
    rounding of float32 vertices (1e-8 here). Other angles revoxelize the
    outline: 1.8e-4 at 45 degrees.
    """
    mask = _rough_cylinder(0.02)
    upright = _outline_fd(mask, _FLAT_SPACING)

    quarter = _outline_fd(np.rot90(mask, axes=(1, 2)), _FLAT_SPACING)
    assert quarter == pytest.approx(upright, rel=1e-6)
    for angle in (30, 45):
        turned = ndimage.rotate(mask, angle, axes=(1, 2), order=0, reshape=True)
        assert _outline_fd(turned, _FLAT_SPACING) == pytest.approx(upright, abs=2e-3)
    axes = (2.0, 6.0, 8.0)
    ellipsoid = _outline_fd(_flat_ellipsoid(axes, _FLAT_SPACING), _FLAT_SPACING)
    for angle in (30, 45):
        turned = _flat_ellipsoid(axes, _FLAT_SPACING, angle)
        assert _outline_fd(turned, _FLAT_SPACING) == pytest.approx(ellipsoid, abs=2e-3)


def test_outline_fractal_dimension_is_measured_along_the_first_axis() -> None:
    """Lying the same rough cylinder on its side changes the footprint and D.

    Seen from the side, the 3 um tall cylinder is a rectangle whose long edges
    take the widest radius of each row, so its outline is nearly straight.
    """
    mask = np.repeat(_rough_cylinder(0.08)[:1], 28, axis=0)
    spacing = (0.108, 0.108, 0.108)

    upright = _outline_fd(mask, spacing)
    on_its_side = _outline_fd(np.swapaxes(mask, 0, 1), spacing)

    assert abs(upright - on_its_side) > 0.02


@pytest.mark.parametrize("plane", [(0.108, 0.108), (0.2, 0.1)])
def test_tight_blur_matches_the_wide_one_on_the_frame(plane, monkeypatch) -> None:
    """The outline's one-voxel blur equals the wide blur on the frame and ring.

    The footprints touch their crop on all four sides, where the ring is
    highest; it stays below 0.5, so the contour closes inside the array. The
    guard of ``_padded_blur`` is off, so the real blurs are compared also
    where it would skip them (the box at the larger scale on the coarser
    plane, whose blur peaks at 0.28).
    """
    monkeypatch.setattr(mesh, "_VANISHING_BOUND", 0.0)
    step = np.asarray(plane)
    box = np.ones((40, 30), dtype=bool)
    rough = _rough_cylinder(0.08)[0]
    rough = rough[np.ix_(rough.any(axis=1), rough.any(axis=0))]
    for footprint in (box, rough):
        for sigma in (4 * max(plane), 16 * max(plane)):
            wide, pad = mesh._padded_blur(footprint, step, sigma)
            tight, ring = mesh._padded_blur(footprint, step, sigma, tight=True)

            assert ring.tolist() == [1, 1]
            inner = wide[pad[0] - 1 : 1 - pad[0], pad[1] - 1 : 1 - pad[1]]
            np.testing.assert_array_equal(tight, inner)
            border = np.concatenate([tight[0], tight[-1], tight[:, 0], tight[:, -1]])
            assert border.max() < 0.5


def test_outline_fractal_window() -> None:
    """The default window is 4 to 16 coarser in-plane voxels.

    outline_fractal_scales moves it.
    """
    mask = _rough_cylinder(0.04)
    default = _outline_fd(mask, _FLAT_SPACING)

    explicit = _outline_fd(mask, _FLAT_SPACING, (4 * 0.108, 16 * 0.108))
    custom = _outline_fd(mask, _FLAT_SPACING, (0.3, 0.9))

    assert explicit == default
    assert np.isfinite(custom)
    assert abs(custom - default) > 1e-3


def test_outline_fractal_dimension_of_a_footprint_narrower_than_the_blur() -> None:
    """A footprint that vanishes at the top scale gives NaN, with a warning.

    The ball's footprint, 16 voxels across, vanishes under the default 16-voxel
    blur; the other features stay finite.
    """
    names = [_OUTLINE, "Fractal Dimension", "Volume"]
    with pytest.warns(UserWarning, match="Outline Fractal Dimension: the mask or its"):
        features = mesh.extract_surface_features(_ball(8, 22), features=names)

    assert np.isnan(features[_OUTLINE])
    assert np.isfinite(features["Fractal Dimension"])
    assert features["Volume"] > 0


def test_outline_fractal_dimension_of_a_sheet_too_thin_for_curvature_sigma() -> None:
    """The outline does not depend on the 3D blur at curvature_sigma.

    A one-plane sheet vanishes under that blur, which makes the curvature
    features NaN, but its footprint is a plain square.
    """
    sheet = np.zeros((3, 64, 64), dtype=bool)
    sheet[1, 2:62, 2:62] = True
    names = [_OUTLINE, "Avg Curvedness", "Fractal Dimension"]

    with pytest.warns(UserWarning, match="vanishes when blurred at curvature_sigma"):
        features = mesh.extract_surface_features(sheet, _NUCLEUS_SPACING, names)

    assert np.isnan(features["Avg Curvedness"])
    assert np.isnan(features["Fractal Dimension"])
    assert np.isfinite(features[_OUTLINE])


def test_fractal_dimension_of_a_sheet_thinner_than_the_blur() -> None:
    """D of a slab thinner than the blur is NaN, with a warning.

    The 3-plane slab survives the blur, so the curvature features stay finite,
    but its thinner twin vanishes when blurred again.
    """
    slab = np.zeros((9, 60, 60), dtype=bool)
    slab[3:6, 10:50, 10:50] = True
    names = ["Fractal Dimension", "Avg Curvedness"]

    with pytest.warns(UserWarning, match="smooth twin vanishes"):
        features = mesh.extract_surface_features(
            slab, spacing=_NUCLEUS_SPACING, features=names
        )

    assert np.isnan(features["Fractal Dimension"])
    assert np.isfinite(features["Avg Curvedness"])


@pytest.mark.parametrize("spacing", [None, _NUCLEUS_SPACING])
def test_smooth_twin_fits_in_the_mask_frame(spacing) -> None:
    """Cropping the twin to the mask's frame loses none of its voxels.

    Both objects touch their frame on every face, where the blur outside the
    frame is highest: a solid box along whole faces, a ball at their centres.
    The twin of the box reaches every face too, so no tighter crop would do.
    """
    step = mesh._step(spacing)
    sigma = mesh._curvature_sigma(spacing, None)
    box = np.ones((12, 30, 30), dtype=bool)
    for mask in (box, _ball(6, 13)):
        blurred, pad = mesh._padded_blur(mask, step, sigma)
        twin = mesh._smooth_twin(blurred, pad)

        assert twin.shape == mask.shape
        assert int(twin.sum()) == int((blurred > 0.5).sum()) > 0
        if mask is box:
            for axis in range(3):
                assert twin.take(0, axis=axis).any()
                assert twin.take(-1, axis=axis).any()


def test_fractal_dimension_needs_a_blur_above_the_finest_spacing() -> None:
    """D is NaN, with a warning, when curvature_sigma is not above the finest spacing.

    There is then no range of scales to fit.
    """
    with pytest.warns(UserWarning, match="finest voxel spacing"):
        features = mesh.extract_surface_features(
            _ball(8, 22), features=["Fractal Dimension", "Volume"], curvature_sigma=1.0
        )

    assert np.isnan(features["Fractal Dimension"])
    assert features["Volume"] > 0


def test_empty_fractal_window_is_decided_before_any_blur(monkeypatch) -> None:
    """An empty default window gives NaN without blurring the mask.

    With a curvature feature too, a mask that vanishes at curvature_sigma
    reports only that feature as vanishing, not Fractal Dimension.
    """

    def fail(*args, **kwargs):
        raise AssertionError("blurred the mask")

    with monkeypatch.context() as patch:
        patch.setattr(mesh, "_padded_blur", fail)
        with pytest.warns(UserWarning, match="finest voxel spacing"):
            alone = mesh.extract_surface_features(
                _ball(8, 22), features=["Fractal Dimension"], curvature_sigma=1.0
            )
    assert np.isnan(alone["Fractal Dimension"])

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mesh.extract_surface_features(
            np.ones((1, 1, 1), dtype=bool),
            features=["Avg Curvedness", "Fractal Dimension"],
            curvature_sigma=1.0,
        )
    vanished = [str(w.message) for w in caught if "vanishes" in str(w.message)]
    assert vanished == [
        "Mask of 1 voxels vanishes when blurred at curvature_sigma=1; "
        "returning NaN for ['Avg Curvedness']."
    ]


def test_curvature_features_of_an_object_too_small_for_the_blur() -> None:
    """A mask that vanishes under the blur gets NaN for those features only."""
    names = [
        "Volume",
        "Total Absolute Gaussian Curvature",
        "Avg Shape Index",
        "Fractal Dimension",
    ]

    with pytest.warns(UserWarning, match="vanishes when blurred at curvature_sigma=2"):
        features = mesh.extract_surface_features(
            np.ones((1, 1, 1), dtype=bool), features=names
        )

    assert features["Volume"] > 0
    assert np.isnan(features["Total Absolute Gaussian Curvature"])
    assert np.isnan(features["Avg Shape Index"])
    assert np.isnan(features["Fractal Dimension"])


def test_a_blur_that_provably_vanishes_does_not_run(monkeypatch) -> None:
    """A blur 1000 times the spacing gives the outcome of a vanished mask unblurred.

    Its wide padding alone would be a 6.3 TB array (6026 x 16136 x 16136
    float32 voxels); the bound shows the blur
    stays below 0.5, so the curvature features and Fractal Dimension are NaN
    with the usual warning and ``mask2mesh`` raises, without blurring.
    """
    ball = _ball(8, 22)
    spacing = (0.29, 0.108, 0.108)
    sigma = 1000 * max(spacing)

    def fail(*args, **kwargs):
        raise AssertionError("ran the blur")

    monkeypatch.setattr(mesh, "gaussian_filter_like_scipy", fail)
    with pytest.warns(UserWarning, match="vanishes when blurred at curvature_sigma"):
        features = mesh.extract_surface_features(
            ball, spacing=spacing, curvature_sigma=sigma
        )
    with pytest.raises(ValueError, match="vanishes when blurred"):
        mesh.mask2mesh(ball, spacing=spacing, sigma=sigma)

    vanished = [*mesh._SMOOTHED_FEATURES, "Fractal Dimension"]
    assert all(np.isnan(features[name]) for name in vanished)
    assert all(np.isfinite(v) for n, v in features.items() if n not in vanished)


@pytest.mark.parametrize(
    ("keyword", "name"),
    [
        ("fractal_scales", "Fractal Dimension"),
        ("outline_fractal_scales", "Outline Fractal Dimension"),
    ],
)
def test_a_fractal_window_far_wider_than_the_object(keyword: str, name: str) -> None:
    """A top scale of 1e6 voxels gives NaN with a warning; only the low rungs blur."""
    low = 1 if keyword == "fractal_scales" else 4
    with pytest.warns(UserWarning, match=f"{name}: the mask or its smooth twin"):
        features = mesh.extract_surface_features(
            _ball(8, 22), features=[name], **{keyword: (low, 1e6)}
        )

    assert np.isnan(features[name])


def _sigma_with_bound(mask: np.ndarray, step: np.ndarray, target: float) -> float:
    """Return the blur scale at which ``_blur_bound`` of ``mask`` is ``target``."""
    low, high = 1e-3, 1e3
    for _ in range(80):
        mid = np.sqrt(low * high)
        if mesh._blur_bound(mask, mid / step) > target:
            low = mid
        else:
            high = mid
    return high


@pytest.mark.parametrize("tight", [False, True], ids=["wide", "tight"])
@pytest.mark.parametrize(
    "step",
    [(1.0, 1.0, 1.0), (2.685, 1.0, 1.0), (0.29, 0.108, 0.108), (1.0, 1.0), (1.0, 0.4)],
    ids=str,
)
def test_the_blur_guard_fires_only_on_masks_that_vanish(
    step, tight: bool, monkeypatch
) -> None:
    """Where the bound is just under the guard's threshold, the real blur is below 0.5.

    One voxel, two voxels along each axis, then seeded random masks of up to
    64 voxels, with the blur scale chosen so that the bound is 0.9 to 1 times
    the threshold; the guard is off, so ``_padded_blur`` runs the blur. For
    one or two voxels the bound is nearly the real maximum, so the largest
    real maximum comes close to the threshold: a bound that is too small is
    caught.
    """
    step = np.asarray(step)
    rng = np.random.default_rng(0)
    threshold = mesh._VANISHING_BOUND
    monkeypatch.setattr(mesh, "_VANISHING_BOUND", 0.0)
    ndim = step.size
    fixed = [(1,) * ndim] + [
        tuple(2 if a == axis else 1 for a in range(ndim)) for axis in range(ndim)
    ]
    masks = [np.ones(shape, dtype=bool) for shape in fixed]
    for _ in range(30 - len(masks)):
        mask = rng.random(rng.integers(1, 5, size=ndim)) < rng.uniform(0.2, 1.0)
        mask.flat[0] = True
        masks.append(mask)
    largest = 0.0
    for case, mask in enumerate(masks):
        sigma = _sigma_with_bound(mask, step, rng.uniform(0.9, 1.0) * threshold)
        bound = mesh._blur_bound(mask, sigma / step)
        assert 0.9 * threshold <= bound < threshold, (case, bound)

        real = float(mesh._padded_blur(mask, step, sigma, tight=tight)[0].max())

        assert real <= 0.5, (case, mask.shape, sigma, real, bound)
        largest = max(largest, real)
    assert largest > 0.4, largest


def test_mesh_is_smoothed_only_for_curvature_features(monkeypatch) -> None:
    """Features that need neither the blur nor the smoothed mesh never compute it."""

    def fail(*args, **kwargs):
        raise AssertionError("blurred the mask")

    monkeypatch.setattr(mesh, "_padded_blur", fail)
    features = mesh.extract_surface_features(
        _ball(8, 22), features=["Volume", "Avg Mean Curvature"]
    )

    assert features["Volume"] > 0


def test_fractal_dimension_alone_builds_no_smoothed_mesh(monkeypatch) -> None:
    """Fractal Dimension reuses the blurred level set but needs no mesh object."""

    def fail(*args, **kwargs):
        raise AssertionError("built the smoothed mesh")

    monkeypatch.setattr(mesh, "_level_set_mesh", fail)
    name = "Fractal Dimension"
    features = mesh.extract_surface_features(_ball(8, 22), features=[name])

    assert features[name] == pytest.approx(2, abs=0.05)


def test_extract_mesh_features_measures_curvature_on_the_smooth_mesh() -> None:
    """Passing the smoothed mesh reproduces the per-mask pipeline exactly."""
    mask = _ball(8, 22)
    raw, smooth = mesh.mask2mesh(mask), mesh.mask2mesh(mask, sigma=2.0)

    from_meshes = mesh.extract_mesh_features(
        raw, _CURVATURE_FEATURES + ["Volume"], smooth_mesh=smooth
    )
    from_mask = mesh.extract_surface_features(
        mask, features=_CURVATURE_FEATURES + ["Volume"]
    )

    assert from_meshes == pytest.approx(from_mask)


def test_principal_curvatures_are_computed_once_per_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Avg Shape Index and Avg Curvedness share one per-vertex curvature pass.

    Each used to compute it. One requested alone gives the registry's value;
    and the caller's mesh is not kept alive by the call.
    """
    calls = []
    principal_curvatures = mesh._principal_curvatures

    def counted(
        surface: trimesh.Trimesh,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        calls.append(surface)
        return principal_curvatures(surface)

    monkeypatch.setattr(mesh, "_principal_curvatures", counted)
    mask = _ball(8, 22)

    mesh.extract_surface_features(mask)
    assert len(calls) == 1

    surface = mesh.mask2mesh(mask, sigma=2.0)
    for name in ("Avg Shape Index", "Avg Curvedness"):
        alone = mesh._measure_mesh(surface, [name])[name]
        assert alone == float(mesh._MESH_FEATURES[name](surface))
    calls.clear()
    reference = weakref.ref(surface)
    mesh.extract_mesh_features(surface)
    assert len(calls) == 1
    calls.clear()
    del surface
    gc.collect()
    assert reference() is None


def test_curvature_sigma_is_validated() -> None:
    """A non-positive blur is rejected, and mask2mesh refuses a vanished mask."""
    with pytest.raises(ValueError, match="curvature_sigma must be finite and positive"):
        mesh.extract_surface_features(
            _ball(8, 22), features=_CURVATURE_FEATURES, curvature_sigma=0
        )
    with pytest.raises(ValueError, match="vanishes when blurred"):
        mesh.mask2mesh(np.ones((1, 1, 1), dtype=bool), sigma=2.0)


@pytest.mark.parametrize(
    ("entry", "sigma", "labelled"),
    [
        ("extract_surface_features", -1.0, True),
        ("extract_features", float("nan"), True),
        ("extract_features", float("inf"), False),
        ("extract_surface_features", 0.6, True),
        ("extract_features", 0.6, True),
        ("extract_features", 2.0, False),
    ],
)
def test_curvature_sigma_is_validated_whichever_features(
    entry: str, sigma: float, labelled: bool
) -> None:
    """curvature_sigma is checked at entry, like the windows and the spacing.

    It used to be checked only when a smoothed feature was requested, so a bad
    value passed with ``["Volume"]`` or on an image without labels. The error
    names the caller's value, not the one divided by the finest spacing.
    """
    mask = _cube_volume(4) == 1 if labelled else np.zeros((6, 6, 6), dtype=bool)
    spacing = (0.5, 0.25, 0.25)

    def call() -> object:
        if entry == "extract_features":
            return mesh.extract_features(
                mask.astype(np.int32), ["Volume"], spacing, curvature_sigma=sigma
            )
        return mesh.extract_surface_features(
            mask, spacing, ["Volume"], curvature_sigma=sigma
        )

    if np.isfinite(sigma) and sigma > 0:
        call()
    else:
        with pytest.raises(
            ValueError,
            match=rf"curvature_sigma must be finite and positive; got {sigma}$",
        ):
            call()


def test_mask2mesh_checks_its_sigma_and_refuses_ignored_keywords() -> None:
    """A bad ``sigma`` is named as such, and keywords it would ignore are refused.

    With ``sigma`` the mask is blurred and meshed without
    ``marching_cubes_kwargs``, so a ``spacing`` passed there would silently
    leave the mesh in voxel units.
    """
    mask = _ball(8, 22)
    with pytest.raises(ValueError, match="^sigma must be finite and positive"):
        mesh.mask2mesh(mask, sigma=0)
    with pytest.raises(ValueError, match="marching_cubes_kwargs cannot be combined"):
        mesh.mask2mesh(
            mask, marching_cubes_kwargs={"spacing": (2.0, 0.5, 0.5)}, sigma=1.0
        )


def test_small_object_does_not_fail_a_label_image() -> None:
    """A tiny object next to a normal one gets NaN where it cannot be measured.

    A ball of radius 3 survives the default blur as a 44-face mesh. With the
    default features the call returns, with NaN only for the tiny object's
    ``Fractal Dimension``.
    """
    labels = np.zeros((22, 22, 44), dtype=np.int32)
    labels[:, :, :22][_ball(3, 22)] = 1
    labels[:, :, 22:][_ball(8, 22)] = 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, values, columns = mesh.extract_features(labels)

    fractal = values[:, columns.index("Fractal Dimension")]
    assert np.isnan(fractal[0]) and np.isfinite(fractal[1])
    assert np.isfinite(np.delete(values, columns.index("Fractal Dimension"), 1)).all()


def test_sphericity_is_one_for_a_sphere_only() -> None:
    """Wadell sphericity is 1 for a sphere and below 1 for a cube."""
    assert mesh.sphericity(4 / 3 * np.pi, 4 * np.pi) == pytest.approx(1)
    assert mesh.sphericity(1.0, 6.0) == pytest.approx((np.pi / 6) ** (1 / 3))


def test_mesh_is_reachable_from_cubic_feature_without_eager_import() -> None:
    """``cubic.feature.mesh`` resolves on access, not on ``import cubic.feature``.

    It used to raise ``AttributeError`` unless the submodule had been imported
    explicitly; importing it eagerly would make every user of the voxel and
    texture features pay for the trimesh import.
    """
    assert cubic.feature.__getattr__("mesh") is mesh
    # Only a fresh interpreter can show that the package import, star import
    # included, skips trimesh.
    code = (
        "import sys\nfrom cubic.feature import *\nassert 'trimesh' not in sys.modules\n"
    )
    # A timeout turns a hung import (e.g. a CUDA probe) into a failure.
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


@pytest.mark.parametrize("version", ["4.6.11", "4.6", "3.23.5"])
def test_trimesh_older_than_the_signed_curvature_is_refused(version: str) -> None:
    """A trimesh without signed dihedral angles fails at import, not in the values."""
    with pytest.raises(ImportError, match=r"trimesh>=4\.6\.12; found"):
        mesh._check_trimesh_version(version)


@pytest.mark.parametrize("version", ["4.6.12", "4.11.2", "5.0.0rc1"])
def test_supported_trimesh_is_accepted(version: str) -> None:
    """The floor itself, later releases and uncomparable tags pass the check."""
    mesh._check_trimesh_version(version)
