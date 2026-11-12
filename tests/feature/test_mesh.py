"""Tests for mesh-based feature extraction."""

import sys
import warnings
import subprocess

import numpy as np
import pytest
from scipy import ndimage
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

    def fake_surface_features(
        mask,
        spacing=None,
        features=None,
        curvature_sigma=None,
        fractal_scales=None,
        outline_fractal_scales=None,
    ) -> dict[str, float]:
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
    z, y, x = _centred_grid(50, 50, 50)
    torus = (np.sqrt(y**2 + x**2) - 14) ** 2 + z**2 <= 6**2
    zz, yy, xx = _centred_grid(24, 24, 36)
    peanut = (zz**2 + yy**2 + (xx - 6) ** 2 <= 8**2) | (
        zz**2 + yy**2 + (xx + 6) ** 2 <= 8**2
    )
    name = ["Total Absolute Gaussian Curvature"]

    torus_k = mesh.extract_surface_features(torus, features=name)[name[0]]
    peanut_k = mesh.extract_surface_features(peanut, features=name)[name[0]]

    assert torus_k == pytest.approx(2, rel=0.05)
    assert peanut_k > 1.1


def test_curvature_distribution_converts_with_the_spacing_unit() -> None:
    """The default blur follows the spacing, so a unit change only rescales.

    Unitless features stay put and curvedness scales as 1 / length.
    """
    mask = _ball(8, 22)
    coarse = mesh.extract_surface_features(mask, (1.0, 1.0, 1.0), _CURVATURE_FEATURES)
    fine = mesh.extract_surface_features(mask, (0.1, 0.1, 0.1), _CURVATURE_FEATURES)

    for name in ("Total Absolute Gaussian Curvature", "Avg Shape Index"):
        assert fine[name] == pytest.approx(coarse[name], rel=1e-6)
    assert fine["Avg Curvedness"] == pytest.approx(10 * coarse["Avg Curvedness"])


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
def test_warnings_point_at_the_caller(case: str, match: str) -> None:
    """A direct call's warnings are attributed to the calling line, not to cubic.

    The _require_trimesh wrapper adds a frame; with stacklevel=2 every warning
    landed on that wrapper in mesh.py. The outline and 3D-twin cases are raised
    inside the shared blur-ladder estimator.
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
        mesh.extract_surface_features(mask, spacing, names, **kwargs)

    matching = [w for w in caught if match in str(w.message)]
    assert len(matching) == 1
    assert matching[0].filename == __file__


def test_thin_object_warning_is_shown_once_for_a_label_image() -> None:
    """Many thin labels give one warning under Python's default filter.

    The slabs differ in thickness and width, so a text naming each object's
    size would be shown once per label.
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
def test_tight_blur_matches_the_wide_one_on_the_frame(plane) -> None:
    """The outline's one-voxel blur equals the wide blur on the frame and ring.

    The footprints touch their crop on all four sides, where the ring is
    highest; it stays below 0.5, so the contour closes inside the array.
    """
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


def test_curvature_sigma_is_validated() -> None:
    """A non-positive blur is rejected, and mask2mesh refuses a vanished mask."""
    with pytest.raises(ValueError, match="curvature_sigma must be finite and positive"):
        mesh.extract_surface_features(
            _ball(8, 22), features=_CURVATURE_FEATURES, curvature_sigma=0
        )
    with pytest.raises(ValueError, match="vanishes when blurred"):
        mesh.mask2mesh(np.ones((1, 1, 1), dtype=bool), sigma=2.0)


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
