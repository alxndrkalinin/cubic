"""GPU mesh features: CUDA marching cubes and ``ArrayMesh`` against the CPU path."""

import warnings
from typing import NoReturn
from collections.abc import Callable

import numpy as np
import pytest
from scipy import ndimage
from skimage.measure import marching_cubes

trimesh = pytest.importorskip("trimesh")

from cubic.cuda import ascupy, asnumpy, get_device  # noqa: E402
from cubic.feature import mesh, _marching_cubes_cuda  # noqa: E402
from cubic.feature._array_mesh import ArrayMesh, face_areas  # noqa: E402
from cubic.feature._marching_cubes_cuda import (  # noqa: E402
    marching_cubes as marching_cubes_gpu,
)

_SPACING_ERROR = r"spacing must be three finite, positive numbers \(z, y, x\); got "


@pytest.fixture
def no_host_trimesh(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail the test if an ``ArrayMesh`` builds a host trimesh."""

    def fail(self: ArrayMesh) -> NoReturn:
        raise AssertionError("fell back to a host trimesh")

    monkeypatch.setattr(ArrayMesh, "to_trimesh", fail)


def _canonical(verts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted unique vertices, and faces as sorted rows of vertex ranks.

    Each face is rotated to start at its lowest-ranked vertex, which keeps its
    orientation, so two meshes compare equal exactly when they have the same
    vertices and the same oriented triangles in any order.
    """
    unique, rank = np.unique(verts, axis=0, return_inverse=True)
    ranked = rank.reshape(-1)[faces]
    start = np.argmin(ranked, axis=1)[:, None]
    rotated = np.take_along_axis(ranked, (start + np.arange(3)) % 3, axis=1)
    return unique, np.unique(rotated, axis=0)


def _assert_same_mesh(v_gpu, f_gpu, v_cpu: np.ndarray, f_cpu: np.ndarray) -> None:
    """Assert a GPU mesh equals a CPU one bit for bit, up to vertex and face order."""
    gpu = _canonical(asnumpy(v_gpu), asnumpy(f_gpu))
    cpu = _canonical(v_cpu, f_cpu)
    np.testing.assert_array_equal(gpu[0], cpu[0])
    np.testing.assert_array_equal(gpu[1], cpu[1])


def _volumes() -> dict[str, tuple[np.ndarray, float, tuple, bool]]:
    rng = np.random.default_rng(0)
    smooth = ndimage.gaussian_filter(rng.normal(size=(28, 28, 28)), 2).astype(
        np.float32
    )
    z, y, x = np.meshgrid(
        *[np.arange(n) - (n - 1) / 2 for n in (22, 26, 24)], indexing="ij"
    )
    ball = np.pad(z**2 + y**2 + x**2 <= 9**2, 1).astype(np.float32)
    return {
        # dense with every Lewiner case, including the ambiguous ones
        "binary noise": (
            (rng.random((20, 20, 20)) > 0.5).astype(np.float32),
            0.5,
            (1.0, 1.0, 1.0),
            True,
        ),
        "continuous noise": (
            rng.normal(size=(20, 20, 20)).astype(np.float32),
            0.0,
            (1.0, 1.0, 1.0),
            True,
        ),
        "anisotropic ball": (ball, 0.5, (0.325, 0.088, 0.088), True),
        # values equal to the level create coincident vertices
        "level hits, merged": (
            (np.round(smooth * 4) / 4).astype(np.float32),
            0.0,
            (1.0, 1.0, 1.0),
            False,
        ),
    }


@pytest.mark.parametrize("name", list(_volumes()))
def test_gpu_marching_cubes_matches_skimage(name: str, gpu_available: bool) -> None:
    """The CUDA port yields skimage's mesh bit for bit, up to vertex and face order."""
    if not gpu_available:
        pytest.skip("GPU not available")
    volume, level, spacing, allow_degenerate = _volumes()[name]
    v_cpu, f_cpu, _, _ = marching_cubes(
        volume, level, spacing=spacing, allow_degenerate=allow_degenerate
    )
    v_gpu, f_gpu = marching_cubes_gpu(ascupy(volume), level, spacing, allow_degenerate)

    assert asnumpy(v_gpu).dtype == v_cpu.dtype
    _assert_same_mesh(v_gpu, f_gpu, v_cpu, f_cpu)


@pytest.mark.parametrize("device", ["CPU", "GPU"])
def test_marching_cubes_faces_wind_outward(device: str, gpu_available: bool) -> None:
    """The faces from ``mesh._marching_cubes`` enclose a positive volume.

    Both implementations wind faces inward for skimage's default gradient
    direction, and ``_marching_cubes`` reverses them; the random mask's mesh is
    not watertight, which does not matter to that rule.
    """
    if device == "GPU" and not gpu_available:
        pytest.skip("GPU not available")
    rng = np.random.default_rng(0)
    z, y, x = np.meshgrid(*[np.arange(24) - 11.5] * 3, indexing="ij")
    r2 = z**2 + y**2 + x**2
    volumes = {
        "ball": (np.pad(r2 <= 9**2, 1), None),
        "hollow ball": (np.pad((r2 <= 10**2) & (r2 > 4**2), 1), None),
        "random mask": (np.pad(rng.random((12, 12, 12)) < 0.35, 1), None),
        "blurred mask": (
            mesh._padded_blur(r2 <= 6**2, np.ones(3), 2.0)[0],
            0.5,
        ),
    }
    for name, (volume, level) in volumes.items():
        on_device = ascupy(volume) if device == "GPU" else volume
        verts, faces = mesh._marching_cubes(on_device, level)
        surface = trimesh.Trimesh(asnumpy(verts), asnumpy(faces), process=False)
        assert surface.volume > 0, name


def test_gpu_default_level_rounds_like_skimage(gpu_available: bool) -> None:
    """``level=None`` is skimage's midpoint, halved from the float32 sum of the extremes.

    Here that sum rounds (0.1 + 3.0 is not a float32), so a midpoint taken in
    float64 sits about 5e-8 away and moves vertices by an ulp.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    volume = np.random.default_rng(2).uniform(0.1, 3.0, (20, 20, 20))
    volume = volume.astype(np.float32)
    volume.flat[:2] = 0.1, 3.0
    low, high = volume.min(), volume.max()
    assert float(low + high) != float(low) + float(high)

    v_cpu, f_cpu, _, _ = marching_cubes(volume)
    v_gpu, f_gpu = marching_cubes_gpu(ascupy(volume))

    _assert_same_mesh(v_gpu, f_gpu, v_cpu, f_cpu)


@pytest.mark.parametrize("spacing", [(0.5,), (0.5, 1.0)])
def test_gpu_marching_cubes_rejects_spacing_of_other_lengths(
    spacing: tuple, gpu_available: bool
) -> None:
    """A spacing of other than three values raises skimage's error on the GPU too.

    Unchecked, one value scaled every axis and two failed to broadcast.
    ``mask2mesh`` rejects it before meshing, with its own message.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    mask = _labels() == 1
    volume = np.pad(mask, 1).astype(np.float32)
    match = "`spacing` must consist of three floats"

    with pytest.raises(ValueError, match=match):
        marching_cubes(volume, 0.5, spacing=spacing)
    with pytest.raises(ValueError, match=match):
        marching_cubes_gpu(ascupy(volume), 0.5, spacing)
    with pytest.raises(ValueError, match=_SPACING_ERROR):
        mesh.mask2mesh(ascupy(mask), spacing=spacing)


def test_missing_skimage_tables_break_only_the_gpu_path(monkeypatch) -> None:
    """Without skimage's private lookup tables only GPU marching cubes fails.

    The CPU path of mask2mesh still meshes, and the GPU entry point names the
    missing module before any CuPy call, so this needs no GPU.
    """
    monkeypatch.setattr(_marching_cubes_cuda, "_mcluts", None)
    monkeypatch.setattr(_marching_cubes_cuda, "_to_array", None)
    z, y, x = np.meshgrid(*[np.arange(n) - 9.5 for n in (20, 20, 20)], indexing="ij")

    assert mesh.mask2mesh(z**2 + y**2 + x**2 <= 8**2).volume > 0
    with pytest.raises(ImportError, match=r"_marching_cubes_lewiner_luts.*_to_array"):
        marching_cubes_gpu(np.zeros((2, 2, 2), dtype=np.float32), 0.5)


def test_array_mesh_matches_trimesh() -> None:
    """On NumPy arrays, ArrayMesh reproduces trimesh's features to rounding.

    It recomputes the trimesh quantities the features read (mass properties,
    adjacency, convexity, angle defects) with trimesh's formulas, and hands
    hull-based features to trimesh itself. The dihedral angles of both come
    from mesh._dihedral_angles.
    """
    z, y, x = np.meshgrid(
        *[np.arange(n) - (n - 1) / 2 for n in (30, 30, 30)], indexing="ij"
    )
    r2 = z**2 + y**2 + x**2
    hollow = np.pad((r2 <= 12**2) & ~(r2 <= 5**2), 1)
    verts, faces = mesh._marching_cubes(hollow, spacing=(2.0, 1.0, 1.0))
    reference = trimesh.Trimesh(verts, faces)

    expected = mesh.extract_mesh_features(reference)
    got = mesh.extract_mesh_features(ArrayMesh(verts, faces))

    assert got == pytest.approx(expected, rel=1e-12)


def test_array_mesh_measures_inward_faces_as_given(gpu_available: bool) -> None:
    """Faces wound inward give minus the outward volume; they are not flipped.

    ``trimesh.Trimesh`` measures its faces as given too, so an inside-out mesh
    reaches the volume check on either device instead of being flipped on one.
    """
    z, y, x = np.meshgrid(*[np.arange(n) - 9.5 for n in (20, 20, 20)], indexing="ij")
    verts, faces = mesh._marching_cubes(np.pad(z**2 + y**2 + x**2 <= 8**2, 1))
    arrays = [(verts, faces)]
    if gpu_available:
        arrays.append((ascupy(verts), ascupy(faces)))
    for v, f in arrays:
        outward = ArrayMesh(v, f).volume
        inward = ArrayMesh(v, f[:, ::-1].copy()).volume

        assert outward > 0
        assert inward == pytest.approx(-outward, rel=1e-12)


def _level_hit_mesh() -> tuple[np.ndarray, np.ndarray]:
    """Mesh of noise rounded to quarters, whose 0.5 level is hit exactly.

    Removing the degenerate faces leaves coincident and unreferenced vertices.
    """
    rng = np.random.default_rng(1)
    volume = np.pad(np.round(rng.random((16, 16, 16)) * 4) / 4, 1).astype(np.float32)
    return mesh._marching_cubes(volume, 0.5, allow_degenerate=False)


def test_array_mesh_merges_vertices_like_trimesh(gpu_available: bool) -> None:
    """Equal and unreferenced vertices are merged as trimesh's processing does.

    Unmerged, each unreferenced vertex added a 2 pi angle defect and each
    duplicate split the defects and edges at a seam: the sum of |defect| / 4 pi
    (total |K|) came out 480.9 against trimesh's 496.1 on this mesh. The mesh
    is not watertight, so the topology-dependent quantities are compared
    directly instead of through extract_mesh_features.

    Defects are compared away from zero-area (collinear) faces: whether trimesh
    or ArrayMesh zeroes such a face's angles depends on the last bit of a dot
    product near 1, which arccos amplifies to its 1e-8 tolerance, on any device.
    """
    verts, faces = _level_hit_mesh()
    assert len(np.unique(faces)) < len(verts)
    assert len(np.unique(verts, axis=0)) < len(verts)
    reference = trimesh.Trimesh(verts, faces)  # process=True, as on the CPU path
    key = {tuple(p): i for i, p in enumerate(np.round(reference.vertices * 1e8))}

    arrays = [(verts, faces)]
    if gpu_available:
        arrays.append((ascupy(verts), ascupy(faces)))
    for v, f in arrays:
        merged = ArrayMesh(v, f)
        m_verts, m_faces = asnumpy(merged.vertices), asnumpy(merged.faces)
        assert len(m_verts) == len(reference.vertices)
        assert len(np.unique(m_faces)) == len(m_verts)
        assert merged.euler_number == reference.euler_number
        assert merged.area == pytest.approx(reference.area, rel=1e-12)
        match = np.array([key[tuple(p)] for p in np.round(m_verts * 1e8)])
        flat = np.zeros(len(m_verts), dtype=bool)
        flat[m_faces[asnumpy(merged.area_faces) == 0].ravel()] = True
        np.testing.assert_allclose(
            asnumpy(merged.vertex_defects)[~flat],
            reference.vertex_defects[match[~flat]],
            atol=1e-9,
        )


def test_array_mesh_refuses_in_place_changes() -> None:
    """A delegated method that would change the mesh raises; reads still work.

    Changing only the host copy would leave the arrays and the cached volume
    and area describing a different mesh from the delegated attributes.
    """
    z, y, x = np.meshgrid(*[np.arange(n) - 9.5 for n in (20, 20, 20)], indexing="ij")
    verts, faces = mesh._marching_cubes(np.pad(z**2 + y**2 + x**2 <= 8**2, 1))
    ball = ArrayMesh(verts, faces)

    with pytest.raises(TypeError, match="in place"):
        ball.apply_scale(2.0)

    assert ball.to_trimesh().volume == pytest.approx(ball.volume, rel=1e-12)
    assert ball.copy().volume == pytest.approx(ball.volume, rel=1e-12)


def test_gpu_marching_cubes_runs_on_the_input_device(gpu_available: bool) -> None:
    """A volume on GPU 1 is meshed on GPU 1 while GPU 0 is current."""
    if not gpu_available:
        pytest.skip("GPU not available")
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("needs two GPUs")
    volume, level, spacing, allow_degenerate = _volumes()["anisotropic ball"]
    v_cpu, f_cpu, _, _ = marching_cubes(volume, level, spacing=spacing)
    # float32 is used as is; float64 makes the call copy it to contiguous float32,
    # which the unfixed function did on GPU 0, returning the mesh there
    for dtype in (cp.float32, cp.float64):
        with cp.cuda.Device(1):
            on_1 = cp.asarray(volume, dtype=dtype)

        with cp.cuda.Device(0):
            v_gpu, f_gpu = marching_cubes_gpu(on_1, level, spacing, allow_degenerate)

        assert v_gpu.device.id == 1 and f_gpu.device.id == 1
        _assert_same_mesh(v_gpu, f_gpu, v_cpu, f_cpu)


def _labels() -> np.ndarray:
    z, y, x = np.meshgrid(*[np.arange(n) for n in (30, 64, 64)], indexing="ij")
    labels = np.zeros((30, 64, 64), dtype=np.int32)
    labels[(z - 14) ** 2 + (y - 18) ** 2 / 2 + (x - 20) ** 2 <= 9**2] = 1
    torus = (np.sqrt((y - 44) ** 2 + (x - 42) ** 2) - 12) ** 2 + (z - 14) ** 2 <= 5**2
    labels[torus] = 2
    labels[0:6, 50:64, 0:14] = 3  # touches the border
    return labels


def test_extract_features_on_gpu_matches_cpu(gpu_available: bool) -> None:
    """A CuPy label image gives the CPU features, to rounding.

    Both meshes are identical: the marching cubes port matches skimage, and the
    curvature blur rounds like SciPy on the GPU. Only summation order differs.
    The third label is too thin for the default blur, so its four
    curvature-distribution features are NaN on both devices.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    labels = _labels()
    spacing = (0.3, 0.1, 0.1)

    with pytest.warns(UserWarning, match="vanishes"):
        cpu_labels, cpu, _ = mesh.extract_features(labels, spacing=spacing)
    with pytest.warns(UserWarning, match="vanishes"):
        gpu_labels, gpu, _ = mesh.extract_features(ascupy(labels), spacing=spacing)

    np.testing.assert_array_equal(gpu_labels, cpu_labels)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-9, equal_nan=True)


def test_ragged_label_on_gpu_matches_cpu(
    ragged_ball: np.ndarray, gpu_available: bool
) -> None:
    """A label whose mesh is not watertight gives the same features on both devices.

    The CPU path used to rely on ``fix_normals``, which skips a mesh trimesh
    does not call watertight: there it raised on the negative volume while the
    GPU path measured the label. Its edges shared by four faces make Euler
    Characteristic and Avg Mean Curvature approximate, with a warning on both.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    labels = np.zeros((41, 41, 60), dtype=np.int32)
    labels[:, :, :41][ragged_ball] = 1
    labels[10:20, 10:20, 45:55] = 2
    assert not mesh.mask2mesh(ragged_ball).is_watertight

    with pytest.warns(UserWarning, match=_SHARED_EDGE_TEXT):
        cpu_labels, cpu, _ = mesh.extract_features(labels)
    with pytest.warns(UserWarning, match=_SHARED_EDGE_TEXT):
        gpu_labels, gpu, _ = mesh.extract_features(ascupy(labels))

    np.testing.assert_array_equal(gpu_labels, cpu_labels)
    np.testing.assert_allclose(gpu, cpu, rtol=1e-9, equal_nan=True)


_SHARED_EDGE_TEXT = "edges shared by more than two faces"
# Requested by the shared-edge tests: two exact features, then the five that
# are approximate on such a mesh (the module's _SHARED_EDGE_FEATURES).
_SHARED_EDGE_REQUEST = [
    "Volume",
    "Area",
    "Euler Characteristic",
    "Avg Mean Curvature",
    "Total Absolute Gaussian Curvature",
    "Avg Shape Index",
    "Avg Curvedness",
]
# On the raw mesh, without smooth_mesh, where the devices agree to rounding:
# _principal_curvatures sets the curvatures of flat vertices to 0 (_FLAT_ANGLE,
# _FLAT_DEFECT) and both devices measure at one unit (_edge_unit).
_RAW_CURVATURE = [
    "Total Absolute Gaussian Curvature",
    "Avg Curvedness",
    "Avg Shape Index",
]


def _shared_edge_warnings(call: Callable[[], object]) -> tuple[object, list]:
    """Run ``call``; return its result and the shared-edge warnings it raised."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = call()
    return result, [w for w in caught if _SHARED_EDGE_TEXT in str(w.message)]


def test_shared_edges_are_counted_alike_on_every_mesh(
    shared_edge_masks: dict[str, np.ndarray], gpu_available: bool
) -> None:
    """The trimesh and ArrayMesh paths, NumPy and CuPy, count the same shared edges.

    Also on the level-hit mesh, whose vertices ArrayMesh merges itself.
    """
    meshes = {
        name: mesh._marching_cubes(np.pad(mask, 1))
        for name, mask in shared_edge_masks.items()
    }
    meshes["level hits, merged"] = _level_hit_mesh()
    expected = {"ragged ball 10%": 3, "ragged ball 30%": 33}
    for name, (verts, faces) in meshes.items():
        reference = mesh._overshared_edge_count(trimesh.Trimesh(verts, faces))
        counts = [mesh._overshared_edge_count(ArrayMesh(verts, faces))]
        if gpu_available:
            counts.append(
                mesh._overshared_edge_count(ArrayMesh(ascupy(verts), ascupy(faces)))
            )
        assert counts == [reference] * len(counts), name
        assert reference == expected.get(name, reference) and reference > 0, name


@pytest.mark.usefixtures("no_host_trimesh")
@pytest.mark.parametrize(
    "entry", ["extract_features", "extract_surface_features", "extract_mesh_features"]
)
def test_shared_edge_warning_on_gpu(
    entry: str,
    ragged_ball_of: Callable[[float], np.ndarray],
    gpu_available: bool,
) -> None:
    """A CuPy ragged ball warns, at the calling line; a clean one does not.

    The edges are counted on the device: no host trimesh is built for the
    GPU-native features.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    names = _SHARED_EDGE_REQUEST
    for fraction, expected in [(0.3, 1), (0.0, 0)]:
        mask = ascupy(ragged_ball_of(fraction))
        calls = {
            "extract_features": lambda m=mask: mesh.extract_features(
                m.astype(np.int32), names
            ),
            "extract_surface_features": lambda m=mask: mesh.extract_surface_features(
                m, features=names
            ),
            "extract_mesh_features": lambda m=mask: mesh.extract_mesh_features(
                mesh.mask2mesh(m), names
            ),
        }
        _, caught = _shared_edge_warnings(calls[entry])
        assert len(caught) == expected, fraction
        assert [w.filename for w in caught] == [__file__] * expected


def test_shared_edge_masks_match_on_both_devices(
    shared_edge_masks: dict[str, np.ndarray],
    gpu_available: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Meshes with shared edges give the same values, count and warning everywhere.

    Three ways through extract_surface_features: a NumPy mask (trimesh), a
    NumPy mask meshed as an ArrayMesh on NumPy, and a CuPy mask. The random and
    noise masks vanish under the default blur, so their curvature-distribution
    features are NaN on every way, and no blurred mesh has a shared edge; the
    raw meshes are measured by extract_mesh_features too, for the curvature
    features on a mesh that has them.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    closed_mesh = mesh._closed_mesh

    def numpy_array_mesh(mask: np.ndarray) -> tuple[ArrayMesh, dict[str, float]]:
        monkeypatch.setattr(mesh, "_closed_mesh", ArrayMesh)
        try:
            return mesh.mask2mesh(mask), mesh.extract_surface_features(
                mask, features=_SHARED_EDGE_REQUEST
            )
        finally:
            monkeypatch.setattr(mesh, "_closed_mesh", closed_mesh)

    for name, mask in shared_edge_masks.items():
        ways = {
            "trimesh": lambda m=mask: (
                mesh.mask2mesh(m),
                mesh.extract_surface_features(m, features=_SHARED_EDGE_REQUEST),
            ),
            "ArrayMesh, NumPy": lambda m=mask: numpy_array_mesh(m),
            "ArrayMesh, CuPy": lambda m=mask: (
                mesh.mask2mesh(ascupy(m)),
                mesh.extract_surface_features(ascupy(m), features=_SHARED_EDGE_REQUEST),
            ),
        }
        results = {way: _shared_edge_warnings(call) for way, call in ways.items()}
        (surface, reference), _ = results["trimesh"]
        count = mesh._overshared_edge_count(surface)
        assert count > 0, name
        with pytest.warns(UserWarning, match=_SHARED_EDGE_TEXT):
            raw_reference = mesh.extract_mesh_features(surface, _RAW_CURVATURE)
        for way, ((surface, values), caught) in results.items():
            assert mesh._overshared_edge_count(surface) == count, (name, way)
            assert len(caught) == 1, (name, way)
            raw, raw_caught = _shared_edge_warnings(
                lambda s=surface: mesh.extract_mesh_features(s, _RAW_CURVATURE)
            )
            assert len(raw_caught) == 1, (name, way)
            values = {**values, **{f"raw {k}": v for k, v in raw.items()}}
            reference = {
                **reference,
                **{f"raw {k}": v for k, v in raw_reference.items()},
            }
            for feature, value in values.items():
                np.testing.assert_allclose(
                    value,
                    reference[feature],
                    rtol=1e-9,
                    equal_nan=True,
                    err_msg=f"{name}, {way}, {feature}",
                )


def test_gpu_principal_curvatures_are_computed_once(
    gpu_available: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A CuPy mask computes the per-vertex curvatures once for both averages."""
    if not gpu_available:
        pytest.skip("GPU not available")
    calls = []
    principal_curvatures = mesh._principal_curvatures

    def counted(
        surface: ArrayMesh, *args: object
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        calls.append(type(surface))
        return principal_curvatures(surface, *args)

    monkeypatch.setattr(mesh, "_principal_curvatures", counted)
    mesh.extract_surface_features(
        ascupy(_labels() == 1),
        spacing=(0.3, 0.1, 0.1),
        features=["Avg Shape Index", "Avg Curvedness"],
    )

    assert calls == [ArrayMesh]


@pytest.mark.parametrize("entry", ["extract_surface_features", "extract_features"])
def test_gpu_warnings_point_at_the_caller(entry: str, gpu_available: bool) -> None:
    """On the GPU too, the module's warnings are attributed to the calling line.

    A slab too thin for its Fractal Dimension window, and a single voxel that
    vanishes under the curvature blur.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    slab = np.zeros((12, 60, 60), dtype=bool)
    slab[3:9, 10:50, 10:50] = True
    cases = [
        (slab, (0.3, 0.1, 0.1), ["Fractal Dimension"], "biased low"),
        (np.ones((1, 1, 1), dtype=bool), None, ["Avg Curvedness"], "vanishes"),
    ]
    for mask, spacing, names, match in cases:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if entry == "extract_features":
                mesh.extract_features(ascupy(mask.astype(np.int32)), names, spacing)
            else:
                mesh.extract_surface_features(ascupy(mask), spacing, names)
        matching = [w for w in caught if match in str(w.message)]
        assert [w.filename for w in matching] == [__file__], match


@pytest.mark.usefixtures("no_host_trimesh")
@pytest.mark.parametrize("fractal_scales", [None, (0.15, 0.45)])
def test_fractal_dimension_stays_on_the_gpu(
    fractal_scales, gpu_available: bool
) -> None:
    """Fractal Dimension of a CuPy mask never builds a host trimesh."""
    if not gpu_available:
        pytest.skip("GPU not available")
    mask = ascupy(_labels() == 1)
    name = "Fractal Dimension"
    kwargs = {"spacing": (0.3, 0.1, 0.1), "fractal_scales": fractal_scales}

    on_gpu = mesh.extract_surface_features(mask, features=[name], **kwargs)[name]
    on_cpu = mesh.extract_surface_features(asnumpy(mask), features=[name], **kwargs)[
        name
    ]

    assert np.isfinite(on_cpu)
    assert on_gpu == pytest.approx(on_cpu, rel=1e-9)


def test_gpu_blur_that_provably_vanishes_does_not_run(
    gpu_available: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The stand-in for a skipped blur is on the GPU, and ``mask2mesh`` raises."""
    if not gpu_available:
        pytest.skip("GPU not available")

    def fail(*args, **kwargs) -> NoReturn:
        raise AssertionError("ran the blur")

    monkeypatch.setattr(mesh, "gaussian_filter_like_scipy", fail)
    mask = ascupy(_labels() == 1)
    for tight in (False, True):
        volume, pad = mesh._padded_blur(mask, np.ones(3), 1e4, tight=tight)

        assert get_device(volume) == "GPU"
        assert volume.dtype == np.float32
        assert volume.shape == tuple(n + 2 for n in mask.shape)
        assert float(volume.max()) == 0
        assert pad.tolist() == [1, 1, 1]
    with pytest.raises(ValueError, match="vanishes when blurred"):
        mesh.mask2mesh(mask, sigma=1e4)


def test_gpu_shape_index_of_a_raw_mesh_matches_numpy(gpu_available: bool) -> None:
    """A raw voxel mesh's shape index agrees between NumPy and CuPy to rounding.

    Its flat vertices have dihedral angles and angle defects that are rounding
    residues, which differ between the devices; ``_principal_curvatures`` sets
    them to 0 (``_FLAT_ANGLE``, ``_FLAT_DEFECT``), and the shape index, which
    differed by about 1e-3 between the devices, then agrees to 1e-12. This ball
    is at the default spacing; ``test_gpu_edge_unit_matches_numpy`` covers the
    unit the devices measure at.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    grid = np.indices((36, 36, 36)) - 17.5
    ball = (grid**2).sum(axis=0) <= 15**2
    name = "Avg Shape Index"

    on_cpu = mesh.extract_mesh_features(mesh.mask2mesh(ball), [name])[name]
    on_gpu = mesh.extract_mesh_features(mesh.mask2mesh(ascupy(ball)), [name])[name]

    assert abs(on_gpu - on_cpu) <= 1e-12


@pytest.mark.parametrize("angle", [1e-9, np.pi - 1e-9])
def test_gpu_dihedral_angle_of_a_hinge_is_exact(
    angle: float, gpu_available: bool
) -> None:
    """On CuPy arrays too, two triangles at ``angle`` give that angle to rounding."""
    if not gpu_available:
        pytest.skip("GPU not available")
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0.5, 1, 0], [0.5, -np.cos(angle), np.sin(angle)]]
    )
    faces = np.array([[0, 1, 2], [1, 0, 3]])
    hinge = ArrayMesh(ascupy(vertices), ascupy(faces), merge=False)

    measured = mesh._dihedral_angles(hinge)

    assert get_device(measured) == "GPU"
    assert float(measured[0]) == pytest.approx(angle, rel=1e-12)


def test_gpu_flat_vertices_of_a_raw_mesh_have_no_curvature(
    flat_vertices_of: Callable[[object], np.ndarray], gpu_available: bool
) -> None:
    """On the GPU too, ``k1 = k2 = 0`` exactly where all faces of a vertex are coplanar.

    The raw anisotropic ball of the CPU test, meshed and measured on the GPU.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    grid = np.indices((36, 36, 36)) - 17.5
    ball = (grid**2).sum(axis=0) <= 15**2
    raw = mesh.mask2mesh(ascupy(ball), spacing=(2.117, 0.7884, 0.7884))
    measured = mesh._scaled_copy(raw, 1 / mesh._edge_unit(raw), center=True)
    flat = flat_vertices_of(measured)

    k1, k2, _ = mesh._principal_curvatures(measured)

    assert get_device(k1) == "GPU"
    assert flat.sum() == 1872
    assert (asnumpy(k1)[flat] == 0).all()
    assert (asnumpy(k2)[flat] == 0).all()


def _areas_by_face(vertices, faces) -> tuple[np.ndarray, np.ndarray]:
    """Faces as sorted rows of sorted vertex ranks, and their areas in that order.

    The areas are computed on the device of the arrays. Two meshes with the
    same vertices and triangles give the same rows, whatever order and rotation
    their faces are listed in, so their areas compare face by face.
    """
    host_vertices, host_faces = asnumpy(vertices), asnumpy(faces)
    rank = np.unique(host_vertices, axis=0, return_inverse=True)[1].reshape(-1)
    rows = np.sort(rank[host_faces], axis=1)
    order = np.lexsort(rows.T[::-1])
    return rows[order], asnumpy(face_areas(vertices, faces))[order]


def test_gpu_edge_unit_matches_numpy(
    noise_blob_of: Callable[[int], np.ndarray], gpu_available: bool
) -> None:
    """A NumPy and a CuPy mesh of one mask are measured at bitwise the same unit.

    ``extract_mesh_features`` measures scaled copies of the meshes at the unit
    ``_edge_unit`` returns, which uses only the vertex extremes and the face
    count. The CuPy meshes have bitwise the same vertices and
    triangles as the NumPy ones; seed 2's is also given with each face rotated
    to start at its next vertex. The test asserts that some face areas differ
    between the devices, as CuPy 14.0.1 computes them for seeds 5, 7 and 11
    and the rotation does on any version, so that a unit computed from the
    face areas would fail it.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    cases = [
        (2, (0.29, 0.108, 0.108), True),
        (5, (0.29, 0.108, 0.108), False),
        (7, (2.685, 1.0, 1.0), False),
        (11, (0.29, 0.108, 0.108), False),
    ]
    differing = 0
    for seed, spacing, rotate in cases:
        mask = noise_blob_of(seed)
        cpu = mesh.mask2mesh(mask, spacing=spacing)
        gpu = mesh.mask2mesh(ascupy(mask), spacing=spacing)
        if rotate:
            gpu = ArrayMesh(gpu.vertices, np.roll(gpu.faces, -1, axis=1), merge=False)
        cpu_rows, cpu_areas = _areas_by_face(cpu.vertices, cpu.faces)
        gpu_rows, gpu_areas = _areas_by_face(gpu.vertices, gpu.faces)
        np.testing.assert_array_equal(gpu_rows, cpu_rows)
        differing += int((gpu_areas != cpu_areas).sum())

        assert mesh._edge_unit(gpu) == mesh._edge_unit(cpu), (seed, spacing)
    assert differing > 0


def test_gpu_bounding_cylinder_matches_numpy(
    noise_blob_of: Callable[[int], np.ndarray], gpu_available: bool
) -> None:
    """The bounding cylinder of a CuPy mesh is that of the NumPy mesh, to rounding.

    Both are measured on the host, from trimesh's convex hull of bitwise the
    same vertices and the inertia tensor. The hull of a CuPy mesh can list or
    triangulate its faces otherwise, and its inertia tensor is summed on the
    device; together they moved the volume by at most 9.8e-16 relative over
    144 masks under CuPy 14.0.1 and 14.2.0, and left 128 of them bitwise
    equal, but not these two (5.1e-16 and 4.4e-16 apart under 14.0.1).
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    names = ["Bounding Cylinder Volume"]
    for seed, spacing in [(3, (0.29, 0.108, 0.108)), (23, (1.0, 1.0, 1.0))]:
        mask = noise_blob_of(seed)
        cpu = mesh.extract_mesh_features(mesh.mask2mesh(mask, spacing=spacing), names)
        gpu = mesh.extract_mesh_features(
            mesh.mask2mesh(ascupy(mask), spacing=spacing), names
        )
        assert gpu[names[0]] == pytest.approx(cpu[names[0]], rel=1e-14), seed


def test_gpu_oriented_bounding_box_matches_numpy(
    lumpy_nucleus_of: Callable[[int], np.ndarray], gpu_available: bool
) -> None:
    """The oriented bounding box of a CuPy mesh is that of the NumPy mesh, to rounding.

    Both are measured on the host, from trimesh's convex hull of the same
    vertices, which the CuPy mesh lists in another order. trimesh's box
    search, used before, depends on that order: it put the box of this mesh
    1.7% apart between the devices, and that of 10 of 240 nuclei up to 9.4%.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    names = ["Oriented Bounding Box Volume", "Extent"]
    spacing = (2.685, 1.0, 1.0)
    mask = lumpy_nucleus_of(6)
    cpu = mesh.extract_mesh_features(mesh.mask2mesh(mask, spacing=spacing), names)
    gpu = mesh.extract_mesh_features(
        mesh.mask2mesh(ascupy(mask), spacing=spacing), names
    )
    for name in names:
        assert gpu[name] == pytest.approx(cpu[name], rel=1e-12), name


@pytest.mark.parametrize("smoothed", [False, True], ids=["raw", "smooth_mesh"])
def test_gpu_mesh_features_match_numpy_at_an_anisotropic_spacing(
    noise_blob_of: Callable[[int], np.ndarray], smoothed: bool, gpu_available: bool
) -> None:
    """Every mesh feature agrees between NumPy and CuPy for a lobed anisotropic mask.

    The mesh of this mask (seed 7, spacing (2.685, 1, 1)) is one where a unit
    an ulp apart moved trimesh's bounding cylinder search, which
    ``Bounding Cylinder Volume`` used before, by 8.7% between the devices.
    Under CuPy 14.0.1 its face areas differ from NumPy's at 2873 of
    16568 faces, and a unit computed from them differs by an ulp; under CuPy
    14.2.0 they are equal, and this test passes with either unit, which
    ``test_gpu_edge_unit_matches_numpy`` covers instead.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    spacing = (2.685, 1.0, 1.0)
    sigma = mesh._curvature_sigma(spacing, None)
    mask = noise_blob_of(7)

    def features(mask: np.ndarray) -> dict[str, float]:
        raw = mesh.mask2mesh(mask, spacing=spacing)
        smooth = (
            mesh.mask2mesh(mask, spacing=spacing, sigma=sigma) if smoothed else None
        )
        return mesh.extract_mesh_features(raw, smooth_mesh=smooth)

    on_cpu = features(mask)
    on_gpu = features(ascupy(mask))

    assert on_gpu.keys() == on_cpu.keys()
    for name, value in on_gpu.items():
        np.testing.assert_allclose(
            value,
            on_cpu[name],
            rtol=1e-9,
            err_msg=name,
        )


def test_gpu_curvatures_are_the_same_on_every_run(
    noise_blob_of: Callable[[int], np.ndarray], gpu_available: bool
) -> None:
    """The GPU curvatures of a blurred mesh, and every feature, repeat to the bit.

    The per-vertex angle defects, mixed areas and edge sums are bin sums that
    ``ordered_bincount`` adds in input order. With CuPy's weighted
    ``bincount``, which adds them with atomics, the per-vertex curvatures of
    this mesh differed between two runs.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    spacing = (2.685, 1.0, 1.0)
    mask = ascupy(noise_blob_of(7))
    blurred = mesh.mask2mesh(
        mask, spacing=spacing, sigma=mesh._curvature_sigma(spacing, None)
    )

    def curvatures() -> list[np.ndarray]:
        # A new mesh each run, so no per-vertex sum comes from a cache.
        fresh = ArrayMesh(blurred.vertices, blurred.faces, merge=False)
        return [asnumpy(a) for a in mesh._principal_curvatures(fresh)]

    for first, second in zip(curvatures(), curvatures()):
        np.testing.assert_array_equal(second, first)
    first = mesh.extract_surface_features(mask, spacing=spacing)
    second = mesh.extract_surface_features(mask, spacing=spacing)
    assert second.keys() == first.keys()
    for name, value in second.items():
        np.testing.assert_array_equal(value, first[name], err_msg=name)


def _rough_flat_ellipsoid(spacing: tuple) -> np.ndarray:
    """Return a flat (2, 6, 8) um ellipsoid with a rough outline, at ``spacing``."""
    shape = [
        2 * int(np.ceil((1.2 * a + 1) / s)) + 1 for a, s in zip((2, 6, 8), spacing)
    ]
    z, y, x = np.meshgrid(
        *[(np.arange(n) - (n - 1) / 2) * s for n, s in zip(shape, spacing)],
        indexing="ij",
    )
    noise = ndimage.gaussian_filter(
        np.random.default_rng(3).standard_normal(shape[1:]),
        [0.5 / s for s in spacing[1:]],
    )
    radius = (z / 2) ** 2 + (y / 6) ** 2 + (x / 8) ** 2
    return radius < 1 + 0.15 * noise / noise.std()


@pytest.mark.usefixtures("no_host_trimesh")
@pytest.mark.parametrize("spacing", [(0.29, 0.108, 0.108), (0.5, 0.2, 0.1)])
@pytest.mark.parametrize("outline_fractal_scales", [None, (0.3, 1.2)])
def test_outline_fractal_dimension_stays_on_the_gpu(
    spacing: tuple, outline_fractal_scales, gpu_available: bool
) -> None:
    """Outline Fractal Dimension of a CuPy mask runs on the GPU and matches the CPU.

    The 2D blur rounds like SciPy and the ribbon comes from the marching cubes
    port, so only summation order differs.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    mask = _rough_flat_ellipsoid(spacing)
    name = "Outline Fractal Dimension"
    kwargs = {"spacing": spacing, "outline_fractal_scales": outline_fractal_scales}

    on_gpu = mesh.extract_surface_features(ascupy(mask), features=[name], **kwargs)
    on_cpu = mesh.extract_surface_features(mask, features=[name], **kwargs)

    assert np.isfinite(on_cpu[name])
    assert on_gpu[name] == pytest.approx(on_cpu[name], rel=1e-9)


@pytest.mark.usefixtures("no_host_trimesh")
@pytest.mark.parametrize("c", [1.0, 1e-8])
def test_extract_mesh_features_stays_on_the_gpu(c: float, gpu_available: bool) -> None:
    """The features computed on the device never build a host trimesh.

    ``extract_mesh_features`` measures scaled copies of both meshes; the copies
    are ArrayMeshes too, at an ordinary spacing and at one in metres.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    on_device = [
        "Area",
        "Volume",
        "Min Axis Length",
        "Med Axis Length",
        "Max Axis Length",
        "Inertia PC1",
        "Inertia PC2",
        "Inertia PC3",
        "Sphericity",
        "Euler Characteristic",
        "Avg Mean Curvature",
        "Total Absolute Gaussian Curvature",
        "Avg Shape Index",
        "Avg Curvedness",
    ]
    mask = ascupy(_labels() == 1)
    spacing = tuple(c * s for s in (0.3, 0.1, 0.1))
    surface = mesh.mask2mesh(mask, spacing=spacing)
    smooth = mesh.mask2mesh(mask, spacing=spacing, sigma=c * 0.31)

    values = mesh.extract_mesh_features(surface, on_device, smooth_mesh=smooth)

    assert isinstance(surface, ArrayMesh) and isinstance(smooth, ArrayMesh)
    assert all(np.isfinite(v) for v in values.values()), values


def test_mask2mesh_on_gpu(gpu_available: bool) -> None:
    """A CuPy mask is meshed on the GPU.

    Of skimage's keywords, only ``level``, ``spacing`` and ``allow_degenerate``
    are accepted there; others raise.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    mask = ascupy(_labels() == 1)

    on_gpu = mesh.mask2mesh(mask, spacing=(0.3, 0.1, 0.1))
    on_cpu = mesh.mask2mesh(asnumpy(mask), spacing=(0.3, 0.1, 0.1))

    assert isinstance(on_gpu, ArrayMesh)
    assert on_gpu.volume == pytest.approx(on_cpu.volume, rel=1e-12)
    with pytest.raises(ValueError, match="does not support"):
        mesh.mask2mesh(mask, marching_cubes_kwargs={"step_size": 2})


@pytest.mark.parametrize("spacing", [(-1.0, -1.0, -1.0), (0.0, 1.0, 1.0)])
@pytest.mark.parametrize("source", ["spacing", "marching_cubes_kwargs"])
def test_mask2mesh_on_gpu_rejects_a_spacing_that_is_not_positive(
    source: str, spacing: tuple, gpu_available: bool
) -> None:
    """A CuPy mask rejects such a spacing from either source, like a NumPy one."""
    if not gpu_available:
        pytest.skip("GPU not available")
    kwargs = {
        "spacing": {"spacing": spacing},
        "marching_cubes_kwargs": {"marching_cubes_kwargs": {"spacing": spacing}},
    }[source]

    with pytest.raises(ValueError, match=_SPACING_ERROR):
        mesh.mask2mesh(ascupy(_labels() == 1), **kwargs)
