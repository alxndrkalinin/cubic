"""Mesh features convert with the unit of ``spacing``, on the CPU and the GPU."""

import re
import warnings
from typing import Any
from functools import cache

import numpy as np
import pytest
from skimage.measure import marching_cubes

trimesh = pytest.importorskip("trimesh")

from cubic.cuda import ascupy, asnumpy, to_same_device  # noqa: E402
from cubic.feature import mesh  # noqa: E402

_SPACING_ERROR = re.escape(
    "spacing must be three finite, positive numbers (z, y, x); got "
)

# Power of the length unit in each feature, measured independently of the
# module's table: log2 of each feature's ratio between isotropic spacings s and
# 2 s (windows scaled with them) is this integer to 1e-5.
DIMENSIONS = {
    "Area": 2,
    "Volume": 3,
    "Min Axis Length": 1,
    "Med Axis Length": 1,
    "Max Axis Length": 1,
    "Inertia PC1": 5,
    "Inertia PC2": 5,
    "Inertia PC3": 5,
    "Oriented Bounding Box Volume": 3,
    "Bounding Cylinder Volume": 3,
    "Bounding Sphere Volume": 3,
    "Convex Hull Volume": 3,
    "Convex Hull Area": 2,
    "Sphericity": 0,
    "Extent": 0,
    "Solidity": 0,
    "Euler Characteristic": 0,
    "Avg Mean Curvature": -1,
    "Total Absolute Gaussian Curvature": 0,
    "Avg Shape Index": 0,
    "Avg Curvedness": -1,
    "Fractal Dimension": 0,
    "Outline Fractal Dimension": 0,
}

BASES = {
    "anisotropic": (2.685, 1.0, 1.0),
    # (0.2e-6 / 0.065e-6) is not 0.2 / 0.065 in floating point, one ulp apart.
    "unit rounding": (0.2, 0.065, 0.065),
}
# (factor, base, explicit windows) of each covariance check. For the
# anisotropic base and one window mode the internal problem is the same at
# every factor, so one factor other than 1 checks each mode; 1e-9 and 1e3 put
# the caller's lengths far below and above trimesh's absolute tolerances. The
# unit-rounding base is the exception that the last cell bounds.
CELLS = [
    (1e-9, "anisotropic", False),
    (1e-9, "anisotropic", True),
    (1e3, "anisotropic", False),
    (1e-6, "unit rounding", False),
]

# Features whose GPU value varies between runs of the same code (atomic sums).
_GPU_JITTER = ("Total Absolute Gaussian Curvature", "Avg Shape Index", "Avg Curvedness")


def _mask() -> np.ndarray:
    """Return a flat ellipsoid turned in plane, with a lobe on top.

    Its footprint is about 72 by 88 voxels, over 4.5 times the top of the default
    outline window (16 in-plane voxels), so ``Outline Fractal Dimension`` is
    finite, and it is 14 voxels thick, enough for the default 3D fractal window.
    """
    z, y, x = np.meshgrid(
        *[np.arange(n) - (n - 1) / 2 for n in (26, 96, 96)], indexing="ij"
    )
    turn = np.deg2rad(30)
    u = y * np.cos(turn) + x * np.sin(turn)
    v = -y * np.sin(turn) + x * np.cos(turn)
    body = (u / 44) ** 2 + (v / 36) ** 2 + (z / 7) ** 2 <= 1
    lobe = (z - 5) ** 2 + (y - 10) ** 2 + (x + 14) ** 2 <= 6**2
    return body | lobe


def _windows(c: float) -> dict[str, object]:
    """Explicit windows, in units of the base spacing times ``c``.

    3 sigma / step stays clear of integers (9.3, and 3.46 on the anisotropic
    axis), so an ulp in the normalised sigma cannot change the blur's padding.
    """
    return {
        "curvature_sigma": c * 3.1,
        "fractal_scales": (c * 1.0, c * 3.1),
        "outline_fractal_scales": (c * 2.0, c * 8.0),
    }


def _features(device: str, base: str, c: float, explicit: bool) -> dict[str, float]:
    """Every feature of ``_mask()`` at spacing ``c * base``, on ``device``."""
    mask = _mask() if device == "cpu" else ascupy(_mask())
    spacing = tuple(c * s for s in BASES[base])
    windows = _windows(c) if explicit else {}
    with warnings.catch_warnings():
        # Only the values are compared here.
        warnings.simplefilter("ignore")
        return mesh.extract_surface_features(
            mask,
            spacing=spacing,
            features=mesh.mesh_feature_list(optional=True),
            **windows,
        )


@cache
def _reference(device: str, base: str, explicit: bool) -> dict[str, float]:
    """``_features`` at ``c = 1``, computed once per combination."""
    return _features(device, base, 1.0, explicit)


def _assert_covariant(device: str, base: str, c: float, explicit: bool) -> None:
    reference = _reference(device, base, explicit)
    assert all(np.isfinite(v) for v in reference.values()), reference
    actual = _features(device, base, c, explicit)
    off = {}
    for name, value in actual.items():
        expected = c ** DIMENSIONS[name] * reference[name]
        if base == "unit rounding":
            rtol = 5e-3 if name == "Bounding Cylinder Volume" else 1e-6
        else:
            rtol = 1e-12 if device == "cpu" else 1e-9
        atol = 0.0
        if device == "gpu" and name in _GPU_JITTER:
            atol = 1e-7 * c ** DIMENSIONS[name]
        if not abs(value - expected) <= atol + rtol * abs(expected):
            off[name] = (value, expected, abs(value - expected) / abs(expected))
    assert not off, f"not covariant at c={c:g}: {off}"


def _cell_id(cell: tuple[float, str, bool]) -> str:
    c, base, explicit = cell
    return f"{c:g}-{base}-{'explicit' if explicit else 'default'}"


@pytest.mark.parametrize(("c", "base", "explicit"), CELLS, ids=map(_cell_id, CELLS))
def test_features_are_covariant_with_the_unit(
    c: float, base: str, explicit: bool
) -> None:
    """Each feature at spacing ``c * base`` is ``c ** dimension`` times its value.

    The reference is the value at ``base``; the tolerance is the rounding of the
    final multiplication. ``explicit`` scales ``curvature_sigma`` and both fractal windows with the
    spacing; the default windows follow the spacing by themselves.

    The ``unit rounding`` cell is looser: the ratio of its spacings rounds
    differently in the two units, so the internal problem differs by one ulp,
    which moved Avg Mean Curvature by 1e-8 relative; Bounding Cylinder Volume
    comes from an approximate search and moved by 4e-4.
    """
    _assert_covariant("cpu", base, c, explicit)


@pytest.mark.parametrize(("c", "base", "explicit"), CELLS, ids=map(_cell_id, CELLS))
def test_gpu_features_are_covariant_with_the_unit(
    c: float, base: str, explicit: bool, gpu_available: bool
) -> None:
    """The covariance of the CPU test, for a CuPy mask meshed on the GPU."""
    if not gpu_available:
        pytest.skip("GPU not available")
    _assert_covariant("gpu", base, c, explicit)


def test_every_feature_has_a_dimension() -> None:
    """The module's dimension table covers every feature, as measured here."""
    names = mesh.mesh_feature_list(optional=True)
    assert set(mesh._FEATURE_DIMENSIONS) == set(names)
    assert mesh._FEATURE_DIMENSIONS == DIMENSIONS


@pytest.mark.parametrize(
    "spacing",
    [(0.0, 1.0, 1.0), (-1.0, 1.0, 1.0), (-1.0, -1.0, -1.0), (np.nan, 1.0, 1.0)]
    + [(np.inf, 1.0, 1.0)]
    + [0.5, (1.0,), (1.0, 1.0), (1.0, 1.0, 1.0, 1.0), np.ones((2, 3))],
)
def test_spacing_must_be_three_finite_positive_numbers(
    spacing: float | tuple[float, ...] | np.ndarray,
) -> None:
    """Each entry rejects such a spacing, even for an image without labels.

    It used to fail deep in the blur, or for an all-negative spacing to return
    the features of the mirrored object, and an image without labels returned
    an empty table. A scalar failed with a TypeError, a spacing of another
    length with skimage's marching cubes error, and an image without labels
    accepted a spacing of any length.
    """
    ball = np.pad(np.ones((5, 5, 5), dtype=bool), 2)
    with pytest.raises(ValueError, match=_SPACING_ERROR):
        mesh.extract_surface_features(ball, spacing=spacing)
    with pytest.raises(ValueError, match=_SPACING_ERROR):
        mesh.extract_features(ball.astype(np.int32), spacing=spacing)
    with pytest.raises(ValueError, match=_SPACING_ERROR):
        mesh.extract_features(np.zeros((4, 4, 4), dtype=np.int32), spacing=spacing)


def _ball(radius: int) -> np.ndarray:
    grid = np.meshgrid(*[np.arange(2 * radius + 3) - radius - 1] * 3, indexing="ij")
    return sum(axis**2 for axis in grid) <= radius**2


@pytest.mark.parametrize(
    ("mask", "kwargs", "text"),
    [
        # Fractal Dimension's default window is empty below the finest spacing.
        (
            _ball(6),
            {"features": ["Fractal Dimension"], "curvature_sigma": 0.2},
            r"spacing 0\.25; got 0\.2\.",
        ),
        # A footprint 3.25 wide vanishes at the outline window's top scale of 2.
        (
            _ball(6),
            {
                "features": ["Outline Fractal Dimension"],
                "outline_fractal_scales": (0.5, 2.0),
            },
            r"scale up to 2 \(",
        ),
        # Two voxels across vanish at the default curvature_sigma, twice 0.25.
        (
            np.pad(np.ones((2, 2, 2), dtype=bool), 2),
            {"features": ["Avg Shape Index"]},
            r"curvature_sigma=0\.5;",
        ),
    ],
    ids=["fractal window", "outline vanishes", "blur vanishes"],
)
def test_warnings_print_lengths_in_the_callers_units(
    mask: np.ndarray, kwargs: dict[str, Any], text: str
) -> None:
    """Scales in warnings are the caller's, not those of the internal unit.

    At spacing 0.25 the internal unit is 0.25, so each scale would print four
    times larger.
    """
    with pytest.warns(UserWarning, match=text):
        mesh.extract_surface_features(mask, spacing=(0.25, 0.25, 0.25), **kwargs)


def test_volume_error_prints_the_callers_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The open-mesh error names the volume in the caller's units.

    ``mask2mesh`` is replaced by one that leaves the surface open, which the
    padding otherwise prevents.
    """

    def open_mask2mesh(
        mask: np.ndarray, spacing: tuple[float, ...] | None = None
    ) -> trimesh.Trimesh:
        verts, faces, _, _ = marching_cubes(mask, 0.5, spacing=spacing)
        return trimesh.Trimesh(vertices=verts, faces=faces)

    mask = np.zeros((6, 6, 6), dtype=bool)
    mask[:3, :3, :3] = True  # touches the border, so its surface is open
    spacing = (0.5, 0.25, 0.25)
    expected = open_mask2mesh(mask, spacing).volume
    monkeypatch.setattr(mesh, "mask2mesh", open_mask2mesh)

    with pytest.raises(ValueError, match="volume must be finite") as error:
        mesh.extract_surface_features(mask, spacing, ["Volume"])

    got = float(re.search(r"got (\S+)\. ", str(error.value)).group(1))
    assert got == pytest.approx(expected, rel=1e-12)


def _on(device: str, mask: np.ndarray, gpu_available: bool) -> Any:
    """``mask`` on ``device``; skips the test for the GPU without one."""
    if device == "cpu":
        return mask
    if not gpu_available:
        pytest.skip("GPU not available")
    return ascupy(mask)


def _arrays(surface: Any) -> tuple[np.ndarray, np.ndarray]:
    """Vertices and faces of a mesh of either type, on the host."""
    return np.asarray(asnumpy(surface.vertices)), np.asarray(asnumpy(surface.faces))


def _old_mask2mesh(
    mask: Any, spacing: tuple[float, ...] | None, sigma: float | None
) -> Any:
    """``mask2mesh`` as it was before it normalised the spacing, from its helpers."""
    if sigma is not None:
        return mesh._smooth_mask2mesh(mask, spacing, sigma)
    step = (1.0, 1.0, 1.0) if spacing is None else spacing
    verts, faces = mesh._marching_cubes(np.pad(mask, 1), spacing=step)
    one_voxel = np.asarray(step, dtype=float)
    return mesh._closed_mesh(verts - to_same_device(one_voxel, verts), faces)


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("spacing", [None, (1.0, 1.0, 1.0), (2.685, 1.0, 1.0)])
@pytest.mark.parametrize("branch", ["spacing", "marching_cubes_kwargs", "sigma"])
def test_mask2mesh_with_a_unit_finest_spacing_is_unchanged(
    device: str,
    spacing: tuple[float, ...] | None,
    branch: str,
    gpu_available: bool,
) -> None:
    """With a finest spacing of 1 (or none), the mesh is the one built before.

    ``extract_surface_features`` meshes at such a spacing, so its values must
    not move: the vertices and faces are bitwise those of the code before the
    spacing was normalised, rebuilt here from the same helpers.
    """
    mask = _on(device, _ball(10), gpu_available)
    sigma = 3.1 if branch == "sigma" else None
    kwargs: dict[str, Any] = {
        "spacing": {"spacing": spacing},
        "marching_cubes_kwargs": {"marching_cubes_kwargs": {"spacing": spacing}},
        "sigma": {"spacing": spacing, "sigma": sigma},
    }[branch]
    if branch == "marching_cubes_kwargs" and spacing is None:
        kwargs = {}

    got = _arrays(mesh.mask2mesh(mask, **kwargs))
    expected = _arrays(_old_mask2mesh(mask, spacing, sigma))

    for g, e in zip(got, expected, strict=True):
        assert g.dtype == e.dtype
        np.testing.assert_array_equal(g, e)


# Factors of the caller's unit, far below and above trimesh's absolute merge
# tolerance (1e-8), and 0.3, which is not a power of ten.
UNIT_FACTORS = [1e-9, 1e-8, 1e-7, 1e-6, 1e-3, 0.3, 1e3]
# Spacing and blur scale at factor 1 of each base. The anisotropic one is the
# unit-rounding base: (0.2 c) / (0.065 c) is not 0.2 / 0.065 at every c, so its
# normalised spacing differs by an ulp between factors. 3 sigma / step stays
# clear of integers (9.3, and 3.02 on the coarse axis), so that ulp cannot
# change the blur's padding.
MESH_BASES = {
    "isotropic": ((1.0, 1.0, 1.0), 3.1),
    "anisotropic": ((0.2, 0.065, 0.065), 0.2015),
}


@pytest.mark.parametrize("device", ["cpu", "gpu"])
@pytest.mark.parametrize("base", list(MESH_BASES))
@pytest.mark.parametrize("smooth", [False, True], ids=["plain", "sigma"])
@pytest.mark.parametrize("shape", ["ball", "ragged ball"])
def test_mask2mesh_is_covariant_with_the_unit(
    device: str,
    base: str,
    smooth: bool,
    shape: str,
    ragged_ball: np.ndarray,
    gpu_available: bool,
) -> None:
    """At spacing ``c * base``, the mesh is ``c`` times the mesh at ``base``.

    Same vertex and face count, watertightness and Euler number, the same faces,
    and vertices ``c`` times those at factor 1 to 4 float64 ulps of the
    coordinate before the pad was removed: the mesh is scaled once by its unit
    at each factor and the normalised spacing may differ by an ulp (measured
    up to 1.8). Unnormalised, the ball at spacing 1e-8 came out with 1,402 of
    its 1,896 vertices and Euler number 566, its vertices merged at trimesh's
    absolute tolerance.
    """
    mask = _on(device, _ball(10) if shape == "ball" else ragged_ball, gpu_available)
    spacing, sigma = MESH_BASES[base]
    step = np.asarray(spacing)

    def mesh_at(c: float) -> Any:
        kwargs = {"sigma": c * sigma} if smooth else {}
        return mesh.mask2mesh(mask, spacing=tuple(c * step), **kwargs)

    reference = mesh_at(1.0)
    ref_verts, ref_faces = _arrays(reference)
    pad = (np.ceil(3 * sigma / step) + 1) if smooth else 1
    padded = np.abs(ref_verts) + pad * step
    for c in UNIT_FACTORS:
        surface = mesh_at(c)
        verts, faces = _arrays(surface)
        topology = (len(verts), len(faces), surface.is_watertight)
        assert topology == (len(ref_verts), len(ref_faces), reference.is_watertight), c
        assert surface.euler_number == reference.euler_number, c
        np.testing.assert_array_equal(faces, ref_faces, err_msg=f"c={c:g}")
        off = np.abs(verts - c * ref_verts) / (c * np.finfo(float).eps * padded)
        assert off.max() <= 4, (c, off.max())
