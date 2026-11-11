"""A closed triangle mesh whose arrays may live on the CPU or the GPU.

``ArrayMesh`` exposes the ``trimesh.Trimesh`` attributes that the mesh feature
registry reads, computed with the same formulas as trimesh from NumPy or CuPy
arrays, so the features run on whichever device holds the mesh. Anything it
does not implement (convex hull, bounding primitives) is delegated
to a ``trimesh.Trimesh`` built on the host on first use.
"""

from __future__ import annotations

from typing import Any
from functools import wraps, cached_property

import numpy as np

from ..cuda import asnumpy

# trimesh.constants.tol: zero for unitize, merge for angles and convexity.
_TOL_ZERO = 1e-13
_TOL_MERGE = 1e-8


def _unitize(vectors: Any) -> tuple[Any, Any]:
    """Row-normalize ``vectors`` as ``trimesh.util.unitize`` does.

    Rows with norm at or below the tolerance keep their direction scaled by
    their norm, which is what trimesh's in-place power leaves them with.
    """
    norm = np.sqrt((vectors * vectors).sum(axis=1))
    valid = norm > _TOL_ZERO
    scale = np.where(valid, 1 / np.where(valid, norm, 1), norm)
    return vectors * scale[:, None], valid


def _group_rows(keys: Any) -> tuple[Any, Any]:
    """Sort the rows of ``keys`` and mark the first row of each run of equal rows.

    Returns ``(order, is_new)``: ``keys[order]`` is sorted lexicographically
    and ``is_new[i]`` is True where row ``i`` of it differs from the row before,
    so ``np.cumsum(is_new) - 1`` numbers the groups. A lexsort and a neighbour
    comparison instead of ``unique(axis=0)``: ``cupy.unique(axis=0)`` took about
    7 s per 1,000 rows, against 1.5 ms per 300,000 rows for the lexsort.
    """
    order = np.lexsort(keys.T[::-1])
    sorted_keys = keys[order]
    is_new = np.ones_like(order, dtype=bool)
    is_new[1:] = (sorted_keys[1:] != sorted_keys[:-1]).any(axis=1)
    return order, is_new


def _merge_vertices(vertices: Any, faces: Any) -> tuple[Any, Any]:
    """Drop unreferenced vertices and merge equal ones, as trimesh's processing does.

    Marching cubes with degenerate faces removed leaves both after an exact level
    hit. trimesh merges them on the CPU path (``process=True``); without this,
    each unreferenced vertex adds a ``2 pi`` defect (0.5 to total |K|) and each
    duplicate splits defects and edges at a seam. Vertices equal to trimesh's
    merge tolerance (8 decimals) are merged. The arrays come back unchanged when
    there is nothing to merge.
    """
    referenced = np.zeros_like(vertices[:, 0], dtype=bool)
    referenced[faces.ravel()] = True
    index = np.flatnonzero(referenced)
    # trimesh.grouping.merge_vertices: vertices * 10**digits(tol.merge), rounded;
    # dividing by 1e-8 instead rounds a few vertices into other groups
    keys = np.round(vertices[index] * 1e8).astype(np.int64)
    order, new = _group_rows(keys)
    if len(index) == len(vertices) and bool(new.all()):
        return vertices, faces
    remap = np.zeros_like(faces, shape=len(vertices))
    remap[index[order]] = np.cumsum(new) - 1
    return vertices[index[order[new]]], remap[faces]


def _triangles_cross(triangles: Any) -> Any:
    """Cross product of each triangle's edges, in trimesh's operand order."""
    vectors = triangles[:, 1:, :] - triangles[:, :2, :]
    return np.cross(vectors[:, 0], vectors[:, 1])


def face_areas(vertices: Any, faces: Any) -> Any:
    """Area of each triangle as trimesh computes it, on the arrays' device.

    Needs neither merging nor orientation, so it is cheaper than an ArrayMesh.
    """
    return np.sqrt((_triangles_cross(vertices[faces]) ** 2).sum(axis=1)) / 2.0


class ArrayMesh:
    """Closed, consistently oriented triangle mesh on the device of its arrays.

    Parameters
    ----------
    vertices : array
        ``(V, 3)`` vertex coordinates, NumPy or CuPy.
    faces : array
        ``(F, 3)`` vertex indices on the same device.

    The faces are flipped if the enclosed volume comes out negative, as
    ``trimesh.repair.fix_normals`` does for a mesh with consistent winding, and
    equal and unreferenced vertices are merged as trimesh's processing does.

    Attributes not computed here are read from a ``trimesh.Trimesh`` copy on the
    host. A delegated method that would change the mesh in place (e.g.
    ``apply_scale``) raises ``TypeError``: the device arrays and cached
    quantities would no longer describe it. Use ``to_trimesh()`` for a mutable
    mesh.
    """

    def __init__(self, vertices: Any, faces: Any) -> None:
        self.vertices, self.faces = _merge_vertices(
            vertices.astype(np.float64), faces.astype(np.int64)
        )
        if self.volume < 0:
            self.faces = self.faces[:, ::-1].copy()
            for name in list(vars(self)):
                if name not in ("vertices", "faces"):
                    delattr(self, name)

    def __getattr__(self, name: str) -> Any:
        # Called only for attributes not defined here: delegate to trimesh.
        if name.startswith("_"):
            raise AttributeError(name)
        host = self.to_trimesh()
        value = getattr(host, name)
        if not callable(value):
            return value

        @wraps(value)
        def guarded(*args: Any, **kwargs: Any) -> Any:
            # trimesh's hash covers vertices and faces, not its caches
            before = hash(host)
            result = value(*args, **kwargs)
            if hash(host) != before:
                del self._host  # rebuilt from the device arrays on next use
                raise TypeError(
                    f"ArrayMesh.{name} would change the mesh in place, which "
                    "ArrayMesh does not support; call to_trimesh() for a mutable "
                    "trimesh.Trimesh."
                )
            return result

        return guarded

    def to_trimesh(self) -> Any:
        """Return (and cache) a ``trimesh.Trimesh`` of this mesh on the host."""
        if "_host" not in vars(self):
            import trimesh

            self._host = trimesh.Trimesh(
                asnumpy(self.vertices), asnumpy(self.faces), process=False
            )
        return self._host

    # -- per-face geometry (trimesh.triangles) --------------------------------

    @cached_property
    def triangles(self) -> Any:
        return self.vertices[self.faces]

    @cached_property
    def triangles_cross(self) -> Any:
        return _triangles_cross(self.triangles)

    @cached_property
    def area_faces(self) -> Any:
        return np.sqrt((self.triangles_cross**2).sum(axis=1)) / 2.0

    @cached_property
    def area(self) -> float:
        return float(self.area_faces.sum())

    @cached_property
    def face_normals(self) -> Any:
        unit, valid = _unitize(self.triangles_cross)
        return np.where(valid[:, None], unit, 0.0)

    @cached_property
    def face_angles(self) -> Any:
        tri = self.triangles
        u, _ = _unitize(tri[:, 1] - tri[:, 0])
        v, _ = _unitize(tri[:, 2] - tri[:, 0])
        w, _ = _unitize(tri[:, 2] - tri[:, 1])
        a0 = np.arccos(np.clip((u * v).sum(axis=1), -1, 1))
        a1 = np.arccos(np.clip((-u * w).sum(axis=1), -1, 1))
        angles = np.stack([a0, a1, np.pi - a0 - a1], axis=1)
        return np.where((angles < _TOL_MERGE).any(axis=1, keepdims=True), 0.0, angles)

    @cached_property
    def vertex_defects(self) -> Any:
        angle_sum = np.bincount(
            self.faces.ravel(),
            weights=self.face_angles.ravel(),
            minlength=len(self.vertices),
        )
        return 2 * np.pi - angle_sum

    # -- mass properties (trimesh.triangles.mass_properties, density 1) -------

    @cached_property
    def _integrated(self) -> np.ndarray:
        tri, crosses = self.triangles, self.triangles_cross
        t0, t1, t2 = tri[:, 0, :], tri[:, 1, :], tri[:, 2, :]
        f1 = t0 + t1 + t2
        f2 = t0**2 + t1**2 + t0 * t1 + t2 * f1
        f3 = t0**3 + t0**2 * t1 + t0 * t1**2 + t1**3 + t2 * f2
        g0 = f2 + (t0 + f1) * t0
        g1 = f2 + (t1 + f1) * t1
        g2 = f2 + (t2 + f1) * t2
        terms = [crosses[:, 0] * f1[:, 0]]
        terms += [(crosses * f2)[:, i] for i in range(3)]
        terms += [(crosses * f3)[:, i] for i in range(3)]
        for i in range(3):
            j = (i + 1) % 3
            terms.append(
                crosses[:, i]
                * (t0[:, j] * g0[:, i] + t1[:, j] * g1[:, i] + t2[:, j] * g2[:, i])
            )
        sums = asnumpy(np.stack(terms).sum(axis=1))
        return sums / np.array(
            [6, 24, 24, 24, 60, 60, 60, 120, 120, 120], dtype=np.float64
        )

    @cached_property
    def volume(self) -> float:
        return float(self._integrated[0])

    @property
    def mass(self) -> float:
        return self.volume

    @cached_property
    def center_mass(self) -> np.ndarray:
        return self._integrated[1:4] / self.volume

    @cached_property
    def moment_inertia(self) -> np.ndarray:
        s, v, c = self._integrated, self.volume, self.center_mass
        inertia = np.zeros((3, 3))
        inertia[0, 0] = s[5] + s[6] - v * (c[[1, 2]] ** 2).sum()
        inertia[1, 1] = s[4] + s[6] - v * (c[[0, 2]] ** 2).sum()
        inertia[2, 2] = s[4] + s[5] - v * (c[[0, 1]] ** 2).sum()
        inertia[0, 1] = inertia[1, 0] = -(s[7] - v * np.prod(c[[0, 1]]))
        inertia[1, 2] = inertia[2, 1] = -(s[8] - v * np.prod(c[[1, 2]]))
        inertia[0, 2] = inertia[2, 0] = -(s[9] - v * np.prod(c[[0, 2]]))
        return inertia

    @cached_property
    def principal_inertia_components(self) -> np.ndarray:
        return np.linalg.eigh(self.moment_inertia)[0]

    # -- edges and adjacency (trimesh.graph) ----------------------------------

    @cached_property
    def _edge_groups(self) -> tuple[Any, Any, Any, Any]:
        """Edges grouped by vertex pair: (edges, order, start of each run, run length).

        ``edges`` holds each face's three edges with sorted vertex indices, in
        face order; ``order`` sorts them so that equal edges form runs.
        """
        edges = np.sort(self.faces[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
        order, is_start = _group_rows(edges)
        starts = np.flatnonzero(is_start)
        lengths = np.diff(starts, append=len(edges))
        return edges, order, starts, lengths

    @cached_property
    def euler_number(self) -> int:
        referenced = int(np.unique(self.faces).size)
        _, _, starts, _ = self._edge_groups
        return referenced - int(starts.size) + len(self.faces)

    @cached_property
    def _face_adjacency(self) -> tuple[Any, Any]:
        """Pairs of faces sharing an edge, and that edge: (pairs, edges)."""
        edges, order, starts, lengths = self._edge_groups
        first = starts[lengths == 2]
        edge_a, edge_b = order[first], order[first + 1]
        pair = np.stack([edge_a // 3, edge_b // 3], axis=1)
        keep = pair[:, 0] != pair[:, 1]
        return np.sort(pair[keep], axis=1), edges[edge_a[keep]]

    @cached_property
    def face_adjacency(self) -> Any:
        return self._face_adjacency[0]

    @cached_property
    def face_adjacency_edges(self) -> Any:
        return self._face_adjacency[1]

    @cached_property
    def face_adjacency_angles(self) -> Any:
        normals = self.face_normals[self.face_adjacency]
        dots = np.clip((normals[:, 0] * normals[:, 1]).sum(axis=1), -1.0, 1.0)
        return np.abs(np.arccos(dots))

    @cached_property
    def face_adjacency_unshared(self) -> Any:
        edges = self.face_adjacency_edges
        out = []
        for side in range(2):
            faces = self.faces[self.face_adjacency[:, side]]
            unshared = (faces != edges[:, :1]) & (faces != edges[:, 1:])
            row_ok = unshared.sum(axis=1) == 1
            column = np.argmax(unshared, axis=1)[:, None]
            picked = np.take_along_axis(faces, column, axis=1)[:, 0]
            out.append(np.where(row_ok, picked, -1))
        return np.stack(out, axis=1)

    @cached_property
    def face_adjacency_convex(self) -> Any:
        normals = self.face_normals[self.face_adjacency[:, 0]]
        origins = self.vertices[self.face_adjacency_edges[:, 0]]
        other = self.vertices[self.face_adjacency_unshared[:, 1]] - origins
        return (other * normals).sum(axis=1) < _TOL_MERGE

    @cached_property
    def integral_mean_curvature(self) -> float:
        edges = self.face_adjacency_edges
        lengths = np.linalg.norm(
            self.vertices[edges[:, 0]] - self.vertices[edges[:, 1]], axis=1
        )
        signs = np.where(self.face_adjacency_convex, 1.0, -1.0)
        return float((self.face_adjacency_angles * signs * lengths).sum() * 0.5)
