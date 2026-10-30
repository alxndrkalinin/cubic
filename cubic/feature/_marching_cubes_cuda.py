"""Lewiner marching cubes on the GPU, matching ``skimage.measure.marching_cubes``.

A port of scikit-image's Cython implementation (Lewiner et al., 2003) to CUDA,
run on scikit-image's own lookup tables. The serial scan of the reference
shares each edge vertex between the cubes around it; here the work is split
into three passes over all cubes at once:

1. classify every cube, pick its triangulation with the same face and interior
   ambiguity tests, and mark the edges (and cube centres) it uses;
2. number the marked edges and centres with a prefix sum and compute their
   vertices;
3. write each cube's triangles as vertex numbers.

An edge vertex depends only on the two corner values of its edge: the
reference's centre-of-mass interpolation sums the same two terms whichever
cube creates the vertex, and addition is commutative. With the corner values,
tests and interpolation kept in float64 in the reference's operation order,
and ``--fmad=false``, the vertices round to the same float32 values as on the
CPU. The mesh is the same up to the order of vertices and faces.
"""

import functools
from typing import Any
from collections.abc import Sequence

import numpy as np
from skimage.measure import _marching_cubes_lewiner_luts as _mcluts
from skimage.measure._marching_cubes_lewiner import _to_array

from ..cuda import CUDAManager

# Lookup tables used by the Lewiner case analysis (the classic table is not).
_TABLES = (
    "CASES",
    "TILING1",
    "TILING2",
    "TILING3_1",
    "TILING3_2",
    "TILING4_1",
    "TILING4_2",
    "TILING5",
    "TILING6_1_1",
    "TILING6_1_2",
    "TILING6_2",
    "TILING7_1",
    "TILING7_2",
    "TILING7_3",
    "TILING7_4_1",
    "TILING7_4_2",
    "TILING8",
    "TILING9",
    "TILING10_1_1",
    "TILING10_1_1_",
    "TILING10_1_2",
    "TILING10_2",
    "TILING10_2_",
    "TILING11",
    "TILING12_1_1",
    "TILING12_1_1_",
    "TILING12_1_2",
    "TILING12_2",
    "TILING12_2_",
    "TILING13_1",
    "TILING13_1_",
    "TILING13_2",
    "TILING13_2_",
    "TILING13_3",
    "TILING13_3_",
    "TILING13_4",
    "TILING13_5_1",
    "TILING13_5_2",
    "TILING14",
    "TEST3",
    "TEST4",
    "TEST6",
    "TEST7",
    "TEST10",
    "TEST12",
    "TEST13",
    "SUBCONFIG13",
)

_KERNELS = r"""
// FLT_EPSILON of the reference is np.spacing(1.0), the double epsilon.
#define EPS 2.220446049250313e-16
#define T1D(N, i) (OFF_##N + (i) * L1_##N)
#define T2D(N, i, j) (OFF_##N + (i) * L1_##N * L2_##N + (j) * L2_##N)
#define G1(N, i) ((int)L[OFF_##N + (i)])
#define G2(N, i, j) ((int)L[OFF_##N + (i) * L1_##N + (j)])

__device__ __forceinline__ double dabs(double a) { return a >= 0 ? a : -a; }

// Corner values v0..v7 are ordered as in the reference's set_cube.
__device__ int test_face(const double* v, int face) {
    int abs_face = face < 0 ? -face : face;
    double A = 0, B = 0, C = 0, D = 0;
    switch (abs_face) {
        case 1: A = v[0]; B = v[4]; C = v[5]; D = v[1]; break;
        case 2: A = v[1]; B = v[5]; C = v[6]; D = v[2]; break;
        case 3: A = v[2]; B = v[6]; C = v[7]; D = v[3]; break;
        case 4: A = v[3]; B = v[7]; C = v[4]; D = v[0]; break;
        case 5: A = v[0]; B = v[3]; C = v[2]; D = v[1]; break;
        case 6: A = v[4]; B = v[7]; C = v[6]; D = v[5]; break;
    }
    double ac_bd = A * C - B * D;
    if (ac_bd > -EPS && ac_bd < EPS) return face >= 0;
    return face * A * ac_bd >= 0;
}

__device__ int test_internal(const double* v, const signed char* L, int cas,
                             int config, int subconfig, int s) {
    double t, At = 0, Bt = 0, Ct = 0, Dt = 0, a, b;
    int test = 0;
    int edge = -1;
    if (cas == 4 || cas == 10) {
        a = (v[4] - v[0]) * (v[6] - v[2]) - (v[7] - v[3]) * (v[5] - v[1]);
        b = v[2] * (v[4] - v[0]) + v[0] * (v[6] - v[2]) - v[1] * (v[7] - v[3])
            - v[3] * (v[5] - v[1]);
        t = -b / (2 * a + EPS);
        if (t < 0 || t > 1) return s > 0;
        At = v[0] + (v[4] - v[0]) * t;
        Bt = v[3] + (v[7] - v[3]) * t;
        Ct = v[2] + (v[6] - v[2]) * t;
        Dt = v[1] + (v[5] - v[1]) * t;
    } else if (cas == 6 || cas == 7 || cas == 12 || cas == 13) {
        if (cas == 6) edge = G2(TEST6, config, 2);
        else if (cas == 7) edge = G2(TEST7, config, 4);
        else if (cas == 12) edge = G2(TEST12, config, 3);
        else edge = (int)L[T2D(TILING13_5_1, config, subconfig)];
        switch (edge) {
            case 0: t = v[0] / (v[0] - v[1] + EPS); At = 0;
                Bt = v[3] + (v[2] - v[3]) * t; Ct = v[7] + (v[6] - v[7]) * t;
                Dt = v[4] + (v[5] - v[4]) * t; break;
            case 1: t = v[1] / (v[1] - v[2] + EPS); At = 0;
                Bt = v[0] + (v[3] - v[0]) * t; Ct = v[4] + (v[7] - v[4]) * t;
                Dt = v[5] + (v[6] - v[5]) * t; break;
            case 2: t = v[2] / (v[2] - v[3] + EPS); At = 0;
                Bt = v[1] + (v[0] - v[1]) * t; Ct = v[5] + (v[4] - v[5]) * t;
                Dt = v[6] + (v[7] - v[6]) * t; break;
            case 3: t = v[3] / (v[3] - v[0] + EPS); At = 0;
                Bt = v[2] + (v[1] - v[2]) * t; Ct = v[6] + (v[5] - v[6]) * t;
                Dt = v[7] + (v[4] - v[7]) * t; break;
            case 4: t = v[4] / (v[4] - v[5] + EPS); At = 0;
                Bt = v[7] + (v[6] - v[7]) * t; Ct = v[3] + (v[2] - v[3]) * t;
                Dt = v[0] + (v[1] - v[0]) * t; break;
            case 5: t = v[5] / (v[5] - v[6] + EPS); At = 0;
                Bt = v[4] + (v[7] - v[4]) * t; Ct = v[0] + (v[3] - v[0]) * t;
                Dt = v[1] + (v[2] - v[1]) * t; break;
            case 6: t = v[6] / (v[6] - v[7] + EPS); At = 0;
                Bt = v[5] + (v[4] - v[5]) * t; Ct = v[1] + (v[0] - v[1]) * t;
                Dt = v[2] + (v[3] - v[2]) * t; break;
            case 7: t = v[7] / (v[7] - v[4] + EPS); At = 0;
                Bt = v[6] + (v[5] - v[6]) * t; Ct = v[2] + (v[1] - v[2]) * t;
                Dt = v[3] + (v[0] - v[3]) * t; break;
            case 8: t = v[0] / (v[0] - v[4] + EPS); At = 0;
                Bt = v[3] + (v[7] - v[3]) * t; Ct = v[2] + (v[6] - v[2]) * t;
                Dt = v[1] + (v[5] - v[1]) * t; break;
            case 9: t = v[1] / (v[1] - v[5] + EPS); At = 0;
                Bt = v[0] + (v[4] - v[0]) * t; Ct = v[3] + (v[7] - v[3]) * t;
                Dt = v[2] + (v[6] - v[2]) * t; break;
            case 10: t = v[2] / (v[2] - v[6] + EPS); At = 0;
                Bt = v[1] + (v[5] - v[1]) * t; Ct = v[0] + (v[4] - v[0]) * t;
                Dt = v[3] + (v[7] - v[3]) * t; break;
            case 11: t = v[3] / (v[3] - v[7] + EPS); At = 0;
                Bt = v[2] + (v[6] - v[2]) * t; Ct = v[1] + (v[5] - v[1]) * t;
                Dt = v[0] + (v[4] - v[0]) * t; break;
        }
    }
    if (At >= 0) test += 1;
    if (Bt >= 0) test += 2;
    if (Ct >= 0) test += 4;
    if (Dt >= 0) test += 8;
    switch (test) {
        case 5: if (At * Ct - Bt * Dt < EPS) return s > 0; return 0;
        case 10: if (At * Ct - Bt * Dt >= EPS) return s > 0; return 0;
        case 7: case 11: case 13: case 14: case 15: return s < 0;
        default: return s > 0;
    }
}

// Offset into L of the cube's triangle list, and its triangle count *nt.
__device__ int big_switch(const double* v, const signed char* L, int cas,
                          int config, int* nt) {
    int sub = 0;
    switch (cas) {
        case 1: *nt = 1; return T1D(TILING1, config);
        case 2: *nt = 2; return T1D(TILING2, config);
        case 3:
            if (test_face(v, G1(TEST3, config))) { *nt = 4; return T1D(TILING3_2, config); }
            *nt = 2; return T1D(TILING3_1, config);
        case 4:
            if (test_internal(v, L, cas, config, sub, G1(TEST4, config))) {
                *nt = 2; return T1D(TILING4_1, config);
            }
            *nt = 6; return T1D(TILING4_2, config);
        case 5: *nt = 3; return T1D(TILING5, config);
        case 6:
            if (test_face(v, G2(TEST6, config, 0))) { *nt = 5; return T1D(TILING6_2, config); }
            if (test_internal(v, L, cas, config, sub, G2(TEST6, config, 1))) {
                *nt = 3; return T1D(TILING6_1_1, config);
            }
            *nt = 9; return T1D(TILING6_1_2, config);
        case 7:
            if (test_face(v, G2(TEST7, config, 0))) sub += 1;
            if (test_face(v, G2(TEST7, config, 1))) sub += 2;
            if (test_face(v, G2(TEST7, config, 2))) sub += 4;
            switch (sub) {
                case 0: *nt = 3; return T1D(TILING7_1, config);
                case 1: *nt = 5; return T2D(TILING7_2, config, 0);
                case 2: *nt = 5; return T2D(TILING7_2, config, 1);
                case 3: *nt = 9; return T2D(TILING7_3, config, 0);
                case 4: *nt = 5; return T2D(TILING7_2, config, 2);
                case 5: *nt = 9; return T2D(TILING7_3, config, 1);
                case 6: *nt = 9; return T2D(TILING7_3, config, 2);
                default:
                    if (test_internal(v, L, cas, config, sub, G2(TEST7, config, 3))) {
                        *nt = 9; return T1D(TILING7_4_2, config);
                    }
                    *nt = 5; return T1D(TILING7_4_1, config);
            }
        case 8: *nt = 2; return T1D(TILING8, config);
        case 9: *nt = 4; return T1D(TILING9, config);
        case 10:
            if (test_face(v, G2(TEST10, config, 0))) {
                if (test_face(v, G2(TEST10, config, 1))) { *nt = 4; return T1D(TILING10_1_1_, config); }
                *nt = 8; return T1D(TILING10_2, config);
            }
            if (test_face(v, G2(TEST10, config, 1))) { *nt = 8; return T1D(TILING10_2_, config); }
            if (test_internal(v, L, cas, config, sub, G2(TEST10, config, 2))) {
                *nt = 4; return T1D(TILING10_1_1, config);
            }
            *nt = 8; return T1D(TILING10_1_2, config);
        case 11: *nt = 4; return T1D(TILING11, config);
        case 12:
            if (test_face(v, G2(TEST12, config, 0))) {
                if (test_face(v, G2(TEST12, config, 1))) { *nt = 4; return T1D(TILING12_1_1_, config); }
                *nt = 8; return T1D(TILING12_2, config);
            }
            if (test_face(v, G2(TEST12, config, 1))) { *nt = 8; return T1D(TILING12_2_, config); }
            if (test_internal(v, L, cas, config, sub, G2(TEST12, config, 2))) {
                *nt = 4; return T1D(TILING12_1_1, config);
            }
            *nt = 8; return T1D(TILING12_1_2, config);
        case 13:
            for (int k = 0; k < 6; ++k) {
                if (test_face(v, G2(TEST13, config, k))) sub += 1 << k;
            }
            sub = G1(SUBCONFIG13, sub);
            if (sub == 0) { *nt = 4; return T1D(TILING13_1, config); }
            if (sub >= 1 && sub <= 6) { *nt = 6; return T2D(TILING13_2, config, sub - 1); }
            if (sub >= 7 && sub <= 18) { *nt = 10; return T2D(TILING13_3, config, sub - 7); }
            if (sub >= 19 && sub <= 22) { *nt = 12; return T2D(TILING13_4, config, sub - 19); }
            if (sub >= 23 && sub <= 26) {
                sub -= 23;
                if (test_internal(v, L, cas, config, sub, G2(TEST13, config, 6))) {
                    *nt = 6; return T2D(TILING13_5_1, config, sub);
                }
                *nt = 10; return T2D(TILING13_5_2, config, sub);
            }
            if (sub >= 27 && sub <= 38) { *nt = 10; return T2D(TILING13_3_, config, sub - 27); }
            if (sub >= 39 && sub <= 44) { *nt = 6; return T2D(TILING13_2_, config, sub - 39); }
            if (sub == 45) { *nt = 4; return T1D(TILING13_1_, config); }
            *nt = 0; return 0;
        case 14: *nt = 4; return T1D(TILING14, config);
    }
    *nt = 0;
    return 0;
}

// Local cube edge -> (axis, dx, dy, dz) of the grid edge it lies on; the
// reference's face-layer slots (get_index_in_facelayer) in grid terms.
__constant__ int EDGE_AXIS[12] = {0, 1, 0, 1, 0, 1, 0, 1, 2, 2, 2, 2};
__constant__ int EDGE_DX[12] = {0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0};
__constant__ int EDGE_DY[12] = {0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1};
__constant__ int EDGE_DZ[12] = {0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0};

__device__ __forceinline__ void load_cube(const float* vol, long long c, int nx,
                                          int ny, int ncx, int ncy, double level,
                                          int* x, int* y, int* z, double* v) {
    *x = (int)(c % ncx);
    *y = (int)((c / ncx) % ncy);
    *z = (int)(c / ((long long)ncx * ncy));
    long long sy = nx, sz = (long long)nx * ny;
    long long o = *z * sz + *y * sy + *x;
    v[0] = (double)vol[o] - level;
    v[1] = (double)vol[o + 1] - level;
    v[2] = (double)vol[o + sy + 1] - level;
    v[3] = (double)vol[o + sy] - level;
    v[4] = (double)vol[o + sz] - level;
    v[5] = (double)vol[o + sz + 1] - level;
    v[6] = (double)vol[o + sz + sy + 1] - level;
    v[7] = (double)vol[o + sz + sy] - level;
}

__device__ __forceinline__ int cube_index(const double* v) {
    int index = 0;
    for (int k = 0; k < 8; ++k) {
        if (v[k] > 0.0) index += 1 << k;
    }
    return index;
}

extern "C" __global__ void mc_classify(
    const float* vol, int nz, int ny, int nx, double level, const signed char* L,
    int* tri_base, int* tri_count, unsigned char* edge_used,
    unsigned char* center_used) {
    int ncx = nx - 1, ncy = ny - 1, ncz = nz - 1;
    long long ncubes = (long long)ncx * ncy * ncz;
    long long n = (long long)nx * ny * nz;
    for (long long c = blockIdx.x * (long long)blockDim.x + threadIdx.x; c < ncubes;
         c += (long long)gridDim.x * blockDim.x) {
        int x, y, z;
        double v[8];
        load_cube(vol, c, nx, ny, ncx, ncy, level, &x, &y, &z, v);
        int index = cube_index(v);
        int cas = G2(CASES, index, 0);
        int nt = 0;
        int base = 0;
        if (cas > 0) {
            base = big_switch(v, L, cas, G2(CASES, index, 1), &nt);
        }
        tri_base[c] = base;
        tri_count[c] = nt;
        for (int k = 0; k < 3 * nt; ++k) {
            int e = (int)L[base + k];
            if (e == 12) {
                center_used[c] = 1;
            } else {
                long long g = (long long)(z + EDGE_DZ[e]) * nx * ny
                              + (long long)(y + EDGE_DY[e]) * nx + (x + EDGE_DX[e]);
                edge_used[EDGE_AXIS[e] * n + g] = 1;
            }
        }
    }
}

// Edge vertex = edge origin + t_far / (t_near + t_far) along the edge, with
// t = 1 / (EPS + |v|): the reference's interpolation reduced to its nonzero
// terms. Written in (z, y, x) order, like skimage's output.
extern "C" __global__ void mc_edge_vertices(
    const float* vol, int nz, int ny, int nx, double level,
    const unsigned char* edge_used, const int* edge_id, float* verts) {
    long long n = (long long)nx * ny * nz;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < 3 * n;
         i += (long long)gridDim.x * blockDim.x) {
        if (!edge_used[i]) continue;
        int axis = (int)(i / n);
        long long g = i - axis * n;
        int x = (int)(g % nx);
        int y = (int)((g / nx) % ny);
        int z = (int)(g / ((long long)nx * ny));
        long long step = axis == 0 ? 1 : (axis == 1 ? nx : (long long)nx * ny);
        double v_near = (double)vol[g] - level;
        double v_far = (double)vol[g + step] - level;
        double t_near = 1.0 / (EPS + dabs(v_near));
        double t_far = 1.0 / (EPS + dabs(v_far));
        double f = t_far / (t_near + t_far);
        double px = x, py = y, pz = z;
        if (axis == 0) px = x + f;
        else if (axis == 1) py = y + f;
        else pz = z + f;
        long long o = (long long)edge_id[i] * 3;
        verts[o + 0] = (float)pz;
        verts[o + 1] = (float)py;
        verts[o + 2] = (float)px;
    }
}

// Cube-centre vertex: the reference's calculate_center_vertex, with its
// summation order.
extern "C" __global__ void mc_center_vertices(
    const float* vol, int nz, int ny, int nx, double level,
    const unsigned char* center_used, const int* center_id, float* verts) {
    int ncx = nx - 1, ncy = ny - 1, ncz = nz - 1;
    long long ncubes = (long long)ncx * ncy * ncz;
    for (long long c = blockIdx.x * (long long)blockDim.x + threadIdx.x; c < ncubes;
         c += (long long)gridDim.x * blockDim.x) {
        if (!center_used[c]) continue;
        int x, y, z;
        double v[8], w[8];
        load_cube(vol, c, nx, ny, ncx, ncy, level, &x, &y, &z, v);
        for (int k = 0; k < 8; ++k) w[k] = 1.0 / (EPS + dabs(v[k]));
        const double cx[8] = {0, 1, 1, 0, 0, 1, 1, 0};
        const double cy[8] = {0, 0, 1, 1, 0, 0, 1, 1};
        const double cz[8] = {0, 0, 0, 0, 1, 1, 1, 1};
        double fx = 0.0, fy = 0.0, fz = 0.0, ff = 0.0;
        for (int k = 0; k < 8; ++k) {
            fx += cx[k] * w[k];
            fy += cy[k] * w[k];
            fz += cz[k] * w[k];
            ff += w[k];
        }
        long long o = (long long)center_id[c] * 3;
        verts[o + 0] = (float)(z + fz / ff);
        verts[o + 1] = (float)(y + fy / ff);
        verts[o + 2] = (float)(x + fx / ff);
    }
}

// Triangles as vertex numbers, each reversed as skimage does for
// gradient_direction='descent'.
extern "C" __global__ void mc_faces(
    int nz, int ny, int nx, const signed char* L, const int* tri_base,
    const int* tri_count, const long long* face_offset, const int* edge_id,
    const int* center_id, int* faces) {
    int ncx = nx - 1, ncy = ny - 1, ncz = nz - 1;
    long long ncubes = (long long)ncx * ncy * ncz;
    long long n = (long long)nx * ny * nz;
    for (long long c = blockIdx.x * (long long)blockDim.x + threadIdx.x; c < ncubes;
         c += (long long)gridDim.x * blockDim.x) {
        int nt = tri_count[c];
        if (nt == 0) continue;
        int x = (int)(c % ncx);
        int y = (int)((c / ncx) % ncy);
        int z = (int)(c / ((long long)ncx * ncy));
        int base = tri_base[c];
        long long out = face_offset[c] * 3;
        for (int t = 0; t < nt; ++t) {
            for (int j = 0; j < 3; ++j) {
                int e = (int)L[base + 3 * t + j];
                int vid;
                if (e == 12) {
                    vid = center_id[c];
                } else {
                    long long g = (long long)(z + EDGE_DZ[e]) * nx * ny
                                  + (long long)(y + EDGE_DY[e]) * nx + (x + EDGE_DX[e]);
                    vid = edge_id[EDGE_AXIS[e] * n + g];
                }
                faces[out + 3 * t + (2 - j)] = vid;
            }
        }
    }
}
"""


@functools.cache
def _lut_layout() -> tuple[np.ndarray, str]:
    """Return the concatenated int8 lookup tables and their offset/shape macros."""
    chunks, defines, offset = [], [], 0
    for name in _TABLES:
        table = np.ascontiguousarray(_to_array(getattr(_mcluts, name)), dtype=np.int8)
        shape = table.shape + (1,) * (3 - table.ndim)
        defines += [
            f"#define OFF_{name} {offset}",
            f"#define L1_{name} {shape[1]}",
            f"#define L2_{name} {shape[2]}",
        ]
        chunks.append(table.ravel())
        offset += table.size
    return np.concatenate(chunks), "\n".join(defines) + "\n"


def _cupy() -> Any:
    """Return the CuPy module, raising if it is unavailable."""
    cp = CUDAManager().get_cp()
    if cp is None:
        raise RuntimeError("CuPy is required for GPU marching cubes")
    return cp


@functools.cache
def _kernels(device_id: int) -> tuple[Any, ...]:
    """Compile the kernels once per device; return them and the device LUTs."""
    cp = _cupy()
    luts, defines = _lut_layout()
    with cp.cuda.Device(device_id):
        # --fmad=false keeps every multiply and add separately rounded, as in
        # the Cython reference.
        module = cp.RawModule(code=defines + _KERNELS, options=("--fmad=false",))
        names = ("mc_classify", "mc_edge_vertices", "mc_center_vertices", "mc_faces")
        return tuple(module.get_function(n) for n in names) + (cp.asarray(luts),)


def _grid(size: int, threads: int = 256) -> tuple[tuple[int], tuple[int]]:
    return (max(1, min((size + threads - 1) // threads, 65535 * 8)),), (threads,)


def marching_cubes(
    volume: Any,
    level: float,
    spacing: Sequence[float] = (1.0, 1.0, 1.0),
    allow_degenerate: bool = True,
) -> tuple[Any, Any]:
    """Return ``(verts, faces)`` of ``volume``'s ``level`` isosurface on the GPU.

    Matches ``skimage.measure.marching_cubes(volume, level, spacing=spacing,
    allow_degenerate=allow_degenerate)`` with the default Lewiner method and
    ``gradient_direction='descent'``, including the output dtypes: float32
    vertices with unit spacing or when degenerate faces are removed, float64
    otherwise. Vertex and face order differ.
    """
    cp = _cupy()
    vol = cp.ascontiguousarray(volume, dtype=cp.float32)
    if vol.ndim != 3 or min(vol.shape) < 2:
        raise ValueError("Input array must be a 3D array of at least 2x2x2.")
    level = float(level)
    if level < float(vol.min()) or level > float(vol.max()):
        raise ValueError("Surface level must be within volume data range.")
    classify, edge_vertices, center_vertices, write_faces, luts = _kernels(
        vol.device.id
    )
    nz, ny, nx = (int(s) for s in vol.shape)
    n = nz * ny * nx
    ncubes = (nz - 1) * (ny - 1) * (nx - 1)
    tri_base = cp.empty(ncubes, dtype=cp.int32)
    tri_count = cp.empty(ncubes, dtype=cp.int32)
    edge_used = cp.zeros(3 * n, dtype=cp.uint8)
    center_used = cp.zeros(ncubes, dtype=cp.uint8)
    args = (vol, np.int32(nz), np.int32(ny), np.int32(nx), np.float64(level))
    classify(*_grid(ncubes), args + (luts, tri_base, tri_count, edge_used, center_used))

    edge_id = cp.cumsum(edge_used, dtype=cp.int32) - 1
    n_edges = int(edge_id[-1]) + 1 if n else 0
    center_id = cp.cumsum(center_used, dtype=cp.int32) - 1 + n_edges
    n_verts = int(center_id[-1]) + 1 if ncubes else n_edges
    face_end = cp.cumsum(tri_count, dtype=cp.int64)
    n_faces = int(face_end[-1]) if ncubes else 0
    if n_verts == 0 or n_faces == 0:
        raise RuntimeError("No surface found at the given iso value.")

    verts = cp.empty((n_verts, 3), dtype=cp.float32)
    edge_vertices(*_grid(3 * n), args + (edge_used, edge_id, verts))
    center_vertices(*_grid(ncubes), args + (center_used, center_id, verts))
    faces = cp.empty((n_faces, 3), dtype=cp.int32)
    write_faces(
        *_grid(ncubes),
        (
            np.int32(nz),
            np.int32(ny),
            np.int32(nx),
            luts,
            tri_base,
            tri_count,
            face_end - tri_count,
            edge_id,
            center_id,
            faces,
        ),
    )

    step = tuple(float(s) for s in spacing)
    if step != (1.0, 1.0, 1.0):
        verts = verts.astype(cp.float64) * cp.asarray(step, dtype=cp.float64)
    if not allow_degenerate:
        verts, faces = _remove_degenerate_faces(verts.astype(cp.float32), faces)
    return verts, faces


def _remove_degenerate_faces(verts: Any, faces: Any) -> tuple[Any, Any]:
    """Merge coincident vertices and drop the faces that collapse.

    skimage merges the vertices of degenerate faces and trimesh then merges any
    remaining coincident ones; merging every exact duplicate does both.
    """
    cp = _cupy()
    # A lexicographic sort brings duplicates together; cp.unique(axis=0) does
    # the same but took hours on a few thousand vertices.
    order = cp.lexsort(verts.T[::-1])
    ordered = verts[order]
    first = cp.ones(len(ordered), dtype=bool)
    first[1:] = (ordered[1:] != ordered[:-1]).any(axis=1)
    inverse = cp.empty(len(verts), dtype=cp.int32)
    inverse[order] = (cp.cumsum(first) - 1).astype(cp.int32)
    unique = ordered[first]
    faces = inverse[faces]
    keep = (
        (faces[:, 0] != faces[:, 1])
        & (faces[:, 1] != faces[:, 2])
        & (faces[:, 0] != faces[:, 2])
    )
    return unique, faces[keep]
