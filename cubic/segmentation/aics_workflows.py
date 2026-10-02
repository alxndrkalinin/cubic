"""Device-agnostic Allen Cell Structure Segmenter classic workflows.

Ports of the ``aicssegmentation`` SEC61B (ER) and TOMM20 (mitochondria)
workflows and the primitives they share. The CPU reference runs ITK's
gradient anisotropic diffusion and a per-slice Python loop for the 2D
vesselness filter; here every step is a whole-volume array operation, so the
same call runs on NumPy or CuPy input.

Each primitive reproduces its reference's arithmetic, dtype, operation order
and boundary handling (see the per-function notes). For float32 input the
masks were bitwise equal to ``aicssegmentation`` on every volume measured
(240 A549 ER volumes, 10 more across ER and mitochondria) on an x86-64 AVX2
host without AVX-512, where NumPy's float32 power returns glibc's FMA
``powf``.

The reference itself is not bitwise across CPUs. On AVX-512 hosts NumPy's
float32 ``np.power`` in its vesselness rounds differently, which moves 1-3
voxels per volume. The CPU path calls the same ``np.power``, so it follows the
host. The GPU kernel always reproduces glibc's ``powf``, so it matches the
AVX2 result.

The match is measured, not guaranteed for every input. The diffusion's
conductance ``K`` replays ITK's serial summation only near a float32 rounding
midpoint (see ``_K_SERIAL_MARGIN``).
"""

import functools
from typing import Any
from collections.abc import Sequence

import numpy as np
from scipy.ndimage import generate_binary_structure

from ..cuda import CUDAManager, asnumpy, get_device, get_array_module
from ..scipy import ndimage as _ndimage
from .segment_utils import _SKIMAGE_USES_MAX_SIZE

# Relative distance from a float32 rounding midpoint below which ITK's serially
# accumulated K could round differently from a pairwise sum. The two sums were
# measured 1.3e-11 to 1.5e-11 apart on 48x640x960 volumes, so this margin is a
# measured heuristic with ~70x headroom, not a bound. The worst-case error of a
# serial float64 sum of n terms is (n - 1) * 2**-53 relative: about 1e-8 for
# such a volume's 3 x 48 x 640 x 960 squared derivatives. An interval that wide
# holds a float32 midpoint for a sixth to a third of all K values. Replaying
# those would mean 2-3 host replays of 1-3 s each per 10-iteration volume,
# against 0.37 s for the whole GPU workflow. An input whose serial sum drifts
# past the margin can therefore round K differently from ITK.
_K_SERIAL_MARGIN = 1e-9


@functools.cache
def _cuda_functions(
    source: str, names: tuple[str, ...], device_id: int
) -> tuple[Any, ...]:
    """Compile ``source`` once per device and return its kernels.

    ``--fmad=false`` keeps every multiply and add separately rounded, as in the
    scalar C/C++ references; contractions the reference does perform are
    spelled out as explicit ``fma`` calls.
    """
    cp = CUDAManager().get_cp()
    if cp is None:
        raise RuntimeError("CuPy is required for the CUDA kernels")
    with cp.cuda.Device(device_id):
        module = cp.RawModule(code=source, options=("--fmad=false",))
        return tuple(module.get_function(name) for name in names)


def _cupy_device(array: np.ndarray) -> Any:
    """CuPy device of ``array``; NumPy's stubs type ``.device`` as ``"cpu"``."""
    return array.device


def _launch_config(size: int, threads: int = 256) -> tuple[tuple[int], tuple[int]]:
    """Grid-stride launch configuration for ``size`` elements."""
    return (min((size + threads - 1) // threads, 65535 * 8),), (threads,)


def intensity_normalization(
    image: np.ndarray, scaling_param: tuple[float, float]
) -> np.ndarray:
    """Clip to ``[mean - a*std, mean + b*std]`` and rescale to ``[0, 1]``.

    Port of ``aicssegmentation.core.pre_processing_utils.intensity_normalization``
    for the two-value auto-contrast form ``[a, b]``. Unlike the reference, the
    input is not modified in place.

    The mean and standard deviation are computed on the host exactly as the
    reference's ``scipy.stats.norm.fit`` does (NumPy's pairwise summation in
    the input dtype). A device reduction rounds differently, and on real
    volumes that moves the stretch bounds by an ulp, enough to flip voxels of
    the final mask.

    Parameters
    ----------
    image : np.ndarray
        Floating-point image.
    scaling_param : tuple[float, float]
        Number of standard deviations ``(a, b)`` kept below and above the mean.

    Returns
    -------
    np.ndarray
        Normalized image in ``[0, 1]`` with the input's dtype.
    """
    if len(scaling_param) != 2:
        raise ValueError(
            f"scaling_param must be (a, b) auto-contrast bounds, got {scaling_param!r}"
        )
    if not np.issubdtype(image.dtype, np.floating):
        raise TypeError(f"image must be floating point, got {image.dtype}")
    if not bool(np.isfinite(image).all()):
        raise ValueError("image contains non-finite values")
    dtype = image.dtype.type
    flat = asnumpy(image).ravel()
    mean, lowest, highest = flat.mean(), flat.min(), flat.max()
    # A device array arrives as a fresh host copy, so square it in place.
    on_cpu = get_device(image) == "CPU"
    sq = flat - mean if on_cpu else np.subtract(flat, mean, out=flat)
    np.multiply(sq, sq, out=sq)
    std = np.sqrt(sq.mean())
    del flat, sq
    # The reference's scalar expressions verbatim, so the installed NumPy's
    # promotion rules apply to both alike: float32 under NEP 50, float64 scalars
    # (rounded to float32 where they meet the array) under NumPy 1.
    stretch_min = max(mean - scaling_param[0] * std, lowest)
    stretch_max = min(mean + scaling_param[1] * std, highest)
    if stretch_min > stretch_max:
        raise ValueError(
            f"scaling_param {scaling_param!r} gives an empty range "
            f"[{stretch_min}, {stretch_max}]"
        )
    denominator = stretch_max - stretch_min + 1e-8
    low, high = dtype(stretch_min), dtype(stretch_max)
    out = np.clip(image, low, high)
    out -= low
    out += dtype(1e-8)
    out /= dtype(denominator)
    return out


def gradient_anisotropic_diffusion(
    image: np.ndarray,
    n_iter: int = 10,
    conductance: float = 1.2,
    time_step: float = 0.0625,
    spacing: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Perona-Malik edge-preserving smoothing, as ITK's GradientAnisotropicDiffusion.

    Port of ``itk.GradientAnisotropicDiffusionImageFilter`` (the
    ``edge_preserving_smoothing_3d`` step of ``aicssegmentation``). As in ITK:

    - the conductance ``K = -2 * conductance**2 * <|grad u|^2>`` is recomputed
      from the current image at the start of every iteration;
    - derivatives use zero-flux Neumann boundaries (edge replication);
    - each update is evaluated in float64 and the image is stored as float32
      between iterations.

    Parameters
    ----------
    image : np.ndarray
        2D or 3D image; cast to float32.
    n_iter : int
        Number of diffusion iterations.
    conductance : float
        Conductance parameter; larger values smooth across stronger edges.
    time_step : float
        Integration step; stable up to ``min(spacing) / 2**(ndim + 1)``.
    spacing : tuple[float, ...] | None
        Voxel size per axis; derivatives are scaled by ``1 / spacing``.
        Defaults to unit spacing.

    Returns
    -------
    np.ndarray
        Smoothed float32 image on the input's device.
    """
    ndim = image.ndim
    scale = [1.0] * ndim if spacing is None else [1.0 / float(s) for s in spacing]
    if len(scale) != ndim:
        raise ValueError(f"spacing has {len(scale)} entries for a {ndim}D image")
    u = image.astype(np.float32, order="C", copy=True)
    use_kernel = ndim == 3 and get_device(u) == "GPU"
    step = _diffusion_step_cuda if use_kernel else _diffusion_step_xp
    for _ in range(n_iter):
        # K == 0 leaves u unchanged, so every later iteration is a no-op too.
        if not step(u, scale, conductance, time_step):
            break
    return u


def _diffusion_step_xp(
    u: np.ndarray, scale: list[float], conductance: float, time_step: float
) -> bool:
    """Apply one ITK diffusion iteration to ``u`` in place; ``False`` if ``K == 0``.

    Array-operation form of ``GradientNDAnisotropicDiffusionFunction``: works
    on NumPy and CuPy arrays of any dimensionality.
    """
    ndim = u.ndim
    inner = tuple(slice(1, -1) for _ in range(ndim))
    p = np.pad(u, 1, mode="edge")
    center = p[inner]

    def shifted(offsets: dict[int, int]) -> np.ndarray:
        """View of ``p`` at ``center + offsets`` over the unpadded region."""
        index = list(inner)
        for axis, offset in offsets.items():
            index[axis] = slice(1 + offset, p.shape[axis] - 1 + offset)
        return p[tuple(index)]

    def sub(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Float32 pixel difference widened to float64, as ITK's ``GetPixel`` math."""
        return (a - b).astype(np.float64)

    # ITK iterates dimensions in index order (x, y, z) = array axes reversed.
    axes = list(reversed(range(ndim)))
    dx = {i: sub(shifted({i: 1}), shifted({i: -1})) / 2.0 * scale[i] for i in axes}
    grad_sq = dx[axes[0]] * dx[axes[0]]
    for i in axes[1:]:
        grad_sq = grad_sq + dx[i] * dx[i]
    k = _conductance_k(grad_sq, [dx[i] for i in axes], conductance)
    if k == 0.0:
        # ITK sets both conductances to zero, so the update vanishes.
        return False
    delta = np.zeros_like(center, dtype=np.float64)
    for i in axes:
        forward = sub(shifted({i: 1}), center) * scale[i]
        backward = sub(center, shifted({i: -1})) * scale[i]
        accum = np.zeros_like(center, dtype=np.float64)
        accum_d = np.zeros_like(center, dtype=np.float64)
        for j in axes:
            if j == i:
                continue
            aug = sub(shifted({i: 1, j: 1}), shifted({i: 1, j: -1})) / 2.0
            dim = sub(shifted({i: -1, j: 1}), shifted({i: -1, j: -1})) / 2.0
            accum += 0.25 * (dx[j] + aug * scale[j]) ** 2
            accum_d += 0.25 * (dx[j] + dim * scale[j]) ** 2
        forward = forward * np.exp((forward * forward + accum) / k)
        backward = backward * np.exp((backward * backward + accum_d) / k)
        delta += forward - backward
    # ITK ApplyUpdate: u += float(dt * double(float(delta))).
    u += (time_step * delta.astype(np.float32).astype(np.float64)).astype(np.float32)
    return True


def _conductance_k(
    grad_sq: np.ndarray, derivatives: Sequence[np.ndarray], conductance: float
) -> float:
    """ITK's ``K = -2 * conductance**2 * mean(|grad u|^2)``, stored as float32.

    ITK keeps ``m_K`` in the pixel type, so the float64 product is rounded, and
    it accumulates the squared derivatives serially in float64. A pairwise sum
    of ``grad_sq`` lands within ~1e-11 of that serial sum, which rounds to the
    same float32 unless K sits next to a rounding midpoint; only then is ITK's
    exact summation order replayed on the host from ``derivatives`` (central
    derivatives in ITK dimension order).
    """
    n = grad_sq.size
    k = float(grad_sq.sum()) / n * conductance * conductance * -2.0
    if _near_float32_midpoint(k, _K_SERIAL_MARGIN):
        total = _itk_serial_sum_of_squares([asnumpy(d) for d in derivatives])
        k = total / n * conductance * conductance * -2.0
    return float(np.float32(k))


def _near_float32_midpoint(value: float, margin: float) -> bool:
    """Whether ``value`` lies within ``margin * |value|`` of a float32 midpoint."""
    if value == 0.0 or not np.isfinite(value):
        return False
    nearest = np.float32(value)
    for direction in (-np.inf, np.inf):
        neighbour = np.nextafter(nearest, np.float32(direction))
        midpoint = (float(nearest) + float(neighbour)) / 2.0
        if abs(value - midpoint) <= margin * abs(value):
            return True
    return False


def _itk_boundary_regions(shape: tuple[int, ...]) -> list[tuple[slice, ...]]:
    """Regions in the order ITK's ``ImageBoundaryFacesCalculator`` visits them.

    For a radius-1 neighbourhood: the interior first, then for each ITK
    dimension (x, y, z = array axes reversed) its lower and upper one-voxel
    faces, each restricted to the part not covered by earlier faces.
    """
    ndim = len(shape)
    start = [0] * ndim
    size = list(shape)
    faces = []

    def region(axis: int, face_start: int, face_size: int) -> tuple[slice, ...]:
        return tuple(
            slice(face_start, face_start + face_size)
            if a == axis
            else slice(start[a], start[a] + size[a])
            for a in range(ndim)
        )

    for d in range(ndim):
        axis = ndim - 1 - d
        low = min(1, size[axis])
        faces.append(region(axis, start[axis], low))
        start[axis] += low
        size[axis] -= low
        high = min(1, size[axis])
        faces.append(region(axis, start[axis] + size[axis] - high, high))
        size[axis] -= high
    interior = tuple(slice(start[a], start[a] + size[a]) for a in range(ndim))
    return [interior, *faces]


def _itk_serial_sum_of_squares(derivatives: Sequence[np.ndarray]) -> float:
    """``sum(d**2)`` accumulated in ITK's order: regions, raster order, then dims."""
    squares = np.stack([d * d for d in derivatives], axis=-1)
    terms = np.concatenate(
        [squares[r].ravel() for r in _itk_boundary_regions(squares.shape[:-1])]
    )
    del squares
    return float(np.add.accumulate(terms, out=terms)[-1])


# Each face flux F_i(c) = fwd * exp((fwd^2 + accum) / K) through (c, c + e_i) is
# computed once. In ITK's per-voxel ComputeUpdate, voxel c's backward term along i
# repeats c - e_i's forward term on the same float32 operands (the cross-derivative
# sum differs only by the commutative order of one addition), and it is exactly
# +0.0 on the lower border. So delta(c) = sum_i [F_i(c) - F_i(c - e_i)] is bitwise
# ITK's update at half the float64 exp/div work.
_DIFFUSION_CUDA_SOURCE = r"""
__device__ __forceinline__ long clamp(long v, long n) {
  return v < 0 ? 0 : (v >= n ? n - 1 : v);
}

// ITK subtracts float pixels in float, then widens the result to double.
__device__ __forceinline__ double sub(float a, float b) {
  return (double)(a - b);
}

// Linear index of the clamped (zero-flux Neumann) neighbour c + s * e_d along
// ITK dimension d (0=x, 1=y, 2=z).
__device__ __forceinline__ long neighbour(
    long z, long y, long x, int d, int s, long nz, long ny, long nx) {
  if (d == 0) x = clamp(x + s, nx);
  if (d == 1) y = clamp(y + s, ny);
  if (d == 2) z = clamp(z + s, nz);
  return (z * ny + y) * nx + x;
}

// D[j] = central derivative along ITK dimension j; g = |grad u|^2 in ITK order.
extern "C" __global__ void gad_gradient(
    const float* u, double* D, double* g, long nz, long ny, long nx,
    double s0, double s1, double s2) {
  const double sc[3] = {s0, s1, s2};
  long n = nz * ny * nx;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += (long)blockDim.x * gridDim.x) {
    long x = idx % nx, y = (idx / nx) % ny, z = idx / (nx * ny);
    double acc = 0.0;
    for (int j = 0; j < 3; ++j) {
      double v = sub(u[neighbour(z, y, x, j, 1, nz, ny, nx)],
                     u[neighbour(z, y, x, j, -1, nz, ny, nx)]) / 2.0;
      v *= sc[j];
      D[j * n + idx] = v;
      acc += v * v;
    }
    g[idx] = acc;
  }
}

// F[i] = flux through the face (c, c + e_i).
extern "C" __global__ void gad_flux(
    const float* u, const double* D, double* F, long nz, long ny, long nx,
    double k, double s0, double s1, double s2) {
  const double sc[3] = {s0, s1, s2};
  long n = nz * ny * nx;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += (long)blockDim.x * gridDim.x) {
    long x = idx % nx, y = (idx / nx) % ny, z = idx / (nx * ny);
    for (int i = 0; i < 3; ++i) {
      long nb = neighbour(z, y, x, i, 1, nz, ny, nx);
      double fwd = sub(u[nb], u[idx]);
      fwd *= sc[i];
      double accum = 0.0;
      for (int j = 0; j < 3; ++j) {
        if (j == i) continue;
        double t = D[j * n + idx] + D[j * n + nb];
        accum += 0.25 * (t * t);
      }
      F[i * n + idx] = fwd * exp((fwd * fwd + accum) / k);
    }
  }
}

// delta = sum_i [F_i(c) - F_i(c - e_i)], then ITK's ApplyUpdate in place.
extern "C" __global__ void gad_apply(
    const double* F, float* u, long nz, long ny, long nx, double dt) {
  long n = nz * ny * nx;
  const long stride[3] = {1, nx, nx * ny};
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += (long)blockDim.x * gridDim.x) {
    const long coord[3] = {idx % nx, (idx / nx) % ny, idx / (nx * ny)};
    double delta = 0.0;
    for (int i = 0; i < 3; ++i) {
      double back = coord[i] > 0 ? F[i * n + idx - stride[i]] : 0.0;
      delta += F[i * n + idx] - back;
    }
    u[idx] = u[idx] + (float)(dt * (double)(float)delta);
  }
}
"""


def _diffusion_step_cuda(
    u: np.ndarray, scale: list[float], conductance: float, time_step: float
) -> bool:
    """Kernel form of :func:`_diffusion_step_xp` for C-contiguous 3D CuPy arrays."""
    xp = get_array_module(u)
    nz, ny, nx = (np.int64(n) for n in u.shape)
    # ITK dimension order (x, y, z) = array axes reversed.
    sx, sy, sz = (np.float64(scale[axis]) for axis in (2, 1, 0))
    launch = _launch_config(u.size)
    device = _cupy_device(u)
    with device:
        gradient_kernel, flux_kernel, apply_kernel = _cuda_functions(
            _DIFFUSION_CUDA_SOURCE,
            ("gad_gradient", "gad_flux", "gad_apply"),
            device.id,
        )
        derivatives = xp.empty((3, *u.shape), dtype=np.float64)
        grad_sq = xp.empty(u.shape, dtype=np.float64)
        gradient_kernel(*launch, (u, derivatives, grad_sq, nz, ny, nx, sx, sy, sz))
        k = _conductance_k(grad_sq, list(derivatives), conductance)
        del grad_sq
        if k == 0.0:
            return False
        flux = xp.empty((3, *u.shape), dtype=np.float64)
        flux_kernel(
            *launch, (u, derivatives, flux, nz, ny, nx, np.float64(k), sx, sy, sz)
        )
        del derivatives
        apply_kernel(*launch, (flux, u, nz, ny, nx, np.float64(time_step)))
    return True


def _gaussian_nearest(image: np.ndarray, sigmas: list[float]) -> np.ndarray:
    """Separable Gaussian (``mode="nearest"``, ``truncate=3``) rounded like SciPy.

    SciPy filters a float32 image one axis at a time, accumulating each pass in
    float64 and storing it as float32. cupyx accumulates float32 input in
    float32, so on GPU each pass runs on a float64 copy to stay on SciPy's
    arithmetic. Axes with ``sigma == 0`` are skipped, as in SciPy.
    """
    if get_device(image) == "CPU":
        return _ndimage.gaussian_filter(
            image, sigma=sigmas, mode="nearest", truncate=3.0
        )
    out = image
    for axis, sigma in enumerate(sigmas):
        if sigma > 1e-15:
            out = _ndimage.gaussian_filter1d(
                out.astype(np.float64), sigma, axis=axis, mode="nearest", truncate=3.0
            ).astype(image.dtype)
    return out


def _hessian_2d_eigen_max(image: np.ndarray, sigma: float) -> np.ndarray:
    """Hessian eigenvalue of largest magnitude in the last two axes.

    Mirrors ``aicssegmentation.core.hessian.absolute_3d_hessian_eigenvalues``
    for 2D planes: Gaussian smoothing (``mode="nearest"``, ``truncate=3``),
    ``np.gradient`` twice, ``sigma**2`` scaling. Leading axes are a batch of
    independent planes.
    """
    y_axis, x_axis = image.ndim - 2, image.ndim - 1
    smoothed = _gaussian_nearest(image, [0.0] * (image.ndim - 2) + [sigma, sigma])
    grad_y, grad_x = np.gradient(smoothed, axis=(y_axis, x_axis))
    del smoothed
    hessian = [
        np.gradient(grad_y, axis=y_axis),
        np.gradient(grad_y, axis=x_axis),
        np.gradient(grad_x, axis=x_axis),
    ]
    del grad_y, grad_x
    if sigma > 0:
        for h in hessian:
            h *= sigma**2
    a, b, c = (h.astype(np.float64) for h in hessian)
    del hessian
    half_trace = (a + c) / 2.0
    radius = np.sqrt(((a - c) / 2.0) ** 2 + b**2)
    del a, b, c
    low = (half_trace - radius).astype(image.dtype)
    high = (half_trace + radius).astype(image.dtype)
    # Ascending |lambda| with a stable tie-break, as ``sortbyabs`` on eigvalsh
    # output: the larger eigenvalue wins when the magnitudes are equal.
    return np.where(np.abs(low) > np.abs(high), low, high)


_RESPONSE_CUDA_SOURCE = r"""
// glibc 2.28 powf(x, 3.0f) for finite x >= 0: sysdeps/ieee754/flt-32/e_powf.c with
// its log2/exp2 tables, as built into the x86-64 FMA ifunc variant (__powf_fma,
// -mfma: every single-use a*b+c contracted). NumPy's float32 power calls it on
// AVX2 hosts, and CUDA's powf rounds differently in ~6% of real inputs.
__constant__ double POWF_INVC[16] = {
  0x1.661ec79f8f3bep+0, 0x1.571ed4aaf883dp+0, 0x1.49539f0f010bp+0, 0x1.3c995b0b80385p+0,
  0x1.30d190c8864a5p+0, 0x1.25e227b0b8eap+0, 0x1.1bb4a4a1a343fp+0, 0x1.12358f08ae5bap+0,
  0x1.0953f419900a7p+0, 0x1p+0, 0x1.e608cfd9a47acp-1, 0x1.ca4b31f026aap-1,
  0x1.b2036576afce6p-1, 0x1.9c2d163a1aa2dp-1, 0x1.886e6037841edp-1, 0x1.767dcf5534862p-1};
__constant__ double POWF_LOGC[16] = {
  -0x1.efec65b963019p-2, -0x1.b0b6832d4fca4p-2, -0x1.7418b0a1fb77bp-2, -0x1.39de91a6dcf7bp-2,
  -0x1.01d9bf3f2b631p-2, -0x1.97c1d1b3b7afp-3, -0x1.2f9e393af3c9fp-3, -0x1.960cbbf788d5cp-4,
  -0x1.a6f9db6475fcep-5, 0x0p+0, 0x1.338ca9f24f53dp-4, 0x1.476a9543891bap-3,
  0x1.e840b4ac4e4d2p-3, 0x1.40645f0c6651cp-2, 0x1.88e9c2c1b9ff8p-2, 0x1.ce0a44eb17bccp-2};
__constant__ double POWF_A[5] = {
  0x1.27616c9496e0bp-2, -0x1.71969a075c67ap-2, 0x1.ec70a6ca7baddp-2,
  -0x1.7154748bef6c8p-1, 0x1.71547652ab82bp0};
__constant__ unsigned long long EXP2F_T[32] = {
  0x3ff0000000000000ULL, 0x3fefd9b0d3158574ULL, 0x3fefb5586cf9890fULL, 0x3fef9301d0125b51ULL,
  0x3fef72b83c7d517bULL, 0x3fef54873168b9aaULL, 0x3fef387a6e756238ULL, 0x3fef1e9df51fdee1ULL,
  0x3fef06fe0a31b715ULL, 0x3feef1a7373aa9cbULL, 0x3feedea64c123422ULL, 0x3feece086061892dULL,
  0x3feebfdad5362a27ULL, 0x3feeb42b569d4f82ULL, 0x3feeab07dd485429ULL, 0x3feea47eb03a5585ULL,
  0x3feea09e667f3bcdULL, 0x3fee9f75e8ec5f74ULL, 0x3feea11473eb0187ULL, 0x3feea589994cce13ULL,
  0x3feeace5422aa0dbULL, 0x3feeb737b0cdc5e5ULL, 0x3feec49182a3f090ULL, 0x3feed503b23e255dULL,
  0x3feee89f995ad3adULL, 0x3feeff76f2fb5e47ULL, 0x3fef199bdd85529cULL, 0x3fef3720dcef9069ULL,
  0x3fef5818dcfba487ULL, 0x3fef7c97337b9b5fULL, 0x3fefa4afa2a490daULL, 0x3fefd0765b6e4540ULL};
__constant__ double EXP2F_C[3] = {
  0x1.c6af84b912394p-5, 0x1.ebfce50fac4f3p-3, 0x1.62e42ff0c52d6p-1};

__device__ float glibc_powf3(float x) {
  unsigned int ix = __float_as_uint(x);
  if (ix - 0x00800000u >= 0x7f800000u - 0x00800000u) {
    if (2u * ix == 0u) return x * x;
    if (ix < 0x00800000u) {  // normalize subnormal x
      ix = __float_as_uint(x * 0x1p23f) & 0x7fffffffu;
      ix -= 23u << 23;
    }
  }
  // log2_inline
  unsigned int tmp = ix - 0x3f330000u;
  int i = (tmp >> 19) % 16;
  unsigned int top = tmp & 0xff800000u;
  unsigned int iz = ix - top;
  int k = (int)top >> 23;
  double z = (double)__uint_as_float(iz);
  double r = fma(z, POWF_INVC[i], -1.0);
  double y0 = POWF_LOGC[i] + (double)k;
  double r2 = r * r;
  double y = fma(POWF_A[0], r, POWF_A[1]);
  double p = fma(POWF_A[2], r, POWF_A[3]);
  double r4 = r2 * r2;
  double q = fma(POWF_A[4], r, y0);
  q = fma(p, r2, q);
  y = fma(y, r4, q);
  double ylogx = 3.0 * y;
  unsigned long long bits = (unsigned long long)__double_as_longlong(ylogx);
  unsigned long long limit = (unsigned long long)__double_as_longlong(126.0);
  if (((bits >> 47) & 0xffffULL) >= (limit >> 47)) {
    if (ylogx > 0x1.fffffffd1d571p+6) return __int_as_float(0x7f800000);
    if (ylogx <= -150.0) return 0.0f;
  }
  // exp2_inline (TOINT_INTRINSICS == 0)
  const double shift = 0x1.8p+52 / 32;
  double kd = ylogx + shift;
  unsigned long long ki = (unsigned long long)__double_as_longlong(kd);
  kd -= shift;
  double rr = ylogx - kd;
  unsigned long long t = EXP2F_T[ki % 32] + (ki << 47);
  double s = __longlong_as_double((long long)t);
  double zz = fma(EXP2F_C[0], rr, EXP2F_C[1]);
  double rr2 = rr * rr;
  double yy = fma(EXP2F_C[2], rr, 1.0);
  yy = fma(zz, rr2, yy);
  yy = yy * s;
  return (float)yy;
}

// aicssegmentation.core.vessel.compute_vesselness2D (tau = 1) on (nz, ny, nx_out)
// columns of an (nz, ny, nx_in) eigenvalue stack with per-plane minima.
extern "C" __global__ void vesselness_response(
    const float* eigen, const float* plane_min, float* out,
    long nz, long ny, long nx_in, long nx_out) {
  long n = nz * ny * nx_out;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += (long)blockDim.x * gridDim.x) {
    long x = idx % nx_out, y = (idx / nx_out) % ny, z = idx / (nx_out * ny);
    float e = eigen[(z * ny + y) * nx_in + x];
    float m = plane_min[z];
    float lambda3 = (e < 0.0f && e >= m) ? m : e;
    float diff = fabsf(lambda3 - e);
    float numerator = 27.0f * ((e * e) * diff);
    float denominator = glibc_powf3(2.0f * fabsf(e) + diff);
    if (denominator == 0.0f) denominator = 1e-10f;
    float response = numerator / denominator;
    if (e < 0.5f * lambda3) response = 1.0f;
    if (e >= 0.0f) response = 0.0f;
    if (isinf(response)) response = 0.0f;
    out[idx] = response;
  }
}
"""


def _vesselness_2d_response(eigen: np.ndarray, width: int) -> np.ndarray:
    """Jerman 2D filament response (tau = 1) over the first ``width`` columns.

    The minimum eigenvalue is taken over each whole plane, as in the reference;
    the response is elementwise, so only the kept columns are computed. On GPU
    one kernel evaluates a 3D float32 stack with glibc's ``powf`` algorithm,
    which the NumPy reference calls on x86-64 AVX2 hosts; other dtypes use the
    array operations on either device.
    """
    if get_device(eigen) == "GPU" and eigen.ndim == 3 and eigen.dtype == np.float32:
        return _vesselness_2d_response_cuda(eigen, width)
    plane_min = eigen.min(axis=(eigen.ndim - 2, eigen.ndim - 1), keepdims=True)
    eigen = eigen[..., :width]
    lambda3 = np.where((eigen < 0) & (eigen >= plane_min), plane_min, eigen)
    diff = np.abs(lambda3 - eigen)
    numerator = 27 * (np.square(eigen) * diff)
    denominator = np.power(2 * np.abs(eigen) + diff, 3)
    denominator = np.where(denominator == 0, eigen.dtype.type(1e-10), denominator)
    response = numerator / denominator
    response = np.where(eigen < 0.5 * lambda3, eigen.dtype.type(1), response)
    response = np.where(eigen >= 0, eigen.dtype.type(0), response)
    return np.where(np.isinf(response), eigen.dtype.type(0), response)


def _vesselness_2d_response_cuda(eigen: np.ndarray, width: int) -> np.ndarray:
    """Kernel form of :func:`_vesselness_2d_response` for 3D float32 CuPy stacks."""
    xp = get_array_module(eigen)
    device = _cupy_device(eigen)
    with device:
        (kernel,) = _cuda_functions(
            _RESPONSE_CUDA_SOURCE, ("vesselness_response",), device.id
        )
        eigen = xp.ascontiguousarray(eigen)
        plane_min = xp.ascontiguousarray(eigen.min(axis=(1, 2)))
        nz, ny, nx_in = eigen.shape
        out = xp.empty((nz, ny, width), dtype=np.float32)
        kernel(
            *_launch_config(out.size),
            (
                eigen,
                plane_min,
                out,
                np.int64(nz),
                np.int64(ny),
                np.int64(nx_in),
                np.int64(width),
            ),
        )
    return out


def vesselness_slice_by_slice(image: np.ndarray, sigmas: list[float]) -> np.ndarray:
    """Multi-scale 2D filament filter on each z-plane of a 3D image.

    Port of ``aicssegmentation.core.vessel.vesselnessSliceBySlice`` for
    bright-on-dark structures. As in the reference, each plane is filtered
    side by side with the z maximum-intensity projection, the last three
    columns are zeroed, and the response uses ``tau = 1`` (the reference
    ignores its ``tau`` argument). All planes are filtered in one batched call.

    Parameters
    ----------
    image : np.ndarray
        3D float image ``(Z, Y, X)``.
    sigmas : list[float]
        Gaussian scales; the response is the maximum over scales.

    Returns
    -------
    np.ndarray
        Float64 response ``(Z, Y, X)``.
    """
    if image.ndim != 3:
        raise ValueError(f"image must be 3D (Z, Y, X), got {image.ndim}D")
    if not sigmas:
        raise ValueError("sigmas must contain at least one scale")
    if any(s < 0 for s in sigmas):
        raise ValueError("Sigma values less than zero are not valid")
    if image.shape[2] <= 3:
        raise ValueError(f"image must be wider than 3 columns, got {image.shape[2]}")
    kept = image.shape[2] - 3
    mip = np.broadcast_to(image.max(axis=0), image.shape)
    stacked = np.concatenate([image, mip], axis=2)
    del mip
    response = None
    for sigma in sigmas:
        r = _vesselness_2d_response(_hessian_2d_eigen_max(stacked, sigma), kept)
        response = r if response is None else np.maximum(response, r)
    assert response is not None  # sigmas is non-empty
    return np.pad(response.astype(np.float64), ((0, 0), (0, 0), (0, 3)))


def remove_small_objects_aics(
    mask: np.ndarray,
    min_size: int,
    *,
    per_slice: bool = False,
    inclusive: bool = False,
) -> np.ndarray:
    """Drop face-connected components below ``min_size`` voxels.

    Matches ``skimage.morphology.remove_small_objects(mask, min_size=min_size,
    connectivity=1)`` as the ``aicssegmentation`` workflows call it. Unlike
    :func:`cubic.segmentation.remove_small_objects`, which keeps the pre-0.26
    ``size >= min_size`` rule for label images through cuCIM, this takes
    binary masks, needs only cupyx on GPU, and can filter per z-plane.

    Parameters
    ----------
    mask : np.ndarray
        Binary mask; NumPy or CuPy.
    min_size : int
        Size threshold in voxels.
    per_slice : bool
        Filter the components of each z-plane of a 3D mask independently
        (in a single labeling pass).
    inclusive : bool
        Also remove components of exactly ``min_size`` voxels. scikit-image
        0.26 forwards the deprecated ``min_size`` argument to the inclusive
        ``max_size`` threshold, so ``aicssegmentation`` run under
        scikit-image >= 0.26 behaves as ``inclusive=True``; older releases
        behave as ``inclusive=False``.

    Returns
    -------
    np.ndarray
        Boolean mask on the input's device.
    """
    mask = mask.astype(bool, copy=False)
    structure = generate_binary_structure(mask.ndim, 1)
    if per_slice:
        if mask.ndim != 3:
            raise ValueError("per_slice requires a 3D mask")
        structure[0] = False
        structure[2] = False
    labels, _ = _ndimage.label(mask, structure=structure)
    sizes = np.bincount(labels.ravel())
    keep = sizes > min_size if inclusive else sizes >= min_size
    keep[0] = False
    return keep[labels]


def _resolve_inclusive(size_filter_inclusive: bool | None) -> bool:
    """``None`` follows the installed scikit-image, as the reference would."""
    if size_filter_inclusive is None:
        return _SKIMAGE_USES_MAX_SIZE
    return size_filter_inclusive


def workflow_sec61b(
    image: np.ndarray, *, size_filter_inclusive: bool | None = None
) -> np.ndarray:
    """SEC61B (endoplasmic reticulum) classic segmentation of one z-stack.

    Port of ``aicssegmentation.structure_wrapper.seg_sec61b.Workflow_sec61b``
    with its fixed parameters (no rescaling).

    Parameters
    ----------
    image : np.ndarray
        3D image ``(Z, Y, X)``; NumPy or CuPy. Converted to float32 first; the
        bitwise match with the reference holds for float32 input (the
        reference normalizes other dtypes in their own precision).
    size_filter_inclusive : bool | None
        Small-object semantics of the reference run: ``True`` reproduces
        ``aicssegmentation`` under scikit-image >= 0.26, ``False`` under older
        releases, ``None`` (default) follows the installed scikit-image. See
        :func:`remove_small_objects_aics`.

    Returns
    -------
    np.ndarray
        Boolean mask on the input's device.
    """
    inclusive = _resolve_inclusive(size_filter_inclusive)
    norm = intensity_normalization(image.astype(np.float32, copy=False), (2.5, 7.5))
    smooth = gradient_anisotropic_diffusion(norm)
    del norm
    bw = vesselness_slice_by_slice(smooth, sigmas=[1.0]) > 0.15
    del smooth
    bw = remove_small_objects_aics(bw, 15, inclusive=inclusive)
    bw = remove_small_objects_aics(bw, 3, per_slice=True, inclusive=inclusive)
    return remove_small_objects_aics(bw, 15, inclusive=inclusive)


def workflow_tomm20(
    image: np.ndarray, *, size_filter_inclusive: bool | None = None
) -> np.ndarray:
    """TOMM20 (mitochondria) classic segmentation of one z-stack.

    Port of ``aicssegmentation.structure_wrapper.seg_tomm20.Workflow_tomm20``
    with its fixed parameters (no rescaling).

    Parameters
    ----------
    image : np.ndarray
        3D image ``(Z, Y, X)``; NumPy or CuPy. Converted to float32 first; the
        bitwise match with the reference holds for float32 input (the
        reference normalizes other dtypes in their own precision).
    size_filter_inclusive : bool | None
        Small-object semantics of the reference run: ``True`` reproduces
        ``aicssegmentation`` under scikit-image >= 0.26, ``False`` under older
        releases, ``None`` (default) follows the installed scikit-image. See
        :func:`remove_small_objects_aics`.

    Returns
    -------
    np.ndarray
        Boolean mask on the input's device.
    """
    norm = intensity_normalization(image.astype(np.float32, copy=False), (3.5, 15.0))
    smooth = _gaussian_nearest(norm, [1.0] * norm.ndim)
    del norm
    bw = vesselness_slice_by_slice(smooth, sigmas=[1.5]) > 0.16
    return remove_small_objects_aics(
        bw, 10, inclusive=_resolve_inclusive(size_filter_inclusive)
    )
