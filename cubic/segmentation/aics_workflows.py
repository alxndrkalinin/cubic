"""Device-agnostic Allen Cell Structure Segmenter classic workflows.

Ports of the ``aicssegmentation`` SEC61B (ER) and TOMM20 (mitochondria)
workflows and the primitives they share. The CPU reference runs ITK's
gradient anisotropic diffusion and a per-slice Python loop for the 2D
vesselness filter; here every step is a whole-volume array operation, so the
same call runs on NumPy or CuPy input.

Each primitive follows its reference's arithmetic, dtype and boundary handling
(see the per-function notes), so a mask from these workflows differs from the
reference only where floating-point rounding moves a voxel across a threshold.
"""

import functools
from itertools import combinations_with_replacement

import numpy as np
from scipy.ndimage import generate_binary_structure

from ..cuda import asnumpy, get_device, get_array_module
from ..scipy import ndimage as _ndimage


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
    mean = flat.mean()
    std = np.sqrt(((flat - mean) ** 2).mean())
    stretch_min = max(mean - dtype(scaling_param[0]) * std, flat.min())
    stretch_max = min(mean + dtype(scaling_param[1]) * std, flat.max())
    del flat
    if stretch_min > stretch_max:
        raise ValueError(
            f"scaling_param {scaling_param!r} gives an empty range "
            f"[{stretch_min}, {stretch_max}]"
        )
    eps = dtype(1e-8)
    clipped = np.clip(image, stretch_min, stretch_max)
    return (clipped - stretch_min + eps) / (stretch_max - stretch_min + eps)


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
    u = image.astype(np.float32, copy=True)
    use_kernel = ndim == 3 and get_device(u) == "GPU"
    step = _diffusion_step_cuda if use_kernel else _diffusion_step_xp
    for _ in range(n_iter):
        delta = step(u, scale, conductance)
        if delta is not None:
            u += (time_step * delta.astype(np.float64)).astype(np.float32)
    return u


def _diffusion_step_xp(
    u: np.ndarray, scale: list[float], conductance: float
) -> np.ndarray | None:
    """One ITK diffusion update ``delta`` (float32), or ``None`` when ``K == 0``.

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
    k = _conductance_k(grad_sq, conductance)
    if k == 0.0:
        # ITK sets both conductances to zero, so the update vanishes.
        return None
    delta = np.zeros(center.shape, dtype=np.float64)
    for i in axes:
        forward = sub(shifted({i: 1}), center) * scale[i]
        backward = sub(center, shifted({i: -1})) * scale[i]
        accum = np.zeros(center.shape, dtype=np.float64)
        accum_d = np.zeros(center.shape, dtype=np.float64)
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
    return delta.astype(np.float32)


def _conductance_k(grad_sq: np.ndarray, conductance: float) -> float:
    """ITK's ``K = -2 * conductance**2 * mean(|grad u|^2)``, stored as float32.

    ITK keeps ``m_K`` in the pixel type, so the float64 product is rounded.
    """
    k = float(grad_sq.mean()) * conductance * conductance * -2.0
    return float(np.float32(k))


_DIFFUSION_CUDA_SOURCE = r"""
// Clamped (zero-flux Neumann) read of u at (z, y, x) + offset.
__device__ __forceinline__ float at(
    const float* u, long z, long y, long x, const int* o,
    long nz, long ny, long nx) {
  z += o[0]; y += o[1]; x += o[2];
  z = z < 0 ? 0 : (z >= nz ? nz - 1 : z);
  y = y < 0 ? 0 : (y >= ny ? ny - 1 : y);
  x = x < 0 ? 0 : (x >= nx ? nx - 1 : x);
  return u[(z * ny + y) * nx + x];
}

// ITK subtracts float pixels in float, then widens the result to double.
__device__ __forceinline__ double sub(float a, float b) {
  return (double)(a - b);
}

// Offsets (dz, dy, dx) of a unit step along ITK dimension d (0=x, 1=y, 2=z),
// optionally combined with a step along dimension e.
__device__ __forceinline__ void step(int* o, int d, int sd, int e, int se) {
  o[0] = 0; o[1] = 0; o[2] = 0;
  o[2 - d] += sd;
  if (e >= 0) o[2 - e] += se;
}

extern "C" __global__ void gad_grad_sq(
    const float* u, double* out, long nz, long ny, long nx,
    double s0, double s1, double s2) {
  const double sc[3] = {s0, s1, s2};
  long n = nz * ny * nx;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += (long)blockDim.x * gridDim.x) {
    long x = idx % nx, y = (idx / nx) % ny, z = idx / (nx * ny);
    int op[3], om[3];
    double acc = 0.0;
    for (int i = 0; i < 3; ++i) {
      step(op, i, 1, -1, 0); step(om, i, -1, -1, 0);
      double v = sub(at(u, z, y, x, op, nz, ny, nx), at(u, z, y, x, om, nz, ny, nx)) / 2.0;
      v *= sc[i];
      acc += v * v;
    }
    out[idx] = acc;
  }
}

// GradientNDAnisotropicDiffusionFunction::ComputeUpdate for one voxel.
extern "C" __global__ void gad_update(
    const float* u, float* out, long nz, long ny, long nx, double k,
    double s0, double s1, double s2) {
  const double sc[3] = {s0, s1, s2};
  const int zero[3] = {0, 0, 0};
  long n = nz * ny * nx;
  for (long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; idx < n;
       idx += (long)blockDim.x * gridDim.x) {
    long x = idx % nx, y = (idx / nx) % ny, z = idx / (nx * ny);
    int a[3], b[3];
    float c = at(u, z, y, x, zero, nz, ny, nx);
    double dx[3];
    for (int i = 0; i < 3; ++i) {
      step(a, i, 1, -1, 0); step(b, i, -1, -1, 0);
      dx[i] = sub(at(u, z, y, x, a, nz, ny, nx), at(u, z, y, x, b, nz, ny, nx)) / 2.0;
      dx[i] *= sc[i];
    }
    double delta = 0.0;
    for (int i = 0; i < 3; ++i) {
      step(a, i, 1, -1, 0); step(b, i, -1, -1, 0);
      double fwd = sub(at(u, z, y, x, a, nz, ny, nx), c);
      fwd *= sc[i];
      double bwd = sub(c, at(u, z, y, x, b, nz, ny, nx));
      bwd *= sc[i];
      double accum = 0.0, accum_d = 0.0;
      for (int j = 0; j < 3; ++j) {
        if (j == i) continue;
        step(a, i, 1, j, 1); step(b, i, 1, j, -1);
        double aug = sub(at(u, z, y, x, a, nz, ny, nx), at(u, z, y, x, b, nz, ny, nx)) / 2.0;
        aug *= sc[j];
        step(a, i, -1, j, 1); step(b, i, -1, j, -1);
        double dim = sub(at(u, z, y, x, a, nz, ny, nx), at(u, z, y, x, b, nz, ny, nx)) / 2.0;
        dim *= sc[j];
        accum += 0.25 * ((dx[j] + aug) * (dx[j] + aug));
        accum_d += 0.25 * ((dx[j] + dim) * (dx[j] + dim));
      }
      fwd = fwd * exp((fwd * fwd + accum) / k);
      bwd = bwd * exp((bwd * bwd + accum_d) / k);
      delta += fwd - bwd;
    }
    out[idx] = (float)delta;
  }
}
"""


@functools.cache
def _diffusion_kernels():
    """Compile the fused diffusion kernels once per process.

    ``--fmad=false`` keeps each multiply and add separately rounded, as in ITK's
    scalar C++ loop.
    """
    import cupy as cp

    module = cp.RawModule(code=_DIFFUSION_CUDA_SOURCE, options=("--fmad=false",))
    return module.get_function("gad_grad_sq"), module.get_function("gad_update")


def _diffusion_step_cuda(
    u: np.ndarray, scale: list[float], conductance: float
) -> np.ndarray | None:
    """Fused-kernel equivalent of :func:`_diffusion_step_xp` for 3D CuPy arrays."""
    xp = get_array_module(u)
    grad_sq_kernel, update_kernel = _diffusion_kernels()
    u = xp.ascontiguousarray(u)
    nz, ny, nx = (np.int64(n) for n in u.shape)
    # ITK dimension order (x, y, z) = array axes reversed.
    sx, sy, sz = (np.float64(scale[axis]) for axis in (2, 1, 0))
    threads = 256
    blocks = (min(int(u.size + threads - 1) // threads, 65535 * 8),)
    grad_sq = xp.empty(u.shape, dtype=np.float64)
    grad_sq_kernel(blocks, (threads,), (u, grad_sq, nz, ny, nx, sx, sy, sz))
    k = _conductance_k(grad_sq, conductance)
    del grad_sq
    if k == 0.0:
        return None
    delta = xp.empty(u.shape, dtype=np.float32)
    update_kernel(blocks, (threads,), (u, delta, nz, ny, nx, np.float64(k), sx, sy, sz))
    return delta


def _gaussian_nearest(image: np.ndarray, sigmas: list[float]) -> np.ndarray:
    """Separable Gaussian (``mode="nearest"``, ``truncate=3``) rounded like SciPy.

    SciPy filters a float32 image one axis at a time, accumulating each pass in
    float64 and storing it as float32. cupyx accumulates float32 input in
    float32, so each pass runs on a float64 copy here to keep both devices on
    SciPy's arithmetic. Axes with ``sigma == 0`` are skipped, as in SciPy.
    """
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
    plane_axes = (image.ndim - 2, image.ndim - 1)
    sigmas = [0.0] * (image.ndim - 2) + [sigma, sigma]
    smoothed = _gaussian_nearest(image, sigmas)
    gradients = np.gradient(smoothed, axis=plane_axes)
    h = {
        (a, b): np.gradient(gradients[a], axis=plane_axes[b])
        for a, b in combinations_with_replacement(range(2), 2)
    }
    if sigma > 0:
        h = {key: (sigma**2) * value for key, value in h.items()}
    a, b, c = (h[0, 0].astype(np.float64), h[0, 1], h[1, 1].astype(np.float64))
    half_trace = (a + c) / 2.0
    radius = np.sqrt(((a - c) / 2.0) ** 2 + b.astype(np.float64) ** 2)
    low = (half_trace - radius).astype(image.dtype)
    high = (half_trace + radius).astype(image.dtype)
    # Ascending |lambda| with a stable tie-break, as ``sortbyabs`` on eigvalsh
    # output: the larger eigenvalue wins when the magnitudes are equal.
    return np.where(np.abs(low) > np.abs(high), low, high)


def _vesselness_2d_response(eigen: np.ndarray, tau: float) -> np.ndarray:
    """Jerman 2D filament response; the minimum eigenvalue is taken per plane."""
    plane_axes = (eigen.ndim - 2, eigen.ndim - 1)
    plane_min = eigen.min(axis=plane_axes, keepdims=True)
    lambda3 = np.where((eigen < 0) & (eigen >= tau * plane_min), tau * plane_min, eigen)
    diff = np.abs(lambda3 - eigen)
    numerator = 27 * (np.square(eigen) * diff)
    denominator = np.power(2 * np.abs(eigen) + diff, 3)
    denominator = np.where(denominator == 0, eigen.dtype.type(1e-10), denominator)
    response = numerator / denominator
    response = np.where(eigen < 0.5 * lambda3, eigen.dtype.type(1), response)
    response = np.where(eigen >= 0, eigen.dtype.type(0), response)
    return np.where(np.isinf(response), eigen.dtype.type(0), response)


def vesselness_slice_by_slice(
    image: np.ndarray, sigmas: list[float], tau: float = 1.0
) -> np.ndarray:
    """Multi-scale 2D filament filter on each z-plane of a 3D image.

    Port of ``aicssegmentation.core.vessel.vesselnessSliceBySlice`` for
    bright-on-dark structures. As in the reference, each plane is filtered
    side by side with the z maximum-intensity projection, and the last three
    columns are zeroed. All planes are filtered in one batched call.

    Parameters
    ----------
    image : np.ndarray
        3D float image ``(Z, Y, X)``.
    sigmas : list[float]
        Gaussian scales; the response is the maximum over scales.
    tau : float
        Response uniformity parameter in ``[0.5, 1]``.

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
    width = image.shape[2]
    mip = np.broadcast_to(image.max(axis=0), image.shape)
    stacked = np.concatenate([image, mip], axis=2)
    response = _vesselness_2d_response(_hessian_2d_eigen_max(stacked, sigmas[0]), tau)
    for sigma in sigmas[1:]:
        r = _vesselness_2d_response(_hessian_2d_eigen_max(stacked, sigma), tau)
        response = np.maximum(response, r)
    xp = get_array_module(image)
    out = xp.zeros(image.shape, dtype=np.float64)
    out[:, :, : width - 3] = response[:, :, : width - 3]
    return out


def remove_small_objects_aics(
    mask: np.ndarray,
    min_size: int,
    *,
    per_slice: bool = False,
    inclusive: bool = False,
) -> np.ndarray:
    """Drop face-connected components below ``min_size`` voxels.

    Matches ``skimage.morphology.remove_small_objects(mask, min_size=min_size,
    connectivity=1)`` as the ``aicssegmentation`` workflows call it.

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
    mask = mask.astype(bool)
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


def workflow_sec61b(
    image: np.ndarray, *, size_filter_inclusive: bool = True
) -> np.ndarray:
    """SEC61B (endoplasmic reticulum) classic segmentation of one z-stack.

    Port of ``aicssegmentation.structure_wrapper.seg_sec61b.Workflow_sec61b``
    with its fixed parameters (no rescaling).

    Parameters
    ----------
    image : np.ndarray
        3D image ``(Z, Y, X)``; NumPy or CuPy.
    size_filter_inclusive : bool
        Small-object semantics of the reference run: ``True`` (default)
        reproduces ``aicssegmentation`` under scikit-image >= 0.26, ``False``
        under older releases. See :func:`remove_small_objects_aics`.

    Returns
    -------
    np.ndarray
        Boolean mask on the input's device.
    """
    rso = functools.partial(remove_small_objects_aics, inclusive=size_filter_inclusive)
    norm = intensity_normalization(image.astype(np.float32), (2.5, 7.5))
    smooth = gradient_anisotropic_diffusion(norm)
    bw = vesselness_slice_by_slice(smooth, sigmas=[1.0], tau=1.0) > 0.15
    bw = rso(bw, 15)
    bw = rso(bw, 3, per_slice=True)
    return rso(bw, 15)


def workflow_tomm20(
    image: np.ndarray, *, size_filter_inclusive: bool = True
) -> np.ndarray:
    """TOMM20 (mitochondria) classic segmentation of one z-stack.

    Port of ``aicssegmentation.structure_wrapper.seg_tomm20.Workflow_tomm20``
    with its fixed parameters (no rescaling).

    Parameters
    ----------
    image : np.ndarray
        3D image ``(Z, Y, X)``; NumPy or CuPy.
    size_filter_inclusive : bool
        Small-object semantics of the reference run: ``True`` (default)
        reproduces ``aicssegmentation`` under scikit-image >= 0.26, ``False``
        under older releases. See :func:`remove_small_objects_aics`.

    Returns
    -------
    np.ndarray
        Boolean mask on the input's device.
    """
    norm = intensity_normalization(image.astype(np.float32), (3.5, 15.0))
    smooth = _gaussian_nearest(norm, [1.0] * norm.ndim)
    bw = vesselness_slice_by_slice(smooth, sigmas=[1.5], tau=1.0) > 0.16
    return remove_small_objects_aics(bw, 10, inclusive=size_filter_inclusive)
