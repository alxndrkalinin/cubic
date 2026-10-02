"""Device-agnostic Allen Cell Structure Segmenter classic workflows.

Ports of the ``aicssegmentation`` SEC61B (ER) and TOMM20 (mitochondria)
workflows and the primitives they share. The CPU reference runs ITK's
gradient anisotropic diffusion and a per-slice Python loop for the 2D
vesselness filter; here every step is a whole-volume array operation, so the
same call runs on NumPy or CuPy input.

Each primitive reproduces its reference's arithmetic, dtype, operation order
and boundary handling (see the per-function notes), so the masks are bitwise
equal to ``aicssegmentation`` run on the same inputs. The reference itself is
not bitwise across CPUs: ``np.power`` in its vesselness rounds differently on
AVX-512 hosts, which moves 1-3 voxels per volume; these ports follow the AVX2
result on both devices.
"""

import functools

import numpy as np
from scipy.ndimage import generate_binary_structure

from ..cuda import CUDAManager, asnumpy, get_device, get_array_module
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
    mean, lowest, highest = flat.mean(), flat.min(), flat.max()
    # A device array arrives as a fresh host copy, so square it in place.
    on_cpu = get_device(image) == "CPU"
    sq = flat - mean if on_cpu else np.subtract(flat, mean, out=flat)
    np.multiply(sq, sq, out=sq)
    std = np.sqrt(sq.mean())
    del flat, sq
    stretch_min = max(mean - dtype(scaling_param[0]) * std, lowest)
    stretch_max = min(mean + dtype(scaling_param[1]) * std, highest)
    if stretch_min > stretch_max:
        raise ValueError(
            f"scaling_param {scaling_param!r} gives an empty range "
            f"[{stretch_min}, {stretch_max}]"
        )
    eps = dtype(1e-8)
    out = np.clip(image, stretch_min, stretch_max)
    out -= stretch_min
    out += eps
    out /= stretch_max - stretch_min + eps
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
    k = _conductance_k(grad_sq, conductance)
    if k == 0.0:
        # ITK sets both conductances to zero, so the update vanishes.
        return False
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
    # ITK ApplyUpdate: u += float(dt * double(float(delta))).
    u += (time_step * delta.astype(np.float32).astype(np.float64)).astype(np.float32)
    return True


def _conductance_k(grad_sq: np.ndarray, conductance: float) -> float:
    """ITK's ``K = -2 * conductance**2 * mean(|grad u|^2)``, stored as float32.

    ITK keeps ``m_K`` in the pixel type, so the float64 product is rounded.
    """
    k = float(grad_sq.mean()) * conductance * conductance * -2.0
    return float(np.float32(k))


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


@functools.cache
def _diffusion_kernels():
    """Compile the diffusion kernels once per process.

    ``--fmad=false`` keeps each multiply and add separately rounded, as in ITK's
    scalar C++ loop.
    """
    cp = CUDAManager().get_cp()
    module = cp.RawModule(code=_DIFFUSION_CUDA_SOURCE, options=("--fmad=false",))
    names = ("gad_gradient", "gad_flux", "gad_apply")
    return tuple(module.get_function(name) for name in names)


def _diffusion_step_cuda(
    u: np.ndarray, scale: list[float], conductance: float, time_step: float
) -> bool:
    """Kernel form of :func:`_diffusion_step_xp` for C-contiguous 3D CuPy arrays."""
    xp = get_array_module(u)
    gradient_kernel, flux_kernel, apply_kernel = _diffusion_kernels()
    nz, ny, nx = (np.int64(n) for n in u.shape)
    # ITK dimension order (x, y, z) = array axes reversed.
    sx, sy, sz = (np.float64(scale[axis]) for axis in (2, 1, 0))
    threads = 256
    launch = ((min(int(u.size + threads - 1) // threads, 65535 * 8),), (threads,))
    derivatives = xp.empty((3, *u.shape), dtype=np.float64)
    grad_sq = xp.empty(u.shape, dtype=np.float64)
    gradient_kernel(*launch, (u, derivatives, grad_sq, nz, ny, nx, sx, sy, sz))
    k = _conductance_k(grad_sq, conductance)
    del grad_sq
    if k == 0.0:
        return False
    flux = xp.empty((3, *u.shape), dtype=np.float64)
    flux_kernel(*launch, (u, derivatives, flux, nz, ny, nx, np.float64(k), sx, sy, sz))
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


def _vesselness_2d_response(eigen: np.ndarray, tau: float, width: int) -> np.ndarray:
    """Jerman 2D filament response over the first ``width`` columns.

    The minimum eigenvalue is taken over each whole plane, as in the reference;
    the response is elementwise, so only the kept columns are computed.
    """
    plane_min = eigen.min(axis=(eigen.ndim - 2, eigen.ndim - 1), keepdims=True)
    eigen = eigen[..., :width]
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
    if image.shape[2] <= 3:
        raise ValueError(f"image must be wider than 3 columns, got {image.shape[2]}")
    kept = image.shape[2] - 3
    mip = np.broadcast_to(image.max(axis=0), image.shape)
    stacked = np.concatenate([image, mip], axis=2)
    del mip
    response = None
    for sigma in sigmas:
        r = _vesselness_2d_response(_hessian_2d_eigen_max(stacked, sigma), tau, kept)
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
    inclusive = size_filter_inclusive
    norm = intensity_normalization(image.astype(np.float32, copy=False), (2.5, 7.5))
    smooth = gradient_anisotropic_diffusion(norm)
    del norm
    bw = vesselness_slice_by_slice(smooth, sigmas=[1.0], tau=1.0) > 0.15
    del smooth
    bw = remove_small_objects_aics(bw, 15, inclusive=inclusive)
    bw = remove_small_objects_aics(bw, 3, per_slice=True, inclusive=inclusive)
    return remove_small_objects_aics(bw, 15, inclusive=inclusive)


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
    norm = intensity_normalization(image.astype(np.float32, copy=False), (3.5, 15.0))
    smooth = _gaussian_nearest(norm, [1.0] * norm.ndim)
    del norm
    bw = vesselness_slice_by_slice(smooth, sigmas=[1.5], tau=1.0) > 0.16
    return remove_small_objects_aics(bw, 10, inclusive=size_filter_inclusive)
