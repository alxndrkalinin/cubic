"""GPU-resident Cellpose mask computation: dynamics, flow QC, fill/size, stitch.

Mirrors ``cellpose.dynamics`` / ``cellpose.utils`` post-processing but keeps the
flow field and masks on the GPU as CuPy arrays, bridging to torch *zero-copy*
only for the pure-torch kernels (Euler integration, histogram seeding). The
``cv2``/``fastremap``/``fill_voids``/``scipy`` CPU glue is replaced with cubic's
device-agnostic wrappers (cupyx.scipy.ndimage, cuCIM) so the only device→host
transfer is the final integer mask.

cellpose is never modified; the pure-torch helpers ``max_pool_nd``/``max_pool1d``
are imported and reused as-is.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ..cuda import CUDAManager, ascupy, asnumpy, to_torch
from .cellpose_sam_gpu import resize_gpu

try:  # cellpose is optional; fail only at call time
    from cellpose.dynamics import max_pool_nd

    _CELLPOSE_AVAILABLE = True
except ImportError:  # pragma: no cover
    max_pool_nd = None  # type: ignore[assignment]
    _CELLPOSE_AVAILABLE = False


def _cp():
    """Return the cupy module (raises if unavailable)."""
    cp = CUDAManager().get_cp()
    if cp is None:
        raise RuntimeError("cupy + a CUDA GPU are required for the GPU-resident path.")
    return cp


def _relabel_sequential(masks: Any) -> Any:
    """Consecutive relabel (cupy), the GPU analogue of ``fastremap.renumber``.

    ``cucim.skimage.segmentation.relabel_sequential`` returns a *copy* (unlike
    ``fastremap.renumber(in_place=True)``), so callers must reassign.
    """
    from cucim.skimage.segmentation import relabel_sequential

    return relabel_sequential(masks)[0]


def _filter_small(masks: Any, min_size: int) -> Any:
    """Drop labels with pixel count < ``min_size`` then relabel (cupy)."""
    cp = _cp()
    uniq, counts = cp.unique(masks, return_counts=True)
    small = uniq[1:][counts[1:] < min_size]
    if int(small.size) > 0:
        masks = cp.where(cp.isin(masks, small), 0, masks)
    return _relabel_sequential(masks)


# --------------------------------------------------------------------------- #
# dynamics (mirror dynamics.steps_interp / follow_flows / get_masks_torch)
# --------------------------------------------------------------------------- #
def _steps_interp(dP: Any, inds: tuple, niter: int, device: Any) -> Any:
    """Euler integration of the flow field (torch on GPU), fed CuPy zero-copy.

    Mirrors ``cellpose.dynamics.steps_interp`` but uses ``torch.as_tensor`` (via
    :func:`~cubic.cuda.to_torch`) instead of ``torch.from_numpy`` so CuPy inputs
    stay on the GPU. Returns a torch tensor of final pixel locations.
    """
    import torch

    shape = dP.shape[1:]
    ndim = len(shape)
    pt = torch.zeros(
        (*[1] * ndim, len(inds[0]), ndim), dtype=torch.float32, device=device
    )
    im = torch.zeros((1, ndim, *shape), dtype=torch.float32, device=device)
    for n in range(ndim):
        if ndim == 3:
            pt[0, 0, 0, :, ndim - n - 1] = to_torch(
                inds[n], device=device, dtype=torch.float32
            )
        else:
            pt[0, 0, :, ndim - n - 1] = to_torch(
                inds[n], device=device, dtype=torch.float32
            )
        im[0, ndim - n - 1] = to_torch(dP[n], device=device, dtype=torch.float32)

    shape_arr = np.array(shape)[::-1].astype("float") - 1
    for k in range(ndim):
        im[:, k] *= 2.0 / shape_arr[k]
        pt[..., k] /= shape_arr[k]
    pt *= 2
    pt -= 1
    for _ in range(int(niter)):
        dPt = torch.nn.functional.grid_sample(im, pt, align_corners=False)
        for k in range(ndim):
            pt[..., k] += dPt[:, k]
            torch.clamp_(pt[..., k], -1.0, 1.0)
    pt += 1
    pt *= 0.5
    for k in range(ndim):
        pt[..., k] *= shape_arr[k]

    order = [2, 1, 0] if ndim == 3 else [1, 0]
    pt = pt[..., order].squeeze()
    pt = pt.unsqueeze(0) if pt.ndim == 1 else pt
    return pt.T


def _follow_flows(dP: Any, inds: tuple, niter: int, device: Any) -> Any:
    return _steps_interp(dP, inds, niter, device=device)


def _get_masks(
    pt: Any, inds: tuple, shape0: tuple, max_size_fraction: float = 0.4, rpad: int = 20
) -> Any:
    """Histogram-seed masks from converged pixels (mirror ``get_masks_torch``).

    The torch histogram/maxpool/seed-extension body is reused verbatim; the
    ``.cpu().numpy()`` + ``fastremap`` tail is replaced with a zero-copy bridge
    to CuPy and ``cp.unique``/``cp.isin``/``relabel_sequential``.
    """
    import torch

    cp = _cp()
    ndim = len(shape0)
    device = pt.device

    pt = pt + rpad
    pt = torch.clamp(pt, min=0)
    for i in range(len(pt)):
        pt[i] = torch.clamp(pt[i], max=shape0[i] + rpad - 1)
    shape = tuple(np.array(shape0) + 2 * rpad)

    coo = torch.sparse_coo_tensor(
        pt, torch.ones(pt.shape[1], device=device, dtype=torch.int), shape
    )
    h1 = coo.to_dense()
    del coo

    hmax1 = max_pool_nd(h1.unsqueeze(0), kernel_size=5).squeeze()
    seeds1 = torch.nonzero((h1 - hmax1 > -1e-6) * (h1 > 10))
    del hmax1
    if len(seeds1) == 0:
        return cp.zeros(shape0, dtype="uint16")

    npts = (
        h1[seeds1[:, 0], seeds1[:, 1]]
        if ndim == 2
        else h1[seeds1[:, 0], seeds1[:, 1], seeds1[:, 2]]
    )
    seeds1 = seeds1[npts.argsort()]
    n_seeds = len(seeds1)

    offset_t = torch.arange(-5, 6, device=device)
    inds_t = torch.meshgrid(ndim * [offset_t], indexing="ij")
    if ndim == 2:
        flat_inds = (inds_t[0] * shape[1] + inds_t[1]).flatten()
        flat_inds = flat_inds + (seeds1[:, 0] * shape[1] + seeds1[:, 1])[:, None]
    else:
        flat_inds = (
            inds_t[0] * shape[1] * shape[2] + inds_t[1] * shape[2] + inds_t[2]
        ).flatten()
        flat_inds = (
            flat_inds
            + (
                seeds1[:, 0] * shape[1] * shape[2]
                + seeds1[:, 1] * shape[2]
                + seeds1[:, 2]
            )[:, None]
        )

    h1 = h1.view(-1)
    h_slc = h1[flat_inds].reshape(n_seeds, *[11] * ndim)
    del h1

    seed_masks = torch.zeros((n_seeds, *[11] * ndim), device=device)
    if ndim == 2:
        seed_masks[:, 5, 5] = 1
    else:
        seed_masks[:, 5, 5, 5] = 1
    for _ in range(5):
        seed_masks = max_pool_nd(seed_masks, kernel_size=3)
        seed_masks *= h_slc > 2
    del h_slc

    dtype = torch.int32 if n_seeds < 2**16 else torch.int64
    M1 = torch.zeros(int(np.prod(shape)), device=device, dtype=dtype)
    ipix = torch.nonzero(seed_masks).to(dtype)
    mask_idx = ipix[:, 0]
    mask_pos = ipix[:, 1:] + seeds1[mask_idx] - 5
    if ndim == 2:
        flat = mask_pos[:, 0] * shape[1] + mask_pos[:, 1]
    else:
        flat = (
            mask_pos[:, 0] * shape[1] * shape[2]
            + mask_pos[:, 1] * shape[2]
            + mask_pos[:, 2]
        )
    M1.scatter_reduce_(0, flat, mask_idx + 1, reduce="amax", include_self=False)
    M1 = M1.reshape(shape)
    M1 = M1[pt[0], pt[1]] if ndim == 2 else M1[pt[0], pt[1], pt[2]]

    # zero-copy bridge to cupy; finish on GPU
    M1 = ascupy(M1)
    out_dtype = cp.uint16 if n_seeds < 2**16 else cp.uint32
    M0 = cp.zeros(shape0, dtype=out_dtype)
    M0[inds] = M1

    uniq, counts = cp.unique(M0, return_counts=True)
    big = np.prod(shape0) * max_size_fraction
    bigc = uniq[counts > big]
    if int(bigc.size) > 0 and (int(bigc.size) > 1 or int(bigc[0]) != 0):
        M0 = cp.where(cp.isin(M0, bigc), 0, M0)
    M0 = _relabel_sequential(M0).reshape(tuple(shape0))
    return M0


# --------------------------------------------------------------------------- #
# flow QC (mirror dynamics.masks_to_flows_gpu / flow_error / remove_bad_flow_masks)
# --------------------------------------------------------------------------- #
def _extend_centers(
    neighbors: Any, meds: Any, isneighbor: Any, shape: tuple, n_iter: int, device: Any
) -> Any:
    """Heat diffusion from mask centers (torch), returning CuPy flows.

    Mirrors ``dynamics._extend_centers_gpu`` but returns a CuPy array instead of
    ``.cpu().numpy()`` so the downstream normalization/scatter stay on the GPU.
    """
    import torch

    big = np.prod(shape) > 4e7 or device.type == "mps"
    dtype = torch.float32 if big else torch.float64
    T_flat = torch.zeros(int(np.prod(shape)), dtype=dtype, device=device)
    ndim = len(shape)
    if ndim == 2:
        _, Lx = shape
        flat_n = (neighbors[0] * Lx + neighbors[1]).long()
        flat_m = (meds[:, 0] * Lx + meds[:, 1]).long()
    else:
        _, Ly, Lx = shape
        flat_n = (neighbors[0] * (Ly * Lx) + neighbors[1] * Lx + neighbors[2]).long()
        flat_m = (meds[:, 0] * (Ly * Lx) + meds[:, 1] * Lx + meds[:, 2]).long()
    flat_center = flat_n[0]
    nneigh = flat_n.shape[0]
    for _ in range(int(n_iter)):
        T_flat[flat_m] += 1
        Tneigh = T_flat[flat_n]
        T_flat[flat_center] = (Tneigh * isneighbor).sum(dim=0) / nneigh
    if ndim == 2:
        grads = T_flat[flat_n[[2, 1, 4, 3]]]
        mu = torch.stack((grads[0] - grads[1], grads[2] - grads[3]))
    else:
        grads = T_flat[flat_n[1:]]
        mu = torch.stack(
            (grads[0] - grads[1], grads[2] - grads[3], grads[4] - grads[5])
        )
    return ascupy(mu)


def _mask_centers_2d(masks: Any, idx: Any) -> tuple[Any, float]:
    """Per-label centroid snapped to the nearest in-mask pixel, + max extent (GPU)."""
    cp = _cp()
    from cupyx.scipy import ndimage as cnd

    Ly, Lx = masks.shape
    yy, xx = cp.meshgrid(
        cp.arange(Ly, dtype=cp.float32), cp.arange(Lx, dtype=cp.float32), indexing="ij"
    )
    cy = cp.asarray(cnd.mean(yy, masks, idx))
    cx = cp.asarray(cnd.mean(xx, masks, idx))
    nmax = int(masks.max()) + 1
    cyf = cp.zeros(nmax, cp.float32)
    cxf = cp.zeros(nmax, cp.float32)
    cyf[idx] = cy
    cxf[idx] = cx
    dist = (yy - cyf[masks]) ** 2 + (xx - cxf[masks]) ** 2
    pos = cnd.minimum_position(dist, masks, idx)
    centers = cp.asarray([[int(p[0]), int(p[1])] for p in pos], dtype=cp.int64)
    ext = float(
        (
            cp.asarray(cnd.maximum(yy, masks, idx))
            - cp.asarray(cnd.minimum(yy, masks, idx))
            + cp.asarray(cnd.maximum(xx, masks, idx))
            - cp.asarray(cnd.minimum(xx, masks, idx))
            + 4
        ).max()
    )
    # cellpose get_centers: ext = span_y + span_x + 2 with span = (max-min+1);
    # on raw (max-min) coordinate differences that constant becomes +4.
    return centers, ext


def _masks_to_flows_gpu(masks: Any, device: Any, niter: int | None = None) -> Any:
    """2D mask→flow diffusion, fully on GPU (mirror ``masks_to_flows_gpu``)."""
    import torch
    import torch.nn.functional as F

    cp = _cp()
    Ly0, Lx0 = masks.shape
    if int(masks.max()) == 0:
        return cp.zeros((2, Ly0, Lx0))

    masks_padded = F.pad(to_torch(masks.astype(cp.int64), device=device), (1, 1, 1, 1))
    shape = tuple(masks_padded.shape)
    y, x = torch.nonzero(masks_padded, as_tuple=True)
    y = y.int()
    x = x.int()
    neighbors = torch.zeros((2, 9, y.shape[0]), dtype=torch.int, device=device)
    yxi = [[0, -1, 1, 0, 0, -1, -1, 1, 1], [0, 0, 0, -1, 1, -1, 1, -1, 1]]
    for i in range(9):
        neighbors[0, i] = y + yxi[0][i]
        neighbors[1, i] = x + yxi[1][i]
    isneighbor = torch.ones((9, y.shape[0]), dtype=torch.bool, device=device)
    m0 = masks_padded[neighbors[0, 0], neighbors[1, 0]]
    for i in range(1, 9):
        isneighbor[i] = masks_padded[neighbors[0, i], neighbors[1, i]] == m0
    del m0, masks_padded

    idx = cp.arange(1, int(masks.max()) + 1)
    centers, ext = _mask_centers_2d(masks, idx)
    meds_p = to_torch(centers, device=device).long() + 1
    n_iter = int(2 * ext) if niter is None else int(niter)

    mu = _extend_centers(neighbors, meds_p, isneighbor, shape, n_iter, device).astype(
        cp.float64
    )
    mu /= 1e-60 + (mu**2).sum(axis=0) ** 0.5
    mu0 = cp.zeros((2, Ly0, Lx0))
    mu0[:, ascupy(y) - 1, ascupy(x) - 1] = mu
    return mu0


def _flow_error(maski: Any, dP_net: Any, device: Any) -> tuple[Any, Any]:
    """Per-mask flow error (mirror ``dynamics.flow_error``), GPU labeled-mean."""
    cp = _cp()
    from cupyx.scipy import ndimage as cnd

    dP_masks = _masks_to_flows_gpu(maski, device=device)
    err = ((dP_masks - dP_net / 5.0) ** 2).sum(axis=0)
    n = int(maski.max())
    flow_errors = cp.asarray(cnd.mean(err, maski, cp.arange(1, n + 1)))
    return flow_errors, dP_masks


def _remove_bad_flow_masks(
    masks: Any, flows: Any, threshold: float, device: Any
) -> Any:
    """Discard masks whose flows disagree with the network (mirror cellpose).

    For very large masks where the on-device flow recomputation would not fit
    in free VRAM, fall back to cellpose's CPU flow-QC (mirrors the
    ``masks.size > 1e8`` guard in ``dynamics.remove_bad_flow_masks``). This
    is the one place a derived array may leave the GPU, and only to avoid an
    out-of-memory error on a huge image.
    """
    cp = _cp()
    import torch

    if masks.size > 10000 * 10000 and device is not None and device.type == "cuda":
        torch.cuda.empty_cache()
        free_mem, _total = torch.cuda.mem_get_info(device.index)
        if masks.size * 32 > free_mem:
            warnings.warn(
                "image is very large; computing flow-QC on CPU to avoid GPU OOM "
                "(set flow_threshold=0 to skip this step)."
            )
            from cellpose.dynamics import remove_bad_flow_masks as _cpu_remove

            out = _cpu_remove(
                asnumpy(masks),
                asnumpy(flows),
                threshold=threshold,
                device=torch.device("cpu"),
            )
            return ascupy(out)

    merrors, _ = _flow_error(masks, flows, device)
    badi = 1 + cp.nonzero(merrors > threshold)[0]
    masks = cp.where(cp.isin(masks, badi), 0, masks)
    return _relabel_sequential(masks)


# --------------------------------------------------------------------------- #
# fill holes + size filter (replaces utils.fill_holes_and_remove_small_masks)
# --------------------------------------------------------------------------- #
def _fill_holes_and_size_filter(masks: Any, min_size: int = 15) -> Any:
    """Fill per-label holes + drop small masks, fully on GPU.

    ``fill_voids`` (CPU-only) is replaced by per-object
    ``cupyx.scipy.ndimage.binary_fill_holes`` over ``find_objects`` slices;
    ``fastremap`` size-filtering by ``cp.unique``/``cp.isin``/relabel.
    """
    from cupyx.scipy import ndimage as cnd

    if masks.ndim > 3 or masks.ndim < 2:
        raise ValueError(f"expected 2D or 3D masks, got {masks.ndim}D")

    if min_size > 0:
        masks = _filter_small(masks, min_size)

    slices = cnd.find_objects(masks)
    for i, slc in enumerate(slices):
        if slc is not None:
            msk = masks[slc] == (i + 1)
            filled = cnd.binary_fill_holes(msk)
            masks[slc][filled] = i + 1

    if min_size > 0:
        masks = _filter_small(masks, min_size)
    return masks


# --------------------------------------------------------------------------- #
# stitching (mirror utils.stitch3D, GPU IoU)
# --------------------------------------------------------------------------- #
def _stitch3D(masks: Any, stitch_threshold: float = 0.25) -> Any:
    """Stitch per-plane 2D masks into a 3D volume by IoU (GPU)."""
    cp = _cp()
    from ..metrics.average_precision import _intersection_over_union

    mmax = int(masks[0].max())
    empty = 0
    for i in range(len(masks) - 1):
        iou = _intersection_over_union(masks[i + 1], masks[i])[1:, 1:]
        if int(iou.size) == 0 and empty == 0:
            mmax = int(masks[i + 1].max())
        elif int(iou.size) == 0 and empty != 0:
            icount = int(masks[i + 1].max())
            istitch = cp.arange(mmax + 1, mmax + icount + 1, dtype=masks.dtype)
            mmax += icount
            istitch = cp.concatenate((cp.array([0], dtype=masks.dtype), istitch))
            masks[i + 1] = istitch[masks[i + 1]]
        else:
            iou[iou < stitch_threshold] = 0.0
            iou[iou < iou.max(axis=0)] = 0.0
            istitch = iou.argmax(axis=1) + 1
            ino = cp.nonzero(iou.max(axis=1) == 0.0)[0]
            istitch[ino] = cp.arange(
                mmax + 1, mmax + int(ino.size) + 1, dtype=istitch.dtype
            )
            mmax += int(ino.size)
            istitch = cp.concatenate((cp.array([0], dtype=istitch.dtype), istitch))
            masks[i + 1] = istitch[masks[i + 1]]
            empty = 1
    return masks


# --------------------------------------------------------------------------- #
# orchestration (mirror dynamics.compute_masks / resize_and_compute_masks and
# CellposeModel._compute_masks)
# --------------------------------------------------------------------------- #
def _compute_masks_single(
    dP: Any,
    cellprob: Any,
    niter: int = 200,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    do_3D: bool = False,
    max_size_fraction: float = 0.4,
    device: Any = None,
) -> Any:
    """Mirror ``dynamics.compute_masks`` (no size filter; that lives in resize step)."""
    cp = _cp()
    if int((cellprob > cellprob_threshold).sum()) == 0:
        return cp.zeros(cellprob.shape, dtype=cp.uint16)
    inds = cp.nonzero(cellprob > cellprob_threshold)
    if int(inds[0].size) == 0:
        return cp.zeros(cellprob.shape, dtype=cp.uint16)

    p_final = _follow_flows(
        dP * (cellprob > cellprob_threshold) / 5.0, inds, niter, device
    )
    p_final = p_final.int()
    mask = _get_masks(p_final, inds, dP.shape[1:], max_size_fraction=max_size_fraction)
    del p_final

    if (
        not do_3D
        and int(mask.max()) > 0
        and flow_threshold is not None
        and flow_threshold > 0
    ):
        mask = _remove_bad_flow_masks(mask, dP, threshold=flow_threshold, device=device)

    if int(mask.max()) < 2**16 and mask.dtype != cp.uint16:
        mask = mask.astype(cp.uint16)
    return mask


def _resize_and_compute_masks(
    dP: Any,
    cellprob: Any,
    niter: int = 200,
    cellprob_threshold: float = 0.0,
    flow_threshold: float = 0.4,
    do_3D: bool = False,
    min_size: int = 15,
    max_size_fraction: float = 0.4,
    resize: Any = None,
    device: Any = None,
) -> Any:
    """Mirror ``dynamics.resize_and_compute_masks`` (cucim nearest resize)."""
    mask = _compute_masks_single(
        dP,
        cellprob,
        niter=niter,
        cellprob_threshold=cellprob_threshold,
        flow_threshold=flow_threshold,
        do_3D=do_3D,
        max_size_fraction=max_size_fraction,
        device=device,
    )

    if resize is not None:
        if len(resize) == 2:
            mask = resize_gpu(mask, resize[0], resize[1], order=0, no_channels=True)
        else:
            Lz, Ly, Lx = resize
            if mask.shape[0] != Lz or mask.shape[1] != Ly:
                if mask.shape[1] != Ly:
                    mask = resize_gpu(mask, Ly=Ly, Lx=Lx, order=0, no_channels=True)
                if mask.shape[0] != Lz:
                    mask = resize_gpu(
                        mask.transpose(1, 0, 2), Ly=Lz, Lx=Lx, order=0, no_channels=True
                    ).transpose(1, 0, 2)

    mask = _fill_holes_and_size_filter(mask, min_size=min_size)
    return mask


def compute_masks(
    shape: tuple,
    dP: Any,
    cellprob: Any,
    *,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_size: int = 15,
    max_size_fraction: float = 0.4,
    niter: int | None = None,
    do_3D: bool = False,
    stitch_threshold: float = 0.0,
    device: Any = None,
) -> Any:
    """GPU-resident ``CellposeModel._compute_masks``; returns a CuPy label array.

    ``dP`` and ``cellprob`` are CuPy arrays on the GPU; the returned mask stays
    on the GPU (the caller does the single device→host transfer).
    """
    cp = _cp()
    niter = 200 if niter is None else niter
    Lz, Ly, Lx = shape[:3]
    if do_3D:
        diff = int((np.array(dP.shape[-3:]) != np.array(shape[:3])).sum())
        masks = _resize_and_compute_masks(
            dP,
            cellprob,
            niter=niter,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            do_3D=True,
            min_size=min_size,
            max_size_fraction=max_size_fraction,
            resize=shape[:3] if diff else None,
            device=device,
        )
        return masks

    nimg = shape[0]
    Ly0, Lx0 = cellprob[0].shape
    resize = None if (Ly0 == Ly and Lx0 == Lx) else [Ly, Lx]
    masks = None
    for i in range(nimg):
        min_size0 = min_size if (stitch_threshold == 0 or nimg == 1) else -1
        out = _resize_and_compute_masks(
            dP[:, i],
            cellprob[i],
            niter=niter,
            cellprob_threshold=cellprob_threshold,
            flow_threshold=flow_threshold,
            resize=resize,
            min_size=min_size0,
            max_size_fraction=max_size_fraction,
            device=device,
        )
        if i == 0 and nimg > 1:
            masks = cp.zeros((nimg, shape[1], shape[2]), out.dtype)
        if nimg > 1:
            masks[i] = out
        else:
            masks = out

    if stitch_threshold > 0 and nimg > 1:
        masks = _stitch3D(masks, stitch_threshold=stitch_threshold)
        masks = _fill_holes_and_size_filter(masks, min_size=min_size)
    return masks
