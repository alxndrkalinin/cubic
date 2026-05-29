"""GPU-resident Cellpose-SAM inference: preprocessing, tiling and network forward.

This module mirrors ``cellpose.transforms`` / ``cellpose.core`` but keeps every
array on the GPU as a CuPy array, bridging to torch *zero-copy* only for the
network forward. The input image is uploaded to the device exactly once
(:func:`segment_cpsam`) and only the final integer mask returns to the
host (mask computation lives in :mod:`cubic.segmentation.cellpose_dynamics`).

cellpose itself is never modified. Heavy CPU dependencies are replaced with
cubic's device-agnostic wrappers: ``cv2.resize`` → :func:`resize_gpu`
(``cubic.skimage.transform.resize`` → cuCIM on GPU); host buffer allocations
(``np.zeros``) → ``xp.zeros`` via :func:`~cubic.cuda.get_array_module`.

The orchestrator is GPU-only by contract (the whole point is GPU residency);
the building-block transforms remain device-agnostic and run on NumPy too.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..cuda import (
    CUDAManager,
    ascupy,
    asnumpy,
    to_torch,
    ascupy_f32,
    get_device,
    get_array_module,
)
from ..skimage import transform as _sk_transform

try:  # cellpose is an optional dependency; fail only at call time
    from cellpose import transforms as _cp_transforms

    _CELLPOSE_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised in no-cellpose envs
    _cp_transforms = None  # type: ignore[assignment]
    _CELLPOSE_AVAILABLE = False


def _require_cellpose() -> None:
    if not _CELLPOSE_AVAILABLE:
        raise ImportError(
            "cellpose>=4 (SAM) is required for the GPU-resident path. "
            "Install with `pip install cubic[cellpose]`."
        )


# --------------------------------------------------------------------------- #
# device-aware resize (replaces cv2-based transforms.resize_image/resize_safe)
# --------------------------------------------------------------------------- #
def _resize2d(img: Any, Ly: int, Lx: int, order: int, no_channels: bool) -> Any:
    """Resize a single 2D (optionally multi-channel) image, preserving dtype."""
    out_shape = (Ly, Lx) if (img.ndim == 2 or no_channels) else (Ly, Lx, img.shape[-1])
    res = _sk_transform.resize(
        img,
        out_shape,
        order=order,
        mode="constant",
        anti_aliasing=False,
        preserve_range=True,
    )
    return res.astype(img.dtype)


def resize_gpu(
    img: Any,
    Ly: int | None = None,
    Lx: int | None = None,
    rsz: Any = None,
    order: int = 1,
    no_channels: bool = False,
) -> Any:
    """Device-aware replacement for ``cellpose.transforms.resize_image``.

    Uses ``cubic.skimage.transform.resize`` (cuCIM on GPU, scikit-image on CPU)
    instead of cv2. ``order=1`` is bilinear (images), ``order=0`` nearest
    (label masks, with integer dtype preserved). Handles ``[Y,X,C]``,
    ``[Z,Y,X,C]`` and ``[Z,Y,X]``/``[Y,X]`` (``no_channels=True``).
    """
    xp = get_array_module(img)
    if Ly is None and rsz is None:
        raise ValueError("must give size to resize to or factor to use for resizing")
    if Ly is None:
        if not isinstance(rsz, (list, np.ndarray)):
            rsz = [rsz, rsz]
        if no_channels:
            Ly, Lx = int(img.shape[-2] * rsz[-2]), int(img.shape[-1] * rsz[-1])
        else:
            Ly, Lx = int(img.shape[-3] * rsz[-2]), int(img.shape[-2] * rsz[-1])
    assert Ly is not None and Lx is not None

    # per-slice loop for stacks: [Z,Y,X] (no_channels) or [Z,Y,X,C]
    if (img.ndim == 3 and no_channels) or (img.ndim == 4 and not no_channels):
        if Ly == 0 or Lx == 0:
            raise ValueError(
                "diameter too high -- not enough pixels to resize to ratio"
            )
        out = None
        for i in range(img.shape[0]):
            ri = _resize2d(img[i], Ly, Lx, order, no_channels)
            if out is None:
                shp = (
                    (img.shape[0], Ly, Lx)
                    if no_channels
                    else (img.shape[0], Ly, Lx, img.shape[-1])
                )
                out = xp.zeros(shp, ri.dtype)
            out[i] = ri if (ri.ndim > 2 or no_channels) else ri[..., None]
        return out
    return _resize2d(img, Ly, Lx, order, no_channels)


def resize_image_3d_gpu(
    img: Any, shape: tuple[int, int, int], order: int = 1, no_channels: bool = False
) -> Any:
    """Device-aware ``cellpose.transforms.resize_image_3d`` (cv2-free)."""
    Lzr, Lyr, Lxr = shape
    Lz, Ly, Lx = img.shape[:3]
    img_rsz = img
    if Lyr != Ly or Lxr != Lx:
        img_rsz = resize_gpu(
            img, Ly=Lyr, Lx=Lxr, order=order, no_channels=no_channels
        ).astype(img.dtype)
    if Lzr != Lz:
        tt = (1, 0, 2) if no_channels else (1, 0, 2, 3)
        img_rsz = (
            resize_gpu(
                img_rsz.transpose(tt),
                Ly=Lzr,
                Lx=Lxr,
                order=order,
                no_channels=no_channels,
            )
            .astype(img.dtype)
            .transpose(tt)
        )
    return img_rsz


# --------------------------------------------------------------------------- #
# tiling (mirror transforms._taper_mask / make_tiles / average_tiles / unaugment)
# --------------------------------------------------------------------------- #
def _taper_mask(xp: Any, ly: int = 224, lx: int = 224, sig: float = 7.5) -> Any:
    bsize = max(224, max(ly, lx))
    xm = xp.arange(bsize)
    xm = xp.abs(xm - xm.mean())
    mask = 1 / (1 + xp.exp((xm - (bsize / 2 - 20)) / sig))
    mask = mask * mask[:, None]
    return mask[
        bsize // 2 - ly // 2 : bsize // 2 + ly // 2 + ly % 2,
        bsize // 2 - lx // 2 : bsize // 2 + lx // 2 + lx % 2,
    ]


def unaugment_tiles(y: Any) -> Any:
    """Reverse test-time flips for tile averaging (``np.flip`` dispatches to cupy)."""
    for j in range(y.shape[0]):
        for i in range(y.shape[1]):
            if j % 2 == 0 and i % 2 == 1:
                y[j, i] = np.flip(y[j, i], axis=-2)
                y[j, i, 0] *= -1
            elif j % 2 == 1 and i % 2 == 0:
                y[j, i] = np.flip(y[j, i], axis=-1)
                y[j, i, 1] *= -1
            elif j % 2 == 1 and i % 2 == 1:
                y[j, i] = np.flip(y[j, i], axis=(-2, -1))
                y[j, i, 0] *= -1
                y[j, i, 1] *= -1
    return y


def make_tiles(
    imgi: Any, bsize: int = 224, augment: bool = False, tile_overlap: float = 0.1
) -> tuple[Any, list, list, int, int]:
    """Device-aware ``cellpose.transforms.make_tiles`` (xp-allocated buffers).

    Flips in the augment branch write into the freshly allocated (contiguous)
    ``IMG`` buffer, so the batch handed to torch is contiguous.
    """
    xp = get_array_module(imgi)
    nchan, Ly, Lx = imgi.shape
    if augment:
        bsize = int(bsize)
        if Ly < bsize:
            imgi = xp.concatenate(
                (imgi, xp.zeros((nchan, bsize - Ly, Lx), imgi.dtype)), axis=1
            )
            Ly = bsize
        if Lx < bsize:
            imgi = xp.concatenate(
                (imgi, xp.zeros((nchan, Ly, bsize - Lx), imgi.dtype)), axis=2
            )
        Ly, Lx = imgi.shape[-2:]
        ny = max(2, int(np.ceil(2.0 * Ly / bsize)))
        nx = max(2, int(np.ceil(2.0 * Lx / bsize)))
        ystart = np.linspace(0, Ly - bsize, ny).astype(int)
        xstart = np.linspace(0, Lx - bsize, nx).astype(int)
        ysub, xsub = [], []
        IMG = xp.zeros((len(ystart), len(xstart), nchan, bsize, bsize), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j] + bsize])
                xsub.append([xstart[i], xstart[i] + bsize])
                IMG[j, i] = imgi[
                    :, ysub[-1][0] : ysub[-1][1], xsub[-1][0] : xsub[-1][1]
                ]
                if j % 2 == 0 and i % 2 == 1:
                    IMG[j, i] = np.flip(IMG[j, i], axis=-2)
                elif j % 2 == 1 and i % 2 == 0:
                    IMG[j, i] = np.flip(IMG[j, i], axis=-1)
                elif j % 2 == 1 and i % 2 == 1:
                    IMG[j, i] = np.flip(IMG[j, i], axis=(-2, -1))
    else:
        tile_overlap = min(0.5, max(0.05, tile_overlap))
        bsizeY, bsizeX = int(min(bsize, Ly)), int(min(bsize, Lx))
        ny = 1 if Ly <= bsize else int(np.ceil((1.0 + 2 * tile_overlap) * Ly / bsize))
        nx = 1 if Lx <= bsize else int(np.ceil((1.0 + 2 * tile_overlap) * Lx / bsize))
        ystart = np.linspace(0, Ly - bsizeY, ny).astype(int)
        xstart = np.linspace(0, Lx - bsizeX, nx).astype(int)
        ysub, xsub = [], []
        IMG = xp.zeros((len(ystart), len(xstart), nchan, bsizeY, bsizeX), np.float32)
        for j in range(len(ystart)):
            for i in range(len(xstart)):
                ysub.append([ystart[j], ystart[j] + bsizeY])
                xsub.append([xstart[i], xstart[i] + bsizeX])
                IMG[j, i] = imgi[
                    :, ysub[-1][0] : ysub[-1][1], xsub[-1][0] : xsub[-1][1]
                ]
    return IMG, ysub, xsub, Ly, Lx


def average_tiles(y: Any, ysub: list, xsub: list, Ly: int, Lx: int) -> Any:
    """Device-aware ``cellpose.transforms.average_tiles`` (taper-weighted mean)."""
    xp = get_array_module(y)
    Navg = xp.zeros((Ly, Lx), np.float32)
    yf = xp.zeros((y.shape[1], Ly, Lx), np.float32)
    mask = _taper_mask(xp, ly=y.shape[-2], lx=y.shape[-1])
    for j in range(len(ysub)):
        yf[:, ysub[j][0] : ysub[j][1], xsub[j][0] : xsub[j][1]] += y[j] * mask
        Navg[ysub[j][0] : ysub[j][1], xsub[j][0] : xsub[j][1]] += mask
    yf /= Navg
    return yf


# --------------------------------------------------------------------------- #
# network forward (mirror core._forward / run_net / run_3D), GPU-resident
# --------------------------------------------------------------------------- #
def _forward_gpu(net: Any, x: Any) -> tuple[Any, Any]:
    """Run the torch net on a CuPy tile batch, returning CuPy fp32 outputs.

    Bridges zero-copy: CuPy fp32 → torch (cast to ``net.dtype``, e.g. bf16,
    on-device) → forward → cast back to fp32 → CuPy. ``ascontiguousarray``
    guards against a negative-stride view reaching ``torch.as_tensor``.
    """
    import torch

    xp = get_array_module(x)
    X = to_torch(xp.ascontiguousarray(x), device=net.device, dtype=net.dtype)
    net.eval()
    with torch.no_grad():
        y, style = net(X)[:2]
    return ascupy_f32(y), ascupy_f32(style)


def run_net_gpu(
    net: Any,
    imgi: Any,
    batch_size: int = 8,
    augment: bool = False,
    tile_overlap: float = 0.1,
    bsize: int = 224,
    rsz: Any = None,
) -> tuple[Any, Any]:
    """Device-aware ``cellpose.core.run_net`` (tiles stay on GPU)."""
    _require_cellpose()
    xp = get_array_module(imgi)
    Lz, Ly0, Lx0, nchan = imgi.shape
    if rsz is not None:
        if not isinstance(rsz, (list, np.ndarray)):
            rsz = [rsz, rsz]
        Lyr, Lxr = int(Ly0 * rsz[0]), int(Lx0 * rsz[1])
    else:
        Lyr, Lxr = Ly0, Lx0

    ly, lx = bsize, bsize
    ypad1, ypad2, xpad1, xpad2 = _cp_transforms.get_pad_yx(
        Lyr, Lxr, min_size=(bsize, bsize)
    )
    Ly, Lx = Lyr + ypad1 + ypad2, Lxr + xpad1 + xpad2
    pads = np.array([[0, 0], [ypad1, ypad2], [xpad1, xpad2]])

    if augment:
        ny = max(2, int(np.ceil(2.0 * Ly / bsize)))
        nx = max(2, int(np.ceil(2.0 * Lx / bsize)))
    else:
        ny = 1 if Ly <= bsize else int(np.ceil((1.0 + 2 * tile_overlap) * Ly / bsize))
        nx = 1 if Lx <= bsize else int(np.ceil((1.0 + 2 * tile_overlap) * Lx / bsize))

    ntiles = ny * nx
    nimgs = max(1, batch_size // ntiles)
    niter = int(np.ceil(Lz / nimgs))
    yf = None
    styles = None
    for k in range(niter):
        inds = np.arange(k * nimgs, min(Lz, (k + 1) * nimgs))
        IMGa = xp.zeros((ntiles * len(inds), nchan, ly, lx), np.float32)
        imgb_shape = None
        for i, b in enumerate(inds):
            imgb = resize_gpu(imgi[b], rsz=rsz) if rsz is not None else imgi[b].copy()
            imgb = np.pad(imgb.transpose(2, 0, 1), pads, mode="constant")
            imgb_shape = imgb.shape
            IMG, ysub, xsub, Lyt, Lxt = make_tiles(
                imgb, bsize=bsize, augment=augment, tile_overlap=tile_overlap
            )
            IMGa[i * ntiles : (i + 1) * ntiles] = np.reshape(
                IMG, (ny * nx, nchan, ly, lx)
            )

        ya = None
        for j in range(0, IMGa.shape[0], batch_size):
            bslc = slice(j, min(j + batch_size, IMGa.shape[0]))
            # CPSAM emits zero style vectors (matching stock eval), so styles
            # stay the allocated zeros; only the flow output is accumulated.
            ya0, _style0 = _forward_gpu(net, IMGa[bslc])
            if ya is None:
                nout = ya0.shape[1]
                ya = xp.zeros((IMGa.shape[0], nout, ly, lx), np.float32)
            ya[bslc] = ya0

        assert imgb_shape is not None and ya is not None
        for i, b in enumerate(inds):
            if yf is None:
                yf = xp.zeros((Lz, nout, Ly, Lx), np.float32)
                styles = xp.zeros((Lz, 256), np.float32)
            y = ya[i * ntiles : (i + 1) * ntiles]
            if augment:
                y = np.reshape(y, (ny, nx, nout, ly, lx))
                y = unaugment_tiles(y)
                y = np.reshape(y, (-1, nout, ly, lx))
            yfi = average_tiles(y, ysub, xsub, Lyt, Lxt)
            yf[b] = yfi[:, : imgb_shape[-2], : imgb_shape[-1]]

    assert yf is not None
    yf = yf[:, :, ypad1 : Ly - ypad2, xpad1 : Lx - xpad2]
    yf = yf.transpose(0, 2, 3, 1)
    return yf, styles


def run_3D_gpu(
    net: Any,
    imgs: Any,
    batch_size: int = 8,
    augment: bool = False,
    tile_overlap: float = 0.1,
    bsize: int = 224,
) -> tuple[Any, Any]:
    """Device-aware ``cellpose.core.run_3D`` (three orthogonal sweeps on GPU)."""
    xp = get_array_module(imgs)
    pm = [(0, 1, 2, 3), (1, 0, 2, 3), (2, 0, 1, 3)]
    ipm = [(0, 1, 2), (1, 0, 2), (1, 2, 0)]
    cp_ax = [(1, 2), (0, 2), (0, 1)]
    cpy = [(0, 1), (0, 1), (0, 1)]
    shape = imgs.shape[:-1]
    yf = xp.zeros((*shape, 4), np.float32)
    style = None
    for p in range(3):
        xsl = imgs.transpose(pm[p])
        y, style = run_net_gpu(
            net,
            xsl,
            batch_size=batch_size,
            augment=augment,
            bsize=bsize,
            tile_overlap=tile_overlap,
            rsz=None,
        )
        yf[..., -1] += y[..., -1].transpose(ipm[p])
        for j in range(2):
            yf[..., cp_ax[p][j]] += y[..., cpy[p][j]].transpose(ipm[p])
        del y
    return yf, style


def _run_net_gpu(
    net: Any,
    x: Any,
    backbone: str = "sam_vitl",
    rescale: float = 1.0,
    resample: bool = True,
    augment: bool = False,
    batch_size: int = 8,
    tile_overlap: float = 0.1,
    bsize: int | None = None,
    anisotropy: float | None = 1.0,
    do_3D: bool = False,
) -> tuple[Any, Any, Any]:
    """Device-aware ``cellpose.models.CellposeModel._run_net``.

    Owns the dP/cellprob split, the flow transpose, the 3D channel reassembly
    and the resample/anisotropy resize that ``run_net``/``run_3D`` do not.
    """
    shape = x.shape
    if bsize is None:
        bsize = 256 if backbone == "sam_vitl" else 384

    if do_3D:
        Lz, Ly, Lx = shape[:-1]
        if rescale != 1.0 or (anisotropy is not None and anisotropy != 1.0):
            anisotropy = 1.0 if anisotropy is None else anisotropy
            new_shape = (
                int(Lz * anisotropy * rescale),
                int(Ly * rescale),
                int(Lx * rescale),
            )
            x = resize_image_3d_gpu(x, new_shape, no_channels=False)
        yf, styles = run_3D_gpu(
            net,
            x,
            batch_size=batch_size,
            augment=augment,
            tile_overlap=tile_overlap,
            bsize=bsize,
        )
        if resample and (rescale != 1.0 or Lz != yf.shape[0]):
            yf = resize_image_3d_gpu(yf, shape[:-1], no_channels=False)
        cellprob = yf[..., -1]
        dP = yf[..., :-1].transpose((3, 0, 1, 2))
    else:
        yf, styles = run_net_gpu(
            net,
            x,
            bsize=bsize,
            augment=augment,
            batch_size=batch_size,
            tile_overlap=tile_overlap,
            rsz=rescale if rescale != 1.0 else None,
        )
        if resample and rescale != 1.0:
            yf = resize_gpu(yf, shape[1], shape[2])
        cellprob = yf[..., -1]
        dP = yf[..., -3:-1].transpose((3, 0, 1, 2))

    return dP, cellprob, styles.squeeze()


# --------------------------------------------------------------------------- #
# normalization (reuse cellpose's percentile path, which dispatches on cupy)
# --------------------------------------------------------------------------- #
_NORMALIZE_DEFAULT = {
    "lowhigh": None,
    "percentile": None,
    "normalize": True,
    "norm3D": True,
    "sharpen_radius": 0,
    "smooth_radius": 0,
    "tile_norm_blocksize": 0,
    "tile_norm_smooth3D": 1,
    "invert": False,
}


def normalize_img_gpu(img: Any, **params: Any) -> Any:
    """Percentile normalization on GPU by reusing ``cellpose.transforms.normalize_img``.

    The default cellpose-SAM normalization (percentile, no tile-norm/sharpen)
    is pure ``np.``/``np.percentile``/``np.ptp`` and dispatches to cupy
    unchanged (verified to match the CPU result to ~1e-9). The tile-norm and
    sharpen/smooth sub-paths use cv2/``np.array(list)`` and are not yet ported
    to the GPU-resident path.
    """
    _require_cellpose()
    if params.get("tile_norm_blocksize", 0):
        raise NotImplementedError(
            "tile_norm_blocksize>0 is not supported on the GPU-resident path yet"
        )
    if params.get("sharpen_radius", 0) or params.get("smooth_radius", 0):
        raise NotImplementedError(
            "sharpen/smooth normalization is not supported on the GPU-resident path yet"
        )
    return _cp_transforms.normalize_img(img, **params)


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
def _check_gpu_precondition(model: Any) -> None:
    """GPU-only contract: require CUDA + cupy + a SAM (v4) model on cuda."""
    if CUDAManager().get_cp() is None:
        raise RuntimeError(
            "segment_cpsam requires cupy + a CUDA GPU; use CellposeModel.eval for CPU."
        )
    if getattr(model, "device", None) is None or model.device.type != "cuda":
        raise RuntimeError(
            "segment_cpsam requires a model on a CUDA device "
            f"(got device={getattr(model, 'device', None)})."
        )
    if not hasattr(model, "backbone") or not hasattr(
        getattr(model, "net", None), "dtype"
    ):
        raise RuntimeError(
            "segment_cpsam requires cellpose>=4 (SAM): "
            "model.backbone / model.net.dtype not found."
        )


def segment_cpsam(
    model: Any,
    x: Any,
    *,
    batch_size: int = 8,
    resample: bool = True,
    normalize: bool | dict = True,
    channel_axis: int | None = None,
    z_axis: int | None = None,
    diameter: float | None = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    do_3D: bool = False,
    anisotropy: float | None = None,
    flow3D_smooth: float | list | int = 0,
    stitch_threshold: float = 0.0,
    min_size: int = 15,
    max_size_fraction: float = 0.4,
    niter: int | None = None,
    augment: bool = False,
    tile_overlap: float = 0.1,
    download: bool = False,
) -> tuple[Any, list, Any]:
    """Segment an image with Cellpose-SAM, GPU-resident (2D / 3D / stitch).

    This is cubic's recommended Cellpose-SAM segmentation entry point: the
    input is uploaded to the GPU once and only the final integer mask returns
    to the host (default). It mirrors ``CellposeModel.eval`` but keeps the flow
    field on the device, replacing the CPU pre/post-processing with cubic's
    device-agnostic wrappers.

    Args:
        model: A pre-built ``cellpose.models.CellposeModel`` (v4/SAM) on CUDA.
        x: A single image (2D/3D/4D), or a list / 5D array of images (processed
            one at a time, each uploaded to the GPU once; returns lists).
        download: If ``False`` (default), return ``masks`` as NumPy (the single
            device→host transfer) and ``[None, dP, cellprob]`` + ``styles`` as
            CuPy (GPU-resident). If ``True``, also move ``dP``/``cellprob``/
            ``styles`` to host and compute ``dx_to_circ`` for drop-in parity
            with stock ``CellposeModel.eval``.

    Returns
    -------
        ``(masks, [flows_hsv_or_None, dP, cellprob], styles)``.
    """
    _require_cellpose()
    _check_gpu_precondition(model)

    # list / 5D-array of images: process one at a time (mirror CellposeModel.eval)
    if isinstance(x, list) or (hasattr(x, "ndim") and x.squeeze().ndim == 5):
        nimg = len(x)
        diam = (
            diameter if isinstance(diameter, (list, np.ndarray)) else [diameter] * nimg
        )
        masks_out, flows_out, styles_out = [], [], []
        for i in range(nimg):
            out = segment_cpsam(
                model,
                x[i],
                batch_size=batch_size,
                resample=resample,
                normalize=normalize,
                channel_axis=channel_axis,
                z_axis=z_axis,
                diameter=diam[i],
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                do_3D=do_3D,
                anisotropy=anisotropy,
                flow3D_smooth=flow3D_smooth,
                stitch_threshold=stitch_threshold,
                min_size=min_size,
                max_size_fraction=max_size_fraction,
                niter=niter,
                augment=augment,
                tile_overlap=tile_overlap,
                download=download,
            )
            masks_out.append(out[0])
            flows_out.append(out[1])
            styles_out.append(out[2])
        return masks_out, flows_out, styles_out

    from .cellpose_dynamics import compute_masks

    # --- host-side layout (pure transpose/slice, no heavy compute) ---
    x = _cp_transforms.convert_image(
        x,
        channel_axis=channel_axis,
        z_axis=z_axis,
        do_3D=(do_3D or stitch_threshold > 0),
    )
    if x.ndim < 4:
        x = x[np.newaxis, ...]
    nimg = x.shape[0]

    rescale = 1.0
    if diameter is not None and diameter > 0:
        rescale = 30.0 / diameter

    # --- resolve normalization params (mirror CellposeModel.eval) ---
    normalize_params = dict(_NORMALIZE_DEFAULT)
    if isinstance(normalize, dict):
        normalize_params = {**normalize_params, **normalize}
    elif isinstance(normalize, bool):
        normalize_params["normalize"] = normalize
    else:
        raise ValueError("normalize parameter must be a bool or a dict")
    do_normalization = bool(normalize_params["normalize"])

    # --- THE SINGLE host -> device upload ---
    x = ascupy(x)

    # pre-normalize the whole stack for 3D / stitching (mirror eval:276-289)
    if nimg > 1 and do_normalization and (stitch_threshold or do_3D):
        normalize_params["norm3D"] = True if do_3D else normalize_params["norm3D"]
        x = normalize_img_gpu(x, **normalize_params)
        do_normalization = False
    elif normalize_params["norm3D"] and nimg > 1 and do_normalization:
        normalize_params["norm3D"] = False
    if do_normalization:
        x = normalize_img_gpu(x, **normalize_params)

    # --- network forward (GPU-resident) ---
    dP, cellprob, styles = _run_net_gpu(
        model.net,
        x,
        backbone=model.backbone,
        rescale=rescale,
        resample=resample,
        augment=augment,
        batch_size=batch_size,
        tile_overlap=tile_overlap,
        do_3D=do_3D,
        anisotropy=anisotropy,
    )

    if do_3D and flow3D_smooth:
        from ..scipy import ndimage as _ndimage

        if isinstance(flow3D_smooth, (int, float)):
            flow3D_smooth = [flow3D_smooth] * 3
        elif isinstance(flow3D_smooth, list) and len(flow3D_smooth) == 1:
            flow3D_smooth = flow3D_smooth * 3
        if len(flow3D_smooth) == 3 and any(v > 0 for v in flow3D_smooth):
            dP = _ndimage.gaussian_filter(dP, [0, *flow3D_smooth])

    # --- mask computation (GPU-resident) ---
    niter_scale = 1 if (rescale is None or not resample) else rescale
    n_iter = int(200 / niter_scale) if (niter is None or niter == 0) else niter
    masks = compute_masks(
        x.shape,
        dP,
        cellprob,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        min_size=min_size,
        max_size_fraction=max_size_fraction,
        niter=n_iter,
        stitch_threshold=stitch_threshold,
        do_3D=do_3D,
        device=model.device,
    )

    masks, dP, cellprob = masks.squeeze(), dP.squeeze(), cellprob.squeeze()
    assert get_device(masks) == "GPU", "internal: mask left the GPU before download"

    # --- return contract ---
    if download:
        dP_h, cellprob_h = asnumpy(dP), asnumpy(cellprob)
        from cellpose import plot

        return (
            asnumpy(masks),
            [plot.dx_to_circ(dP_h), dP_h, cellprob_h],
            asnumpy(styles),
        )
    return asnumpy(masks), [None, dP, cellprob], styles
