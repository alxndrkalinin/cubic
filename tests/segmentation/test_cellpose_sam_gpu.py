"""Tests for the GPU-resident Cellpose-SAM eval path.

Module-level imports stay free of optional deps (cellpose/cupy/torch) so the
suite collects on CPU/no-GPU CI; GPU/cellpose-dependent tests skip cleanly.
"""

import numpy as np
import pytest


def _require_cellpose_v4() -> None:
    pytest.importorskip("cellpose")
    from cellpose import models

    if not hasattr(models, "CellposeModel"):  # pragma: no cover
        pytest.skip("requires cellpose>=4 (SAM)")


def _disks(h: int = 224, w: int = 224, centers=None, r: int = 15) -> np.ndarray:
    centers = centers or [
        (40, 40),
        (40, 110),
        (40, 180),
        (110, 40),
        (110, 110),
        (110, 180),
        (180, 40),
        (180, 110),
        (180, 180),
    ]
    img = np.zeros((h, w), np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    for cy, cx in centers:
        img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 200.0
    return img + np.random.default_rng(0).normal(0, 3, img.shape).astype(np.float32)


def _volume_3d(lz: int = 20, h: int = 80, w: int = 80) -> np.ndarray:
    """Small 3D grayscale volume with a few spherical blobs."""
    vol = np.zeros((lz, h, w), np.float32)
    zz, yy, xx = np.mgrid[0:lz, 0:h, 0:w]
    for cz, cy, cx in [(10, 28, 28), (10, 55, 55), (7, 30, 58)]:
        vol[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= 12**2] = 200.0
    return vol + np.random.default_rng(2).normal(0, 3, vol.shape).astype(np.float32)


def _zstack(nz: int = 6) -> np.ndarray:
    """Z-stack of 2D disk images (same objects per plane) for stitching."""
    return np.stack(
        [_disks(centers=[(60, 60), (60, 150), (150, 100)], r=16) for _ in range(nz)]
    )


def test_segmentation_imports_without_optional_deps() -> None:
    """``import cubic.segmentation`` must succeed even without cellpose/cupy/torch."""
    import importlib

    mod = importlib.import_module("cubic.segmentation")
    assert hasattr(mod, "segment_cpsam")


def test_resident_requires_cellpose(monkeypatch: pytest.MonkeyPatch) -> None:
    """A clear ImportError is raised when cellpose (v4) is unavailable."""
    from cubic.segmentation import cellpose_sam_gpu as g

    monkeypatch.setattr(g, "_CELLPOSE_AVAILABLE", False)
    with pytest.raises(ImportError):
        g.segment_cpsam(object(), np.zeros((32, 32), np.float32))


def test_resident_rejects_non_cuda_model() -> None:
    """The GPU-only orchestrator rejects a CPU model with a clear RuntimeError."""
    _require_cellpose_v4()
    import torch

    from cubic.segmentation import cellpose_sam_gpu as g

    class _Net:
        dtype = torch.float32

    class _Model:
        backbone = "sam_vitl"
        net = _Net()
        device = torch.device("cpu")

    with pytest.raises(RuntimeError):
        g.segment_cpsam(_Model(), np.zeros((32, 32), np.float32))


def test_sam_model_exposes_v4_api(gpu_available: bool) -> None:
    """Fail loudly on a v3 env: SAM model must expose backbone + net.dtype."""
    if not gpu_available:
        pytest.skip("requires a CUDA GPU")
    _require_cellpose_v4()
    from cellpose.models import CellposeModel

    model = CellposeModel(gpu=True, pretrained_model="cpsam")
    assert hasattr(model, "backbone")
    assert hasattr(model.net, "dtype")
    assert model.device.type == "cuda"


def test_building_blocks_run_on_cpu() -> None:
    """Device-agnostic leaves run on NumPy (CPU), independent of the orchestrator."""
    _require_cellpose_v4()
    from cubic.segmentation import cellpose_sam_gpu as g

    img = np.random.default_rng(0).random((3, 100, 120)).astype(np.float32)
    IMG, ysub, xsub, Ly, Lx = g.make_tiles(img, bsize=64, augment=False)
    assert isinstance(IMG, np.ndarray)
    assert IMG.shape[1] == 3

    x = (np.random.default_rng(1).random((1, 64, 64, 3)) * 1000).astype(np.float32)
    out = g.normalize_img_gpu(x, **g._NORMALIZE_DEFAULT)
    assert isinstance(out, np.ndarray) and out.shape == x.shape


def test_parity_2d_vs_stock(gpu_available: bool) -> None:
    """Resident masks match stock CellposeModel.eval (AP >= 0.95 @ IoU 0.5)."""
    if not gpu_available:
        pytest.skip("requires a CUDA GPU")
    _require_cellpose_v4()
    from cellpose.models import CellposeModel

    from cubic.segmentation import segment_cpsam
    from cubic.metrics.average_precision import average_precision

    img = _disks()
    model = CellposeModel(gpu=True, pretrained_model="cpsam")
    m_stock, _, _ = model.eval(img, diameter=None, do_3D=False)
    m_gpu, _, _ = segment_cpsam(model, img, do_3D=False, download=True)

    m_stock = m_stock.astype(np.int32)
    m_gpu = np.asarray(m_gpu).astype(np.int32)
    ap, _, _, _ = average_precision(m_stock, m_gpu, [0.5])
    assert ap[0] >= 0.95, (
        f"AP@0.5={ap[0]:.3f} (stock n={m_stock.max()}, gpu n={m_gpu.max()})"
    )


def test_list_input_returns_per_image_lists(gpu_available: bool) -> None:
    """A list of images is processed per-image (mirrors eval) and matches stock."""
    if not gpu_available:
        pytest.skip("requires a CUDA GPU")
    _require_cellpose_v4()
    from cellpose.models import CellposeModel

    from cubic.segmentation import segment_cpsam
    from cubic.metrics.average_precision import average_precision

    imgs = [_disks(), _disks(centers=[(60, 60), (160, 160), (60, 160)], r=18)]
    model = CellposeModel(gpu=True, pretrained_model="cpsam")
    m_stock, _, _ = model.eval(imgs, diameter=None, do_3D=False)
    m_gpu, flows, styles = segment_cpsam(model, imgs, do_3D=False, download=True)

    assert (
        isinstance(m_gpu, list) and isinstance(flows, list) and isinstance(styles, list)
    )
    assert len(m_gpu) == len(imgs)
    for ms, mg in zip(m_stock, m_gpu):
        ap, _, _, _ = average_precision(
            np.asarray(ms).astype(np.int32), np.asarray(mg).astype(np.int32), [0.5]
        )
        assert ap[0] >= 0.95, f"AP@0.5={ap[0]:.3f}"


def test_parity_3d_vs_stock(gpu_available: bool) -> None:
    """3D (do_3D=True) resident masks match stock eval (AP >= 0.95 @ IoU 0.5)."""
    if not gpu_available:
        pytest.skip("requires a CUDA GPU")
    _require_cellpose_v4()
    from cellpose.models import CellposeModel

    from cubic.segmentation import segment_cpsam
    from cubic.metrics.average_precision import average_precision

    vol = _volume_3d()
    model = CellposeModel(gpu=True, pretrained_model="cpsam")
    m_stock, _, _ = model.eval(vol, z_axis=0, diameter=None, do_3D=True)
    m_gpu, _, _ = segment_cpsam(model, vol, z_axis=0, do_3D=True, download=True)

    m_stock = np.asarray(m_stock).astype(np.int32)
    m_gpu = np.asarray(m_gpu).astype(np.int32)
    assert m_stock.shape == m_gpu.shape == vol.shape
    ap, _, _, _ = average_precision(m_stock, m_gpu, [0.5])
    assert ap[0] >= 0.95, (
        f"AP@0.5={ap[0]:.3f} (stock n={m_stock.max()}, gpu n={m_gpu.max()})"
    )


def test_parity_stitch_vs_stock(gpu_available: bool) -> None:
    """Stitched 3D masks (stitch_threshold>0) match stock eval (AP >= 0.95)."""
    if not gpu_available:
        pytest.skip("requires a CUDA GPU")
    _require_cellpose_v4()
    from cellpose.models import CellposeModel

    from cubic.segmentation import segment_cpsam
    from cubic.metrics.average_precision import average_precision

    stack = _zstack()
    model = CellposeModel(gpu=True, pretrained_model="cpsam")
    m_stock, _, _ = model.eval(
        stack, z_axis=0, diameter=None, do_3D=False, stitch_threshold=0.5
    )
    m_gpu, _, _ = segment_cpsam(
        model, stack, z_axis=0, do_3D=False, stitch_threshold=0.5, download=True
    )

    m_stock = np.asarray(m_stock).astype(np.int32)
    m_gpu = np.asarray(m_gpu).astype(np.int32)
    assert m_stock.shape == m_gpu.shape == stack.shape
    ap, _, _, _ = average_precision(m_stock, m_gpu, [0.5])
    assert ap[0] >= 0.95, (
        f"AP@0.5={ap[0]:.3f} (stock n={m_stock.max()}, gpu n={m_gpu.max()})"
    )


def test_fill_holes_fills_interior(gpu_available: bool) -> None:
    """_fill_holes_and_size_filter actually fills interior holes (writes back)."""
    if not gpu_available:
        pytest.skip("requires a CUDA GPU")
    cp = pytest.importorskip("cupy")
    from cubic.segmentation.cellpose_dynamics import _fill_holes_and_size_filter

    lbl = cp.zeros((40, 40), dtype=cp.uint16)
    yy, xx = cp.mgrid[0:40, 0:40]
    r2 = (yy - 20) ** 2 + (xx - 20) ** 2
    lbl[(r2 <= 12**2) & (r2 > 4**2)] = 1  # annulus: solid disk with a hole
    assert int(lbl[20, 20]) == 0  # hole present before filling

    out = _fill_holes_and_size_filter(lbl.copy(), min_size=5)
    assert int(out[20, 20]) == 1  # interior hole filled (slice write-back works)


def test_residency_single_download(gpu_available: bool) -> None:
    """download=False: only the final mask leaves the GPU; flows stay CuPy."""
    if not gpu_available:
        pytest.skip("requires a CUDA GPU")
    _require_cellpose_v4()
    from cellpose.models import CellposeModel

    from cubic.cuda import get_device
    from cubic.segmentation import cellpose_sam_gpu as g

    img = _disks()
    model = CellposeModel(gpu=True, pretrained_model="cpsam")

    calls = {"n": 0}
    orig = g.asnumpy

    def _spy(arr):
        calls["n"] += 1
        return orig(arr)

    g.asnumpy = _spy
    try:
        masks, flows, _ = g.segment_cpsam(model, img, do_3D=False, download=False)
    finally:
        g.asnumpy = orig

    assert isinstance(masks, np.ndarray)  # mask downloaded
    assert flows[0] is None  # dx_to_circ skipped (CPU-only)
    assert get_device(flows[1]) == "GPU"  # dP stays resident
    assert get_device(flows[2]) == "GPU"  # cellprob stays resident
    assert calls["n"] == 1  # exactly one device->host array transfer
