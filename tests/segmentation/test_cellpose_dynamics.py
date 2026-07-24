"""Tests for the GPU-resident Cellpose mask post-processing.

Module-level imports stay free of optional deps (cellpose/cupy/torch) so the
suite collects on CPU/no-GPU CI; GPU-dependent tests skip cleanly.
"""

import numpy as np
import pytest

from cubic.segmentation import cellpose_dynamics as cd


def test_compute_masks_requires_cellpose(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without cellpose, ``compute_masks`` raises ImportError up front.

    ``_CELLPOSE_AVAILABLE`` was set but never read, so ``max_pool_nd`` stayed
    None and the failure surfaced deep inside ``_get_masks`` as
    ``TypeError: 'NoneType' object is not callable``.
    """
    monkeypatch.setattr(cd, "_CELLPOSE_AVAILABLE", False)
    monkeypatch.setattr(cd, "max_pool_nd", None)

    with pytest.raises(ImportError, match="cellpose"):
        cd.compute_masks(
            (1, 8, 8),
            np.zeros((2, 1, 8, 8), dtype=np.float32),
            np.zeros((1, 8, 8), dtype=np.float32),
        )


def test_fill_holes_and_size_filter_compacts_label_ids(gpu_available: bool) -> None:
    """Gapped input ids come back compacted, as upstream cellpose does.

    Upstream writes ``j + 1`` with ``j`` counting only the labels present; cubic
    wrote ``i + 1``, leaving gaps that make ``_stitch3D``'s IoU rows NaN when
    ``min_size <= 0`` skips the trailing relabel.
    """
    if not gpu_available:
        pytest.skip("requires a CUDA GPU")
    cp = pytest.importorskip("cupy")

    masks = cp.zeros((16, 16), dtype=cp.uint16)
    masks[2:5, 2:5] = 1
    masks[9:12, 9:12] = 3  # label 2 is absent

    out = cd._fill_holes_and_size_filter(masks, min_size=-1)

    assert set(np.unique(cp.asnumpy(out)).tolist()) == {0, 1, 2}


def test_fill_holes_and_size_filter_fills_after_compacting(
    gpu_available: bool,
) -> None:
    """Compacting the ids does not stop the per-label hole filling."""
    if not gpu_available:
        pytest.skip("requires a CUDA GPU")
    cp = pytest.importorskip("cupy")

    masks = cp.zeros((24, 24), dtype=cp.uint16)
    masks[2:9, 2:9] = 3  # label 1 and 2 absent
    masks[5, 5] = 0  # interior hole

    out = cd._fill_holes_and_size_filter(masks, min_size=-1)

    assert int(out[5, 5]) == 1
    assert set(np.unique(cp.asnumpy(out)).tolist()) == {0, 1}
