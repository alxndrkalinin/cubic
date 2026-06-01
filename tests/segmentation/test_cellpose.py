"""Tests for the Cellpose segmentation wrapper."""

import numpy as np
import pytest

from cubic.segmentation import cellpose as cellpose_mod
from cubic.segmentation.cellpose import cellpose_segment


def test_cellpose_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cellpose should raise if the library is unavailable.

    Force the unavailable branch so the test is deterministic regardless of
    whether cellpose is installed in the running environment.
    """
    monkeypatch.setattr(cellpose_mod, "_CELLPOSE_AVAILABLE", False)
    image = np.zeros((1, 32, 32), dtype=np.float32)
    with pytest.raises(ImportError):
        cellpose_segment(image)
