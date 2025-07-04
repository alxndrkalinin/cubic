"""Tests for thresholding utilities."""

import numpy as np

from morphocell.preprocessing.thresholding import (
    get_threshold_otsu,
    select_nonempty_patches,
)


def test_threshold_and_patch_selection() -> None:
    """Otsu threshold and patch selection on a toy image."""
    img_top = np.zeros((10, 10), dtype=np.float32)
    img_bottom = np.ones((10, 10), dtype=np.float32)
    img = np.stack([img_top, img_bottom])
    th = get_threshold_otsu(img, blur_sigma=0)
    assert 0.0 < th < 1.0
    patches = select_nonempty_patches(
        img, patch_size=10, min_nonzeros=0.5, threshold=th
    )
    assert len(patches) >= 1


def test_select_nonempty_patches_empty() -> None:
    """No patches should be returned when threshold is high."""
    img = np.zeros((2, 10, 10), dtype=np.float32)
    patches = select_nonempty_patches(
        img, patch_size=10, min_nonzeros=0.5, threshold=1.0
    )
    assert patches == []


def test_select_nonempty_patches_threshold_bounds() -> None:
    """Boundary threshold values should behave as expected."""
    img = np.ones((2, 10, 10), dtype=np.float32)
    patches_lo = select_nonempty_patches(
        img, patch_size=10, min_nonzeros=1.0, threshold=0.0
    )
    assert len(patches_lo) == 1
    patches_hi = select_nonempty_patches(
        img, patch_size=10, min_nonzeros=1.0, threshold=1.0
    )
    assert patches_hi == []
