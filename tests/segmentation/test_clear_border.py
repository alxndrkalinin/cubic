"""Tests for border-clearing segmentation helper."""

import numpy as np
import pytest

from cubic.segmentation._clear_border import clear_border


def test_clear_border_simple() -> None:
    """Remove objects touching the image border."""
    labels = np.array(
        [
            [1, 1, 0],
            [0, 2, 0],
            [0, 0, 0],
        ],
        dtype=int,
    )
    result = clear_border(labels.copy())
    assert 1 not in np.unique(result)
    assert 2 in np.unique(result)


def test_clear_border_3d() -> None:
    """Handle 3D labeled volumes."""
    labels = np.zeros((3, 3, 3), dtype=int)
    labels[1, 1, 1] = 1  # interior object
    labels[0, 0, 0] = 2  # border object
    result = clear_border(labels.copy())
    assert 2 not in np.unique(result)
    assert 1 in np.unique(result)


def test_clear_border_all_border_objects() -> None:
    """All objects removed when touching border."""
    labels = np.array([[1, 0], [0, 2]], dtype=int)
    result = clear_border(labels.copy())
    assert np.all(result == 0)


def test_clear_border_mask_shape_mismatch_raises_valueerror() -> None:
    """A mismatched mask raises ValueError (previously raised TypeError)."""
    labels = np.zeros((4, 4), dtype=int)
    bad_mask = np.ones((3, 3), dtype=bool)
    with pytest.raises(ValueError, match="same shape"):
        clear_border(labels, mask=bad_mask)


def test_clear_border_leaves_input_untouched() -> None:
    """Without ``out``, the caller's array is copied rather than modified."""
    labels = np.array([[1, 1, 0], [0, 2, 0], [0, 0, 0]], dtype=int)
    original = labels.copy()
    clear_border(labels)
    assert np.array_equal(labels, original)


def test_clear_border_writes_into_out() -> None:
    """``out`` receives the result and is returned."""
    labels = np.array([[1, 1, 0], [0, 2, 0], [0, 0, 0]], dtype=int)
    out = np.empty_like(labels)
    result = clear_border(labels, out=out)
    assert result is out
    assert 1 not in np.unique(out)
    assert 2 in np.unique(out)


def test_clear_border_rejects_removed_in_place_argument() -> None:
    """``in_place`` was removed from skimage in 0.20 and is gone here too."""
    labels = np.zeros((4, 4), dtype=int)
    with pytest.raises(TypeError):
        clear_border(labels, in_place=True)
