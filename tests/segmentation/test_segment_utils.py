"""Tests for segmentation post-processing utilities."""

import numpy as np

from cubic.segmentation.segment_utils import remove_touching_objects


def test_remove_touching_objects_keeps_isolated_high_labels() -> None:
    """Labels above the legacy border_value are not misclassified as touching.

    Two adjacent objects (150, 151) must be removed; an isolated object (200)
    must be kept. The previous additive-offset implementation treated every
    label id > border_value (=100) as touching and deleted everything.
    """
    volume = np.zeros((3, 9, 9), dtype=np.int32)
    volume[1, 1:3, 1:3] = 150  # touches 151
    volume[1, 3:5, 1:3] = 151  # touches 150
    volume[1, 7:9, 7:9] = 200  # isolated

    result = remove_touching_objects(volume.copy())
    remaining = set(np.unique(result).tolist())

    assert 200 in remaining
    assert 150 not in remaining
    assert 151 not in remaining


def test_remove_touching_objects_keeps_all_isolated() -> None:
    """All non-touching objects survive regardless of label magnitude."""
    volume = np.zeros((3, 9, 9), dtype=np.int32)
    volume[1, 1:3, 1:3] = 105
    volume[1, 6:8, 6:8] = 250

    result = remove_touching_objects(volume.copy())
    remaining = set(np.unique(result).tolist())

    assert remaining == {0, 105, 250}
