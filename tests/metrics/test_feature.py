"""Tests for morphology feature-correlation helpers."""

import numpy as np

from cubic.metrics.feature import get_true_features


def test_get_true_features_accepts_vector_property_names() -> None:
    """A precomputed dict with expanded columns matches base feature names.

    ``regionprops_table`` expands vector properties into ``centroid-0`` /
    ``centroid-1`` columns. Requesting ``features=["centroid"]`` must not raise:
    the validation compares base names, not expanded column names.
    """
    gt_feature_dict = {
        "label": np.array([1, 2]),
        "area": np.array([10.0, 20.0]),
        "centroid-0": np.array([1.0, 5.0]),
        "centroid-1": np.array([2.0, 6.0]),
    }

    labels, features_array, names = get_true_features(
        None,
        features=["centroid", "area"],
        gt_feature_dict=gt_feature_dict,
    )

    np.testing.assert_array_equal(labels, [1, 2])
    # numeric columns: area, centroid-0, centroid-1 (sorted) -> 3 columns
    assert features_array.shape == (2, 3)
    assert names == ["centroid", "area"]
