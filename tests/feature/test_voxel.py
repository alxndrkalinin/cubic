"""Tests for the voxel feature module."""

import numpy as np
import pytest

from cubic.feature import voxel


def test_regionprops_extract_features() -> None:
    """Extract simple features and check results."""
    labels = np.array([[0, 1], [1, 1]], dtype=np.int32)
    props = voxel.regionprops_table(labels, properties=["area"])
    assert props["label"].tolist() == [1]
    assert int(props["area"][0]) == 3

    labels_out, feature_values, names = voxel.extract_features(labels, ["area"])
    assert labels_out.tolist() == [1]
    assert feature_values.shape == (1, 1)
    assert feature_values[0, 0] == 3
    assert names == ["area"]


def test_regionprops_multiple_labels() -> None:
    """Extract multiple features from a multi-label image."""
    labels = np.array(
        [
            [0, 1, 1],
            [2, 2, 0],
        ],
        dtype=np.int32,
    )

    props = voxel.regionprops_table(labels, properties=["area", "centroid", "bbox"])
    assert sorted(props["label"].tolist()) == [1, 2]
    assert props["area"].tolist() == [2, 2]
    assert "centroid-0" in props and "centroid-1" in props
    assert "bbox-0" in props and "bbox-3" in props

    labels_out, feats, names = voxel.extract_features(labels, ["area", "centroid"])
    assert labels_out.tolist() == [1, 2]
    assert feats.shape == (2, 3)
    assert names == ["area", "centroid-0", "centroid-1"]


def test_extract_features_returns_expanded_column_names() -> None:
    """Column names are the expanded, alphabetically sorted regionprops names.

    A 3D ``centroid`` becomes three columns, so neither the count nor the order
    of columns follows the requested ``features`` list. Callers must key results
    by the returned names.
    """
    labels = np.zeros((6, 6, 6), dtype=np.int32)
    labels[1:3, 1:3, 1:3] = 1
    labels[3:5, 3:5, 3:5] = 2

    labels_out, feats, names = voxel.extract_features(labels, ["centroid", "area"])

    assert labels_out.tolist() == [1, 2]
    # Requested order is ["centroid", "area"]; output order is alphabetical.
    assert names == ["area", "centroid-0", "centroid-1", "centroid-2"]
    assert feats.shape == (2, 4)
    np.testing.assert_allclose(feats[:, 0], [8, 8])
    np.testing.assert_allclose(feats[0, 1:], [1.5, 1.5, 1.5])


def test_extract_features_range_mismatch_raises() -> None:
    """A missing per-column range raises instead of asserting.

    ``feature_ranges`` must be keyed by expanded column name; passing the base
    property name raises ``ValueError`` (asserts are stripped under ``python -O``).
    """
    labels = np.zeros((6, 6, 6), dtype=np.int32)
    labels[1:3, 1:3, 1:3] = 1

    with pytest.raises(ValueError, match="centroid-0"):
        voxel.extract_features(
            labels, ["centroid", "area"], {"area": (0.0, 10.0), "centroid": (0.0, 6.0)}
        )


def test_norm_features_by_range_integer_zero_range() -> None:
    """An integer zero-width range does not truncate the epsilon to 0.

    ``np.asarray([(0, 0)])`` is int64, so assigning ``np.finfo(np.float32).eps``
    into it used to store 0 and yield infinities.
    """
    out = voxel.norm_features_by_range([[5.0]], ["a"], {"a": (0, 0)})
    assert np.isfinite(out).all()
    np.testing.assert_allclose(out, [[5.0 / np.finfo(np.float32).eps]])
    # Float-typed ranges must give the same answer as integer ones.
    np.testing.assert_allclose(
        out, voxel.norm_features_by_range([[5.0]], ["a"], {"a": (0.0, 0.0)})
    )


def test_regionprops_table_default_properties() -> None:
    """``properties=None`` falls through to the regionprops_table default."""
    labels = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int32)
    out = voxel.regionprops_table(labels)

    assert "label" in out, "an empty table without a label column is unusable"
    assert sorted(out["label"].tolist()) == [1, 2]
    assert "bbox-0" in out  # skimage's default is ("label", "bbox")


def test_regionprops_table_preserves_property_order() -> None:
    """Output columns follow the requested order with a deduped label last."""
    labels = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int32)
    # Duplicate "area" and omit "label": dedup must keep first occurrence
    # and append "label" once at the end.
    props = ["area", "perimeter", "area", "centroid"]
    out = voxel.regionprops_table(labels, properties=props)
    keys = list(out.keys())

    assert sum(k == "area" for k in keys) == 1
    assert keys.index("area") < keys.index("perimeter")
    assert keys.index("perimeter") < keys.index("centroid-0")
    assert keys.index("centroid-0") < keys.index("label")


def test_regionprops_table_extra_properties() -> None:
    """extra_properties callables are forwarded and keyed by function name."""

    def intensity_range(regionmask: np.ndarray, intensity: np.ndarray) -> float:
        values = intensity[regionmask]
        return float(values.max() - values.min())

    labels = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int32)
    intensity = np.array([[0, 10, 20], [5, 5, 0]], dtype=np.float32)

    out = voxel.regionprops_table(
        labels,
        intensity_image=intensity,
        properties=["area"],
        extra_properties=(intensity_range,),
    )

    assert "intensity_range" in out
    by_label = dict(zip(out["label"].tolist(), out["intensity_range"].tolist()))
    assert by_label[1] == 10.0  # intensities {10, 20}
    assert by_label[2] == 0.0  # intensities {5, 5}


def test_regionprops_extra_properties() -> None:
    """extra_properties callables are forwarded and exposed as region attributes."""

    def intensity_range(regionmask: np.ndarray, intensity: np.ndarray) -> float:
        values = intensity[regionmask]
        return float(values.max() - values.min())

    labels = np.array([[0, 1, 1], [2, 2, 0]], dtype=np.int32)
    intensity = np.array([[0, 10, 20], [5, 5, 0]], dtype=np.float32)

    regions = voxel.regionprops(
        labels,
        intensity_image=intensity,
        extra_properties=(intensity_range,),
    )

    by_label = {region.label: region.intensity_range for region in regions}
    assert by_label[1] == 10.0  # intensities {10, 20}
    assert by_label[2] == 0.0  # intensities {5, 5}
