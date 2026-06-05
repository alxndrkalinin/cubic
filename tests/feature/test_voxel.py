"""Tests for the voxel feature module."""

import numpy as np

from cubic.feature import voxel


def test_regionprops_extract_features() -> None:
    """Extract simple features and check results."""
    labels = np.array([[0, 1], [1, 1]], dtype=np.int32)
    props = voxel.regionprops_table(labels, properties=["area"])
    assert props["label"].tolist() == [1]
    assert int(props["area"][0]) == 3

    labels_out, feature_values = voxel.extract_features(labels, ["area"])
    assert labels_out.tolist() == [1]
    assert feature_values.shape == (1, 1)
    assert feature_values[0, 0] == 3


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

    labels_out, feats = voxel.extract_features(labels, ["area", "centroid"])
    assert labels_out.tolist() == [1, 2]
    assert feats.shape == (2, 3)


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
