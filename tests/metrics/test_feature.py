"""Tests for morphology feature-correlation helpers."""

import numpy as np
import pytest

from cubic.cuda import ascupy
from cubic.feature.voxel import extract_features
from cubic.metrics.feature import (
    cosine_median,
    get_true_features,
    _calculate_correlations,
    morphology_correlations,
)


def _labels_3d() -> np.ndarray:
    """Return a 3D label image with five differently sized, placed objects."""
    labels = np.zeros((24, 24, 24), dtype=np.int32)
    for idx, (z, y, x, size) in enumerate(
        [(1, 1, 1, 2), (6, 3, 8, 3), (12, 14, 2, 4), (17, 6, 15, 2), (3, 17, 18, 5)],
        start=1,
    ):
        labels[z : z + size, y : y + size, x : x + size] = idx
    return labels


@pytest.mark.parametrize("use_gpu", [False, True])
def test_morphology_correlations_identical_inputs_are_one(
    use_gpu: bool, gpu_available: bool
) -> None:
    """Identical true and pred masks must correlate exactly 1.0 per feature.

    ``extract_features`` expands ``centroid`` into three columns in 3D, so the
    ``corrcoef`` block to slice is ``[:4, 4:]`` and not ``[:2, 2:]``. Slicing by
    the number of *requested* properties returned off-block entries, which are
    true-vs-true correlations of unrelated features: this case used to give
    ``{'area': -0.380, 'centroid': 0.915}``.
    """
    labels = _labels_3d()
    matched = np.arange(1, 6)
    if use_gpu:
        if not gpu_available:
            pytest.skip("GPU not available")
        labels = ascupy(labels)
        matched = ascupy(matched)
    correlations = morphology_correlations(
        labels,
        labels,
        features=["area", "centroid"],
        thresholds=[0.5],
        matches_per_threshold={0.5: (matched, matched)},
    )

    assert list(correlations) == [0.5]
    assert sorted(correlations[0.5]) == [
        "area",
        "centroid-0",
        "centroid-1",
        "centroid-2",
    ]
    for name, value in correlations[0.5].items():
        # Plain host floats on both devices, as annotated: a GPU run used to
        # leak 0-d CuPy scalars.
        assert isinstance(value, float), name
        assert value == pytest.approx(1.0), name


def test_morphology_correlations_keys_and_values_match_columns() -> None:
    """Each reported correlation is the true-vs-pred one for that column."""
    labels_true = _labels_3d()
    # Shift one object so the predicted features differ from the true ones.
    labels_pred = labels_true.copy()
    labels_pred[labels_pred == 3] = 0
    labels_pred[13:17, 15:19, 3:7] = 3

    matched = np.arange(1, 6)
    correlations = morphology_correlations(
        labels_true,
        labels_pred,
        features=["area", "centroid"],
        thresholds=[0.5],
        matches_per_threshold={0.5: (matched, matched)},
    )[0.5]

    _, true_feats, names = extract_features(labels_true, ["area", "centroid"])
    _, pred_feats, _ = extract_features(labels_pred, ["area", "centroid"])
    for column, name in enumerate(names):
        expected = np.corrcoef(true_feats[:, column], pred_feats[:, column])[0, 1]
        assert correlations[name] == pytest.approx(expected), name


def test_morphology_correlations_unmatched_threshold_is_nan() -> None:
    """A threshold with no matches yields NaN for every expanded column."""
    labels = _labels_3d()
    empty = np.array([], dtype=np.int32)
    correlations = morphology_correlations(
        labels,
        labels,
        features=["area", "centroid"],
        thresholds=[0.9],
        matches_per_threshold={0.9: (empty, empty)},
    )[0.9]

    assert sorted(correlations) == ["area", "centroid-0", "centroid-1", "centroid-2"]
    assert all(np.isnan(value) for value in correlations.values())


def test_calculate_correlations_slices_by_matrix_width() -> None:
    """The correlation block is sized from the matrix, not the name list."""
    rng = np.random.default_rng(0)
    true_matrix = rng.random((10, 4))
    pred_matrix = true_matrix + 0.01 * rng.random((10, 4))
    names = ["a", "b", "c", "d"]

    out = _calculate_correlations(true_matrix, pred_matrix, names)

    assert list(out) == names
    for column, name in enumerate(names):
        expected = np.corrcoef(true_matrix[:, column], pred_matrix[:, column])[0, 1]
        assert out[name] == pytest.approx(expected)


def test_cosine_median_identical_masks_is_zero() -> None:
    """Identical masks have zero cosine distance, with and without thresholds."""
    labels = _labels_3d()
    matched = np.arange(1, 6)

    assert cosine_median(labels, labels, ["area", "centroid"]) == pytest.approx(0.0)

    distances = cosine_median(
        labels,
        labels,
        ["area", "centroid"],
        thresholds=[0.5],
        matches_per_threshold={0.5: (matched, matched)},
    )
    assert distances[0.5] == pytest.approx(0.0)


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
    # Without ranges the values are returned unnormalized, matching the
    # ``extract_features`` branch.
    np.testing.assert_allclose(features_array[:, 0], [10.0, 20.0])


def test_get_true_features_validates_precomputed_dict() -> None:
    """Bad precomputed dicts raise ValueError, not AssertionError.

    ``assert`` statements vanish under ``python -O``, and the missing-label check
    used to run after the value it validates had already been consumed.
    """
    with pytest.raises(ValueError, match="Label column not found"):
        get_true_features(None, gt_feature_dict={"area": np.array([1.0])})

    with pytest.raises(ValueError, match="not found in precomputed"):
        get_true_features(
            None,
            features=["perimeter"],
            gt_feature_dict={"label": np.array([1]), "area": np.array([1.0])},
        )
