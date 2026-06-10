"""Calculate feature-based metrics for morphology comparison."""

import numpy as np

from cubic.cuda import to_same_device
from cubic.feature.voxel import extract_features, norm_features_by_range
from cubic.metrics.average_precision import compute_matches


def _calculate_cosine(true_features: np.ndarray, pred_features: np.ndarray) -> float:
    """Calculate cosine distance between true and pred features."""
    norm_true = np.linalg.norm(true_features)
    norm_pred = np.linalg.norm(pred_features)
    if norm_true == 0 or norm_pred == 0:
        return 1
    return 1 - np.dot(true_features, pred_features) / (norm_true * norm_pred)


def filter_nan_features(
    true_features: np.ndarray, pred_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Remove features that contain NaNs from both true and predicted feature sets."""
    nan_mask = np.isnan(true_features).any(axis=0) | np.isnan(pred_features).any(axis=0)
    return true_features[:, ~nan_mask], pred_features[:, ~nan_mask]


def get_true_features(
    label_image_true: np.ndarray,
    features: list[str] | None = None,
    feature_ranges: dict[str, tuple[float, float]] | None = None,
    gt_feature_dict: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract or retrieve true features based on precomputed or provided label images."""
    if gt_feature_dict:
        gt_features = gt_feature_dict.keys()
        numeric_features = sorted([feat for feat in gt_features if feat != "label"])
        assert "label" in gt_features, "Label column not found in precomputed features."

        if not features:
            features = sorted(
                list(
                    set([feat.split("-")[0] for feat in gt_features if feat != "label"])
                )
            )
        else:
            # gt_features holds expanded column names (e.g. "centroid-0"); compare
            # against their base names so vector properties like "centroid" match.
            gt_base = {feat.split("-")[0] for feat in gt_features if feat != "label"}
            assert all(feat in gt_base for feat in features if feat != "label"), (
                "Some features not found in precomputed features."
            )

        true_labels = gt_feature_dict["label"]

        if not feature_ranges:
            feature_ranges = {}
            for feat in numeric_features:
                arr = gt_feature_dict[feat]
                feature_ranges[feat] = (arr.min(), arr.max())
            true_features = np.stack(
                [gt_feature_dict[f] for f in numeric_features], axis=1
            )
        else:
            true_features = norm_features_by_range(
                np.stack([gt_feature_dict[f] for f in numeric_features], axis=1),
                numeric_features,
                feature_ranges,
            )
    else:
        true_labels, true_features = extract_features(
            label_image_true,
            features,  # type: ignore
            feature_ranges,  # type: ignore
        )

    return true_labels, true_features, features  # type: ignore


def cosine_median(
    label_image_true: np.ndarray,
    label_image_pred: np.ndarray,
    features: list[str],
    thresholds: list[float] | None = None,
    feature_ranges: dict[str, tuple[float, float]] | None = None,
    matches_per_threshold: dict[float, tuple[np.ndarray, np.ndarray]] | None = None,
    precomputed_gt_feature_df: dict[str, np.ndarray] | None = None,
    return_features: bool = False,
) -> float | dict[float, float] | tuple[dict[float, float], np.ndarray, np.ndarray]:
    """Calculate cosine distance between median features of true and pred masks."""
    true_labels, true_features, features = get_true_features(
        label_image_true, features, feature_ranges, precomputed_gt_feature_df
    )
    pred_labels, pred_features = extract_features(
        label_image_pred, features, feature_ranges
    )
    true_features = to_same_device(true_features, pred_features)
    true_labels = to_same_device(true_labels, pred_labels)
    true_features, pred_features = filter_nan_features(true_features, pred_features)

    if thresholds is None:
        return _calculate_cosine(
            np.median(true_features, axis=0), np.median(pred_features, axis=0)
        )

    if matches_per_threshold is None:
        matches_per_threshold = compute_matches(
            label_image_true, label_image_pred, thresholds
        )  # type: ignore[assignment]

    distances = {}
    for th in thresholds:
        true_ind, pred_ind = matches_per_threshold[th]  # type: ignore
        if true_ind.size and pred_ind.size:
            true_ind = to_same_device(true_ind, true_labels)
            pred_ind = to_same_device(pred_ind, pred_labels)
            distances[th] = _calculate_cosine(
                np.median(true_features[np.isin(true_labels, true_ind)], axis=0),
                np.median(pred_features[np.isin(pred_labels, pred_ind)], axis=0),
            )
        else:
            distances[th] = np.nan

    if not return_features:
        return distances
    else:
        return distances, true_features, pred_features


def morphology_correlations(
    label_image_true: np.ndarray,
    label_image_pred: np.ndarray,
    features: list[str],
    thresholds: list[float],
    matches_per_threshold: dict[float, tuple[np.ndarray, np.ndarray]],
    min_max_ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[float, dict[str, float]]:
    """Calculate morphology correlations for matched masks using precomputed matches."""
    true_labels, true_features = extract_features(
        label_image_true, features, min_max_ranges
    )
    pred_labels, pred_features = extract_features(
        label_image_pred, features, min_max_ranges
    )

    correlations: dict[float, dict[str, float]] = {}
    for th in thresholds:
        true_ind, pred_ind = matches_per_threshold[th]
        true_indices = np.isin(true_labels, true_ind)
        pred_indices = np.isin(pred_labels, pred_ind)
        if true_indices.any() and pred_indices.any():
            filtered_true_feats = true_features[true_indices]
            filtered_pred_feats = pred_features[pred_indices]
            if (
                filtered_true_feats.shape[0] > 1
                and filtered_true_feats.shape == filtered_pred_feats.shape
            ):
                correlations[th] = _calculate_correlations(
                    filtered_true_feats, filtered_pred_feats, features
                )
            else:
                correlations[th] = dict(zip(features, [np.nan] * len(features)))
        else:
            correlations[th] = dict(zip(features, [np.nan] * len(features)))

    return correlations


def _calculate_correlations(
    true_matrix: np.ndarray, pred_matrix: np.ndarray, features: list[str]
) -> dict[str, float]:
    """Calculate Pearson correlation between true and pred features."""
    if true_matrix.size > 0 and pred_matrix.size > 0:
        correlation_matrix = np.corrcoef(true_matrix.T, pred_matrix.T)[
            : len(features), len(features) :
        ]
        return dict(zip(features, np.diag(correlation_matrix)))
    else:
        return dict(zip(features, [np.nan] * len(features)))
