"""Calculate feature-based metrics for morphology comparison."""

import numpy as np

from cubic.cuda import asnumpy, to_same_device
from cubic.feature.voxel import extract_features, norm_features_by_range
from cubic.metrics.average_precision import compute_matches


def _calculate_cosine(true_features: np.ndarray, pred_features: np.ndarray) -> float:
    """Calculate cosine distance between true and pred features."""
    norm_true = np.linalg.norm(true_features)
    norm_pred = np.linalg.norm(pred_features)
    if norm_true == 0 or norm_pred == 0:
        return 1.0
    # float() also brings a GPU result back to the host, so the returned metric
    # is a plain scalar on both devices.
    return float(1 - np.dot(true_features, pred_features) / (norm_true * norm_pred))


def _nan_dict(feature_names: list[str]) -> dict[str, float]:
    """Return a NaN value for every feature column."""
    return dict.fromkeys(feature_names, np.nan)


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
    """Extract or retrieve true features based on precomputed or provided label images.

    The returned feature names are the *base* property names (as accepted by
    ``regionprops_table``), not the expanded column names of the feature matrix.
    """
    if gt_feature_dict:
        gt_features = gt_feature_dict.keys()
        if "label" not in gt_features:
            raise ValueError("Label column not found in precomputed features.")
        numeric_features = sorted([feat for feat in gt_features if feat != "label"])

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
            missing = [
                feat for feat in features if feat != "label" and feat not in gt_base
            ]
            if missing:
                raise ValueError(
                    f"Some features not found in precomputed features: {missing}"
                )

        true_labels = gt_feature_dict["label"]

        stacked = np.stack([gt_feature_dict[f] for f in numeric_features], axis=1)
        if not feature_ranges:
            # No ranges given: return raw values, matching the ``extract_features``
            # branch below, which likewise does not normalize when
            # ``feature_ranges is None``. Deriving ranges from the ground truth
            # here would scale the true features while callers extract the
            # predicted ones unnormalized, putting the two on different scales.
            true_features = stacked
        else:
            true_features = norm_features_by_range(
                stacked, numeric_features, feature_ranges
            )
    else:
        true_labels, true_features, _ = extract_features(
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
    pred_labels, pred_features, _ = extract_features(
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
    """Calculate morphology correlations for matched masks using precomputed matches.

    The returned dicts are keyed by the *expanded* feature column names, since
    ``extract_features`` splits vector properties into one column each (3D
    ``"centroid"`` becomes ``"centroid-0"``, ``"centroid-1"``, ``"centroid-2"``).
    """
    true_labels, true_features, feature_names = extract_features(
        label_image_true, features, min_max_ranges
    )
    pred_labels, pred_features, pred_feature_names = extract_features(
        label_image_pred, features, min_max_ranges
    )
    if feature_names != pred_feature_names:
        raise ValueError(
            "True and predicted label images yielded different feature columns: "
            f"{feature_names} vs {pred_feature_names}"
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
                    filtered_true_feats, filtered_pred_feats, feature_names
                )
            else:
                correlations[th] = _nan_dict(feature_names)
        else:
            correlations[th] = _nan_dict(feature_names)

    return correlations


def _calculate_correlations(
    true_matrix: np.ndarray, pred_matrix: np.ndarray, feature_names: list[str]
) -> dict[str, float]:
    """Calculate Pearson correlation between true and pred features.

    ``np.corrcoef`` of the two stacked matrices is ``(2 * n, 2 * n)`` for ``n``
    feature columns; its upper-right ``(n, n)`` block holds the true-vs-pred
    correlations, whose diagonal is the per-feature correlation. ``n`` must be the
    matrix width and not the number of requested properties: vector properties are
    expanded into several columns each, and too small an ``n`` silently returns
    off-block entries, which are true-vs-true correlations.
    """
    n = true_matrix.shape[1]
    correlation_matrix = np.corrcoef(true_matrix.T, pred_matrix.T)[:n, n:]
    # asnumpy keeps the returned values plain floats for GPU inputs too.
    diagonal = asnumpy(np.diag(correlation_matrix))
    return dict(zip(feature_names, (float(value) for value in diagonal)))
