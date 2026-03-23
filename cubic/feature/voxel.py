"""Implement extraction of 3D voxel-based morphological features.

Executing regionprops on GPU currently does not provide performance gains, see:
https://github.com/rapidsai/cucim/issues/241
"""

import numpy as np
import numpy.typing as npt

from ..skimage import measure


def regionprops(
    label_image: npt.ArrayLike,
    intensity_image: npt.ArrayLike | None = None,
    spacing: list[float] | None = None,
) -> list:
    """Extract region-based morphological features."""
    # cucim requires spacing as tuple for kernel memoization
    spacing_arg = tuple(spacing) if spacing is not None else None
    return measure.regionprops(label_image, intensity_image, spacing=spacing_arg)


def regionprops_table(
    label_image: npt.ArrayLike,
    intensity_image: npt.ArrayLike | None = None,
    properties: list[str] | None = None,
    spacing: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Extract region-based morphological features and return in pandas-compatible format."""
    if properties is not None:
        properties = list(set(properties + ["label"]))
    else:
        properties = []
    # cucim requires spacing as tuple for kernel memoization
    spacing_arg = tuple(spacing) if spacing is not None else None
    return measure.regionprops_table(
        label_image, intensity_image, properties=properties, spacing=spacing_arg
    )


def extract_features(
    label_image: npt.ArrayLike,
    features: list[str],
    feature_ranges: dict[str, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features from a label image using regionprops."""
    image_props = regionprops_table(label_image, properties=features)
    selected_features = sorted(
        [
            feat
            for feat in image_props.keys()
            if feat != "label" and feat.split("-")[0] in features
        ]
    )
    feature_values = np.column_stack([image_props[feat] for feat in selected_features])
    if feature_ranges is not None:
        assert all(feature in feature_ranges for feature in selected_features), (
            "Mismatch between extracted features and provided ranges."
        )
        feature_values = norm_features_by_range(
            feature_values, selected_features, feature_ranges
        )
    labels = image_props["label"]
    return labels, feature_values


def norm_features_by_range(
    feature_values: npt.ArrayLike,
    features: list[str],
    min_max_ranges: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Normalize feature values using min-max scaling."""
    mins = np.asarray([min_max_ranges[feature][0] for feature in features])
    ranges = np.asarray(
        [
            min_max_ranges[feature][1] - min_max_ranges[feature][0]
            for feature in features
        ]
    )
    ranges[ranges == 0] = np.finfo(np.float32).eps
    return (feature_values - mins) / ranges


def calculate_feature_percentiles(
    regionprops_dict: dict[str, np.ndarray],
    percentiles: tuple[float, ...] = (0.01, 0.99),
    features_include: list[str] | None = None,
    features_exclude: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Calculate percentiles of features in regionprops."""
    percentile_values = np.asarray(percentiles) * 100
    if features_include is not None:
        features = [f for f in features_include if f in regionprops_dict]
    else:
        features = list(regionprops_dict.keys())
    if features_exclude is not None:
        features = [f for f in features if f not in features_exclude]
    return {f: np.percentile(regionprops_dict[f], percentile_values) for f in features}
