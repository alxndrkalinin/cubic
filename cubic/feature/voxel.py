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
    extra_properties: tuple | None = None,
) -> list:
    """Extract region-based morphological features.

    Parameters
    ----------
    extra_properties : tuple of callable, optional
        User-defined property functions ``func(regionmask, intensity)``
        forwarded to ``measure.regionprops`` (both skimage and cucim
        accept them).
    """
    # cucim requires spacing as tuple for kernel memoization
    spacing_arg = tuple(spacing) if spacing is not None else None
    return measure.regionprops(
        label_image,
        intensity_image,
        spacing=spacing_arg,
        extra_properties=extra_properties,
    )


def regionprops_table(
    label_image: npt.ArrayLike,
    intensity_image: npt.ArrayLike | None = None,
    properties: list[str] | None = None,
    spacing: list[float] | None = None,
    extra_properties: tuple | None = None,
) -> dict[str, np.ndarray]:
    """Extract region-based morphological features and return in pandas-compatible format.

    Parameters
    ----------
    properties : list of str, optional
        Properties to compute. ``"label"`` is appended when missing so the
        table can always be keyed by label. When ``None``, the underlying
        ``measure.regionprops_table`` default (``("label", "bbox")``) is used.
    extra_properties : tuple of callable, optional
        User-defined property functions ``func(regionmask, intensity)``
        forwarded to ``measure.regionprops_table`` (both skimage and
        cucim accept them).
    """
    # cucim requires spacing as tuple for kernel memoization
    spacing_arg = tuple(spacing) if spacing is not None else None
    # Leaving the argument out lets ``regionprops_table`` apply its own default.
    # Passing an empty list instead returns an empty dict without even a ``label``
    # column, which is not a usable table.
    properties_kwarg: dict[str, list[str]] = {}
    if properties is not None:
        # Order-preserving dedup. A bare ``set()`` iterates in an order that
        # depends on the interpreter hash seed, which differs per spawned
        # process worker, so it would scramble the output column order across
        # multiprocessing workers and misalign the pooled feature matrix.
        properties_kwarg["properties"] = list(dict.fromkeys([*properties, "label"]))
    return measure.regionprops_table(
        label_image,
        intensity_image,
        spacing=spacing_arg,
        extra_properties=extra_properties,
        **properties_kwarg,
    )


def extract_features(
    label_image: npt.ArrayLike,
    features: list[str],
    feature_ranges: dict[str, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Extract features from a label image using regionprops.

    Returns
    -------
    labels : np.ndarray
        Label of each row, background excluded.
    feature_values : np.ndarray
        ``(n_labels, n_columns)`` feature matrix.
    column_names : list of str
        Name of each column, in column order. ``regionprops_table`` expands
        vector properties into one column per component (3D ``"centroid"``
        becomes ``"centroid-0"``, ``"centroid-1"``, ``"centroid-2"``), and the
        columns are sorted alphabetically, so neither the number of columns nor
        their order matches ``features``. Callers that key results by feature
        must use these names.
    """
    image_props = regionprops_table(label_image, properties=features)
    column_names = sorted(
        [
            feat
            for feat in image_props.keys()
            if feat != "label" and feat.split("-")[0] in features
        ]
    )
    feature_values = np.column_stack([image_props[feat] for feat in column_names])
    if feature_ranges is not None:
        missing = [feat for feat in column_names if feat not in feature_ranges]
        if missing:
            raise ValueError(
                "Mismatch between extracted features and provided ranges; "
                f"no range given for {missing}. Note that vector properties are "
                "expanded into per-component columns, which must be keyed "
                "individually (e.g. 'centroid-0', not 'centroid')."
            )
        feature_values = norm_features_by_range(
            feature_values, column_names, feature_ranges
        )
    labels = image_props["label"]
    return labels, feature_values, column_names


def norm_features_by_range(
    feature_values: npt.ArrayLike,
    features: list[str],
    min_max_ranges: dict[str, tuple[float, float]],
) -> np.ndarray:
    """Normalize feature values using min-max scaling."""
    # dtype=float is required: integer ranges would truncate the zero-range
    # epsilon below back to 0 and produce infinities.
    mins = np.asarray([min_max_ranges[feature][0] for feature in features], dtype=float)
    ranges = np.asarray(
        [
            min_max_ranges[feature][1] - min_max_ranges[feature][0]
            for feature in features
        ],
        dtype=float,
    )
    ranges[ranges == 0] = np.finfo(np.float32).eps
    return (feature_values - mins) / ranges


def calculate_feature_percentiles(
    regionprops_dict: dict[str, np.ndarray],
    percentiles: tuple[float, ...] = (0.01, 0.99),
    features_include: list[str] | None = None,
    features_exclude: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Calculate percentiles of features in regionprops.

    ``percentiles`` are fractions in ``[0, 1]`` (they are multiplied by 100
    before being passed to ``np.percentile``), so the 1st and 99th percentiles
    are ``(0.01, 0.99)`` and not ``(1, 99)``.
    """
    percentile_values = np.asarray(percentiles) * 100
    if features_include is not None:
        features = [f for f in features_include if f in regionprops_dict]
    else:
        features = list(regionprops_dict.keys())
    if features_exclude is not None:
        features = [f for f in features if f not in features_exclude]
    return {f: np.percentile(regionprops_dict[f], percentile_values) for f in features}
