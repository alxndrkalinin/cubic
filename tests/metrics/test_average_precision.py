"""Tests for detection metrics based on IoU."""

import numpy as np

from cubic.metrics.average_precision import compute_matches, average_precision


def test_average_precision_perfect_match() -> None:
    """Perfect overlap should yield perfect precision."""
    mask_true = np.array([[0, 1], [0, 2]], dtype=np.int32)
    mask_pred = mask_true.copy()
    thresholds = [0.5]
    matches, _ = compute_matches(mask_true, mask_pred, thresholds, return_iou=True)
    ap, tp, fp, fn = average_precision(mask_true, mask_pred, thresholds, matches)
    assert ap[0] == 1
    assert tp[0] == 2
    assert fp[0] == 0
    assert fn[0] == 0


def test_average_precision_return_iou_matches_compute_matches() -> None:
    """return_iou should pass through the same IoU compute_matches produces."""
    mask_true = np.zeros((10, 10), dtype=np.int32)
    mask_true[1:4, 1:4] = 1
    mask_true[6:9, 6:9] = 2
    mask_pred = np.zeros((10, 10), dtype=np.int32)
    mask_pred[1:4, 1:5] = 1  # larger than true #1 -> IoU 0.75
    mask_pred[6:9, 6:9] = 2  # exact match -> IoU 1.0
    thresholds = [0.5]

    _, iou_ref = compute_matches(mask_true, mask_pred, thresholds, return_iou=True)

    # IoU computed internally (matches_per_threshold not supplied).
    result = average_precision(mask_true, mask_pred, thresholds, return_iou=True)
    assert len(result) == 5
    ap, tp, fp, fn, iou = result
    assert iou.shape == (2, 2)
    np.testing.assert_allclose(iou, iou_ref)
    np.testing.assert_allclose(np.diag(iou), [0.75, 1.0])

    # Default (return_iou=False) preserves the 4-tuple contract.
    assert len(average_precision(mask_true, mask_pred, thresholds)) == 4


def test_average_precision_return_iou_recomputes_for_supplied_matches() -> None:
    """When matches are supplied, return_iou recomputes the same overlap."""
    mask_true = np.zeros((8, 8), dtype=np.int32)
    mask_true[1:4, 1:4] = 1
    mask_pred = np.zeros((8, 8), dtype=np.int32)
    mask_pred[1:4, 1:5] = 1
    thresholds = [0.5]

    matches, iou_ref = compute_matches(
        mask_true, mask_pred, thresholds, return_iou=True
    )
    *_, iou = average_precision(
        mask_true, mask_pred, thresholds, matches_per_threshold=matches, return_iou=True
    )
    np.testing.assert_allclose(iou, iou_ref)
