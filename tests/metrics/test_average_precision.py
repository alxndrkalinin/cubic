"""Tests for detection metrics based on IoU."""

import warnings

import numpy as np
import pytest

from cubic.metrics.average_precision import (
    compute_matches,
    average_precision,
    _intersection_over_union,
)


def test_average_precision_supplied_matches_nonsequential_labels() -> None:
    """Supplied matches with gapped label ids give correct FP/FN/AP.

    ``.max()`` over-counted objects when label ids were non-sequential and the
    sequential-label check (only in ``compute_matches``) was bypassed.
    """
    mask_true = np.array([[0, 5], [5, 9]], dtype=np.int32)  # 2 objects: 5, 9
    mask_pred = mask_true.copy()
    matches = {0.5: (np.array([5, 9]), np.array([5, 9]))}
    ap, tp, fp, fn = average_precision(
        mask_true, mask_pred, [0.5], matches_per_threshold=matches
    )
    assert tp[0] == 2
    assert fp[0] == 0
    assert fn[0] == 0
    assert ap[0] == 1.0


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


# ===================================================================
# Regression tests — label validation and dtype handling
# ===================================================================


def _two_object_labels(dtype) -> np.ndarray:
    """Return an 8x8 label image with background and two objects."""
    lab = np.zeros((8, 8), dtype=dtype)
    lab[1:4, 1:4] = 1
    lab[5:7, 5:7] = 2
    return lab


@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int32, np.uint16])
def test_compute_matches_accepts_integral_float_labels(dtype) -> None:
    """Integer-valued float labels work on CPU, as they already did on GPU.

    ``_label_overlap_gpu`` cast to ``uint32`` but ``_label_overlap_cpu``
    did not, so a float label image (common after an I/O round-trip)
    raised ``numba.TypingError: Unsupported array index type float64`` on
    CPU while succeeding on GPU.
    """
    lab = _two_object_labels(dtype)
    matches = compute_matches(lab, lab.copy(), [0.5])
    np.testing.assert_array_equal(matches[0.5][0], [1, 2])
    np.testing.assert_array_equal(matches[0.5][1], [1, 2])


def test_compute_matches_cpu_gpu_agree_on_float_labels(gpu_available: bool) -> None:
    """The same float label image gives the same result on both devices."""
    if not gpu_available:
        pytest.skip("GPU not available")
    from cubic.cuda import ascupy, asnumpy

    lab = _two_object_labels(np.float64)
    cpu = _intersection_over_union(lab, lab.copy())
    gpu = asnumpy(_intersection_over_union(ascupy(lab), ascupy(lab.copy())))
    np.testing.assert_allclose(cpu, gpu)


def test_label_overlap_gpu_kernel_is_reused_across_calls(gpu_available: bool) -> None:
    """``_label_overlap_gpu`` calls the cached kernel factory, not indexes it.

    The raw kernel is compiled once behind ``@cache`` and must be *called*
    to get the kernel before the launch-config subscript; indexing the
    factory itself raises ``TypeError: _label_overlap_kernel() takes 0
    positional arguments but 4 were given``. Two calls also confirm the
    same compiled kernel object is reused rather than re-wrapped.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    from cubic.cuda import ascupy, asnumpy
    from cubic.metrics.average_precision import (
        _label_overlap_cpu,
        _label_overlap_gpu,
        _label_overlap_kernel,
    )

    lab_a = _two_object_labels(np.uint32)
    lab_b = _two_object_labels(np.uint32)
    expected = _label_overlap_cpu(lab_a, lab_b)

    first = asnumpy(_label_overlap_gpu(ascupy(lab_a), ascupy(lab_b)))
    second = asnumpy(_label_overlap_gpu(ascupy(lab_a), ascupy(lab_b)))
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)
    assert _label_overlap_kernel() is _label_overlap_kernel()


def test_non_integral_float_labels_raise() -> None:
    """Fractional label values are rejected rather than silently truncated.

    ``_intersection_over_union`` is checked directly because
    ``compute_matches`` first rejects the same input on contiguity
    grounds — fractional ids are not consecutive either.
    """
    lab = _two_object_labels(np.float64)
    lab[1, 1] = 1.7
    with pytest.raises(ValueError, match="integer values"):
        _intersection_over_union(lab, lab.copy())
    with pytest.raises(ValueError):
        compute_matches(lab, lab.copy(), [0.5])


def test_negative_labels_raise() -> None:
    """Negative labels would wrap around in the ``uint32`` cast."""
    lab = _two_object_labels(np.int32)
    lab[1, 1] = -1
    with pytest.raises(ValueError, match="negative labels"):
        compute_matches(lab, lab.copy(), [0.5])


def test_fully_labelled_mask_reports_missing_background() -> None:
    """A mask with no background says so, instead of blaming label ordering.

    ``compute_matches(np.array([[1, 1], [2, 2]]), ...)`` used to raise
    "mask_true should have sequential labels" even though the labels 1, 2
    *are* sequential; the real complaint was the absent background.
    """
    full = np.array([[1, 1], [2, 2]], dtype=np.int32)
    with pytest.raises(ValueError, match="mask_true has no background pixels"):
        compute_matches(full, full.copy(), [0.5])


def test_fully_labelled_pred_reports_missing_background() -> None:
    """The same check names ``mask_pred`` when only the prediction is full."""
    ok = np.array([[0, 1], [0, 2]], dtype=np.int32)
    full = np.array([[1, 1], [2, 2]], dtype=np.int32)
    with pytest.raises(ValueError, match="mask_pred has no background pixels"):
        compute_matches(ok, full, [0.5])


def test_gapped_labels_still_report_non_sequential() -> None:
    """A genuine gap in the label ids still raises the contiguity error."""
    gapped = np.array([[0, 1], [0, 5]], dtype=np.int32)
    with pytest.raises(ValueError, match="sequential labels"):
        compute_matches(gapped, gapped.copy(), [0.5])


def test_all_background_mask_reports_no_masks_to_match() -> None:
    """An entirely-background mask has nothing to assign (documented)."""
    empty = np.zeros((4, 4), dtype=np.int32)
    with pytest.raises(ValueError, match="No masks to match"):
        compute_matches(empty, empty.copy(), [0.5])


# ===================================================================
# Regression tests — IoU matrix shape and background handling
# ===================================================================


def test_intersection_over_union_drops_background() -> None:
    """The IoU matrix is object-to-object; callers no longer slice it."""
    lab = _two_object_labels(np.int32)
    iou = _intersection_over_union(lab, lab.copy())
    assert iou.shape == (2, 2)
    np.testing.assert_allclose(np.diag(iou), [1.0, 1.0])


def test_intersection_over_union_no_background_emits_no_warning() -> None:
    """A mask with no background no longer divides 0/0 at the background entry.

    The background row/column is consumed by the per-label pixel counts and
    dropped before the division, so the ``RuntimeWarning: invalid value
    encountered in divide`` is gone.
    """
    full = np.array([[1, 1], [2, 2]], dtype=np.int32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        iou = _intersection_over_union(full, full.copy())
    assert iou.shape == (2, 2)
    np.testing.assert_allclose(np.diag(iou), [1.0, 1.0])


def test_average_precision_counts_match_unique_based_counting() -> None:
    """Deriving object counts from the IoU shape matches the old ``np.unique``.

    ``average_precision`` used to run two full ``np.unique`` sorts over the
    input volumes on every call; when it computes the matches itself the
    counts are already in the IoU matrix shape.
    """
    mask_true = np.zeros((12, 12), dtype=np.int32)
    mask_true[1:4, 1:4] = 1
    mask_true[6:9, 6:9] = 2
    mask_true[1:4, 6:9] = 3
    mask_pred = np.zeros((12, 12), dtype=np.int32)
    mask_pred[1:4, 1:5] = 1  # overlaps true 1
    mask_pred[6:9, 6:9] = 2  # exact match with true 2

    thresholds = [0.5]
    ap, tp, fp, fn = average_precision(mask_true, mask_pred, thresholds)
    n_true = int(np.count_nonzero(np.unique(mask_true)))
    n_pred = int(np.count_nonzero(np.unique(mask_pred)))
    assert tp[0] == 2
    assert fp[0] == n_pred - tp[0]
    assert fn[0] == n_true - tp[0]
    assert ap[0] == pytest.approx(tp[0] / (tp[0] + fp[0] + fn[0]))


def test_average_precision_return_iou_false_still_four_tuple() -> None:
    """Computing the IoU internally for its shape must not change the return."""
    lab = _two_object_labels(np.int32)
    assert len(average_precision(lab, lab.copy(), [0.5])) == 4
    assert len(average_precision(lab, lab.copy(), [0.5], return_iou=True)) == 5
