"""Tests for ``cubic.metrics.microssim.image_processing``."""

import numpy as np
import pytest

from cubic.metrics.microssim.image_processing import (
    linearize_list,
    normalize_min_max,
    compute_norm_parameters,
)


class TestLinearizeList:
    """Tests for ``linearize_list``."""

    def test_ndarray_passes_through(self) -> None:
        """An ndarray input is returned unchanged (no flattening)."""
        arr = np.arange(12).reshape(3, 4)
        out = linearize_list(arr)
        # Same object — no copy, no ravel.
        assert out is arr

    def test_list_of_uniform_shapes_concatenates_and_flattens(self) -> None:
        """A list of three (2, 3) arrays produces a length-18 1-D array."""
        rng = np.random.default_rng(0)
        items = [rng.random((2, 3)) for _ in range(3)]
        out = linearize_list(items)
        assert out.ndim == 1
        assert out.shape == (18,)
        # Order matters: the concatenation flattens each in order.
        expected = np.concatenate([x.ravel() for x in items])
        np.testing.assert_array_equal(out, expected)

    def test_list_of_ragged_shapes_concatenates(self) -> None:
        """Ragged inputs (5x5 + 3x7) still concatenate via ravel."""
        rng = np.random.default_rng(1)
        a = rng.random((5, 5))
        b = rng.random((3, 7))
        out = linearize_list([a, b])
        assert out.ndim == 1
        assert out.shape == (25 + 21,)
        np.testing.assert_array_equal(out[:25], a.ravel())
        np.testing.assert_array_equal(out[25:], b.ravel())

    def test_empty_list_raises(self) -> None:
        """An empty list cannot be concatenated; ``np.concatenate`` raises."""
        with pytest.raises(ValueError):
            linearize_list([])


class TestComputeNormParameters:
    """Tests for ``compute_norm_parameters``."""

    def test_defaults_compute_from_data(self) -> None:
        """Default ``None`` parameters derive percentile + max from data."""
        rng = np.random.default_rng(2)
        gt = rng.random((4, 5)).astype(np.float64)
        pred = rng.random((4, 5)).astype(np.float64) * 2.0
        offset_gt, offset_pred, max_val = compute_norm_parameters(
            gt, pred, bg_percentile=3
        )
        expected_offset_gt = float(np.percentile(gt, 3))
        expected_offset_pred = float(np.percentile(pred, 3))
        expected_max_val = float((gt - expected_offset_gt).max())
        assert offset_gt == pytest.approx(expected_offset_gt)
        assert offset_pred == pytest.approx(expected_offset_pred)
        assert max_val == pytest.approx(expected_max_val)

    def test_bg_percentile_changes_offset(self) -> None:
        """``bg_percentile`` propagates to ``np.percentile`` calls."""
        rng = np.random.default_rng(3)
        gt = rng.random((10, 10))
        pred = rng.random((10, 10))
        offset_gt, offset_pred, _ = compute_norm_parameters(gt, pred, bg_percentile=50)
        assert offset_gt == pytest.approx(float(np.percentile(gt, 50)))
        assert offset_pred == pytest.approx(float(np.percentile(pred, 50)))

    def test_explicit_offsets_bypass_computation(self) -> None:
        """Supplied offsets are returned unchanged and skip the percentile call."""
        gt = np.full((3, 3), 5.0)
        pred = np.full((3, 3), 7.0)
        offset_gt, offset_pred, max_val = compute_norm_parameters(
            gt, pred, offset_gt=1.25, offset_pred=2.5
        )
        assert offset_gt == 1.25
        assert offset_pred == 2.5
        # ``max_val`` still gets computed from data using ``offset_gt``.
        assert max_val == pytest.approx(float((gt - 1.25).max()))

    def test_explicit_max_val_bypasses_computation(self) -> None:
        """Supplied ``max_val`` is returned unchanged."""
        gt = np.arange(20.0).reshape(4, 5)
        pred = np.arange(20.0).reshape(4, 5)
        _, _, max_val = compute_norm_parameters(gt, pred, max_val=42.0)
        assert max_val == 42.0

    def test_all_explicit_short_circuits(self) -> None:
        """All three params supplied → returned as-is, untouched by data."""
        gt = np.zeros((2, 2))
        pred = np.zeros((2, 2))
        out = compute_norm_parameters(
            gt, pred, offset_gt=1.0, offset_pred=2.0, max_val=3.0
        )
        assert out == (1.0, 2.0, 3.0)

    def test_list_input_uses_linearized_data(self) -> None:
        """List inputs are linearized before percentile/max computations."""
        rng = np.random.default_rng(4)
        gt_list = [rng.random((2, 3)) for _ in range(3)]
        pred_list = [rng.random((2, 3)) for _ in range(3)]
        offset_gt, _, max_val = compute_norm_parameters(
            gt_list, pred_list, bg_percentile=10
        )
        gt_flat = np.concatenate([x.ravel() for x in gt_list])
        expected_offset = float(np.percentile(gt_flat, 10))
        expected_max = float((gt_flat - expected_offset).max())
        assert offset_gt == pytest.approx(expected_offset)
        assert max_val == pytest.approx(expected_max)

    def test_cupy_percentile(self) -> None:
        """``np.percentile`` operates on CuPy arrays via duck typing."""
        cp = pytest.importorskip("cupy")
        rng = np.random.default_rng(5)
        gt_cpu = rng.random((6, 6))
        pred_cpu = rng.random((6, 6))
        gt = cp.asarray(gt_cpu)
        pred = cp.asarray(pred_cpu)
        offset_gt, offset_pred, max_val = compute_norm_parameters(
            gt, pred, bg_percentile=5
        )
        # All three are Python floats (cast via ``float(...)``).
        assert isinstance(offset_gt, float)
        assert isinstance(offset_pred, float)
        assert isinstance(max_val, float)
        # Values match the CPU computation within float tolerance.
        cpu_off_gt, cpu_off_pred, cpu_max = compute_norm_parameters(
            gt_cpu, pred_cpu, bg_percentile=5
        )
        assert offset_gt == pytest.approx(cpu_off_gt, rel=1e-6, abs=1e-9)
        assert offset_pred == pytest.approx(cpu_off_pred, rel=1e-6, abs=1e-9)
        assert max_val == pytest.approx(cpu_max, rel=1e-6, abs=1e-9)


class TestNormalizeMinMax:
    """Tests for ``normalize_min_max``."""

    def test_scalar_offset_and_max(self) -> None:
        """``(x - offset) / max_val`` is applied element-wise."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = normalize_min_max(x, offset=1.0, max_val=2.0)
        expected = (x - 1.0) / 2.0
        np.testing.assert_array_equal(out, expected)

    def test_divisor_is_max_val_not_max_minus_offset(self) -> None:
        """The divisor is ``max_val`` exactly — *not* ``max_val - offset``."""
        x = np.array([10.0, 20.0])
        out = normalize_min_max(x, offset=5.0, max_val=10.0)
        np.testing.assert_array_equal(out, (x - 5.0) / 10.0)
        # Sanity: confirm the (incorrect) alternative diverges here.
        wrong = (x - 5.0) / (10.0 - 5.0)
        assert not np.allclose(out, wrong)

    def test_list_input_returns_list(self) -> None:
        """A list input returns a list of normalized arrays."""
        rng = np.random.default_rng(6)
        items = [rng.random((2, 3)) for _ in range(4)]
        out = normalize_min_max(items, offset=0.1, max_val=2.0)
        assert isinstance(out, list)
        assert len(out) == 4
        for original, normalized in zip(items, out):
            np.testing.assert_array_equal(normalized, (original - 0.1) / 2.0)

    def test_list_recursion_is_one_level(self) -> None:
        """Each list element is normalized independently."""
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0, 5.0])
        out = normalize_min_max([a, b], offset=1.0, max_val=2.0)
        assert isinstance(out, list)
        np.testing.assert_array_equal(out[0], (a - 1.0) / 2.0)
        np.testing.assert_array_equal(out[1], (b - 1.0) / 2.0)

    def test_max_val_zero_yields_inf_or_nan(self) -> None:
        """``max_val=0`` does not raise — produces inf/nan via float division."""
        # NumPy float division by zero issues a RuntimeWarning and produces
        # inf/nan rather than raising. Document this behavior.
        x = np.array([1.0, 2.0])
        with np.errstate(divide="ignore", invalid="ignore"):
            out = normalize_min_max(x, offset=1.0, max_val=0.0)
        # First element: (1 - 1) / 0 = 0 / 0 → nan; second: 1 / 0 → inf.
        assert np.isnan(out[0])
        assert np.isinf(out[1])
