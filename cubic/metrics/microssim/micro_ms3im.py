"""MicroMS3IM: multi-scale variant of MicroSSIM.

Port of ``juglab/microssim@8bccb17d`` ``MicroMS3IM`` (``micro_ms3im.py:127-200``).
Inherits fit-time behavior from :class:`MicroSSIM` and overrides ``score()`` to
delegate the multi-scale SSIM computation to :func:`cubic.metrics.ms_ssim`,
which is numerically faithful to ``torchmetrics``'s
``MultiScaleStructuralSimilarityIndexMeasure``.
"""

from __future__ import annotations

import warnings
from typing import Any, cast

import numpy as np

from ...cuda import asnumpy
from ..ms_ssim import (
    _MS_SSIM_DEFAULTS,
    ms_ssim,
    _ms_ssim_per_image,
    _validate_ms_ssim_shape,
)
from .micro_ssim import MicroSSIM

# Pixels scored per batch in :meth:`MicroMS3IM.score_stack`. Each batch holds
# ~15 float64 temporaries of its size, ~2 GB at this bound.
_STACK_BATCH_PIXELS = 1 << 24


class MicroMS3IM(MicroSSIM):
    """Multi-scale MicroSSIM. Inherits fit; overrides score to use MS-SSIM."""

    def score(
        self,
        gt: np.ndarray,
        pred: np.ndarray,
        return_individual_components: bool = False,
        **ms_ssim_kwargs: Any,
    ) -> float:
        """Compute MicroMS3IM between two 2-D images.

        Parameters
        ----------
        gt, pred : np.ndarray
            2-D ground-truth and prediction images with matching shapes.
        return_individual_components : bool, default=False
            Accepted for upstream signature parity but ignored for MS-SSIM,
            which has no per-component decomposition to return. Passing
            ``True`` emits a ``UserWarning`` and is otherwise a no-op
            (matches upstream ``micro_ms3im.py:162-166``; the single-scale
            :meth:`MicroSSIM.score` raises ``NotImplementedError`` instead).
        **ms_ssim_kwargs
            Forwarded to :func:`cubic.metrics.ms_ssim` (e.g.,
            ``kernel_size``, ``sigma``, ``betas``).

        Returns
        -------
        float
            Multi-scale SSIM score on the per-call data range.

        Raises
        ------
        ValueError
            If ``fit()`` has not been called, gt/pred shapes differ, or
            ``gt.ndim != 2``.
        """
        if return_individual_components:
            warnings.warn(
                "`return_individual_components` is not supported for "
                "the MS-SSIM metric. Ignoring it.",
                stacklevel=2,
            )
        gt_norm, pred_scaled, data_range = self._prepare(gt, pred)
        # Upstream calls torchmetrics with (pred_torch, gt_torch); we mirror
        # the argument order (micro_ms3im.py:200). SSIM is symmetric in its
        # two image arguments, so order doesn't affect the score.
        return ms_ssim(pred_scaled, gt_norm, data_range=data_range, **ms_ssim_kwargs)

    def score_stack(
        self,
        gt: np.ndarray,
        pred: np.ndarray,
        *,
        degenerate: float = np.nan,
        **ms_ssim_kwargs: Any,
    ) -> np.ndarray:
        """Score every slice of a ``(N, H, W)`` stack in batched passes.

        Each slice gets the score :meth:`score` gives it alone, with its own
        normalized ground-truth data range. Batching only changes the order of
        floating-point reductions, and the scores agree to about 1e-15. A slice
        whose ground-truth range is not finite and positive, which makes
        :meth:`score` raise, scores ``degenerate`` instead.

        Parameters
        ----------
        gt, pred : np.ndarray
            ``(N, H, W)`` ground-truth and prediction stacks of matching shape.
        degenerate : float, default=nan
            Score of a slice whose normalized ground truth is constant or
            non-finite.
        **ms_ssim_kwargs
            Forwarded to the MS-SSIM computation, as in :meth:`score` (e.g.
            ``kernel_size``, ``sigma``, ``betas``).

        Returns
        -------
        np.ndarray
            ``(N,)`` float64 scores, on the host.

        Raises
        ------
        ValueError
            If ``fit()`` has not been called, the shapes differ, the stacks are
            not 3-D, or the slices are too small for ``len(betas)`` scales.
        TypeError
            If ``ms_ssim_kwargs`` names a parameter :func:`ms_ssim` lacks, or
            ``data_range``, which is computed per slice.
        """
        self._check_pair(gt, pred)
        if gt.ndim != 3:
            raise ValueError(f"Expected a (N, H, W) stack; got ndim={gt.ndim}.")
        unknown = set(ms_ssim_kwargs) - set(_MS_SSIM_DEFAULTS)
        if unknown:
            raise TypeError(f"Unexpected MS-SSIM arguments: {sorted(unknown)}")
        params = {**_MS_SSIM_DEFAULTS, **ms_ssim_kwargs}
        _validate_ms_ssim_shape(
            gt.shape, len(params["betas"]), params["kernel_size"], params["sigma"]
        )
        n, h, w = gt.shape
        scores = np.full(n, degenerate, dtype=np.float64)
        step = max(1, _STACK_BATCH_PIXELS // (h * w))
        for start in range(0, n, step):
            batch = slice(start, start + step)
            gt_norm, pred_scaled = self._normalize_pair(gt[batch], pred[batch])
            data_range = gt_norm.max(axis=(1, 2)) - gt_norm.min(axis=(1, 2))
            valid = asnumpy(np.isfinite(data_range) & (data_range > 0))
            if not valid.any():
                continue
            if not valid.all():
                gt_norm, pred_scaled = gt_norm[valid], pred_scaled[valid]
                data_range = data_range[valid]
            # Argument order as in :meth:`score` (prediction first).
            batch_scores = _ms_ssim_per_image(
                pred_scaled, gt_norm, data_range=data_range, **params
            )
            scores[start : start + len(valid)][valid] = asnumpy(
                cast(np.ndarray, batch_scores)
            )
        return scores


def micro_multiscale_structural_similarity(
    gt: np.ndarray | list[np.ndarray],
    pred: np.ndarray | list[np.ndarray],
) -> float | list[float]:
    """Fit a :class:`MicroMS3IM` on ``(gt, pred)`` then score each slice.

    Thin wrapper over the inherited :meth:`MicroSSIM.fit_and_score`; see
    that method for the list / 3-D / 2-D return contract.
    """
    return MicroMS3IM.fit_and_score(gt, pred)
