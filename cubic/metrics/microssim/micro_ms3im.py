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

from ..ms_ssim import ms_ssim
from .micro_ssim import MicroSSIM
from .image_processing import normalize_min_max


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
            Accepted for upstream signature parity but ignored for MS-SSIM
            (matches upstream ``micro_ms3im.py:162-166``).
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
        if not self._initialized:
            raise ValueError("MicroSSIM was not initialized, call `fit()` first.")
        if gt.shape != pred.shape:
            raise ValueError("Groundtruth and prediction must have the same shape.")
        if gt.ndim != 2:
            raise ValueError("Only 2D images are supported.")
        if return_individual_components:
            warnings.warn(
                "`return_individual_components` is not supported for "
                "the MS-SSIM metric. Ignoring it.",
                stacklevel=2,
            )

        # After the _initialized check above, all four params are non-None
        # (enforced by __init__ + fit). Cast to silence mypy union narrowing.
        offset_gt = cast(float, self._offset_gt)
        offset_pred = cast(float, self._offset_pred)
        max_val = cast(float, self._max_val)
        ri_factor = cast(float, self._ri_factor)
        gt_norm = cast(np.ndarray, normalize_min_max(gt, offset_gt, max_val))
        pred_norm = (
            cast(np.ndarray, normalize_min_max(pred, offset_pred, max_val)) * ri_factor
        )
        data_range = float(gt_norm.max() - gt_norm.min())
        # Upstream calls torchmetrics with (pred_torch, gt_torch); we mirror
        # the argument order (micro_ms3im.py:200). SSIM is symmetric in its
        # two image arguments, so order doesn't affect the score.
        return ms_ssim(pred_norm, gt_norm, data_range=data_range, **ms_ssim_kwargs)


def micro_multiscale_structural_similarity(
    gt: np.ndarray | list[np.ndarray],
    pred: np.ndarray | list[np.ndarray],
) -> float | list[float]:
    """Fit a MicroMS3IM on ``(gt, pred)`` then score each slice.

    Mirrors upstream's stack/list semantics from
    ``micro_ssim.py:172-203`` adapted for the multi-scale subclass.
    """
    m3 = MicroMS3IM().fit(gt, pred)
    if isinstance(gt, list):
        pred_l = cast(list[np.ndarray], pred)
        return [float(m3.score(g, p)) for g, p in zip(gt, pred_l)]
    pred_a = cast(np.ndarray, pred)
    if gt.ndim == 3:
        return [float(m3.score(gt[i], pred_a[i])) for i in range(gt.shape[0])]
    return float(m3.score(gt, pred_a))
