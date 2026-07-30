"""MicroMS3IM: multi-scale variant of MicroSSIM.

Port of ``juglab/microssim@8bccb17d`` ``MicroMS3IM`` (``micro_ms3im.py:127-200``).
Inherits fit-time behavior from :class:`MicroSSIM` and overrides ``score()`` to
delegate the multi-scale SSIM computation to :func:`cubic.metrics.ms_ssim`,
which is numerically faithful to ``torchmetrics``'s
``MultiScaleStructuralSimilarityIndexMeasure``.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from ..ms_ssim import ms_ssim
from .micro_ssim import MicroSSIM


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


def micro_multiscale_structural_similarity(
    gt: np.ndarray | list[np.ndarray],
    pred: np.ndarray | list[np.ndarray],
) -> float | list[float]:
    """Fit a :class:`MicroMS3IM` on ``(gt, pred)`` then score each slice.

    Thin wrapper over the inherited :meth:`MicroSSIM.fit_and_score`; see
    that method for the list / 3-D / 2-D return contract.
    """
    return MicroMS3IM.fit_and_score(gt, pred)
