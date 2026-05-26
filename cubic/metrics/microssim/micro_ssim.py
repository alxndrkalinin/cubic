"""MicroSSIM class and single-image convenience wrapper.

Port of ``juglab/microssim@8bccb17d`` ``MicroSSIM`` (``micro_ssim.py:223-437``)
with the cubic-side substitutions documented in the design note: a
device-agnostic SSIM-element compute, percentile-normalization helpers,
and a scipy-free bisection RI-factor solver.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from .ri_factor import (
    ALPHA_MAX_DEFAULT,
    ALPHA_MIN_DEFAULT,
    get_global_ri_factor,
    _validate_alpha_bounds,
)
from .ssim_elements import compute_ssim_elements
from .image_processing import (
    normalize_min_max,
    compute_norm_parameters,
)


class MicroSSIM:
    """MicroSSIM: SSIM with a learned per-dataset range-invariance factor.

    Mirrors upstream ``juglab/microssim.MicroSSIM`` (`micro_ssim.py:223-437`):
    fit normalization parameters (percentile background offset and max value)
    plus a single RI factor on a dataset, then score individual 2-D image
    pairs against those fit parameters.

    Parameters
    ----------
    bg_percentile : float, default=3
        Percentile used to estimate the background offset for both ground
        truth and prediction.
    offset_gt, offset_pred : float, optional
        Pre-computed background offsets; bypass the percentile estimate when
        provided.
    max_val : float, optional
        Pre-computed max value used as the normalization divisor.
    ri_factor : float, optional
        Pre-computed RI factor. If supplied, all of ``offset_gt``,
        ``offset_pred``, and ``max_val`` must also be supplied; in that
        case ``score()`` can be called without first calling ``fit()``.
        Use this to pin ``alpha`` from an external calibration (e.g.
        load a per-(model, organelle) RI factor fit once and reuse it
        across all evaluation calls).
    alpha_min : float, default=:data:`ALPHA_MIN_DEFAULT` (``1e-6``)
        Lower bracket cap forwarded to :func:`get_global_ri_factor` during
        ``fit()``. Has no effect when ``ri_factor`` is supplied directly.
    alpha_max : float, default=:data:`ALPHA_MAX_DEFAULT` (``1e6``)
        Upper bracket cap forwarded to :func:`get_global_ri_factor` during
        ``fit()``. Has no effect when ``ri_factor`` is supplied directly.

    Raises
    ------
    ValueError
        If ``ri_factor`` is provided but any of the normalization
        parameters is missing (matches upstream ``micro_ssim.py:270-279``),
        or if ``alpha_min`` / ``alpha_max`` are out of range.
    """

    def __init__(
        self,
        bg_percentile: float = 3,
        offset_gt: float | None = None,
        offset_pred: float | None = None,
        max_val: float | None = None,
        ri_factor: float | None = None,
        *,
        alpha_min: float = ALPHA_MIN_DEFAULT,
        alpha_max: float = ALPHA_MAX_DEFAULT,
    ) -> None:
        _validate_alpha_bounds(alpha_min, alpha_max)
        self._bg_percentile = bg_percentile
        self._offset_pred = offset_pred
        self._offset_gt = offset_gt
        self._max_val = max_val
        self._ri_factor = ri_factor
        self._alpha_min = alpha_min
        self._alpha_max = alpha_max
        self._initialized = ri_factor is not None
        if self._initialized and any(
            p is None for p in (offset_gt, offset_pred, max_val)
        ):
            raise ValueError(
                "Please specify offset_pred, offset_gt and max_val "
                "if ri_factor is provided."
            )

    def fit(
        self,
        gt: np.ndarray | list[np.ndarray],
        pred: np.ndarray | list[np.ndarray],
    ) -> "MicroSSIM":
        """Fit normalization parameters and the RI factor on a dataset.

        Parameters
        ----------
        gt, pred : np.ndarray or list of np.ndarray
            Ground-truth and prediction images. If arrays, must share the
            same shape with ``ndim`` in ``{2, 3}``. If lists, must have the
            same length.

        Returns
        -------
        MicroSSIM
            ``self``, to enable chaining.

        Raises
        ------
        ValueError
            If gt/pred are different types (one list, one array), have
            different list lengths, different array shapes, or are arrays
            with ndim outside ``{2, 3}``. Mirrors upstream
            ``micro_ssim.py:335-352``.
        """
        if isinstance(gt, list) != isinstance(pred, list):
            raise ValueError("Images must be of the same type (list or numpy.ndarray).")
        if isinstance(gt, list):
            pred_l = cast(list[np.ndarray], pred)
            if len(gt) != len(pred_l):
                raise ValueError("Lists must have the same length.")
        else:
            pred_a = cast(np.ndarray, pred)
            if gt.shape != pred_a.shape:
                raise ValueError(
                    f"Images must have the same shape "
                    f"(got {gt.shape} and {pred_a.shape})."
                )
            if gt.ndim < 2 or gt.ndim > 3:
                raise ValueError("Only 2D or 3D images are supported.")

        self._offset_gt, self._offset_pred, self._max_val = compute_norm_parameters(
            gt,
            pred,
            bg_percentile=self._bg_percentile,
            offset_gt=self._offset_gt,
            offset_pred=self._offset_pred,
            max_val=self._max_val,
        )
        gt_norm = normalize_min_max(gt, self._offset_gt, self._max_val)
        pred_norm = normalize_min_max(pred, self._offset_pred, self._max_val)

        if isinstance(gt_norm, list):
            pred_norm_list = cast(list[np.ndarray], pred_norm)
            shapes = {x.shape for x in gt_norm}
            if len(shapes) > 1:
                raise NotImplementedError(
                    f"Ragged input shapes not supported in v1: {shapes}"
                )
            gt_norm = np.concatenate([x[None] if x.ndim == 2 else x for x in gt_norm])
            pred_norm = np.concatenate(
                [x[None] if x.ndim == 2 else x for x in pred_norm_list]
            )

        gt_norm = cast(np.ndarray, gt_norm)
        pred_norm = cast(np.ndarray, pred_norm)
        if gt_norm.ndim == 2:
            gt_norm = gt_norm[None]
            pred_norm = pred_norm[None]

        # Upstream-faithful per-slice elements + pooled RI fit: each slice
        # gets its own data_range; C1, C2 come from the LAST slice
        # (ri_factor.py:123-131). gaussian_weights=False, win_size=7,
        # crop=True is the fit-time path — these are the defaults of
        # get_global_ri_factor (matches upstream ri_factor.py:89).
        self._ri_factor = get_global_ri_factor(
            gt_norm,
            pred_norm,
            alpha_min=self._alpha_min,
            alpha_max=self._alpha_max,
        )
        self._initialized = True
        return self

    def score(
        self,
        gt: np.ndarray,
        pred: np.ndarray,
        return_individual_components: bool = False,
        **kwargs: Any,
    ) -> float:
        """Compute MicroSSIM between two 2-D images.

        Parameters
        ----------
        gt, pred : np.ndarray
            2-D ground-truth and prediction images with matching shapes.
        return_individual_components : bool, default=False
            Currently unused; accepted for upstream-signature parity.
        **kwargs
            Forwarded to :func:`compute_ssim_elements` (e.g., ``sigma``,
            ``K1``, ``K2``) to match upstream's pass-through contract at
            ``micro_ssim.py:377-383``.

        Returns
        -------
        float
            Scalar mean of the SSIM map on the cropped extent.

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

        # After the _initialized check, params are guaranteed non-None;
        # cast to silence mypy union narrowing.
        offset_gt = cast(float, self._offset_gt)
        offset_pred = cast(float, self._offset_pred)
        max_val = cast(float, self._max_val)
        ri_factor = cast(float, self._ri_factor)
        gt_norm = cast(np.ndarray, normalize_min_max(gt, offset_gt, max_val))
        pred_norm = cast(np.ndarray, normalize_min_max(pred, offset_pred, max_val))
        data_range = float(gt_norm.max() - gt_norm.min())
        # Score path: gaussian_weights=True matches _compute_micro_ssim's
        # default at micro_ssim.py:31 (gaussian filter, sigma=1.5, win_size=11).
        e = compute_ssim_elements(
            gt_norm,
            pred_norm * ri_factor,
            data_range=data_range,
            gaussian_weights=True,
            crop=True,
            **kwargs,
        )
        a1 = 2.0 * e.ux * e.uy + e.C1
        a2 = 2.0 * e.vxy + e.C2
        b1 = e.ux**2 + e.uy**2 + e.C1
        b2 = e.vx + e.vy + e.C2
        return float(((a1 * a2) / (b1 * b2)).mean())

    def get_parameters(self) -> dict[str, float | None]:
        """Return the fitted parameters.

        Returns
        -------
        dict
            Keys ``bg_percentile``, ``offset_pred``, ``offset_gt``,
            ``max_val``, ``ri_factor`` matching upstream
            ``micro_ssim.py:296-302`` exactly.

        Notes
        -----
        The key set intentionally mirrors upstream ``MicroSSIM``. The
        cubic-only ``alpha_min`` / ``alpha_max`` knobs are **not**
        round-tripped here — a ``MicroSSIM(**ms.get_parameters())``
        re-instantiation reverts them to :data:`ALPHA_MIN_DEFAULT` /
        :data:`ALPHA_MAX_DEFAULT`. Save them separately if non-default
        caps are part of your configuration.
        """
        return {
            "bg_percentile": self._bg_percentile,
            "offset_pred": self._offset_pred,
            "offset_gt": self._offset_gt,
            "max_val": self._max_val,
            "ri_factor": self._ri_factor,
        }


def micro_structural_similarity(
    gt: np.ndarray | list[np.ndarray],
    pred: np.ndarray | list[np.ndarray],
) -> float | list[float]:
    """Fit a MicroSSIM on ``(gt, pred)`` then score each slice.

    Mirrors upstream ``micro_ssim.py:172-203``: returns a list of per-slice
    floats for list / 3-D inputs, or a single float for a 2-D pair. Do NOT
    call ``MicroSSIM.score`` on a 3-D stack — upstream raises on
    ``ndim != 2``.
    """
    ms = MicroSSIM().fit(gt, pred)
    if isinstance(gt, list):
        pred_l = cast(list[np.ndarray], pred)
        return [float(ms.score(g, p)) for g, p in zip(gt, pred_l)]
    pred_a = cast(np.ndarray, pred)
    if gt.ndim == 3:
        return [float(ms.score(gt[i], pred_a[i])) for i in range(gt.shape[0])]
    return float(ms.score(gt, pred_a))
