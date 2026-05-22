"""Device-agnostic Pearson correlation coefficient."""

from __future__ import annotations

import numpy as np

from .skimage_metrics import scale_invariant


@scale_invariant
def pcc(
    image_true: np.ndarray,
    image_test: np.ndarray,
    mask: np.ndarray | None = None,
    **_unused_kwargs,
) -> float:
    """Compute the Pearson correlation coefficient between two images.

    Device-agnostic: accepts NumPy or CuPy arrays. The result is always
    a Python ``float``.

    Parameters
    ----------
    image_true, image_test : np.ndarray
        Images to compare. Must have the same shape.
    mask : np.ndarray, optional
        Boolean mask selecting voxels to include in the correlation. If
        ``None``, all voxels are used.

    Returns
    -------
    float
        Pearson *r* in [-1, 1]. Returns ``nan`` when either input has
        zero variance over the (optionally masked) region.

    Notes
    -----
    The ``@scale_invariant`` decorator adds a ``scale_invariant`` kwarg
    (default ``False``) and injects ``data_range`` into ``**kwargs`` when
    set. PCC is itself affine-invariant, so the flag is a no-op
    numerically but lets callers share the decorator API with
    ``psnr``/``nrmse``. ``**_unused_kwargs`` absorbs the injected
    ``data_range``.
    """
    if image_true.shape != image_test.shape:
        raise ValueError(
            f"Shape mismatch: image_true {image_true.shape} vs "
            f"image_test {image_test.shape}"
        )
    x = image_true[mask] if mask is not None else image_true
    y = image_test[mask] if mask is not None else image_test
    x = x.ravel().astype(np.float64)
    y = y.ravel().astype(np.float64)
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = float(np.sqrt(float((x_c * x_c).sum()) * float((y_c * y_c).sum())))
    if denom < 1e-12:
        return float("nan")
    return float(np.clip(float((x_c * y_c).sum()) / denom, -1.0, 1.0))
