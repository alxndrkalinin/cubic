"""Implement the Richardson-Lucy deconvolution algorithm using either NumPy or CuPy.

Modified from https://github.com/True-North-Intelligent-Algorithms/tnia-python/blob/main/tnia/deconvolution/richardson_lucy.py

Original license:
--------------------
BSD 3-Clause License

Copyright (c) 2021, True-North-Intelligent-Algorithms
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from collections.abc import Callable

import numpy as np

from cubic.cuda import check_same_device
from cubic.skimage import util
from cubic.image_utils import crop_center, pad_image_to_shape


def _check_psf_fits(image: np.ndarray, psf: np.ndarray, name: str = "psf") -> None:
    """Ensure ``psf`` can be centered into ``image`` without cropping.

    ``pad_image_to_shape`` only pads, so an oversized PSF would otherwise trip
    its bare shape assertion with no explanation.
    """
    if psf.ndim != image.ndim:
        raise ValueError(
            f"{name} must have the same number of axes as the image; got "
            f"{name}.shape={tuple(psf.shape)} and image.shape={tuple(image.shape)}."
        )
    if any(p > i for p, i in zip(psf.shape, image.shape)):
        raise ValueError(
            f"{name}.shape={tuple(psf.shape)} must not exceed "
            f"image.shape={tuple(image.shape)} on any axis (use noncirc=True to "
            "extend the image instead)."
        )


def richardson_lucy_xp(
    image: np.ndarray,
    psf: np.ndarray,
    n_iter: int = 10,
    *,
    noncirc: bool = False,
    mask: np.ndarray | None = None,
    observer_fn: Callable | None = None,
    backprojector: np.ndarray | None = None,
    small_value: float | None = None,
) -> np.ndarray:
    """Lucy-Richardson deconvolution implemented with NumPy or CuPy.

    Parameters
    ----------
    image : np.ndarray
        Input image to deconvolve.
    psf : np.ndarray
        Forward point spread function (forward projector).
    n_iter : int, default=10
        Number of iterations.
    noncirc : bool, default=False
        Enable non-circulant edge handling (matched path only).
    mask : np.ndarray | None, default=None
        Mask array (matched path only). Must have ``image``'s shape.
    observer_fn : Callable | None, default=None
        Function called after each iteration with ``(estimate, i)``.
    backprojector : np.ndarray | None, default=None
        Optional back projector PSF (e.g. Wiener-Butterworth). When provided,
        an unmatched Richardson-Lucy update is used so that an unmatched back
        projector can drive ~1-2 iteration convergence. The unmatched path is
        circulant only (it raises ``NotImplementedError`` when ``noncirc`` is
        set or a ``mask`` is given) and returns an array of ``image.shape``.

        Note: passing a matched (``traditional``-type) back projector does NOT
        reproduce the default matched path bit-for-bit, because the unmatched
        branch sum-normalizes the forward PSF (the matched path never
        normalizes ``psf``). This is expected.
    small_value : float | None, default=None
        Positive floor applied to the image and to every estimate on the
        unmatched path (see :func:`_richardson_lucy_unmatched`). ``None`` scales
        it with the data as ``1e-6 * image.max()``, evaluated after
        ``img_as_float`` conversion. Unmatched path only; passing it without a
        ``backprojector`` raises ``ValueError``.

    Raises
    ------
    ValueError
        If ``psf`` (or ``backprojector``) does not fit inside ``image``, if
        ``mask`` does not have ``image``'s shape, or if ``small_value`` is given
        without a ``backprojector``.

    """
    if backprojector is not None:
        if noncirc or mask is not None:
            raise NotImplementedError(
                "Unmatched/Wiener-Butterworth back projector currently supports "
                "circulant mode without a mask (noncirc=False, mask=None)."
            )
        return _richardson_lucy_unmatched(
            image, psf, backprojector, n_iter, observer_fn, small_value
        )
    if small_value is not None:
        raise ValueError(
            "small_value applies to the unmatched back-projector path only; "
            "pass a backprojector or leave small_value as None."
        )

    check_same_device(image, psf)

    image = util.img_as_float(image)
    psf = util.img_as_float(psf)

    if not noncirc:
        # noncirc extends the image by the PSF support, so any PSF size fits.
        _check_psf_fits(image, psf)
    if not noncirc and image.shape != psf.shape:
        psf = pad_image_to_shape(psf, image.shape, mode="constant")

    mask_values = None
    if mask is not None:
        if mask.shape != image.shape:
            raise ValueError(
                f"mask.shape={tuple(mask.shape)} must equal "
                f"image.shape={tuple(image.shape)}."
            )
        mask = util.img_as_float(mask)
        mask_values = image * (1 - mask)
        # Not in-place: ``img_as_float`` returns the caller's array unchanged for
        # float input, so ``image *= mask`` would zero the caller's data.
        image = image * mask

    if noncirc:
        orig_size = image.shape
        ext_size = [image.shape[i] + psf.shape[i] - 1 for i in range(image.ndim)]
        psf = pad_image_to_shape(psf, ext_size, mode="constant")

    psf = np.fft.fftn(np.fft.ifftshift(psf))
    otf_conj = np.conjugate(psf)

    if noncirc:
        image = pad_image_to_shape(image, ext_size, mode="constant")
        estimate = np.full_like(image, np.mean(image))
    else:
        estimate = image

    if mask is not None:
        htones = np.ones_like(image) * mask
    else:
        htones = np.ones_like(image)

    htones = np.real(np.fft.ifftn(np.fft.fftn(htones) * otf_conj))
    htones[htones < 1e-6] = 1

    for i in range(1, n_iter + 1):
        reblurred = np.real(np.fft.ifftn(np.fft.fftn(estimate) * psf))
        ratio = image / (reblurred + 1e-6)
        correction = np.real(np.fft.ifftn(np.fft.fftn(ratio) * otf_conj))

        correction[correction < 0] = 0
        estimate = estimate * correction / htones

        if observer_fn is not None:
            if noncirc:
                unpadded_estimate = crop_center(estimate, orig_size)
                observer_fn(unpadded_estimate, i)
            else:
                observer_fn(estimate, i)

    if noncirc:
        # Free the extended-size intermediates before crop_center allocates.
        del psf, otf_conj, htones
        estimate = crop_center(estimate, orig_size)

    if mask is not None:
        estimate = estimate * mask + mask_values

    return estimate


def _richardson_lucy_unmatched(
    image: np.ndarray,
    psf: np.ndarray,
    backprojector: np.ndarray,
    n_iter: int,
    observer_fn: Callable | None,
    small_value: float | None = None,
) -> np.ndarray:
    """Unmatched Richardson-Lucy update with a separate back projector.

    Faithful port of the reference ``DeconSingleView.m`` unmatched RL loop with
    ``ConvFFT3_S(x, OTF) = real(ifftn(fftn(x) * OTF))``: no ``H^T 1`` term, no
    ``correction < 0`` clip and no epsilon on the ratio denominator. The
    division is safe without an epsilon because the forward PSF sums to one, so
    the reblurred estimate is a convex combination of values >= ``small_value``
    and therefore strictly positive. Circulant only; returns ``image.shape``.

    ``small_value`` is that strictly positive floor. It defaults to
    ``1e-6 * image.max()`` because ``img_as_float`` rescales integer input into
    ``[0, 1]``: the reference's absolute ``1e-3`` would clamp a sizeable
    fraction of a uint16 image (``1e-3`` is 65.5 counts there).
    """
    check_same_device(image, psf, backprojector)

    image = util.img_as_float(image)
    psf = util.img_as_float(psf)
    bp = util.img_as_float(backprojector)

    if bp.shape != psf.shape:
        raise ValueError(
            f"backprojector shape {bp.shape} must match psf shape {psf.shape}"
        )
    _check_psf_fits(image, psf)

    # Normalize the forward PSF and back projector to sum=1 (only in this branch).
    psf = psf / psf.sum()
    bp = bp / bp.sum()

    if image.shape != psf.shape:
        psf = pad_image_to_shape(psf, image.shape, mode="constant")
    if image.shape != bp.shape:
        bp = pad_image_to_shape(bp, image.shape, mode="constant")

    otf_f = np.fft.fftn(np.fft.ifftshift(psf))
    otf_bp = np.fft.fftn(np.fft.ifftshift(bp))

    if small_value is None:
        small_value = 1e-6 * float(image.max())
    if not small_value > 0.0:
        raise ValueError(
            f"small_value must be > 0, got {small_value}; the epsilon-free ratio "
            "needs a strictly positive floor (an all-zero or all-negative image "
            "cannot be deconvolved on this path)."
        )
    image = np.maximum(image, small_value)
    estimate = image

    for i in range(1, n_iter + 1):
        reblurred = np.real(np.fft.ifftn(np.fft.fftn(estimate) * otf_f))
        ratio = image / reblurred
        correction = np.real(np.fft.ifftn(np.fft.fftn(ratio) * otf_bp))
        estimate = estimate * correction
        estimate = np.maximum(estimate, small_value)

        if observer_fn is not None:
            observer_fn(estimate, i)

    return estimate
