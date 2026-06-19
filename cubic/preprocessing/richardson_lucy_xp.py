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

from cubic.cuda import (
    get_array_module,
    check_same_device,
)
from cubic.skimage import util
from cubic.image_utils import crop_center, pad_image_to_shape


def richardson_lucy_xp(
    image: np.ndarray,
    psf: np.ndarray,
    n_iter: int = 10,
    *,
    noncirc: bool = False,
    mask: np.ndarray | None = None,
    observer_fn: Callable | None = None,
    backprojector: np.ndarray | None = None,
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
        Mask array (matched path only).
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

    """
    if backprojector is not None:
        if noncirc or mask is not None:
            raise NotImplementedError(
                "Unmatched/Wiener-Butterworth back projector currently supports "
                "circulant mode without a mask (noncirc=False, mask=None)."
            )
        return _richardson_lucy_unmatched(
            image, psf, backprojector, n_iter, observer_fn
        )

    xp = get_array_module(image)
    check_same_device(image, psf)

    image = util.img_as_float(image)
    psf = util.img_as_float(psf)

    if not noncirc and image.shape != psf.shape:
        psf = pad_image_to_shape(psf, image.shape, mode="constant")

    mask_values = None
    if mask is not None:
        mask = util.img_as_float(mask)
        mask_values = image * (1 - mask)
        image *= mask

    if noncirc:
        orig_size = image.shape
        ext_size = [image.shape[i] + psf.shape[i] - 1 for i in range(image.ndim)]
        psf = pad_image_to_shape(psf, ext_size, mode="constant")

    psf = xp.fft.fftn(xp.fft.ifftshift(psf))
    otf_conj = xp.conjugate(psf)

    if noncirc:
        image = pad_image_to_shape(image, ext_size, mode="constant")
        estimate = xp.full_like(image, xp.mean(image))
    else:
        estimate = image

    if mask is not None:
        htones = xp.ones_like(image) * mask
    else:
        htones = xp.ones_like(image)

    htones = xp.real(xp.fft.ifftn(xp.fft.fftn(htones) * otf_conj))
    htones[htones < 1e-6] = 1

    for i in range(1, n_iter + 1):
        reblurred = xp.real(xp.fft.ifftn(xp.fft.fftn(estimate) * psf))
        ratio = image / (reblurred + 1e-6)
        correction = xp.real(xp.fft.ifftn(xp.fft.fftn(ratio) * otf_conj))

        correction[correction < 0] = 0
        estimate = estimate * correction / htones

        if observer_fn is not None:
            if noncirc:
                unpadded_estimate = crop_center(estimate, orig_size)
                observer_fn(unpadded_estimate, i)
            else:
                observer_fn(estimate, i)

    del psf, otf_conj, htones

    if noncirc:
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
) -> np.ndarray:
    """Unmatched Richardson-Lucy update with a separate back projector.

    Faithful port of the reference ``DeconSingleView.m`` unmatched RL loop with
    ``ConvFFT3_S(x, OTF) = real(ifftn(fftn(x) * OTF))``: no ``H^T 1`` term, no
    ``correction < 0`` clip and no epsilon on the ratio denominator. The
    division is safe without an epsilon because the forward PSF sums to one, so
    the reblurred estimate is a convex combination of values >= ``small_value``
    and therefore strictly positive. Circulant only; returns ``image.shape``.
    """
    xp = get_array_module(image)
    check_same_device(image, psf, backprojector)

    image = util.img_as_float(image)
    psf = util.img_as_float(psf)
    bp = util.img_as_float(backprojector)

    if bp.shape != psf.shape:
        raise ValueError(
            f"backprojector shape {bp.shape} must match psf shape {psf.shape}"
        )

    # Normalize the forward PSF and back projector to sum=1 (only in this branch).
    psf = psf / psf.sum()
    bp = bp / bp.sum()

    if image.shape != psf.shape:
        psf = pad_image_to_shape(psf, image.shape, mode="constant")
    if image.shape != bp.shape:
        bp = pad_image_to_shape(bp, image.shape, mode="constant")

    otf_f = xp.fft.fftn(xp.fft.ifftshift(psf))
    otf_bp = xp.fft.fftn(xp.fft.ifftshift(bp))

    small_value = 1e-3
    image = xp.maximum(image, small_value)
    estimate = image

    for i in range(1, n_iter + 1):
        reblurred = xp.real(xp.fft.ifftn(xp.fft.fftn(estimate) * otf_f))
        ratio = image / reblurred
        correction = xp.real(xp.fft.ifftn(xp.fft.fftn(ratio) * otf_bp))
        estimate = estimate * correction
        estimate = xp.maximum(estimate, small_value)

        if observer_fn is not None:
            observer_fn(estimate, i)

    return estimate
