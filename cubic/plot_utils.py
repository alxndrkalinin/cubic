"""Implements functions to display images and graphs."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from .cuda import asnumpy


def show_image(
    img: np.ndarray,
    figsize: tuple[int, int] = (10, 10),
    bit_depth: int = 14,
    cmap: str = "gray",
) -> Figure:
    """Display 2D image (NumPy or CuPy).

    Color limits are fixed to the integer range of ``bit_depth``
    (``vmin=0, vmax=2**bit_depth - 1``), which assumes raw camera counts. A
    float image in ``[0, 1]`` — what ``Image(as_float=True)`` and
    ``normalize_min_max`` produce — therefore renders as solid black with the
    default ``bit_depth=14``. For such images rescale to counts first, or call
    ``plt.imshow`` directly with your own ``vmin``/``vmax``.
    """
    fig = plt.figure(figsize=figsize)
    plt.imshow(asnumpy(img), cmap=cmap, vmin=0, vmax=2**bit_depth - 1)
    plt.axis("off")
    return fig


def show_2d(
    img: np.ndarray,
    axis: int = 0,
    figsize: tuple[int, int] = (10, 10),
    bit_depth: int = 14,
    cmap: str = "gray",
) -> Figure:
    """Max project and display 3D image (NumPy or CuPy).

    See :func:`show_image` for the ``bit_depth`` color-limit caveat.
    """
    return show_image(img.max(axis), figsize=figsize, bit_depth=bit_depth, cmap=cmap)


def show_image_error(
    img: np.ndarray,
    figsize: tuple[int, int] = (20, 20),
    bit_depth: int = 14,
    cmap: str = "bwr",
) -> Figure:
    """Display error map between two images (NumPy or CuPy).

    Color limits are symmetric around zero over the integer range of
    ``bit_depth``, so — as in :func:`show_image` — a float error map in
    ``[-1, 1]`` renders as a flat mid-tone with the default ``bit_depth=14``.
    """
    fig = plt.figure(figsize=figsize)
    plt.imshow(
        asnumpy(img), cmap=cmap, vmin=-(2**bit_depth - 1), vmax=(2**bit_depth - 1)
    )
    plt.axis("off")
    return fig
