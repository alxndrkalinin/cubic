"""Implements functions to display images and graphs."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


def show_image(
    img: np.ndarray,
    figsize: tuple[int, int] = (10, 10),
    bit_depth: int = 14,
    cmap: str = "gray",
) -> Figure:
    """Display 2D image."""
    fig = plt.figure(figsize=figsize)
    plt.imshow(img, cmap=cmap, vmin=0, vmax=2**bit_depth - 1)
    plt.axis("off")
    return fig


def show_2d(
    img: np.ndarray,
    axis: int = 0,
    figsize: tuple[int, int] = (10, 10),
    bit_depth: int = 14,
    cmap: str = "gray",
) -> Figure:
    """Max project and display 3D image."""
    return show_image(img.max(axis), figsize=figsize, bit_depth=bit_depth, cmap=cmap)


def show_image_error(
    img: np.ndarray,
    figsize: tuple[int, int] = (20, 20),
    bit_depth: int = 14,
    cmap: str = "bwr",
) -> Figure:
    """Display error map between teo images."""
    fig = plt.figure(figsize=figsize)
    plt.imshow(img, cmap=cmap, vmin=-(2**bit_depth - 1), vmax=(2**bit_depth - 1))
    plt.axis("off")
    return fig
