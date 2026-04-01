"""Implement 3D image segmentation using Cellpose."""

import warnings

import numpy as np

try:
    from cellpose import models

    _CELLPOSE_AVAILABLE = True
except ImportError:
    _CELLPOSE_AVAILABLE = False
    warnings.warn("Cellpose is not available. Cellpose segmentation will not work.")

from .segment_utils import (
    clear_xy_borders,
    downscale_and_filter,
    remove_small_objects,
    remove_touching_objects,
)


def cellpose_eval(
    image: np.ndarray,
    model_type: str = "cyto",
    omni: bool = False,
    channels: list[int] | None = None,
    diameter: int | None = None,
    do_3D: bool = True,
) -> np.ndarray:
    """Run pre-trained Cellpose model and return masks."""
    model = models.Cellpose(gpu=True, model_type=model_type, omni=omni)
    masks, _, _, _ = model.eval(
        image,
        channels=channels,
        diameter=diameter,
        do_3D=do_3D,
    )
    return masks


def cellpose_segment(
    image,
    downscale_factor: float = 0.5,
    model_type: str = "cyto",
    omni: bool = False,
    channels: list[int] | None = None,
    diameter: int | None = None,
    do_3D: bool = True,
    border_value: int = 100,
    min_size: int = 500,
) -> np.ndarray:
    """Preprocess image, run Cellpose and postprocessing."""
    if not _CELLPOSE_AVAILABLE:
        raise ImportError(
            "Cellpose is required for this function, but not available. "
            "Try re-installing with `pip install cubic[cellpose]`."
        )

    image = downscale_and_filter(image, downscale_factor=downscale_factor)
    masks = cellpose_eval(
        image,
        model_type=model_type,
        omni=omni,
        channels=channels,
        diameter=diameter,
        do_3D=do_3D,
    )
    masks = remove_touching_objects(masks, border_value=border_value)
    masks = clear_xy_borders(masks)
    masks = remove_small_objects(masks, min_size=min_size)
    return masks
