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
    pretrained_model: str = "cpsam",
    channel_axis: int | None = None,
    diameter: float | None = None,
    do_3D: bool = True,
    batch_size: int = 8,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
) -> np.ndarray:
    """Run a pretrained Cellpose-SAM (v4) model and return masks.

    Cellpose 4 removed the ``Cellpose`` class and the ``model_type``/``omni``/
    ``channels`` arguments; this uses ``CellposeModel`` with a
    ``pretrained_model`` (default the Cellpose-SAM weights ``"cpsam"``) and
    ``channel_axis`` for multi-channel inputs.
    """
    if not _CELLPOSE_AVAILABLE:
        raise ImportError(
            "Cellpose is required for this function, but not available. "
            "Try re-installing with `pip install cubic[cellpose]`."
        )

    model = models.CellposeModel(gpu=True, pretrained_model=pretrained_model)
    masks, _, _ = model.eval(
        image,
        channel_axis=channel_axis,
        diameter=diameter,
        do_3D=do_3D,
        batch_size=batch_size,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    return masks


def cellpose_segment(
    image,
    downscale_factor: float = 0.5,
    pretrained_model: str = "cpsam",
    channel_axis: int | None = None,
    diameter: float | None = None,
    do_3D: bool = True,
    batch_size: int = 8,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    border_value: int = 100,
    min_size: int = 500,
) -> np.ndarray:
    """Preprocess image, run Cellpose-SAM and postprocessing."""
    if not _CELLPOSE_AVAILABLE:
        raise ImportError(
            "Cellpose is required for this function, but not available. "
            "Try re-installing with `pip install cubic[cellpose]`."
        )

    image = downscale_and_filter(image, downscale_factor=downscale_factor)
    masks = cellpose_eval(
        image,
        pretrained_model=pretrained_model,
        channel_axis=channel_axis,
        diameter=diameter,
        do_3D=do_3D,
        batch_size=batch_size,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
    )
    masks = remove_touching_objects(masks, border_value=border_value)
    masks = clear_xy_borders(masks)
    masks = remove_small_objects(masks, min_size=min_size)
    return masks
