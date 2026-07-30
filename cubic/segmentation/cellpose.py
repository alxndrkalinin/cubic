"""Implement 3D image segmentation using Cellpose."""

from typing import Any

import numpy as np

try:
    from cellpose import models

    _CELLPOSE_AVAILABLE = True
except ImportError:
    _CELLPOSE_AVAILABLE = False

from .segment_utils import (
    clear_xy_borders,
    downscale_and_filter,
    remove_small_objects,
    remove_touching_objects,
)

# Loading Cellpose weights takes seconds, so keep one model per
# (pretrained_model, gpu) pair instead of rebuilding it on every call.
_MODEL_CACHE: dict[tuple[str, bool], Any] = {}


def _get_model(pretrained_model: str, gpu: bool) -> Any:
    """Return a cached ``CellposeModel`` for ``pretrained_model`` on ``gpu``."""
    key = (pretrained_model, gpu)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = models.CellposeModel(
            gpu=gpu, pretrained_model=pretrained_model
        )
    return _MODEL_CACHE[key]


def cellpose_eval(
    image: np.ndarray,
    pretrained_model: str = "cpsam",
    channel_axis: int | None = None,
    diameter: float | None = None,
    do_3D: bool = True,
    batch_size: int = 8,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    *,
    model: Any = None,
    gpu: bool = True,
) -> np.ndarray:
    """Run a pretrained Cellpose v4 model and return masks.

    Cellpose 4 removed the ``Cellpose`` class and the ``model_type``/``omni``/
    ``channels`` arguments; this uses ``CellposeModel`` with a
    ``pretrained_model`` and ``channel_axis`` for multi-channel inputs. Any v4
    model name is accepted -- the SAM backbone (``"cpsam"`` (default),
    ``"cpsam_v2"``) or the DINOv3 backbones (``"cpdino"``, ``"cpdino-vitb"``) --
    or a path to a custom model.

    Parameters
    ----------
    model : cellpose.models.CellposeModel, optional
        A pre-built model to run (as :func:`~cubic.segmentation.segment_cellpose`
        accepts). When ``None`` (default), a model is built for
        ``pretrained_model`` and cached, so repeated calls do not reload the
        weights.
    gpu : bool, optional
        Whether the cached model runs on the GPU, by default True. Ignored when
        ``model`` is given.

    """
    if model is None:
        if not _CELLPOSE_AVAILABLE:
            raise ImportError(
                "Cellpose is required for this function, but not available. "
                "Try re-installing with `pip install cubic[cellpose]`."
            )
        model = _get_model(pretrained_model, gpu)

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
    min_size: int = 500,
    *,
    model: Any = None,
    gpu: bool = True,
) -> np.ndarray:
    """Preprocess image, run a Cellpose v4 model (SAM or DINO) and postprocess.

    .. warning::
        The returned masks are on the ``downscale_factor`` grid, **not** the
        input grid: the image is downscaled before Cellpose runs and the masks
        are never upscaled back. With the default ``downscale_factor=0.5`` the
        returned label image has half the XY size of ``image``. Upscale with
        ``cubic.image_utils.rescale_xy(masks, 1 / downscale_factor, order=0,
        preserve_range=True)`` if you need the original grid, or pass
        ``downscale_factor=1.0`` to skip downscaling.

    Parameters
    ----------
    model : cellpose.models.CellposeModel, optional
        A pre-built model to run; see :func:`cellpose_eval`.
    gpu : bool, optional
        Whether the cached model runs on the GPU, by default True. Ignored when
        ``model`` is given.

    """
    if model is None and not _CELLPOSE_AVAILABLE:
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
        model=model,
        gpu=gpu,
    )
    masks = remove_touching_objects(masks)
    masks = clear_xy_borders(masks)
    masks = remove_small_objects(masks, min_size=min_size)
    return masks
