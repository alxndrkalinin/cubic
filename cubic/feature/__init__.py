"""Expose feature extraction functions."""

from types import ModuleType
from importlib import import_module

from .voxel import regionprops, regionprops_table
from .texture import glcm_features, glcm_features_by_label

__all__ = [
    "mesh",
    "regionprops",
    "glcm_features",
    "glcm_features_by_label",
    "regionprops_table",
]


def __getattr__(name: str) -> ModuleType:
    """Import the ``mesh`` submodule on first access.

    Importing trimesh takes 0.75-3 s, which users of the voxel and texture
    features should not pay on ``import cubic.feature``.
    """
    if name == "mesh":
        return import_module(f"{__name__}.mesh")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
