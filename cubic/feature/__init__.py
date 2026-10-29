"""Expose feature extraction functions."""

from types import ModuleType
from typing import TYPE_CHECKING
from importlib import import_module

from .voxel import regionprops, regionprops_table
from .texture import glcm_features, glcm_features_by_label

if TYPE_CHECKING:
    from . import mesh as mesh

# ``mesh`` stays out of ``__all__`` so that a star import does not load trimesh.
__all__ = [
    "regionprops",
    "glcm_features",
    "glcm_features_by_label",
    "regionprops_table",
]

# Hidden from type checkers, which would otherwise type every unknown attribute
# of this package, typos included, as a module.
if not TYPE_CHECKING:

    def __getattr__(name: str) -> ModuleType:
        """Import the ``mesh`` submodule on first access.

        Importing trimesh takes 0.75-3 s, which users of the voxel and texture
        features should not pay on ``import cubic.feature``.
        """
        if name == "mesh":
            return import_module(f"{__name__}.mesh")
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
