"""Wraps scikit-image submodules dynamically for device awareness."""

from types import ModuleType
from typing import Any
from importlib import import_module
from collections.abc import Callable

from .cuda import dispatch_device_call


class SkimageProxy(ModuleType):
    """Proxy module for dynamic wrapping of skimage functions for device awareness."""

    _loaded_modules: dict[str, "SkimageProxy"] = {}

    def __getattr__(self, func_name: str) -> Callable:
        """Dynamically wrap skimage or cucim.skimage functions based on device capability."""

        def func_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrap skimage or cucim functions based on device capability."""
            # The io submodule routes to cubic.cucim, which manages device
            # placement itself, so it bypasses device dispatch entirely.
            if self.__name__ == "io":
                func = getattr(import_module(f"cubic.cucim.{self.__name__}"), func_name)
                return func(*args, **kwargs)
            # Everything else dispatches on whether *any* argument is a GPU
            # array — not just the first positional one, since skimage functions
            # accept the array under varying names (e.g. ``label_image``,
            # ``intensity_image``, ``coords``).
            return dispatch_device_call(
                func_name,
                args,
                kwargs,
                gpu_module=f"cucim.skimage.{self.__name__}",
                cpu_module=f"skimage.{self.__name__}",
            )

        self.__dict__[func_name] = func_wrapper
        return func_wrapper

    @classmethod
    def load_module(cls, name: str) -> "SkimageProxy":
        """Load the module if not already loaded."""
        if name not in cls._loaded_modules:
            cls._loaded_modules[name] = SkimageProxy(name)
        return cls._loaded_modules[name]


def __getattr__(name: str) -> SkimageProxy:
    """Load skimage proxy module."""
    return SkimageProxy.load_module(name)
