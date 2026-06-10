"""Wraps scikit-image submodules dynamically for device awareness."""

from types import ModuleType
from typing import Any
from importlib import import_module
from collections.abc import Callable

from .cuda import CUDAManager, asnumpy, is_gpu_array


class SkimageProxy(ModuleType):
    """Proxy module for dynamic wrapping of skimage functions for device awareness."""

    _loaded_modules: dict[str, "SkimageProxy"] = {}

    def __init__(self, name: str) -> None:
        """Initialize the proxy module."""
        super().__init__(name)
        self.cp = CUDAManager().get_cp()

    def __getattr__(self, func_name: str) -> Callable:
        """Dynamically wrap skimage or cucim.skimage functions based on device capability."""
        if func_name in self.__dict__:
            return self.__dict__[func_name]

        def func_wrapper(*args: Any, **kwargs: Any) -> Any:
            """Wrap skimage or cucim functions based on device capability."""
            # The io submodule routes to cubic.cucim, which manages device
            # placement itself. Everything else dispatches on whether *any*
            # argument is a GPU array — not just the first positional one,
            # since skimage functions accept the array under varying names
            # (e.g. ``label_image``, ``intensity_image``, ``coords``).
            if self.__name__ == "io":
                base_module = "cubic.cucim"
                use_gpu = False
            else:
                use_gpu = self.cp is not None and (
                    any(is_gpu_array(a) for a in args)
                    or any(is_gpu_array(v) for v in kwargs.values())
                )
                base_module = "cucim.skimage" if use_gpu else "skimage"

            full_func_name = f"{base_module}.{self.__name__}.{func_name}"
            module_name, method_name = full_func_name.rsplit(".", maxsplit=1)
            func = getattr(import_module(module_name), method_name)

            if use_gpu or self.__name__ == "io":
                return func(*args, **kwargs)
            # CPU route: defensively coerce any stray GPU array to CPU so a raw
            # CuPy array never reaches a host scikit-image function.
            cpu_args = [asnumpy(a) if is_gpu_array(a) else a for a in args]
            cpu_kwargs = {
                k: asnumpy(v) if is_gpu_array(v) else v for k, v in kwargs.items()
            }
            return func(*cpu_args, **cpu_kwargs)

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
