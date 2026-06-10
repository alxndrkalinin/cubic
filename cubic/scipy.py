"""Wraps SciPy submodules dynamically for device awareness."""

import warnings
from types import ModuleType
from typing import Any
from importlib import import_module
from collections.abc import Callable

from .cuda import CUDAManager, asnumpy, to_device, is_gpu_array


def _any_gpu_arg(args: tuple, kwargs: dict) -> bool:
    """Return True if any positional or keyword argument is a GPU array.

    Device routing must scan *every* argument, not just ``args[0]`` /
    ``kwargs["input"]``: SciPy functions take the array under varying names
    (e.g. ``coordinates``, ``weights``), so a GPU array elsewhere would
    otherwise route to CPU SciPy and crash on the raw CuPy input.
    """
    return any(is_gpu_array(a) for a in args) or any(
        is_gpu_array(v) for v in kwargs.values()
    )


class SciPyProxy(ModuleType):
    """Proxy module for dynamic wrapping of SciPy functions."""

    _loaded_modules: dict[str, "SciPyProxy"] = {}

    def __init__(self, name: str) -> None:
        """Initialize the proxy module."""
        super().__init__(name)
        self.cp = CUDAManager().get_cp()

    def __getattr__(self, func_name: str) -> Callable:
        """Dynamically wrap scipy or cupyx.scipy functions based on device."""
        if func_name in self.__dict__:
            return self.__dict__[func_name]

        def func_wrapper(*args: Any, **kwargs: Any) -> Any:
            use_gpu = self.cp is not None and _any_gpu_arg(args, kwargs)
            base_module = "cupyx.scipy" if use_gpu else "scipy"
            module_name = f"{base_module}.{self.__name__}"

            def _on_cpu(func: Callable, return_to_gpu: bool) -> Any:
                # Coerce every GPU array argument to CPU before calling SciPy,
                # then optionally move array results back to the GPU.
                cpu_args = [asnumpy(a) if is_gpu_array(a) else a for a in args]
                cpu_kwargs = {
                    k: asnumpy(v) if is_gpu_array(v) else v for k, v in kwargs.items()
                }
                result = func(*cpu_args, **cpu_kwargs)
                if not return_to_gpu:
                    return result
                if isinstance(result, tuple):
                    return tuple(
                        to_device(r, "GPU") if hasattr(r, "dtype") else r
                        for r in result
                    )
                if hasattr(result, "dtype"):
                    return to_device(result, "GPU")
                return result

            try:
                module = import_module(module_name)
                func = getattr(module, func_name)
            except (ModuleNotFoundError, AttributeError):
                warnings.warn(
                    f"cupyx.scipy.{self.__name__}.{func_name} is unavailable, falling back to CPU."
                )
                func = getattr(import_module(f"scipy.{self.__name__}"), func_name)
                return _on_cpu(func, return_to_gpu=use_gpu)

            if not use_gpu:
                # CPU route: defensively coerce any stray GPU array to CPU so a
                # raw CuPy array never reaches a host SciPy function.
                return _on_cpu(func, return_to_gpu=False)
            return func(*args, **kwargs)

        self.__dict__[func_name] = func_wrapper
        return func_wrapper

    @classmethod
    def load_module(cls, name: str) -> "SciPyProxy":
        """Load the module if not already loaded."""
        if name not in cls._loaded_modules:
            cls._loaded_modules[name] = SciPyProxy(name)
        return cls._loaded_modules[name]


def __getattr__(name: str) -> SciPyProxy:
    """Load scipy proxy module."""
    return SciPyProxy.load_module(name)
