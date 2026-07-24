"""Wraps SciPy submodules dynamically for device awareness."""

from types import ModuleType
from typing import Any
from collections.abc import Callable

from .cuda import dispatch_device_call


class SciPyProxy(ModuleType):
    """Proxy module for dynamic wrapping of SciPy functions."""

    _loaded_modules: dict[str, "SciPyProxy"] = {}

    def __getattr__(self, func_name: str) -> Callable:
        """Dynamically wrap scipy or cupyx.scipy functions based on device."""

        def func_wrapper(*args: Any, **kwargs: Any) -> Any:
            return dispatch_device_call(
                func_name,
                args,
                kwargs,
                gpu_module=f"cupyx.scipy.{self.__name__}",
                cpu_module=f"scipy.{self.__name__}",
            )

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
