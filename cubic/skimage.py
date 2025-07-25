"""Wraps scikit-image submodules dynamically for device awareness."""

from types import ModuleType
from typing import Any
from functools import wraps
from importlib import import_module
from collections.abc import Callable

from .cuda import CUDAManager


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
            base_module = "skimage"

            if self.__name__ == "io":
                base_module = "cubic.cucim"
                array = args[1] if len(args) > 1 else kwargs.get("arr", None)
            else:
                array = args[0] if args else kwargs.get("image", None)

                if self.cp is not None and hasattr(array, "device"):
                    device_val = getattr(array, "device", None)
                    if hasattr(device_val, "id") or (
                        isinstance(device_val, str) and device_val != "cpu"
                    ):
                        base_module = "cucim.skimage"

            full_func_name = f"{base_module}.{self.__name__}.{func_name}"
            module_name, method_name = full_func_name.rsplit(".", maxsplit=1)
            module = import_module(module_name)
            func = getattr(module, method_name)

            @wraps(func)
            def inner_func(*args: Any, **kwargs: Any) -> Any:
                """Inner function to call the wrapped function."""
                return func(*args, **kwargs)

            return inner_func(*args, **kwargs)

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
