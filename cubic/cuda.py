"""Contains a class for accessing CUDA-accelerated libraries."""

from __future__ import annotations

import os
import warnings
from types import ModuleType
from typing import Any
from collections.abc import Callable

import numpy as np


class CUDAManager:
    """Manages CUDA resources."""

    _instance: CUDAManager | None = None
    cp: ModuleType | None
    cucim: ModuleType | None
    num_gpus: int

    def __new__(cls):
        """Ensure only one instance of CUDAManager is created."""
        if cls._instance is None:
            cls._instance = super(CUDAManager, cls).__new__(cls)
            cls._instance.init_gpu()
        return cls._instance

    def init_gpu(self) -> None:
        """Initialize GPU resources."""
        try:
            import cupy as cp
            import cucim

            self.cp = cp
            self.cucim = cucim
            self.num_gpus = cp.cuda.runtime.getDeviceCount()
        except ImportError:
            self.cp = self.cucim = None
            self.num_gpus = 0
            warnings.warn("CuPy or CuCIM is not installed. Falling back to CPU.")
        except Exception:
            self.cp = self.cucim = None
            self.num_gpus = 0
            warnings.warn(
                "Unable to detect CUDA-compatible GPU at the runtime. Check that driver is installed and GPU is visible. Falling back to CPU."
            )

    def get_cp(self) -> ModuleType | None:
        """Return CuPy if available."""
        return self.cp

    def get_num_gpus(self) -> int:
        """Return number of available GPUs."""
        return self.num_gpus


def get_device(array: np.ndarray) -> str:
    """Return current image device."""
    cp = CUDAManager().get_cp()
    if cp is not None and hasattr(array, "device") and array.device != "cpu":
        return "GPU"
    return "CPU"


def to_device(array: np.ndarray, device: str) -> np.ndarray:
    """Move array to the requested device."""
    cp = CUDAManager().get_cp()
    if device == "GPU":
        if cp is not None:
            return cp.asarray(array)
        else:
            raise RuntimeError("GPU requested but not available.")
    elif device == "CPU":
        return np.asarray(array)
    else:
        raise ValueError(
            f"Device should be 'CPU' or 'GPU', unknown requested: {device}."
        )


def to_same_device(source_array: np.ndarray, reference_array: np.ndarray) -> np.ndarray:
    """Move the source_array to the same device as reference_array."""
    target_device = get_device(reference_array)
    return to_device(source_array, target_device)


def check_same_device(*arrays: np.ndarray) -> None:
    """Check all provided arrays are on the same device."""
    devices = [get_device(a) for a in arrays]
    unique_devices = sorted(set(devices))
    if len(unique_devices) != 1:
        raise ValueError(
            f"All inputs must be on the same device, but found: {unique_devices}"
        )


def get_array_module(array: np.ndarray) -> ModuleType:
    """Get the NumPy or CuPy method based on argument location."""
    cp = CUDAManager().get_cp()
    if cp is not None:
        return cp.get_array_module(array)
    return np


def asnumpy(array: np.ndarray) -> np.ndarray:
    """Move (or keep) array to CPU."""
    cp = CUDAManager().get_cp()
    if isinstance(array, np.ndarray):
        return np.asarray(array)
    elif cp is not None and hasattr(array, "device"):
        device_val = getattr(array, "device", None)
        if hasattr(device_val, "id") or (
            isinstance(device_val, str) and device_val != "cpu"
        ):
            return cp.asnumpy(array)
    return np.asarray(array)


def ascupy(array: np.ndarray) -> object:
    """Move (or keep) array to GPU."""
    cp = CUDAManager().get_cp()
    if cp is not None:
        return cp.asarray(array)
    raise RuntimeError("GPU requested but not available.")
