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


def _is_torch_tensor(array: Any) -> bool:
    """Return True if ``array`` is a torch.Tensor.

    Uses a lazy import: returns False when torch is not installed so the
    cubic runtime never hard-depends on torch.
    """
    try:
        import torch
    except ImportError:
        return False
    return isinstance(array, torch.Tensor)


def get_device(array: np.ndarray) -> str:
    """Return current image device.

    Recognizes:
    - NumPy arrays → "CPU".
    - CuPy arrays (with ``device`` attribute) → "GPU".
    - torch.Tensor → "GPU" iff ``tensor.device.type == "cuda"``, else "CPU".
    """
    if _is_torch_tensor(array):
        return "GPU" if array.device.type == "cuda" else "CPU"  # type: ignore[attr-defined]
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
    """Move (or keep) array to CPU.

    Recognizes:
    - NumPy arrays → returned as-is.
    - CuPy arrays → moved via ``cupy.asnumpy``.
    - torch.Tensor → moved via ``.detach().cpu().numpy()`` (zero-copy
      when already on CPU).
    """
    if _is_torch_tensor(array):
        return array.detach().cpu().numpy()  # type: ignore[attr-defined]
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


def ascupy(array: Any) -> Any:
    """Move (or keep) array to GPU.

    For torch CUDA tensors and other CUDA Array Interface (CAI) producers,
    ``cupy.asarray`` returns a zero-copy view that shares storage with the
    source. Callers should treat the returned array as read-only: cubic
    metrics never mutate their inputs (verified by
    ``tests/test_cuda.py::test_metrics_do_not_mutate_input``); third-party
    code that wants to write into the GPU buffer should explicitly copy
    with ``cp.array(x, copy=True)`` first.

    Note:
        This is **float-dtype-agnostic at the bridge**: ``cupy`` has no
        ``bfloat16`` dtype, so ``cp.asarray`` on a torch ``bfloat16`` tensor
        silently produces an opaque ``|V2`` (2-byte void) array rather than
        raising. Route network outputs through :func:`ascupy_f32` instead.
    """
    cp = CUDAManager().get_cp()
    if cp is not None:
        return cp.asarray(array)
    raise RuntimeError("GPU requested but not available.")


def to_torch(array: Any, device: Any = None, dtype: Any = None) -> Any:
    """Move (or wrap) an array as a torch tensor, zero-copy from CuPy when possible.

    The inverse of :func:`ascupy` for the cubic↔torch bridge:

    - CuPy array → ``torch.as_tensor`` (zero-copy view via the CUDA Array
      Interface; shares storage with the source).
    - NumPy array → ``torch.from_numpy`` then moved to ``device``.
    - torch.Tensor → moved/cast to ``device``/``dtype`` as requested.

    torch is imported lazily so ``cubic`` still imports without torch installed
    (mirrors :func:`_is_torch_tensor`).

    Args:
        array: CuPy/NumPy array or torch tensor.
        device: Target torch device (e.g. ``"cuda"`` or ``torch.device(...)``).
            For CuPy inputs, defaults to ``"cuda"``; for tensors, left as-is
            unless provided.
        dtype: Optional torch dtype to cast to. The cast happens on-device, so
            a CuPy float32 input can be cast to ``bfloat16`` for the network
            without leaving the GPU.

    Returns
    -------
        torch.Tensor on the requested device/dtype.
    """
    import torch  # lazy: keep base cubic importable without torch

    if isinstance(array, torch.Tensor):
        if device is not None or dtype is not None:
            return array.to(device=device, dtype=dtype)
        return array

    cp = CUDAManager().get_cp()
    if cp is not None and isinstance(array, cp.ndarray):
        # zero-copy view via CUDA Array Interface; default to the array's own
        # CUDA device (e.g. cuda:1) rather than a hard-coded "cuda"
        dev = f"cuda:{array.device.id}" if device is None else device
        tensor = torch.as_tensor(array, device=dev)
        return tensor.to(dtype) if dtype is not None else tensor

    tensor = torch.from_numpy(np.asarray(array))
    if device is not None:
        tensor = tensor.to(device)
    return tensor.to(dtype) if dtype is not None else tensor


def ascupy_f32(array: Any) -> Any:
    """Move (or keep) an array on the GPU as a CuPy array, fp32-safe at the bridge.

    Identical to :func:`ascupy` except that torch tensors are first cast to
    ``float32`` **on-device**. This guards the silent-corruption failure mode
    where ``cupy.asarray`` on a torch ``bfloat16`` tensor yields an opaque
    ``|V2`` array (CuPy has no ``bfloat16``). Use at the network-output
    boundary, where activations may be ``bfloat16``.

    Non-torch inputs are forwarded to :func:`ascupy` unchanged.
    """
    if _is_torch_tensor(array):
        import torch  # lazy

        array = array.to(torch.float32)
    return ascupy(array)
