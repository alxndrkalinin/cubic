"""Tests for CUDA helper utilities."""

import time
import threading

import numpy as np
import pytest

from cubic.cuda import (
    CUDAManager,
    ascupy,
    asnumpy,
    to_device,
    get_device,
    is_gpu_array,
    to_same_device,
    check_same_device,
)


def test_is_gpu_array_classification(gpu_available: bool) -> None:
    """``is_gpu_array`` is True only for GPU arrays, not NumPy or scalars."""
    assert is_gpu_array(np.ones((2, 2))) is False
    assert is_gpu_array(5) is False
    assert is_gpu_array("not an array") is False
    assert is_gpu_array([1, 2, 3]) is False

    if gpu_available:
        assert is_gpu_array(ascupy(np.ones((2, 2)))) is True


@pytest.mark.parametrize("device", ["CPU", "GPU"])
def test_to_device_roundtrip(device: str, gpu_available: bool) -> None:
    """Move array to the specified device and verify roundtrip."""
    if device == "GPU" and not gpu_available:
        pytest.skip("GPU not available")

    arr = np.ones((2, 2), dtype=np.float32)
    res = to_device(arr, device)
    if device == "GPU":
        assert np.allclose(asnumpy(res), arr)
    else:
        assert np.allclose(res, arr)


def test_to_device_cpu_accepts_gpu_array(gpu_available: bool) -> None:
    """``to_device(gpu_arr, "CPU")`` materializes on the host.

    It used to call ``np.asarray``, which raises ``TypeError: Implicit
    conversion to a NumPy array is not allowed`` for CuPy input — breaking
    ``Image.to_cpu()`` and ``to_same_device(gpu_arr, cpu_ref)``.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = to_device(ascupy(arr), "CPU")
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, arr)


def test_to_same_device_moves_gpu_source_to_cpu_reference(gpu_available: bool) -> None:
    """A GPU source with a host reference lands on the host."""
    if not gpu_available:
        pytest.skip("GPU not available")
    arr = np.arange(4, dtype=np.float32)
    out = to_same_device(ascupy(arr), arr)
    assert get_device(out) == "CPU"
    np.testing.assert_array_equal(out, arr)


def test_to_device_cpu_accepts_cuda_torch_tensor(gpu_available: bool) -> None:
    """``to_device(..., "CPU")`` also handles CUDA torch tensors via ``asnumpy``."""
    torch = pytest.importorskip("torch")
    if not gpu_available or not torch.cuda.is_available():
        pytest.skip("GPU not available")
    arr = np.arange(4, dtype=np.float32)
    out = to_device(torch.from_numpy(arr).cuda(), "CPU")
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, arr)


def test_to_device_invalid_device_raises() -> None:
    """Ensure to_device raises ``ValueError`` for an invalid device string."""
    arr = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        to_device(arr, "INVALID_DEVICE")


def test_check_same_device(gpu_available: bool) -> None:
    """Ensure mismatched devices trigger an error when GPU is present."""
    arr = np.ones((2, 2), dtype=np.float32)
    if gpu_available:
        gpu_arr = ascupy(arr)
        with pytest.raises(ValueError):
            check_same_device(arr, gpu_arr)
    else:
        check_same_device(arr, arr)


def test_asnumpy_accepts_cpu_torch_tensor() -> None:
    """``asnumpy`` round-trips a CPU torch.Tensor to a numpy.ndarray."""
    torch = pytest.importorskip("torch")
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = torch.from_numpy(arr)
    out = asnumpy(t)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, arr)


def test_asnumpy_accepts_cuda_torch_tensor(gpu_available: bool) -> None:
    """``asnumpy`` materializes a CUDA torch.Tensor on the host."""
    torch = pytest.importorskip("torch")
    if not gpu_available or not torch.cuda.is_available():
        pytest.skip("GPU not available")
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    t = torch.from_numpy(arr).cuda()
    out = asnumpy(t)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, arr)


def test_asnumpy_detaches_tensor_with_grad() -> None:
    """``asnumpy`` succeeds on tensors that require gradients (detach path)."""
    torch = pytest.importorskip("torch")
    t = torch.ones(3, requires_grad=True)
    out = asnumpy(t)
    assert isinstance(out, np.ndarray)
    np.testing.assert_array_equal(out, np.ones(3, dtype=np.float32))


def test_get_device_cpu_torch_tensor() -> None:
    """``get_device`` returns 'CPU' for a CPU torch.Tensor."""
    torch = pytest.importorskip("torch")
    assert get_device(torch.zeros(3)) == "CPU"


def test_get_device_cuda_torch_tensor(gpu_available: bool) -> None:
    """``get_device`` returns 'GPU' for a CUDA torch.Tensor."""
    torch = pytest.importorskip("torch")
    if not gpu_available or not torch.cuda.is_available():
        pytest.skip("GPU not available")
    assert get_device(torch.zeros(3).cuda()) == "GPU"


def test_cuda_manager_has_class_level_defaults() -> None:
    """``cp``/``cucim``/``num_gpus`` live on the class, not just on instances.

    They used to be annotation-only, so a thread that reached the singleton
    before ``init_gpu()`` finished raised ``AttributeError`` instead of falling
    back to CPU.
    """
    assert hasattr(CUDAManager, "cp")
    assert hasattr(CUDAManager, "cucim")
    assert CUDAManager.num_gpus >= 0


def test_cuda_manager_concurrent_construction_is_atomic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent ``CUDAManager()`` calls all get one fully initialized instance.

    ``__new__`` used to publish ``cls._instance`` *before* ``init_gpu()`` ran, so
    a second thread inside that window saw a manager with no ``cp``/``num_gpus``.
    """
    original_instance = CUDAManager._instance
    real_init = CUDAManager.init_gpu

    def slow_init(self: CUDAManager) -> None:
        time.sleep(0.05)  # widen the race window
        real_init(self)

    monkeypatch.setattr(CUDAManager, "init_gpu", slow_init)
    seen: list[tuple] = []
    errors: list[Exception] = []

    def build() -> None:
        try:
            manager = CUDAManager()
            seen.append((manager, manager.get_cp(), manager.get_num_gpus()))
        except Exception as exc:
            errors.append(exc)

    try:
        CUDAManager._instance = None
        threads = [threading.Thread(target=build) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        CUDAManager._instance = original_instance

    assert not errors, errors
    assert len(seen) == len(threads)
    assert all(entry == seen[0] for entry in seen)


def test_metrics_do_not_mutate_input(gpu_available: bool) -> None:
    """Calling pixel metrics through ``ascupy`` must not mutate the source.

    Documents the contract that cubic's metric pipeline treats inputs as
    read-only — important for callers that hand torch CUDA tensors to
    cubic via ``ascupy``'s zero-copy CAI view.
    """
    if not gpu_available:
        pytest.skip("GPU not available")
    from cubic.metrics import pcc, psnr, ssim, nrmse

    rng = np.random.default_rng(0)
    arr = rng.random((16, 16)).astype(np.float32)
    gpu = ascupy(arr)
    baseline = gpu.copy()

    pcc(gpu, gpu)
    ssim(gpu, gpu, data_range=1.0, win_size=3)
    psnr(gpu, gpu, data_range=1.0)
    nrmse(gpu, gpu, normalize="min_max")

    from cubic.cuda import CUDAManager

    cp = CUDAManager().get_cp()
    assert cp.array_equal(gpu, baseline), "ascupy view was mutated by a metric call"
