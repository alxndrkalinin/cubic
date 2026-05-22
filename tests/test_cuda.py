"""Tests for CUDA helper utilities."""

import numpy as np
import pytest

from cubic.cuda import (
    ascupy,
    asnumpy,
    to_device,
    get_device,
    check_same_device,
)


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
