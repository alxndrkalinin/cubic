"""Tests for ``image_utils`` helper functions."""

import numpy as np
import pytest

from cubic.cuda import ascupy, asnumpy
from cubic.image_utils import (
    rotate_image,
    pad_image_to_cube,
    checkerboard_split,
    reverse_checkerboard_split,
    select_max_contrast_slices,
)


def test_pad_image_to_cube() -> None:
    """Pad image to a cube and verify shape."""
    img = np.zeros((2, 4, 6), dtype=np.float32)
    padded = pad_image_to_cube(img)
    assert padded.shape == (6, 6, 6)


@pytest.mark.parametrize("use_gpu", [False, True])
def test_rotate_image_cpu_vs_gpu(use_gpu: bool, gpu_available: bool) -> None:
    """CPU vs GPU results for ``rotate_image``."""
    img = np.arange(9, dtype=np.float32).reshape(1, 3, 3)
    cpu_res = rotate_image(img, 90)

    if use_gpu:
        if not gpu_available:
            pytest.skip("GPU not available")
        gpu_res = rotate_image(ascupy(img), 90)
        assert np.allclose(asnumpy(gpu_res), cpu_res)
    else:
        gpu_res = rotate_image(img, 90)
        assert np.allclose(gpu_res, cpu_res)


def test_select_max_contrast_slices() -> None:
    """Ensure the function finds the highest-contrast slice block."""
    rng = np.random.default_rng(0)
    img = rng.random((5, 4, 4), dtype=np.float32)
    img[2:4] *= 2  # higher contrast region
    result, sl = select_max_contrast_slices(img, num_slices=2, return_indices=True)
    assert result.shape[0] == 2
    assert sl.stop - sl.start == 2


def test_select_max_contrast_edge_cases() -> None:
    """Test ``select_max_contrast_slices`` edge conditions."""
    rng = np.random.default_rng(1)
    img = rng.random((3, 2, 2), dtype=np.float32)

    # num_slices = 1
    res, sl = select_max_contrast_slices(img, num_slices=1, return_indices=True)
    assert res.shape[0] == 1
    assert sl.stop - sl.start == 1

    # num_slices equal to number of slices
    res, sl = select_max_contrast_slices(img, num_slices=3, return_indices=True)
    assert res.shape[0] == 3
    assert sl.stop - sl.start == 3

    # num_slices greater than number of slices should return full volume
    res, sl = select_max_contrast_slices(img, num_slices=5, return_indices=True)
    assert res.shape[0] == img.shape[0]
    assert sl.start == 0

    # uniform contrast image should return first slices
    uniform = np.ones((4, 2, 2), dtype=np.float32)
    res, sl = select_max_contrast_slices(uniform, num_slices=2, return_indices=True)
    assert sl.start == 0
    assert np.allclose(res, uniform[:2])


def test_checkerboard_split() -> None:
    """Test checkerboard_split and reverse_checkerboard_split functions."""
    # 2D regular checkerboard
    img_2d = np.arange(16, dtype=np.float32).reshape(4, 4)
    img1, img2 = checkerboard_split(img_2d)
    assert img1.shape == (2, 2)
    assert img2.shape == (2, 2)
    assert np.array_equal(img1, np.array([[5, 7], [13, 15]], dtype=np.float32))
    assert np.array_equal(img2, np.array([[0, 2], [8, 10]], dtype=np.float32))
    assert img1.dtype == img_2d.dtype

    # 2D reverse checkerboard
    img1_rev, img2_rev = reverse_checkerboard_split(img_2d)
    assert np.array_equal(img1_rev, np.array([[4, 6], [12, 14]], dtype=np.float32))
    assert np.array_equal(img2_rev, np.array([[1, 3], [9, 11]], dtype=np.float32))

    # 3D with Z-summing (Koho strategy)
    img_3d = np.arange(64, dtype=np.float32).reshape(4, 4, 4)
    img1_3d, img2_3d = checkerboard_split(img_3d, disable_3d_sum=False)
    assert img1_3d.shape == (2, 2, 2)
    assert img2_3d.shape == (2, 2, 2)
    # Verify Z-summing: z_summed = img[0::2] + img[1::2]
    z_summed = img_3d[0::2] + img_3d[1::2]
    expected_img1 = z_summed[:, 1::2, 1::2]
    expected_img2 = z_summed[:, 0::2, 0::2]
    assert np.allclose(img1_3d, expected_img1)
    assert np.allclose(img2_3d, expected_img2)

    # 3D full checkerboard (disable_3d_sum=True)
    img1_full, img2_full = checkerboard_split(img_3d, disable_3d_sum=True)
    assert img1_full.shape == (2, 2, 2)
    assert np.array_equal(img1_full, img_3d[1::2, 1::2, 1::2])
    assert np.array_equal(img2_full, img_3d[0::2, 0::2, 0::2])

    # Integer dtype conversion (preserve_range=False)
    img_int = np.random.randint(0, 255, (4, 4, 4), dtype=np.uint8)
    img1_int, img2_int = checkerboard_split(img_int, preserve_range=False)
    assert img1_int.dtype == np.float32
    assert img2_int.dtype == np.float32

    # preserve_range=True preserves dtype (should warn for integer types with Z-summing)
    with pytest.warns(UserWarning, match="preserve_range=True with integer dtype"):
        img1_preserve, img2_preserve = checkerboard_split(img_int, preserve_range=True)
    assert img1_preserve.dtype == np.uint8
    assert img2_preserve.dtype == np.uint8

    # Float types are preserved regardless of preserve_range
    img_float64 = np.random.rand(4, 4, 4).astype(np.float64)
    img1_f64, img2_f64 = checkerboard_split(img_float64, preserve_range=False)
    assert img1_f64.dtype == np.float64
    assert img2_f64.dtype == np.float64
    img1_f64_preserve, img2_f64_preserve = checkerboard_split(
        img_float64, preserve_range=True
    )
    assert img1_f64_preserve.dtype == np.float64
    assert img2_f64_preserve.dtype == np.float64

    # Reverse with 3D Z-summing
    img1_rev_3d, img2_rev_3d = reverse_checkerboard_split(img_3d, disable_3d_sum=False)
    z_summed_rev = img_3d[0::2] + img_3d[1::2]
    expected_img1_rev = z_summed_rev[:, 1::2, 0::2]
    expected_img2_rev = z_summed_rev[:, 0::2, 1::2]
    assert np.allclose(img1_rev_3d, expected_img1_rev)
    assert np.allclose(img2_rev_3d, expected_img2_rev)
