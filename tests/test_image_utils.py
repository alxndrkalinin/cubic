"""Tests for ``image_utils`` helper functions."""

import numpy as np
import pytest

from cubic.cuda import ascupy, asnumpy
from cubic.image_utils import (
    rotate_image,
    binomial_split,
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


def test_pad_image_to_cube_default_constant() -> None:
    """Verify default mode is 'constant' (zero-padding, not reflect)."""
    img = np.ones((2, 4, 6), dtype=np.float32)
    padded = pad_image_to_cube(img)
    # Padded regions along the first axis (original size 2, padded to 6)
    # should be zero (constant), not reflected ones
    assert padded[0, 0, 0] == 0.0
    assert padded[-1, 0, 0] == 0.0


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
    # 2D regular checkerboard - matches miplib implementation (Koho et al. 2019)
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


class TestBinomialSplit:
    """Tests for ``binomial_split``."""

    def test_conservation_counts_mode(self) -> None:
        """img1 + img2 == integer counts (no readout correction)."""
        rng = np.random.default_rng(42)
        img = rng.poisson(100, size=(64, 64)).astype(np.float32)
        img1, img2 = binomial_split(img, rng=0)
        np.testing.assert_array_equal(img1 + img2, np.rint(img).astype(np.float32))

    def test_expectation(self) -> None:
        """Mean of img1 ≈ p * mean(img)."""
        rng_img = np.random.default_rng(1)
        img = rng_img.poisson(500, size=(128, 128)).astype(np.float32)
        p = 0.5
        img1, _ = binomial_split(img, p=p, rng=2)
        assert abs(img1.mean() - p * img.mean()) / img.mean() < 0.02

    def test_independence_poisson_thinning(self) -> None:
        """Poisson thinning draws should be independent (uncorrelated residuals)."""
        img = np.full((128, 128), 200.0, dtype=np.float32)
        p = 0.5
        img1, img2 = binomial_split(img, p=p, counts_mode="poisson_thinning", rng=4)
        # In poisson_thinning mode, n1 and n2 are independent Poisson draws
        corr = np.corrcoef(img1.ravel(), img2.ravel())[0, 1]
        assert abs(corr) < 0.1

    def test_variance_law(self) -> None:
        """Var(img1) per pixel follows p*(1-p)*n scaling."""
        n_val = 1000
        p = 0.5
        img = np.full((256, 256), n_val, dtype=np.float32)
        img1, _ = binomial_split(img, p=p, rng=5)
        expected_var = p * (1 - p) * n_val
        actual_var = float(np.var(img1))
        assert abs(actual_var - expected_var) / expected_var < 0.1

    def test_shape_preservation(self) -> None:
        """Output shape matches input shape."""
        img = np.random.default_rng(6).poisson(50, size=(32, 64)).astype(np.float32)
        img1, img2 = binomial_split(img, rng=7)
        assert img1.shape == img.shape
        assert img2.shape == img.shape

    def test_deterministic_with_seed(self) -> None:
        """Same seed produces same split."""
        img = np.random.default_rng(8).poisson(100, size=(32, 32)).astype(np.float32)
        img1a, img2a = binomial_split(img, rng=42)
        img1b, img2b = binomial_split(img, rng=42)
        np.testing.assert_array_equal(img1a, img1b)
        np.testing.assert_array_equal(img2a, img2b)

    def test_readout_noise_nonneg(self) -> None:
        """With readout noise correction, outputs are non-negative."""
        rng_img = np.random.default_rng(9)
        img = rng_img.poisson(10, size=(64, 64)).astype(np.float32)
        img1, img2 = binomial_split(img, readout_noise_rms=3.0, rng=10)
        assert np.all(img1 >= 0)
        assert np.all(img2 >= 0)

    def test_poisson_thinning_nonneg(self) -> None:
        """Poisson thinning mode: outputs non-negative."""
        img = (
            np.random.default_rng(11).uniform(0, 100, size=(32, 32)).astype(np.float32)
        )
        img1, img2 = binomial_split(img, counts_mode="poisson_thinning", rng=12)
        assert np.all(img1 >= 0)
        assert np.all(img2 >= 0)

    def test_poisson_thinning_independence(self) -> None:
        """Poisson thinning draws are independent (no exact conservation)."""
        img = np.full((128, 128), 200.0, dtype=np.float32)
        img1, img2 = binomial_split(img, counts_mode="poisson_thinning", rng=13)
        # Not exactly conserved (unlike counts mode)
        diff = img1 + img2 - img
        assert np.std(diff) > 0  # there should be variation
        # But means should be close
        assert abs((img1 + img2).mean() - img.mean()) / img.mean() < 0.05

    def test_float_input_warning(self) -> None:
        """Float input with default gain/offset in counts mode triggers warning."""
        img = (
            np.random.default_rng(14)
            .uniform(0.5, 10.5, size=(32, 32))
            .astype(np.float32)
        )
        with pytest.warns(UserWarning, match="non-integer pixels"):
            binomial_split(img, rng=15)

    def test_invalid_p(self) -> None:
        """P outside (0, 1) raises ValueError."""
        img = np.ones((4, 4), dtype=np.float32)
        with pytest.raises(ValueError, match="p must be in"):
            binomial_split(img, p=0.0)
        with pytest.raises(ValueError, match="p must be in"):
            binomial_split(img, p=1.0)

    def test_3d_input(self) -> None:
        """3D input produces valid 3D output."""
        img = np.random.default_rng(16).poisson(50, size=(8, 32, 32)).astype(np.float32)
        img1, img2 = binomial_split(img, rng=17)
        assert img1.shape == img.shape
        assert img1.ndim == 3
        np.testing.assert_array_equal(img1 + img2, np.rint(img).astype(np.float32))
