"""Tests for ``image_utils`` helper functions."""

import numpy as np
import pytest

from cubic.cuda import ascupy, asnumpy
from cubic.image_utils import (
    clahe,
    label,
    crop_bl,
    crop_br,
    crop_tl,
    crop_tr,
    img_mse,
    pad_image,
    crop_center,
    crop_corner,
    image_stats,
    random_crop,
    rotate_image,
    tukey_window,
    binomial_split,
    hamming_window,
    crop_to_divisor,
    pad_image_to_cube,
    checkerboard_split,
    pad_image_to_shape,
    get_xy_block_coords,
    pad_to_matching_shape,
    distance_transform_edt,
    reverse_checkerboard_split,
    select_max_contrast_slices,
)


def test_crop_corner_all_four_corners() -> None:
    """Each corner crop selects the right 2x2 block (br previously returned all)."""
    img = np.arange(1, 17).reshape(4, 4)
    np.testing.assert_array_equal(crop_tl(img, 2, axes=[0, 1]), [[1, 2], [5, 6]])
    np.testing.assert_array_equal(crop_tr(img, 2, axes=[0, 1]), [[3, 4], [7, 8]])
    np.testing.assert_array_equal(crop_bl(img, 2, axes=[0, 1]), [[9, 10], [13, 14]])
    np.testing.assert_array_equal(crop_br(img, 2, axes=[0, 1]), [[11, 12], [15, 16]])


def test_crop_corner_single_axis_bottom_right_crops_from_end() -> None:
    """With one axis, "b"/"r" crop the far edge instead of aliasing "tl".

    The far-edge test used to be gated on ``len(axes) >= 2``, so a single-axis
    ``crop_bl`` silently returned the same slice as ``crop_tl``.
    """
    img = np.arange(4)
    np.testing.assert_array_equal(crop_tl(img, 2, axes=[0]), [0, 1])
    np.testing.assert_array_equal(crop_bl(img, 2, axes=[0]), [2, 3])
    np.testing.assert_array_equal(crop_tr(img, 2, axes=[0]), [2, 3])
    np.testing.assert_array_equal(crop_br(img, 2, axes=[0]), [2, 3])


def test_crop_corner_rejects_descending_axes() -> None:
    """Corner selection keys off ``axes[-1]``/``axes[-2]``, so order matters."""
    img = np.zeros((4, 4, 4))
    with pytest.raises(ValueError, match="strictly ascending"):
        crop_corner(img, 2, axes=[2, 1], corner="br")
    with pytest.raises(ValueError, match="strictly ascending"):
        crop_corner(img, 2, axes=[1, 1], corner="br")


def test_crop_to_divisor_honors_axes_for_corner_crops() -> None:
    """Corner crop types must crop the caller's axes, not the default [1, 2].

    ``crop_size`` is computed for ``axes``, but the corner branches used to call
    ``crop_tl(img, crop_size)`` without forwarding ``axes``, so the sizes were
    applied to the wrong axes: (7, 7, 7) with ``axes=[0, 1]`` returned
    (7, 6, 6) instead of (6, 6, 7).
    """
    img = np.zeros((7, 7, 7))
    for crop_type in ("tl", "bl", "tr", "br", "center"):
        assert crop_to_divisor(img, 2, axes=[0, 1], crop_type=crop_type).shape == (
            6,
            6,
            7,
        ), crop_type


class TestPadImage:
    """Tests for the ``pad_image`` per-axis padding contract."""

    def test_int_pads_every_axis_symmetrically(self) -> None:
        """An int pads before and after on every axis in ``axes``."""
        padded = pad_image(np.zeros((4, 4, 4)), 3, axes=[0, 2], mode="constant")
        assert padded.shape == (10, 4, 10)

    def test_sequence_aligns_positionally_with_axes(self) -> None:
        """A sequence is indexed by position in ``axes``, not by axis number.

        ``pad_size[ax]`` used to index by axis number, so ``(4, 3)`` on
        ``axes=[0, 1]`` padded axis 0 by 4 and axis 1 by 4 → (15, 12) instead of
        (15, 18), and ``[1, 2]`` on ``axes=[1, 2]`` raised IndexError.
        """
        assert pad_image(
            np.zeros((7, 12)), (4, 3), axes=[0, 1], mode="constant"
        ).shape == (15, 18)
        assert pad_image(
            np.zeros((4, 4, 4)), [1, 2], axes=[1, 2], mode="constant"
        ).shape == (4, 6, 8)

    def test_before_after_pair_is_asymmetric(self) -> None:
        """A ``(before, after)`` entry pads each side independently."""
        padded = pad_image(np.ones((4, 4)), [(1, 2)], axes=[0], mode="constant")
        assert padded.shape == (7, 4)
        assert padded[0, 0] == 0 and padded[-1, 0] == 0 and padded[-2, 0] == 0
        np.testing.assert_array_equal(padded[1:5], np.ones((4, 4)))

    def test_mixed_int_and_pair_entries(self) -> None:
        """Int and pair entries can be mixed within one sequence."""
        padded = pad_image(
            np.zeros((4, 4, 4)), [2, (0, 3)], axes=[0, 2], mode="constant"
        )
        assert padded.shape == (8, 4, 7)

    def test_length_mismatch_raises(self) -> None:
        """A sequence ``pad_size`` must have one entry per axis."""
        with pytest.raises(ValueError, match="must align positionally"):
            pad_image(np.zeros((4, 4)), [1, 2, 3], axes=[0, 1], mode="constant")
        with pytest.raises(ValueError, match="must align positionally"):
            pad_image(np.zeros((4, 4)), [1, 2], axes=0, mode="constant")

    def test_numpy_integers_accepted(self) -> None:
        """NumPy integer scalars count as ints, not sequences."""
        assert pad_image(
            np.zeros((4, 4)), np.int64(2), axes=[0], mode="constant"
        ).shape == (8, 4)
        assert pad_image(
            np.zeros((4, 4)), [np.int64(2)], axes=[0], mode="constant"
        ).shape == (8, 4)


def test_pad_image_to_shape_rejects_smaller_target() -> None:
    """A target smaller than the input is a ValueError, not a bare assert."""
    with pytest.raises(ValueError, match="padding cannot shrink"):
        pad_image_to_shape(np.zeros((8, 8)), (4, 4))
    with pytest.raises(ValueError, match="entries but the image is"):
        pad_image_to_shape(np.zeros((8, 8, 8)), (8, 8))


def test_pad_image_to_cube_rejects_smaller_target() -> None:
    """``cube_size`` below an existing axis is a ValueError, not a bare assert."""
    with pytest.raises(ValueError, match="padding cannot shrink"):
        pad_image_to_cube(np.zeros((8, 8)), cube_size=4)


def test_pad_to_matching_shape() -> None:
    """Both images are padded up to the elementwise-max shape."""
    img1, img2 = pad_to_matching_shape(np.ones((4, 7)), np.ones((6, 3)))
    assert img1.shape == img2.shape == (6, 7)


def test_image_stats_percentile_key_spelling() -> None:
    """The upper-percentile key is ``percentile_max`` (was ``precentile_max``)."""
    stats = image_stats(np.arange(100.0))
    assert set(stats) == {"min", "max", "mean", "percentile_min", "percentile_max"}


def test_img_mse_shape_mismatch_raises() -> None:
    """Full shapes are compared, not just ``len()`` of the first axis."""
    assert img_mse(np.zeros((4, 4)), np.ones((4, 4))) == 1.0
    with pytest.raises(ValueError, match="same shape"):
        img_mse(np.zeros((4, 4)), np.zeros((4, 5)))


def test_random_crop_and_block_coords_accept_2d() -> None:
    """2D input uses the trailing two axes (``shape[1:]`` used to unpack-fail)."""
    assert random_crop(np.zeros((10, 10)), 4).shape == (4, 4)
    assert random_crop(np.zeros((3, 10, 10)), 4).shape == (3, 4, 4)
    assert get_xy_block_coords((8, 8), 4).shape == (4, 4)
    assert get_xy_block_coords((2, 8, 8), 4).shape == (4, 4)


def test_tukey_window_single_element_axis() -> None:
    """A length-1 axis returns 1.0 instead of dividing by ``n - 1`` (NaN).

    ``tukey_window(np.ones((1, 8, 8)))`` used to raise because the length-1 axis
    produced a 2-element window that could not be reshaped.
    """
    out = tukey_window(np.ones((1, 8, 8), dtype=np.float32))
    assert out.shape == (1, 8, 8)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(tukey_window(np.ones((1, 1), dtype=np.float32)), [[1.0]])


def test_tukey_window_matches_scipy() -> None:
    """The 1D window still matches ``scipy.signal.windows.tukey``."""
    from scipy.signal.windows import tukey

    from cubic.image_utils import _tukey_window_1d

    for n in (1, 2, 8, 17):
        for alpha in (0.0, 0.1, 0.5, 1.0):
            np.testing.assert_allclose(
                np.asarray(_tukey_window_1d(n, alpha, np)),
                tukey(n, alpha),
                atol=1e-6,
                err_msg=f"n={n} alpha={alpha}",
            )


@pytest.mark.parametrize("use_gpu", [False, True])
def test_hamming_window_cpu_vs_gpu(use_gpu: bool, gpu_available: bool) -> None:
    """``_nd_window`` still scales each axis by ``window ** (1 / ndim)``."""
    img = np.ones((4, 6), dtype=np.float32)
    expected = np.outer(np.hamming(4) ** 0.5, np.hamming(6) ** 0.5)

    if use_gpu:
        if not gpu_available:
            pytest.skip("GPU not available")
        res = asnumpy(hamming_window(ascupy(img)))
    else:
        res = hamming_window(img)
    assert res.dtype == np.float32
    np.testing.assert_allclose(res, expected, atol=1e-6)


def test_select_max_contrast_slices_validation() -> None:
    """Bad input raises ``ValueError`` and ``num_slices`` is clamped."""
    with pytest.raises(ValueError, match="more than 2 dimensions"):
        select_max_contrast_slices(np.zeros((4, 4)))
    with pytest.raises(ValueError, match="must be >= 1"):
        select_max_contrast_slices(np.zeros((4, 2, 2)), num_slices=0)
    # more slices requested than available returns the whole volume
    img = np.random.default_rng(0).random((3, 2, 2))
    res, sl = select_max_contrast_slices(img, num_slices=99, return_indices=True)
    assert res.shape == img.shape
    assert (sl.start, sl.stop) == (0, 3)


def test_checkerboard_split_rejects_unsupported_ndim() -> None:
    """Only 2D and 3D are supported (1D raised IndexError, 4D silently misbehaved)."""
    for ndim in (1, 4):
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            checkerboard_split(np.zeros((4,) * ndim))
        with pytest.raises(ValueError, match="Expected 2D or 3D"):
            reverse_checkerboard_split(np.zeros((4,) * ndim))


def test_checkerboard_split_odd_shapes_truncate_to_even() -> None:
    """Odd axes are truncated so both halves match, for 2D and both 3D modes."""
    img = np.arange(5 * 7 * 9, dtype=np.float32).reshape(5, 7, 9)
    for kwargs in ({"disable_3d_sum": False}, {"disable_3d_sum": True}):
        a, b = checkerboard_split(img, **kwargs)
        assert a.shape == b.shape == (2, 3, 4)
    a2d, b2d = checkerboard_split(img[0])
    assert a2d.shape == b2d.shape == (3, 4)


class TestClahe:
    """Tests for ``clahe``'s ``n_tiles`` contract."""

    def test_too_small_volume_raises_in_n_tiles_vocabulary(self) -> None:
        """A volume finer than ``n_tiles`` raises naming ``n_tiles``, not ``kernel_size``.

        Previously ``img.shape // n_tiles`` floored to 0 and skimage raised
        ``ValueError: Incorrect value of kernel_size: [0 0 0]`` — a parameter
        cubic does not expose. The message must name ``n_tiles``, the input
        shape and the computed per-axis tile size.
        """
        img = np.random.default_rng(0).random((1, 2, 4))
        with pytest.raises(ValueError) as excinfo:
            clahe(img)  # default n_tiles=(2, 3, 5)
        message = str(excinfo.value)
        assert "n_tiles" in message
        assert "kernel_size" not in message
        assert "(2, 3, 5)" in message  # the offending tile counts
        assert "(1, 2, 4)" in message  # the input shape
        assert "(0, 0, 0)" in message  # the computed tile size
        assert "[0, 1, 2]" in message  # the offending axes

    def test_partially_too_fine_volume_raises(self) -> None:
        """Only some axes need to be too fine for the request to be rejected."""
        img = np.random.default_rng(1).random((1, 2, 5))
        with pytest.raises(ValueError, match=r"below 1 on axes \[0, 1\]"):
            clahe(img)  # 5 // 5 == 1 is fine; axes 0 and 1 floor to 0

    def test_coarse_tiling_still_works(self) -> None:
        """A tiling the volume can support is unaffected by the new check."""
        img = np.random.default_rng(2).random((8, 32, 32))
        assert clahe(img).shape == img.shape

    def test_tile_count_semantics(self) -> None:
        """``n_tiles`` is a tile count per axis, so it accepts the full shape."""
        img = np.random.default_rng(1).random((4, 8, 9))
        assert clahe(img, n_tiles=(4, 8, 9)).shape == img.shape

    def test_rank_mismatch_raises(self) -> None:
        """``n_tiles`` must have one entry per image axis."""
        with pytest.raises(ValueError, match="entries but the image is"):
            clahe(np.random.default_rng(2).random((2, 2)), n_tiles=(1, 1, 1))

    def test_non_positive_tiles_raise(self) -> None:
        """A zero tile count would divide by zero."""
        with pytest.raises(ValueError, match="must be >= 1"):
            clahe(np.random.default_rng(3).random((2, 2)), n_tiles=(0, 1))


class TestDistanceTransformEdt:
    """Tests for ``distance_transform_edt`` dispatch and argument forwarding."""

    def test_host_route_rejects_gpu_only_args(self) -> None:
        """``block_params``/``float64_distances`` are GPU-only."""
        img = np.array([[0, 1], [1, 1]], dtype=np.uint8)
        with pytest.raises(ValueError, match="only be used with CuPy"):
            distance_transform_edt(img, float64_distances=True)
        with pytest.raises(ValueError, match="only be used with CuPy"):
            distance_transform_edt(img, block_params=(1, 1, 1))

    def test_array_like_dispatches_to_host(self) -> None:
        """A list is ``ArrayLike`` but not ``np.ndarray``, so it took the cuCIM branch."""
        res = distance_transform_edt([[0, 1], [1, 1]])
        np.testing.assert_allclose(res, [[0.0, 1.0], [1.0, np.sqrt(2)]])

    def test_gpu_forwards_block_params(self, gpu_available: bool) -> None:
        """``block_params`` reaches cuCIM instead of being replaced by ``None``.

        A valid value matches the default result; an out-of-range value must
        raise from cuCIM — with the old hard-coded ``None`` it never could.
        """
        if not gpu_available:
            pytest.skip("GPU not available")
        img = ascupy(np.random.default_rng(0).integers(0, 2, (64, 64)).astype(np.uint8))
        reference = distance_transform_edt(img)
        np.testing.assert_allclose(
            asnumpy(distance_transform_edt(img, block_params=(1, 1, 2))),
            asnumpy(reference),
        )
        with pytest.raises(ValueError, match="m3 too large"):
            distance_transform_edt(img, block_params=(1, 1, 32))

    def test_gpu_forwards_float64_distances(self, gpu_available: bool) -> None:
        """``float64_distances`` reaches cuCIM's dtype check for ``distances``.

        cuCIM itself never threads the flag down to its kernels, so the returned
        dtype stays float32 (documented in the docstring); what this asserts is
        that the caller's value is no longer silently replaced by ``False``.
        """
        if not gpu_available:
            pytest.skip("GPU not available")
        from cubic.cuda import CUDAManager

        cp = CUDAManager().get_cp()
        img = ascupy(np.random.default_rng(1).integers(0, 2, (32, 32)).astype(np.uint8))
        out = cp.zeros(img.shape, dtype=cp.float64)
        with pytest.raises(RuntimeError, match="must have dtype"):
            distance_transform_edt(img, distances=out, float64_distances=True)


def test_label_is_device_agnostic(gpu_available: bool) -> None:
    """``label`` routes through the skimage proxy on both devices."""
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[1:3, 1:3] = 1
    mask[5:7, 5:7] = 1
    assert label(mask).max() == 2
    if gpu_available:
        assert int(label(ascupy(mask)).max()) == 2


def test_pad_image_to_shape_odd_difference_round_trips() -> None:
    """Odd size differences reach the exact target and invert via crop_center."""
    img = np.arange(5 * 7 * 7, dtype=np.float64).reshape(5, 7, 7)
    target = (10, 8, 8)  # odd diff on axis 0 (5), odd on axes 1, 2 (1)
    padded = pad_image_to_shape(img, target, mode="constant")
    assert padded.shape == target
    np.testing.assert_array_equal(crop_center(padded, img.shape), img)


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
