"""Tests for segmentation post-processing utilities."""

import numpy as np
import pytest
from skimage import morphology as skmorph
from skimage.segmentation import watershed

from cubic.cuda import ascupy, asnumpy, get_device
from cubic.image_utils import distance_transform_edt
from cubic.segmentation.segment_utils import (
    find_objects,
    clear_xy_borders,
    fill_label_holes,
    fill_holes_slicer,
    segment_watershed,
    remove_thin_objects,
    cleanup_segmentation,
    remove_large_objects,
    remove_touching_objects,
)


def _to_device(array: np.ndarray, use_gpu: bool, gpu_available: bool) -> np.ndarray:
    """Return ``array`` on the GPU, skipping the test when there is none."""
    if not use_gpu:
        return array
    if not gpu_available:
        pytest.skip("GPU not available")
    return ascupy(array)


def _reference_remove_touching(label_image: np.ndarray) -> np.ndarray:
    """Pre-rewrite implementation: dilate every object over the whole image.

    ``dilation`` stands in for the deprecated ``binary_dilation`` the old code
    used; the two agree for the symmetric ``(3,) * ndim`` footprint.
    """
    exclude: set[int] = set()
    footprint = skmorph.footprint_rectangle((3,) * label_image.ndim)
    for mask_idx in np.unique(label_image)[1:]:
        if int(mask_idx) in exclude:
            continue
        binary_mask = label_image == mask_idx
        outline = skmorph.dilation(binary_mask, footprint) & ~binary_mask
        neighbor_ids = np.unique(label_image[outline])
        neighbor_ids = neighbor_ids[neighbor_ids != 0]
        if neighbor_ids.size > 0:
            exclude.add(int(mask_idx))
            exclude.update(int(n) for n in neighbor_ids)
    out = label_image.copy()
    for label_id in exclude:
        out[out == label_id] = 0
    return out


def _random_labels(shape=(24, 64, 64), n=120, seed=0) -> np.ndarray:
    """Scatter small blocks, some of which end up touching."""
    rng = np.random.default_rng(seed)
    lbl = np.zeros(shape, dtype=np.int32)
    placed = 0
    for _ in range(n * 40):
        if placed >= n:
            break
        start = [int(rng.integers(0, s - e)) for s, e in zip(shape, (3, 4, 4))]
        sl = tuple(slice(a, a + e) for a, e in zip(start, (3, 4, 4)))
        if lbl[sl].any():
            continue
        placed += 1
        lbl[sl] = placed
    return lbl


# --------------------------------------------------------------------------- #
# remove_touching_objects
# --------------------------------------------------------------------------- #
def test_remove_touching_objects_keeps_isolated_high_labels() -> None:
    """Labels above the legacy border_value are not misclassified as touching.

    Two adjacent objects (150, 151) must be removed; an isolated object (200)
    must be kept. The previous additive-offset implementation treated every
    label id > border_value (=100) as touching and deleted everything.
    """
    volume = np.zeros((3, 9, 9), dtype=np.int32)
    volume[1, 1:3, 1:3] = 150  # touches 151
    volume[1, 3:5, 1:3] = 151  # touches 150
    volume[1, 7:9, 7:9] = 200  # isolated

    result = remove_touching_objects(volume.copy())
    remaining = set(np.unique(result).tolist())

    assert 200 in remaining
    assert 150 not in remaining
    assert 151 not in remaining


def test_remove_touching_objects_keeps_all_isolated() -> None:
    """All non-touching objects survive regardless of label magnitude."""
    volume = np.zeros((3, 9, 9), dtype=np.int32)
    volume[1, 1:3, 1:3] = 105
    volume[1, 6:8, 6:8] = 250

    result = remove_touching_objects(volume.copy())
    remaining = set(np.unique(result).tolist())

    assert remaining == {0, 105, 250}


@pytest.mark.parametrize("use_gpu", [False, True])
def test_remove_touching_objects_device_agnostic(
    use_gpu: bool, gpu_available: bool
) -> None:
    """Runs on GPU label images and matches the CPU result.

    The old implementation built the ``cube(3)`` footprint on the host (no array
    argument, so the skimage proxy never routed to cuCIM) and cuCIM rejected the
    NumPy footprint with ``ValueError: footprint must be either an ndarray or
    Sequence`` for *every* GPU label image.
    """
    volume = np.zeros((3, 9, 9), dtype=np.int32)
    volume[1, 1:3, 1:3] = 1  # touches 2
    volume[1, 3:5, 1:3] = 2
    volume[1, 7:9, 7:9] = 3  # isolated

    result = remove_touching_objects(_to_device(volume, use_gpu, gpu_available))
    if use_gpu:
        assert get_device(result) == "GPU"
    assert set(np.unique(asnumpy(result)).tolist()) == {0, 3}


@pytest.mark.parametrize("use_gpu", [False, True])
def test_remove_touching_objects_2d(use_gpu: bool, gpu_available: bool) -> None:
    """2D label images work; the hard-coded 3D ``cube(3)`` footprint used to fail.

    ``cleanup``/``cellpose_segment`` call this unconditionally, so a 2D input
    raised ``RuntimeError: structure and input must have same dimensionality``.
    """
    image = np.zeros((32, 32), dtype=np.int32)
    image[2:6, 2:6] = 1  # touches 2
    image[6:10, 2:6] = 2
    image[20:24, 20:24] = 3  # isolated

    result = remove_touching_objects(_to_device(image, use_gpu, gpu_available))
    assert set(np.unique(asnumpy(result)).tolist()) == {0, 3}


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_remove_touching_objects_matches_per_label_dilation(seed: int) -> None:
    """The vectorized rewrite reproduces the per-label dilation exactly."""
    labels = _random_labels(seed=seed)
    assert np.array_equal(
        remove_touching_objects(labels), _reference_remove_touching(labels)
    )


def test_remove_touching_objects_diagonal_contact_counts() -> None:
    """Corner-only contact counts as touching (26-connectivity, as before)."""
    volume = np.zeros((5, 20, 20), dtype=np.int32)
    volume[1, 2:4, 2:4] = 1
    volume[2, 4:6, 4:6] = 2  # shares only a corner with 1
    assert set(np.unique(remove_touching_objects(volume)).tolist()) == {0}


def test_remove_touching_objects_without_background() -> None:
    """An image fully covered by labels is handled (no background pixel).

    The shifted-comparison pass builds its ``touching`` flags by indexing on
    raw label values, so it must not assume a 0 is present. Note this does not
    guard the ``np.unique(...)[1:]`` slice: for *this* function a skipped
    lowest id is still flagged as some other label's neighbour, so that slice
    was never a bug here — see the ``_label_ids`` tests, which do guard it.
    """
    image = np.ones((8, 8), dtype=np.int32)
    image[4:, :] = 2
    assert set(np.unique(remove_touching_objects(image)).tolist()) == {0}


def test_remove_touching_objects_rejects_negative_labels() -> None:
    """Negative ids are not labels and must be rejected up front.

    ``touching`` is indexed by raw label value, so a negative id wrapped around
    into ``IndexError: index -4 is out of bounds`` (and an all-negative image
    hit ``np.zeros(negative)``).
    """
    image = np.zeros((4, 8, 8), dtype=np.int16)
    image[1:3, 1:3, 1:3] = 1
    image[1:3, 5:7, 5:7] = -4

    with pytest.raises(ValueError, match="must not contain negative values"):
        remove_touching_objects(image)


def test_remove_touching_objects_border_value_is_inert() -> None:
    """``border_value`` is still accepted for back-compat and changes nothing."""
    volume = np.zeros((3, 9, 9), dtype=np.int32)
    volume[1, 1:3, 1:3] = 150  # touches 151
    volume[1, 3:5, 1:3] = 151
    volume[1, 7:9, 7:9] = 200  # isolated

    assert np.array_equal(
        remove_touching_objects(volume, border_value=100),
        remove_touching_objects(volume, border_value=1000),
    )
    assert set(np.unique(remove_touching_objects(volume, border_value=1)).tolist()) == {
        0,
        200,
    }


# --------------------------------------------------------------------------- #
# input is never mutated
# --------------------------------------------------------------------------- #
def test_filters_do_not_mutate_input() -> None:
    """The label filters return a copy instead of corrupting the caller's array."""
    volume = np.zeros((5, 9, 9), dtype=np.int32)
    volume[1:3, 1:3, 1:3] = 1  # touches 2
    volume[1:3, 3:5, 1:3] = 2
    original = volume.copy()

    remove_touching_objects(volume)
    assert np.array_equal(volume, original)

    remove_large_objects(volume, max_size=1)
    assert np.array_equal(volume, original)

    remove_thin_objects(volume, min_z=4)
    assert np.array_equal(volume, original)

    fill_holes_slicer(volume, area_threshold=2)
    assert np.array_equal(volume, original)


# --------------------------------------------------------------------------- #
# cleanup_segmentation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("max_hole_size,filled", [(4, False), (5, True)])
def test_cleanup_segmentation_hole_threshold_cpu_gpu_parity(
    max_hole_size: int, filled: bool, gpu_available: bool
) -> None:
    """CPU and GPU agree on a hole of area exactly ``max_hole_size``.

    skimage 0.26 ``max_size=N`` fills area ≤ N while cucim ``area_threshold=N``
    fills area < N, so without the ``- 1`` shift the two devices returned
    different label images for a 4-pixel hole at ``max_hole_size=4``.
    """
    label_img = np.ones((10, 10), dtype=np.int32)
    label_img[4:6, 4:6] = 0  # 4-pixel hole

    cpu = cleanup_segmentation(label_img.copy(), max_hole_size=max_hole_size)
    assert bool((cpu[4:6, 4:6] != 0).all()) is filled

    if not gpu_available:
        pytest.skip("GPU not available")
    gpu = cleanup_segmentation(ascupy(label_img.copy()), max_hole_size=max_hole_size)
    assert np.array_equal(asnumpy(gpu), cpu)


def test_cleanup_segmentation_widens_dtype_past_uint16() -> None:
    """More than 65535 objects must not wrap around to uint16."""
    n = 70000
    ids = np.arange(1, n + 1, dtype=np.int64).reshape(-1, 1)
    label_img = np.concatenate([ids, np.zeros_like(ids)], axis=1)

    result = cleanup_segmentation(label_img)

    assert int(result.max()) == n
    assert len(np.unique(result)) == n + 1
    assert result.dtype == np.uint32


def test_cleanup_segmentation_keeps_uint16_for_small_counts() -> None:
    """The historical uint16 output dtype is kept when the labels fit."""
    label_img = np.zeros((4, 8, 8), dtype=np.int32)
    label_img[1, 1:3, 1:3] = 1
    label_img[1, 5:7, 5:7] = 2
    assert cleanup_segmentation(label_img).dtype == np.uint16


def test_cleanup_segmentation_fills_holes_of_lowest_label() -> None:
    """The lowest label is not skipped when the image has no background pixel.

    ``np.unique(label_img)[1:]`` treated label 1 as background here, so its
    4-pixel hole (a patch of label 2) was never filled.
    """
    label_img = np.ones((8, 8), dtype=np.int32)
    label_img[3:5, 3:5] = 2

    result = cleanup_segmentation(label_img, max_hole_size=5)

    assert len(np.unique(result)) == 1  # one object, label 2 absorbed


def test_cleanup_segmentation_does_not_mutate_input() -> None:
    """Hole filling works on a copy of the caller's label image."""
    label_img = np.ones((8, 8), dtype=np.int32)
    label_img[3:5, 3:5] = 2
    original = label_img.copy()
    cleanup_segmentation(label_img, max_hole_size=5)
    assert np.array_equal(label_img, original)


# --------------------------------------------------------------------------- #
# clear_xy_borders
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("use_gpu", [False, True])
def test_clear_xy_borders_preserves_label_ids_3d(
    use_gpu: bool, gpu_available: bool
) -> None:
    """3D border clearing keeps label ids, like the 2D branch already did.

    The 3D branch used to end with ``label()``, renumbering every id (7 → 1) and
    contradicting ``cleanup_segmentation``'s label-preserving contract.
    """
    volume = np.zeros((3, 9, 9), dtype=np.int32)
    volume[1, 4:6, 4:6] = 7  # interior
    volume[1, 0:2, 0:2] = 9  # touches the XY border

    result = clear_xy_borders(_to_device(volume, use_gpu, gpu_available))
    assert set(np.unique(asnumpy(result)).tolist()) == {0, 7}
    assert asnumpy(result).shape == volume.shape


def test_clear_xy_borders_2d_and_3d_agree_on_ids() -> None:
    """A single interior object keeps its id in both the 2D and 3D branch."""
    plane = np.zeros((9, 9), dtype=np.int32)
    plane[4:6, 4:6] = 7
    volume = np.zeros((3, 9, 9), dtype=np.int32)
    volume[1] = plane
    assert np.array_equal(np.unique(clear_xy_borders(plane)), np.array([0, 7]))
    assert np.array_equal(np.unique(clear_xy_borders(volume)), np.array([0, 7]))


# --------------------------------------------------------------------------- #
# remove_thin_objects
# --------------------------------------------------------------------------- #
def test_remove_thin_objects_keeps_object_spanning_exactly_min_z() -> None:
    """``min_z`` is the minimum kept thickness, matching the docstring."""
    volume = np.zeros((6, 8, 8), dtype=np.int32)
    volume[1:3, 1:3, 1:3] = 1  # spans exactly 2 planes
    volume[1:2, 5:7, 5:7] = 2  # spans 1 plane

    result = remove_thin_objects(volume, min_z=2)
    assert set(np.unique(result).tolist()) == {0, 1}


def test_remove_thin_objects_rejects_2d() -> None:
    """A 2D input gets a clear error instead of a numpy axis error."""
    with pytest.raises(ValueError, match="3D"):
        remove_thin_objects(np.zeros((8, 8), dtype=np.int32))


@pytest.mark.parametrize("use_gpu", [False, True])
def test_remove_thin_objects_device_agnostic(
    use_gpu: bool, gpu_available: bool
) -> None:
    """Thin-object removal gives the same result on CPU and GPU."""
    volume = np.zeros((6, 8, 8), dtype=np.int32)
    volume[1:5, 1:3, 1:3] = 1
    volume[1:2, 5:7, 5:7] = 2

    result = remove_thin_objects(_to_device(volume, use_gpu, gpu_available), min_z=3)
    assert set(np.unique(asnumpy(result)).tolist()) == {0, 1}


def test_remove_thin_objects_handles_gaps_in_label_ids() -> None:
    """Non-contiguous label ids are handled (bbox lookup is indexed by id)."""
    volume = np.zeros((6, 8, 8), dtype=np.int32)
    volume[1:5, 1:3, 1:3] = 4  # thick
    volume[1:2, 5:7, 5:7] = 9  # thin
    assert set(np.unique(remove_thin_objects(volume, min_z=3)).tolist()) == {0, 4}


# --------------------------------------------------------------------------- #
# find_objects
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("use_gpu", [False, True])
def test_find_objects_returns_bounding_boxes(
    use_gpu: bool, gpu_available: bool
) -> None:
    """Bounding boxes are indexed by ``label - 1``, with None for absent labels."""
    volume = np.zeros((6, 8, 8), dtype=np.int32)
    volume[1:3, 2:5, 3:4] = 1
    volume[4:5, 0:2, 6:8] = 3  # label 2 is absent

    boxes = find_objects(_to_device(volume, use_gpu, gpu_available))

    assert len(boxes) == 3
    assert boxes[0] == (slice(1, 3), slice(2, 5), slice(3, 4))
    assert boxes[1] is None
    assert boxes[2] == (slice(4, 5), slice(0, 2), slice(6, 8))


def test_find_objects_honours_max_label() -> None:
    """``max_label`` truncates the returned list, as in scipy.ndimage."""
    volume = np.zeros((4, 8, 8), dtype=np.int32)
    volume[1, 1:3, 1:3] = 1
    volume[2, 5:7, 5:7] = 2
    assert len(find_objects(volume, max_label=1)) == 1
    assert len(find_objects(volume)) == 2


# --------------------------------------------------------------------------- #
# fill_label_holes / fill_holes_slicer
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("use_gpu", [False, True])
def test_fill_label_holes_fills_interior(use_gpu: bool, gpu_available: bool) -> None:
    """Interior holes are filled with the surrounding label id."""
    label_img = np.zeros((8, 8), dtype=np.int32)
    label_img[1:6, 1:6] = 1
    label_img[3:4, 3:4] = 0  # interior hole

    result = asnumpy(fill_label_holes(_to_device(label_img, use_gpu, gpu_available)))
    assert result[3, 3] == 1


def test_fill_label_holes_forwards_kwargs() -> None:
    """Extra kwargs reach ``binary_fill_holes`` instead of raising TypeError.

    ``_binary_fill_holes`` took exactly one argument, so *any* keyword passed to
    ``fill_label_holes`` raised ``TypeError: unexpected keyword argument``.
    """
    label_img = np.zeros((8, 8), dtype=np.int32)
    label_img[1:6, 1:6] = 1
    label_img[3:4, 3:4] = 0

    structure = np.ones((3, 3), dtype=bool)
    result = fill_label_holes(label_img, structure=structure)
    assert result[3, 3] == 1
    assert np.array_equal(result, fill_label_holes(label_img, output=None))


@pytest.mark.parametrize("use_gpu", [False, True])
def test_fill_holes_slicer_device_agnostic(use_gpu: bool, gpu_available: bool) -> None:
    """Runs on GPU arrays; ``np.asarray`` used to reject CuPy input outright."""
    volume = np.zeros((4, 10, 10), dtype=np.int32)
    volume[1:3, 2:8, 2:8] = 1
    volume[1:3, 4:6, 4:6] = 0  # interior hole, 4 px per slice

    result = asnumpy(
        fill_holes_slicer(_to_device(volume, use_gpu, gpu_available), area_threshold=5)
    )
    assert result[1, 4, 4] == 1
    assert result[1, 5, 5] == 1


def test_fill_holes_slicer_fills_holes_of_lowest_label() -> None:
    """The lowest label is not skipped when there is no background pixel."""
    volume = np.ones((2, 8, 8), dtype=np.int32)
    volume[:, 3:5, 3:5] = 2  # a 4-px "hole" in label 1 on every slice

    result = fill_holes_slicer(volume, area_threshold=5, axes=(0,))

    assert set(np.unique(result).tolist()) == {1}


# --------------------------------------------------------------------------- #
# segment_watershed
# --------------------------------------------------------------------------- #
def _two_spheres() -> np.ndarray:
    """Two overlapping spheres forming one connected binary blob."""
    z, y, x = np.mgrid[0:32, 0:48, 0:32]
    left = (z - 16) ** 2 + (y - 16) ** 2 + (x - 16) ** 2 <= 9**2
    right = (z - 16) ** 2 + (y - 32) ** 2 + (x - 16) ** 2 <= 9**2
    return left | right


@pytest.mark.parametrize("use_gpu", [False, True])
@pytest.mark.parametrize("dilate_seeds", [False, True])
def test_segment_watershed_distance_splits_touching_spheres(
    use_gpu: bool, dilate_seeds: bool, gpu_available: bool
) -> None:
    """Distance-based watershed splits the blob into its two spheres."""
    blob = _two_spheres().astype(np.uint8)

    result = segment_watershed(
        _to_device(blob, use_gpu, gpu_available), ball_size=8, dilate_seeds=dilate_seeds
    )
    if use_gpu:
        assert get_device(result) == "GPU"
    labels = asnumpy(result)
    assert int(labels.max()) == 2
    # labels cover exactly the blob
    assert np.array_equal(labels > 0, blob > 0)


def test_segment_watershed_marker_based_landscape_is_offset_invariant() -> None:
    """The negated EDT needs no min-subtraction: a constant offset is inert.

    ``segment_watershed`` used to shift the landscape by ``-ws_image.min()``
    before flooding, which cannot change the result; this pins that down.
    """
    blob = _two_spheres()
    markers = np.zeros(blob.shape, dtype=np.int32)
    markers[16, 16, 16] = 1
    markers[16, 32, 16] = 2

    result = segment_watershed(blob, markers=markers, mask=blob)

    distance = distance_transform_edt(blob)
    shifted = watershed(-distance - (-distance).min(), markers=markers, mask=blob)
    assert np.array_equal(result, shifted)
    assert int(result.max()) == 2
