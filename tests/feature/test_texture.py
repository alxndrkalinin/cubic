"""Tests for the GLCM texture feature module."""

import numpy as np
import pytest
from skimage.feature import graycoprops, graycomatrix

from cubic.cuda import ascupy
from cubic.feature import glcm_features
from cubic.feature.texture import _PROPS, _unit_offsets

_ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def _skimage_reference(image: np.ndarray, levels: int) -> dict[str, float]:
    """Direction-averaged graycoprops for an already-quantized integer image."""
    glcm = graycomatrix(
        image, distances=[1], angles=_ANGLES, levels=levels, symmetric=True, normed=True
    )
    ref = {
        prop: float(graycoprops(glcm, prop).mean())
        for prop in (
            "contrast",
            "dissimilarity",
            "homogeneity",
            "ASM",
            "energy",
            "correlation",
        )
    }
    # Hand-computed entropy (natural log), averaged over the four angles.
    entropies = []
    for a in range(glcm.shape[3]):
        plane = glcm[:, :, 0, a]
        nonzero = plane > 0
        entropies.append(float(-np.sum(plane[nonzero] * np.log(plane[nonzero]))))
    ref["entropy"] = float(np.mean(entropies))
    return ref


def test_glcm_2d_matches_graycoprops() -> None:
    """All seven properties match skimage's graycoprops on a 2D image."""
    rng = np.random.default_rng(0)
    levels = 8
    quant = rng.integers(0, levels, size=(20, 25)).astype(np.uint8)

    # value_range=(0, levels) makes quantization the identity for an integer
    # image already in [0, levels - 1], so both paths see the same array.
    feats = glcm_features(
        quant.astype(np.float64), levels=levels, value_range=(0.0, levels)
    )
    ref = _skimage_reference(quant, levels)

    assert set(feats) == set(ref)
    for prop in ref:
        assert abs(feats[prop] - ref[prop]) < 1e-9, prop


def test_unit_offsets_counts_and_half_space() -> None:
    """2D has 4 directions, 3D has 13, with no antipodal duplicates."""
    assert len(_unit_offsets(2)) == 4
    assert len(_unit_offsets(3)) == 13
    for ndim in (2, 3):
        offsets = set(_unit_offsets(ndim))
        for off in offsets:
            first_nonzero = next(o for o in off if o != 0)
            assert first_nonzero > 0
            assert tuple(-o for o in off) not in offsets


def test_glcm_3d_runs_and_is_finite() -> None:
    """3D images produce all properties as finite scalars."""
    rng = np.random.default_rng(2)
    vol = rng.random((12, 16, 18)).astype(np.float32)
    feats = glcm_features(vol, levels=16)
    assert set(feats) == set(_PROPS)
    for value in feats.values():
        assert np.isfinite(value)


@pytest.mark.parametrize("scale,shift", [(2.0, 5.0), (0.5, -3.0)])
def test_glcm_scale_invariance(scale: float, shift: float) -> None:
    """Affine intensity changes leave features unchanged with per-image range."""
    rng = np.random.default_rng(3)
    img = rng.random((30, 40)).astype(np.float64)
    base = glcm_features(img, levels=16)
    scaled = glcm_features(img * scale + shift, levels=16)
    for prop in base:
        assert abs(base[prop] - scaled[prop]) < 1e-9, prop


def test_glcm_mask_excludes_background_pairs() -> None:
    """Masked-out voxels do not contribute co-occurrence pairs."""
    rng = np.random.default_rng(4)
    levels = 8
    img = rng.random((24, 24)).astype(np.float64)
    mask = np.zeros(img.shape, dtype=bool)
    mask[4:20, 4:20] = True

    masked = glcm_features(img, mask=mask, levels=levels, value_range=(0.0, 1.0))

    # Cropping to the mask bounding box and computing without a mask must
    # match, since every in-mask pair lies inside that crop.
    crop = img[4:20, 4:20]
    cropped = glcm_features(crop, levels=levels, value_range=(0.0, 1.0))
    for prop in masked:
        assert abs(masked[prop] - cropped[prop]) < 1e-9, prop


def test_glcm_constant_region_correlation_is_one() -> None:
    """A constant image yields correlation 1.0 (matching skimage's std guard)."""
    img = np.full((16, 16), 7.0)
    feats = glcm_features(img, levels=8)
    assert feats["correlation"] == 1.0
    assert feats["contrast"] == 0.0


def test_glcm_rejects_bad_input() -> None:
    """Invalid ndim, mask shape and levels raise ValueError."""
    img = np.zeros((4, 4, 4, 4))
    with pytest.raises(ValueError, match="2D"):
        glcm_features(img)
    img2d = np.zeros((8, 8))
    with pytest.raises(ValueError, match="mask shape"):
        glcm_features(img2d, mask=np.zeros((4, 4), dtype=bool))
    with pytest.raises(ValueError, match="levels"):
        glcm_features(img2d, levels=1)


@pytest.mark.parametrize("use_gpu", [False, True])
def test_glcm_cpu_vs_gpu(use_gpu: bool, gpu_available: bool) -> None:
    """GLCM features agree between CPU and GPU within tolerance."""
    rng = np.random.default_rng(5)
    img = rng.random((40, 50)).astype(np.float32)
    cpu = glcm_features(img, levels=16)

    if use_gpu:
        if not gpu_available:
            pytest.skip("GPU not available")
        gpu = glcm_features(ascupy(img), levels=16)
        for prop in cpu:
            assert abs(cpu[prop] - gpu[prop]) < 1e-4, prop
    else:
        for value in cpu.values():
            assert np.isfinite(value)
