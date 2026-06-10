"""Tests for mesh-based feature extraction."""

import numpy as np
import pytest

pytest.importorskip("trimesh")

from cubic.feature import mesh  # noqa: E402


def test_extract_features_excludes_background_label(monkeypatch) -> None:
    """Background (label 0) produces no row, matching voxel.regionprops.

    Previously ``np.unique`` kept 0, so a meaningless background mesh and a
    spurious ``label == 0`` row were emitted alongside the real objects. The
    per-label surface computation is stubbed so the test targets the label
    filtering without depending on trimesh's optional ``rtree`` backend.
    """
    names = mesh.mesh_feature_list()
    seen_masks: list[int] = []

    def fake_surface_features(mask: np.ndarray) -> dict[str, float]:
        # Record which label each call corresponds to (mask sum is unique here).
        seen_masks.append(int(mask.sum()))
        return dict.fromkeys(names, 0.0)

    monkeypatch.setattr(mesh, "extract_surface_features", fake_surface_features)

    volume = np.zeros((12, 12, 12), dtype=np.int32)
    volume[2:6, 2:6, 2:6] = 1  # 64 voxels
    volume[7:11, 7:11, 7:11] = 2  # 64 voxels

    labels, feature_values = mesh.extract_features(volume)

    assert sorted(labels.tolist()) == [1, 2]
    assert 0 not in labels.tolist()
    assert feature_values.shape[0] == 2
    # Background (label 0) mask sum would be 12**3 - 128 = 1600; never visited.
    assert 1600 not in seen_masks
