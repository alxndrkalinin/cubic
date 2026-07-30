"""Tests for radial bin-edge construction."""

import numpy as np
import pytest

from cubic.metrics.spectral.radial import (
    radial_edges,
    reduce_power,
    frc_from_sums,
    radial_bin_id,
    reduce_frc_sums,
    sectioned_bin_id,
    _validate_angle_delta,
)
from cubic.metrics.spectral.iterators import FourierShellIterator


def test_radial_edges_use_max_nyquist_is_a_noop_without_spacing() -> None:
    """Without spacing every axis has the same Nyquist, so the flag does nothing.

    ``use_max_nyquist`` distinguishes the coarsest axis from the finest, which
    only differ when the axes have different *physical* spacing. With no spacing
    every axis is sampled once per pixel, so all of them top out at 0.5 cycles
    per pixel however long they are.

    This used to differ: the index-unit grid scaled each axis by its own length,
    giving a (16, 64, 64) volume a min Nyquist of 8 against a max of 32, so the
    flag mattered and a constant-radius ring was an ellipse in physical
    frequency. See ``radial._spacing_or_unit``.
    """
    shape = (16, 64, 64)

    edges_min, _ = radial_edges(shape, spacing=None, use_max_nyquist=False)
    edges_max, _ = radial_edges(shape, spacing=None, use_max_nyquist=True)

    assert edges_min[-1] == 0.5
    assert edges_max[-1] == 0.5
    # Identical to spelling the same grid as a spacing of one.
    edges_one, _ = radial_edges(shape, spacing=[1.0, 1.0, 1.0])
    np.testing.assert_allclose(edges_min, edges_one)


def test_radial_edges_use_max_nyquist_splits_on_anisotropic_spacing() -> None:
    """With genuinely anisotropic spacing the flag still separates the axes."""
    shape = (16, 64, 64)
    spacing = [0.5, 0.1, 0.1]  # Z Nyquist 1.0, XY Nyquist 5.0

    edges_min, _ = radial_edges(shape, spacing=spacing, use_max_nyquist=False)
    edges_max, _ = radial_edges(shape, spacing=spacing, use_max_nyquist=True)

    assert edges_min[-1] == pytest.approx(1.0)
    assert edges_max[-1] == pytest.approx(5.0)


def test_radial_edges_isotropic_unaffected() -> None:
    """For isotropic shapes the flag is a no-op (min == max Nyquist)."""
    shape = (64, 64)
    edges_min, _ = radial_edges(shape, spacing=None, use_max_nyquist=False)
    edges_max, _ = radial_edges(shape, spacing=None, use_max_nyquist=True)
    assert edges_min[-1] == edges_max[-1] == 0.5


def test_radial_bin_id_spacing_one_fills_all_bins() -> None:
    """A spacing of exactly 1.0 must bin in the same units as the edges.

    Regression guard: ``radial_bin_id`` treated an all-ones spacing as "no
    spacing" and switched to index units (K up to n/2) while ``radial_edges``
    stayed in physical units (kmax ~= 0.5), so ``np.digitize`` clipped every
    non-DC voxel into the last bin — 1 of 32 bins occupied.
    """
    shape = (64, 64)
    edges, radii = radial_edges(shape, spacing=[1.0, 1.0])
    bin_id = radial_bin_id(shape, edges, spacing=[1.0, 1.0])

    # Every bin but the innermost, which spans [0, edges[1]) and so holds only
    # the excluded DC term.
    occupied = np.unique(bin_id[bin_id >= 0])
    np.testing.assert_array_equal(occupied, np.arange(1, len(radii)))

    # ... and identical to the index-unit path, which is the same geometry.
    edges_idx, _ = radial_edges(shape, spacing=None)
    bin_id_idx = radial_bin_id(shape, edges_idx, spacing=None)
    np.testing.assert_array_equal(bin_id, bin_id_idx)


def test_sectioned_bin_id_spacing_one_fills_all_bins() -> None:
    """The 3D sectioned binning had the same spacing==1.0 shortcut."""
    shape = (32, 32, 32)
    edges, radii = radial_edges(shape, spacing=[1.0, 1.0, 1.0])
    angle_edges = np.array([0.0, 45.0, 90.0], dtype=np.float32)
    radial_id, angle_id = sectioned_bin_id(
        shape, edges, angle_edges, spacing=[1.0, 1.0, 1.0]
    )

    occupied = np.unique(radial_id[radial_id >= 0])
    np.testing.assert_array_equal(occupied, np.arange(1, len(radii)))
    assert np.unique(angle_id[angle_id >= 0]).size == 2


def test_radial_bin_id_innermost_ring_bins_upward() -> None:
    """The first non-DC ring sits exactly on a bin edge and belongs above it.

    ``k = 1 / (n * spacing)`` equals ``edges[1]``, but the frequency grid is
    float32 and the edges float64, so it could round into bin 0 — shifting the
    whole low-frequency end relative to the mask backend.
    """
    shape = (64, 64)
    spacing = [0.26, 0.26]
    edges, _ = radial_edges(shape, spacing=spacing)
    bin_id = radial_bin_id(shape, edges, spacing=spacing)

    # Bin 0 spans [0, edges[1]) and holds only DC, which is excluded.
    assert not np.any(bin_id == 0)


def test_radial_bin_id_exclude_overflow() -> None:
    """exclude_overflow drops the corner frequencies instead of folding them."""
    shape = (32, 32)
    edges, radii = radial_edges(shape, spacing=None)

    folded = radial_bin_id(shape, edges, spacing=None)
    dropped = radial_bin_id(shape, edges, spacing=None, exclude_overflow=True)

    last = len(radii) - 1
    assert np.sum(folded == last) > np.sum(dropped == last)
    # Only the last bin loses voxels; the rest are untouched.
    np.testing.assert_array_equal(folded[dropped >= 0], dropped[dropped >= 0])
    assert np.sum(dropped == -1) > np.sum(folded == -1)


def test_shell_iterator_exclude_overflow() -> None:
    """The mask backend exposes the same opt-out."""
    shape = (16, 16, 16)
    folded = FourierShellIterator(shape, 1)
    dropped = FourierShellIterator(shape, 1, exclude_overflow=True)

    counts_folded = [len(idx[0]) for idx, _ in folded]
    counts_dropped = [len(idx[0]) for idx, _ in dropped]

    assert counts_folded[:-1] == counts_dropped[:-1]
    assert counts_folded[-1] > counts_dropped[-1]


def test_reduce_power_returns_one_value_per_bin() -> None:
    """An empty trailing bin must not shorten the curve.

    ``nbins`` used to be derived from the data (``bin_id.max() + 1``), so a
    trailing empty bin produced a curve shorter than ``radii`` — which callers
    then paired positionally with the bin centers.
    """
    shape = (32, 32)
    edges, radii = radial_edges(shape, spacing=None)
    bin_id = radial_bin_id(shape, edges, spacing=None, exclude_overflow=True)
    # Empty the last bin so the derived count would come out short.
    bin_id[bin_id == len(radii) - 1] = -1

    F = np.fft.fftn(np.random.default_rng(0).normal(size=shape))
    S2, N = reduce_power(F, bin_id, nbins=len(radii))
    assert len(S2) == len(radii) == len(N)
    assert N[-1] == 0

    Sx2, Sy2, Sxy, N2 = reduce_frc_sums(F, F, bin_id, len(radii))
    assert len(Sx2) == len(Sy2) == len(Sxy) == len(N2) == len(radii)


def test_frc_from_sums_keeps_the_cross_spectrum_sign() -> None:
    """Anticorrelated bins must stay negative, and empty bins must read 0."""
    Sx2 = np.array([4.0, 4.0, 0.0])
    Sy2 = np.array([9.0, 9.0, 0.0])
    Sxy = np.array([6.0, -6.0, 0.0])

    signed = frc_from_sums(Sx2, Sy2, Sxy)
    np.testing.assert_allclose(signed, [1.0, -1.0, 0.0], rtol=1e-6)

    magnitude = frc_from_sums(Sx2, Sy2, Sxy, signed=False)
    np.testing.assert_allclose(magnitude, [1.0, 1.0, 0.0], rtol=1e-6)


def test_validate_angle_delta() -> None:
    """Only positive divisors of 90 are accepted."""
    assert _validate_angle_delta(45) == 2
    assert _validate_angle_delta(15) == 6
    assert _validate_angle_delta(90) == 1
    for bad in (20, 100, 0, -15):
        with pytest.raises(ValueError, match="angle_delta"):
            _validate_angle_delta(bad)
    with pytest.raises(ValueError, match="separate XY from Z"):
        _validate_angle_delta(90, min_sectors=2)
