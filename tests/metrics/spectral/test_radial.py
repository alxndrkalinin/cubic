"""Tests for radial bin-edge construction."""

import numpy as np

from cubic.metrics.spectral.radial import radial_edges


def test_radial_edges_index_units_honor_use_max_nyquist() -> None:
    """use_max_nyquist extends index-unit edges to the max axis Nyquist.

    For an anisotropic shape with no spacing the radial bins must reach
    ``max(n // 2)`` when requested, so sectioned callers that normalize by the
    XY Nyquist span the full [0, 1] range instead of being compressed into the
    low-frequency quarter.
    """
    shape = (16, 64, 64)  # min(n//2)=8, max(n//2)=32

    edges_min, _ = radial_edges(shape, spacing=None, use_max_nyquist=False)
    edges_max, _ = radial_edges(shape, spacing=None, use_max_nyquist=True)

    assert edges_min[-1] == 8.0
    assert edges_max[-1] == 32.0


def test_radial_edges_isotropic_unaffected() -> None:
    """For isotropic shapes the flag is a no-op (min == max Nyquist)."""
    shape = (64, 64)
    edges_min, _ = radial_edges(shape, spacing=None, use_max_nyquist=False)
    edges_max, _ = radial_edges(shape, spacing=None, use_max_nyquist=True)
    assert edges_min[-1] == edges_max[-1] == 32.0
