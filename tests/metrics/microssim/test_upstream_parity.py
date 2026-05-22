"""Frozen-fixture parity vs juglab/microssim@8bccb17d.

Compares the cubic port against the frozen upstream golden outputs
**unconditionally**: the comparison runs regardless of installed
numpy/scipy/skimage versions. Pin metadata embedded in the npz is
surfaced in failure messages for diagnostic value only — it is *not*
used to skip the comparison. If the port drifts beyond 1e-3 on a
modern numpy/scipy, that is a real signal worth surfacing.

Regenerate the fixture by running ``_generate_golden.py`` with the
pinned upstream microssim installed (see that file's docstring).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cubic.metrics import MicroSSIM, MicroMS3IM

GOLDEN = Path(__file__).parent / "data" / "microssim_golden.npz"

PARITY_TOL = 1e-3
SCORE_ATOL = 1e-3


def _inputs_from_seed(
    seed: int, n: int, h: int, w: int
) -> tuple[np.ndarray, np.ndarray]:
    """Regenerate the exact GT/pred arrays the generator used.

    Kept in sync with ``_generate_golden._generate_inputs``.
    """
    rng = np.random.default_rng(int(seed))
    gt = rng.exponential(0.5, (int(n), int(h), int(w))).astype(np.float32) * 1000
    pred = gt * 1.2 + 0.1 * float(gt.max()) * rng.standard_normal(
        (int(n), int(h), int(w))
    ).astype(np.float32)
    return gt, pred


def _format_pins(g) -> str:
    return (
        f"\n  Fixture pins: microssim={g['_pin_microssim'].item()!r}, "
        f"numpy={g['_pin_numpy'].item()!r}, "
        f"scipy={g['_pin_scipy'].item()!r}, "
        f"skimage={g['_pin_skimage'].item()!r}"
    )


@pytest.fixture(scope="module")
def golden():
    """Load the frozen golden fixture; skip module if absent."""
    if not GOLDEN.exists():
        pytest.skip(
            f"Frozen fixture not present at {GOLDEN}; "
            "run tests/metrics/microssim/_generate_golden.py "
            "with the pinned upstream microssim installed."
        )
    return np.load(GOLDEN, allow_pickle=False)


@pytest.fixture(scope="module")
def inputs(golden):
    """Regenerate the GT/pred inputs from the stored seed."""
    return _inputs_from_seed(
        golden["_seed"].item(),
        golden["_N"].item(),
        golden["_H"].item(),
        golden["_W"].item(),
    )


@pytest.fixture(scope="module")
def ms_fitted(golden, inputs):
    """Cubic MicroSSIM fit on the regenerated inputs."""
    gt, pred = inputs
    return MicroSSIM().fit(gt, pred)


@pytest.fixture(scope="module")
def m3_fitted(golden, inputs):
    """Cubic MicroMS3IM fit on the regenerated inputs."""
    gt, pred = inputs
    return MicroMS3IM().fit(gt, pred)


def test_micro_ssim_fit_parameters(golden, ms_fitted):
    """Match upstream to 1e-3 on bg offsets, max_val, and ri_factor."""
    p = ms_fitted.get_parameters()
    pins = _format_pins(golden)
    assert abs(p["offset_gt"] - float(golden["ms_offset_gt"])) < PARITY_TOL, pins
    assert abs(p["offset_pred"] - float(golden["ms_offset_pred"])) < PARITY_TOL, pins
    assert abs(p["max_val"] - float(golden["ms_max_val"])) < PARITY_TOL, pins
    assert abs(p["ri_factor"] - float(golden["ms_ri_factor"])) < PARITY_TOL, pins


def test_micro_ssim_scores(golden, inputs, ms_fitted):
    """Per-slice MicroSSIM scores match upstream to 1e-3."""
    gt, pred = inputs
    cubic_scores = np.array(
        [float(ms_fitted.score(gt[i], pred[i])) for i in range(gt.shape[0])]
    )
    np.testing.assert_allclose(
        cubic_scores,
        np.asarray(golden["ms_scores"]),
        atol=SCORE_ATOL,
        err_msg=_format_pins(golden),
    )


def test_micro_ms3im_ri_factor(golden, m3_fitted):
    """MS3IM RI factor matches upstream to 1e-3."""
    pins = _format_pins(golden)
    diff = abs(m3_fitted.get_parameters()["ri_factor"] - float(golden["m3_ri_factor"]))
    assert diff < PARITY_TOL, f"diff={diff}{pins}"


def test_micro_ms3im_scores(golden, inputs, m3_fitted):
    """Per-slice MicroMS3IM scores match upstream to 1e-3."""
    gt, pred = inputs
    cubic_scores = np.array(
        [float(m3_fitted.score(gt[i], pred[i])) for i in range(gt.shape[0])]
    )
    np.testing.assert_allclose(
        cubic_scores,
        np.asarray(golden["m3_scores"]),
        atol=SCORE_ATOL,
        err_msg=_format_pins(golden),
    )


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Inline rank-Pearson Spearman correlation (no scipy)."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def test_micro_ssim_ranking_preserved(golden, inputs, ms_fitted):
    """Per-slice MicroSSIM ranking vs upstream stays > 0.99 Spearman."""
    gt, pred = inputs
    cubic_scores = np.array(
        [float(ms_fitted.score(gt[i], pred[i])) for i in range(gt.shape[0])]
    )
    rho = _spearman(cubic_scores, np.asarray(golden["ms_scores"]))
    assert rho > 0.99, f"Spearman={rho:.4f}{_format_pins(golden)}"


def test_micro_ms3im_ranking_preserved(golden, inputs, m3_fitted):
    """Per-slice MicroMS3IM ranking vs upstream stays > 0.99 Spearman."""
    gt, pred = inputs
    cubic_scores = np.array(
        [float(m3_fitted.score(gt[i], pred[i])) for i in range(gt.shape[0])]
    )
    rho = _spearman(cubic_scores, np.asarray(golden["m3_scores"]))
    assert rho > 0.99, f"Spearman={rho:.4f}{_format_pins(golden)}"
