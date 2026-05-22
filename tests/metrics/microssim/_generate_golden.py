r"""Generate the frozen upstream MicroSSIM parity fixture.

Run **once**, on a host with the pinned upstream microssim installed.
The output is committed and treated as ground truth; the cubic port is
required to match it within 1e-3 regardless of installed
numpy/scipy/skimage versions.

Recommended invocation::

    uv run --with 'microssim @ git+https://github.com/juglab/microssim@8bccb17d' \\
           --with 'numpy==1.26.4' --with 'scipy==1.11.4' --with 'scikit-image==0.22.0' \\
           python tests/metrics/microssim/_generate_golden.py

Output: ``tests/metrics/microssim/data/microssim_golden.npz`` (~1 kB).

Only the seed and the golden outputs are stored; inputs are regenerated
from the seed at test time.

This file is **not collected by pytest** (filename starts with ``_`` and
``conftest.py`` lists it in ``collect_ignore`` for defense in depth).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _generate_inputs(seed: int = 42, n: int = 60, h: int = 256, w: int = 256):
    """Generate the seeded exponential GT + scaled-and-noised prediction.

    Kept in sync with ``test_upstream_parity._inputs_from_seed`` — both
    must produce identical arrays from the same seed.
    """
    rng = np.random.default_rng(seed)
    gt = rng.exponential(0.5, (n, h, w)).astype(np.float32) * 1000
    pred = gt * 1.2 + 0.1 * float(gt.max()) * rng.standard_normal((n, h, w)).astype(
        np.float32
    )
    return gt, pred


def main() -> None:
    """Run the upstream MicroSSIM and MicroMS3IM pipelines and write the npz."""
    import scipy  # noqa: F401  imported for version metadata
    import skimage  # noqa: F401  imported for version metadata
    from microssim import MicroSSIM, MicroMS3IM

    seed = 42
    n, h, w = 60, 256, 256
    gt, pred = _generate_inputs(seed, n, h, w)

    print("Fitting upstream MicroSSIM...")
    ms = MicroSSIM().fit(gt, pred)
    ms_params = ms.get_parameters()
    ms_scores = np.array(
        [float(ms.score(gt[i], pred[i])) for i in range(n)], dtype=np.float64
    )

    print("Fitting upstream MicroMS3IM...")
    m3 = MicroMS3IM().fit(gt, pred)
    m3_params = m3.get_parameters()
    m3_scores = np.array(
        [float(m3.score(gt[i], pred[i])) for i in range(n)], dtype=np.float64
    )

    out = Path(__file__).parent / "data" / "microssim_golden.npz"
    out.parent.mkdir(exist_ok=True, parents=True)
    np.savez_compressed(
        out,
        _seed=np.int64(seed),
        _N=np.int64(n),
        _H=np.int64(h),
        _W=np.int64(w),
        _dtype="float32",
        _pin_microssim="8bccb17d",
        _pin_numpy=np.__version__,
        _pin_scipy=scipy.__version__,
        _pin_skimage=skimage.__version__,
        ms_bg_percentile=float(ms_params["bg_percentile"]),
        ms_offset_gt=float(ms_params["offset_gt"]),
        ms_offset_pred=float(ms_params["offset_pred"]),
        ms_max_val=float(ms_params["max_val"]),
        ms_ri_factor=float(ms_params["ri_factor"]),
        ms_scores=ms_scores,
        m3_ri_factor=float(m3_params["ri_factor"]),
        m3_scores=m3_scores,
    )
    print(f"Wrote {out} (size = {out.stat().st_size} bytes)")
    print(f"ms_ri_factor = {ms_params['ri_factor']:.10f}")
    print(f"m3_ri_factor = {m3_params['ri_factor']:.10f}")
    print(f"ms_scores mean = {ms_scores.mean():.6f}, std = {ms_scores.std():.6f}")
    print(f"m3_scores mean = {m3_scores.mean():.6f}, std = {m3_scores.std():.6f}")


if __name__ == "__main__":
    main()
