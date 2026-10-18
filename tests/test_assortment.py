import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlrec.slate.assortment import solve_bruteforce, solve_lp, solve_parametric  # noqa: E402


def test_solvers_match_bruteforce():
    rng = np.random.default_rng(0)
    for trial in range(300):
        N = int(rng.integers(2, 11))
        K = int(rng.integers(1, min(N, 6) + 1))
        w = np.exp(rng.normal(size=N))
        f = rng.uniform(0, 1, N)
        w0, f0 = 1.0, float(rng.uniform(0, 1))
        vb, _ = solve_bruteforce(w0, f0, w, f, K)
        vl, Al = solve_lp(w0, f0, w, f, K)
        vp, mask = solve_parametric(np.array([w0]), np.array([f0]), w[None], f[None], K)
        assert len(Al) <= K and mask.sum() <= K and mask.sum() >= 1
        assert abs(vl - vb) < 1e-6, (trial, vl, vb)
        assert abs(vp[0] - vb) < 1e-6, (trial, vp, vb)
