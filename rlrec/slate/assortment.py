"""Assortment (slate) optimisation under the MNL choice model.

Problem (Lee & Oh 2025, Eq. 8 / Appendix C): given attraction weights w_a = exp(utility) and
item values f_a, plus the outside option a0 with weight w0 and value f0, choose an assortment A
with 1 <= |A| <= K items to maximise

    F(A) = (w0 f0 + sum_{a in A} w_a f_a) / (w0 + sum_{a in A} w_a).

Two exact solvers:
  solve_lp          the Charnes-Cooper linear program of Appendix C (scipy, used to verify and in the
                    neural re-ranker later)
  solve_parametric  F(A) >= lam  <=>  w0 (f0 - lam) + sum_{a in A} w_a (f_a - lam) >= 0, so bisect on lam;
                    the inner max is "take the top-K positive terms". Vectorised over many states.
Both are checked against brute-force enumeration in tests/test_assortment.py.
"""
from itertools import combinations

import numpy as np
from scipy.optimize import linprog


def value(w0, f0, w, f, A):
    A = list(A)
    return (w0 * f0 + np.sum(w[A] * f[A])) / (w0 + np.sum(w[A]))


def solve_bruteforce(w0, f0, w, f, K):
    best, bestA = -np.inf, None
    for m in range(1, K + 1):
        for A in combinations(range(len(w)), m):
            v = value(w0, f0, w, f, A)
            if v > best + 1e-12:
                best, bestA = v, A
    return best, list(bestA)


def solve_lp(w0, f0, w, f, K):
    """Charnes-Cooper LP: variables y_1..y_N, t.
    max sum f_a w_a y_a + f0 w0 t   s.t.  sum w_a y_a + w0 t = 1,  sum y_a <= K t,  0 <= y_a <= t."""
    N = len(w)
    c = -np.concatenate([f * w, [f0 * w0]])
    A_eq = np.concatenate([w, [w0]])[None]
    A_ub = np.zeros((N + 1, N + 1))
    A_ub[0, :N], A_ub[0, N] = 1.0, -K
    A_ub[1:, :N] = np.eye(N)
    A_ub[1:, N] = -1.0
    res = linprog(c, A_ub=A_ub, b_ub=np.zeros(N + 1), A_eq=A_eq, b_eq=[1.0],
                  bounds=[(0, None)] * (N + 1), method="highs")
    y, t = res.x[:N], res.x[N]
    x = y / t
    A = [int(i) for i in np.argsort(-x)[:K] if x[i] > 0.5]
    if not A:  # the LP prefers the empty assortment; the problem requires >= 1 item
        A = [int(np.argmax((w0 * f0 + w * f) / (w0 + w)))]
    return value(w0, f0, w, f, A), A


def _inner(w0, f0, w, f, lam, K):
    """g(lam) and the maximising assortment mask; arrays are (B, N), w0/f0 are (B,)."""
    t = w * (f - lam[:, None])
    order = np.argsort(-t, axis=1)[:, :K]
    top = np.take_along_axis(t, order, 1)
    take = top > 0
    take[:, 0] = True  # at least one item
    g = w0 * (f0 - lam) + np.where(take, top, 0.0).sum(1)
    return g, order, take


def solve_parametric(w0, f0, w, f, K, iters=60):
    """Vectorised: w, f are (B, N); w0, f0 are (B,). Returns values (B,) and item mask (B, N)."""
    w, f = np.atleast_2d(w), np.atleast_2d(f)
    B = w.shape[0]
    w0 = np.broadcast_to(np.asarray(w0, float), (B,))
    f0 = np.broadcast_to(np.asarray(f0, float), (B,))
    lo = np.minimum(f.min(1), f0) - 1e-9
    hi = np.maximum(f.max(1), f0) + 1e-9
    for _ in range(iters):
        mid = (lo + hi) / 2
        g, _, _ = _inner(w0, f0, w, f, mid, K)
        up = g >= 0
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    _, order, take = _inner(w0, f0, w, f, lo, K)
    mask = np.zeros_like(w, dtype=bool)
    np.put_along_axis(mask, order, take, 1)
    vals = (w0 * f0 + (w * f * mask).sum(1)) / (w0 + (w * mask).sum(1))
    return vals, mask
