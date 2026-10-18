"""Agents for combinatorial RL with MNL preference feedback (Lee & Oh 2025), tabular-state version
used on the synthetic budget MDP.

  Myopic      OFU-MNL+ adapted to unknown rewards: optimistic MNL utilities, item value = estimated
              immediate reward only (ignores transitions). This is the gamma = 0 policy.
  LSVIUCB     LSVI-UCB with each assortment as one atomic action (one-hot features = tabular),
              which needs all sum_{m<=K} C(N, m) assortments enumerated.
  MNLVQL      MNL-VQL for linear MDPs with sigma_bar = 1 (unweighted regression), as in the paper's
              experiments: online-mirror-descent MNL estimation, optimistic / overly optimistic /
              pessimistic item-level Q-values from ridge regression on psi(s,a) with elliptical bonuses,
              Eq. (7) utility switching, assortment selection by exact MNL optimisation, Eq. (9) policy switch.

Implementation choices (documented in the S5 notes): one MNL parameter shared across steps h
(the environment's preference model is stationary), bonus scales are hyper-parameters instead of the
theoretical constants, and plans are recomputed every `replan` episodes.
"""
from itertools import combinations

import numpy as np

from rlrec.slate.assortment import solve_parametric


class MNLEstimator:
    """Online mirror descent for the MNL parameter (Lee & Oh 2024; Lee & Oh 2025 Eq. 1-2)."""

    def __init__(self, d, lam=1.0, eta=1.0, alpha=0.5):
        self.theta = np.zeros(d)
        self.Hm = lam * np.eye(d)
        self.eta, self.alpha, self.t = eta, alpha, 0

    @staticmethod
    def _grad_hess(theta, X, y):
        u = X @ theta
        e = np.exp(u - max(u.max(), 0.0))
        p = e / (np.exp(-max(u.max(), 0.0)) + e.sum())
        g = X.T @ (p - y)
        m = X.T @ p
        Hs = (X.T * p) @ X - np.outer(m, m)
        return g, Hs

    def update(self, X, y):
        """X: (n, d) features of the offered items, y: one-hot of the chosen item (all zeros = a0)."""
        g, Hs = self._grad_hess(self.theta, X, y)
        Ht = self.Hm + self.eta * Hs
        self.theta = self.theta - self.eta * np.linalg.solve(Ht, g)
        _, Hs_new = self._grad_hess(self.theta, X, y)
        self.Hm = self.Hm + Hs_new
        self.t += 1

    def utilities(self, phi, sign=0):
        """phi: (..., d). sign=+1 optimistic, -1 pessimistic, 0 point estimate."""
        u = phi @ self.theta
        if sign == 0:
            return u
        Hinv = np.linalg.inv(self.Hm)
        width = np.sqrt(np.einsum("...i,ij,...j->...", phi, Hinv, phi))
        return u + sign * self.alpha * np.sqrt(np.log(2 + self.t)) * width


def _feasible_utilities(est, phi, f):
    """Eq. (7): optimistic utilities where some item's value >= the outside option's, else pessimistic."""
    opt = (f[:, 1:] >= f[:, :1]).any(1, keepdims=True)
    return np.where(opt, est.utilities(phi, +1), est.utilities(phi, -1))


class Myopic:
    name = "Myopic"

    def __init__(self, env, alpha=0.5, beta=0.05, **_):
        self.env = env
        self.est = MNLEstimator(env.d, alpha=alpha)
        self.rsum = np.zeros((env.S, env.N + 1))
        self.n = np.zeros((env.S, env.N + 1))
        self.beta = beta

    def plan(self, k):
        f = np.where(self.n > 0, self.rsum / np.maximum(self.n, 1), 0.0) + self.beta / np.sqrt(self.n + 1)
        f[:, 0] = 0.0  # buying nothing pays nothing now; the myopic agent ignores the budget increase
        u = _feasible_utilities(self.est, self.env.phi, f)
        w = np.exp(u)
        _, self.mask = solve_parametric(w[:, 0], f[:, 0], w[:, 1:], f[:, 1:], self.env.K)

    def act(self, h, s):
        return list(np.flatnonzero(self.mask[s]) + 1)

    def observe(self, h, s, A, a, r, s2):
        X = self.env.phi[s, A]
        self.est.update(X, (np.asarray(A) == a).astype(float))
        self.rsum[s, a] += r
        self.n[s, a] += 1


class LSVIUCB:
    name = "LSVI-UCB"

    def __init__(self, env, beta=0.2, **_):
        self.env = env
        acts = [c for m in range(1, env.K + 1) for c in combinations(range(1, env.N + 1), m)]
        self.acts = acts
        self.index = {a: i for i, a in enumerate(acts)}
        nA = len(acts)
        self.n = np.zeros((env.H, env.S, nA))
        self.rsum = np.zeros((env.H, env.S, nA))
        self.trans = np.zeros((env.H, env.S, nA, env.S))
        self.beta = beta

    def plan(self, k):
        H, S = self.env.H, self.env.S
        V = np.zeros((H + 1, S))
        self.best = np.zeros((H, S), int)
        for h in range(H - 1, -1, -1):
            n = np.maximum(self.n[h], 1)
            Q = self.rsum[h] / n + (self.trans[h] / n[..., None]) @ V[h + 1] + self.beta / np.sqrt(self.n[h] + 1)
            Q = np.where(self.n[h] > 0, np.minimum(Q, 1.0), 1.0)  # unvisited: maximally optimistic
            self.best[h] = Q.argmax(1)
            V[h] = Q.max(1)

    def act(self, h, s):
        return list(self.acts[self.best[h, s]])

    def observe(self, h, s, A, a, r, s2):
        i = self.index[tuple(A)]
        self.n[h, s, i] += 1
        self.rsum[h, s, i] += r
        self.trans[h, s, i, s2] += 1


class MNLVQL:
    name = "MNL-VQL"

    def __init__(self, env, alpha=0.5, beta=0.05, lam=1.0, u_switch=0.5, **_):
        self.env = env
        H, S, dl = env.H, env.S, env.psi.shape[-1]
        self.est = MNLEstimator(env.d, alpha=alpha)
        self.Lam = np.repeat(lam * np.eye(dl)[None], H, 0)
        self.Rv = np.zeros((H, dl))  # sum psi * r
        self.Bm = np.zeros((H, dl, S))  # sum psi * 1[s' = .]
        self.beta, self.u = beta, u_switch

    def plan(self, k):
        env, H, S, N = self.env, self.env.H, self.env.S, self.env.N
        psi = env.psi  # (S, N+1, dl)
        V = {j: np.zeros(S) for j in (1, 2, -2)}
        self.f = np.zeros((H, 3, S, N + 1))
        self.masks = np.zeros((H, 3, S, N), bool)
        for h in range(H - 1, -1, -1):
            Linv = np.linalg.inv(self.Lam[h])
            b = self.beta * np.sqrt(np.einsum("sai,ij,saj->sa", psi, Linv, psi))
            newV = {}
            for idx, j in enumerate((1, 2, -2)):
                wvec = Linv @ (self.Rv[h] + self.Bm[h] @ V[j])
                fhat = psi @ wvec
                if j == 1:
                    f = np.minimum(fhat + b, 1.0)
                elif j == 2:
                    f = np.minimum(fhat + 3 * b, 1.0)  # \hat f_{h,2} + 2 b_{h,1} + b_{h,2}, sigma_bar = 1
                else:
                    f = np.maximum(fhat - b, 0.0)
                u = _feasible_utilities(self.est, env.phi, f)
                w = np.exp(u)
                newV[j], self.masks[h, idx] = solve_parametric(w[:, 0], f[:, 0], w[:, 1:], f[:, 1:], env.K)
                self.f[h, idx] = f
            V = newV
        self.switched = False

    def act(self, h, s):
        if h == 0:
            self.switched = False
        A1 = np.flatnonzero(self.masks[h, 0, s]) + 1
        gap = (self.f[h, 1, s, A1] - self.f[h, 0, s, A1]).max()
        if gap > self.u:  # Eq. (9): optimistic and overly optimistic estimates diverge
            self.switched = True
        idx = 1 if self.switched else 0
        return list(np.flatnonzero(self.masks[h, idx, s]) + 1)

    def observe(self, h, s, A, a, r, s2):
        self.est.update(self.env.phi[s, A], (np.asarray(A) == a).astype(float))
        p = self.env.psi[s, a]
        self.Lam[h] += np.outer(p, p)
        self.Rv[h] += p * r
        self.Bm[h, :, s2] += p


AGENTS = {"myopic": Myopic, "lsvi": LSVIUCB, "mnlvql": MNLVQL}
