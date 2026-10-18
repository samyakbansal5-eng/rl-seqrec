"""'Online shopping with budget' combinatorial MDP with MNL preference feedback (Lee & Oh 2025, App. G).

States s_1..s_S are budget levels (index 0..S-1 here); items a_1..a_N, outside option a_0.
At most M items are offered including a_0, i.e. K = M - 1 real items.
  buy a_i in s_j :  reward (i/(100N) + j/S)/H ; budget up w.p. 1 - i/N, down w.p. i/N
  buy nothing    :  reward 0                  ; budget up deterministically
Choice: MNL with utility phi(s,a)^T theta*, phi(s,a_0) = 0. Initial state: middle budget.
"""
import numpy as np

from rlrec.slate.assortment import solve_parametric


class BudgetEnv:
    def __init__(self, N=10, S=5, H=5, M=6, d=5, seed=0):
        rng = np.random.default_rng(seed)
        self.N, self.S, self.H, self.K, self.d = N, S, H, M - 1, d
        self.s0 = (S + 1) // 2 - 1  # s_ceil(S/2), 0-based
        # MNL features: uniform, scaled to ||phi|| <= 1; theta* uniform in [-1, 1]^d scaled by 2
        phi = rng.uniform(0, 1, (S, N + 1, d)) / np.sqrt(d)
        phi[:, 0] = 0.0
        self.phi = phi
        self.theta = rng.uniform(-1, 1, d) * 2.0
        # rewards r[s, a] and transitions P[s, a, s'] (a = 0 is the outside option)
        self.r = np.zeros((S, N + 1))
        self.P = np.zeros((S, N + 1, S))
        for j in range(S):
            up, down = min(j + 1, S - 1), max(j - 1, 0)
            self.P[j, 0, up] = 1.0
            for i in range(1, N + 1):
                self.r[j, i] = (i / (100 * N) + (j + 1) / S) / H
                self.P[j, i, up] += 1 - i / N
                self.P[j, i, down] += i / N
        # linear-MDP features psi(s,a) = [P(.|s,a), r(s,a)], d_lin = S + 1 (exact linear MDP)
        self.psi = np.concatenate([self.P, self.r[..., None]], -1)
        self.rng = np.random.default_rng(seed + 1000)

    def utilities(self, theta=None):
        return self.phi @ (self.theta if theta is None else theta)  # (S, N+1)

    def choice_probs(self, s, A):
        u = self.utilities()[s]
        w = np.exp(u[A])
        return w / (1.0 + w.sum())

    def reset(self):
        self.s, self.h = self.s0, 0
        return self.s

    def step(self, A):
        """A: list of item ids in 1..N. Returns (chosen item or 0, reward, next state, done)."""
        assert 1 <= len(A) <= self.K
        p = self.choice_probs(self.s, np.asarray(A))
        k = self.rng.choice(len(A) + 1, p=np.concatenate([[1 - p.sum()], p]))
        a = 0 if k == 0 else A[k - 1]
        r = self.r[self.s, a]
        s2 = self.rng.choice(self.S, p=self.P[self.s, a])
        self.s, self.h = s2, self.h + 1
        return a, r, s2, self.h >= self.H

    def optimal_policy(self, gamma=1.0):
        """Exact DP with the true model. Returns V[h, s], masks[h, s, N] over items 1..N."""
        S, N, H = self.S, self.N, self.H
        w = np.exp(self.utilities())  # (S, N+1); w[:, 0] = 1
        V = np.zeros((H + 1, S))
        masks = np.zeros((H, S, N), bool)
        for h in range(H - 1, -1, -1):
            Qi = self.r + gamma * self.P @ V[h + 1]  # item-level Q (S, N+1)
            V[h], masks[h] = solve_parametric(w[:, 0], Qi[:, 0], w[:, 1:], Qi[:, 1:], self.K)
        return V, masks
