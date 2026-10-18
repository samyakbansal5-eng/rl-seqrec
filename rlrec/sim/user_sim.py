"""Simulator v1 (step S7): MovieLens-1M users with MNL choice and choice-dependent long-term dynamics.

Batched: B users are simulated in parallel with numpy.

Observation (what a recommender sees): user id, last `hist_len` clicked items, step index.
Hidden state: satisfaction sigma in [0, 1] and genre fatigue f in R^18.

Each step the recommender shows a slate of up to K items (item ids 0..n-1, -1 = empty slot).
  Choice (MNL, Lee & Oh Assumption 3.1), outside option utility 0:
      v(u, i) = b0 + beta_r (r_hat[u, i] - 3.5) + kappa * junk_i - s * lambda_f * overlap(f, genres_i)
  Reward: 1 per click (the short-term signal a CTR model optimises).
  Transition (scaled by the long-term-effect strength s in [0, 1]):
      click on i : sigma += s * eta * (q_ui - q_ref),   q_ui = (r_hat[u,i] - 1) / 4
                   junk click additionally: sigma -= s * eta_junk, and the user leaves w.p. s * p_junk
      no click   : nothing changes
      fatigue    : f <- decay * f + genres_i / |genres_i|   (only enters utility through s * lambda_f)
  Exit: each step the user leaves w.p.  base_exit + s * c_exit * (1 - sigma); sessions are capped at H steps.

With s = 0 neither the utilities nor the exit probability depend on what the user picked, so by
Proposition 2 the myopic (supervised) policy is optimal; s > 0 switches the long-term effects on.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SimConfig:
    strength: float = 0.5  # s: 0 = no long-term effects
    K: int = 1  # slate size
    H: int = 30
    hist_len: int = 10
    b0: float = -2.0
    beta_r: float = 1.0
    kappa: float = 1.0  # junk items are more attractive ("clickbait")
    lambda_f: float = 1.5
    fatigue_decay: float = 0.8
    sigma0: float = 0.6
    eta: float = 0.4
    q_ref: float = 0.6
    eta_junk: float = 0.15
    p_junk: float = 0.3
    base_exit: float = 0.05
    c_exit: float = 0.4
    data: str = str(ROOT / "data/processed/ml1m_sim.npz")
    extra: dict = field(default_factory=dict)


class UserSim:
    def __init__(self, cfg: SimConfig, seed=0):
        self.cfg = cfg
        d = np.load(cfg.data, allow_pickle=False)
        self.r_hat = d["r_hat"]
        self.genres = d["genres"]
        self.gnorm = self.genres / np.maximum(self.genres.sum(1, keepdims=True), 1)
        self.junk = d["junk"].astype(bool)
        self.titles = d["titles"]
        self.n_users, self.n_items = self.r_hat.shape
        self.quality = (self.r_hat - 1.0) / 4.0
        self.base_util = cfg.b0 + cfg.beta_r * (self.r_hat - 3.5) + cfg.kappa * self.junk[None]
        self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------ state
    def reset(self, users=None, B=None):
        if users is None:
            users = self.rng.integers(0, self.n_users, B)
        self.u = np.asarray(users)
        B = len(self.u)
        self.sigma = np.full(B, self.cfg.sigma0)
        self.f = np.zeros((B, self.genres.shape[1]))
        self.hist = np.full((B, self.cfg.hist_len), -1, dtype=np.int64)
        self.t = 0
        self.active = np.ones(B, bool)
        return self.obs()

    def obs(self):
        return {"user": self.u.copy(), "history": self.hist.copy(), "t": self.t, "active": self.active.copy()}

    def utilities(self):
        """True utilities v(u, i) for every active user and item, shape (B, n_items). For oracles / analysis."""
        over = self.f @ self.gnorm.T
        return self.base_util[self.u] - self.cfg.strength * self.cfg.lambda_f * over

    # ------------------------------------------------------------------ dynamics
    def choice_probs(self, slates):
        """slates: (B, K) item ids, -1 = empty. Returns (B, K+1) probabilities, column 0 = no click."""
        v = self.utilities()
        valid = slates >= 0
        w = np.where(valid, np.exp(np.take_along_axis(v, np.maximum(slates, 0), 1)), 0.0)
        p = np.concatenate([np.ones((len(w), 1)), w], 1)
        return p / p.sum(1, keepdims=True)

    def step(self, slates):
        cfg, s = self.cfg, self.cfg.strength
        slates = np.asarray(slates).reshape(len(self.u), -1)
        assert slates.shape[1] <= cfg.K
        P = self.choice_probs(slates)
        k = (self.rng.random(len(P))[:, None] > np.cumsum(P, 1)).sum(1)  # sampled column
        k = np.minimum(k, P.shape[1] - 1)
        clicked = np.where(k > 0, np.take_along_axis(slates, np.maximum(k - 1, 0)[:, None], 1)[:, 0], -1)
        clicked = np.where(self.active, clicked, -1)
        reward = (clicked >= 0).astype(float)

        c = clicked >= 0
        idx = np.maximum(clicked, 0)
        q = self.quality[self.u, idx]
        jk = self.junk[idx] & c
        self.sigma = np.clip(self.sigma + c * s * cfg.eta * (q - cfg.q_ref) - jk * s * cfg.eta_junk, 0.0, 1.0)
        self.f = np.where(c[:, None], cfg.fatigue_decay * self.f + self.gnorm[idx], self.f)
        self.hist = np.where(c[:, None], np.concatenate([self.hist[:, 1:], clicked[:, None]], 1), self.hist)

        p_exit = cfg.base_exit + s * cfg.c_exit * (1 - self.sigma)
        leave = (self.rng.random(len(P)) < p_exit) | (jk & (self.rng.random(len(P)) < s * cfg.p_junk))
        was_active = self.active.copy()
        self.t += 1
        self.active = self.active & ~leave & (self.t < cfg.H)
        info = {"clicked": clicked, "sigma": self.sigma.copy(), "left": was_active & ~self.active,
                "junk_click": jk, "p_click": 1 - P[:, 0]}
        return self.obs(), reward * was_active, ~self.active, info


def rollout(sim, policy, users=None, B=1000):
    """Run full sessions. policy(sim, obs) -> (B, K) slates. Returns per-user dict of metrics."""
    obs = sim.reset(users=users, B=B)
    n = len(sim.u)
    ret, length, first_ctr, junk = np.zeros(n), np.zeros(n), None, np.zeros(n)
    while obs["active"].any():
        act = obs["active"]
        obs, r, done, info = sim.step(policy(sim, obs))
        if first_ctr is None:
            first_ctr = info["p_click"].copy()
        ret += r
        length += act
        junk += info["junk_click"] * act
    return {"return": ret, "length": length, "ctr_first": first_ctr, "ctr_step": ret / np.maximum(length, 1),
            "junk_clicks": junk}


def calibrate_base_exit(cfg, policy, target_len=10.0, B=2000, seed=0):
    """Bisection on base_exit so that the mean session length under `policy` matches `target_len`."""
    lo, hi = 0.0, 0.5
    for _ in range(14):
        mid = (lo + hi) / 2
        cfg.base_exit = mid
        m = rollout(UserSim(cfg, seed), policy, B=B)["length"].mean()
        lo, hi = (mid, hi) if m > target_len else (lo, mid)
    cfg.base_exit = (lo + hi) / 2
    return cfg.base_exit
