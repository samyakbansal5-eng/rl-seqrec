"""Reference policies for the simulator (oracles use hidden information; they are for sanity checks only).

  random     : K random items
  popular    : K most-rated items (non-personalised)
  myopic     : top-K by the TRUE current utility = the ideal supervised next-click recommender (Lemma 1)
  quality    : top-K by predicted rating among non-junk items (a hand-made long-term-aware heuristic)
"""
import numpy as np


def _no_repeat(sim, scores):
    hist = sim.hist
    rows = np.arange(len(scores))[:, None]
    s = scores.copy()
    s[rows, np.maximum(hist, 0)] = np.where(hist >= 0, -np.inf, s[rows, np.maximum(hist, 0)])
    return s


def topk(scores, K):
    return np.argsort(-scores, 1)[:, :K]


def random_policy(sim, obs):
    return topk(sim.rng.random((len(sim.u), sim.n_items)), sim.cfg.K)


def popular_policy(sim, obs):
    pop = -np.arange(sim.n_items, dtype=float)  # catalogue is sorted by popularity
    return topk(_no_repeat(sim, np.broadcast_to(pop, (len(sim.u), sim.n_items))), sim.cfg.K)


def myopic_oracle(sim, obs):
    return topk(_no_repeat(sim, sim.utilities()), sim.cfg.K)


def quality_oracle(sim, obs):
    q = sim.r_hat[sim.u] - 10.0 * sim.junk[None]
    return topk(_no_repeat(sim, q), sim.cfg.K)


POLICIES = {"random": random_policy, "popular": popular_policy, "myopic": myopic_oracle, "quality": quality_oracle}
