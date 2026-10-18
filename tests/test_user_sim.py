import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rlrec.sim.policies import myopic_oracle, popular_policy, quality_oracle  # noqa: E402
from rlrec.sim.user_sim import SimConfig, UserSim, rollout  # noqa: E402


def test_choice_probs_sum_to_one_and_ignore_empty_slots():
    sim = UserSim(SimConfig(K=3), seed=0)
    sim.reset(B=50)
    slates = np.tile(np.array([[3, -1, 7]]), (50, 1))
    P = sim.choice_probs(slates)
    assert np.allclose(P.sum(1), 1) and np.all(P[:, 2] == 0)


def test_strength_zero_makes_dynamics_choice_independent():
    """Proposition 2's condition: with s = 0, utilities and exit probabilities do not depend on past choices."""
    cfg = SimConfig(strength=0.0, K=1)
    sim = UserSim(cfg, seed=0)
    sim.reset(users=np.zeros(200, int))
    u0 = sim.utilities().copy()
    for _ in range(5):
        sim.step(myopic_oracle(sim, sim.obs()))
    sim.active[:] = True
    assert np.allclose(sim.utilities(), u0)
    assert np.allclose(sim.sigma, cfg.sigma0)


def test_myopic_wins_ctr_and_quality_wins_return_with_strong_effects():
    cfg = SimConfig(strength=1.0, K=1)
    m = rollout(UserSim(cfg, 1), myopic_oracle, B=3000)
    q = rollout(UserSim(cfg, 1), quality_oracle, B=3000)
    assert m["ctr_first"].mean() > q["ctr_first"].mean()
    assert q["return"].mean() > m["return"].mean()


def test_reproducible_with_seed():
    a = rollout(UserSim(SimConfig(), 5), popular_policy, users=np.arange(100))
    b = rollout(UserSim(SimConfig(), 5), popular_policy, users=np.arange(100))
    assert np.array_equal(a["return"], b["return"])
