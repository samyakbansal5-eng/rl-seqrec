"""Train and save the two agents the demo shows side by side (supervised next-click model and offline RL, MNL-Q),
for each long-term-effect strength offered in the demo. Output: results/checkpoints/{kind}-s{strength}.pt
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rlrec.agents.sim_agents import collect_logs, train_offline  # noqa: E402
from rlrec.sim.user_sim import SimConfig, UserSim  # noqa: E402

DEMO_STRENGTHS = (0.0, 0.5, 1.0)

if __name__ == "__main__":
    torch.set_num_threads(1)
    out = ROOT / "results/checkpoints"
    out.mkdir(parents=True, exist_ok=True)
    for s in DEMO_STRENGTHS:
        cfg = SimConfig(strength=s, K=1)
        sim = UserSim(cfg, seed=0)
        users = np.random.default_rng(0).permutation(sim.n_users)[: int(0.8 * sim.n_users)]
        logs = collect_logs(sim, users, sessions_per_user=2, seed=0)
        torch.save(train_offline(sim, logs, "supervised", seed=0).state_dict(), out / f"supervised-s{s}.pt")
        torch.save(train_offline(sim, logs, "mnlq", seed=0).state_dict(), out / f"mnlq-s{s}.pt")
        print("saved strength", s, flush=True)
