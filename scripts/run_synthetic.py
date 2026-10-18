"""Step S5: reproduce Lee & Oh (2025) Figure 1 on the synthetic budget MDP.

python scripts/run_synthetic.py --N 10 --agents myopic lsvi mnlvql --seeds 0 1 2 3 4 --K 30000
Saves per-episode returns to results/synthetic/N{N}-{agent}-s{seed}.npy and the optimal value.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rlrec.agents.combinatorial import AGENTS  # noqa: E402
from rlrec.sim.budget_env import BudgetEnv  # noqa: E402


def run(N, agent, seed, K, replan):
    env = BudgetEnv(N=N, seed=seed)
    ag = AGENTS[agent](env)
    rets = np.zeros(K)
    t0 = time.time()
    for k in range(K):
        if k % replan == 0:
            ag.plan(k)
        s, done, h, G = env.reset(), False, 0, 0.0
        while not done:
            A = ag.act(h, s)
            a, r, s2, done = env.step(A)
            ag.observe(h, s, A, a, r, s2)
            G += r
            s, h = s2, h + 1
        rets[k] = G
    return rets, (time.time() - t0) / K


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=10)
    ap.add_argument("--agents", nargs="+", default=["myopic", "mnlvql", "lsvi"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--K", type=int, default=30000)
    ap.add_argument("--replan", type=int, default=5)
    a = ap.parse_args()
    out = ROOT / "results/synthetic"
    out.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        env = BudgetEnv(N=a.N, seed=seed)
        V, _ = env.optimal_policy()
        json.dump({"optimal_value": float(V[0, env.s0])}, open(out / f"N{a.N}-optimal-s{seed}.json", "w"))
        for ag in a.agents:
            rets, spe = run(a.N, ag, seed, a.K, a.replan)
            np.save(out / f"N{a.N}-{ag}-s{seed}.npy", rets)
            print(f"N={a.N} seed={seed} {ag}: last-2000 mean return {rets[-2000:].mean():.4f} "
                  f"(optimal {V[0, env.s0]:.4f}), {spe * 1000:.2f} ms/episode", flush=True)
