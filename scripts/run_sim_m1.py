"""Step S8: supervised vs RL in the simulator at slate size K = 1, across long-term-effect strengths.

python scripts/run_sim_m1.py --strengths 0 0.25 0.5 1.0 --seeds 0 1 2
Writes results/sim_m1/<strength>-s<seed>.json with first-slate CTR, clicks/session, session length
and junk clicks for every agent on held-out users, and results/sim_m1_table.md.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rlrec.agents.sim_agents import Candidates, collect_logs, net_policy, train_dqn, train_offline  # noqa: E402
from rlrec.sim.policies import POLICIES, _no_repeat, topk  # noqa: E402
from rlrec.sim.user_sim import SimConfig, UserSim, rollout  # noqa: E402

AGENTS = ["popular", "rating", "supervised", "sqn", "cql", "mnlq", "dqn", "myopic", "quality"]
LABEL = {"popular": "Popularity", "rating": "Highest estimated rating (history model)", "supervised": "Supervised (CE)",
         "sqn": "SQN-style (CE + TD aux)", "cql": "Offline RL (CQL)", "mnlq": "Offline RL (MNL-Q, ours)",
         "dqn": "Online RL (DQN re-ranking top-20)", "myopic": "Oracle myopic", "quality": "Oracle quality"}


def rating_policy():
    """Non-learned heuristic: recommend the highest rating predicted by the history model (data/history_features.py)."""
    d = np.load(ROOT / "data/processed/ml1m_hist_feats.npz")
    est = d["mu"] + d["user_feat"] @ d["item_feat"].T
    return lambda sim, obs: topk(_no_repeat(sim, est[sim.u]), sim.cfg.K)


def evaluate(cfg, policy, users, seed):
    m = rollout(UserSim(cfg, seed=10_000 + seed), policy, users=np.repeat(users, 2))
    return {k: float(v.mean()) for k, v in m.items()} | {"return_se": float(m["return"].std() / np.sqrt(len(m["return"])))}


def run(strength, seed, dqn_steps, log):
    torch.set_num_threads(1)
    cfg = SimConfig(strength=strength, K=1)
    sim = UserSim(cfg, seed=seed)
    rng = np.random.default_rng(seed)
    users = rng.permutation(sim.n_users)
    train_u, test_u = users[: int(0.8 * len(users))], users[int(0.8 * len(users)):]
    out = {}
    t0 = time.time()
    logs = collect_logs(sim, train_u, sessions_per_user=2, seed=seed)
    log(f"s={strength} seed={seed}: {len(logs['reward'])} logged transitions ({time.time() - t0:.0f}s)")
    pols = {name: POLICIES[name] for name in ("popular", "myopic", "quality")} | {"rating": rating_policy()}
    nets = {}
    for kind in ("supervised", "sqn", "cql", "mnlq"):
        nets[kind] = train_offline(sim, logs, kind, seed=seed)
        pols[kind] = net_policy(nets[kind], {"cql": "q", "mnlq": "mnlq"}.get(kind, "choice"))
    cand = Candidates(nets["supervised"], 20)
    dqn = train_dqn(UserSim(cfg, seed=seed + 1), train_u, steps=dqn_steps, seed=seed, log=log, candidates=cand)
    pols["dqn"] = net_policy(dqn, "q", cand)
    for name in AGENTS:
        out[name] = evaluate(cfg, pols[name], test_u, seed)
        log(f"  {name:10s} ctr1 {out[name]['ctr_first']:.3f}  return {out[name]['return']:.2f}  "
            f"len {out[name]['length']:.1f}  junk {out[name]['junk_clicks']:.2f}")
    out["_meta"] = {"secs": time.time() - t0, "n_logged": int(len(logs["reward"])), "dqn_steps": dqn_steps}
    return out


def table():
    rows = ["| strength s | agent | first-slate CTR | clicks / session | session length | junk clicks | seeds |",
            "| --- | --- | --- | --- | --- | --- | --- |"]
    files = sorted((ROOT / "results/sim_m1").glob("*.json"))
    by_s = {}
    for f in files:
        d = json.load(open(f))
        by_s.setdefault(d["_strength"], []).append(d)
    for s in sorted(by_s):
        runs = by_s[s]
        for a in AGENTS:
            g = lambda k: np.array([r[a][k] for r in runs])
            rows.append(f"| {s} | {LABEL[a]} | {g('ctr_first').mean():.3f} | {g('return').mean():.2f} ± {g('return').std():.2f} | "
                        f"{g('length').mean():.1f} | {g('junk_clicks').mean():.2f} | {len(runs)} |")
    txt = "\n".join(rows) + "\n"
    (ROOT / "results/sim_m1_table.md").write_text(txt)
    return txt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--strengths", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--dqn_steps", type=int, default=3000)
    a = ap.parse_args()
    (ROOT / "results/sim_m1").mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        for s in a.strengths:
            res = run(s, seed, a.dqn_steps, lambda m: print(m, flush=True))
            res["_strength"] = s
            json.dump(res, open(ROOT / f"results/sim_m1/{s}-s{seed}.json", "w"), indent=1)
    print(table())
