"""What the learned recommenders know about each user before the simulated sessions start (step S8).

A real recommender is trained on the users' past interactions. We give it that past as a separate model:
ALS (rank 32) fitted on a random half of each user's MovieLens-1M ratings. The simulator's ground truth
(data/preprocess_ml1m.py) uses all ratings, so these features are informative but not exact.
Output: data/processed/ml1m_hist_feats.npz with user_feat (users, rank), item_feat (catalogue items, rank)
and the global mean, so that mu + user_feat @ item_feat.T is the recommender's own rating estimate.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "data"))
from preprocess_ml1m import als  # noqa: E402


def main(a):
    R = pd.read_csv(ROOT / "data/raw/ml-1m/ml1m_ratings.dat", sep="\t", header=None, names=["u", "i", "r", "t"])
    uids, iids = np.sort(R.u.unique()), np.sort(R.i.unique())
    R["uu"] = R.u.map({u: k for k, u in enumerate(uids)})
    R["ii"] = R.i.map({i: k for k, i in enumerate(iids)})
    sim = np.load(ROOT / "data/processed/ml1m_sim.npz")
    cat = np.searchsorted(iids, sim["movie_ids"])

    keep = np.random.default_rng(a.seed).random(len(R)) < a.frac
    H = R[keep].copy()
    mu = H.r.mean()
    H["c"] = H.r - mu
    by_u, by_i = dict(list(H.groupby("uu"))), dict(list(H.groupby("ii")))
    empty = (np.zeros(0, int), np.zeros(0))
    rows = lambda g, col: (g[col].to_numpy(), g.c.to_numpy()) if g is not None else empty
    U, V = als([rows(by_u.get(u), "ii") for u in range(len(uids))], [rows(by_i.get(i), "uu") for i in range(len(iids))],
               len(uids), len(iids), a.rank, a.lam, a.iters, seed=a.seed + 1)
    est = np.clip(mu + U @ V[cat].T, 1, 5)
    corr = np.mean([np.corrcoef(est[u], sim["r_hat"][u])[0, 1] for u in range(len(uids))])
    out = ROOT / "data/processed/ml1m_hist_feats.npz"
    np.savez_compressed(out, user_feat=U.astype(np.float32), item_feat=V[cat].astype(np.float32), mu=mu)
    print(f"history ALS on {keep.mean():.0%} of ratings; mean per-user correlation with simulator r_hat {corr:.3f}; saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lam", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
