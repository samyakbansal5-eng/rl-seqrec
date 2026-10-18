"""Build the item/user tables the MovieLens-1M user simulator needs (step S7).

- catalogue: the `--n_items` most-rated movies (default 500) so the simulator stays fast
- matrix factorisation (ALS, rank 32) on all ratings -> predicted rating r_hat[u, i] for every user and catalogue item
- genres (18, multi-hot) for the fatigue model
- junk flag: catalogue items in the bottom `--junk_frac` by mean rating (popular but poorly rated, i.e.
  "attractive but low quality", the Lee & Oh junk-item protocol)
Output: data/processed/ml1m_sim.npz
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
          "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


def als(R_users, R_items, n_users, n_items, rank, lam, iters, seed=0):
    """Explicit-feedback ALS with per-user / per-item bias absorbed into a global mean."""
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 0.1, (n_users, rank))
    V = rng.normal(0, 0.1, (n_items, rank))
    I = lam * np.eye(rank)
    for _ in range(iters):
        for u, (idx, r) in enumerate(R_users):
            Vi = V[idx]
            U[u] = np.linalg.solve(Vi.T @ Vi + I * max(len(idx), 1), Vi.T @ r)
        for i, (idx, r) in enumerate(R_items):
            Uu = U[idx]
            V[i] = np.linalg.solve(Uu.T @ Uu + I * max(len(idx), 1), Uu.T @ r)
    return U, V


def main(a):
    raw = ROOT / "data/raw/ml-1m"
    R = pd.read_csv(raw / "ml1m_ratings.dat", sep="\t", header=None, names=["u", "i", "r", "t"])
    M = pd.read_csv(raw / "ml1m_items.dat", sep="\t", header=None, names=["i", "title", "genres"], encoding="latin-1")
    uids = np.sort(R.u.unique())
    iids = np.sort(R.i.unique())
    umap = {u: k for k, u in enumerate(uids)}
    imap = {i: k for k, i in enumerate(iids)}
    R["uu"], R["ii"] = R.u.map(umap), R.i.map(imap)
    mu = R.r.mean()
    R["c"] = R.r - mu

    R_users = [(g.ii.to_numpy(), g.c.to_numpy()) for _, g in R.groupby("uu")]
    R_items = [(g.uu.to_numpy(), g.c.to_numpy()) for _, g in R.groupby("ii")]
    U, V = als(R_users, R_items, len(uids), len(iids), a.rank, a.lam, a.iters)

    # held-out sanity check is not needed for a simulator, but report training RMSE
    pred = mu + np.einsum("ij,ij->i", U[R.uu], V[R.ii])
    rmse = float(np.sqrt(np.mean((pred - R.r) ** 2)))

    counts = R.groupby("ii").size()
    cat = counts.sort_values(ascending=False).index[: a.n_items].to_numpy()
    r_hat = np.clip(mu + U @ V[cat].T, 1.0, 5.0).astype(np.float32)  # (users, n_items)
    mean_rating = R.groupby("ii").r.mean().loc[cat].to_numpy()
    pop = counts.loc[cat].to_numpy()
    junk = mean_rating <= np.quantile(mean_rating, a.junk_frac)

    meta = M.set_index("i").loc[iids[cat]]
    G = np.array([[g in s.split("|") for g in GENRES] for s in meta.genres], dtype=np.float32)
    out = ROOT / "data/processed/ml1m_sim.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, r_hat=r_hat, genres=G, junk=junk, pop=pop, mean_rating=mean_rating,
                        titles=meta.title.to_numpy().astype(str), movie_ids=iids[cat])
    print(f"ALS train RMSE {rmse:.3f}; catalogue {len(cat)} items, {junk.sum()} junk "
          f"(mean rating <= {np.quantile(mean_rating, a.junk_frac):.2f}); users {len(uids)}; saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_items", type=int, default=500)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--lam", type=float, default=0.05)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--junk_frac", type=float, default=0.25)
    main(ap.parse_args())
