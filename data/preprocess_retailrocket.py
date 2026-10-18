"""Preprocess RetailRocket following SQN (Xin et al. 2020) and silva24b.

- views are clicks, add-to-cart events are purchases (is_buy = 1); transaction rows are dropped
- the visitor id is the session id (as in the SQN preprocessing)
- sessions and items with <= 2 interactions are removed
- time-based split (silva24b, Hidasi & Czapp 2023): sessions are ordered by their first
  timestamp; first 80% train, next 10% validation, last 10% test

Output: data/processed/retailrocket/{train,val,test}.pkl, each a list of
(item_ids, is_buy) numpy arrays, plus stats.json.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main(raw, out):
    df = pd.read_csv(raw)
    df.columns = ["timestamp", "session_id", "behavior", "item_id", "transid"]
    df = df[df["transid"].isnull()].drop(columns="transid")
    df = df[df["behavior"].isin(["view", "addtocart"])]

    sess_size = df.groupby("session_id")["item_id"].transform("size")
    df = df[sess_size > 2]
    item_size = df.groupby("item_id")["session_id"].transform("size")
    df = df[item_size > 2]

    # item ids 1..N, 0 is padding
    items = np.sort(df["item_id"].unique())
    item_map = {it: i + 1 for i, it in enumerate(items)}
    df["item"] = df["item_id"].map(item_map)
    df["is_buy"] = (df["behavior"] == "addtocart").astype(np.int8)
    df = df.sort_values(["session_id", "timestamp"], kind="stable")

    starts = df.groupby("session_id")["timestamp"].min().sort_values(kind="stable")
    order = starts.index.to_numpy()
    n = len(order)
    cut1, cut2 = int(0.8 * n), int(0.9 * n)
    split_of = {}
    for i, s in enumerate(order):
        split_of[s] = "train" if i < cut1 else ("val" if i < cut2 else "test")

    seqs = {"train": [], "val": [], "test": []}
    for sid, g in df.groupby("session_id", sort=False):
        seqs[split_of[sid]].append((g["item"].to_numpy(np.int32), g["is_buy"].to_numpy(np.int8)))

    out.mkdir(parents=True, exist_ok=True)
    for k, v in seqs.items():
        with open(out / f"{k}.pkl", "wb") as f:
            pickle.dump(v, f)
    stats = {
        "sessions": int(n),
        "events": int(len(df)),
        "items": int(len(items)),
        "buys": int(df["is_buy"].sum()),
        "clicks": int(len(df) - df["is_buy"].sum()),
        "train_sessions": len(seqs["train"]),
        "val_sessions": len(seqs["val"]),
        "test_sessions": len(seqs["test"]),
        "avg_session_len": float(len(df) / n),
    }
    with open(out / "stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=ROOT / "data/raw/retailrocket/events.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "data/processed/retailrocket")
    a = ap.parse_args()
    main(a.raw, a.out)
