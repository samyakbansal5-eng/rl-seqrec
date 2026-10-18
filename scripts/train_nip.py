"""Train one backbone/head/seed on RetailRocket and report NIP metrics (steps S3-S4).

Example:  python scripts/train_nip.py --backbone sasrec --head sqn --seed 0
Model selection: best validation purchase NDCG@20 across epochs (early stopping, patience 3);
the test metrics of that epoch are saved to results/nip/<backbone>-<head>-s<seed>.json
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rlrec.data import build_transitions, load_split  # noqa: E402
from rlrec.eval.nip_metrics import evaluate  # noqa: E402
from rlrec.models.seqrec import SeqRecModel, losses  # noqa: E402


def main(a):
    torch.manual_seed(a.seed)
    torch.set_num_threads(a.threads)
    proc = ROOT / "data/processed/retailrocket"
    n_items = json.load(open(proc / "stats.json"))["items"]
    tr = build_transitions(load_split(proc, "train"), a.state_len)
    va = build_transitions(load_split(proc, "val"), a.state_len)
    te = build_transitions(load_split(proc, "test"), a.state_len)
    if a.subsample < 1.0:  # for quick smoke tests only
        keep = torch.rand(len(tr["action"])) < a.subsample
        tr = {k: v[keep] for k, v in tr.items()}

    dev = torch.device("cuda" if torch.cuda.is_available() and a.device != "cpu" else "cpu")
    tr, va, te = ({k: v.to(dev) for k, v in d.items()} for d in (tr, va, te))
    model = SeqRecModel(a.backbone, n_items, a.head, a.hidden, a.state_len).to(dev)
    target = copy.deepcopy(model).eval() if a.head in ("sqn", "eval") else None
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    n = len(tr["action"])
    best, best_ep, history, bad = None, -1, [], 0
    for ep in range(a.epochs):
        t0 = time.time()
        perm = torch.randperm(n, device=dev)
        tot = {}
        for bi, i in enumerate(range(0, n, a.batch)):
            idx = perm[i : i + a.batch]
            batch = {k: v[idx] for k, v in tr.items()}
            loss, parts = losses(model, target, batch, a.gamma)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if target is not None:
                with torch.no_grad():
                    for p, pt in zip(model.parameters(), target.parameters()):
                        pt.mul_(1 - a.tau).add_(p, alpha=a.tau)
            for k, v in parts.items():
                tot[k] = tot.get(k, 0.0) + v
        val = evaluate(model, va)
        rec = {"epoch": ep, "secs": time.time() - t0, **{k: v / (bi + 1) for k, v in tot.items()}, **{"val_" + k: v for k, v in val.items()}}
        score = val["buy_ndcg@20"]
        if best is None or score > best:
            best, best_ep, bad = score, ep, 0
            rec.update({"test_" + k: v for k, v in evaluate(model, te).items()})
            best_test = {k: v for k, v in rec.items() if k.startswith("test_")}
        else:
            bad += 1
        history.append(rec)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if "@10" not in k}), flush=True)
        if bad >= a.patience:
            break
    out = ROOT / "results/nip" / f"{a.backbone}-{a.head}-s{a.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"args": vars(a), "best_epoch": best_ep, "test": best_test, "history": history}, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="sasrec", choices=["gru", "sasrec"])
    ap.add_argument("--head", default="none", choices=["none", "sqn", "eval", "hist"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--gamma", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--state_len", type=int, default=10)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--subsample", type=float, default=1.0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"])
    main(ap.parse_args())
