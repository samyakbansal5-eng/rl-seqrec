"""Build the S4 and S5 result tables and figures from results/.

python scripts/summarize.py   ->  results/nip_table.md, results/synthetic_table.md, results/fig_synthetic.png
"""
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

# silva24b Table 3 (RetailRocket, purchase targets) for GRU and SASRec
SILVA = {
    ("gru", "none"): (0.1601, 0.1248, 0.2306, 0.1450), ("gru", "sqn"): (0.1921, 0.1519, 0.2698, 0.1743),
    ("gru", "eval"): (0.1962, 0.1545, 0.2718, 0.1762), ("gru", "hist"): (0.1980, 0.1549, 0.2747, 0.1770),
    ("sasrec", "none"): (0.2458, 0.1872, 0.3509, 0.2176), ("sasrec", "sqn"): (0.3012, 0.2280, 0.4227, 0.2634),
    ("sasrec", "eval"): (0.2963, 0.2242, 0.4195, 0.2600), ("sasrec", "hist"): (0.2960, 0.2231, 0.3847, 0.2573),
}
HEADS = ["none", "sqn", "eval", "hist"]
NAMES = {"gru": "GRU", "sasrec": "SAS", "none": "", "sqn": "-SQN", "eval": "-EVAL", "hist": "-HIST"}


def nip_table():
    runs = defaultdict(list)
    for f in glob.glob(str(RES / "nip/*.json")):
        d = json.load(open(f))
        runs[(d["args"]["backbone"], d["args"]["head"])].append(d)
    if not runs:
        return "No NIP runs yet.\n"
    cols = ["buy_hr@5", "buy_ndcg@5", "buy_hr@20", "buy_ndcg@20", "click_hr@20", "click_ndcg@20"]
    lines = ["| Model | seeds | epochs | " + " | ".join(c.replace("_", " ") for c in cols) + " | Silva buy HR@5 / NG@5 / HR@20 / NG@20 |",
             "|" + " --- |" * (len(cols) + 4)]
    for b in ["gru", "sasrec"]:
        for h in HEADS:
            rs = runs.get((b, h))
            if not rs:
                continue
            vals = []
            for c in cols:
                x = np.array([r["test"]["test_" + c] for r in rs])
                vals.append(f"{x.mean():.4f}" + (f" ± {x.std():.4f}" if len(x) > 1 else ""))
            ep = ",".join(str(r["best_epoch"] + 1) for r in rs)
            sil = " / ".join(f"{v:.4f}" for v in SILVA[(b, h)])
            lines.append(f"| {NAMES[b]}{NAMES[h]} | {len(rs)} | {ep} | " + " | ".join(vals) + f" | {sil} |")
    return "\n".join(lines) + "\n"


def synthetic():
    out, curves = [], {}
    for N in (10, 20, 40):
        opt = [json.load(open(f))["optimal_value"] for f in sorted(glob.glob(str(RES / f"synthetic/N{N}-optimal-s*.json")))]
        if not opt:
            continue
        row = {"Optimal": (np.mean(opt), np.std(opt))}
        for ag, name in (("myopic", "Myopic"), ("lsvi", "LSVI-UCB"), ("mnlvql", "MNL-VQL")):
            fs = sorted(glob.glob(str(RES / f"synthetic/N{N}-{ag}-s*.npy")))
            if not fs:
                continue
            R = np.stack([np.load(f) for f in fs])
            last = R[:, -2000:].mean(1)
            row[name] = (last.mean(), last.std())
            curves[(N, name)] = R
        out.append((N, row))
    lines = ["| N | " + " | ".join(["Optimal", "Myopic", "LSVI-UCB", "MNL-VQL"]) + " |", "| --- | --- | --- | --- | --- |"]
    for N, row in out:
        cell = lambda k: f"{row[k][0]:.3f} ± {row[k][1]:.3f}" if k in row else "not run"
        lines.append(f"| {N} | " + " | ".join(cell(k) for k in ["Optimal", "Myopic", "LSVI-UCB", "MNL-VQL"]) + " |")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        Ns = [N for N, _ in out]
        fig, axes = plt.subplots(1, len(Ns), figsize=(4.2 * len(Ns), 3.2), sharey=True)
        axes = np.atleast_1d(axes)
        colors = {"MNL-VQL": "#2a6fdb", "LSVI-UCB": "#e08a1e", "Myopic": "#c0392b"}
        for ax, (N, row) in zip(axes, out):
            for name, c in colors.items():
                if (N, name) not in curves:
                    continue
                R = curves[(N, name)]
                w = 500
                sm = np.stack([np.convolve(r, np.ones(w) / w, "valid") for r in R])
                x = np.arange(sm.shape[1]) + w
                ax.plot(x, sm.mean(0), color=c, label=name, lw=1.5)
                ax.fill_between(x, sm.mean(0) - sm.std(0), sm.mean(0) + sm.std(0), color=c, alpha=0.15)
            ax.axhline(row["Optimal"][0], color="k", ls="--", lw=1, label="Optimal")
            ax.set_title(f"N = {N}")
            ax.set_xlabel("Episodes")
        axes[0].set_ylabel("Episodic return (moving avg 500)")
        axes[0].legend(fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(RES / "fig_synthetic.png", dpi=150)
    except ImportError:
        pass
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    t1, t2 = nip_table(), synthetic()
    (RES / "nip_table.md").write_text(t1)
    (RES / "synthetic_table.md").write_text(t2)
    print(t1)
    print(t2)
