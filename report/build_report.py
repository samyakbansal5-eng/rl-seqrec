"""Fill the result tables into report/midsem_report.md and build report/midsem_report.pdf.

python report/build_report.py      (needs pandoc and Chromium; set CHROME to the browser binary)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_sim_m1  # noqa: E402

SHOW = ["popular", "rating", "supervised", "sqn", "cql", "mnlq", "dqn", "myopic", "quality"]


def s8_runs():
    by_s = {}
    for f in sorted((ROOT / "results/sim_m1").glob("*.json")):
        d = json.load(open(f))
        by_s.setdefault(d["_strength"], []).append(d)
    return by_s


def s8_table(by_s):
    """Clicks per session (mean ± std over seeds) for every agent and strength, then CTR / junk at the extremes."""
    S = sorted(by_s)
    m = lambda a, s, k: np.array([r[a][k] for r in by_s[s]])
    rows = ["**Clicks per session** (long-term metric; ± std over %d seeds)\n" % len(by_s[S[0]]),
            "| agent | " + " | ".join(f"s = {s}" for s in S) + " |", "|---" * (len(S) + 1) + "|"]
    for a in SHOW:
        rows.append(f"| {run_sim_m1.LABEL[a]} | " + " | ".join(f"{m(a, s, 'return').mean():.2f} ± {m(a, s, 'return').std():.2f}" for s in S) + " |")
    lo, hi = S[0], S[-1]
    rows += ["", f"**First-slate CTR** (short-term metric) **and junk clicks per session**, at s = {lo} and s = {hi}\n",
             f"| agent | CTR, s = {lo} | CTR, s = {hi} | junk clicks, s = {lo} | junk clicks, s = {hi} | session length, s = {hi} |",
             "|---|---|---|---|---|---|"]
    for a in SHOW:
        rows.append(f"| {run_sim_m1.LABEL[a]} | {m(a, lo, 'ctr_first').mean():.3f} | {m(a, hi, 'ctr_first').mean():.3f} | "
                    f"{m(a, lo, 'junk_clicks').mean():.2f} | {m(a, hi, 'junk_clicks').mean():.2f} | {m(a, hi, 'length').mean():.1f} |")
    return "\n".join(rows)


def s8_figure(by_s):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = sorted(by_s)
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    style = {"supervised": ("#c0392b", "o", "Supervised (CE)"), "mnlq": ("#2a6fdb", "s", "Offline RL (MNL-Q)"),
             "dqn": ("#8e44ad", "^", "Online RL (DQN, top-20)"), "rating": ("#7f8c8d", "D", "Est.-rating heuristic"),
             "quality": ("#27ae60", "", "Oracle quality"), "myopic": ("#e08a1e", "", "Oracle myopic")}
    for a, (c, m, lab) in style.items():
        y = [np.mean([r[a]["return"] for r in by_s[s]]) for s in S]
        e = [np.std([r[a]["return"] for r in by_s[s]]) for s in S]
        ls = "--" if a in ("quality", "myopic") else "-"
        ax.errorbar(S, y, yerr=e, color=c, marker=m or None, ls=ls, label=lab, capsize=3, lw=1.6)
    ax.set_xlabel("long-term effect strength s")
    ax.set_ylabel("clicks per session")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, ncol=2)
    fig.tight_layout()
    fig.savefig(ROOT / "results/fig_sim_m1.png", dpi=180)


if __name__ == "__main__":
    by_s = s8_runs()
    s8_figure(by_s)
    src = (ROOT / "report/midsem_report.md").read_text()
    s5 = (ROOT / "results/synthetic_table.md").read_text().strip()
    s4 = (ROOT / "results/nip_table.md").read_text().strip() if any((ROOT / "results/nip").glob("*.json")) else \
        (ROOT / "results/nip_partial_cpu.md").read_text().strip()
    filled = src.replace("{{S8_TABLE}}", s8_table(by_s)).replace("{{S5_TABLE}}", s5).replace("{{S4_TABLE}}", s4)
    out = ROOT / "report/build"
    out.mkdir(exist_ok=True)
    (out / "midsem_report.md").write_text(filled.replace("](../results/", "](" + str(ROOT / "results") + "/"))
    html = out / "midsem_report.html"
    subprocess.run(["pandoc", str(out / "midsem_report.md"), "-s", "--mathml", "--toc", "--toc-depth=1",
                    "-c", str(ROOT / "report/report.css"), "--embed-resources", "-o", str(html)], check=True)
    chrome = os.environ.get("CHROME", "/opt/pw-browsers/chromium")
    subprocess.run([chrome, "--headless", "--no-sandbox", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={ROOT / 'report/midsem_report.pdf'}", html.as_uri()], check=True, capture_output=True, timeout=300)
    print("wrote", ROOT / "report/midsem_report.pdf")
