"""Step S7 sanity check: reference policies across long-term-effect strengths and slate sizes.

python scripts/sim_sanity.py -> results/sim_sanity.md
Shows the short-term (first-slate CTR) vs long-term (clicks per session) trade-off before any learning.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rlrec.sim.policies import POLICIES  # noqa: E402
from rlrec.sim.user_sim import SimConfig, UserSim, rollout  # noqa: E402

rows = ["| strength s | slate K | policy | first-slate CTR | clicks / session | session length | junk clicks |",
        "| --- | --- | --- | --- | --- | --- | --- |"]
for s in (0.0, 0.25, 0.5, 1.0):
    for K in (1, 5):
        for name in ("popular", "myopic", "quality"):
            m = rollout(UserSim(SimConfig(strength=s, K=K), seed=0), POLICIES[name], B=4000)
            rows.append(f"| {s} | {K} | {name} | {m['ctr_first'].mean():.3f} | {m['return'].mean():.2f} ± "
                        f"{m['return'].std() / np.sqrt(len(m['return'])):.2f} | {m['length'].mean():.1f} | "
                        f"{m['junk_clicks'].mean():.2f} |")
txt = "\n".join(rows) + "\n"
(ROOT / "results").mkdir(exist_ok=True)
(ROOT / "results/sim_sanity.md").write_text(txt)
print(txt)
