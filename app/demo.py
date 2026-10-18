"""Streamlit demo v0 (step S9).

    streamlit run app/demo.py

Page "Live session": one simulated MovieLens user, two recommenders side by side, same random draws.
Page "Results": the reproduction tables and simulator results produced by scripts/.
"""
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rlrec.agents.sim_agents import RecNet, net_policy  # noqa: E402
from rlrec.sim.policies import POLICIES  # noqa: E402
from rlrec.sim.user_sim import SimConfig, UserSim  # noqa: E402

st.set_page_config(page_title="RL vs supervised recommender", layout="wide")
CKPT = ROOT / "results/checkpoints"
STRENGTHS = [0.0, 0.5, 1.0]


@st.cache_resource
def load_net(kind, strength):
    f = CKPT / f"{kind}-s{strength}.pt"
    if not f.exists():
        return None
    sim = UserSim(SimConfig(strength=strength))
    net = RecNet(sim.n_users, sim.n_items, ("choice",) if kind == "supervised" else ("choice", "q", "n"), H=sim.cfg.H)
    net.load_state_dict(torch.load(f))
    return net_policy(net.eval(), "choice" if kind == "supervised" else "mnlq")


def agent_options(strength):
    opts = {"Supervised next-click model": load_net("supervised", strength),
            "RL (MNL-Q, offline)": load_net("mnlq", strength),
            "Oracle: most clickable now (myopic)": POLICIES["myopic"],
            "Oracle: good items, no junk": POLICIES["quality"],
            "Popularity": POLICIES["popular"]}
    return {k: v for k, v in opts.items() if v is not None}


def run_session(policy, strength, user, seed):
    sim = UserSim(SimConfig(strength=strength, K=1), seed=seed)
    obs = sim.reset(users=np.array([user]))
    steps = []
    while obs["active"][0]:
        slate = policy(sim, obs)
        obs, r, done, info = sim.step(slate)
        item = int(slate[0, 0])
        steps.append({"step": len(steps) + 1, "shown": sim.titles[item], "junk": bool(sim.junk[item]),
                      "p_click": float(info["p_click"][0]), "clicked": bool(r[0] > 0),
                      "satisfaction": float(info["sigma"][0]), "left": bool(info["left"][0])})
    return steps


page = st.sidebar.radio("Page", ["Live session", "Results"])

if page == "Live session":
    st.title("Clicks now vs engagement later")
    st.caption("One simulated MovieLens user, two recommenders, identical random draws. "
               "Junk = popular but low-rated movies: easy to click, but they lower satisfaction and can make the user leave.")
    c1, c2, c3 = st.columns(3)
    strength = c1.select_slider("Long-term effect strength s", STRENGTHS, value=0.5)
    user = c2.number_input("User id (0-6039)", 0, 6039, 42)
    seed = c3.number_input("Random seed", 0, 10_000, 7)
    opts = agent_options(strength)
    names = list(opts)
    if len(names) < 5:
        st.info("Trained agents not found; run `python scripts/train_demo_models.py` to add them. Showing oracles only.")
    left, right = st.columns(2)
    for col, default in ((left, 0), (right, 1 if len(names) > 1 else 0)):
        with col:
            name = st.selectbox("Recommender", names, index=default, key=f"agent{default}")
            steps = run_session(opts[name], strength, int(user), int(seed))
            clicks = sum(s["clicked"] for s in steps)
            m1, m2, m3 = st.columns(3)
            m1.metric("Clicks in session", clicks)
            m2.metric("Session length", len(steps))
            m3.metric("First-step click prob.", f"{steps[0]['p_click']:.2f}")
            st.line_chart({"satisfaction": [s["satisfaction"] for s in steps]}, height=160)
            st.dataframe([{"step": s["step"], "shown": ("⚠ " if s["junk"] else "") + s["shown"],
                           "P(click)": round(s["p_click"], 2), "clicked": "yes" if s["clicked"] else "",
                           "user left": "LEFT" if s["left"] else ""} for s in steps],
                         hide_index=True, width="stretch")
else:
    st.title("Results so far")
    for title, f in [("Simulator: supervised vs RL, slate size 1 (S8)", "results/sim_m1_table.md"),
                     ("Simulator: reference policies (S7)", "results/sim_sanity.md"),
                     ("Labarca et al. reproduction on RetailRocket (S4)", "results/nip_table.md"),
                     ("Lee & Oh synthetic experiment (S5)", "results/synthetic_table.md")]:
        st.subheader(title)
        p = ROOT / f
        st.markdown(p.read_text() if p.exists() else "_not produced yet_")
    fig = ROOT / "results/fig_synthetic.png"
    if fig.exists():
        st.image(str(fig), caption="Episodic return on the synthetic budget MDP (Lee & Oh, Figure 1 setting)")
