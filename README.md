# RL for Sequential Recommendation: when does RL beat a supervised recommender on long-term utility?

Course project, topic 3.9. We compare a supervised next-click recommender (short-term, CTR) with RL recommenders
(long-term, clicks per session) inside one model: an episodic combinatorial MDP with multinomial-logit (MNL) user
choice (Lee & Oh, ICML 2025). There, the supervised recommender is the γ = 0 (myopic) policy and RL is the γ > 0
policy. The slate size K is the extension knob (K = 1 sequential recommendation, K > 1 re-ranking).

## Layout

```
data/              download.sh, preprocess_retailrocket.py, preprocess_ml1m.py, history_features.py
rlrec/
  data.py          sessions -> (state, action, reward, next_state) transitions
  models/seqrec.py GRU4Rec, SASRec + heads: CE, SQN, EVAL, HIST (Labarca et al. 2024)
  eval/            full-ranking HR@k / NDCG@k
  slate/           MNL assortment optimisation: Charnes-Cooper LP (Lee & Oh App. C) and exact bisection
  sim/             budget_env.py (Lee & Oh synthetic MDP), user_sim.py (MovieLens-1M simulator), policies.py
  agents/          combinatorial.py (Myopic, LSVI-UCB, MNL-VQL), sim_agents.py (supervised, SQN-style, CQL, MNL-Q, DQN)
scripts/           one script per experiment; summarize.py builds the tables
app/demo.py        Streamlit demo
notebooks/         colab_s4.ipynb (GPU run of the RetailRocket reproduction)
report/            midsem_report.md/.pdf (build_report.py), formulation.tex (propositions with proofs), slides, viva prep
tests/             solver, simulator tests (pytest)
results/           tables, figures, per-run JSON
```

## Reproduce

```bash
pip install -r requirements.txt
bash data/download.sh
python data/preprocess_retailrocket.py && python data/preprocess_ml1m.py && python data/history_features.py
pytest -q tests

bash scripts/run_nip_grid.sh            # S4: Labarca et al. reproduction (EPOCHS, SEEDS, PAR env vars)
bash scripts/run_synthetic_grid.sh      # S5: Lee & Oh synthetic experiment
python scripts/sim_sanity.py            # S7: simulator reference policies
python scripts/run_sim_m1.py            # S8: supervised vs RL at slate size 1
python scripts/summarize.py             # S4/S5 tables + figure
python report/build_report.py           # mid-sem report PDF (pandoc + Chromium)
python scripts/train_demo_models.py && streamlit run app/demo.py
```

## Deviations from the papers (and why)

- **Compute.** Built on a 4-core CPU without a GPU. The RetailRocket reproduction uses 1 seed and at most 5 epochs
  with early stopping on validation purchase NDCG@20 (paper: 5 seeds, 50 epochs). `train_nip.py` uses a GPU when present.
- **SQN target network.** Polyak-averaged target (τ = 0.01) with double-DQN action selection, instead of SQN's two
  alternating networks.
- **HIST target** is clipped at 50 past interactions to keep the MSE stable on very long sessions.
- **MNL-VQL** is the linear-MDP version with σ̄ = 1 (as in the paper's own experiments), one MNL parameter shared
  across steps, tuned bonus scales, and plans recomputed every 5 episodes. Assortments are solved exactly
  (bisection, verified against the LP and brute force in `tests/test_assortment.py`).
- **Simulator agents** start from user/item factors of a rating model fitted on half of each user's ratings
  (`data/history_features.py`), standing in for the platform's interaction history; the simulator's ground truth
  uses all ratings. The online DQN re-ranks the supervised model's top-20 candidates (a DQN over all 500 items
  overestimates and fails).
- **Datasets.** Kaggle/GroupLens/Zenodo were not reachable from the build environment; RetailRocket (byte-identical
  events.csv) and MovieLens-1M come from public GitHub mirrors. KuaiRec was not available, so the simulator is
  calibrated on MovieLens-1M (as Lee & Oh do).
