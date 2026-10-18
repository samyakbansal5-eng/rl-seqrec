"""Learning agents for the MovieLens user simulator (step S8, slate size K = 1 first).

All agents see only what a real recommender sees: user id, the last clicked items, the step index, and
the user/item factors of a rating model fitted on half of each user's past ratings (data/history_features.py),
which initialise the user and item embeddings (warm start).

  Supervised (CE)   next-click model trained on logged data with the MNL choice log-likelihood
                    (outside option = no click); recommends top-K scores. This is the short-term
                    recommender of the topic brief (Lemma 1: it targets the myopic policy).
  SQN-style         same model plus a Q head trained by TD on the logs (RL as an auxiliary loss,
                    Xin et al. 2020 / Labarca et al. 2024); still recommends from the CE head.
  CQL (offline RL)  Q-network trained on the same logs with double-DQN targets plus the
                    conservative penalty of Kumar et al. 2020; recommends argmax Q.
  DQN (online RL)   Q-network trained by interacting with the simulator (epsilon-greedy, replay,
                    double DQN, target network); recommends argmax Q.
"""
import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


HIST_FEATS = Path(__file__).resolve().parents[2] / "data/processed/ml1m_hist_feats.npz"


def load_hist_feats(path=HIST_FEATS):
    """(user_feat, item_feat) from the history rating model, or None when the file is missing."""
    if not Path(path).exists():
        return None
    d = np.load(path)
    return torch.as_tensor(d["user_feat"]), torch.as_tensor(d["item_feat"])


class StateEncoder(nn.Module):
    def __init__(self, n_users, n_items, dim=32, hidden=64, H=30, feats=None):
        super().__init__()
        self.ue = nn.Embedding(n_users, dim)
        self.ie = nn.Embedding(n_items + 1, dim, padding_idx=0)  # item i -> i + 1, 0 = padding
        if feats is not None:
            with torch.no_grad():
                self.ue.weight.copy_(feats[0])
                self.ie.weight[1:].copy_(feats[1])
        self.te = nn.Embedding(H + 1, 8)
        self.mlp = nn.Sequential(nn.Linear(2 * dim + 8 + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())

    def forward(self, user, hist, t):
        h = self.ie(hist + 1)  # hist uses -1 for empty
        n = (hist >= 0).sum(1, keepdim=True).float()
        pooled = h.sum(1) / n.clamp(min=1)
        x = torch.cat([self.ue(user), pooled, self.te(t.clamp(max=self.te.num_embeddings - 1)), n / hist.size(1)], 1)
        return self.mlp(x)


class ItemHead(nn.Module):
    """score(s, i) = <W z_s + w_u e_u, e_i> + b_i: item scores factorise over the item embeddings."""

    def __init__(self, hid, dim):
        super().__init__()
        self.proj = nn.Linear(hid, dim)
        self.user_gain = nn.Parameter(torch.ones(1))

    def forward(self, z, e_u, E, b):
        return (self.proj(z) + self.user_gain * e_u) @ E.T + b


class RecNet(nn.Module):
    """feats="history" (default) warm-starts the embeddings from the history rating model."""

    def __init__(self, n_users, n_items, heads=("choice",), feats="history", **kw):
        super().__init__()
        feats = load_hist_feats() if feats == "history" else feats
        self.enc = StateEncoder(n_users, n_items, feats=feats, **kw)
        hid, dim = self.enc.mlp[-2].out_features, self.enc.ie.embedding_dim
        item_heads = [h for h in heads if h != "n"]
        self.heads = nn.ModuleDict({h: ItemHead(hid, dim) for h in item_heads})
        self.bias = nn.ParameterDict({h: nn.Parameter(torch.zeros(n_items)) for h in item_heads})
        self.n_head = nn.Linear(hid, 1) if "n" in heads else None  # MNL-Q: value of a no-click step

    def forward(self, user, hist, t):
        z = self.enc(user, hist, t)
        e_u, E = self.enc.ue(user), self.enc.ie.weight[1:]
        out = {k: h(z, e_u, E, self.bias[k]) for k, h in self.heads.items()}
        if self.n_head is not None:
            out["n"] = self.n_head(z).squeeze(1)
        return out


def to_t(obs):
    t = obs["t"]
    t = t if isinstance(t, np.ndarray) else np.full(len(obs["user"]), t)
    return (torch.as_tensor(obs["user"]).long(), torch.as_tensor(obs["history"]).long(), torch.as_tensor(t).long())


def masked_topk(scores, hist, K, allowed=None):
    """Top-K items, never re-showing items already clicked in the session (and only `allowed` ones if given)."""
    s = scores.clone() if allowed is None else scores.masked_fill(~allowed, -float("inf"))
    rows = torch.arange(len(s))[:, None].expand_as(hist)
    m = hist >= 0
    s[rows[m], hist[m]] = -float("inf")
    return s.topk(K, 1).indices


# ---------------------------------------------------------------- logged data
def collect_logs(sim, users, sessions_per_user=1, eps=0.3, temp=1.0, seed=0):
    """Behaviour policy: softmax over (negative) popularity rank with prob 1 - eps, uniform otherwise.
    Returns transition arrays for slates of size sim.cfg.K."""
    rng = np.random.default_rng(seed)
    K = sim.cfg.K
    buf = {k: [] for k in ["user", "hist", "t", "slate", "choice", "reward", "user2", "hist2", "t2", "done"]}
    pop_logit = -np.arange(sim.n_items) / (temp * 50.0)
    for _ in range(sessions_per_user):
        obs = sim.reset(users=users)
        while obs["active"].any():
            act = obs["active"]
            B = len(sim.u)
            p = np.exp(pop_logit - pop_logit.max())
            p /= p.sum()
            slates = np.stack([rng.choice(sim.n_items, K, replace=False, p=p) if rng.random() > eps
                               else rng.choice(sim.n_items, K, replace=False) for _ in range(B)])
            t0 = np.full(B, obs["t"])
            obs2, r, done, info = sim.step(slates)
            clicked = info["clicked"]
            choice = np.where(clicked >= 0, (slates == clicked[:, None]).argmax(1) + 1, 0)  # 0 = no click
            for k, v in (("user", obs["user"]), ("hist", obs["history"]), ("t", t0), ("slate", slates),
                         ("choice", choice), ("reward", r), ("user2", obs2["user"]), ("hist2", obs2["history"]),
                         ("t2", t0 + 1), ("done", done)):
                buf[k].append(v[act])
            obs = obs2
    return {k: torch.as_tensor(np.concatenate(v)) for k, v in buf.items()}


def choice_nll(util, slate, choice):
    """MNL log-loss: util (B, n_items), slate (B, K), choice in 0..K (0 = outside option, utility 0)."""
    u = torch.cat([torch.zeros(len(util), 1), util.gather(1, slate)], 1)
    return F.cross_entropy(u, choice)


# ---------------------------------------------------------------- MNL-Q (Lee & Oh structure, K = 1)
def mnlq_scores(out):
    """Score of showing item a:  Q(s, a) = p(a|s) Qc(s, a) + (1 - p(a|s)) N(s), where p is the MNL click
    probability of a shown alone, Qc the return given a click on a (click included) and N the return
    given no click. Returns Q(s, .) for every item."""
    p = torch.sigmoid(out["choice"])  # K = 1: P(click a | {a}) = e^v / (1 + e^v)
    return p * out["q"] + (1 - p) * out["n"][:, None]


def mnlq_value(out, hist):
    q = mnlq_scores(out)
    return q.gather(1, masked_topk(q, hist, 1)).squeeze(1)


# ---------------------------------------------------------------- training
def train_offline(sim, logs, kind, epochs=8, batch=512, lr=3e-3, gamma=0.9, cql_alpha=1.0, seed=0, log=None):
    """kind in {'supervised', 'sqn', 'cql'}. Trains on logged transitions; K = 1 for Q-learning kinds."""
    torch.manual_seed(seed)
    heads = {"supervised": ("choice",), "sqn": ("choice", "q"), "cql": ("q",), "mnlq": ("choice", "q", "n")}[kind]
    net = RecNet(sim.n_users, sim.n_items, heads, H=sim.cfg.H)
    tgt = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    n = len(logs["reward"])
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            b = {k: v[perm[i:i + batch]] for k, v in logs.items()}
            out = net(b["user"].long(), b["hist"].long(), b["t"].long())
            loss = 0.0
            if "choice" in out:
                loss = loss + choice_nll(out["choice"], b["slate"].long(), b["choice"].long())
            if kind == "mnlq":
                a = b["slate"][:, 0].long()
                click = (b["choice"] > 0).float()
                with torch.no_grad():
                    s2 = (b["user2"].long(), b["hist2"].long(), b["t2"].long())
                    y = b["reward"].float() + gamma * (1 - b["done"].float()) * mnlq_value(tgt(*s2), s2[1])
                qc = out["q"].gather(1, a[:, None]).squeeze(1)
                loss = loss + (click * (qc - y) ** 2).sum() / click.sum().clamp(min=1) \
                    + ((1 - click) * (out["n"] - y) ** 2).sum() / (1 - click).sum().clamp(min=1)
            elif "q" in out:
                a = b["slate"][:, 0].long()  # K = 1: the shown item is the action
                q_sa = out["q"].gather(1, a[:, None]).squeeze(1)
                with torch.no_grad():
                    o2 = net(b["user2"].long(), b["hist2"].long(), b["t2"].long())["q"]
                    a2 = masked_topk(o2, b["hist2"].long(), 1)[:, 0]
                    v2 = tgt(b["user2"].long(), b["hist2"].long(), b["t2"].long())["q"].gather(1, a2[:, None]).squeeze(1)
                    y = b["reward"].float() + gamma * (1 - b["done"].float()) * v2
                loss = loss + F.mse_loss(q_sa, y)
                if kind == "cql":
                    loss = loss + cql_alpha * (torch.logsumexp(out["q"], 1) - q_sa).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(0.99).add_(p, alpha=0.01)
            tot += loss.item()
        if log:
            log(f"{kind} epoch {ep}: loss {tot / (n // batch + 1):.4f}")
    return net


class Candidates:
    """Candidate generation by a trained supervised model: its top-n items for the state (two-stage
    recommender; the RL agent then re-ranks the candidates). Keeps exploration and the max in the
    Q-target inside a small, sensible set instead of the whole catalogue."""

    def __init__(self, net, n=20):
        self.net, self.n = net.eval(), n

    @torch.no_grad()
    def __call__(self, u, h, t):
        top = masked_topk(self.net(u, h, t)["choice"], h, self.n)
        return torch.zeros(len(u), self.net.bias["choice"].numel(), dtype=torch.bool).scatter_(1, top, True)


def train_dqn(sim, users, steps=4000, B=256, gamma=0.9, lr=1e-3, eps0=1.0, eps1=0.05, buffer=200_000,
              batch=512, updates_per_step=2, seed=0, log=None, candidates=None):
    """Online double DQN in the batched simulator (K = 1). One step = one recommendation to each of B users.
    candidates: optional Candidates; actions (random or greedy) and target maxima are restricted to them."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = RecNet(sim.n_users, sim.n_items, ("q",), H=sim.cfg.H)
    tgt = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    hl = sim.cfg.hist_len
    R = {"user": np.zeros(buffer, np.int64), "hist": np.zeros((buffer, hl), np.int64), "t": np.zeros(buffer, np.int64),
         "a": np.zeros(buffer, np.int64), "reward": np.zeros(buffer, np.float32), "user2": np.zeros(buffer, np.int64),
         "hist2": np.zeros((buffer, hl), np.int64), "t2": np.zeros(buffer, np.int64), "done": np.zeros(buffer, np.float32)}
    ptr, n, obs = 0, 0, sim.reset(users=rng.choice(users, B))
    for step in range(steps):
        eps = max(eps1, eps0 + (eps1 - eps0) * step / (0.6 * steps))
        u, h, t = to_t(obs)
        allowed = candidates(u, h, t) if candidates else None
        with torch.no_grad():
            a = masked_topk(net(u, h, t)["q"], h, 1, allowed)[:, 0].numpy()
        rand = rng.random(B) < eps
        if allowed is None:
            a_rand = rng.integers(0, sim.n_items, B)
        else:
            a_rand = masked_topk(torch.rand(B, sim.n_items), h, 1, allowed)[:, 0].numpy()
        a = np.where(rand, a_rand, a)
        act = obs["active"]
        obs2, r, done, _ = sim.step(a[:, None])
        new = {"user": obs["user"], "hist": obs["history"], "t": np.full(B, obs["t"]), "a": a, "reward": r,
               "user2": obs2["user"], "hist2": obs2["history"], "t2": np.full(B, obs2["t"]), "done": done}
        m = int(act.sum())
        pos = (ptr + np.arange(m)) % buffer
        for k, v in new.items():
            R[k][pos] = np.asarray(v)[act]
        ptr, n = (ptr + m) % buffer, min(n + m, buffer)
        obs = obs2 if obs2["active"].any() else sim.reset(users=rng.choice(users, B))
        if n < 2000:
            continue
        data = R
        for _ in range(updates_per_step):
            idx = rng.integers(0, n, batch)
            b = {k: torch.as_tensor(v[idx]) for k, v in data.items()}
            q = net(b["user"].long(), b["hist"].long(), b["t"].long())["q"]
            q_sa = q.gather(1, b["a"].long()[:, None]).squeeze(1)
            with torch.no_grad():
                s2 = (b["user2"].long(), b["hist2"].long(), b["t2"].long())
                o2 = net(*s2)["q"]
                a2 = masked_topk(o2, s2[1], 1, candidates(*s2) if candidates else None)[:, 0]
                v2 = tgt(*s2)["q"].gather(1, a2[:, None]).squeeze(1)
                y = b["reward"].float() + gamma * (1 - b["done"].float()) * v2
            loss = F.smooth_l1_loss(q_sa, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            with torch.no_grad():
                for p, pt in zip(net.parameters(), tgt.parameters()):
                    pt.mul_(0.995).add_(p, alpha=0.005)
        if log and step % 500 == 0:
            log(f"dqn step {step}: eps {eps:.2f} loss {loss.item():.4f} buffer {n}")
    return net


def net_policy(net, head, candidates=None):
    """Wrap a trained network as a simulator policy (greedy top-K on `head`, within `candidates` if given).
    head="mnlq" ranks by the MNL-Q score."""
    @torch.no_grad()
    def policy(sim, obs):
        u, h, t = to_t(obs)
        out = net(u, h, t)
        sc = mnlq_scores(out) if head == "mnlq" else out[head]
        return masked_topk(sc, h, sim.cfg.K, candidates(u, h, t) if candidates else None).numpy()
    return policy
