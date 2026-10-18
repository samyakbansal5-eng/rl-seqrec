"""Learning agents for the MovieLens user simulator (step S8, slate size K = 1 first).

All agents see only what a real recommender sees: user id, the last clicked items, the step index.

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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class StateEncoder(nn.Module):
    def __init__(self, n_users, n_items, dim=32, hidden=64, H=30):
        super().__init__()
        self.ue = nn.Embedding(n_users, dim)
        self.ie = nn.Embedding(n_items + 1, dim, padding_idx=0)  # item i -> i + 1, 0 = padding
        self.te = nn.Embedding(H + 1, 8)
        self.mlp = nn.Sequential(nn.Linear(2 * dim + 8 + 1, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU())

    def forward(self, user, hist, t):
        h = self.ie(hist + 1)  # hist uses -1 for empty
        n = (hist >= 0).sum(1, keepdim=True).float()
        pooled = h.sum(1) / n.clamp(min=1)
        x = torch.cat([self.ue(user), pooled, self.te(t.clamp(max=self.te.num_embeddings - 1)), n / hist.size(1)], 1)
        return self.mlp(x)


class RecNet(nn.Module):
    def __init__(self, n_users, n_items, heads=("choice",), **kw):
        super().__init__()
        self.enc = StateEncoder(n_users, n_items, **kw)
        hid = self.enc.mlp[-2].out_features
        self.heads = nn.ModuleDict({h: nn.Linear(hid, n_items) for h in heads})

    def forward(self, user, hist, t):
        z = self.enc(user, hist, t)
        return {k: h(z) for k, h in self.heads.items()}


def to_t(obs):
    t = obs["t"]
    t = t if isinstance(t, np.ndarray) else np.full(len(obs["user"]), t)
    return (torch.as_tensor(obs["user"]).long(), torch.as_tensor(obs["history"]).long(), torch.as_tensor(t).long())


def masked_topk(scores, hist, K):
    """Top-K items, never re-showing items already clicked in the session."""
    s = scores.clone()
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


# ---------------------------------------------------------------- training
def train_offline(sim, logs, kind, epochs=8, batch=512, lr=3e-3, gamma=0.9, cql_alpha=1.0, seed=0, log=None):
    """kind in {'supervised', 'sqn', 'cql'}. Trains on logged transitions; K = 1 for Q-learning kinds."""
    torch.manual_seed(seed)
    heads = {"supervised": ("choice",), "sqn": ("choice", "q"), "cql": ("q",)}[kind]
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
            if "q" in out:
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


def train_dqn(sim, users, steps=4000, B=256, gamma=0.9, lr=1e-3, eps0=1.0, eps1=0.05, buffer=200_000,
              batch=512, updates_per_step=2, seed=0, log=None):
    """Online double DQN in the batched simulator (K = 1). One step = one recommendation to each of B users."""
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
        with torch.no_grad():
            a = masked_topk(net(u, h, t)["q"], h, 1)[:, 0].numpy()
        rand = rng.random(B) < eps
        a = np.where(rand, rng.integers(0, sim.n_items, B), a)
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
                o2 = net(b["user2"].long(), b["hist2"].long(), b["t2"].long())["q"]
                a2 = masked_topk(o2, b["hist2"].long(), 1)[:, 0]
                v2 = tgt(b["user2"].long(), b["hist2"].long(), b["t2"].long())["q"].gather(1, a2[:, None]).squeeze(1)
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


def net_policy(net, head):
    """Wrap a trained network as a simulator policy (greedy top-K on `head`)."""
    @torch.no_grad()
    def policy(sim, obs):
        u, h, t = to_t(obs)
        return masked_topk(net(u, h, t)[head], h, sim.cfg.K).numpy()
    return policy
