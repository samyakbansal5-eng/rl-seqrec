"""Sequential backbones (GRU4Rec, SASRec) with the heads compared in silva24b.

Every model has the supervised action-logits head (cross-entropy). The auxiliary head is one of
  none : plain self-supervised model
  sqn  : Q-values over all items, trained with the (double) DQN loss       (Xin et al. 2020)
  eval : Q-values over all items, trained with TD on the logged next item  (silva24b, L_eval)
  hist : per-item regression of the number of past interactions          (silva24b, hist)
Recommendations always come from the action-logits head, as in SQN.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU4Rec(nn.Module):
    def __init__(self, n_items, hidden=64, state_len=10, dropout=0.0):
        super().__init__()
        self.emb = nn.Embedding(n_items + 1, hidden, padding_idx=0)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)

    def forward(self, s):
        out, h = self.gru(self.drop(self.emb(s)))
        return h[-1]


class SASRec(nn.Module):
    def __init__(self, n_items, hidden=64, state_len=10, dropout=0.1, heads=1, blocks=1):
        super().__init__()
        self.emb = nn.Embedding(n_items + 1, hidden, padding_idx=0)
        self.pos = nn.Embedding(state_len, hidden)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(hidden, heads, hidden, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(hidden)
        mask = torch.triu(torch.ones(state_len, state_len, dtype=torch.bool), 1)
        self.register_buffer("causal", mask)

    def forward(self, s):
        pos = torch.arange(s.size(1), device=s.device)
        x = self.drop(self.emb(s) * self.emb.embedding_dim ** 0.5 + self.pos(pos))
        pad = s == 0
        # an all-padding state (first event of a session) would give NaN attention; let it see itself
        pad = pad & ~(pad.all(1, keepdim=True) & (pos == s.size(1) - 1))
        x = x * (~pad).unsqueeze(-1)
        h = self.enc(x, mask=self.causal, src_key_padding_mask=pad)
        return self.norm(h[:, -1])


BACKBONES = {"gru": GRU4Rec, "sasrec": SASRec}


class SeqRecModel(nn.Module):
    def __init__(self, backbone, n_items, head="none", hidden=64, state_len=10, dropout=None):
        super().__init__()
        kw = {} if dropout is None else {"dropout": dropout}
        self.body = BACKBONES[backbone](n_items, hidden, state_len, **kw)
        self.logits = nn.Linear(hidden, n_items + 1)
        self.head = head
        self.aux = nn.Linear(hidden, n_items + 1) if head != "none" else None

    def forward(self, s):
        z = self.body(s)
        return z, self.logits(z), (self.aux(z) if self.aux is not None else None)

    def aux_at(self, z, a):
        """Auxiliary head output for item a only: same value as aux(z)[a], without the |I|-wide matmul."""
        return (z * self.aux.weight[a]).sum(1) + self.aux.bias[a]


def losses(model, target, batch, gamma=0.5, hist_clip=50.0):
    """Return (total_loss, dict of parts). `target` is the target network (sqn/eval only)."""
    s, a = batch["state"], batch["action"]
    z = model.body(s)
    logits = model.logits(z)
    logits[:, 0] = -1e9
    ce = F.cross_entropy(logits, a)
    parts = {"ce": ce.item()}
    if model.head == "none":
        return ce, parts
    q_sa = model.aux_at(z, a)
    if model.head == "hist":
        aux_loss = F.mse_loss(q_sa, batch["hist_len"].clamp(max=hist_clip))
    else:
        with torch.no_grad():
            s2, done, r = batch["next_state"], batch["done"], batch["reward"]
            if model.head == "sqn":  # double DQN: argmax from online net, value from target net
                q_next_o = model.aux(model.body(s2))
                q_next_o[:, 0] = -1e9
                a2 = q_next_o.argmax(1)
            else:  # eval: SARSA on the logged next item
                a2 = batch["next_action"]
            v_next = target.aux_at(target.body(s2), a2)
            y = r + gamma * (1 - done) * v_next
        aux_loss = F.mse_loss(q_sa, y)
    parts["aux"] = aux_loss.item()
    return ce + aux_loss, parts
