"""Next-item-prediction metrics (HR@k, NDCG@k) with full ranking over all items,
reported separately for click and purchase targets as in SQN / silva24b."""
import numpy as np
import torch


@torch.no_grad()
def evaluate(model, data, ks=(5, 10, 20), batch_size=1024):
    model.eval()
    ranks = []
    for i in range(0, len(data["action"]), batch_size):
        s = data["state"][i : i + batch_size]
        a = data["action"][i : i + batch_size]
        _, logits, _ = model(s)
        logits[:, 0] = -float("inf")
        tgt = logits.gather(1, a[:, None])
        ranks.append((logits > tgt).sum(1))  # 0-based rank of the true item
    model.train()
    ranks = torch.cat(ranks).cpu().numpy()
    buy = data["is_buy"].cpu().numpy()
    out = {}
    for name, m in (("click", ~buy), ("buy", buy)):
        r = ranks[m]
        for k in ks:
            hit = r < k
            out[f"{name}_hr@{k}"] = float(hit.mean())
            out[f"{name}_ndcg@{k}"] = float((hit / np.log2(r + 2)).mean())
    return out
