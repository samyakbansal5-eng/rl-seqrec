"""Turn session lists into fixed-length (state, action, reward, next_state, ...) transitions."""
import pickle
from pathlib import Path

import numpy as np
import torch

CLICK_REWARD, BUY_REWARD = 1.0, 5.0  # silva24b Table 5


def load_split(proc_dir, split):
    with open(Path(proc_dir) / f"{split}.pkl", "rb") as f:
        return pickle.load(f)


def build_transitions(sessions, state_len=10):
    """One transition per event.

    state      = last `state_len` items before the event, left-padded with 0
    action     = the event's item;   is_buy = its type
    next_state = state after appending the action
    next_action= the following item (for EVAL / SARSA), 0 if the session ends
    hist_len   = number of earlier interactions in the session (target of the HIST head)
    """
    n = sum(len(s[0]) for s in sessions)
    S = np.zeros((n, state_len), np.int32)
    S2 = np.zeros((n, state_len), np.int32)
    A = np.zeros(n, np.int64)
    A2 = np.zeros(n, np.int64)
    B = np.zeros(n, np.int8)
    D = np.zeros(n, np.int8)
    L = np.zeros(n, np.float32)
    i = 0
    for items, buys in sessions:
        T = len(items)
        padded = np.concatenate([np.zeros(state_len, np.int32), items])
        for t in range(T):
            S[i] = padded[t : t + state_len]
            S2[i] = padded[t + 1 : t + 1 + state_len]
            A[i] = items[t]
            A2[i] = items[t + 1] if t + 1 < T else 0
            B[i] = buys[t]
            D[i] = t == T - 1
            L[i] = t
            i += 1
    R = np.where(B == 1, BUY_REWARD, CLICK_REWARD).astype(np.float32)
    return {
        "state": torch.from_numpy(S).long(),
        "action": torch.from_numpy(A),
        "reward": torch.from_numpy(R),
        "next_state": torch.from_numpy(S2).long(),
        "next_action": torch.from_numpy(A2),
        "done": torch.from_numpy(D).float(),
        "is_buy": torch.from_numpy(B).bool(),
        "hist_len": torch.from_numpy(L),
    }
