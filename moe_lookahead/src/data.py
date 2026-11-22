"""Batch construction shared by train / eval / tests.

Sequence i: inputs stream[i*1024:(i+1)*1024], targets stream[i*1024+1:(i+1)*1024+1].
Teacher for the input token at stream index j, with g = j // 64:
  r3 (future teacher): cluster[g]
  r2 (past teacher):   cluster[g-1], or cluster[0] when g == 0
"""
import numpy as np
import torch

from common import BLOCK, BLOCKS_PER_SEQ, SEQ_LEN

_AR = np.arange(SEQ_LEN + 1, dtype=np.int64)


def seq_batch(tokens, seq_ids):
    seq_ids = np.asarray(seq_ids, dtype=np.int64)
    idx = seq_ids[:, None] * SEQ_LEN + _AR[None, :]
    arr = np.asarray(tokens[idx], dtype=np.int64)
    return arr[:, :-1], arr[:, 1:]


def teacher_ids(clusters, seq_ids, mode):
    """[B, 1024] teacher expert ids for r2/r3 (None for other modes)."""
    if mode not in ("r2", "r3"):
        return None
    seq_ids = np.asarray(seq_ids, dtype=np.int64)
    g = seq_ids[:, None] * BLOCKS_PER_SEQ + np.arange(BLOCKS_PER_SEQ, dtype=np.int64)[None, :]
    if mode == "r2":
        g = np.maximum(g - 1, 0)
    t = np.asarray(clusters[g], dtype=np.int64)
    return np.repeat(t, BLOCK, axis=1)


def train_order(n_seq, data_seed=0):
    """Fixed permutation of training sequences; identical for every run."""
    return np.random.default_rng(int(data_seed)).permutation(n_seq)


def micro_seq_ids(perm, step, micro, grad_accum, micro_bs):
    k = (int(step) * grad_accum + int(micro)) * micro_bs
    return perm[(k + np.arange(micro_bs)) % len(perm)]


def coin(data_seed, step, micro, shape):
    """Deterministic per-token coin, seeded by (data seed, step, micro-step); shared across layers."""
    seed = (int(data_seed) << 40) | (int(step) << 8) | int(micro)
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.rand(shape, generator=g)


def to_device(arr, device):
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if device.type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)
