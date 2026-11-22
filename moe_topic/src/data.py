"""Batch construction shared by train / eval / tests.

Sequence i: inputs stream[i*1024 : (i+1)*1024], targets stream[i*1024+1 : (i+1)*1024+1].
Per-token labels (topic / format / confidence / document) belong to the document a token is part of;
the EOT appended after a document belongs to that document.
"""
import numpy as np
import torch

from common import SEQ_LEN

_AR = np.arange(SEQ_LEN + 1, dtype=np.int64)


def seq_positions(seq_ids, with_target=True):
    seq_ids = np.asarray(seq_ids, dtype=np.int64)
    ar = _AR if with_target else _AR[:-1]
    return seq_ids[:, None] * SEQ_LEN + ar[None, :]


def seq_batch(tokens, seq_ids):
    idx = seq_positions(seq_ids)
    arr = np.asarray(tokens[idx], dtype=np.int64)
    return arr[:, :-1], arr[:, 1:]


def topic_targets(split, seq_ids):
    """[B, 1024, 24] float32 soft topic distribution of each input token's document (P3 targets only)."""
    pos = seq_positions(seq_ids, with_target=False)
    d = split.doc_index(pos.ravel())
    return split.soft[d].astype(np.float32).reshape(pos.shape[0], pos.shape[1], -1)


def train_order(n_seq, data_seed=0):
    """Fixed permutation of training sequences; identical for every run (init_seed never touches it)."""
    return np.random.default_rng(int(data_seed)).permutation(n_seq)


def micro_seq_ids(perm, step, micro, grad_accum, micro_bs):
    k = (int(step) * grad_accum + int(micro)) * micro_bs
    return perm[(k + np.arange(micro_bs)) % len(perm)]


def to_device(arr, device):
    t = torch.from_numpy(np.ascontiguousarray(arr))
    if device.type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)
