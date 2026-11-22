"""Specialization metrics on routing decisions (pure numpy; unit-tested in tests.py).

Conventions
  ids    [N, k] expert ids of one router pool at one layer (k = 2 for r1f: both slots count as samples)
  topic  [N] topic id of the routed (input) token's document; fmt [N] format id; tok [N] input token id
  All information quantities are in bits.
"""
import numpy as np

from common import conditional_mi_bits, entropy_bits, joint_counts, mi_from_counts


def pairs(ids, *labels):
    """Flatten (token, slot) pairs: expert ids + labels repeated per slot."""
    ids = np.asarray(ids)
    if ids.ndim == 1:
        ids = ids[:, None]
    k = ids.shape[1]
    return (ids.reshape(-1).astype(np.int64),) + tuple(np.repeat(np.asarray(l, dtype=np.int64), k) for l in labels)


def topic_info(ids, topic, n_exp, n_topics):
    e, t = pairs(ids, topic)
    j = joint_counts(e, t, n_exp, n_topics)
    mi = mi_from_counts(j)
    h = entropy_bits(j.sum(0))
    return dict(mi=mi, h_topic=h, nmi=mi / h if h > 0 else 0.0, n=int(len(e)))


def common_word(ids, topic, tok, n_exp, n_topics, top_ids, min_count=2000):
    """I(E;T) restricted to tokens whose id is in top_ids, and I(E;T|W) averaged over token ids with
    >= min_count occurrences (frequency weighted; occurrences counted in tokens, not slots)."""
    tok = np.asarray(tok, dtype=np.int64)
    mask = np.isin(tok, np.asarray(top_ids, dtype=np.int64))
    ids = np.asarray(ids)
    ids2 = ids if ids.ndim == 2 else ids[:, None]
    res = dict(top_ids_share=float(mask.mean()) if len(mask) else 0.0)
    r = topic_info(ids2[mask], np.asarray(topic)[mask], n_exp, n_topics)
    res.update(mi_top=r["mi"], h_topic_top=r["h_topic"], nmi_top=r["nmi"], n_top=r["n"])
    uniq, inv, cnt = np.unique(tok, return_inverse=True, return_counts=True)
    keep = cnt >= min_count
    sel = keep[inv]
    if sel.sum() == 0:
        res.update(cmi=0.0, h_topic_given_word=0.0, ncmi=0.0, n_words=0, cmi_token_share=0.0)
        return res
    remap = np.full(len(uniq), -1, dtype=np.int64)
    remap[keep] = np.arange(keep.sum())
    w = remap[inv[sel]]
    e, t, wp = pairs(ids2[sel], np.asarray(topic)[sel], w)
    cmi, hb, nw, _ = conditional_mi_bits(e, t, wp, n_exp, n_topics, int(keep.sum()))
    res.update(cmi=cmi, h_topic_given_word=hb, ncmi=cmi / hb if hb > 0 else 0.0, n_words=nw,
               cmi_token_share=float(sel.mean()))
    return res


def format_control(ids, topic, fmt, n_exp, n_topics, n_formats):
    e, t, f = pairs(ids, topic, fmt)
    j = joint_counts(e, f, n_exp, n_formats)
    mi_f = mi_from_counts(j)
    hf = entropy_bits(j.sum(0))
    cmi, ht, _, _ = conditional_mi_bits(e, t, f, n_exp, n_topics, n_formats)
    return dict(mi_format=mi_f, nmi_format=mi_f / hf if hf > 0 else 0.0, cmi_topic_given_format=cmi,
                ncmi_topic_given_format=cmi / ht if ht > 0 else 0.0)


def purity(ids, topic, n_exp, n_topics):
    e, t = pairs(ids, topic)
    j = joint_counts(e, t, n_exp, n_topics).astype(np.float64)
    load = j.sum(1)
    base = j.sum(0) / max(j.sum(), 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        share = j / np.maximum(load[:, None], 1)
        srt = -np.sort(-share, axis=1)
        lift = np.where(base[None, :] > 0, share / base[None, :], 0.0)
        tshare = j / np.maximum(j.sum(0, keepdims=True), 1)
    top = share.argmax(1)
    lw = load / max(load.sum(), 1)
    return dict(
        top1_share=srt[:, 0].tolist(), top3_share=srt[:, :3].sum(1).tolist(), top_topic=top.tolist(),
        top_lift=lift[np.arange(n_exp), top].tolist(), lift=lift.tolist(), load=lw.tolist(),
        topic_max_expert_share=tshare.max(0).tolist(),
        mean_top1_share=float((srt[:, 0] * lw).sum()), mean_top3_share=float((srt[:, :3].sum(1) * lw).sum()),
        mean_top_lift=float((lift[np.arange(n_exp), top] * lw).sum()),
        mean_topic_max_expert_share=float(np.nanmean(np.where(j.sum(0) > 0, tshare.max(0), np.nan))),
    )


def consistency(ids, same_doc_prev):
    """P(slot-0 expert at t == slot-0 expert at t-1) over consecutive tokens of the same document.
    same_doc_prev[i] is True when token i-1 (same sequence) belongs to the same document as token i."""
    ids = np.asarray(ids)
    e0 = ids[:, 0] if ids.ndim == 2 else ids
    m = np.asarray(same_doc_prev, dtype=bool)
    m[0] = False
    prev = np.roll(e0, 1)
    same = (e0 == prev)[m]
    load = np.bincount(e0.astype(np.int64)) / max(len(e0), 1)
    return dict(p_same=float(same.mean()) if len(same) else 0.0, chance=float((load ** 2).sum()), n=int(m.sum()))


def health(ids, n_exp, dead_threshold=0.01):
    e = pairs(ids)[0]
    load = np.bincount(e, minlength=n_exp) / max(len(e), 1)
    return dict(load=load.tolist(), max_load=float(load.max()), dead=int((load < dead_threshold).sum()))


def shuffled_topic(doc_topic, doc_of_token, seed=0):
    """Topic labels permuted across documents (not tokens): the chance level for topic information."""
    perm = np.random.default_rng(seed).permutation(len(doc_topic))
    return np.asarray(doc_topic)[perm][doc_of_token]


def selectivity(delta):
    """delta [E, T]: loss increase per topic when expert e is disabled. Returns per-expert selectivity
    (largest positive increase / sum of positive increases; NaN if none positive) and most-hurt topic."""
    d = np.asarray(delta, dtype=np.float64)
    pos = np.clip(d, 0, None)
    tot = pos.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sel = np.where(tot > 0, pos.max(1) / tot, np.nan)
    return sel, d.argmax(1)
