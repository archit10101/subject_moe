"""Tokenize the downloaded shards into flat GPT-2 token streams + per-token metadata + eval sets.

  python src/prep_data.py [--threads N]                 real data from data/raw (manifest from download.py)
  python src/prep_data.py --synthetic [...]             synthetic topic corpus (tests / smoke test)

Outputs (data/): for each split in train, val (natural), bal (balanced), abl (ablation subset of bal):
  {split}_tokens.bin uint16   documents in order, EOT (50256) appended after each
  {split}_topic.bin / _format.bin / _conf.bin   uint8 per token (label of the token's document; conf = 1 if
                              the topic classifier's top softmax probability >= 0.5)
  {split}_doc_offsets.npy     int64 [n_docs + 1] stream start of every document
  {split}_doc_soft.npy        float16 [n_docs, 24] softmax(topic logits) (P3 targets)
  {split}_doc_info.npy        int32 [n_docs, 5] topic, format, conf, shard id, index within the shard
  meta.json                   lengths, counts per topic/format, names, shards, fingerprint

Documents inside a raw shard are sorted by URL (measured: 99.99% of adjacent pairs ascending), so documents
are shuffled within each train shard (seed = shard id) before concatenation; eval documents are pooled
across the eval shards and shuffled with a fixed seed before the eval sets are drawn.
"""
import argparse
import hashlib
import io
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CONF_THRESHOLD, EOT, FORMAT_NAMES, N_FORMATS, N_TOPICS, SHARD_FILES, SPLITS, THE_TOKEN,  # noqa: E402
                    TOPIC_NAMES, Paths, log, n_seq, read_json, shard_name, stream_fingerprint, write_json)

EVAL_SEED = 1234


def softmax(x):
    x = np.asarray(x, dtype=np.float64)
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


# ----------------------------------------------------------------------------- real shards
def load_shard(args):
    """-> dict with concatenated tokens (EOT after each doc), per-doc lengths and labels (in shuffled order
    if shuffle) plus alignment statistics."""
    raw, sid, shuffle = args
    import tiktoken
    import zstandard

    enc = tiktoken.get_encoding("gpt2")
    nm = shard_name(sid)
    f = {k: os.path.join(raw, v.format(name=nm)) for k, v in SHARD_FILES.items()}
    neox = np.load(f["tokens"])
    topic = np.load(f["topic"])
    tlog = np.load(f["topic_logits"])
    fmt = np.load(f["format"])
    flog = np.load(f["format_logits"])
    texts = []
    with open(f["documents"], "rb") as fh:
        rd = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        for line in rd:
            if line.strip():
                texts.append(json.loads(line)["text"])
    n = len(texts)
    lens = {"documents": n, "tokens": len(neox), "topic": len(topic), "topic_logits": len(tlog),
            "format": len(fmt), "format_logits": len(flog)}
    if len(set(lens.values())) != 1:
        raise RuntimeError(f"shard {sid}: label/document counts differ: {lens}")
    if tlog.shape[1] != N_TOPICS or flog.shape[1] != N_FORMATS:
        raise RuntimeError(f"shard {sid}: unexpected logit shapes {tlog.shape} {flog.shape}")
    agree_t = float((tlog.argmax(1) == topic).mean())
    agree_f = float((flog.argmax(1) == fmt).mean())
    if agree_t < 0.999 or agree_f < 0.999:
        raise RuntimeError(f"shard {sid}: argmax(logits) agrees with __choice only {agree_t:.4f}/{agree_f:.4f}")
    toks = enc.encode_ordinary_batch(texts, num_threads=1)
    glen = np.array([len(t) for t in toks], dtype=np.int64)
    soft = softmax(tlog)
    order = np.random.default_rng(sid).permutation(n) if shuffle else np.arange(n)
    out = np.empty(int(glen.sum()) + n, dtype=np.uint16)
    p = 0
    for i in order:
        L = len(toks[i])
        out[p:p + L] = toks[i]
        out[p + L] = EOT
        p += L + 1
    return dict(sid=sid, tokens=out, lens=glen[order] + 1, topic=topic[order].astype(np.uint8),
                fmt=fmt[order].astype(np.uint8), conf=(soft.max(1) >= CONF_THRESHOLD)[order].astype(np.uint8),
                soft=soft[order].astype(np.float16), index=order.astype(np.int32),
                stats=dict(shard=sid, docs=n, gpt2_tokens=int(glen.sum()), neox_tokens=int(neox.sum()),
                           ratio=float(glen.sum() / max(neox.sum(), 1)), topic_agree=agree_t, format_agree=agree_f,
                           empty_docs=int((glen == 0).sum())))


# ----------------------------------------------------------------------------- synthetic corpus
class SynCorpus:
    """Fake documents from 24 'topics' with distinct vocabularies + shared common words (incl. ' the'),
    24 'formats' with their own marker tokens, realistic topic logits (some low-confidence documents)."""

    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        self.common = np.unique(np.concatenate([[THE_TOKEN, 11, 13, 284, 290, 286, 257, 318, 287, 262],
                                                rng.choice(np.arange(300, 5000), 190, replace=False)]))
        pool = rng.permutation(np.arange(5000, 50000))
        self.topic_vocab = [pool[i * 600:(i + 1) * 600] for i in range(N_TOPICS)]
        self.format_vocab = [pool[20000 + i * 80:20000 + (i + 1) * 80] for i in range(N_FORMATS)]
        z = 1.0 / np.arange(1, len(self.common) + 1)
        self.pc = z / z.sum()
        zt = 1.0 / np.arange(1, 601) ** 0.8
        self.pt = zt / zt.sum()
        prior = rng.dirichlet(np.ones(N_TOPICS) * 2)
        self.topic_prior = 0.5 * prior + 0.5 / N_TOPICS
        self.partner = {}

    def doc(self, rng, sid, i):
        t = int(rng.choice(N_TOPICS, p=self.topic_prior))
        f = int(rng.integers(N_FORMATS))
        L = int(np.clip(rng.lognormal(5.7, 0.8), 20, 6000))
        u = rng.random(L)
        out = np.where(u < 0.45, rng.choice(self.common, L, p=self.pc),
                       np.where(u < 0.88, rng.choice(self.topic_vocab[t], L, p=self.pt),
                                rng.choice(self.format_vocab[f], L)))
        # local structure: a topic word is often followed by a fixed partner word of the same topic
        tv = self.topic_vocab[t]
        follow = (u[:-1] >= 0.45) & (u[:-1] < 0.88) & (rng.random(L - 1) < 0.5)
        out[1:][follow] = tv[(np.searchsorted(np.sort(tv), out[:-1][follow]) * 7 + 3) % len(tv)]
        logits = rng.normal(0, 1, N_TOPICS)
        logits[t] += rng.choice([5.5, 1.0], p=[0.9, 0.1])
        return dict(tokens=np.append(out, EOT).astype(np.uint16), topic=t, fmt=f, logits=logits, sid=sid, index=i)

    def shard(self, sid, n_tokens):
        rng = np.random.default_rng(10_000 + sid)
        docs, tot, i = [], 0, 0
        while tot < n_tokens:
            d = self.doc(rng, sid, i)
            docs.append(d)
            tot += len(d["tokens"])
            i += 1
        return docs


def syn_shard_record(docs):
    soft = softmax(np.stack([d["logits"] for d in docs]))
    return dict(sid=docs[0]["sid"], tokens=np.concatenate([d["tokens"] for d in docs]),
                lens=np.array([len(d["tokens"]) for d in docs], dtype=np.int64),
                topic=np.array([d["topic"] for d in docs], dtype=np.uint8),
                fmt=np.array([d["fmt"] for d in docs], dtype=np.uint8),
                conf=(soft.max(1) >= CONF_THRESHOLD).astype(np.uint8), soft=soft.astype(np.float16),
                index=np.array([d["index"] for d in docs], dtype=np.int32),
                stats=dict(shard=docs[0]["sid"], docs=len(docs), gpt2_tokens=int(sum(len(d["tokens"]) - 1 for d in docs))))


# ----------------------------------------------------------------------------- writing
class SplitWriter:
    def __init__(self, paths, split):
        self.paths, self.split = paths, split
        self.f = {k: open(p + ".tmp", "wb") for k, p in
                  (("tokens", paths.tokens(split)), ("topic", paths.tok_meta(split, "topic")),
                   ("format", paths.tok_meta(split, "format")), ("conf", paths.tok_meta(split, "conf")))}
        self.n = 0
        self.offsets = [0]
        self.soft, self.info = [], []
        self.topic_tok = np.zeros(N_TOPICS, np.int64)
        self.topic_doc = np.zeros(N_TOPICS, np.int64)
        self.format_tok = np.zeros(N_FORMATS, np.int64)
        self.conf_tok = 0

    def add(self, rec, sel=None):
        """Append documents of a shard record (all, or the doc indices in sel, in that order)."""
        lens = rec["lens"]
        if sel is None:
            toks = rec["tokens"]
            sel = np.arange(len(lens))
        else:
            sel = np.asarray(sel, dtype=np.int64)
            starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
            toks = np.concatenate([rec["tokens"][starts[i]:starts[i] + lens[i]] for i in sel]) if len(sel) else \
                np.zeros(0, np.uint16)
        L = lens[sel]
        self.f["tokens"].write(np.ascontiguousarray(toks, dtype=np.uint16).tobytes())
        for k, src in (("topic", "topic"), ("format", "fmt"), ("conf", "conf")):
            self.f[k].write(np.repeat(rec[src][sel], L).astype(np.uint8).tobytes())
        self.offsets.extend((self.n + np.cumsum(L)).tolist())
        self.n += int(L.sum())
        self.soft.append(rec["soft"][sel])
        self.info.append(np.stack([rec["topic"][sel], rec["fmt"][sel], rec["conf"][sel],
                                   rec["sids"][sel] if "sids" in rec else np.full(len(sel), rec["sid"]),
                                   rec["index"][sel]], 1).astype(np.int32))
        self.topic_tok += np.bincount(rec["topic"][sel], weights=L, minlength=N_TOPICS).astype(np.int64)
        self.topic_doc += np.bincount(rec["topic"][sel], minlength=N_TOPICS)
        self.format_tok += np.bincount(rec["fmt"][sel], weights=L, minlength=N_FORMATS).astype(np.int64)
        self.conf_tok += int((rec["conf"][sel].astype(np.int64) * L).sum())

    def close(self):
        for k, fh in self.f.items():
            fh.close()
        p = self.paths
        for k, path in (("tokens", p.tokens(self.split)), ("topic", p.tok_meta(self.split, "topic")),
                        ("format", p.tok_meta(self.split, "format")), ("conf", p.tok_meta(self.split, "conf"))):
            os.replace(path + ".tmp", path)
        offs = np.asarray(self.offsets[:-1] + [self.n], dtype=np.int64)
        np.save(p.doc_offsets(self.split), offs)
        np.save(p.doc_soft(self.split), np.concatenate(self.soft) if self.soft else np.zeros((0, N_TOPICS), np.float16))
        np.save(p.doc_info(self.split), np.concatenate(self.info) if self.info else np.zeros((0, 5), np.int32))
        n_docs = len(offs) - 1
        return dict(n_tokens=self.n, n_seq=n_seq(self.n), n_docs=n_docs, tokens_in_windows=n_seq(self.n) * 1024,
                    topic_tokens=self.topic_tok.tolist(), topic_docs=self.topic_doc.tolist(),
                    format_tokens=self.format_tok.tolist(), high_conf_token_share=self.conf_tok / max(self.n, 1),
                    fingerprint=stream_fingerprint(p.tokens(self.split), 2 * self.n) if self.n else "empty")


def pick_budget(order, topic, lens, t, budget, used):
    """Documents of topic t in the given order, skipping ones that would overflow the token budget."""
    sel, tot = [], 0
    for i in order:
        if used[i] or topic[i] != t:
            continue
        if tot + lens[i] > budget:
            continue
        sel.append(i)
        tot += lens[i]
        if tot >= budget:
            break
    return sel, tot


def build_eval(paths, recs, args):
    """Natural val (random documents up to --nat_val_tokens), balanced val (<= --bal_tokens per topic),
    ablation subset (<= --abl_tokens per topic, drawn from the balanced set), all from the eval shards."""
    # pool the eval shards into one record
    allr = dict(sid=-1, tokens=np.concatenate([r["tokens"] for r in recs]),
                lens=np.concatenate([r["lens"] for r in recs]),
                **{k: np.concatenate([r[k] for r in recs]) for k in ("topic", "fmt", "conf", "soft", "index")})
    sids = np.concatenate([np.full(len(r["lens"]), r["sid"]) for r in recs])
    n = len(allr["lens"])
    order = np.random.default_rng(EVAL_SEED).permutation(n)
    used = np.zeros(n, bool)
    nat, tot = [], 0
    for i in order:
        if tot >= args.nat_val_tokens:
            break
        nat.append(i)
        tot += allr["lens"][i]
    used[nat] = True
    bal_by_topic, abl_by_topic = {}, {}
    for t in range(N_TOPICS):
        sel, _ = pick_budget(order, allr["topic"], allr["lens"], t, args.bal_tokens, used)
        bal_by_topic[t] = sel
    bal_set = set(i for s in bal_by_topic.values() for i in s)
    bal = [i for i in order if i in bal_set]  # balanced set in shuffled order (topics interleaved)
    for t in range(N_TOPICS):
        sel, _ = pick_budget(bal_by_topic[t], allr["topic"], allr["lens"], t, args.abl_tokens, np.zeros(n, bool))
        abl_by_topic[t] = sel
    abl_set = set(i for s in abl_by_topic.values() for i in s)
    abl = [i for i in bal if i in abl_set]
    out = {}
    for split, sel in (("val", nat), ("bal", bal), ("abl", abl)):
        w = SplitWriter(paths, split)
        w.add(dict(allr, sids=sids), sel)
        out[split] = w.close()
    low = [TOPIC_NAMES[t] for t in range(N_TOPICS) if out["bal"]["topic_docs"][t] < 200]
    out["bal"]["topics_below_200_docs"] = low
    return out


def fingerprint(meta):
    h = hashlib.sha256()
    for s in SPLITS:
        h.update(f"{s}:{meta[s]['n_tokens']}:{meta[s]['fingerprint']}:{meta[s]['n_docs']}".encode())
    h.update(json.dumps(meta.get("shards", {}), sort_keys=True).encode())
    return h.hexdigest()[:16]


def check_names(paths):
    for name, ref in (("TopicClassifier", TOPIC_NAMES), ("FormatClassifier", FORMAT_NAMES)):
        c = read_json(os.path.join(paths.raw, "classifiers", name, "config.json"))
        if c is None:
            raise SystemExit(f"{paths.raw}/classifiers/{name}/config.json missing: run setup_login.sh")
        names = [c["id2label"][str(i)] for i in range(len(c["id2label"]))]
        if names != ref:
            raise SystemExit(f"{name} id2label differs from src/common.py: {names}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--nat_val_tokens", type=float, default=10e6)
    ap.add_argument("--bal_tokens", type=float, default=400e3)
    ap.add_argument("--abl_tokens", type=float, default=50e3)
    ap.add_argument("--max_train_shards", type=int, default=0, help="testing: use only the first N train shards")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--syn_train_tokens", type=float, default=3e6)
    ap.add_argument("--syn_eval_tokens", type=float, default=0, help="default: enough for the eval budgets")
    ap.add_argument("--force", action="store_true", help="rebuild even if data/meta.json exists")
    args = ap.parse_args()
    paths = Paths(args.work, raw=args.raw).mk()
    old = read_json(paths.meta)
    if old and not args.force:
        log(f"{paths.meta} exists (fingerprint {old['fingerprint']}); use --force to rebuild")
        return
    if os.path.exists(paths.meta):
        os.remove(paths.meta)  # an interrupted rebuild must not look finished
    t0 = time.time()
    meta = dict(created=time.strftime("%Y-%m-%d %H:%M:%S"), synthetic=args.synthetic, topic_names=TOPIC_NAMES,
                format_names=FORMAT_NAMES, eot=EOT, seq_len=1024, conf_threshold=CONF_THRESHOLD,
                soft_targets="softmax of the TopicClassifier logits (domains_topics__logits)",
                settings=dict(nat_val_tokens=args.nat_val_tokens, bal_tokens=args.bal_tokens,
                              abl_tokens=args.abl_tokens, eval_seed=EVAL_SEED))
    if args.synthetic:
        syn = SynCorpus(0)
        per = 2_000_000
        nsh = max(1, int(np.ceil(args.syn_train_tokens / per)))
        w = SplitWriter(paths, "train")
        stats = []
        for s in range(nsh):
            rec = syn_shard_record(syn.shard(s, min(per, args.syn_train_tokens - s * per)))
            w.add(rec)
            stats.append(rec["stats"])
        meta["train"] = w.close()
        ev_tok = args.syn_eval_tokens or (1.15 * args.nat_val_tokens + 1.6 * N_TOPICS * args.bal_tokens / (
            syn.topic_prior.min() * N_TOPICS))
        recs = [syn_shard_record(syn.shard(1000, ev_tok))]
        meta.update(build_eval(paths, recs, args))
        meta["shards"] = dict(train=list(range(nsh)), eval=[1000])
        meta["shard_stats"] = stats + [recs[0]["stats"]]
    else:
        man = read_json(paths.manifest)
        if not man or not man.get("complete"):
            raise SystemExit(f"{paths.manifest} missing/incomplete: run ./setup_login.sh (src/download.py)")
        check_names(paths)
        tr = man["train_shards"][:args.max_train_shards or None]
        log(f"tokenizing {len(tr)} train shards with {args.threads} processes")
        w = SplitWriter(paths, "train")
        stats = []
        with Pool(args.threads) as pool:
            for k, rec in enumerate(pool.imap(load_shard, [(paths.raw, s, True) for s in tr])):
                w.add(rec)
                stats.append(rec["stats"])
                if (k + 1) % 5 == 0 or k + 1 == len(tr):
                    log(f"  train shard {k + 1}/{len(tr)}: {w.n / 1e9:.3f}B GPT-2 tokens so far "
                        f"({time.time() - t0:.0f}s)")
            meta["train"] = w.close()
            ev = man["eval_shards"]
            log(f"tokenizing eval shards {ev}")
            recs = pool.map(load_shard, [(paths.raw, s, False) for s in ev])
        for r in recs:
            stats.append(r["stats"])
        meta.update(build_eval(paths, recs, args))
        meta["shards"] = dict(train=tr, eval=ev)
        meta["shard_stats"] = stats
        ratio = sum(s["gpt2_tokens"] for s in stats) / max(sum(s["neox_tokens"] for s in stats), 1)
        meta["gpt2_over_neox"] = ratio
        log(f"GPT-2 / NeoX token ratio over all shards: {ratio:.4f}")
    meta["fingerprint"] = fingerprint(meta)
    write_json(paths.meta, meta)
    for s in SPLITS:
        m = meta[s]
        log(f"{s:5s}: {m['n_tokens'] / 1e6:10.2f}M tokens, {m['n_docs']:8d} docs, {m['n_seq']} windows, "
            f"high-conf {m['high_conf_token_share']:.3f}")
    if meta["bal"]["topics_below_200_docs"]:
        log(f"WARNING: balanced val topics with < 200 documents: {meta['bal']['topics_below_200_docs']}")
    log(f"wrote {paths.meta} (fingerprint {meta['fingerprint']}) in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
