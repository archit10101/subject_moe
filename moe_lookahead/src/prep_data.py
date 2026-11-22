"""Tokenize parquet shards into flat token streams (CPU job).

Outputs in <work>/data/:
  {train,val}_tokens.bin  uint16 GPT-2 ids, EOT (50256) after every document
  {train,val}_src.bin     uint8 source id per token (an EOT belongs to the doc it ends)
  meta.json               lengths, n_seq = (len-1)//1024, sources, fingerprint

Sequence i: inputs stream[i*1024:(i+1)*1024], targets shifted by one.
Block g:    stream[g*64:(g+1)*64]  (16 blocks per sequence).

--synthetic writes the same files from a fake 7-"source" phrase language (for tests).
"""
import argparse
import glob
import itertools
import os
import queue
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    BLOCK, EOT, N_SOURCES, SOURCES, Paths, log, n_blocks, n_seq, read_json,
    stream_fingerprint, write_json,
)


class StreamWriter:
    def __init__(self, paths, split):
        self.paths, self.split = paths, split
        self.tok_tmp = paths.tokens(split) + ".tmp"
        self.src_tmp = paths.src(split) + ".tmp"
        self.ft = open(self.tok_tmp, "wb")
        self.fs = open(self.src_tmp, "wb")
        self.n = 0
        self.docs = 0
        self.per_source = np.zeros(N_SOURCES, dtype=np.int64)

    def write(self, toks, srcs, ndocs):
        assert toks.dtype == np.uint16 and srcs.dtype == np.uint8 and len(toks) == len(srcs)
        self.ft.write(toks.tobytes())
        self.fs.write(srcs.tobytes())
        self.n += len(toks)
        self.docs += ndocs
        self.per_source += np.bincount(srcs, minlength=N_SOURCES)

    def close(self):
        self.ft.close()
        self.fs.close()
        os.replace(self.tok_tmp, self.paths.tokens(self.split))
        os.replace(self.src_tmp, self.paths.src(self.split))
        return dict(
            n_tokens=int(self.n), n_seq=int(n_seq(self.n)), n_blocks=int(n_blocks(self.n)),
            n_docs=int(self.docs), tokens_per_source=[int(x) for x in self.per_source],
        )


def iter_parquet_batches(files, batch_rows):
    """Yield (texts, source_ids) batches; parquet reading runs in a background thread."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    q = queue.Queue(maxsize=4)
    name_to_id = {n: i for i, n in enumerate(SOURCES)}

    def producer():
        try:
            for f in files:
                pf = pq.ParquetFile(f)
                for b in pf.iter_batches(batch_size=batch_rows, columns=["text", "meta"]):
                    texts = b.column("text").to_pylist()
                    names = pc.struct_field(b.column("meta"), "redpajama_set_name").to_pylist()
                    try:
                        sids = np.array([name_to_id[n] for n in names], dtype=np.uint8)
                    except KeyError as e:
                        raise RuntimeError(f"unknown redpajama_set_name {e} in {f}") from e
                    q.put((texts, sids))
            q.put(None)
        except BaseException as e:  # noqa: BLE001
            q.put(e)

    t = threading.Thread(target=producer, daemon=True)
    t.start()
    while True:
        item = q.get()
        if item is None:
            return
        if isinstance(item, BaseException):
            raise item
        yield item


def tokenize_split(paths, split, files, limit, threads, batch_rows):
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    w = StreamWriter(paths, split)
    t0 = time.time()
    last = t0
    for texts, sids in iter_parquet_batches(files, batch_rows):
        texts = [t if t is not None else "" for t in texts]
        ids = enc.encode_ordinary_batch(texts, num_threads=threads)
        lens = np.fromiter((len(x) for x in ids), dtype=np.int64, count=len(ids))
        flat = np.fromiter(itertools.chain.from_iterable(ids), dtype=np.uint16, count=int(lens.sum()))
        # insert EOT after each document
        toks = np.insert(flat, np.cumsum(lens), EOT).astype(np.uint16)
        srcs = np.repeat(sids, lens + 1)
        ndocs = len(ids)
        if limit is not None and w.n + len(toks) >= limit:
            keep = limit - w.n
            toks, srcs = toks[:keep], srcs[:keep]
            w.write(toks, srcs, ndocs)
            break
        w.write(toks, srcs, ndocs)
        if time.time() - last > 30:
            last = time.time()
            rate = w.n / (last - t0)
            eta = "" if limit is None else f", ETA {(limit - w.n) / max(rate, 1) / 60:.1f} min"
            log(f"  {split}: {w.n / 1e6:.0f}M tokens, {w.docs} docs, {rate / 1e6:.2f}M tok/s{eta}")
    info = w.close()
    log(f"{split}: {info['n_tokens']:,} tokens, {info['n_docs']:,} docs in {time.time() - t0:.0f}s")
    return info


# ---------------------------------------------------------------- synthetic data
def synthetic_split(paths, split, n_tokens, seed, mean_doc=700):
    """Fake language: each source has its own pool of 'phrases' (short token n-grams) drawn
    from its own sub-vocabulary; documents are Zipf-weighted phrase sequences. Learnable,
    and source-dependent, so routing and clustering have something to find."""
    g = np.random.default_rng(1234)  # phrase tables shared by train and val
    n_phr, max_len, sub_vocab = 400, 8, 4000
    tables, tlens = [], []
    for s in range(N_SOURCES):
        vocab = g.choice(50000, size=sub_vocab, replace=False)
        tables.append(vocab[g.integers(0, sub_vocab, size=(n_phr, max_len))].astype(np.uint16))
        tlens.append(g.integers(2, max_len + 1, size=n_phr))
    zipf = 1.0 / np.arange(1, n_phr + 1) ** 1.1
    zipf /= zipf.sum()
    src_w = np.array([0.52, 0.27, 0.05, 0.04, 0.04, 0.04, 0.04])

    rng = np.random.default_rng(seed)
    w = StreamWriter(paths, split)
    chunk_docs = 2000
    while w.n < n_tokens:
        dsrc = rng.choice(N_SOURCES, size=chunk_docs, p=src_w).astype(np.uint8)
        dlen = np.maximum(16, rng.geometric(1.0 / mean_doc, size=chunk_docs)).astype(np.int64)
        # global token layout of this chunk
        doc_start = np.concatenate([[0], np.cumsum(dlen)[:-1]])
        toks = np.empty(int(dlen.sum()), dtype=np.uint16)
        for s in range(N_SOURCES):
            m = np.flatnonzero(dsrc == s)
            if len(m) == 0:
                continue
            need = int(dlen[m].sum())
            ph = rng.choice(n_phr, size=need // 2 + 16, p=zipf)
            mask = np.arange(max_len)[None, :] < tlens[s][ph][:, None]
            stream = tables[s][ph][mask][:need]
            # scatter this source's stream into its documents, in order
            off_in_src = np.concatenate([[0], np.cumsum(dlen[m])[:-1]])
            rep = np.repeat(np.arange(len(m)), dlen[m])
            pos_in_doc = np.arange(need) - off_in_src[rep]
            toks[doc_start[m][rep] + pos_in_doc] = stream
        toks[doc_start + dlen - 1] = EOT
        srcs = np.repeat(dsrc, dlen)
        if w.n + len(toks) >= n_tokens:
            keep = n_tokens - w.n
            w.write(toks[:keep], srcs[:keep], chunk_docs)
            break
        w.write(toks, srcs, chunk_docs)
    info = w.close()
    log(f"synthetic {split}: {info['n_tokens']:,} tokens")
    return info


def finalize(paths, info, extra):
    fp = {}
    for split in ("train", "val"):
        fp[split] = stream_fingerprint(paths.tokens(split), info[split]["n_tokens"], 2)
    meta = dict(
        train=info["train"], val=info["val"], sources=SOURCES, block=BLOCK, seq_len=1024,
        fingerprint=f"{fp['train']}-{fp['val']}", **extra,
    )
    write_json(paths.meta, meta)
    log(f"wrote {paths.meta} (fingerprint {meta['fingerprint']})")
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    ap.add_argument("--train_tokens", type=float, default=1.2e9)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4)))
    ap.add_argument("--batch_rows", type=int, default=4096)
    ap.add_argument("--raw_dir", default=None, help="parquet folder (default <project>/data/raw)")
    ap.add_argument("--val_files", nargs="*", default=None, help="explicit validation parquet files")
    ap.add_argument("--train_files", nargs="*", default=None, help="explicit train parquet files")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--syn_val_tokens", type=float, default=2e6)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    paths = Paths(args.work).mk()

    if os.path.exists(paths.meta) and not args.force:
        log(f"{paths.meta} already exists; nothing to do (use --force to redo)")
        return
    if os.path.exists(paths.meta):
        os.remove(paths.meta)  # invalidate before rewriting the streams

    t0 = time.time()
    train_limit = int(args.train_tokens)
    if args.synthetic:
        info = dict(
            train=synthetic_split(paths, "train", train_limit, seed=1),
            val=synthetic_split(paths, "val", int(args.syn_val_tokens), seed=2),
        )
        finalize(paths, info, dict(synthetic=True, shards=[], prep_seconds=time.time() - t0))
        return

    raw = args.raw_dir or paths.raw
    man = read_json(os.path.join(raw, "manifest.json"), {}) or {}
    if args.val_files is not None:
        val_files = args.val_files
    elif man.get("validation"):
        val_files = [os.path.join(raw, f) for f in man["validation"]]
    else:
        val_files = sorted(glob.glob(os.path.join(raw, "data", "validation-*.parquet")))
    if args.train_files is not None:
        train_files = args.train_files
    elif man.get("train"):
        train_files = [os.path.join(raw, f) for f in man["train"]]
    else:
        train_files = sorted(glob.glob(os.path.join(raw, "data", "train-*.parquet")))
    if not val_files or not train_files:
        sys.exit(f"no parquet shards found under {raw} (run setup_login.sh first)")
    log(f"tokenizing {len(train_files)} train shards (limit {train_limit:,}) and "
        f"{len(val_files)} val shards with {args.threads} threads")
    info = {}
    info["val"] = tokenize_split(paths, "val", val_files, None, args.threads, args.batch_rows)
    info["train"] = tokenize_split(paths, "train", train_files, train_limit, args.threads, args.batch_rows)
    if info["train"]["n_tokens"] < train_limit:
        log(f"WARNING: only {info['train']['n_tokens']:,} train tokens available (< {train_limit:,})")
    finalize(paths, info, dict(
        synthetic=False, shards=[os.path.basename(f) for f in val_files + train_files],
        prep_seconds=time.time() - t0, train_token_limit=train_limit,
    ))


if __name__ == "__main__":
    main()
