"""Subject labels: one k-means cluster id (K=8) per 64-token block (GPU job).

1. Embed each block's decoded text with MiniLM (fp16 on GPU, max_length 128, mean pooling, L2 norm).
   CPU threads decode + tokenize the next batches while the GPU embeds the current one.
2. Spherical k-means on 200k random train blocks (50 iters, 3 restarts, keep best mean cosine).
3. Balance: per-centroid bias b, assignment argmax(cos + b), b -= lr (frac - 1/8), lr 0.1 halved every
   1000 iters, until every cluster holds 9.4%-15.6% of the sample.
4. Assign all train + val blocks by streaming -> data/{train,val}_cluster.npy (uint8).
5. Fallback (MiniLM fails to load, or projected assignment time > 30 min): IDF-weighted bag of GPT-2
   ids, random-projected to 256 dims, refit.
6. Sanity report: results/clusters.md + clusters.json (clusters.json is written last = stage marker).
"""
import argparse
import math
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    BLOCK, GPU_MISSING_MSG, MINILM_REPO, gpu_expected, N_CLUSTERS, N_SOURCES, SOURCE_SHORT, VOCAB_PAD, Paths, load_meta, log,
    mi_from_counts, write_json,
)

PAD = VOCAB_PAD - 1  # a padded-vocab id that never occurs in data


def get_blocks(tok, src, idx):
    """[n, 64] int64 tokens (PAD past the stream end), lengths, majority source per block."""
    idx = np.asarray(idx, dtype=np.int64)
    n_tok = len(tok)
    pos = idx[:, None] * BLOCK + np.arange(BLOCK)[None, :]
    valid = pos < n_tok
    posc = np.minimum(pos, n_tok - 1)
    if len(idx) > 1 and np.all(np.diff(idx) == 1) and valid.all():
        a, b = int(idx[0]) * BLOCK, (int(idx[-1]) + 1) * BLOCK
        blocks = np.asarray(tok[a:b], dtype=np.int64).reshape(-1, BLOCK)
        srcb = np.asarray(src[a:b]).reshape(-1, BLOCK)
    else:
        blocks = np.asarray(tok[posc], dtype=np.int64)
        srcb = np.asarray(src[posc])
    blocks = np.where(valid, blocks, PAD)
    lens = valid.sum(1)
    onehot = (srcb[:, :, None] == np.arange(N_SOURCES)[None, None, :]) & valid[:, :, None]
    major = onehot.sum(1).argmax(1).astype(np.int64)
    return blocks, lens, major


# ---------------------------------------------------------------- CPU prep in worker processes (no GIL)
_W = {}


def _winit(split_files, kind):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.set_num_threads(1)
    _W["kind"] = kind
    _W["data"] = {s: (np.memmap(t, dtype=np.uint16, mode="r", shape=(n,)), np.memmap(r, dtype=np.uint8, mode="r", shape=(n,)))
                  for s, (t, r, n) in split_files.items()}
    if kind == "minilm":
        import tiktoken
        from transformers import AutoTokenizer

        _W["enc"] = tiktoken.get_encoding("gpt2")
        _W["bt"] = AutoTokenizer.from_pretrained(MINILM_REPO)


def _wprep(split, idx):
    tok, src = _W["data"][split]
    blocks, lens, major = get_blocks(tok, src, idx)
    if _W["kind"] == "minilm":
        texts = _W["enc"].decode_batch([b[:n].tolist() for b, n in zip(blocks, lens)], errors="replace", num_threads=1)
        bt = _W["bt"](texts, max_length=128, truncation=True, padding=True, return_tensors="np")
        return (bt["input_ids"].astype(np.int32), bt["attention_mask"].astype(np.int8)), major
    return blocks.astype(np.int32), major


class Prep:
    """Pool of CPU workers preparing batches in order, with a bounded look-ahead window."""

    def __init__(self, split_files, kind, workers):
        self.workers = workers
        if workers > 0:
            ctx = mp.get_context("spawn")
            self.pool = ctx.Pool(workers, initializer=_winit, initargs=(split_files, kind))
        else:
            _winit(split_files, kind)
            self.pool = None

    def run(self, split, index_chunks):
        if self.pool is None:
            for idx in index_chunks:
                yield _wprep(split, idx)
            return
        it = iter(index_chunks)
        pending = []
        for idx in it:
            pending.append(self.pool.apply_async(_wprep, (split, idx)))
            if len(pending) >= 2 * self.workers:
                break
        while pending:
            r = pending.pop(0)
            nxt = next(it, None)
            if nxt is not None:
                pending.append(self.pool.apply_async(_wprep, (split, nxt)))
            yield r.get()

    def close(self):
        if self.pool is not None:
            self.pool.close()
            self.pool.join()


class MiniLMFeat:
    name = "minilm"
    kind = "minilm"

    def __init__(self, device):
        from transformers import AutoModel, AutoTokenizer

        self.device = device
        AutoTokenizer.from_pretrained(MINILM_REPO)  # fail early (offline cache) before starting workers
        model = AutoModel.from_pretrained(MINILM_REPO)
        if device.type == "cuda":
            model = model.half()
        self.model = model.to(device).eval()
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def embed(self, payload):
        ids, mask = (torch.from_numpy(a).long() for a in payload)
        if self.device.type == "cuda":
            ids, mask = ids.pin_memory().to(self.device, non_blocking=True), mask.pin_memory().to(self.device, non_blocking=True)
        hs = self.model(input_ids=ids, attention_mask=mask).last_hidden_state.float()
        m = mask.unsqueeze(-1).float()
        emb = (hs * m).sum(1) / m.sum(1).clamp(min=1.0)
        return F.normalize(emb, dim=-1)


class IdfFeat:
    name = "idf_bag_rp256"
    kind = "idf"

    def __init__(self, device, dim=256, seed=0):
        self.device, self.dim = device, dim
        g = torch.Generator().manual_seed(seed)
        self.R = (torch.randn(VOCAB_PAD, dim, generator=g) / math.sqrt(dim)).to(device)
        self.idf = None

    def fit(self, blocks):
        """IDF from block document frequency on the sample."""
        s = np.sort(blocks, axis=1)
        first = np.ones_like(s, dtype=bool)
        first[:, 1:] = s[:, 1:] != s[:, :-1]
        df = np.bincount(s[first], minlength=VOCAB_PAD).astype(np.float64)
        n = len(blocks)
        idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
        idf[PAD] = 0.0
        self.idf = torch.tensor(idf, dtype=torch.float32, device=self.device)

    @torch.no_grad()
    def embed(self, blocks):
        t = torch.from_numpy(blocks).long()
        t = t.pin_memory().to(self.device, non_blocking=True) if self.device.type == "cuda" else t
        e = F.embedding_bag(t, self.R, per_sample_weights=self.idf[t], mode="sum")
        return F.normalize(e, dim=-1)


def featurize(feat, prep, split, index_chunks):
    """Yield (embeddings [n,d] on device, majority source [n]) per chunk; CPU prep runs in workers."""
    for payload, major in prep.run(split, index_chunks):
        yield feat.embed(payload), major


def spherical_kmeans(X, K, iters, restarts, seed):
    n = X.shape[0]
    best = None
    scores = []
    for r in range(restarts):
        g = torch.Generator().manual_seed(seed * 1000 + r)
        # k-means++ seeding with cosine distance
        C = X[torch.randint(0, n, (1,), generator=g)].clone()
        for _ in range(1, K):
            d = (1.0 - (X @ C.T).max(1).values).clamp(min=0).double().cpu()
            p = d / d.sum() if d.sum() > 0 else torch.full_like(d, 1.0 / n)
            C = torch.cat([C, X[torch.multinomial(p, 1, generator=g)]], 0)
        for _ in range(iters):
            a = (X @ C.T).argmax(1)
            newC = torch.zeros_like(C).index_add_(0, a, X)
            cnt = torch.bincount(a, minlength=K)
            empty = (cnt == 0).nonzero().flatten()
            for j in empty.tolist():  # re-seed an empty cluster at a random point
                newC[j] = X[torch.randint(0, n, (1,), generator=g)]
            C = F.normalize(newC, dim=-1)
        score = (X @ C.T).max(1).values.mean().item()
        scores.append(score)
        log(f"  k-means restart {r}: mean cosine {score:.4f}")
        if best is None or score > best[0]:
            best = (score, C)
    return best[1], scores


def balance(sims, K, lo=0.094, hi=0.156, lr=0.1, max_iter=20000):
    b = torch.zeros(K, device=sims.device)
    n = sims.shape[0]
    best = (float("inf"), b.clone(), None)
    for it in range(max_iter):
        frac = torch.bincount((sims + b).argmax(1), minlength=K).float() / n
        dev = float(torch.clamp(lo - frac, min=0).max().item() + torch.clamp(frac - hi, min=0).max().item())
        if dev < best[0]:
            best = (dev, b.clone(), frac.clone())
        if dev == 0.0:
            return b, frac, it, True
        b -= lr * (frac - 1.0 / K)
        if (it + 1) % 1000 == 0:
            lr /= 2
    return best[1], best[2], max_iter, False


def fit(feat, prep, sample_idx, args):
    t0 = time.time()
    chunks = [sample_idx[i:i + args.batch] for i in range(0, len(sample_idx), args.batch)]
    X = torch.cat([e for e, _ in featurize(feat, prep, "train", chunks)], 0).float()
    if X.device.type == "cuda":
        torch.cuda.synchronize()
    embed_s = time.time() - t0
    C, scores = spherical_kmeans(X, N_CLUSTERS, args.iters, args.restarts, args.seed)
    sims = X @ C.T
    b, frac, iters, ok = balance(sims, N_CLUSTERS)
    log(f"  balance: {'converged' if ok else 'NOT converged'} after {iters} iters; "
        f"fractions {[round(f, 3) for f in frac.tolist()]}")
    labels = (sims + b).argmax(1).cpu().numpy()
    return dict(C=C, b=b, scores=scores, frac=frac.tolist(), balance_iters=iters, balance_ok=ok,
                embed_seconds=embed_s, rate=len(sample_idx) / max(embed_s, 1e-9), sample_labels=labels)


def assign_split(feat, prep, split, model, nb, args, out_path):
    C, b = model["C"], model["b"]
    labels = np.empty(nb, dtype=np.uint8)
    joint = np.zeros((N_CLUSTERS, N_SOURCES), dtype=np.int64)
    starts = list(range(0, nb, args.batch))
    chunks = (np.arange(s, min(s + args.batch, nb)) for s in starts)
    t0, last, done = time.time(), time.time(), 0
    for (emb, major), s in zip(featurize(feat, prep, split, chunks), starts):
        lab = (emb @ C.T + b).argmax(1).to(torch.uint8).cpu().numpy()
        labels[s:s + len(lab)] = lab
        joint += np.bincount(lab.astype(np.int64) * N_SOURCES + major,
                             minlength=N_CLUSTERS * N_SOURCES).reshape(N_CLUSTERS, N_SOURCES)
        done += len(lab)
        if time.time() - last > 60:
            last = time.time()
            rate = done / (last - t0)
            log(f"  assigned {done:,}/{nb:,} blocks ({rate:,.0f}/s, ETA {(nb - done) / rate / 60:.1f} min)")
    tmp = out_path + ".tmp.npy"
    np.save(tmp, labels)
    os.replace(tmp, out_path)
    log(f"  wrote {out_path} ({nb:,} blocks, {time.time() - t0:.0f}s)")
    return labels, joint


def md_escape(s, n=240):
    s = s.replace("\r", " ").replace("\n", " ⏎ ").replace("|", "\\|").replace("`", "'")
    s = " ".join(s.split())
    return s[:n] + ("…" if len(s) > n else "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    ap.add_argument("--featurizer", default="auto", choices=["auto", "minilm", "idf"])
    ap.add_argument("--sample", type=int, default=200_000)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--prep_threads", type=int, default=max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", 4)) - 1),
                    help="CPU worker processes for decode/tokenize (0 = in-process)")
    ap.add_argument("--max_assign_minutes", type=float, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--selftest_minilm", type=int, default=0,
                    help="only load MiniLM offline and embed N train blocks, print the rate (smoke test)")
    args = ap.parse_args()
    t_start = time.time()
    if args.selftest_minilm:
        paths = Paths(args.work)
        meta = load_meta(paths)
        dev = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
        tok = np.memmap(paths.tokens("train"), dtype=np.uint16, mode="r", shape=(meta["train"]["n_tokens"],))
        src = np.memmap(paths.src("train"), dtype=np.uint8, mode="r", shape=(meta["train"]["n_tokens"],))
        feat = MiniLMFeat(dev)
        prep = Prep({"train": (paths.tokens("train"), paths.src("train"), meta["train"]["n_tokens"])}, feat.kind, args.prep_threads)
        n = min(args.selftest_minilm, meta["train"]["n_tokens"] // BLOCK)
        chunks = [np.arange(s, min(s + args.batch, n)) for s in range(0, n, args.batch)]
        t0 = time.time()
        k = sum(e.shape[0] for e, _ in featurize(feat, prep, "train", chunks))
        rate = k / (time.time() - t0)
        prep.close()
        print(f"MINILM_OK blocks={k} rate={rate:.0f}/s device={dev}")
        return
    paths = Paths(args.work).mk()
    meta = load_meta(paths)
    dev = torch.device("cuda" if (args.device != "cpu" and torch.cuda.is_available()) else "cpu")
    if dev.type != "cuda" and (args.device == "cuda" or (args.device == "auto" and gpu_expected())):
        sys.exit(f"ERROR: {GPU_MISSING_MSG}")
    if os.path.exists(paths.clusters_json):
        os.remove(paths.clusters_json)  # invalidate the stage marker while we rebuild
    torch.set_num_threads(max(1, args.prep_threads))
    toks = {s: np.memmap(paths.tokens(s), dtype=np.uint16, mode="r", shape=(meta[s]["n_tokens"],)) for s in ("train", "val")}
    srcs = {s: np.memmap(paths.src(s), dtype=np.uint8, mode="r", shape=(meta[s]["n_tokens"],)) for s in ("train", "val")}
    nbs = {s: meta[s]["n_blocks"] for s in ("train", "val")}
    total_blocks = nbs["train"] + nbs["val"]
    rng = np.random.default_rng(args.seed)
    n_full = meta["train"]["n_tokens"] // BLOCK
    sample_idx = np.sort(rng.choice(n_full, size=min(args.sample, n_full), replace=False))
    log(f"device {dev}; {total_blocks:,} blocks to assign; sample {len(sample_idx):,}")

    split_files = {s: (paths.tokens(s), paths.src(s), meta[s]["n_tokens"]) for s in ("train", "val")}
    fallback_reason = None
    feat = model = prep = None
    if args.featurizer in ("auto", "minilm"):
        try:
            feat = MiniLMFeat(dev)
            log(f"MiniLM loaded; embedding the sample ({args.prep_threads} CPU prep workers)")
            prep = Prep(split_files, feat.kind, args.prep_threads)
            model = fit(feat, prep, sample_idx, args)
            proj = total_blocks / model["rate"] / 60
            log(f"MiniLM: {model['rate']:,.0f} blocks/s on the sample -> ~{proj:.1f} min to assign all blocks")
            model["projected_minutes"] = proj
            if args.featurizer == "auto" and proj > args.max_assign_minutes:
                fallback_reason = f"MiniLM projected {proj:.1f} min > {args.max_assign_minutes} min to assign all blocks"
        except Exception as e:  # noqa: BLE001
            if args.featurizer == "minilm":
                raise
            fallback_reason = f"MiniLM failed: {type(e).__name__}: {e}"
        if fallback_reason:
            log(f"FALLBACK to IDF features: {fallback_reason}")
            if prep is not None:
                prep.close()
            feat = model = prep = None
    if feat is None:
        feat = IdfFeat(dev, seed=args.seed)
        blocks, _, _ = get_blocks(toks["train"], srcs["train"], sample_idx)
        feat.fit(blocks)
        prep = Prep(split_files, feat.kind, args.prep_threads)
        model = fit(feat, prep, sample_idx, args)
        model["projected_minutes"] = total_blocks / model["rate"] / 60

    joints, sizes, mi = {}, {}, {}
    for s in ("train", "val"):
        log(f"assigning {s} blocks")
        labels, joint = assign_split(feat, prep, s, model, nbs[s], args, paths.cluster(s))
        joints[s] = joint
        sizes[s] = np.bincount(labels, minlength=N_CLUSTERS).tolist()
        mi[s] = mi_from_counts(joint)
    prep.close()

    # sample texts: 3 per cluster from the fitting sample
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    samples = {}
    for c in range(N_CLUSTERS):
        pick = sample_idx[model["sample_labels"] == c][:3]
        texts = []
        for g in pick:
            ids = np.asarray(toks["train"][g * BLOCK:(g + 1) * BLOCK], dtype=np.int64).tolist()
            texts.append(enc.decode(ids, errors="replace"))
        samples[c] = texts

    jt = joints["train"].astype(np.float64)
    row_pct = (100.0 * jt / np.maximum(jt.sum(1, keepdims=True), 1)).round(1).tolist()
    info = dict(
        featurizer=feat.name, fallback_reason=fallback_reason, k=N_CLUSTERS, sample=len(sample_idx),
        kmeans_scores=model["scores"], bias=model["b"].tolist(), balance_iters=model["balance_iters"],
        balance_converged=model["balance_ok"], sample_fractions=model["frac"],
        sample_rate_blocks_per_s=model["rate"], projected_assign_minutes=model["projected_minutes"],
        sizes=sizes, fractions={s: (np.array(v) / max(sum(v), 1)).round(4).tolist() for s, v in sizes.items()},
        table_train_row_pct=row_pct, joint_counts={s: j.tolist() for s, j in joints.items()},
        mi_bits_block_majority_source=mi, samples={str(k): v for k, v in samples.items()},
        n_blocks=nbs, data_fingerprint=meta["fingerprint"], synthetic=meta.get("synthetic", False),
        device=str(dev), seconds=time.time() - t_start,
    )

    lines = ["# Subject clusters: sanity report", ""]
    if meta.get("synthetic"):
        lines += ["> **SYNTHETIC DATA (smoke/local test): these clusters mean nothing.**", ""]
    lines += [f"- featurizer: **{feat.name}**" + (f" (fallback: {fallback_reason})" if fallback_reason else ""),
              f"- k-means mean cosine per restart: {[round(x, 4) for x in model['scores']]}",
              f"- balance converged: {model['balance_ok']} after {model['balance_iters']} iterations; "
              f"bias = {[round(x, 3) for x in model['b'].tolist()]}",
              f"- I(cluster; block majority source): train **{mi['train']:.3f} bits**, val {mi['val']:.3f} bits "
              f"(max possible log2(7) = 2.807)",
              f"- time: {info['seconds'] / 60:.1f} min ({model['rate']:,.0f} blocks/s on the sample)", "",
              "## Cluster sizes", "", "| cluster | train blocks | train % | val blocks | val % |",
              "|---:|---:|---:|---:|---:|"]
    for c in range(N_CLUSTERS):
        lines.append(f"| {c} | {sizes['train'][c]:,} | {100 * info['fractions']['train'][c]:.1f} | "
                     f"{sizes['val'][c]:,} | {100 * info['fractions']['val'][c]:.1f} |")
    lines += ["", "## Cluster vs. block majority source (train, row %)", "",
              "| cluster | " + " | ".join(SOURCE_SHORT) + " |", "|---:|" + "---:|" * N_SOURCES]
    for c in range(N_CLUSTERS):
        lines.append(f"| {c} | " + " | ".join(f"{v:.1f}" for v in row_pct[c]) + " |")
    lines += ["", "## Sample blocks", ""]
    for c in range(N_CLUSTERS):
        lines.append(f"**cluster {c}**")
        lines.append("")
        for t in samples[c]:
            lines.append(f"- `{md_escape(t)}`")
        lines.append("")
    with open(os.path.join(paths.results, "clusters.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    write_json(paths.clusters_json, info)  # stage marker, written last
    log(f"clusters done in {info['seconds'] / 60:.1f} min; featurizer {feat.name}; I(cluster;source) {mi['train']:.3f} bits")


if __name__ == "__main__":
    main()
