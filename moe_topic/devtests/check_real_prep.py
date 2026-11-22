"""Independent check of a real-data prep output against the raw shards.

  python devtests/check_real_prep.py <work dir> <raw dir> [n_docs_per_split]

For random documents of every split: re-read the raw document by (shard, index), tokenize it again and
compare with the stream slice; compare topic / format / confidence / soft targets with the raw label files;
check per-token metadata, EOT placement, split disjointness, budgets and names.
"""
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from common import EOT, FORMAT_NAMES, SHARD_FILES, SPLITS, TOPIC_NAMES, Paths, Split, load_meta, shard_name  # noqa: E402


def main():
    work, raw = sys.argv[1], sys.argv[2]
    k = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    import tiktoken
    import zstandard

    enc = tiktoken.get_encoding("gpt2")
    paths = Paths(work, raw=raw)
    meta = load_meta(paths)
    assert meta["topic_names"] == TOPIC_NAMES and meta["format_names"] == FORMAT_NAMES
    docs_cache = {}

    def raw_shard(sid):
        if sid not in docs_cache:
            nm = shard_name(sid)
            f = {kk: os.path.join(raw, v.format(name=nm)) for kk, v in SHARD_FILES.items()}
            with open(f["documents"], "rb") as fh:
                rd = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
                texts = [json.loads(line)["text"] for line in rd if line.strip()]
            docs_cache[sid] = dict(texts=texts, topic=np.load(f["topic"]), fmt=np.load(f["format"]),
                                   tlog=np.load(f["topic_logits"]))
        return docs_cache[sid]

    rng = np.random.default_rng(0)
    keys = {}
    for s in SPLITS:
        sp = Split(paths, s, meta)
        nd = len(sp.offsets) - 1
        assert sp.info.shape == (nd, 5) and sp.soft.shape == (nd, 24)
        assert sp.offsets[0] == 0 and sp.offsets[-1] == sp.n_tokens and np.all(np.diff(sp.offsets) > 0)
        ends = sp.offsets[1:] - 1
        assert np.all(sp.tokens[ends] == EOT), f"{s}: every document must end with EOT"
        n_eot = int((np.asarray(sp.tokens) == EOT).sum())
        assert n_eot == nd, f"{s}: {n_eot} EOTs for {nd} documents (raw text contained the EOT id?)"
        # per-token metadata is the document label
        lens = np.diff(sp.offsets)
        for kind, col in (("topic", 0), ("format", 1), ("conf", 2)):
            arr = getattr(sp, kind)
            assert np.array_equal(np.asarray(arr), np.repeat(sp.info[:, col], lens).astype(np.uint8)), f"{s} {kind}"
        tt = np.bincount(sp.info[:, 0], weights=lens, minlength=24)
        assert np.array_equal(tt.astype(np.int64), np.asarray(meta[s]["topic_tokens"]))
        keys[s] = set(map(tuple, sp.info[:, 3:5].tolist()))
        for d in rng.choice(nd, min(k, nd), replace=False):
            sid, idx = int(sp.info[d, 3]), int(sp.info[d, 4])
            r = raw_shard(sid)
            want = enc.encode_ordinary(r["texts"][idx]) + [EOT]
            got = np.asarray(sp.tokens[sp.offsets[d]:sp.offsets[d + 1]]).tolist()
            assert got == want, f"{s} doc {d} (shard {sid} #{idx}): tokens differ"
            assert sp.info[d, 0] == r["topic"][idx] and sp.info[d, 1] == r["fmt"][idx], f"{s} doc {d}: labels"
            p = np.exp(r["tlog"][idx] - r["tlog"][idx].max())
            p /= p.sum()
            assert np.allclose(sp.soft[d].astype(np.float64), p, atol=2e-3), f"{s} doc {d}: soft targets"
            assert sp.info[d, 2] == int(p.max() >= 0.5)
        print(f"{s:5s} OK: {nd} docs, {sp.n_tokens} tokens; {min(k, nd)} docs re-tokenized from raw and matched")
    tr = set(meta["shards"]["train"])
    ev = set(meta["shards"]["eval"])
    assert not (tr & ev), "train and eval shards overlap"
    assert {kk[0] for kk in keys["train"]} <= tr and all({kk[0] for kk in keys[s]} <= ev for s in ("val", "bal", "abl"))
    assert not (keys["val"] & keys["bal"]), "natural val and balanced val share documents"
    assert keys["abl"] <= keys["bal"], "ablation subset must come from the balanced set"
    bt = np.asarray(meta["bal"]["topic_tokens"])
    at = np.asarray(meta["abl"]["topic_tokens"])
    assert np.all(bt <= meta["settings"]["bal_tokens"]) and np.all(at <= meta["settings"]["abl_tokens"])
    print("per-topic balanced tokens:", bt.tolist())
    print("per-topic balanced docs:  ", meta["bal"]["topic_docs"])
    print("per-topic ablation tokens:", at.tolist())
    print("ALL REAL-DATA CHECKS PASSED")


if __name__ == "__main__":
    main()
