"""Login-node downloads (the only stage that touches the internet).

  * SlimPajama-6B parquet: all validation shards + train shards one at a time until
    ~1.3B estimated tokens (characters / 4, via pyarrow utf8_length).
  * sentence-transformers/all-MiniLM-L6-v2 (json, txt, safetensors) into $HF_HOME.
  * tiktoken's GPT-2 vocab into $TIKTOKEN_CACHE_DIR.

Resumable: shards already present (and already counted in the manifest) are skipped.
`--verify` re-checks everything in offline mode, the way compute nodes will see it.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import DATASET_REPO, MINILM_REPO, Paths, log, read_json, write_json  # noqa: E402


def count_chars(path):
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(path)
    total = 0
    for batch in pf.iter_batches(batch_size=8192, columns=["text"]):
        total += int(pc.sum(pc.utf8_length(batch.column(0))).as_py() or 0)
    return total, pf.metadata.num_rows


def download_dataset(paths, target_tokens, max_val_shards=None):
    from huggingface_hub import hf_hub_download, list_repo_files

    os.makedirs(paths.raw, exist_ok=True)
    man_path = os.path.join(paths.raw, "manifest.json")
    man = read_json(man_path, {}) or {}
    man.setdefault("shards", {})
    files = sorted(list_repo_files(DATASET_REPO, repo_type="dataset"))
    val = [f for f in files if f.startswith("data/validation-") and f.endswith(".parquet")]
    train = [f for f in files if f.startswith("data/train-") and f.endswith(".parquet")]
    if not val or not train:
        raise RuntimeError(f"unexpected repo layout: {len(val)} validation / {len(train)} train shards")
    if max_val_shards:
        val = val[:max_val_shards]
    log(f"repo has {len(val)} validation and {len(train)} train shards")

    def get(fname):
        local = os.path.join(paths.raw, fname)
        info = man["shards"].get(fname)
        if info and os.path.exists(local) and os.path.getsize(local) == info.get("bytes"):
            return info
        log(f"downloading {fname}")
        p = hf_hub_download(DATASET_REPO, fname, repo_type="dataset", local_dir=paths.raw)
        chars, rows = count_chars(p)
        info = dict(bytes=os.path.getsize(p), chars=chars, rows=rows, est_tokens=chars // 4)
        man["shards"][fname] = info
        write_json(man_path, man)
        log(f"  {fname}: {rows} docs, {chars / 1e6:.0f}M chars, ~{chars / 4e6:.0f}M tokens")
        return info

    for f in val:
        get(f)
    est = 0
    used = []
    for f in train:
        if est >= target_tokens:
            break
        info = get(f)
        est += info["est_tokens"]
        used.append(f)
        log(f"  cumulative train estimate: {est / 1e9:.3f}B tokens (target {target_tokens / 1e9:.2f}B)")
    man["validation"] = val
    man["train"] = used
    man["target_tokens"] = target_tokens
    man["est_train_tokens"] = est
    man["complete"] = est >= target_tokens or len(used) == len(train)
    write_json(man_path, man)
    log(f"dataset done: {len(used)} train shards, ~{est / 1e9:.2f}B estimated tokens")


def download_minilm():
    from huggingface_hub import snapshot_download

    p = snapshot_download(MINILM_REPO, allow_patterns=["*.json", "*.txt", "*.safetensors"])
    log(f"MiniLM snapshot at {p}")


def download_tiktoken():
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    assert enc.n_vocab == 50257 and enc.eot_token == 50256
    log(f"tiktoken gpt2 cached in {os.environ.get('TIKTOKEN_CACHE_DIR', '(default cache)')}")


def verify(paths):
    """Offline check of every cached artefact, the way a compute node sees it."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    ok = True
    man = read_json(os.path.join(paths.raw, "manifest.json"))
    if not man or not man.get("complete"):
        log("FAIL: dataset manifest missing or incomplete")
        ok = False
    else:
        for f in man["validation"] + man["train"]:
            if not os.path.exists(os.path.join(paths.raw, f)):
                log(f"FAIL: missing shard {f}")
                ok = False
        log(f"dataset: {len(man['validation'])} val + {len(man['train'])} train shards, "
            f"~{man['est_train_tokens'] / 1e9:.2f}B est. train tokens")
    try:
        import tiktoken

        tiktoken.get_encoding("gpt2").encode_ordinary("offline check")
        log("tiktoken gpt2: OK (offline)")
    except Exception as e:  # noqa: BLE001
        log(f"FAIL: tiktoken offline: {e!r}")
        ok = False
    try:
        from transformers import AutoModel, AutoTokenizer

        AutoTokenizer.from_pretrained(MINILM_REPO)
        AutoModel.from_pretrained(MINILM_REPO)
        log("MiniLM: OK (offline)")
    except Exception as e:  # noqa: BLE001
        log(f"WARN: MiniLM does not load offline ({e!r}); clustering will use the IDF fallback")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_tokens", type=float, default=1.3e9)
    ap.add_argument("--max_val_shards", type=int, default=None)
    ap.add_argument("--skip_dataset", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    paths = Paths()
    if args.verify:
        sys.exit(0 if verify(paths) else 1)
    download_tiktoken()
    download_minilm()
    if not args.skip_dataset:
        download_dataset(paths, int(args.target_tokens), args.max_val_shards)


if __name__ == "__main__":
    main()
