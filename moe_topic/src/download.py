"""Login-node downloads (needs internet): WebOrganizer/Corpus-200B shards + label shards, classifier configs,
GPT-2 vocab. Everything lands inside the project (data/raw, cache/), so compute nodes can run offline.

  python src/download.py                 select + download shards, write data/raw/manifest.json
  python src/download.py --verify        offline check that every file in the manifest is present

Shard selection (deterministic, whole shards, no overlap):
  train: shards 0, 1, 2, ... in sorted order until their GPT-NeoX token counts (tokens/*.npy) reach
         --train_tokens * (1 + --margin)   (NeoX ~= GPT-2 within ~0.5% on this corpus, measured)
  eval:  the last shards (9888, 9887, ...) until the natural-val budget plus, for every topic, the
         balanced-val budget (tokens and >= 200 documents) are covered with headroom, max --eval_max_shards.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (DATASET_REPO, FORMAT_CLF_REPO, FORMAT_NAMES, N_TOPICS, SHARD_FILES, TOPIC_CLF_REPO,  # noqa: E402
                    TOPIC_NAMES, Paths, log, read_json, shard_name, write_json)

TAXONOMY_URLS = {
    "topics": "https://raw.githubusercontent.com/CodeCreator/WebOrganizer/main/define_domains/taxonomies/topics.yaml",
    "formats": "https://raw.githubusercontent.com/CodeCreator/WebOrganizer/main/define_domains/taxonomies/formats.yaml",
}


def hf_get(path, raw, repo=DATASET_REPO, repo_type="dataset", tries=6):
    from huggingface_hub import hf_hub_download

    for k in range(tries):
        try:
            return hf_hub_download(repo, path, repo_type=repo_type, local_dir=raw)
        except Exception as e:  # noqa: BLE001
            msg = str(e)
            if "gated" in msg.lower() or "401" in msg or "403" in msg:
                raise SystemExit(f"ACCESS DENIED for {repo}/{path}: the dataset may be gated. Accept its terms on "
                                 f"huggingface.co and run `huggingface-cli login`, then re-run.\n{msg}")
            if k == tries - 1:
                raise
            wait = 60 * (k + 1) if "429" in msg or "rate" in msg.lower() else 10 * (k + 1)
            log(f"download of {path} failed ({msg[:200]}); retry {k + 1}/{tries - 1} in {wait}s "
                f"(an HF token via `huggingface-cli login` raises rate limits)")
            time.sleep(wait)


def fetch_many(paths, raw, workers):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda p: hf_get(p, raw), paths))


def check_layout(files):
    """Fail loudly if the repo layout differs from what prep_data.py expects."""
    folders = {f.split("/")[0] for f in files if "/" in f}
    need = {v.split("/")[0] for v in SHARD_FILES.values()}
    missing = need - folders
    if missing:
        raise SystemExit(f"repo layout changed: folders {sorted(missing)} not found (have {sorted(folders)}). "
                         f"Update SHARD_FILES in src/common.py.")
    sets = []
    for key, pat in SHARD_FILES.items():
        pre, suf = pat.split("{name}")
        ids = set()
        for f in files:
            if f.startswith(pre) and f.endswith(suf):
                nm = f[len(pre):len(f) - len(suf)]
                if nm.startswith("CC_shard_") and nm.endswith("_processed"):
                    ids.add(int(nm.split("_")[2]))
        sets.append(ids)
    common = sorted(set.intersection(*sets))
    if not common:
        raise SystemExit("no shard has all of documents/tokens/topic/format files")
    return common


def taxonomy_check(raw):
    """Optional, non-fatal: compare label names with the taxonomy definitions in CodeCreator/WebOrganizer."""
    import urllib.request

    out = {}
    os.makedirs(os.path.join(raw, "taxonomies"), exist_ok=True)
    for kind, url in TAXONOMY_URLS.items():
        names = TOPIC_NAMES if kind == "topics" else FORMAT_NAMES
        try:
            txt = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", "replace")
        except Exception as e:  # noqa: BLE001
            out[kind] = f"not checked ({type(e).__name__}: {str(e)[:120]})"
            continue
        with open(os.path.join(raw, "taxonomies", f"{kind}.yaml"), "w") as f:
            f.write(txt)
        lines = txt.split("\n")
        try:
            i = lines.index("choices:")
        except ValueError:
            out[kind] = "not checked (no 'choices:' list in the YAML)"
            continue
        full = []
        for j in range(i + 1, len(lines) - 1):
            if lines[j].startswith("demonstrations"):
                break
            if lines[j].startswith("- "):
                full.append(lines[j + 1].strip())
        if len(full) != len(names):
            out[kind] = f"MISMATCH: {len(full)} taxonomy choices vs {len(names)} classifier labels"
            continue
        words = lambda x: {w.strip(".,()&/").lower()[:4] for w in x.split() if len(w.strip(".,()&/")) > 2}  # noqa: E731
        weak = [f"{a} ~ {b}" for a, b in zip(names, full) if not (words(a) & words(b))]
        out[kind] = dict(pairs=[f"{a} = {b}" for a, b in zip(names, full)],
                         verdict="same order, every label shares a word with its taxonomy name" if not weak
                         else f"check by hand: {weak}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None, help="download dir (default $MOE_ROOT/data/raw)")
    ap.add_argument("--train_tokens", type=float, default=1.25e9)
    ap.add_argument("--margin", type=float, default=0.03)
    ap.add_argument("--nat_val_tokens", type=float, default=10e6)
    ap.add_argument("--bal_tokens", type=float, default=400e3)
    ap.add_argument("--bal_min_docs", type=int, default=200)
    ap.add_argument("--eval_max_shards", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    paths = Paths(raw=args.raw)
    raw = paths.raw
    os.makedirs(raw, exist_ok=True)
    man_path = os.path.join(raw, "manifest.json")

    if args.verify:
        man = read_json(man_path)
        if not man or not man.get("complete"):
            raise SystemExit("manifest missing or incomplete: run src/download.py online first")
        bad = [p for p, s in man["files"].items() if not os.path.exists(os.path.join(raw, p)) or
               (s is not None and os.path.getsize(os.path.join(raw, p)) != s)]
        if bad:
            raise SystemExit(f"{len(bad)} files missing or wrong size, e.g. {bad[:3]}")
        for r in ("TopicClassifier", "FormatClassifier"):
            c = read_json(os.path.join(raw, "classifiers", r, "config.json"))
            assert c and len(c["id2label"]) == 24, f"classifier config {r} missing"
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        assert enc.encode(" the") == [262]
        log(f"verify OK: {len(man['files'])} files, train shards {len(man['train_shards'])} "
            f"(~{man['train_neox_tokens'] / 1e9:.2f}B NeoX tokens), eval shards {man['eval_shards']}; tiktoken offline OK")
        return

    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info(DATASET_REPO)
    gated = getattr(info, "gated", False)
    log(f"{DATASET_REPO}: gated={gated} private={getattr(info, 'private', None)}")
    files = api.list_repo_files(DATASET_REPO, repo_type="dataset")
    ids = check_layout(files)
    log(f"repo has {len(files)} files; {len(ids)} complete shards ({ids[0]} .. {ids[-1]})")

    # ---- train shards from the front
    train, ntok, k = [], 0, 0
    goal = args.train_tokens * (1 + args.margin)
    while ntok < goal:
        batch = ids[k:k + 16]
        if not batch:
            raise SystemExit("ran out of shards for training")
        got = fetch_many([SHARD_FILES["tokens"].format(name=shard_name(i)) for i in batch], raw, args.workers)
        for i, p in zip(batch, got):
            if ntok >= goal:
                break
            c = int(np.load(p).sum())
            train.append(dict(id=i, neox_tokens=c))
            ntok += c
        k += 16
    log(f"train: {len(train)} shards ({train[0]['id']}..{train[-1]['id']}), {ntok / 1e9:.3f}B NeoX tokens")

    # ---- eval shards from the back
    used = {t["id"] for t in train}
    rest = [i for i in reversed(ids) if i not in used]
    ev, tok_t, doc_t = [], np.zeros(N_TOPICS), np.zeros(N_TOPICS)
    reason = ""
    for i in rest[:args.eval_max_shards]:
        tp, cp = fetch_many([SHARD_FILES["tokens"].format(name=shard_name(i)),
                             SHARD_FILES["topic"].format(name=shard_name(i))], raw, 2)
        t, c = np.load(tp), np.load(cp)
        assert len(t) == len(c), f"shard {i}: tokens/topic length mismatch"
        tok_t += np.bincount(c, weights=t, minlength=N_TOPICS)
        doc_t += np.bincount(c, minlength=N_TOPICS)
        ev.append(dict(id=i, neox_tokens=int(t.sum()), docs=int(len(t))))
        total = tok_t.sum()
        f_rest = max(0.0, 1 - 1.1 * args.nat_val_tokens / total)
        ok_tok = tok_t * f_rest >= 1.25 * args.bal_tokens
        ok_doc = doc_t * f_rest >= 1.2 * args.bal_min_docs
        if total >= 1.2 * args.nat_val_tokens and ok_tok.all() and ok_doc.all():
            reason = "budgets covered"
            break
    else:
        reason = f"hit --eval_max_shards {args.eval_max_shards}; topics short: " + str(
            [TOPIC_NAMES[t] for t in range(N_TOPICS) if tok_t[t] * f_rest < 1.25 * args.bal_tokens])
    log(f"eval: {len(ev)} shards {[e['id'] for e in ev]}, {tok_t.sum() / 1e6:.1f}M NeoX tokens ({reason})")

    # ---- full file sets for the selected shards
    want = []
    for s in train + ev:
        for key in SHARD_FILES:
            want.append(SHARD_FILES[key].format(name=shard_name(s["id"])))
    sizes = {}
    for k0 in range(0, len(want), 500):
        for it in api.get_paths_info(DATASET_REPO, want[k0:k0 + 500], repo_type="dataset"):
            sizes[it.path] = getattr(it, "size", None)
    gb = sum(v or 0 for v in sizes.values()) / 1e9
    log(f"downloading {len(want)} files ({gb:.2f} GB) with {args.workers} workers (finished files are skipped)")
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(args.workers) as ex:
        for _ in ex.map(lambda p: hf_get(p, raw), want):
            done += 1
            if done % 50 == 0 or done == len(want):
                log(f"  {done}/{len(want)} files ({time.time() - t0:.0f}s)")
    bad = [p for p in want if sizes.get(p) is not None and os.path.getsize(os.path.join(raw, p)) != sizes[p]]
    if bad:
        raise SystemExit(f"size mismatch after download: {bad[:5]}")

    clf = {}
    for repo in (TOPIC_CLF_REPO, FORMAT_CLF_REPO):
        name = repo.split("/")[1]
        d = os.path.join(raw, "classifiers", name)
        from huggingface_hub import hf_hub_download

        for fn in ("config.json", "README.md"):
            hf_hub_download(repo, fn, local_dir=d)
        c = read_json(os.path.join(d, "config.json"))
        names = [c["id2label"][str(i)] for i in range(len(c["id2label"]))]
        ref = TOPIC_NAMES if "Topic" in name else FORMAT_NAMES
        clf[name] = "matches src/common.py" if names == ref else f"DIFFERS: {names}"
        log(f"{repo} id2label: {clf[name]}")
        if names != ref:
            raise SystemExit(f"{repo} labels differ from src/common.py; update TOPIC_NAMES / FORMAT_NAMES")
    tax = taxonomy_check(raw)
    log(f"taxonomy cross-check (GitHub, optional): {tax}")

    import tiktoken

    assert tiktoken.get_encoding("gpt2").encode(" the") == [262]
    write_json(man_path, dict(
        complete=True, created=time.strftime("%Y-%m-%d %H:%M:%S"), repo=DATASET_REPO, gated=gated,
        train_shards=[t["id"] for t in train], train_neox_tokens=ntok, train=train,
        eval_shards=[e["id"] for e in ev], eval=ev, eval_topic_neox_tokens=tok_t.tolist(),
        eval_topic_docs=doc_t.tolist(), eval_selection=reason, classifiers=clf, taxonomy_check=tax,
        settings=vars(args), files=sizes, gigabytes=gb))
    log(f"wrote {man_path}")


if __name__ == "__main__":
    main()
