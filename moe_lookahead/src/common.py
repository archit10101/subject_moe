"""Shared constants, paths and small helpers used by every stage.

Directory layout (all relative to the work root, which is $MOE_WORK or $MOE_ROOT):
  data/      token streams, source ids, cluster labels, meta.json   (data/raw/ holds parquet)
  ckpt/      <tag>_last.pt (+ <tag>_last.json sidecar), R1_r4ref.pt
  logs/      <tag>.jsonl training logs, slurm_*.out, jobs.tsv
  results/   config.json, clusters.{md,json}, eval/<tag>.json, REPORT.md, summary.json, plots/
Caches (HF, tiktoken) always live under $MOE_ROOT/cache, also for the smoke test.
"""
import hashlib
import json
import os
import time

import numpy as np

SEQ_LEN = 1024
BLOCK = 64
BLOCKS_PER_SEQ = SEQ_LEN // BLOCK  # 16
EOT = 50256
VOCAB = 50257
VOCAB_PAD = 50304
N_CLUSTERS = 8

DATASET_REPO = "DKYoon/SlimPajama-6B"
MINILM_REPO = "sentence-transformers/all-MiniLM-L6-v2"

SOURCES = [
    "RedPajamaCommonCrawl",
    "RedPajamaC4",
    "RedPajamaGithub",
    "RedPajamaBook",
    "RedPajamaArXiv",
    "RedPajamaWikipedia",
    "RedPajamaStackExchange",
]
SOURCE_SHORT = ["CC", "C4", "Github", "Book", "ArXiv", "Wiki", "StackEx"]
N_SOURCES = len(SOURCES)

# (tag, mode, init_seed, tier) in submission priority order.
RUNS = [
    ("R1", "r1", 0, 1),
    ("R3", "r3", 0, 1),
    ("R2", "r2", 0, 1),
    ("R4", "r4", 0, 2),
    ("R1_seed1", "r1", 1, 2),
    ("R3_seed1", "r3", 1, 3),
    ("R2_seed1", "r2", 1, 3),
]
RUN_INFO = {t: dict(tag=t, mode=m, seed=s, tier=k) for (t, m, s, k) in RUNS}


def project_root():
    env = os.environ.get("MOE_ROOT")
    if env:
        return os.path.abspath(env)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def work_root(arg=None):
    if arg:
        return os.path.abspath(arg)
    env = os.environ.get("MOE_WORK")
    if env:
        return os.path.abspath(env)
    return project_root()


class Paths:
    def __init__(self, work=None):
        self.work = work_root(work)
        self.root = project_root()
        self.data = os.path.join(self.work, "data")
        self.raw = os.path.join(self.root, "data", "raw")  # raw parquet is always under the project
        self.ckpt = os.path.join(self.work, "ckpt")
        self.logs = os.path.join(self.work, "logs")
        self.results = os.path.join(self.work, "results")
        self.cache = os.path.join(self.root, "cache")

    def mk(self):
        for d in (self.data, self.ckpt, self.logs, self.results):
            os.makedirs(d, exist_ok=True)
        return self

    # data files
    def tokens(self, split):
        return os.path.join(self.data, f"{split}_tokens.bin")

    def src(self, split):
        return os.path.join(self.data, f"{split}_src.bin")

    def cluster(self, split):
        return os.path.join(self.data, f"{split}_cluster.npy")

    @property
    def meta(self):
        return os.path.join(self.data, "meta.json")

    @property
    def config(self):
        return os.path.join(self.results, "config.json")

    @property
    def clusters_json(self):
        return os.path.join(self.results, "clusters.json")

    def ckpt_last(self, tag):
        return os.path.join(self.ckpt, f"{tag}_last.pt")

    def ckpt_side(self, tag):
        return os.path.join(self.ckpt, f"{tag}_last.json")

    def log(self, tag):
        return os.path.join(self.logs, f"{tag}.jsonl")

    @property
    def jobs_tsv(self):
        return os.path.join(self.logs, "jobs.tsv")


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path, obj):
    """Atomic JSON write (tmp + rename) so readers never see half a file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)
        f.write("\n")
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def n_blocks(n_tokens):
    return (n_tokens + BLOCK - 1) // BLOCK


def n_seq(n_tokens):
    return max(0, (n_tokens - 1) // SEQ_LEN)


def stream_fingerprint(path, n_items, itemsize):
    """Cheap but sensitive fingerprint: length + 256 strided 4 KiB chunks."""
    h = hashlib.sha256()
    h.update(str(n_items).encode())
    nbytes = n_items * itemsize
    chunk = 4096
    with open(path, "rb") as f:
        if nbytes <= 256 * chunk:
            h.update(f.read())
        else:
            for i in range(256):
                f.seek((nbytes - chunk) * i // 255)
                h.update(f.read(chunk))
    return h.hexdigest()[:16]


def load_meta(paths):
    meta = read_json(paths.meta)
    if meta is None:
        raise FileNotFoundError(f"{paths.meta} missing: run the prep stage first")
    return meta


def tokens_memmap(paths, split, meta=None, in_ram=False):
    meta = meta or load_meta(paths)
    n = meta[split]["n_tokens"]
    if in_ram:
        arr = np.fromfile(paths.tokens(split), dtype=np.uint16, count=n)
    else:
        arr = np.memmap(paths.tokens(split), dtype=np.uint16, mode="r", shape=(n,))
    return arr


def src_memmap(paths, split, meta=None, in_ram=False):
    meta = meta or load_meta(paths)
    n = meta[split]["n_tokens"]
    if in_ram:
        return np.fromfile(paths.src(split), dtype=np.uint8, count=n)
    return np.memmap(paths.src(split), dtype=np.uint8, mode="r", shape=(n,))


def load_clusters(paths, split):
    return np.load(paths.cluster(split))


def mutual_info_bits(a, b, na, nb):
    """I(A;B) in bits from two aligned integer arrays, via bincount on combined indices."""
    a = np.asarray(a, dtype=np.int64).ravel()
    b = np.asarray(b, dtype=np.int64).ravel()
    joint = np.bincount(a * nb + b, minlength=na * nb).reshape(na, nb).astype(np.float64)
    return mi_from_counts(joint)


def mi_from_counts(joint):
    joint = np.asarray(joint, dtype=np.float64)
    tot = joint.sum()
    if tot <= 0:
        return 0.0
    p = joint / tot
    pa = p.sum(1, keepdims=True)
    pb = p.sum(0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (pa @ pb)[nz])).sum())


def gpu_expected():
    """True inside a SLURM job that was allocated GPUs (then silently using the CPU would be wrong)."""
    if os.environ.get("MOE_ALLOW_CPU") == "1" or not os.environ.get("SLURM_JOB_ID"):
        return False
    return any(os.environ.get(k) for k in ("SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_STEP_GPUS",
                                           "CUDA_VISIBLE_DEVICES"))


GPU_MISSING_MSG = ("this SLURM job was given a GPU but torch.cuda.is_available() is False. The torch CUDA build "
                   "probably does not match the node's driver: see RUN_ON_ORCD.md section 4 (reinstall torch).")


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{now_str()}] {msg}", flush=True)
