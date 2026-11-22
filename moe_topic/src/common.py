"""Shared constants, paths and small helpers used by every stage.

Directory layout (relative to the work root = $MOE_WORK, else $MOE_ROOT):
  data/       {split}_tokens.bin (uint16), {split}_topic.bin / _format.bin / _conf.bin (uint8, per token),
              {split}_doc_offsets.npy (int64, n_docs+1), {split}_doc_soft.npy (float16 [n_docs, 24]),
              {split}_doc_info.npy (int32 [n_docs, 5]: topic, format, conf, shard, index in shard), meta.json
              splits: train, val (natural), bal (balanced), abl (ablation subset of bal)
  ckpt/       <tag>_last.pt (+ <tag>_last.json sidecar), <tag>_p25.pt / _p50.pt / _p75.pt (weights only)
  logs/       <tag>.jsonl training logs, slurm_*.out, jobs.tsv
  results/    config.json, eval/<tag>.json, examples/<tag>.md, REPORT.md, summary.json, plots/
Raw downloads always live in $MOE_ROOT/data/raw and caches in $MOE_ROOT/cache (also for the smoke test).
"""
import hashlib
import json
import os
import time

import numpy as np

SEQ_LEN = 1024
EOT = 50256
VOCAB = 50257
VOCAB_PAD = 50304
N_TOPICS = 24
N_FORMATS = 24
SUMMARY_WINDOW = 50   # s_t = mean of h over max(t-49, d_t) .. t
CONF_THRESHOLD = 0.5  # "high-confidence" documents: top-topic softmax probability >= 0.5
THE_TOKEN = 262       # GPT-2 id of " the"

DATASET_REPO = "WebOrganizer/Corpus-200B"
TOPIC_CLF_REPO = "WebOrganizer/TopicClassifier"
FORMAT_CLF_REPO = "WebOrganizer/FormatClassifier"

# id2label of WebOrganizer/TopicClassifier and /FormatClassifier (verified 2026-11-22 against both model
# configs and both model cards). prep_data.py re-reads the downloaded configs and asserts they still agree.
TOPIC_NAMES = [
    "Adult", "Art & Design", "Software Dev.", "Crime & Law", "Education & Jobs", "Hardware",
    "Entertainment", "Social Life", "Fashion & Beauty", "Finance & Business", "Food & Dining", "Games",
    "Health", "History", "Home & Hobbies", "Industrial", "Literature", "Politics", "Religion",
    "Science & Tech.", "Software", "Sports & Fitness", "Transportation", "Travel",
]
FORMAT_NAMES = [
    "Academic Writing", "Content Listing", "Creative Writing", "Customer Support", "Comment Section", "FAQ",
    "Truncated", "Knowledge Article", "Legal Notices", "Listicle", "News Article", "Nonfiction Writing",
    "About (Org.)", "News (Org.)", "About (Pers.)", "Personal Blog", "Product Page", "Q&A Forum",
    "Spam / Ads", "Structured Data", "Documentation", "Audio Transcript", "Tutorial", "User Review",
]

# Shard file layout actually found in the repo (the README says "domain_topics__logits/"; the repo has
# "domains_topics__logits/", and format logits live in their own "domains_formats__logits/" folder).
SHARD_FILES = {
    "documents": "documents/{name}.jsonl.zst",
    "tokens": "tokens/{name}.npy",
    "topic": "domains_topics/{name}__choice.npy",
    "topic_logits": "domains_topics__logits/{name}__logits.npy",
    "format": "domains_formats/{name}__choice.npy",
    "format_logits": "domains_formats__logits/{name}__logits.npy",
}


def shard_name(i):
    return f"CC_shard_{int(i):08d}_processed"


# (tag, mode, init_seed) in submission priority order.
RUNS = [
    ("R1", "r1", 0),
    ("P2", "p2", 0),
    ("P4", "p4", 0),
    ("R1_seed1", "r1", 1),
    ("R1f", "r1f", 0),
    ("P3", "p3", 0),
    ("P1", "p1", 0),
    ("P2_seed1", "p2", 1),
    ("P4_seed1", "p4", 1),
]
RUN_INFO = {t: dict(tag=t, mode=m, seed=s, prio=i) for i, (t, m, s) in enumerate(RUNS)}
MODE_TAG = {"r1": "R1", "r1f": "R1f", "p1": "P1", "p2": "P2", "p3": "P3", "p4": "P4"}
SPLITS = ("train", "val", "bal", "abl")
MILESTONES = (25, 50, 75)  # % of training: weights-only checkpoints for specialization over time


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
    def __init__(self, work=None, raw=None):
        self.work = work_root(work)
        self.root = project_root()
        self.data = os.path.join(self.work, "data")
        self.raw = os.path.abspath(raw) if raw else os.environ.get("MOE_RAW") or os.path.join(self.root, "data", "raw")
        self.ckpt = os.path.join(self.work, "ckpt")
        self.logs = os.path.join(self.work, "logs")
        self.results = os.path.join(self.work, "results")
        self.cache = os.path.join(self.root, "cache")

    def mk(self):
        for d in (self.data, self.ckpt, self.logs, self.results):
            os.makedirs(d, exist_ok=True)
        return self

    def tokens(self, split):
        return os.path.join(self.data, f"{split}_tokens.bin")

    def tok_meta(self, split, kind):  # kind: topic | format | conf
        return os.path.join(self.data, f"{split}_{kind}.bin")

    def doc_offsets(self, split):
        return os.path.join(self.data, f"{split}_doc_offsets.npy")

    def doc_soft(self, split):
        return os.path.join(self.data, f"{split}_doc_soft.npy")

    def doc_info(self, split):
        return os.path.join(self.data, f"{split}_doc_info.npy")

    @property
    def meta(self):
        return os.path.join(self.data, "meta.json")

    @property
    def manifest(self):
        return os.path.join(self.raw, "manifest.json")

    @property
    def config(self):
        return os.path.join(self.results, "config.json")

    def ckpt_last(self, tag):
        return os.path.join(self.ckpt, f"{tag}_last.pt")

    def ckpt_side(self, tag):
        return os.path.join(self.ckpt, f"{tag}_last.json")

    def ckpt_milestone(self, tag, pct):
        return os.path.join(self.ckpt, f"{tag}_p{pct}.pt")

    def log(self, tag):
        return os.path.join(self.logs, f"{tag}.jsonl")

    def eval_json(self, tag):
        return os.path.join(self.results, "eval", f"{tag}.json")

    def examples(self, tag):
        return os.path.join(self.results, "examples", f"{tag}.md")

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
        json.dump(obj, f, indent=1, default=_json_default)
        f.write("\n")
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")


def n_seq(n_tokens):
    return max(0, (n_tokens - 1) // SEQ_LEN)


def stream_fingerprint(path, nbytes):
    """Cheap but sensitive fingerprint: length + 256 strided 4 KiB chunks."""
    h = hashlib.sha256()
    h.update(str(nbytes).encode())
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


class Split:
    """All arrays of one split (token stream memmapped or in RAM, per-token metadata, per-document arrays)."""

    def __init__(self, paths, split, meta=None, in_ram=False, max_tokens=None):
        meta = meta or load_meta(paths)
        n = meta[split]["n_tokens"]
        if max_tokens:
            n = min(n, int(max_tokens))
        self.name, self.n_tokens = split, n
        rd = lambda p, dt: np.memmap(p, dtype=dt, mode="r", shape=(n,))  # noqa: E731
        # in_ram: only the token stream is read into RAM (per-token metadata stays memory-mapped)
        self.tokens = np.fromfile(paths.tokens(split), dtype=np.uint16, count=n) if in_ram else \
            rd(paths.tokens(split), np.uint16)
        self.topic = rd(paths.tok_meta(split, "topic"), np.uint8)
        self.format = rd(paths.tok_meta(split, "format"), np.uint8)
        self.conf = rd(paths.tok_meta(split, "conf"), np.uint8)
        self.offsets = np.load(paths.doc_offsets(split))
        self.soft = np.load(paths.doc_soft(split))
        self.info = np.load(paths.doc_info(split))
        self.n_seq = n_seq(n)

    def doc_index(self, positions):
        return np.searchsorted(self.offsets, positions, side="right") - 1


# ----------------------------------------------------------------------------- information theory
def entropy_bits(counts):
    c = np.asarray(counts, dtype=np.float64).ravel()
    t = c.sum()
    if t <= 0:
        return 0.0
    p = c[c > 0] / t
    return float(-(p * np.log2(p)).sum())


def mi_from_counts(joint):
    joint = np.asarray(joint, dtype=np.float64)
    tot = joint.sum()
    if tot <= 0:
        return 0.0
    p = joint / tot
    pa = p.sum(1, keepdims=True)
    pb = p.sum(0, keepdims=True)
    nz = p > 0
    return float(max(0.0, (p[nz] * np.log2(p[nz] / (pa @ pb)[nz])).sum()))


def joint_counts(a, b, na, nb):
    a = np.asarray(a, dtype=np.int64).ravel()
    b = np.asarray(b, dtype=np.int64).ravel()
    return np.bincount(a * nb + b, minlength=na * nb).reshape(na, nb)


def mutual_info_bits(a, b, na, nb):
    """I(A;B) in bits from two aligned integer arrays, via bincount on combined indices."""
    return mi_from_counts(joint_counts(a, b, na, nb))


def conditional_mi_bits(a, b, c, na, nb, nc, min_count=0):
    """I(A;B|C) = sum_c p(c) I(A;B|C=c), over values of c with >= min_count samples (weights renormalized).
    Returns (cmi, H(B|C) over the same values, number of c values used, fraction of samples used)."""
    a = np.asarray(a, dtype=np.int64).ravel()
    b = np.asarray(b, dtype=np.int64).ravel()
    c = np.asarray(c, dtype=np.int64).ravel()
    j = np.bincount((c * na + a) * nb + b, minlength=nc * na * nb).reshape(nc, na, nb).astype(np.float64)
    nper = j.sum((1, 2))
    use = np.nonzero(nper >= max(min_count, 1))[0]
    if len(use) == 0:
        return 0.0, 0.0, 0, 0.0
    w = nper[use] / nper[use].sum()
    cmi = sum(wi * mi_from_counts(j[ci]) for wi, ci in zip(w, use))
    hb = sum(wi * entropy_bits(j[ci].sum(0)) for wi, ci in zip(w, use))
    return float(cmi), float(hb), int(len(use)), float(nper[use].sum() / max(nper.sum(), 1))


def gpu_expected():
    """True inside a SLURM job that was allocated GPUs (then silently using the CPU would be wrong)."""
    if os.environ.get("MOE_ALLOW_CPU") == "1" or not os.environ.get("SLURM_JOB_ID"):
        return False
    return any(os.environ.get(k) for k in ("SLURM_JOB_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_STEP_GPUS",
                                           "CUDA_VISIBLE_DEVICES"))


GPU_MISSING_MSG = ("this SLURM job was given a GPU but torch.cuda.is_available() is False. The torch CUDA build "
                   "probably does not match the node's driver: see RUN_ON_ORCD.md (smoke test section).")


def now_str():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(msg):
    print(f"[{now_str()}] {msg}", flush=True)
