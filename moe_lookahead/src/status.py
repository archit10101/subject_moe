"""Which stages / runs are finished, based only on files (no torch import: fast on the login node).

  python src/status.py                  human summary (add --live for squeue/sacct info)
  python src/status.py --shell          KEY=VALUE lines for submit_all.sh
  python src/status.py --runs [--max_tier N]   "tag mode seed tier" lines in priority order
  python src/status.py --active --prefix P [--kind train]   ids of this project's pending/running jobs
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RUN_INFO, RUNS, Paths, read_json  # noqa: E402

ACTIVE_STATES = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "REQUEUED", "RESIZING", "SUSPENDED",
                 "REQUEUE_HOLD", "REQUEUE_FED"}
BAD_STATES = ("FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED", "BOOT_FAIL", "DEADLINE",
              "PREEMPTED")


def file_ok(path, size):
    return os.path.exists(path) and os.path.getsize(path) == size


def collect(paths):
    st = dict(stale=None)
    man = read_json(os.path.join(paths.raw, "manifest.json"))
    st["download"] = "done" if man and man.get("complete") else "missing"
    meta = read_json(paths.meta)
    if meta and all(file_ok(paths.tokens(s), 2 * meta[s]["n_tokens"]) and file_ok(paths.src(s), meta[s]["n_tokens"])
                    for s in ("train", "val")):
        st["prep"] = "done"
    else:
        st["prep"] = "missing"
        meta = None
    clus = read_json(paths.clusters_json)
    if meta and clus and clus.get("data_fingerprint") == meta["fingerprint"] and all(
            os.path.exists(paths.cluster(s)) for s in ("train", "val")):
        st["clusters"] = "done"
    else:
        st["clusters"] = "stale" if clus and meta else "missing"
    conf = read_json(paths.config)
    chash = conf.get("config_hash") if conf else None
    if conf and meta and conf.get("data_fingerprint") == meta["fingerprint"]:
        st["bench"] = "done"
    elif conf and meta:
        st["bench"] = "stale"
    else:
        st["bench"] = "missing"
    runs = {}
    any_ckpt = False
    for tag in RUN_INFO:
        side = read_json(paths.ckpt_side(tag))
        has_pt = os.path.exists(paths.ckpt_last(tag))
        any_ckpt = any_ckpt or has_pt or side is not None
        if side is None and not has_pt:
            runs[tag] = dict(state="missing")
        elif side is None:
            runs[tag] = dict(state="partial", step=0, total_steps=None)
        elif chash and side.get("config_hash") != chash:
            runs[tag] = dict(state="stale", step=side.get("step"), total_steps=side.get("total_steps"))
        elif side.get("done") and has_pt:
            runs[tag] = dict(state="done", step=side["step"], total_steps=side["total_steps"])
        else:
            runs[tag] = dict(state="partial", step=side.get("step"), total_steps=side.get("total_steps"))
    st["runs"] = runs
    any_ckpt = any_ckpt or os.path.exists(os.path.join(paths.ckpt, "R1_r4ref.pt"))
    # staleness: never silently reuse checkpoints made under different settings / data
    if any_ckpt and conf is None:
        st["stale"] = "checkpoints exist in ckpt/ but results/config.json does not"
    elif any_ckpt and meta is None:
        st["stale"] = "checkpoints exist but the token streams (data/meta.json) are missing or incomplete"
    elif any_ckpt and st["bench"] == "stale":
        st["stale"] = "results/config.json was made from different data than data/meta.json"
    elif any_ckpt and st["clusters"] != "done":
        st["stale"] = "checkpoints exist but the cluster labels are missing or were made from different data"
    elif any(r["state"] == "stale" for r in runs.values()):
        bad = [t for t, r in runs.items() if r["state"] == "stale"]
        st["stale"] = f"checkpoints {bad} were trained under a different results/config.json"
    report = os.path.join(paths.results, "REPORT.md")
    done_times = [os.path.getmtime(paths.ckpt_side(t)) for t, r in runs.items() if r["state"] == "done"]
    if os.path.exists(report) and (not done_times or os.path.getmtime(report) >= max(done_times)):
        st["report"] = "done"
    else:
        st["report"] = "missing"
    st["synthetic"] = bool(meta and meta.get("synthetic"))
    return st


def last_record(path, kind="train"):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 65536))
            lines = f.read().decode(errors="replace").splitlines()
    except OSError:
        return None
    for line in reversed(lines):
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("type") == kind:
            return r
    return None


def squeue_jobs():
    """[(id, name, state)] for the current user, or [] if squeue is unavailable."""
    try:
        out = subprocess.run(["squeue", "--me", "-h", "-o", "%i|%j|%T"], capture_output=True, text=True,
                             timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    jobs = []
    for line in out.stdout.splitlines():
        parts = line.strip().split("|")
        if len(parts) >= 3:
            jobs.append((parts[0], parts[1], parts[2]))
    return jobs


def ours(name, prefix):
    """Job belongs to this project view (the real project excludes the smoke test's jobs)."""
    if not name.startswith(prefix):
        return False
    return prefix.endswith("smoke_") or not name.startswith(prefix + "smoke_")


def active_jobs(prefix, kind=None):
    me = os.environ.get("SLURM_JOB_ID")
    res = []
    for jid, name, state in squeue_jobs():
        if not ours(name, prefix) or state not in ACTIVE_STATES or jid == me:
            continue
        stage = name[len(prefix):]
        if kind and not (stage == kind or stage.startswith(kind + "_")):
            continue
        res.append((jid, name, state))
    return res


def read_jobs_tsv(paths):
    rows = []
    if os.path.exists(paths.jobs_tsv):
        with open(paths.jobs_tsv) as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 4 and p[0] != "time":
                    rows.append(dict(time=p[0], jobid=p[1], name=p[2], stage=p[3], tag=p[4] if len(p) > 4 else "",
                                     dep=p[5] if len(p) > 5 else "", note=p[6] if len(p) > 6 else ""))
    return rows


def sacct_states(ids):
    if not ids:
        return {}
    try:
        out = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(ids), "--format=JobID,JobName,State,ExitCode,NodeList"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return {}
    if out.returncode != 0:
        return {}
    res = {}
    for line in out.stdout.splitlines():
        p = line.split("|")
        if len(p) >= 4:
            res[p[0]] = dict(name=p[1], state=p[2], exit=p[3], node=p[4] if len(p) > 4 else "")
    return res


def human(paths, st, live, prefix):
    print(f"project work dir: {paths.work}" + ("   [SYNTHETIC DATA]" if st["synthetic"] else ""))
    print(f"  download  {st['download']}")
    for k in ("prep", "clusters", "bench", "report"):
        print(f"  {k:9s} {st[k]}")
    conf = read_json(paths.config) or {}
    if conf:
        print(f"  config: {conf.get('total_steps')} steps/run ({conf.get('total_tokens', 0) / 1e6:.0f}M tokens), "
              f"R4 {conf.get('r4_steps')} steps; bench {conf.get('tps', 0):,.0f} tok/s on {conf.get('gpu')}")
    if st["stale"]:
        print(f"  !! STALE: {st['stale']}")
    print("runs:")
    for tag, mode, seed, tier in RUNS:
        r = st["runs"][tag]
        s = f"  {tag:9s} tier {tier} {r['state']:8s}"
        if r.get("step") is not None and r["state"] != "done":
            s += f" ckpt step {r['step']}/{r.get('total_steps')}"
        rec = last_record(paths.log(tag))
        if rec and r["state"] != "done":
            age = (time.time() - rec.get("time", time.time())) / 60
            s += (f" | log step {rec['step']} loss {rec['loss']:.4f} {rec.get('tok_s', 0):,.0f} tok/s "
                  f"ETA {rec.get('eta_min', 0):.0f} min (updated {age:.0f} min ago)")
        elif r["state"] == "done":
            v = last_record(paths.log(tag), "val")
            if v:
                s += f" | final quick-val {v['val_loss']:.4f}"
        print(s)
    if not live:
        return
    print("queue (this project):")
    act = [j for j in squeue_jobs() if ours(j[1], prefix)]
    if not act:
        print("  (no pending/running jobs)")
    for jid, name, state in act:
        print(f"  {jid:>10s} {state:10s} {name}")
    rows = read_jobs_tsv(paths)
    states = sacct_states([r["jobid"] for r in rows])
    bad = [(r, states[r["jobid"]]) for r in rows if r["jobid"] in states and
           states[r["jobid"]]["state"].split()[0].startswith(BAD_STATES)]
    if bad:
        print("failed / cancelled jobs:")
        for r, s in bad:
            print(f"  {r['jobid']:>10s} {s['state']:12s} exit {s['exit']:6s} {r['name']} node {s['node']}\n"
                  f"             log: {os.path.join(paths.logs, 'slurm_' + r['name'] + '_' + r['jobid'] + '.out')}")
    elif rows and not states:
        print("(sacct unavailable: cannot list failed jobs)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    ap.add_argument("--shell", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--runs", action="store_true")
    ap.add_argument("--max_tier", type=int, default=3)
    ap.add_argument("--active", action="store_true")
    ap.add_argument("--kind", default=None)
    ap.add_argument("--prefix", default=os.environ.get("JOB_PREFIX", "mla_"))
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--long", action="store_true", help="with --active: 'id|name|state' lines")
    args = ap.parse_args()
    paths = Paths(args.work)
    if args.runs:
        for tag, mode, seed, tier in RUNS:
            if tier <= args.max_tier:
                print(tag, mode, seed, tier)
        return
    if args.active:
        jobs = active_jobs(args.prefix, args.kind)
        if args.long:
            for j in jobs:
                print("|".join(j))
        else:
            print(" ".join(j[0] for j in jobs))
        return
    st = collect(paths)
    if args.json:
        print(json.dumps(st, indent=2))
    elif args.shell:
        for k in ("download", "prep", "clusters", "bench", "report"):
            print(f"ST_{k.upper()}={st[k]}")
        for tag, r in st["runs"].items():
            print(f"ST_RUN_{tag}={r['state']}")
        print("ST_STALE=" + ("'" + st["stale"].replace("'", "") + "'" if st["stale"] else "''"))
        print(f"ST_SYNTHETIC={int(st['synthetic'])}")
    else:
        human(paths, st, args.live, args.prefix)


if __name__ == "__main__":
    main()
