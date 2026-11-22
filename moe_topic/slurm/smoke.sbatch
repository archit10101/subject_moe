#!/usr/bin/env bash
# Smoke test: one GPU job (~25 min), everything under $MOE_ROOT/smoke/ (cannot touch real outputs).
# GPU check -> tests -> synthetic prep -> bench at the REAL model size -> ~1.5 min of each mode (R1 with a
# forced checkpoint + resume) -> eval with every metric -> report -> sbatch --test-only from a compute node.
set -euo pipefail
source "$MOE_ROOT/slurm/env.sh"
moe_activate
cd "$MOE_ROOT"
case "$MOE_WORK" in */smoke) ;; *) echo "refusing: MOE_WORK must end in /smoke (got $MOE_WORK)"; exit 1 ;; esac
moe_job_header
WARN=""
# MOE_SMOKE_LOCAL=1 (local testing only): no CUDA requirement, tiny model, small data
if [ "${MOE_SMOKE_LOCAL:-0}" = "1" ]; then
  PREP_ARGS="--syn_train_tokens 2e6 --nat_val_tokens 1e5 --bal_tokens 4e3 --abl_tokens 2e3"
  BENCH_ARGS="--tiny --micro_bs 2 --grad_accum 2 --bench_steps 3 --bench_warmup 1 --force_steps 24 --quick_val_tokens 1e4"
  EVAL_ARGS="--batch 4 --max_tokens 3e4 --abl_max_tokens 1e4 --min_word_count 100"
  PAUSE_MIN=0.05
else
  PREP_ARGS="--syn_train_tokens 40e6 --nat_val_tokens 2e6 --bal_tokens 4e4 --abl_tokens 5e3"
  BENCH_ARGS="--bench_steps 10 --bench_warmup 3 --minutes_per_run 1.5 --quick_val_tokens 5e5 --micro_bs $MICRO_BS --grad_accum $GRAD_ACCUM"
  EVAL_ARGS="--batch $EVAL_BATCH --min_word_count 200"
  PAUSE_MIN=0.75
fi
step() { echo; echo "=== [smoke $(date '+%T')] $*"; }

step "python / torch / CUDA"
python - <<'PY'
import os, sys, torch
print("python", sys.version.split()[0], "torch", torch.__version__, "cuda build", torch.version.cuda)
ok = torch.cuda.is_available()
print("cuda available:", ok)
if not ok and os.environ.get("MOE_SMOKE_LOCAL") == "1":
    print("(local smoke test: running on CPU)"); sys.exit(0)
if not ok:
    print("CUDA UNAVAILABLE: reinstall torch for this node's driver, see RUN_ON_ORCD.md"); sys.exit(1)
print("device:", torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16); (x @ x).sum().item()
PY

step "correctness tests"
python -u src/tests.py

step "synthetic data"
rm -rf "$MOE_WORK/data" "$MOE_WORK/ckpt" "$MOE_WORK/results"
mkdir -p "$MOE_WORK/logs"
rm -f "$MOE_WORK"/logs/*.jsonl
# shellcheck disable=SC2086
python -u src/prep_data.py --synthetic --force $PREP_ARGS

step "tiktoken offline (used by eval examples)"
python -c "import tiktoken; assert tiktoken.get_encoding('gpt2').encode(' the') == [262]; print('tiktoken OK')"

step "bench at the real model size"
# shellcheck disable=SC2086
python -u src/train.py --bench $BENCH_ARGS | tee "$MOE_WORK/logs/bench.out"

step "R1 with a forced checkpoint after $PAUSE_MIN min, then resume"
set +e
python -u src/train.py --tag R1 --mode r1 --init_seed 0 --max_minutes "$PAUSE_MIN"
rc=$?
set -e
if [ "$rc" -ne 3 ] && [ "$rc" -ne 0 ]; then echo "R1 first segment failed with rc=$rc"; exit 1; fi
[ "$rc" -eq 3 ] || WARN="$WARN R1-finished-before-forced-pause"
python -u src/train.py --tag R1 --mode r1 --init_seed 0
for spec in "R1f r1f" "P1 p1" "P2 p2" "P3 p3" "P4 p4"; do
  set -- $spec
  step "train $1"
  python -u src/train.py --tag "$1" --mode "$2" --init_seed 0
done

step "status"
python -u src/status.py

for t in R1 R1f P1 P2 P3 P4; do
  step "eval $t"
  # shellcheck disable=SC2086
  python -u src/eval.py --tag "$t" $EVAL_ARGS
done
step "report"
python -u src/report.py

step "sbatch from a compute node (--test-only; continuations, retries and waits need this)"
if MOE_TEST_ONLY=1 moe_submit train R1 "" train.sbatch R1 r1 0; then :; else
  WARN="$WARN sbatch-from-compute-node-failed(self-continuation would not work)"
fi

python - <<'PY'
import json, os
w = os.environ["MOE_WORK"]
c = json.load(open(os.path.join(w, "results", "config.json")))
print()
print("SMOKE TEST PASSED")
print(f"GPU: {c['gpu']}")
print("bench tokens/sec: " + ", ".join(f"{m}: {r['tps']:,.0f}" for m, r in c["bench"].items()))
mem = {m: r["peak_mem_gib"] for m, r in c["bench"].items() if r["peak_mem_gib"]}
if mem:
    print("peak GPU memory (GiB): " + ", ".join(f"{m}: {v:.1f}" for m, v in mem.items()))
tr = {}
for t in ("R1", "R1f", "P1", "P2", "P3", "P4"):
    recs = [json.loads(l) for l in open(os.path.join(w, "logs", f"{t}.jsonl")) if '"train"' in l]
    if recs:
        tr[t] = recs[-1]["tok_s"]
print("training tokens/sec (last log window): " + ", ".join(f"{t}: {v:,.0f}" for t, v in tr.items()))
print("estimated minutes per real run (1.2B tokens): " + ", ".join(
    f"{t}: {1.2e9 / v / 60:.0f}" for t, v in tr.items()))
PY
if [ -n "$WARN" ]; then echo "WARNINGS:$WARN"; fi
echo "report: $MOE_WORK/results/REPORT.md"
