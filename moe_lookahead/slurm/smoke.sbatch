#!/usr/bin/env bash
# Smoke test (one GPU job, ~20-25 min, everything under $MOE_ROOT/smoke/; cannot touch real outputs).
# tests -> synthetic prep (~40M tokens) -> IDF clustering -> MiniLM offline check -> bench at the REAL
# model size -> ~1.5 min each of R1 (with a forced checkpoint/resume), R3, R2, R4 -> eval -> report.
set -euo pipefail
source "$MOE_ROOT/slurm/env.sh"
moe_activate
cd "$MOE_ROOT"
case "$MOE_WORK" in */smoke) ;; *) echo "refusing: MOE_WORK must end in /smoke (got $MOE_WORK)"; exit 1 ;; esac
moe_job_header
WARN=""
# MOE_SMOKE_LOCAL=1 (local testing only): no CUDA requirement, tiny model, small data
if [ "${MOE_SMOKE_LOCAL:-0}" = "1" ]; then
  SMOKE_TOKENS=3e6; SMOKE_VAL=3e5
  BENCH_EXTRA="--tiny --micro_bs 2 --grad_accum 2 --bench_steps 4 --bench_warmup 1 --force_steps 30 --force_r4_steps 10 --quick_val_tokens 2e4"
  EVAL_EXTRA="--batch 4 --max_val_tokens 5e4"
else
  SMOKE_TOKENS=40e6; SMOKE_VAL=2e6
  BENCH_EXTRA="--target_minutes 1.5 --r4_minutes 1.5 --quick_val_tokens 5e5"
  EVAL_EXTRA=""
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
python -u src/prep_data.py --synthetic --train_tokens "$SMOKE_TOKENS" --syn_val_tokens "$SMOKE_VAL" --force

step "clusters (IDF fallback featurizer)"
python -u src/clusters.py --featurizer idf --prep_threads "$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))"

step "MiniLM offline load on this node (non-fatal)"
if python -u src/clusters.py --selftest_minilm "${SMOKE_MINILM_BLOCKS:-20000}" --prep_threads "$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))"; then :; else
  WARN="$WARN MiniLM-failed(real run would use the IDF fallback)"
fi

step "bench at the real model size"
python -u src/train.py --bench $BENCH_EXTRA | tee "$MOE_WORK/logs/bench.out"

step "R1 with a forced checkpoint after 0.75 min, then resume"
set +e
python -u src/train.py --tag R1 --mode r1 --init_seed 0 --max_minutes 0.75
rc=$?
set -e
if [ "$rc" -ne 3 ] && [ "$rc" -ne 0 ]; then echo "R1 first segment failed with rc=$rc"; exit 1; fi
[ "$rc" -eq 3 ] || WARN="$WARN R1-finished-before-forced-pause"
python -u src/train.py --tag R1 --mode r1 --init_seed 0
for spec in "R3 r3" "R2 r2" "R4 r4"; do
  set -- $spec
  step "train $1"
  python -u src/train.py --tag "$1" --mode "$2" --init_seed 0
done

step "status"
python -u src/status.py

step "eval + report"
python -u src/eval.py $EVAL_EXTRA
python -u src/report.py

step "sbatch from a compute node (--test-only; continuations/retries need this)"
if MOE_TEST_ONLY=1 moe_submit train R1 "" train.sbatch R1 r1 0; then :; else
  WARN="$WARN sbatch-from-compute-node-failed(self-continuation would not work)"
fi

python - <<'PY'
import json, os
c = json.load(open(os.path.join(os.environ["MOE_WORK"], "results", "config.json")))
tps = c["tps"]
proj = tps * 120 * 60
print()
print("SMOKE TEST PASSED")
print(f"GPU: {c['gpu']}")
print(f"measured tokens/sec (min of R1, R3): {tps:,.0f}   (R1 {c['bench']['r1']['tps']:,.0f}, R3 {c['bench']['r3']['tps']:,.0f})")
mem = [c["bench"][m]["peak_mem_gib"] for m in ("r1", "r3")]
if None not in mem:
    print(f"peak GPU memory in bench: R1 {mem[0]:.1f} GiB, R3 {mem[1]:.1f} GiB")
print(f"projected tokens per 2-hour run: {proj / 1e9:.2f}B  (the real run is capped at the 1.2B tokenized tokens"
      f"{' -> runs will be shorter than 2 h' if proj > 1.2e9 else ''})")
PY
if [ -n "$WARN" ]; then echo "WARNINGS:$WARN"; fi
echo "report: $MOE_WORK/results/REPORT.md"
