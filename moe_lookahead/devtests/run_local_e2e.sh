#!/usr/bin/env bash
# Local end-to-end on synthetic data with the tiny model on CPU (~10-15 min):
# prep --synthetic -> clusters (IDF fallback) -> bench --tiny -> R1, R3, R2, R4 (~100 steps) -> eval -> report
#   devtests/run_local_e2e.sh <work dir>      (python with the requirements must be on PATH)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MOE_ROOT="$(cd "$HERE/.." && pwd)"
export MOE_WORK="${1:?work dir}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$MOE_ROOT/cache/tiktoken}" HF_HOME="${HF_HOME:-$MOE_ROOT/cache/hf}"
cd "$MOE_ROOT"
rm -rf "$MOE_WORK"; mkdir -p "$MOE_WORK"
t0=$(date +%s)
python src/tests.py
python src/prep_data.py --synthetic --train_tokens 3e6 --syn_val_tokens 3e5
python src/clusters.py --featurizer idf --sample 20000
python src/train.py --bench --tiny --micro_bs 2 --grad_accum 2 --bench_steps 8 --bench_warmup 3 \
  --force_steps 100 --force_r4_steps 50 --quick_val_tokens 3e4
for spec in "R1 r1" "R3 r3" "R2 r2" "R4 r4"; do
  set -- $spec
  python src/train.py --tag "$1" --mode "$2" --init_seed 0 --log_every 10
done
python src/status.py
python src/eval.py --batch 4 --max_val_tokens 1e5
python src/report.py
echo "local e2e finished in $(( $(date +%s) - t0 ))s; report: $MOE_WORK/results/REPORT.md"
