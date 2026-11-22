#!/usr/bin/env bash
# Local end-to-end on synthetic data with the tiny model on CPU (resumable; ~40 min on 4 cores):
# tests -> prep --synthetic -> bench --tiny -> all 9 runs (100 steps) -> eval (every metric, small sets)
# -> report -> status.   devtests/run_local_e2e.sh <work dir> [--fresh]
# (python with the requirements must be on PATH)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MOE_ROOT="$(cd "$HERE/.." && pwd)"
export MOE_WORK="${1:?work dir}"
export TIKTOKEN_CACHE_DIR="${TIKTOKEN_CACHE_DIR:-$MOE_ROOT/cache/tiktoken}" HF_HOME="${HF_HOME:-$MOE_ROOT/cache/hf}"
cd "$MOE_ROOT"
if [ "${2:-}" = "--fresh" ]; then rm -rf "$MOE_WORK"; fi
mkdir -p "$MOE_WORK"
t0=$(date +%s)
python src/tests.py
python src/prep_data.py --synthetic --syn_train_tokens 3e6 --nat_val_tokens 2e5 --bal_tokens 8e3 --abl_tokens 2e3
python src/train.py --bench --tiny --micro_bs 2 --grad_accum 2 --bench_steps 4 --bench_warmup 1 \
  --force_steps 100 --quick_val_tokens 1e4
python src/status.py --runs | while read -r tag mode seed; do
  python src/train.py --tag "$tag" --mode "$mode" --init_seed "$seed" --log_every 10 </dev/null
done
python src/status.py
python src/status.py --runs | while read -r tag mode seed; do
  if python src/status.py --json | python -c "import json,sys; sys.exit(0 if json.load(sys.stdin)['evals']['$tag']=='done' else 1)"; then
    echo "eval $tag already done"; continue
  fi
  python src/eval.py --tag "$tag" --batch 4 --max_tokens 6e4 --abl_max_tokens 1.2e4 --min_word_count 100 </dev/null
done
python src/report.py
python src/status.py
echo "local e2e finished in $(( $(date +%s) - t0 ))s; report: $MOE_WORK/results/REPORT.md"
