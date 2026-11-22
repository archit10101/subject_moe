#!/usr/bin/env bash
# Correctness tests (the job fails if any fails), then a 25-step throughput bench -> results/config.json.
# bench never overwrites a valid config.json once training has started.
set -euo pipefail
source "$MOE_ROOT/slurm/env.sh"
moe_activate
cd "$MOE_ROOT"
moe_job_header
python -u src/tests.py
python -u src/train.py --bench --total_tokens "$TOTAL_TOKENS" --micro_bs "$MICRO_BS" --grad_accum "$GRAD_ACCUM"
