#!/usr/bin/env bash
# Correctness tests (job fails if any fails), then throughput bench -> results/config.json.
set -euo pipefail
source "$MOE_ROOT/slurm/env.sh"
moe_activate
cd "$MOE_ROOT"
moe_job_header
python -u src/tests.py
python -u src/train.py --bench --target_minutes "$TARGET_MINUTES" --r4_minutes "$R4_MINUTES" \
  --r4_max_tokens "$R4_MAX_TOKENS"
