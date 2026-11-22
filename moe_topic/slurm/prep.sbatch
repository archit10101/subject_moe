#!/usr/bin/env bash
# Tokenize train + eval shards, build per-token metadata and the eval sets (CPU job on mit_normal).
set -euo pipefail
source "$MOE_ROOT/slurm/env.sh"
moe_activate
cd "$MOE_ROOT"
moe_job_header
python -u src/prep_data.py --force --threads "${SLURM_CPUS_PER_TASK:-16}" --nat_val_tokens "$NAT_VAL_TOKENS" \
  --bal_tokens "$BAL_TOKENS_PER_TOPIC" --abl_tokens "$ABL_TOKENS_PER_TOPIC"
python -u src/status.py
