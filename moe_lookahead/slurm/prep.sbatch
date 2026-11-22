#!/usr/bin/env bash
# Tokenize parquet -> flat uint16 token streams (CPU job on mit_normal). Resources come from submit_all.sh.
set -euo pipefail
source "$MOE_ROOT/slurm/env.sh"
moe_activate
cd "$MOE_ROOT"
moe_job_header
python -u src/prep_data.py --force --train_tokens "$TRAIN_TOKENS" --threads "${SLURM_CPUS_PER_TASK:-16}"
python -u src/status.py
