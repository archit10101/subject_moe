#!/usr/bin/env bash
# Subject labels: MiniLM embeddings (IDF fallback) -> balanced spherical k-means, K=8.
set -euo pipefail
source "$MOE_ROOT/slurm/env.sh"
moe_activate
cd "$MOE_ROOT"
moe_job_header
python -u src/clusters.py --prep_threads "$(( ${SLURM_CPUS_PER_TASK:-8} - 1 ))"
