#!/usr/bin/env bash
# Final report (CPU). If any train/eval job of this project is still pending or running (e.g. a
# continuation or a re-queued eval), re-queue itself with afterany on those jobs and exit 0.
set -euo pipefail
source "$MOE_ROOT/slurm/env.sh"
moe_activate
cd "$MOE_ROOT"
moe_job_header
if [ "${MOE_NOWAIT:-0}" != "1" ]; then
  active="$(moe_active_kind train,eval)" || active=""
  if [ -n "$active" ]; then
    waits="${MOE_WAITS:-0}"
    if [ "$waits" -lt "$REPORT_MAX_WAITS" ]; then
      # shellcheck disable=SC2086
      dep="$(moe_dep_list afterany $active)"
      echo "train/eval jobs still active ($active): re-queueing the report with --dependency=$dep"
      if jid="$(MOE_NEXT_WAITS="$((waits + 1))" MOE_NOTE="report waiting for $active" moe_submit report "" "$dep" report.sbatch)"; then
        echo "submitted report job $jid"; exit 0
      fi
      echo "WARNING: could not submit the waiting report job; building the report now instead"
    else
      echo "WARNING: the report has waited $waits times already; building it from what is finished now"
    fi
  fi
fi
python -u src/status.py
python -u src/report.py
echo "=== report: $MOE_WORK/results/REPORT.md"
