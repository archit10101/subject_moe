#!/usr/bin/env bash
# Progress overview (login node, prints only): this project's queue, done/partial/missing per run and
# eval, latest log line + ETA of running runs, failed jobs with the path to their .out file.
#   ./status.sh [--smoke]
set -euo pipefail
MOE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MOE_ROOT
source "$MOE_ROOT/config.sh"
if [ "${1:-}" = "--smoke" ]; then export MOE_WORK="$MOE_ROOT/smoke"; JOB_PREFIX="${JOB_PREFIX}smoke_"; else export MOE_WORK="$MOE_ROOT"; fi
export JOB_PREFIX
source "$MOE_ROOT/slurm/env.sh"
moe_activate
python "$MOE_ROOT/src/status.py" --live --prefix "$JOB_PREFIX"
if [ "${1:-}" = "--smoke" ]; then
  f="$(ls -t "$MOE_WORK"/logs/slurm_"${JOB_PREFIX}"smoke_*.out 2>/dev/null | head -1 || true)"
  if [ -n "$f" ]; then
    echo "latest smoke log: $f"; tail -n 8 "$f"
    if grep -q "SMOKE TEST PASSED" "$f"; then echo ">>> SMOKE TEST PASSED"; fi
  fi
fi
if [ -f "$MOE_WORK/results/REPORT.md" ]; then echo "report: $MOE_WORK/results/REPORT.md"; fi
