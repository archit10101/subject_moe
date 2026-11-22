#!/usr/bin/env bash
# Evaluate every finished run + write the report. If training jobs of this project are still
# pending/running (e.g. continuations), re-queue itself behind them (afterany) and exit 0.
set -euo pipefail
source "$MOE_ROOT/slurm/env.sh"
moe_activate
cd "$MOE_ROOT"
moe_job_header
if [ "${MOE_EVAL_NOWAIT:-0}" != "1" ]; then
  active="$(moe_active train)" || active=""
  if [ -n "$active" ]; then
    waits="${MOE_EVAL_WAITS:-0}"
    if [ "$waits" -lt "$EVAL_MAX_WAITS" ]; then
      # shellcheck disable=SC2086
      dep="$(moe_dep_list afterany $active)"
      echo "training jobs still active ($active): re-queueing eval with --dependency=$dep"
      if jid="$(MOE_NEXT_EVAL_WAITS="$((waits + 1))" MOE_NOTE="eval waiting for $active" moe_submit eval "" "$dep" eval.sbatch)"; then
        echo "submitted eval job $jid"; exit 0
      fi
      echo "WARNING: could not submit the waiting eval job; evaluating now instead"
    else
      echo "WARNING: eval has waited $waits times already; evaluating what is finished now"
    fi
  fi
fi
python -u src/status.py
set +e
python -u src/eval.py --batch "$EVAL_BATCH" ${EVAL_MAX_VAL_TOKENS:+--max_val_tokens "$EVAL_MAX_VAL_TOKENS"}
rc=$?
python -u src/report.py
rc2=$?
set -e
echo "=== report: $MOE_WORK/results/REPORT.md"
if [ "$rc" -ne 0 ] || [ "$rc2" -ne 0 ]; then exit 1; fi
