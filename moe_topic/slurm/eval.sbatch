#!/usr/bin/env bash
# Evaluate one run: eval.sbatch <tag>. If the run's checkpoint is not done (e.g. a continuation is still
# queued), re-queue itself with afterany on that run's active train job(s) and exit 0.
set -euo pipefail
source "$MOE_ROOT/slurm/env.sh"
moe_activate
cd "$MOE_ROOT"
moe_job_header
TAG="$1"
state="$(python src/status.py --run_state "$TAG")"
if [ "$state" != "done" ]; then
  active="$(moe_active_name "$(moe_job_name train "$TAG")")" || active=""
  waits="${MOE_WAITS:-0}"
  if [ -n "$active" ] && [ "$waits" -lt "$EVAL_MAX_WAITS" ]; then
    # shellcheck disable=SC2086
    dep="$(moe_dep_list afterany $active)"
    echo "$TAG is $state; its train job(s) $active are active: re-queueing eval with --dependency=$dep"
    jid="$(MOE_NEXT_WAITS="$((waits + 1))" MOE_NOTE="eval $TAG waiting for $active" \
           moe_submit eval "$TAG" "$dep" eval.sbatch "$TAG")" || { echo "ERROR: could not re-queue"; exit 1; }
    echo "submitted eval job $jid"
    exit 0
  fi
  echo "ERROR: $TAG is $state and no train job for it is active (waits=$waits); nothing to evaluate."
  echo "       Re-run ./submit_all.sh to resume the run."
  exit 5
fi
# shellcheck disable=SC2086
python -u src/eval.py --tag "$TAG" --batch "$EVAL_BATCH" $EVAL_EXTRA
