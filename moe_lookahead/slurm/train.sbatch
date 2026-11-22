#!/usr/bin/env bash
# One training run: train.sbatch <tag> <mode> <init_seed>. Auto-resumes from ckpt/<tag>_last.pt.
#   train.py exit 0 -> done | 3 -> hit TRAIN_MAX_MINUTES: submit a continuation of itself, exit 0
#   2 -> config/staleness error (no retry) | other -> crash: resubmit once excluding this node, exit rc
set -euo pipefail
source "$MOE_ROOT/slurm/env.sh"
moe_activate
cd "$MOE_ROOT"
moe_job_header
TAG="$1"; MODE="$2"; SEED="$3"
set +e
python -u src/train.py --tag "$TAG" --mode "$MODE" --init_seed "$SEED" \
  --max_minutes "$TRAIN_MAX_MINUTES" --ckpt_minutes "$CKPT_MINUTES"
rc=$?
set -e
echo "=== $(date '+%F %T') train.py exited with $rc"
case "$rc" in
  0) exit 0 ;;
  3)
    jid="$(MOE_NEXT_ATTEMPT="${MOE_ATTEMPT:-0}" MOE_NOTE="continuation of ${SLURM_JOB_ID:-?}" \
           moe_submit train "$TAG" "" train.sbatch "$TAG" "$MODE" "$SEED")" || { echo "ERROR: could not submit continuation"; exit 1; }
    echo "submitted continuation job $jid"
    exit 0 ;;
  2) echo "configuration error: not retrying"; exit 2 ;;
  143) echo "stopped by SIGTERM (checkpoint saved); not resubmitting"; exit 143 ;;
  *)
    attempt="${MOE_ATTEMPT:-0}"
    if [ "$attempt" -lt "$TRAIN_MAX_RETRIES" ]; then
      node="${SLURMD_NODENAME:-$(hostname -s)}"
      excl="${MOE_EXCLUDE:-}"
      if [ -n "$excl" ]; then excl="$excl,$node"; else excl="$node"; fi
      jid="$(MOE_EXCLUDE="$excl" MOE_NEXT_ATTEMPT="$((attempt + 1))" \
             MOE_NOTE="retry $((attempt + 1)) after rc=$rc on $node (job ${SLURM_JOB_ID:-?})" \
             moe_submit train "$TAG" "" train.sbatch "$TAG" "$MODE" "$SEED")" && echo "submitted retry job $jid (excluding $excl)"
    else
      echo "retries exhausted ($attempt/$TRAIN_MAX_RETRIES)"
    fi
    exit "$rc" ;;
esac
