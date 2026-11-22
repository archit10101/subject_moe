#!/usr/bin/env bash
# Submit whatever is missing, wired with dependencies. Safe to re-run at any time (idempotent):
# finished stages/runs are skipped, queued/running jobs are reused, unfinished runs resume.
#
#   ./submit_all.sh               prep -> clusters -> bench -> train (tiers 1+2) -> eval
#   ./submit_all.sh --smoke       only the smoke test, in ./smoke/ (never touches real outputs)
#   ./submit_all.sh --more-seeds  also tier 3 (R3_seed1, R2_seed1)
#   ./submit_all.sh --report-now  just an eval/report job for whatever has finished (no waiting)
set -euo pipefail
MOE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MOE_ROOT

SMOKE=0; MORE=0; REPORT_NOW=0
for a in "$@"; do
  case "$a" in
    --smoke) SMOKE=1 ;;
    --more-seeds) MORE=1 ;;
    --report-now) REPORT_NOW=1 ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "unknown option: $a" >&2; exit 2 ;;
  esac
done

source "$MOE_ROOT/config.sh"
if [ "$SMOKE" = "1" ]; then
  export MOE_WORK="$MOE_ROOT/smoke"
  JOB_PREFIX="${JOB_PREFIX}smoke_"
else
  export MOE_WORK="$MOE_ROOT"
fi
export JOB_PREFIX
source "$MOE_ROOT/slurm/env.sh"
moe_activate
mkdir -p "$MOE_WORK/logs"

die() { echo "ERROR: $*" >&2; exit 1; }
command -v sbatch >/dev/null 2>&1 || die "sbatch not found: run this on the ORCD login node"
command -v python >/dev/null 2>&1 || die "python not found (did setup_login.sh create the conda env?)"
[ -n "$GPU_TYPE" ] || die "GPU_TYPE is empty in config.sh: an untyped GPU request gets a slow L40S"
[ -d "$MOE_ROOT/cache/tiktoken" ] || die "no tiktoken cache in $MOE_ROOT/cache: run ./setup_login.sh first"

# Pre-flight: let the scheduler validate partition / GPU type / account without submitting anything.
if ! MOE_TEST_ONLY=1 moe_submit bench "" "" bench.sbatch >/dev/null 2>&1; then
  echo "sbatch --test-only rejected the GPU job flags; details:" >&2
  MOE_TEST_ONLY=1 moe_submit bench "" "" bench.sbatch || true
  die "fix GPU_PARTITION / GPU_TYPE / MOE_ACCOUNT in config.sh"
fi

ACTIVE="$(python "$MOE_ROOT/src/status.py" --active --long --prefix "$JOB_PREFIX")"
active_id() {  # job name -> id of an active job with that name ("" if none)
  echo "$ACTIVE" | awk -F'|' -v n="$1" '$2 == n { print $1; exit }'
}
SUBMITTED=""
note() { printf '  %-22s %s\n' "$1" "$2"; }

# ------------------------------------------------------------------ smoke test
if [ "$SMOKE" = "1" ]; then
  a="$(active_id "$(moe_job_name smoke)")"
  [ -z "$a" ] || die "a smoke test is already queued/running (job $a)"
  jid="$(moe_submit smoke "" "" smoke.sbatch)"
  echo "submitted smoke test: job $jid"
  echo "  log: $MOE_WORK/logs/slurm_$(moe_job_name smoke)_${jid}.out"
  echo "  it should end with 'SMOKE TEST PASSED' (about 20-25 min after it starts)"
  exit 0
fi

eval "$(python "$MOE_ROOT/src/status.py" --shell)"
if [ -n "$ST_STALE" ]; then
  cat >&2 <<MSG
ERROR: refusing to submit: $ST_STALE
Stale checkpoints must never be reused silently. If you really want to start the runs over:
    mv $MOE_WORK/ckpt $MOE_WORK/ckpt_old_\$(date +%s); mv $MOE_WORK/logs $MOE_WORK/logs_old_\$(date +%s)
and re-run ./submit_all.sh. (If you changed the data on purpose, also delete results/config.json.)
MSG
  exit 1
fi

# ------------------------------------------------------------------ report only
if [ "$REPORT_NOW" = "1" ]; then
  [ "$ST_BENCH" = "done" ] || die "no results/config.json yet: nothing to evaluate"
  jid="$(MOE_EVAL_NOWAIT=1 MOE_NOTE="report-now" moe_submit eval "" "" eval.sbatch)"
  echo "submitted eval/report job $jid (evaluates whatever has finished, does not wait)"
  exit 0
fi

echo "submitting from $MOE_ROOT (GPU $GPU_TYPE on $GPU_PARTITION)"
DEP=""
for stage in prep clusters bench; do
  case "$stage" in prep) st="$ST_PREP" ;; clusters) st="$ST_CLUSTERS" ;; bench) st="$ST_BENCH" ;; esac
  name="$(moe_job_name "$stage")"
  if [ "$st" = "done" ]; then note "$stage" "done"; continue; fi
  a="$(active_id "$name")"
  if [ -n "$a" ]; then
    note "$stage" "already queued/running as job $a"
    DEP="afterok:$a"; continue
  fi
  if [ "$stage" = "prep" ] && [ "$ST_DOWNLOAD" != "done" ]; then
    die "the dataset download is incomplete: run ./setup_login.sh first"
  fi
  jid="$(moe_submit "$stage" "" "$DEP" "$stage.sbatch")"
  note "$stage" "submitted job $jid${DEP:+ (dependency $DEP)}"
  SUBMITTED="$SUBMITTED $jid"
  DEP="afterok:$jid"
done

MAXTIER=2
[ "$MORE" = "1" ] && MAXTIER=3
TRAIN_IDS=""
RUNLIST="$(python "$MOE_ROOT/src/status.py" --runs --max_tier "$MAXTIER")"
while read -r tag mode seed tier; do
  [ -n "$tag" ] || continue
  eval "st=\${ST_RUN_${tag}:-missing}"
  name="$(moe_job_name train "$tag")"
  if [ "$st" = "done" ]; then note "train $tag" "done"; continue; fi
  a="$(active_id "$name")"
  if [ -n "$a" ]; then
    note "train $tag" "already queued/running as job $a"
    TRAIN_IDS="$TRAIN_IDS $a"; continue
  fi
  jid="$(moe_submit train "$tag" "$DEP" train.sbatch "$tag" "$mode" "$seed")"
  what="submitted"; [ "$st" = "partial" ] && what="resubmitted (resumes from checkpoint)"
  note "train $tag" "$what job $jid (tier $tier)${DEP:+ (dependency $DEP)}"
  SUBMITTED="$SUBMITTED $jid"
  TRAIN_IDS="$TRAIN_IDS $jid"
done <<RUNS
$RUNLIST
RUNS

# ------------------------------------------------------------------ eval
EVAL_ACTIVE="$(echo "$ACTIVE" | awk -F'|' -v n="$(moe_job_name eval)" '$2 == n { print $1 "|" $3 }')"
if [ -n "$SUBMITTED" ] || { [ -z "$EVAL_ACTIVE" ] && [ "$ST_REPORT" != "done" ]; }; then
  # a previously queued (pending) eval is superseded by the new one
  for row in $EVAL_ACTIVE; do
    id="${row%%|*}"; state="${row#*|}"
    if [ "$state" = "PENDING" ]; then scancel "$id" && note "eval" "cancelled superseded pending eval job $id"; fi
  done
  # shellcheck disable=SC2086
  EDEP="$(moe_dep_list afterany $TRAIN_IDS)"
  if ! jid="$(moe_submit eval "" "$EDEP" eval.sbatch)"; then
    echo "WARNING: eval submission with dependency failed; submitting it without (it waits by itself)" >&2
    jid="$(moe_submit eval "" "" eval.sbatch)"
  fi
  note "eval" "submitted job $jid${EDEP:+ (dependency $EDEP)}"
  SUBMITTED="$SUBMITTED $jid"
elif [ -n "$EVAL_ACTIVE" ]; then
  note "eval" "already queued/running (${EVAL_ACTIVE%%|*})"
else
  note "eval" "report is up to date (use --report-now to rebuild it)"
fi

if [ -z "$SUBMITTED" ]; then
  echo "nothing new to submit."
else
  echo "submitted:$SUBMITTED   (all job ids are recorded in $MOE_WORK/logs/jobs.tsv)"
fi
echo "check progress with ./status.sh"
