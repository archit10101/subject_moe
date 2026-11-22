# Job submission helpers (bash 3.2 compatible: no mapfile / assoc arrays / ${x,,}; empty arrays guarded).
# All sbatch flags are built here, so continuations, retries and re-queued waits use identical resources.

moe_now() { date '+%Y-%m-%dT%H:%M:%S'; }

moe_record() {  # jobid name stage tag dep note
  mkdir -p "$MOE_WORK/logs"
  if [ ! -f "$MOE_WORK/logs/jobs.tsv" ]; then
    printf 'time\tjobid\tname\tstage\ttag\tdependency\tnote\n' > "$MOE_WORK/logs/jobs.tsv"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(moe_now)" "$1" "$2" "$3" "$4" "$5" "$6" >> "$MOE_WORK/logs/jobs.tsv"
}

moe_job_name() {  # stage [tag]
  if [ -n "${2:-}" ]; then echo "${JOB_PREFIX}${1}_${2}"; else echo "${JOB_PREFIX}${1}"; fi
}

# moe_submit <stage> <tag> <dependency or ""> <script in slurm/> [script args...]
# Prints the job id on stdout. Optional env: MOE_NOTE, MOE_NEXT_ATTEMPT, MOE_EXCLUDE (comma list),
# MOE_NEXT_WAITS, MOE_NOWAIT, MOE_TEST_ONLY=1 (sbatch --test-only, nothing recorded).
moe_submit() {
  local stage="$1" tag="$2" dep="$3" script="$4"
  shift 4
  local name
  name="$(moe_job_name "$stage" "$tag")"
  local exp="ALL,MOE_ROOT=$MOE_ROOT,MOE_WORK=$MOE_WORK,MOE_ATTEMPT=${MOE_NEXT_ATTEMPT:-0}"
  exp="$exp,MOE_WAITS=${MOE_NEXT_WAITS:-0},MOE_NOWAIT=${MOE_NOWAIT:-0}"
  exp="$exp,MOE_EXCLUDE=${MOE_EXCLUDE:-},JOB_PREFIX=$JOB_PREFIX"
  local flags
  flags=( -J "$name" --chdir="$MOE_ROOT" -o "$MOE_WORK/logs/slurm_%x_%j.out" --export="$exp" --nodes=1 --ntasks=1 )
  local gpu
  gpu=( -p "$GPU_PARTITION" -G "${GPU_TYPE}:1" -c "$GPU_CPUS" --mem="$GPU_MEM" )
  case "$stage" in
    prep)   flags+=( -p "$CPU_PARTITION" -c "$PREP_CPUS" --mem="$PREP_MEM" -t "$PREP_TIME" ) ;;
    bench)  flags+=( "${gpu[@]}" -t "$BENCH_TIME" ) ;;
    train)  flags+=( "${gpu[@]}" -t "$TRAIN_TIME" --requeue --open-mode=append ) ;;
    eval)   flags+=( "${gpu[@]}" -t "$EVAL_TIME" ) ;;
    report) flags+=( -p "$CPU_PARTITION" -c "$REPORT_CPUS" --mem="$REPORT_MEM" -t "$REPORT_TIME" ) ;;
    smoke)  flags+=( "${gpu[@]}" -t "$SMOKE_TIME" ) ;;
    *) echo "moe_submit: unknown stage $stage" >&2; return 1 ;;
  esac
  if [ -n "$MOE_ACCOUNT" ]; then flags+=( -A "$MOE_ACCOUNT" ); fi
  if [ -n "$MAIL_USER" ]; then
    flags+=( --mail-user="$MAIL_USER" )
    if [ "$stage" = "report" ] || [ "$stage" = "smoke" ]; then flags+=( --mail-type=END,FAIL ); else flags+=( --mail-type=FAIL ); fi
  fi
  if [ -n "${MOE_EXCLUDE:-}" ] && [ "$stage" = "train" ]; then flags+=( --exclude="$MOE_EXCLUDE" ); fi
  if [ -n "$dep" ]; then flags+=( --dependency="$dep" --kill-on-invalid-dep=yes ); fi
  if [ -n "$MOE_EXTRA_SBATCH" ]; then
    local extra
    set -f; extra=( $MOE_EXTRA_SBATCH ); set +f
    flags+=( "${extra[@]}" )
  fi
  if [ "${MOE_TEST_ONLY:-0}" = "1" ]; then
    sbatch --test-only "${flags[@]}" "$MOE_ROOT/slurm/$script" "$@"
    return $?
  fi
  local out jid
  if ! out="$(sbatch --parsable "${flags[@]}" "$MOE_ROOT/slurm/$script" "$@")"; then
    echo "moe_submit: sbatch failed for $name" >&2
    return 1
  fi
  jid="${out%%;*}"
  case "$jid" in
    ''|*[!0-9_]*) echo "moe_submit: unexpected sbatch output '$out'" >&2; return 1 ;;
  esac
  moe_record "$jid" "$name" "$stage" "$tag" "$dep" "${MOE_NOTE:-}"
  echo "$jid"
}

# Ids of this project's active (pending/running) jobs, space separated (excludes this job).
moe_active_kind() {  # kinds, comma separated (e.g. "train,eval")
  python "$MOE_ROOT/src/status.py" --active --prefix "$JOB_PREFIX" --kind "$1"
}
moe_active_name() {  # exact job name
  python "$MOE_ROOT/src/status.py" --active --prefix "$JOB_PREFIX" --name "$1"
}

# Dependency string "afterany:1:2:3" from a space separated id list ("" if empty).
moe_dep_list() {  # type ids...
  local t="$1"; shift
  local s="" i
  for i in "$@"; do s="$s:$i"; done
  if [ -n "$s" ]; then echo "${t}${s}"; fi
}
