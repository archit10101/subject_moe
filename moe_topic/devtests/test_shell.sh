#!/usr/bin/env bash
# Local test of the SLURM layer with mock sbatch / squeue / scancel / sacct / sinfo / module / conda
# (devtests/mockbin). Every case runs in a fresh copy of the project.
#   devtests/test_shell.sh <scratch dir> <synthetic work dir from run_local_e2e.sh> [bash binary]
# The optional bash binary (e.g. a bash 3.2 build) runs every script and is first on PATH as `bash`.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
T="${1:?scratch dir}"; SYN="${2:?synthetic work dir}"; B="${3:-bash}"
B="$(command -v "$B")"
mkdir -p "$T/bin"
ln -sf "$B" "$T/bin/bash"
export PATH="$T/bin:$HERE/mockbin:$PATH"
unset SLURM_JOB_ID MAIL_USER MOE_ACCOUNT GPU_TYPE MOE_WORK JOB_PREFIX
echo "shell under test: $("$B" -c 'echo $BASH_VERSION')"
NPASS=0; NFAIL=0
pass() { echo "  ok   $*"; NPASS=$((NPASS + 1)); }
fail() { echo "  FAIL $*"; NFAIL=$((NFAIL + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }

copy_project() {  # src dest: project files without data / caches / outputs (no rsync needed)
  mkdir -p "$2"
  tar -C "$1" -cf - --exclude ./data --exclude ./cache --exclude ./ckpt --exclude ./logs --exclude ./results \
    --exclude ./smoke --exclude '__pycache__' . | tar -C "$2" -xf -
}
fresh() {  # new project copy + empty mock state
  rm -rf "$T/proj" "$T/mock"; mkdir -p "$T/mock"
  copy_project "$SRC" "$T/proj"
  mkdir -p "$T/proj/cache/tiktoken"
  cp -r "${TIKTOKEN_CACHE_DIR:?}/." "$T/proj/cache/tiktoken/"
  export MOCK_DIR="$T/mock"
  P="$T/proj"
}
# real (non --test-only) sbatch calls, one per line, args separated by " ¦ "
calls() { grep -v -- '--test-only' "$MOCK_DIR/sbatch.log" 2>/dev/null | sed $'s/\x1f$//; s/\x1f/ ¦ /g'; }
call_for() { calls | grep -- "-J ¦ $1 ¦" ; }
ncalls() { calls | wc -l | tr -d ' '; }
jobid_of() { awk -F'\t' -v n="$1" '$3 == n { id = $2 } END { print id }' "$P/logs/jobs.tsv"; }  # latest id
with_data() {  # data + config.json from the synthetic work dir
  mkdir -p "$P/data/raw" "$P/results/eval" "$P/ckpt" "$P/logs"
  echo '{"complete": true, "train_shards": [], "eval_shards": [], "files": {}}' > "$P/data/raw/manifest.json"
  cp "$SYN"/data/* "$P/data/"
  cp "$SYN"/results/config.json "$P/results/"
}
with_run() {  # with_run TAG [eval]: finished checkpoint (+ log, + eval json)
  cp "$SYN"/ckpt/"$1"_last.* "$P/ckpt/"
  cp "$SYN"/ckpt/"$1"_p*.pt "$P/ckpt/" 2>/dev/null || true
  cp "$SYN"/logs/"$1".jsonl "$P/logs/" 2>/dev/null || true
  if [ "${2:-}" = "eval" ]; then cp "$SYN"/results/eval/"$1".json "$P/results/eval/"; fi
}
with_partial() {  # R1 + P2 done and evaluated, P4 done but not evaluated
  with_data; with_run R1 eval; with_run P2 eval; with_run P4
}
ALL9="R1 P2 P4 R1_seed1 R1f P3 P1 P2_seed1 P4_seed1"

echo "== 1. empty project, no downloads -> refuses"
fresh
out="$("$B" "$P/submit_all.sh" 2>&1)"; rc=$?
check "exits non-zero" "[ $rc -ne 0 ]"
check "says to run setup_login.sh" "echo \"\$out\" | grep -q 'setup_login.sh'"
check "no jobs submitted" "[ \$(ncalls) -eq 0 ]"

echo "== 2. fresh submit with downloads -> full chain"
mkdir -p "$P/data/raw"; echo '{"complete": true}' > "$P/data/raw/manifest.json"
out="$("$B" "$P/submit_all.sh" 2>&1)"; rc=$?
echo "$out" | sed 's/^/     | /'
check "exit 0" "[ $rc -eq 0 ]"
check "21 jobs (prep, bench, 9 train, 9 eval, report)" "[ \$(ncalls) -eq 21 ]"
check "module load miniforge + conda activate moe_topic were used" "grep -q 'module load miniforge' \$MOCK_DIR/module.log && grep -q 'activate moe_topic' \$MOCK_DIR/conda.log"
check "prep: CPU partition, 32 cores, 64G, 2h, no dependency" "call_for mtp_prep | grep -q -- '-p ¦ mit_normal ¦ -c ¦ 32 ¦ --mem=64G ¦ -t ¦ 02:00:00' && ! call_for mtp_prep | grep -q dependency"
check "bench afterok:prep + kill-on-invalid-dep, GPU h100, 45 min" "call_for mtp_bench | grep -q -- '-G ¦ h100:1 ¦ -c ¦ 8 ¦ --mem=64G ¦ -t ¦ 00:45:00' && call_for mtp_bench | grep -q -- '--dependency=afterok:1000 ¦ --kill-on-invalid-dep=yes'"
check "train R1: afterok:bench, 6h, requeue, 8 cores, 64G" "call_for mtp_train_R1 | grep -q -- '-c ¦ 8 ¦ --mem=64G ¦ -t ¦ 06:00:00 ¦ --requeue' && call_for mtp_train_R1 | grep -q 'afterok:1001'"
check "train args (tag mode seed)" "call_for mtp_train_R1 | grep -q 'train.sbatch ¦ R1 ¦ r1 ¦ 0$' && call_for mtp_train_P4_seed1 | grep -q 'train.sbatch ¦ P4_seed1 ¦ p4 ¦ 1$' && call_for mtp_train_R1f | grep -q 'R1f ¦ r1f ¦ 0$'"
order="$(calls | grep -o 'mtp_train_[A-Za-z0-9_]*' | tr '\n' ' ')"
check "train priority order R1,P2,P4,R1_seed1,R1f,P3,P1,P2_seed1,P4_seed1" "[ \"$order\" = 'mtp_train_R1 mtp_train_P2 mtp_train_P4 mtp_train_R1_seed1 mtp_train_R1f mtp_train_P3 mtp_train_P1 mtp_train_P2_seed1 mtp_train_P4_seed1 ' ]"
okdeps=1
for t in $ALL9; do
  tid="$(jobid_of "mtp_train_$t")"
  call_for "mtp_eval_$t" | grep -q -- "--dependency=afterok:$tid ¦ --kill-on-invalid-dep=yes" || okdeps=0
  call_for "mtp_eval_$t" | grep -q -- "eval.sbatch ¦ $t$" || okdeps=0
done
check "each eval_<tag> afterok on its own train job, arg = tag" "[ $okdeps -eq 1 ]"
check "eval: GPU, 1.5 h" "call_for mtp_eval_P3 | grep -q -- '-G ¦ h100:1 ¦ -c ¦ 8 ¦ --mem=64G ¦ -t ¦ 01:30:00'"
eids="$(awk -F'\t' '$3 ~ /^mtp_eval_/ {printf "%s%s", s, $2; s=":"}' "$P/logs/jobs.tsv")"
check "report afterany on all 9 eval jobs, CPU partition" "call_for mtp_report | grep -q -- \"--dependency=afterany:$eids ¦\" && call_for mtp_report | grep -q -- '-p ¦ mit_normal ¦ -c ¦ 4 ¦ --mem=16G ¦ -t ¦ 00:30:00'"
check "exports + --chdir + log path" "call_for mtp_report | grep -q -- \"--chdir=$P ¦ -o ¦ $P/logs/slurm_%x_%j.out ¦ --export=ALL,MOE_ROOT=$P,MOE_WORK=$P,MOE_ATTEMPT=0,MOE_WAITS=0,MOE_NOWAIT=0,MOE_EXCLUDE=,JOB_PREFIX=mtp_ ¦\""
check "every job exports MOE_ROOT" "[ \$(calls | grep -c -- \"--export=ALL,MOE_ROOT=$P,\") -eq 21 ]"
check "no mail flags when MAIL_USER is empty" "! calls | grep -q mail"
check "jobs.tsv has 21 rows" "[ \$(grep -c '^20' \$P/logs/jobs.tsv) -eq 21 ]"

echo "== 3. re-run while everything is queued -> nothing new"
before=$(ncalls)
out="$("$B" "$P/submit_all.sh" 2>&1)"; rc=$?
check "exit 0, no new submissions" "[ $rc -eq 0 ] && [ \$(ncalls) -eq $before ]"
check "reports queued jobs" "echo \"\$out\" | grep -q 'already queued/running as job 1000' && echo \"\$out\" | grep -q 'nothing new'"

echo "== 4. partly finished project -> only the missing work"
fresh; with_partial
out="$("$B" "$P/submit_all.sh" 2>&1)"; rc=$?
echo "$out" | sed 's/^/     | /'
check "exit 0" "[ $rc -eq 0 ]"
check "skips prep/bench, train+eval of R1/P2, train of P4" "! calls | grep -qE 'mtp_(prep|bench|train_R1 |train_P2 |train_P4 |eval_R1 |eval_P2 )'"
check "P4: eval only, without dependency" "call_for mtp_eval_P4 | grep -qv dependency && ! call_for mtp_train_P4 | grep -q ."
check "6 new trains without dependency" "[ \$(calls | grep -c 'train.sbatch') -eq 6 ] && ! calls | grep 'train.sbatch' | grep -q dependency"
check "14 jobs (6 train, 7 eval, report)" "[ \$(ncalls) -eq 14 ]"
rdep="$(call_for mtp_report | grep -o 'afterany:[0-9:]*')"
check "report afterany on the 7 new evals" "[ \"\$(echo $rdep | tr ':' '\n' | grep -c '^[0-9]')\" -eq 7 ]"

echo "== 5. --only"
fresh; with_partial
out="$("$B" "$P/submit_all.sh" --only P2,P1 2>&1)"; rc=$?
check "only P1 (train + eval) and a report; P2 already done" "[ $rc -eq 0 ] && [ \$(ncalls) -eq 3 ] && call_for mtp_train_P1 | grep -q 'P1 ¦ p1 ¦ 0' && call_for mtp_eval_P1 | grep -q afterok:1000 && call_for mtp_report | grep -q afterany:1001"
out="$("$B" "$P/submit_all.sh" --only P9 2>&1)"; rc=$?
check "unknown tag refused" "[ $rc -ne 0 ] && echo \"\$out\" | grep -q 'unknown tag'"

echo "== 6. --report-now"
fresh; with_partial
"$B" "$P/submit_all.sh" --report-now >/dev/null 2>&1
check "one report job, no dependency, MOE_NOWAIT=1" "[ \$(ncalls) -eq 1 ] && call_for mtp_report | grep -q 'MOE_NOWAIT=1' && ! call_for mtp_report | grep -q dependency"

echo "== 7. --smoke"
fresh
"$B" "$P/submit_all.sh" --smoke >/dev/null 2>&1
check "one smoke job in smoke/ with its own prefix, GPU, 45 min" "[ \$(ncalls) -eq 1 ] && call_for mtp_smoke_smoke | grep -q -- \"MOE_WORK=$P/smoke,\" && call_for mtp_smoke_smoke | grep -q -- '-G ¦ h100:1' && call_for mtp_smoke_smoke | grep -q -- '-t ¦ 00:45:00' && call_for mtp_smoke_smoke | grep -q 'JOB_PREFIX=mtp_smoke_'"
check "smoke logs go to smoke/logs" "call_for mtp_smoke_smoke | grep -q -- \"-o ¦ $P/smoke/logs/slurm_%x_%j.out\""
out="$("$B" "$P/submit_all.sh" --smoke 2>&1)"; rc=$?
check "second --smoke refused while queued" "[ $rc -ne 0 ] && echo \"\$out\" | grep -q 'already queued'"

echo "== 8. stale checkpoints -> refuse"
fresh; with_partial; rm "$P/results/config.json"
out="$("$B" "$P/submit_all.sh" 2>&1)"; rc=$?
check "checkpoints without config.json: refuses, nothing submitted" "[ $rc -ne 0 ] && echo \"\$out\" | grep -q 'refusing' && [ \$(ncalls) -eq 0 ]"
fresh; with_partial
python - "$P/results/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["data_fingerprint"] = "deadbeef"; json.dump(c, open(sys.argv[1], "w"))
PY
out="$("$B" "$P/submit_all.sh" 2>&1)"; rc=$?
check "config from different data: refuses" "[ $rc -ne 0 ] && echo \"\$out\" | grep -q 'different data' && [ \$(ncalls) -eq 0 ]"
fresh; with_partial
python - "$P/ckpt/R1_last.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["config_hash"] = "0000"; json.dump(c, open(sys.argv[1], "w"))
PY
out="$("$B" "$P/submit_all.sh" 2>&1)"; rc=$?
check "checkpoint from a different config.json: refuses" "[ $rc -ne 0 ] && echo \"\$out\" | grep -q 'R1. were trained under a different' && [ \$(ncalls) -eq 0 ]"

echo "== 9. pre-flight --test-only failure stops everything"
fresh; with_partial
out="$(MOCK_TESTONLY_FAIL=1 "$B" "$P/submit_all.sh" 2>&1)"; rc=$?
check "refuses with a clear message" "[ $rc -ne 0 ] && echo \"\$out\" | grep -q 'GPU_TYPE' && [ \$(ncalls) -eq 0 ]"

echo "== 10. MAIL_USER + account + GPU type override"
fresh; with_partial
MAIL_USER=me@example.com MOE_ACCOUNT=acct GPU_TYPE=h200 "$B" "$P/submit_all.sh" >/dev/null 2>&1
check "train: mail FAIL only, account, h200" "call_for mtp_train_P3 | grep -q -- '--mail-user=me@example.com ¦ --mail-type=FAIL ¦' && call_for mtp_train_P3 | grep -q -- '-A ¦ acct' && call_for mtp_train_P3 | grep -q 'h200:1'"
check "eval: mail FAIL only" "call_for mtp_eval_P3 | grep -q -- '--mail-type=FAIL ¦'"
check "report: mail END,FAIL" "call_for mtp_report | grep -q -- '--mail-type=END,FAIL'"

echo "== 11. train job: time limit -> self-continuation with identical resources"
fresh; with_data; with_run R1 eval
out="$(cd "$P" && MOE_ROOT="$P" SLURM_JOB_ID=777 TRAIN_MAX_MINUTES=0.01 "$B" slurm/train.sbatch P4 p4 0 2>&1)"; rc=$?
echo "$out" | tail -3 | sed 's/^/     | /'
check "wrapper exits 0 after checkpointing" "[ $rc -eq 0 ] && echo \"\$out\" | grep -q 'exited with 3'"
check "continuation: same flags + args, no dependency" "call_for mtp_train_P4 | grep -q -- '-G ¦ h100:1 ¦ -c ¦ 8 ¦ --mem=64G ¦ -t ¦ 06:00:00 ¦ --requeue ¦ --open-mode=append' && call_for mtp_train_P4 | grep -q 'train.sbatch ¦ P4 ¦ p4 ¦ 0$' && ! call_for mtp_train_P4 | grep -q dependency"
check "jobs.tsv notes the continuation" "grep -q 'continuation of 777' \$P/logs/jobs.tsv"
check "checkpoint not done, step > 0" "python -c \"import json,sys; s=json.load(open('$P/ckpt/P4_last.json')); sys.exit(0 if (not s['done'] and s['step']>0) else 1)\""
n0=$(ncalls)
out="$(cd "$P" && MOE_ROOT="$P" SLURM_JOB_ID=778 "$B" slurm/train.sbatch P4 p4 0 2>&1)"; rc=$?
check "continuation resumes and finishes (no new job)" "[ $rc -eq 0 ] && echo \"\$out\" | grep -q 'resumed from step' && grep -q '\"done\": true' \$P/ckpt/P4_last.json && [ \$(ncalls) -eq $n0 ]"
check "25/50/75% checkpoints exist" "[ -f \$P/ckpt/P4_p25.pt ] && [ -f \$P/ckpt/P4_p50.pt ] && [ -f \$P/ckpt/P4_p75.pt ]"
out="$(cd "$P" && MOE_ROOT="$P" SLURM_JOB_ID=779 "$B" slurm/train.sbatch P4 p4 0 2>&1)"; rc=$?
check "re-running a finished run is a no-op" "[ $rc -eq 0 ] && echo \"\$out\" | grep -q 'already finished'"

echo "== 12. train job crash -> one retry excluding the node (+ eval behind it), then give up"
n0=$(ncalls)
head -c 1000 "$P/ckpt/P4_last.pt" > "$P/ckpt/P2_last.pt"  # corrupt checkpoint -> torch.load crashes
out="$(cd "$P" && MOE_ROOT="$P" SLURM_JOB_ID=780 SLURMD_NODENAME=node42 "$B" slurm/train.sbatch P2 p2 0 2>&1)"; rc=$?
check "wrapper exits non-zero (FAIL mail)" "[ $rc -ne 0 ]"
rid="$(jobid_of mtp_train_P2)"
check "retry submitted with --exclude=node42 and attempt 1" "call_for mtp_train_P2 | tail -1 | grep -q -- '--exclude=node42' && call_for mtp_train_P2 | tail -1 | grep -q 'MOE_ATTEMPT=1'"
check "eval_P2 submitted afterok on the retry" "call_for mtp_eval_P2 | tail -1 | grep -q -- \"--dependency=afterok:$rid ¦\" && call_for mtp_eval_P2 | tail -1 | grep -q 'MOE_EXCLUDE=,'"
n1=$(ncalls)
out="$(cd "$P" && MOE_ROOT="$P" SLURM_JOB_ID=781 MOE_ATTEMPT=1 SLURMD_NODENAME=node43 "$B" slurm/train.sbatch P2 p2 0 2>&1)"; rc=$?
check "second crash: no further retry" "[ $rc -ne 0 ] && [ \$(ncalls) -eq $n1 ] && echo \"\$out\" | grep -q 'retries exhausted'"
rm -f "$P/ckpt/P2_last.pt"
out="$(cd "$P" && MOE_ROOT="$P" SLURM_JOB_ID=782 "$B" slurm/train.sbatch R1 r2 0 2>&1)"; rc=$?
check "config error (wrong mode for tag): exit 2, no retry" "[ $rc -eq 2 ] && [ \$(ncalls) -eq $n1 ]"

echo "== 13. eval waits for an unfinished run"
fresh; with_data; with_run R1 eval
cp "$SYN"/ckpt/P4_last.pt "$P/ckpt/"
python - "$SYN/ckpt/P4_last.json" "$P/ckpt/P4_last.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["done"] = False; c["step"] = 50; json.dump(c, open(sys.argv[2], "w"))
PY
echo "4242|mtp_train_P4|PENDING" > "$MOCK_DIR/queue"
echo "4243|mtp_train_P4_seed1|RUNNING" >> "$MOCK_DIR/queue"
echo "4244|mtp_smoke_train_P4|RUNNING" >> "$MOCK_DIR/queue"
out="$(cd "$P" && MOE_ROOT="$P" SLURM_JOB_ID=900 "$B" slurm/eval.sbatch P4 2>&1)"; rc=$?
echo "$out" | tail -2 | sed 's/^/     | /'
check "exit 0 and re-queued behind exactly its own train job (4242)" "[ $rc -eq 0 ] && call_for mtp_eval_P4 | grep -q -- '--dependency=afterany:4242 ¦ --kill-on-invalid-dep=yes' && call_for mtp_eval_P4 | grep -q 'eval.sbatch ¦ P4$'"
check "wait counter incremented, GPU flags kept" "call_for mtp_eval_P4 | grep -q 'MOE_WAITS=1' && call_for mtp_eval_P4 | grep -q -- '-t ¦ 01:30:00'"
check "did not evaluate" "[ ! -f \$P/results/eval/P4.json ]"
: > "$MOCK_DIR/queue"
out="$(cd "$P" && MOE_ROOT="$P" SLURM_JOB_ID=901 "$B" slurm/eval.sbatch P4 2>&1)"; rc=$?
check "unfinished and no active train job: exit 5 with advice" "[ $rc -eq 5 ] && echo \"\$out\" | grep -q 'submit_all.sh'"
echo "4245|mtp_train_P4|PENDING" > "$MOCK_DIR/queue"
out="$(cd "$P" && MOE_ROOT="$P" SLURM_JOB_ID=902 MOE_WAITS=20 "$B" slurm/eval.sbatch P4 2>&1)"; rc=$?
check "after EVAL_MAX_WAITS: stops waiting (exit 5)" "[ $rc -eq 5 ]"
with_run P4; : > "$MOCK_DIR/queue"; n0=$(ncalls)
out="$(cd "$P" && MOE_ROOT="$P" SLURM_JOB_ID=903 EVAL_BATCH=4 EVAL_EXTRA='--max_tokens 3e4 --abl_max_tokens 4e3 --min_word_count 50' "$B" slurm/eval.sbatch P4 2>&1)"; rc=$?
echo "$out" | tail -2 | sed 's/^/     | /'
check "finished run: evaluates, writes eval json + examples, no resubmission" "[ $rc -eq 0 ] && [ -f \$P/results/eval/P4.json ] && [ -f \$P/results/examples/P4.md ] && [ \$(ncalls) -eq $n0 ]"
check "status: P4 eval done" "python \$P/src/status.py --work \$P --json | python -c \"import json,sys; sys.exit(0 if json.load(sys.stdin)['evals']['P4']=='done' else 1)\""

echo "== 14. report waits for active train/eval jobs"
echo "5000|mtp_train_P3|RUNNING" > "$MOCK_DIR/queue"
echo "5001|mtp_eval_P3|PENDING" >> "$MOCK_DIR/queue"
echo "5002|mtp_report|RUNNING" >> "$MOCK_DIR/queue"
echo "5003|mtp_smoke_eval_R1|RUNNING" >> "$MOCK_DIR/queue"
echo "5004|other_job|RUNNING" >> "$MOCK_DIR/queue"
out="$(cd "$P" && MOE_ROOT="$P" SLURM_JOB_ID=5002 "$B" slurm/report.sbatch 2>&1)"; rc=$?
echo "$out" | tail -2 | sed 's/^/     | /'
check "exit 0, re-queued afterany:5000:5001 only, CPU" "[ $rc -eq 0 ] && call_for mtp_report | grep -q -- '--dependency=afterany:5000:5001 ¦' && call_for mtp_report | grep -q -- '-p ¦ mit_normal'"
check "wait counter incremented, report not built" "call_for mtp_report | grep -q 'MOE_WAITS=1' && [ ! -f \$P/results/REPORT.md ]"
out="$(cd "$P" && MOE_ROOT="$P" SLURM_JOB_ID=5002 MOE_NOWAIT=1 "$B" slurm/report.sbatch 2>&1)"; rc=$?
check "--report-now (MOE_NOWAIT=1) builds it despite active jobs" "[ $rc -eq 0 ] && [ -f \$P/results/REPORT.md ]"
rm -f "$P/results/REPORT.md"; : > "$MOCK_DIR/queue"; n0=$(ncalls)
out="$(cd "$P" && MOE_ROOT="$P" SLURM_JOB_ID=5010 "$B" slurm/report.sbatch 2>&1)"; rc=$?
check "nothing active: builds the report, no resubmission" "[ $rc -eq 0 ] && [ -f \$P/results/REPORT.md ] && [ \$(ncalls) -eq $n0 ]"
check "report flags missing runs + how to resume, synthetic banner, claim section, plots" "grep -q 'Missing or unfinished runs' \$P/results/REPORT.md && grep -q 'submit_all.sh' \$P/results/REPORT.md && head -1 \$P/results/REPORT.md | grep -q SYNTHETIC && grep -q 'Claim check' \$P/results/REPORT.md && ls \$P/results/plots/*.png >/dev/null"
check "status: report done" "python \$P/src/status.py --work \$P --shell | grep -q ST_REPORT=done"

echo "== 15. status.sh"
echo "6000|mtp_train_P3|RUNNING" > "$MOCK_DIR/queue"
printf '2026-11-20T10:00:00\t4100\tmtp_train_R1_seed1\ttrain\tR1_seed1\t\t\n' >> "$P/logs/jobs.tsv"
echo 4100 > "$MOCK_DIR/failed"
out="$("$B" "$P/status.sh" 2>&1)"; rc=$?
echo "$out" | sed 's/^/     | /'
check "lists queue, done/missing per run + eval, failed job log path" "[ $rc -eq 0 ] && echo \"\$out\" | grep -q 'mtp_train_P3' && echo \"\$out\" | grep -qE 'R1 +r1 +seed 0 +done +done' && echo \"\$out\" | grep -qE 'P3 .*missing +missing' && echo \"\$out\" | grep -q 'slurm_mtp_train_R1_seed1_4100.out'"
out="$("$B" "$P/status.sh" --smoke 2>&1)"; rc=$?
check "status.sh --smoke works on an empty smoke dir" "[ $rc -eq 0 ]"

echo "== 16. check_cluster.sh (mock sinfo; no network assertions)"
out="$("$B" "$P/check_cluster.sh" 2>&1)"; rc=$?
check "runs to the summary, GPU type found" "echo \"\$out\" | grep -q 'pass,' && echo \"\$out\" | grep -q \"GPU type 'h100'\""

echo
echo "shell tests ($("$B" -c 'echo $BASH_VERSION')): $NPASS passed, $NFAIL failed"
[ "$NFAIL" -eq 0 ]
