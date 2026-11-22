#!/usr/bin/env bash
# Local test of the SLURM layer with mock sbatch/squeue/scancel/sacct (devtests/mockbin).
#   devtests/test_shell.sh <scratch dir> <synthetic work dir with finished R1,R3,R4 (from run_local_e2e.sh)>
# Runs with macOS bash 3.2 as well as bash 5.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
T="${1:?scratch dir}"; SYN="${2:?synthetic work dir}"
export PATH="$HERE/mockbin:$PATH" MOE_SKIP_CONDA=1
NPASS=0; NFAIL=0
pass() { echo "  ok   $*"; NPASS=$((NPASS + 1)); }
fail() { echo "  FAIL $*"; NFAIL=$((NFAIL + 1)); }
check() { if eval "$2"; then pass "$1"; else fail "$1"; fi; }

fresh() {  # new project copy + empty mock state
  rm -rf "$T"; mkdir -p "$T"
  rsync -a --exclude data --exclude cache --exclude ckpt --exclude logs --exclude results --exclude smoke \
    --exclude '__pycache__' "$SRC/" "$T/proj/"
  mkdir -p "$T/proj/cache/tiktoken" "$T/mock"
  export MOCK_DIR="$T/mock"
  P="$T/proj"
}
# real (non --test-only) sbatch calls, one per line, args separated by " ¦ "
calls() { grep -v -- '--test-only' "$MOCK_DIR/sbatch.log" 2>/dev/null | sed $'s/\x1f/ ¦ /g'; }
call_for() { calls | grep -- "-J ¦ $1 ¦" ; }
ncalls() { calls | wc -l | tr -d ' '; }
with_partial() {  # project with finished data/clusters/bench and R1, R3, R4 done
  mkdir -p "$P/data" "$P/results" "$P/ckpt" "$P/logs"
  cp "$SYN"/data/*.bin "$SYN"/data/*.npy "$SYN"/data/meta.json "$P/data/"
  cp "$SYN"/results/config.json "$SYN"/results/clusters.json "$P/results/"
  cp "$SYN"/ckpt/R1_last.* "$SYN"/ckpt/R3_last.* "$SYN"/ckpt/R4_last.* "$SYN"/ckpt/R1_r4ref.pt "$P/ckpt/"
  cp "$SYN"/logs/R1.jsonl "$SYN"/logs/R3.jsonl "$SYN"/logs/R4.jsonl "$P/logs/" 2>/dev/null || true
}

echo "== 1. empty project, no downloads -> refuses"
fresh
out="$(bash "$P/submit_all.sh" 2>&1)"; rc=$?
check "exits non-zero" "[ $rc -ne 0 ]"
check "says to run setup_login.sh" "echo \"\$out\" | grep -q 'setup_login.sh'"
check "no jobs submitted" "[ \$(ncalls) -eq 0 ]"

echo "== 2. empty project with downloads -> full chain"
mkdir -p "$P/data/raw"; echo '{"complete": true, "validation": [], "train": []}' > "$P/data/raw/manifest.json"
out="$(bash "$P/submit_all.sh" 2>&1)"; rc=$?
echo "$out" | sed 's/^/     | /'
check "exit 0" "[ $rc -eq 0 ]"
check "9 jobs (prep, clusters, bench, 5 train, eval)" "[ \$(ncalls) -eq 9 ]"
check "prep: CPU partition, 32 cores, 2h, no dependency" "call_for mla_prep | grep -q -- '-p ¦ mit_normal ¦ -c ¦ 32 ¦ --mem=64G ¦ -t ¦ 02:00:00' && ! call_for mla_prep | grep -q dependency"
check "clusters afterok:prep + kill-on-invalid-dep" "call_for mla_clusters | grep -q -- '--dependency=afterok:1000 ¦ --kill-on-invalid-dep=yes'"
check "clusters: GPU job with 16 cores" "call_for mla_clusters | grep -q -- '-G ¦ h100:1 ¦ -c ¦ 16 ¦ --mem=64G ¦ -t ¦ 01:30:00'"
check "bench afterok:clusters" "call_for mla_bench | grep -q -- '--dependency=afterok:1001'"
check "GPU type always set (-G h100:1)" "call_for mla_bench | grep -q -- '-G ¦ h100:1'"
check "train R1 afterok:bench, 6h, requeue" "call_for mla_train_R1 | grep -q -- '-t ¦ 06:00:00 ¦ --requeue' && call_for mla_train_R1 | grep -q 'afterok:1002'"
check "train args R1 r1 0" "call_for mla_train_R1 | grep -q 'train.sbatch ¦ R1 ¦ r1 ¦ 0'"
check "priority order R1,R3,R2,R4,R1_seed1" "[ \"\$(calls | grep -o 'mla_train_[A-Za-z0-9_]*' | tr '\n' ' ')\" = 'mla_train_R1 mla_train_R3 mla_train_R2 mla_train_R4 mla_train_R1_seed1 ' ]"
check "no tier-3 runs by default" "! calls | grep -q 'R3_seed1'"
check "eval afterany on all 5 train jobs" "call_for mla_eval | grep -q -- '--dependency=afterany:1003:1004:1005:1006:1007'"
check "MOE_ROOT exported, --chdir, log path" "call_for mla_eval | grep -q -- \"--chdir=$P ¦ -o ¦ $P/logs/slurm_%x_%j.out ¦ --export=ALL,MOE_ROOT=$P,\""
check "no mail flags when MAIL_USER empty" "! calls | grep -q mail"
check "jobs.tsv has 9 rows" "[ \$(grep -c '^20' \$P/logs/jobs.tsv) -eq 9 ]"

echo "== 3. re-run while everything is queued -> nothing new"
before=$(ncalls)
out="$(bash "$P/submit_all.sh" 2>&1)"; rc=$?
check "exit 0" "[ $rc -eq 0 ]"
check "no new submissions" "[ \$(ncalls) -eq $before ]"
check "reports queued jobs" "echo \"\$out\" | grep -q 'already queued/running as job 1000'"

echo "== 4. partly finished project -> only the missing runs"
fresh; with_partial
out="$(bash "$P/submit_all.sh" 2>&1)"; rc=$?
echo "$out" | sed 's/^/     | /'
check "exit 0" "[ $rc -eq 0 ]"
check "skips prep/clusters/bench and R1/R3/R4" "! calls | grep -qE 'mla_(prep|clusters|bench|train_R1 |train_R3 |train_R4 )'"
check "submits R2 and R1_seed1 without dependency" "call_for mla_train_R2 | grep -qv dependency && call_for mla_train_R1_seed1 | grep -qv dependency && [ \$(ncalls) -eq 3 ]"
check "eval afterany on the 2 new runs" "call_for mla_eval | grep -q -- '--dependency=afterany:1000:1001'"

echo "== 5. --more-seeds adds tier 3"
fresh; with_partial
bash "$P/submit_all.sh" --more-seeds >/dev/null 2>&1
check "R3_seed1 and R2_seed1 submitted (seed 1)" "call_for mla_train_R3_seed1 | grep -q 'R3_seed1 ¦ r3 ¦ 1' && call_for mla_train_R2_seed1 | grep -q 'R2_seed1 ¦ r2 ¦ 1'"
check "eval waits for 4 runs" "call_for mla_eval | grep -q 'afterany:1000:1001:1002:1003'"

echo "== 6. --report-now"
fresh; with_partial
bash "$P/submit_all.sh" --report-now >/dev/null 2>&1
check "one eval job, no dependency, MOE_EVAL_NOWAIT=1" "[ \$(ncalls) -eq 1 ] && call_for mla_eval | grep -q 'MOE_EVAL_NOWAIT=1' && ! call_for mla_eval | grep -q dependency"

echo "== 7. --smoke"
fresh
bash "$P/submit_all.sh" --smoke >/dev/null 2>&1
check "one smoke job in smoke/ with its own prefix, 30 min" "[ \$(ncalls) -eq 1 ] && call_for mla_smoke_smoke | grep -q -- \"MOE_WORK=$P/smoke,\" && call_for mla_smoke_smoke | grep -q -- '-t ¦ 00:30:00'"
check "smoke logs go to smoke/logs" "call_for mla_smoke_smoke | grep -q -- \"-o ¦ $P/smoke/logs/slurm_%x_%j.out\""
out="$(bash "$P/submit_all.sh" --smoke 2>&1)"; rc=$?
check "second --smoke refused while queued" "[ $rc -ne 0 ] && echo \"\$out\" | grep -q 'already queued'"

echo "== 8. stale checkpoints -> refuse"
fresh; with_partial; rm "$P/results/config.json"
out="$(bash "$P/submit_all.sh" 2>&1)"; rc=$?
check "missing config.json with checkpoints: refuses" "[ $rc -ne 0 ] && echo \"\$out\" | grep -q 'refusing' && [ \$(ncalls) -eq 0 ]"
fresh; with_partial
python - "$P/results/config.json" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["data_fingerprint"] = "deadbeef"; json.dump(c, open(sys.argv[1], "w"))
PY
out="$(bash "$P/submit_all.sh" 2>&1)"; rc=$?
check "config from different data: refuses" "[ $rc -ne 0 ] && echo \"\$out\" | grep -q 'different data' && [ \$(ncalls) -eq 0 ]"

echo "== 9. pre-flight --test-only failure stops everything"
fresh; with_partial
out="$(MOCK_TESTONLY_FAIL=1 bash "$P/submit_all.sh" 2>&1)"; rc=$?
check "refuses with a clear message" "[ $rc -ne 0 ] && echo \"\$out\" | grep -q 'GPU_TYPE' && [ \$(ncalls) -eq 0 ]"

echo "== 10. MAIL_USER + account + GPU type override"
fresh; with_partial
MAIL_USER=me@example.com MOE_ACCOUNT=acct GPU_TYPE=h200 bash "$P/submit_all.sh" >/dev/null 2>&1
check "train: mail FAIL, account, h200" "call_for mla_train_R2 | grep -q -- '--mail-type=FAIL' && call_for mla_train_R2 | grep -q -- '-A ¦ acct' && call_for mla_train_R2 | grep -q 'h200:1'"
check "eval: mail END,FAIL" "call_for mla_eval | grep -q -- '--mail-type=END,FAIL'"

echo "== 11. train job: time limit -> self-continuation with identical resources"
fresh; with_partial
rm "$P/ckpt/R4_last.pt" "$P/ckpt/R4_last.json" "$P/logs/R4.jsonl"
out="$(cd "$P" && MOE_ROOT="$P" SLURM_JOB_ID=777 TRAIN_MAX_MINUTES=0.01 bash slurm/train.sbatch R4 r4 0 2>&1)"; rc=$?
echo "$out" | tail -4 | sed 's/^/     | /'
check "wrapper exits 0 after checkpointing" "[ $rc -eq 0 ] && echo \"\$out\" | grep -q 'exited with 3'"
check "continuation submitted, no dependency, same flags" "call_for mla_train_R4 | grep -q -- '-t ¦ 06:00:00 ¦ --requeue ¦ --open-mode=append' && call_for mla_train_R4 | grep -q 'train.sbatch ¦ R4 ¦ r4 ¦ 0' && ! call_for mla_train_R4 | grep -q dependency"
check "jobs.tsv notes the continuation" "grep -q 'continuation of 777' \$P/logs/jobs.tsv"
check "checkpoint not done, step > 0" "python -c \"import json,sys; s=json.load(open('$P/ckpt/R4_last.json')); sys.exit(0 if (not s['done'] and s['step']>0) else 1)\""
out="$(cd "$P" && MOE_ROOT="$P" SLURM_JOB_ID=778 bash slurm/train.sbatch R4 r4 0 2>&1)"; rc=$?
check "continuation resumes and finishes" "[ $rc -eq 0 ] && echo \"\$out\" | grep -q 'resumed from step' && grep -q '\"done\": true' \$P/ckpt/R4_last.json"
out="$(cd "$P" && MOE_ROOT="$P" SLURM_JOB_ID=779 bash slurm/train.sbatch R4 r4 0 2>&1)"; rc=$?
check "re-running a finished run is a no-op" "[ $rc -eq 0 ] && echo \"\$out\" | grep -q 'already finished'"

echo "== 12. train job crash -> one retry excluding the node, then give up"
n0=$(ncalls)
head -c 1000 "$P/ckpt/R4_last.pt" > "$P/ckpt/R2_last.pt"  # corrupt checkpoint -> torch.load crashes
out="$(cd "$P" && MOE_ROOT="$P" SLURM_JOB_ID=780 SLURMD_NODENAME=node42 bash slurm/train.sbatch R2 r2 0 2>&1)"; rc=$?
check "wrapper exits non-zero (FAIL mail)" "[ $rc -ne 0 ]"
check "retry submitted with --exclude=node42 and attempt 1" "call_for mla_train_R2 | tail -1 | grep -q -- '--exclude=node42' && call_for mla_train_R2 | tail -1 | grep -q 'MOE_ATTEMPT=1'"
n1=$(ncalls)
out="$(cd "$P" && MOE_ROOT="$P" SLURM_JOB_ID=781 MOE_ATTEMPT=1 SLURMD_NODENAME=node43 bash slurm/train.sbatch R2 r2 0 2>&1)"; rc=$?
check "second crash: no further retry" "[ $rc -ne 0 ] && [ \$(ncalls) -eq $n1 ] && echo \"\$out\" | grep -q 'retries exhausted'"
rm -f "$P/ckpt/R2_last.pt"

echo "== 13. eval waits for active training jobs"
fresh; with_partial
echo "4242|mla_train_R2|PENDING" > "$MOCK_DIR/queue"
echo "4243|mla_train_R1_seed1|RUNNING" >> "$MOCK_DIR/queue"
echo "5000|other_job|RUNNING" >> "$MOCK_DIR/queue"
out="$(cd "$P" && MOE_ROOT="$P" SLURM_JOB_ID=900 bash slurm/eval.sbatch 2>&1)"; rc=$?
echo "$out" | tail -2 | sed 's/^/     | /'
check "exit 0 and re-queued behind 4242:4243" "[ $rc -eq 0 ] && call_for mla_eval | grep -q -- '--dependency=afterany:4242:4243'"
check "wait counter incremented" "call_for mla_eval | grep -q 'MOE_EVAL_WAITS=1'"
check "did not evaluate yet" "[ ! -f \$P/results/REPORT.md ]"
out="$(cd "$P" && MOE_ROOT="$P" SLURM_JOB_ID=901 MOE_EVAL_WAITS=40 EVAL_BATCH=4 EVAL_MAX_VAL_TOKENS=20000 bash slurm/eval.sbatch 2>&1)"; rc=$?
check "after EVAL_MAX_WAITS it evaluates anyway" "echo \"\$out\" | grep -q 'evaluating what is finished' && [ -f \$P/results/REPORT.md ]"
: > "$MOCK_DIR/queue"; rm -f "$P/results/REPORT.md"
n0=$(ncalls)
out="$(cd "$P" && MOE_ROOT="$P" SLURM_JOB_ID=902 EVAL_BATCH=4 EVAL_MAX_VAL_TOKENS=20000 bash slurm/eval.sbatch 2>&1)"; rc=$?
echo "$out" | tail -3 | sed 's/^/     | /'
check "nothing active: evaluates + report, no resubmission" "[ $rc -eq 0 ] && [ -f \$P/results/REPORT.md ] && [ \$(ncalls) -eq $n0 ]"
check "report flags missing runs and how to resume" "grep -q 'Missing or unfinished runs' \$P/results/REPORT.md && grep -q 'submit_all.sh' \$P/results/REPORT.md"
check "report has synthetic banner" "head -1 \$P/results/REPORT.md | grep -q SYNTHETIC"

echo "== 14. status.sh"
echo "4300|mla_train_R2|RUNNING" > "$MOCK_DIR/queue"
printf '2026-11-20T10:00:00\t4100\tmla_train_R1_seed1\ttrain\tR1_seed1\t\t\n' >> "$P/logs/jobs.tsv"
echo 4100 > "$MOCK_DIR/failed"
out="$(bash "$P/status.sh" 2>&1)"; rc=$?
echo "$out" | sed 's/^/     | /'
check "lists queue, stages, runs, failed job log path" "[ $rc -eq 0 ] && echo \"\$out\" | grep -q 'mla_train_R2' && echo \"\$out\" | grep -q 'R1        tier 1 done' && echo \"\$out\" | grep -q 'slurm_mla_train_R1_seed1_4100.out'"

echo
echo "shell tests: $NPASS passed, $NFAIL failed"
[ "$NFAIL" -eq 0 ]
