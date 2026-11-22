#!/usr/bin/env bash
# Run slurm/smoke.sbatch locally on CPU (tiny model, small synthetic data) in a copy of the project, with
# the mock SLURM/module/conda binaries.   devtests/run_smoke_local.sh <scratch dir> [bash binary]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$(cd "$HERE/.." && pwd)"
T="${1:?scratch dir}"; B="$(command -v "${2:-bash}")"
copy_project() {  # src dest: project files without data / caches / outputs (no rsync needed)
  mkdir -p "$2"
  tar -C "$1" -cf - --exclude ./data --exclude ./cache --exclude ./ckpt --exclude ./logs --exclude ./results \
    --exclude ./smoke --exclude '__pycache__' . | tar -C "$2" -xf -
}
rm -rf "$T"; mkdir -p "$T/mock" "$T/bin"
ln -sf "$B" "$T/bin/bash"
copy_project "$SRC" "$T/proj"
mkdir -p "$T/proj/cache/tiktoken"
cp -r "${TIKTOKEN_CACHE_DIR:?}/." "$T/proj/cache/tiktoken/"
export PATH="$T/bin:$HERE/mockbin:$PATH" MOCK_DIR="$T/mock" MOE_ROOT="$T/proj" MOE_WORK="$T/proj/smoke" \
  JOB_PREFIX=mtp_smoke_ MOE_SMOKE_LOCAL=1 SLURM_JOB_ID=4711
mkdir -p "$MOE_WORK/logs"
cd "$T/proj"
"$B" slurm/smoke.sbatch 2>&1 | tee "$MOE_WORK/logs/slurm_mtp_smoke_smoke_4711.out"
grep -q "SMOKE TEST PASSED" "$MOE_WORK/logs/slurm_mtp_smoke_smoke_4711.out"
echo "local smoke OK"
