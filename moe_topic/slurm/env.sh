# Sourced by every job script and by the login-node helpers. Needs MOE_ROOT.
# Sets offline caches, sources config.sh + lib.sh; call moe_activate to get the conda env.
: "${MOE_ROOT:?MOE_ROOT is not set (submit via submit_all.sh)}"
# shellcheck source=../config.sh
source "$MOE_ROOT/config.sh"
export MOE_ROOT
export MOE_WORK="${MOE_WORK:-$MOE_ROOT}"
export JOB_PREFIX
export HF_HOME="$MOE_ROOT/cache/hf"
export TIKTOKEN_CACHE_DIR="$MOE_ROOT/cache/tiktoken"
export PYTHONUNBUFFERED=1
if [ "${MOE_ONLINE:-0}" != "1" ]; then
  export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
fi
if [ -n "${SLURM_CPUS_PER_TASK:-}" ]; then
  export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
fi

moe_activate() {
  set +u
  if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/lmod/lmod/init/bash; do
      if [ -f "$f" ]; then source "$f"; break; fi
    done
  fi
  if type module >/dev/null 2>&1; then
    module load "$CONDA_MODULE" >/dev/null 2>&1 || module load "$CONDA_MODULE" || { echo "ERROR: module load $CONDA_MODULE failed" >&2; set -u; return 1; }
  fi
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found after 'module load $CONDA_MODULE'" >&2; set -u; return 1
  fi
  eval "$(conda shell.bash hook)"
  conda activate "$CONDA_ENV" || { echo "ERROR: conda activate $CONDA_ENV failed (run setup_login.sh)" >&2; set -u; return 1; }
  set -u
}

moe_job_header() {
  echo "=== $(date '+%F %T') job ${SLURM_JOB_ID:-local} (${SLURM_JOB_NAME:-}) on $(hostname)"
  echo "    MOE_ROOT=$MOE_ROOT MOE_WORK=$MOE_WORK attempt=${MOE_ATTEMPT:-0} waits=${MOE_WAITS:-0}"
  if command -v nvidia-smi >/dev/null 2>&1 && [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  fi
}

# shellcheck source=lib.sh
source "$MOE_ROOT/slurm/lib.sh"
