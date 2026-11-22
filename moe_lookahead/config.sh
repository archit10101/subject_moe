# moe_lookahead: every cluster-specific setting lives here.
# Each value can also be overridden from the environment, e.g.  GPU_TYPE=h200 ./submit_all.sh
# (sourced by every script; plain bash, no side effects)

# ---------------------------------------------------------------- scheduler
: "${GPU_PARTITION:=mit_normal_gpu}"   # 6 h max/job; 2 GPUs + 32 cores per user at a time
: "${CPU_PARTITION:=mit_normal}"       # CPU-only, 12 h max (tokenization)
: "${QUICK_PARTITION:=mit_quicktest}"  # CPU-only, 15 min max (only reported by check_cluster.sh)
: "${GPU_TYPE:=h100}"                  # h100 or h200. NEVER leave empty: untyped requests get a slow L40S
: "${MOE_ACCOUNT:=}"                   # optional SLURM account (-A); empty = default account
: "${MAIL_USER:=}"                     # e-mail for FAIL (all jobs) and END (eval); empty = no mail
: "${JOB_PREFIX:=mla_}"                # job-name prefix; used to find this project's jobs in squeue
: "${MOE_EXTRA_SBATCH:=}"              # extra sbatch flags for every job, e.g. "--qos=foo" (space separated)

# ---------------------------------------------------------------- software
: "${CONDA_MODULE:=miniforge}"
: "${CONDA_ENV:=moe_lookahead}"        # env name (or an absolute path to a prefix env)
: "${PYTHON_VERSION:=3.11}"
: "${TORCH_VERSION:=2.14.1}"
: "${TORCH_INDEX_URL:=https://download.pytorch.org/whl/cu126}"  # cu126 runs on drivers >= 525 (CUDA 12 minor compat)

# ---------------------------------------------------------------- resources per stage
: "${PREP_CPUS:=32}"
: "${PREP_MEM:=64G}"
: "${PREP_TIME:=02:00:00}"
: "${GPU_CPUS:=8}"
: "${GPU_MEM:=64G}"
: "${CLUSTERS_CPUS:=16}"                # clusters job: CPU workers decode + tokenize for MiniLM
: "${CLUSTERS_TIME:=01:30:00}"
: "${BENCH_TIME:=00:45:00}"
: "${TRAIN_TIME:=06:00:00}"
: "${EVAL_TIME:=01:30:00}"
: "${SMOKE_TIME:=00:30:00}"

# ---------------------------------------------------------------- self-healing
: "${TRAIN_MAX_MINUTES:=330}"          # checkpoint + submit a continuation after this many minutes
: "${CKPT_MINUTES:=15}"
: "${TRAIN_MAX_RETRIES:=1}"            # automatic resubmits of a crashed train job (excluding the bad node)
: "${EVAL_MAX_WAITS:=40}"              # how often eval may re-queue itself behind still-active train jobs

# ---------------------------------------------------------------- eval
: "${EVAL_BATCH:=32}"                  # sequences per eval forward pass
: "${EVAL_MAX_VAL_TOKENS:=}"           # testing only: cap on validation tokens (empty = full stream)

# ---------------------------------------------------------------- experiment sizing (identical for all runs)
: "${DOWNLOAD_TOKENS:=1300000000}"     # estimated tokens (chars/4) of train shards to download
: "${TRAIN_TOKENS:=1200000000}"        # tokens to tokenize for training
: "${TARGET_MINUTES:=120}"             # per main run, on the bench GPU
: "${R4_MINUTES:=25}"
: "${R4_MAX_TOKENS:=150000000}"
