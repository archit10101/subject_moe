# moe_topic: every cluster-specific setting lives here.
# Each value can also be overridden from the environment, e.g.  GPU_TYPE=h200 ./submit_all.sh
# (sourced by every script; plain bash 3.2-compatible, no side effects)
#
# MIT ORCD Engaging facts this project relies on:
#   login:  ssh USERNAME@orcd-login.mit.edu   (Kerberos password + Duo). Login nodes are shared and have
#           internet: only pip installs, downloads, editing and sbatch/squeue run there.
#   compute nodes are assumed to have NO internet: all downloads happen in setup_login.sh, into caches
#           inside the project; jobs run with HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 (slurm/env.sh).
#   software: module load miniforge + a conda env (Python 3.11).
#   mit_normal_gpu: 6 h max per job, 2 GPUs and 32 cores per user at a time. Always request a GPU type
#           (-G h100:1 or -G h200:1): an untyped request gets an L40S, which is much slower.
#   mit_normal: CPU only, 12 h max (tokenization, report).

# ---------------------------------------------------------------- scheduler
: "${GPU_PARTITION:=mit_normal_gpu}"   # 6 h max/job; 2 GPUs + 32 cores per user at a time
: "${CPU_PARTITION:=mit_normal}"       # CPU-only, 12 h max (prep, report)
: "${QUICK_PARTITION:=mit_quicktest}"  # CPU-only, 15 min max (only reported by check_cluster.sh)
: "${GPU_TYPE:=h100}"                  # h100 or h200. NEVER leave empty: untyped requests get a slow L40S
: "${MOE_ACCOUNT:=}"                   # optional SLURM account (-A); empty = default account
: "${MAIL_USER:=}"                     # e-mail: FAIL on every job, END on the final report job; empty = no mail
: "${JOB_PREFIX:=mtp_}"                # job-name prefix; used to find this project's jobs in squeue
: "${MOE_EXTRA_SBATCH:=}"              # extra sbatch flags for every job, e.g. "--qos=foo" (space separated)

# ---------------------------------------------------------------- software
: "${CONDA_MODULE:=miniforge}"
: "${CONDA_ENV:=moe_topic}"            # env name (or an absolute path to a prefix env)
: "${PYTHON_VERSION:=3.11}"
: "${TORCH_VERSION:=2.14.1}"           # the version the previous project ran on ORCD H100s
: "${TORCH_INDEX_URL:=https://download.pytorch.org/whl/cu126}"  # cu126 runs on drivers >= 525

# ---------------------------------------------------------------- resources per stage
: "${PREP_CPUS:=32}"
: "${PREP_MEM:=64G}"
: "${PREP_TIME:=02:00:00}"
: "${GPU_CPUS:=8}"
: "${GPU_MEM:=64G}"
: "${BENCH_TIME:=00:45:00}"
: "${TRAIN_TIME:=06:00:00}"
: "${EVAL_TIME:=01:30:00}"
: "${REPORT_CPUS:=4}"
: "${REPORT_MEM:=16G}"
: "${REPORT_TIME:=00:30:00}"
: "${SMOKE_TIME:=00:45:00}"            # the smoke test itself takes ~25 min

# ---------------------------------------------------------------- self-healing
: "${TRAIN_MAX_MINUTES:=330}"          # checkpoint + submit a continuation after this many minutes
: "${CKPT_MINUTES:=15}"
: "${TRAIN_MAX_RETRIES:=1}"            # automatic resubmits of a crashed train job (excluding the bad node)
: "${EVAL_MAX_WAITS:=20}"              # how often an eval may re-queue itself behind its run's train job
: "${REPORT_MAX_WAITS:=40}"            # how often the report may re-queue itself behind active train/eval jobs

# ---------------------------------------------------------------- experiment sizing (identical for all runs)
: "${TOTAL_TOKENS:=1200000000}"        # tokens per run, floored to whole optimizer steps (262,144 tokens)
: "${TRAIN_DOWNLOAD_TOKENS:=1250000000}"  # train shards are downloaded until their token counts reach this
: "${NAT_VAL_TOKENS:=10000000}"        # natural val
: "${BAL_TOKENS_PER_TOPIC:=400000}"    # balanced val (also >= 200 documents per topic where available)
: "${ABL_TOKENS_PER_TOPIC:=50000}"     # ablation subset (taken from the balanced set)
: "${EVAL_MAX_SHARDS:=8}"              # at most this many held-out shards for the eval sets
: "${MICRO_BS:=32}"                    # 32 x 1024 x 8 = 262,144 tokens per optimizer step
: "${GRAD_ACCUM:=8}"

# ---------------------------------------------------------------- eval
: "${EVAL_BATCH:=32}"                  # windows per eval forward pass
: "${EVAL_EXTRA:=}"                    # testing only, e.g. "--max_tokens 1e5"
