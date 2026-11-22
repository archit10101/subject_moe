#!/usr/bin/env bash
# Login-node setup: conda env + all downloads (WebOrganizer shards + labels, classifier configs, GPT-2 vocab)
# into the project. Safe to re-run: the existing env and finished downloads are reused.
# Takes ~15-30 min (≈3 GB torch + ≈2.6 GB of shards).
set -euo pipefail
MOE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MOE_ROOT
source "$MOE_ROOT/config.sh"
cd "$MOE_ROOT"
mkdir -p cache/hf cache/tiktoken data/raw logs results ckpt

set +u
if ! type module >/dev/null 2>&1; then
  for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do [ -f "$f" ] && source "$f" && break; done
fi
module load "$CONDA_MODULE"
eval "$(conda shell.bash hook)"
if conda env list | awk '{print $1}' | grep -qx "$CONDA_ENV"; then
  echo "== conda env $CONDA_ENV exists"
else
  echo "== creating conda env $CONDA_ENV (python $PYTHON_VERSION)"
  conda create -y -n "$CONDA_ENV" "python=$PYTHON_VERSION"
fi
conda activate "$CONDA_ENV"
set -u

if python -c "import torch, sys; sys.exit(0 if torch.__version__.split('+')[0] == '$TORCH_VERSION' else 1)" 2>/dev/null; then
  echo "== torch $TORCH_VERSION already installed"
else
  echo "== installing torch $TORCH_VERSION from $TORCH_INDEX_URL"
  python -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"
fi
python -m pip install -r requirements.txt
python - <<'PY'
import torch, numpy, tiktoken, huggingface_hub, pyarrow, zstandard, transformers, matplotlib
print("== imports OK: torch", torch.__version__, "(CUDA build", torch.version.cuda, ") numpy", numpy.__version__)
PY

echo "== downloads (online)"
export HF_HOME="$MOE_ROOT/cache/hf" TIKTOKEN_CACHE_DIR="$MOE_ROOT/cache/tiktoken" HF_HUB_DISABLE_PROGRESS_BARS=1
python -u src/download.py --train_tokens "$TRAIN_DOWNLOAD_TOKENS" --nat_val_tokens "$NAT_VAL_TOKENS" \
  --bal_tokens "$BAL_TOKENS_PER_TOPIC" --eval_max_shards "$EVAL_MAX_SHARDS"
echo "== offline verification (what compute nodes will see)"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -u src/download.py --verify
if python src/tests.py >/dev/null 2>&1; then echo "== correctness tests pass (CPU)"; else echo "tests FAILED"; python src/tests.py; exit 1; fi
du -sh data/raw cache 2>/dev/null || true
echo
echo "setup complete. Next:  ./submit_all.sh --smoke   (then ./status.sh --smoke)"
