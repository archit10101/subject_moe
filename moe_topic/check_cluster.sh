#!/usr/bin/env bash
# Read-only environment check for the ORCD login node. Prints only; changes nothing.
set -uo pipefail
MOE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$MOE_ROOT/config.sh"
PASS=0; WARN=0; FAIL=0; SUMMARY=""
ok()   { echo "  PASS  $*"; PASS=$((PASS + 1)); }
warn() { echo "  WARN  $*"; WARN=$((WARN + 1)); SUMMARY="$SUMMARY
  WARN  $*"; }
bad()  { echo "  FAIL  $*"; FAIL=$((FAIL + 1)); SUMMARY="$SUMMARY
  FAIL  $*"; }
hdr()  { echo; echo "== $*"; }

hdr "host"
echo "  $(hostname)  user=$(whoami)  project=$MOE_ROOT"
if [ -n "${SLURM_JOB_ID:-}" ]; then warn "running inside a SLURM job; this script is meant for the login node"; fi

hdr "bash"
echo "  this shell: ${BASH_VERSION:-unknown}"
bash --version | head -1 | sed 's/^/  login bash: /'
ok "bash available (scripts are bash 3.2 compatible)"

hdr "SLURM partitions and GPU types"
if command -v sinfo >/dev/null 2>&1; then
  for p in "$GPU_PARTITION" "$CPU_PARTITION" "$QUICK_PARTITION"; do
    if sinfo -h -p "$p" -o "%P" 2>/dev/null | grep -q .; then ok "partition $p exists"; else
      if [ "$p" = "$QUICK_PARTITION" ]; then warn "partition $p not found (not needed)"; else bad "partition $p not found"; fi
    fi
  done
  echo "  -- sinfo -p $GPU_PARTITION -O Partition,Nodes,Gres,Timelimit,StateLong"
  sinfo -p "$GPU_PARTITION" -O Partition,Nodes,Gres:40,Timelimit,StateLong 2>&1 | sed 's/^/     /'
  echo "  -- sinfo -p $CPU_PARTITION -O Partition,Nodes,CPUs,Memory,Timelimit,StateLong"
  sinfo -p "$CPU_PARTITION" -O Partition,Nodes,CPUs,Memory,Timelimit,StateLong 2>&1 | sed 's/^/     /'
  if sinfo -h -p "$GPU_PARTITION" -O Gres:60 2>/dev/null | grep -qi "$GPU_TYPE"; then
    ok "GPU type '$GPU_TYPE' (config.sh) is present in $GPU_PARTITION"
  else
    bad "GPU type '$GPU_TYPE' not seen in $GPU_PARTITION gres (check the list above, set GPU_TYPE in config.sh)"
  fi
  if command -v sacctmgr >/dev/null 2>&1; then
    echo "  -- your associations / QOS limits"
    sacctmgr -n show assoc user="$(whoami)" format=Account%20,Partition%20,QOS%40,GrpTRES%40 2>/dev/null | sed 's/^/     /' | head -20
  fi
else
  bad "sinfo not found (are you on orcd-login.mit.edu?)"
fi
if command -v sbatch >/dev/null 2>&1; then ok "sbatch available"; else bad "sbatch not found"; fi
if command -v sacct >/dev/null 2>&1; then ok "sacct available (used for the timeline + failed-job detection)"; else warn "sacct not available"; fi

hdr "modules / conda"
set +u
if ! type module >/dev/null 2>&1; then
  for f in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh; do [ -f "$f" ] && source "$f" && break; done
fi
if type module >/dev/null 2>&1; then
  if ( module load "$CONDA_MODULE" >/dev/null 2>&1 && command -v conda >/dev/null 2>&1 ); then
    ok "module load $CONDA_MODULE works (conda found)"
    if ( module load "$CONDA_MODULE" >/dev/null 2>&1; conda env list 2>/dev/null | awk '{print $1}' | grep -qx "$CONDA_ENV" ); then
      ok "conda env '$CONDA_ENV' already exists"
    else
      echo "  info  conda env '$CONDA_ENV' does not exist yet (setup_login.sh creates it)"
    fi
  else
    bad "module load $CONDA_MODULE failed (try 'module avail miniforge' and set CONDA_MODULE in config.sh)"
  fi
else
  bad "'module' command not available"
fi
set -u

hdr "disk space and quota"
df -h "$HOME" 2>/dev/null | sed 's/^/  /'
df -h "$MOE_ROOT" 2>/dev/null | tail -1 | sed 's/^/  project: /'
if command -v quota >/dev/null 2>&1; then quota -s 2>/dev/null | sed 's/^/  /' | head -15; fi
for d in "$HOME/orcd/scratch" "$HOME/orcd/pool" /orcd/scratch/*/"$(whoami)" /orcd/pool/*/"$(whoami)" /home/"$(whoami)"/orcd/*; do
  [ -e "$d" ] || continue
  echo "  -- $d"; df -h "$d" 2>/dev/null | tail -1 | sed 's/^/     /'
done
avail_kb="$(df -Pk "$MOE_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')"
if [ -n "$avail_kb" ] && [ "$avail_kb" -gt 52428800 ]; then
  ok "project filesystem has $((avail_kb / 1048576)) GB free (need ~40 GB, 50 GB+ recommended)"
elif [ -n "$avail_kb" ]; then
  warn "project filesystem has only $((avail_kb / 1048576)) GB free; the project needs ~40 GB (+ ~6 GB conda env in home)"
fi
echo "  ORCD home is backed up but limited; pool storage (~/orcd/pool) is large and persistent;"
echo "  scratch (~/orcd/scratch) is fast but may be purged. Prefer pool if you have it, else home if it has room."

hdr "internet from the login node"
code="$(curl -sS -m 20 -o /dev/null -w '%{http_code}' https://huggingface.co/api/datasets/WebOrganizer/Corpus-200B 2>/dev/null || true)"
if [ "$code" = "200" ]; then
  ok "huggingface.co reachable; WebOrganizer/Corpus-200B metadata readable"
  gated="$(curl -sS -m 20 https://huggingface.co/api/datasets/WebOrganizer/Corpus-200B 2>/dev/null | grep -o '"gated":[^,}]*' | head -1)"
  case "$gated" in
    *false*) ok "dataset is not gated ($gated)" ;;
    "") warn "could not read the dataset's gated flag" ;;
    *) warn "dataset is gated ($gated): accept its terms on huggingface.co and run 'huggingface-cli login'" ;;
  esac
else
  bad "huggingface.co dataset API not reachable (HTTP '$code'): downloads in setup_login.sh will fail"
fi
if curl -sS -m 20 -o /dev/null https://download.pytorch.org/whl/ 2>/dev/null; then ok "download.pytorch.org reachable"; else warn "download.pytorch.org not reachable"; fi
if curl -sS -m 20 -o /dev/null https://openaipublic.blob.core.windows.net/gpt-2/encodings/main/vocab.bpe 2>/dev/null; then
  ok "GPT-2 vocab host reachable (tiktoken)"
else
  bad "openaipublic.blob.core.windows.net not reachable (tiktoken needs it once)"
fi

hdr "summary"
echo "  $PASS pass, $WARN warn, $FAIL fail$SUMMARY"
if [ "$FAIL" -eq 0 ]; then echo "  => OK to run ./setup_login.sh"; else echo "  => fix the FAIL items (or adjust config.sh) before continuing"; fi
