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
bash --version | head -1 | sed 's/^/  /'
ok "bash available"

hdr "SLURM partitions"
if command -v sinfo >/dev/null 2>&1; then
  for p in "$GPU_PARTITION" "$CPU_PARTITION" "$QUICK_PARTITION"; do
    if sinfo -h -p "$p" -o "%P" 2>/dev/null | grep -q .; then ok "partition $p exists"; else bad "partition $p not found"; fi
  done
  echo "  -- sinfo -p $GPU_PARTITION -O Partition,Nodes,Gres,StateLong"
  sinfo -p "$GPU_PARTITION" -O Partition,Nodes,Gres:40,StateLong 2>&1 | sed 's/^/     /'
  echo "  -- sinfo -p $CPU_PARTITION,$QUICK_PARTITION"
  sinfo -p "$CPU_PARTITION,$QUICK_PARTITION" -O Partition,Nodes,CPUs,Memory,Timelimit,StateLong 2>&1 | sed 's/^/     /'
  if sinfo -h -p "$GPU_PARTITION" -O Gres:60 2>/dev/null | grep -qi "$GPU_TYPE"; then
    ok "GPU type '$GPU_TYPE' (config.sh) is present in $GPU_PARTITION"
  else
    bad "GPU type '$GPU_TYPE' not seen in $GPU_PARTITION gres (check the list above, set GPU_TYPE in config.sh)"
  fi
  if command -v sacctmgr >/dev/null 2>&1; then
    echo "  -- your associations / QOS limits"
    sacctmgr -n show assoc user="$(whoami)" format=Account%20,Partition%20,QOS%40 2>/dev/null | sed 's/^/     /' | head -20
  fi
else
  bad "sinfo not found (are you on orcd-login.mit.edu?)"
fi
command -v sbatch >/dev/null 2>&1 && ok "sbatch available" || bad "sbatch not found"
command -v sacct >/dev/null 2>&1 && ok "sacct available (used for timing + failed-job detection)" || warn "sacct not available"

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
    bad "module load $CONDA_MODULE failed (try 'module avail miniforge' and set CONDA_MODULE)"
  fi
else
  bad "'module' command not available"
fi
set -u

hdr "disk space and quota"
df -h "$HOME" 2>/dev/null | sed 's/^/  /'
df -h "$MOE_ROOT" 2>/dev/null | tail -1 | sed 's/^/  project: /'
command -v quota >/dev/null 2>&1 && quota -s 2>/dev/null | sed 's/^/  /' | head -15
for d in "$HOME/orcd/scratch" "$HOME/orcd/pool" /orcd/scratch/*/"$(whoami)" /orcd/pool/*/"$(whoami)" /home/"$(whoami)"/orcd/*; do
  [ -e "$d" ] || continue
  echo "  -- $d"; df -h "$d" 2>/dev/null | tail -1 | sed 's/^/     /'
done
avail_kb="$(df -Pk "$MOE_ROOT" 2>/dev/null | awk 'NR==2 {print $4}')"
if [ -n "$avail_kb" ] && [ "$avail_kb" -gt 41943040 ]; then
  ok "project filesystem has $((avail_kb / 1048576)) GB free (need ~30 GB, 40 GB+ recommended)"
elif [ -n "$avail_kb" ]; then
  warn "project filesystem has only $((avail_kb / 1048576)) GB free; the project needs ~30 GB (+ ~6 GB conda env in home)"
fi
echo "  recommendation: keep the project where you have >= 40 GB free and no purge policy."
echo "  ORCD home is backed up but limited; pool storage (~/orcd/pool) is large and persistent;"
echo "  scratch (~/orcd/scratch) is fast but may be purged. Prefer pool if you have it, else home if it has room."

hdr "internet from the login node"
if command -v curl >/dev/null 2>&1 && curl -sS -m 15 -o /dev/null -w '%{http_code}' https://huggingface.co/api/datasets/DKYoon/SlimPajama-6B 2>/dev/null | grep -q '^2'; then
  ok "huggingface.co reachable"
else
  bad "huggingface.co not reachable (downloads in setup_login.sh will fail)"
fi
if curl -sS -m 15 -o /dev/null https://download.pytorch.org/whl/ 2>/dev/null; then ok "download.pytorch.org reachable"; else warn "download.pytorch.org not reachable"; fi

hdr "summary"
echo "  $PASS pass, $WARN warn, $FAIL fail$SUMMARY"
if [ "$FAIL" -eq 0 ]; then echo "  => OK to run ./setup_login.sh"; else echo "  => fix the FAIL items (or adjust config.sh) before continuing"; fi
