# Running moe_lookahead on MIT ORCD Engaging

Replace `USERNAME` with your Kerberos name. Every command below is copy-paste ready. Run **everything on
the login node**. The heavy work runs as SLURM jobs.

Your whole workflow: `check_cluster.sh` → `setup_login.sh` → `submit_all.sh --smoke` → `submit_all.sh`
→ wait (~6–7 h plus queue) → copy `results/` back.

---

## 1. Copy the project over and unpack it

On **your computer**, in the folder that holds `moe_lookahead.tar.gz`:
```bash
scp moe_lookahead.tar.gz USERNAME@orcd-login.mit.edu:~/
```
On the **login node** (`ssh USERNAME@orcd-login.mit.edu`, Kerberos password + Duo):
```bash
cd ~ && tar xzf moe_lookahead.tar.gz && cd ~/moe_lookahead
```
The project needs ~30 GB (≈3.2 GB parquet, ≈3.3 GB token streams, ≈1.3 GB per checkpoint × 5–7) plus a
~6 GB conda env in your home directory. Step 2 shows free space. If home is tight and you have pool
storage, unpack there instead (e.g. `cd ~/orcd/pool && tar xzf ~/moe_lookahead.tar.gz`) and use that
path everywhere below. Avoid purgeable scratch for a multi-hour unattended run.

## 2. Check the cluster (prints only, changes nothing)

```bash
cd ~/moe_lookahead
./check_cluster.sh
```
Read the summary at the end. It should say `0 fail`. If the GPU type check fails, look at the `sinfo`
GRES list it prints and set `GPU_TYPE` in `config.sh` (`h100` or `h200`; never empty, or you get a slow
L40S). If you need a specific SLURM account, set `MOE_ACCOUNT`. To get e-mail on failures and at the
end, set `MAIL_USER`:
```bash
nano config.sh        # GPU_TYPE, MOE_ACCOUNT, MAIL_USER are the usual edits
```

## 3. Set up the environment and download everything (login node, ~15–30 min)

```bash
./setup_login.sh 2>&1 | tee setup_login.log
```
This creates the conda env `moe_lookahead` (Python 3.11, torch 2.14.1 built for CUDA 12.6) and
downloads ~11 SlimPajama-6B shards, MiniLM and the GPT-2 vocab into the project. It then verifies
everything in offline mode and runs the CPU tests. It is safe to re-run, because finished downloads are
skipped. It should end with `setup complete`.

## 4. Smoke test (one GPU job, ~20–25 min once it starts)

```bash
./submit_all.sh --smoke
./status.sh --smoke                 # repeat until the job is gone from the queue
```
It works only inside `smoke/`, using synthetic data, the real model size on a real GPU, the IDF fallback
clustering, ~1.5 min of each run (including a forced checkpoint → resume), eval and the report.
**It passed if the log ends with `SMOKE TEST PASSED`**:
```bash
tail -n 15 smoke/logs/slurm_mla_smoke_smoke_*.out
```
The last lines show the GPU name, the measured tokens/sec, and the projected tokens per 2-hour run.
`WARNINGS:` lines are not fatal, but tell me about them:
- `MiniLM-failed` means the real run will use the IDF fallback features.
- `sbatch-from-compute-node-failed` means self-continuations would not work.

**If it fails with `CUDA UNAVAILABLE`**, the torch build doesn't match the node's driver. Reinstall torch
from a different PyTorch index, then re-run the smoke test:
```bash
module load miniforge && eval "$(conda shell.bash hook)" && conda activate moe_lookahead
pip install --force-reinstall "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu124   # or cu118 / cu128
```
If you change the index, also put it in `config.sh` (`TORCH_INDEX_URL`). Note that torch 2.14.1 may not
exist for every index; if pip says so, drop the version pin in that one command.

## 5. Submit the real experiment

```bash
./submit_all.sh                     # tiers 1+2: R1, R3, R2, R4, R1_seed1
# or: ./submit_all.sh --more-seeds  # also R3_seed1, R2_seed1 (+~2 h)
```
It prints every job id with its dependency and records them in `logs/jobs.tsv`. You can log out now:
everything runs unattended (prep → clusters → bench → training → eval/report). The timeline is
~6–7 h plus queue waits.

## 6. Check progress (any time)

```bash
cd ~/moe_lookahead && ./status.sh
```
It shows this project's queue, which stages and runs are done, the latest log line of each running run
(step/S, loss, tokens/s, ETA), and any failed jobs with the path to their `.out` file. Live logs:
```bash
tail -f logs/slurm_mla_train_R1_*.out
```
**If the queue is slow**, you can drop lower tiers (the report will mark them as missing):
```bash
scancel --name=mla_train_R1_seed1     # tier 2: noise estimate
scancel --name=mla_train_R4           # tier 2: leak demo
scancel --name=mla_train_R3_seed1; scancel --name=mla_train_R2_seed1   # tier 3
```
To get a report from whatever has finished so far, without waiting for the rest:
```bash
./submit_all.sh --report-now
```

## 7. Copy the results back to your computer

When `./status.sh` shows `report done` and nothing is queued, run this on **your computer**:
```bash
scp -r USERNAME@orcd-login.mit.edu:~/moe_lookahead/results ./results_orcd
scp USERNAME@orcd-login.mit.edu:~/moe_lookahead/logs/{jobs.tsv,*.jsonl} ./results_orcd/
```
`results_orcd/REPORT.md` is the report, with plots under `plots/`. Bring the whole `results_orcd` folder
back into this project for analysis.

## 8. If something fails: what to send me

Run `./status.sh`. It lists failed jobs with their log paths. Then send:
1. the output of `./status.sh`
2. `logs/jobs.tsv`
3. the `.out` file of each failed job (`logs/slurm_<jobname>_<jobid>.out`); for the smoke test,
   `smoke/logs/slurm_mla_smoke_smoke_<jobid>.out`
4. for a training failure, also `logs/<tag>.jsonl` (e.g. `logs/R2.jsonl`)
5. for a clustering problem, `results/clusters.md`

A one-liner that bundles all of it:
```bash
cd ~/moe_lookahead && ./status.sh > logs/status_now.txt 2>&1; tar czf ~/moe_debug.tgz logs results/*.json results/*.md 2>/dev/null; ls -lh ~/moe_debug.tgz
# then on your computer:  scp USERNAME@orcd-login.mit.edu:~/moe_debug.tgz .
```

## 9. Resuming after a failure

Fix the cause if needed (e.g. `config.sh`), then **just re-run**:
```bash
./submit_all.sh
```
It submits only what is missing and wires the dependencies to what it just submitted:
- finished stages and runs are skipped
- queued jobs are reused
- unfinished runs resume from their last checkpoint
- a new eval/report job follows

It **refuses to run** if checkpoints exist but `results/config.json` doesn't, or if the config was made
from different data. Stale checkpoints are never reused silently. The message tells you which folders to
move aside to start over.

Notes:
- Trouble with the env: `conda env remove -n moe_lookahead` and re-run `./setup_login.sh`.
- Eval sends an `END` mail (when `MAIL_USER` is set) each time an eval job ends. If an eval job
  re-queued itself behind a continuation, you'll get an extra END mail from that short job.
