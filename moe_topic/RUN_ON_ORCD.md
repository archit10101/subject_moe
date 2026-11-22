# Running moe_topic on MIT ORCD Engaging

Replace `USERNAME` with your Kerberos name. Every command is copy-paste ready. Run **everything on the
login node**; the heavy work runs as SLURM jobs.

Workflow: `check_cluster.sh` → `setup_login.sh` → `submit_all.sh --smoke` → `submit_all.sh` → wait
(~7 h plus queue) → copy `results/` back.

The dataset (`WebOrganizer/Corpus-200B`) is **public and not gated**: no Hugging Face login is needed.
(If that changes, `setup_login.sh` stops with "ACCESS DENIED … may be gated": accept the terms on
huggingface.co, run `huggingface-cli login` on the login node, and re-run `./setup_login.sh`.)

---

## 1. Copy the project over and unpack it

On **your computer**, in the folder that holds `moe_topic.tar.gz`:
```bash
scp moe_topic.tar.gz USERNAME@orcd-login.mit.edu:~/
```
On the **login node** (`ssh USERNAME@orcd-login.mit.edu`, Kerberos password + Duo):
```bash
cd ~ && tar xzf moe_topic.tar.gz && cd ~/moe_topic
```
The project needs **~40 GB**: ≈2.9 GB of downloaded shards, ≈6.6 GB of token streams + metadata, and
≈2.6 GB of checkpoints per run × 9 (final checkpoint with optimizer state + 25/50/75% weights). The conda
env (~6 GB) goes to your home directory. Step 2 shows free space. If home is tight and you have pool
storage, unpack there instead (e.g. `cd ~/orcd/pool && tar xzf ~/moe_topic.tar.gz`) and use that path
everywhere below. Avoid purgeable scratch for a multi-hour unattended run.

## 2. Check the cluster (prints only, changes nothing)

```bash
cd ~/moe_topic
./check_cluster.sh
```
Read the summary at the end; it should say `0 fail`. It checks partitions and GPU types, `module load
miniforge`, the bash version, disk space and quota (home, `~/orcd/pool`, `~/orcd/scratch`), and that
huggingface.co, download.pytorch.org and the GPT-2 vocab host are reachable.

If the GPU type check fails, look at the `sinfo` GRES list it prints and set `GPU_TYPE` in `config.sh`
(`h100` or `h200`; never empty, or you get a slow L40S). If you need a specific SLURM account, set
`MOE_ACCOUNT`. For e-mail on failures (every job) and at the end (report job), set `MAIL_USER`:
```bash
nano config.sh        # GPU_TYPE, MOE_ACCOUNT, MAIL_USER are the usual edits
```

## 3. Set up the environment and download everything (login node, ~15–30 min)

```bash
./setup_login.sh 2>&1 | tee setup_login.log
```
This creates the conda env `moe_topic` (Python 3.11, torch 2.14.1 built for CUDA 12.6, the same build the
previous project ran on ORCD H100s), then downloads into the project:
- train shards 0–82 (≈1.29B tokens) and eval shards 9888–9886 with their topic/format label + logit files
  (≈2.9 GB; the exact shards are chosen from the `tokens/` counts and recorded in `data/raw/manifest.json`),
- the two classifier configs (topic/format names) and the GPT-2 vocab.

It then verifies everything in offline mode (what compute nodes see) and runs the CPU tests. It is safe to
re-run: finished downloads are skipped. It should end with `setup complete`. The line `taxonomy
cross-check (GitHub, optional)` is informational; it may say `check by hand` for formats whose short label
differs from the taxonomy wording (FAQ, Truncated, News (Org.), Documentation). That is expected (see
`DATA_NOTES.md`).

## 4. Smoke test (one GPU job, ~25 min once it starts)

```bash
./submit_all.sh --smoke
./status.sh --smoke                 # repeat until the job is gone from the queue
```
It works only inside `smoke/` with synthetic data: tests, synthetic prep, bench at the **real model size**
on a real GPU, ~1.5 min of each of the 6 modes (R1 with a forced checkpoint → resume), eval with every
metric on small sets, the report, and an `sbatch --test-only` from the compute node.
**It passed if the log ends with `SMOKE TEST PASSED`**:
```bash
tail -n 15 smoke/logs/slurm_mtp_smoke_smoke_*.out
```
The last lines show the GPU name, bench tokens/sec per mode, training tokens/sec per mode, and the
estimated minutes per real run (1.2B tokens). R1 should be roughly 250k tok/s (≈80 min) on an H100; P3 is
slower (≈1.5×) because its topic loss needs a second backward pass. A `WARNINGS:` line is not fatal, but
tell me about it; `sbatch-from-compute-node-failed` means self-continuations and waits would not work.
The smoke report is at `smoke/results/REPORT.md` (it says SYNTHETIC at the top).

**If it fails with `CUDA UNAVAILABLE`**, the torch build doesn't match the node's driver. Reinstall torch
from a different PyTorch index, then re-run the smoke test:
```bash
module load miniforge && eval "$(conda shell.bash hook)" && conda activate moe_topic
pip install --force-reinstall "torch==2.14.1" --index-url https://download.pytorch.org/whl/cu124   # or cu118 / cu128
```
If you change the index, also put it in `config.sh` (`TORCH_INDEX_URL`).

## 5. Submit the real experiment

```bash
./submit_all.sh
```
It prints every job id with its dependency and records them in `logs/jobs.tsv`. You can log out now:
everything runs unattended:

| stage | job(s) | resources | expected time |
|---|---|---|---|
| prep | `mtp_prep` | CPU, 32 cores, 64 GB | ~10–15 min |
| bench | `mtp_bench` (afterok prep) | 1 GPU, 45 min limit | ~15 min (tests, then 25 steps each of R1, R1f, P2, P4, P3) |
| train | `mtp_train_<tag>` × 9 (afterok bench), priority R1, P2, P4, R1_seed1, R1f, P3, P1, P2_seed1, P4_seed1 | 1 GPU, 8 cores, 64 GB, 6 h, `--requeue` | ~80 min each (P3 ~120 min); 2 at a time → ~6 h |
| eval | `mtp_eval_<tag>` × 9 (afterok its train job) | 1 GPU, 1.5 h | ~10–20 min each, overlapping with training |
| report | `mtp_report` (afterany all evals) | CPU, 30 min | ~5 min |

Total ≈ 7 h plus queue waits (target < 16 h, hard max 24 h). Self-healing:
- a train job that reaches 330 min checkpoints and submits a continuation of itself (exits 0);
- an eval job whose run is not finished yet re-queues itself behind that run's active train job;
- the report re-queues itself while any train/eval job of this project is still pending or running;
- a crashed train job is resubmitted once, excluding the node it crashed on, with a fresh eval job behind it;
- jobs whose dependency failed are cancelled automatically (`--kill-on-invalid-dep=yes`), and the report
  then flags the missing runs.

Other options:
```bash
./submit_all.sh --only R1,P2        # only these runs (train + eval) plus a report job
./submit_all.sh --report-now        # a report from whatever has been evaluated, without waiting
```

## 6. Check progress (any time)

```bash
cd ~/moe_topic && ./status.sh
```
It shows this project's queue, which stages are done, train and eval state per run
(done / partial / missing), the latest log line of running runs (step, loss, tokens/s, ETA), and any failed
jobs with the path to their `.out` file. Live logs:
```bash
tail -f logs/slurm_mtp_train_R1_*.out
```

### If the queue is slow: cancel the seed runs

The seed runs are last in the priority order. Cancelling them (train and the eval waiting for it) costs
the seed means for P2/P4 but keeps everything else; R1_seed1 also sets the noise margin δ (without it,
δ = 0.01 nats):
```bash
scancel --name=mtp_train_P4_seed1; scancel --name=mtp_eval_P4_seed1
```
```bash
scancel --name=mtp_train_P2_seed1; scancel --name=mtp_eval_P2_seed1
```
Only if really needed (this also removes the noise estimate):
```bash
scancel --name=mtp_train_R1_seed1; scancel --name=mtp_eval_R1_seed1
``` The report job
does not need to be touched: it runs when the remaining evals finish and marks the cancelled runs as missing.
To get a report right now from whatever has been evaluated:
```bash
./submit_all.sh --report-now
```
To run cancelled seeds later: `./submit_all.sh --only P2_seed1,P4_seed1`.

## 7. Copy the results back to your computer

When `./status.sh` shows `report done` and nothing is queued, run this on **your computer**:
```bash
scp -r USERNAME@orcd-login.mit.edu:~/moe_topic/results ./results_topic
```
```bash
scp USERNAME@orcd-login.mit.edu:'~/moe_topic/logs/*.jsonl' USERNAME@orcd-login.mit.edu:'~/moe_topic/logs/jobs.tsv' ./results_topic/
```
`results_topic/REPORT.md` is the report (plots under `plots/`, per-run metrics in `eval/<tag>.json`,
routing examples in `examples/<tag>.md`, everything machine-readable in `summary.json`). Bring the whole
folder back to me for `RESULTS_SUMMARY.md`.

## 8. If something fails: what to send me

Run `./status.sh`; it lists failed jobs with their log paths. Then send:
1. the output of `./status.sh`
2. `logs/jobs.tsv`
3. the `.out` file of each failed job (`logs/slurm_<jobname>_<jobid>.out`); for the smoke test,
   `smoke/logs/slurm_mtp_smoke_smoke_<jobid>.out`
4. for a training failure, also `logs/<tag>.jsonl` (e.g. `logs/P3.jsonl`)
5. for a prep problem, `data/meta.json` (if it exists) and `data/raw/manifest.json`

A one-liner that bundles all of it:
```bash
cd ~/moe_topic && ./status.sh > logs/status_now.txt 2>&1; tar czf ~/moe_topic_debug.tgz logs results/*.json results/*.md data/meta.json data/raw/manifest.json 2>/dev/null; ls -lh ~/moe_topic_debug.tgz
```
Then on your computer:
```bash
scp USERNAME@orcd-login.mit.edu:~/moe_topic_debug.tgz .
```

## 9. Resuming after a failure

Fix the cause if needed (e.g. `config.sh`), then **just re-run**:
```bash
./submit_all.sh
```
It submits only what is missing and wires the dependencies to what it just submitted:
- finished stages, runs and evals are skipped;
- queued or running jobs are reused;
- unfinished runs resume from their last checkpoint (every 15 min);
- a new report job follows (a pending older one is cancelled).

It **refuses to run** if checkpoints exist but `results/config.json` doesn't, if the config was made from
different data (fingerprint), or if a checkpoint was trained under a different config. Stale checkpoints are
never reused silently; the message tells you which folders to move aside to start over. `bench` never
overwrites a valid `config.json` once training has started.

Notes:
- Trouble with the env: `conda env remove -n moe_topic` and re-run `./setup_login.sh`.
- A run that diverges (non-finite loss) stops without retry and keeps its last good checkpoint; send me
  its `logs/<tag>.jsonl`.
