# moe_lookahead: does knowing the *subject* of upcoming text help MoE routing?

A self-contained SLURM experiment for MIT ORCD Engaging. Copy-paste instructions for running it are in
[RUN_ON_ORCD.md](RUN_ON_ORCD.md). This file explains the design, the files, and the timeline.

## The experiment

**Hypothesis.** An MoE router that knows the subject of upcoming text can learn subject-based routing
that improves language modelling over a standard MoE.

**Why the naive version is broken (R4).** If the router at position t can see tokens after t, the
expert choice becomes a side channel. The network learns to smuggle the next token through which expert
fires. Teacher-forced loss looks great, but generation is impossible because the future doesn't exist yet.
R4 does exactly this on purpose (its router sees the mean hidden state of positions t+2 … t+51) as a
demonstration. Its `zero_router` evaluation is the generation-time view.

**The leak-free design (R2, R3).**
- A **teacher** gives routing decisions early in training. It is a frozen, precomputed subject label per
  64-token block: the k-means cluster (K=8) of a MiniLM sentence embedding of that block. The LM loss
  never trains it, so it can't learn to smuggle tokens, and it carries at most 3 bits per 64 tokens.
- A **student** router in every layer sees only the current hidden state (causal) and learns to imitate
  the teacher through a distillation loss (CE, weight 0.1). The LM loss has no gradient path to the
  student. Control moves from teacher to student via α: α = 1 for the first 10% of steps, then
  falls linearly to 0 at 60%. A per-token coin picks the teacher if u < α.
- **Evaluation uses the student only**, so the validation numbers are honest.
- **R2 (past teacher)** uses the previous block's cluster. **R3 (future teacher)** uses the current
  block's cluster, which includes up to 63 tokens the model hasn't seen yet. They are otherwise
  identical, so R3 vs R2 isolates whether seeing the future matters.
- **R1** is the baseline: a standard learned top-1 router with a switch load-balance loss.

All runs are identical except for the routing mode and, for the seed runs, the init seed: same data order,
same token count, same hyperparameters, and identical initial weights for the same seed. The routed
weight is exactly 1.0 in the forward pass in every mode (R1/R4 use `p_e / p_e.detach()`).

| Tier | Runs | When |
|---|---|---|
| 1 | R1, R3, R2 (seed 0) | always |
| 2 | R4 (leak demo, ≤150M tokens / ~25 min), R1_seed1 (noise estimate) | always |
| 3 | R3_seed1, R2_seed1 | `./submit_all.sh --more-seeds` |

**Decision rule** (`results/REPORT.md`): δ = max(0.01, 2·|R1 − R1_seed1|), or 0.01 with one seed.
- R3 < R2 − δ means seeing the future helped.
- |R3 − R2| ≤ δ with both below R1 − δ means subject supervision helped and the future didn't matter.
- If neither beats R1 by more than δ, subject routing didn't help at this scale.
- Anything else is described as it is.

## Model

A GPT-style decoder: 8 layers, d=512, 8 heads, context 1024, pre-norm RMSNorm (fp32), learned positions,
tied embeddings, vocab 50304. Every FFN is an MoE with 1 shared and 8 routed experts (512→1024→512, GELU),
top-1, with **dropless** dispatch (argsort by expert, contiguous slices, `index_copy` back). That's about
106M parameters in total, about 50M active per token. Training uses AdamW (0.9, 0.95), wd 0.1, peak lr
1e-3, 2% warmup, cosine to 1e-4, clip 1.0, 262,144 tokens/step (32 × 1024 × 8), bf16 autocast and TF32.
Only the attention modules are compiled.

## Pipeline (all SLURM jobs, chained, unattended)

```
setup_login.sh (login node): conda env, ~11 SlimPajama-6B shards (~3.2 GB), MiniLM, GPT-2 vocab
   └─ submit_all.sh
        prep      (mit_normal CPU, 32 cores)  tokenize 1.2B train tokens + full validation (~9.5M)
        clusters  (GPU)  MiniLM block embeddings → balanced spherical k-means → labels per block
        bench     (GPU)  correctness tests (fail = stop) → throughput → results/config.json
        train_<tag> × 5 (GPU, 6 h limit, --requeue)  each sized to ~120 min on the bench GPU
        eval      (GPU, afterany)  full-validation eval of every finished run → REPORT.md
```
Stages are chained with `afterok` + `--kill-on-invalid-dep=yes`, so a failed stage cancels everything
downstream instead of leaving it pending forever. Eval uses `afterany` and then checks for itself.

**Self-healing.**
- Checkpoints are written every 15 min and at the end. Every training job auto-resumes from
  `ckpt/<tag>_last.pt`.
- After 330 min a run checkpoints, submits a continuation of itself with identical resources (recorded
  in `logs/jobs.tsv`), and exits 0.
- A crashed training job is resubmitted once, excluding the node it crashed on (`TRAIN_MAX_RETRIES`).
- A diverging run (non-finite loss or grad norm) stops immediately without overwriting its last good
  checkpoint (exit code 4). It is retried once from that checkpoint like any crash, then left for you to inspect.
- Node failures are covered by `--requeue` plus auto-resume.
- If eval finds training jobs of this project still pending or running (for example a continuation), it
  re-queues itself behind them with `afterany`.
- `submit_all.sh` is idempotent. It submits only what is missing, reuses queued jobs, and refuses to
  touch stale checkpoints (checkpoints without `results/config.json`, or a config made from different data).

**Fallbacks.** Clustering switches to an IDF-weighted bag of GPT-2 ids (random-projected to 256 dims) if
MiniLM fails to load, or if MiniLM would take more than 30 min to label every block. If `torch.compile`
breaks, the model falls back to eager mode.

## Expected timeline (2 GPUs at a time)

| stage | wall-clock (excluding queue waits) |
|---|---|
| prep | ~10–20 min (tokenization runs at ~10M tokens/s on 8 threads locally; most of the time is reading parquet) |
| clusters | ~20–30 min (MiniLM is bound by CPU decode + tokenize: 15 worker processes on 16 cores; measured ~22k blocks/s with 6 workers locally → ~15 min for 19M blocks; the IDF fallback takes ~5 min) |
| bench | ~10 min (tests + 2 × 25 steps incl. compile) |
| training | R1 + R3 in parallel (~2 h), then R2 + R4 (R4 ~25 min), then R1_seed1 → **~4.5 h** |
| eval + report | ~15–30 min |
| **total** | **~6–7 h plus queue waits** (target < 16 h). Tier 3 adds ~2 h. |

If the GPU is faster than ~170k tokens/s, a 2-hour run would need more than the 1.2B tokenized training
tokens. In that case the runs are **capped at 1.2B tokens** (about 4,577 steps) and finish sooner. The
report says so.

**If the queue is slow**, cancel the lower tiers and the report will mark them as missing:
```bash
scancel --name=mla_train_R1_seed1      # tier 2 noise estimate (then δ falls back to 0.01)
scancel --name=mla_train_R4            # tier 2 leak demo
scancel --name=mla_train_R3_seed1; scancel --name=mla_train_R2_seed1   # tier 3
```
You can resume them later with `./submit_all.sh` (or `--more-seeds`).

## Files

```
config.sh          every cluster-specific value (partitions, GPU type, account, mail, times, sizes)
check_cluster.sh   read-only login-node check (sinfo, modules, disk/quota, internet) → PASS/WARN/FAIL
setup_login.sh     conda env + downloads (login node; the only step that uses the internet)
submit_all.sh      idempotent submitter (--smoke, --more-seeds, --report-now)
status.sh          queue + stage/run status + last log line per run + failed jobs (with .out paths)
slurm/             env.sh (offline caches, conda), lib.sh (all sbatch flags), *.sbatch job scripts
src/
  download.py      dataset shards (resumable, ~1.3B est. tokens), MiniLM, tiktoken vocab
  prep_data.py     parquet → {train,val}_{tokens,src}.bin + meta.json (--synthetic for tests)
  clusters.py      block embeddings → balanced spherical k-means → {train,val}_cluster.npy + sanity report
  model.py         the MoE GPT and the four routing modes
  data.py          batch construction, teacher alignment, deterministic coins
  train.py         training with auto-resume / time-limit continuation; --bench writes config.json
  tests.py         12 correctness tests (causality, the test's teeth, teacher alignment, dropless MoE, …)
  eval.py          full-validation evaluation (cached per checkpoint)
  report.py        REPORT.md, summary.json, plots/
  status.py        file-based status (used by submit_all.sh, status.sh, eval.sbatch)
devtests/          local test harness: mock sbatch/squeue/scancel/sacct, shell tests, local e2e
```

Outputs: `data/` (~6.5 GB with raw parquet), `ckpt/` (~1.3 GB per run), `logs/` (`<tag>.jsonl`, `slurm_*.out`,
`jobs.tsv`), `results/` (`config.json`, `clusters.{md,json}`, `eval/*.json`, `REPORT.md`, `summary.json`,
`plots/`). The smoke test writes the same structure under `smoke/` only.

## Local testing (what was run before hand-over)

```bash
python src/tests.py                                   # 12 tests, ~3 s on CPU
devtests/run_local_e2e.sh /tmp/e2e                    # synthetic data, tiny model, all stages, ~5 min
devtests/test_shell.sh /tmp/shelltest /tmp/e2e        # 53 checks of the SLURM layer with mocks (run under macOS bash 3.2)
```
