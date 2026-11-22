# moe_topic: can an MoE router learn topic-level expert specialization without losing accuracy?

A self-contained SLURM package for MIT ORCD Engaging. Copy it to the login node and follow
**`RUN_ON_ORCD.md`**. Data facts are in **`DATA_NOTES.md`**.

## The experiment

Nine runs of the same 8-layer GPT-style decoder (d 512, 8 heads, context 1024, tied embeddings, vocab
50304) on 1.2B GPT-2 tokens of the topic-labelled WebOrganizer web corpus. Every layer has a shared expert
(d_ff 1024) plus routed experts with 1024 active and 8192 total routed hidden units. Runs differ **only**
in the routing:

| tag | routed experts per layer | router input |
|---|---|---|
| R1 (+ R1_seed1) | 8 × 1024, top-1 | h_t |
| R1f | 16 × 512, top-2 | h_t |
| P1 | 8 × 1024, top-1 | s_t only |
| P2 (+ P2_seed1) | 8 × 1024, top-1 | [h_t ; s_t] |
| P3 | as P2 + auxiliary topic loss (0.05) on s_t | [h_t ; s_t] |
| P4 (+ P4_seed1) | 8 "topic" × 512 (router on s_t) + 8 "token" × 512 (router on h_t), top-1 each | s_t / h_t |

`s_t` = mean of the layer's `h` over the last ≤ 50 positions of the current document (resets after EOT,
includes t; causal by construction, tested). Topic labels are used only as P3's training target and for
evaluation; they never enter the forward pass (tested). Routed outputs are scaled by exactly 1.0
(`p/p.detach()`); dropless dispatch; Switch load-balance loss 0.01 per router.

Evaluation (`results/eval/<tag>.json`): natural/balanced val loss, perplexity and accuracy (overall, per
topic, per format, high-confidence documents); per router pool and layer: I(expert; topic) with a
document-level shuffled baseline, the common-word test (top-100 token ids; I(E;T|token id)), the format
control (I(E;F), I(E;T|F)), purity / lift, within-document consistency, ablation selectivity (disable an
expert → loss change per topic), routing health; specialization over training from the 25/50/75%
checkpoints; routing examples (`results/examples/<tag>.md`). The report (`results/REPORT.md`) applies the
pre-registered claim check.

## Layout

```
config.sh            every cluster setting (partitions, GPU type, resources, token counts)
check_cluster.sh     read-only environment check (login node)
setup_login.sh       conda env + all downloads (login node, online) + offline verification
submit_all.sh        idempotent submission: --smoke | --only TAG[,TAG] | --report-now
status.sh            progress, ETA, failed jobs with log paths
slurm/               env.sh + lib.sh (sbatch flag builder) and the job scripts:
                     prep, bench, train (self-continuation, retry), eval (waits for continuations),
                     report (waits for active jobs), smoke
src/
  common.py          constants, paths, topic/format names, information-theory helpers
  download.py        shard selection + download, manifest, --verify (offline)
  prep_data.py       tokenization, per-token metadata, natural/balanced/ablation sets; --synthetic
  model.py           GPT + MoE (6 routing modes), context summary, per-name init
  data.py            batches, fixed data order, P3 topic targets
  train.py           training loop (auto-resume, checkpoints, milestones) and the bench
  metrics.py         MI / conditional MI / purity / consistency / selectivity
  eval.py            all evaluation metrics for one run
  report.py          REPORT.md, summary.json, plots
  status.py          file-based stage detection + squeue helpers (used by the shell scripts)
  tests.py           correctness tests (CPU, tiny model): run locally and in the bench job
devtests/            local-only: mock SLURM binaries, shell tests, CPU end-to-end, real-shard checker
```

## Outputs (after the cluster run)

```
results/config.json        settings, TOTAL_TOKENS, throughput, minutes per run, GPU, parameter counts, data fingerprint
results/eval/<tag>.json    every metric for one run
results/examples/<tag>.md  strongest experts with snippets, where " the" is routed per topic
results/REPORT.md          tables, claim check, timeline, deviations; plots in results/plots/
results/summary.json       the report's numbers in machine-readable form
logs/<tag>.jsonl           training logs (every 10 steps; quick val every 10%)
logs/jobs.tsv              every submitted job with dependency and note
```

## Local testing (what was run before hand-over)

```bash
python src/tests.py                                              # 12 correctness tests
bash devtests/run_local_e2e.sh /tmp/e2e --fresh                  # synthetic data, all 9 runs, eval, report
python devtests/check_real_prep.py <work> <raw> 300             # real-shard prep vs. raw documents
bash devtests/test_shell.sh /tmp/shell /tmp/e2e [bash-3.2]       # SLURM layer with mock sbatch/squeue/module/conda
MOE_SMOKE_LOCAL=1 ...                                            # smoke job script on CPU (see devtests/run_smoke_local.sh)
bash devtests/make_tarball.sh                                    # ../moe_topic.tar.gz
```
