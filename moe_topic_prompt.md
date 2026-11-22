 z# Task: topic-aware MoE routing experiment (SLURM package for MIT ORCD, which I submit myself)

You are running on **my own computer**. You can't reach the cluster. The work has two phases:

- **Phase 1 (now):** build a complete, tested, self-contained project that I copy to the MIT ORCD Engaging cluster and submit with a few commands. Once submitted it runs **unattended from start to finish** through chained SLURM jobs, recovers from common failures by itself, and stops safely when something is truly wrong.
- **Phase 2 (later):** I'll bring the `results/` folder back to you. You write `RESULTS_SUMMARY.md` (spec in §13) for a separate analyst to interpret. Report facts; don't reach conclusions beyond what was measured.

Work autonomously. Ask me only if something is truly ambiguous and a wrong guess would waste cluster hours.

If a previous project from me exists on this computer (e.g. `moe_lookahead/`, a subject-routing experiment on SlimPajama), reuse its infrastructure where it fits: SLURM chain, self-continuation, `status.sh`, `check_cluster.sh`, dropless MoE, tests. Build this as a **new** project, `moe_topic/`, and don't modify the old one.

---

## 0. Research question and claim

**Question:** can an MoE router learn **topic-level expert specialization** (science, history, literature, ...) **without losing accuracy** compared with a standard MoE?

**The claim we hope to support (or refute):** "Run X matches the baseline's validation loss within a pre-set margin, while its experts carry substantially more information about document topic than the baseline's. The specialization holds even on common words like *the*, and disabling an expert mainly hurts its own topics."

**Everything is causal.** No router ever sees future tokens; topic labels are used only as training targets (P3) and for evaluation, never as model inputs.

**Background from the previous experiment** (so you understand the design choices):
- Routing by frozen subject clusters, with the same label at every layer and a router trained only to copy the label, cost about 0.05–0.07 nats.
- Standard MoE routers specialize by token-level features, not topic.
- This experiment keeps every router trained by the LM loss, lets each layer route freely, and gives some routers an explicit summary of recent context.

---

## 1. Runs

All runs share the same data, data order, hyperparameters, token count, attention stack and shared expert. They differ **only** in the MoE routing. Each run is identical in **active** FFN compute per token and in **total** routed-expert parameters.

| tag | routed experts per layer | router(s) and input | purpose |
|---|---|---|---|
| **R1** | 8 × d_ff 1024, top-1 | one router on `h_t` (current hidden state) | baseline |
| **R1f** | 16 × d_ff 512, top-2 | one router on `h_t` | fine-grained baseline; control for P4's expert size |
| **P1** | 8 × d_ff 1024, top-1 | router on `s_t` only (summary of recent context) | pure topic-style routing (expected to lose accuracy; a contrast) |
| **P2** | 8 × d_ff 1024, top-1 | router on `[h_t ; s_t]` (concatenated) | token + context routing |
| **P3** | as P2 | as P2, plus an auxiliary topic loss on `s_t` (§4) | does explicit topic supervision help? |
| **P4** | 8 "topic" experts + 8 "token" experts, each d_ff 512 | topic router on `s_t` picks 1 topic expert; token router on `h_t` picks 1 token expert | split routing |
| **R1_seed1** | as R1, init seed 1 | | noise estimate |
| **P2_seed1**, **P4_seed1** | as P2/P4, init seed 1 | | seeds for the main candidates |

Compute matching: every run has a shared expert (d_ff 1024) plus 1024 active routed hidden units per token, and 8192 routed hidden units in total per layer.

Submission priority (SLURM runs 2 at a time): R1, P2, P4, R1_seed1, R1f, P3, P1, P2_seed1, P4_seed1.

---

## 2. Data: WebOrganizer topic-labelled web corpus

**Source:** Hugging Face dataset `WebOrganizer/Corpus-200B` (Wettig et al., "Organize the Web", ICML 2025). According to its README it contains:
- `documents/`: `.jsonl.zst` shards
- `tokens/`: GPT-NeoX token counts per document
- `domains_topics/`: top topic index per document
- `domain_topics__logits/`: per-topic logits
- `domains_formats/`: top format index per document, plus format logits
- quality scores and k=24 clusters

There are 24 topics (e.g. Science & Tech., History, Literature, Health, Finance & Business, Politics, Software Dev., ...) and 24 formats (e.g. Academic Writing, News, Tutorial, ...).

**Verify everything above locally before relying on it:**
1. `list_repo_files("WebOrganizer/Corpus-200B", repo_type="dataset")`, then inspect the folder names, shard naming and file formats.
2. Download **one** document shard plus the matching label shards. Confirm how labels align with documents (same shard names? same order? an id field?), and assert that the counts match.
3. Get the topic and format **names** from the classifiers' model configs (`id2label` in `WebOrganizer/TopicClassifier` and `WebOrganizer/FormatClassifier`). Cross-check against the taxonomy definitions in the GitHub repo `CodeCreator/WebOrganizer` (`define_domains/taxonomies`).
4. If the dataset is gated, say so in `RUN_ON_ORCD.md`: I'd need to accept its terms on Hugging Face and run `huggingface-cli login` on the login node.
5. `.jsonl.zst` needs the `zstandard` package.

If the folder names differ from the README, follow what's actually in the repo. Record what you found in `DATA_NOTES.md`.

**Splits** (by whole shards, deterministic, no overlap):
- **Train:** enough shards for ≥ 1.25B GPT-2 tokens. Estimate before downloading using the `tokens/` counts (NeoX tokens ≈ GPT-2 tokens, ±10%).
- **Eval:** separate held-out shards (e.g. the last shards in sorted order). From them build:
  - **natural val:** about 10M tokens in the natural topic mix.
  - **balanced val:** up to 400k tokens per topic, at least 200 documents per topic where available. Report the actual tokens and document counts per topic, and flag topics with fewer than 200 documents.
  - **ablation subset:** 50k tokens per topic, taken from the balanced set.

**Per-token metadata** (uint8 arrays aligned with the token streams):
- topic id of the token's document
- format id of the token's document
- topic confidence bin: whether the classifier's top-topic softmax probability is ≥ 0.5, used for a "high-confidence documents only" analysis

Also store per-document soft topic targets (softmax of the topic logits, float16) for P3. If the logits aren't available, fall back to one-hot labels and note it.

**Format:** tokenize with tiktoken GPT-2 and append EOT after each document. Use the same flat stream layout as before: `data/{split}_tokens.bin` (uint16), per-token metadata `.bin` files, and `meta.json` with lengths, topic and format names, and a data fingerprint. Sequences are 1024-token windows of the stream; targets are shifted by one.

---

## 3. Model (shared by all runs)

**Architecture:**
- GPT-style decoder: pre-norm RMSNorm (fp32), learned absolute positions, context 1024, 8 layers, d_model 512, 8 heads, `F.scaled_dot_product_attention(is_causal=True)` with no document masking, tied embeddings, vocab 50304, no dropout.
- Every layer: `x = x + attn(norm(x))`; then `h = norm(x)`; then `x = x + shared(h) + Σ_k w_k · expert_k(h)` over the chosen routed experts.
- Experts are `Linear → GELU → Linear` with no bias. Store each pool as stacked parameters, e.g. `w1: [E, 512, d_ff]` and `w2: [E, d_ff, 512]`.

**Routed weights:** every chosen expert's output is multiplied by exactly **1.0** in the forward pass, using `w = p_e / p_e.detach()`, so all routers still get LM-loss gradients. This keeps output scale identical across runs: one full-width expert in R1/P1–P3, two half-width experts in R1f and P4.

**Dropless dispatch:** argsort by expert id, run each expert on its contiguous slice, then un-sort with `index_copy`. For top-2, dispatch the (token, slot) pairs.

**Load-balance loss:** Switch loss `E · Σ_e f_e · P_e` for **each** router, weight 0.01, averaged over routers and layers.

**Context summary `s_t`** (P1–P4):
- At each MoE layer, `s_t` is the mean of that layer's `h` over positions `max(t−49, d_t) … t`, where `d_t` is the first position of the current document within the sequence: the position after the most recent EOT strictly before t, or 0 if there is none.
- So the window **resets at document boundaries** and includes the current token.
- Compute it with an fp32 cumulative sum plus a per-position start index derived causally from EOT positions.
- It is causal by construction, and the tests must prove it (§7).

**Routers** (bias-free linear layers, logits in fp32):
- R1, R1f: `W[E, 512] · h_t`
- P1: `W[8, 512] · s_t`
- P2, P3: `W[8, 1024] · [h_t ; s_t]`
- P4: `W_topic[8, 512] · s_t` and `W_token[8, 512] · h_t`

**Init:** normal(0, 0.02), with the output projections of attention and experts scaled by 1/sqrt(2·n_layers). **Initialize every parameter with its own generator, seeded from (parameter name, init_seed),** so parameters with the same name and shape are identical across modes regardless of construction order.

**Engineering notes** (each of these caused real bugs before):
- Compile only the attention modules, in place (`block.attn.compile()`), so `state_dict` keys don't change. Set `torch._dynamo.config.suppress_errors = True`, and compile only on CUDA.
- Compute the cross-entropy in fp32.
- Use `torch.load(..., weights_only=False)` for your own checkpoints.
- Use `np.bincount` on combined indices for mutual information.

---

## 4. P3's auxiliary topic loss

- Each MoE layer gets a small topic head `Linear(512 → 24)` applied to `s_t`.
- Loss: cross-entropy against the document's **soft** topic distribution (or one-hot if logits are unavailable), averaged over tokens and layers, **weight 0.05**.
- The loss flows into the backbone through `s_t`; that's intended. The router itself is still trained only by the LM loss and load balancing.
- Log topic-head accuracy, which is low near document starts by nature.
- The label is a target only. It never enters the forward pass.

---

## 5. Training

- **Hyperparameters, identical across runs:** AdamW β = (0.9, 0.95), eps 1e-8, weight decay 0.1 on ≥2-D params only, fused on CUDA. Peak LR 1e-3, 2% warmup, cosine decay to 1e-4, gradient clip 1.0. 262,144 tokens per optimizer step (micro-batch 32 × 1024 × 8 accumulation steps, or an equivalent split, identical for all runs). bf16 autocast, TF32 on.
- **Tokens per run:** `TOTAL_TOKENS` in `config.sh`, default 1.2B (floored to whole steps), the same for every run. The previous experiment's model ran at about 250k tokens/s on an H100, about 65–70 minutes per run.
- **Data order:** a fixed permutation of sequences (seed 0), the same for all runs. `init_seed` changes only the initialization.
- **Logging:** JSONL every 10 steps: step, tokens, lr, LM loss, aux and topic losses, max load per router per layer, tokens/sec. Every 10% of training, also a quick validation loss and next-token accuracy on 2M natural-val tokens.
- **Checkpoints:** every 15 minutes and at the end (model, optimizer, step, `done`). Auto-resume. At `--max_minutes` (330), checkpoint and submit a continuation of itself (§6).
- Save checkpoints at 25%, 50% and 75% of training as well, for specialization-over-time analysis (model weights only).
- A `--tiny` config (2 layers, d=64, 4 heads, expert widths scaled down proportionally) runs on CPU for local tests.

---

## 6. Pipeline and SLURM (unattended, ORCD)

**ORCD facts.** Put all of these in `config.sh`:
- Login: `ssh USERNAME@orcd-login.mit.edu` (Kerberos + Duo). Login nodes are shared and have internet; only `pip install`, downloads, editing and `sbatch`/`squeue` run there.
- **Assume compute nodes have no internet.** All downloads happen in `setup_login.sh`, into caches inside the project. Jobs set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HOME` and `TIKTOKEN_CACHE_DIR`.
- `module load miniforge`, plus a conda env with Python 3.11 and `torch numpy tiktoken huggingface_hub pyarrow zstandard transformers matplotlib`.
- `mit_normal_gpu`: **6-hour max per job, 2 GPUs and 32 cores per user at a time.** Request `-G h100:1` by default, or `-G h200:1`. Without a type you get an L40S, which is much slower; never use it.
- `mit_normal`: CPU, 12-hour max. Use it for tokenization.
- `#SBATCH` lines can't expand variables, so pass partition, GPU, account, `--chdir`, `-o $PROJECT/logs/slurm_%x_%j.out` and `--export=ALL,MOE_ROOT=$PROJECT` from `submit_all.sh`.
- Optional `MAIL_USER`: `--mail-type=FAIL` on all jobs, `END` on the final report job.

**Jobs:**
1. `prep`: CPU. Tokenize train and eval, and build the per-token metadata and the eval sets.
2. `bench`: GPU, 45 min. Run the tests (the job fails if any fail), then a 25-step throughput measurement for R1, R1f, P2 and P4. Write `results/config.json` with all settings, TOTAL_TOKENS, throughput, estimated minutes per run, GPU name, parameter counts and the data fingerprint. It must never overwrite a valid config once training has started.
3. `train_<tag>` for each run: GPU, 8 cores, 64 GB, `-t 06:00:00`, `--requeue`.
4. `eval_<tag>`, one per run, `afterok` on its train job: GPU, 1.5 h limit, the metrics in §8. These run in parallel as GPUs free up.
5. `report`: `afterany` on all eval jobs. If any train or eval job of this project is still pending or running (e.g. a continuation), it resubmits itself with `afterany` on those jobs and exits 0. Otherwise it builds the report from whatever finished and flags missing runs.

**Dependencies:** `afterok` between stages, plus `--kill-on-invalid-dep=yes`.

**Self-continuation:** a train job that reaches `--max_minutes` checkpoints, submits a continuation with the same arguments, appends its id to `logs/jobs.tsv`, and exits 0. The eval for that run must wait for the continuation: the eval job checks the checkpoint's `done` flag; if it's not done, it resubmits itself with `afterany` on the active train job for that tag.

**`submit_all.sh` is idempotent:**
- `status.py` detects finished stages and runs; only missing work is submitted.
- It refuses to run if checkpoints exist without a matching `config.json` or data fingerprint.
- Flags:
  - `--smoke`: a 30-minute synthetic end-to-end test at the real model size, in `$PROJECT/smoke/`, ending with `SMOKE TEST PASSED`, GPU name and tokens/sec per mode
  - `--only TAG[,TAG]`
  - `--report-now`
- `status.sh`: this project's jobs, done/partial/missing per run with the last log line and ETA, and failed jobs with the paths to their `.out` files.
- `check_cluster.sh`: partitions and GPU types, module availability, bash version, disk space and quota (home and ORCD scratch/pool), and Hugging Face reachability. Prints a PASS/WARN summary.
- **Shell compatibility:** I'll test on macOS (bash 3.2) and the cluster has a newer bash. Avoid expanding empty arrays under `set -u`, `mapfile`, associative arrays and `${var,,}`. Use `#!/usr/bin/env bash`.

**Expected timeline:**
- prep about 15 min
- bench about 15 min
- 9 runs × about 70 min, 2 at a time: about 5.5 h
- evals overlapping with training, about 15 min each
- report about 5 min

That's about 6.5–7 hours plus queue waits. Target: under 16 h end to end, hard maximum 24 h. `RUN_ON_ORCD.md` should explain how to cancel the seed runs if the queue is slow.

---

## 7. Tests (`src/tests.py`; tiny model, fp32, CPU)

These run locally and at the start of the bench job.
1. **Causality, every mode** (R1, R1f, P1, P2, P3, P4): change every token at positions ≥ T/2 and assert that logits at positions < T/2 change by less than 1e-4. Include EOTs in the test sequence.
2. **Context summary:** `s_t` matches a naive loop, including the window reset right after an EOT, windows shorter than 50 at document and sequence starts, and an EOT at the very end.
3. **The test has teeth:** a deliberately leaky variant (summary over `t+1…t+50`, test-only) must **fail** the causality test.
4. **Dropless dispatch:** top-1 and top-2 match a naive per-token loop (tolerance 1e-4), and every (token, slot) pair is routed exactly once.
5. **Compute matching:** active routed hidden units per token = 1024 and total routed hidden units per layer = 8192 for every mode. The shared expert is identical in all modes.
6. **Init:** all same-name, same-shape parameters are identical across modes for the same init seed; a different seed changes them.
7. **Gradients:** every router gets a nonzero gradient from the LM loss. P3's topic head gets a gradient only from the topic loss. Topic labels are not used anywhere in the forward pass (assert that the logits are unchanged when the labels are shuffled).
8. Routed weights are exactly 1.0 in the forward pass.
9. **Ablation hook:** disabling expert e reroutes its tokens to their next-best expert in the same pool, and changes nothing for other tokens.
10. **Metrics:** MI, conditional MI and selectivity give known values on small synthetic inputs (e.g. a perfectly topic-aligned router → MI = H(topic); random routing → MI ≈ 0).

---

## 8. Evaluation (`eval_<tag>` jobs; write everything to `results/eval/<tag>.json`)

Use the final checkpoint. Also evaluate the 25/50/75% checkpoints for metrics 1 and 3 only, as a specialization-over-time curve.

### Accuracy (natural val and balanced val)
- Validation loss (nats), perplexity, and next-token top-1 accuracy: overall, per topic, and per format.
- Also for high-confidence documents only.

### Specialization
Computed on the balanced val set. For every router pool separately: R1f counts both slots; P4 reports topic pool and token pool separately. Compute per layer and average.

1. **Topic information:** I(expert; topic) in bits, and normalized by H(topic). Include a **shuffled-label baseline**: permute topic labels across *documents* (not tokens) and recompute, to get the chance level.
2. **Common-word test:**
   - I(expert; topic) restricted to tokens among the 100 most frequent token ids in the balanced set.
   - Also the conditional I(expert; topic | token id), averaged over token ids with ≥ 2,000 occurrences, weighted by frequency.
   - This separates topic routing from topic-specific vocabulary.
3. **Format control:** I(expert; format), and I(expert; topic | format), estimated per format and weighted. This checks whether "topic" specialization is really writing style.
4. **Purity:**
   - per expert: the share of its tokens from its top-1 and top-3 topics, and the **lift** (topic share ÷ topic base rate) of its top topic
   - per topic: the share of its tokens going to its single most-used expert
5. **Within-document consistency:** P(same expert as the previous token), for consecutive tokens in the same document, per layer.
6. **Ablation selectivity**, on the ablation subset (50k tokens per topic):
   - For each layer and each expert, disable it (reroute to next-best) and measure the loss change per topic.
   - Selectivity of an expert = (largest positive per-topic loss increase) ÷ (sum of positive increases); 1/24 means no selectivity.
   - Also record whether the most-hurt topic equals the expert's highest-lift topic.
   - Report the load-weighted mean selectivity per run and the match rate, and compare with the same numbers for R1.
7. **Routing health:** load per expert per layer, max load, and count of dead experts (< 1% load).
8. **Examples:** for P2, P3 and P4 (and R1 for contrast), pick the 4 experts with the highest top-topic lift in a middle layer. For each, give its top 3 topics and 3 short snippets (40 tokens of context, with the routed token marked). Also show a "same word" example: where the token *the* is routed in Science & Tech vs. History vs. Literature documents. Save to `results/examples/<tag>.md`.

---

## 9. Report (`report` job → `results/REPORT.md`, `results/summary.json`, `results/plots/`)

- **Main table:** run, seed(s), natural-val loss, balanced-val loss, Δ vs. R1 (mean of R1 seeds), next-token accuracy, and a **no-worse verdict** against margin δ = max(0.01 nats, 2 × |R1 − R1_seed1|). Where a mode has two seeds, use the mean and show both.
- **Specialization table:** per run, rows for each metric in §8 (pools separate), next to R1's values and the shuffled-label baseline.
- **Per-topic table:** loss Δ vs. R1 for each of the 24 topics, with token counts.
- **Plots:**
  - validation loss over training
  - topic MI over training (from the 25/50/75/100% checkpoints)
  - per-layer topic MI per run
  - an expert × topic heatmap of lift for one middle layer (R1, P2, P4)
  - an ablation-selectivity distribution per run
- **The claim check**, using this rule, fixed in advance:
  - A run **supports the claim** if it passes the no-worse test on natural val **and** its normalized topic MI on the common-word test is at least 3× R1's **and** above the shuffled baseline **and** its ablation match rate beats R1's.
  - List which runs pass which parts. Don't round in favour of the claim.
- Also: compute and timeline per stage (from `sacct` if available, otherwise log timestamps), failures and fixes, deviations from this spec, and a banner if the data is synthetic.

---

## 10. Local testing before handing over

You can't run the real thing, so test everything you can, using a local virtual environment. Don't download more than a few shards.
1. `tests.py` passes on CPU.
2. **Real data path:** with one document shard plus its label shards, run prep in a temporary folder. Verify label alignment, topic and format names, soft targets, and the per-token metadata. Build a tiny balanced set from it.
3. **End-to-end on CPU with the tiny model:** use a synthetic-data mode (fake documents from several "topics" with distinct vocabularies, plus topic labels, logits and formats), or the single real shard. Run bench `--tiny`, train every mode for about 100 steps, run eval with every metric (on small sets), and build the report. Check that every table and plot renders and that `status.py` reports correctly.
4. **Shell scripts with mock `sbatch`, `squeue`, `module` and `conda`** on PATH (log the arguments, return incrementing ids). Cover:
   - a fresh submit
   - a partly finished project
   - stale-checkpoint refusal
   - `--smoke`, `--only` and `--report-now`
   - train self-continuation
   - eval waiting for an unfinished run
   - report waiting for active jobs

   Check every dependency string and exported variable.
5. **Smoke test job** (`submit_all.sh --smoke`) for me to run on the cluster first: tests, synthetic prep, bench, about 1.5 minutes per mode at the **real model size**, eval and the report.

Fix everything these tests surface. Don't hand over anything you haven't run at least once, except the parts that truly need the cluster; list those explicitly.

## 11. Deliverables

- `moe_topic/` (code, SLURM scripts, `README.md`, `DATA_NOTES.md`)
- `moe_topic.tar.gz` (without data, caches or checkpoints)
- `RUN_ON_ORCD.md`, with copy-paste commands for: `scp` + unpack, `check_cluster.sh`, `setup_login.sh` (env + downloads), `submit_all.sh --smoke` and how to read its result, `submit_all.sh`, `status.sh`, cancelling the seed runs if needed, resuming after failure, what to send you if something fails, and copying `results/` back (`scp -r`)

## 12. Your reply at the end of Phase 1

Reply with:
- the tarball path
- a summary of the pipeline and expected timeline
- what you tested and how
- what could only be tested on the cluster, with the likeliest failure points
- anything about the dataset that differed from this spec

---

## 13. Phase 2: when I bring back `results/`

Write `RESULTS_SUMMARY.md` for a separate analyst who has seen this spec but not the files.

- **Facts and numbers only.** Mark anything that is interpretation as interpretation, and keep it to a short final section.
- **Don't round in favour of the claim.**
- **Flag anything suspicious,** e.g. a run more than 0.3 nats off the others, dead experts, a metric that contradicts another, or very small topic counts.

Contents:
1. **What ran:** each run with status, GPU, tokens, wall-clock time, and failures or deviations.
2. **Data:** shards used; train and eval token counts; topic distribution in train, natural val and balanced val; topics with few documents.
3. **Accuracy table:** natural and balanced val loss, accuracy, Δ vs. R1 with seeds, and the no-worse verdict with δ.
4. **Specialization table:** every §8 metric for every run and pool, with R1's values and the shuffled baseline side by side.
5. **Per-topic Δ loss table** (24 topics).
6. **Specialization over training** (25/50/75/100%).
7. **Ablation results:** selectivity distribution summary and match rates.
8. **Examples:** for the two most specialized runs, the strongest experts' top topics and the "same word" routing example, copied from `results/examples/`.
9. **Claim check:** which runs pass which parts of the rule.
10. **Training dynamics:** loss gap vs. R1 at about 10%, 25%, 50%, 75% and 100% of training, and anything unusual in the logs.
11. **Anomalies and caveats.**
12. **Brief interpretation**, clearly labelled.

Keep it compact: tables over prose, under about 400 lines.
