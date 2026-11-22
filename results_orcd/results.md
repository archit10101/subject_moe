# Subject-aware MoE routing: results

*Run on MIT ORCD (1× H100 per run), 21–22 Nov 2026. Source data: this folder (report, eval files, training
logs). Tier-3 seeds (R2_seed1, R3_seed1) were submitted afterwards and are not included yet.*

## Summary

- **Subject routing made language modelling worse at this scale.** Against the standard MoE baseline (R1):
  - the future-teacher model (R3) is **+0.050 nats** worse (≈5% higher perplexity)
  - the past-teacher model (R2) is **+0.070 nats** worse (≈7%)
  - seed noise is 0.004 nats, so both gaps are far outside it
- **Seeing the future helped relative to the past.** R3 beats R2 by 0.021 nats, which is more than the
  decision threshold δ = 0.01. By the pre-registered rule the verdict is "seeing the future helped", but only
  relative to R2: neither model beats the baseline. R2 and R3 have one seed each so far.
- **The routers did become subject-level, but only coarsely.** Their experts carry 0.45–0.51 bits of
  information about the data source, versus 0.07 bits for the baseline router. Of the 8 "subjects", 3 are
  real domains (code, academic text, Wikipedia/non-English); the other 5 are fuzzy splits of general web text.
- **Subject routing helps where the domain is distinct and hurts everywhere else.** R3 beats the baseline on
  ArXiv (−0.038) and Github (−0.012). It loses on web text (CommonCrawl, C4: +0.05–0.06), which is 78% of the
  validation tokens.
- **The evaluation is honest.** In the leak check, routing by the teacher is *worse* than routing by the causal
  student, so the R3 student is not benefiting from future information.

## What was tested

**Hypothesis.** An MoE router that knows the subject of upcoming text can learn subject-based routing that
improves language modelling over a standard MoE.

All models are the same 8-layer GPT-style decoder (d = 512, 110M parameters, 51M active per token). Every
feed-forward layer is an MoE with 1 shared expert and 8 routed experts, and each token goes through exactly
one routed expert per layer. All runs saw the same 1.2B tokens in the same order with identical
hyperparameters. They differ only in how the routed expert is chosen:

| run | routing |
|---|---|
| **R1** baseline | A learned router on the current hidden state, trained by the LM loss, plus a load-balance loss. |
| **R2** past teacher | A *student* router imitates a *teacher* label: the subject cluster of the **previous** 64-token block. |
| **R3** future teacher | The same, but the teacher label is the cluster of the **current** block, which includes up to 63 tokens the model hasn't read yet. |
| **R4** leak demo | The router sees the average hidden state of the next ~50 tokens. It is deliberately leaky, and was trained on only 150M tokens. |

**Teacher labels.** Each 64-token block of text was turned into a vector, and k-means grouped all blocks into
8 clusters, one per expert. A block's cluster number (0–7) is its "subject". The labels are computed before
training and frozen, so the language model cannot train them into a hidden channel. They carry at most
3 bits per 64 tokens.

**Teacher-to-student handoff.** In R2 and R3 a per-token coin routes by the teacher with probability α:
- α = 1 for the first 10% of training
- α falls linearly to 0 at 60%
- the student routes alone for the remaining 40%

The student only sees the current hidden state, so it is causal. It learns only from a distillation loss
(predict the teacher's label) and gets no gradient from the LM loss.

**Evaluation.** All numbers are on the full validation stream (9.5M tokens). R2 and R3 are always evaluated
with **student-only** routing, so there is no access to the future.

## Main result

| run | val loss (nats) | perplexity | Δ vs R1 (mean of seeds) | first 64 tokens of a doc | rest |
|---|---:|---:|---:|---:|---:|
| R1 (seed 0) | 3.5087 | 33.41 | — | 4.173 | 3.464 |
| R1 (seed 1) | 3.5051 | 33.28 | — | 4.171 | 3.460 |
| **R1 mean** | **3.5069** | 33.34 | — | | |
| **R3** future teacher | **3.5564** | 35.04 | **+0.0495** (+5.1% ppl) | 4.265 | 3.509 |
| **R2** past teacher | **3.5773** | 35.78 | **+0.0704** (+7.3% ppl) | 4.306 | 3.529 |

**Decision rule.**
- δ = max(0.01, 2 × |R1 − R1_seed1|) = max(0.01, 0.0072) = **0.01**.
- R3 − R2 = −0.021 < −δ, so the rule outputs **"seeing the future helped"**.
- R2 − R1 = +0.070 and R3 − R1 = +0.050: **neither subject-routed model beats the baseline**. The fallback
  verdict, "subject routing didn't help at this scale", also describes the result.

## Where the difference comes from

### By data source (Δ loss vs R1 seed 0; negative = better than baseline)

| source | share of val tokens | R3 − R1 | R2 − R1 | R1 seed noise |
|---|---:|---:|---:|---:|
| ArXiv | 4.6% | **−0.038** | **−0.027** | 0.003 |
| Github | 6.6% | **−0.012** | +0.004 | 0.004 |
| StackExchange | 4.5% | +0.009 | +0.036 | 0.003 |
| Book | 1.3% | +0.023 | +0.049 | 0.004 |
| C4 | 27.0% | +0.054 | +0.076 | 0.003 |
| CommonCrawl | 51.5% | +0.060 | +0.081 | 0.005 |
| Wikipedia | 4.6% | +0.088 | +0.119 | 0.005 |

Subject routing helps on the domains that have their own cluster (ArXiv, code). It hurts on general web text,
which is spread across five fuzzy clusters, and on Wikipedia, which shares a cluster with non-English text.
Book is a single validation document, so treat that row as anecdotal.

### Document starts

Every model does worst on the first 64 tokens after a document boundary (6.3% of tokens), where it has the
least context. The penalty relative to the rest of the document is:

| | R1 | R3 | R2 |
|---|---:|---:|---:|
| first-64 loss − rest loss | 0.708 | 0.756 | 0.777 |

R2 is hurt most. At a document start its "past" teacher is the cluster of the *previous, unrelated*
document, so it is systematically wrong there. R3 still beats R2 on the rest of the document (−0.020), so
document boundaries don't explain the whole R3 vs R2 gap.

### Over training (training-loss gap vs R1, averaged over windows)

| tokens seen | R3 − R1 | R2 − R1 | phase |
|---|---:|---:|---|
| 20–80M | +0.006 | +0.021 | teacher routes everything (α = 1) |
| 180–220M | +0.051 | +0.055 | mostly teacher (α ≈ 0.9) |
| 600–700M | +0.061 | +0.075 | just before handoff (α → 0 at 720M) |
| 1100–1200M | +0.047 | +0.067 | student only |

The gap is already ~0.05 nats by 200M tokens, while the teacher still does nearly all the routing. So most of
the cost comes from routing by these subject labels at all, not from a failed handoff. The handoff adds a
smaller bump around 650–750M tokens, which partly recovers.

R3 is close to R1 in the first ~80M tokens. That is where training-time routing uses the future teacher's
label, which has seen the upcoming text. Its advantage disappears quickly, and it never shows up in
validation, which uses the student.

Plots:
- [`train_loss_diff.png`](results/plots/train_loss_diff.png): loss gap vs R1, with α in its own panel
- [`agreement.png`](results/plots/agreement.png): student–teacher agreement
- [`val_loss.png`](results/plots/val_loss.png): validation loss during training

## Do the routers become subject-level?

**Yes, coarsely.** The evidence:

**1. The students reproduce the subject labels.** On held-out data with student-only routing, the student
picks the teacher's expert far more often than chance (12.5%):

| layer | 1 | 2 | 3–8 |
|---|---:|---:|---:|
| R2 (predicts the *previous* block's cluster) | 0.60 | 0.78 | 0.81 |
| R3 (predicts the *current* block's cluster) | 0.61 | 0.62 | 0.62–0.63 |

The past block's label is easier to predict, because its text has already been read. The future label is
harder.

**2. The experts specialize by source; the baseline's do not.** I(expert; source) is the information an
expert choice carries about which dataset a token came from (max 2.81 bits):

| | mean over layers | per layer |
|---|---:|---|
| R1 baseline | 0.069 bits | 0.18, 0.05, **0.00**, 0.04, 0.06, 0.09, 0.05, 0.08 |
| R1 seed 1 | 0.074 bits | similar |
| R2 | 0.445 bits | 0.50, then 0.43–0.44 |
| R3 | 0.508 bits | 0.50–0.52 in every layer |
| *reference: the cluster labels themselves* | *0.415 bits* | |

The baseline router learned to split tokens by something other than subject (unmeasured; probably
token-level features). R2 and R3 split by domain, as designed. Their source information is the same in every
layer because every layer receives the same teacher label: in effect, each token picks one of 8 fixed paths
through the network.

**3. But the "subjects" are coarse.** The clusters came from the fallback featurizer (IDF bag of words), not
MiniLM:

| cluster | content (train blocks, by majority source) |
|---|---|
| 1 | **code**: 51% Github, 21% StackExchange |
| 4 | **academic / technical**: 36% ArXiv |
| 3 | **Wikipedia + non-English** text: 26% Wikipedia |
| 0, 2, 5, 6, 7 | **general web prose** (CommonCrawl + C4: 75–95% each), split by vocabulary/style rather than clear topic |

## Leak checks

**R2 and R3, teacher vs student.** If the routing decision carried future information that a causal router
cannot reproduce, validation loss with teacher routing would be much *lower* than with student routing. It
is the opposite:

| | student loss | teacher-routed loss | gap (student − teacher) |
|---|---:|---:|---:|
| R2 | 3.5773 | 3.5830 | −0.006 |
| R3 | 3.5564 | 3.5677 | −0.011 |

**R4 demo (naive leaky router).**

| | val loss |
|---|---:|
| R4 with its leaky routing (future visible) | 5.527 |
| R4 with the router fed zeros (generation-time view: no future) | 6.646 |
| R1 at the same 150M tokens, for reference | 4.532 |

R4 depends heavily on its view of the future: without it, loss rises by 1.1 nats. But R4 never got the
"suspiciously good" teacher-forced loss the demo was meant to show. It stayed ~1 nat *worse* than the
baseline at the same token count, so in 150M tokens it did not learn to exploit the leak. The zeros
comparison also sends every token to expert 0, so part of the 1.1-nat collapse is that distribution shift,
not just lost future information.

## Why is subject routing worse? (interpretation)

These explanations fit the numbers but have not been tested directly.

1. **The router is not optimized for prediction.** R1's router learns whatever split lowers the LM loss. The
   R2/R3 students are trained only to reproduce the k-means label, so their routing is fixed by a label made
   for clustering, not prediction. (Keeping the LM loss out of the student is an experimental control that
   keeps routing *about subject*. It is not what prevents leaks; the frozen teacher and causal student do
   that.)
2. **Subject is largely redundant with context.** After a few tokens, attention already tells the model
   whether it is reading code or prose. Routing by subject adds little new information and gives up the
   token-level specialization the baseline uses. This matches the per-source pattern: gains only on distinct
   domains, losses on the fuzzy-cluster majority.
3. **The routing is less flexible.** A subject-routed token takes the same expert in all 8 layers (8 paths),
   while R1 can combine experts freely across layers (up to 8⁸ paths). Each routed expert also mostly trains
   on one domain.
4. **The student took a shortcut during the teacher phase.** While α was high, layers 2–8 agreed with the
   teacher 94–98% of the time, but layer 1 only 47–52%. Layers 2 onwards could read the teacher's choice off
   the output of the expert the teacher picked in the layer before. Only layer 1 had to predict subject from
   the text. Agreement fell to ~56% at the handoff, when that signal disappeared, and part of the loss bump
   near 700M tokens likely comes from this.

## Caveats

- **Weaker teacher than planned.** MiniLM was projected to need 43 min to label all ~19M blocks, over the
  30-min threshold, so clustering fell back to IDF features. The clusters job actually finished in under a
  minute of its 1.5-hour limit, so MiniLM would have fit. In a local test on real data, MiniLM clusters
  carried more information about source (0.525 vs 0.321 bits), so the hypothesis has only been tested with
  the weaker subject signal.
- **One seed for R2 and R3.** The 0.021 R3-vs-R2 gap is judged against R1's seed noise. The tier-3 seed runs
  will show whether it holds.
- **Small scale.** The model has 110M parameters and saw 1.2B tokens per run, with 8 experts and 8 subject
  clusters. Conclusions may not transfer to larger models or finer-grained subjects.
- **Runs were token-capped, not time-capped.** The H100 measured ~252k tokens/s, so a 2-hour run would have
  needed 1.8B tokens. Runs stopped at the 1.2B tokens prepared, after about 62–64 min each.

## Compute and timeline

| stage | run time | queue wait (after dependencies) |
|---|---:|---:|
| tokenization (1.2B train + 9.5M val tokens) | 2 min | 18 min |
| clustering (IDF fallback) | 1 min | 10 h 27 min (likely the manual `scontrol hold`) |
| tests + throughput bench | 1 min | 1 h 08 min |
| R1 / R3 / R2 / R1_seed1 training | 61–64 min each | 7 min – 4 h 26 min |
| R4 training | 10 min | 4 h 16 min |

End-to-end, from first submission to the last training job finishing, took 17 h 29 min. About 5.5 h of that
was compute; the rest was queue time, mostly the clustering job waiting.

## Pending and possible next steps

1. **Tier-3 seeds (submitted).** R2_seed1 and R3_seed1 will give two seeds per mode and confirm or refute the
   R3 > R2 result. Download the new `results/` and logs when `./status.sh` shows `report done`.
2. **Use MiniLM for the teacher.** It fits easily in the time limit. Re-cluster, then retrain R2/R3; R1 and
   R4 can be reused.
3. **Remove the student shortcut.** Make the student predict the label from a representation not shaped by
   the teacher's earlier routing choices, so it must infer subject from text.
4. **Let the LM loss also train the student**, keeping the distillation loss. This tests subject knowledge
   as a *starting point* for a router that is free to optimize prediction, rather than a rule it must follow.
   It is still leak-free, because the student remains causal.
5. **Direct diagnostics** on the existing checkpoints: what R1's router actually splits on (token identity,
   position), and how often R2/R3 routing switches within a block or document.
