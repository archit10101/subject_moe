# Data notes: WebOrganizer/Corpus-200B (verified 2026-11-22)

Everything below was checked against the live Hugging Face repo with `huggingface_hub` and two downloaded
shards (0 and 9888) plus their label files. `src/download.py` and `src/prep_data.py` re-check the parts
that matter on every run and stop with a clear message if something changed.

## Access

- `WebOrganizer/Corpus-200B`: **not gated, public** (`gated=False`, `private=False`). No Hugging Face login
  is needed. If that ever changes, `download.py` stops with "ACCESS DENIED … may be gated": accept the
  terms on huggingface.co and run `huggingface-cli login` on the login node, then re-run `./setup_login.sh`.
- The README recommends `git lfs clone`; we download individual files with `hf_hub_download` instead
  (resumable, only the shards we need).

## Repo layout (what is actually there)

118,670 files; 9,889 shards `CC_shard_00000000_processed` … `CC_shard_00009888_processed` (no gaps), each
present in all 12 folders:

| folder | file | content |
|---|---|---|
| `documents/` | `<shard>.jsonl.zst` | one JSON object per line: `text`, `url`, `metadata` (dict), `warcinfo`, `previous_word_count`, `language_id_whole_page_fasttext`. **No document id field.** |
| `tokens/` | `<shard>.npy` | int64 [n_docs]: GPT-NeoX token count per document |
| `domains_topics/` | `<shard>__choice.npy` | int64 [n_docs]: top topic index (0–23) |
| `domains_topics__logits/` | `<shard>__logits.npy` | float32 [n_docs, 24]: topic logits |
| `domains_formats/` | `<shard>__choice.npy` | int64 [n_docs]: top format index |
| `domains_formats__logits/` | `<shard>__logits.npy` | float32 [n_docs, 24]: format logits |
| `domains_clusters-k24/`, `-k576/`, `-k13824/` | `<shard>.npy` | k-means cluster ids (unused) |
| `scores_dclm-fasttext/`, `scores_fineweb-edu/`, `scores_fineweb-edu__rounded/` | `<shard>.npy` / `__rounded.npy` | quality scores (unused) |

**Differences from the README / the spec:**
- topic logits live in `domains_topics__logits/` (README and spec say `domain_topics__logits/`);
- format logits live in their own folder `domains_formats__logits/` (the README lists them under
  `domains_formats/`);
- the README mentions a `domain_statistics/` folder: it is not in the repo (not needed).

## Alignment of labels with documents

- There is no id field; labels align with documents **by order**. For shards 0 and 9888 the number of JSON
  lines equals the length of every label array (15,398 and 27,215 documents). `prep_data.py` asserts equal
  counts for every shard it reads.
- `argmax(topic logits) == topic __choice` and `argmax(format logits) == format __choice` for 100% of
  documents in both shards (prep asserts ≥ 99.9% per shard).
- `devtests/check_real_prep.py` re-reads documents from the raw shard by (shard, index), re-tokenizes them
  and compares with the built streams, labels, soft targets and confidence bits (passed on shards 0 + 9888).

## Ordering inside a shard

Documents inside a shard are **sorted by URL** (99.99% of adjacent pairs ascending in both shards), so
neighbouring documents often come from the same website. To get a natural stream, `prep_data.py` shuffles
documents within each train shard (seed = shard id) and pools + shuffles the eval-shard documents
(seed 1234) before drawing the eval sets. The training sequence order is a further fixed permutation
(seed 0) of 1024-token windows.

## Token counts

- GPT-2 (tiktoken) vs. GPT-NeoX counts: ratio 1.0006 (shard 0) and 1.0023 (shard 9888); per document the
  5th–95th percentile ratio is 0.95–1.04. So the `tokens/` counts are an accurate size estimate.
- Shards vary from ~10M to ~26M tokens (mean of the first 90 shards: 15.7M).
- For ≥ 1.25B GPT-2 tokens (+3% margin) the train split needs **shards 0–82 (83 shards, ≈1.29B tokens,
  ≈2.8 GB download)**, measured from the `tokens/` files.
- Eval: shards 9888, 9887, 9886 (≈68M tokens) cover the natural-val budget (10M) and ≥ 400k tokens / ≥ 200
  documents for every topic, with headroom (the scarcest topic is Adult, ≈0.6M tokens left after natural val).
  `download.py` picks these automatically from the back of the shard list (max 8).

## Topic and format names

From `id2label` of `WebOrganizer/TopicClassifier` and `WebOrganizer/FormatClassifier` (`config.json`),
identical to the lists in both model cards. Cross-check with the taxonomy definitions in the GitHub repo
`CodeCreator/WebOrganizer` (`define_domains/taxonomies/{topics,formats}.yaml`, fetched by
`download.py`'s optional check): same 24 entries in the same order; the classifier labels are short forms of
the taxonomy names:

| id | topic (id2label) | taxonomy name | | id | format (id2label) | taxonomy name |
|---:|---|---|---|---:|---|---|
| 0 | Adult | Adult Content | | 0 | Academic Writing | Academic Writing |
| 1 | Art & Design | Art & Design | | 1 | Content Listing | Content Listing |
| 2 | Software Dev. | Software Development | | 2 | Creative Writing | Creative Writing |
| 3 | Crime & Law | Crime & Law | | 3 | Customer Support | Customer Support Page |
| 4 | Education & Jobs | Education & Jobs | | 4 | Comment Section | Discussion Forum / Comment Section |
| 5 | Hardware | Electronics & Hardware | | 5 | FAQ | FAQs |
| 6 | Entertainment | Entertainment | | 6 | Truncated | Incomplete Content |
| 7 | Social Life | Social Life | | 7 | Knowledge Article | Knowledge Article |
| 8 | Fashion & Beauty | Fashion & Beauty | | 8 | Legal Notices | Legal Notices |
| 9 | Finance & Business | Finance & Business | | 9 | Listicle | Listicle |
| 10 | Food & Dining | Food & Dining | | 10 | News Article | News Article |
| 11 | Games | Games | | 11 | Nonfiction Writing | Nonfiction Writing |
| 12 | Health | Health | | 12 | About (Org.) | Organizational About Page |
| 13 | History | History & Geography | | 13 | News (Org.) | Organizational Announcement |
| 14 | Home & Hobbies | Home & Hobbies | | 14 | About (Pers.) | Personal About Page |
| 15 | Industrial | Industrial | | 15 | Personal Blog | Personal Blog |
| 16 | Literature | Literature | | 16 | Product Page | Product Page |
| 17 | Politics | Politics | | 17 | Q&A Forum | Q&A Forum |
| 18 | Religion | Religion | | 18 | Spam / Ads | Spam / Ads |
| 19 | Science & Tech. | Science, Math & Technology | | 19 | Structured Data | Structured Data |
| 20 | Software | Software | | 20 | Documentation | Technical Writing |
| 21 | Sports & Fitness | Sports & Fitness | | 21 | Audio Transcript | Transcript / Interview |
| 22 | Transportation | Transportation | | 22 | Tutorial | Tutorial / How-To Guide |
| 23 | Travel | Travel & Tourism | | 23 | User Review | User Reviews |

The corpus README says the labels come from `TopicClassifier` / `FormatClassifier` (the URL-aware
versions, not the `-NoURL` variants).

## Labels used in this project

- **topic id / format id** per document: the `__choice` arrays.
- **soft topic targets** (P3): softmax of the topic logits, stored per document as float16. The logits are
  available for every document, so the one-hot fallback is never needed.
- **confidence bin**: 1 if the topic softmax top probability ≥ 0.5. Share of documents: 95.7% (shard 0),
  95.1% (shard 9888).
- Topic distribution is skewed: e.g. shard 9888 has 2,502 Entertainment vs. 260 Adult documents.

## Splits (whole shards, deterministic, no overlap)

| split | source | size |
|---|---|---|
| train | shards 0–82 (sorted order) | ≈1.29B tokens (≥ 1.25B) |
| natural val (`val`) | random documents of the eval shards | ≈10M tokens, natural topic mix |
| balanced val (`bal`) | remaining eval documents | ≤ 400k tokens per topic (documents added in random order, skipping any that would exceed the budget) |
| ablation (`abl`) | subset of `bal` | ≤ 50k tokens per topic |

Per-topic token and document counts of every split are in `data/meta.json` and in the report; topics with
fewer than 200 balanced documents are flagged (expected: none at full size).

## Stream format

`data/{split}_tokens.bin` (uint16): GPT-2 tokens of each document followed by EOT (50256). Per-token
metadata `{split}_topic.bin`, `_format.bin`, `_conf.bin` (uint8) carry the label of the document the token
belongs to (the trailing EOT belongs to the document it ends). Per-document arrays: `_doc_offsets.npy`,
`_doc_soft.npy`, `_doc_info.npy` (topic, format, conf, shard, index in shard). Sequences are 1024-token
windows of the stream; targets are shifted by one. `meta.json` holds lengths, names, counts and the data
fingerprint that `config.json` and every checkpoint are tied to.
