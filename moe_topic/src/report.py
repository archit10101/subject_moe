"""Build results/REPORT.md, results/summary.json and results/plots/ from whatever has been evaluated.

  python src/report.py [--work DIR]
"""
import argparse
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import FORMAT_NAMES, N_TOPICS, RUN_INFO, RUNS, TOPIC_NAMES, Paths, read_json, write_json  # noqa: E402

MODE_ORDER = ["r1", "r1f", "p1", "p2", "p3", "p4"]
MODE_NAME = {"r1": "R1", "r1f": "R1f", "p1": "P1", "p2": "P2", "p3": "P3", "p4": "P4"}
CLAIM_POOL = {"p4": "topic"}  # pool used for P4's claim check (the token pool is listed too)


def f(x, d=4):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:.{d}f}"


def jsonl(path):
    out = []
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def sacct_rows(ids):
    if not ids:
        return None
    try:
        r = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(ids),
                            "--format=JobID,JobName,State,Submit,Eligible,Start,End,NodeList,ExitCode"],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    rows = []
    for line in r.stdout.splitlines():
        p = line.split("|")
        if len(p) >= 9:
            rows.append(dict(id=p[0], name=p[1], state=p[2], submit=p[3], eligible=p[4], start=p[5], end=p[6],
                             node=p[7], exit=p[8]))
    return rows


def tdiff_min(a, b):
    try:
        ta = time.mktime(time.strptime(a, "%Y-%m-%dT%H:%M:%S"))
        tb = time.mktime(time.strptime(b, "%Y-%m-%dT%H:%M:%S"))
        return (tb - ta) / 60
    except (ValueError, TypeError):
        return None


def mean_or_none(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.mean(v)) if v else None


def pool_metric(ev, pool, key):
    try:
        return ev["specialization"][pool]["mean"][key]
    except (KeyError, TypeError):
        return None


def abl_metric(ev, pool, key):
    try:
        return ev["ablation"][pool][key]
    except (KeyError, TypeError):
        return None


def main():
    warnings.filterwarnings("ignore", message="Mean of empty slice")
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    args = ap.parse_args()
    paths = Paths(args.work).mk()
    conf = read_json(paths.config) or {}
    meta = read_json(paths.meta) or {}
    display = sorted(RUN_INFO, key=lambda t: (MODE_ORDER.index(RUN_INFO[t]["mode"]), RUN_INFO[t]["seed"]))
    evals = {t: read_json(paths.eval_json(t)) for t in display}
    evals = {t: e for t, e in evals.items() if e}  # display order: R1, R1_seed1, R1f, P1, P2, P2_seed1, ...
    plots = os.path.join(paths.results, "plots")
    os.makedirs(plots, exist_ok=True)
    L = []
    w = L.append
    synthetic = meta.get("synthetic") or conf.get("synthetic")
    if synthetic:
        w("# ⚠️ SYNTHETIC DATA: these numbers come from the synthetic test corpus, not WebOrganizer ⚠️\n")
    w("# Topic-aware MoE routing: report\n")
    w(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} from `{paths.work}`. Facts only; no interpretation._\n")

    # ------------------------------------------------------------------ main table
    by_mode = {}
    for tag, mode, seed in RUNS:
        if tag in evals:
            by_mode.setdefault(mode, []).append(tag)
    nat = lambda t: evals[t]["natural"]["overall"]["loss"]  # noqa: E731
    bal = lambda t: evals[t]["balanced"]["overall"]["loss"]  # noqa: E731
    r1_tags = by_mode.get("r1", [])
    r1_nat = mean_or_none([nat(t) for t in r1_tags])
    r1_bal = mean_or_none([bal(t) for t in r1_tags])
    noise = abs(nat("R1") - nat("R1_seed1")) if {"R1", "R1_seed1"} <= set(evals) else None
    delta = max(0.01, 2 * noise) if noise is not None else 0.01
    summary = dict(generated=time.strftime("%Y-%m-%d %H:%M:%S"), synthetic=bool(synthetic), delta=delta,
                   noise_r1=noise, r1_natural=r1_nat, r1_balanced=r1_bal, runs={}, modes={}, claim={})
    w("## Accuracy (natural val; balanced val)\n")
    w(f"Margin δ = max(0.01, 2 × |R1 − R1_seed1|) = **{f(delta)}** nats"
      + (f" (|R1 − R1_seed1| = {f(noise)})" if noise is not None else " (R1_seed1 missing: δ falls back to 0.01)")
      + ". Δ is against the mean of the R1 seeds. No-worse ⇔ Δ(natural) ≤ δ.\n")
    w("| mode | seeds (natural loss) | natural loss | natural ppl | natural acc | balanced loss | balanced acc | Δ natural | Δ balanced | no-worse |")
    w("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for mode in MODE_ORDER:
        tags = by_mode.get(mode)
        if not tags:
            continue
        n = mean_or_none([nat(t) for t in tags])
        b = mean_or_none([bal(t) for t in tags])
        acc = mean_or_none([evals[t]["natural"]["overall"]["acc"] for t in tags])
        bacc = mean_or_none([evals[t]["balanced"]["overall"]["acc"] for t in tags])
        dn = n - r1_nat if (n is not None and r1_nat is not None) else None
        db = b - r1_bal if (b is not None and r1_bal is not None) else None
        ok = None if dn is None else bool(dn <= delta)
        seeds = " / ".join(f"{t}: {f(nat(t))}" for t in tags)
        w(f"| {MODE_NAME[mode]} | {seeds} | {f(n)} | {f(np.exp(n), 2)} | {f(acc)} | {f(b)} | {f(bacc)} | "
          f"{f(dn) if mode != 'r1' else '0 (ref)'} | {f(db) if mode != 'r1' else '0 (ref)'} | "
          f"{'–' if mode == 'r1' else ('yes' if ok else 'NO')} |")
        summary["modes"][MODE_NAME[mode]] = dict(tags=tags, natural=n, balanced=b, acc=acc, balanced_acc=bacc,
                                                 delta_natural=dn, delta_balanced=db, no_worse=ok)
    w("")
    w("High-confidence documents only (topic classifier top probability ≥ 0.5; label of the target token):\n")
    w("| run | natural loss (high-conf) | tokens | balanced loss (high-conf) | tokens |")
    w("|---|---:|---:|---:|---:|")
    for t in evals:
        e = evals[t]
        w(f"| {t} | {f(e['natural']['high_conf']['loss'])} | {e['natural']['high_conf']['tokens']:,} | "
          f"{f(e['balanced']['high_conf']['loss'])} | {e['balanced']['high_conf']['tokens']:,} |")
        summary["runs"][t] = dict(natural=nat(t), balanced=bal(t), step=e["step"], tokens=e["tokens"],
                                  natural_acc=e["natural"]["overall"]["acc"], gpu=e.get("gpu"),
                                  natural_high_conf=e["natural"]["high_conf"]["loss"],
                                  train_minutes=e.get("train_minutes"))
    w("")

    # ------------------------------------------------------------------ specialization table
    w("## Specialization (balanced val; per router pool; mean over layers)\n")
    w("nmi = I(expert; topic) / H(topic). `shuf` = topic labels permuted across documents (chance level). "
      "common-word = tokens among the 100 most frequent ids; cmi = I(E;T | token id) over ids with ≥ "
      f"{next(iter(evals.values()))['conventions']['min_word_count'] if evals else 2000} occurrences. "
      "Selectivity / match: ablation on the ablation subset (chance 1/24 = 0.042). R1 columns are the reference; "
      "`shuf` rows are each run's own shuffled-label baseline. Conditional MI estimated from finite counts is "
      "biased upwards (roughly (E−1)·23/(2 n ln 2) bits per conditioning value with n samples); the shuffled rows "
      "measure that bias directly.\n")
    cols = [("I(E;T) bits", "mi", 3), ("nmi", "nmi", 4), ("nmi shuf", "nmi_shuffled", 4),
            ("common-word nmi", "cw_nmi_top", 4), ("cw nmi shuf", "cw_nmi_top_shuffled", 4),
            ("I(E;T\\|W) bits", "cw_cmi", 4), ("ncmi", "cw_ncmi", 4), ("ncmi shuf", "cw_ncmi_shuffled", 4),
            ("I(E;F) bits", "mi_format", 3), ("I(E;T\\|F) bits", "cmi_topic_given_format", 3),
            ("top-1 share", "mean_top1_share", 3), ("top-3 share", "mean_top3_share", 3),
            ("top lift", "mean_top_lift", 2), ("topic→top expert", "mean_topic_max_expert_share", 3),
            ("P(same as prev)", "p_same_prev", 3), ("chance", "p_same_prev_chance", 3),
            ("max load", "max_load", 3)]
    order = list(evals)
    colsp = [(t, pool) for t in order for pool in evals[t]["specialization"]]
    w("| metric | " + " | ".join(f"{t}" + ("" if pool == "main" else f" ({pool})") for t, pool in colsp) + " |")
    w("|---|" + "---:|" * len(colsp))
    rowsx = [(lab, (lambda e, pool, k=k: e["specialization"][pool]["mean"].get(k)), dd) for lab, k, dd in cols]
    rowsx += [("dead experts (all layers)", lambda e, pool: e["specialization"][pool]["mean"]["dead_total"], 0),
              ("ablation selectivity (load-weighted)",
               lambda e, pool: abl_metric(e, pool, "mean_selectivity_load_weighted"), 3),
              ("ablation selectivity (unweighted)", lambda e, pool: abl_metric(e, pool, "mean_selectivity"), 3),
              ("ablation match rate", lambda e, pool: abl_metric(e, pool, "match_rate"), 3),
              ("ablation match rate (load-weighted)", lambda e, pool: abl_metric(e, pool, "match_rate_load_weighted"), 3)]
    for lab, get, dd in rowsx:
        vals = []
        for t, pool in colsp:
            v = get(evals[t], pool)
            vals.append(str(v) if dd == 0 and v is not None else f(v, dd))
        w(f"| {lab} | " + " | ".join(vals) + " |")
    for t, pool in colsp:
        e = evals[t]
        m = e["specialization"][pool]["mean"]
        summary["runs"][t].setdefault("pools", {})[pool] = dict(
            **{k: m.get(k) for _, k, _ in cols}, dead=m["dead_total"],
            selectivity=abl_metric(e, pool, "mean_selectivity_load_weighted"),
            match_rate=abl_metric(e, pool, "match_rate"))
    w("")

    # ------------------------------------------------------------------ per-topic table
    if r1_tags:
        w("## Per-topic balanced-val loss, Δ vs R1 (mean of seeds; label of the target token)\n")
        r1pt = np.nanmean([evals[t]["balanced"]["per_topic"]["loss"] for t in r1_tags], axis=0)
        modes = [m for m in MODE_ORDER if m in by_mode and m != "r1"]
        w("| topic | bal tokens | nat tokens | R1 loss | " + " | ".join(f"Δ {MODE_NAME[m]}" for m in modes) + " |")
        w("|---|---:|---:|---:|" + "---:|" * len(modes))
        e0 = evals[r1_tags[0]]
        for i, tn in enumerate(TOPIC_NAMES):
            row = f"| {tn} | {e0['balanced']['per_topic']['tokens'][i]:,} | {e0['natural']['per_topic']['tokens'][i]:,} | {f(r1pt[i])} | "
            for m in modes:
                v = np.nanmean([evals[t]["balanced"]["per_topic"]["loss"][i] for t in by_mode[m]])
                row += f"{f(v - r1pt[i])} | "
            w(row)
        w("")

    # ------------------------------------------------------------------ claim check
    w("## Claim check (rule fixed in advance)\n")
    w("A run supports the claim iff (a) no-worse on natural val (Δ ≤ δ) **and** (b) its common-word normalized "
      "topic MI is ≥ 3 × R1's **and** (c) above its own shuffled-label baseline **and** (d) its ablation match "
      "rate beats R1's. Seeds are averaged where two exist. P4 is checked on its topic pool; its token pool is "
      "listed for completeness. Values are not rounded before comparison.\n")
    r1_cw = mean_or_none([pool_metric(evals[t], "main", "cw_nmi_top") for t in r1_tags])
    r1_match = mean_or_none([abl_metric(evals[t], "main", "match_rate") for t in r1_tags])
    w(f"R1 reference: common-word nmi = {f(r1_cw)}, match rate = {f(r1_match, 3)}.\n")
    w("| mode | pool | Δ natural | (a) no-worse | cw nmi | ×R1 | (b) ≥3×R1 | cw nmi shuf | (c) > shuf | match rate | (d) > R1 | supports claim |")
    w("|---|---|---:|---|---:|---:|---|---:|---|---:|---|---|")
    for mode in MODE_ORDER:
        if mode == "r1" or mode not in by_mode:
            continue
        tags = by_mode[mode]
        pools = list(evals[tags[0]]["specialization"].keys())
        for pool in pools:
            dn = summary["modes"][MODE_NAME[mode]]["delta_natural"]
            a = dn is not None and dn <= delta
            cw = mean_or_none([pool_metric(evals[t], pool, "cw_nmi_top") for t in tags])
            cws = mean_or_none([pool_metric(evals[t], pool, "cw_nmi_top_shuffled") for t in tags])
            mr = mean_or_none([abl_metric(evals[t], pool, "match_rate") for t in tags])
            ratio = cw / r1_cw if (cw is not None and r1_cw) else None
            b = ratio is not None and ratio >= 3
            c = cw is not None and cws is not None and cw > cws
            dd = mr is not None and r1_match is not None and mr > r1_match
            primary = CLAIM_POOL.get(mode, "main") == pool
            sup = a and b and c and dd
            yn = lambda v: "yes" if v else "no"  # noqa: E731
            w(f"| {MODE_NAME[mode]} | {pool}{'' if primary else ' (not used)'} | {f(dn)} | {yn(a)} | {f(cw)} | "
              f"{f(ratio, 2)} | {yn(b)} | {f(cws)} | {yn(c)} | {f(mr, 3)} | {yn(dd)} | "
              f"{('**YES**' if sup else 'no') if primary else '–'} |")
            if primary:
                summary["claim"][MODE_NAME[mode]] = dict(pool=pool, a_no_worse=a, b_3x_r1=b, c_above_shuffled=c,
                                                          d_match_beats_r1=dd, supports=sup, cw_nmi=cw,
                                                          cw_ratio_vs_r1=ratio, match_rate=mr)
    w("")
    if not r1_tags:
        w("**R1 has not been evaluated: the claim cannot be checked.**\n")

    # ------------------------------------------------------------------ ablation summary
    w("## Ablation selectivity\n")
    w("| run | pool | experts | mean selectivity | load-weighted | match rate | match (load-wt) | experts with no positive increase |")
    w("|---|---|---:|---:|---:|---:|---:|---:|")
    for t, e in evals.items():
        for pool, a in (e.get("ablation") or {}).items():
            w(f"| {t} | {pool} | {a['n_experts']} | {f(a['mean_selectivity'], 3)} | "
              f"{f(a['mean_selectivity_load_weighted'], 3)} | {f(a['match_rate'], 3)} | "
              f"{f(a['match_rate_load_weighted'], 3)} | {a['n_experts_no_positive_increase']} |")
    w("")

    # ------------------------------------------------------------------ over time
    w("## Specialization over training (balanced val; nmi per pool, mean over layers)\n")
    w("| run | pool | 25% | 50% | 75% | 100% | shuffled (100%) |")
    w("|---|---|---:|---:|---:|---:|---:|")
    for t, e in evals.items():
        ot = e.get("over_time", {})
        for pool in e["specialization"]:
            vals = [ot.get(p, {}).get(pool, {}).get("nmi") for p in ("25", "50", "75", "100")]
            w(f"| {t} | {pool} | " + " | ".join(f(v) for v in vals) + f" | {f(ot.get('100', {}).get(pool, {}).get('nmi_shuffled'))} |")
    w("")
    w("Quick validation loss during training (natural val, 2M tokens), gap vs R1 at each logged point:\n")
    curves = {}
    for t in RUN_INFO:
        v = [r for r in jsonl(paths.log(t)) if r.get("type") == "val"]
        if v:
            curves[t] = v
    if curves:
        steps = sorted({r["step"] for v in curves.values() for r in v})
        w("| run | " + " | ".join(str(s) for s in steps) + " |")
        w("|---|" + "---:|" * len(steps))
        r1c = {r["step"]: r["val_loss"] for r in curves.get("R1", [])}
        for t, v in curves.items():
            dv = {r["step"]: r["val_loss"] for r in v}
            w(f"| {t} | " + " | ".join(f(dv[s] - r1c[s]) if (s in dv and s in r1c and t != 'R1') else
                                      (f(dv.get(s)) if s in dv else "–") for s in steps) + " |")
        w("\n(R1 row: absolute loss; other rows: difference to R1 at the same step.)\n")

    # ------------------------------------------------------------------ routing health
    w("## Routing health (balanced val)\n")
    w("| run | pool | max load (worst layer) | dead experts (<1% load), all layers |")
    w("|---|---|---:|---:|")
    for t, e in evals.items():
        for pool, d in e["specialization"].items():
            w(f"| {t} | {pool} | {f(max(r['max_load'] for r in d['layers']), 3)} | {d['mean']['dead_total']} |")
    w("")

    # ------------------------------------------------------------------ data
    w("## Data\n")
    if meta:
        sh = meta.get("shards", {})
        w(f"- train shards: {len(sh.get('train', []))} ({sh.get('train', [None])[0]}…{sh.get('train', [None])[-1]}); "
          f"eval shards: {sh.get('eval')}")
        for s in ("train", "val", "bal", "abl"):
            m = meta.get(s, {})
            w(f"- {s}: {m.get('n_tokens', 0):,} tokens, {m.get('n_docs', 0):,} documents, "
              f"high-confidence token share {f(m.get('high_conf_token_share'), 3)}")
        if conf:
            w(f"- tokens per run: {conf.get('total_tokens', 0):,} ({conf.get('total_steps')} steps × "
              f"{conf.get('hparams', {}).get('tokens_per_step')}); passes over the training data: {f(conf.get('data_epochs'), 3)}")
        low = meta.get("bal", {}).get("topics_below_200_docs", [])
        w(f"- balanced-val topics with < 200 documents: {low if low else 'none'}")
        w("")
        w("| topic | train share | natural-val share | balanced tokens | balanced docs | ablation tokens |")
        w("|---|---:|---:|---:|---:|---:|")
        tt = np.asarray(meta["train"]["topic_tokens"], dtype=float)
        vt = np.asarray(meta["val"]["topic_tokens"], dtype=float)
        for i, tn in enumerate(TOPIC_NAMES):
            w(f"| {tn} | {tt[i] / tt.sum():.4f} | {vt[i] / max(vt.sum(), 1):.4f} | {meta['bal']['topic_tokens'][i]:,} | "
              f"{meta['bal']['topic_docs'][i]:,} | {meta['abl']['topic_tokens'][i]:,} |")
        w("")
        summary["data"] = dict(shards=sh, **{s: {k: meta[s][k] for k in ("n_tokens", "n_docs", "topic_tokens", "topic_docs")}
                                             for s in ("train", "val", "bal", "abl")}, topics_below_200_docs=low)

    # ------------------------------------------------------------------ compute / timeline / failures
    w("## Compute and timeline\n")
    if conf:
        w(f"- bench GPU: {conf.get('gpu')}; torch {conf.get('torch')}; throughput (tok/s): " +
          ", ".join(f"{k}: {v['tps']:,.0f}" for k, v in conf.get("bench", {}).items()))
        w("- estimated minutes per run: " + ", ".join(f"{k} {v:.0f}" for k, v in conf.get("est_minutes_per_run", {}).items()))
    rows = []
    if os.path.exists(paths.jobs_tsv):
        with open(paths.jobs_tsv) as fh:
            for line in fh:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 4 and p[0] != "time":
                    rows.append(dict(time=p[0], id=p[1], name=p[2], stage=p[3], tag=p[4] if len(p) > 4 else "",
                                     dep=p[5] if len(p) > 5 else "", note=p[6] if len(p) > 6 else ""))
    sa = sacct_rows([r["id"] for r in rows])
    if sa:
        w("\n| job id | name | state | submit | start | end | wait (min) | run (min) | node | note |")
        w("|---|---|---|---|---|---|---:|---:|---|---|")
        notes = {r["id"]: r["note"] for r in rows}
        for r in sa:
            w(f"| {r['id']} | {r['name']} | {r['state']} | {r['submit']} | {r['start']} | {r['end']} | "
              f"{f(tdiff_min(r['submit'], r['start']), 0)} | {f(tdiff_min(r['start'], r['end']), 0)} | {r['node']} | "
              f"{notes.get(r['id'], '')} |")
        bad = [r for r in sa if not r["state"].startswith(("COMPLETED", "RUNNING", "PENDING"))]
        w("\nFailed / cancelled jobs: " + (", ".join(f"{r['name']} ({r['id']}: {r['state']}, exit {r['exit']})"
                                                     for r in bad) if bad else "none") + "\n")
        summary["jobs"] = sa
    else:
        w("\n(sacct unavailable: timeline from training logs)\n")
        w("| run | start | end | train minutes |")
        w("|---|---|---|---:|")
        for t in RUN_INFO:
            lg = jsonl(paths.log(t))
            st = [r for r in lg if r.get("type") == "start"]
            en = [r for r in lg if r.get("type") == "end"]
            if st:
                w(f"| {t} | {time.strftime('%F %T', time.localtime(st[0]['time']))} | "
                  f"{time.strftime('%F %T', time.localtime(en[-1]['time'])) if en else '–'} | "
                  f"{f(en[-1].get('elapsed_min'), 1) if en else '–'} |")
    cont = [r for r in rows if "continuation" in r["note"] or "retry" in r["note"] or "waiting" in r["note"]]
    if cont:
        w("Continuations / retries / waits recorded in logs/jobs.tsv:\n")
        for r in cont:
            w(f"- {r['time']} {r['name']} job {r['id']}: {r['note']}")
        w("")
    errs = [(t, r) for t in RUN_INFO for r in jsonl(paths.log(t)) if r.get("type") in ("error", "pause")]
    if errs:
        w("Training log events (pauses / errors):\n")
        for t, r in errs:
            w(f"- {t} step {r.get('step')}: {r.get('type')} ({r.get('reason')})")
        w("")

    # ------------------------------------------------------------------ missing
    missing = [t for t in RUN_INFO if t not in evals]
    w("## Missing or unfinished runs\n")
    if missing:
        w("Not evaluated: " + ", ".join(missing) + ". To finish them, run `./submit_all.sh` again on the login node "
          "(it resumes unfinished runs from their checkpoints and evaluates them), then the report is rebuilt.\n")
    else:
        w("none\n")
    summary["missing"] = missing

    # ------------------------------------------------------------------ deviations
    w("## Deviations from the spec and implementation choices\n")
    for line in [
        "Dataset folders: topic logits are in `domains_topics__logits/` (README says `domain_topics__logits/`); "
        "format logits are in `domains_formats__logits/`. Labels align with documents by order (no id field); "
        "counts are asserted equal per shard.",
        "Raw shards are sorted by URL; documents are shuffled within each train shard (seed = shard id) and eval "
        "documents are pooled and shuffled (seed 1234) before building the eval sets.",
        "Balanced/ablation sets: documents of a topic are added in random order, skipping any that would exceed the "
        "per-topic token budget.",
        "P3: the topic loss is back-propagated into all parameters except the routers (one extra backward pass), so "
        "routers are trained only by the LM and load-balance losses; this makes P3 slower per step.",
        "Accuracy per topic/format/confidence uses the target token's document label; routing metrics use the "
        "routed (input) token's label. EOT tokens belong to the document they end.",
        "Within-document consistency uses the slot-0 (highest-probability) expert for top-2 routing.",
        "Example experts are restricted to experts with ≥ 1% load (avoids noise-driven lifts of near-dead experts).",
        "The report job runs on the CPU partition (no GPU needed).",
        "Bench also measures P3 (slowest mode) in addition to R1, R1f, P2, P4; P1's time is estimated from P2.",
    ]:
        w(f"- {line}")
    if conf.get("data_epochs", 0) > 1:
        w(f"- **Training data is shorter than TOTAL_TOKENS: {conf['data_epochs']:.3f} passes over the training stream.**")
    w("")
    write_json(os.path.join(paths.results, "summary.json"), summary)

    # ------------------------------------------------------------------ plots
    try:
        make_plots(plots, evals, curves, summary)
        w("## Plots\n")
        for p in sorted(os.listdir(plots)):
            if p.endswith(".png"):
                w(f"![{p}](plots/{p})")
        w("")
    except Exception as e:  # noqa: BLE001
        w(f"(plots failed: {type(e).__name__}: {e})\n")
    with open(os.path.join(paths.results, "REPORT.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {os.path.join(paths.results, 'REPORT.md')} ({len(evals)} runs evaluated; missing: {missing})")


def make_plots(plots, evals, curves, summary):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(plots, name), dpi=110)
        plt.close(fig)

    if curves:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for t, v in curves.items():
            ax.plot([r["tokens"] / 1e6 for r in v], [r["val_loss"] for r in v], marker="o", ms=3, label=t)
        ax.set_xlabel("training tokens (M)")
        ax.set_ylabel("natural-val loss (quick, 2M tokens)")
        ax.set_title("validation loss over training")
        ax.legend(fontsize=7, ncol=2)
        save(fig, "val_loss.png")
    if evals:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for t, e in evals.items():
            ot = e.get("over_time", {})
            for pool in e["specialization"]:
                xs, ys = [], []
                for p in ("25", "50", "75", "100"):
                    v = ot.get(p, {}).get(pool, {}).get("nmi")
                    if v is not None:
                        xs.append(int(p))
                        ys.append(v)
                if xs:
                    ax.plot(xs, ys, marker="o", label=f"{t}:{pool}")
        ax.set_xlabel("% of training")
        ax.set_ylabel("I(expert; topic) / H(topic)")
        ax.set_title("topic information over training (balanced val)")
        ax.legend(fontsize=7, ncol=2)
        save(fig, "topic_mi_over_training.png")
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for t, e in evals.items():
            for pool, d in e["specialization"].items():
                ax.plot([r["layer"] for r in d["layers"]], [r["nmi"] for r in d["layers"]], marker="o", label=f"{t}:{pool}")
        ax.set_xlabel("layer")
        ax.set_ylabel("normalized topic MI")
        ax.set_title("per-layer topic MI (balanced val)")
        ax.legend(fontsize=7, ncol=2)
        save(fig, "topic_mi_per_layer.png")
        panels = [(t, p) for t in ("R1", "P2", "P4") if t in evals for p in evals[t]["specialization"]]
        if panels:
            fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 6), squeeze=False)
            for ax, (t, pool) in zip(axes[0], panels):
                d = evals[t]["specialization"][pool]
                mid = len(d["layers"]) // 2
                lift = np.asarray(d["layers"][mid]["purity"]["lift"])
                im = ax.imshow(lift.T, aspect="auto", cmap="viridis", vmin=0, vmax=max(2.0, float(np.nanmax(lift))))
                ax.set_yticks(range(N_TOPICS))
                ax.set_yticklabels(TOPIC_NAMES, fontsize=6)
                ax.set_xticks(range(lift.shape[0]))
                ax.set_xlabel("expert")
                ax.set_title(f"{t} {pool}: lift, layer {mid}", fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.04)
            save(fig, "lift_heatmap_mid_layer.png")
        data, labs = [], []
        for t, e in evals.items():
            for pool, a in (e.get("ablation") or {}).items():
                s = [x for L in a["layers"] for x in L["selectivity"] if x is not None and np.isfinite(x)]
                if s:
                    data.append(s)
                    labs.append(f"{t}:{pool}")
        if data:
            fig, ax = plt.subplots(figsize=(max(6, 0.7 * len(data)), 4.2))
            ax.boxplot(data)
            ax.set_xticks(range(1, len(labs) + 1))
            ax.set_xticklabels(labs, rotation=60, fontsize=7)
            ax.axhline(1 / 24, ls="--", c="gray", lw=1)
            ax.set_ylabel("selectivity (max / sum of positive Δloss)")
            ax.set_title("ablation selectivity per expert (dashed: 1/24)")
            save(fig, "ablation_selectivity.png")


if __name__ == "__main__":
    main()
