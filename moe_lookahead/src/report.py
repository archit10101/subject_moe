"""Build results/REPORT.md, results/summary.json and results/plots/*.png from eval + logs."""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import RUN_INFO, RUNS, SOURCE_SHORT, Paths, read_json, write_json  # noqa: E402
from status import collect, read_jobs_tsv  # noqa: E402

# fixed entity colors (reference categorical palette, light mode); seed-1 runs: same hue, dashed
COLORS = {"r1": "#2a78d6", "r2": "#eb6834", "r3": "#1baf7a", "r4": "#eda100"}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def fnum(x, nd=4):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def load_log(path):
    recs = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return recs


# ----------------------------------------------------------------------------- timing
def parse_t(s):
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
    except (TypeError, ValueError):
        return None


def sacct_rows(ids):
    if not ids:
        return None
    try:
        out = subprocess.run(["sacct", "-X", "-n", "-P", "-j", ",".join(ids),
                              "--format=JobID,JobName,State,Submit,Eligible,Start,End,NodeList,ExitCode"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0 or not out.stdout.strip():
        return None
    rows = []
    for line in out.stdout.splitlines():
        p = line.split("|")
        if len(p) >= 9:
            rows.append(dict(id=p[0], name=p[1], state=p[2], submit=parse_t(p[3]), eligible=parse_t(p[4]),
                             start=parse_t(p[5]), end=parse_t(p[6]), node=p[7], exit=p[8]))
    return rows


def hm(seconds):
    if seconds is None:
        return "—"
    m = int(round(seconds / 60))
    return f"{m // 60}h{m % 60:02d}m"


# ----------------------------------------------------------------------------- plots
def make_plots(paths, logs, conf):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
    out = os.path.join(paths.results, "plots")
    os.makedirs(out, exist_ok=True)
    made = []

    def style(tag):
        info = RUN_INFO[tag]
        return dict(color=COLORS[info["mode"]], linestyle="--" if info["seed"] else "-", linewidth=2)

    # 1. validation loss vs tokens
    fig, ax = plt.subplots(figsize=(8, 4.5))
    any_line = False
    for tag, recs in logs.items():
        v = [r for r in recs if r.get("type") == "val"]
        if not v:
            continue
        x = [r["tokens"] / 1e6 for r in v]
        ax.plot(x, [r["val_loss"] for r in v], marker="o", markersize=4, label=tag, **style(tag))
        if RUN_INFO[tag]["mode"] == "r4" and "val_loss_zero_router" in v[0]:
            ax.plot(x, [r["val_loss_zero_router"] for r in v], marker="o", markersize=4, color=COLORS["r4"],
                    linestyle=":", linewidth=2, label=f"{tag} zero_router")
        any_line = True
    if any_line:
        ax.set_xlabel("training tokens (M)")
        qv = (conf.get("hparams") or {}).get("quick_val_tokens", 2_000_000)
        ax.set_ylabel(f"quick val loss (nats, first {qv / 1e6:g}M val tokens)")
        ax.set_title("Validation loss during training (R2/R3: student-only routing)", loc="left", color=INK)
        ax.legend(ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "val_loss.png"), dpi=130)
        made.append("val_loss.png")
    plt.close(fig)

    # 2. smoothed train-loss difference vs R1 (top) and alpha (bottom), shared x: no dual axis
    def series(tag):
        tr = [r for r in logs.get(tag, []) if r.get("type") == "train"]
        return {r["step"]: r for r in tr}

    base = series("R1")
    if base:
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 5.5), sharex=True, gridspec_kw=dict(height_ratios=[3, 1]))
        drew = False
        alpha_curve = None
        for tag in ("R2", "R3"):
            s = series(tag)
            common = sorted(set(s) & set(base))
            if len(common) < 2:
                continue
            d = np.array([s[k]["loss"] - base[k]["loss"] for k in common])
            win = max(1, len(d) // 25)
            sm = np.convolve(d, np.ones(win) / win, mode="valid")
            x = np.array([s[k]["tokens"] / 1e6 for k in common])[win - 1:]
            ax.plot(x, sm, label=f"{tag} − R1", **style(tag))
            if alpha_curve is None:
                alpha_curve = ([s[k]["tokens"] / 1e6 for k in common], [s[k].get("alpha") for k in common])
            drew = True
        if drew:
            ax.axhline(0, color=INK2, linewidth=1)
            ax.set_ylabel("Δ train loss vs R1 (nats)")
            ax.set_title(f"Smoothed training-loss difference (window ≈ {max(1, len(base) // 25)} log points)",
                         loc="left", color=INK)
            ax.legend()
            ax2.plot(alpha_curve[0], alpha_curve[1], color=INK2, linewidth=2)
            ax2.set_ylabel("teacher α")
            ax2.set_ylim(-0.05, 1.05)
            ax2.set_xlabel("training tokens (M)")
            fig.tight_layout()
            fig.savefig(os.path.join(out, "train_loss_diff.png"), dpi=130)
            made.append("train_loss_diff.png")
        plt.close(fig)

    # 3. student-teacher agreement vs tokens (mean over layers, band = min..max layer)
    fig, ax = plt.subplots(figsize=(8, 4))
    drew = False
    for tag in ("R2", "R3", "R2_seed1", "R3_seed1"):
        tr = [r for r in logs.get(tag, []) if r.get("type") == "train" and r.get("agree")]
        if not tr:
            continue
        x = np.array([r["tokens"] / 1e6 for r in tr])
        a = np.array([r["agree"] for r in tr])
        st = style(tag)
        ax.plot(x, a.mean(1), label=f"{tag} (mean of layers)", **st)
        ax.fill_between(x, a.min(1), a.max(1), color=st["color"], alpha=0.15, linewidth=0)
        drew = True
    if drew:
        ax.set_ylim(0, 1)
        ax.set_xlabel("training tokens (M)")
        ax.set_ylabel("student = teacher (fraction of tokens)")
        ax.set_title("Student–teacher agreement during training (band: min–max over layers)", loc="left", color=INK)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out, "agreement.png"), dpi=130)
        made.append("agreement.png")
    plt.close(fig)
    return made


# ----------------------------------------------------------------------------- decision
def headline(ev):
    return ev["variants"][ev["headline"]]["val_loss"] if ev else None


def decide(evals):
    def vals(mode_tags):
        return [headline(evals[t]) for t in mode_tags if evals.get(t)]

    r1v, r2v, r3v = vals(["R1", "R1_seed1"]), vals(["R2", "R2_seed1"]), vals(["R3", "R3_seed1"])
    out = dict(r1_values=r1v, r2_values=r2v, r3_values=r3v)
    if evals.get("R1") and evals.get("R1_seed1"):
        noise = abs(headline(evals["R1"]) - headline(evals["R1_seed1"]))
        out["noise"] = noise
        out["delta"] = max(0.01, 2 * noise)
        out["single_seed"] = False
    else:
        out["noise"] = None
        out["delta"] = 0.01
        out["single_seed"] = True
    if not (r1v and r2v and r3v):
        out["verdict"] = "incomplete"
        out["text"] = ("Cannot apply the decision rule: R1, R2 and R3 must all be finished and evaluated. "
                       "See 'Missing or failed runs' below.")
        return out
    R1, R2, R3, d = float(np.mean(r1v)), float(np.mean(r2v)), float(np.mean(r3v)), out["delta"]
    out.update(R1=R1, R2=R2, R3=R3)
    facts = (f"R1 = {R1:.4f}, R2 = {R2:.4f}, R3 = {R3:.4f} nats; δ = {d:.4f}. "
             f"R2 − R1 = {R2 - R1:+.4f}, R3 − R1 = {R3 - R1:+.4f}, R3 − R2 = {R3 - R2:+.4f}.")
    if R3 < R2 - d:
        out["verdict"] = "future_helped"
        extra = "" if R3 < R1 - d else (" Note: R3 does not beat the R1 baseline by more than δ, so the future "
                                          "teacher only helped relative to the past teacher.")
        out["text"] = f"**Seeing the future helped** (R3 < R2 − δ). {facts}{extra}"
    elif abs(R3 - R2) <= d and R2 < R1 - d and R3 < R1 - d:
        out["verdict"] = "subject_helped_future_irrelevant"
        out["text"] = f"**Subject supervision helped, and seeing the future did not matter** (|R3 − R2| ≤ δ, both < R1 − δ). {facts}"
    elif not (R2 < R1 - d) and not (R3 < R1 - d):
        out["verdict"] = "no_help"
        out["text"] = f"**Subject routing did not help at this scale** (neither R2 nor R3 beats R1 by more than δ). {facts}"
    else:
        out["verdict"] = "other"
        parts = []
        parts.append("R2 beats R1 by more than δ" if R2 < R1 - d else "R2 does not beat R1 by more than δ")
        parts.append("R3 beats R1 by more than δ" if R3 < R1 - d else "R3 does not beat R1 by more than δ")
        parts.append("R3 is worse than R2 by more than δ" if R3 > R2 + d else "R3 and R2 are within δ")
        out["text"] = f"**No clean category:** {'; '.join(parts)}. {facts}"
    return out


# ----------------------------------------------------------------------------- report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    args = ap.parse_args()
    paths = Paths(args.work)
    conf = read_json(paths.config, {}) or {}
    meta = read_json(paths.meta, {}) or {}
    clus = read_json(paths.clusters_json, {}) or {}
    ev_dir = os.path.join(paths.results, "eval")
    ref = read_json(os.path.join(ev_dir, "reference.json"), {}) or {}
    st = collect(paths)
    evals = {t: read_json(os.path.join(ev_dir, f"{t}.json")) for t in RUN_INFO}
    # drop eval files that do not belong to the current config/checkpoint
    for t, e in list(evals.items()):
        if e and (e.get("config_hash") != conf.get("config_hash") or st["runs"][t]["state"] != "done"):
            evals[t] = None
    r4ref = read_json(os.path.join(ev_dir, "R1_r4ref.json"))
    logs = {t: load_log(paths.log(t)) for t in RUN_INFO if os.path.exists(paths.log(t))}
    plots = make_plots(paths, logs, conf)
    dec = decide(evals)

    L = []
    w = L.append
    if meta.get("synthetic"):
        w("> ⚠️ **SYNTHETIC DATA — SMOKE / LOCAL TEST.** These numbers test the pipeline only; they say nothing "
          "about the hypothesis.\n")
    w("# Subject-aware MoE routing: report\n")
    w(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} from `{paths.work}`._\n")

    missing = [(t, st["runs"][t]) for t, _, _, tier in RUNS if st["runs"][t]["state"] != "done" and (tier <= 2 or st["runs"][t]["state"] != "missing")]
    if missing:
        w("> ❗ **Missing or unfinished runs:** " + ", ".join(
            f"{t} ({r['state']}" + (f", step {r.get('step')}/{r.get('total_steps')}" if r.get("step") is not None else "") + ")"
            for t, r in missing) + ". Re-run `./submit_all.sh` on the login node to resume them "
          "(it only submits what is missing), then the report is rebuilt automatically.\n")

    w("## Verdict\n")
    w(dec["text"] + "\n")
    if dec["single_seed"]:
        w(f"_Single seed: R1_seed1 is not available, so δ = 0.01 nats is a fixed guess, not a measured noise level._\n")
    else:
        w(f"_Noise estimate |R1 − R1_seed1| = {dec['noise']:.4f} nats → δ = max(0.01, 2 × noise) = {dec['delta']:.4f}._\n")
    w("Decision rule: R3 < R2 − δ → the future helped; |R3 − R2| ≤ δ and both < R1 − δ → subject supervision "
      "helped and the future did not matter; neither beats R1 by more than δ → no help at this scale; "
      "anything else is described as is. Values are means over seeds where two seeds exist. "
      "All R2/R3 numbers use **student-only routing** (honest, causal).\n")

    # main table
    w("## Main results (full validation stream)\n")
    w("| run | val loss | ppl | Δ vs R1 | first 64 tokens of doc | rest | step |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    r1 = headline(evals.get("R1"))
    for t in ("R1", "R2", "R3", "R1_seed1", "R2_seed1", "R3_seed1"):
        e = evals.get(t)
        if not e:
            if st["runs"][t]["state"] != "missing" or RUN_INFO[t]["tier"] <= 2:
                w(f"| {t} | _{st['runs'][t]['state']}_ | | | | | |")
            continue
        v = e["variants"][e["headline"]]
        dv = None if r1 is None else v["val_loss"] - r1
        w(f"| {t} | {fnum(v['val_loss'])} | {fnum(v['ppl'], 2)} | {fnum(dv, 4) if dv is not None else '—'} | "
          f"{fnum(v['first64'])} | {fnum(v['rest'])} | {e['step']} |")
    w("")
    for mode, tags in (("R1", ["R1", "R1_seed1"]), ("R2", ["R2", "R2_seed1"]), ("R3", ["R3", "R3_seed1"])):
        vs = [headline(evals[t]) for t in tags if evals.get(t)]
        if len(vs) == 2:
            w(f"- {mode}: two seeds {vs[0]:.4f} / {vs[1]:.4f} → mean {np.mean(vs):.4f}")
    w("\n'First 64 tokens of doc' = target tokens at most 64 positions after the latest preceding EOT "
      "(where subject information is scarcest). R3's teacher, for a token at offset k within its 64-token "
      "block, has seen 63 − k future tokens; R2's teacher only sees the past.\n")

    # per source
    w("## Loss by source (source of the target token)\n")
    w("| run | " + " | ".join(SOURCE_SHORT) + " |")
    w("|---|" + "---:|" * len(SOURCE_SHORT))
    any_e = None
    for t in RUN_INFO:
        e = evals.get(t)
        if not e:
            continue
        any_e = e
        by = e["variants"][e["headline"]]["by_source"]
        w(f"| {t} ({e['headline']}) | " + " | ".join(fnum(by.get(s, {}).get("loss")) for s in SOURCE_SHORT) + " |")
    if any_e:
        by = any_e["variants"][any_e["headline"]]["by_source"]
        w("| _tokens_ | " + " | ".join(f"{by.get(s, {}).get('n', 0):,}" for s in SOURCE_SHORT) + " |")
    w("")

    # leak check
    w("## Leak check (R2/R3): teacher routing vs student routing\n")
    w("A large gap (teacher ≪ student) means the routing decision carries information the causal student "
      "cannot reproduce; for R3 that includes the future.\n")
    w("| run | student loss | teacher (α=1) loss | gap (student − teacher) |")
    w("|---|---:|---:|---:|")
    for t in ("R2", "R3", "R2_seed1", "R3_seed1"):
        e = evals.get(t)
        if e:
            w(f"| {t} | {fnum(e['variants']['student']['val_loss'])} | {fnum(e['variants']['teacher']['val_loss'])} | "
              f"{fnum(e['leak_gap'])} |")
    w("")

    # routing diagnostics
    w("## Routing diagnostics (full validation stream, headline routing)\n")
    if ref:
        w(f"Reference: I(cluster; source) = **{ref.get('mi_cluster_source_bits', float('nan')):.3f} bits** "
          f"(token level, cluster of the token's own block). Max for 7 sources: 2.807 bits.\n")
    w("| run | I(expert; source) mean | per layer | max expert load per layer | load entropy (bits, max 3) | agreement per layer |")
    w("|---|---:|---|---|---|---|")
    for t in RUN_INFO:
        e = evals.get(t)
        if not e or "routing" not in e:
            continue
        r = e["routing"]
        ag = ", ".join(f"{x:.2f}" for x in r["agreement"]) if r.get("agreement") else "—"
        w(f"| {t} | {r['mi_expert_source_mean']:.3f} | {', '.join(f'{x:.2f}' for x in r['mi_expert_source_bits'])} | "
          f"{', '.join(f'{x:.2f}' for x in r['load_max'])} | {', '.join(f'{x:.2f}' for x in r['load_entropy_bits'])} | {ag} |")
    w("")

    # R4 demo
    w("## R4 demo: the naive leaky router\n")
    e4 = evals.get("R4")
    if e4:
        w("R4's router sees the mean hidden state of positions t+2 … t+51 (the future). Teacher-forced loss "
          "looks good; at generation time those positions don't exist (zero_router).\n")
        w("| model | val loss |")
        w("|---|---:|")
        w(f"| R4 leaky routing (teacher-forced view) | {fnum(e4['variants']['leaky']['val_loss'])} |")
        w(f"| R4 zero_router (generation-time view) | {fnum(e4['variants']['zero_router']['val_loss'])} |")
        if r4ref:
            w(f"| R1 at the same token count (R1_r4ref, step {r4ref['step']}) | {fnum(r4ref['variants']['normal']['val_loss'])} |")
        w("\n_R1_r4ref is a rough reference only: R1's learning rate had not finished decaying at that point, "
          "while R4 ran its own full schedule._\n")
    else:
        w(f"_R4 not available ({st['runs']['R4']['state']})._\n")

    # plots
    w("## Plots\n")
    for p in plots:
        w(f"![{p}](plots/{p})\n")
    if not plots:
        w("_No training logs found._\n")
    w("_The α schedule is drawn in its own panel under the loss difference (shared x axis) instead of on a "
      "second y axis._\n")

    # setup
    w("## Setup\n")
    if conf:
        hp = conf.get("hparams", {})
        pc = conf.get("param_counts") or {}
        w(f"- model: {conf.get('model')} {conf.get('model_cfg')}")
        w(f"- parameters: total {pc.get('total', 0) / 1e6:.1f}M, active per token {pc.get('active_per_token', 0) / 1e6:.1f}M, "
          f"embedding {pc.get('embedding', 0) / 1e6:.1f}M")
        w(f"- tokens per run: {conf.get('total_tokens', 0) / 1e6:.1f}M ({conf.get('total_steps')} steps × "
          f"{hp.get('tokens_per_step'):,}); R4: {conf.get('r4_tokens', 0) / 1e6:.1f}M ({conf.get('r4_steps')} steps)"
          + (" — **capped by available training tokens**" if conf.get("capped_by_data") else ""))
        w(f"- throughput (bench, min of R1/R3): {conf.get('tps', 0):,.0f} tokens/s on {conf.get('gpu')}; "
          f"projected {conf.get('projected_tokens_per_2h', 0) / 1e9:.2f}B tokens per 2 h")
        w(f"- optimizer: AdamW β={hp.get('betas')} eps={hp.get('eps')} wd={hp.get('weight_decay')} (≥2-D params), "
          f"peak lr {hp.get('lr')} → {hp.get('min_lr')}, {hp.get('lr_schedule')}, clip {hp.get('grad_clip')}; "
          f"micro-batch {hp.get('micro_bs')} × {hp.get('seq_len')} × {hp.get('grad_accum')} accumulation; {hp.get('precision')}")
        w(f"- routing: LB coef {hp.get('lb_coef')} (R1/R4), distillation coef {hp.get('distill_coef')} (R2/R3); α: {hp.get('alpha_schedule')}")
        w(f"- torch {conf.get('torch')} / CUDA {conf.get('cuda')}; data fingerprint {conf.get('data_fingerprint')}")
    if meta:
        w(f"- data: {meta['train']['n_tokens'] / 1e6:,.0f}M train tokens ({meta['train']['n_seq']:,} sequences), "
          f"{meta['val']['n_tokens'] / 1e6:,.1f}M validation tokens" + (" (synthetic)" if meta.get("synthetic") else ""))
    w("")
    w("### Clusters (teacher labels)\n")
    if clus:
        w(f"- featurizer: **{clus.get('featurizer')}**" + (f" — fallback: {clus['fallback_reason']}" if clus.get("fallback_reason") else ""))
        w(f"- train cluster sizes (%): {', '.join(f'{100 * x:.1f}' for x in clus['fractions']['train'])}; "
          f"balance converged: {clus.get('balance_converged')}")
        w(f"- I(cluster; block majority source): {clus['mi_bits_block_majority_source']['train']:.3f} bits (train). "
          f"Full table and sample texts: [clusters.md](clusters.md)\n")

    # GPUs + timing
    w("### GPUs and wall-clock\n")
    w("| run | GPU(s) / node(s) | job segments | training minutes |")
    w("|---|---|---:|---:|")
    for t in RUN_INFO:
        starts = [r for r in logs.get(t, []) if r.get("type") == "start"]
        if not starts:
            continue
        gpus = sorted({f"{r.get('gpu')} @ {r.get('host')}" for r in starts})
        side = read_json(paths.ckpt_side(t)) or {}
        w(f"| {t} | {'; '.join(gpus)} | {len(starts)} | {fnum(side.get('train_minutes'), 1)} |")
    w("")
    jobs = read_jobs_tsv(paths)
    rows = sacct_rows([j["jobid"] for j in jobs])
    timing = {}
    if rows:
        w("Per job (from `sacct`; wait = start − eligible, i.e. queue time after dependencies were met):\n")
        w("| job | state | node | submitted | wait | run |")
        w("|---|---|---|---|---:|---:|")
        for r in rows:
            wait = (r["start"] - (r["eligible"] or r["submit"])).total_seconds() if r["start"] and (r["eligible"] or r["submit"]) else None
            run = (r["end"] - r["start"]).total_seconds() if r["end"] and r["start"] else None
            w(f"| {r['name']} ({r['id']}) | {r['state']} | {r['node']} | {r['submit'] or '—'} | {hm(wait)} | {hm(run)} |")
        subs = [r["submit"] for r in rows if r["submit"]]
        ends = [r["end"] for r in rows if r["end"]]
        if subs and ends:
            timing["end_to_end_seconds"] = (max(ends) - min(subs)).total_seconds()
            w(f"\nFirst submission → last finished job: **{hm(timing['end_to_end_seconds'])}** "
              f"(the running eval job itself is not finished yet when this is written).\n")
    else:
        w("_`sacct` unavailable; stage durations from logs:_\n")
        w(f"- prep {hm(meta.get('prep_seconds'))}, clusters {hm(clus.get('seconds'))}, bench {hm(conf.get('bench_seconds'))}")
        ev_s = sum((e or {}).get("eval_seconds", 0) for e in evals.values())
        w(f"- eval (model passes) {hm(ev_s)}\n")

    w("## Missing or failed runs\n")
    if missing:
        for t, r in missing:
            w(f"- **{t}**: {r['state']}" + (f" (checkpoint at step {r.get('step')}/{r.get('total_steps')})" if r.get("step") is not None else ""))
        w("\nTo resume: on the login node run `./submit_all.sh` (add `--more-seeds` for tier 3). It resubmits only "
          "unfinished runs, which resume from their last checkpoint, followed by a new eval + report. "
          "For failures, send `logs/jobs.tsv` and the `logs/slurm_*_<jobid>.out` files of the failed jobs.\n")
    else:
        w("None.\n")

    summary = dict(generated=time.time(), synthetic=meta.get("synthetic", False), decision=dec,
                   runs={t: dict(state=st["runs"][t]["state"],
                                 val_loss=headline(evals.get(t)),
                                 eval=evals.get(t)) for t in RUN_INFO},
                   r1_r4ref=r4ref, reference=ref, config=conf, clusters=dict(
                       featurizer=clus.get("featurizer"), fallback_reason=clus.get("fallback_reason"),
                       mi=clus.get("mi_bits_block_majority_source"), fractions=clus.get("fractions")),
                   timing=timing, plots=plots)
    write_json(os.path.join(paths.results, "summary.json"), summary)
    with open(os.path.join(paths.results, "REPORT.md"), "w") as f:
        f.write("\n".join(L) + "\n")
    print(f"wrote {os.path.join(paths.results, 'REPORT.md')} (verdict: {dec['verdict']})")


if __name__ == "__main__":
    main()
