"""Evaluate one finished run -> results/eval/<tag>.json and results/examples/<tag>.md.

  python src/eval.py --tag R1 [--batch 32] [--max_tokens N] [--abl_max_tokens N]

Exit codes: 0 ok | 5 the run is not finished (no done checkpoint) | 2 config/staleness error

Label conventions
  accuracy (loss / next-token accuracy) per topic / format / confidence: label of the TARGET token's document
  routing metrics: label of the ROUTED (input) token's document
  balanced val is used for every specialization metric; the ablation subset for ablation selectivity.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M  # noqa: E402
from common import (MILESTONES, N_FORMATS, N_TOPICS, RUN_INFO, SEQ_LEN, THE_TOKEN, TOPIC_NAMES, Paths, Split,  # noqa: E402
                    load_meta, log, read_json, write_json)
from data import seq_batch, to_device  # noqa: E402
from model import GPT, ModelConfig  # noqa: E402
from train import autocast, config_hash, gpu_name, setup_device  # noqa: E402

EXIT_NOT_DONE = 5
EXAMPLE_TOPICS = ("Science & Tech.", "History", "Literature")


class Pass:
    """Results of one forward pass over a set: per-position loss / correctness, routes per (layer, pool)."""


@torch.no_grad()
def forward_set(model, split, n_seq, batch, device, keep_ids=True):
    model.eval()
    N = n_seq * SEQ_LEN
    loss = np.empty(N, np.float32)
    correct = np.empty(N, bool)
    pools = [(li, n, p) for li, n, p in model.pools()]
    ids = {(li, n): np.empty((N, p.top_k), np.uint8) for li, n, p in pools} if keep_ids else None
    counts = {(li, n): np.zeros(p.n_exp, np.int64) for li, n, p in pools}
    for s in range(0, n_seq, batch):
        sid = np.arange(s, min(s + batch, n_seq))
        x, y = seq_batch(split.tokens, sid)
        x, y = to_device(x, device), to_device(y, device)
        with autocast(device):
            out = model(x, y, reduction="none", keep_ids=keep_ids, return_logits=True)
        a, b = s * SEQ_LEN, (s + len(sid)) * SEQ_LEN
        loss[a:b] = out.loss.float().cpu().numpy()
        correct[a:b] = (out.logits.argmax(-1) == y).reshape(-1).cpu().numpy()
        for li, st in enumerate(out.stats):
            for n, c in st["counts"].items():
                counts[(li, n)] += c.cpu().numpy()
                if keep_ids:
                    ids[(li, n)][a:b] = st["ids"][n].cpu().numpy().astype(np.uint8)
    r = Pass()
    r.loss, r.correct, r.ids, r.counts = loss, correct, ids, counts
    return r


def labels(split, n_seq):
    """Per-position labels for the first n_seq windows of a split."""
    N = n_seq * SEQ_LEN
    pos = np.arange(N, dtype=np.int64)
    L = Pass()
    L.tok = np.asarray(split.tokens[:N], dtype=np.int64)
    L.topic_in = np.asarray(split.topic[:N], dtype=np.int64)
    L.format_in = np.asarray(split.format[:N], dtype=np.int64)
    L.topic_tg = np.asarray(split.topic[1:N + 1], dtype=np.int64)
    L.format_tg = np.asarray(split.format[1:N + 1], dtype=np.int64)
    L.conf_tg = np.asarray(split.conf[1:N + 1], dtype=bool)
    L.doc = split.doc_index(pos)
    L.same_doc_prev = np.concatenate([[False], L.doc[1:] == L.doc[:-1]]) & (pos % SEQ_LEN != 0)
    L.doc_topic = split.info[:, 0].astype(np.int64)
    return L


def accuracy(p, L):
    def agg(mask=None):
        lo, co = (p.loss, p.correct) if mask is None else (p.loss[mask], p.correct[mask])
        n = len(lo)
        m = float(lo.astype(np.float64).mean()) if n else float("nan")
        return dict(loss=m, ppl=float(np.exp(m)) if n else float("nan"), acc=float(co.mean()) if n else float("nan"),
                    tokens=int(n))

    def by(lab, k):
        cnt = np.bincount(lab, minlength=k)
        ls = np.bincount(lab, weights=p.loss.astype(np.float64), minlength=k)
        ac = np.bincount(lab, weights=p.correct.astype(np.float64), minlength=k)
        with np.errstate(invalid="ignore", divide="ignore"):
            return dict(loss=(ls / cnt).tolist(), acc=(ac / cnt).tolist(), tokens=cnt.tolist())

    hc = L.conf_tg
    res = dict(overall=agg(), per_topic=by(L.topic_tg, N_TOPICS), per_format=by(L.format_tg, N_FORMATS),
               high_conf=agg(hc))
    res["high_conf_per_topic"] = by(np.where(hc, L.topic_tg, N_TOPICS), N_TOPICS + 1)
    for k in ("loss", "acc", "tokens"):
        res["high_conf_per_topic"][k] = res["high_conf_per_topic"][k][:N_TOPICS]
    return res


def layer_mean(rows, key):
    v = [r[key] for r in rows if r.get(key) is not None and np.isfinite(r[key])]
    return float(np.mean(v)) if v else None


def specialization(model, p, L, shuffled_topic, min_word_count, n_top=100):
    """Metrics 1-5, 7 per (layer, pool), plus layer means per pool."""
    uniq, cnt = np.unique(L.tok, return_counts=True)
    top_ids = uniq[np.argsort(-cnt, kind="stable")[:n_top]]
    out = {}
    for name in model.blocks[0].moe.pools.keys():
        rows = []
        for li in range(len(model.blocks)):
            pool = model.blocks[li].moe.pools[name]
            E = pool.n_exp
            ids = p.ids[(li, name)]
            r = dict(layer=li)
            ti = M.topic_info(ids, L.topic_in, E, N_TOPICS)
            r.update(mi=ti["mi"], nmi=ti["nmi"], h_topic=ti["h_topic"])
            ts = M.topic_info(ids, shuffled_topic, E, N_TOPICS)
            r.update(mi_shuffled=ts["mi"], nmi_shuffled=ts["nmi"])
            cw = M.common_word(ids, L.topic_in, L.tok, E, N_TOPICS, top_ids, min_word_count)
            cws = M.common_word(ids, shuffled_topic, L.tok, E, N_TOPICS, top_ids, min_word_count)
            r.update({f"cw_{k}": v for k, v in cw.items()})
            r.update({f"cw_{k}_shuffled": cws[k] for k in ("mi_top", "nmi_top", "cmi", "ncmi")})
            r.update(M.format_control(ids, L.topic_in, L.format_in, E, N_TOPICS, N_FORMATS))
            pu = M.purity(ids, L.topic_in, E, N_TOPICS)
            r["purity"] = pu
            r.update(mean_top1_share=pu["mean_top1_share"], mean_top3_share=pu["mean_top3_share"],
                     mean_top_lift=pu["mean_top_lift"], mean_topic_max_expert_share=pu["mean_topic_max_expert_share"])
            co = M.consistency(ids, L.same_doc_prev)
            r.update(p_same_prev=co["p_same"], p_same_prev_chance=co["chance"])
            r["health"] = M.health(ids, E)
            r.update(max_load=r["health"]["max_load"], dead=r["health"]["dead"])
            rows.append(r)
        keys = ["mi", "nmi", "mi_shuffled", "nmi_shuffled", "cw_mi_top", "cw_nmi_top", "cw_mi_top_shuffled",
                "cw_nmi_top_shuffled", "cw_cmi", "cw_ncmi", "cw_cmi_shuffled", "cw_ncmi_shuffled", "mi_format",
                "nmi_format", "cmi_topic_given_format", "ncmi_topic_given_format", "mean_top1_share",
                "mean_top3_share", "mean_top_lift", "mean_topic_max_expert_share", "p_same_prev",
                "p_same_prev_chance", "max_load"]
        mean = {k: layer_mean(rows, k) for k in keys}
        mean["dead_total"] = int(sum(r["dead"] for r in rows))
        mean["cw_n_words"] = rows[0]["cw_n_words"]
        mean["cw_top_ids_share"] = rows[0]["cw_top_ids_share"]
        out[name] = dict(n_exp=model.blocks[0].moe.pools[name].n_exp,
                         top_k=model.blocks[0].moe.pools[name].top_k, layers=rows, mean=mean)
    return out, top_ids


@torch.no_grad()
def ablation(model, split, n_seq, batch, device, spec):
    """Disable each expert of each pool in each layer (rerouted to next-best); loss change per topic."""
    N = n_seq * SEQ_LEN
    tg = np.asarray(split.topic[1:N + 1], dtype=np.int64)
    cnt = np.bincount(tg, minlength=N_TOPICS).astype(np.float64)

    def per_topic():
        p = forward_set(model, split, n_seq, batch, device, keep_ids=False)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.bincount(tg, weights=p.loss.astype(np.float64), minlength=N_TOPICS) / cnt, p.loss.mean()

    base, base_all = per_topic()
    res = {}
    t0 = time.time()
    for name in model.blocks[0].moe.pools.keys():
        layers = []
        for li in range(len(model.blocks)):
            pool = model.blocks[li].moe.pools[name]
            delta = np.zeros((pool.n_exp, N_TOPICS))
            dall = np.zeros(pool.n_exp)
            for e in range(pool.n_exp):
                pool.disabled = e
                try:
                    lt, la = per_topic()
                finally:
                    pool.disabled = None
                delta[e] = lt - base
                dall[e] = la - base_all
            sel, hurt = M.selectivity(np.nan_to_num(delta, nan=0.0))
            pu = spec[name]["layers"][li]["purity"]
            top_lift_topic = np.asarray(pu["top_topic"])
            load = np.asarray(pu["load"])
            layers.append(dict(layer=li, delta=delta.tolist(), delta_overall=dall.tolist(), selectivity=sel.tolist(),
                               most_hurt=hurt.tolist(), top_lift_topic=top_lift_topic.tolist(), load=load.tolist(),
                               match=(hurt == top_lift_topic).tolist()))
            log(f"  ablation {name} layer {li}: mean selectivity {np.nanmean(sel):.3f} "
                f"({time.time() - t0:.0f}s)")
        sel = np.array([s for L in layers for s in L["selectivity"]], dtype=np.float64)
        w = np.array([x for L in layers for x in L["load"]], dtype=np.float64)
        match = np.array([m for L in layers for m in L["match"]], dtype=np.float64)
        ok = np.isfinite(sel)
        res[name] = dict(
            layers=layers, base_loss_per_topic=base.tolist(), base_loss=float(base_all),
            mean_selectivity_load_weighted=float((sel[ok] * w[ok]).sum() / max(w[ok].sum(), 1e-12)) if ok.any() else None,
            mean_selectivity=float(sel[ok].mean()) if ok.any() else None,
            match_rate=float(match[ok].mean()) if ok.any() else None,
            match_rate_load_weighted=float((match[ok] * w[ok]).sum() / max(w[ok].sum(), 1e-12)) if ok.any() else None,
            n_experts=int(len(sel)), n_experts_no_positive_increase=int((~ok).sum()),
            selectivity_hist=np.histogram(sel[ok], bins=np.linspace(0, 1, 21))[0].tolist(),
            chance_selectivity=1 / N_TOPICS, chance_match=1 / N_TOPICS)
    return res


def examples_md(tag, model, p, L, spec, split, n_seq, rng):
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    mid = len(model.blocks) // 2
    lines = [f"# Routing examples: {tag} (layer {mid}, balanced val)", "",
             "Experts with the highest top-topic lift among experts with >= 1% load. Snippets: 40 tokens of "
             "context ending at the routed token, which is marked ⟦like this⟧; drawn from documents of the "
             "expert's top topic.", ""]
    tok = L.tok

    def snippet(i):
        a = max(0, (i // SEQ_LEN) * SEQ_LEN, i - 39)
        ctx = enc.decode(tok[a:i].tolist()).replace("\n", "⏎")
        return f"…{ctx}⟦{enc.decode([int(tok[i])]).replace(chr(10), '⏎')}⟧"

    for name, d in spec.items():
        pu = d["layers"][mid]["purity"]
        load = np.asarray(pu["load"])
        lift = np.asarray(pu["lift"])
        cand = [e for e in np.argsort(-np.asarray(pu["top_lift"])) if load[e] >= 0.01][:4]
        lines += [f"## pool `{name}` ({d['n_exp']} experts, top-{d['top_k']})", ""]
        ids = p.ids[(mid, name)]
        for e in cand:
            tops = np.argsort(-lift[e])[:3]
            lines.append(f"### expert {e}: load {load[e]:.3f}, top topics " + ", ".join(
                f"{TOPIC_NAMES[t]} (lift {lift[e, t]:.2f})" for t in tops))
            hit = np.nonzero((ids == e).any(1) & (L.topic_in == tops[0]) & (np.arange(len(tok)) % SEQ_LEN >= 39))[0]
            for i in (rng.choice(hit, min(3, len(hit)), replace=False) if len(hit) else []):
                lines.append(f"- `{snippet(int(i))}`")
            lines.append("")
        lines += [f"### where the token ' the' goes ({name} pool, layer {mid})", "",
                  "| document topic | ' the' tokens | top experts (share of slots) |", "|---|---:|---|"]
        is_the = tok == THE_TOKEN
        for tn in EXAMPLE_TOPICS + ("(all topics)",):
            m = is_the if tn == "(all topics)" else is_the & (L.topic_in == TOPIC_NAMES.index(tn))
            e = ids[m].reshape(-1).astype(np.int64)
            if len(e) == 0:
                lines.append(f"| {tn} | 0 | |")
                continue
            c = np.bincount(e, minlength=d["n_exp"]) / len(e)
            lines.append(f"| {tn} | {int(m.sum())} | " + ", ".join(f"e{k}: {c[k]:.2f}" for k in np.argsort(-c)[:4]) + " |")
        lines.append("")
    return "\n".join(lines)


def load_model(path, cfg, mode, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = GPT(cfg, mode).to(device)
    m.load_state_dict(ck["model"])
    m.eval()
    return m, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--max_tokens", type=float, default=0, help="testing: cap tokens of val / bal")
    ap.add_argument("--abl_max_tokens", type=float, default=0, help="testing: cap tokens of the ablation set")
    ap.add_argument("--min_word_count", type=int, default=2000)
    ap.add_argument("--skip_ablation", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    paths = Paths(args.work).mk()
    info = RUN_INFO[args.tag]
    conf = read_json(paths.config)
    meta = load_meta(paths)
    if conf is None or conf.get("data_fingerprint") != meta["fingerprint"]:
        log("CONFIG ERROR: results/config.json missing or made from different data")
        sys.exit(2)
    chash = config_hash(conf)
    side = read_json(paths.ckpt_side(args.tag))
    if not side or not side.get("done") or not os.path.exists(paths.ckpt_last(args.tag)):
        log(f"{args.tag} is not finished (checkpoint step {side and side.get('step')}); nothing to evaluate")
        sys.exit(EXIT_NOT_DONE)
    if side.get("config_hash") != chash:
        log(f"CONFIG ERROR: {args.tag} checkpoint was trained under a different config.json")
        sys.exit(2)
    device = setup_device(args.device)
    cfg = ModelConfig(**conf["model_cfg"])
    model, ck = load_model(paths.ckpt_last(args.tag), cfg, info["mode"], device)
    cap = lambda s, c: min(s.n_seq, int(c // SEQ_LEN) if c else s.n_seq)  # noqa: E731
    val = Split(paths, "val", meta)
    bal = Split(paths, "bal", meta)
    abl = Split(paths, "abl", meta)
    nv, nb, na = cap(val, args.max_tokens), cap(bal, args.max_tokens), cap(abl, args.abl_max_tokens)
    log(f"eval {args.tag} ({info['mode']}, step {ck['step']}) on {gpu_name(device)}: val {nv} / bal {nb} / abl {na} windows")
    res = dict(tag=args.tag, mode=info["mode"], init_seed=info["seed"], step=int(ck["step"]),
               total_steps=int(ck["total_steps"]), tokens=int(ck["step"]) * conf["hparams"]["tokens_per_step"],
               train_minutes=ck.get("train_minutes"), config_hash=chash, ckpt_saved=side.get("saved"),
               gpu=gpu_name(device), data_fingerprint=meta["fingerprint"], synthetic=meta.get("synthetic", False),
               sets=dict(val_tokens=nv * SEQ_LEN, bal_tokens=nb * SEQ_LEN, abl_tokens=na * SEQ_LEN),
               conventions=dict(accuracy_labels="target token's document", routing_labels="routed token's document",
                                min_word_count=args.min_word_count, common_word_top_n=100,
                                consistency="slot-0 (highest-probability) expert", dead_threshold=0.01))
    pv = forward_set(model, val, nv, args.batch, device, keep_ids=False)
    Lv = labels(val, nv)
    res["natural"] = accuracy(pv, Lv)
    res["natural"]["load"] = {f"L{li}.{n}": (c / max(c.sum(), 1)).tolist() for (li, n), c in pv.counts.items()}
    log(f"natural val: loss {res['natural']['overall']['loss']:.4f} acc {res['natural']['overall']['acc']:.4f}")
    del pv, Lv
    pb = forward_set(model, bal, nb, args.batch, device, keep_ids=True)
    Lb = labels(bal, nb)
    res["balanced"] = accuracy(pb, Lb)
    log(f"balanced val: loss {res['balanced']['overall']['loss']:.4f} acc {res['balanced']['overall']['acc']:.4f}")
    shuf = M.shuffled_topic(Lb.doc_topic, Lb.doc, seed=0)
    spec, top_ids = specialization(model, pb, Lb, shuf, args.min_word_count)
    res["specialization"] = spec
    res["common_word_top_ids"] = top_ids.tolist()
    for name, d in spec.items():
        m = d["mean"]
        log(f"  {name}: I(E;T) {m['mi']:.3f} bits (nmi {m['nmi']:.3f}, shuffled {m['nmi_shuffled']:.4f}); "
            f"common words nmi {m['cw_nmi_top']:.3f} cmi {m['cw_cmi']:.3f}; I(E;F) {m['mi_format']:.3f} "
            f"I(E;T|F) {m['cmi_topic_given_format']:.3f}; dead {m['dead_total']}")
    os.makedirs(os.path.dirname(paths.examples(args.tag)), exist_ok=True)
    with open(paths.examples(args.tag), "w") as f:
        f.write(examples_md(args.tag, model, pb, Lb, spec, bal, nb, np.random.default_rng(0)))
    del pb
    if not args.skip_ablation:
        log("ablation selectivity ...")
        res["ablation"] = ablation(model, abl, na, args.batch, device, spec)
        for name, a in res["ablation"].items():
            log(f"  {name}: load-weighted selectivity {a['mean_selectivity_load_weighted']}, match rate {a['match_rate']}")
    # specialization over training (metrics 1 and 3 only)
    res["over_time"] = {}
    for pct in MILESTONES:
        pth = paths.ckpt_milestone(args.tag, pct)
        if not os.path.exists(pth):
            continue
        mm, ckm = load_model(pth, cfg, info["mode"], device)
        if ckm.get("config_hash") != chash:
            log(f"  skipping {pth}: different config")
            continue
        pm = forward_set(mm, bal, nb, args.batch, device, keep_ids=True)
        row = dict(step=int(ckm["step"]), tokens=int(ckm.get("tokens", 0)), bal_loss=float(pm.loss.mean()))
        for name in mm.blocks[0].moe.pools.keys():
            E = mm.blocks[0].moe.pools[name].n_exp
            per = []
            for li in range(len(mm.blocks)):
                ids = pm.ids[(li, name)]
                ti = M.topic_info(ids, Lb.topic_in, E, N_TOPICS)
                fc = M.format_control(ids, Lb.topic_in, Lb.format_in, E, N_TOPICS, N_FORMATS)
                per.append(dict(mi=ti["mi"], nmi=ti["nmi"], nmi_shuffled=M.topic_info(ids, shuf, E, N_TOPICS)["nmi"],
                                **fc))
            row[name] = dict(layers=per, **{k: float(np.mean([r[k] for r in per])) for k in per[0]})
        res["over_time"][str(pct)] = row
        log(f"  {pct}% checkpoint (step {row['step']}): " + ", ".join(
            f"{n} nmi {row[n]['nmi']:.3f}" for n in mm.blocks[0].moe.pools.keys()))
        del mm, pm
    final = dict(step=res["step"], tokens=res["tokens"], bal_loss=res["balanced"]["overall"]["loss"])
    for name, d in spec.items():
        final[name] = {k: d["mean"][k] for k in ("mi", "nmi", "nmi_shuffled", "mi_format", "cmi_topic_given_format")}
    res["over_time"]["100"] = final
    res["eval_seconds"] = time.time() - t0
    write_json(paths.eval_json(args.tag), res)
    log(f"wrote {paths.eval_json(args.tag)} and {paths.examples(args.tag)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
