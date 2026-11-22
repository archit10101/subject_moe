"""Evaluate every finished run on the full validation stream (eval job).

The validation stream is chunked in order into 1024-token sequences (no shuffling).
Headline numbers: R1 normal routing; R2/R3 STUDENT-ONLY routing (alpha = 0); R4 leaky routing.
Extra variants: R2/R3 teacher routing (alpha = 1) for the leak check; R4 zero_router (generation view).
Results: results/eval/<tag>.json (cached; recomputed only when the checkpoint changes) and
results/eval/reference.json (I(cluster; source) and run status at eval time).
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    EOT, N_CLUSTERS, N_SOURCES, RUN_INFO, SEQ_LEN, SOURCE_SHORT, Paths, log, mutual_info_bits,
    read_json, write_json,
)
from data import seq_batch, teacher_ids, to_device  # noqa: E402
from model import GPT, ModelConfig  # noqa: E402
from train import ConfigError, autocast, config_hash, gpu_name, load_run_config, setup_device  # noqa: E402


class ValData:
    def __init__(self, paths, meta, max_tokens=None):
        n = meta["val"]["n_tokens"]
        self.tok = np.fromfile(paths.tokens("val"), dtype=np.uint16, count=n)
        self.src = np.fromfile(paths.src("val"), dtype=np.uint8, count=n)
        self.clusters = np.load(paths.cluster("val"))
        self.n_seq = meta["val"]["n_seq"]
        if max_tokens:
            self.n_seq = max(1, min(self.n_seq, int(max_tokens) // SEQ_LEN))
        N = self.n_seq * SEQ_LEN
        self.N = N
        self.in_src = self.src[:N].astype(np.int64)
        self.tgt_src = self.src[1:N + 1].astype(np.int64)
        # target k is "first 64 of a document" if k - (latest EOT strictly before k) <= 64
        idx = np.arange(N + 1, dtype=np.int64)
        e = np.where(self.tok[:N + 1] == EOT, idx, -1)
        last_incl = np.maximum.accumulate(e)
        k = idx[1:]
        prev = last_incl[:-1]
        self.first64 = (k - prev) <= 64
        # token-level future-teacher cluster for the reference I(cluster; source)
        self.cluster_tok = np.repeat(self.clusters[: (N + 63) // 64], 64)[:N].astype(np.int64)


@torch.no_grad()
def run_variant(model, vd, mode, device, batch, alpha=0.0, zero_router=False, keep_ids=True):
    L = model.cfg.n_layer
    losses = np.empty(vd.N, dtype=np.float32)
    ids = np.empty((L, vd.N), dtype=np.uint8) if keep_ids else None
    agree = np.zeros(L, dtype=np.int64)
    for s in range(0, vd.n_seq, batch):
        seqs = np.arange(s, min(s + batch, vd.n_seq))
        x, y = seq_batch(vd.tok, seqs)
        t = teacher_ids(vd.clusters, seqs, mode)
        x, y = to_device(x, device), to_device(y, device)
        if t is not None:
            t = to_device(t, device)
        with autocast(device):
            out = model(x, y, teacher=t, alpha=alpha, zero_router=zero_router, reduction="none",
                        keep_ids=keep_ids)
        a, b = s * SEQ_LEN, (s + len(seqs)) * SEQ_LEN
        losses[a:b] = out.loss.float().cpu().numpy()
        if keep_ids:
            for li, st in enumerate(out.stats):
                ids[li, a:b] = st["ids"].to(torch.uint8).cpu().numpy()
                if "agree" in st:
                    agree[li] += int(st["agree"].item())
    return losses, ids, agree


def loss_metrics(losses, vd):
    m = dict(val_loss=float(losses.mean()), ppl=float(np.exp(losses.mean())), n_tokens=int(len(losses)))
    by = {}
    for s in range(N_SOURCES):
        sel = vd.tgt_src == s
        if sel.any():
            by[SOURCE_SHORT[s]] = dict(loss=float(losses[sel].mean()), n=int(sel.sum()))
    m["by_source"] = by
    m["first64"] = float(losses[vd.first64].mean()) if vd.first64.any() else None
    m["rest"] = float(losses[~vd.first64].mean()) if (~vd.first64).any() else None
    m["n_first64"] = int(vd.first64.sum())
    return m


def routing_metrics(ids, vd, n_exp, agree=None):
    L = ids.shape[0]
    load_max, load_ent, mi = [], [], []
    for li in range(L):
        c = np.bincount(ids[li], minlength=n_exp).astype(np.float64)
        f = c / c.sum()
        load_max.append(float(f.max()))
        nz = f[f > 0]
        load_ent.append(float(-(nz * np.log2(nz)).sum()))
        mi.append(mutual_info_bits(ids[li], vd.in_src, n_exp, N_SOURCES))
    r = dict(load_max=load_max, load_entropy_bits=load_ent, mi_expert_source_bits=mi,
             mi_expert_source_mean=float(np.mean(mi)))
    if agree is not None and agree.sum() > 0:
        r["agreement"] = (agree / vd.N).tolist()
        r["agreement_mean"] = float(np.mean(agree / vd.N))
    return r


def evaluate_run(tag, mode, ck_path, cfg, vd, device, batch):
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = GPT(cfg, mode)
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    t0 = time.time()
    res = dict(variants={})
    if mode == "r1":
        losses, ids, _ = run_variant(model, vd, mode, device, batch)
        res["variants"]["normal"] = loss_metrics(losses, vd)
        res["routing"] = routing_metrics(ids, vd, cfg.n_exp)
        res["headline"] = "normal"
    elif mode in ("r2", "r3"):
        losses, ids, agree = run_variant(model, vd, mode, device, batch, alpha=0.0)
        res["variants"]["student"] = loss_metrics(losses, vd)
        res["routing"] = routing_metrics(ids, vd, cfg.n_exp, agree)
        losses_t, ids_t, _ = run_variant(model, vd, mode, device, batch, alpha=1.0, keep_ids=False)
        res["variants"]["teacher"] = loss_metrics(losses_t, vd)
        res["leak_gap"] = res["variants"]["student"]["val_loss"] - res["variants"]["teacher"]["val_loss"]
        res["headline"] = "student"
    else:  # r4
        losses, ids, _ = run_variant(model, vd, mode, device, batch)
        res["variants"]["leaky"] = loss_metrics(losses, vd)
        res["routing"] = routing_metrics(ids, vd, cfg.n_exp)
        losses_z, _, _ = run_variant(model, vd, mode, device, batch, zero_router=True, keep_ids=False)
        res["variants"]["zero_router"] = loss_metrics(losses_z, vd)
        res["headline"] = "leaky"
    res.update(tag=tag, mode=mode, step=int(ck.get("step", 0)), total_steps=ck.get("total_steps"),
               tokens=int(ck.get("step", 0)) * int(ck.get("tokens_per_step", 0) or 0),
               config_hash=ck.get("config_hash"), ckpt_saved=ck.get("saved"),
               eval_seconds=time.time() - t0, eval_gpu=gpu_name(device))
    del model, ck
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return res


def run_status(paths, chash):
    st = {}
    for tag in RUN_INFO:
        side = read_json(paths.ckpt_side(tag))
        if side is None:
            st[tag] = dict(state="missing")
        elif side.get("config_hash") != chash:
            st[tag] = dict(state="stale", step=side.get("step"), total_steps=side.get("total_steps"))
        elif side.get("done"):
            st[tag] = dict(state="done", step=side["step"], total_steps=side["total_steps"])
        else:
            st[tag] = dict(state="incomplete", step=side.get("step"), total_steps=side.get("total_steps"))
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    ap.add_argument("--tags", nargs="*", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--max_val_tokens", type=float, default=None, help="testing only")
    args = ap.parse_args()
    paths = Paths(args.work).mk()
    out_dir = os.path.join(paths.results, "eval")
    os.makedirs(out_dir, exist_ok=True)
    try:
        conf, meta = load_run_config(paths)
    except ConfigError as e:
        log(f"CONFIG ERROR: {e}")
        sys.exit(2)
    chash = config_hash(conf)
    cfg = ModelConfig(**conf["model_cfg"])
    device = setup_device(args.device)
    vd = ValData(paths, meta, args.max_val_tokens)
    log(f"validation: {vd.n_seq} sequences = {vd.N:,} target tokens; device {gpu_name(device)}")
    status = run_status(paths, chash)
    ref = dict(
        mi_cluster_source_bits=mutual_info_bits(vd.cluster_tok, vd.in_src, N_CLUSTERS, N_SOURCES),
        n_tokens=vd.N, first64_fraction=float(vd.first64.mean()), status=status, time=time.time(),
        config_hash=chash,
    )
    tags = args.tags or list(RUN_INFO)
    for tag in tags:
        info = RUN_INFO[tag]
        if status[tag]["state"] != "done":
            log(f"{tag}: {status[tag]['state']} -> not evaluated")
            continue
        out = os.path.join(out_dir, f"{tag}.json")
        side = read_json(paths.ckpt_side(tag))
        old = read_json(out)
        if (old and not args.force and old.get("config_hash") == chash and old.get("step") == side["step"]
                and abs((old.get("ckpt_saved") or 0) - (side.get("saved") or 0)) < 5 and old.get("n_tokens_eval") == vd.N):
            log(f"{tag}: cached result is current")
            continue
        log(f"{tag}: evaluating ({info['mode']})")
        res = evaluate_run(tag, info["mode"], paths.ckpt_last(tag), cfg, vd, device, args.batch)
        res.update(seed=info["seed"], n_tokens_eval=vd.N)
        write_json(out, res)
        hv = res["variants"][res["headline"]]["val_loss"]
        log(f"{tag}: val loss {hv:.4f} ({res['headline']}) in {res['eval_seconds']:.0f}s")
    ref_path = os.path.join(paths.ckpt, "R1_r4ref.pt")
    if os.path.exists(ref_path):
        out = os.path.join(out_dir, "R1_r4ref.json")
        old = read_json(out)
        if args.force or not old or old.get("config_hash") != chash or old.get("n_tokens_eval") != vd.N:
            ck = torch.load(ref_path, map_location="cpu", weights_only=False)
            if ck.get("config_hash") == chash:
                model = GPT(cfg, "r1")
                model.load_state_dict(ck["model"])
                model = model.to(device).eval()
                losses, _, _ = run_variant(model, vd, "r1", device, args.batch, keep_ids=False)
                write_json(out, dict(tag="R1_r4ref", step=ck["step"], tokens=ck.get("tokens"), config_hash=chash,
                                     n_tokens_eval=vd.N, variants=dict(normal=loss_metrics(losses, vd))))
                log(f"R1_r4ref: val loss {losses.mean():.4f}")
    write_json(os.path.join(out_dir, "reference.json"), ref)
    log("eval done")


if __name__ == "__main__":
    main()
