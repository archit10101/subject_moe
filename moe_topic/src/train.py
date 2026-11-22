"""Training (one run per invocation) and the throughput bench.

  python src/train.py --tag R1 --mode r1 --init_seed 0          # train / auto-resume
  python src/train.py --bench [--tiny ...]                         # measure tok/s, write results/config.json

Exit codes: 0 finished (or already done) | 3 checkpointed at --max_minutes, needs a continuation job
            2 configuration / staleness error (do not retry) | 4 diverged (non-finite loss)
            143 stopped by SIGTERM (checkpoint saved)
Everything that defines a run (model size, batch, hyperparameters, token count) comes from
results/config.json, so all runs are identical except for --mode and --init_seed.
"""
import argparse
import hashlib
import json
import math
import os
import platform
import signal
import socket
import sys
import time

T_START = time.time()

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (GPU_MISSING_MSG, MILESTONES, MODE_TAG, RUN_INFO, SEQ_LEN, Paths, Split, gpu_expected,  # noqa: E402
                    load_meta, log, read_json, write_json)
from data import micro_seq_ids, seq_batch, to_device, topic_targets, train_order  # noqa: E402
from model import GPT, MODES, ModelConfig, base_config, lr_at, param_counts, pool_specs, tiny_config  # noqa: E402

EXIT_DONE, EXIT_CONFIG, EXIT_CONTINUE, EXIT_DIVERGED, EXIT_SIGTERM = 0, 2, 3, 4, 143
BENCH_MODES = ("r1", "r1f", "p2", "p4", "p3")
_STOP = {"flag": False}


def _on_sigterm(signum, frame):
    _STOP["flag"] = True
    log(f"received signal {signum}: will checkpoint after the current step and exit")


class ConfigError(Exception):
    pass


class NonFinite(Exception):
    pass


# ----------------------------------------------------------------------------- setup
def setup_device(pref="auto"):
    if pref == "auto" and not torch.cuda.is_available() and gpu_expected():
        raise RuntimeError(GPU_MISSING_MSG)
    if pref == "cpu" or (pref == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    import torch._dynamo as dynamo

    dynamo.config.suppress_errors = True  # broken toolchain -> eager, not a crash
    return torch.device("cuda")


def gpu_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(0)
    return f"cpu ({platform.processor() or platform.machine()})"


def config_hash(conf):
    keep = {k: conf[k] for k in ("model_cfg", "hparams", "total_steps", "data_fingerprint")}
    return hashlib.sha256(json.dumps(keep, sort_keys=True).encode()).hexdigest()[:16]


def load_run_config(paths):
    conf = read_json(paths.config)
    if conf is None:
        raise ConfigError(f"{paths.config} missing: the bench stage has not run")
    meta = load_meta(paths)
    if conf.get("data_fingerprint") != meta["fingerprint"]:
        raise ConfigError(f"config.json was made from different data ({conf.get('data_fingerprint')} != "
                          f"{meta['fingerprint']}); refusing to train on stale settings")
    return conf, meta


def build_model(cfg, mode, seed, device):
    return GPT(cfg, mode).init_weights(seed).to(device)


def make_optimizer(model, hp, device):
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [dict(params=decay, weight_decay=hp["weight_decay"]), dict(params=no_decay, weight_decay=0.0)]
    kw = dict(lr=hp["lr"], betas=tuple(hp["betas"]), eps=hp["eps"])
    if device.type == "cuda":
        kw["fused"] = True
    return torch.optim.AdamW(groups, **kw)


def autocast(device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return torch.autocast(device_type="cpu", enabled=False)


@torch.no_grad()
def quick_val(model, val, n_seqs, mb, device):
    """Mean loss and next-token accuracy on the first n_seqs natural-val windows."""
    model.eval()
    tot = torch.zeros((), device=device, dtype=torch.float64)
    cor = torch.zeros((), device=device, dtype=torch.float64)
    cnt = 0
    for s in range(0, n_seqs, mb):
        ids = np.arange(s, min(s + mb, n_seqs))
        x, y = seq_batch(val.tokens, ids)
        x, y = to_device(x, device), to_device(y, device)
        with autocast(device):
            out = model(x, y, reduction="none", return_logits=True)
        tot += out.loss.double().sum()
        cor += (out.logits.argmax(-1) == y).double().sum()
        cnt += y.numel()
    model.train()
    return float(tot.item() / max(cnt, 1)), float(cor.item() / max(cnt, 1))


# ----------------------------------------------------------------------------- checkpoints
def save_ckpt(paths, tag, model, opt, step, total_steps, done, conf_hash, extra):
    state = dict(model=model.state_dict(), optim=opt.state_dict(), step=step, total_steps=total_steps,
                 done=done, tag=tag, config_hash=conf_hash, saved=time.time(), **extra)
    path = paths.ckpt_last(tag)
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)
    write_json(paths.ckpt_side(tag), dict(tag=tag, step=step, total_steps=total_steps, done=done,
                                          config_hash=conf_hash, saved=time.time(), **extra))


def truncate_log(path, step):
    """On resume, drop train/val records beyond the checkpointed step (they will be redone)."""
    if not os.path.exists(path):
        return
    keep = []
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("type") in ("train", "val") and r.get("step", 0) > step:
                continue
            keep.append(line if line.endswith("\n") else line + "\n")
    with open(path + ".tmp", "w") as f:
        f.writelines(keep)
    os.replace(path + ".tmp", path)


class JsonlLog:
    def __init__(self, path):
        self.f = open(path, "a")

    def write(self, **rec):
        rec.setdefault("time", time.time())
        self.f.write(json.dumps(rec) + "\n")
        self.f.flush()


# ----------------------------------------------------------------------------- one optimizer step
def run_step(model, opt, step, total_steps, hp, train, perm, device, acc):
    lr = lr_at(step, total_steps, hp["lr"], hp["min_lr"], hp["warmup_frac"])
    for g in opt.param_groups:
        g["lr"] = lr
    mb, accum = hp["micro_bs"], hp["grad_accum"]
    p3 = model.mode == "p3"
    for micro in range(accum):
        ids = micro_seq_ids(perm, step, micro, accum, mb)
        x, y = seq_batch(train.tokens, ids)
        x, y = to_device(x, device), to_device(y, device)
        tt = to_device(topic_targets(train, ids), device) if p3 else None
        with autocast(device):
            out = model(x, y, topic_targets=tt)
        model.backward(out, 1.0 / accum)
        if acc is not None:
            acc.add(out)
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), hp["grad_clip"])
    opt.step()
    opt.zero_grad(set_to_none=True)
    if acc is not None:
        acc.gnorm += gn.detach().float()
    return lr


class Acc:
    """Device-side accumulators for the logging window (no host syncs until flush)."""

    def __init__(self, model, device):
        self.keys = [f"L{li}.{n}" for li, n, _ in model.pools()]
        self.device = device
        self.reset()

    def reset(self):
        z = lambda *s: torch.zeros(*s, device=self.device)  # noqa: E731
        self.lm, self.aux, self.topic, self.tacc, self.gnorm = z(()), z(()), z(()), z(()), z(())
        self.maxload = z(len(self.keys))
        self.n_micro = 0
        self.steps = 0
        self.t0 = time.time()

    def add(self, out):
        self.lm += out.loss.detach().float()
        self.aux += out.aux.detach().float()
        if out.topic_loss is not None:
            self.topic += out.topic_loss.detach().float()
            self.tacc += out.topic_acc.detach().float()
        ml = [c.max().float() / c.sum().clamp(min=1) for st in out.stats for c in st["counts"].values()]
        self.maxload += torch.stack(ml)
        self.n_micro += 1

    def flush(self, mode):
        n = max(self.n_micro, 1)
        rec = dict(loss=float(self.lm / n), aux=float(self.aux / n),
                   max_load={k: round(v, 4) for k, v in zip(self.keys, (self.maxload / n).tolist())},
                   grad_norm=float(self.gnorm / max(self.steps, 1)))
        if mode == "p3":
            rec["topic_loss"] = float(self.topic / n)
            rec["topic_acc"] = float(self.tacc / n)
        return rec


# ----------------------------------------------------------------------------- training
def milestone_steps(total_steps):
    return {pct: max(1, int(round(total_steps * pct / 100))) for pct in MILESTONES}


def train(args):
    paths = Paths(args.work).mk()
    info = RUN_INFO.get(args.tag)
    if info and (info["mode"] != args.mode or info["seed"] != args.init_seed):
        raise ConfigError(f"tag {args.tag} must be mode={info['mode']} seed={info['seed']}")
    conf, meta = load_run_config(paths)
    chash = config_hash(conf)
    cfg = ModelConfig(**conf["model_cfg"])
    hp = conf["hparams"]
    total_steps = conf["total_steps"]
    tps_step = hp["tokens_per_step"]
    side = read_json(paths.ckpt_side(args.tag))
    if side and side.get("done") and side.get("config_hash") == chash and os.path.exists(paths.ckpt_last(args.tag)):
        log(f"{args.tag} is already finished ({side['step']}/{side['total_steps']} steps); nothing to do")
        return EXIT_DONE

    device = setup_device(args.device)
    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGUSR1, _on_sigterm)
    torch.manual_seed(0)
    model = build_model(cfg, args.mode, args.init_seed, device)
    opt = make_optimizer(model, hp, device)
    start_step, prev_minutes = 0, 0.0
    ck_path = paths.ckpt_last(args.tag)
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location=device, weights_only=False)
        if ck.get("config_hash") != chash:
            raise ConfigError(f"{ck_path} was trained with a different config.json; move it away to restart")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optim"])
        start_step = int(ck["step"])
        prev_minutes = float(ck.get("train_minutes", 0.0))
        if ck.get("done"):
            log(f"{args.tag}: checkpoint says done")
            return EXIT_DONE
        del ck
        log(f"{args.tag}: resumed from step {start_step}/{total_steps}")
    truncate_log(paths.log(args.tag), start_step)
    if device.type == "cuda" and not args.no_compile:
        model.compile_attention()

    log(f"loading training tokens into RAM ({meta['train']['n_tokens'] / 1e9:.3f}B)")
    tr = Split(paths, "train", meta, in_ram=True)
    perm = train_order(tr.n_seq, hp["data_seed"])
    val = Split(paths, "val", meta)
    val_n = min(val.n_seq, max(1, hp["quick_val_tokens"] // SEQ_LEN))
    val_steps = sorted({max(1, int(round(total_steps * k / 10))) for k in range(1, 11)})
    miles = milestone_steps(total_steps)

    jl = JsonlLog(paths.log(args.tag))
    jl.write(type="start", step=start_step, total_steps=total_steps, tag=args.tag, mode=args.mode,
             init_seed=args.init_seed, gpu=gpu_name(device), host=socket.gethostname(),
             job_id=os.environ.get("SLURM_JOB_ID"), tokens_per_step=tps_step)
    model.train()
    acc = Acc(model, device)
    last_ckpt = time.time()
    t_loop = time.time()
    run_minutes = lambda: prev_minutes + (time.time() - t_loop) / 60  # noqa: E731
    extra = lambda: dict(mode=args.mode, init_seed=args.init_seed, tokens_per_step=tps_step,  # noqa: E731
                         train_minutes=run_minutes())
    step = start_step
    while step < total_steps:
        lr = run_step(model, opt, step, total_steps, hp, tr, perm, device, acc)
        acc.steps += 1
        step += 1
        if step % args.log_every == 0 or step == total_steps:
            rec = acc.flush(args.mode)  # host sync
            if not (math.isfinite(rec["loss"]) and math.isfinite(rec["grad_norm"])):
                jl.write(type="error", step=step, reason="non-finite loss/grad norm", loss=str(rec["loss"]))
                raise NonFinite(f"{args.tag}: non-finite loss ({rec['loss']}) or grad norm ({rec['grad_norm']}) "
                                f"at step {step}; last good checkpoint is kept")
            dt = time.time() - acc.t0
            tok_s = acc.steps * tps_step / max(dt, 1e-9)
            eta_min = (total_steps - step) * (dt / max(acc.steps, 1)) / 60
            jl.write(type="train", step=step, tokens=step * tps_step, lr=lr, tok_s=tok_s,
                     elapsed_min=run_minutes(), eta_min=eta_min, **rec)
            msg = f"{args.tag} step {step}/{total_steps} loss {rec['loss']:.4f} aux {rec['aux']:.3f} " \
                  f"tok/s {tok_s:,.0f} ETA {eta_min:.1f}m"
            if "topic_loss" in rec:
                msg += f" topic {rec['topic_loss']:.3f} acc {rec['topic_acc']:.3f}"
            log(msg)
            acc.reset()
        if step in val_steps:
            v, a = quick_val(model, val, val_n, hp["micro_bs"], device)
            jl.write(type="val", step=step, tokens=step * tps_step, val_loss=v, val_acc=a)
            log(f"{args.tag} val @ step {step}: loss {v:.4f} acc {a:.4f}")
            acc.t0 = time.time()  # keep tok/s honest
        for pct, ms in miles.items():
            if step == ms and ms < total_steps:
                p = paths.ckpt_milestone(args.tag, pct)
                torch.save(dict(model=model.state_dict(), step=step, pct=pct, tag=args.tag, mode=args.mode,
                                init_seed=args.init_seed, config_hash=chash, tokens=step * tps_step), p + ".tmp")
                os.replace(p + ".tmp", p)
                log(f"saved {pct}% checkpoint at step {step}")
        if step == total_steps:
            break
        elapsed = (time.time() - T_START) / 60
        need = _STOP["flag"] or elapsed >= args.max_minutes or time.time() - last_ckpt >= args.ckpt_minutes * 60
        if need and not (torch.isfinite(acc.lm).item() and torch.isfinite(acc.gnorm).item()):
            jl.write(type="error", step=step, reason="non-finite loss/grad norm before checkpoint")
            raise NonFinite(f"{args.tag}: non-finite loss/grad norm at step {step}; last good checkpoint is kept")
        if _STOP["flag"] or elapsed >= args.max_minutes:
            save_ckpt(paths, args.tag, model, opt, step, total_steps, False, chash, extra())
            why = "SIGTERM" if _STOP["flag"] else f"--max_minutes {args.max_minutes}"
            jl.write(type="pause", step=step, reason=why)
            log(f"{args.tag}: checkpointed at step {step}/{total_steps} ({why})")
            return EXIT_SIGTERM if _STOP["flag"] else EXIT_CONTINUE
        if time.time() - last_ckpt >= args.ckpt_minutes * 60:
            save_ckpt(paths, args.tag, model, opt, step, total_steps, False, chash, extra())
            jl.write(type="ckpt", step=step)
            last_ckpt = time.time()
    save_ckpt(paths, args.tag, model, opt, step, total_steps, True, chash, extra())
    jl.write(type="end", step=step, elapsed_min=run_minutes())
    log(f"{args.tag}: finished {step} steps ({step * tps_step / 1e6:.1f}M tokens)")
    return EXIT_DONE


# ----------------------------------------------------------------------------- bench
def run_started(paths):
    for tag in RUN_INFO:
        for p in (paths.ckpt_last(tag), paths.ckpt_side(tag), paths.log(tag)):
            if os.path.exists(p):
                return True
    return False


def bench(args):
    paths = Paths(args.work).mk()
    meta = load_meta(paths)
    old = read_json(paths.config)
    if old is not None and run_started(paths):
        if old.get("data_fingerprint") == meta["fingerprint"]:
            log("results/config.json already exists and training has started: keeping it unchanged")
            return EXIT_DONE
        raise ConfigError("config.json is stale (different data) but checkpoints/logs exist; refusing")

    device = setup_device(args.device)
    cfg = tiny_config() if args.tiny else base_config()
    hp = dict(lr=1e-3, min_lr=1e-4, warmup_frac=0.02, betas=[0.9, 0.95], eps=1e-8, weight_decay=0.1,
              grad_clip=1.0, micro_bs=args.micro_bs, grad_accum=args.grad_accum, seq_len=SEQ_LEN,
              tokens_per_step=args.micro_bs * args.grad_accum * SEQ_LEN, data_seed=0,
              lb_coef=cfg.lb_coef, topic_coef=cfg.topic_coef, quick_val_tokens=int(args.quick_val_tokens),
              precision="bf16 autocast + TF32" if device.type == "cuda" else "fp32 (cpu)",
              lr_schedule="linear warmup 2% of steps, cosine to min_lr")
    tps_step = hp["tokens_per_step"]
    t0 = time.time()
    tr = Split(paths, "train", meta, in_ram=True)
    perm = train_order(tr.n_seq, 0)
    res, pcs = {}, {}
    for mode in MODES:
        with torch.device("meta"):
            pcs[mode] = param_counts(GPT(cfg, mode))
    modes = [m for m in BENCH_MODES if m in args.bench_modes.split(",")]
    for mode in modes:
        model = build_model(cfg, mode, 0, device)
        opt = make_optimizer(model, hp, device)
        if device.type == "cuda":
            if not args.no_compile:
                model.compile_attention()
            torch.cuda.reset_peak_memory_stats()
        times = []
        for s in range(args.bench_steps):
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts = time.time()
            run_step(model, opt, s, 1000, hp, tr, perm, device, None)
            if device.type == "cuda":
                torch.cuda.synchronize()
            times.append(time.time() - ts)
            log(f"bench {mode} step {s}: {times[-1]:.3f}s")
        meas = times[args.bench_warmup:] or times
        tps = tps_step * len(meas) / sum(meas)
        mem = torch.cuda.max_memory_allocated() / 2 ** 30 if device.type == "cuda" else None
        res[mode] = dict(tps=tps, step_seconds=sum(meas) / len(meas), peak_mem_gib=mem)
        log(f"bench {mode}: {tps:,.0f} tokens/s" + (f", peak mem {mem:.1f} GiB" if mem else ""))
        del model, opt
        if device.type == "cuda":
            torch.cuda.empty_cache()
    tps_min = min(r["tps"] for r in res.values())
    if args.force_steps:
        total_steps = args.force_steps
    elif args.minutes_per_run:
        total_steps = max(10, int(tps_min * args.minutes_per_run * 60 // tps_step))
    else:
        total_steps = int(args.total_tokens // tps_step)
    total_tokens = total_steps * tps_step
    est = {}
    for mode in MODES:
        b = res.get(mode) or res.get({"p1": "p2", "p3": "p2"}.get(mode, "r1")) or res[modes[0]]
        est[MODE_TAG[mode]] = total_tokens / b["tps"] / 60
    epochs = total_tokens / max(tr.n_seq * SEQ_LEN, 1)
    conf = dict(
        version=1, created=time.strftime("%Y-%m-%d %H:%M:%S"), model="tiny" if args.tiny else "base",
        model_cfg=cfg.to_dict(), pools={m: pool_specs(cfg, m) for m in MODES}, hparams=hp,
        total_steps=total_steps, total_tokens=total_tokens, requested_total_tokens=args.total_tokens,
        tps=tps_min, bench=res, est_minutes_per_run=est, train_tokens_available=tr.n_seq * SEQ_LEN,
        data_epochs=epochs, gpu=gpu_name(device), host=socket.gethostname(), torch=torch.__version__,
        cuda=torch.version.cuda, param_counts=pcs, data_fingerprint=meta["fingerprint"],
        synthetic=meta.get("synthetic", False), bench_seconds=time.time() - t0, forced_steps=bool(args.force_steps),
        minutes_per_run=args.minutes_per_run or None,
        bench_note="P3 is benchmarked in addition to R1/R1f/P2/P4 because its topic loss needs a second "
                   "backward pass (so routers get no topic-loss gradient); P1 uses P2's speed as estimate.")
    conf["config_hash"] = config_hash(conf)
    write_json(paths.config, conf)
    log(f"wrote {paths.config}: {total_steps} steps = {total_tokens / 1e6:.1f}M tokens per run; "
        + ", ".join(f"{k} ~{v:.0f} min" for k, v in est.items())
        + (f"; WARNING: {epochs:.2f} passes over the training data" if epochs > 1 else ""))
    print("BENCH_RESULT gpu='%s' " % conf["gpu"] + " ".join(f"{m}={r['tps']:.0f}tok/s" for m, r in res.items()))
    return EXIT_DONE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default=None)
    ap.add_argument("--tag")
    ap.add_argument("--mode", choices=list(MODES))
    ap.add_argument("--init_seed", type=int, default=0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--max_minutes", type=float, default=330)
    ap.add_argument("--ckpt_minutes", type=float, default=15)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--no_compile", action="store_true")
    # bench
    ap.add_argument("--bench", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="bench: tiny model config (CPU-friendly tests)")
    ap.add_argument("--micro_bs", type=int, default=32)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--bench_steps", type=int, default=25)
    ap.add_argument("--bench_warmup", type=int, default=5)
    ap.add_argument("--bench_modes", default=",".join(BENCH_MODES))
    ap.add_argument("--total_tokens", type=float, default=1.2e9, help="tokens per run (floored to whole steps)")
    ap.add_argument("--minutes_per_run", type=float, default=0, help="smoke test: size runs by measured speed")
    ap.add_argument("--quick_val_tokens", type=float, default=2e6, help="tokens for the in-training val loss")
    ap.add_argument("--force_steps", type=int, default=0, help="testing only: fix total_steps")
    args = ap.parse_args()
    try:
        if args.bench:
            rc = bench(args)
        else:
            if not args.tag or not args.mode:
                ap.error("--tag and --mode are required (or use --bench)")
            rc = train(args)
    except ConfigError as e:
        log(f"CONFIG ERROR: {e}")
        rc = EXIT_CONFIG
    except NonFinite as e:
        log(f"DIVERGED: {e}")
        rc = EXIT_DIVERGED
    sys.exit(rc)


if __name__ == "__main__":
    main()
