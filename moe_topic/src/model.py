"""GPT-style decoder whose every FFN is a shared expert + routed MoE pool(s) (dropless, routed weight 1.0).

Modes (all identical in active routed hidden units per token (1024) and total routed units per layer (8192)):
  r1   1 pool:  8 x d_ff 1024, top-1, router W[8, d]  . h_t
  r1f  1 pool: 16 x d_ff  512, top-2, router W[16, d] . h_t
  p1   1 pool:  8 x d_ff 1024, top-1, router W[8, d]  . s_t
  p2   1 pool:  8 x d_ff 1024, top-1, router W[8, 2d] . [h_t ; s_t]
  p3   as p2, plus a topic head Linear(d -> 24) on s_t per layer, trained with CE against the document's
       soft topic distribution (weight 0.05). The head only feeds that loss; labels never enter the forward.
  p4   2 pools, 8 x d_ff 512 each, top-1 each: "topic" pool routed by W[8, d] . s_t, "token" pool by
       W[8, d] . h_t

s_t at layer l = mean of that layer's h over positions max(t-49, d_t) .. t, d_t = first position of the
current document inside the sequence (position after the latest EOT strictly before t; 0 if none).

Every chosen expert's output is multiplied by w = p_e / p_e.detach() (exactly 1.0 forward), so routers
get LM-loss gradients. Switch load-balance loss E * sum_e f_e P_e per router, averaged over routers + layers.
"""
import hashlib
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("r1", "r1f", "p1", "p2", "p3", "p4")
EOT_ID = 50256


@dataclass
class ModelConfig:
    n_layer: int = 8
    d_model: int = 512
    n_head: int = 8
    d_shared: int = 1024
    d_ff_full: int = 1024   # one full-width routed expert
    d_ff_half: int = 512    # half-width routed expert (r1f, p4)
    vocab: int = 50304
    ctx: int = 1024
    lb_coef: float = 0.01
    topic_coef: float = 0.05
    n_topics: int = 24
    window: int = 50
    eot: int = EOT_ID

    def to_dict(self):
        return asdict(self)


def base_config():
    return ModelConfig()


def tiny_config():
    # d_model 512 -> 64 (/8): every width scales by the same factor
    return ModelConfig(n_layer=2, d_model=64, n_head=4, d_shared=128, d_ff_full=128, d_ff_half=64)


def pool_specs(cfg, mode):
    """[(name, n_experts, d_ff, top_k, router input)] for a mode. Router input: h | s | hs."""
    F_, H_ = cfg.d_ff_full, cfg.d_ff_half
    return {
        "r1": [("main", 8, F_, 1, "h")],
        "r1f": [("main", 16, H_, 2, "h")],
        "p1": [("main", 8, F_, 1, "s")],
        "p2": [("main", 8, F_, 1, "hs")],
        "p3": [("main", 8, F_, 1, "hs")],
        "p4": [("topic", 8, H_, 1, "s"), ("token", 8, H_, 1, "h")],
    }[mode]


def uses_summary(mode):
    return mode in ("p1", "p2", "p3", "p4")


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        hd = C // self.n_head
        q = q.view(B, T, self.n_head, hd).transpose(1, 2)
        k = k.view(B, T, self.n_head, hd).transpose(1, 2)
        v = v.view(B, T, self.n_head, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


# ----------------------------------------------------------------------------- context summary
def summary_start(idx, eot, window):
    """[B, T] long: first position of the summary window of each t, max(t - window + 1, d_t), where d_t is
    the position after the latest EOT strictly before t (0 if none). Uses only idx[:, :t] -> causal."""
    B, T = idx.shape
    pos = torch.arange(T, device=idx.device)
    marks = torch.where(idx == eot, pos + 1, torch.zeros_like(pos))          # an EOT at p starts a doc at p+1
    prev = torch.cat([marks.new_zeros(B, 1), marks[:, :-1]], dim=1)          # strictly before t
    d = torch.cummax(prev, dim=1).values
    return torch.maximum(d, (pos - (window - 1)).clamp(min=0)[None, :])


def context_summary(h, start):
    """s[b, t] = mean(h[b, start[b, t] .. t]) via an fp32 cumulative sum."""
    B, T, D = h.shape
    hf = h.float()
    c = torch.cat([hf.new_zeros(B, 1, D), torch.cumsum(hf, dim=1)], dim=1)  # c[k] = sum h[:k]
    end = c[:, 1:]                                                           # sum h[:t+1]
    beg = c.gather(1, start[:, :, None].expand(B, T, D))
    cnt = (torch.arange(T, device=h.device)[None, :] + 1 - start).to(hf.dtype)
    return (end - beg) / cnt[:, :, None]


# ----------------------------------------------------------------------------- experts
class ExpertPool(nn.Module):
    def __init__(self, d, n_exp, d_ff, top_k, inp):
        super().__init__()
        self.n_exp, self.d_ff, self.top_k, self.inp = n_exp, d_ff, top_k, inp
        self.w1 = nn.Parameter(torch.empty(n_exp, d, d_ff))
        self.w2 = nn.Parameter(torch.empty(n_exp, d_ff, d))
        self.router = nn.Parameter(torch.empty(n_exp, 2 * d if inp == "hs" else d))
        self.disabled = None     # eval ablation: expert id whose tokens are rerouted to their next-best expert
        self.keep_debug = False  # tests: keep the routed weights of the last forward
        self.debug_w = None

    def route(self, rin):
        with torch.autocast(device_type=rin.device.type, enabled=False):
            logits = F.linear(rin.float(), self.router.float())
        if self.disabled is not None:
            logits = logits.clone()
            logits[:, self.disabled] = float("-inf")
        p = torch.softmax(logits, dim=-1)
        top_p, top_i = torch.topk(p, self.top_k, dim=-1)  # [N, k], best first
        return p, top_p, top_i

    def dispatch(self, x, top_i):
        """Dropless: argsort the (token, slot) pairs by expert, run each expert on its contiguous slice,
        un-sort with index_copy. Returns [N, k, D] expert outputs and per-expert pair counts."""
        N, D = x.shape
        k = self.top_k
        flat_e = top_i.reshape(-1)
        order = torch.argsort(flat_e, stable=True)
        counts = torch.bincount(flat_e, minlength=self.n_exp)
        tok = order // k
        xs = x[tok]
        outs = []
        start = 0
        for i, c in enumerate(counts.tolist()):
            if c == 0:
                continue
            seg = xs[start:start + c]
            outs.append(F.gelu(seg @ self.w1[i]) @ self.w2[i])
            start += c
        if not outs:
            return x.new_zeros(N, k, D), counts
        ys = torch.cat(outs, dim=0)
        y = torch.zeros_like(ys).index_copy(0, order, ys)
        return y.view(N, k, D), counts

    def forward(self, x, rin):
        p, top_p, top_i = self.route(rin)
        w = top_p / top_p.detach()  # forward value exactly 1.0; gradient d log p_e
        if self.keep_debug:
            self.debug_w = w.detach()
        y, counts = self.dispatch(x, top_i)
        y = (y * w[:, :, None].to(y.dtype)).sum(1)
        frac = counts.float() / top_i.numel()
        aux = self.n_exp * (frac * p.mean(0)).sum()
        return y, aux, counts, top_i


class MoE(nn.Module):
    def __init__(self, cfg, mode):
        super().__init__()
        self.cfg, self.mode = cfg, mode
        d = cfg.d_model
        self.shared_w1 = nn.Parameter(torch.empty(d, cfg.d_shared))
        self.shared_w2 = nn.Parameter(torch.empty(cfg.d_shared, d))
        self.pools = nn.ModuleDict({n: ExpertPool(d, E, f, k, inp) for n, E, f, k, inp in pool_specs(cfg, mode)})
        self.topic_head = nn.Linear(d, cfg.n_topics) if mode == "p3" else None

    def forward(self, h, start, rc):
        B, T, D = h.shape
        hf = h.reshape(B * T, D)
        s = None
        if uses_summary(self.mode):
            fn = rc.get("summary_fn") or context_summary
            s = fn(h, start)  # [B, T, D] fp32
        out = F.gelu(hf @ self.shared_w1) @ self.shared_w2
        auxs, st = [], {"counts": {}, "ids": {}}
        for name, pool in self.pools.items():
            if pool.inp == "h":
                rin = hf
            elif pool.inp == "s":
                rin = s.reshape(B * T, D)
            else:
                rin = torch.cat([hf.float(), s.reshape(B * T, D)], dim=-1)
            y, aux, counts, top_i = pool(hf, rin)
            out = out + y
            auxs.append(aux)
            st["counts"][name] = counts
            if rc.get("keep_ids"):
                st["ids"][name] = top_i.detach()
        topic = None
        if self.topic_head is not None:
            with torch.autocast(device_type=h.device.type, enabled=False):
                tl = self.topic_head(s.reshape(B * T, D).float())
            tt = rc.get("topic_targets")
            if tt is not None:
                tt = tt.reshape(B * T, -1).float()
                topic = dict(loss=-(tt * F.log_softmax(tl, dim=-1)).sum(-1).mean(),
                             acc=(tl.argmax(-1) == tt.argmax(-1)).float().mean())
        return out.reshape(B, T, D), auxs, topic, st


class Block(nn.Module):
    def __init__(self, cfg, mode):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.moe = MoE(cfg, mode)

    def forward(self, x, start, rc):
        x = x + self.attn(self.ln1(x))
        m, auxs, topic, st = self.moe(self.ln2(x), start, rc)
        return x + m, auxs, topic, st


class Out:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def param_seed(name, init_seed):
    h = hashlib.sha256(f"{name}|{int(init_seed)}".encode()).hexdigest()
    return int(h[:15], 16)


class GPT(nn.Module):
    def __init__(self, cfg, mode):
        super().__init__()
        assert mode in MODES, mode
        self.cfg, self.mode = cfg, mode
        self.wte = nn.Embedding(cfg.vocab, cfg.d_model)
        self.wpe = nn.Embedding(cfg.ctx, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, mode) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.d_model)
        self.summary_fn = None  # tests only: replace the context summary (e.g. a deliberately leaky one)

    @torch.no_grad()
    def init_weights(self, seed):
        """normal(0, 0.02); attention / expert output projections scaled by 1/sqrt(2 n_layer); norms 1,
        biases 0. Every parameter is drawn on CPU from its own generator seeded by (name, init_seed), so a
        parameter with the same name and shape is identical across modes, independent of construction order."""
        scaled = (".attn.proj.weight", ".moe.shared_w2", ".w2")
        for name, p in self.named_parameters():
            if name.endswith(("ln1.weight", "ln2.weight")) or name == "ln_f.weight":
                p.fill_(1.0)
                continue
            if name.endswith(".bias"):
                p.zero_()
                continue
            std = 0.02
            if name.endswith(scaled):
                std /= math.sqrt(2 * self.cfg.n_layer)
            g = torch.Generator(device="cpu").manual_seed(param_seed(name, seed))
            p.copy_((torch.randn(p.shape, generator=g) * std).to(p.device, p.dtype))
        return self

    def compile_attention(self):
        for b in self.blocks:
            b.attn.compile()  # in place: state_dict keys unchanged

    def pools(self):
        """[(layer, pool name, ExpertPool)]"""
        return [(li, n, p) for li, b in enumerate(self.blocks) for n, p in b.moe.pools.items()]

    def forward(self, idx, targets=None, topic_targets=None, reduction="mean", keep_ids=False,
                return_logits=False):
        B, T = idx.shape
        rc = dict(topic_targets=topic_targets, keep_ids=keep_ids, summary_fn=self.summary_fn)
        start = summary_start(idx, self.cfg.eot, self.cfg.window) if uses_summary(self.mode) else None
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None]
        auxs, topics, stats = [], [], []
        for blk in self.blocks:
            x, a, topic, st = blk(x, start, rc)
            auxs.extend(a)
            if topic is not None:
                topics.append(topic)
            stats.append(st)
        x = self.ln_f(x)
        logits = F.linear(x, self.wte.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1),
                                   reduction=reduction)
        return Out(logits=logits if return_logits else None, loss=loss, aux=torch.stack(auxs).mean(),
                   topic_loss=torch.stack([t["loss"] for t in topics]).mean() if topics else None,
                   topic_acc=torch.stack([t["acc"] for t in topics]).mean() if topics else None,
                   stats=stats)

    def total_loss(self, out):
        loss = out.loss + self.cfg.lb_coef * out.aux
        if out.topic_loss is not None:
            loss = loss + self.cfg.topic_coef * out.topic_loss
        return loss

    def router_params(self):
        return [p.router for _, _, p in self.pools()]

    def backward(self, out, scale=1.0):
        """Accumulate gradients of total_loss(out) * scale. P3's topic loss is back-propagated into every
        parameter EXCEPT the routers (it reaches earlier routers through the residual stream otherwise),
        so routers are trained only by the LM and load-balance losses. Costs one extra backward for P3."""
        main = (out.loss + self.cfg.lb_coef * out.aux) * scale
        if out.topic_loss is not None:
            rid = {id(p) for p in self.router_params()}
            others = [p for p in self.parameters() if id(p) not in rid and p.requires_grad]
            (self.cfg.topic_coef * out.topic_loss * scale).backward(inputs=others, retain_graph=True)
        main.backward()


def routed_units(cfg, mode):
    """(active routed hidden units per token, total routed hidden units per layer)."""
    sp = pool_specs(cfg, mode)
    return sum(k * f for _, E, f, k, _ in sp), sum(E * f for _, E, f, k, _ in sp)


def param_counts(model):
    cfg = model.cfg
    total = sum(p.numel() for p in model.parameters())
    routed_total = sum(p.w1.numel() + p.w2.numel() for _, _, p in model.pools())
    routed_active = sum(p.top_k * 2 * cfg.d_model * p.d_ff for _, _, p in model.pools())
    emb = model.wte.weight.numel() + model.wpe.weight.numel()
    routers = sum(p.router.numel() for _, _, p in model.pools())
    return dict(total=total, active_per_token=total - routed_total + routed_active, embedding=emb,
                non_embedding=total - emb, routed_total=routed_total, routers=routers)


def lr_at(step, total_steps, peak=1e-3, min_lr=1e-4, warmup_frac=0.02):
    warm = max(1, int(round(warmup_frac * total_steps)))
    if step < warm:
        return peak * (step + 1) / warm
    prog = min(1.0, (step - warm) / max(1, total_steps - warm))
    return min_lr + 0.5 * (peak - min_lr) * (1.0 + math.cos(math.pi * prog))
