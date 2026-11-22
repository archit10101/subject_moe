"""GPT-style decoder whose every FFN is an MoE (1 shared + 8 routed experts, top-1, dropless).

Routing modes
  r1  baseline learned router:  p = softmax(W_r h), e = argmax p, w = p_e / p_e.detach()  (+ switch LB loss)
  r2  past teacher:   teacher id = cluster[g-1] (g = j // 64; cluster[0] when g == 0)
  r3  future teacher: teacher id = cluster[g]
      r2/r3: student logits s = W_s h (fp32). One coin per token (shared across layers):
      route to the teacher id if u < alpha else argmax(s). w == 1. The LM loss has no path into
      the student; it learns only from the distillation loss CE(s, teacher) (which does reach the backbone).
  r4  naive leaky router (demo): router input at t = mean(h[t+2 .. min(t+51, T-1)]); zero_router=True
      feeds zeros instead (the generation-time view).

Every mode creates the same parameters in the same order (the r1/r4 router and the r2/r3 student are
both `moe.router` of shape [8, d]), so the same init seed gives identical initial weights.
"""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

MODES = ("r1", "r2", "r3", "r4")


@dataclass
class ModelConfig:
    n_layer: int = 8
    d_model: int = 512
    n_head: int = 8
    d_ff: int = 1024
    n_exp: int = 8
    vocab: int = 50304
    ctx: int = 1024
    lb_coef: float = 0.01
    distill_coef: float = 0.1
    r4_lo: int = 2
    r4_hi: int = 51

    def to_dict(self):
        return asdict(self)


def base_config():
    return ModelConfig()


def tiny_config():
    return ModelConfig(n_layer=2, d_model=64, n_head=4, d_ff=128)


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


def future_mean(h, lo, hi):
    """Mean of h over positions t+lo .. min(t+hi, T-1) (zeros when empty), fp32 cumsum."""
    B, T, D = h.shape
    hf = h.float()
    c = torch.cat([hf.new_zeros(B, 1, D), torch.cumsum(hf, dim=1)], dim=1)  # c[k] = sum h[:k]
    t = torch.arange(T, device=h.device)
    a = t + lo
    b = torch.clamp(t + hi, max=T - 1)
    valid = a <= b
    a_i = torch.where(valid, a, torch.zeros_like(a))
    b_i = torch.where(valid, b + 1, torch.zeros_like(b))
    s = c[:, b_i] - c[:, a_i]
    cnt = torch.where(valid, b - a + 1, torch.ones_like(a)).to(hf.dtype)
    return s / cnt[None, :, None] * valid.to(hf.dtype)[None, :, None]


class RouteCtx:
    """Per-forward routing inputs shared by all layers."""

    def __init__(self, mode, teacher=None, alpha=0.0, coin=None, zero_router=False, keep_ids=False):
        self.mode = mode
        self.teacher = teacher  # [B, T] long or None
        self.alpha = float(alpha)
        self.coin = coin  # [B, T] float in [0, 1) or None
        self.zero_router = zero_router
        self.keep_ids = keep_ids


class MoE(nn.Module):
    def __init__(self, cfg, mode):
        super().__init__()
        self.cfg, self.mode = cfg, mode
        d, f, E = cfg.d_model, cfg.d_ff, cfg.n_exp
        self.shared_w1 = nn.Parameter(torch.empty(d, f))
        self.shared_w2 = nn.Parameter(torch.empty(f, d))
        self.w1 = nn.Parameter(torch.empty(E, d, f))
        self.w2 = nn.Parameter(torch.empty(E, f, d))
        self.router = nn.Parameter(torch.empty(E, d))  # r1/r4 router, r2/r3 student
        self.keep_debug = False  # tests: keep the routed weight w of the last forward
        self.debug_w = None

    def dispatch(self, x, e):
        """Dropless top-1: sort tokens by expert, run each expert on its contiguous slice, un-sort."""
        E = self.cfg.n_exp
        order = torch.argsort(e, stable=True)
        counts = torch.bincount(e, minlength=E)
        xs = x[order]
        outs = []
        start = 0
        for i, c in enumerate(counts.tolist()):
            if c == 0:
                continue
            seg = xs[start:start + c]
            outs.append(F.gelu(seg @ self.w1[i]) @ self.w2[i])
            start += c
        ys = torch.cat(outs, dim=0)
        return torch.zeros_like(ys).index_copy(0, order, ys), counts

    def forward(self, h, rc):
        B, T, D = h.shape
        E = self.cfg.n_exp
        hf = h.reshape(B * T, D)
        st = {}
        aux = None
        distill = None
        w = None
        if self.mode in ("r1", "r4"):
            if self.mode == "r4":
                rin = torch.zeros_like(h, dtype=torch.float32) if rc.zero_router else \
                    future_mean(h, self.cfg.r4_lo, self.cfg.r4_hi)
                rin = rin.reshape(B * T, D)
            else:
                rin = hf
            with torch.autocast(device_type=h.device.type, enabled=False):
                logits = F.linear(rin.float(), self.router.float())
            p = torch.softmax(logits, dim=-1)
            e = torch.argmax(p, dim=-1)
            pe = p.gather(1, e[:, None]).squeeze(1)
            w = pe / pe.detach()  # forward value exactly 1.0, gradient = d log p_e
            frac = torch.bincount(e, minlength=E).float() / e.numel()
            aux = E * (frac * p.mean(0)).sum()
        else:
            with torch.autocast(device_type=h.device.type, enabled=False):
                s = F.linear(hf.float(), self.router.float())
            student = torch.argmax(s, dim=-1)
            teacher = rc.teacher.reshape(-1) if rc.teacher is not None else None
            if teacher is not None:
                distill = F.cross_entropy(s, teacher)
                st["agree"] = (student == teacher).sum()
            if teacher is None or rc.alpha <= 0.0:
                e = student
            elif rc.alpha >= 1.0:
                e = teacher
            else:
                e = torch.where(rc.coin.reshape(-1) < rc.alpha, teacher, student)
        y, counts = self.dispatch(hf, e)
        if w is not None:
            if self.keep_debug:
                self.debug_w = w.detach()
            y = y * w[:, None].to(y.dtype)
        shared = F.gelu(hf @ self.shared_w1) @ self.shared_w2
        out = (shared + y).reshape(B, T, D)
        st["counts"] = counts
        if rc.keep_ids:
            st["ids"] = e.detach()
            if self.mode in ("r2", "r3"):
                st["student"] = student.detach()
        return out, aux, distill, st


class Block(nn.Module):
    def __init__(self, cfg, mode):
        super().__init__()
        self.ln1 = RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.ln2 = RMSNorm(cfg.d_model)
        self.moe = MoE(cfg, mode)

    def forward(self, x, rc):
        x = x + self.attn(self.ln1(x))
        m, aux, distill, st = self.moe(self.ln2(x), rc)
        return x + m, aux, distill, st


class Out:
    def __init__(self, **kw):
        self.__dict__.update(kw)


class GPT(nn.Module):
    def __init__(self, cfg, mode):
        super().__init__()
        assert mode in MODES, mode
        self.cfg, self.mode = cfg, mode
        self.wte = nn.Embedding(cfg.vocab, cfg.d_model)
        self.wpe = nn.Embedding(cfg.ctx, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg, mode) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.d_model)

    @torch.no_grad()
    def init_weights(self, seed):
        """normal(0, 0.02) everywhere, output projections scaled by 1/sqrt(2 n_layer), norms = 1.
        Drawn on CPU from one generator in parameter order -> identical across modes and devices."""
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        scaled = (".attn.proj.weight", ".moe.shared_w2", ".moe.w2")
        for name, p in self.named_parameters():
            if name.endswith("ln1.weight") or name.endswith("ln2.weight") or name == "ln_f.weight":
                p.fill_(1.0)
                continue
            std = 0.02
            if name.endswith(scaled):
                std /= math.sqrt(2 * self.cfg.n_layer)
            p.copy_((torch.randn(p.shape, generator=g) * std).to(p.device, p.dtype))
        return self

    def compile_attention(self):
        for b in self.blocks:
            b.attn.compile()

    def forward(self, idx, targets=None, teacher=None, alpha=0.0, coin=None, zero_router=False,
                reduction="mean", keep_ids=False, return_logits=False):
        B, T = idx.shape
        rc = RouteCtx(self.mode, teacher=teacher, alpha=alpha, coin=coin,
                      zero_router=zero_router, keep_ids=keep_ids)
        pos = torch.arange(T, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)[None]
        auxs, dists, stats = [], [], []
        for blk in self.blocks:
            x, aux, distill, st = blk(x, rc)
            if aux is not None:
                auxs.append(aux)
            if distill is not None:
                dists.append(distill)
            stats.append(st)
        x = self.ln_f(x)
        logits = F.linear(x, self.wte.weight)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1),
                                   reduction=reduction)
        aux = torch.stack(auxs).mean() if auxs else None
        distill = torch.stack(dists).mean() if dists else None
        return Out(logits=logits if return_logits else None, loss=loss, aux=aux, distill=distill,
                   stats=stats)

    def total_loss(self, out):
        """LM loss + the mode's auxiliary term (identical coefficients for every run)."""
        loss = out.loss
        if self.mode in ("r1", "r4"):
            loss = loss + self.cfg.lb_coef * out.aux
        elif out.distill is not None:
            loss = loss + self.cfg.distill_coef * out.distill
        return loss


def param_counts(model):
    cfg = model.cfg
    total = sum(p.numel() for p in model.parameters())
    per_layer_routed = cfg.n_exp * 2 * cfg.d_model * cfg.d_ff
    active = total - cfg.n_layer * per_layer_routed * (cfg.n_exp - 1) // cfg.n_exp
    emb = model.wte.weight.numel() + model.wpe.weight.numel()
    return dict(total=total, active_per_token=active, embedding=emb, non_embedding=total - emb,
                router_per_layer=cfg.n_exp * cfg.d_model)


def alpha_at(step, total_steps):
    """1 for step < 0.1 S; linear 1 -> 0 over [0.1 S, 0.6 S); 0 afterwards."""
    S = float(total_steps)
    a0, a1 = 0.10 * S, 0.60 * S
    if step < a0:
        return 1.0
    if step >= a1:
        return 0.0
    return 1.0 - (step - a0) / (a1 - a0)


def lr_at(step, total_steps, peak=1e-3, min_lr=1e-4, warmup_frac=0.02):
    warm = max(1, int(round(warmup_frac * total_steps)))
    if step < warm:
        return peak * (step + 1) / warm
    prog = min(1.0, (step - warm) / max(1, total_steps - warm))
    return min_lr + 0.5 * (peak - min_lr) * (1.0 + math.cos(math.pi * prog))
