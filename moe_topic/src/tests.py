"""Correctness tests (tiny model, fp32, CPU, well under a minute). Exit code 1 if any test fails.

Run locally and at the start of the bench job; training never starts unless all pass.
"""
import os
import sys
import time
import traceback

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import EOT, mutual_info_bits  # noqa: E402
import metrics as M  # noqa: E402
from data import micro_seq_ids, train_order  # noqa: E402
from model import (GPT, MODES, base_config, context_summary, pool_specs, routed_units, summary_start,  # noqa: E402
                   tiny_config, uses_summary)

torch.set_num_threads(min(4, os.cpu_count() or 1))
CFG = tiny_config()
T = 256
B = 2


def make(mode, seed=0, cfg=CFG):
    m = GPT(cfg, mode).init_weights(seed)
    m.eval()
    return m


def tokens_with_eots(g, b=B, t=T, every=(17, 40)):
    """Random tokens with EOTs at irregular positions, including one at the very end of row 0."""
    x = torch.randint(0, 50000, (b, t), generator=g)
    for r in range(b):
        p = int(torch.randint(1, every[0], (1,), generator=g))
        while p < t:
            x[r, p] = EOT
            p += int(torch.randint(every[0], every[1], (1,), generator=g))
    x[0, t - 1] = EOT
    x[1, T // 2 - 1] = EOT  # an EOT right before the changed half
    return x


def logits_of(model, idx):
    with torch.no_grad():
        return model(idx, return_logits=True).logits


def change_second_half(g, x):
    x2 = x.clone()
    y = tokens_with_eots(g, x.shape[0], x.shape[1], every=(5, 30))
    x2[:, T // 2:] = y[:, T // 2:]
    assert (x2[:, T // 2:] != x[:, T // 2:]).float().mean() > 0.9
    return x2


def leaky_summary(h, start):
    """TEST ONLY: mean over t+1 .. t+50 (sees the future)."""
    Bh, Th, D = h.shape
    hf = h.float()
    c = torch.cat([hf.new_zeros(Bh, 1, D), torch.cumsum(hf, 1)], 1)
    t = torch.arange(Th)
    a = (t + 1).clamp(max=Th)
    b = (t + 51).clamp(max=Th)
    cnt = (b - a).clamp(min=1).to(hf.dtype)
    return (c[:, b] - c[:, a]) / cnt[None, :, None]


# ------------------------------------------------------------------ 1. causality, every mode
def test_causality_all_modes():
    g = torch.Generator().manual_seed(1)
    x = tokens_with_eots(g)
    x2 = change_second_half(g, x)
    for mode in MODES:
        m = make(mode)
        d = (logits_of(m, x)[:, :T // 2] - logits_of(m, x2)[:, :T // 2]).abs().max().item()
        assert d < 1e-4, f"{mode}: earlier logits changed by {d:.2e} when later tokens changed"
        later = (logits_of(m, x)[:, T // 2:] - logits_of(m, x2)[:, T // 2:]).abs().max().item()
        assert later > 1e-3, f"{mode}: sanity: later logits should change"


# ------------------------------------------------------------------ 2. context summary vs a naive loop
def naive_summary(h, idx, window=50):
    Bh, Th, D = h.shape
    out = torch.zeros(Bh, Th, D, dtype=torch.float64)
    for b in range(Bh):
        for t in range(Th):
            d = 0
            for p in range(t - 1, -1, -1):  # latest EOT strictly before t
                if int(idx[b, p]) == EOT:
                    d = p + 1
                    break
            s = max(t - window + 1, d)
            out[b, t] = h[b, s:t + 1].double().mean(0)
    return out


def test_context_summary():
    g = torch.Generator().manual_seed(3)
    idx = torch.randint(0, 50000, (3, 130), generator=g)
    idx[0, [0, 5, 6, 60, 129]] = EOT   # EOT at the start, two in a row, one at the very end
    idx[1, [100]] = EOT                # long document (> 50) then a reset
    # row 2: no EOT at all (windows shorter than 50 only at the sequence start)
    h = torch.randn(3, 130, 8, generator=g)
    st = summary_start(idx, EOT, 50)
    s = context_summary(h, st)
    ref = naive_summary(h, idx)
    err = (s.double() - ref).abs().max().item()
    assert err < 1e-5, f"summary differs from naive loop by {err:.2e}"
    # explicit spot checks
    assert int(st[0, 6]) == 6 and int(st[0, 7]) == 7, "window must reset right after an EOT"
    assert int(st[0, 5]) == 1, "the EOT itself belongs to its document"
    assert int(st[1, 99]) == 50 and int(st[1, 100]) == 51 and int(st[1, 101]) == 101
    assert int(st[2, 10]) == 0 and int(st[2, 49]) == 0 and int(st[2, 50]) == 1
    assert torch.allclose(s[2, 0], h[2, 0])


# ------------------------------------------------------------------ 3. the causality test has teeth
def test_causality_has_teeth():
    g = torch.Generator().manual_seed(2)
    x = tokens_with_eots(g)
    x2 = change_second_half(g, x)
    for mode in ("p1", "p2", "p4"):
        m = make(mode)
        m.summary_fn = leaky_summary
        d = (logits_of(m, x)[:, :T // 2] - logits_of(m, x2)[:, :T // 2]).abs().max().item()
        assert d > 1e-3, f"{mode}: the leaky summary should fail the causality test, diff only {d:.2e}"


# ------------------------------------------------------------------ 4. dropless dispatch vs naive loop
def test_dispatch():
    torch.manual_seed(0)
    for mode in ("r1", "r1f", "p4"):
        m = make(mode)
        for _, _, pool in m.pools():
            N = 300
            x = torch.randn(N, CFG.d_model)
            rin = torch.randn(N, pool.router.shape[1])
            with torch.no_grad():
                y, aux, counts, top_i = pool(x, rin)
                p, top_p, ti = pool.route(rin)
                ref = torch.zeros_like(x)
                for n in range(N):
                    for s in range(pool.top_k):
                        e = int(ti[n, s])
                        ref[n] += torch.nn.functional.gelu(x[n] @ pool.w1[e]) @ pool.w2[e]
            err = (y - ref).abs().max().item()
            assert err < 1e-4, f"{mode}: dispatch differs from naive loop by {err:.2e}"
            assert int(counts.sum()) == N * pool.top_k, "every (token, slot) pair routed exactly once"
            pairs = set((n, int(top_i[n, s])) for n in range(N) for s in range(pool.top_k))
            assert len(pairs) == N * pool.top_k, "top-k slots of a token must be distinct experts"
            assert torch.equal(counts, torch.bincount(top_i.reshape(-1), minlength=pool.n_exp))


# ------------------------------------------------------------------ 5. compute matching
def test_compute_matching():
    base = base_config()
    shared = None
    for mode in MODES:
        act, tot = routed_units(base, mode)
        assert (act, tot) == (1024, 8192), f"{mode}: routed units {act}/{tot}, want 1024/8192"
        with torch.device("meta"):
            m = GPT(base, mode)
        sh = tuple(m.blocks[0].moe.shared_w1.shape) + tuple(m.blocks[0].moe.shared_w2.shape)
        assert sh == (512, 1024, 1024, 512), sh
        shared = shared or sh
        assert sh == shared
        a = sum(p.top_k * p.d_ff for li, _, p in m.pools() if li == 0)
        t = sum(p.n_exp * p.d_ff for li, _, p in m.pools() if li == 0)
        assert (a, t) == (1024, 8192)
        act_t, tot_t = routed_units(CFG, mode)
        assert (act_t, tot_t) == (128, 1024), f"tiny {mode}: {act_t}/{tot_t}"


# ------------------------------------------------------------------ 6. init
def test_init():
    models = {mode: make(mode, 0) for mode in MODES}
    sd = {mode: dict(m.named_parameters()) for mode, m in models.items()}
    names_compared = 0
    for a in MODES:
        for b in MODES:
            for n, p in sd[a].items():
                q = sd[b].get(n)
                if q is not None and q.shape == p.shape:
                    assert torch.equal(p, q), f"{n} differs between {a} and {b}"
                    names_compared += 1
    assert names_compared > 0
    r1 = dict(make("r1", 0).named_parameters())
    assert all(torch.equal(r1[n], p) for n, p in sd["r1"].items()), "init must be deterministic"
    r1s = dict(make("r1", 1).named_parameters())
    for n, p in sd["r1"].items():
        if "ln" in n:
            continue
        assert not torch.equal(p, r1s[n]), f"init seed 1 must change {n}"
    # construction order must not matter: build p4 pools in the opposite order
    m = GPT(CFG, "p4")
    for blk in m.blocks:
        blk.moe.pools = torch.nn.ModuleDict(list(blk.moe.pools.items())[::-1])
    m.init_weights(0)
    for n, p in m.named_parameters():
        assert torch.equal(p, sd["p4"][n]), f"{n} depends on construction order"


# ------------------------------------------------------------------ 7. gradients + labels never enter forward
def test_gradients():
    g = torch.Generator().manual_seed(4)
    x = tokens_with_eots(g)
    y = torch.randint(0, 50000, x.shape, generator=g)
    soft = torch.softmax(torch.randn(B, T, 24, generator=g) * 3, -1)
    for mode in MODES:
        m = make(mode)
        m.train()
        out = m(x, y, topic_targets=soft if mode == "p3" else None)
        out.loss.backward()  # LM loss only
        for li, name, pool in m.pools():
            gn = pool.router.grad
            assert gn is not None and gn.abs().sum() > 0, f"{mode} layer {li} {name}: router has no LM gradient"
        if mode == "p3":
            for blk in m.blocks:
                th = blk.moe.topic_head
                assert th.weight.grad is None or th.weight.grad.abs().sum() == 0, "topic head got an LM gradient"
            m.zero_grad(set_to_none=True)
            out = m(x, y, topic_targets=soft)
            (out.loss + m.cfg.lb_coef * out.aux).backward()
            ref = {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
            m.zero_grad(set_to_none=True)
            out = m(x, y, topic_targets=soft)
            out.topic_loss.backward()
            naive_router = [p.router.grad for _, _, p in m.pools()]
            assert any(g is not None and g.abs().sum() > 0 for g in naive_router), \
                "sanity: a plain backward of the topic loss does reach earlier routers"
            m.zero_grad(set_to_none=True)
            out = m(x, y, topic_targets=soft)
            m.backward(out)  # the training path
            for n, p in m.named_parameters():
                if n.endswith(".router"):
                    assert torch.allclose(p.grad, ref[n], atol=1e-7, rtol=1e-5), f"{n}: router trained by topic loss"
            for blk in m.blocks:
                assert blk.moe.topic_head.weight.grad.abs().sum() > 0, "topic head must learn from the topic loss"
            d = (m.wte.weight.grad - ref["wte.weight"]).abs().sum()
            assert d > 0, "topic loss should reach the backbone through s_t"
            m.eval()
            with torch.no_grad():
                a = m(x, y, topic_targets=soft, return_logits=True)
                perm = torch.randperm(B * T, generator=g)
                b = m(x, y, topic_targets=soft.reshape(B * T, 24)[perm].reshape(B, T, 24), return_logits=True)
                c = m(x, y, topic_targets=None, return_logits=True)
            assert torch.equal(a.logits, b.logits) and torch.equal(a.logits, c.logits), \
                "logits must not depend on topic labels"
            assert not torch.allclose(a.topic_loss, b.topic_loss), "sanity: shuffled labels change the topic loss"


# ------------------------------------------------------------------ 8. routed weights are exactly 1.0
def test_routed_weight_one():
    g = torch.Generator().manual_seed(5)
    x = tokens_with_eots(g)
    for mode in MODES:
        m = make(mode)
        for _, _, p in m.pools():
            p.keep_debug = True
        with torch.no_grad():
            m(x)
        for _, _, p in m.pools():
            assert torch.equal(p.debug_w, torch.ones_like(p.debug_w)), f"{mode}: routed weights not exactly 1.0"


# ------------------------------------------------------------------ 9. ablation hook
def test_ablation():
    torch.manual_seed(1)
    for mode in ("r1", "r1f", "p4"):
        m = make(mode)
        for _, _, pool in m.pools():
            N = 400
            x = torch.randn(N, CFG.d_model)
            rin = torch.randn(N, pool.router.shape[1]) * 5
            with torch.no_grad():
                y0, _, _, i0 = pool(x, rin)
                e = int(torch.bincount(i0[:, 0], minlength=pool.n_exp).argmax())
                p, _, _ = pool.route(rin)
                pool.disabled = e
                y1, _, c1, i1 = pool(x, rin)
                pool.disabled = None
            assert int(c1[e]) == 0, "disabled expert must get no tokens"
            hit = (i0 == e).any(1)
            assert hit.sum() > 0
            # rerouted to the next-best expert(s) in the same pool
            masked = p.clone()
            masked[:, e] = -1
            want = torch.topk(masked, pool.top_k, dim=-1).indices
            assert torch.equal(i1, want), "tokens must go to their next-best expert"
            assert torch.equal(i1[~hit], i0[~hit]), "other tokens' routing must not change"
            assert torch.allclose(y1[~hit], y0[~hit], atol=1e-6), "other tokens' outputs must not change"
            assert (y1[hit] - y0[hit]).abs().max() > 1e-6


# ------------------------------------------------------------------ 10. metrics on known inputs
def test_metrics():
    rng = np.random.default_rng(0)
    n = 200000
    topic = rng.integers(0, 24, n)
    h = M.topic_info(topic[:, None], topic, 24, 24)
    hb = -np.sum((np.bincount(topic) / n) * np.log2(np.bincount(topic) / n))
    assert abs(h["mi"] - hb) < 1e-9 and abs(h["nmi"] - 1) < 1e-9, h
    assert abs(mutual_info_bits(topic, topic, 24, 24) - hb) < 1e-9
    r = M.topic_info(rng.integers(0, 8, (n, 1)), topic, 8, 24)
    assert r["mi"] < 0.002, r
    r2 = M.topic_info(rng.integers(0, 16, (n, 2)), topic, 16, 24)  # two slots
    assert r2["mi"] < 0.002 and r2["n"] == 2 * n
    # 8 experts, each owning 3 topics -> I = log2(8) = 3 bits
    r3 = M.topic_info((topic // 3)[:, None], topic, 8, 24)
    assert abs(r3["mi"] - np.log2(8)) < 0.01, r3
    # token-driven routing: expert = f(token), topic correlated with token -> MI > 0 but I(E;T|W) = 0
    tok = rng.integers(0, 30, n)
    topic_t = np.where(rng.random(n) < 0.7, tok % 24, rng.integers(0, 24, n))
    cw = M.common_word((tok % 8)[:, None], topic_t, tok, 8, 24, top_ids=np.arange(30), min_count=100)
    assert cw["mi_top"] > 0.3 and cw["cmi"] < 0.005, cw
    cw2 = M.common_word((topic_t // 3)[:, None], topic_t, tok, 8, 24, top_ids=np.arange(30), min_count=100)
    assert cw2["cmi"] > 0.5, cw2
    # conditional on format
    fmt = rng.integers(0, 24, n)
    fc = M.format_control((fmt // 3)[:, None], topic, fmt, 8, 24, 24)
    assert fc["mi_format"] > 2.9 and fc["cmi_topic_given_format"] < 0.01, fc
    # purity: perfect experts
    pu = M.purity((topic // 3)[:, None], topic, 8, 24)
    assert all(abs(s - 1) < 1e-9 for s in pu["top3_share"]) and abs(pu["mean_topic_max_expert_share"] - 1) < 1e-9
    assert all(abs(l - 8) < 0.5 for l in pu["top_lift"]), pu["top_lift"]
    # consistency
    e = np.repeat(np.arange(100) % 8, 10)
    same = np.ones(1000, bool)
    same[::10] = False
    c = M.consistency(e[:, None], same)
    assert abs(c["p_same"] - 1) < 1e-9 and c["n"] == 900
    hl = M.health(np.zeros((100, 1), int), 8)
    assert hl["dead"] == 7 and hl["max_load"] == 1.0
    # shuffled labels across documents: chance level
    doc_topic = rng.integers(0, 24, 2000)
    doc_of_tok = np.repeat(np.arange(2000), 100)
    tt = doc_topic[doc_of_tok]
    ids = (tt // 3)[:, None]
    sh = M.shuffled_topic(doc_topic, doc_of_tok, 0)
    assert M.topic_info(ids, tt, 8, 24)["mi"] > 2.9 and M.topic_info(ids, sh, 8, 24)["mi"] < 0.1
    # selectivity
    d = np.zeros((3, 24))
    d[0, 5] = 0.2
    d[1] = 0.01
    d[2] = -0.01
    sel, hurt = M.selectivity(d)
    assert abs(sel[0] - 1) < 1e-12 and abs(sel[1] - 1 / 24) < 1e-12 and np.isnan(sel[2]) and hurt[0] == 5


# ------------------------------------------------------------------ misc
def test_data_order():
    perm = train_order(10, 0)
    assert sorted(perm.tolist()) == list(range(10))
    assert np.array_equal(perm, train_order(10, 0)), "data order must be deterministic"
    ids = micro_seq_ids(perm, step=1, micro=1, grad_accum=2, micro_bs=3)
    assert np.array_equal(ids, perm[[9, 0, 1]])


def test_pool_specs():
    for mode in MODES:
        sp = pool_specs(CFG, mode)
        assert all(inp in ("h", "s", "hs") for *_, inp in sp)
        assert uses_summary(mode) == any(inp != "h" for *_, inp in sp)


TESTS = [test_causality_all_modes, test_context_summary, test_causality_has_teeth, test_dispatch,
         test_compute_matching, test_init, test_gradients, test_routed_weight_one, test_ablation, test_metrics,
         test_data_order, test_pool_specs]


def main():
    fails = 0
    t00 = time.time()
    for t in TESTS:
        t0 = time.time()
        try:
            t()
            print(f"PASS {t.__name__} ({time.time() - t0:.1f}s)", flush=True)
        except Exception:  # noqa: BLE001
            fails += 1
            print(f"FAIL {t.__name__}", flush=True)
            traceback.print_exc()
    print(f"{len(TESTS) - fails}/{len(TESTS)} tests passed in {time.time() - t00:.1f}s")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
