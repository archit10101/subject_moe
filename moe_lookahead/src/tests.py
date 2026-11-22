"""Correctness tests (tiny model, fp32, CPU, a few seconds). Exit code 1 if any test fails.

Run locally and at the start of the bench job; training never starts unless all pass.
"""
import os
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BLOCK, mutual_info_bits  # noqa: E402
from data import coin, micro_seq_ids, seq_batch, teacher_ids, train_order  # noqa: E402
from model import GPT, MODES, alpha_at, future_mean, lr_at, tiny_config  # noqa: E402

torch.set_num_threads(min(4, os.cpu_count() or 1))
CFG = tiny_config()
T = 256  # 4 blocks of 64
B = 2


def make(mode, seed=0):
    m = GPT(CFG, mode).init_weights(seed)
    m.eval()
    return m


def rand_tokens(g, b=B, t=T):
    return torch.randint(0, 50257, (b, t), generator=g)


def rand_teacher(g, b=B, t=T):
    blocks = torch.randint(0, 8, (b, t // BLOCK), generator=g)
    return blocks.repeat_interleave(BLOCK, dim=1)


def logits_of(model, idx, **kw):
    with torch.no_grad():
        return model(idx, return_logits=True, **kw).logits


# ------------------------------------------------------------------ 1 + 2
def test_causality():
    g = torch.Generator().manual_seed(1)
    x = rand_tokens(g)
    x2 = x.clone()
    x2[:, T // 2:] = rand_tokens(g)[:, T // 2:]
    for mode in ("r1", "r2", "r3"):
        m = make(mode)
        # student-only routing (alpha = 0); the teacher is irrelevant then
        d = (logits_of(m, x)[:, :T // 2] - logits_of(m, x2)[:, :T // 2]).abs().max().item()
        assert d < 1e-4, f"{mode}: earlier logits changed by {d:.2e} when later tokens changed"


def test_causality_has_teeth():
    g = torch.Generator().manual_seed(2)
    x = rand_tokens(g)
    x2 = x.clone()
    x2[:, T // 2:] = rand_tokens(g)[:, T // 2:]
    m = make("r4")
    d = (logits_of(m, x)[:, :T // 2] - logits_of(m, x2)[:, :T // 2]).abs().max().item()
    assert d > 1e-3, f"R4 should leak the future but earlier logits changed by only {d:.2e}"
    dz = (logits_of(m, x, zero_router=True)[:, :T // 2] -
          logits_of(m, x2, zero_router=True)[:, :T // 2]).abs().max().item()
    assert dz < 1e-4, f"R4 with zero_router must be causal, diff {dz:.2e}"


# ------------------------------------------------------------------ 3
def test_teacher_alignment():
    rng = np.random.default_rng(0)
    n_seq = 7
    stream = rng.integers(0, 50257, size=n_seq * 1024 + 1).astype(np.uint16)
    clusters = rng.integers(0, 8, size=(len(stream) + 63) // 64).astype(np.uint8)
    perm = train_order(n_seq, 0)
    assert sorted(perm.tolist()) == list(range(n_seq))
    assert np.array_equal(perm, train_order(n_seq, 0)), "data order must be deterministic"
    ids = micro_seq_ids(perm, step=1, micro=0, grad_accum=1, micro_bs=3)
    assert np.array_equal(ids, perm[3:6])
    x, y = seq_batch(stream, ids)
    t3 = teacher_ids(clusters, ids, "r3")
    t2 = teacher_ids(clusters, ids, "r2")
    for b, i in enumerate(ids):
        assert np.array_equal(x[b], stream[i * 1024:(i + 1) * 1024])
        assert np.array_equal(y[b], stream[i * 1024 + 1:(i + 1) * 1024 + 1])
        for t in range(1024):
            j = i * 1024 + t
            assert t3[b, t] == clusters[j // 64]
            assert t2[b, t] == clusters[max(j // 64 - 1, 0)]
    # sequence 0: R2 uses cluster[0] for the first block
    t2_0 = teacher_ids(clusters, np.array([0]), "r2")
    assert np.all(t2_0[0, :128] == clusters[0])


# ------------------------------------------------------------------ 4
def test_dropless_moe():
    m = make("r1")
    moe = m.blocks[0].moe
    g = torch.Generator().manual_seed(3)
    N = 300
    x = torch.randn(N, CFG.d_model, generator=g)
    e = torch.randint(0, 8, (N,), generator=g)
    e[e == 5] = 4  # an expert with zero tokens must be handled
    y, counts = moe.dispatch(x, e)
    ref = torch.stack([F.gelu(x[i] @ moe.w1[e[i]]) @ moe.w2[e[i]] for i in range(N)])
    d = (y - ref).abs().max().item()
    assert d < 1e-4, f"dispatch differs from naive loop by {d:.2e}"
    assert counts.sum().item() == N and counts[5].item() == 0
    assert torch.equal(counts, torch.bincount(e, minlength=8))
    order = torch.argsort(e, stable=True)
    assert torch.equal(torch.sort(order).values, torch.arange(N)), "every token routed exactly once"


# ------------------------------------------------------------------ 5
def test_alpha_schedule():
    S = 1000
    vals = {0: 1.0, 100: 1.0, 350: 0.5, 600: 0.0, 1000: 0.0}
    for s, v in vals.items():
        assert abs(alpha_at(s, S) - v) < 1e-9, (s, alpha_at(s, S), v)
    assert abs(alpha_at(99, S) - 1.0) < 1e-9 and 0 < alpha_at(599, S) < 0.01
    # lr: warmup then cosine to 1e-4
    assert abs(lr_at(19, S) - 1e-3) < 1e-12 and abs(lr_at(S, S) - 1e-4) < 1e-12 and lr_at(0, S) < 1e-4


# ------------------------------------------------------------------ 6
def test_r4_window():
    g = torch.Generator().manual_seed(4)
    Tt = 128
    h = torch.randn(2, Tt, 16, generator=g)
    fm = future_mean(h, CFG.r4_lo, CFG.r4_hi)
    for t in range(Tt):
        a, b = t + 2, min(t + 51, Tt - 1)
        ref = h[:, a:b + 1].mean(1) if a <= b else torch.zeros(2, 16)
        d = (fm[:, t] - ref).abs().max().item()
        assert d < 1e-4, f"t={t}: future mean off by {d:.2e}"
    assert fm[:, Tt - 1].abs().max() == 0 and fm[:, Tt - 2].abs().max() == 0
    assert torch.allclose(fm[:, Tt - 3], h[:, Tt - 1], atol=1e-5)


# ------------------------------------------------------------------ 7
def test_identical_init():
    sds = {mode: GPT(CFG, mode).init_weights(0).state_dict() for mode in MODES}
    keys = list(sds["r1"].keys())
    for mode in MODES:
        assert list(sds[mode].keys()) == keys, f"{mode}: different parameter names/order"
        for k in keys:
            assert torch.equal(sds[mode][k], sds["r1"][k]), f"{mode}: {k} differs from r1"
    assert sds["r1"]["blocks.0.moe.router"].shape == (8, CFG.d_model)
    other = GPT(CFG, "r1").init_weights(1).state_dict()
    assert not torch.equal(other["blocks.0.moe.w1"], sds["r1"]["blocks.0.moe.w1"]), "seed must matter"
    for k in keys:  # no _orig_mod. prefix and no extra buffers
        assert "_orig_mod" not in k


# ------------------------------------------------------------------ 8
def test_gradients():
    g = torch.Generator().manual_seed(5)
    x, y = rand_tokens(g), rand_tokens(g)
    t = rand_teacher(g)
    u = torch.rand(B, T, generator=g)
    for mode in ("r1", "r4"):
        m = GPT(CFG, mode).init_weights(0)
        out = m(x, y)
        out.loss.backward()  # LM loss alone, no load-balance term
        gr = m.blocks[0].moe.router.grad
        assert gr is not None and gr.abs().max() > 0, f"{mode}: router got no LM gradient"
    for mode in ("r2", "r3"):
        for alpha in (0.0, 0.5, 1.0):
            m = GPT(CFG, mode).init_weights(0)
            out = m(x, y, teacher=t, alpha=alpha, coin=u)
            out.loss.backward()
            for blk in m.blocks:
                gr = blk.moe.router.grad
                assert gr is None or gr.abs().max() == 0, f"{mode} alpha={alpha}: student got an LM gradient"
            assert m.wte.weight.grad.abs().max() > 0
        m = GPT(CFG, mode).init_weights(0)
        out = m(x, y, teacher=t, alpha=0.5, coin=u)
        out.distill.backward()
        for blk in m.blocks:
            assert blk.moe.router.grad.abs().max() > 0, f"{mode}: student got no distillation gradient"
        # distillation also reaches the backbone (intended)
        assert m.blocks[0].attn.qkv.weight.grad.abs().max() > 0


# ------------------------------------------------------------------ 9
def test_alpha1_loads_equal_teacher():
    g = torch.Generator().manual_seed(6)
    x = rand_tokens(g)
    t = rand_teacher(g)
    for mode in ("r2", "r3"):
        m = make(mode)
        with torch.no_grad():
            out = m(x, teacher=t, alpha=1.0, keep_ids=True)
        ref = torch.bincount(t.reshape(-1), minlength=8)
        for st in out.stats:
            assert torch.equal(st["counts"], ref), f"{mode}: loads {st['counts'].tolist()} != {ref.tolist()}"
            assert torch.equal(st["ids"], t.reshape(-1))
        # alpha=0 -> student; mixed alpha uses the coin
        with torch.no_grad():
            o0 = m(x, teacher=t, alpha=0.0, keep_ids=True)
            u = torch.rand(B, T, generator=g)
            oh = m(x, teacher=t, alpha=0.5, coin=u, keep_ids=True)
        st0 = o0.stats[0]
        assert torch.equal(st0["ids"], st0["student"])
        sh = oh.stats[0]
        use_t = (u.reshape(-1) < 0.5)
        assert torch.equal(sh["ids"][use_t], t.reshape(-1)[use_t])
        assert torch.equal(sh["ids"][~use_t], sh["student"][~use_t])


# ------------------------------------------------------------------ 10
def test_routed_weight_is_one():
    g = torch.Generator().manual_seed(7)
    x = rand_tokens(g)
    for mode in ("r1", "r4"):
        m = make(mode)
        for blk in m.blocks:
            blk.moe.keep_debug = True
        with torch.no_grad():
            m(x)
        for blk in m.blocks:
            w = blk.moe.debug_w
            assert w is not None and torch.all(w == 1.0), f"{mode}: routed weight is not exactly 1.0"


# ------------------------------------------------------------------ extras
def test_coin_and_mi():
    a = coin(0, 5, 1, (2, 8))
    assert torch.equal(a, coin(0, 5, 1, (2, 8))) and not torch.equal(a, coin(0, 5, 2, (2, 8)))
    z = np.random.default_rng(0).integers(0, 8, 100000)
    assert abs(mutual_info_bits(z, z, 8, 8) - 3.0) < 0.01
    assert mutual_info_bits(z, np.random.default_rng(1).integers(0, 7, 100000), 8, 7) < 0.01


def test_losses_finite_all_modes():
    g = torch.Generator().manual_seed(8)
    x, y, t = rand_tokens(g), rand_tokens(g), rand_teacher(g)
    for mode in MODES:
        m = GPT(CFG, mode).init_weights(0)
        out = m(x, y, teacher=t, alpha=1.0)
        loss = m.total_loss(out)
        assert torch.isfinite(loss), mode
        assert abs(out.loss.item() - np.log(50304)) < 0.5, f"{mode}: initial loss {out.loss.item():.3f}"
        if mode in ("r1", "r4"):
            assert out.aux is not None and out.aux.item() >= 0.99  # 8 * sum f P >= 1 at balance
        else:
            assert out.distill is not None


TESTS = [
    test_causality, test_causality_has_teeth, test_teacher_alignment, test_dropless_moe,
    test_alpha_schedule, test_r4_window, test_identical_init, test_gradients,
    test_alpha1_loads_equal_teacher, test_routed_weight_is_one, test_coin_and_mi,
    test_losses_finite_all_modes,
]


def main():
    t0 = time.time()
    failed = 0
    for i, t in enumerate(TESTS, 1):
        ts = time.time()
        try:
            t()
            print(f"PASS {i:2d} {t.__name__} ({time.time() - ts:.2f}s)", flush=True)
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {i:2d} {t.__name__}", flush=True)
            traceback.print_exc()
    print(f"{len(TESTS) - failed}/{len(TESTS)} tests passed in {time.time() - t0:.1f}s")
    if failed:
        print("TESTS FAILED")
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
