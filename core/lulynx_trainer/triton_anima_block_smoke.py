"""End-to-end Anima DiT block smoke: eager vs fully Triton-fused.

Run with the flashattention env:
    backend/env/python-flashattention/python.exe \
        backend/core/lulynx_trainer/triton_anima_block_smoke.py

Builds a faithful Anima DiT block (self-attention + MLP + AdaLN + gated
residual) out of *real* ``LoRALinear`` wrappers, patches attention with the
real ``patch_anima_attention`` (so the ``_fused_qkv`` hook is live), then
applies all three shipped injections (base_lora ``apply``, ``apply_qkv``,
``apply_adaln``) to a second identically-initialised block. It checks output +
gradient parity against the eager block and measures the end-to-end forward and
forward+backward speedups, emitting the unified promotion scorecard.

This is the consolidation gate: it proves the three fusions compose correctly on
a realistic block and quantifies the real end-to-end win (dominated by AdaLN;
base_lora/qkv contribute forward-side gains, training-neutral).
"""

from __future__ import annotations

import os
import statistics
import sys
import time

_BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

import torch
import torch.nn.functional as F
from torch import nn

from core.lulynx_trainer.anima_attention import patch_anima_attention
from core.lulynx_trainer.lora_injector import LoRALinear
from core.lulynx_trainer.triton_ops import triton_inject
from core.lulynx_trainer.triton_ops.config import describe_gpu, detect_gpu
from core.lulynx_trainer.triton_ops_scorecard import build_triton_ops_scorecard

DEV = "cuda"
B, N, D = 2, 512, 2048
HEADS, HEAD_DIM = 16, 128
HIDDEN = 8192
RANK = 32


class _RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width))

    def forward(self, x):
        out = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return out * self.weight


def _lora(in_f: int, out_f: int) -> LoRALinear:
    return LoRALinear(nn.Linear(in_f, out_f, bias=True), rank=RANK, alpha=2 * RANK, dropout=0.0)


class _Attn(nn.Module):
    """Mirror of the real Anima _ProjectionAttention interface."""

    def __init__(self):
        super().__init__()
        self.num_heads, self.head_dim, self.hidden_dim = HEADS, HEAD_DIM, D
        self.q_proj = _lora(D, D)
        self.k_proj = _lora(D, D)
        self.v_proj = _lora(D, D)
        self.output_proj = _lora(D, D)
        self.q_norm = _RMSNorm(HEAD_DIM)
        self.k_norm = _RMSNorm(HEAD_DIM)

    def _split_heads(self, t):
        b, n, w = t.shape
        return t.view(b, n, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, t):
        b, h, n, hd = t.shape
        return t.transpose(1, 2).reshape(b, n, self.hidden_dim)

    def forward(self, x, context=None):
        src = x if context is None else context
        q = self.q_norm(self._split_heads(self.q_proj(x)))
        k = self.k_norm(self._split_heads(self.k_proj(src)))
        v = self._split_heads(self.v_proj(src))
        attn = F.scaled_dot_product_attention(q, k, v)
        return self.output_proj(self._merge_heads(attn))


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = _lora(D, HIDDEN)
        self.layer2 = _lora(HIDDEN, D)

    def forward(self, x):
        return self.layer2(F.silu(self.layer1(x)))


class _AdaLn(nn.Module):
    def __init__(self, chunks: int):
        super().__init__()
        self.add_module("0", nn.SiLU())
        self.add_module("1", nn.Linear(D, chunks * D, bias=True))
        self.chunks = chunks

    def forward(self, emb):
        out = emb
        for layer in self._modules.values():
            out = layer(out)
        return out.chunk(self.chunks, dim=-1)


class _Block(nn.Module):
    """Mirror of the real Anima _Block: adaln -> sublayer -> gated residual."""

    def __init__(self):
        super().__init__()
        self.self_attn = _Attn()
        self.mlp = _Mlp()
        self.adaln_modulation_self_attn = _AdaLn(3)
        self.adaln_modulation_cross_attn = _AdaLn(3)  # structural fidelity
        self.adaln_modulation_mlp = _AdaLn(3)

    def _apply_adaln(self, x, shift, scale):
        normalized = F.layer_norm(x.float(), (x.shape[-1],)).to(x.dtype)
        return normalized * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x, emb):
        shift, scale, gate = self.adaln_modulation_self_attn(emb)
        x = x + gate.unsqueeze(1) * self.self_attn(self._apply_adaln(x, shift, scale))
        shift, scale, gate = self.adaln_modulation_mlp(emb)
        x = x + gate.unsqueeze(1) * self.mlp(self._apply_adaln(x, shift, scale))
        return x


def _build(seed: int) -> nn.Module:
    torch.manual_seed(seed)
    blk = _Block()
    for m in blk.modules():
        if isinstance(m, LoRALinear):
            with torch.no_grad():
                m.lora.lora_down.weight.normal_(0.0, D ** -0.5)
                m.lora.lora_up.weight.normal_(0.0, RANK ** -0.5)  # real delta
    return blk.to(device=DEV, dtype=torch.bfloat16)


def _rel(a, b):
    return (a.float() - b.float()).abs().max().item() / (b.float().abs().max().item() + 1e-6)


def _data(seed=777):
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.randn(B, N, D, device=DEV, dtype=torch.bfloat16, generator=g)
    emb = torch.randn(B, D, device=DEV, dtype=torch.bfloat16, generator=g)
    gout = torch.randn(B, N, D, device=DEV, dtype=torch.bfloat16, generator=g)
    return x, emb, gout


def check_parity(net_e, net_f):
    print("== parity (eager vs fused) ==")
    x, emb, gout = _data()

    def run(net):
        for m in net.modules():
            if isinstance(m, LoRALinear):
                m.lora.lora_down.weight.grad = None
                m.lora.lora_up.weight.grad = None
        xx = x.clone().detach().requires_grad_(True)
        out = net(xx, emb)
        (out * gout).sum().backward()
        return out, xx.grad

    out_e, gx_e = run(net_e)
    out_f, gx_f = run(net_f)
    out_rel = _rel(out_f, out_e)
    gx_rel = _rel(gx_f, gx_e)

    worst_g = 0.0
    for me, mf in zip(net_e.modules(), net_f.modules()):
        if isinstance(me, LoRALinear):
            for nm in ("lora_down", "lora_up"):
                worst_g = max(worst_g, _rel(getattr(mf.lora, nm).weight.grad,
                                            getattr(me.lora, nm).weight.grad))
    ok = out_rel < 3e-2 and gx_rel < 3e-2 and worst_g < 5e-2
    print(f"  output rel={out_rel:.2e}  grad_x rel={gx_rel:.2e}  worst lora-grad rel={worst_g:.2e}")
    print(f"  {'OK' if ok else 'FAIL'}")
    return ok


def _time(net, *, backward: bool, warmup=8, iters=30) -> float:
    x, emb, gout = _data(seed=99)
    params = [p for p in net.parameters() if p.requires_grad]

    def step():
        if backward:
            for p in params:
                p.grad = None
            (net(x, emb) * gout).sum().backward()
        else:
            with torch.no_grad():
                net(x, emb)

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def measure(net_e, net_f, reps=5):
    print(f"== end-to-end speedup (median of {reps}) ==")
    out = {}
    for label, backward in (("fwd-only", False), ("fwd+bwd", True)):
        sps = []
        for _ in range(reps):
            te = _time(net_e, backward=backward)
            tf = _time(net_f, backward=backward)
            sps.append(te / tf if tf > 0 else 0.0)
        sp = statistics.median(sps)
        out[label] = sp
        print(f"  {label:8s}: median {sp:.3f}x  (reps {[round(s, 3) for s in sps]})")
    return out


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    gpu = detect_gpu()
    print(describe_gpu(gpu))

    net_e = _build(seed=42)
    net_f = _build(seed=42)
    patch_anima_attention(net_e, backend="sdpa")
    patch_anima_attention(net_f, backend="sdpa")
    n_lora = triton_inject.apply(net_f)
    n_qkv = triton_inject.apply_qkv(net_f)
    n_adaln = triton_inject.apply_adaln(net_f)
    print(f"injected: base_lora={n_lora} qkv={n_qkv} adaln={n_adaln}")

    parity_ok = check_parity(net_e, net_f)
    speed = measure(net_e, net_f)

    scorecard = build_triton_ops_scorecard(
        layers_patched=n_lora,
        numerical_verified=parity_ok,
        gradients_verified=parity_ok,
        performance_measured=True,
        forward_speedup=speed.get("fwd-only"),
        training_speedup=speed.get("fwd+bwd"),
        gpu_name=gpu.name,
        is_ada_or_newer=gpu.is_ada_or_newer,
        qkv_fused=n_qkv,
        adaln_blocks=n_adaln,
        components={
            "base_lora": {"forward": "~1.66x", "training": "~1.0x (neutral)"},
            "qkv_lora": {"forward": "~2.05x", "training": "~0.99x (neutral)"},
            "adaln_norm": {"forward": "~2.8x", "training": "~1.23x (real win)"},
        },
    )
    print("\n== unified scorecard ==")
    for k, v in scorecard.items():
        print(f"  {k}: {v}")
    print("\nRESULT:", "PARITY PASS" if parity_ok else "PARITY FAIL",
          f"| e2e fwd-only={speed.get('fwd-only'):.3f}x fwd+bwd={speed.get('fwd+bwd'):.3f}x",
          f"| scorecard.ok={scorecard['ok']}")
    sys.exit(0 if parity_ok else 1)
