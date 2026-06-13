# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""CPU smoke for modulation_guidance_reserve_seam.py (full pooled-text route).

Proves the seam is a genuine default-off opt-in reserve on a minimal AdaLN-shaped
stand-in (mirrors the real `_Block` / `_AdaLn` modulation layout):

* ``method="none"`` wraps nothing -> bitwise parity.
* installed-but-idle (no pooled text published, or zero-init projector) -> parity.
* a trained (non-zero) projector + published pooled text really steers the output,
  finite, and leaves the gate chunk untouched.
* the distillation loss is off==base / on adds a real term whose gradient reaches
  the projector.
* the replay request round-trips and the readiness report stays honest while the
  *default* reconciliation is unchanged (still partial).
* ``handle.remove()`` restores the original forward bit-for-bit.

Run:
    backend/env/python-flashattention/python.exe \
        backend/core/lulynx_trainer/modulation_guidance_reserve_smoke.py
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

import torch
import torch.nn as nn

from core.lulynx_trainer.modulation_guidance_reconciliation import (
    build_modulation_guidance_reconciliation,
)
from core.lulynx_trainer.modulation_guidance_reserve_seam import (
    PooledTextModulationProjector,
    build_modulation_guidance_replay_request,
    compose_modulation_guidance_distill_loss,
    install_modulation_guidance_reserve,
    modulation_guidance_pooled_text_context,
    modulation_guidance_reserve_readiness,
)

HIDDEN = 16
TEXT_DIM = 12


class _AdaLnStandIn(nn.Module):
    """SiLU + linear -> chunked (shift, scale, gate), like the real `_AdaLn`."""

    def __init__(self, hidden: int, chunks: int = 3):
        super().__init__()
        self.chunks = chunks
        self.act = nn.SiLU()
        self.lin = nn.Linear(hidden, hidden * chunks)

    def forward(self, emb):
        return self.lin(self.act(emb)).chunk(self.chunks, dim=-1)


class _Block(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.adaln_modulation_self_attn = _AdaLnStandIn(hidden, 3)
        self.proj = nn.Linear(hidden, hidden)

    def forward(self, x, emb):
        shift, scale, gate = self.adaln_modulation_self_attn(emb)
        normalized = torch.nn.functional.layer_norm(x.float(), (x.shape[-1],)).to(dtype=x.dtype)
        h = normalized * (1.0 + scale) + shift
        return x + gate * self.proj(h)


class _FakeDiT(nn.Module):
    def __init__(self, hidden: int, n_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(hidden) for _ in range(n_blocks)])

    def forward(self, x, emb):
        out = x
        for blk in self.blocks:
            out = blk(out, emb)
        return out


def _fixture(seed: int = 0):
    torch.manual_seed(seed)
    model = _FakeDiT(HIDDEN, n_blocks=3)
    x = torch.randn(2, HIDDEN)
    emb = torch.randn(2, HIDDEN)
    pooled = torch.randn(2, TEXT_DIM)
    return model, x, emb, pooled


def test_method_none_is_bitwise_parity():
    model, x, emb, _ = _fixture()
    baseline = model(x, emb).clone()
    handle = install_modulation_guidance_reserve(model, "none")
    assert handle.wrapped_count == 0
    assert torch.equal(model(x, emb), baseline)
    print("PASS: method=none wraps nothing and is bitwise parity")


def test_installed_but_no_pooled_text_is_parity():
    model, x, emb, _ = _fixture()
    baseline = model(x, emb).clone()
    install_modulation_guidance_reserve(model, "pooled_text", text_dim=TEXT_DIM)
    # No pooled-text context active -> wrapped sites fall back to original output.
    assert torch.equal(model(x, emb), baseline)
    print("PASS: installed reserve with no published pooled text is parity")


def test_zero_init_projector_is_parity_even_with_pooled_text():
    model, x, emb, pooled = _fixture()
    baseline = model(x, emb).clone()
    install_modulation_guidance_reserve(model, "pooled_text", text_dim=TEXT_DIM)
    with modulation_guidance_pooled_text_context(pooled):
        out = model(x, emb)
    assert torch.allclose(out, baseline, atol=1e-6)
    print("PASS: zero-init projector emits zero deltas -> parity")


def test_trained_projector_steers_output_and_leaves_gate_untouched():
    model, x, emb, pooled = _fixture()
    baseline = model(x, emb).clone()
    handle = install_modulation_guidance_reserve(model, "pooled_text", text_dim=TEXT_DIM)
    # Make the projector non-trivial (training would do this).
    torch.manual_seed(1)
    nn.init.normal_(handle.projector.out_proj.weight, std=0.5)
    nn.init.normal_(handle.projector.out_proj.bias, std=0.5)
    nn.init.normal_(handle.projector.site_embed.weight, std=0.5)

    # Gate-level check on a single wrapped modulation site.
    site = model.blocks[0].adaln_modulation_self_attn
    off = site(emb)
    with modulation_guidance_pooled_text_context(pooled):
        on = site(emb)
    assert torch.allclose(on[2], off[2]), "gate chunk must be untouched"
    assert not torch.allclose(on[0], off[0]), "shift chunk must change"
    assert not torch.allclose(on[1], off[1]), "scale chunk must change"

    with modulation_guidance_pooled_text_context(pooled):
        out = model(x, emb)
    assert torch.isfinite(out).all()
    assert not torch.allclose(out, baseline, atol=1e-5)
    print("PASS: trained projector steers output, finite, gate untouched")


def test_distill_loss_off_is_base_on_adds_term_with_projector_grad():
    model, x, emb, pooled = _fixture()
    handle = install_modulation_guidance_reserve(model, "pooled_text", text_dim=TEXT_DIM)
    nn.init.normal_(handle.projector.out_proj.weight, std=0.3)

    with modulation_guidance_pooled_text_context(pooled):
        student = model(x, emb)
    teacher = student.detach() + 0.25
    base = student.new_tensor(0.5)

    off = compose_modulation_guidance_distill_loss(student, teacher, enabled=False, base_loss=base)
    assert off["applied"] is False
    assert torch.allclose(off["total"], base)

    on = compose_modulation_guidance_distill_loss(
        student, teacher, enabled=True, base_loss=base,
        projector=handle.projector, distill_weight=1.0, reg_weight=1e-3,
    )
    assert on["applied"] is True
    assert not torch.allclose(on["total"], base)
    on["total"].backward()
    grads = [p.grad for p in handle.projector.parameters() if p.grad is not None]
    assert grads and any(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    print("PASS: distill off=base, on adds term and projector receives gradient")


def test_replay_request_round_trips():
    req = build_modulation_guidance_replay_request(
        {"method": "pooled_text", "text_dim": TEXT_DIM, "target_sites": 9, "distill_weight": 0.7}
    )
    again = build_modulation_guidance_replay_request(req)
    assert again == req
    meta = req.as_metadata()
    assert meta["ss_feature_type"] == "modulation_guidance"
    assert meta["ss_modulation_guidance_method"] == "pooled_text"
    assert meta["ss_modulation_guidance_pooled_text_projection_contract"] == "true"
    assert meta["ss_training_path_enabled"] == "false"
    print("PASS: replay request round-trips and stamps metadata")


def test_readiness_is_honest_and_default_reconciliation_stays_partial():
    rep = modulation_guidance_reserve_readiness()
    assert rep["wired"] is True
    assert rep["full_route_ready"] is True
    for flag in (
        "runtime_activation_enabled", "request_fields_emitted", "trainer_wiring_allowed",
        "training_path_enabled", "default_behavior_changed", "promotion_ready",
    ):
        assert rep[flag] is False, f"{flag} must stay False"
    # The default reconciliation (no seam) is unchanged -> still partial.
    default_rec = build_modulation_guidance_reconciliation()
    assert default_rec["full_route_ready"] is False
    assert "pooled_text_projection_contract_missing" in default_rec["blocked_reasons"]
    print("PASS: readiness honest (full_route_ready, operator flags False); default stays partial")


def test_remove_restores_forward_bitwise():
    model, x, emb, pooled = _fixture()
    baseline = model(x, emb).clone()
    handle = install_modulation_guidance_reserve(model, "pooled_text", text_dim=TEXT_DIM)
    nn.init.normal_(handle.projector.out_proj.weight, std=0.5)
    handle.remove()
    # After remove, even with pooled text published the output is the original.
    with modulation_guidance_pooled_text_context(pooled):
        out = model(x, emb)
    assert torch.equal(out, baseline)
    print("PASS: handle.remove() restores original forward bit-for-bit")


if __name__ == "__main__":
    test_method_none_is_bitwise_parity()
    test_installed_but_no_pooled_text_is_parity()
    test_zero_init_projector_is_parity_even_with_pooled_text()
    test_trained_projector_steers_output_and_leaves_gate_untouched()
    test_distill_loss_off_is_base_on_adds_term_with_projector_grad()
    test_replay_request_round_trips()
    test_readiness_is_honest_and_default_reconciliation_stays_partial()
    test_remove_restores_forward_bitwise()
    print("\n8/8 modulation_guidance_reserve smoke checks passed!")
