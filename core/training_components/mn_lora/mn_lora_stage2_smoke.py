"""Smoke tests for MN-LoRA subspace preconditioning."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

BACKEND_ROOT = Path(__file__).resolve().parents[3]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.training_components.mn_lora.gradient_subspace import GradientSubspaceProjection


def _seed_identity_subspace(gsp: GradientSubspaceProjection, layer: str = "layer") -> None:
    gsp.V_cache[layer] = torch.eye(2)
    gsp.S_cache[layer] = torch.tensor([10.0, 1.0])


def test_precondition_none_preserves_projection() -> None:
    gsp = GradientSubspaceProjection(
        precondition_mode="none",
        enable_residual_monitor=False,
        precision_guard=True,
    )
    _seed_identity_subspace(gsp)
    grad = torch.tensor([[1.0, 4.0]])
    projected = gsp.project_gradient("layer", grad)
    assert torch.allclose(projected, grad), "identity GSP without preconditioning should preserve gradient"


def test_svd_preconditioner_uses_cached_spectrum() -> None:
    gsp = GradientSubspaceProjection(
        precondition_mode="svd",
        svd_precond_beta=1.0,
        precond_min_scale=0.01,
        precond_max_scale=100.0,
        precond_clip=100.0,
        enable_residual_monitor=False,
        precision_guard=True,
    )
    _seed_identity_subspace(gsp)
    grad = torch.tensor([[1.0, 1.0]])
    projected = gsp.project_gradient("layer", grad)
    assert projected[0, 0] < grad[0, 0], "large singular direction should be damped"
    assert projected[0, 1] > grad[0, 1], "small singular direction should be amplified"


def test_grad_ema_preconditioner_normalizes_subspace_energy() -> None:
    gsp = GradientSubspaceProjection(
        precondition_mode="grad_ema",
        precond_min_scale=0.01,
        precond_max_scale=100.0,
        precond_clip=100.0,
        precond_eps=1e-8,
        enable_residual_monitor=False,
        precision_guard=True,
    )
    _seed_identity_subspace(gsp)
    grad = torch.tensor([[1.0, 4.0]])
    projected = gsp.project_gradient("layer", grad)
    assert "layer" in gsp.coord_curv_cache
    assert torch.allclose(projected[0, 0], projected[0, 1], atol=1e-4), (
        "grad_ema preconditioner should equalize one-sample subspace coordinate energy"
    )
    telemetry = gsp.get_telemetry_snapshot()
    assert telemetry["mode"] == "grad_ema"
    assert telemetry["coord_curv_layers"] == 1
    assert telemetry["precondition_calls"] == 1
    assert float(telemetry["precondition_norm_ratio_avg"]) > 0.0


def test_hybrid_preconditioner_remains_clipped() -> None:
    gsp = GradientSubspaceProjection(
        precondition_mode="hybrid",
        svd_precond_beta=1.0,
        precond_min_scale=0.01,
        precond_max_scale=100.0,
        precond_clip=1.25,
        enable_residual_monitor=False,
        precision_guard=True,
    )
    _seed_identity_subspace(gsp)
    grad = torch.tensor([[1.0, 4.0]])
    projected = gsp.project_gradient("layer", grad)
    assert projected.norm() <= grad.norm() * 1.2501
    telemetry = gsp.get_telemetry_snapshot()
    assert telemetry["precondition_clip_count"] >= 0
    assert float(telemetry["precondition_clip_rate"]) >= 0.0


def test_zero_norm_precondition_telemetry_is_separate() -> None:
    gsp = GradientSubspaceProjection(
        precondition_mode="grad_ema",
        precond_eps=1e-8,
        enable_residual_monitor=False,
        precision_guard=True,
    )
    _seed_identity_subspace(gsp)
    projected = gsp.project_gradient("layer", torch.zeros(1, 2))
    assert torch.count_nonzero(projected) == 0
    telemetry = gsp.get_telemetry_snapshot()
    assert telemetry["precondition_calls"] == 1
    assert telemetry["precondition_ratio_count"] == 0
    assert telemetry["precondition_zero_norm_calls"] == 1
    assert float(telemetry["precondition_norm_ratio_min"]) == 1.0


def test_adaptive_sparse_skips_cold_layers() -> None:
    gsp = GradientSubspaceProjection(
        precondition_mode="grad_ema",
        adaptive_sparse_enabled=True,
        adaptive_sparse_warmup_steps=0,
        adaptive_sparse_refresh_interval=1,
        adaptive_sparse_hot_ratio=0.0,
        adaptive_sparse_warm_ratio=0.0,
        adaptive_sparse_min_hot_layers=0,
        adaptive_sparse_cold_interval=0,
        enable_residual_monitor=False,
        precision_guard=True,
    )
    _seed_identity_subspace(gsp, "cold")
    gsp.step(1)
    grad = torch.tensor([[2.0, 5.0]])
    projected = gsp.project_gradient("cold", grad)
    assert torch.allclose(projected, grad), "cold tier should skip projection when cold interval is disabled"
    telemetry = gsp.get_telemetry_snapshot()
    assert telemetry["adaptive_sparse_enabled"] is True
    assert telemetry["adaptive_sparse_skip_count"] == 1
    assert telemetry["precondition_calls"] == 0


def main() -> int:
    test_precondition_none_preserves_projection()
    print("  precondition=none projection -- PASS")
    test_svd_preconditioner_uses_cached_spectrum()
    print("  SVD spectrum preconditioner -- PASS")
    test_grad_ema_preconditioner_normalizes_subspace_energy()
    print("  grad-EMA subspace preconditioner -- PASS")
    test_hybrid_preconditioner_remains_clipped()
    print("  hybrid precondition clipping -- PASS")
    test_zero_norm_precondition_telemetry_is_separate()
    print("  zero-norm telemetry separation -- PASS")
    test_adaptive_sparse_skips_cold_layers()
    print("  adaptive sparse cold-layer skip -- PASS")
    print("MN-LoRA stage-2 smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
