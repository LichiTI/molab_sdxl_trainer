"""Smoke tests for MN-LoRA P3 trust-region update clipping."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

BACKEND_ROOT = Path(__file__).resolve().parents[3]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.training_components.mn_lora.mn_optimizer import MNLoRAOptimizer
from core.training_components.mn_lora.mn_lora_trust_region import MNLoRATrustRegionController
from core.training_components.mn_lora.effective_delta import MNLoRAEffectiveDeltaController
from core.training_components.mn_lora.lora_kfac_lite import LoRAKFACLiteController
from core.training_components.mn_lora.fisher_ewc import MNLoRAFisherEWCController
from core.training_components.mn_lora.gradient_conflict import MNLoRAGradientConflictController
from core.training_components.mn_lora.ghost_replay import ReferenceOutputCache, GhostReplayRegularizer


class _TinyLoRAWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.original = nn.Linear(3, 2, bias=False)
        self.lora = nn.Module()
        self.lora.rank = 2
        self.lora.alpha = 2.0
        self.lora.scaling = 1.0
        self.lora.lora_down = nn.Linear(3, 2, bias=False)
        self.lora.lora_up = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.original.weight.fill_(1.0)
            self.lora.lora_down.weight.fill_(1.0)
            self.lora.lora_up.weight.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + self.lora.lora_up(self.lora.lora_down(x)) * self.lora.scaling


class _FastPathLoRA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.rank = 2
        self.alpha = 2.0
        self.scaling = 1.0
        self.dropout = nn.Identity()
        self.lora_down = nn.Linear(3, 2, bias=False)
        self.lora_up = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.lora_down.weight.fill_(1.0)
            self.lora_up.weight.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.linear(x, self.lora_down.weight)
        return F.linear(hidden, self.lora_up.weight) * self.scaling


class _FastPathLoRAWrapper(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.original = nn.Linear(3, 2, bias=False)
        self.lora = _FastPathLoRA()
        with torch.no_grad():
            self.original.weight.fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.original(x) + self.lora(x)


class _SelectiveReplayModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([2.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if bool((x < 0).any()):
            raise RuntimeError("intentional replay failure")
        return x * self.weight


def test_trust_region_clips_large_update() -> None:
    param = nn.Parameter(torch.ones(4, 4))
    old = param.detach().clone()
    param.data.add_(10.0)

    controller = MNLoRATrustRegionController(
        enabled=True,
        max_update_rms_ratio=0.10,
        max_update_norm_ratio=0.10,
        param_names={id(param): "block.lora_down.weight"},
    )
    controller.apply(param, old)

    delta_rms = (param.detach() - old).norm() / (param.numel() ** 0.5)
    base_rms = old.norm() / (old.numel() ** 0.5)
    assert float(delta_rms) <= float(base_rms) * 0.1001

    telemetry = controller.get_telemetry_snapshot()
    assert telemetry["calls"] == 1
    assert telemetry["clipped"] == 1
    assert float(telemetry["scale_min"]) < 1.0


def test_optimizer_trust_region_telemetry_and_state() -> None:
    param = nn.Parameter(torch.ones(2, 2))
    optimizer = MNLoRAOptimizer(
        torch.optim.SGD([param], lr=1.0),
        enable_tgwd=False,
        enable_gsp=False,
        enable_pilot=False,
        trust_region_config={
            "enabled": True,
            "max_update_rms_ratio": 0.05,
            "max_update_norm_ratio": 0.05,
        },
        param_names={id(param): "block.lora_up.weight"},
    )

    before = param.detach().clone()
    param.grad = torch.ones_like(param)
    optimizer.step()
    delta_rms = (param.detach() - before).norm() / (param.numel() ** 0.5)
    base_rms = before.norm() / (before.numel() ** 0.5)
    assert float(delta_rms) <= float(base_rms) * 0.0501

    telemetry = optimizer.get_telemetry_snapshot()
    assert telemetry["trust_region_enabled"] is True
    assert telemetry["trust_region"]["clipped"] == 1

    state = optimizer.state_dict()
    assert "mn_lora_trust_region" in state


def test_zero_base_tensor_is_not_over_clipped() -> None:
    param = nn.Parameter(torch.zeros(2, 2))
    optimizer = MNLoRAOptimizer(
        torch.optim.SGD([param], lr=1e-3),
        enable_tgwd=False,
        enable_gsp=False,
        enable_pilot=False,
        trust_region_config={
            "enabled": True,
            "max_update_rms_ratio": 0.01,
            "max_update_norm_ratio": 0.10,
        },
        param_names={id(param): "block.lora_up.weight"},
    )

    param.grad = torch.ones_like(param)
    optimizer.step()
    assert torch.allclose(param.detach(), torch.full_like(param, -1e-3))
    telemetry = optimizer.get_telemetry_snapshot()["trust_region"]
    assert telemetry["calls"] == 1
    assert telemetry["clipped"] == 0


def test_effective_delta_clips_paired_lora_weight() -> None:
    module = _TinyLoRAWrapper()
    controller = MNLoRAEffectiveDeltaController(
        enabled=True,
        max_norm_ratio=0.10,
        max_rms_ratio=0.10,
        modules={"tiny": module},
    )
    before = module.lora.lora_up.weight.detach().clone()
    controller.apply_all()
    after = module.lora.lora_up.weight.detach()
    assert after.norm() < before.norm(), "effective ΔW clip should scale the B/up matrix"
    telemetry = controller.get_telemetry_snapshot()
    assert telemetry["registered_pairs"] == 1
    assert telemetry["pairs_seen"] == 1
    assert telemetry["pairs_clipped"] == 1
    assert telemetry["norm_ratio_max"] > telemetry["max_norm_ratio"]


def test_optimizer_effective_delta_telemetry_and_state() -> None:
    module = _TinyLoRAWrapper()
    params = list(module.lora.lora_down.parameters()) + list(module.lora.lora_up.parameters())
    optimizer = MNLoRAOptimizer(
        torch.optim.SGD(params, lr=0.1),
        enable_tgwd=False,
        enable_gsp=False,
        enable_pilot=False,
        effective_delta_config={
            "enabled": True,
            "max_norm_ratio": 0.10,
            "max_rms_ratio": 0.10,
        },
        lora_modules={"tiny": module},
    )
    for param in params:
        param.grad = torch.ones_like(param)
    optimizer.step()
    telemetry = optimizer.get_telemetry_snapshot()
    assert telemetry["effective_delta_enabled"] is True
    assert telemetry["effective_delta"]["registered_pairs"] == 1
    assert telemetry["effective_delta"]["pairs_seen"] == 1
    assert "mn_lora_effective_delta" in optimizer.state_dict()


def test_fisher_weighted_effective_delta_tracks_gradient_importance() -> None:
    module = _TinyLoRAWrapper()
    module.lora.lora_down.weight.grad = torch.full_like(module.lora.lora_down.weight, 2.0)
    module.lora.lora_up.weight.grad = torch.full_like(module.lora.lora_up.weight, 2.0)

    controller = MNLoRAEffectiveDeltaController(
        enabled=True,
        clip_enabled=False,
        fisher_weighted=True,
        fisher_strength=1.0,
        fisher_max_weight=8.0,
        modules={"tiny": module},
    )
    controller.apply_all()
    telemetry = controller.get_telemetry_snapshot()
    assert telemetry["fisher_weighted"] is True
    assert telemetry["fisher_pairs"] == 1
    assert telemetry["fisher_updates"] == 1
    assert telemetry["fisher_weight_avg"] > 1.0
    state = controller.state_dict()
    assert "tiny" in state["fisher_ema"]


def test_lora_kfac_lite_preconditions_paired_lora_grads() -> None:
    module = _TinyLoRAWrapper()
    controller = LoRAKFACLiteController(
        enabled=True,
        modules={"tiny": module},
        damping=1e-2,
        max_samples=16,
        grad_clip=3.0,
        active_ratio=1.0,
    )
    x = torch.randn(5, 3, requires_grad=True)
    loss = module(x).pow(2).mean()
    loss.backward()

    before_down = module.lora.lora_down.weight.grad.detach().clone()
    before_up = module.lora.lora_up.weight.grad.detach().clone()
    controller.pre_step(1)
    after_down = module.lora.lora_down.weight.grad.detach()
    after_up = module.lora.lora_up.weight.grad.detach()

    assert not torch.allclose(before_down, after_down)
    assert not torch.allclose(before_up, after_up)
    telemetry = controller.get_telemetry_snapshot()
    assert telemetry["registered_modules"] == 1
    assert telemetry["factor_layers"] == 1
    assert telemetry["updates"] == 1
    assert telemetry["preconditioned"] == 1
    assert telemetry["preconditioned_params"] == 2
    assert telemetry["last_updates"] == 1
    assert telemetry["last_preconditioned"] == 1
    assert telemetry["last_preconditioned_params"] == 2
    assert telemetry["precondition_hit_rate"] == 1.0
    assert telemetry["grad_norm_ratio_min"] > 0.0
    assert "factors" in controller.state_dict()
    controller.close()


def test_lora_kfac_lite_skip_reasons_for_unsupported_modules() -> None:
    unsupported = nn.Linear(3, 2, bias=False)
    controller = LoRAKFACLiteController(enabled=True, modules={"unsupported": unsupported})
    telemetry = controller.get_telemetry_snapshot()
    assert telemetry["registered_modules"] == 0
    assert telemetry["skipped"] == 1
    assert telemetry["skip_reasons"]["register_missing_lora_pair"] == 1
    controller.close()


def test_lora_kfac_lite_captures_fast_path_lora() -> None:
    module = _FastPathLoRAWrapper()
    controller = LoRAKFACLiteController(
        enabled=True,
        modules={"fast": module},
        damping=1e-2,
        max_samples=16,
        grad_clip=3.0,
        active_ratio=1.0,
    )
    x = torch.randn(5, 3, requires_grad=True)
    loss = module(x).pow(2).mean()
    loss.backward()
    controller.pre_step(1)
    telemetry = controller.get_telemetry_snapshot()
    assert telemetry["registered_modules"] == 1
    assert telemetry["factor_layers"] == 1
    assert telemetry["updates"] == 1
    assert telemetry["preconditioned"] == 1
    controller.close()


def test_optimizer_lora_kfac_lite_telemetry_and_state() -> None:
    module = _TinyLoRAWrapper()
    params = list(module.lora.lora_down.parameters()) + list(module.lora.lora_up.parameters())
    optimizer = MNLoRAOptimizer(
        torch.optim.SGD(params, lr=0.01),
        enable_tgwd=False,
        enable_gsp=False,
        enable_pilot=False,
        kfac_lite_config={"enabled": True, "damping": 1e-2, "max_samples": 16, "active_ratio": 1.0},
        lora_modules={"tiny": module},
    )
    x = torch.randn(4, 3)
    loss = module(x).pow(2).mean()
    loss.backward()
    optimizer.step()
    telemetry = optimizer.get_telemetry_snapshot()
    assert telemetry["kfac_lite_enabled"] is True
    assert telemetry["kfac_lite"]["registered_modules"] == 1
    assert telemetry["kfac_lite"]["preconditioned"] == 1
    assert telemetry["kfac_lite"]["stacked_with_gsp"] is False
    assert "mn_lora_kfac_lite" in optimizer.state_dict()
    optimizer.kfac_lite.close()


def test_optimizer_lora_kfac_lite_stacked_gsp_guard() -> None:
    module = _TinyLoRAWrapper()
    params = list(module.lora.lora_down.parameters()) + list(module.lora.lora_up.parameters())
    optimizer = MNLoRAOptimizer(
        torch.optim.SGD(params, lr=0.01),
        enable_tgwd=False,
        enable_gsp=True,
        enable_pilot=False,
        gsp_config={"precondition_mode": "none", "update_interval": 1, "k_ratio": 1.0},
        kfac_lite_config={
            "enabled": True,
            "damping": 1e-6,
            "max_samples": 16,
            "grad_clip": 10.0,
            "stacked_grad_clip": 1.25,
            "active_ratio": 1.0,
        },
        lora_modules={"tiny": module},
    )
    x = torch.randn(4, 3)
    loss = module(x).pow(2).mean()
    loss.backward()
    optimizer.step()
    telemetry = optimizer.get_telemetry_snapshot()["kfac_lite"]
    assert telemetry["stacked_with_gsp"] is True
    assert telemetry["stacked_grad_clip"] == 1.25
    assert telemetry["grad_norm_ratio_max"] <= 1.2501
    optimizer.kfac_lite.close()


def test_fisher_ewc_builds_penalty_after_weight_drift() -> None:
    param = nn.Parameter(torch.ones(2, 2))
    controller = MNLoRAFisherEWCController(
        enabled=True,
        lambda_ewc=0.5,
        fisher_beta=0.0,
        params=[param],
        param_names={id(param): "block.lora_up.weight"},
    )
    param.grad = torch.ones_like(param)
    first, _ = controller.build_penalty_grads([param], step=1)
    assert first["block.lora_up.weight"].abs().sum().item() == 0.0
    with torch.no_grad():
        param.add_(1.0)
    param.grad = torch.ones_like(param) * 2.0
    second, stats = controller.build_penalty_grads([param], step=2)
    assert second["block.lora_up.weight"].abs().sum().item() > 0.0
    assert stats["fisher_layers"] == 1
    telemetry = controller.get_telemetry_snapshot()
    assert telemetry["penalty_applications"] == 1


def test_optimizer_fisher_ewc_state_and_telemetry() -> None:
    param = nn.Parameter(torch.ones(2, 2))
    optimizer = MNLoRAOptimizer(
        torch.optim.SGD([param], lr=0.1),
        enable_tgwd=False,
        enable_gsp=False,
        enable_pilot=False,
        fisher_ewc_config={
            "enabled": True,
            "lambda_ewc": 0.5,
            "fisher_beta": 0.0,
            "max_penalty_norm_ratio": 1.0,
        },
        param_names={id(param): "block.lora_up.weight"},
    )
    param.grad = torch.ones_like(param)
    optimizer.step()
    param.grad = torch.ones_like(param)
    optimizer.step()
    telemetry = optimizer.get_telemetry_snapshot()
    assert telemetry["fisher_ewc_enabled"] is True
    assert telemetry["fisher_ewc"]["fisher_layers"] == 1
    assert telemetry["fisher_ewc"]["penalty_applications"] >= 1
    assert "mn_lora_fisher_ewc" in optimizer.state_dict()


def test_gradient_conflict_projects_regularizer_map() -> None:
    controller = MNLoRAGradientConflictController(enabled=True, conflict_threshold=0.0)
    main = {"w": torch.tensor([1.0, 0.0])}
    regularizer = {"w": torch.tensor([-1.0, 1.0])}
    resolved, stats = controller.resolve([main, regularizer])
    assert stats["conflict_pairs"] == 1
    assert stats["projections"] == 1
    assert resolved["w"][0].item() > 0.99
    assert resolved["w"][1].item() > 0.99


def test_optimizer_fisher_ewc_conflict_surgery() -> None:
    param = nn.Parameter(torch.ones(2))
    optimizer = MNLoRAOptimizer(
        torch.optim.SGD([param], lr=0.1),
        enable_tgwd=False,
        enable_gsp=False,
        enable_pilot=False,
        fisher_ewc_config={
            "enabled": True,
            "lambda_ewc": 1.0,
            "fisher_beta": 0.0,
            "max_penalty_norm_ratio": 10.0,
        },
        gradient_conflict_config={"enabled": True, "conflict_threshold": 0.0},
        param_names={id(param): "block.lora_up.weight"},
    )
    param.grad = torch.ones_like(param)
    optimizer.step()
    param.grad = torch.ones_like(param)
    optimizer.step()
    telemetry = optimizer.get_telemetry_snapshot()
    assert telemetry["gradient_conflict_enabled"] is True
    assert telemetry["gradient_conflict"]["conflict_pairs"] >= 1
    assert "mn_lora_gradient_conflict" in optimizer.state_dict()


def test_ghost_replay_divides_by_success_count() -> None:
    cache = ReferenceOutputCache(cache_size=4)
    cache.add_sample({"x": torch.tensor([1.0])}, torch.tensor([1.0]))
    cache.add_sample({"x": torch.tensor([-1.0])}, torch.tensor([0.0]))

    regularizer = GhostReplayRegularizer(
        cache,
        lambda_replay=1.0,
        replay_interval=1,
        max_deviation=0.0,
    )
    loss = regularizer.compute_loss(_SelectiveReplayModel(), step=1, batch_size=4, device="cpu")
    assert loss.requires_grad
    assert torch.allclose(loss.detach(), torch.tensor(1.0)), "loss should divide by successful forwards only"
    telemetry = regularizer.get_telemetry_snapshot()
    assert telemetry["success_count"] == 1
    assert telemetry["failure_count"] == 1


def test_ghost_replay_balanced_timestep_buckets() -> None:
    cache = ReferenceOutputCache(cache_size=6)
    for timestep in (10, 120, 420, 520, 800, 920):
        cache.add_sample({"x": torch.tensor([1.0])}, torch.tensor([0.0]), timestep=timestep)

    regularizer = GhostReplayRegularizer(
        cache,
        lambda_replay=1.0,
        replay_interval=1,
        max_deviation=0.0,
        timestep_strategy="balanced",
        num_train_timesteps=1000,
    )
    loss = regularizer.compute_loss(_SelectiveReplayModel(), step=1, batch_size=3, device="cpu")
    assert loss.requires_grad
    telemetry = regularizer.get_telemetry_snapshot()
    assert telemetry["timestep_strategy"] == "balanced"
    assert telemetry["success_count"] == 3
    assert telemetry["selected_timestep_buckets"] == {"low": 1, "mid": 1, "high": 1}
    assert telemetry["cache_timestep_buckets"] == {"low": 2, "mid": 2, "high": 2}


def main() -> int:
    test_trust_region_clips_large_update()
    print("  trust-region direct clipping -- PASS")
    test_optimizer_trust_region_telemetry_and_state()
    print("  optimizer trust-region telemetry/state -- PASS")
    test_zero_base_tensor_is_not_over_clipped()
    print("  zero-base tensor warm start -- PASS")
    test_effective_delta_clips_paired_lora_weight()
    print("  effective-delta paired LoRA clipping -- PASS")
    test_optimizer_effective_delta_telemetry_and_state()
    print("  optimizer effective-delta telemetry/state -- PASS")
    test_fisher_weighted_effective_delta_tracks_gradient_importance()
    print("  fisher-weighted effective-delta importance -- PASS")
    test_lora_kfac_lite_preconditions_paired_lora_grads()
    print("  LoRA-KFAC-Lite direct preconditioner -- PASS")
    test_lora_kfac_lite_skip_reasons_for_unsupported_modules()
    print("  LoRA-KFAC-Lite skip reasons -- PASS")
    test_lora_kfac_lite_captures_fast_path_lora()
    print("  LoRA-KFAC-Lite fast-path capture -- PASS")
    test_optimizer_lora_kfac_lite_telemetry_and_state()
    print("  optimizer LoRA-KFAC-Lite telemetry/state -- PASS")
    test_optimizer_lora_kfac_lite_stacked_gsp_guard()
    print("  optimizer LoRA-KFAC-Lite stacked GSP guard -- PASS")
    test_fisher_ewc_builds_penalty_after_weight_drift()
    print("  Fisher/EWC direct penalty -- PASS")
    test_optimizer_fisher_ewc_state_and_telemetry()
    print("  optimizer Fisher/EWC telemetry/state -- PASS")
    test_gradient_conflict_projects_regularizer_map()
    print("  MN-LoRA gradient conflict projection -- PASS")
    test_optimizer_fisher_ewc_conflict_surgery()
    print("  optimizer Fisher/EWC conflict surgery -- PASS")
    test_ghost_replay_divides_by_success_count()
    print("  ghost replay success-count normalization -- PASS")
    test_ghost_replay_balanced_timestep_buckets()
    print("  ghost replay timestep buckets -- PASS")
    print("MN-LoRA stage-3 smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
