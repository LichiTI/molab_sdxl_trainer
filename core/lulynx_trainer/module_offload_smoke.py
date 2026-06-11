"""Smoke tests for Warehouse module_offload config, planning, and runtime guards."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn

TRAINER_ROOT = Path(__file__).resolve().parent
CORE_ROOT = TRAINER_ROOT.parent
BACKEND_ROOT = CORE_ROOT.parent
for _path in (BACKEND_ROOT, CORE_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    package = type(sys)(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package


def _import_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_ensure_package("core", CORE_ROOT)
_ensure_package("core.lulynx_trainer", TRAINER_ROOT)
_ensure_package("core.warehouse", CORE_ROOT / "warehouse")
_ensure_package("core.warehouse.training_features", CORE_ROOT / "warehouse" / "training_features")

_constants = _import_from_file("core.constants", CORE_ROOT / "constants.py")
_configs = _import_from_file("core.configs", CORE_ROOT / "configs.py")
_facade_config = _import_from_file("core.lulynx_trainer.config", TRAINER_ROOT / "config.py")
_module_contract = _import_from_file(
    "core.lulynx_trainer.module_offload_contract",
    TRAINER_ROOT / "module_offload_contract.py",
)
_module_offload = _import_from_file("core.lulynx_trainer.module_offload", TRAINER_ROOT / "module_offload.py")
_config_adapter = _import_from_file("core.lulynx_trainer.config_adapter", TRAINER_ROOT / "config_adapter.py")
_training_checks = _import_from_file(
    "core.warehouse.training_features.training_config_checks",
    CORE_ROOT / "warehouse" / "training_features" / "training_config_checks.py",
)
_training_loop = _import_from_file("core.lulynx_trainer.training_loop", TRAINER_ROOT / "training_loop.py")

TrainingPreflightConfig = _training_checks.TrainingPreflightConfig
run_training_config_checks = _training_checks.run_training_config_checks
UnifiedTrainingConfig = _configs.UnifiedTrainingConfig
ConfigAdapter = _config_adapter.ConfigAdapter
ModuleResidencyManager = _module_offload.ModuleResidencyManager
build_module_offload_plan = _module_offload.build_module_offload_plan
resolve_module_offload_config = _module_contract.resolve_module_offload_config
TrainingLoop = _training_loop.TrainingLoop


class _TieBreakBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj_b = nn.Linear(8, 8, bias=False)
        self.proj_a = nn.Linear(8, 8, bias=False)
        self.conv = nn.Conv2d(4, 4, 1, bias=False)
        for param in self.parameters():
            param.requires_grad_(False)


class _TinyTextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc_large = nn.Linear(8, 8, bias=False)
        self.fc_small = nn.Linear(4, 4, bias=False)
        for param in self.parameters():
            param.requires_grad_(False)


class _TinyModuleOffloadModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Linear(4, 4)
        self.trainable = nn.Linear(4, 1)
        for param in self.frozen.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.trainable(torch.relu(self.frozen(x)))


class _TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class _TinyUNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down_blocks = nn.ModuleList([_TinyBlock(), _TinyBlock()])
        self.mid_block = _TinyBlock()
        self.up_blocks = nn.ModuleList([_TinyBlock(), _TinyBlock()])


class _TinyVAE(nn.Module):
    config = SimpleNamespace(scaling_factor=1.0)


class _TinyScheduler:
    config = SimpleNamespace(num_train_timesteps=1000)
    alphas_cumprod = torch.linspace(1.0, 0.0, 1000)


def test_config_defaults_and_normalization() -> None:
    config = UnifiedTrainingConfig()
    assert config.module_offload_enabled is False
    assert config.module_offload_ratio == 0
    assert config.module_offload_backbone_ratio is None
    assert config.module_offload_text_encoder_ratio is None
    assert config.module_offload_profile == "custom"
    assert config.module_offload_profile_enabled is False

    parsed = ConfigAdapter.from_frontend_dict(
        {
            "module_offload_enabled": "true",
            "module_offload_ratio": "120",
            "module_offload_backbone_ratio": "",
            "module_offload_text_encoder_ratio": "-5",
            "module_offload_profile_enabled": "true",
            "module_offload_profile": "balanced",
            "module_offload_min_param_mb": "0.001",
            "module_offload_include_patterns": "proj",
            "module_offload_exclude_patterns": "proj_b",
            "module_offload_prefetch_enabled": "true",
        }
    )
    assert parsed.module_offload_enabled is True
    assert parsed.module_offload_ratio == 100
    assert parsed.module_offload_backbone_ratio is None
    assert parsed.module_offload_text_encoder_ratio == 0
    assert parsed.module_offload_profile_enabled is True
    assert parsed.module_offload_profile == "balanced"
    assert parsed.module_offload_min_param_mb == 0.001
    assert parsed.module_offload_include_patterns == "proj"
    assert parsed.module_offload_exclude_patterns == "proj_b"
    assert parsed.module_offload_prefetch_enabled is True

    zero_view = resolve_module_offload_config(
        {
            "module_offload_enabled": True,
            "module_offload_ratio": 0,
        }
    )
    assert zero_view.requested is False
    profile_view = resolve_module_offload_config(
        {
            "module_offload_enabled": True,
            "module_offload_profile_enabled": True,
            "module_offload_profile": "aggressive",
            "module_offload_backbone_ratio": 20,
        }
    )
    assert profile_view.effective_backbone_ratio == 20
    assert profile_view.effective_text_encoder_ratio == 50
    print("  [PASS] module_offload config defaults + normalization")


def test_planner_order_and_scope_overrides() -> None:
    backbone = _TieBreakBackbone()
    te1 = _TinyTextEncoder()
    te2 = _TinyTextEncoder()
    plan = build_module_offload_plan(
        backbone,
        te1,
        te2,
        {
            "module_offload_enabled": True,
            "module_offload_ratio": 50,
            "module_offload_backbone_ratio": 50,
            "module_offload_text_encoder_ratio": 50,
        },
    )
    backbone_scope = plan.scopes["backbone"]
    te1_scope = plan.scopes["text_encoder_1"]
    te2_scope = plan.scopes["text_encoder_2"]
    assert backbone_scope.selected_count == 2
    assert backbone_scope.selected_paths[:2] == ["proj_a", "proj_b"]
    assert te1_scope.selected_count == 1
    assert te2_scope.selected_count == 1
    print("  [PASS] module_offload planner ordering + per-scope selection")


def test_planner_filters_and_summary() -> None:
    backbone = _TieBreakBackbone()
    plan = build_module_offload_plan(
        backbone,
        None,
        None,
        {
            "module_offload_enabled": True,
            "module_offload_profile_enabled": True,
            "module_offload_profile": "balanced",
            "module_offload_min_param_mb": 0.0001,
            "module_offload_include_patterns": "proj*",
            "module_offload_exclude_patterns": "proj_b",
            "module_offload_prefetch_enabled": True,
        },
    )
    state = plan.as_dict()
    assert plan.config.effective_backbone_ratio == 50
    assert plan.config.effective_text_encoder_ratio == 25
    assert plan.scopes["backbone"].selected_paths == ["proj_a"]
    assert state["profile"] == "balanced"
    assert state["min_param_mb"] == 0.0001
    assert state["include_patterns"] == "proj*"
    assert state["exclude_patterns"] == "proj_b"
    assert state["estimated_transfer_mb"] > 0
    print("  [PASS] module_offload planner filters + summary")


def test_manager_forward_backward_on_cpu() -> None:
    model = _TinyModuleOffloadModel()
    plan = build_module_offload_plan(
        model,
        None,
        None,
        {
            "module_offload_enabled": True,
            "module_offload_ratio": 100,
        },
    )
    assert plan.enabled is True
    manager = ModuleResidencyManager(plan, device="cpu")
    x = torch.randn(2, 4, requires_grad=True)
    with manager.step_context():
        loss = model(x).sum()
        loss.backward()
    stats = manager.stats_dict()
    assert stats["materialize_count"] > 0
    assert stats["materialize_bytes"] > 0
    assert stats["top_modules"]
    assert x.grad is not None
    assert model.trainable.weight.grad is not None
    assert model.frozen.weight.device.type == "cpu"
    manager.close()
    print("  [PASS] module_offload manager forward + backward on CPU")


def test_prefetch_cpu_safe_degrade() -> None:
    model = _TinyModuleOffloadModel()
    plan = build_module_offload_plan(
        model,
        None,
        None,
        {
            "module_offload_enabled": True,
            "module_offload_ratio": 100,
            "module_offload_prefetch_enabled": True,
        },
    )
    manager = ModuleResidencyManager(plan, device="cpu")
    with manager.step_context():
        _ = model(torch.randn(2, 4)).sum()
    stats = manager.stats_dict()
    assert stats["prefetch_requested"] is True
    assert stats["prefetch_enabled"] is False
    assert "CUDA" in stats["prefetch_degraded_reason"]
    manager.close()
    print("  [PASS] module_offload prefetch CPU safe degrade")


def test_preflight_conflicts() -> None:
    report = run_training_config_checks(
        TrainingPreflightConfig(
            schema_id="sdxl-lora",
            training_type="lora",
            config={
                "schema_id": "sdxl-lora",
                "training_type": "lora",
                "module_offload_enabled": True,
                "module_offload_ratio": 50,
                "torch_compile": True,
                "gradient_checkpointing": True,
            },
        )
    )
    assert any("torch_compile" in error for error in report.errors)
    assert any("gradient_checkpointing" in error for error in report.errors)
    print("  [PASS] module_offload preflight conflicts")


def test_training_loop_cpu_guard() -> None:
    unet = _TinyUNet()
    optimizer = torch.optim.SGD(unet.parameters(), lr=0.01)
    loop = TrainingLoop(
        unet=unet,
        text_encoder_1=_TinyTextEncoder(),
        text_encoder_2=None,
        vae=_TinyVAE(),
        tokenizer_1=None,
        tokenizer_2=None,
        noise_scheduler=_TinyScheduler(),
        lora_injector=None,
        optimizer=optimizer,
        lr_scheduler=None,
        device="cpu",
        dtype=torch.float32,
        module_offload_enabled=True,
        module_offload_ratio=50,
    )
    assert getattr(loop, "_module_offload_manager", None) is None
    assert loop.memory_optimization_state.get("mode") == "none"
    assert "single CUDA GPU" in str(loop.memory_optimization_state.get("reason", ""))
    print("  [PASS] module_offload runtime CPU guard")


def main() -> int:
    tests = [
        test_config_defaults_and_normalization,
        test_planner_order_and_scope_overrides,
        test_planner_filters_and_summary,
        test_manager_forward_backward_on_cpu,
        test_prefetch_cpu_safe_degrade,
        test_preflight_conflicts,
        test_training_loop_cpu_guard,
    ]
    results = []
    for test_fn in tests:
        try:
            test_fn()
            results.append((test_fn.__name__, True))
        except Exception as exc:
            import traceback

            traceback.print_exc()
            print(f"FAIL: {test_fn.__name__} - {exc}")
            results.append((test_fn.__name__, False))
    passed = sum(1 for _, ok in results if ok)
    print("\n" + "=" * 60)
    print("Module Offload Smoke Test Results")
    print("=" * 60)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")
    print(f"\n{passed}/{len(results)} tests passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

