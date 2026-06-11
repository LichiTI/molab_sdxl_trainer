"""Smoke-test advanced optimizer strategy resolution."""

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from backend.core.configs import UnifiedTrainingConfig
from backend.core.lulynx_trainer.config_adapter import ConfigAdapter
from backend.core.lulynx_trainer.advanced_optimizer_strategy import (
    apply_galore_runtime_outcome,
    apply_lora_plus_runtime_outcome,
    normalize_advanced_optimizer_strategy,
    resolve_advanced_optimizer_strategy,
)


def test_normalize_aliases() -> None:
    assert normalize_advanced_optimizer_strategy("LoRA+") == "lora_plus"
    assert normalize_advanced_optimizer_strategy("RSLoRA") == "rs_lora"
    assert normalize_advanced_optimizer_strategy("gradient-low-rank") == "galore"
    assert normalize_advanced_optimizer_strategy("unknown") == "auto"


def test_config_normalization() -> None:
    cfg = UnifiedTrainingConfig.from_dict({"advanced_optimizer_strategy": "LoRA+"})
    assert cfg.advanced_optimizer_strategy == "lora_plus"
    cfg = UnifiedTrainingConfig.from_dict({"advanced_optimizer_strategy": "lora"})
    assert cfg.advanced_optimizer_strategy == "lora_plus"
    frontend_cfg = ConfigAdapter.from_frontend_dict({"advancedOptimizerStrategy": "lora"})
    assert frontend_cfg.advanced_optimizer_strategy == "lora_plus"


def test_lora_plus_enables_existing_route() -> None:
    cfg = SimpleNamespace(advanced_optimizer_strategy="lora_plus", lora_plus_enabled=False, lora_plus_lr_ratio=8.0)
    profile = resolve_advanced_optimizer_strategy(cfg).as_dict()
    assert profile["resolved"] == "lora_plus"
    assert profile["active"] is True
    assert cfg.lora_plus_enabled is True
    assert profile["capabilities"]["lora_plus_param_groups"] is True
    assert profile["capabilities"]["rs_lora_adapter_scaling"] is True
    assert "set lora_plus_enabled=True" in profile["config_effects"]
    runtime_profile = apply_lora_plus_runtime_outcome(
        profile,
        applied=True,
        note="Applied LoRA+ param-group split during optimizer construction.",
    )
    assert runtime_profile["resolved"] == "lora_plus"
    assert runtime_profile["active"] is True
    assert runtime_profile["fallback_reason"] == ""
    assert "Applied LoRA+ param-group split during optimizer construction." in runtime_profile["notes"]


def test_rs_lora_enables_adapter_scaling_route() -> None:
    cfg = SimpleNamespace(
        advanced_optimizer_strategy="rs_lora",
        lora_plus_enabled=False,
        rs_lora_enabled=False,
    )
    profile = resolve_advanced_optimizer_strategy(cfg).as_dict()
    assert profile["requested"] == "rs_lora"
    assert profile["resolved"] == "rs_lora"
    assert profile["active"] is True
    assert cfg.rs_lora_enabled is True
    assert profile["fallback_reason"] == ""
    assert profile["capabilities"]["rs_lora_adapter_scaling"] is True
    assert "set rs_lora_enabled=True" in profile["config_effects"]


def test_rs_lora_preserves_existing_flag() -> None:
    cfg = SimpleNamespace(
        advanced_optimizer_strategy="rs_lora",
        lora_plus_enabled=False,
        rs_lora_enabled=True,
    )
    profile = resolve_advanced_optimizer_strategy(cfg).as_dict()
    assert profile["resolved"] == "rs_lora"
    assert profile["active"] is True
    assert cfg.rs_lora_enabled is True
    assert any(
        effect in profile["config_effects"]
        for effect in ("preserved rs_lora_enabled=True", "set rs_lora_enabled=True")
    )


def test_rs_lora_unsupported_routes_fallback_to_profile_only() -> None:
    cfg = SimpleNamespace(
        advanced_optimizer_strategy="rs_lora",
        lora_plus_enabled=False,
        rs_lora_enabled=False,
        network_module="lycoris.locon",
    )
    profile = resolve_advanced_optimizer_strategy(cfg).as_dict()
    assert profile["requested"] == "rs_lora"
    assert profile["resolved"] == "profile_only"
    assert profile["active"] is False
    assert "LyCORIS" in profile["fallback_reason"]
    assert cfg.rs_lora_enabled is False


def test_galore_enables_existing_projection_wrapper() -> None:
    cfg = SimpleNamespace(
        advanced_optimizer_strategy="galore",
        lora_plus_enabled=False,
        svd_grad_proj_rank=8,
        svd_grad_proj_update_interval=10,
    )
    profile = resolve_advanced_optimizer_strategy(cfg).as_dict()
    assert profile["requested"] == "galore"
    assert profile["resolved"] == "galore"
    assert profile["active"] is True
    assert profile["fallback_reason"] == ""
    assert cfg.svd_grad_proj_enabled is True
    assert profile["capabilities"]["galore_optimizer_projection"] is True
    assert "set svd_grad_proj_enabled=True" in profile["config_effects"]


def test_galore_runtime_skip_falls_back_to_profile_only() -> None:
    cfg = SimpleNamespace(advanced_optimizer_strategy="galore", lora_plus_enabled=False)
    profile = resolve_advanced_optimizer_strategy(cfg).as_dict()
    runtime_profile = apply_galore_runtime_outcome(
        profile,
        applied=False,
        fallback_reason="Projection wrapper disabled by runtime guard.",
        note="GaLore stayed profile-only because projection wrapping was skipped.",
    )
    assert runtime_profile["resolved"] == "profile_only"
    assert runtime_profile["active"] is False
    assert runtime_profile["fallback_reason"] == "Projection wrapper disabled by runtime guard."
    assert "GaLore stayed profile-only because projection wrapping was skipped." in runtime_profile["notes"]


def test_lora_plus_runtime_skip_falls_back_to_profile_only() -> None:
    cfg = SimpleNamespace(advanced_optimizer_strategy="lora_plus", lora_plus_enabled=False, lora_plus_lr_ratio=16.0)
    profile = resolve_advanced_optimizer_strategy(cfg).as_dict()
    runtime_profile = apply_lora_plus_runtime_outcome(
        profile,
        applied=False,
        fallback_reason="LoRA+ param-group split conflicts with pre-grouped optimizer params; keeping existing groups.",
        note="LoRA+ stayed profile-only because grouped optimizer params were already active.",
    )
    assert runtime_profile["resolved"] == "profile_only"
    assert runtime_profile["active"] is False
    assert runtime_profile["fallback_reason"] == "LoRA+ param-group split conflicts with pre-grouped optimizer params; keeping existing groups."
    assert "LoRA+ stayed profile-only because grouped optimizer params were already active." in runtime_profile["notes"]


def test_auto_does_not_change_training_by_default() -> None:
    cfg = SimpleNamespace(advanced_optimizer_strategy="auto", lora_plus_enabled=False)
    profile = resolve_advanced_optimizer_strategy(cfg).as_dict()
    assert profile["resolved"] == "off"
    assert profile["active"] is False
    assert profile["config_effects"] == []


def test_auto_preserves_existing_projection_request() -> None:
    cfg = SimpleNamespace(advanced_optimizer_strategy="auto", lora_plus_enabled=False, svd_grad_proj_enabled=True)
    profile = resolve_advanced_optimizer_strategy(cfg).as_dict()
    assert profile["resolved"] == "galore"
    assert profile["active"] is True
    assert "preserved svd_grad_proj_enabled=True" in profile["config_effects"]


def main() -> None:
    test_normalize_aliases()
    test_config_normalization()
    test_lora_plus_enables_existing_route()
    test_rs_lora_enables_adapter_scaling_route()
    test_rs_lora_preserves_existing_flag()
    test_rs_lora_unsupported_routes_fallback_to_profile_only()
    test_galore_enables_existing_projection_wrapper()
    test_galore_runtime_skip_falls_back_to_profile_only()
    test_lora_plus_runtime_skip_falls_back_to_profile_only()
    test_auto_does_not_change_training_by_default()
    test_auto_preserves_existing_projection_request()
    print("advanced_optimizer_strategy_smoke: ok")


if __name__ == "__main__":
    main()
