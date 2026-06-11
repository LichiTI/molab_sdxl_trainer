# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke tests for module_offload_enhanced toggle."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = BACKEND_ROOT / "core"
TRAINER_ROOT = CORE_ROOT / "lulynx_trainer"


def _ensure_namespace(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_module(name: str, path: Path):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ensure_namespace("core", CORE_ROOT)
_ensure_namespace("core.lulynx_trainer", TRAINER_ROOT)
configs_mod = _load_module("core.configs", CORE_ROOT / "configs.py")
contract_mod = _load_module(
    "core.lulynx_trainer.module_offload_contract",
    TRAINER_ROOT / "module_offload_contract.py",
)

resolve_module_offload_config = contract_mod.resolve_module_offload_config
UnifiedTrainingConfig = configs_mod.UnifiedTrainingConfig


def test_enhanced_false_noop() -> None:
    """Default: enhanced=False changes nothing."""
    view = resolve_module_offload_config({"module_offload_enhanced": False})
    assert view.enhanced is False
    assert view.enabled is False
    assert view.profile == "custom"
    assert view.profile_enabled is False
    assert view.prefetch_enabled is False
    assert view.requested is False
    print("  PASS: test_enhanced_false_noop")


def test_enhanced_sets_aggressive() -> None:
    """enhanced=True auto-enables aggressive profile + prefetch."""
    view = resolve_module_offload_config({"module_offload_enhanced": True})
    assert view.enhanced is True
    assert view.enabled is True
    assert view.profile == "aggressive"
    assert view.profile_enabled is True
    assert view.prefetch_enabled is True
    assert view.effective_backbone_ratio == 75
    assert view.effective_text_encoder_ratio == 50
    assert view.requested is True
    print("  PASS: test_enhanced_sets_aggressive")


def test_enhanced_explicit_profile_wins() -> None:
    """Explicit profile=balanced overrides enhanced default of aggressive."""
    view = resolve_module_offload_config({
        "module_offload_enhanced": True,
        "module_offload_profile": "balanced",
        "module_offload_profile_enabled": True,
    })
    assert view.profile == "balanced", f"Expected balanced, got {view.profile}"
    assert view.effective_backbone_ratio == 50
    assert view.effective_text_encoder_ratio == 25
    print("  PASS: test_enhanced_explicit_profile_wins")


def test_enhanced_explicit_prefetch_false_wins() -> None:
    """Explicit prefetch_enabled=False overrides enhanced default."""
    view = resolve_module_offload_config({
        "module_offload_enhanced": True,
        "module_offload_prefetch_enabled": False,
    })
    assert view.prefetch_enabled is False
    print("  PASS: test_enhanced_explicit_prefetch_false_wins")


def test_config_default() -> None:
    """UnifiedTrainingConfig default has module_offload_enhanced=False."""
    config = UnifiedTrainingConfig()
    assert getattr(config, "module_offload_enhanced", None) is False
    print("  PASS: test_config_default")


def main() -> int:
    print("Offload Enhanced Smoke Tests")
    print("=" * 40)
    test_enhanced_false_noop()
    test_enhanced_sets_aggressive()
    test_enhanced_explicit_profile_wins()
    test_enhanced_explicit_prefetch_false_wins()
    test_config_default()
    print("=" * 40)
    print("All offload enhanced smoke tests passed!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
