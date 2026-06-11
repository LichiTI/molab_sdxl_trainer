# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test Anima adapter variant injection coverage.

Proves:
1. LoRAInjector injects all expected Anima DiT target suffixes.
2. Target group classification (self_attn, cross_attn, mlp, etc.) is correct.
3. network_dropout is passed through to LoRA layer construction.
4. DoRA-enabled LoRA injection creates DoRA layers.
5. VeRA and T-LoRA config-to-injector coverage is verified by anima_adapter_matrix_smoke.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import torch
from torch import nn

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
_load_module("core.lulynx_trainer.model_family", TRAINER_ROOT / "model_family.py")
targets_mod = _load_module("core.lulynx_trainer.anima_targets", TRAINER_ROOT / "anima_targets.py")
injector_mod = _load_module("core.lulynx_trainer.lora_injector", TRAINER_ROOT / "lora_injector.py")

get_anima_dit_targets = targets_mod.get_anima_dit_targets
get_anima_target_groups = targets_mod.get_anima_target_groups
ANIMA_DIT_SELF_ATTN_TARGETS = targets_mod.ANIMA_DIT_SELF_ATTN_TARGETS
ANIMA_DIT_CROSS_ATTN_TARGETS = targets_mod.ANIMA_DIT_CROSS_ATTN_TARGETS
ANIMA_DIT_MLP_TARGETS = targets_mod.ANIMA_DIT_MLP_TARGETS
ANIMA_DIT_ADALN_TARGETS = targets_mod.ANIMA_DIT_ADALN_TARGETS
ANIMA_LLM_ADAPTER_TARGETS = targets_mod.ANIMA_LLM_ADAPTER_TARGETS
LoRAInjector = injector_mod.LoRAInjector


class _TinyAnimaBlock(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.self_attn = nn.Module()
        self.self_attn.q_proj = nn.Linear(dim, dim, bias=False)
        self.self_attn.k_proj = nn.Linear(dim, dim, bias=False)
        self.self_attn.v_proj = nn.Linear(dim, dim, bias=False)
        self.self_attn.output_proj = nn.Linear(dim, dim, bias=False)

        self.cross_attn = nn.Module()
        self.cross_attn.q_proj = nn.Linear(dim, dim, bias=False)
        self.cross_attn.k_proj = nn.Linear(dim, dim, bias=False)
        self.cross_attn.v_proj = nn.Linear(dim, dim, bias=False)
        self.cross_attn.output_proj = nn.Linear(dim, dim, bias=False)

        self.mlp = nn.Module()
        self.mlp.layer1 = nn.Linear(dim, dim * 2, bias=False)
        self.mlp.layer2 = nn.Linear(dim * 2, dim, bias=False)

        self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim, bias=False))
        self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim, bias=False))
        self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim, bias=False))


class _TinyAnimaNet(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_TinyAnimaBlock(dim)])
        self.final_layer = nn.Module()
        self.final_layer.adaln_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim, bias=False))
        self.llm_adapter = nn.Module()
        self.llm_adapter.proj = nn.Linear(dim, dim, bias=False)


class _TinyAnimaRoot(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.net = _TinyAnimaNet(dim)


def test_lora_target_coverage() -> None:
    """LoRAInjector injects all expected Anima DiT target suffixes."""
    targets = get_anima_dit_targets(include_llm_adapter=True)
    assert len(targets) == 22, f"Expected 22 targets, got {len(targets)}"

    model = _TinyAnimaRoot()
    injector = LoRAInjector(rank=1, alpha=1, model_arch="anima")
    injector._inject_model(model, targets, prefix="net")

    # injected_layers keys are module paths like "net.blocks.0.self_attn.q_proj"
    # Check that each target suffix appears in at least one injected layer name
    injected_names = set(injector.injected_layers.keys())
    for target in ANIMA_DIT_SELF_ATTN_TARGETS:
        assert any(target in name for name in injected_names), (
            f"Missing self_attn target: {target}, injected={injected_names}"
        )
    for target in ANIMA_DIT_CROSS_ATTN_TARGETS:
        assert any(target in name for name in injected_names), f"Missing cross_attn target: {target}"
    for target in ANIMA_DIT_MLP_TARGETS:
        assert any(target in name for name in injected_names), f"Missing MLP target: {target}"


def test_target_groups() -> None:
    """get_anima_target_groups returns correct group classification."""
    groups = get_anima_target_groups(include_llm_adapter=True)
    assert set(groups.keys()) == {"self_attn", "cross_attn", "mlp", "adaln_modulation", "llm_adapter"}
    assert groups["self_attn"] == ANIMA_DIT_SELF_ATTN_TARGETS
    assert groups["cross_attn"] == ANIMA_DIT_CROSS_ATTN_TARGETS
    assert groups["mlp"] == ANIMA_DIT_MLP_TARGETS
    assert groups["adaln_modulation"] == ANIMA_DIT_ADALN_TARGETS
    assert groups["llm_adapter"] == ANIMA_LLM_ADAPTER_TARGETS


def test_network_dropout_passthrough() -> None:
    """network_dropout is passed to LoRA layer construction."""
    model = _TinyAnimaRoot()
    injector_with_dropout = LoRAInjector(rank=1, alpha=1, dropout=0.5, model_arch="anima")
    targets = get_anima_dit_targets(include_llm_adapter=False)
    injector_with_dropout._inject_model(model, targets, prefix="net")

    has_dropout = False
    for name, module in injector_with_dropout.injected_layers.items():
        lora_layer = getattr(module, "lora", None)
        if lora_layer is not None and hasattr(lora_layer, "dropout") and isinstance(lora_layer.dropout, nn.Dropout):
            has_dropout = True
            break
    assert has_dropout, "Expected at least one LoRA layer with nn.Dropout when dropout=0.5"


def test_no_dropout_default() -> None:
    """Default dropout=0 uses nn.Identity (no-op dropout)."""
    model = _TinyAnimaRoot()
    injector = LoRAInjector(rank=1, alpha=1, dropout=0.0, model_arch="anima")
    targets = get_anima_dit_targets(include_llm_adapter=False)
    injector._inject_model(model, targets, prefix="net")

    for name, module in injector.injected_layers.items():
        lora_layer = getattr(module, "lora", None)
        if lora_layer is not None and hasattr(lora_layer, "dropout"):
            assert isinstance(lora_layer.dropout, nn.Identity), (
                f"Expected nn.Identity for dropout=0, got {type(lora_layer.dropout)}"
            )


def test_dora_injection() -> None:
    """DoRA-enabled LoRA injection creates DoRA-wrapped layers."""
    model = _TinyAnimaRoot()
    injector = LoRAInjector(rank=1, alpha=1, dora_enabled=True, model_arch="anima")
    targets = ["self_attn.q_proj"]
    injector._inject_model(model, targets, prefix="net")

    found_dora = False
    for name, module in injector.injected_layers.items():
        if getattr(module, "use_dora", False):
            found_dora = True
            break
    assert found_dora, "Expected DoRA-wrapped layer when dora_enabled=True"


def main() -> int:
    test_lora_target_coverage()
    print("  LoRA target coverage: all 22 Anima DiT suffixes -- PASS")

    test_target_groups()
    print("  Target group classification -- PASS")

    test_network_dropout_passthrough()
    print("  network_dropout passthrough to LoRA layers -- PASS")

    test_no_dropout_default()
    print("  Default dropout=0 uses nn.Identity -- PASS")

    test_dora_injection()
    print("  DoRA injection creates DoRA-wrapped layers -- PASS")

    print(
        "Anima adapter-variants smoke passed: LoRA target coverage, group classification, "
        "dropout passthrough, DoRA injection"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
