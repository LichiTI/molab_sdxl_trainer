"""Smoke tests for Warehouse frozen weight compression."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
import torch.nn as nn

_here = Path(__file__).resolve().parent
_root = _here.parent
_backend = _root.parent
if str(_backend) not in sys.path:
    sys.path.insert(0, str(_backend))
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _import_from_file(module_name: str, file_path: Path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


_fp8 = _import_from_file("core.lulynx_trainer.fp8_quantize", _here / "fp8_quantize.py")
_wc = _import_from_file("core.lulynx_trainer.weight_compression", _here / "weight_compression.py")
_constants = _import_from_file("core.constants", _root / "constants.py")
_configs = _import_from_file("core.configs", _root / "configs.py")
_lulynx_pkg = types.ModuleType("core.lulynx_trainer")
_lulynx_pkg.__path__ = [str(_here)]
sys.modules["core.lulynx_trainer"] = _lulynx_pkg
_import_from_file(
    "core.lulynx_trainer.module_offload_contract",
    _here / "module_offload_contract.py",
)
from core.warehouse.training_features import training_config_checks as _checks


class TinyBundle(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = nn.Sequential(nn.Linear(8, 8), nn.LayerNorm(8))
        self.text_encoder_1 = nn.Sequential(nn.Linear(8, 8), nn.Linear(8, 8))
        self.text_encoder_2 = nn.Linear(8, 8)


class FakeInjector:
    def __init__(self, params):
        self._params = list(params)
        self.injected_layers = {}

    def get_trainable_params(self):
        return self._params


def freeze(module: nn.Module) -> None:
    for param in module.parameters():
        param.requires_grad = False


def test_config_defaults() -> None:
    cfg = _configs.UnifiedTrainingConfig()
    assert cfg.weight_compression_enabled is False
    assert cfg.weight_compression_target == "none"
    assert cfg.weight_compression_format == "fp8_e4m3"
    assert "torchao_uint4" in _wc.SUPPORTED_COMPRESSION_FORMATS
    assert _wc.normalize_weight_compression_format("int4") == "torchao_uint4"
    assert _wc.normalize_weight_compression_format("quanto_qfloat8") == "quanto_float8"
    assert cfg.weight_compression_preset == "off"
    assert _wc.normalize_weight_compression_preset("safe") == "stable_backbone_int8"
    assert _wc.resolve_weight_compression_preset("text_encoder")["target"] == "text_encoder"
    assert cfg.compression_companion_type == "lora"
    assert cfg.compression_companion_mode == "merge_into_base"
    assert cfg.compression_companion_scale == 1.0
    print("  [PASS] config defaults")


def test_runtime_config_resolves_preset_defaults() -> None:
    cfg = types.SimpleNamespace(
        weight_compression_enabled=True,
        weight_compression_preset="stable_backbone_int8",
        weight_compression_target="none",
        weight_compression_format="fp8_e4m3",
        fp8_base=False,
    )
    resolved = _wc.resolve_weight_compression_runtime_config(cfg)
    assert resolved.requested is True
    assert resolved.enabled is True
    assert resolved.target == "backbone"
    assert resolved.format == "torchao_int8"
    assert resolved.preset == "stable_backbone_int8"
    assert resolved.preset_applied is True

    explicit = _wc.resolve_weight_compression_runtime_config(
        types.SimpleNamespace(
            weight_compression_enabled=True,
            weight_compression_preset="stable_backbone_int8",
            weight_compression_target="text_encoder",
            weight_compression_format="quanto_int8",
            fp8_base=False,
        )
    )
    assert explicit.target == "text_encoder"
    assert explicit.format == "quanto_int8"
    print("  [PASS] runtime config resolves preset defaults")


def test_legacy_fp8_maps_to_backbone() -> None:
    bundle = TinyBundle()
    freeze(bundle)
    result = _wc.apply_weight_compression(bundle, legacy_fp8_base=True)
    assert result.enabled is True
    assert result.target == "backbone"
    assert [c.name for c in result.components] == ["backbone"]
    print("  [PASS] legacy fp8_base maps to backbone")


def test_text_encoder_compression_and_trainable_skip() -> None:
    bundle = TinyBundle()
    freeze(bundle)
    result = _wc.apply_weight_compression(
        bundle,
        enabled=True,
        target="text_encoder",
        train_text_encoder=False,
    )
    assert {c.name for c in result.components} == {"text_encoder_1", "text_encoder_2"}
    assert result.compressed_count > 0

    bundle2 = TinyBundle()
    freeze(bundle2)
    result2 = _wc.apply_weight_compression(
        bundle2,
        enabled=True,
        target="text_encoder",
        train_text_encoder=True,
    )
    assert result2.components == []
    assert any("Text encoder compression skipped" in w for w in result2.warnings)
    print("  [PASS] text encoder compression + trainable guard")


def test_patterns_and_adapter_skip() -> None:
    bundle = TinyBundle()
    freeze(bundle)
    adapter = nn.Parameter(torch.randn(4, 4), requires_grad=True)
    injector = FakeInjector([adapter])
    result = _wc.apply_weight_compression(
        bundle,
        enabled=True,
        target="both",
        lora_injector=injector,
        include_patterns="text_encoder_1.0.weight,backbone.0.weight",
        exclude_patterns="text_encoder_2",
    )
    names = {c.name for c in result.components}
    assert names == {"backbone", "text_encoder_1", "text_encoder_2"}
    assert result.compressed_count == 2
    assert adapter.dtype != torch.float8_e4m3fn
    print("  [PASS] include/exclude patterns + adapter skip")




def test_lora_wrapped_original_remains_compressible() -> None:
    injector_mod = _import_from_file("core.lulynx_trainer.lora_injector", _here / "lora_injector.py")
    model = nn.Sequential(nn.Linear(8, 8))
    injector = injector_mod.LoRAInjector(rank=2, alpha=2.0)
    injector.inject(model, ["0"], prefix="unet")
    for param in model.parameters():
        param.requires_grad = False
    result = _wc.apply_weight_compression(
        type("Bundle", (), {"unet": model})(),
        enabled=True,
        target="backbone",
        format="fp8_e4m3",
        lora_injector=injector,
    )
    assert result.compressed_count >= 1
    layer = injector.injected_layers["unet.0"]
    assert layer.original.weight.dtype == torch.float8_e4m3fn
    assert layer.lora.lora_down.weight.dtype != torch.float8_e4m3fn
    print("  [PASS] lora wrapped original remains compressible")
def test_optional_backend_capability_probe() -> None:
    formats = _wc.available_weight_compression_formats()
    assert formats["fp8_e4m3"]["backend"] == "native"
    assert "torchao_int8" in formats
    assert "quanto_float8" in formats
    torchao = _wc.get_weight_compression_format_info("torchao_int8")
    if not torchao.available:
        assert "torchao" in torchao.unavailable_reason
    quanto = _wc.get_weight_compression_format_info("quanto_float8")
    if not quanto.available:
        assert "quanto" in quanto.unavailable_reason
    print("  [PASS] optional backend capability probe")
def test_preflight_conflicts() -> None:
    report = _checks.check_weight_compression(
        {
            "weight_compression_enabled": True,
            "weight_compression_target": "text_encoder",
            "train_text_encoder": True,
            "torch_compile": True,
            "module_offload_enabled": True,
        },
        "sdxl-lora",
    )
    assert len(report.errors) == 3
    missing_backend = _checks.check_weight_compression({"weight_compression_enabled": True, "weight_compression_target": "backbone", "weight_compression_format": "torchao_uint4"}, "anima-lora")
    if not _wc.get_weight_compression_format_info("torchao_uint4").available:
        assert any("torchao" in err for err in missing_backend.errors)
    preset_report = _checks.check_weight_compression({"weight_compression_preset": "stable_backbone_int8", "weight_compression_verify": False}, "anima-lora")
    if _wc.get_weight_compression_format_info("torchao_int8").available:
        assert preset_report.errors == []
    else:
        assert any("torchao" in err for err in preset_report.errors)
    companion_missing = _checks.check_weight_compression({"weight_compression_enabled": True, "weight_compression_target": "backbone", "compression_companion_enabled": True}, "anima-lora")
    assert any("compression_companion_path" in err for err in companion_missing.errors)
    legacy = _checks.check_weight_compression({"fp8_base": True}, "anima-lora")
    assert legacy.errors == []
    assert legacy.notes
    print("  [PASS] preflight conflicts")


def main() -> int:
    tests = [
        test_config_defaults,
        test_runtime_config_resolves_preset_defaults,
        test_legacy_fp8_maps_to_backbone,
        test_text_encoder_compression_and_trainable_skip,
        test_patterns_and_adapter_skip,
        test_lora_wrapped_original_remains_compressible,
        test_optional_backend_capability_probe,
        test_preflight_conflicts,
    ]
    failed = 0
    for test in tests:
        try:
            test()
        except Exception as exc:
            failed += 1
            print(f"  [FAIL] {test.__name__}: {exc}")
    print(f"\nweight_compression smoke: {len(tests) - failed} passed, {failed} failed out of {len(tests)} tests")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())











