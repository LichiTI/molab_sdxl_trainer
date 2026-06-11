from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType
from types import SimpleNamespace
from unittest.mock import patch

torch_stub = ModuleType("torch")
torch_stub.backends = SimpleNamespace(
    mps=SimpleNamespace(is_available=lambda: False, is_built=lambda: False),
)
sys.modules.setdefault("torch", torch_stub)

_HERE = os.path.dirname(os.path.abspath(__file__))
_SPEC = importlib.util.spec_from_file_location(
    "core.lulynx_trainer.mps_runtime",
    os.path.join(_HERE, "mps_runtime.py"),
)
_MPS = importlib.util.module_from_spec(_SPEC)
sys.modules["core.lulynx_trainer.mps_runtime"] = _MPS
_SPEC.loader.exec_module(_MPS)


def test_mps_guard_forces_safe_overrides():
    cfg = SimpleNamespace(
        execution_profile_id="apple-mps",
        schema_id="anima-lora",
        mixed_precision="bf16",
        optimizer_type="PagedAdamW8bit",
        attention_backend="flash2",
        anima_attn_mode="flash2",
        torch_compile=True,
        xformers=True,
        dataloader_num_workers=8,
        persistent_data_loader_workers=True,
    )
    with patch.object(_MPS, "mps_backend_available", return_value=True), patch.object(_MPS, "mps_backend_built", return_value=True):
        guard = _MPS.build_mps_runtime_guard(cfg)
    assert guard.is_mps is True
    assert guard.forced_overrides["device"] == "mps"
    assert guard.forced_overrides["attention_backend"] == "sdpa"
    assert guard.forced_overrides["optimizer_type"] == "AdamW"
    assert guard.forced_overrides["torch_compile"] is False
    assert guard.forced_overrides["mixed_precision"] == "no"
    _MPS.apply_mps_runtime_guard(cfg, guard)
    assert cfg.device == "mps"
    assert cfg.attention_backend == "sdpa"
    assert cfg.optimizer_type == "AdamW"
    assert cfg.mixed_precision == "no"


def test_mps_guard_rejects_flux_and_lumina_routes():
    for schema_id in ("flux-lora", "lumina-lora"):
        cfg = SimpleNamespace(execution_profile_id="apple-mps", schema_id=schema_id)
        guard = _MPS.build_mps_runtime_guard(cfg)
        assert guard.route_supported is False
        assert "does not currently expose" in guard.route_reason


def test_non_mps_runtime_is_noop():
    cfg = SimpleNamespace(execution_profile_id="standard", optimizer_type="PagedAdamW8bit")
    guard = _MPS.build_mps_runtime_guard(cfg)
    assert guard.is_mps is False
    assert guard.forced_overrides == {}


if __name__ == "__main__":
    test_mps_guard_forces_safe_overrides()
    test_mps_guard_rejects_flux_and_lumina_routes()
    test_non_mps_runtime_is_noop()
    print("All Apple MPS runtime smoke tests passed.")
