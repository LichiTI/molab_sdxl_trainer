"""Smoke for the lulynx_native Rust JSON flat AdamW owner prototype."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_flat_adamw_state import FlatAdamWConfig, PersistentFlatAdamW  # noqa: E402
from core.turbocore_flat_buffer_descriptor import build_reference_flat_adamw_owner_descriptor  # noqa: E402


def _inject_native_artifact_dir_from_env() -> None:
    raw = str(os.environ.get("LULYNX_NATIVE_ARTIFACT_DIR", "") or "").strip()
    if not raw:
        return
    path = Path(raw).expanduser()
    if path.is_dir():
        resolved = str(path.resolve())
        if resolved not in sys.path:
            sys.path.insert(0, resolved)


def _require_native():
    _inject_native_artifact_dir_from_env()
    spec = importlib.util.find_spec("lulynx_native")
    if spec is None:
        raise RuntimeError("lulynx_native is not importable; run devtools/run_turbocore_native_bridge_smoke.bat first")
    import lulynx_native  # type: ignore

    return lulynx_native


def _make_flat_values(seed: int, count: int, scale: float) -> list[float]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tensor = torch.randn(count, generator=generator, dtype=torch.float64) * scale
    return tensor.tolist()


def _max_diff(left: list[float], right: list[float]) -> float:
    lhs = torch.tensor(left, dtype=torch.float64)
    rhs = torch.tensor(right, dtype=torch.float64)
    return float((lhs - rhs).abs().max().item()) if lhs.numel() else 0.0


PARITY_ATOL = 2e-7


def _state_params(state: dict[str, Any]) -> list[float]:
    params = state.get("param_flat")
    if not isinstance(params, list):
        raise AssertionError(state)
    return [float(item) for item in params]


def _python_owner(param_flat: list[float], cfg: FlatAdamWConfig) -> PersistentFlatAdamW:
    tensor = torch.tensor(param_flat, dtype=torch.float64)
    return PersistentFlatAdamW([tensor], cfg)


def run_smoke() -> dict[str, Any]:
    native = _require_native()
    required = [
        "create_flat_adamw_owner",
        "flat_adamw_set_grad_buffer",
        "flat_adamw_step",
        "flat_adamw_zero_grad",
        "flat_adamw_state_dict",
        "flat_adamw_load_state_dict",
        "flat_adamw_snapshot",
        "destroy_flat_adamw_owner",
    ]
    missing = [name for name in required if not hasattr(native, name)]
    if missing:
        raise AssertionError({"missing_flat_owner_entrypoints": missing})

    config = {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "finite_check": True,
    }
    cfg = FlatAdamWConfig(
        lr=config["lr"],
        betas=tuple(config["betas"]),
        eps=config["eps"],
        weight_decay=config["weight_decay"],
        max_grad_norm=config["max_grad_norm"],
        finite_check=config["finite_check"],
    )
    param_flat = _make_flat_values(8100, 13, 0.01)
    descriptor = build_reference_flat_adamw_owner_descriptor(numel=len(param_flat), handle_kind="json_reference")
    py_owner = _python_owner(param_flat, cfg)
    create = native.create_flat_adamw_owner(json.dumps({"param_flat": param_flat, "descriptor": descriptor, **config}))
    assert create["ok"] is True, create
    assert create["snapshot"]["descriptor_present"] is True, create
    owner_id = int(create["owner_id"])
    restored_id: int | None = None
    restored_checked = False

    for step_index in range(3):
        grad_flat = _make_flat_values(8200 + step_index, 13, 2.0)
        py_owner.set_grads([torch.tensor(grad_flat, dtype=torch.float64)])
        py_report = py_owner.step()
        step = native.flat_adamw_step(owner_id, json.dumps({"grad_flat": grad_flat}))
        assert step["ok"] is True and step["skipped"] is False, step
        assert py_report.step_index == step_index + 1
        state = native.flat_adamw_state_dict(owner_id)
        diff = _max_diff(_state_params(state), py_owner.param_flat.detach().double().cpu().tolist())
        assert diff < PARITY_ATOL, {"step": step_index, "diff": diff, "state": state}
        if step_index == 1:
            saved_state = native.flat_adamw_state_dict(owner_id)
            assert saved_state["descriptor"] is not None, saved_state
            restored = native.create_flat_adamw_owner(json.dumps({"param_flat": _state_params(saved_state), "descriptor": descriptor, **config}))
            restored_id = int(restored["owner_id"])
            loaded = native.flat_adamw_load_state_dict(restored_id, json.dumps(saved_state))
            assert loaded["ok"] is True, loaded
        elif restored_id is not None:
            restored_step = native.flat_adamw_step(restored_id, json.dumps({"grad_flat": grad_flat}))
            assert restored_step["ok"] is True and restored_step["skipped"] is False, restored_step
            restored_state = native.flat_adamw_state_dict(restored_id)
            main_state = native.flat_adamw_state_dict(owner_id)
            resume_diff = _max_diff(_state_params(restored_state), _state_params(main_state))
            assert resume_diff < PARITY_ATOL, {"resume_diff": resume_diff}
            restored_checked = True
        zeroed = native.flat_adamw_zero_grad(owner_id)
        assert zeroed["ok"] is True, zeroed
        if restored_id is not None:
            restored_zeroed = native.flat_adamw_zero_grad(restored_id)
            assert restored_zeroed["ok"] is True, restored_zeroed

    assert restored_id is not None and restored_checked is True
    before_bad = native.flat_adamw_state_dict(owner_id)
    bad_grad = _make_flat_values(8300, 13, 1.0)
    bad_grad[0] = "NaN"  # type: ignore[assignment]
    skipped = native.flat_adamw_step(owner_id, json.dumps({"grad_flat": bad_grad}))
    after_bad = native.flat_adamw_state_dict(owner_id)
    assert skipped["ok"] is True and skipped["skipped"] is True, skipped
    assert _state_params(before_bad) == _state_params(after_bad)
    snapshot = native.flat_adamw_snapshot(owner_id)
    assert snapshot["training_path_enabled"] is False, snapshot
    assert snapshot["native_kernel_present"] is False, snapshot
    assert snapshot["reference_flat_owner"] is True, snapshot
    assert snapshot["descriptor_present"] is True, snapshot
    bad_descriptor = build_reference_flat_adamw_owner_descriptor(numel=len(param_flat) + 1, handle_kind="json_reference")
    bad_create = native.create_flat_adamw_owner(json.dumps({"param_flat": param_flat, "descriptor": bad_descriptor, **config}))
    assert bad_create["ok"] is False, bad_create
    assert "descriptor_param_flat_numel_mismatch" in bad_create["reason"], bad_create
    native.destroy_flat_adamw_owner(owner_id)
    native.destroy_flat_adamw_owner(restored_id)
    return {
        "schema_version": 1,
        "probe": "turbocore_native_flat_adamw_owner_smoke",
        "ok": True,
        "native_kernel_present": False,
        "reference_flat_owner": True,
        "training_path_enabled": False,
    }


if __name__ == "__main__":
    result = run_smoke()
    print(json.dumps(result, ensure_ascii=False, indent=2))
