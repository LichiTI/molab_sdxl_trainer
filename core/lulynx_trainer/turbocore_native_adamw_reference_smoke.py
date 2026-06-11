"""Smoke for lulynx_native Rust JSON AdamW reference microkernel.

This is a developer-only probe.  It validates Rust-side AdamW math/state
against PyTorch on tiny JSON arrays; it does not pass PyTorch tensor pointers to
native code and does not enable training dispatch.
"""

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


def _flatten_params(params: list[torch.nn.Parameter]) -> list[list[float]]:
    return [param.detach().double().reshape(-1).cpu().tolist() for param in params]


def _flatten_grads(params: list[torch.nn.Parameter]) -> list[list[float]]:
    values: list[list[float]] = []
    for param in params:
        grad = param.grad
        if grad is None:
            values.append([])
        else:
            values.append(grad.detach().double().reshape(-1).cpu().tolist())
    return values


def _assign_grads(params: list[torch.nn.Parameter], seed: int, *, scale: float = 1.0) -> list[list[float]]:
    torch.manual_seed(seed)
    for param in params:
        param.grad = torch.randn_like(param) * scale
    return _flatten_grads(params)


def _make_params() -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    torch.manual_seed(2026)
    shapes = [(2, 3), (4,)]
    ref: list[torch.nn.Parameter] = []
    for shape in shapes:
        ref.append(torch.nn.Parameter(torch.randn(*shape, dtype=torch.float64) * 0.01))
    cand = [torch.nn.Parameter(param.detach().clone()) for param in ref]
    return ref, cand


def _max_diff(native_params: list[list[float]], ref_params: list[torch.nn.Parameter]) -> float:
    max_diff = 0.0
    for native, ref in zip(native_params, ref_params):
        actual = torch.tensor(native, dtype=torch.float64)
        expected = ref.detach().reshape(-1).double().cpu()
        max_diff = max(max_diff, float((actual - expected).abs().max().item()))
    return max_diff


def _state_params(state: dict[str, Any]) -> list[list[float]]:
    params = state.get("params")
    if not isinstance(params, list):
        raise AssertionError(state)
    return params


def _max_rows_diff(left: list[list[float]], right: list[list[float]]) -> float:
    max_diff = 0.0
    for lhs, rhs in zip(left, right):
        lhs_t = torch.tensor(lhs, dtype=torch.float64)
        rhs_t = torch.tensor(rhs, dtype=torch.float64)
        max_diff = max(max_diff, float((lhs_t - rhs_t).abs().max().item()))
    return max_diff


def run_smoke() -> dict[str, Any]:
    native = _require_native()
    config = {
        "lr": 1e-4,
        "betas": [0.9, 0.999],
        "eps": 1e-8,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "finite_check": True,
        "set_to_none": True,
    }
    ref_params, cand_params = _make_params()
    optimizer = torch.optim.AdamW(ref_params, lr=config["lr"], betas=tuple(config["betas"]), eps=config["eps"], weight_decay=config["weight_decay"])
    create = native.create_stateful_adamw_optimizer(json.dumps({"params": _flatten_params(cand_params), **config}))
    assert create["ok"] is True, create
    optimizer_id = int(create["optimizer_id"])
    restored_id: int | None = None
    restored_checked = False

    for step_index in range(3):
        grads = _assign_grads(ref_params, 7000 + step_index, scale=2.0)
        torch.nn.utils.clip_grad_norm_(ref_params, config["max_grad_norm"])
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step = native.optimizer_step(optimizer_id, json.dumps({"grads": grads}))
        assert step["ok"] is True and step["skipped"] is False, step
        state = native.optimizer_state_dict(optimizer_id)
        diff = _max_diff(_state_params(state), ref_params)
        assert diff < 1e-10, {"step": step_index, "diff": diff, "state": state}
        if step_index == 1:
            saved_state = native.optimizer_state_dict(optimizer_id)
            restored = native.create_stateful_adamw_optimizer(json.dumps({"params": _state_params(state), **config}))
            restored_id = int(restored["optimizer_id"])
            loaded = native.optimizer_load_state_dict(restored_id, json.dumps(saved_state))
            assert loaded["ok"] is True, loaded
        elif restored_id is not None:
            restored_step = native.optimizer_step(restored_id, json.dumps({"grads": grads}))
            assert restored_step["ok"] is True and restored_step["skipped"] is False, restored_step
            restored_state = native.optimizer_state_dict(restored_id)
            main_state = native.optimizer_state_dict(optimizer_id)
            resume_diff = _max_rows_diff(_state_params(restored_state), _state_params(main_state))
            assert resume_diff < 1e-10, {"resume_diff": resume_diff}
            restored_checked = True
        zeroed = native.optimizer_zero_grad(optimizer_id)
        assert zeroed["ok"] is True, zeroed
        if restored_id is not None:
            restored_zeroed = native.optimizer_zero_grad(restored_id)
            assert restored_zeroed["ok"] is True, restored_zeroed

    assert restored_id is not None and restored_checked is True

    before_bad = native.optimizer_state_dict(optimizer_id)
    bad_grads = _assign_grads(ref_params, 7201, scale=1.0)
    bad_grads[0][0] = "NaN"
    skipped = native.optimizer_step(optimizer_id, json.dumps({"grads": bad_grads}))
    after_bad = native.optimizer_state_dict(optimizer_id)
    assert skipped["ok"] is True and skipped["skipped"] is True, skipped
    assert _state_params(before_bad) == _state_params(after_bad)

    native.destroy_optimizer(optimizer_id)
    native.destroy_optimizer(restored_id)
    return {
        "schema_version": 1,
        "probe": "turbocore_native_adamw_reference_smoke",
        "ok": True,
        "native_kernel_present": False,
        "reference_microkernel": True,
        "training_path_enabled": False,
    }


if __name__ == "__main__":
    result = run_smoke()
    print(json.dumps(result, ensure_ascii=False, indent=2))
