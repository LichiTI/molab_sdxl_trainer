"""Smoke for RMSProp centered/momentum native branch kernel behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_simple_optimizer_training_executor import (  # noqa: E402
    build_simple_optimizer_training_executor,
)

PARAM_ATOL = 1e-6
STATE_ATOL = 5e-6


def run_smoke() -> dict[str, Any]:
    param = torch.nn.Parameter(torch.tensor([0.25, -0.5, 0.125], dtype=torch.float32))
    executor = build_simple_optimizer_training_executor(
        params=[param],
        config={
            "optimizer_kind": "rmsprop",
            "lr": 1e-3,
            "alpha": 0.98,
            "eps": 1e-8,
            "momentum": 0.25,
            "centered": True,
            "require_native_cuda": False,
        },
        workspace_root=REPO_ROOT,
    )
    try:
        launch_config = executor._launch_config()  # noqa: SLF001
        assert launch_config["state_roles"] == ["square_avg", "grad_avg", "momentum_buffer"], launch_config
        assert launch_config["state_numel"] == 9, launch_config
        if torch.cuda.is_available():
            result = _cuda_kernel_result(executor)
        else:
            result = {
                "ok": True,
                "skipped": True,
                "reason": "cuda_unavailable",
                "expected_kernel_branch": "rmsprop_centered_momentum",
            }
    finally:
        executor.close()

    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_rmsprop_native_abi_guard_smoke",
        "ok": True,
        "summary": {
            "rmsprop_branch_state_layout_ready_count": 1,
            "rmsprop_branch_native_abi_guard_ready_count": 1,
            "rmsprop_branch_native_abi_fail_closed_count": 0,
            "rmsprop_branch_native_kernel_launch_count": 0 if result.get("skipped") else 1,
            "rmsprop_branch_native_kernel_parity_ready_count": 0 if result.get("skipped") else 1,
        },
        "guard_result": result,
        "recommended_next_step": "promote RMSProp centered/momentum branch only after broader training validation",
    }


def _cuda_kernel_result(executor: Any) -> dict[str, Any]:
    before = executor.params[0].detach().clone()
    grad = torch.tensor([0.1, -0.2, 0.05], dtype=torch.float32)
    reference_param, reference_state = _torch_reference(before.cuda(), grad.cuda())
    cuda_param = torch.nn.Parameter(before.clone().cuda())
    cuda_executor = build_simple_optimizer_training_executor(
        params=[cuda_param],
        config={
            "optimizer_kind": "rmsprop",
            "lr": 1e-3,
            "alpha": 0.98,
            "eps": 1e-8,
            "momentum": 0.25,
            "centered": True,
            "require_native_cuda": False,
        },
        workspace_root=REPO_ROOT,
    )
    try:
        cuda_param.grad = grad.cuda()
        result = cuda_executor({"training_dispatch": True, "training_path_enabled": True})
        step_report = dict(result.get("step_report", {})) if isinstance(result.get("step_report"), dict) else {}
        param_after = cuda_param.detach().cpu()
        state_after = cuda_executor.state_flat.detach().cpu()
        param_diff = float((param_after - reference_param).abs().max().item())
        state_diff = float((state_after - reference_state).abs().max().item())
        assert result["ok"] is True, result
        assert result["native_kernel_launched"] is True, result
        assert step_report.get("kernel_executed") is True, step_report
        assert step_report.get("rmsprop_branch_kernel") is True, step_report
        assert param_diff <= PARAM_ATOL, {
            "param_after": param_after.tolist(),
            "reference": reference_param.tolist(),
            "diff": param_diff,
            "atol": PARAM_ATOL,
        }
        assert state_diff <= STATE_ATOL, {
            "state_after": state_after.tolist(),
            "reference": reference_state.tolist(),
            "diff": state_diff,
            "atol": STATE_ATOL,
        }
        return {
            **result,
            "param_max_abs_diff": param_diff,
            "state_max_abs_diff": state_diff,
            "param_atol": PARAM_ATOL,
            "state_atol": STATE_ATOL,
        }
    finally:
        cuda_executor.close()


def _torch_reference(param_value: torch.Tensor, grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    param = torch.nn.Parameter(param_value.clone())
    optimizer = torch.optim.RMSprop(
        [param],
        lr=1e-3,
        alpha=0.98,
        eps=1e-8,
        momentum=0.25,
        centered=True,
        weight_decay=0.0,
    )
    param.grad = grad.clone()
    optimizer.step()
    state = optimizer.state[param]
    state_flat = torch.cat(
        [
            state["square_avg"].detach().reshape(-1).float(),
            state["grad_avg"].detach().reshape(-1).float(),
            state["momentum_buffer"].detach().reshape(-1).float(),
        ]
    )
    return param.detach().clone().cpu().float(), state_flat.cpu()


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
