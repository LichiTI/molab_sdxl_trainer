"""Smoke for SGDP projection/decoupled-decay native branch behavior."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
PYTORCH_OPTIMIZER_ROOT = REPO_ROOT / "plugin" / "pytorch_optimizer-main"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT), str(PYTORCH_OPTIMIZER_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from pytorch_optimizer import SGDP  # noqa: E402

from core.turbocore_simple_optimizer_training_executor import (  # noqa: E402
    build_simple_optimizer_training_executor,
)


PARAM_ATOL = 2e-6
STATE_ATOL = 2e-6


def run_smoke() -> dict[str, Any]:
    param = torch.nn.Parameter(torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32))
    executor = build_simple_optimizer_training_executor(
        params=[param],
        config=_config(),
        workspace_root=REPO_ROOT,
    )
    try:
        launch_config = executor._launch_config()  # noqa: SLF001
        assert launch_config["state_roles"] == ["momentum"], launch_config
        assert launch_config["state_numel"] == 4, launch_config
        assert launch_config["delta"] == 1.0, launch_config
        assert launch_config["wd_ratio"] == 0.1, launch_config
        assert launch_config["weight_decouple"] is True, launch_config
        assert launch_config["fixed_decay"] is False, launch_config
        assert launch_config["nesterov"] is True, launch_config
        if torch.cuda.is_available():
            result = _cuda_kernel_result()
        else:
            result = {
                "ok": True,
                "skipped": True,
                "reason": "cuda_unavailable",
                "expected_kernel_branch": "sgdp_projection_decoupled_decay",
            }
    finally:
        executor.close()

    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_sgdp_native_abi_guard_smoke",
        "ok": True,
        "summary": {
            "sgdp_projection_state_layout_ready_count": 1,
            "sgdp_projection_native_abi_guard_ready_count": 1,
            "sgdp_decoupled_decay_native_abi_guard_ready_count": 1,
            "sgdp_branch_native_kernel_launch_count": 0 if result.get("skipped") else 1,
            "sgdp_branch_native_kernel_parity_ready_count": 0 if result.get("skipped") else 1,
        },
        "guard_result": result,
        "recommended_next_step": "keep SGDP branch default-off until broader training validation",
    }


def _cuda_kernel_result() -> dict[str, Any]:
    before = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device="cuda")
    grads = [
        torch.tensor([[0.0, 0.25, -0.5, 0.125]], dtype=torch.float32, device="cuda"),
        torch.tensor([[0.05, -0.125, 0.25, -0.375]], dtype=torch.float32, device="cuda"),
    ]
    reference_param, reference_state = _torch_reference(before, grads)
    cuda_param = torch.nn.Parameter(before.detach().clone())
    cuda_executor = build_simple_optimizer_training_executor(
        params=[cuda_param],
        config=_config(),
        workspace_root=REPO_ROOT,
    )
    try:
        result: dict[str, Any] = {}
        for grad in grads:
            cuda_param.grad = grad.detach().clone()
            result = cuda_executor({"training_dispatch": True, "training_path_enabled": True})
        step_report = dict(result.get("step_report", {})) if isinstance(result.get("step_report"), dict) else {}
        param_after = cuda_param.detach().cpu()
        state_after = cuda_executor.state_flat.detach().cpu()
        param_diff = float((param_after - reference_param).abs().max().item())
        state_diff = float((state_after - reference_state).abs().max().item())
        assert result["ok"] is True, result
        assert result["native_kernel_launched"] is True, result
        assert step_report.get("kernel_executed") is True, step_report
        assert step_report.get("sgdp_projection_branch_kernel") is True, step_report
        assert step_report.get("sgdp_decoupled_decay_branch_kernel") is True, step_report
        assert step_report.get("weight_decouple") is True, step_report
        assert step_report.get("nesterov") is True, step_report
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
            "step_count": len(grads),
            "param_max_abs_diff": param_diff,
            "state_max_abs_diff": state_diff,
            "param_atol": PARAM_ATOL,
            "state_atol": STATE_ATOL,
        }
    finally:
        cuda_executor.close()


def _torch_reference(param_value: torch.Tensor, grads: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    param = torch.nn.Parameter(param_value.detach().clone())
    optimizer = SGDP([param], **_optimizer_kwargs())
    for grad in grads:
        param.grad = grad.detach().clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    state = optimizer.state[param]
    state_flat = state["momentum"].detach().reshape(-1).float()
    return param.detach().clone().cpu().float(), state_flat.cpu()


def _config() -> dict[str, Any]:
    return {
        "optimizer_kind": "sgdp",
        "require_native_cuda": False,
        **_optimizer_kwargs(),
    }


def _optimizer_kwargs() -> dict[str, Any]:
    return {
        "lr": 1e-3,
        "momentum": 0.9,
        "dampening": 0.1,
        "weight_decay": 0.01,
        "weight_decouple": True,
        "fixed_decay": False,
        "delta": 1.0,
        "wd_ratio": 0.1,
        "nesterov": True,
        "eps": 1e-8,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
