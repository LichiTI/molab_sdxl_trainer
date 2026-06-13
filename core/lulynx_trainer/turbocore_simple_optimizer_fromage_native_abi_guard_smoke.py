"""Smoke for Fromage per-tensor norm and p_bound native branch behavior."""

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

from pytorch_optimizer import Fromage  # noqa: E402

from core.turbocore_simple_optimizer_training_executor import (  # noqa: E402
    build_simple_optimizer_training_executor,
)


PARAM_ATOL = 2e-6
STATE_ATOL = 2e-6


def run_smoke() -> dict[str, Any]:
    params = _initial_params(device="cpu")
    executor = build_simple_optimizer_training_executor(
        params=params,
        config=_config(),
        workspace_root=REPO_ROOT,
    )
    try:
        launch_config = executor._launch_config()  # noqa: SLF001
        assert launch_config["state_roles"] == [
            "param_group_offsets",
            "per_tensor_param_norm",
            "per_tensor_grad_norm",
            "p_bound",
            "per_tensor_post_norm",
        ], launch_config
        assert launch_config["state_numel"] == 11, launch_config
        assert launch_config["param_group_offsets"] == [0, 2, 5], launch_config
        if torch.cuda.is_available():
            result = _cuda_kernel_result()
        else:
            result = {
                "ok": True,
                "skipped": True,
                "reason": "cuda_unavailable",
                "expected_kernel_branch": "fromage_per_tensor_norm_p_bound",
            }
    finally:
        executor.close()

    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_fromage_native_abi_guard_smoke",
        "ok": True,
        "summary": {
            "fromage_per_tensor_norm_state_layout_ready_count": 1,
            "fromage_p_bound_state_layout_ready_count": 1,
            "fromage_per_tensor_norm_native_abi_guard_ready_count": 1,
            "fromage_p_bound_native_abi_guard_ready_count": 1,
            "fromage_branch_native_kernel_launch_count": 0 if result.get("skipped") else 1,
            "fromage_branch_native_kernel_parity_ready_count": 0 if result.get("skipped") else 1,
        },
        "guard_result": result,
        "recommended_next_step": "keep Fromage branch default-off until broader training validation",
    }


def _cuda_kernel_result() -> dict[str, Any]:
    reference_param, reference_p_bound = _torch_reference()
    cuda_params = _initial_params(device="cuda")
    cuda_executor = build_simple_optimizer_training_executor(
        params=cuda_params,
        config=_config(),
        workspace_root=REPO_ROOT,
    )
    try:
        result: dict[str, Any] = {}
        for grads in _grads(device="cuda"):
            for param, grad in zip(cuda_params, grads):
                param.grad = grad.detach().clone()
            result = cuda_executor({"training_dispatch": True, "training_path_enabled": True})
        step_report = dict(result.get("step_report", {})) if isinstance(result.get("step_report"), dict) else {}
        param_after = torch.cat([param.detach().reshape(-1).cpu().float() for param in cuda_params])
        state_after = cuda_executor.state_flat.detach().cpu().float()
        p_bound_after = state_after[7:9]
        param_diff = float((param_after - reference_param).abs().max().item())
        state_diff = float((p_bound_after - reference_p_bound).abs().max().item())
        assert result["ok"] is True, result
        assert result["native_kernel_launched"] is True, result
        assert step_report.get("kernel_executed") is True, step_report
        assert step_report.get("fromage_per_tensor_norm_branch_kernel") is True, step_report
        assert step_report.get("fromage_p_bound_branch_kernel") is True, step_report
        assert step_report.get("per_tensor_norm") is True, step_report
        assert step_report.get("p_bound") == 1.0, step_report
        assert step_report.get("param_group_offsets") == [0, 2, 5], step_report
        assert param_diff <= PARAM_ATOL, {
            "param_after": param_after.tolist(),
            "reference": reference_param.tolist(),
            "diff": param_diff,
            "atol": PARAM_ATOL,
        }
        assert state_diff <= STATE_ATOL, {
            "p_bound_after": p_bound_after.tolist(),
            "reference": reference_p_bound.tolist(),
            "diff": state_diff,
            "atol": STATE_ATOL,
        }
        return {
            **result,
            "step_count": len(_grads(device="cpu")),
            "param_max_abs_diff": param_diff,
            "state_max_abs_diff": state_diff,
            "param_atol": PARAM_ATOL,
            "state_atol": STATE_ATOL,
        }
    finally:
        cuda_executor.close()


def _torch_reference() -> tuple[torch.Tensor, torch.Tensor]:
    params = _initial_params(device="cuda")
    optimizer = Fromage(params, lr=0.1, p_bound=1.0)
    for grads in _grads(device="cuda"):
        for param, grad in zip(params, grads):
            param.grad = grad.detach().clone()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    state = torch.stack([optimizer.state[param]["max"].detach().reshape(()).float().cpu() for param in params])
    param_flat = torch.cat([param.detach().reshape(-1).cpu().float() for param in params])
    return param_flat, state


def _initial_params(*, device: str) -> list[torch.nn.Parameter]:
    return [
        torch.nn.Parameter(torch.tensor([1.0, -0.5], dtype=torch.float32, device=device)),
        torch.nn.Parameter(torch.tensor([[0.25, -0.75, 0.5]], dtype=torch.float32, device=device)),
    ]


def _grads(*, device: str) -> list[list[torch.Tensor]]:
    return [
        [
            torch.tensor([0.4, -0.8], dtype=torch.float32, device=device),
            torch.tensor([[1.0, 0.25, -0.5]], dtype=torch.float32, device=device),
        ],
        [
            torch.tensor([-0.2, 0.5], dtype=torch.float32, device=device),
            torch.tensor([[0.5, -1.0, 0.25]], dtype=torch.float32, device=device),
        ],
    ]


def _config() -> dict[str, Any]:
    return {
        "optimizer_kind": "fromage",
        "lr": 0.1,
        "p_bound": 1.0,
        "require_native_cuda": False,
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
