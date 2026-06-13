"""Smoke checks for Fromage per-tensor norm and p_bound launch config propagation."""

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


def run_smoke() -> dict[str, Any]:
    params = [
        torch.nn.Parameter(torch.tensor([1.0, -0.5], dtype=torch.float32)),
        torch.nn.Parameter(torch.tensor([[0.25, -0.75, 0.5]], dtype=torch.float32)),
    ]
    executor = build_simple_optimizer_training_executor(
        params=params,
        config={
            "optimizer_kind": "fromage",
            "lr": 0.1,
            "p_bound": 1.0,
            "require_native_cuda": False,
        },
        workspace_root=REPO_ROOT,
    )
    try:
        launch_config = executor._launch_config()  # noqa: SLF001
        state = executor.state_flat.detach().cpu()
        assert launch_config["optimizer_kind"] == "fromage", launch_config
        assert launch_config["p_bound"] == 1.0, launch_config
        assert launch_config["per_tensor_norm"] is True, launch_config
        assert launch_config["tensor_count"] == 2, launch_config
        assert launch_config["param_group_offsets"] == [0, 2, 5], launch_config
        assert launch_config["training_dispatch"] is True, launch_config
        assert launch_config["training_path_enabled"] is True, launch_config
        assert launch_config["state_roles"] == [
            "param_group_offsets",
            "per_tensor_param_norm",
            "per_tensor_grad_norm",
            "p_bound",
            "per_tensor_post_norm",
        ], launch_config
        assert [item["numel"] for item in launch_config["state_layout"]] == [3, 2, 2, 2, 2], launch_config
        assert launch_config["state_numel"] == 11, launch_config
        assert executor.state_flat.numel() == 11, executor.state_flat
        assert state[:3].tolist() == [0.0, 2.0, 5.0], state.tolist()
        assert state[7:9].tolist() == [
            torch.linalg.vector_norm(params[0].detach().float()).item(),
            torch.linalg.vector_norm(params[1].detach().float()).item(),
        ], state.tolist()
    finally:
        executor.close()

    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_fromage_launch_config_smoke",
        "ok": True,
        "summary": {
            "fromage_per_tensor_norm_launch_config_ready_count": 1,
            "fromage_p_bound_launch_config_ready_count": 1,
            "fromage_branch_state_layout_ready_count": 1,
        },
        "recommended_next_step": "validate Fromage per-tensor norm and p_bound native kernel parity before broader training validation",
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
