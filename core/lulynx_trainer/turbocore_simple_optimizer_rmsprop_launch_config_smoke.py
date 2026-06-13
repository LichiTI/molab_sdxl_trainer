"""Smoke checks for RMSProp centered/momentum launch config propagation."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.turbocore_simple_optimizer_training_executor import (  # noqa: E402
    build_simple_optimizer_training_executor,
)


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
        assert launch_config["centered"] is True, launch_config
        assert launch_config["momentum"] == 0.25, launch_config
        assert launch_config["optimizer_kind"] == "rmsprop", launch_config
        assert launch_config["training_dispatch"] is True, launch_config
        assert launch_config["training_path_enabled"] is True, launch_config
        assert launch_config["max_numel"] == 3, launch_config
        assert launch_config["parameter_numel"] == 3, launch_config
        assert launch_config["state_numel"] == 9, launch_config
        assert launch_config["state_roles"] == ["square_avg", "grad_avg", "momentum_buffer"], launch_config
        assert [item["offset"] for item in launch_config["state_layout"]] == [0, 3, 6], launch_config
        assert executor.state_flat.numel() == 9, executor.state_flat
    finally:
        executor.close()

    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_rmsprop_launch_config_smoke",
        "ok": True,
        "summary": {
            "rmsprop_centered_launch_config_ready_count": 1,
            "rmsprop_momentum_launch_config_ready_count": 1,
            "rmsprop_branch_state_layout_ready_count": 1,
        },
        "recommended_next_step": "validate RMSProp centered/momentum native kernel parity before broader training validation",
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
