"""Smoke checks for PID momentum three-buffer launch config propagation."""

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
    param = torch.nn.Parameter(torch.tensor([0.25, -0.5, 0.125], dtype=torch.float32))
    executor = build_simple_optimizer_training_executor(
        params=[param],
        config={
            "optimizer_kind": "pid",
            "lr": 1e-3,
            "momentum": 0.9,
            "dampening": 0.1,
            "derivative": 10.0,
            "integral": 5.0,
            "weight_decay": 0.01,
            "require_native_cuda": False,
        },
        workspace_root=REPO_ROOT,
    )
    try:
        launch_config = executor._launch_config()  # noqa: SLF001
        assert launch_config["optimizer_kind"] == "pid", launch_config
        assert launch_config["momentum"] == 0.9, launch_config
        assert launch_config["dampening"] == 0.1, launch_config
        assert launch_config["derivative"] == 10.0, launch_config
        assert launch_config["integral"] == 5.0, launch_config
        assert launch_config["training_dispatch"] is True, launch_config
        assert launch_config["training_path_enabled"] is True, launch_config
        assert launch_config["max_numel"] == 3, launch_config
        assert launch_config["parameter_numel"] == 3, launch_config
        assert launch_config["state_numel"] == 9, launch_config
        assert launch_config["state_roles"] == ["integral_buffer", "previous_grad", "momentum_buffer"], launch_config
        assert [item["offset"] for item in launch_config["state_layout"]] == [0, 3, 6], launch_config
        assert executor.state_flat.numel() == 9, executor.state_flat
    finally:
        executor.close()

    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_pid_launch_config_smoke",
        "ok": True,
        "summary": {
            "pid_momentum_three_buffer_launch_config_ready_count": 1,
            "pid_momentum_three_buffer_state_layout_ready_count": 1,
        },
        "recommended_next_step": "validate PID momentum three-buffer native kernel parity before broader training validation",
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
