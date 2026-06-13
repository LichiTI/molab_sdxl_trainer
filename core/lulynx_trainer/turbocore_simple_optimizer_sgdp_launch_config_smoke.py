"""Smoke checks for SGDP projection/decay launch config propagation."""

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
    param = torch.nn.Parameter(torch.tensor([[0.25, -0.5, 0.125, -0.375]], dtype=torch.float32))
    executor = build_simple_optimizer_training_executor(
        params=[param],
        config={
            "optimizer_kind": "sgdp",
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
            "require_native_cuda": False,
        },
        workspace_root=REPO_ROOT,
    )
    try:
        launch_config = executor._launch_config()  # noqa: SLF001
        assert launch_config["optimizer_kind"] == "sgdp", launch_config
        assert launch_config["momentum"] == 0.9, launch_config
        assert launch_config["dampening"] == 0.1, launch_config
        assert launch_config["weight_decay"] == 0.01, launch_config
        assert launch_config["weight_decouple"] is True, launch_config
        assert launch_config["fixed_decay"] is False, launch_config
        assert launch_config["delta"] == 1.0, launch_config
        assert launch_config["wd_ratio"] == 0.1, launch_config
        assert launch_config["nesterov"] is True, launch_config
        assert launch_config["training_dispatch"] is True, launch_config
        assert launch_config["training_path_enabled"] is True, launch_config
        assert launch_config["max_numel"] == 4, launch_config
        assert launch_config["parameter_numel"] == 4, launch_config
        assert launch_config["state_numel"] == 4, launch_config
        assert launch_config["state_roles"] == ["momentum"], launch_config
        assert [item["offset"] for item in launch_config["state_layout"]] == [0], launch_config
        assert executor.state_flat.numel() == 4, executor.state_flat
    finally:
        executor.close()

    return {
        "schema_version": 1,
        "probe": "turbocore_simple_optimizer_sgdp_launch_config_smoke",
        "ok": True,
        "summary": {
            "sgdp_projection_launch_config_ready_count": 1,
            "sgdp_decoupled_decay_launch_config_ready_count": 1,
            "sgdp_branch_state_layout_ready_count": 1,
        },
        "recommended_next_step": "validate SGDP projection/decoupled decay native kernel parity before broader training validation",
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
