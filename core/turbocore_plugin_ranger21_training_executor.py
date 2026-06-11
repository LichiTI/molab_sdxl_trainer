"""Default-off training executor wrapper for selected plugin Ranger21."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import torch

from core.turbocore_plugin_ranger_family_training_support import (
    RangerFamilyTrainingExecutor,
    RangerFamilyTrainingExecutorConfig,
    build_ranger_family_training_executor,
)


def build_plugin_ranger21_training_executor(
    *,
    optimizer: torch.optim.Optimizer,
    params: Iterable[torch.nn.Parameter],
    config: RangerFamilyTrainingExecutorConfig | Mapping[str, Any] | None = None,
    workspace_root: str | None = None,
) -> RangerFamilyTrainingExecutor:
    return build_ranger_family_training_executor(
        optimizer=optimizer,
        params=params,
        config=config,
        workspace_root=workspace_root,
        kind="ranger21",
    )


__all__ = ["build_plugin_ranger21_training_executor"]
