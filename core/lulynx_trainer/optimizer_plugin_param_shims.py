# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Parameter shims for optional optimizer plugin bridges."""

from __future__ import annotations

from typing import Any

import torch


def params_to_list(params) -> list[torch.nn.Parameter]:
    if params is None:
        return []
    if isinstance(params, torch.Tensor):
        return [params]
    return [param for param in params if param is not None]


def source_param_groups(trainable_params, lr: float, weight_decay: float) -> list[dict[str, Any]]:
    if isinstance(trainable_params, (dict, torch.Tensor)):
        items = [trainable_params]
    else:
        items = list(trainable_params)

    loose_params = []
    groups: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict) and "params" in item:
            groups.append(dict(item))
        else:
            loose_params.append(item)

    if loose_params:
        groups.insert(0, {"params": loose_params, "lr": lr, "weight_decay": weight_decay})
    return groups


class TrainableParamModule(torch.nn.Module):
    """Model shim for plugin optimizers that require named_parameters()."""

    def __init__(self, trainable_params, *, lr: float, weight_decay: float) -> None:
        super().__init__()
        params: list[torch.nn.Parameter] = []
        seen: set[int] = set()
        for group in source_param_groups(trainable_params, lr=lr, weight_decay=weight_decay):
            for param in params_to_list(group.get("params")):
                if not isinstance(param, torch.nn.Parameter):
                    param = torch.nn.Parameter(param)
                key = id(param)
                if key in seen:
                    continue
                seen.add(key)
                params.append(param)
        if not params:
            raise ValueError("Model-aware optimizer requires at least one trainable parameter.")
        self.params = torch.nn.ParameterList(params)


def create_model_aware_param_list_optimizer(
    optimizer_class: type,
    trainable_params,
    *,
    lr: float,
    weight_decay: float,
    kwargs: dict[str, Any],
) -> torch.optim.Optimizer:
    module = TrainableParamModule(trainable_params, lr=lr, weight_decay=weight_decay)
    optimizer = optimizer_class(module, lr=lr, weight_decay=weight_decay, **kwargs)
    groups = source_param_groups(trainable_params, lr=lr, weight_decay=weight_decay)
    if _has_non_default_group_options(groups, lr=lr, weight_decay=weight_decay):
        _apply_group_overrides_by_param_identity(optimizer, groups, lr=lr, weight_decay=weight_decay)
    return optimizer


def _has_non_default_group_options(groups: list[dict[str, Any]], *, lr: float, weight_decay: float) -> bool:
    return any(
        float(group.get("lr", lr)) != float(lr)
        or float(group.get("weight_decay", weight_decay)) != float(weight_decay)
        for group in groups
    )


def _apply_group_overrides_by_param_identity(
    optimizer: torch.optim.Optimizer,
    source_groups: list[dict[str, Any]],
    *,
    lr: float,
    weight_decay: float,
) -> None:
    overrides: dict[int, dict[str, float]] = {}
    for source in source_groups:
        values = {
            "lr": float(source.get("lr", lr)),
            "weight_decay": float(source.get("weight_decay", weight_decay)),
        }
        for param in params_to_list(source.get("params")):
            overrides[id(param)] = values

    for group in optimizer.param_groups:
        matched = [overrides[id(param)] for param in params_to_list(group.get("params")) if id(param) in overrides]
        if not matched:
            continue
        first = matched[0]
        if all(item == first for item in matched):
            group.update(first)
