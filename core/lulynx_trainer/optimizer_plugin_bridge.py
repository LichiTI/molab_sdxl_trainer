# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Optional optimizer plugin bridge.

This module intentionally stays small: it discovers local Lulynx optimizer
provider plugins and imports their upstream packages through normal Python
imports. It does not vendor or copy optimizer implementations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import MethodType
from typing import Any

import torch

from .optimizer_plugin_param_shims import (
    TrainableParamModule,
    create_model_aware_param_list_optimizer,
    params_to_list,
    source_param_groups,
)
from .optimizer_plugin_wrappers import ClosureRequiredOptimizer, FusedBackwardOptimizer, LossValueClosureOptimizer


_PLUGIN_CAPABILITY = "optimizer_provider:pytorch_optimizer"
_MUON_PARAM_GROUP_OPTIMIZERS = frozenset({"muon", "adamuon", "adago"})
_MATRIX_ONLY_WITH_ADAMW_FALLBACK = frozenset({"spectralsphere"})
_ALICE_OPTIMIZERS = frozenset({"alice"})
_CLOSURE_REQUIRED_OPTIMIZERS = frozenset({"bsam", "lbfgs"})
_CLOSURE_INITIAL_BACKWARD_OPTIMIZERS = frozenset({"bsam"})
_CREATE_GRAPH_BACKWARD_OPTIMIZERS = frozenset({"adahessian"})
_DEMO_OPTIMIZERS = frozenset({"demo"})
_DISTRIBUTED_MUON_OPTIMIZERS = frozenset({"distributedmuon"})
_KRON_OPTIMIZERS = frozenset({"kron"})
_LOSS_VALUE_CLOSURE_OPTIMIZERS = frozenset({"alig"})
_LOMO_FAMILY_OPTIMIZERS = frozenset({"adalomo", "lomo"})
_MODEL_AWARE_PARAM_LIST_OPTIMIZERS = frozenset({"adammini"})
_TORCH_DTYPE_ARG_NAMES = frozenset({"momentum_dtype", "mu_dtype", "precondition_dtype"})
_TORCH_DTYPE_BY_NAME = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "torch.bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "torch.float16": torch.float16,
    "half": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "torch.float32": torch.float32,
    "float": torch.float32,
    "fp64": torch.float64,
    "float64": torch.float64,
    "torch.float64": torch.float64,
    "double": torch.float64,
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _enabled_overrides(root: Path) -> dict[str, bool]:
    path = root / "data" / "plugins" / "enabled.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    raw = data.get("plugins", {})
    if not isinstance(raw, dict):
        return {}
    return {str(key): bool(value) for key, value in raw.items()}


def _iter_plugin_dirs(root: Path):
    for base in (root / "plugin", root / "plugins", root / "extensions"):
        if not base.is_dir():
            continue
        for child in sorted(base.iterdir(), key=lambda p: p.name.lower()):
            if child.is_dir():
                yield child


def _read_manifest(plugin_dir: Path) -> dict[str, Any] | None:
    for name in ("plugin_manifest.json", "manifest.json"):
        path = plugin_dir / name
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
            return data if isinstance(data, dict) else None
    return None


def _find_pytorch_optimizer_plugin() -> Path:
    root = _project_root()
    overrides = _enabled_overrides(root)
    for plugin_dir in _iter_plugin_dirs(root):
        manifest = _read_manifest(plugin_dir)
        if not manifest:
            continue
        capabilities = manifest.get("capabilities", [])
        if _PLUGIN_CAPABILITY not in [str(item) for item in capabilities if item is not None]:
            continue
        plugin_id = str(manifest.get("id") or plugin_dir.name).strip()
        enabled = bool(overrides.get(plugin_id, bool(manifest.get("enabled_by_default", False))))
        if not enabled:
            raise RuntimeError(f"Optimizer plugin {plugin_id} is installed but disabled.")
        return plugin_dir
    raise RuntimeError("pytorch-optimizer plugin is not installed or does not expose optimizer_provider:pytorch_optimizer.")


def _ensure_plugin_import_path() -> Path:
    plugin_dir = _find_pytorch_optimizer_plugin()
    plugin_path = str(plugin_dir)
    if plugin_path not in sys.path:
        sys.path.insert(0, plugin_path)
    return plugin_dir


def create_pytorch_optimizer(
    trainable_params,
    *,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    optimizer_args: dict[str, Any],
):
    """Instantiate an optimizer from the local pytorch-optimizer plugin."""
    _ensure_plugin_import_path()
    from pytorch_optimizer import get_supported_optimizers, load_optimizer

    name = str(optimizer_name or "").strip()
    if not name:
        raise ValueError("PytorchOptimizer requires optimizer_args name=<optimizer class>, e.g. name=Ranger21.")
    supported = set(get_supported_optimizers())
    if name.lower() not in supported:
        raise ValueError(f"Unsupported pytorch-optimizer optimizer {name!r}.")

    optimizer_class = load_optimizer(name)
    kwargs = dict(optimizer_args)
    for key in ("name", "optimizer_name", "optimizer", "lr", "learning_rate", "weight_decay"):
        kwargs.pop(key, None)
    kwargs = _coerce_plugin_optimizer_kwargs(name, kwargs)
    if name.lower() in _DISTRIBUTED_MUON_OPTIMIZERS:
        trainable_params = _build_muon_param_groups(trainable_params, lr=lr, weight_decay=weight_decay)
        optimizer = _create_distributed_muon_optimizer(optimizer_class, trainable_params, lr, weight_decay, kwargs)
        return _finalize_plugin_optimizer(optimizer, lr=lr, name=name)
    if name.lower() in _CLOSURE_REQUIRED_OPTIMIZERS:
        optimizer = _instantiate_optimizer_class(optimizer_class, trainable_params, lr, weight_decay, kwargs)
        return _finalize_plugin_optimizer(
            ClosureRequiredOptimizer(
                optimizer,
                name=name,
                requires_initial_backward=name.lower() in _CLOSURE_INITIAL_BACKWARD_OPTIMIZERS,
            ),
            lr=lr,
            name=name,
        )
    if name.lower() in _LOMO_FAMILY_OPTIMIZERS:
        optimizer = _create_lomo_family_optimizer(
            optimizer_class,
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            kwargs=kwargs,
            name=name,
        )
        return _finalize_plugin_optimizer(optimizer, lr=lr, name=name)
    if name.lower() in _LOSS_VALUE_CLOSURE_OPTIMIZERS:
        kwargs.setdefault("max_lr", lr)
        optimizer = _instantiate_optimizer_class(optimizer_class, trainable_params, lr, weight_decay, kwargs)
        return _finalize_plugin_optimizer(LossValueClosureOptimizer(optimizer), lr=lr, name=name)
    if name.lower() in _MODEL_AWARE_PARAM_LIST_OPTIMIZERS:
        optimizer = create_model_aware_param_list_optimizer(
            optimizer_class,
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            kwargs=kwargs,
        )
        return _finalize_plugin_optimizer(optimizer, lr=lr, name=name)
    if name.lower() in _MUON_PARAM_GROUP_OPTIMIZERS:
        trainable_params = _build_muon_param_groups(trainable_params, lr=lr, weight_decay=weight_decay)
    if name.lower() in _ALICE_OPTIMIZERS:
        trainable_params, kwargs = _build_alice_param_groups(trainable_params, lr=lr, weight_decay=weight_decay, kwargs=kwargs)
    if name.lower() in _MATRIX_ONLY_WITH_ADAMW_FALLBACK:
        return _create_matrix_only_with_adamw_fallback(
            optimizer_class,
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            kwargs=kwargs,
            name=name,
        )
    try:
        optimizer = optimizer_class(trainable_params, lr=lr, weight_decay=weight_decay, **kwargs)
    except TypeError:
        optimizer = optimizer_class(trainable_params, lr=lr, **kwargs)
    return _finalize_plugin_optimizer(optimizer, lr=lr, name=name)


def list_pytorch_optimizer_capabilities() -> dict[str, list[str]]:
    """Return optimizer/scheduler names exposed by the installed plugin."""
    _ensure_plugin_import_path()
    from pytorch_optimizer import get_supported_lr_schedulers, get_supported_optimizers

    return {
        "optimizers": list(get_supported_optimizers()),
        "lr_schedulers": list(get_supported_lr_schedulers()),
    }


def is_schedulefree_like(optimizer: torch.optim.Optimizer) -> bool:
    return hasattr(optimizer, "train") and hasattr(optimizer, "eval") and "schedule" in type(optimizer).__name__.lower()


def _instantiate_optimizer_class(
    cls: type,
    trainable_params,
    lr: float,
    weight_decay: float,
    kwargs: dict[str, Any],
) -> torch.optim.Optimizer:
    try:
        return cls(trainable_params, lr=lr, weight_decay=weight_decay, **kwargs)
    except TypeError:
        return cls(trainable_params, lr=lr, **kwargs)


def _create_distributed_muon_optimizer(
    optimizer_class: type,
    trainable_params,
    lr: float,
    weight_decay: float,
    kwargs: dict[str, Any],
) -> torch.optim.Optimizer:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return _instantiate_optimizer_class(optimizer_class, trainable_params, lr, weight_decay, kwargs)

    module = sys.modules.get(getattr(optimizer_class, "__module__", ""))
    if module is None:
        return _instantiate_optimizer_class(optimizer_class, trainable_params, lr, weight_decay, kwargs)
    original_get_world_size = getattr(module, "get_world_size", None)
    original_get_rank = getattr(module, "get_rank", None)
    original_all_gather = getattr(module, "all_gather", None)

    def get_world_size_identity():
        return 1

    def get_rank_identity():
        return 0

    def all_gather_identity(output_tensors, input_tensors):
        for out, src in zip(output_tensors, input_tensors):
            out.copy_(src)
        return None

    try:
        module.get_world_size = get_world_size_identity
        module.get_rank = get_rank_identity
        module.all_gather = all_gather_identity
        return _instantiate_optimizer_class(optimizer_class, trainable_params, lr, weight_decay, kwargs)
    finally:
        if original_get_world_size is not None:
            module.get_world_size = original_get_world_size
        if original_get_rank is not None:
            module.get_rank = original_get_rank
        if original_all_gather is not None:
            module.all_gather = original_all_gather


def _coerce_plugin_optimizer_kwargs(name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    result = dict(kwargs)
    # AdaFactor's upstream default uses bfloat16 momentum, which is fragile for
    # CPU state_dict roundtrips and mixed dtype foreach ops. FP32 is the safe
    # Lulynx bridge default unless the user explicitly opts into another dtype.
    if name.lower() == "adafactor":
        result.setdefault("momentum_dtype", torch.float32)
    for key in _TORCH_DTYPE_ARG_NAMES:
        value = result.get(key)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in _TORCH_DTYPE_BY_NAME:
                result[key] = _TORCH_DTYPE_BY_NAME[normalized]
    return result


def _finalize_plugin_optimizer(optimizer: torch.optim.Optimizer, *, lr: float, name: str = "") -> torch.optim.Optimizer:
    for group in optimizer.param_groups:
        group.setdefault("lr", lr)
    if str(name).strip().lower() in _CREATE_GRAPH_BACKWARD_OPTIMIZERS:
        optimizer._lulynx_requires_create_graph_backward = True
    if str(name).strip().lower() in _ALICE_OPTIMIZERS:
        _patch_alice_optimizer(optimizer)
    if str(name).strip().lower() in _DEMO_OPTIMIZERS:
        _patch_demo_optimizer(optimizer)
    if str(name).strip().lower() in _DISTRIBUTED_MUON_OPTIMIZERS:
        _patch_distributed_muon_optimizer(optimizer)
    if str(name).strip().lower() in _KRON_OPTIMIZERS:
        _patch_kron_optimizer(optimizer)
    _patch_plugin_optimizer_state_loading(optimizer, name=name)
    return optimizer


def _patch_alice_optimizer(optimizer: torch.optim.Optimizer) -> None:
    original_switch = optimizer.switch

    def switch(self, q: torch.Tensor, u_prev: torch.Tensor, rank: int, leading_basis: int) -> torch.Tensor:
        if int(rank) == int(q.shape[0]):
            return original_switch(q, u_prev, rank, leading_basis)
        vals, vecs = self.subspace_iteration(q.to(torch.float32), u_prev.to(torch.float32), num_steps=1)
        leading = max(1, min(int(leading_basis), int(rank)))
        leading_indices = torch.argsort(vals, descending=True)[:leading]
        projected = u_prev.to(torch.float32) @ vecs
        u_t1 = projected[:, leading_indices]

        tail = max(int(rank) - u_t1.shape[1], 0)
        if tail == 0:
            return u_t1.to(dtype=q.dtype)

        eye = torch.eye(q.shape[0], device=q.device, dtype=torch.float32)
        u_c, _ = torch.linalg.qr(eye - u_t1 @ u_t1.T)
        u_t2 = u_c[:, :tail]
        return torch.cat([u_t1, u_t2], dim=1).to(dtype=q.dtype)

    optimizer.switch = MethodType(switch, optimizer)


def _patch_demo_optimizer(optimizer: torch.optim.Optimizer) -> None:
    original_demo_all_gather = optimizer.demo_all_gather

    def demo_all_gather(self, sparse_idx, sparse_val):
        process_group = getattr(self, "process_group", None)
        distributed_ready = torch.distributed.is_available() and torch.distributed.is_initialized()
        if process_group is None and not distributed_ready:
            return [sparse_idx], [sparse_val]
        return original_demo_all_gather(sparse_idx, sparse_val)

    original_state_dict = optimizer.state_dict
    original_load_state_dict = optimizer.load_state_dict

    def state_dict_with_demo_state():
        state_dict = original_state_dict()
        state_dict["lulynx_demo_state"] = [
            _clone_state_tensors(getattr(optimizer, "demo_state", {}).get(param, {}))
            for param in _optimizer_ordered_params(optimizer)
        ]
        return state_dict

    def load_state_dict_with_demo_state(state_dict):
        demo_state = state_dict.get("lulynx_demo_state") if isinstance(state_dict, dict) else None
        base_state_dict = dict(state_dict)
        base_state_dict.pop("lulynx_demo_state", None)
        result = original_load_state_dict(base_state_dict)
        if isinstance(demo_state, list):
            optimizer.demo_state = getattr(optimizer, "demo_state", {}) or {}
            for param, saved in zip(_optimizer_ordered_params(optimizer), demo_state):
                restored = _clone_state_tensors(saved, param=param) if isinstance(saved, dict) else {}
                optimizer.demo_state[param] = restored
        return result

    optimizer.demo_all_gather = MethodType(demo_all_gather, optimizer)
    optimizer.state_dict = state_dict_with_demo_state
    optimizer.load_state_dict = load_state_dict_with_demo_state


def _patch_distributed_muon_optimizer(optimizer: torch.optim.Optimizer) -> None:
    original_step = optimizer.step

    def step_with_identity_distributed(self, closure=None):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return original_step(closure)
        module = sys.modules.get(type(self).__module__)
        if module is None:
            return original_step(closure)
        original_all_gather = getattr(module, "all_gather", None)

        def all_gather_identity(output_tensors, input_tensors):
            for out, src in zip(output_tensors, input_tensors):
                out.copy_(src)
            return None

        try:
            module.all_gather = all_gather_identity
            return original_step(closure)
        finally:
            if original_all_gather is not None:
                module.all_gather = original_all_gather

    optimizer.step = MethodType(step_with_identity_distributed, optimizer)


def _optimizer_ordered_params(optimizer: torch.optim.Optimizer) -> list[torch.nn.Parameter]:
    return [param for group in optimizer.param_groups for param in group.get("params", [])]


def _clone_state_tensors(state: dict[str, Any], *, param: torch.Tensor | None = None) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in state.items():
        if isinstance(value, torch.Tensor):
            tensor = value.detach().clone()
            if param is not None:
                tensor = tensor.to(device=param.device, dtype=param.dtype if tensor.is_floating_point() else tensor.dtype)
            cloned[key] = tensor
        else:
            cloned[key] = value
    return cloned


def _clone_kron_expression_tree(value: Any) -> Any:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [_clone_kron_expression_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_kron_expression_tree(item) for item in value)
    return value


def _patch_kron_optimizer(optimizer: torch.optim.Optimizer) -> None:
    original_state_dict = optimizer.state_dict
    original_load_state_dict = optimizer.load_state_dict

    def state_dict_with_kron_state():
        state_dict = original_state_dict()
        state_dict["lulynx_kron_expressions"] = [
            _clone_kron_expression_tree(getattr(optimizer, "state", {}).get(param, {}).get("expressions"))
            for param in _optimizer_ordered_params(optimizer)
        ]
        state_dict["lulynx_kron_runtime"] = {
            "prob_step": int(getattr(optimizer, "prob_step", 0) or 0),
            "update_counter": int(getattr(optimizer, "update_counter", 0) or 0),
        }
        return state_dict

    def load_state_dict_with_kron_state(state_dict):
        expressions = state_dict.get("lulynx_kron_expressions") if isinstance(state_dict, dict) else None
        runtime = state_dict.get("lulynx_kron_runtime") if isinstance(state_dict, dict) else None
        base_state_dict = dict(state_dict)
        base_state_dict.pop("lulynx_kron_expressions", None)
        base_state_dict.pop("lulynx_kron_runtime", None)
        result = original_load_state_dict(base_state_dict)
        if isinstance(expressions, list):
            for param, saved in zip(_optimizer_ordered_params(optimizer), expressions):
                state = getattr(optimizer, "state", {}).get(param)
                if isinstance(state, dict) and saved is not None:
                    state["expressions"] = _clone_kron_expression_tree(saved)
        if isinstance(runtime, dict):
            optimizer.prob_step = int(runtime.get("prob_step", getattr(optimizer, "prob_step", 0)) or 0)
            optimizer.update_counter = int(runtime.get("update_counter", getattr(optimizer, "update_counter", 0)) or 0)
        return result

    optimizer.state_dict = state_dict_with_kron_state
    optimizer.load_state_dict = load_state_dict_with_kron_state


def _patch_plugin_optimizer_state_loading(optimizer: torch.optim.Optimizer, *, name: str) -> None:
    canonical = str(name or type(optimizer).__name__).strip().lower()
    if canonical not in {"spam", "sgdsai"}:
        return

    original_load_state_dict = optimizer.load_state_dict

    def load_state_dict_with_plugin_fixes(state_dict):
        result = original_load_state_dict(state_dict)
        if canonical == "spam":
            for state in optimizer.state.values():
                mask = state.get("mask") if isinstance(state, dict) else None
                if isinstance(mask, torch.Tensor) and mask.dtype is not torch.bool:
                    state["mask"] = mask.to(dtype=torch.bool)
        elif canonical == "sgdsai":
            optimizer.has_warmup = any(
                isinstance(state, dict) and "gsnr" in state
                for state in optimizer.state.values()
            )
        return result

    optimizer.load_state_dict = load_state_dict_with_plugin_fixes


def _build_muon_param_groups(trainable_params, *, lr: float, weight_decay: float) -> list[dict[str, Any]]:
    """Split params for Muon-family optimizers without requiring model names.

    pytorch-optimizer's Muon/AdaMuon/AdaGO require every param group to declare
    use_muon. Lulynx usually has trainable LoRA tensors, not full module names,
    so the bridge uses tensor rank: matrices use Muon; scalars/vectors use AdamW.
    """
    result: list[dict[str, Any]] = []
    for source in source_param_groups(trainable_params, lr=lr, weight_decay=weight_decay):
        params = params_to_list(source.get("params"))
        if not params:
            continue

        base_group = {key: value for key, value in source.items() if key not in {"params", "use_muon"}}
        base_group.setdefault("lr", lr)
        base_group.setdefault("weight_decay", weight_decay)
        muon_params = [param for param in params if getattr(param, "ndim", 0) >= 2]
        adamw_params = [param for param in params if getattr(param, "ndim", 0) < 2]

        if muon_params:
            group = dict(base_group)
            group.update({"params": muon_params, "use_muon": True})
            result.append(group)
        if adamw_params:
            group = dict(base_group)
            group.update({"params": adamw_params, "use_muon": False})
            result.append(group)

    if not result:
        raise ValueError("Muon-family optimizers require at least one trainable tensor.")
    return result


def _build_alice_param_groups(
    trainable_params,
    *,
    lr: float,
    weight_decay: float,
    kwargs: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kwargs = dict(kwargs)
    rank_cap = int(kwargs.pop("rank_cap", kwargs.get("rank", 8)) or 8)
    requested_leading = kwargs.get("leading_basis")
    groups: list[dict[str, Any]] = []
    fallback_params = []

    for source in source_param_groups(trainable_params, lr=lr, weight_decay=weight_decay):
        params = params_to_list(source.get("params"))
        if not params:
            continue
        base_group = {key: value for key, value in source.items() if key != "params"}
        base_group.setdefault("lr", lr)
        base_group.setdefault("weight_decay", weight_decay)
        for param in params:
            rows = int(param.shape[0]) if getattr(param, "ndim", 0) >= 1 else 0
            if rows < 1:
                fallback_params.append(param)
                continue
            rank = max(1, min(rank_cap, rows))
            if requested_leading is None:
                leading_basis = max(1, min(rank - 1, max(1, rank // 2))) if rank > 1 else 1
            else:
                leading_basis = int(requested_leading)
                if rank > 1:
                    leading_basis = max(1, min(leading_basis, rank - 1))
                else:
                    leading_basis = 1
            group = dict(base_group)
            group.update({"params": [param], "rank": rank, "leading_basis": leading_basis})
            groups.append(group)

    if fallback_params:
        groups.append({"params": fallback_params, "lr": lr, "weight_decay": weight_decay, "rank": 1, "leading_basis": 1})
    return groups, kwargs


def _split_matrix_and_fallback_param_groups(
    trainable_params,
    *,
    lr: float,
    weight_decay: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    matrix_groups: list[dict[str, Any]] = []
    fallback_groups: list[dict[str, Any]] = []
    for source in source_param_groups(trainable_params, lr=lr, weight_decay=weight_decay):
        params = params_to_list(source.get("params"))
        if not params:
            continue
        base_group = {key: value for key, value in source.items() if key != "params"}
        base_group.setdefault("lr", lr)
        base_group.setdefault("weight_decay", weight_decay)
        matrix_params = [param for param in params if getattr(param, "ndim", 0) == 2]
        fallback_params = [param for param in params if getattr(param, "ndim", 0) != 2]
        if matrix_params:
            group = dict(base_group)
            group["params"] = matrix_params
            matrix_groups.append(group)
        if fallback_params:
            group = dict(base_group)
            group["params"] = fallback_params
            fallback_groups.append(group)
    return matrix_groups, fallback_groups


class _CompositeOptimizer(torch.optim.Optimizer):
    def __init__(self, optimizers: list[torch.optim.Optimizer]) -> None:
        if not optimizers:
            raise ValueError("Composite optimizer requires at least one optimizer.")
        self.optimizers = optimizers
        params = []
        for optimizer in optimizers:
            for group in optimizer.param_groups:
                params.extend(group.get("params", []))
        super().__init__(params, defaults={})
        self.param_groups = [group for optimizer in optimizers for group in optimizer.param_groups]
        self.state = {}
        for optimizer in optimizers:
            self.state.update(optimizer.state)

    def __str__(self) -> str:
        return "+".join(type(optimizer).__name__ for optimizer in self.optimizers)

    def zero_grad(self, set_to_none: bool = True) -> None:  # type: ignore[override]
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):  # type: ignore[override]
        loss = None
        for optimizer in self.optimizers:
            result = optimizer.step(closure)
            if result is not None:
                loss = result
        self.state = {}
        for optimizer in self.optimizers:
            self.state.update(optimizer.state)
        return loss

    def state_dict(self):  # type: ignore[override]
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict):  # type: ignore[override]
        states = state_dict.get("optimizers") if isinstance(state_dict, dict) else None
        if not isinstance(states, list) or len(states) != len(self.optimizers):
            raise ValueError("Composite optimizer state_dict does not match optimizer parts.")
        for optimizer, optimizer_state in zip(self.optimizers, states):
            optimizer.load_state_dict(optimizer_state)
        self.param_groups = [group for optimizer in self.optimizers for group in optimizer.param_groups]
        self.state = {}
        for optimizer in self.optimizers:
            self.state.update(optimizer.state)


def _create_lomo_family_optimizer(
    optimizer_class: type,
    trainable_params,
    *,
    lr: float,
    weight_decay: float,
    kwargs: dict[str, Any],
    name: str,
) -> torch.optim.Optimizer:
    module = TrainableParamModule(trainable_params, lr=lr, weight_decay=weight_decay)
    lomo_kwargs = dict(kwargs)
    if str(name).strip().lower() == "adalomo":
        lomo_kwargs.setdefault("weight_decay", weight_decay)
    optimizer = optimizer_class(module, lr=lr, **lomo_kwargs)
    return FusedBackwardOptimizer(optimizer, name=name)


def _create_matrix_only_with_adamw_fallback(
    optimizer_class: type,
    trainable_params,
    *,
    lr: float,
    weight_decay: float,
    kwargs: dict[str, Any],
    name: str,
) -> torch.optim.Optimizer:
    matrix_groups, fallback_groups = _split_matrix_and_fallback_param_groups(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
    )
    optimizers: list[torch.optim.Optimizer] = []
    if matrix_groups:
        optimizers.append(_finalize_plugin_optimizer(
            _instantiate_optimizer_class(optimizer_class, matrix_groups, lr, weight_decay, kwargs),
            lr=lr,
            name=name,
        ))
    if fallback_groups:
        optimizers.append(torch.optim.AdamW(fallback_groups, lr=lr, weight_decay=weight_decay))
    if len(optimizers) == 1:
        return optimizers[0]
    return _CompositeOptimizer(optimizers)


_RESERVED_KEYS = frozenset({"name", "optimizer_name", "optimizer", "lr", "learning_rate", "weight_decay"})


def create_generic_optimizer(
    trainable_params,
    *,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    optimizer_args: dict[str, Any],
) -> torch.optim.Optimizer:
    """Resolve and instantiate an optimizer by name through a three-tier chain.

    Resolution order:
    1. pytorch_optimizer plugin (if installed and enabled)
    2. torch.optim built-ins (case-insensitive)
    3. Dotted-path dynamic import (e.g. "mypackage.optim.MyOpt")
    """
    name = str(optimizer_name or "").strip()
    if not name:
        raise ValueError("GenericOptimizer requires optimizer_args name=<class>.")

    kwargs = {k: v for k, v in optimizer_args.items() if k not in _RESERVED_KEYS}
    is_dotted = "." in name

    # Tier 1: pytorch_optimizer plugin
    if not is_dotted:
        try:
            _ensure_plugin_import_path()
            from pytorch_optimizer import get_supported_optimizers, load_optimizer
            if name.lower() in set(get_supported_optimizers()):
                cls = load_optimizer(name)
                kwargs = _coerce_plugin_optimizer_kwargs(name, kwargs)
                if name.lower() in _DISTRIBUTED_MUON_OPTIMIZERS:
                    trainable_params = _build_muon_param_groups(trainable_params, lr=lr, weight_decay=weight_decay)
                    optimizer = _create_distributed_muon_optimizer(cls, trainable_params, lr, weight_decay, kwargs)
                    return _finalize_plugin_optimizer(optimizer, lr=lr, name=name)
                if name.lower() in _CLOSURE_REQUIRED_OPTIMIZERS:
                    optimizer = _instantiate_optimizer_class(cls, trainable_params, lr, weight_decay, kwargs)
                    return _finalize_plugin_optimizer(
                        ClosureRequiredOptimizer(
                            optimizer,
                            name=name,
                            requires_initial_backward=name.lower() in _CLOSURE_INITIAL_BACKWARD_OPTIMIZERS,
                        ),
                        lr=lr,
                        name=name,
                    )
                if name.lower() in _LOMO_FAMILY_OPTIMIZERS:
                    optimizer = _create_lomo_family_optimizer(
                        cls,
                        trainable_params,
                        lr=lr,
                        weight_decay=weight_decay,
                        kwargs=kwargs,
                        name=name,
                    )
                    return _finalize_plugin_optimizer(optimizer, lr=lr, name=name)
                if name.lower() in _LOSS_VALUE_CLOSURE_OPTIMIZERS:
                    kwargs.setdefault("max_lr", lr)
                    optimizer = _instantiate_optimizer_class(cls, trainable_params, lr, weight_decay, kwargs)
                    return _finalize_plugin_optimizer(LossValueClosureOptimizer(optimizer), lr=lr, name=name)
                if name.lower() in _MODEL_AWARE_PARAM_LIST_OPTIMIZERS:
                    optimizer = create_model_aware_param_list_optimizer(
                        cls,
                        trainable_params,
                        lr=lr,
                        weight_decay=weight_decay,
                        kwargs=kwargs,
                    )
                    return _finalize_plugin_optimizer(optimizer, lr=lr, name=name)
                if name.lower() in _MUON_PARAM_GROUP_OPTIMIZERS:
                    trainable_params = _build_muon_param_groups(trainable_params, lr=lr, weight_decay=weight_decay)
                if name.lower() in _ALICE_OPTIMIZERS:
                    trainable_params, kwargs = _build_alice_param_groups(
                        trainable_params,
                        lr=lr,
                        weight_decay=weight_decay,
                        kwargs=kwargs,
                    )
                if name.lower() in _MATRIX_ONLY_WITH_ADAMW_FALLBACK:
                    return _create_matrix_only_with_adamw_fallback(
                        cls,
                        trainable_params,
                        lr=lr,
                        weight_decay=weight_decay,
                        kwargs=kwargs,
                        name=name,
                    )
                optimizer = _instantiate_optimizer_class(cls, trainable_params, lr, weight_decay, kwargs)
                return _finalize_plugin_optimizer(optimizer, lr=lr, name=name)
        except Exception:
            pass

    # Tier 2: torch.optim built-ins
    if not is_dotted:
        _torch_optim_map = {
            k.lower(): v for k, v in torch.optim.__dict__.items()
            if isinstance(v, type) and issubclass(v, torch.optim.Optimizer)
            and v is not torch.optim.Optimizer
        }
        cls = _torch_optim_map.get(name.lower())
        if cls is not None:
            return _instantiate_optimizer_class(cls, trainable_params, lr, weight_decay, kwargs)

    # Tier 3: dotted-path dynamic import
    if is_dotted:
        import importlib
        module_path, _, class_name = name.rpartition(".")
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name, None)
        if cls is not None and isinstance(cls, type):
            return _instantiate_optimizer_class(cls, trainable_params, lr, weight_decay, kwargs)

    raise ValueError(
        f"GenericOptimizer: cannot resolve {name!r}. "
        f"Tried: pytorch_optimizer plugin, torch.optim built-ins, dotted import path."
    )
