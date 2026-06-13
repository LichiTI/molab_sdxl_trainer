# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""TrainingLoop boundary matrix for selected pytorch_optimizer plugin routes."""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from types import MethodType
from typing import Any, Callable

import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[3]
    backend_root = repo_root / "backend"
    for import_root in (str(repo_root), str(backend_root)):
        if import_root not in sys.path:
            sys.path.insert(0, import_root)

import core.lulynx_trainer.training_loop as training_loop_module
from core.lulynx_trainer.optimizer_plugin_resume_matrix_smoke import (  # noqa: E402
    PluginResumeCase,
    _make_trainer,
    plugin_resume_cases,
)
from core.lulynx_trainer.optimizer_step_contracts import (  # noqa: E402
    optimizer_requires_create_graph_backward,
    optimizer_requires_step_closure,
    optimizer_step_closure_requires_initial_backward,
    optimizer_uses_fused_backward,
    run_optimizer_fused_backward,
)
from core.lulynx_trainer.training_loop import TrainingLoop  # noqa: E402


EXPECTED_PLUGIN_TRAINING_LOOP_CASE_COUNT = 124
ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"


class _SilentProgress:
    def __init__(self, iterable: Any, *_args: Any, **_kwargs: Any) -> None:
        self._iterable = iterable

    def __iter__(self):
        return iter(self._iterable)

    def __len__(self) -> int:
        try:
            return len(self._iterable)
        except TypeError:
            return 0

    def set_postfix(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def update(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def close(self) -> None:
        return None


class _OptimizerCallProbe:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self.optimizer = optimizer
        self.step_calls = 0
        self.zero_grad_calls = 0
        self._original_step = optimizer.step
        self._original_zero_grad = optimizer.zero_grad

    def __enter__(self) -> "_OptimizerCallProbe":
        def _step(*args: Any, **kwargs: Any) -> Any:
            self.step_calls += 1
            return self._original_step(*args, **kwargs)

        def _zero_grad(*args: Any, **kwargs: Any) -> Any:
            self.zero_grad_calls += 1
            return self._original_zero_grad(*args, **kwargs)

        self.optimizer.step = _step  # type: ignore[method-assign]
        self.optimizer.zero_grad = _zero_grad  # type: ignore[method-assign]
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.optimizer.step = self._original_step  # type: ignore[method-assign]
        self.optimizer.zero_grad = self._original_zero_grad  # type: ignore[method-assign]


def run_smoke() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    original_tqdm = training_loop_module.tqdm
    training_loop_module.tqdm = _SilentProgress
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Using backward\\(\\) with create_graph=True will create a reference cycle.*",
                category=UserWarning,
            )
            for source_case in plugin_resume_cases():
                case = PluginResumeCase(**source_case.__dict__)
                try:
                    row = _run_case(case)
                except Exception as exc:
                    row = _failure_row(case, exc)
                    failures.append({"name": case.name, "error": row["error"]})
                rows.append(row)
    finally:
        training_loop_module.tqdm = original_tqdm

    summary = _summary(rows)
    ok = (
        summary["plugin_training_loop_total_count"] == EXPECTED_PLUGIN_TRAINING_LOOP_CASE_COUNT
        and summary["plugin_training_loop_passed_count"] == EXPECTED_PLUGIN_TRAINING_LOOP_CASE_COUNT
        and summary["plugin_training_loop_failed_count"] == 0
        and summary["plugin_training_loop_step_route_executed_count"] == EXPECTED_PLUGIN_TRAINING_LOOP_CASE_COUNT
        and summary["plugin_training_loop_finite_loss_count"] == EXPECTED_PLUGIN_TRAINING_LOOP_CASE_COUNT
    )
    return {
        "schema_version": 1,
        "probe": "optimizer_plugin_training_loop_matrix_smoke",
        "ok": ok,
        "roadmap": ROADMAP,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "summary": summary,
        "cases": rows,
        "failures": failures,
        "notes": [
            "Each case uses a real TrainingLoop.train_epoch boundary with a toy train_step implementation.",
            "This proves TrainingLoop optimizer-step routing for selected plugin optimizers, not CUDA native dispatch.",
            "Fused-backward optimizers update during train_step and are expected not to call public optimizer.step.",
        ],
    }


def _run_case(case: PluginResumeCase) -> dict[str, Any]:
    torch.manual_seed(20260608)
    initial = _matrix(case.shape, -0.25, 0.35)
    vector_initial = _vector(case.shape[0], -0.2, 0.1) if case.include_vector_param or case.expected_param_group_flags else None
    trainer = _make_trainer(case, initial, vector_initial)
    optimizer = trainer._create_optimizer()
    loop = _make_loop(trainer.lora_injector, optimizer)
    route = _route(optimizer)
    initial_params = [param.detach().clone() for param in loop.lora_injector.get_trainable_params()]
    counters = {"train_step_calls": 0, "toy_impl_calls": 0, "fused_backward_calls": 0}
    loop._train_step_impl = MethodType(_toy_train_step_impl(counters), loop)  # type: ignore[method-assign]
    loop.train_step = MethodType(_counted_train_step(counters), loop)  # type: ignore[method-assign]

    with _OptimizerCallProbe(optimizer) as probe:
        result = loop.train_epoch([{"case": case.name}], epoch=0)

    final_params = [param.detach() for param in loop.lora_injector.get_trainable_params()]
    param_changed = any(not torch.allclose(before, after, atol=0.0, rtol=0.0) for before, after in zip(initial_params, final_params))
    finite_params = all(torch.isfinite(param).all().item() for param in final_params)
    grad_cleared = all(param.grad is None for param in loop.lora_injector.get_trainable_params())
    result_steps = int(result.get("steps", 0) or 0)
    step_route_executed = _step_route_executed(route, probe.step_calls, counters["fused_backward_calls"])
    passed = (
        result_steps == 1
        and int(loop.global_step) == 1
        and counters["train_step_calls"] == 1
        and step_route_executed
        and finite_params
        and grad_cleared
    )
    return {
        "name": case.name,
        "optimizer_class": type(optimizer).__name__,
        "route": route,
        "requires_step_closure": optimizer_requires_step_closure(optimizer),
        "requires_initial_backward": optimizer_step_closure_requires_initial_backward(optimizer),
        "uses_fused_backward": optimizer_uses_fused_backward(optimizer),
        "requires_create_graph": optimizer_requires_create_graph_backward(optimizer),
        "result_steps": result_steps,
        "global_step": int(loop.global_step),
        "train_step_calls": counters["train_step_calls"],
        "toy_impl_calls": counters["toy_impl_calls"],
        "optimizer_step_calls": probe.step_calls,
        "optimizer_zero_grad_calls": probe.zero_grad_calls,
        "fused_backward_calls": counters["fused_backward_calls"],
        "step_route_executed": step_route_executed,
        "param_changed": param_changed,
        "finite_params": finite_params,
        "grad_cleared": grad_cleared,
        "finite_loss": torch.isfinite(torch.tensor(float(result.get("avg_loss", 0.0) or 0.0))).item(),
        "passed": passed,
    }


def _make_loop(injector: Any, optimizer: torch.optim.Optimizer) -> TrainingLoop:
    loop = TrainingLoop(
        unet=torch.nn.Identity(),
        text_encoder_1=torch.nn.Identity(),
        text_encoder_2=None,
        vae=torch.nn.Identity(),
        tokenizer_1=None,
        tokenizer_2=None,
        noise_scheduler=None,
        lora_injector=injector,
        optimizer=optimizer,
        lr_scheduler=None,
        device="cpu",
        dtype=torch.float32,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0e9,
        layer_monitor_enabled=False,
        vram_smart_sensing_enabled=False,
    )
    loop.total_steps = 1
    return loop


def _counted_train_step(counters: dict[str, int]) -> Callable[..., float | torch.Tensor]:
    def _train_step(
        self: TrainingLoop,
        batch: dict[str, Any],
        accumulation_steps: int | None = None,
        return_loss_tensor: bool = False,
    ) -> float | torch.Tensor:
        counters["train_step_calls"] += 1
        return TrainingLoop.train_step(self, batch, accumulation_steps, return_loss_tensor)

    return _train_step


def _toy_train_step_impl(counters: dict[str, int]) -> Callable[..., float | torch.Tensor]:
    def _impl(
        self: TrainingLoop,
        batch: dict[str, Any],
        accumulation_steps: int | None = None,
        do_backward: bool = True,
        return_loss_tensor: bool = False,
    ) -> float | torch.Tensor:
        del batch
        counters["toy_impl_calls"] += 1
        accumulation = max(int(accumulation_steps or self.gradient_accumulation_steps or 1), 1)
        loss = _toy_loss(self.lora_injector.get_trainable_params())
        closure_active = bool(getattr(self, "_optimizer_step_closure_active", False))
        deferred_closure = (
            optimizer_requires_step_closure(self.optimizer)
            and not closure_active
            and not optimizer_step_closure_requires_initial_backward(self.optimizer)
        )
        fused = False
        if do_backward and not deferred_closure:
            fused = run_optimizer_fused_backward(
                self.optimizer,
                loss,
                float(self.optimizer.param_groups[0].get("lr", 0.0)) if self.optimizer.param_groups else 0.0,
            )
            if fused:
                counters["fused_backward_calls"] += 1
            else:
                (loss / accumulation).backward(create_graph=optimizer_requires_create_graph_backward(self.optimizer))
        if return_loss_tensor:
            return loss.detach()
        return float(loss.detach().float().item())

    return _impl


def _toy_loss(params: list[torch.nn.Parameter]) -> torch.Tensor:
    losses = []
    for index, param in enumerate(params):
        target = torch.full_like(param, 0.05 * (index + 1))
        losses.append(((param * (index + 1)) - target).pow(2).mean())
    return torch.stack([loss.float() for loss in losses]).mean()


def _route(optimizer: torch.optim.Optimizer) -> str:
    if optimizer_uses_fused_backward(optimizer):
        return "fused_backward"
    if optimizer_requires_step_closure(optimizer):
        return "closure"
    if optimizer_requires_create_graph_backward(optimizer):
        return "create_graph"
    return "standard"


def _step_route_executed(route: str, step_calls: int, fused_backward_calls: int) -> bool:
    if route == "fused_backward":
        return fused_backward_calls > 0 and step_calls == 0
    return step_calls > 0


def _failure_row(case: PluginResumeCase, exc: Exception) -> dict[str, Any]:
    return {
        "name": case.name,
        "optimizer_class": "",
        "route": "failed_before_route",
        "requires_step_closure": False,
        "requires_initial_backward": False,
        "uses_fused_backward": False,
        "requires_create_graph": False,
        "result_steps": 0,
        "global_step": 0,
        "train_step_calls": 0,
        "toy_impl_calls": 0,
        "optimizer_step_calls": 0,
        "optimizer_zero_grad_calls": 0,
        "fused_backward_calls": 0,
        "step_route_executed": False,
        "param_changed": False,
        "finite_params": False,
        "grad_cleared": False,
        "finite_loss": False,
        "passed": False,
        "error": f"{type(exc).__name__}: {exc}",
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "selected_plugin_optimizer_count": len(rows),
        "plugin_optimizer_count": len(rows),
        "plugin_training_loop_total_count": len(rows),
        "plugin_training_loop_passed_count": sum(1 for row in rows if row.get("passed") is True),
        "plugin_training_loop_failed_count": sum(1 for row in rows if row.get("passed") is not True),
        "plugin_training_loop_skipped_count": 0,
        "plugin_training_loop_step_route_executed_count": sum(1 for row in rows if row.get("step_route_executed") is True),
        "plugin_training_loop_standard_step_called_count": sum(1 for row in rows if row.get("route") == "standard" and int(row.get("optimizer_step_calls", 0) or 0) > 0),
        "plugin_training_loop_closure_step_called_count": sum(1 for row in rows if row.get("route") == "closure" and int(row.get("optimizer_step_calls", 0) or 0) > 0),
        "plugin_training_loop_fused_backward_route_count": sum(1 for row in rows if row.get("route") == "fused_backward"),
        "plugin_training_loop_fused_backward_call_count": sum(1 for row in rows if int(row.get("fused_backward_calls", 0) or 0) > 0),
        "plugin_training_loop_create_graph_backward_count": sum(1 for row in rows if row.get("requires_create_graph") is True),
        "plugin_training_loop_zero_grad_called_count": sum(1 for row in rows if int(row.get("optimizer_zero_grad_calls", 0) or 0) > 0),
        "plugin_training_loop_param_updated_count": sum(1 for row in rows if row.get("param_changed") is True),
        "plugin_training_loop_grad_cleared_count": sum(1 for row in rows if row.get("grad_cleared") is True),
        "plugin_training_loop_global_step_advanced_count": sum(1 for row in rows if int(row.get("global_step", 0) or 0) == 1),
        "plugin_training_loop_finite_loss_count": sum(1 for row in rows if row.get("finite_loss") is True),
        "plugin_training_loop_route_counts": _count_by(rows, "route"),
        "training_path_enabled_count": 0,
        "native_dispatch_allowed_count": 0,
        "product_native_ready_count": 0,
    }


def _count_by(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or "")
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _matrix(shape: tuple[int, ...], start: float, end: float) -> torch.Tensor:
    numel = max(1, int(torch.tensor(shape).prod().item()))
    return torch.linspace(start, end, steps=numel, dtype=torch.float32).reshape(shape)


def _vector(size: int, start: float, end: float) -> torch.Tensor:
    return torch.linspace(start, end, steps=max(int(size), 1), dtype=torch.float32)


def main() -> int:
    report = run_smoke()
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
