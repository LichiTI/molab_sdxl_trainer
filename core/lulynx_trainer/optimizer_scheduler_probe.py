# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Probe optional optimizer and scheduler availability on a tiny CPU model.

The probe is intentionally separate from the trainer. It checks whether hidden
optimizer/scheduler routes can instantiate, run a tiny step, and serialize state
without touching a real training job.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.optimizer_plugin_bridge import (  # noqa: E402
    create_pytorch_optimizer,
    list_pytorch_optimizer_capabilities,
)


DEFAULT_OPTIMIZER_CANDIDATES = [
    "CAME",
    "Ranger21",
    "AdEMAMix",
    "StableAdamW",
    "ScheduleFreeAdamW",
    "FAdam",
    "FOCUS",
    "SCION",
    "Lion",
    "AdaFactor",
    "Prodigy",
    "Muon",
    "SOAP",
    "Shampoo",
    "SophiaH",
    "GaLore",
    "OrthoGrad",
]

DEFAULT_SCHEDULER_CANDIDATES = [
    "cosine",
    "linear",
    "poly",
    "proportion",
    "rex",
    "cosine_annealing",
    "cosine_annealing_with_warmup",
    "warmup_stable_decay",
    "chebyshev",
]


@dataclass
class ProbeResult:
    kind: str
    name: str
    status: str
    stage: str
    category: str
    notes: str = ""
    class_name: str = ""


def _tiny_model() -> nn.Module:
    torch.manual_seed(7)
    return nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 2))


def _tiny_loss(model: nn.Module) -> torch.Tensor:
    x = torch.randn(3, 4)
    return model(x).square().mean()


def _short_error(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    return text.replace("\n", " ")[:220]


def _optimizer_category(name: str, passed: bool, notes: str) -> str:
    if not passed:
        lower_notes = notes.lower()
        if "cuda" in lower_notes or "bitsandbytes" in lower_notes or "torchao" in lower_notes:
            return "gpu_or_dependency"
        return "needs_review"
    lower = name.lower()
    if lower in {
        "came",
        "ranger21",
        "ademamix",
        "stableadamw",
        "schedulefreeadamw",
        "fadam",
        "focus",
        "scion",
        "lion",
        "adafactor",
        "prodigy",
    }:
        return "worth_trying"
    if any(token in lower for token in ("shampoo", "soap", "kron", "muon", "galore", "sophia")):
        return "heavy_or_special"
    if any(token in lower for token in ("sam", "pcgrad", "look", "lomo", "distributed")):
        return "special_loop"
    return "available"


def probe_optimizer(name: str) -> ProbeResult:
    model = _tiny_model()
    optimizer_args: dict[str, Any] = {"name": name}
    if name.lower() == "ranger21":
        optimizer_args["num_iterations"] = 8
    try:
        optimizer = create_pytorch_optimizer(
            model.parameters(),
            optimizer_name=name,
            lr=1e-4,
            weight_decay=0.0,
            optimizer_args=optimizer_args,
        )
    except Exception as exc:
        notes = _short_error(exc)
        return ProbeResult("optimizer", name, "fail", "instantiate", _optimizer_category(name, False, notes), notes)

    class_name = type(optimizer).__name__
    try:
        if hasattr(optimizer, "train"):
            optimizer.train()
        optimizer.zero_grad(set_to_none=True)
        loss = _tiny_loss(model)
        loss.backward()
        if isinstance(optimizer, torch.optim.LBFGS):
            optimizer.step(lambda: _tiny_loss(model))
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    except Exception as exc:
        notes = _short_error(exc)
        return ProbeResult("optimizer", name, "fail", "step", _optimizer_category(name, False, notes), notes, class_name)

    try:
        state = optimizer.state_dict()
        optimizer.load_state_dict(state)
    except Exception as exc:
        notes = _short_error(exc)
        return ProbeResult("optimizer", name, "fail", "state_dict", "needs_review", notes, class_name)

    category = _optimizer_category(name, True, "")
    notes = "tiny CPU step passed"
    return ProbeResult("optimizer", name, "pass", "ok", category, notes, class_name)


def _load_scheduler(name: str) -> Callable[..., Any]:
    if "." in name:
        module_name, _, attr = name.rpartition(".")
        module = importlib.import_module(module_name)
        loaded = getattr(module, attr)
    else:
        from pytorch_optimizer import load_lr_scheduler

        loaded = load_lr_scheduler(name)
    return loaded


def _scheduler_kwargs(factory: Callable[..., Any], total_steps: int) -> dict[str, Any]:
    try:
        params = inspect.signature(factory).parameters
    except (TypeError, ValueError):
        return {}

    kwargs: dict[str, Any] = {}
    if "T_max" in params:
        kwargs["T_max"] = total_steps
    if "T_0" in params:
        kwargs["T_0"] = max(total_steps // 2, 1)
    if "first_cycle_steps" in params:
        kwargs["first_cycle_steps"] = total_steps
    if "num_training_steps" in params:
        kwargs["num_training_steps"] = total_steps
    if "num_warmup_steps" in params:
        kwargs["num_warmup_steps"] = 1
    if "total_steps" in params:
        kwargs["total_steps"] = total_steps
    if "t_max" in params:
        kwargs["t_max"] = total_steps
    if "total_iters" in params:
        kwargs["total_iters"] = total_steps
    if "max_lr" in params:
        kwargs["max_lr"] = 1e-3
    if "min_lr" in params:
        kwargs["min_lr"] = 0.0
    if "init_lr" in params:
        kwargs["init_lr"] = 0.0
    if "warmup_steps" in params:
        kwargs["warmup_steps"] = 1
    if "milestones" in params:
        kwargs["milestones"] = [max(total_steps // 2, 1)]
    if "step_size" in params:
        kwargs["step_size"] = 1
    if "lr_lambda" in params:
        kwargs["lr_lambda"] = lambda step: 1.0
    if "gamma" in params:
        kwargs["gamma"] = 0.95
    if "num_epochs" in params:
        kwargs["num_epochs"] = total_steps
    if "num_warmup_steps" in params:
        kwargs["num_warmup_steps"] = 1
    if "num_stable_steps" in params:
        kwargs["num_stable_steps"] = 3
    if "num_decay_steps" in params:
        kwargs["num_decay_steps"] = 4
    if "num_cycles" in params:
        kwargs["num_cycles"] = 1
    return kwargs


def _scheduler_category(name: str, passed: bool, notes: str) -> str:
    if not passed:
        if "required positional" in notes.lower() or "missing" in notes.lower():
            return "needs_args"
        return "needs_review"
    lower = name.lower()
    if lower in {"cosine", "linear", "poly", "rex", "cosine_annealing_with_warmup"}:
        return "worth_trying"
    if lower in {"chebyshev", "warmup_stable_decay", "proportion"}:
        return "special_shape"
    return "available"


def probe_scheduler(name: str) -> ProbeResult:
    model = _tiny_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    total_steps = 8
    try:
        factory = _load_scheduler(name)
    except Exception as exc:
        notes = _short_error(exc)
        return ProbeResult("scheduler", name, "fail", "load", _scheduler_category(name, False, notes), notes)

    class_name = getattr(factory, "__name__", type(factory).__name__)
    try:
        kwargs = _scheduler_kwargs(factory, total_steps)
        if str(name).lower() in {"cosine", "linear", "poly"}:
            kwargs.update(
                {
                    "t_max": total_steps,
                    "max_lr": 1e-3,
                    "min_lr": 0.0,
                    "init_lr": 0.0,
                    "warmup_steps": 1,
                }
            )
        if str(name).lower() == "proportion":
            base = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
            scheduler = factory(base, max_lr=1e-3, min_lr=0.0)
        else:
            scheduler = factory(optimizer, **kwargs)
    except Exception as exc:
        notes = _short_error(exc)
        return ProbeResult("scheduler", name, "fail", "instantiate", _scheduler_category(name, False, notes), notes, class_name)

    try:
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = _tiny_loss(model)
            loss.backward()
            optimizer.step()
            scheduler.step()
    except Exception as exc:
        notes = _short_error(exc)
        return ProbeResult("scheduler", name, "fail", "step", _scheduler_category(name, False, notes), notes, class_name)

    try:
        if hasattr(scheduler, "state_dict"):
            state = scheduler.state_dict()
            scheduler.load_state_dict(state)
    except Exception as exc:
        notes = _short_error(exc)
        return ProbeResult("scheduler", name, "fail", "state_dict", "needs_review", notes, class_name)

    category = _scheduler_category(name, True, "")
    return ProbeResult("scheduler", name, "pass", "ok", category, "tiny CPU schedule passed", class_name)


def _select_names(available: list[str], requested: list[str], defaults: list[str], all_items: bool) -> list[str]:
    if requested:
        requested_lut = {item.lower(): item for item in available}
        return [requested_lut.get(item.lower(), item) for item in requested]
    if all_items:
        return available
    available_lut = {item.lower(): item for item in available}
    return [available_lut[item.lower()] for item in defaults if item.lower() in available_lut]


def _print_table(title: str, results: list[ProbeResult]) -> None:
    print()
    print(title)
    print("=" * len(title))
    if not results:
        print("(no results)")
        return
    rows = [
        [r.status, r.name, r.category, r.stage, r.class_name or "-", r.notes]
        for r in results
    ]
    headers = ["status", "name", "category", "stage", "class", "notes"]
    widths = [
        min(max(len(str(row[i])) for row in [headers] + rows), 44)
        for i in range(len(headers))
    ]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        cells = [str(row[i])[: widths[i]].ljust(widths[i]) for i in range(len(headers))]
        print(" | ".join(cells))


def _print_summary(results: list[ProbeResult]) -> None:
    print()
    print("Summary")
    print("=======")
    for kind in ("optimizer", "scheduler"):
        subset = [r for r in results if r.kind == kind]
        passed = [r for r in subset if r.status == "pass"]
        worth = [r.name for r in passed if r.category == "worth_trying"]
        special = [r.name for r in passed if r.category in {"heavy_or_special", "special_loop", "special_shape"}]
        failed = [r.name for r in subset if r.status != "pass"]
        print(f"{kind}: pass={len(passed)} fail={len(failed)}")
        if worth:
            print(f"  worth_trying: {', '.join(worth)}")
        if special:
            print(f"  special_care: {', '.join(special)}")
        if failed:
            print(f"  failed: {', '.join(failed[:12])}" + (" ..." if len(failed) > 12 else ""))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe hidden optimizer and scheduler routes.")
    parser.add_argument("--all", action="store_true", help="Probe every optimizer/scheduler exposed by pytorch_optimizer.")
    parser.add_argument("--optimizers", nargs="*", default=[], help="Specific optimizer names to probe.")
    parser.add_argument("--schedulers", nargs="*", default=[], help="Specific scheduler names or dotted paths to probe.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of tables.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when any selected probe fails.")
    args = parser.parse_args(argv)

    try:
        caps = list_pytorch_optimizer_capabilities()
    except Exception as exc:
        print(f"Probe failed before discovery: {_short_error(exc)}")
        return 2

    optimizer_names = _select_names(
        caps.get("optimizers", []),
        args.optimizers,
        DEFAULT_OPTIMIZER_CANDIDATES,
        args.all,
    )
    scheduler_names = _select_names(
        caps.get("lr_schedulers", []),
        args.schedulers,
        DEFAULT_SCHEDULER_CANDIDATES,
        args.all,
    )

    optimizer_results = [probe_optimizer(name) for name in optimizer_names]
    scheduler_results = [probe_scheduler(name) for name in scheduler_names]
    results = optimizer_results + scheduler_results

    if args.json:
        print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
        return 0 if (not args.strict or all(result.status == "pass" for result in results)) else 1

    _print_table("Optimizer Probe", optimizer_results)
    _print_table("Scheduler Probe", scheduler_results)
    _print_summary(results)
    return 0 if (not args.strict or all(result.status == "pass" for result in results)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
