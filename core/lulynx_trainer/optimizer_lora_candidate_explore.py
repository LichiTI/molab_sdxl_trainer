# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Explore selected optimizer candidates on a tiny LoRA-shaped task."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.optimizer_plugin_bridge import create_pytorch_optimizer  # noqa: E402


DEFAULT_CANDIDATES = ["CAME", "StableAdamW", "SCION", "ScheduleFreeAdamW", "Ranger21"]


@dataclass
class LoRACandidateResult:
    name: str
    status: str
    initial_loss: float | None = None
    final_loss: float | None = None
    best_loss: float | None = None
    loss_ratio: float | None = None
    delta_norm: float | None = None
    state_numel: int = 0
    trainable_numel: int = 0
    state_ratio: float | None = None
    grad_accum_ok: bool = False
    state_dict_ok: bool = False
    has_train_eval: bool = False
    notes: str = ""


class TinyLoRALinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float) -> None:
        super().__init__()
        self.base = nn.Linear(in_features, out_features, bias=False)
        self.base.weight.requires_grad_(False)
        self.lora_down = nn.Linear(in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, out_features, bias=False)
        self.scale = float(alpha) / float(rank)
        nn.init.normal_(self.lora_down.weight, std=0.02)
        nn.init.zeros_(self.lora_up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scale

    def lora_parameters(self) -> list[nn.Parameter]:
        return [self.lora_down.weight, self.lora_up.weight]

    def delta_norm(self) -> float:
        with torch.no_grad():
            delta = self.lora_up.weight @ self.lora_down.weight * self.scale
            return float(delta.float().norm())


class TinyLoRANet(nn.Module):
    def __init__(self, rank: int = 4) -> None:
        super().__init__()
        self.l1 = TinyLoRALinear(12, 24, rank=rank, alpha=rank)
        self.act = nn.SiLU()
        self.l2 = TinyLoRALinear(24, 8, rank=rank, alpha=rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.l2(self.act(self.l1(x)))

    def lora_parameters(self) -> list[nn.Parameter]:
        return [*self.l1.lora_parameters(), *self.l2.lora_parameters()]

    def delta_norm(self) -> float:
        return math.sqrt(self.l1.delta_norm() ** 2 + self.l2.delta_norm() ** 2)


def _dataset() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(2026)
    x = torch.randn(64, 12)
    teacher = nn.Sequential(nn.Linear(12, 24, bias=False), nn.SiLU(), nn.Linear(24, 8, bias=False))
    with torch.no_grad():
        y = teacher(x) + 0.02 * torch.randn(64, 8)
    return x, y


def _short_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}".replace("\n", " ")[:240]


def _optimizer_args(name: str, steps: int) -> dict[str, Any]:
    args: dict[str, Any] = {"name": name}
    if name.lower() == "ranger21":
        args["num_iterations"] = steps
    return args


def _state_numel(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for state in optimizer.state_dict().get("state", {}).values():
        if not isinstance(state, dict):
            continue
        for value in state.values():
            if torch.is_tensor(value):
                total += int(value.numel())
    return total


def explore(name: str, *, steps: int, lr: float, rank: int) -> LoRACandidateResult:
    torch.manual_seed(99)
    model = TinyLoRANet(rank=rank)
    x, y = _dataset()
    params = model.lora_parameters()
    trainable_numel = sum(int(param.numel()) for param in params)

    try:
        optimizer = create_pytorch_optimizer(
            params,
            optimizer_name=name,
            lr=lr,
            weight_decay=0.0,
            optimizer_args=_optimizer_args(name, steps),
        )
    except Exception as exc:
        return LoRACandidateResult(name=name, status="fail", trainable_numel=trainable_numel, notes=_short_error(exc))

    has_train_eval = hasattr(optimizer, "train") and hasattr(optimizer, "eval")
    if hasattr(optimizer, "train"):
        optimizer.train()

    loss_fn = nn.MSELoss()
    losses: list[float] = []
    try:
        with torch.no_grad():
            initial_loss = float(loss_fn(model(x), y))
        for _step in range(max(int(steps), 1)):
            optimizer.zero_grad(set_to_none=True)
            for micro_x, micro_y in ((x[:32], y[:32]), (x[32:], y[32:])):
                loss = loss_fn(model(micro_x), micro_y) / 2.0
                loss.backward()
            optimizer.step()
            with torch.no_grad():
                losses.append(float(loss_fn(model(x), y)))

        if hasattr(optimizer, "eval"):
            optimizer.eval()
        state = optimizer.state_dict()
        optimizer.load_state_dict(state)
        state_dict_ok = True
        state_numel = _state_numel(optimizer)
    except Exception as exc:
        return LoRACandidateResult(
            name=name,
            status="fail",
            initial_loss=initial_loss if "initial_loss" in locals() else None,
            trainable_numel=trainable_numel,
            has_train_eval=has_train_eval,
            notes=_short_error(exc),
        )

    final_loss = losses[-1] if losses else initial_loss
    best_loss = min(losses) if losses else initial_loss
    ratio = final_loss / initial_loss if initial_loss > 0 else None
    status = "pass" if ratio is not None and ratio < 0.98 and model.delta_norm() > 0 else "warn"
    state_ratio = state_numel / trainable_numel if trainable_numel else None
    return LoRACandidateResult(
        name=name,
        status=status,
        initial_loss=initial_loss,
        final_loss=final_loss,
        best_loss=best_loss,
        loss_ratio=ratio,
        delta_norm=model.delta_norm(),
        state_numel=state_numel,
        trainable_numel=trainable_numel,
        state_ratio=state_ratio,
        grad_accum_ok=True,
        state_dict_ok=state_dict_ok,
        has_train_eval=has_train_eval,
        notes="LoRA loss decreased" if status == "pass" else "LoRA loss did not clearly decrease",
    )


def _print_table(results: list[LoRACandidateResult]) -> None:
    headers = ["status", "name", "loss_ratio", "best_loss", "delta_norm", "state/param", "train_eval", "notes"]
    rows = []
    for result in results:
        rows.append([
            result.status,
            result.name,
            "-" if result.loss_ratio is None else f"{result.loss_ratio:.3f}",
            "-" if result.best_loss is None else f"{result.best_loss:.5f}",
            "-" if result.delta_norm is None else f"{result.delta_norm:.5f}",
            "-" if result.state_ratio is None else f"{result.state_ratio:.2f}x",
            "yes" if result.has_train_eval else "no",
            result.notes,
        ])
    widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explore optimizer candidates on a frozen-base tiny LoRA task.")
    parser.add_argument("names", nargs="*", default=DEFAULT_CANDIDATES)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    results = [explore(name, steps=args.steps, lr=args.lr, rank=args.rank) for name in args.names]
    if args.json:
        print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    else:
        _print_table(results)
    return 0 if (not args.strict or all(result.status == "pass" for result in results)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
