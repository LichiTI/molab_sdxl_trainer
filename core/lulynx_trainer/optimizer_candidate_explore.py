# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Run a deeper tiny-CPU exploration for selected optimizer candidates."""

from __future__ import annotations

import argparse
import json
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


DEFAULT_CANDIDATES = ["CAME", "StableAdamW", "ScheduleFreeAdamW", "Ranger21", "SCION"]


@dataclass
class CandidateResult:
    name: str
    status: str
    initial_loss: float | None = None
    final_loss: float | None = None
    best_loss: float | None = None
    loss_ratio: float | None = None
    grad_accum_ok: bool = False
    state_dict_ok: bool = False
    has_train_eval: bool = False
    state_tensors: int = 0
    state_numel: int = 0
    param_numel: int = 0
    notes: str = ""


class TinyRegression(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.SiLU(),
            nn.Linear(16, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _dataset() -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(42)
    x = torch.randn(32, 8)
    true_w = torch.randn(8, 4) * 0.5
    y = torch.tanh(x @ true_w) + 0.03 * torch.randn(32, 4)
    return x, y


def _short_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}".replace("\n", " ")[:240]


def _optimizer_args(name: str, steps: int) -> dict[str, Any]:
    args: dict[str, Any] = {"name": name}
    lower = name.lower()
    if lower == "ranger21":
        args["num_iterations"] = steps
    return args


def _state_stats(optimizer: torch.optim.Optimizer) -> tuple[int, int]:
    tensor_count = 0
    numel = 0
    state = optimizer.state_dict().get("state", {})
    for item in state.values():
        if not isinstance(item, dict):
            continue
        for value in item.values():
            if torch.is_tensor(value):
                tensor_count += 1
                numel += int(value.numel())
    return tensor_count, numel


def explore_candidate(name: str, steps: int, lr: float) -> CandidateResult:
    torch.manual_seed(123)
    model = TinyRegression()
    x, y = _dataset()
    param_numel = sum(p.numel() for p in model.parameters() if p.requires_grad)
    try:
        optimizer = create_pytorch_optimizer(
            model.parameters(),
            optimizer_name=name,
            lr=lr,
            weight_decay=0.0,
            optimizer_args=_optimizer_args(name, steps),
        )
    except Exception as exc:
        return CandidateResult(name=name, status="fail", param_numel=param_numel, notes=_short_error(exc))

    has_train_eval = hasattr(optimizer, "train") and hasattr(optimizer, "eval")
    if hasattr(optimizer, "train"):
        optimizer.train()

    loss_fn = nn.MSELoss()
    losses: list[float] = []
    try:
        with torch.no_grad():
            initial_loss = float(loss_fn(model(x), y))

        for step in range(steps):
            optimizer.zero_grad(set_to_none=True)
            # Two micro-batches to catch basic gradient-accumulation compatibility.
            for micro_x, micro_y in ((x[:16], y[:16]), (x[16:], y[16:])):
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
        state_tensors, state_numel = _state_stats(optimizer)
    except Exception as exc:
        return CandidateResult(
            name=name,
            status="fail",
            initial_loss=initial_loss if "initial_loss" in locals() else None,
            param_numel=param_numel,
            has_train_eval=has_train_eval,
            notes=_short_error(exc),
        )

    final_loss = losses[-1] if losses else initial_loss
    best_loss = min(losses) if losses else initial_loss
    ratio = final_loss / initial_loss if initial_loss > 0 else None
    status = "pass" if ratio is not None and ratio < 0.98 else "warn"
    note = "loss decreased" if status == "pass" else "loss did not clearly decrease"
    return CandidateResult(
        name=name,
        status=status,
        initial_loss=initial_loss,
        final_loss=final_loss,
        best_loss=best_loss,
        loss_ratio=ratio,
        grad_accum_ok=True,
        state_dict_ok=state_dict_ok,
        has_train_eval=has_train_eval,
        state_tensors=state_tensors,
        state_numel=state_numel,
        param_numel=param_numel,
        notes=note,
    )


def _print_table(results: list[CandidateResult]) -> None:
    headers = ["status", "name", "loss_ratio", "final_loss", "state/param", "train_eval", "notes"]
    rows = []
    for result in results:
        ratio = "-" if result.loss_ratio is None else f"{result.loss_ratio:.3f}"
        final = "-" if result.final_loss is None else f"{result.final_loss:.5f}"
        state_ratio = "-" if not result.param_numel else f"{result.state_numel / result.param_numel:.2f}x"
        rows.append([
            result.status,
            result.name,
            ratio,
            final,
            state_ratio,
            "yes" if result.has_train_eval else "no",
            result.notes,
        ])
    widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers))))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Explore optimizer candidates on a tiny CPU regression task.")
    parser.add_argument("names", nargs="*", default=DEFAULT_CANDIDATES)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    results = [explore_candidate(name, max(args.steps, 1), args.lr) for name in args.names]
    if args.json:
        print(json.dumps([asdict(result) for result in results], ensure_ascii=False, indent=2))
    else:
        _print_table(results)
    return 0 if (not args.strict or all(result.status == "pass" for result in results)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
