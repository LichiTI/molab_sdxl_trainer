# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Smoke-test the dynamo/AOT budget knobs.

* pin_recompile_limit raises BOTH the context-local override and the
  canonical entry's cross-context ``.default`` (the ContextVar trap), never
  lowers an existing budget, and 0 = no-op.
* apply_activation_memory_budget applies in (0,1], refuses out-of-range,
  is skipped under gradient_checkpointing, and 0 = no-op.
* plan-level wiring: apply_dynamo_budgets_if_requested records reasons and
  honors the mutual-exclusion guard.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BACKEND = ROOT / "backend"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import torch._dynamo as _dynamo
import torch._functorch.config as functorch_config

from backend.core.lulynx_trainer.dynamo_budget import (
    apply_activation_memory_budget,
    pin_recompile_limit,
)
from backend.core.lulynx_trainer.runtime_optimizations import (
    RuntimeOptimizationPlan,
    apply_dynamo_budgets_if_requested,
)


def _limit_name() -> str:
    return "recompile_limit" if hasattr(_dynamo.config, "recompile_limit") else "cache_size_limit"


def test_pin_raises_override_and_cross_context_default() -> None:
    name = _limit_name()
    target = max(int(getattr(_dynamo.config, name)), 8) + 56
    effective = pin_recompile_limit(target)
    assert effective == target
    assert int(getattr(_dynamo.config, name)) == target
    entry = _dynamo.config._config[name]
    canonical = (getattr(entry, "alias", None) or name).rsplit(".", 1)[-1]
    # the cross-context fallback every compile thread reads
    assert _dynamo.config._config[canonical].default == target


def test_pin_never_lowers_and_zero_is_noop() -> None:
    name = _limit_name()
    current = int(getattr(_dynamo.config, name))
    assert pin_recompile_limit(1) == current  # max() keeps the raised budget
    assert int(getattr(_dynamo.config, name)) == current
    assert pin_recompile_limit(0) == 0
    assert int(getattr(_dynamo.config, name)) == current


def test_activation_memory_budget_semantics() -> None:
    original = functorch_config.activation_memory_budget
    try:
        logs: list[str] = []
        # 0 = no-op
        assert apply_activation_memory_budget(0.0, gradient_checkpointing=False, log=logs.append) is False
        assert functorch_config.activation_memory_budget == original
        # out of range rejected
        assert apply_activation_memory_budget(1.5, gradient_checkpointing=False, log=logs.append) is False
        assert functorch_config.activation_memory_budget == original
        # grad-ckpt guard
        assert apply_activation_memory_budget(0.85, gradient_checkpointing=True, log=logs.append) is False
        assert functorch_config.activation_memory_budget == original
        assert any("skipped" in line for line in logs)
        # applies
        assert apply_activation_memory_budget(0.85, gradient_checkpointing=False, log=logs.append) is True
        assert functorch_config.activation_memory_budget == 0.85
    finally:
        functorch_config.activation_memory_budget = original


def test_plan_level_wiring() -> None:
    original = functorch_config.activation_memory_budget
    try:
        plan = RuntimeOptimizationPlan(
            attention_backend="sdpa",
            requested_attention_backend="sdpa",
            dynamo_recompile_limit=96,
            activation_memory_budget=0.9,
            gradient_checkpointing=False,
        )
        apply_dynamo_budgets_if_requested(plan)
        assert functorch_config.activation_memory_budget == 0.9
        assert any("pinned" in r for r in plan.reasons), plan.reasons
        assert any("activation_memory_budget" in r for r in plan.reasons), plan.reasons

        functorch_config.activation_memory_budget = original
        guarded = RuntimeOptimizationPlan(
            attention_backend="sdpa",
            requested_attention_backend="sdpa",
            activation_memory_budget=0.9,
            gradient_checkpointing=True,
        )
        apply_dynamo_budgets_if_requested(guarded)
        assert functorch_config.activation_memory_budget == original
        assert any("skipped" in r for r in guarded.reasons), guarded.reasons
    finally:
        functorch_config.activation_memory_budget = original


def main() -> int:
    test_pin_raises_override_and_cross_context_default()
    test_pin_never_lowers_and_zero_is_noop()
    test_activation_memory_budget_semantics()
    test_plan_level_wiring()
    print("dynamo_budget_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
