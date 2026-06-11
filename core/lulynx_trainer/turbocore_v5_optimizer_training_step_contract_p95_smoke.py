"""Smoke checks for V5-P95 optimizer training-step contract."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
CORE_ROOT = BACKEND_ROOT / "core"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT), str(CORE_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_optimizer_training_step_contract_p95 import (  # noqa: E402
    SPEC,
    build_v5_optimizer_training_step_contract_p95,
)
from lulynx_trainer.turbocore_v5_optimizer_late_stage_smoke_utils import (  # noqa: E402
    build_optimizer_late_stage_ready_report,
    run_optimizer_late_stage_smoke,
)
from lulynx_trainer.turbocore_v5_optimizer_parity_contract_p94_smoke import (  # noqa: E402
    _gate as _p94_gate,
    _p93_ready,
)


def run_smoke() -> dict[str, Any]:
    return run_optimizer_late_stage_smoke(spec=SPEC, gate=_gate, previous_ready=_p94_ready)


def _gate(
    previous: dict[str, Any] | None,
    evidence: dict[str, Any] | None,
    review: dict[str, Any] | None,
    failure_history: list[Any] | None,
    rollback_history: list[Any] | None,
) -> dict[str, Any]:
    return build_v5_optimizer_training_step_contract_p95(
        p94_optimizer_parity_contract=previous,
        optimizer_training_step_evidence=evidence,
        optimizer_training_step_signed_review=review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p94_ready() -> dict[str, Any]:
    return _p94_gate(_p93_ready())


def _p95_ready() -> dict[str, Any]:
    return build_optimizer_late_stage_ready_report(spec=SPEC, gate=_gate, previous_ready=_p94_ready)


if __name__ == "__main__":
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


__all__ = ["_p95_ready", "run_smoke"]
