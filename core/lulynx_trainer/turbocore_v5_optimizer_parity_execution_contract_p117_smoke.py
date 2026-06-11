"""Smoke checks for V5-P117 optimizer parity execution contract."""

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

from core.turbocore_v5_optimizer_parity_execution_contract_p117 import (  # noqa: E402
    SPEC,
    build_v5_optimizer_parity_execution_contract_p117,
)
from lulynx_trainer.turbocore_v5_optimizer_kernel_launch_execution_contract_p116_smoke import _p116_ready  # noqa: E402
from lulynx_trainer.turbocore_v5_optimizer_late_stage_smoke_utils import (  # noqa: E402
    build_optimizer_late_stage_ready_report,
    run_optimizer_late_stage_smoke,
)


def run_smoke() -> dict[str, Any]:
    return run_optimizer_late_stage_smoke(spec=SPEC, gate=_gate, previous_ready=_p116_ready)


def _gate(
    previous: dict[str, Any] | None,
    evidence: dict[str, Any] | None,
    review: dict[str, Any] | None,
    failure_history: list[Any] | None,
    rollback_history: list[Any] | None,
) -> dict[str, Any]:
    return build_v5_optimizer_parity_execution_contract_p117(
        p116_optimizer_kernel_launch_execution_contract=previous,
        optimizer_parity_execution_evidence=evidence,
        optimizer_parity_execution_signed_review=review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p117_ready() -> dict[str, Any]:
    return build_optimizer_late_stage_ready_report(spec=SPEC, gate=_gate, previous_ready=_p116_ready)


if __name__ == "__main__":
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


__all__ = ["_p117_ready", "run_smoke"]
