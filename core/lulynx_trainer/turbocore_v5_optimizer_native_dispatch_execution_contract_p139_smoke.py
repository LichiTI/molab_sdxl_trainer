"""Smoke checks for V5-P139 optimizer native-dispatch execution contract."""

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

from core.turbocore_v5_optimizer_native_dispatch_execution_contract_p139 import (  # noqa: E402
    SPEC,
    build_v5_optimizer_native_dispatch_execution_contract_p139,
)
from lulynx_trainer.turbocore_v5_optimizer_late_stage_smoke_utils import (  # noqa: E402
    build_optimizer_late_stage_ready_report,
    run_optimizer_late_stage_smoke,
)
from lulynx_trainer.turbocore_v5_optimizer_runtime_dispatch_execution_contract_p138_smoke import _p138_ready  # noqa: E402


def run_smoke() -> dict[str, Any]:
    return run_optimizer_late_stage_smoke(spec=SPEC, gate=_gate, previous_ready=_p138_ready)


def _gate(
    previous: dict[str, Any] | None,
    evidence: dict[str, Any] | None,
    review: dict[str, Any] | None,
    failure_history: list[Any] | None,
    rollback_history: list[Any] | None,
) -> dict[str, Any]:
    return build_v5_optimizer_native_dispatch_execution_contract_p139(
        p138_optimizer_runtime_dispatch_execution_contract=previous,
        optimizer_native_dispatch_execution_evidence=evidence,
        optimizer_native_dispatch_execution_signed_review=review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _p139_ready() -> dict[str, Any]:
    return build_optimizer_late_stage_ready_report(spec=SPEC, gate=_gate, previous_ready=_p138_ready)


if __name__ == "__main__":
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


__all__ = ["_p139_ready", "run_smoke"]

