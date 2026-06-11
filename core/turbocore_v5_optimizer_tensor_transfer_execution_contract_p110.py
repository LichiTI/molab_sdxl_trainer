"""Optimizer tensor-transfer execution contract for TurboCore V5-P110."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

BACKEND_ROOT, REPO_ROOT = Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_optimizer_late_stage_contract_utils import build_optimizer_late_stage_contract
from core.turbocore_v5_optimizer_tensor_transfer_execution_specs_p110 import P110_SPEC as SPEC
from core.turbocore_v5_owner_review_evidence_package import load_json

P110_READY_DECISION = SPEC.ready_decision
P110_BLOCKED_DECISION = SPEC.blocked_decision
P110_HOLD_DECISION = SPEC.hold_decision
P110_REJECTED_DECISION = SPEC.rejected_decision
P110_SCOPE = SPEC.scope
REQUIRED_REVIEW_ACKS = SPEC.review_acks
REQUIRED_SECTIONS = SPEC.required_sections
UNSAFE_TRUE_FIELDS = SPEC.all_unsafe_true_fields
UNSAFE_NON_EMPTY_FIELDS = SPEC.all_unsafe_non_empty_fields


def build_v5_optimizer_tensor_transfer_execution_contract_p110(
    *,
    p109_optimizer_parity_execution_contract: Mapping[str, Any] | None = None,
    optimizer_tensor_transfer_execution_evidence: Mapping[str, Any] | None = None,
    optimizer_tensor_transfer_execution_signed_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    return build_optimizer_late_stage_contract(
        SPEC,
        previous_contract=p109_optimizer_parity_execution_contract,
        evidence=optimizer_tensor_transfer_execution_evidence,
        signed_review=optimizer_tensor_transfer_execution_signed_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _load_optional(path: str | None) -> dict[str, Any]:
    return load_json(Path(path)) if path else {}


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p109-optimizer-parity-execution-contract")
    parser.add_argument("--optimizer-tensor-transfer-execution-evidence")
    parser.add_argument("--optimizer-tensor-transfer-execution-signed-review")
    args = parser.parse_args(argv)
    report = build_v5_optimizer_tensor_transfer_execution_contract_p110(
        p109_optimizer_parity_execution_contract=_load_optional(args.p109_optimizer_parity_execution_contract),
        optimizer_tensor_transfer_execution_evidence=_load_optional(args.optimizer_tensor_transfer_execution_evidence),
        optimizer_tensor_transfer_execution_signed_review=_load_optional(
            args.optimizer_tensor_transfer_execution_signed_review
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "P110_SCOPE",
    "REQUIRED_REVIEW_ACKS",
    "REQUIRED_SECTIONS",
    "SPEC",
    "UNSAFE_NON_EMPTY_FIELDS",
    "UNSAFE_TRUE_FIELDS",
    "build_v5_optimizer_tensor_transfer_execution_contract_p110",
]
