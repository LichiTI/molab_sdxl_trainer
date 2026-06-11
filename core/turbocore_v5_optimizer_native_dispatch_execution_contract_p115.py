"""Optimizer native-dispatch execution contract for TurboCore V5-P115."""

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
from core.turbocore_v5_optimizer_native_dispatch_execution_specs_p115 import P115_SPEC as SPEC
from core.turbocore_v5_owner_review_evidence_package import load_json

P115_READY_DECISION = SPEC.ready_decision
P115_BLOCKED_DECISION = SPEC.blocked_decision
P115_HOLD_DECISION = SPEC.hold_decision
P115_REJECTED_DECISION = SPEC.rejected_decision
P115_SCOPE = SPEC.scope
REQUIRED_REVIEW_ACKS = SPEC.review_acks
REQUIRED_SECTIONS = SPEC.required_sections
UNSAFE_TRUE_FIELDS = SPEC.all_unsafe_true_fields
UNSAFE_NON_EMPTY_FIELDS = SPEC.all_unsafe_non_empty_fields


def build_v5_optimizer_native_dispatch_execution_contract_p115(
    *,
    p114_optimizer_runtime_dispatch_execution_contract: Mapping[str, Any] | None = None,
    optimizer_native_dispatch_execution_evidence: Mapping[str, Any] | None = None,
    optimizer_native_dispatch_execution_signed_review: Mapping[str, Any] | None = None,
    failure_history: Sequence[Any] | None = None,
    rollback_history: Sequence[Any] | None = None,
) -> dict[str, Any]:
    return build_optimizer_late_stage_contract(
        SPEC,
        previous_contract=p114_optimizer_runtime_dispatch_execution_contract,
        evidence=optimizer_native_dispatch_execution_evidence,
        signed_review=optimizer_native_dispatch_execution_signed_review,
        failure_history=failure_history,
        rollback_history=rollback_history,
    )


def _load_optional(path: str | None) -> dict[str, Any]:
    return load_json(Path(path)) if path else {}


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--p114-optimizer-runtime-dispatch-execution-contract")
    parser.add_argument("--optimizer-native-dispatch-execution-evidence")
    parser.add_argument("--optimizer-native-dispatch-execution-signed-review")
    args = parser.parse_args(argv)
    report = build_v5_optimizer_native_dispatch_execution_contract_p115(
        p114_optimizer_runtime_dispatch_execution_contract=_load_optional(
            args.p114_optimizer_runtime_dispatch_execution_contract
        ),
        optimizer_native_dispatch_execution_evidence=_load_optional(args.optimizer_native_dispatch_execution_evidence),
        optimizer_native_dispatch_execution_signed_review=_load_optional(
            args.optimizer_native_dispatch_execution_signed_review
        ),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(_main())


__all__ = [
    "P115_SCOPE",
    "REQUIRED_REVIEW_ACKS",
    "REQUIRED_SECTIONS",
    "SPEC",
    "UNSAFE_NON_EMPTY_FIELDS",
    "UNSAFE_TRUE_FIELDS",
    "build_v5_optimizer_native_dispatch_execution_contract_p115",
]
