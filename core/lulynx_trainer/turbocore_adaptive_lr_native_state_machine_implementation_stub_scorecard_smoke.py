"""Smoke checks for built-in adaptive-LR implementation stub artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_adaptive_lr_native_state_machine_cpu_reference_guard_scorecard import (  # noqa: E402
    build_adaptive_lr_native_state_machine_cpu_reference_guard_scorecard,
)
from core.turbocore_adaptive_lr_native_state_machine_implementation_stub_scorecard import (  # noqa: E402
    STUB_ENTRYPOINTS,
    build_adaptive_lr_native_state_machine_implementation_stub_scorecard,
)
from core.turbocore_adaptive_lr_state_machine_replay_executor_scorecard import TARGET_CASES  # noqa: E402


def run_smoke() -> dict[str, Any]:
    cpu_guard = build_adaptive_lr_native_state_machine_cpu_reference_guard_scorecard()
    payload = build_adaptive_lr_native_state_machine_implementation_stub_scorecard(cpu_guard_report=cpu_guard)
    rows = {str(row["optimizer_type"]): row for row in payload["rows"]}
    summary = payload["summary"]

    assert payload["scorecard"] == "turbocore_adaptive_lr_native_state_machine_implementation_stub_scorecard_v0", payload
    assert payload["ok"] is True, payload
    assert payload["native_state_machine_implementation_stub_ready"] is True, payload
    assert payload["promotion_ready"] is False, payload
    assert payload["stub_entrypoints"] == list(STUB_ENTRYPOINTS), payload
    assert payload["training_path_enabled"] is False, payload
    assert payload["runtime_dispatch_ready"] is False, payload
    assert payload["native_dispatch_allowed"] is False, payload
    assert payload["product_native_ready_count"] == 0, payload
    assert set(rows) == {case.optimizer.value for case in TARGET_CASES}, rows
    assert summary["target_count"] == len(TARGET_CASES), summary
    assert summary["implementation_stub_ready_count"] == len(TARGET_CASES), summary
    assert summary["stub_entrypoint_contract_ready_count"] == len(TARGET_CASES), summary
    assert summary["stub_state_transition_contract_ready_count"] == len(TARGET_CASES), summary
    assert summary["stub_dispatch_disabled_assertion_ready_count"] == len(TARGET_CASES), summary
    assert summary["state_machine_abi_implementation_ready_count"] == 0, summary
    assert summary["product_native_ready_count"] == 0, summary
    assert summary["runtime_dispatch_ready_count"] == 0, summary
    assert summary["native_dispatch_allowed_count"] == 0, summary

    for case in TARGET_CASES:
        row = rows[case.optimizer.value]
        stub = row["implementation_stub"]
        assert row["state_machine_status"] == "implementation_stub_ready", row
        assert row["implementation_stub_ready"] is True, row
        assert row["stub_entrypoint_contract_ready"] is True, row
        assert row["stub_state_transition_contract_ready"] is True, row
        assert row["stub_dispatch_disabled_assertion_ready"] is True, row
        assert row["state_machine_abi_implementation_ready"] is False, row
        assert row["native_route"] == "none_report_only", row
        assert row["training_path_enabled"] is False, row
        assert row["runtime_dispatch_ready"] is False, row
        assert row["native_dispatch_allowed"] is False, row
        assert stub["entrypoint_contract"]["entrypoints"] == STUB_ENTRYPOINTS, stub
        assert stub["entrypoint_contract"]["registration_policy"] == "not_registered_report_only", stub

    _write_real_artifact(payload)
    return {
        "schema_version": 1,
        "probe": "turbocore_adaptive_lr_native_state_machine_implementation_stub_scorecard_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "summary": summary,
        "recommended_next_step": payload["recommended_next_step"],
    }


def _write_real_artifact(payload: dict[str, Any]) -> None:
    temp_dir = REPO_ROOT / "temp" / "turbocore_optimizer"
    temp_dir.mkdir(parents=True, exist_ok=True)
    path = temp_dir / "turbocore_adaptive_lr_native_state_machine_implementation_stub_scorecard.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
