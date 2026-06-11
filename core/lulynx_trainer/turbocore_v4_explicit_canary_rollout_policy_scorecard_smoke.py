"""Smoke checks for V4 explicit canary rollout policy."""

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

from core.turbocore_v4_explicit_canary_rollout_policy_scorecard import (  # noqa: E402
    build_v4_explicit_canary_rollout_policy_scorecard,
)


def run_smoke() -> dict[str, Any]:
    p2_current = {"milestone_completed": True, "real_benchmark_result_ready": False}
    single = build_v4_explicit_canary_rollout_policy_scorecard(p2_audit=p2_current)
    assert single["ok"] is True, single
    assert single["explicit_canary_allowed"] is True, single
    assert single["larger_manual_canary_allowed"] is False, single
    assert single["default_rollout_allowed"] is False, single
    assert single["auto_rollout_allowed"] is False, single

    wider_blocked = build_v4_explicit_canary_rollout_policy_scorecard(
        p2_audit=p2_current,
        requested_scope="wider_manual_canary",
    )
    assert wider_blocked["ok"] is False, wider_blocked
    assert wider_blocked["route_decision"]["decision"] == "wider_canary_blocked_until_real_benchmark", wider_blocked
    assert "v4_p3_real_benchmark_result_missing" in wider_blocked["blocked_reasons"], wider_blocked

    p2_perf_blocked = {
        "milestone_completed": True,
        "real_benchmark_result_ready": False,
        "real_benchmark_input_present": True,
        "real_benchmark_executed": True,
        "real_benchmark_contract_ready": True,
        "real_benchmark_performance_gate_ready": False,
        "real_benchmark_status": "performance_gate_blocked",
        "real_benchmark_performance_blockers": ["end_to_end_speedup_below_threshold"],
    }
    wider_perf_blocked = build_v4_explicit_canary_rollout_policy_scorecard(
        p2_audit=p2_perf_blocked,
        requested_scope="wider_manual_canary",
    )
    assert wider_perf_blocked["ok"] is False, wider_perf_blocked
    assert (
        wider_perf_blocked["route_decision"]["decision"] == "wider_canary_blocked_until_performance_gate"
    ), wider_perf_blocked
    assert "v4_p3_real_benchmark_performance_gate_blocked" in wider_perf_blocked["blocked_reasons"], wider_perf_blocked
    assert "end_to_end_speedup_below_threshold" in wider_perf_blocked["blocked_reasons"], wider_perf_blocked

    p2_real = {"milestone_completed": True, "real_benchmark_result_ready": True}
    wider_ready = build_v4_explicit_canary_rollout_policy_scorecard(
        p2_audit=p2_real,
        requested_scope="wider_manual_canary",
    )
    assert wider_ready["ok"] is True, wider_ready
    assert wider_ready["larger_manual_canary_allowed"] is True, wider_ready
    assert wider_ready["auto_rollout_allowed"] is False, wider_ready

    auto = build_v4_explicit_canary_rollout_policy_scorecard(
        p2_audit=p2_real,
        native_training_mode="auto",
    )
    assert auto["ok"] is False, auto
    assert auto["route_decision"]["decision"] == "auto_blocked", auto
    assert "v4_p3_auto_rollout_blocked" in auto["blocked_reasons"], auto

    return {
        "schema_version": 1,
        "probe": "turbocore_v4_explicit_canary_rollout_policy_scorecard_smoke",
        "ok": True,
        "single_run_decision": single["route_decision"]["decision"],
        "wider_blocked_decision": wider_blocked["route_decision"]["decision"],
        "wider_perf_blocked_decision": wider_perf_blocked["route_decision"]["decision"],
        "wider_ready_decision": wider_ready["route_decision"]["decision"],
        "auto_decision": auto["route_decision"]["decision"],
        "recommended_next_step": single["recommended_next_step"],
    }


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
