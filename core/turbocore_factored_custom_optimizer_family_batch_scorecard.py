"""Batch scorecard for built-in factored/custom TurboCore optimizer gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from core.turbocore_adafactor_e2e_shadow_matrix_scorecard import (
    build_adafactor_e2e_shadow_matrix_scorecard,
)
from core.turbocore_adafactor_explicit_canary_rollout_policy_scorecard import (
    build_adafactor_explicit_canary_rollout_policy_scorecard,
)
from core.turbocore_adafactor_native_scratch_kernel_scorecard import (
    build_adafactor_native_scratch_kernel_scorecard,
)
from core.turbocore_adafactor_real_dispatch_integration_review_scorecard import (
    build_adafactor_real_dispatch_integration_review_scorecard,
)
from core.turbocore_adafactor_runtime_dispatch_adapter_shadow_scorecard import (
    build_adafactor_runtime_dispatch_adapter_shadow_scorecard,
)
from core.turbocore_adafactor_training_loop_canary_scorecard import (
    build_adafactor_training_loop_canary_scorecard,
)
from core.turbocore_adafactor_training_tensor_binding_canary_scorecard import (
    build_adafactor_training_tensor_binding_canary_scorecard,
)
from core.turbocore_anima_factored_adamw_e2e_shadow_matrix_scorecard import (
    build_anima_factored_adamw_e2e_shadow_matrix_scorecard,
)
from core.turbocore_anima_factored_adamw_explicit_canary_rollout_policy_scorecard import (
    build_anima_factored_adamw_explicit_canary_rollout_policy_scorecard,
)
from core.turbocore_anima_factored_adamw_native_scratch_kernel_scorecard import (
    build_anima_factored_adamw_native_scratch_kernel_scorecard,
)
from core.turbocore_anima_factored_adamw_real_dispatch_integration_review_scorecard import (
    build_anima_factored_adamw_real_dispatch_integration_review_scorecard,
)
from core.turbocore_anima_factored_adamw_runtime_dispatch_adapter_shadow_scorecard import (
    build_anima_factored_adamw_runtime_dispatch_adapter_shadow_scorecard,
)
from core.turbocore_anima_factored_adamw_training_loop_canary_scorecard import (
    build_anima_factored_adamw_training_loop_canary_scorecard,
)
from core.turbocore_anima_factored_adamw_training_tensor_binding_canary_scorecard import (
    build_anima_factored_adamw_training_tensor_binding_canary_scorecard,
)
from core.turbocore_automagicpp_e2e_shadow_matrix_scorecard import (
    build_automagicpp_e2e_shadow_matrix_scorecard,
)
from core.turbocore_automagicpp_explicit_canary_rollout_policy_scorecard import (
    build_automagicpp_explicit_canary_rollout_policy_scorecard,
)
from core.turbocore_automagicpp_native_scratch_kernel_scorecard import (
    build_automagicpp_native_scratch_kernel_scorecard,
)
from core.turbocore_automagicpp_real_dispatch_integration_review_scorecard import (
    build_automagicpp_real_dispatch_integration_review_scorecard,
)
from core.turbocore_automagicpp_runtime_dispatch_adapter_shadow_scorecard import (
    build_automagicpp_runtime_dispatch_adapter_shadow_scorecard,
)
from core.turbocore_automagicpp_training_loop_canary_scorecard import (
    build_automagicpp_training_loop_canary_scorecard,
)
from core.turbocore_automagicpp_training_tensor_binding_canary_scorecard import (
    build_automagicpp_training_tensor_binding_canary_scorecard,
)
from core.turbocore_factored_custom_optimizer_state_layout_scorecard import (
    build_factored_custom_optimizer_state_layout_scorecard,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "temp" / "turbocore_optimizer" / "turbocore_factored_custom_optimizer_family_batch_scorecard.json"


def build_factored_custom_optimizer_family_batch_scorecard(
    *,
    workspace_root: str | Path | None = None,
    arch: str | None = None,
    run_live_tensor_binding_canaries: bool = True,
    include_live_training_loop_canaries: bool = True,
    write_artifact: bool = False,
) -> dict[str, Any]:
    """Aggregate built-in factored/custom evidence without enabling dispatch."""

    root = Path(workspace_root or REPO_ROOT).resolve()
    state_layout = build_factored_custom_optimizer_state_layout_scorecard(write_artifact=write_artifact)
    rows = [
        _adafactor_row(root, arch, run_live_tensor_binding_canaries, include_live_training_loop_canaries),
        _automagicpp_row(root, arch, run_live_tensor_binding_canaries, include_live_training_loop_canaries),
        _anima_row(root, arch, run_live_tensor_binding_canaries, include_live_training_loop_canaries),
    ]
    unsafe = [reason for row in rows for reason in row["unsafe_reasons"]]
    blockers = [reason for row in rows for reason in row["blocked_reasons"]]
    ready = all(row["dispatch_integration_review_ready"] for row in rows) and not unsafe
    report = {
        "schema_version": 1,
        "scorecard": "turbocore_factored_custom_optimizer_family_batch_scorecard_v0",
        "gate": "factored_custom_optimizer_family_batch",
        "ok": ready,
        "promotion_ready": False,
        "factored_custom_family_batch_ready": ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_ready_count": 0,
        "state_layout_summary": dict(state_layout.get("summary") or {}),
        "rows": rows,
        "summary": _summary(rows),
        "promotion_blockers": _dedupe(blockers + ["factored_custom_owner_release_approval_missing"]),
        "blocked_reasons": _dedupe(blockers + unsafe),
        "recommended_next_step": (
            "record explicit factored/custom owner/release approval before any product dispatch"
            if ready
            else "finish factored/custom native canary chain blockers"
        ),
        "notes": [
            "This batch aggregates existing per-optimizer gates only.",
            "Real runtime dispatch, product TrainingLoop dispatch, request fields, schema, and UI exposure stay disabled.",
            "Dispatch review readiness is not owner/release approval.",
        ],
    }
    if write_artifact:
        ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
        ARTIFACT.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return report


def _adafactor_row(
    root: Path,
    arch: str | None,
    run_live_tensor_binding_canaries: bool,
    include_live_training_loop_canaries: bool,
) -> dict[str, Any]:
    scratch = build_adafactor_native_scratch_kernel_scorecard(workspace_root=root, arch=arch)
    tensor = build_adafactor_training_tensor_binding_canary_scorecard(
        scratch_report=scratch,
        run_live_probe=run_live_tensor_binding_canaries,
        workspace_root=root,
        arch=arch,
    )
    runtime = build_adafactor_runtime_dispatch_adapter_shadow_scorecard(
        p35_audit_report=_tensor_audit(
            "build_p35_adafactor_training_tensor_binding_audit",
            "p35_audit_builder",
            "adafactor_training_tensor_binding_canary",
            tensor,
        )
    )
    loop = build_adafactor_training_loop_canary_scorecard() if include_live_training_loop_canaries else _skipped_loop()
    e2e = build_adafactor_e2e_shadow_matrix_scorecard(
        adapter_report=runtime,
        p37_audit_report=_stage_audit("build_p37_adafactor_training_loop_canary_audit", "p37_audit_builder", loop),
    )
    rollout = build_adafactor_explicit_canary_rollout_policy_scorecard(
        p38_audit_report=_stage_audit("build_p38_adafactor_e2e_shadow_matrix_audit", "p38_audit_builder", e2e)
    )
    review = build_adafactor_real_dispatch_integration_review_scorecard(
        p39_audit_report=_stage_audit("build_p39_adafactor_explicit_canary_rollout_policy_audit", "p39_audit_builder", rollout)
    )
    return _row("adafactor", scratch, tensor, runtime, loop, e2e, rollout, review)


def _automagicpp_row(
    root: Path,
    arch: str | None,
    run_live_tensor_binding_canaries: bool,
    include_live_training_loop_canaries: bool,
) -> dict[str, Any]:
    scratch = build_automagicpp_native_scratch_kernel_scorecard(workspace_root=root, arch=arch)
    tensor = build_automagicpp_training_tensor_binding_canary_scorecard(
        scratch_report=scratch,
        run_live_probe=run_live_tensor_binding_canaries,
        workspace_root=root,
        arch=arch,
    )
    runtime = build_automagicpp_runtime_dispatch_adapter_shadow_scorecard(tensor_binding_report=tensor)
    loop = build_automagicpp_training_loop_canary_scorecard() if include_live_training_loop_canaries else _skipped_loop()
    e2e = build_automagicpp_e2e_shadow_matrix_scorecard(
        adapter_report=runtime,
        p23_audit_report=_stage_audit("build_p23_automagicpp_training_loop_canary_audit", "p23_audit_builder", loop),
    )
    rollout = build_automagicpp_explicit_canary_rollout_policy_scorecard(
        p31_audit_report=_stage_audit("build_p31_automagicpp_e2e_shadow_matrix_audit", "p31_audit_builder", e2e)
    )
    review = build_automagicpp_real_dispatch_integration_review_scorecard(
        p32_audit_report=_stage_audit("build_p32_automagicpp_explicit_canary_rollout_policy_audit", "p32_audit_builder", rollout)
    )
    return _row("Automagic++", scratch, tensor, runtime, loop, e2e, rollout, review)


def _anima_row(
    root: Path,
    arch: str | None,
    run_live_tensor_binding_canaries: bool,
    include_live_training_loop_canaries: bool,
) -> dict[str, Any]:
    scratch = build_anima_factored_adamw_native_scratch_kernel_scorecard(workspace_root=root, arch=arch)
    tensor = build_anima_factored_adamw_training_tensor_binding_canary_scorecard(
        scratch_report=scratch,
        run_live_probe=run_live_tensor_binding_canaries,
        workspace_root=root,
        arch=arch,
    )
    runtime = build_anima_factored_adamw_runtime_dispatch_adapter_shadow_scorecard(
        p25_audit_report=_tensor_audit(
            "build_p25_anima_factored_adamw_training_tensor_binding_audit",
            "p25_audit_builder",
            "anima_factored_adamw_training_tensor_binding_canary",
            tensor,
        )
    )
    loop = build_anima_factored_adamw_training_loop_canary_scorecard() if include_live_training_loop_canaries else _skipped_loop()
    e2e = build_anima_factored_adamw_e2e_shadow_matrix_scorecard(
        adapter_report=runtime,
        p27_audit_report=_stage_audit(
            "build_p27_anima_factored_adamw_training_loop_canary_audit",
            "p27_audit_builder",
            loop,
        ),
    )
    rollout = build_anima_factored_adamw_explicit_canary_rollout_policy_scorecard(
        p28_audit_report=_stage_audit("build_p28_anima_factored_adamw_e2e_shadow_matrix_audit", "p28_audit_builder", e2e)
    )
    review = build_anima_factored_adamw_real_dispatch_integration_review_scorecard(
        p29_audit_report=_stage_audit(
            "build_p29_anima_factored_adamw_explicit_canary_rollout_policy_audit",
            "p29_audit_builder",
            rollout,
        )
    )
    return _row("AnimaFactoredAdamW", scratch, tensor, runtime, loop, e2e, rollout, review)


def _row(
    optimizer_type: str,
    scratch: Mapping[str, Any],
    tensor: Mapping[str, Any],
    runtime: Mapping[str, Any],
    loop: Mapping[str, Any],
    e2e: Mapping[str, Any],
    rollout: Mapping[str, Any],
    review: Mapping[str, Any],
) -> dict[str, Any]:
    stage_ready = {
        "native_scratch_kernel_ready": _ready(scratch, "native_kernel_ready"),
        "training_tensor_binding_canary_ready": _ready(tensor, "training_tensor_binding_canary_ready"),
        "runtime_dispatch_adapter_shadow_ready": _ready(runtime, "runtime_dispatch_adapter_shadow_ready"),
        "training_loop_canary_ready": _ready(loop, "ok"),
        "e2e_shadow_matrix_ready": _ready(e2e, "e2e_shadow_matrix_ready"),
        "canary_rollout_policy_ready": _ready(rollout, "canary_rollout_policy_ready"),
        "dispatch_integration_review_ready": _ready(review, "review_gate_ready"),
    }
    chain = _chain_ready(stage_ready)
    reports = {
        "native_scratch_kernel": scratch,
        "training_tensor_binding": tensor,
        "runtime_dispatch_adapter_shadow": runtime,
        "training_loop_canary": loop,
        "e2e_shadow_matrix": e2e,
        "canary_rollout_policy": rollout,
        "dispatch_integration_review": review,
    }
    unsafe = _unsafe_reasons(reports)
    return {
        "schema_version": 1,
        "optimizer_type": optimizer_type,
        "optimizer_family": "factored_custom",
        "batch_status": _batch_status(chain),
        "next_gate": "record_explicit_factored_custom_owner_release_approval",
        **chain,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "product_native_ready": False,
        "report_summaries": {name: dict(_as_dict(report.get("summary"))) for name, report in reports.items()},
        "unsafe_reasons": unsafe,
        "blocked_reasons": _dedupe(
            reason for report in reports.values() for reason in _strings(report.get("blocked_reasons"))
        ),
    }


def _chain_ready(stage_ready: Mapping[str, bool]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    previous = True
    for key, ready in stage_ready.items():
        previous = previous and bool(ready)
        out[key] = previous
    return out


def _batch_status(chain: Mapping[str, bool]) -> str:
    for key in (
        "dispatch_integration_review_ready",
        "canary_rollout_policy_ready",
        "e2e_shadow_matrix_ready",
        "training_loop_canary_ready",
        "runtime_dispatch_adapter_shadow_ready",
        "training_tensor_binding_canary_ready",
        "native_scratch_kernel_ready",
    ):
        if chain.get(key):
            return f"factored_custom_{key}"
    return "factored_custom_state_layout_reference_ready"


def _summary(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    keys = (
        "native_scratch_kernel_ready",
        "training_tensor_binding_canary_ready",
        "runtime_dispatch_adapter_shadow_ready",
        "training_loop_canary_ready",
        "e2e_shadow_matrix_ready",
        "canary_rollout_policy_ready",
        "dispatch_integration_review_ready",
    )
    summary = {f"{key}_count": sum(1 for row in rows if row.get(key) is True) for key in keys}
    summary.update(
        {
            "optimizer_count": len(rows),
            "product_native_ready_count": 0,
            "runtime_dispatch_ready_count": 0,
            "native_dispatch_allowed_count": 0,
            "training_path_enabled_count": 0,
            "unsafe_claim_count": sum(len(row.get("unsafe_reasons", []) or []) for row in rows),
        }
    )
    return summary


def _tensor_audit(builder: str, summary_key: str, section_key: str, report: Mapping[str, Any]) -> dict[str, Any]:
    ready = bool(report.get("training_tensor_binding_canary_ready", False))
    return {
        "schema_version": 1,
        "ok": ready,
        "milestone_completed": ready,
        "dependency_builder": builder,
        "audit_builder": builder,
        "progress_gates": {"training_tensor_binding_canary": ready},
        "sections": {section_key: dict(report)},
        "summary": {summary_key: builder},
        "training_path_enabled": False,
        "default_behavior_changed": False,
    }


def _stage_audit(builder: str, summary_key: str, report: Mapping[str, Any]) -> dict[str, Any]:
    ready = bool(report.get("ok", False))
    return {
        "schema_version": 1,
        "ok": ready,
        "milestone_completed": ready,
        "dependency_builder": builder,
        "audit_builder": builder,
        "progress_gates": {
            "training_loop_native_canary": ready,
            "training_loop_native_canary_dependency_named": ready,
            "e2e_shadow_matrix_scaffold": ready,
            "explicit_canary_policy": ready,
            "manual_review_required": True,
            "fallback_rollback_ready": True,
            "runtime_dispatch_not_enabled": True,
            "default_behavior_unchanged": True,
        },
        "summary": {summary_key: builder, "fallback_backend_authoritative": True},
        "manual_review_required": True,
        "fallback_rollback_ready": True,
        "canary_auto_enabled": False,
        "runtime_dispatch_not_enabled": True,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "default_behavior_unchanged": True,
    }


def _skipped_loop() -> dict[str, Any]:
    return {"schema_version": 1, "ok": False, "blocked_reasons": ["training_loop_canary_skipped"]}


def _ready(report: Mapping[str, Any], key: str) -> bool:
    return bool(report.get("ok", False)) and bool(report.get(key, False))


def _unsafe_reasons(reports: Mapping[str, Mapping[str, Any]]) -> list[str]:
    reasons: list[str] = []
    for name, report in reports.items():
        if report.get("training_path_enabled") is True:
            reasons.append(f"{name}:training_path_enabled")
        if report.get("default_behavior_changed") is True:
            reasons.append(f"{name}:default_behavior_changed")
        if report.get("runtime_dispatch_ready") is True:
            reasons.append(f"{name}:runtime_dispatch_ready")
        if report.get("native_dispatch_allowed") is True:
            reasons.append(f"{name}:native_dispatch_allowed")
    return reasons


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _strings(value: Any) -> list[str]:
    return [str(item) for item in value or [] if str(item or "")]


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["ARTIFACT", "build_factored_custom_optimizer_family_batch_scorecard"]
