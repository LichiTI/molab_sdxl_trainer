"""Profiled TurboCore optimizer smoke suite."""

from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import json
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
SCRIPT_ROOT = Path(__file__).resolve().parent
for import_root in (str(SCRIPT_ROOT), str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from turbocore_optimizer_smoke_scaffold import (  # noqa: E402
    build_scaffold_audit as build_scaffold_audit_payload,
    specialized_individual_turbocore_smoke_file_count,
    turbocore_related_smoke_files,
)


ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"


@dataclass(frozen=True)
class SmokeSpec:
    smoke_id: str
    module: str
    tier: str
    description: str


SMOKES: tuple[SmokeSpec, ...] = (
    SmokeSpec("selected_default_off_matrix", "turbocore_plugin_selected_default_off_matrix_scorecard_smoke", "quick", "Selected plugin family default-off matrix"),
    SmokeSpec(
        "runtime_rehearsal_matrix",
        "turbocore_plugin_runtime_rehearsal_matrix_smoke",
        "runtime",
        "Selected plugin runtime/precondition rehearsal matrix",
    ),
    SmokeSpec(
        "factored_custom_optimizer_family_batch",
        "turbocore_factored_custom_optimizer_family_batch_scorecard_smoke",
        "runtime",
        "Built-in factored/custom native canary chain batch",
    ),
    SmokeSpec(
        "native_kernel_inventory",
        "turbocore_optimizer_native_kernel_inventory_scorecard_smoke",
        "quick",
        "Selected plugin native kernel/probe inventory",
    ),
    SmokeSpec(
        "optimizer_family_kernel_contract",
        "turbocore_optimizer_family_kernel_contract_scorecard_smoke",
        "quick",
        "Shared optimizer family kernel contract native entrypoint",
    ),
    SmokeSpec(
        "optimizer_coverage_artifact",
        "turbocore_optimizer_coverage_artifact_scorecard_smoke",
        "coverage",
        "Optimizer coverage artifact validation",
    ),
    SmokeSpec(
        "optimizer_native_readiness_gap",
        "turbocore_optimizer_native_readiness_gap_scorecard_smoke",
        "coverage",
        "Optimizer native readiness gap by selected route family",
    ),
    SmokeSpec(
        "optimizer_multitensor_release_hold",
        "turbocore_native_update_optimizer_multitensor_release_hold_scorecard_smoke",
        "release",
        "Multi-tensor native optimizer release-hold evidence",
    ),
    SmokeSpec(
        "representative_performance_importer",
        "turbocore_native_update_representative_performance_importer_smoke",
        "release",
        "Imported representative performance standard artifacts",
    ),
    SmokeSpec(
        "representative_performance_summary",
        "turbocore_native_update_representative_performance_summary_smoke",
        "release",
        "Representative performance evidence summary",
    ),
    SmokeSpec(
        "release_review_package",
        "turbocore_native_update_release_review_package_smoke",
        "release",
        "Native-update release review package",
    ),
    SmokeSpec(
        "owner_release_handoff_summary",
        "turbocore_native_update_owner_release_handoff_summary_smoke",
        "release",
        "Compact owner-release handoff summary",
    ),
    SmokeSpec(
        "owner_release_review_packet",
        "turbocore_native_update_owner_release_review_packet_smoke",
        "release",
        "Owner release-review signable action packet",
    ),
    SmokeSpec("owner_release_review_record", "turbocore_native_update_owner_release_review_record_smoke", "release", "Signed owner release-review record validator"),
    SmokeSpec("product_training_route_binding_preflight", "turbocore_optimizer_product_training_route_binding_preflight_smoke", "release", "Default-off product training-route binding preflight"),
    SmokeSpec("product_training_route_binding_training_loop_contract", "turbocore_optimizer_product_training_route_binding_training_loop_contract_smoke", "release", "TrainingLoop post-approval route-binding context contract"),
    SmokeSpec("product_training_route_binding_config_adapter", "turbocore_optimizer_product_training_route_binding_config_adapter_smoke", "release", "TrainingLoop constructor kwargs route-binding adapter"),
    SmokeSpec(
        "release_review_archive",
        "turbocore_native_update_release_review_archive_smoke",
        "release",
        "Native-update release review archive",
    ),
    SmokeSpec(
        "promotion_scorecard",
        "turbocore_native_update_promotion_scorecard_smoke",
        "release",
        "Native-update promotion scorecard",
    ),
    SmokeSpec("p6_audit", "native_training_performance_p6_audit_smoke", "full", "Native training performance P6 audit"),
)

PROFILE_TIERS = {
    "quick": ("quick",),
    "runtime": ("quick", "runtime"),
    "coverage": ("quick", "coverage"),
    "release": ("quick", "release"),
    "batch": ("quick", "runtime", "coverage", "release"),
    "full": ("quick", "runtime", "coverage", "release", "full"),
}

PROFILE_GUIDANCE = {
    "quick": "Daily optimizer backend loop; validates default-off, inventory, and native family contract artifacts.",
    "runtime": "Use after optimizer runtime/precondition wiring changes.",
    "coverage": "Use after optimizer family coverage changes; validates existing coverage aggregation artifacts.",
    "release": "Use before owner/release handoff; keeps P6 and heavy coverage rebuilds out of the default path.",
    "batch": "Use after a batch of optimizer work; runs quick, runtime, coverage, and release without P6.",
    "full": "Use only when P6 native-training audit evidence is required.",
}

INDIVIDUAL_SMOKE_POLICY = [
    "Do not run per-family or per-optimizer smoke files during the normal loop.",
    "Start with this suite and a profile: quick, runtime, coverage, release, batch, or full.",
    "Use an individual smoke only after the suite identifies a failing smoke_id/module.",
    "Use rebuild entrypoints only when the underlying evidence artifact intentionally needs refresh.",
]


def build_suite(profile: str, *, include: set[str] | None = None, exclude: set[str] | None = None) -> list[SmokeSpec]:
    tiers = set(PROFILE_TIERS[profile])
    selected = [spec for spec in SMOKES if spec.tier in tiers]
    if include:
        selected = [spec for spec in selected if spec.smoke_id in include or spec.module in include]
    if exclude:
        selected = [spec for spec in selected if spec.smoke_id not in exclude and spec.module not in exclude]
    return selected


def build_suite_plan(profile: str = "quick") -> dict[str, Any]:
    selected = build_suite(profile)
    return {
        "schema_version": 1,
        "suite": "turbocore_optimizer_smoke_suite",
        "roadmap": ROADMAP,
        "profile": profile,
        "guidance": PROFILE_GUIDANCE[profile],
        "selected_count": len(selected),
        "selected_smokes": [spec.__dict__ for spec in selected],
        "profiles": {
            name: {
                "tiers": list(tiers),
                "guidance": PROFILE_GUIDANCE[name],
                "smoke_count": len(build_suite(name)),
            }
            for name, tiers in PROFILE_TIERS.items()
        },
        "artifact_policy": [
            *INDIVIDUAL_SMOKE_POLICY,
            "quick/coverage/release are artifact-first by default.",
        ],
        "explicit_rebuild_entrypoints": [
            "turbocore_plugin_selected_default_off_matrix_scorecard_smoke.py --rebuild-artifact",
            "turbocore_optimizer_native_kernel_inventory_scorecard_smoke.py --rebuild-artifact",
            "turbocore_optimizer_coverage_scorecard_smoke.py",
            "turbocore_optimizer_coverage_scorecard_smoke.py --rebuild-artifacts",
        ],
    }


def build_scaffold_audit(profile: str = "quick") -> dict[str, Any]:
    selected = build_suite(profile)
    return build_scaffold_audit_payload(
        repo_root=REPO_ROOT,
        script_root=SCRIPT_ROOT,
        roadmap=ROADMAP,
        profile=profile,
        profile_guidance=PROFILE_GUIDANCE[profile],
        selected_smokes=selected,
        all_smokes=SMOKES,
        profiles={
            name: {
                "tiers": list(tiers),
                "guidance": PROFILE_GUIDANCE[name],
                "smoke_count": len(build_suite(name)),
            }
            for name, tiers in PROFILE_TIERS.items()
        },
        workflow=INDIVIDUAL_SMOKE_POLICY,
    )


def run_suite(
    profile: str = "quick",
    *,
    include: set[str] | None = None,
    exclude: set[str] | None = None,
    continue_on_failure: bool = False,
    write_artifact: bool = True,
    artifact_path: Path | None = None,
) -> dict[str, Any]:
    selected = build_suite(profile, include=include, exclude=exclude)
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    for spec in selected:
        result = _run_one(spec)
        results.append(result)
        if not result["ok"] and not continue_on_failure:
            break
    failed = [item for item in results if not item["ok"]]
    status_summary = _suite_status_summary(results)
    report = {
        "schema_version": 1,
        "suite": "turbocore_optimizer_smoke_suite",
        "profile": profile,
        "roadmap": ROADMAP,
        "ok": not failed and len(results) == len(selected),
        "selected_count": len(selected),
        "executed_count": len(results),
        "passed_count": sum(1 for item in results if item["ok"]),
        "failed_count": len(failed),
        "skipped_count": len(selected) - len(results),
        "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
        "status_summary": status_summary,
        "results": results,
        "profile_guidance": PROFILE_GUIDANCE[profile],
        "artifact_policy": [
            *INDIVIDUAL_SMOKE_POLICY,
            "quick/coverage/release are artifact-first by default.",
        ],
        "notes": ["Use profile=batch after batched optimizer work; individual smokes follow suite failures only."],
    }
    if write_artifact:
        path = artifact_path or DEFAULT_ARTIFACT_DIR / f"turbocore_optimizer_smoke_suite_{profile}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        report["artifact_path"] = str(path)
    return report


def _run_one(spec: SmokeSpec) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        module = importlib.import_module(spec.module)
        payload = _call_smoke(module)
        ok = _payload_ok(payload)
        return {
            "smoke_id": spec.smoke_id,
            "module": spec.module,
            "tier": spec.tier,
            "description": spec.description,
            "ok": ok,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "payload_summary": _payload_summary(payload),
        }
    except Exception as exc:
        return {
            "smoke_id": spec.smoke_id,
            "module": spec.module,
            "tier": spec.tier,
            "description": spec.description,
            "ok": False,
            "elapsed_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }


def _call_smoke(module: Any) -> dict[str, Any]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        if hasattr(module, "run_smoke"):
            payload = module.run_smoke()
            if isinstance(payload, dict):
                payload = dict(payload)
                output = stdout.getvalue().strip()
                if output:
                    payload["_captured_stdout"] = output[-2000:]
                return payload
            return {"ok": False, "payload_type": type(payload).__name__}
        if hasattr(module, "main"):
            try:
                code = module.main()
            except SystemExit as exc:
                code = exc.code
            ok = code in (None, 0)
            payload = {
                "ok": ok,
                "probe": getattr(module, "__name__", ""),
                "main_exit_code": 0 if code is None else code,
            }
            output = stdout.getvalue().strip()
            if output:
                payload["_captured_stdout"] = output[-2000:]
            return payload
    return {"ok": False, "error": "smoke_module_has_no_run_smoke_or_main"}


def _payload_ok(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    return payload.get("ok") is not False


def _payload_summary(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {"payload_type": type(payload).__name__}
    keys = (
        "probe",
        "ok",
        "skipped",
        "roadmap",
        "artifact_mode",
        "case_count",
        "optimizer_count",
        "selected_plugin_optimizer_case_count",
        "family_specific_runtime_launch_adapter_ready_count",
        "runtime_dispatch_ready_count",
        "runtime_precondition_ready_count",
        "milestone_completed",
        "recommended_next_step",
    )
    summary = {key: payload[key] for key in keys if key in payload}
    nested = payload.get("summary")
    if isinstance(nested, dict):
        for key in (
            "case_count",
            "optimizer_count",
            "plugin_optimizer_count",
            "product_native_ready_count",
            "top_level_native_dispatch_allowed_count",
            "native_kernel_launch_count",
            "optimizer_family_contract_count",
            "required_family_present_count",
            "entrypoint_present_count",
            "kernel_source_present_count",
            "rust_probe_present_count",
            "family_count",
            "route_family_counts",
            "selected_family_counts",
            "selected_plugin_family_count",
            "product_native_ready_count",
            "runtime_dispatch_ready_count",
            "native_dispatch_allowed_count",
            "training_path_enabled_count",
            "route_family_count",
            "family_evidence_ready_count",
            "runtime_rehearsal_ready_family_count",
            "runtime_precondition_ready_family_count",
            "family_specific_runtime_launch_adapter_ready_count",
            "family_specific_runtime_launch_adapter_ready_family_count",
            "family_specific_runtime_launch_adapter_ready_optimizer_count",
            "runtime_launch_coverage_ready_family_count",
            "owner_release_hold_ready_family_count",
            "request_schema_ui_non_exposure_ready_family_count",
            "representative_runtime_rehearsal_ready_count",
            "representative_runtime_ready_family_count",
            "runtime_launch_absent_family_count",
            "family_specific_runtime_launch_missing_count",
            "product_training_route_missing_count",
            "owner_release_approval_missing_count",
            "owner_release_approval_recorded_count",
            "product_exposure_decision_recorded_count",
            "product_training_route_binding_ready_count",
            "open_training_path_enabled",
            "candidate_switch_count",
            "product_route_binding_preflight_ready_count", "training_loop_route_candidate_switch_count",
            "product_training_route_binding_config_patch_ready_count",
            "product_native_ready_family_count",
            "runtime_dispatch_ready_family_count",
            "native_dispatch_allowed_family_count",
            "training_path_enabled_family_count",
            "native_scratch_kernel_ready_count",
            "training_tensor_binding_canary_ready_count",
            "runtime_dispatch_adapter_shadow_ready_count",
            "training_loop_canary_ready_count",
            "e2e_shadow_matrix_ready_count",
            "canary_rollout_policy_ready_count",
            "dispatch_integration_review_ready_count",
        ):
            if key in nested:
                summary[key] = nested[key]
    return summary


def _suite_status_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    payloads = [
        item.get("payload_summary")
        for item in results
        if item.get("ok") and isinstance(item.get("payload_summary"), dict)
    ]
    return {
        "roadmap": ROADMAP,
        "profile_entrypoint": "turbocore_optimizer_smoke_suite.py",
        "turbocore_related_smoke_file_count": len(turbocore_related_smoke_files(SCRIPT_ROOT)),
        "optimizer_suite_registered_smoke_count": len(SMOKES),
        "factored_custom_optimizer_count": _max_int(
            [payload for payload in payloads if _probe(payload) == "turbocore_factored_custom_optimizer_family_batch_scorecard_smoke"],
            "optimizer_count",
        ),
        "factored_custom_training_loop_canary_ready_count": _max_int(
            payloads,
            "training_loop_canary_ready_count",
        ),
        "factored_custom_dispatch_review_ready_count": _max_int(
            payloads,
            "dispatch_integration_review_ready_count",
        ),
        "specialized_individual_turbocore_smoke_file_count": specialized_individual_turbocore_smoke_file_count(
            SCRIPT_ROOT,
            (spec.module for spec in SMOKES),
        ),
        "selected_plugin_optimizer_count": _max_int(payloads, "plugin_optimizer_count", "optimizer_count"),
        "selected_plugin_family_count": _max_int(
            payloads,
            "selected_plugin_family_count",
            "family_count",
            "case_count",
            "required_family_present_count",
            "optimizer_family_contract_count",
        ),
        "optimizer_family_kernel_contract_ready_count": _max_int(
            payloads,
            "optimizer_family_contract_count",
            "optimizer_family_kernel_contract_ready_count",
        ),
        "native_readiness_family_evidence_ready_count": _max_int(payloads, "family_evidence_ready_count"),
        "native_readiness_runtime_rehearsal_ready_family_count": _max_int(
            payloads,
            "runtime_rehearsal_ready_family_count",
        ),
        "native_readiness_runtime_precondition_ready_family_count": _max_int(
            payloads,
            "runtime_precondition_ready_family_count",
        ),
        "native_readiness_runtime_launch_adapter_ready_family_count": _max_int(payloads, "family_specific_runtime_launch_adapter_ready_family_count"),
        "native_readiness_runtime_launch_adapter_ready_optimizer_count": _max_int(payloads, "family_specific_runtime_launch_adapter_ready_optimizer_count", "family_specific_runtime_launch_adapter_ready_count"),
        "native_readiness_runtime_launch_coverage_ready_family_count": _max_int(payloads, "runtime_launch_coverage_ready_family_count"),
        "native_readiness_owner_release_hold_ready_family_count": _max_int(payloads, "owner_release_hold_ready_family_count"),
        "native_readiness_request_schema_ui_non_exposure_ready_family_count": _max_int(payloads, "request_schema_ui_non_exposure_ready_family_count"),
        "native_readiness_representative_runtime_rehearsal_ready_count": _max_int(
            payloads,
            "representative_runtime_rehearsal_ready_count",
        ),
        "native_readiness_representative_runtime_ready_family_count": _max_int(
            payloads,
            "representative_runtime_ready_family_count",
        ),
        "native_readiness_runtime_launch_absent_family_count": _max_int(
            payloads,
            "runtime_launch_absent_family_count",
        ),
        "native_readiness_runtime_launch_missing_count": _max_int(
            payloads,
            "family_specific_runtime_launch_missing_count",
        ),
        "native_readiness_product_training_route_missing_count": _max_int(
            payloads,
            "product_training_route_missing_count",
        ),
        "native_readiness_owner_release_approval_missing_count": _max_int(
            payloads,
            "owner_release_approval_missing_count",
        ),
        "owner_release_approval_recorded_count": _max_int(payloads, "owner_release_approval_recorded_count"),
        "product_exposure_decision_recorded_count": _max_int(payloads, "product_exposure_decision_recorded_count"),
        "product_training_route_binding_ready_count": _max_int(payloads, "product_training_route_binding_ready_count"),
        "training_loop_contract_open_training_path_enabled_count": _max_int(payloads, "open_training_path_enabled"),
        "training_loop_contract_candidate_switch_count": _max_int(payloads, "candidate_switch_count"),
        "native_kernel_inventory_source_ready_count": _max_int(
            payloads,
            "optimizer_native_kernel_inventory_source_ready_count",
            "kernel_source_present_count",
        ),
        "native_kernel_inventory_probe_ready_count": _max_int(
            payloads,
            "optimizer_native_kernel_inventory_probe_ready_count",
            "rust_probe_present_count",
        ),
        "product_native_ready_count": _max_int(payloads, "product_native_ready_count"),
        "runtime_dispatch_ready_count": _max_int(payloads, "runtime_dispatch_ready_count"),
        "native_dispatch_allowed_count": _max_int(payloads, "native_dispatch_allowed_count"),
        "training_path_enabled_count": _max_int(payloads, "training_path_enabled_count"),
        "route_family_counts": _first_mapping(payloads, "route_family_counts"),
        "selected_family_counts": _first_mapping(payloads, "selected_family_counts"),
        "workflow": (
            "Use this profiled suite first. Run individual smoke files only after "
            "the suite identifies the failing area; use explicit rebuild entrypoints "
            "only when artifacts must be refreshed."
        ),
    }


def _max_int(payloads: list[Any], *keys: str) -> int:
    values: list[int] = []
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        for key in keys:
            try:
                values.append(int(payload.get(key, 0) or 0))
            except (TypeError, ValueError):
                continue
    return max(values, default=0)


def _first_mapping(payloads: list[Any], key: str) -> dict[str, int]:
    for payload in payloads:
        value = payload.get(key) if isinstance(payload, dict) else None
        if not isinstance(value, dict):
            continue
        out: dict[str, int] = {}
        for item_key, item_value in value.items():
            try:
                out[str(item_key)] = int(item_value or 0)
            except (TypeError, ValueError):
                continue
        if out:
            return dict(sorted(out.items()))
    return {}


def _probe(payload: Any) -> str:
    return str(payload.get("probe", "") or "") if isinstance(payload, dict) else ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(PROFILE_TIERS), default="quick")
    parser.add_argument("--include", action="append", default=[], help="Smoke id or module to include.")
    parser.add_argument("--exclude", action="append", default=[], help="Smoke id or module to exclude.")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--no-artifact", action="store_true")
    parser.add_argument("--artifact-path", type=Path)
    parser.add_argument("--list", action="store_true", help="List selected smokes without running them.")
    parser.add_argument("--plan", action="store_true", help="Print profile guidance without running smokes.")
    parser.add_argument("--audit-scaffold", action="store_true", help="Inspect smoke fragmentation without running smokes.")
    args = parser.parse_args(argv)
    include = set(args.include) or None
    exclude = set(args.exclude) or None
    if args.plan:
        print(json.dumps(build_suite_plan(args.profile), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.audit_scaffold:
        print(json.dumps(build_scaffold_audit(args.profile), ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    if args.list:
        selected = build_suite(args.profile, include=include, exclude=exclude)
        print(json.dumps([spec.__dict__ for spec in selected], ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    report = run_suite(
        args.profile,
        include=include,
        exclude=exclude,
        continue_on_failure=bool(args.continue_on_failure),
        write_artifact=not bool(args.no_artifact),
        artifact_path=args.artifact_path,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
