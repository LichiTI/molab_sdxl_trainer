"""Scaffold audit helpers for the TurboCore optimizer smoke suite."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Sequence


_SMOKE_REF_RE = re.compile(r"[\w./\\-]*_smoke\.py")
_SPECIALIZED_BUCKETS = (
    (
        "explicit_artifact_or_release_refresh",
        (
            "artifact",
            "archive",
            "approval",
            "coverage",
            "handoff",
            "owner",
            "package",
            "promotion",
            "release",
            "review",
            "summary",
        ),
    ),
    (
        "runtime_native_failure_localization",
        (
            "cuda",
            "dispatch",
            "kernel",
            "native",
            "runtime",
            "tensor",
            "training_launch",
            "training_loop",
        ),
    ),
    (
        "family_optimizer_failure_localization",
        (
            "adam",
            "candidate",
            "factored",
            "family",
            "optimizer",
            "plugin",
            "schedule",
            "simple",
            "variant",
        ),
    ),
)


def turbocore_related_smoke_files(script_root: Path) -> list[str]:
    return sorted(path.name for path in script_root.glob("*turbocore*smoke*.py") if path.is_file())


def specialized_individual_turbocore_smoke_file_count(script_root: Path, smoke_modules: Iterable[str]) -> int:
    suite_module_files = {f"{module}.py" for module in smoke_modules}
    suite_module_files.add("turbocore_optimizer_smoke_suite.py")
    return sum(1 for name in turbocore_related_smoke_files(script_root) if name not in suite_module_files)


def build_scaffold_audit(
    *,
    repo_root: Path,
    script_root: Path,
    roadmap: str,
    profile: str,
    profile_guidance: str,
    selected_smokes: Sequence[Any],
    all_smokes: Sequence[Any],
    profiles: dict[str, dict[str, Any]],
    include_groups: dict[str, list[str]],
    available_include_groups: dict[str, list[str]] | None = None,
    include_group_policies: dict[str, dict[str, Any]] | None = None,
    workflow: Sequence[str],
) -> dict[str, Any]:
    suite_modules = tuple(str(spec.module) for spec in all_smokes)
    suite_module_files = {f"{module}.py" for module in suite_modules}
    suite_module_files.add("turbocore_optimizer_smoke_suite.py")

    turbocore_files = turbocore_related_smoke_files(script_root)
    suite_managed_turbocore_files = sorted(name for name in turbocore_files if name in suite_module_files)
    specialized_files = sorted(name for name in turbocore_files if name not in suite_module_files)

    roadmap_files = _roadmap_referenced_smoke_files(repo_root, script_root, roadmap)
    roadmap_suite_files = sorted(name for name in roadmap_files if name in suite_module_files)
    roadmap_specialized_files = sorted(name for name in roadmap_files if name not in suite_module_files)
    recommended_command_policy_table = _recommended_command_policy_table()
    default_recommended_commands = _recommended_commands_by_default_policy(
        recommended_command_policy_table,
        execute_by_default=True,
    )
    deferred_recommended_commands = _recommended_commands_by_default_policy(
        recommended_command_policy_table,
        execute_by_default=False,
    )

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_smoke_scaffold_audit",
        "ok": True,
        "roadmap": roadmap,
        "suite_entrypoint": "backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py",
        "profile": profile,
        "profile_guidance": profile_guidance,
        "selected_profile_smoke_count": len(selected_smokes),
        "include_groups": include_groups,
        "available_include_groups": _include_group_summary(
            available_include_groups or include_groups,
            include_group_policies or {},
        ),
        "optimizer_suite_registered_smoke_count": len(all_smokes),
        "turbocore_related_smoke_file_count": len(turbocore_files),
        "suite_managed_turbocore_smoke_file_count": len(suite_managed_turbocore_files),
        "specialized_individual_turbocore_smoke_file_count": len(specialized_files),
        "roadmap_referenced_smoke_file_count": len(roadmap_files),
        "roadmap_referenced_suite_registered_smoke_file_count": len(roadmap_suite_files),
        "roadmap_referenced_specialized_smoke_file_count": len(roadmap_specialized_files),
        "roadmap_referenced_specialized_smoke_sample": roadmap_specialized_files[:25],
        "roadmap_referenced_specialized_smoke_buckets": _bucket_smoke_files(roadmap_specialized_files),
        "all_specialized_smoke_buckets": _bucket_smoke_files(specialized_files),
        "fragmentation_status": "too_many_specialized_entrypoints_use_profiled_suite_first",
        "simplification_status": (
            "routine optimizer validation is profile-based; roadmap-specialized smokes "
            "remain failure-localization or explicit artifact-refresh entrypoints"
        ),
        "simplification_decision_table": [
            {
                "workflow": "daily_development",
                "entrypoint": "turbocore_optimizer_smoke_suite.py --profile quick",
                "policy": "run_suite_first",
            },
            {
                "workflow": "optimizer_runtime_or_precondition_change",
                "entrypoint": "turbocore_optimizer_smoke_suite.py --profile runtime",
                "policy": "run_suite_first",
            },
            {
                "workflow": "optimizer_family_coverage_change",
                "entrypoint": "turbocore_optimizer_smoke_suite.py --profile coverage",
                "policy": "artifact_first_no_nested_rebuild_by_default",
            },
            {
                "workflow": "v2_runtime_boundary_batch",
                "entrypoint": "turbocore_optimizer_smoke_suite.py --profile release --include v2_runtime_boundary_batch",
                "policy": "run_once_after_runtime_boundary_module_changes",
            },
            {
                "workflow": "v2_release_handoff_batch",
                "entrypoint": "turbocore_optimizer_smoke_suite.py --profile release --include v2_release_handoff_batch",
                "policy": "run_once_after_release_handoff_packet_changes",
            },
            {
                "workflow": "v2_real_gate_phase1_record_batch",
                "entrypoint": (
                    "turbocore_optimizer_smoke_suite.py --profile release "
                    "--include v2_real_gate_phase1_record_batch --allow-real-record-gate "
                    "--real-record-input-manifest <real_record_input_manifest.json>"
                ),
                "policy": "run_once_after_phase1_real_signed_records_are_available",
                "real_reviewer_input_required": True,
                "release_evidence_role": "real_record_gate_validation_only",
                "real_record_gate_phase": "phase1",
                "execution_confirmation_flag": "--allow-real-record-gate",
                "real_record_input_manifest_required": True,
                "real_record_input_manifest_flag": "--real-record-input-manifest",
                "deferred_until_real_record_input_manifest_ready": True,
                "allow_real_record_gate_by_default": False,
                "real_record_gate_execution_allowed_by_default": False,
                "real_record_gate_execution_blocked_by_default_count": 1,
                "real_record_gate_execution_requirement": (
                    "phase1 real reviewer-returned signed bundle and record inputs plus "
                    "--allow-real-record-gate and --real-record-input-manifest"
                ),
            },
            {
                "workflow": "v2_real_gate_record_batch",
                "entrypoint": (
                    "turbocore_optimizer_smoke_suite.py --profile release "
                    "--include v2_real_gate_record_batch --allow-real-record-gate "
                    "--real-record-input-manifest <real_record_input_manifest.json>"
                ),
                "policy": "run_once_after_real_signed_records_are_available",
                "real_reviewer_input_required": True,
                "release_evidence_role": "real_record_gate_validation_only",
                "execution_confirmation_flag": "--allow-real-record-gate",
                "real_record_input_manifest_required": True,
                "real_record_input_manifest_flag": "--real-record-input-manifest",
                "deferred_until_real_record_input_manifest_ready": True,
                "allow_real_record_gate_by_default": False,
                "real_record_gate_execution_allowed_by_default": False,
                "real_record_gate_execution_blocked_by_default_count": 1,
                "real_record_gate_execution_requirement": (
                    "real reviewer-returned signed bundle and record inputs plus --allow-real-record-gate "
                    "and --real-record-input-manifest"
                ),
            },
            {
                "workflow": "v2_approval_chain",
                "entrypoint": "turbocore_optimizer_smoke_suite.py --profile release --include v2_approval_chain",
                "policy": "run_once_after_approval_chain_module_changes",
            },
            {
                "workflow": "v2_freshness_sequence_batch",
                "entrypoint": "turbocore_optimizer_smoke_suite.py --profile release --include v2_freshness_sequence_batch",
                "policy": "run_once_after_freshness_or_sequence_guard_changes",
            },
            {
                "workflow": "v2_support_guard_batch",
                "entrypoint": "turbocore_optimizer_smoke_suite.py --profile release --include v2_support_guard_batch",
                "policy": "run_once_after_support_guard_shape_changes",
            },
            {
                "workflow": "batched_optimizer_work",
                "entrypoint": "turbocore_optimizer_smoke_suite.py --profile batch",
                "policy": "run_once_after_theoretical_changes_are_done",
            },
            {
                "workflow": "suite_failure_localization",
                "entrypoint": "individual smoke named by failing smoke_id/module",
                "policy": "only_after_suite_failure",
            },
            {
                "workflow": "intentional_evidence_refresh",
                "entrypoint": "explicit --rebuild-artifact or --rebuild-artifacts command",
                "policy": "only_when_refreshing_underlying_artifacts",
            },
        ],
        "selected_profile_smokes": [_spec_dict(spec) for spec in selected_smokes],
        "profiles": profiles,
        "workflow": list(workflow),
        "recommended_commands": [
            "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile quick",
            "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile runtime",
            "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile coverage",
            (
                "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile release "
                "--exclude v2_real_gate_record_batch"
            ),
            "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile release --include v2_runtime_boundary_batch",
            "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile release --include v2_release_handoff_batch",
            (
                "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile release "
                "--include v2_real_gate_phase1_record_batch --allow-real-record-gate "
                "--real-record-input-manifest <real_record_input_manifest.json>"
            ),
            (
                "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile release "
                "--include v2_real_gate_record_batch --allow-real-record-gate "
                "--real-record-input-manifest <real_record_input_manifest.json>"
            ),
            "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile release --include v2_approval_chain",
            "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile release --include v2_freshness_sequence_batch",
            "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile release --include v2_support_guard_batch",
            "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile batch",
            "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --audit-scaffold",
        ],
        "recommended_command_policy_table": recommended_command_policy_table,
        "default_recommended_commands": default_recommended_commands,
        "deferred_recommended_commands": deferred_recommended_commands,
        "default_recommended_command_count": len(default_recommended_commands),
        "deferred_recommended_command_count": len(deferred_recommended_commands),
        "notes": [
            "This audit does not execute optimizer kernels or rebuild artifacts.",
            "The specialized files remain available for failure localization.",
            "For batched optimizer work, use --profile batch after changes are done.",
        ],
    }


def _roadmap_referenced_smoke_files(repo_root: Path, script_root: Path, roadmap: str) -> list[str]:
    roadmap_path = repo_root / roadmap
    if not roadmap_path.exists():
        return []
    text = roadmap_path.read_text(encoding="utf-8", errors="ignore")
    found: set[str] = set()
    for match in _SMOKE_REF_RE.findall(text):
        name = Path(match.replace("\\", "/")).name
        if name.endswith("_smoke.py") and (script_root / name).exists():
            found.add(name)
    return sorted(found)


def _bucket_smoke_files(files: Sequence[str]) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[str]] = {name: [] for name, _ in _SPECIALIZED_BUCKETS}
    buckets["uncategorized_failure_localization"] = []
    for file_name in files:
        lowered = file_name.lower()
        for bucket, markers in _SPECIALIZED_BUCKETS:
            if any(marker in lowered for marker in markers):
                buckets[bucket].append(file_name)
                break
        else:
            buckets["uncategorized_failure_localization"].append(file_name)
    return {
        name: {"count": len(items), "sample": items[:12]}
        for name, items in buckets.items()
        if items
    }


def _include_group_summary(
    include_groups: dict[str, list[str]],
    include_group_policies: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "smoke_count": len(items),
            "smoke_ids": list(items),
            "policy": _include_group_policy(name, include_group_policies),
        }
        for name, items in sorted(include_groups.items())
    }


def _include_group_policy(
    group_name: str,
    include_group_policies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    policy = {
        "real_reviewer_input_required": False,
        "approval_artifact_allowed": False,
        "policy": "module_batch_validation",
        "release_evidence_role": "synthetic_or_read_only_validation",
    }
    policy.update(include_group_policies.get(group_name, {}))
    if (
        policy.get("real_reviewer_input_required") is True
        and policy.get("release_evidence_role") == "real_record_gate_validation_only"
    ):
        policy.update(
            {
                "allow_real_record_gate_by_default": False,
                "real_record_gate_execution_allowed_by_default": False,
                "real_record_gate_execution_blocked_by_default_count": 1,
                "real_record_input_manifest_required": True,
                "real_record_input_manifest_flag": "--real-record-input-manifest",
                "deferred_until_real_record_input_manifest_ready": True,
                "real_record_gate_execution_requirement": (
                    "real reviewer-returned signed bundle and record inputs plus --allow-real-record-gate "
                    "and --real-record-input-manifest"
                ),
            }
        )
    return policy


def _recommended_command_policy_table() -> list[dict[str, Any]]:
    commands = [
        ("daily_development", "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile quick"),
        ("optimizer_runtime_or_precondition_change", "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile runtime"),
        ("optimizer_family_coverage_change", "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile coverage"),
        (
            "release_profile",
            (
                "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile release "
                "--exclude v2_real_gate_record_batch"
            ),
        ),
        (
            "v2_runtime_boundary_batch",
            "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile release --include v2_runtime_boundary_batch",
        ),
        (
            "v2_release_handoff_batch",
            "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile release --include v2_release_handoff_batch",
        ),
        (
            "v2_real_gate_phase1_record_batch",
            (
                "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile release "
                "--include v2_real_gate_phase1_record_batch --allow-real-record-gate "
                "--real-record-input-manifest <real_record_input_manifest.json>"
            ),
        ),
        (
            "v2_real_gate_record_batch",
            (
                "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile release "
                "--include v2_real_gate_record_batch --allow-real-record-gate "
                "--real-record-input-manifest <real_record_input_manifest.json>"
            ),
        ),
        (
            "v2_approval_chain",
            "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile release --include v2_approval_chain",
        ),
        (
            "v2_freshness_sequence_batch",
            "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile release --include v2_freshness_sequence_batch",
        ),
        (
            "v2_support_guard_batch",
            "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile release --include v2_support_guard_batch",
        ),
        ("batched_optimizer_work", "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile batch"),
        ("scaffold_audit", "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --audit-scaffold"),
    ]
    out: list[dict[str, Any]] = []
    for workflow, command in commands:
        real_gate = workflow in {"v2_real_gate_record_batch", "v2_real_gate_phase1_record_batch"}
        entry = {
            "workflow": workflow,
            "command": command,
            "execute_by_default": not real_gate,
            "real_reviewer_input_required": real_gate,
        }
        if real_gate:
            entry.update(
                {
                    "deferred_until_real_record_inputs": True,
                    "deferred_until_real_record_input_manifest_ready": True,
                    "execution_confirmation_flag": "--allow-real-record-gate",
                    "real_record_input_manifest_required": True,
                    "real_record_input_manifest_flag": "--real-record-input-manifest",
                    "release_evidence_role": "real_record_gate_validation_only",
                    "real_record_gate_phase": "phase1"
                    if workflow == "v2_real_gate_phase1_record_batch"
                    else "full",
                    "real_record_gate_execution_requirement": (
                        (
                            "phase1 real reviewer-returned signed bundle and record inputs plus "
                            "--allow-real-record-gate and --real-record-input-manifest"
                        )
                        if workflow == "v2_real_gate_phase1_record_batch"
                        else (
                            "real reviewer-returned signed bundle and record inputs plus "
                            "--allow-real-record-gate and --real-record-input-manifest"
                        )
                    ),
                }
            )
        out.append(entry)
    return out


def _recommended_commands_by_default_policy(
    recommended_command_policy_table: Sequence[dict[str, Any]],
    *,
    execute_by_default: bool,
) -> list[str]:
    return [
        item["command"]
        for item in recommended_command_policy_table
        if item["execute_by_default"] is execute_by_default
    ]


def _spec_dict(spec: Any) -> dict[str, Any]:
    if hasattr(spec, "__dict__"):
        return dict(spec.__dict__)
    return {
        "smoke_id": str(getattr(spec, "smoke_id", "")),
        "module": str(getattr(spec, "module", "")),
        "tier": str(getattr(spec, "tier", "")),
        "description": str(getattr(spec, "description", "")),
    }
