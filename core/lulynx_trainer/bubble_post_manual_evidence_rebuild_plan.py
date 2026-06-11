"""JSON-only post-manual-run rebuild plan for GPU-bubble evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


POST_MANUAL_EVIDENCE_REBUILD_PLAN_REPORT = "bubble_post_manual_evidence_rebuild_plan_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
REQUIRED_REBUILD_REFRESH_SEQUENCE = [
    "refresh_external_input_admission",
    "refresh_external_input_intake_registry",
    "refresh_external_input_replay_plan",
    "refresh_source_axis_freshness_dedupe_audit",
    "refresh_source_cache_axis_identity_registry",
    "refresh_source_cache_axis_pipeline_readiness",
    "refresh_external_input_handoff_packet",
    "refresh_gpu_bubble_readiness_next_actions",
    "refresh_gpu_bubble_terminal_self_check",
    "run_gpu_bubble_release_readiness_guard",
]

def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}

def _list(value: Any) -> list[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return list(value)

def _strings(value: Any) -> list[str]:
    return [str(item) for item in _list(value) if item is not None]

def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value if value is not None else default)))
    except (TypeError, ValueError, OverflowError):
        return int(default)

def _repo_path(repo_root: Path, value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str(repo_root / path)

def _python(repo_root: Path, python_exe: str) -> str:
    return _repo_path(repo_root, python_exe)

def _command(
    *,
    command_id: str,
    description: str,
    command: list[str],
    status: str,
    stage_id: str,
    stage_order: int,
    expected_outputs: Sequence[str] = (),
    prerequisites: Sequence[str] = (),
    depends_on: Sequence[str] = (),
) -> dict[str, Any]:
    output_status = _expected_output_status(expected_outputs, command_status=status)
    return {
        "id": command_id,
        "command_id": command_id,
        "description": description,
        "status": status,
        "ready": status == "ready",
        "stage_id": stage_id,
        "stage_order": stage_order,
        "command": command,
        "expected_outputs": list(expected_outputs),
        "expected_output_status": output_status,
        "expected_output_count": len(output_status),
        "existing_expected_output_count": sum(1 for item in output_status if item["exists"]),
        "missing_expected_output_count": sum(1 for item in output_status if not item["exists"]),
        "prerequisites": list(prerequisites),
        "blockers": list(prerequisites) if status != "ready" else [],
        "depends_on_command_ids": list(depends_on),
        "requires_gpu_if_executed": False,
        "safe_to_auto_start": False,
        "release_claim_allowed_after_success": False,
        "not_release_evidence": True,
        "roadmap": ROADMAP,
    }

def _expected_output_status(outputs: Sequence[str], *, command_status: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in outputs:
        path = Path(str(raw))
        exists = path.is_file()
        if command_status == "ready":
            state = "already_present_may_be_refreshed" if exists else "pending_rebuild"
            missing_reason = "" if exists else "output_missing_until_rebuild_command_runs"
        elif exists:
            state = "existing_output_not_refreshed_after_manual_evidence"
            missing_reason = "blocked_until_manual_evidence_is_admitted"
        else:
            state = "blocked_missing_output"
            missing_reason = command_status
        rows.append(
            {
                "path": str(path),
                "exists": exists,
                "state": state,
                "missing_reason": missing_reason,
            }
        )
    return rows

def _stage_rows(commands: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for command in sorted(commands, key=lambda item: _safe_int(_mapping(item).get("stage_order"), 999)):
        item = _mapping(command)
        stage_id = str(item.get("stage_id") or "")
        if not stage_id or stage_id in seen:
            continue
        seen.add(stage_id)
        same_stage = [cmd for cmd in commands if str(_mapping(cmd).get("stage_id") or "") == stage_id]
        output_rows = [
            status
            for cmd in same_stage
            for status in _list(_mapping(cmd).get("expected_output_status"))
            if isinstance(status, Mapping)
        ]
        rows.append(
            {
                "stage_id": stage_id,
                "stage_order": _safe_int(item.get("stage_order")),
                "status": "ready" if all(str(_mapping(cmd).get("status") or "") == "ready" for cmd in same_stage) else "blocked",
                "command_ids": [str(_mapping(cmd).get("id") or "") for cmd in same_stage],
                "expected_output_count": len(output_rows),
                "existing_expected_output_count": sum(1 for row in output_rows if bool(_mapping(row).get("exists"))),
                "missing_expected_output_count": sum(1 for row in output_rows if not bool(_mapping(row).get("exists"))),
            }
        )
    return rows

def _next_rebuild_stage(stages: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for raw in sorted(stages, key=lambda item: _safe_int(_mapping(item).get("stage_order"), 999)):
        stage = _mapping(raw)
        if str(stage.get("status") or "") != "ready":
            return dict(stage)
    return {"stage_id": "all_rebuild_stages_ready", "stage_order": 999, "status": "ready"}

def _refresh_sequence_contract(commands: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    command_ids = [str(_mapping(item).get("command_id") or _mapping(item).get("id") or "") for item in commands]
    positions = {command_id: index for index, command_id in enumerate(command_ids) if command_id}
    missing = [command_id for command_id in REQUIRED_REBUILD_REFRESH_SEQUENCE if command_id not in positions]
    out_of_order: list[str] = []
    for left, right in zip(REQUIRED_REBUILD_REFRESH_SEQUENCE, REQUIRED_REBUILD_REFRESH_SEQUENCE[1:]):
        if left in positions and right in positions and positions[left] >= positions[right]:
            out_of_order.extend([left, right])
    unsafe = [
        command_id
        for command_id, raw in zip(command_ids, commands)
        if bool(_mapping(raw).get("safe_to_auto_start"))
        or bool(_mapping(raw).get("release_claim_allowed_after_success"))
        or not bool(_mapping(raw).get("not_release_evidence"))
    ]
    roadmap_mismatch = [
        command_id
        for command_id, raw in zip(command_ids, commands)
        if str(_mapping(raw).get("roadmap") or "") != ROADMAP
    ]
    return {
        "summary_version": 1,
        "roadmap": ROADMAP,
        "artifact_role": "gpu_bubble_post_manual_refresh_sequence_contract",
        "required_command_ids": list(REQUIRED_REBUILD_REFRESH_SEQUENCE),
        "observed_command_ids": command_ids,
        "missing_command_count": len(missing),
        "missing_command_ids": missing,
        "out_of_order_command_count": len(set(out_of_order)),
        "out_of_order_command_ids": sorted(set(out_of_order)),
        "unsafe_command_count": len(unsafe),
        "unsafe_command_ids": unsafe,
        "roadmap_mismatch_count": len(roadmap_mismatch),
        "roadmap_mismatch_command_ids": roadmap_mismatch,
        "sequence_ok": not missing and not out_of_order and not unsafe and not roadmap_mismatch,
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "not_release_evidence": True,
    }

def _manual_plan_outputs(plan: Mapping[str, Any]) -> list[str]:
    outputs: list[str] = []
    for raw in _list(plan.get("commands")):
        item = _mapping(raw)
        out_dir = str(item.get("out_dir") or "").strip()
        if out_dir:
            outputs.extend(
                [
                    str(Path(out_dir) / "real_material_canary_results.json"),
                    str(Path(out_dir) / "evidence_pack" / "natural_load_canary.json"),
                    str(Path(out_dir) / "evidence_pack" / "release_claims.json"),
                ]
            )
    return outputs

def _sd15_expected_outputs(sd15_readiness: Mapping[str, Any]) -> list[str]:
    expected = _mapping(sd15_readiness.get("expected_outputs"))
    return [str(value) for value in expected.values() if value]


def _manual_evidence_ready(
    *,
    manual_plan: Mapping[str, Any],
    sd15_readiness: Mapping[str, Any],
) -> bool:
    return bool(manual_plan.get("preflight_admitted")) or str(sd15_readiness.get("status") or "") in {
        "manual_gpu_evidence_ready",
        "evidence_available_pending_release_claim_refresh",
    }


def _base_rebuild_status(evidence_ready: bool) -> tuple[str, list[str]]:
    if evidence_ready:
        return "ready", []
    return "blocked_waiting_for_manual_evidence", ["manual_gpu_evidence_required"]


def _current_gap_summary(release_claims: Mapping[str, Any], natural_load_canary: Mapping[str, Any]) -> dict[str, Any]:
    gaps = [_mapping(item) for item in _list(release_claims.get("evidence_gaps"))]
    coverage = [_mapping(item) for item in _list(release_claims.get("coverage"))]
    return {
        "release_readiness": str(release_claims.get("release_readiness") or ""),
        "evidence_gap_count": len(gaps),
        "uncovered_case_ids": [str(item.get("case_id") or "") for item in coverage if not bool(item.get("covered"))],
        "natural_load_status": str(natural_load_canary.get("status") or ""),
        "natural_load_ready_family_count": _safe_int(natural_load_canary.get("ready_family_count")),
        "natural_load_family_count": _safe_int(natural_load_canary.get("family_count")),
        "natural_load_blocked_families": _strings(natural_load_canary.get("blocked_families")),
        "natural_load_missing_families": _strings(natural_load_canary.get("missing_families")),
    }


def _manual_evidence_blocking_summary(
    *,
    manual_ready: bool,
    sd15_manual_blocked: bool,
    evidence_ready: bool,
    blockers: Sequence[str],
    gap_summary: Mapping[str, Any],
) -> dict[str, Any]:
    natural_blocked = str(gap_summary.get("natural_load_status") or "") == "blocked_pending_canary"
    release_blocked = str(gap_summary.get("release_readiness") or "") == "blocked_pending_evidence"
    source_plan_blocked = "source_cache_axis_manual_canary_plan_not_ready" in blockers
    manual_evidence_required = "manual_gpu_evidence_required" in blockers
    return {
        "summary_version": 1,
        "roadmap": ROADMAP,
        "manual_gpu_evidence_ready": evidence_ready,
        "manual_gpu_evidence_required": manual_evidence_required,
        "source_cache_axis_manual_canary_plan_ready": manual_ready,
        "source_cache_axis_manual_canary_plan_required": source_plan_blocked,
        "sd15_checkpoint_required": sd15_manual_blocked,
        "natural_load_canary_pending": natural_blocked,
        "release_claims_rebuild_required": release_blocked or evidence_ready,
        "release_gate_blockers": [
            *(["sd15_lora_512"] if sd15_manual_blocked else []),
            *(["natural_load_canary_pending"] if natural_blocked else []),
        ],
        "next_required_inputs": [
            *(["sd15_checkpoint"] if sd15_manual_blocked else []),
            *(["source_cache_axis_manual_canary_evidence"] if source_plan_blocked else []),
            *(["manual_gpu_evidence"] if manual_evidence_required else []),
        ],
        "next_json_rebuild_stage_id": "rebuild_current_combined_evidence_pack",
        "blocked_actions": [
            "do_not_run_post_manual_rebuild_before_manual_gpu_evidence",
            "do_not_publish_release_claim_before_natural_load_and_release_claim_rebuild",
            "do_not_auto_start_gpu_from_post_manual_summary",
        ],
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "not_release_evidence": True,
    }


def _rebuild_commands(repo_root: Path, python_exe: str, *, status: str, prerequisites: Sequence[str]) -> list[dict[str, Any]]:
    py = _python(repo_root, python_exe)
    runtime = _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime")
    current_combined = _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/current_combined")
    return [
        _command(
            command_id="rebuild_current_combined_evidence_pack",
            description="Rebuild current_combined evidence pack after manual GPU evidence is present.",
            stage_id="rebuild_current_combined_evidence_pack",
            stage_order=10,
            command=[
                py,
                _repo_path(repo_root, "devtools/build_bubble_runtime_evidence_pack.py"),
                runtime,
                "--out-dir",
                current_combined,
            ],
            status=status,
            prerequisites=prerequisites,
            expected_outputs=[
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/current_combined/evidence_pack.json"),
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/current_combined/natural_load_canary.json"),
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/current_combined/release_claims.json"),
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/current_combined/review_queue.json"),
            ],
        ),
        _command(
            command_id="rebuild_followup_run_plan",
            description="Rebuild protected follow-up run plan from the refreshed current_combined follow-up plan.",
            command=[py, _repo_path(repo_root, "devtools/plan_bubble_runtime_followup_runs.py")],
            stage_id="rebuild_followup_run_plan",
            stage_order=20,
            status=status,
            prerequisites=[*prerequisites, "current_combined_followup_plan_required"],
            depends_on=["rebuild_current_combined_evidence_pack"],
            expected_outputs=[
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/real_material_canary_p60_followup_run_plan.json")
            ],
        ),
        _command(
            command_id="rebuild_followup_run_readiness",
            description="Rebuild follow-up run readiness after plan/evidence refresh.",
            command=[py, _repo_path(repo_root, "devtools/check_bubble_runtime_followup_run_readiness.py")],
            stage_id="rebuild_followup_run_readiness",
            stage_order=30,
            status=status,
            prerequisites=[*prerequisites, "followup_run_plan_required"],
            depends_on=["rebuild_followup_run_plan"],
            expected_outputs=[
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/real_material_canary_p60_followup_run_readiness.json")
            ],
        ),
        _command(
            command_id="refresh_source_axis_requirement",
            description="Refresh source/cache-axis requirement after post-run evidence rebuild.",
            command=[py, _repo_path(repo_root, "devtools/build_bubble_p60_source_axis_requirement.py")],
            stage_id="refresh_source_cache_admission_chain",
            stage_order=40,
            status=status,
            prerequisites=prerequisites,
            depends_on=["rebuild_followup_run_readiness"],
            expected_outputs=[
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/p60_source_axis_requirement.json")
            ],
        ),
        _command(
            command_id="refresh_newbie_warm_cache_inventory",
            description="Refresh Newbie warm-cache inventory against rebuilt natural-load gate.",
            command=[py, _repo_path(repo_root, "devtools/build_bubble_newbie_warm_cache_inventory.py")],
            stage_id="refresh_source_cache_admission_chain",
            stage_order=40,
            status=status,
            prerequisites=prerequisites,
            depends_on=["rebuild_current_combined_evidence_pack"],
            expected_outputs=[
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/newbie_warm_cache_inventory.json")
            ],
        ),
        _command(
            command_id="refresh_external_input_admission",
            description="Refresh external input admission after rebuilt release evidence.",
            command=[py, _repo_path(repo_root, "devtools/build_bubble_external_input_admission.py")],
            stage_id="refresh_source_cache_admission_chain",
            stage_order=40,
            status=status,
            prerequisites=prerequisites,
            depends_on=["refresh_source_axis_requirement", "refresh_newbie_warm_cache_inventory"],
            expected_outputs=[
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/external_input_admission.json")
            ],
        ),
        _command(
            command_id="refresh_external_input_intake_registry",
            description="Refresh external input intake registry after rebuilt evidence.",
            command=[py, _repo_path(repo_root, "devtools/build_bubble_external_input_intake_status.py")],
            stage_id="refresh_external_input_replay_chain",
            stage_order=60,
            status=status,
            prerequisites=prerequisites,
            depends_on=["refresh_external_input_admission"],
            expected_outputs=[
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/external_input_intake_registry.json")
            ],
        ),
        _command(
            command_id="refresh_external_input_replay_plan",
            description="Refresh external input replay plan after rebuilt evidence.",
            command=[py, _repo_path(repo_root, "devtools/build_bubble_external_input_replay_plan.py")],
            stage_id="refresh_external_input_replay_chain",
            stage_order=60,
            status=status,
            prerequisites=prerequisites,
            depends_on=["refresh_external_input_intake_registry"],
            expected_outputs=[
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/external_input_replay_plan.json")
            ],
        ),
        _command(
            command_id="refresh_source_axis_freshness_dedupe_audit",
            description="Refresh source-axis freshness/dedupe audit after rebuilt evidence.",
            command=[py, _repo_path(repo_root, "devtools/build_bubble_source_axis_freshness_dedupe_audit.py")],
            stage_id="refresh_external_input_replay_chain",
            stage_order=60,
            status=status,
            prerequisites=prerequisites,
            depends_on=["refresh_external_input_replay_plan"],
            expected_outputs=[
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/source_axis_freshness_dedupe_audit.json")
            ],
        ),
        _command(
            command_id="refresh_source_cache_axis_identity_registry",
            description="Refresh source/cache-axis identity registry after rebuilt freshness and dedupe audit.",
            command=[py, _repo_path(repo_root, "devtools/build_bubble_source_cache_axis_identity_registry.py")],
            stage_id="refresh_external_input_replay_chain",
            stage_order=60,
            status=status,
            prerequisites=prerequisites,
            depends_on=["refresh_source_axis_freshness_dedupe_audit"],
            expected_outputs=[
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/source_cache_axis_identity_registry.json")
            ],
        ),
        _command(
            command_id="refresh_source_cache_axis_pipeline_readiness",
            description="Refresh source/cache-axis pipeline readiness after rebuilt identity registry.",
            command=[py, _repo_path(repo_root, "devtools/build_bubble_source_cache_axis_pipeline_readiness.py")],
            stage_id="refresh_source_cache_pipeline_readiness",
            stage_order=70,
            status=status,
            prerequisites=prerequisites,
            depends_on=["refresh_source_cache_axis_identity_registry"],
            expected_outputs=[
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/source_cache_axis_pipeline_readiness.json")
            ],
        ),
        _command(
            command_id="refresh_external_input_handoff_packet",
            description="Refresh external-input handoff packet before top-level readiness consumes it.",
            command=[py, _repo_path(repo_root, "devtools/build_bubble_external_input_handoff_packet.py")],
            stage_id="refresh_external_input_handoff_packet",
            stage_order=75,
            status=status,
            prerequisites=prerequisites,
            depends_on=["refresh_source_cache_axis_pipeline_readiness"],
            expected_outputs=[
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/external_input_handoff_packet.json")
            ],
        ),
        _command(
            command_id="refresh_gpu_bubble_readiness_next_actions",
            description="Refresh top-level GPU bubble readiness after all post-run artifacts.",
            command=[py, _repo_path(repo_root, "devtools/build_gpu_bubble_experiment_readiness_next_actions.py")],
            stage_id="refresh_top_level_readiness",
            stage_order=80,
            status=status,
            prerequisites=prerequisites,
            depends_on=["refresh_external_input_handoff_packet"],
            expected_outputs=[
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/gpu_bubble_experiment_readiness_next_actions.json")
            ],
        ),
        _command(
            command_id="refresh_gpu_bubble_terminal_self_check",
            description="Refresh terminal GPU bubble readiness self-check after top-level readiness.",
            command=[py, _repo_path(repo_root, "devtools/build_bubble_gpu_bubble_readiness_terminal_self_check.py")],
            stage_id="refresh_terminal_readiness_guard",
            stage_order=90,
            status=status,
            prerequisites=prerequisites,
            depends_on=["refresh_gpu_bubble_readiness_next_actions"],
            expected_outputs=[
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/gpu_bubble_readiness_terminal_self_check.json")
            ],
        ),
        _command(
            command_id="run_gpu_bubble_release_readiness_guard",
            description="Run read-only release readiness guard after readiness and terminal self-check refresh.",
            command=[
                py,
                _repo_path(repo_root, "devtools/guard_gpu_bubble_release_readiness.py"),
                "--out",
                _repo_path(
                    repo_root,
                    "devtools/benchmark_evidence/bubble_runtime/gpu_bubble_release_readiness_guard_report.json",
                ),
            ],
            stage_id="refresh_terminal_readiness_guard",
            stage_order=90,
            status=status,
            prerequisites=prerequisites,
            depends_on=["refresh_gpu_bubble_terminal_self_check"],
            expected_outputs=[
                _repo_path(
                    repo_root,
                    "devtools/benchmark_evidence/bubble_runtime/gpu_bubble_release_readiness_guard_report.json",
                )
            ],
        ),
    ]


def build_post_manual_evidence_rebuild_plan(
    *,
    repo_root: Path,
    source_cache_axis_manual_canary_plan: Mapping[str, Any] | None = None,
    sd15_readiness: Mapping[str, Any] | None = None,
    release_claims: Mapping[str, Any] | None = None,
    natural_load_canary: Mapping[str, Any] | None = None,
    input_sources: Sequence[Mapping[str, Any]] | None = None,
    python_exe: str = "backend/env/python-flashattention/python.exe",
) -> dict[str, Any]:
    """Build a fail-closed rebuild plan for evidence produced by manual GPU work."""

    repo = Path(repo_root)
    manual_plan = _mapping(source_cache_axis_manual_canary_plan)
    sd15 = _mapping(sd15_readiness)
    claims = _mapping(release_claims)
    canary = _mapping(natural_load_canary)
    manual_ready = str(manual_plan.get("status") or "") == "protected_manual_canary_plan_ready"
    sd15_manual_blocked = "sd15_base_checkpoint_missing" in _strings(sd15.get("blockers"))
    evidence_ready = _manual_evidence_ready(manual_plan=manual_plan, sd15_readiness=sd15)
    command_status, blockers = _base_rebuild_status(evidence_ready)
    if not manual_ready:
        blockers.append("source_cache_axis_manual_canary_plan_not_ready")
    if sd15_manual_blocked:
        blockers.append("sd15_checkpoint_required")
    if evidence_ready:
        status = "post_manual_rebuild_ready"
    elif sd15_manual_blocked or not manual_ready:
        status = "waiting_for_manual_evidence"
    else:
        status = "manual_review_required"
    gap_summary = _current_gap_summary(claims, canary)
    commands = _rebuild_commands(repo, python_exe, status=command_status, prerequisites=blockers)
    rebuild_stages = _stage_rows(commands)
    refresh_sequence_contract = _refresh_sequence_contract(commands)
    return {
        "schema_version": 1,
        "report": POST_MANUAL_EVIDENCE_REBUILD_PLAN_REPORT,
        "roadmap": ROADMAP,
        "status": status,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "release_claim_allowed": False,
        "publishable": False,
        "not_release_evidence": True,
        "manual_canary_plan_ready": manual_ready,
        "manual_canary_command_count": _safe_int(manual_plan.get("command_count")),
        "manual_canary_expected_outputs": _manual_plan_outputs(manual_plan),
        "sd15_status": str(sd15.get("status") or ""),
        "sd15_expected_outputs": _sd15_expected_outputs(sd15),
        "input_sources": [dict(_mapping(item)) for item in _list(input_sources)],
        "current_gap_summary": gap_summary,
        "manual_evidence_blocking_summary": _manual_evidence_blocking_summary(
            manual_ready=manual_ready,
            sd15_manual_blocked=sd15_manual_blocked,
            evidence_ready=evidence_ready,
            blockers=blockers,
            gap_summary=gap_summary,
        ),
        "command_count": len(commands),
        "ready_command_count": sum(1 for item in commands if item["status"] == "ready"),
        "rebuild_stage_count": len(rebuild_stages),
        "ready_rebuild_stage_count": sum(1 for item in rebuild_stages if item["status"] == "ready"),
        "rebuild_stages": rebuild_stages,
        "next_rebuild_stage": _next_rebuild_stage(rebuild_stages),
        "refresh_sequence_contract": refresh_sequence_contract,
        "commands": commands,
        "blockers": sorted(set(blockers)),
        "blocked_actions": [
            "auto_start_post_manual_rebuild_plan",
            "promote_manual_gpu_output_without_rebuilding_natural_load_canary",
            "promote_manual_gpu_output_without_rebuilding_release_claims",
            "write_universal_gpu_utilization_claim_after_manual_run",
        ],
        "acceptance_gates": [
            "manual_gpu_evidence_present_before_rebuild",
            "natural_load_canary_rebuilt_after_manual_run",
            "release_claims_rebuilt_after_manual_run",
            "gpu_bubble_readiness_rebuilt_after_claims",
            "case_specific_release_wording_only",
        ],
        "notes": [
            "This plan is JSON-only and does not start GPU work or rebuild artifacts automatically.",
            "Manual GPU evidence must be rebuilt into current_combined before any release claim review.",
            "A rebuilt release claim still must remain case-specific and pass natural-load coverage gates.",
        ],
    }


__all__ = [
    "POST_MANUAL_EVIDENCE_REBUILD_PLAN_REPORT",
    "ROADMAP",
    "build_post_manual_evidence_rebuild_plan",
]
