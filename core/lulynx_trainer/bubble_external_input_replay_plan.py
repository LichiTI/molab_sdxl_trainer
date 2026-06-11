"""JSON-only replay plan for GPU-bubble external-input intake."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .bubble_post_input_refresh_contract import (
    REPLAY_PLAN_REFRESH_SEQUENCE as REQUIRED_REFRESH_SEQUENCE,
)


EXTERNAL_INPUT_REPLAY_PLAN_REPORT = "bubble_external_input_replay_plan_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"


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


def _slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(value).name.strip())
    return text.strip("._") or "registered_source"


def _command(
    *,
    command_id: str,
    description: str,
    command: list[str],
    status: str,
    expected_outputs: Sequence[str] = (),
    prerequisites: Sequence[str] = (),
    family: str = "",
    candidate_root: str = "",
    template: bool = False,
) -> dict[str, Any]:
    return {
        "id": command_id,
        "command_id": command_id,
        "description": description,
        "status": status,
        "ready": status == "ready",
        "family": family,
        "candidate_root": candidate_root,
        "template": template,
        "command": command,
        "expected_outputs": list(expected_outputs),
        "expected_output_count": len(expected_outputs),
        "prerequisites": list(prerequisites),
        "requires_gpu_if_executed": False,
        "safe_to_auto_start": False,
        "release_claim_allowed_after_success": False,
        "not_release_evidence": True,
        "roadmap": ROADMAP,
    }


def _sd15_commands(repo_root: Path, python_exe: str, checkpoint_exists: bool) -> list[dict[str, Any]]:
    py = _python(repo_root, python_exe)
    status = "ready" if checkpoint_exists else "blocked_missing_sd15_checkpoint"
    prereq = [] if checkpoint_exists else ["sd15_checkpoint_required"]
    return [
        _command(
            command_id="refresh_sd15_lora512_release_gap_readiness",
            description="Refresh SD15 LoRA512 release-gap readiness after checkpoint intake.",
            command=[py, _repo_path(repo_root, "devtools/build_bubble_sd15_lora512_release_gap_readiness.py")],
            status=status,
            prerequisites=prereq,
            expected_outputs=[
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/sd15_lora512_release_gap_readiness.json")
            ],
        )
    ]


def _source_root_commands(
    *,
    repo_root: Path,
    python_exe: str,
    root: str,
    families: Sequence[str],
    ready: bool,
) -> list[dict[str, Any]]:
    py = _python(repo_root, python_exe)
    slug = _slug(root)
    scan_out = _repo_path(repo_root, f"devtools/benchmark_evidence/bubble_runtime/real_material_source_windows_{slug}.json")
    source_status = "ready" if ready else "blocked_missing_new_source_root"
    prereq = [] if ready else ["new_source_root_required"]
    family_args: list[str] = []
    for family in families:
        family_args.extend(["--family", family])
    commands = [
        _command(
            command_id=f"scan_registered_source_root_{slug}",
            description="Scan registered source root sample windows without GPU work.",
            command=[
                py,
                _repo_path(repo_root, "devtools/scan_bubble_real_material_sources.py"),
                root or "<new_source_root>",
                "--scan-windows",
                "--samples",
                "8",
                "--window-stride",
                "8",
                *family_args,
                "--out",
                scan_out,
            ],
            status=source_status,
            prerequisites=prereq,
            candidate_root=root,
            expected_outputs=[scan_out],
        ),
        _command(
            command_id=f"rank_registered_source_axis_{slug}",
            description="Rank scanned source/cache axes for P60 natural-load canary admission.",
            command=[
                py,
                _repo_path(repo_root, "devtools/plan_bubble_p60_source_axes.py"),
                "--source-scan",
                scan_out,
                "--out",
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/p60_source_axis_scout.json"),
            ],
            status=source_status,
            prerequisites=[*prereq, "source_scan_report_required"],
            candidate_root=root,
            expected_outputs=[
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/p60_source_axis_scout.json")
            ],
        ),
        _command(
            command_id=f"refresh_source_axis_requirement_{slug}",
            description="Refresh source-axis requirement after scout ranking.",
            command=[py, _repo_path(repo_root, "devtools/build_bubble_p60_source_axis_requirement.py")],
            status=source_status,
            prerequisites=[*prereq, "p60_source_axis_scout_required"],
            candidate_root=root,
            expected_outputs=[
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/p60_source_axis_requirement.json")
            ],
        ),
    ]
    for family in families:
        commands.append(
            _command(
                command_id=f"preflight_registered_source_axis_{family}_{slug}",
                description="Template preflight for a ranked source/cache axis selected from scout output.",
                command=[
                    py,
                    _repo_path(repo_root, "devtools/build_bubble_source_cache_axis_admission_preflight.py"),
                    "--candidate-root",
                    root or "<new_source_root>",
                    "--family",
                    family,
                    "--sample-offset",
                    "<sample_offset_from_scout>",
                    "--source-manifest-sha1",
                    "<source_manifest_sha1_from_scout>",
                ],
                status="template_waiting_for_scout_selection" if ready else source_status,
                prerequisites=[*prereq, "ranked_axis_selection_required"],
                family=family,
                candidate_root=root,
                template=True,
                expected_outputs=[
                    _repo_path(
                        repo_root,
                        "devtools/benchmark_evidence/bubble_runtime/source_cache_axis_admission_preflight.json",
                    )
                ],
            )
        )
    return commands


def _common_refresh_commands(repo_root: Path, python_exe: str, *, ready: bool) -> list[dict[str, Any]]:
    py = _python(repo_root, python_exe)
    status = "ready" if ready else "blocked_missing_external_input"
    prereq = [] if ready else ["external_input_required"]
    return [
        _command(
            command_id="refresh_external_input_admission",
            description="Refresh external-input admission after SD15/source-axis intake changes.",
            command=[py, _repo_path(repo_root, "devtools/build_bubble_external_input_admission.py")],
            status=status,
            prerequisites=prereq,
            expected_outputs=[
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/external_input_admission.json")
            ],
        ),
        _command(
            command_id="refresh_source_cache_axis_manual_canary_plan",
            description="Regenerate protected manual canary plan after an admitted preflight.",
            command=[py, _repo_path(repo_root, "devtools/build_bubble_source_cache_axis_manual_canary_plan.py")],
            status=status,
            prerequisites=[*prereq, "source_cache_axis_preflight_admitted"],
            expected_outputs=[
                _repo_path(
                    repo_root,
                    "devtools/benchmark_evidence/bubble_runtime/source_cache_axis_manual_canary_plan.json",
                )
            ],
        ),
        _command(
            command_id="refresh_external_input_intake_registry",
            description="Refresh intake registry after replayed JSON artifacts.",
            command=[py, _repo_path(repo_root, "devtools/build_bubble_external_input_intake_status.py")],
            status=status,
            prerequisites=prereq,
            expected_outputs=[
                _repo_path(repo_root, "devtools/benchmark_evidence/bubble_runtime/external_input_intake_registry.json")
            ],
        ),
        _command(
            command_id="refresh_source_axis_freshness_dedupe_audit",
            description="Refresh source-axis freshness/dedupe audit after replayed intake and pipeline artifacts.",
            command=[py, _repo_path(repo_root, "devtools/build_bubble_source_axis_freshness_dedupe_audit.py")],
            status=status,
            prerequisites=prereq,
            expected_outputs=[
                _repo_path(
                    repo_root,
                    "devtools/benchmark_evidence/bubble_runtime/source_axis_freshness_dedupe_audit.json",
                )
            ],
        ),
        _command(
            command_id="refresh_source_cache_axis_identity_registry",
            description="Refresh source/cache-axis identity registry after freshness and dedupe audit.",
            command=[py, _repo_path(repo_root, "devtools/build_bubble_source_cache_axis_identity_registry.py")],
            status=status,
            prerequisites=prereq,
            expected_outputs=[
                _repo_path(
                    repo_root,
                    "devtools/benchmark_evidence/bubble_runtime/source_cache_axis_identity_registry.json",
                )
            ],
        ),
        _command(
            command_id="refresh_source_cache_axis_pipeline_readiness",
            description="Refresh source/cache-axis pipeline readiness after identity registry refresh.",
            command=[py, _repo_path(repo_root, "devtools/build_bubble_source_cache_axis_pipeline_readiness.py")],
            status=status,
            prerequisites=prereq,
            expected_outputs=[
                _repo_path(
                    repo_root,
                    "devtools/benchmark_evidence/bubble_runtime/source_cache_axis_pipeline_readiness.json",
                )
            ],
        ),
        _command(
            command_id="refresh_external_input_handoff_packet",
            description="Refresh external-input handoff packet before top-level readiness consumes it.",
            command=[py, _repo_path(repo_root, "devtools/build_bubble_external_input_handoff_packet.py")],
            status=status,
            prerequisites=prereq,
            expected_outputs=[
                _repo_path(
                    repo_root,
                    "devtools/benchmark_evidence/bubble_runtime/external_input_handoff_packet.json",
                )
            ],
        ),
        _command(
            command_id="refresh_gpu_bubble_readiness_next_actions",
            description="Refresh top-level GPU bubble readiness after handoff and replay artifacts.",
            command=[py, _repo_path(repo_root, "devtools/build_gpu_bubble_experiment_readiness_next_actions.py")],
            status=status,
            prerequisites=prereq,
            expected_outputs=[
                _repo_path(
                    repo_root,
                    "devtools/benchmark_evidence/bubble_runtime/gpu_bubble_experiment_readiness_next_actions.json",
                )
            ],
        ),
        _command(
            command_id="refresh_gpu_bubble_terminal_self_check",
            description="Refresh terminal GPU bubble readiness self-check after top-level readiness.",
            command=[py, _repo_path(repo_root, "devtools/build_bubble_gpu_bubble_readiness_terminal_self_check.py")],
            status=status,
            prerequisites=prereq,
            expected_outputs=[
                _repo_path(
                    repo_root,
                    "devtools/benchmark_evidence/bubble_runtime/gpu_bubble_readiness_terminal_self_check.json",
                )
            ],
        ),
        _command(
            command_id="run_gpu_bubble_release_readiness_guard",
            description="Run the read-only release readiness guard after replay and terminal self-check refresh.",
            command=[
                py,
                _repo_path(repo_root, "devtools/guard_gpu_bubble_release_readiness.py"),
                "--out",
                _repo_path(
                    repo_root,
                    "devtools/benchmark_evidence/bubble_runtime/gpu_bubble_release_readiness_guard_report.json",
                ),
            ],
            status=status,
            prerequisites=prereq,
            expected_outputs=[
                _repo_path(
                    repo_root,
                    "devtools/benchmark_evidence/bubble_runtime/gpu_bubble_release_readiness_guard_report.json",
                )
            ],
        ),
    ]


def _refresh_sequence_contract(commands: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    command_ids = [str(_mapping(item).get("command_id") or _mapping(item).get("id") or "") for item in commands]
    positions = {command_id: index for index, command_id in enumerate(command_ids) if command_id}
    missing = [command_id for command_id in REQUIRED_REFRESH_SEQUENCE if command_id not in positions]
    out_of_order: list[str] = []
    for left, right in zip(REQUIRED_REFRESH_SEQUENCE, REQUIRED_REFRESH_SEQUENCE[1:]):
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
        "artifact_role": "gpu_bubble_external_input_replay_refresh_sequence_contract",
        "required_command_ids": list(REQUIRED_REFRESH_SEQUENCE),
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


def _families(intake: Mapping[str, Any]) -> list[str]:
    families = sorted(
        {
            str(_mapping(item).get("family") or "")
            for item in _list(intake.get("candidate_registration_slots"))
            if str(_mapping(item).get("family") or "")
        }
    )
    return families or ["anima", "newbie", "sdxl"]


def _new_source_roots(intake: Mapping[str, Any]) -> list[str]:
    source_axis = _mapping(intake.get("source_axis"))
    roots = []
    for raw in _list(source_axis.get("roots")):
        item = _mapping(raw)
        if str(item.get("intake_status") or "") == "new_root_available":
            roots.append(str(item.get("root") or ""))
    return [root for root in roots if root]


def _input_resolution_summary(
    *,
    intake: Mapping[str, Any],
    checkpoint_exists: bool,
    roots: Sequence[str],
) -> dict[str, Any]:
    upstream = _mapping(intake.get("input_resolution_summary"))
    missing_inputs = _strings(upstream.get("missing_external_inputs")) or _strings(
        intake.get("missing_external_inputs")
    )
    if not missing_inputs:
        missing_inputs = [
            *([] if checkpoint_exists else ["sd15_checkpoint"]),
            *([] if roots else ["new_source_root"]),
        ]
    new_source_root_count = _safe_int(
        upstream.get("new_source_root_count"),
        len(roots),
    )
    return {
        "summary_version": 1,
        "roadmap": ROADMAP,
        "external_input_detected": bool(
            upstream.get("external_input_detected", intake.get("external_input_detected"))
        )
        or checkpoint_exists
        or bool(roots),
        "external_input_required": bool(missing_inputs),
        "missing_external_inputs": missing_inputs,
        "sd15_checkpoint_exists": bool(
            upstream.get("sd15_checkpoint_exists", checkpoint_exists)
        ),
        "sd15_checkpoint_required": bool(
            upstream.get("sd15_checkpoint_required", not checkpoint_exists)
        ),
        "new_source_root_count": new_source_root_count,
        "new_source_root_required": bool(
            upstream.get("new_source_root_required", new_source_root_count <= 0)
        ),
        "warm_cache_or_caption_repair_required": bool(
            upstream.get(
                "warm_cache_or_caption_repair_required",
                set(missing_inputs).intersection({"warm_cache_axis", "caption_repair_axis"}),
            )
        ),
        "intake_next_json_refresh_sequence": _strings(upstream.get("next_json_refresh_sequence")),
        "replay_refresh_sequence": list(REQUIRED_REFRESH_SEQUENCE),
        "safe_to_auto_start": False,
        "release_claim_allowed": False,
        "not_release_evidence": True,
    }


def build_external_input_replay_plan(
    *,
    repo_root: Path,
    external_input_intake_registry: Mapping[str, Any] | None = None,
    python_exe: str = "backend/env/python-flashattention/python.exe",
) -> dict[str, Any]:
    """Build a deterministic JSON-only command plan for newly provided inputs."""

    repo = Path(repo_root)
    intake = _mapping(external_input_intake_registry)
    sd15 = _mapping(intake.get("sd15"))
    checkpoint_exists = bool(sd15.get("checkpoint_exists"))
    roots = _new_source_roots(intake)
    families = _families(intake)
    external_detected = bool(intake.get("external_input_detected")) or checkpoint_exists or bool(roots)
    input_resolution_summary = _input_resolution_summary(
        intake=intake,
        checkpoint_exists=checkpoint_exists,
        roots=roots,
    )
    status = "json_replay_ready" if external_detected else "waiting_for_external_input"
    commands: list[dict[str, Any]] = []
    commands.extend(_sd15_commands(repo, python_exe, checkpoint_exists))
    if roots:
        for root in roots:
            commands.extend(_source_root_commands(repo_root=repo, python_exe=python_exe, root=root, families=families, ready=True))
    else:
        commands.extend(
            _source_root_commands(
                repo_root=repo,
                python_exe=python_exe,
                root="",
                families=families,
                ready=False,
            )
        )
    commands.extend(_common_refresh_commands(repo, python_exe, ready=external_detected))
    ready_commands = [item for item in commands if str(item.get("status")) in {"ready", "template_waiting_for_scout_selection"}]
    refresh_sequence_contract = _refresh_sequence_contract(commands)
    return {
        "schema_version": 1,
        "report": EXTERNAL_INPUT_REPLAY_PLAN_REPORT,
        "roadmap": ROADMAP,
        "status": status,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "release_claim_allowed": False,
        "publishable": False,
        "not_release_evidence": True,
        "external_input_detected": external_detected,
        "sd15_checkpoint_exists": checkpoint_exists,
        "new_source_root_count": len(roots),
        "input_resolution_summary": input_resolution_summary,
        "families": families,
        "command_count": len(commands),
        "ready_command_count": len(ready_commands),
        "template_command_count": sum(1 for item in commands if bool(item.get("template"))),
        "refresh_sequence_contract": refresh_sequence_contract,
        "commands": commands,
        "blocked_actions": [
            "auto_start_gpu_heavy_from_replay_plan",
            "promote_replay_plan_as_release_evidence",
            "skip_preflight_after_source_scan",
            "skip_manual_review_before_canary_execution",
        ],
        "acceptance_gates": [
            "replay_plan_is_json_only",
            "source_scan_and_scout_required_before_preflight_template_materialization",
            "manual_canary_plan_requires_admitted_preflight",
            "release_claims_must_be_rebuilt_after_manual_gpu_evidence",
            "terminal_self_check_and_release_guard_required_after_replay",
        ],
        "notes": [
            "This plan only lists JSON-only replay commands and templates.",
            "Template preflight commands must be filled from scout output before execution.",
            "GPU-heavy manual canaries remain separate protected actions and are never auto-started by this plan.",
        ],
    }


__all__ = [
    "EXTERNAL_INPUT_REPLAY_PLAN_REPORT",
    "ROADMAP",
    "build_external_input_replay_plan",
]
