"""Readiness report for the SD15 LoRA 512 GPU-bubble release gap."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .bubble_runtime_ab_matrix import build_bubble_ab_case_command, default_bubble_advisor_ab_cases

REPORT = "bubble_sd15_lora512_release_gap_readiness_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
CASE_ID = "sd15_data_workers_smoke"
RELEASE_CASE_ID = "sd15_lora_512"
SD15_MODEL_DIR_NAMES = ("sd15", "sd1.5")
SD15_CHECKPOINT_CANDIDATES = (
    "v1-5-pruned-emaonly.safetensors",
    "v1-5-pruned.safetensors",
    "sd15.safetensors",
    "model.safetensors",
)
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _digest(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _coerce_arg_value(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        return value


def _args_to_request_config(args: tuple[str, ...]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    index = 0
    while index < len(args):
        token = args[index]
        if not token.startswith("--"):
            index += 1
            continue
        key = token[2:].replace("-", "_")
        next_index = index + 1
        if next_index < len(args) and not args[next_index].startswith("--"):
            config[key] = _coerce_arg_value(args[next_index])
            index += 2
        else:
            config[key] = True
            index += 1
    return config


def sd15_checkpoint_candidates(model_root: Path) -> list[Path]:
    candidates: list[Path] = []
    for dir_name in SD15_MODEL_DIR_NAMES:
        sd15_dir = model_root / dir_name
        candidates.extend(sd15_dir / name for name in SD15_CHECKPOINT_CANDIDATES)
        if sd15_dir.is_dir():
            candidates.extend(path for path in sorted(sd15_dir.glob("*.safetensors")) if path not in candidates)
    return candidates


def _find_sd15_checkpoint(model_root: Path) -> tuple[Path | None, list[Path]]:
    candidates = sd15_checkpoint_candidates(model_root)
    checkpoint = next((path for path in candidates if path.is_file()), None)
    return checkpoint, candidates


def _count_images(source_data: Path) -> int:
    if not source_data.is_dir():
        return 0
    return sum(1 for path in source_data.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def _coverage_state(release_claims: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = _mapping(release_claims)
    rows = payload.get("coverage")
    gaps = payload.get("evidence_gaps")
    coverage_rows = rows if isinstance(rows, list) else []
    gap_rows = gaps if isinstance(gaps, list) else []
    coverage = next(
        (
            _mapping(item)
            for item in coverage_rows
            if str(_mapping(item).get("case_id") or "") == RELEASE_CASE_ID
        ),
        {},
    )
    matching_gaps = [
        {
            "id": str(_mapping(item).get("id") or ""),
            "case_id": str(_mapping(item).get("case_id") or ""),
            "label": str(_mapping(item).get("label") or ""),
        }
        for item in gap_rows
        if str(_mapping(item).get("case_id") or "") == RELEASE_CASE_ID
    ]
    return {
        "release_claims_present": bool(payload),
        "case_id": RELEASE_CASE_ID,
        "covered": bool(coverage.get("covered")),
        "coverage_row": dict(coverage),
        "matching_evidence_gaps": matching_gaps,
    }


def _sd15_manual_ab_envelope(
    *,
    case: Any,
    out_dir: Path,
    source_data: Path,
    checkpoint: Path | None,
    execution_ready: bool,
    evidence_ready: bool,
    dry_run_command: list[str],
    execute_command: list[str],
    side_commands: Mapping[str, list[str]],
    expected_outputs: Mapping[str, str],
) -> dict[str, Any]:
    before_config = _args_to_request_config(tuple(case.before_args))
    after_config = _args_to_request_config(tuple(case.after_args))
    before_digest = _digest(before_config)
    after_digest = _digest(after_config)
    contract_ready = bool(execution_ready)
    if evidence_ready:
        status = "evidence_available_pending_release_claim_refresh"
    elif execution_ready:
        status = "protected_manual_ready"
    else:
        status = "blocked_by_prerequisite"
    post_success_rebuild_contract = {
        "schema_version": 1,
        "contract": "sd15_lora512_post_success_rebuild_contract_v1",
        "contract_ready": contract_ready,
        "post_manual_plan_artifact": (
            "devtools/benchmark_evidence/bubble_runtime/post_manual_evidence_rebuild_plan.json"
        ),
        "required_post_manual_command_ids": [
            "rebuild_current_combined_evidence_pack",
            "refresh_post_manual_evidence_rebuild_plan",
            "refresh_sd15_lora512_release_gap_readiness",
            "refresh_gpu_bubble_readiness_next_actions",
            "refresh_gpu_bubble_terminal_self_check",
            "run_gpu_bubble_release_readiness_guard",
        ],
        "required_rebuild_artifact_ids": [
            "current_combined/evidence_pack.json",
            "current_combined/natural_load_canary.json",
            "current_combined/release_claims.json",
            "post_manual_evidence_rebuild_plan.json",
            "gpu_bubble_experiment_readiness_next_actions.json",
            "gpu_bubble_readiness_terminal_self_check.json",
            "gpu_bubble_release_readiness_guard_report.json",
        ],
        "first_rebuild_command_id": "rebuild_current_combined_evidence_pack",
        "release_claims_rebuild_required": True,
        "natural_load_rebuild_required": True,
        "manual_start_required_after_ready": True,
        "safe_to_auto_start": False,
        "release_claim_allowed_after_success": False,
        "not_release_evidence": True,
    }
    case_config_contract = {
        "schema_version": 1,
        "contract": "sd15_lora512_manual_ab_case_config_contract_v1",
        "case_id": CASE_ID,
        "release_case_id": RELEASE_CASE_ID,
        "family": "sd15",
        "allowed_side_diff_keys": ["dataloader_workers"],
        "manual_runner": "devtools/run_bubble_advisor_ab_matrix.py",
        "planned_out_dir": str(out_dir),
        "source_data": str(source_data),
        "sd15_checkpoint": str(checkpoint) if checkpoint is not None else "",
        "before": {
            "side": "before",
            "dataloader_workers": before_config.get("dataloader_workers"),
            "config_digest": before_digest,
            "request_config": before_config,
        },
        "after": {
            "side": "after",
            "dataloader_workers": after_config.get("dataloader_workers"),
            "config_digest": after_digest,
            "request_config": after_config,
        },
        "common": {
            "profiles": before_config.get("profiles"),
            "steps": before_config.get("steps"),
            "steady_warmup": before_config.get("steady_warmup"),
            "samples": before_config.get("samples"),
            "resolution": before_config.get("resolution"),
            "network_dim": before_config.get("network_dim"),
            "train_batch_size": before_config.get("train_batch_size"),
            "dataloader_prefetch_factor": before_config.get("dataloader_prefetch_factor"),
            "phase_profile": bool(before_config.get("phase_profile")),
        },
        "contract_ready": contract_ready,
    }
    return {
        "schema_version": 1,
        "envelope": "sd15_lora512_manual_ab_envelope_v1",
        "status": status,
        "case_id": CASE_ID,
        "release_case_id": RELEASE_CASE_ID,
        "family": "sd15",
        "manual_start_required_after_ready": True,
        "safe_to_auto_start": False,
        "release_claim_allowed_after_success": False,
        "requires_gpu_if_executed": True,
        "planned_out_dir": str(out_dir),
        "manual_execute_command": execute_command,
        "manual_dry_run_command": dry_run_command,
        "before_command": list(side_commands.get("before") or []),
        "after_command": list(side_commands.get("after") or []),
        "expected_outputs": [str(value) for value in expected_outputs.values() if value],
        "case_config_contract": case_config_contract,
        "post_success_rebuild_contract": post_success_rebuild_contract,
        "post_run_refresh_sequence": [
            "refresh_post_manual_evidence_rebuild_plan",
            "refresh_sd15_lora512_release_gap_readiness",
            "refresh_gpu_bubble_readiness_next_actions",
            "refresh_gpu_bubble_terminal_self_check",
            "run_gpu_bubble_release_readiness_guard",
        ],
    }


def build_sd15_lora512_release_gap_readiness(
    *,
    repo_root: Path,
    out_dir: Path,
    python_executable: Path,
    release_claims: Mapping[str, Any] | None = None,
    source_data: Path | None = None,
) -> dict[str, Any]:
    """Build a JSON-only readiness report without starting training or CUDA."""

    repo_root = repo_root.resolve()
    out_dir = out_dir.resolve()
    python_executable = python_executable.resolve()
    source_data = (source_data or repo_root / "sucai" / "6_lulu").resolve()
    case = default_bubble_advisor_ab_cases()[CASE_ID]
    case_dir = out_dir / case.case_id
    commands = {
        side: build_bubble_ab_case_command(
            case,
            side=side,
            case_dir=case_dir,
            python_executable=python_executable,
            repo_root=repo_root,
        )
        for side in ("before", "after")
    }
    before_report = case_dir / "before" / "gpu_bubble_experiment_report.json"
    after_report = case_dir / "after" / "gpu_bubble_experiment_report.json"
    ab_evidence = case_dir / "bubble_advisor_ab_evidence.json"
    checkpoint, checkpoint_candidates = _find_sd15_checkpoint(repo_root / "models")
    source_image_count = _count_images(source_data)

    blockers: list[str] = []
    if not python_executable.is_file():
        blockers.append("python_executable_missing")
    if checkpoint is None:
        blockers.append("sd15_base_checkpoint_missing")
    if not source_data.is_dir():
        blockers.append("source_data_missing")
    elif source_image_count <= 0:
        blockers.append("source_images_missing")

    coverage = _coverage_state(release_claims)
    evidence_ready = before_report.is_file() and after_report.is_file() and ab_evidence.is_file()
    execution_ready = not blockers
    if bool(coverage["covered"]):
        status = "release_gap_covered"
    elif evidence_ready:
        status = "evidence_available_pending_release_claim_refresh"
    elif execution_ready:
        status = "ready_for_manual_gpu_run"
    else:
        status = "blocked_missing_prerequisite"

    script = repo_root / "devtools" / "run_bubble_advisor_ab_matrix.py"
    dry_run_command = [
        str(python_executable),
        str(script),
        "--case",
        CASE_ID,
        "--dry-run",
        "--out-dir",
        str(out_dir),
    ]
    execute_command = [
        str(python_executable),
        str(script),
        "--case",
        CASE_ID,
        "--reuse-existing",
        "--out-dir",
        str(out_dir),
    ]
    expected_outputs = {
        "before_report": str(before_report),
        "after_report": str(after_report),
        "ab_evidence": str(ab_evidence),
        "matrix_results": str(out_dir / "ab_matrix_results.json"),
        "evidence_pack": str(out_dir / "evidence_pack" / "bubble_runtime_evidence_pack.json"),
    }
    manual_ab_envelope = _sd15_manual_ab_envelope(
        case=case,
        out_dir=out_dir,
        source_data=source_data,
        checkpoint=checkpoint,
        execution_ready=execution_ready,
        evidence_ready=evidence_ready,
        dry_run_command=dry_run_command,
        execute_command=execute_command,
        side_commands=commands,
        expected_outputs=expected_outputs,
    )
    return {
        "schema_version": 1,
        "report": REPORT,
        "roadmap": ROADMAP,
        "status": status,
        "case_id": CASE_ID,
        "release_case_id": RELEASE_CASE_ID,
        "family": "sd15",
        "description": case.description,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "requires_explicit_manual_gpu_run": True,
        "requires_gpu_if_executed": True,
        "release_claim_allowed": False,
        "publishable": False,
        "not_release_evidence": True,
        "execution_ready": execution_ready,
        "evidence_ready": evidence_ready,
        "blockers": blockers,
        "prerequisites": {
            "python_executable": str(python_executable),
            "python_executable_exists": python_executable.is_file(),
            "sd15_checkpoint": str(checkpoint) if checkpoint is not None else "",
            "sd15_checkpoint_exists": checkpoint is not None,
            "sd15_checkpoint_candidates": [str(path) for path in checkpoint_candidates],
            "source_data": str(source_data),
            "source_data_exists": source_data.is_dir(),
            "source_image_count": source_image_count,
        },
        "release_coverage": coverage,
        "commands": {
            "dry_run": dry_run_command,
            "execute_manual_gpu": execute_command,
            "before": commands["before"],
            "after": commands["after"],
        },
        "expected_outputs": expected_outputs,
        "manual_ab_envelope": manual_ab_envelope,
        "output_status": {
            "before_report_exists": before_report.is_file(),
            "after_report_exists": after_report.is_file(),
            "ab_evidence_exists": ab_evidence.is_file(),
        },
        "acceptance_gates": [
            "explicit_manual_gpu_run_only",
            "before_and_after_gpu_bubble_reports_exist",
            "bubble_advisor_ab_evidence_exists",
            "loss_delta_present",
            "vram_evidence_present",
            "positive_throughput_gain_required_for_release_claim",
            "rebuild_release_claims_and_clear_sd15_lora_512_gap",
        ],
        "blocked_actions": [
            "claim_sd15_lora_512_release_coverage_without_ab_evidence",
            "promote_universal_gpu_utilization_claim",
            "auto_start_gpu_heavy_ab_matrix",
        ],
    }


__all__ = [
    "CASE_ID",
    "RELEASE_CASE_ID",
    "REPORT",
    "ROADMAP",
    "SD15_MODEL_DIR_NAMES",
    "build_sd15_lora512_release_gap_readiness",
    "sd15_checkpoint_candidates",
]
