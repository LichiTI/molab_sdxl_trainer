"""Protected manual canary plan from a source/cache-axis admission preflight."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


SOURCE_CACHE_AXIS_MANUAL_CANARY_PLAN_REPORT = "bubble_source_cache_axis_manual_canary_plan_v0"
ROADMAP = "gpu_bubble_elimination_roadmap.md"
PREFLIGHT_REPORT = "bubble_source_cache_axis_admission_preflight_v0"


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


def _family_key(value: Any) -> str:
    family = str(value or "").strip().lower().replace("-", "_")
    return "newbie" if family in {"dit", "newbie_dit"} else family


def _default_python_executable(repo_root: Path) -> Path:
    flashattention = repo_root / "backend" / "env" / "python-flashattention" / "python.exe"
    if flashattention.is_file():
        return flashattention
    return repo_root / "backend" / "env" / "python_launcher" / "python.exe"


def _family_flags(family: str) -> tuple[str, ...]:
    if family == "sdxl":
        return (
            "--sdxl-resolution",
            "1024",
            "--sdxl-samples",
            "8",
            "--sdxl-steps",
            "16",
            "--sdxl-warmup",
            "4",
            "--sdxl-tune-interval",
            "4",
            "--min-throughput-gain",
            "0.03",
        )
    if family == "anima":
        return (
            "--anima-resolution",
            "64",
            "--anima-samples",
            "8",
            "--anima-steps",
            "24",
            "--anima-warmup",
            "4",
            "--anima-tune-interval",
            "8",
            "--min-throughput-gain",
            "0.03",
        )
    if family == "newbie":
        return (
            "--newbie-resolution",
            "64",
            "--newbie-samples",
            "8",
            "--newbie-steps",
            "16",
            "--newbie-warmup",
            "4",
            "--newbie-tune-interval",
            "4",
            "--min-throughput-gain",
            "0.03",
        )
    return ("--min-throughput-gain", "0.03")


def _short_key(value: Any) -> str:
    text = "".join(char for char in str(value or "") if char.isalnum()).lower()
    return (text[:10] or "nohash")


def _out_dir(out_root: Path, *, family: str, sample_offset: int, source_manifest_sha1: str) -> Path:
    key = _short_key(source_manifest_sha1)
    return out_root / f"real_material_canary_{family}_offset{sample_offset}_{key}_preflight_manual"


def _existing_evidence(out_dir: Path) -> dict[str, Any]:
    required = [
        out_dir / "real_material_canary_results.json",
        out_dir / "evidence_pack" / "evidence_pack.json",
        out_dir / "evidence_pack" / "natural_load_canary.json",
        out_dir / "evidence_pack" / "release_claims.json",
    ]
    ab_files = sorted(out_dir.glob("*_real_material_canary_ab_evidence.json")) if out_dir.exists() else []
    missing = [str(path) for path in required if not path.exists()]
    if not ab_files:
        missing.append(str(out_dir / "*_real_material_canary_ab_evidence.json"))
    evidence_paths = [str(path) for path in required if path.exists()]
    evidence_paths.extend(str(path) for path in ab_files)
    return {
        "completed": not missing,
        "evidence_paths": evidence_paths,
        "missing_paths": missing,
    }


def _command(
    *,
    repo_root: Path,
    python_executable: Path,
    family: str,
    source_data: Path,
    sample_offset: int,
    out_dir: Path,
) -> list[str]:
    return [
        str(python_executable),
        str(repo_root / "devtools" / "run_bubble_real_material_canary.py"),
        "--family",
        family,
        "--source-data",
        str(source_data),
        "--sample-offset",
        str(max(sample_offset, 0)),
        *_family_flags(family),
        "--out-dir",
        str(out_dir),
    ]


def build_source_cache_axis_manual_canary_plan(
    preflight: Mapping[str, Any],
    *,
    repo_root: Path,
    out_root: Path | None = None,
    python_executable: Path | None = None,
) -> dict[str, Any]:
    """Build a protected manual canary command only when preflight admits the axis."""

    repo = Path(repo_root)
    out_base = out_root or repo / "devtools" / "benchmark_evidence" / "bubble_runtime"
    python = python_executable or _default_python_executable(repo)
    preflight_status = str(preflight.get("status") or "")
    candidate = _mapping(preflight.get("candidate"))
    family = _family_key(candidate.get("family"))
    sample_offset = _safe_int(candidate.get("sample_offset"))
    source_root = str(candidate.get("root") or "")
    source_manifest_sha1 = str(candidate.get("source_manifest_sha1") or "")
    preflight_admitted = (
        str(preflight.get("report") or "") == PREFLIGHT_REPORT
        and preflight_status == "admitted"
        and bool(preflight.get("admission_allows_protected_manual_gpu_plan"))
    )
    blockers: list[str] = []
    if str(preflight.get("report") or "") != PREFLIGHT_REPORT:
        blockers.append("preflight_report_missing_or_unrecognized")
    if not preflight_admitted:
        blockers.append("source_cache_axis_preflight_not_admitted")
        blockers.extend(_strings(preflight.get("blockers")))
    if not family:
        blockers.append("candidate_family_missing")
    if not source_root:
        blockers.append("candidate_root_missing")

    commands: list[dict[str, Any]] = []
    if preflight_admitted and family and source_root:
        target_out = _out_dir(
            out_base,
            family=family,
            sample_offset=sample_offset,
            source_manifest_sha1=source_manifest_sha1,
        )
        command = _command(
            repo_root=repo,
            python_executable=python,
            family=family,
            source_data=Path(source_root),
            sample_offset=sample_offset,
            out_dir=target_out,
        )
        commands.append(
            {
                "id": f"{family}_offset{sample_offset}_{_short_key(source_manifest_sha1)}_manual_canary",
                "family": family,
                "profile": "release_relevant_conservative",
                "release_relevant": True,
                "diagnostic_only": False,
                "source_data": source_root,
                "sample_offset": sample_offset,
                "source_manifest_sha1": source_manifest_sha1,
                "out_dir": str(target_out),
                "python_executable": str(python),
                "manual_execute_command": command,
                "dry_run_command": [*command, "--dry-run"],
                "requires_gpu_if_executed": True,
                "manual_start_required": True,
                "safe_to_auto_start": False,
                "release_claim_allowed_after_success": False,
                "post_run_review_required": True,
                "existing_evidence": _existing_evidence(target_out),
            }
        )

    status = "protected_manual_canary_plan_ready" if commands else "blocked_by_preflight"
    return {
        "schema_version": 1,
        "report": SOURCE_CACHE_AXIS_MANUAL_CANARY_PLAN_REPORT,
        "roadmap": ROADMAP,
        "status": status,
        "preflight_report": str(preflight.get("report") or ""),
        "preflight_status": preflight_status,
        "preflight_admitted": preflight_admitted,
        "safe_to_auto_start": False,
        "does_not_run_training": True,
        "does_not_run_cuda": True,
        "requires_gpu_if_executed": bool(commands),
        "release_claim_allowed": False,
        "publishable": False,
        "not_release_evidence": True,
        "command_count": len(commands),
        "blocked_command_count": 0 if commands else 1,
        "blockers": sorted(set(blockers)),
        "commands": commands,
        "blocked_actions": [
            "auto_start_manual_canary_plan",
            "promote_manual_canary_plan_as_release_evidence",
            "skip_natural_load_release_claim_rebuild_after_manual_run",
        ],
        "acceptance_gates_after_manual_run": [
            "real_material_canary_results_present",
            "family_ab_evidence_present",
            "natural_load_canary_rebuilt",
            "release_claims_rebuilt",
            "gpu_bubble_readiness_rebuilt",
            "case_specific_release_wording_only",
        ],
        "notes": [
            "This plan is JSON-only and does not start GPU work.",
            "A ready plan still requires explicit manual execution and post-run evidence rebuild.",
        ],
    }


__all__ = [
    "ROADMAP",
    "SOURCE_CACHE_AXIS_MANUAL_CANARY_PLAN_REPORT",
    "build_source_cache_axis_manual_canary_plan",
]
