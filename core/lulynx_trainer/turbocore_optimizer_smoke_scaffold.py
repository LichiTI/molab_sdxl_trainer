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

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_smoke_scaffold_audit",
        "ok": True,
        "roadmap": roadmap,
        "suite_entrypoint": "backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py",
        "profile": profile,
        "profile_guidance": profile_guidance,
        "selected_profile_smoke_count": len(selected_smokes),
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
            "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile release",
            "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --profile batch",
            "python backend/core/lulynx_trainer/turbocore_optimizer_smoke_suite.py --audit-scaffold",
        ],
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


def _spec_dict(spec: Any) -> dict[str, Any]:
    if hasattr(spec, "__dict__"):
        return dict(spec.__dict__)
    return {
        "smoke_id": str(getattr(spec, "smoke_id", "")),
        "module": str(getattr(spec, "module", "")),
        "tier": str(getattr(spec, "tier", "")),
        "description": str(getattr(spec, "description", "")),
    }
