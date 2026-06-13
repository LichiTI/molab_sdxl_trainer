"""Real-training matrix helpers for P7 bubble closed-loop evidence."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Mapping

from .bubble_natural_data_wait_evidence import (
    NATURAL_DATA_WAIT_EVIDENCE_REPORT,
    build_bubble_natural_data_wait_evidence_report,
)
from .bubble_runtime_closed_loop_case_defs import BubbleClosedLoopCase, default_bubble_closed_loop_cases
from .bubble_runtime_closed_loop_evidence import build_bubble_closed_loop_evidence_report
from .bubble_runtime_evidence_pack import build_bubble_runtime_evidence_pack, write_bubble_runtime_evidence_pack
from .bubble_real_material_canary import (
    REAL_MATERIAL_CANARY_FIXTURE,
    prepare_real_material_canary_source,
)
from .bubble_runtime_source_fixtures import (
    prepare_heavy_raw_decode_cache_miss_mixed_sidecar_source,
    prepare_heavy_raw_decode_mixed_sidecar_source,
    prepare_heavy_raw_decode_source,
)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _source_fixture_dir(case: BubbleClosedLoopCase, case_dir: Path) -> Path | None:
    if not case.source_fixture:
        return None
    return case_dir / "source_data"


def _benchmark_args_for_case(case: BubbleClosedLoopCase, case_dir: Path) -> list[str]:
    args = list(case.benchmark_args)
    source_dir = _source_fixture_dir(case, case_dir)
    if source_dir is not None:
        args.extend(["--source-data", str(source_dir)])
    return args


def _arg_value(args: tuple[str, ...], name: str, default: str = "") -> str:
    try:
        index = args.index(name)
    except ValueError:
        return default
    if index + 1 >= len(args):
        return default
    return str(args[index + 1])


def _arg_int(args: tuple[str, ...], name: str, default: int = 0) -> int:
    try:
        return int(_arg_value(args, name, str(default)))
    except (TypeError, ValueError):
        return int(default)


def _resolve_fixture_source(case: BubbleClosedLoopCase, repo_root: Path) -> Path:
    source = Path(case.source_fixture_source or "sucai/6_lulu")
    if not source.is_absolute():
        source = repo_root / source
    return source


def _prepare_source_fixture(case: BubbleClosedLoopCase, case_dir: Path, *, repo_root: Path) -> dict[str, Any]:
    source_dir = _source_fixture_dir(case, case_dir)
    if source_dir is None:
        return {}
    if case.source_fixture == REAL_MATERIAL_CANARY_FIXTURE:
        return prepare_real_material_canary_source(
            _resolve_fixture_source(case, repo_root),
            source_dir,
            family=case.family,
            samples=case.source_fixture_samples,
            sample_offset=case.source_fixture_sample_offset,
            native_cache_mode=_arg_value(case.benchmark_args, "--native-cache-mode", "cache_first"),
            label=case.case_id,
        )
    if case.source_fixture == "heavy_raw_decode_png_v0":
        return prepare_heavy_raw_decode_source(
            source_dir,
            samples=case.source_fixture_samples,
            size=case.source_fixture_size,
            seed=case.source_fixture_seed,
        )
    if case.source_fixture == "heavy_raw_decode_mixed_sidecars_v0":
        return prepare_heavy_raw_decode_mixed_sidecar_source(
            source_dir,
            samples=case.source_fixture_samples,
            size=case.source_fixture_size,
            seed=case.source_fixture_seed,
        )
    if case.source_fixture == "heavy_raw_decode_cache_miss_mixed_sidecars_v0":
        return prepare_heavy_raw_decode_cache_miss_mixed_sidecar_source(
            source_dir,
            samples=case.source_fixture_samples,
            size=case.source_fixture_size,
            seed=case.source_fixture_seed,
        )
    raise ValueError(f"unsupported source fixture {case.source_fixture!r}")


def build_bubble_closed_loop_case_command(
    case: BubbleClosedLoopCase,
    *,
    case_dir: Path,
    python_executable: Path,
    repo_root: Path,
) -> list[str]:
    run_dir = case_dir / "run"
    return [
        str(python_executable),
        str(repo_root / "backend" / "core" / "lulynx_trainer" / "gpu_bubble_experiment.py"),
        "--out-dir",
        str(run_dir),
        "--",
        str(python_executable),
        str(repo_root / "backend" / "core" / "lulynx_trainer" / "native_runtime_profile_benchmark.py"),
        *_benchmark_args_for_case(case, case_dir),
        "--out",
        str(run_dir),
    ]


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"unsupported JSON root in {path}")
    return data


def _annotate_gpu_bubble_report_case(path: Path, case: BubbleClosedLoopCase) -> None:
    report = _load_json(path)
    if str(report.get("report") or "") != "gpu_bubble_experiment_report_v0":
        return
    benchmark = dict(_mapping(report.get("benchmark")))
    updated = False
    for key, value in (
        ("case_id", case.case_id),
        ("case", case.case_id),
        ("family", case.family),
        ("description", case.description),
    ):
        if benchmark.get(key):
            continue
        benchmark[key] = value
        updated = True
    if updated or not isinstance(report.get("benchmark"), Mapping):
        report["benchmark"] = benchmark
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _manifest_has_closed_loop_state(path: Path) -> bool:
    try:
        extra = _mapping(_load_json(path).get("extra"))
    except Exception:
        return False
    controller = _mapping(extra.get("bubble_controller"))
    closed_loop = _mapping(controller.get("closed_loop"))
    executor = _mapping(closed_loop.get("executor"))
    return bool(_mapping(extra.get("bubble_closed_loop_state")) or executor)


def _manifest_paths(case_dir: Path) -> list[Path]:
    return sorted(path for path in case_dir.rglob("run_manifest.json") if path.is_file())


def _manifest_evidence_status(path: Path) -> str:
    try:
        report = build_bubble_closed_loop_evidence_report(_load_json(path))
    except Exception:
        return ""
    return str(report.get("status") or "")


def _real_material_missing_family_cache(case: BubbleClosedLoopCase, fixture: Mapping[str, Any]) -> bool:
    if case.source_fixture != REAL_MATERIAL_CANARY_FIXTURE:
        return False
    if str(case.family or "").strip().lower() == "sd15":
        return False
    return fixture.get("cache_has_family_cache") is not True


def _real_material_axes(case: BubbleClosedLoopCase, fixture: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "family": case.family,
        "source_fixture": str(fixture.get("fixture") or case.source_fixture),
        "cache_state": str(fixture.get("cache_state") or ""),
        "cache_present_before": bool(fixture.get("cache_present_before")),
        "cache_has_family_cache": bool(fixture.get("cache_has_family_cache")),
        "native_cache_mode": str(fixture.get("native_cache_mode") or _arg_value(case.benchmark_args, "--native-cache-mode", "")),
        "fixture_samples": int(fixture.get("source_image_count") or fixture.get("samples") or 0),
        "source_file_count": int(fixture.get("source_file_count") or 0),
        "source_manifest_sha1": str(fixture.get("source_manifest_sha1") or ""),
        "dataloader_workers": _arg_int(case.benchmark_args, "--dataloader-workers", 0),
        "dataloader_prefetch_factor": _arg_int(case.benchmark_args, "--dataloader-prefetch-factor", 0),
        "pin_memory": "--no-pin-memory" not in case.benchmark_args,
    }


def _write_real_material_cache_blocked_evidence(
    case: BubbleClosedLoopCase,
    case_dir: Path,
    fixture: Mapping[str, Any],
) -> dict[str, Any]:
    axes = _real_material_axes(case, fixture)
    cache_state = str(axes.get("cache_state") or "missing")
    reasons = [
        "real_material_family_cache_missing",
        f"real_material_cache_state_{cache_state}",
    ]
    report = {
        "schema_version": 1,
        "report": NATURAL_DATA_WAIT_EVIDENCE_REPORT,
        "status": "blocked_real_material_family_cache_missing",
        "case_id": case.case_id,
        "family": case.family,
        "metrics": {
            "dominant_bottleneck": "cache_readiness_blocked",
            "data_wait_share": 0.0,
            "h2d_transfer_share": 0.0,
            "optimizer_share": 0.0,
            "host_gap_share": 0.0,
            "steady_samples_per_second": 0.0,
            "peak_vram_mb": 0.0,
        },
        "loss_stability": {"status": "missing"},
        "analysis": {
            "matrix_axes": axes,
            "cache_inventory": dict(_mapping(fixture.get("cache_inventory"))),
        },
        "action_chain": [],
        "decision": {
            "cache_probe_only": True,
            "dataloader_rebuild_observed": False,
            "recommended_action": "prepare_family_cache_before_canary",
            "reasons": sorted(set(reasons)),
        },
        "benchmark_injection_blockers": [],
        "release_claim": {
            "eligible": False,
            "scope": "not_eligible",
            "reason": "real-material canary requires warm family cache before running release-candidate training evidence",
        },
    }
    out_path = case_dir / "natural_data_wait_evidence.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "status": str(report["status"]),
        "path": str(out_path),
        "summary": "",
        "manifest": "",
        "metrics": dict(_mapping(report.get("metrics"))),
    }


def _summary_paths(case_dir: Path, family: str) -> list[Path]:
    preferred = case_dir / "run" / f"{family}_summary.json"
    if preferred.is_file():
        return [preferred]
    return sorted(path for path in case_dir.rglob("*_summary.json") if path.is_file())


def _write_natural_data_wait_evidence(
    case: BubbleClosedLoopCase,
    case_dir: Path,
    *,
    manifest_paths: list[Path],
) -> dict[str, Any]:
    summaries = _summary_paths(case_dir, case.family)
    if not summaries:
        return {"status": "", "reason": "missing_summary"}
    summary_path = summaries[0]
    manifest_path = manifest_paths[0] if manifest_paths else None
    source_fixture_path = _source_fixture_dir(case, case_dir)
    source_fixture_manifest = (
        source_fixture_path / "fixture_manifest.json"
        if source_fixture_path is not None
        else None
    )
    report = build_bubble_natural_data_wait_evidence_report(
        _load_json(summary_path),
        manifest=_load_json(manifest_path) if manifest_path else None,
        source_fixture=_load_json(source_fixture_manifest)
        if source_fixture_manifest is not None and source_fixture_manifest.is_file()
        else None,
        case_id=case.case_id,
        preferred_label="standard",
    )
    out_path = case_dir / "natural_data_wait_evidence.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "status": str(report.get("status") or ""),
        "path": str(out_path),
        "summary": str(summary_path),
        "manifest": str(manifest_path) if manifest_path else "",
        "metrics": dict(_mapping(report.get("metrics"))),
    }


def run_bubble_closed_loop_case(
    case: BubbleClosedLoopCase,
    *,
    output_dir: Path,
    python_executable: Path,
    repo_root: Path,
    reuse_existing: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    case_dir = output_dir / case.case_id
    report_path = case_dir / "run" / "gpu_bubble_experiment_report.json"
    command = build_bubble_closed_loop_case_command(
        case,
        case_dir=case_dir,
        python_executable=python_executable,
        repo_root=repo_root,
    )
    result: dict[str, Any] = {
        "case_id": case.case_id,
        "family": case.family,
        "description": case.description,
        "command": command,
        "gpu_bubble_report": str(report_path),
        "status": "dry_run" if dry_run else "pending",
    }
    source_dir = _source_fixture_dir(case, case_dir)
    if source_dir is not None:
        result["source_fixture"] = {
            "fixture": case.source_fixture,
            "source_data": str(source_dir),
            "prepared": False,
        }
    if dry_run:
        return result

    if not (reuse_existing and report_path.is_file()):
        report_path.parent.mkdir(parents=True, exist_ok=True)
        fixture = _prepare_source_fixture(case, case_dir, repo_root=repo_root)
        if fixture:
            result["source_fixture"] = {**fixture, "source_data": str(source_dir), "prepared": True}
        if _real_material_missing_family_cache(case, fixture):
            result["status"] = "blocked_missing_family_cache"
            result["reason"] = "real_material_family_cache_missing"
            result["natural_data_wait_evidence"] = _write_real_material_cache_blocked_evidence(
                case,
                case_dir,
                fixture,
            )
            return result
        completed = subprocess.run(command, cwd=str(repo_root), check=False)
        if completed.returncode != 0:
            result["status"] = "failed"
            result["return_code"] = int(completed.returncode)
            return result
    if not report_path.is_file():
        result["status"] = "failed"
        result["reason"] = "missing_gpu_bubble_report"
        return result
    _annotate_gpu_bubble_report_case(report_path, case)

    manifests = _manifest_paths(case_dir)
    closed_loop_manifests = [path for path in manifests if _manifest_has_closed_loop_state(path)]
    result["run_manifests"] = [str(path) for path in manifests]
    result["closed_loop_manifests"] = [str(path) for path in closed_loop_manifests]
    result["closed_loop_manifest_count"] = len(closed_loop_manifests)
    if not closed_loop_manifests:
        result["status"] = "failed"
        result["reason"] = "missing_closed_loop_state"
        return result
    evidence_statuses = [_manifest_evidence_status(path) for path in closed_loop_manifests]
    evidence_statuses = [status for status in evidence_statuses if status]
    result["closed_loop_evidence_statuses"] = evidence_statuses
    expected = set(case.expected_evidence_statuses or ())
    if expected and not any(status in expected for status in evidence_statuses):
        result["status"] = "failed"
        result["reason"] = "unexpected_closed_loop_evidence_status"
        result["expected_closed_loop_evidence_statuses"] = list(case.expected_evidence_statuses)
        return result
    if case.build_natural_data_wait_evidence or case.expected_natural_data_wait_statuses:
        natural = _write_natural_data_wait_evidence(
            case,
            case_dir,
            manifest_paths=closed_loop_manifests,
        )
        result["natural_data_wait_evidence"] = natural
        natural_status = str(natural.get("status") or "")
        expected_natural = set(case.expected_natural_data_wait_statuses or ())
        if expected_natural and natural_status not in expected_natural:
            result["status"] = "failed"
            result["reason"] = "unexpected_natural_data_wait_evidence_status"
            result["expected_natural_data_wait_statuses"] = list(case.expected_natural_data_wait_statuses)
            return result
    result["status"] = "ok"
    return result


def build_closed_loop_matrix_pack(output_dir: Path, *, copy_evidence: bool = True) -> dict[str, Any]:
    pack_dir = output_dir / "evidence_pack"
    pack = build_bubble_runtime_evidence_pack([output_dir], output_dir=pack_dir, copy_evidence=copy_evidence)
    paths = write_bubble_runtime_evidence_pack(pack, pack_dir)
    return {"pack": pack, "paths": paths}


__all__ = [
    "BubbleClosedLoopCase",
    "build_bubble_closed_loop_case_command",
    "build_closed_loop_matrix_pack",
    "default_bubble_closed_loop_cases",
    "prepare_heavy_raw_decode_cache_miss_mixed_sidecar_source",
    "prepare_heavy_raw_decode_source",
    "run_bubble_closed_loop_case",
]
