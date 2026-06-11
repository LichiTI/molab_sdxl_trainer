"""A/B matrix helpers for bubble advisor benchmark evidence."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .bubble_runtime_ab_evidence import build_bubble_advisor_ab_evidence_report
from .bubble_runtime_evidence_pack import build_bubble_runtime_evidence_pack, write_bubble_runtime_evidence_pack


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


@dataclass(frozen=True)
class BubbleAdvisorAbCase:
    case_id: str
    family: str
    description: str
    before_args: tuple[str, ...]
    after_args: tuple[str, ...]


def _base_args(
    family: str,
    *,
    steps: int,
    warmup: int,
    samples: int,
    resolution: int,
    batch: int,
    workers: int,
) -> tuple[str, ...]:
    return (
        "--family",
        family,
        "--profiles",
        "standard",
        "--steps",
        str(steps),
        "--steady-warmup",
        str(warmup),
        "--samples",
        str(samples),
        "--resolution",
        str(resolution),
        "--network-dim",
        "1",
        "--train-batch-size",
        str(batch),
        "--dataloader-workers",
        str(workers),
        "--dataloader-prefetch-factor",
        "2",
        "--phase-profile",
    )


def default_bubble_advisor_ab_cases() -> dict[str, BubbleAdvisorAbCase]:
    """Return conservative real-training A/B cases.

    These cases use the existing product trainer benchmark path.  They are short
    enough for smoke evidence but still run the actual local models/datasets.
    """

    anima_data_before = _base_args("anima", steps=16, warmup=4, samples=32, resolution=64, batch=16, workers=0)
    anima_data_after = _base_args("anima", steps=16, warmup=4, samples=32, resolution=64, batch=16, workers=2)
    anima_streaming_before = _base_args(
        "anima",
        steps=16,
        warmup=4,
        samples=32,
        resolution=64,
        batch=16,
        workers=2,
    ) + (
        "--data-transfer-profile",
        "--anima-block-residency",
        "streaming_offload",
    )
    anima_streaming_after = anima_streaming_before + (
        "--anima-block-prefetch",
        "--anima-block-prefetch-depth",
        "1",
    )
    newbie_data_before = _base_args("newbie", steps=8, warmup=2, samples=16, resolution=64, batch=4, workers=0)
    newbie_data_after = _base_args("newbie", steps=8, warmup=2, samples=16, resolution=64, batch=4, workers=2)
    sd15_data_before = _base_args("sd15", steps=4, warmup=1, samples=4, resolution=512, batch=1, workers=0)
    sd15_data_after = _base_args("sd15", steps=4, warmup=1, samples=4, resolution=512, batch=1, workers=2)
    sdxl_data_before = _base_args("sdxl", steps=4, warmup=1, samples=4, resolution=1024, batch=1, workers=0)
    sdxl_data_after = _base_args("sdxl", steps=4, warmup=1, samples=4, resolution=1024, batch=1, workers=2)
    cases = [
        BubbleAdvisorAbCase(
            case_id="anima_data_workers_smoke",
            family="anima",
            description="Anima cache-first data supply: workers 0 vs 2.",
            before_args=anima_data_before,
            after_args=anima_data_after,
        ),
        BubbleAdvisorAbCase(
            case_id="anima_streaming_prefetch_smoke",
            family="anima",
            description="Anima streaming offload transfer overlap: block prefetch off vs depth1.",
            before_args=anima_streaming_before,
            after_args=anima_streaming_after,
        ),
        BubbleAdvisorAbCase(
            case_id="newbie_data_workers_smoke",
            family="newbie",
            description="Newbie/DiT cache-first data supply: workers 0 vs 2.",
            before_args=newbie_data_before,
            after_args=newbie_data_after,
        ),
        BubbleAdvisorAbCase(
            case_id="sd15_data_workers_smoke",
            family="sd15",
            description="SD15 LoRA 512 data supply: workers 0 vs 2.",
            before_args=sd15_data_before,
            after_args=sd15_data_after,
        ),
        BubbleAdvisorAbCase(
            case_id="sdxl_data_workers_smoke",
            family="sdxl",
            description="SDXL LoRA 1024 data supply: workers 0 vs 2.",
            before_args=sdxl_data_before,
            after_args=sdxl_data_after,
        ),
    ]
    return {case.case_id: case for case in cases}


def _report_path(case_dir: Path, side: str) -> Path:
    return case_dir / side / "gpu_bubble_experiment_report.json"


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise ValueError(f"unsupported JSON root in {path}")
    return data


def build_bubble_ab_case_command(
    case: BubbleAdvisorAbCase,
    *,
    side: str,
    case_dir: Path,
    python_executable: Path,
    repo_root: Path,
) -> list[str]:
    if side not in {"before", "after"}:
        raise ValueError(f"unsupported side: {side}")
    side_dir = case_dir / side
    benchmark_args = case.before_args if side == "before" else case.after_args
    return [
        str(python_executable),
        str(repo_root / "backend" / "core" / "lulynx_trainer" / "gpu_bubble_experiment.py"),
        "--out-dir",
        str(side_dir),
        "--",
        str(python_executable),
        str(repo_root / "backend" / "core" / "lulynx_trainer" / "native_runtime_profile_benchmark.py"),
        *benchmark_args,
        "--out",
        str(side_dir),
    ]


def run_bubble_ab_case(
    case: BubbleAdvisorAbCase,
    *,
    output_dir: Path,
    python_executable: Path,
    repo_root: Path,
    reuse_existing: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    case_dir = output_dir / case.case_id
    before_report = _report_path(case_dir, "before")
    after_report = _report_path(case_dir, "after")
    commands = {
        "before": build_bubble_ab_case_command(
            case,
            side="before",
            case_dir=case_dir,
            python_executable=python_executable,
            repo_root=repo_root,
        ),
        "after": build_bubble_ab_case_command(
            case,
            side="after",
            case_dir=case_dir,
            python_executable=python_executable,
            repo_root=repo_root,
        ),
    }
    result: dict[str, Any] = {
        "case_id": case.case_id,
        "family": case.family,
        "description": case.description,
        "commands": commands,
        "before_report": str(before_report),
        "after_report": str(after_report),
        "status": "dry_run" if dry_run else "pending",
    }
    if dry_run:
        return result

    for side, report_path in (("before", before_report), ("after", after_report)):
        if reuse_existing and report_path.is_file():
            continue
        report_path.parent.mkdir(parents=True, exist_ok=True)
        completed = subprocess.run(commands[side], cwd=str(repo_root), check=False)
        if completed.returncode != 0:
            result["status"] = "failed"
            result["failed_side"] = side
            result["return_code"] = int(completed.returncode)
            return result
        if not report_path.is_file():
            result["status"] = "failed"
            result["failed_side"] = side
            result["reason"] = "missing_gpu_bubble_report"
            return result

    ab_report = build_bubble_advisor_ab_evidence_report(_load_json(before_report), _load_json(after_report))
    ab_report["matrix_case"] = {
        "case_id": case.case_id,
        "family": case.family,
        "description": case.description,
    }
    ab_path = case_dir / "bubble_advisor_ab_evidence.json"
    ab_path.write_text(json.dumps(ab_report, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    result["ab_evidence"] = str(ab_path)
    result["ab_status"] = ab_report.get("status")
    result["steady_gain_pct"] = _mapping(ab_report.get("comparison")).get("steady_samples_per_second_gain_pct")
    result["status"] = "ok"
    return result


def build_matrix_pack(output_dir: Path, *, copy_evidence: bool = True) -> dict[str, Any]:
    pack_dir = output_dir / "evidence_pack"
    pack = build_bubble_runtime_evidence_pack([output_dir], output_dir=pack_dir, copy_evidence=copy_evidence)
    paths = write_bubble_runtime_evidence_pack(pack, pack_dir)
    return {"pack": pack, "paths": paths}


__all__ = [
    "BubbleAdvisorAbCase",
    "build_bubble_ab_case_command",
    "build_matrix_pack",
    "default_bubble_advisor_ab_cases",
    "run_bubble_ab_case",
]
