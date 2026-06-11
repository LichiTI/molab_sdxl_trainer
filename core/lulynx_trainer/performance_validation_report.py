# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Aggregate staged performance validation reports for the roadmap.

This is a report-only helper. It does not launch training by default; it reads
existing smoke/A-B JSON files and produces one compact summary for roadmap
items:
- 1-4: short real benchmarks, low-VRAM boundary, Newbie cache-first profile,
  and experimental strategy A/B.
- 5.x / >40-step: optional extended reports for dependency/long-train status.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"ok": False, "path": str(path), "error": f"{type(exc).__name__}: {exc}"}


def _losses_from_log_tail(log_tail: Iterable[str]) -> List[float]:
    losses: List[float] = []
    for line in log_tail or []:
        text = str(line)
        marker = "Loss:"
        if marker not in text:
            continue
        try:
            losses.append(float(text.split(marker, 1)[1].strip().split()[0]))
        except Exception:
            pass
    return losses


def _case_summary(case: Dict[str, Any]) -> Dict[str, Any]:
    payload = case.get("payload") if isinstance(case.get("payload"), dict) else {}
    result = (payload.get("results") or [{}])[0] if isinstance(payload.get("results"), list) else {}
    copy_report = result.get("copy_report") if isinstance(result.get("copy_report"), dict) else {}
    return {
        "case": case.get("case", ""),
        "ok": bool(case.get("ok", False)),
        "duration_seconds": case.get("duration_seconds", 0.0),
        "payload_duration_seconds": payload.get("duration_seconds", 0.0),
        "resolved_steps": payload.get("resolved_steps", result.get("resolved_steps", 0)),
        "artifact": result.get("artifact", ""),
        "cached_pairs": copy_report.get("cached_pairs", 0),
        "source_data_reused": bool(case.get("source_data_reused", False)),
        "optimizer_runtime": result.get("optimizer_runtime", {}),
        "memory_optimization": result.get("memory_optimization", {}),
        "newbie_cache_first_profile": result.get("newbie_cache_first_profile", {}),
        "losses": _losses_from_log_tail(result.get("log_tail") or []),
    }


def _ab_summary(path: Path) -> Dict[str, Any]:
    data = _read_json(path)
    results = data.get("results") if isinstance(data.get("results"), list) else []
    cases = [_case_summary(case) for case in results if isinstance(case, dict)]
    return {
        "path": str(path),
        "ok": bool(data.get("ok", False)),
        "probe": data.get("probe", ""),
        "experiment": data.get("experiment", ""),
        "family": data.get("family", ""),
        "duration_seconds": data.get("duration_seconds", 0.0),
        "reuse_newbie_cache": bool(data.get("reuse_newbie_cache", False)),
        "reusable_newbie_source": data.get("reusable_newbie_source", ""),
        "cases": cases,
    }


def _long_quality_summary(path: Path) -> Dict[str, Any]:
    data = _read_json(path)
    results = data.get("results") if isinstance(data.get("results"), list) else []
    first = results[0] if results and isinstance(results[0], dict) else {}
    return {
        "path": str(path),
        "category": "long_quality",
        "ok": bool(data.get("ok", False)),
        "resolved_steps": int(data.get("resolved_steps", first.get("resolved_steps", 0)) or 0),
        "duration_seconds": float(data.get("duration_seconds", first.get("duration_seconds", 0.0)) or 0.0),
        "artifact": str(first.get("artifact", "")),
        "family": str(first.get("family", "")),
        "adapter": str(first.get("adapter", "")),
    }


def _high_dependency_summary(path: Path) -> Dict[str, Any]:
    data = _read_json(path)
    fp8 = data.get("fp8_transformer_engine") if isinstance(data.get("fp8_transformer_engine"), dict) else {}
    fp8_profile = fp8.get("profile") if isinstance(fp8.get("profile"), dict) else {}
    return {
        "path": str(path),
        "category": "high_dependency_probe",
        "ok": bool(data.get("ok", False)),
        "scope": str(data.get("scope", "")),
        "fp8_requested": bool(fp8_profile.get("requested", False)),
        "fp8_resolved": str(fp8_profile.get("resolved", "")),
        "fp8_fallback_reason": str(fp8_profile.get("fallback_reason", "")),
        "te_available": bool(
            (data.get("capabilities") or {}).get("transformer_engine", {}).get("available", False)
            if isinstance(data.get("capabilities"), dict)
            else False
        ),
    }


def _te_install_probe_summary(path: Path) -> Dict[str, Any]:
    data = _read_json(path)
    decision = data.get("decision") if isinstance(data.get("decision"), dict) else {}
    commands = data.get("commands") if isinstance(data.get("commands"), list) else []
    return {
        "path": str(path),
        "category": "te_install_probe",
        "ok": bool(data.get("transformer_engine_importable", False)),
        "resolved": str(decision.get("resolved", "")),
        "reason": str(decision.get("reason", "")),
        "training_ab_status": str(decision.get("training_ab_status", "")),
        "command_count": len(commands),
        "failing_commands": [
            str(item.get("command", ""))
            for item in commands
            if isinstance(item, dict) and not bool(item.get("ok", False))
        ],
    }


def _generic_summary(path: Path) -> Dict[str, Any]:
    payload = _read_json(path)
    return {
        "path": str(path),
        "category": "generic",
        "ok": bool(payload.get("ok", True)),
        "payload": payload,
    }


def _summarize_path(path: Path) -> Dict[str, Any]:
    name = path.name
    if name.startswith("real_strategy_"):
        item = _ab_summary(path)
        item["category"] = "short_ab"
        return item
    if name.startswith("real_quality_long_"):
        return _long_quality_summary(path)
    if "high_dependency_performance_probe" in name:
        return _high_dependency_summary(path)
    if "transformer_engine_install_probe" in name:
        return _te_install_probe_summary(path)
    return _generic_summary(path)


def _collect_summaries(root: Path, raw_paths: Iterable[str]) -> tuple[List[Dict[str, Any]], List[str]]:
    summaries: List[Dict[str, Any]] = []
    missing: List[str] = []
    for raw in raw_paths:
        path = Path(raw)
        if not path.is_absolute():
            path = root / path
        if not path.exists():
            missing.append(str(path))
            summaries.append({"path": str(path), "ok": False, "missing": True})
            continue
        summaries.append(_summarize_path(path))
    return summaries, missing


def _collect_blockers(extended: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    blockers: List[Dict[str, Any]] = []
    for item in extended:
        category = str(item.get("category", ""))
        if category == "te_install_probe":
            resolved = str(item.get("resolved", ""))
            if resolved.startswith("blocked"):
                blockers.append(
                    {
                        "type": "te_fp8_training",
                        "source": str(item.get("path", "")),
                        "resolved": resolved,
                        "reason": str(item.get("reason", "")),
                    }
                )
        if category == "high_dependency_probe":
            if str(item.get("fp8_resolved", "")) != "fp8_te":
                reason = str(item.get("fp8_fallback_reason", "")).strip()
                blockers.append(
                    {
                        "type": "te_fp8_runtime",
                        "source": str(item.get("path", "")),
                        "resolved": str(item.get("fp8_resolved", "")),
                        "reason": reason or "fp8_te_not_resolved",
                    }
                )
    # 去重，避免同一来源重复写入
    dedup: Dict[tuple[str, str], Dict[str, Any]] = {}
    for blocker in blockers:
        key = (str(blocker.get("type", "")), str(blocker.get("source", "")))
        dedup[key] = blocker
    return list(dedup.values())


def _vram_probe() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "platform": platform.platform(),
        "cuda_available": False,
        "classification": "cpu_only_or_cuda_unavailable",
        "notes": [],
    }
    try:
        import torch

        report["torch_version"] = str(torch.__version__)
        report["cuda_available"] = bool(torch.cuda.is_available())
        if not torch.cuda.is_available():
            report["notes"].append("CUDA is unavailable; low-VRAM behavior must be tested on target hardware.")
            return report
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
        total_gb = total_bytes / (1024 ** 3)
        free_gb = free_bytes / (1024 ** 3)
        if total_gb <= 8:
            classification = "low_vram_target"
        elif total_gb <= 12:
            classification = "mid_vram_pressure_relevant"
        else:
            classification = "not_low_vram_target"
        report.update(
            {
                "device_index": idx,
                "device_name": props.name,
                "total_vram_gb": round(total_gb, 3),
                "free_vram_gb": round(free_gb, 3),
                "compute_capability": f"{props.major}.{props.minor}",
                "classification": classification,
            }
        )
        if classification == "not_low_vram_target":
            report["notes"].append("Current GPU is not a low-VRAM proxy; treat pressure results as guardrail only.")
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="temp/performance_validation_report.json")
    parser.add_argument(
        "--reports",
        nargs="*",
        default=[
            "temp/real_strategy_selective_anima_ab_40step.json",
            "temp/real_strategy_selective_newbie_ab_40step_reuse_cache.json",
            "temp/real_strategy_galore_anima_ab_40step.json",
            "temp/real_newbie_cache_profile_reuse_smoke_v2.json",
            "temp/galore_svd_ab_probe_cuda_40step.json",
        ],
        help="Core roadmap reports (items 1-4).",
    )
    parser.add_argument(
        "--extended-reports",
        nargs="*",
        default=[
            "temp/real_quality_long_anima_80step.json",
            "temp/real_quality_long_newbie_80step.json",
            "temp/real_quality_long_sdxl_80step.json",
            "temp/high_dependency_performance_probe_fp8_final.json",
            "temp/transformer_engine_install_probe_windows.json",
        ],
        help="Optional extended reports (5.x and >40-step).",
    )
    args = parser.parse_args()

    root = _repo_root()
    summaries, missing_reports = _collect_summaries(root, args.reports)
    extended_summaries, missing_extended = _collect_summaries(root, args.extended_reports)
    blockers = _collect_blockers(extended_summaries)

    core_ok = (not missing_reports) and all(bool(item.get("ok", False)) for item in summaries)
    extended_complete = not missing_extended
    te_fp8_ready = not any(str(item.get("type", "")) == "te_fp8_training" for item in blockers)

    output = {
        "probe": "performance_validation_report",
        "scope": "roadmap_items_1_to_5_with_longtrain",
        "ok": core_ok,
        "complete": not missing_reports,
        "core_ok": core_ok,
        "extended_complete": extended_complete,
        "missing_reports": missing_reports,
        "missing_extended_reports": missing_extended,
        "vram_probe": _vram_probe(),
        "reports": summaries,
        "extended_reports": extended_summaries,
        "known_blockers": blockers,
        "te_fp8_training_ready": te_fp8_ready,
        "notes": [
            "Training is not launched by this helper; it aggregates bounded smoke/A-B reports.",
            "Low-VRAM classification is hardware-bound and should be repeated on target low-end GPUs.",
            "If complete=false, the report is useful as a partial progress snapshot but should not be treated as full validation.",
            "ok/core_ok only tracks roadmap 1-4 core closure; extended reports carry long-train/dependency status.",
        ],
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0 if output["core_ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())



