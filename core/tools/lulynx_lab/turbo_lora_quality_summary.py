"""Summarize Turbo/LCM LoRA validation and sample-report readiness."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .report_turbo_lora_samples import report_turbo_lora_samples
    from .validate_turbo_lora_output import validate_turbo_lora_output
except ImportError:  # pragma: no cover - direct script execution
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from report_turbo_lora_samples import report_turbo_lora_samples
    from validate_turbo_lora_output import validate_turbo_lora_output


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def summarize_turbo_lora_quality(
    output_path: Path,
    *,
    samples_dir: Path | None = None,
    write_sidecar: bool = False,
) -> dict[str, Any]:
    """Build a conservative quality-readiness summary from existing reports."""

    validation_result = validate_turbo_lora_output(output_path, write_sidecar=write_sidecar)
    sample_result = report_turbo_lora_samples(output_path, samples_dir=samples_dir, write_sidecar=write_sidecar)
    validation = dict(validation_result.get("validation") or {})
    sample_evaluation = dict(sample_result.get("sample_evaluation") or {})
    gate = dict(validation.get("experimental_quality_gate") or {})

    validation_status = str(validation_result.get("status") or "unknown")
    sample_status = str(sample_result.get("status") or "unknown")
    sample_count = int(sample_evaluation.get("sample_count") or 0)
    review_items = list(gate.get("review") or [])
    fatal_items = list(gate.get("fatal") or [])

    if validation_status == "failed" or fatal_items:
        readiness = "blocked"
    elif validation_status not in {"passed", "warning"}:
        readiness = "blocked"
    elif sample_status == "basic_report" and not review_items:
        readiness = "ready_for_human_review"
    elif sample_status == "basic_report":
        readiness = "needs_review"
    elif sample_status == "no_samples":
        readiness = "needs_samples"
    else:
        readiness = "needs_review"

    summary = {
        "status": readiness,
        "checked_at": _utc_now(),
        "output_path": str(Path(output_path).resolve()),
        "metadata_sidecar": str(validation_result.get("metadata_sidecar") or sample_result.get("metadata_sidecar") or ""),
        "validation_status": validation_status,
        "sample_status": sample_status,
        "sample_count": sample_count,
        "quality_boundary": "not_final_quality_validation",
        "requires_human_review": readiness in {"ready_for_human_review", "needs_review"},
        "requires_more_samples": readiness == "needs_samples",
        "fatal": fatal_items[:16],
        "review": review_items[:16],
        "issues": list(validation.get("issues") or [])[:16] + list(sample_evaluation.get("issues") or [])[:16],
        "validation": validation,
        "sample_evaluation": sample_evaluation,
        "note": "Readiness summary only; final visual quality validation still requires human or model-based review.",
    }

    if write_sidecar:
        _write_summary_sidecar(Path(output_path), summary)
    return summary


def _write_summary_sidecar(output_path: Path, summary: dict[str, Any]) -> None:
    sidecar_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8")) if sidecar_path.is_file() else {}
    except Exception:
        sidecar = {}
    if not isinstance(sidecar, dict):
        sidecar = {}
    sidecar.update(
        {
            "lulynx.quality_readiness_status": summary["status"],
            "lulynx.quality_readiness_checked_at": summary["checked_at"],
            "lulynx.quality_readiness_requires_human_review": bool(summary["requires_human_review"]),
            "lulynx.quality_readiness_requires_more_samples": bool(summary["requires_more_samples"]),
            "lulynx.quality_boundary": summary["quality_boundary"],
            "lulynx.quality_readiness_note": summary["note"],
            "quality_readiness_summary": summary,
        }
    )
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize Turbo/LCM LoRA validation and sample readiness")
    parser.add_argument("output", help="Path to the output .safetensors file")
    parser.add_argument("--samples-dir", default="", help="Directory containing generated sample images")
    parser.add_argument("--write-sidecar", action="store_true", help="Write quality readiness fields back to metadata sidecar")
    args = parser.parse_args()

    samples_dir = Path(args.samples_dir) if str(args.samples_dir).strip() else None
    result = summarize_turbo_lora_quality(Path(args.output), samples_dir=samples_dir, write_sidecar=args.write_sidecar)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] != "blocked" else 1


if __name__ == "__main__":
    raise SystemExit(main())
