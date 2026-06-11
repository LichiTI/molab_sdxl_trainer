"""Validate experimental SDXL Turbo/LCM LoRA smoke outputs.

This is deliberately narrower than quality evaluation.  It checks that the
saved LoRA tensors are finite and that the sidecar contains usable step
diagnostics with valid latent/prediction/target shapes and distributions.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from .run_turbo_lora import _loss_diagnostics, _summarize_safetensors_file
except ImportError:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_turbo_lora import _loss_diagnostics, _summarize_safetensors_file


_STEP_TENSOR_KEYS = ("prediction", "target", "latent", "noisy_latent")
_EXPERIMENTAL_QUALITY_GATE_LEVEL = "smoke_diagnostics"
_EXPERIMENTAL_QUALITY_GATE_NOTE = (
    "Experimental diagnostic gate only; this is not final image-quality validation."
)


def _load_sidecar(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"sidecar root must be an object: {path}")
    return data


def _sidecar_path(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".metadata.json")


def _finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _validate_tensor_stats(stats: Any) -> list[str]:
    issues: list[str] = []
    if not isinstance(stats, dict):
        return ["missing stats"]

    shape = stats.get("shape")
    if not isinstance(shape, list) or len(shape) != 4:
        issues.append(f"invalid shape {shape!r}")
    else:
        dims = []
        for raw in shape:
            try:
                dims.append(int(raw))
            except (TypeError, ValueError):
                dims.append(0)
        if any(dim <= 0 for dim in dims):
            issues.append(f"non-positive shape {shape!r}")
        if len(dims) == 4 and dims[1] != 4:
            issues.append(f"expected latent channel 4, got {dims[1]}")
        if len(dims) == 4 and (dims[2] % 8 != 0 or dims[3] % 8 != 0):
            issues.append(f"latent spatial dims should align to 8, got {shape!r}")

    finite_ratio = stats.get("finite_ratio")
    if not _finite_number(finite_ratio) or float(finite_ratio) < 1.0:
        issues.append(f"finite_ratio={finite_ratio!r}")

    total = stats.get("total")
    try:
        if int(total) <= 0:
            issues.append(f"total={total!r}")
    except (TypeError, ValueError):
        issues.append(f"total={total!r}")

    for key in ("mean", "std", "min", "max", "abs_max"):
        if key in stats and not _finite_number(stats.get(key)):
            issues.append(f"{key}={stats.get(key)!r}")

    abs_max = stats.get("abs_max")
    if _finite_number(abs_max) and float(abs_max) > 100.0:
        issues.append(f"abs_max unusually high: {abs_max!r}")

    return issues


def _validate_steps(sidecar: dict[str, Any]) -> dict[str, Any]:
    diagnostics = sidecar.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return {
            "status": "missing",
            "valid": False,
            "count": 0,
            "issues": ["diagnostics object missing"],
        }
    steps = diagnostics.get("steps")
    if not isinstance(steps, list) or not steps:
        return {
            "status": "missing",
            "valid": False,
            "count": 0,
            "issues": ["diagnostic steps missing"],
        }

    issues: list[str] = []
    valid_count = 0
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            issues.append(f"step {index}: not an object")
            continue
        loss = step.get("loss")
        if not _finite_number(loss):
            issues.append(f"step {index}: invalid loss {loss!r}")
        step_ok = True
        shapes: dict[str, Any] = {}
        for key in _STEP_TENSOR_KEYS:
            tensor_issues = _validate_tensor_stats(step.get(key))
            if tensor_issues:
                step_ok = False
                issues.extend(f"step {index} {key}: {issue}" for issue in tensor_issues)
            else:
                shapes[key] = step[key].get("shape")
        if shapes:
            reference = shapes.get("prediction")
            for key, shape in shapes.items():
                if reference is not None and shape != reference:
                    step_ok = False
                    issues.append(f"step {index}: {key} shape {shape!r} != prediction {reference!r}")
        if step_ok:
            valid_count += 1

    status = "passed" if valid_count == len(steps) and not issues else "warning"
    return {
        "status": status,
        "valid": status == "passed",
        "count": len(steps),
        "valid_count": valid_count,
        "issues": issues[:32],
    }


def _experimental_quality_gate(
    sidecar: dict[str, Any],
    output_summary: dict[str, Any],
    loss_summary: dict[str, Any],
    step_validation: dict[str, Any],
    *,
    output_valid: bool,
    metadata_ok: bool,
) -> dict[str, Any]:
    """Promote smoke diagnostics into an explicit next-step readiness gate.

    The gate intentionally avoids judging final sample quality.  It only says
    whether a real short smoke has enough finite tensors, step diagnostics, and
    metadata to be worth using in the next evaluation stage.
    """

    diagnostics = sidecar.get("diagnostics")
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    teacher_lora = diagnostics.get("teacher_lora")
    if not isinstance(teacher_lora, dict):
        teacher_lora = {}

    fatal: list[str] = []
    review: list[str] = []
    observations: list[str] = []

    if not output_valid:
        fatal.append("output tensors are missing, empty, or non-finite")
    if step_validation.get("status") != "passed":
        fatal.append("sidecar step diagnostics did not pass")
    if loss_summary.get("status") != "passed" or int(loss_summary.get("finite_count") or 0) <= 0:
        fatal.append("finite loss diagnostics did not pass")

    smoke_status = str(sidecar.get("lulynx.smoke_status") or diagnostics.get("status") or "")
    if smoke_status != "passed":
        review.append(f"smoke_status={smoke_status or 'missing'}")
    validation_level = str(sidecar.get("lulynx.validation_level") or diagnostics.get("validation_level") or "")
    if validation_level != "real_short_smoke":
        review.append(f"validation_level={validation_level or 'missing'}")
    if diagnostics.get("mode") != "real_short_smoke":
        review.append(f"diagnostics.mode={diagnostics.get('mode') or 'missing'}")
    if not metadata_ok:
        review.append("lulynx Turbo/LCM metadata incomplete")

    step_count = int(step_validation.get("count") or 0)
    if step_count < 2:
        review.append(f"step_count={step_count}; at least 2 steps are recommended for the gate")

    if sidecar.get("lulynx.teacher_lora_path") and not diagnostics.get("teacher_lora_loaded"):
        review.append("teacher LoRA path exists but teacher adapter was not loaded")
    teacher_warnings = teacher_lora.get("warnings")
    if isinstance(teacher_warnings, list) and teacher_warnings:
        review.extend(f"teacher warning: {warning}" for warning in teacher_warnings[:4])

    finite_ratio = output_summary.get("finite_ratio")
    tensor_count = output_summary.get("tensor_count")
    loss_trend = loss_summary.get("trend")
    if loss_trend:
        observations.append(f"loss_trend={loss_trend}")
    if finite_ratio is not None:
        observations.append(f"output_finite_ratio={finite_ratio}")
    if tensor_count is not None:
        observations.append(f"output_tensor_count={tensor_count}")

    if fatal:
        status = "failed"
    elif review:
        status = "needs_review"
    else:
        status = "experimental_pass"

    return {
        "status": status,
        "level": _EXPERIMENTAL_QUALITY_GATE_LEVEL,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fatal": fatal[:16],
        "review": review[:16],
        "observations": observations[:16],
        "step_count": step_count,
        "note": _EXPERIMENTAL_QUALITY_GATE_NOTE,
    }


def validate_turbo_lora_output(output_path: Path, *, write_sidecar: bool = False) -> dict[str, Any]:
    output_path = output_path.resolve()
    sidecar_path = _sidecar_path(output_path)
    sidecar = _load_sidecar(sidecar_path)
    output_summary = _summarize_safetensors_file(output_path)

    issues: list[str] = []
    if output_summary.get("summary_error"):
        issues.append(str(output_summary["summary_error"]))

    finite_ratio = output_summary.get("finite_ratio")
    output_valid = (
        output_path.is_file()
        and not output_summary.get("summary_error")
        and int(output_summary.get("tensor_count") or 0) > 0
        and int(output_summary.get("bad_count") or 0) == 0
        and _finite_number(finite_ratio)
        and float(finite_ratio) == 1.0
    )
    if not output_valid:
        issues.append("output tensors are missing, empty, or non-finite")

    losses = sidecar.get("losses")
    if not isinstance(losses, list):
        diagnostics = sidecar.get("diagnostics")
        if isinstance(diagnostics, dict):
            loss_summary = diagnostics.get("loss")
            losses = []
            if isinstance(loss_summary, dict):
                for key in ("initial", "final"):
                    if _finite_number(loss_summary.get(key)):
                        losses.append(float(loss_summary[key]))
    loss_summary = _loss_diagnostics([float(loss) for loss in losses if _finite_number(loss)]) if isinstance(losses, list) else {
        "status": "failed",
        "count": 0,
        "finite_count": 0,
        "finite_ratio": 0.0,
    }
    if loss_summary.get("status") == "failed" or int(loss_summary.get("count") or 0) <= 0:
        issues.append("finite loss diagnostics missing")

    step_validation = _validate_steps(sidecar)
    if step_validation["status"] == "missing":
        issues.append("sidecar step diagnostics missing")
    elif step_validation["status"] != "passed":
        issues.append("sidecar step diagnostics have warnings")

    metadata_ok = (
        sidecar.get("lulynx.schema_id") == "sdxl-turbo-lora"
        and sidecar.get("lulynx.artifact_kind") == "acceleration_lora"
    )
    if sidecar and not metadata_ok:
        issues.append("lulynx Turbo/LCM metadata is incomplete")

    if not output_valid:
        status = "failed"
    elif step_validation["status"] == "passed" and loss_summary.get("status") == "passed" and metadata_ok:
        status = "passed"
    else:
        status = "warning"

    quality_gate = _experimental_quality_gate(
        sidecar,
        output_summary,
        loss_summary,
        step_validation,
        output_valid=output_valid,
        metadata_ok=metadata_ok,
    )

    validation = {
        "status": status,
        "checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "output": output_summary,
        "loss": loss_summary,
        "sidecar_steps": step_validation,
        "experimental_quality_gate": quality_gate,
        "metadata_ok": metadata_ok,
        "issues": issues[:32],
        "note": "Finite tensor and sidecar diagnostics check only; quality is still not validated.",
    }

    if write_sidecar:
        updated = dict(sidecar)
        updated.update({
            "lulynx.output_validation_status": status,
            "lulynx.output_validation_level": "tensor_sidecar",
            "lulynx.sidecar_steps_valid": bool(step_validation["valid"]),
            "lulynx.sidecar_step_count": int(step_validation.get("count") or 0),
            "lulynx.sidecar_distribution_status": step_validation["status"],
            "lulynx.output_validation_checked_at": validation["checked_at"],
            "lulynx.output_validation_note": validation["note"],
            "lulynx.experimental_quality_gate_status": quality_gate["status"],
            "lulynx.experimental_quality_gate_level": quality_gate["level"],
            "lulynx.experimental_quality_gate_checked_at": quality_gate["checked_at"],
            "lulynx.experimental_quality_gate_note": quality_gate["note"],
            "lulynx.experimental_quality_gate_review_count": len(quality_gate["review"]),
            "lulynx.output_tensor_count": output_summary.get("tensor_count", updated.get("lulynx.output_tensor_count", "")),
            "lulynx.output_finite_ratio": output_summary.get("finite_ratio", updated.get("lulynx.output_finite_ratio", "")),
            "output_validation": validation,
            "experimental_quality_gate": quality_gate,
        })
        sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        sidecar_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "status": status,
        "output_path": str(output_path),
        "metadata_sidecar": str(sidecar_path),
        "validation": validation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate a Lulynx SDXL Turbo/LCM LoRA smoke output")
    parser.add_argument("output", help="Path to the output .safetensors file")
    parser.add_argument("--write-sidecar", action="store_true", help="Write validation fields back to the metadata sidecar")
    args = parser.parse_args()

    result = validate_turbo_lora_output(Path(args.output), write_sidecar=args.write_sidecar)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["status"] in {"passed", "warning"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
