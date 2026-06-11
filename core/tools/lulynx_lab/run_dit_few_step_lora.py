"""Experimental Anima/Newbie few-step acceleration LoRA contract runner.

This runner is intentionally conservative.  It validates the launcher contract
and writes a metadata-only safetensors artifact that Resource Center can
recognize as an acceleration LoRA.  Real family-specific DiT/flow distillation
objectives should replace this dry-run body after separate quality validation.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_SUPPORTED_FAMILIES = {"anima", "newbie"}


def _load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config root must be an object")
    return data


def _write_safetensors_metadata_stub(path: Path, metadata: dict[str, Any]) -> None:
    header = {"__metadata__": {str(k): str(v) for k, v in metadata.items()}}
    payload = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(len(payload).to_bytes(8, "little"))
        fh.write(payload)


def _int_value(config: dict[str, Any], key: str, default: int, *, minimum: int = 1) -> int:
    raw = config.get(key, default)
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return value


def _float_value(config: dict[str, Any], key: str, default: float, *, minimum: float | None = None) -> float:
    raw = config.get(key, default)
    try:
        value = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key} must be a number") from exc
    if minimum is not None and value < minimum:
        raise ValueError(f"{key} must be >= {minimum}")
    return value


def _string_value(config: dict[str, Any], key: str, default: str = "") -> str:
    return str(config.get(key) or default).strip()


def _metadata_for_contract(config: dict[str, Any], output_path: Path) -> dict[str, Any]:
    family = _string_value(config, "model_family", "anima").lower()
    if family not in _SUPPORTED_FAMILIES:
        raise ValueError(f"unsupported model_family: {family}")
    schema_id = _string_value(config, "schema_id", f"{family}-few-step-lora")
    student_steps = _int_value(config, "student_steps", 4)
    teacher_steps = _int_value(config, "teacher_steps", 28)
    network_dim = _int_value(config, "network_dim", 16)
    network_alpha = _int_value(config, "network_alpha", network_dim)
    seed = _int_value(config, "seed", 42, minimum=0)
    guidance_scale = _float_value(config, "guidance_scale", 1.0, minimum=0.0)
    distill_method = _string_value(config, "distill_method", "family_flow_consistency")
    objective = _string_value(config, "few_step_objective", "contract_probe")
    schedule = _string_value(config, "sigma_schedule", "family_default")
    now = datetime.now(timezone.utc).isoformat()
    return {
        "modelspec.title": output_path.stem,
        "model_type": "lora",
        "model_family": family,
        "artifact_kind": "acceleration_lora",
        "ss_network_module": _string_value(config, "network_module", "networks.lora"),
        "ss_base_model_version": family,
        "lulynx.schema_id": schema_id,
        "lulynx.artifact_kind": "acceleration_lora",
        "lulynx.model_family": family,
        "lulynx.dit_family": family,
        "lulynx.distill_method": distill_method,
        "lulynx.few_step_objective": objective,
        "lulynx.sigma_schedule": schedule,
        "lulynx.teacher_steps": teacher_steps,
        "lulynx.student_steps": student_steps,
        "lulynx.guidance_scale": guidance_scale,
        "lulynx.seed": seed,
        "lulynx.network_dim": network_dim,
        "lulynx.network_alpha": network_alpha,
        "lulynx.adapter_type": _string_value(config, "adapter_type", "lora"),
        "lulynx.teacher_adapter_path": _string_value(config, "teacher_adapter_path"),
        "lulynx.base_model_path": _string_value(config, "base_model_path"),
        "lulynx.transformer_path": _string_value(config, "transformer_path"),
        "lulynx.validation_level": "contract_dry_run",
        "lulynx.contract_status": "passed",
        "lulynx.quality_status": "not_quality_validated",
        "lulynx.quality_note": "contract only; real family-specific few-step quality validation is deferred",
        "lulynx.recommended_usage": f"{family} few-step scheduler, {student_steps} steps",
        "lulynx.contract_version": "1",
        "lulynx.checked_at": now,
        "lulynx.dry_run": True,
    }


def run_contract(config: dict[str, Any]) -> dict[str, Any]:
    output_path = Path(_string_value(config, "output_path", "./output/dit_few_step_lora/contract.safetensors")).resolve()
    if output_path.suffix.lower() != ".safetensors":
        raise ValueError("output_path must end with .safetensors")
    if not bool(config.get("dry_run", True)):
        raise RuntimeError(
            "Real Anima/Newbie few-step acceleration LoRA training is not enabled in this contract runner yet."
        )

    metadata = _metadata_for_contract(config, output_path)
    _write_safetensors_metadata_stub(output_path, metadata)
    sidecar_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    sidecar = {
        **metadata,
        "status": "success",
        "mode": "contract_dry_run",
        "diagnostics": {
            "family": metadata["lulynx.dit_family"],
            "schema_id": metadata["lulynx.schema_id"],
            "contract_status": "passed",
            "quality_status": "not_quality_validated",
            "note": "This validates launcher wiring and metadata only, not generated image quality.",
        },
    }
    sidecar_path.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "status": "success",
        "dry_run": True,
        "output_path": str(output_path),
        "metadata_sidecar": str(sidecar_path),
        "metadata_keys": sorted(metadata.keys()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Lulynx DiT few-step acceleration LoRA contract runner")
    parser.add_argument("--config", required=True, help="Path to launcher-generated runner config JSON")
    args = parser.parse_args()
    result = run_contract(_load_config(Path(args.config).resolve()))
    print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
