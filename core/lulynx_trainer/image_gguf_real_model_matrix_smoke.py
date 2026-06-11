"""Real-weight image GGUF compatibility matrix.

This script scans ``models/`` and runs a bounded shadow experiment:

- read-only probe for every discovered safetensors candidate;
- export-plan estimation for every candidate/group;
- optional small real GGUF exports for low-risk components only;
- Python/native descriptor validation when an export is attempted.

It never marks a model runtime-loadable and never touches training paths.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from core.image_gguf_probe import probe_image_gguf_manifest  # noqa: E402
from core.services.native_module_loader import native_with_entrypoints  # noqa: E402
from core.tools.image_gguf_exporter import export_image_gguf_component, plan_image_gguf_export  # noqa: E402
from core.tools.image_gguf_forward_quality_gate import run_image_gguf_forward_quality_gate  # noqa: E402
from core.tools.image_gguf_payload_parity import check_image_gguf_payload_parity  # noqa: E402
from core.tools.image_gguf_quality_gate import run_image_gguf_lightweight_quality_gate  # noqa: E402
from core.tools.image_gguf_state_dict_loader import summarize_image_gguf_state_dict_load  # noqa: E402
from core.tools.image_gguf_validator import validate_image_gguf_container  # noqa: E402


SHARD_RE = re.compile(r"^(?P<prefix>.+)-(?P<index>\d+)-of-(?P<total>\d+)\.safetensors$", re.IGNORECASE)
LOW_RISK_EXPORT_COMPONENTS = {"vae", "clip"}


def _default_report_path() -> Path:
    return Path(
        os.environ.get(
            "LULYNX_IMAGE_GGUF_REAL_MATRIX_REPORT",
            str(ROOT / ".runs" / "image_gguf_real_matrix" / "real_model_matrix.json"),
        )
    )


def _default_export_dir() -> Path:
    return Path(
        os.environ.get(
            "LULYNX_IMAGE_GGUF_REAL_MATRIX_EXPORT_DIR",
            str(ROOT / ".runs" / "image_gguf_real_matrix" / "exports"),
        )
    )


def _discover_safetensors(models_dir: Path) -> list[Path]:
    if not models_dir.is_dir():
        return []
    files = [path for path in models_dir.rglob("*.safetensors") if path.is_file()]
    return sorted(files, key=lambda item: str(item).lower())


def _build_candidates(files: list[Path], models_dir: Path) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[Path]] = {}
    consumed: set[Path] = set()
    for path in files:
        match = SHARD_RE.match(path.name)
        if not match:
            continue
        key = (str(path.parent).lower(), match.group("prefix").lower(), match.group("total"))
        groups.setdefault(key, []).append(path)
        consumed.add(path)

    candidates: list[dict[str, Any]] = []
    for paths in groups.values():
        ordered = sorted(paths, key=lambda item: item.name.lower())
        candidates.append(_candidate_payload(ordered, models_dir, grouped=True))
    for path in files:
        if path in consumed:
            continue
        candidates.append(_candidate_payload([path], models_dir, grouped=False))
    return sorted(candidates, key=lambda item: str(item["display_name"]).lower())


def _candidate_payload(paths: list[Path], models_dir: Path, *, grouped: bool) -> dict[str, Any]:
    family_hint = _guess_family_hint(paths[0], models_dir)
    if grouped and family_hint == "":
        family_hint = _guess_family_hint(paths[0].parent, models_dir)
    rel_paths = [_relative(path, ROOT) for path in paths]
    return {
        "display_name": _display_name(paths, models_dir),
        "family_hint": family_hint,
        "grouped_shards": grouped,
        "path_count": len(paths),
        "paths": paths,
        "relative_paths": rel_paths,
        "input_size_bytes": sum(path.stat().st_size for path in paths),
    }


def _display_name(paths: list[Path], models_dir: Path) -> str:
    if len(paths) == 1:
        return _relative(paths[0], models_dir)
    return f"{_relative(paths[0].parent, models_dir)}/{paths[0].stem.rsplit('-', 2)[0]} ({len(paths)} shards)"


def _guess_family_hint(path: Path, models_dir: Path) -> str:
    text = _relative(path, models_dir).replace("\\", "/").lower()
    if "/anima/diffusion_models/" in f"/{text}" or text.startswith("anima/diffusion_models/"):
        return "anima"
    if text == "newbie/diffusion_pytorch_model.safetensors":
        return "newbie"
    if text.startswith("sdxl/"):
        return "sdxl"
    if "/vae/" in f"/{text}/":
        return "qwen_image_vae" if "qwen_image_vae" in text else "vae"
    if "clip_model/" in text:
        return "clip"
    if "text_encoder_2/" in text:
        return "t5"
    if "flux.1-schnell/text_encoder/" in text:
        return "clip"
    if "transformer/" in text and "flux" in text:
        return "flux_transformer"
    if "text_encoders/" in text and "anima" in text:
        return "qwen_text_encoder"
    if "text_encoder/" in text and "newbie" in text:
        return "gemma_text_encoder"
    if text.startswith("lora/"):
        return "lora"
    return ""


def _probe_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    manifests = []
    errors = []
    for path in candidate["paths"]:
        try:
            manifest = probe_image_gguf_manifest(path, family_hint=candidate["family_hint"]).to_dict()
        except Exception as exc:  # pragma: no cover - real file diagnostics
            errors.append(f"{type(exc).__name__}: {exc}")
            continue
        manifests.append(_manifest_summary(manifest))
    components = sorted({str(item["component"]) for item in manifests})
    families = sorted({str(item["family"]) for item in manifests})
    return {
        "ok": bool(manifests) and not errors,
        "manifest_count": len(manifests),
        "components": components,
        "families": families,
        "all_manifests_ok": all(bool(item["ok"]) for item in manifests) if manifests else False,
        "any_manifest_ok": any(bool(item["ok"]) for item in manifests),
        "total_tensor_count": sum(int(item["tensor_count"]) for item in manifests),
        "total_matched_tensors": sum(int(item["matched_tensors"]) for item in manifests),
        "missing_required_tensor_count": sum(len(item["missing_required_tensors"]) for item in manifests),
        "missing_required_prefix_count": sum(len(item["missing_required_prefixes"]) for item in manifests),
        "manifests": manifests,
        "errors": errors,
    }


def _manifest_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    shape_summary = manifest.get("shape_summary") if isinstance(manifest.get("shape_summary"), dict) else {}
    return {
        "source_path": _relative(Path(str(manifest.get("source_path") or "")), ROOT),
        "adapter_id": manifest.get("adapter_id"),
        "component": manifest.get("component"),
        "family": manifest.get("family"),
        "compatibility": manifest.get("compatibility"),
        "ok": bool(manifest.get("ok")),
        "tensor_count": int(manifest.get("tensor_count") or 0),
        "matched_tensors": int(manifest.get("matched_tensors") or 0),
        "missing_required_tensors": list(manifest.get("missing_required_tensors") or []),
        "missing_required_prefixes": list(manifest.get("missing_required_prefixes") or []),
        "unexpected_tensors_sample": list(manifest.get("unexpected_tensors_sample") or [])[:12],
        "dtype_counts": dict(manifest.get("dtype_counts") or {}),
        "rank_counts": dict(manifest.get("rank_counts") or {}),
        "total_numel": int(shape_summary.get("total_numel") or 0),
        "warnings": list(manifest.get("warnings") or [])[:4],
        "notes": list(manifest.get("notes") or [])[:4],
    }


def _plan_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    try:
        plan = plan_image_gguf_export(candidate["paths"], family_hint=candidate["family_hint"], file_type="f16").to_dict()
    except Exception as exc:  # pragma: no cover - real file diagnostics
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "ok": bool(plan["ok"]),
        "component": plan["component"],
        "family": plan["family"],
        "compatibility": plan["compatibility"],
        "unique_tensor_count": plan["unique_tensor_count"],
        "duplicate_tensor_count": plan["duplicate_tensor_count"],
        "converted_tensors": plan["converted_tensors"],
        "skipped_tensors": plan["skipped_tensors"],
        "estimated_output_size_bytes": plan["estimated_output_size_bytes"],
        "gguf_file_type": plan["gguf_file_type"],
        "dtype_counts": plan["dtype_counts"],
        "rank_counts": plan["rank_counts"],
        "warnings": list(plan["warnings"] or [])[:8],
        "errors": list(plan["errors"] or [])[:8],
    }


def _should_export(plan: dict[str, Any], export_count: int, args: argparse.Namespace) -> tuple[bool, str]:
    if args.no_export:
        return False, "real export disabled by --no-export"
    if export_count >= args.max_export_targets:
        return False, "real export target limit reached"
    if not plan.get("ok"):
        return False, "export plan is not ok"
    component = str(plan.get("component") or "")
    if component not in LOW_RISK_EXPORT_COMPONENTS:
        return False, f"component {component or '<missing>'} is not in low-risk export allowlist"
    estimated = int(plan.get("estimated_output_size_bytes") or 0)
    if estimated > args.max_export_mb * 1024 * 1024:
        return False, f"estimated output exceeds limit: {estimated} bytes"
    return True, "selected for bounded real export"


def _export_and_validate(candidate: dict[str, Any], export_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    export_dir.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(candidate["display_name"]))[:120].strip("._") or "image_gguf_model"
    output_path = export_dir / f"{safe_name}.gguf"
    try:
        result = export_image_gguf_component(
            candidate["paths"],
            output_path,
            family_hint=candidate["family_hint"],
            name=safe_name,
            file_type="f16",
            overwrite=True,
        )
        validation = validate_image_gguf_container(result.output_path)
        parity = check_image_gguf_payload_parity(
            candidate["paths"],
            result.output_path,
            sidecar_path=result.sidecar_path,
            max_tensors=8,
            max_elements_per_tensor=4096,
        )
        state_dict_load = summarize_image_gguf_state_dict_load(result.output_path, sidecar_path=result.sidecar_path, max_tensors=8)
        quality_gate = run_image_gguf_lightweight_quality_gate(
            candidate["paths"],
            result.output_path,
            sidecar_path=result.sidecar_path,
        )
        forward_quality_gate = run_image_gguf_forward_quality_gate(
            candidate["paths"],
            result.output_path,
            sidecar_path=result.sidecar_path,
            component=result.component,
            family=result.family,
            enable_reference_forward=args.enable_forward_quality,
            allow_trust_remote_code=args.allow_forward_trust_remote_code,
        )
        native = _native_descriptor(result.output_path)
    except Exception as exc:  # pragma: no cover - real file diagnostics
        return {"attempted": True, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "attempted": True,
        "ok": bool(result.ok) and bool(validation.get("container_contract", {}).get("ok")),
        "output_path": result.output_path,
        "sidecar_path": result.sidecar_path,
        "output_size_bytes": result.output_size_bytes,
        "component": result.component,
        "family": result.family,
        "tensor_count": result.tensor_count,
        "converted_tensors": result.converted_tensors,
        "validation_ok": bool(validation.get("ok")),
        "shape_contract_ok": bool(validation.get("shape_contract", {}).get("ok")),
        "runtime_loadable": bool(validation.get("runtime_loadable")),
        "runtime_loader_abi": validation.get("runtime_contract", {}).get("runtime_loader_abi", {}),
        "payload_parity": _payload_parity_summary(parity),
        "state_dict_loader": _state_dict_loader_summary(state_dict_load),
        "quality_gate": _quality_gate_summary(quality_gate),
        "forward_quality_gate": _forward_quality_gate_summary(forward_quality_gate),
        "runtime_blockers": list(validation.get("runtime_blockers") or [])[:8],
        "validation_issues": list(validation.get("issues") or [])[:8],
        "native_descriptor": native,
    }


def _forward_quality_gate_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok")),
        "gate": report.get("gate"),
        "quality_stage": report.get("quality_stage"),
        "status": report.get("status"),
        "status_reason": report.get("status_reason"),
        "reference_forward_enabled": bool(report.get("reference_forward_enabled")),
        "allow_trust_remote_code": bool(report.get("allow_trust_remote_code")),
        "required_dependencies": list(report.get("required_dependencies") or []),
        "missing_dependencies": list(report.get("missing_dependencies") or []),
        "state_dict_loader_ok": bool(report.get("state_dict_loader_ok")),
        "state_dict_tensor_count": int(report.get("state_dict_tensor_count") or 0),
        "shape_contract_ok": bool(report.get("shape_contract_ok")),
        "config_probe": dict(report.get("config_probe") or {}),
        "forward_plan": dict(report.get("forward_plan") or {}),
        "reference_forward_result": dict(report.get("reference_forward_result") or {}),
        "reads_tensor_payloads": bool(report.get("reads_tensor_payloads")),
        "builds_model_modules": bool(report.get("builds_model_modules")),
        "runs_forward_pass": bool(report.get("runs_forward_pass")),
        "runtime_loadable_enabled": bool(report.get("runtime_loadable_enabled")),
    }


def _quality_gate_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok")),
        "gate": report.get("gate"),
        "quality_stage": report.get("quality_stage"),
        "state_dict_tensor_count": int(report.get("state_dict_tensor_count") or 0),
        "dtype_counts": dict(report.get("dtype_counts") or {}),
        "payload_parity_ok": bool(report.get("payload_parity_ok")),
        "payload_parity_max_abs_error": float(report.get("payload_parity_max_abs_error") or 0.0),
        "check_count": int(report.get("check_count") or 0),
        "failed_check_count": int(report.get("failed_check_count") or 0),
        "failed_checks": list(report.get("failed_checks") or [])[:8],
        "reads_tensor_payloads": bool(report.get("reads_tensor_payloads")),
        "builds_model_modules": bool(report.get("builds_model_modules")),
        "runs_forward_pass": bool(report.get("runs_forward_pass")),
        "runtime_loadable_enabled": bool(report.get("runtime_loadable_enabled")),
    }


def _state_dict_loader_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok")),
        "loader": report.get("loader"),
        "state_dict_tensor_count": int(report.get("state_dict_tensor_count") or 0),
        "gguf_tensor_count": int(report.get("gguf_tensor_count") or 0),
        "truncated": bool(report.get("truncated")),
        "dtype_counts": dict(report.get("dtype_counts") or {}),
        "memory_estimate_bytes": int(report.get("memory_estimate_bytes") or 0),
        "records": list(report.get("records") or [])[:8],
        "reads_tensor_payloads": bool(report.get("reads_tensor_payloads")),
        "builds_model_modules": bool(report.get("builds_model_modules")),
        "runs_forward_pass": bool(report.get("runs_forward_pass")),
        "runtime_loadable_enabled": bool(report.get("runtime_loadable_enabled")),
    }


def _payload_parity_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok")),
        "checker": report.get("checker"),
        "sampled_tensor_count": int(report.get("sampled_tensor_count") or 0),
        "max_abs_error": float(report.get("max_abs_error") or 0.0),
        "max_rel_error": float(report.get("max_rel_error") or 0.0),
        "failed_tensor_count": int(report.get("failed_tensor_count") or 0),
        "failed_tensors": list(report.get("failed_tensors") or [])[:8],
        "records": list(report.get("records") or [])[:8],
        "reads_tensor_payloads": bool(report.get("reads_tensor_payloads")),
        "runs_forward_pass": bool(report.get("runs_forward_pass")),
        "runtime_loadable_enabled": bool(report.get("runtime_loadable_enabled")),
    }


def _native_descriptor(path: str) -> dict[str, Any]:
    native = native_with_entrypoints("read_image_gguf_descriptor")
    if native is None:
        return {"available": False, "ok": False, "reason": "lulynx_native descriptor entrypoint is unavailable"}
    try:
        report = native.read_image_gguf_descriptor(path, 64, 64)
    except Exception as exc:  # pragma: no cover - native diagnostics
        return {"available": True, "ok": False, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "available": True,
        "ok": bool(report.get("ok")),
        "provider": report.get("provider"),
        "component": report.get("component"),
        "family": report.get("family"),
        "tensor_count": report.get("tensor_count"),
        "container_contract_ok": bool(report.get("container_contract", {}).get("ok")),
        "shape_contract_ok": bool(report.get("shape_contract", {}).get("ok")),
        "runtime_loadable": bool(report.get("runtime_loadable")),
        "runtime_loader_abi": report.get("runtime_contract", {}).get("runtime_loader_abi", {}),
        "runtime_blockers": list(report.get("runtime_blockers") or [])[:8],
        "issues": list(report.get("issues") or [])[:8],
    }


def _run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    models_dir = Path(args.models_dir).resolve()
    files = _discover_safetensors(models_dir)
    candidates = _build_candidates(files, models_dir)
    export_dir = Path(args.export_dir).resolve()
    export_count = 0
    rows = []
    for candidate in candidates:
        probe = _probe_candidate(candidate)
        plan = _plan_candidate(candidate)
        should_export, export_reason = _should_export(plan, export_count, args)
        export = {"attempted": False, "ok": False, "reason": export_reason}
        if should_export:
            export = _export_and_validate(candidate, export_dir, args)
            export_count += 1
        rows.append(
            {
                "display_name": candidate["display_name"],
                "family_hint": candidate["family_hint"],
                "grouped_shards": candidate["grouped_shards"],
                "path_count": candidate["path_count"],
                "relative_paths": candidate["relative_paths"],
                "input_size_bytes": candidate["input_size_bytes"],
                "probe": probe,
                "export_plan": plan,
                "real_export": export,
            }
        )
    return {
        "schema_version": 1,
        "experiment": "image_gguf_real_model_matrix_shadow_v1",
        "models_dir": str(models_dir),
        "report_only": True,
        "runtime_loadable_enabled": False,
        "training_path_enabled": False,
        "scanned_safetensors": len(files),
        "candidate_count": len(candidates),
        "real_export_count": export_count,
        "real_export_dir": str(export_dir),
        "max_export_mb": args.max_export_mb,
        "max_export_targets": args.max_export_targets,
        "enable_forward_quality": bool(args.enable_forward_quality),
        "allow_forward_trust_remote_code": bool(args.allow_forward_trust_remote_code),
        "rows": rows,
        "summary": _summary(rows),
    }


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_component: dict[str, int] = {}
    plan_ok = 0
    export_ok = 0
    shape_ok = 0
    parity_ok = 0
    state_dict_ok = 0
    quality_ok = 0
    forward_passed = 0
    forward_skipped = 0
    unsupported = 0
    for row in rows:
        component = str(row.get("export_plan", {}).get("component") or "<missing>")
        by_component[component] = by_component.get(component, 0) + 1
        if row.get("export_plan", {}).get("ok"):
            plan_ok += 1
        if row.get("real_export", {}).get("ok"):
            export_ok += 1
        if row.get("real_export", {}).get("shape_contract_ok"):
            shape_ok += 1
        if row.get("real_export", {}).get("payload_parity", {}).get("ok"):
            parity_ok += 1
        if row.get("real_export", {}).get("state_dict_loader", {}).get("ok"):
            state_dict_ok += 1
        if row.get("real_export", {}).get("quality_gate", {}).get("ok"):
            quality_ok += 1
        forward_status = str(row.get("real_export", {}).get("forward_quality_gate", {}).get("status") or "")
        if row.get("real_export", {}).get("forward_quality_gate", {}).get("ok"):
            forward_passed += 1
        if forward_status.startswith("skipped_") or forward_status == "not_supported":
            forward_skipped += 1
        if component in {"generic_tensor_bundle", "unknown", "<missing>"}:
            unsupported += 1
    return {
        "by_component": dict(sorted(by_component.items())),
        "export_plan_ok": plan_ok,
        "real_export_ok": export_ok,
        "real_export_shape_contract_ok": shape_ok,
        "real_export_payload_parity_ok": parity_ok,
        "real_export_state_dict_loader_ok": state_dict_ok,
        "real_export_lightweight_quality_gate_ok": quality_ok,
        "real_export_forward_quality_gate_passed": forward_passed,
        "real_export_forward_quality_gate_skipped": forward_skipped,
        "unsupported_or_generic_candidates": unsupported,
    }


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a bounded real-weight image GGUF compatibility matrix.")
    parser.add_argument("--models-dir", default=str(ROOT / "models"))
    parser.add_argument("--report-path", default=str(_default_report_path()))
    parser.add_argument("--export-dir", default=str(_default_export_dir()))
    parser.add_argument("--max-export-mb", type=int, default=int(os.environ.get("LULYNX_IMAGE_GGUF_REAL_MATRIX_MAX_EXPORT_MB", "1024")))
    parser.add_argument("--max-export-targets", type=int, default=int(os.environ.get("LULYNX_IMAGE_GGUF_REAL_MATRIX_MAX_EXPORT_TARGETS", "2")))
    parser.add_argument("--enable-forward-quality", action="store_true", help="Run optional tiny reference forward comparisons when dependencies and adapters are available.")
    parser.add_argument("--allow-forward-trust-remote-code", action="store_true", help="Allow forward-quality probes to use trust_remote_code adapters. Disabled by default.")
    parser.add_argument("--no-export", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    report = _run_matrix(args)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({**report["summary"], "report_path": str(report_path), "candidate_count": report["candidate_count"]}, indent=2, ensure_ascii=False))
    if report["candidate_count"] <= 0:
        print("SKIP: no safetensors candidates found under models directory")
    else:
        print("\nImage GGUF real model matrix shadow experiment completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
