"""Forward-readiness quality gate for image GGUF shadow exports.

This gate is intentionally report-only. It checks whether a GGUF export is
ready for a future model forward comparison, but it does not enable runtime
loading, training dispatch, or image model execution by itself.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Iterable

try:
    from core.tools.image_gguf_state_dict_loader import load_image_gguf_state_dict_with_report, summarize_image_gguf_state_dict_load
except ImportError:
    from backend.core.tools.image_gguf_state_dict_loader import load_image_gguf_state_dict_with_report, summarize_image_gguf_state_dict_load


SUPPORTED_COMPONENT_DEPENDENCIES = {
    "vae": ["diffusers"],
    "clip": ["transformers"],
}

MODEL_CONFIG_FILENAMES = {
    "vae": ["config.json", "vae/config.json"],
    "clip": [
        "config.json",
        "text_encoder/config.json",
        "clip_model/config.json",
        "tokenizer.json",
        "vocab.json",
        "merges.txt",
        "tokenizer/tokenizer.json",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        "clip_model/tokenizer.json",
    ],
}

FORWARD_THRESHOLDS = {
    "vae": {"max_abs_error": 0.25, "max_rel_error": 0.25, "min_cosine_similarity": 0.995},
    "clip": {"max_abs_error": 0.1, "max_rel_error": 0.1, "min_cosine_similarity": 0.999},
}


def run_image_gguf_forward_quality_gate(
    source_paths: str | Path | Iterable[str | Path],
    gguf_path: str | Path,
    *,
    sidecar_path: str | Path | None = None,
    component: str = "",
    family: str = "",
    max_tensors: int = 8,
    enable_reference_forward: bool = False,
    allow_trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Return a machine-readable forward quality gate report.

    The current implementation stops before model construction. If optional
    forward dependencies are missing, it reports a skipped status instead of
    failing the export matrix.
    """

    sources = _normalize_sources(source_paths)
    state_report = _load_state_summary(gguf_path, sidecar_path=sidecar_path, max_tensors=max_tensors)
    resolved_component = component or str(state_report.get("component") or "")
    resolved_family = family or str(state_report.get("family") or "")
    required_dependencies = list(SUPPORTED_COMPONENT_DEPENDENCIES.get(resolved_component, []))
    dependency_status = _dependency_status(required_dependencies)
    missing_dependencies = [name for name, available in dependency_status.items() if not available]
    config_probe = _probe_model_config(sources, resolved_component)
    forward_plan = _build_forward_plan(
        component=resolved_component,
        family=resolved_family,
        config_probe=config_probe,
        allow_trust_remote_code=allow_trust_remote_code,
    )

    status, reason = _status_for_gate(
        component=resolved_component,
        state_report=state_report,
        missing_dependencies=missing_dependencies,
        config_probe=config_probe,
        forward_plan=forward_plan,
        enable_reference_forward=enable_reference_forward,
    )
    reference_forward_result: dict[str, Any] = {}
    if status == "ready_reference_forward":
        reference_forward_result = _run_reference_forward_comparison(
            sources,
            gguf_path,
            sidecar_path=sidecar_path,
            component=resolved_component,
            forward_plan=forward_plan,
        )
        status = "passed" if reference_forward_result.get("ok") else "failed_reference_forward"
        reason = str(reference_forward_result.get("status_reason") or reference_forward_result.get("status") or reason)

    runtime_blockers = [
        "image GGUF forward quality gate is report-only",
        "image GGUF runtime loading remains disabled",
    ]
    if missing_dependencies:
        runtime_blockers.append(f"missing forward dependency: {', '.join(missing_dependencies)}")
    if status == "skipped_missing_model_config":
        runtime_blockers.append("missing colocated model config for forward reference construction")
    if status == "skipped_reference_forward_disabled":
        runtime_blockers.append("reference forward comparison requires enable_reference_forward=true")
    if status == "skipped_requires_trust_remote_code":
        runtime_blockers.append("forward adapter requires explicit allow_trust_remote_code=true")

    built_model_modules = bool(reference_forward_result.get("builds_model_modules"))
    ran_forward_pass = bool(reference_forward_result.get("runs_forward_pass"))

    return {
        "schema_version": 1,
        "gate": "image_gguf_forward_quality_gate_v1",
        "quality_stage": "forward_quality_gate",
        "ok": status == "passed",
        "status": status,
        "status_reason": reason,
        "component": resolved_component,
        "family": resolved_family,
        "source_paths": [str(path) for path in sources],
        "gguf_path": str(gguf_path),
        "sidecar_path": str(sidecar_path or ""),
        "report_only": True,
        "reads_tensor_payloads": bool(state_report.get("reads_tensor_payloads")),
        "builds_model_modules": built_model_modules,
        "runs_forward_pass": ran_forward_pass,
        "runtime_loadable_enabled": False,
        "training_path_enabled": False,
        "reference_forward_enabled": bool(enable_reference_forward),
        "allow_trust_remote_code": bool(allow_trust_remote_code),
        "required_dependencies": required_dependencies,
        "dependency_status": dependency_status,
        "missing_dependencies": missing_dependencies,
        "config_probe": config_probe,
        "forward_plan": forward_plan,
        "reference_forward_result": reference_forward_result,
        "state_dict_loader_ok": bool(state_report.get("ok")),
        "state_dict_tensor_count": int(state_report.get("state_dict_tensor_count") or 0),
        "gguf_tensor_count": int(state_report.get("gguf_tensor_count") or 0),
        "state_dict_truncated": bool(state_report.get("truncated")),
        "dtype_counts": dict(state_report.get("dtype_counts") or {}),
        "memory_estimate_bytes": int(state_report.get("memory_estimate_bytes") or 0),
        "shape_contract_ok": bool(state_report.get("shape_contract_ok")),
        "runtime_loadable": False,
        "runtime_blockers": runtime_blockers,
    }


def _load_state_summary(gguf_path: str | Path, *, sidecar_path: str | Path | None, max_tensors: int) -> dict[str, Any]:
    try:
        return summarize_image_gguf_state_dict_load(gguf_path, sidecar_path=sidecar_path, max_tensors=max_tensors)
    except Exception as exc:  # pragma: no cover - diagnostics for real model runs
        return {
            "ok": False,
            "component": "",
            "family": "",
            "reads_tensor_payloads": False,
            "runs_forward_pass": False,
            "runtime_loadable_enabled": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _status_for_gate(
    *,
    component: str,
    state_report: dict[str, Any],
    missing_dependencies: list[str],
    config_probe: dict[str, Any],
    forward_plan: dict[str, Any],
    enable_reference_forward: bool,
) -> tuple[str, str]:
    if not state_report.get("ok"):
        error = str(state_report.get("error") or "state_dict loader did not pass")
        return "failed_state_dict_loader", error
    if component not in SUPPORTED_COMPONENT_DEPENDENCIES:
        return "not_supported", f"forward quality gate is not implemented for component: {component or '<missing>'}"
    if missing_dependencies:
        return "skipped_missing_dependency", "install Launcher support dependency forward packages before running model forward checks"
    if not config_probe.get("has_model_config"):
        return "skipped_missing_model_config", "forward reference construction needs a colocated model config"
    if forward_plan.get("requires_trust_remote_code") and not forward_plan.get("trust_remote_code_allowed"):
        return "skipped_requires_trust_remote_code", "model config requires trust_remote_code and the gate keeps it disabled by default"
    if not forward_plan.get("supported"):
        return "skipped_unsupported_model_class", str(forward_plan.get("reason") or "no supported forward adapter")
    if not enable_reference_forward:
        return "skipped_reference_forward_disabled", "dependency, config, and adapter probes passed; reference forward is disabled by default"
    return "ready_reference_forward", "dependency, config, and adapter probes passed"


def _dependency_status(names: list[str]) -> dict[str, bool]:
    return {name: importlib.util.find_spec(name) is not None for name in names}


def _probe_model_config(sources: list[Path], component: str) -> dict[str, Any]:
    filenames = MODEL_CONFIG_FILENAMES.get(component, [])
    roots = _candidate_config_roots(sources)
    found = []
    for root in roots:
        for filename in filenames:
            path = root / filename
            if path.is_file():
                found.append({"name": filename, "path": str(path)})
    return {
        "component": component,
        "searched_roots": [str(root) for root in roots[:12]],
        "searched_filename_count": len(filenames),
        "found": found[:12],
        "has_model_config": any(item["name"].endswith("config.json") for item in found),
    }


def _build_forward_plan(
    *,
    component: str,
    family: str,
    config_probe: dict[str, Any],
    allow_trust_remote_code: bool,
) -> dict[str, Any]:
    config_path = _first_config_path(config_probe)
    if config_path is None:
        return _forward_plan(
            component=component,
            family=family,
            supported=False,
            reason="missing model config",
        )
    config = _read_json_config(config_path)
    if component == "vae":
        return _vae_forward_plan(config_path, config, family=family)
    if component == "clip":
        return _clip_forward_plan(config_path, config, family=family, allow_trust_remote_code=allow_trust_remote_code)
    return _forward_plan(
        component=component,
        family=family,
        supported=False,
        config_path=config_path,
        reason=f"component does not have a forward adapter: {component or '<missing>'}",
    )


def _vae_forward_plan(config_path: Path, config: dict[str, Any], *, family: str) -> dict[str, Any]:
    class_name = str(config.get("_class_name") or "")
    if class_name == "AutoencoderKL":
        return _forward_plan(
            component="vae",
            family=family,
            supported=True,
            config_path=config_path,
            adapter="diffusers_autoencoder_kl",
            model_class=class_name,
            input_spec={
                "sample_shape": [1, _safe_int(config.get("in_channels"), 3), 32, 32],
                "dtype": "float32",
                "device": "cpu",
            },
        )
    return _forward_plan(
        component="vae",
        family=family,
        supported=False,
        config_path=config_path,
        model_class=class_name,
        reason=f"unsupported VAE class for reference forward: {class_name or '<missing>'}",
    )


def _clip_forward_plan(
    config_path: Path,
    config: dict[str, Any],
    *,
    family: str,
    allow_trust_remote_code: bool,
) -> dict[str, Any]:
    architectures = [str(item) for item in config.get("architectures", []) or []]
    model_type = str(config.get("model_type") or "")
    requires_trust_remote_code = bool(config.get("auto_map"))
    if requires_trust_remote_code:
        return _forward_plan(
            component="clip",
            family=family,
            supported=False,
            config_path=config_path,
            model_class=architectures[0] if architectures else model_type,
            requires_trust_remote_code=True,
            trust_remote_code_allowed=allow_trust_remote_code,
            reason="custom CLIP config needs an explicit trust_remote_code adapter before reference forward",
        )
    if model_type == "clip_text_model" or any(item in {"CLIPTextModel", "CLIPTextModelWithProjection"} for item in architectures):
        return _forward_plan(
            component="clip",
            family=family,
            supported=True,
            config_path=config_path,
            adapter="transformers_clip_text_model",
            model_class="CLIPTextModelWithProjection" if "CLIPTextModelWithProjection" in architectures else "CLIPTextModel",
            input_spec={
                "input_ids_shape": [1, min(max(_safe_int(config.get("max_position_embeddings"), 77), 2), 8)],
                "vocab_size": _safe_int(config.get("vocab_size"), 49408),
                "dtype": "int64",
                "device": "cpu",
            },
        )
    return _forward_plan(
        component="clip",
        family=family,
        supported=False,
        config_path=config_path,
        model_class=architectures[0] if architectures else model_type,
        reason=f"unsupported CLIP config for reference forward: {model_type or architectures or '<missing>'}",
    )


def _run_reference_forward_comparison(
    sources: list[Path],
    gguf_path: str | Path,
    *,
    sidecar_path: str | Path | None,
    component: str,
    forward_plan: dict[str, Any],
) -> dict[str, Any]:
    try:
        gguf_report = load_image_gguf_state_dict_with_report(gguf_path, sidecar_path=sidecar_path, max_tensors=0)
        source_state = _load_source_state_dict(sources)
        gguf_state = dict(gguf_report.get("state_dict") or {})
        if component == "vae":
            return _run_vae_reference_forward(source_state, gguf_state, forward_plan)
        if component == "clip":
            return _run_clip_reference_forward(source_state, gguf_state, forward_plan)
        return {"ok": False, "status": "not_supported", "status_reason": f"component is not supported: {component}"}
    except Exception as exc:  # pragma: no cover - optional dependency path
        return {
            "ok": False,
            "status": "failed_exception",
            "status_reason": f"{type(exc).__name__}: {exc}",
            "builds_model_modules": False,
            "runs_forward_pass": False,
        }


def _run_vae_reference_forward(source_state: dict[str, Any], gguf_state: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    import torch
    from diffusers import AutoencoderKL

    config = _read_json_config(Path(str(plan["config_path"])))
    source_model = AutoencoderKL.from_config(config).eval()
    gguf_model = AutoencoderKL.from_config(config).eval()
    source_prepared = _prepare_state_dict_for_model(source_state, source_model)
    gguf_prepared = _prepare_state_dict_for_model(gguf_state, gguf_model)
    source_incompatible = source_model.load_state_dict(source_prepared["state_dict"], strict=False)
    gguf_incompatible = gguf_model.load_state_dict(gguf_prepared["state_dict"], strict=False)
    sample_shape = list(plan.get("input_spec", {}).get("sample_shape") or [1, 3, 32, 32])
    sample = torch.zeros(sample_shape, dtype=torch.float32)
    with torch.inference_mode():
        source_output = _extract_output_tensors(source_model(sample), prefix="vae")
        gguf_output = _extract_output_tensors(gguf_model(sample), prefix="vae")
    comparisons = _compare_outputs(source_output, gguf_output, FORWARD_THRESHOLDS["vae"])
    return _forward_result("vae", comparisons, source_incompatible, gguf_incompatible, source_prepared, gguf_prepared)


def _run_clip_reference_forward(source_state: dict[str, Any], gguf_state: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    import torch
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTextModelWithProjection

    config = CLIPTextConfig.from_json_file(str(plan["config_path"]))
    model_class = CLIPTextModelWithProjection if plan.get("model_class") == "CLIPTextModelWithProjection" else CLIPTextModel
    source_model = model_class(config).eval()
    gguf_model = model_class(config).eval()
    source_prepared = _prepare_state_dict_for_model(source_state, source_model)
    gguf_prepared = _prepare_state_dict_for_model(gguf_state, gguf_model)
    source_incompatible = source_model.load_state_dict(source_prepared["state_dict"], strict=False)
    gguf_incompatible = gguf_model.load_state_dict(gguf_prepared["state_dict"], strict=False)
    seq_len = int(list(plan.get("input_spec", {}).get("input_ids_shape") or [1, 8])[-1])
    vocab_size = max(int(getattr(config, "vocab_size", 49408) or 49408), 4)
    input_ids = torch.arange(seq_len, dtype=torch.long).remainder(vocab_size).unsqueeze(0)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        source_output = _extract_output_tensors(source_model(input_ids=input_ids, attention_mask=attention_mask), prefix="clip")
        gguf_output = _extract_output_tensors(gguf_model(input_ids=input_ids, attention_mask=attention_mask), prefix="clip")
    comparisons = _compare_outputs(source_output, gguf_output, FORWARD_THRESHOLDS["clip"])
    return _forward_result("clip", comparisons, source_incompatible, gguf_incompatible, source_prepared, gguf_prepared)


def _prepare_state_dict_for_model(state: dict[str, Any], model: Any) -> dict[str, Any]:
    target_keys = set(model.state_dict().keys())
    exact = sum(1 for key in state if key in target_keys)
    stripped = _strip_prefix_state_dict(state, "text_model.")
    stripped_matches = sum(1 for key in stripped if key in target_keys)
    prefixed = _add_prefix_state_dict(state, "text_model.")
    prefixed_matches = sum(1 for key in prefixed if key in target_keys)
    variants = [
        ("identity", state, exact),
        ("strip_text_model_prefix", stripped, stripped_matches),
        ("add_text_model_prefix", prefixed, prefixed_matches),
    ]
    name, selected, matches = max(variants, key=lambda item: item[2])
    return {
        "state_dict": selected,
        "adapter": name,
        "target_key_count": len(target_keys),
        "source_key_count": len(state),
        "matched_key_count": matches,
    }


def _strip_prefix_state_dict(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {key.removeprefix(prefix): value for key, value in state.items()}


def _add_prefix_state_dict(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    return {key if key.startswith(prefix) else f"{prefix}{key}": value for key, value in state.items()}


def _load_source_state_dict(sources: list[Path]) -> dict[str, Any]:
    from safetensors import safe_open

    state: dict[str, Any] = {}
    for source in sources:
        with safe_open(str(source), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                state.setdefault(str(key), handle.get_tensor(key).detach().cpu().contiguous())
    return state


def _extract_output_tensors(output: Any, *, prefix: str) -> dict[str, Any]:
    tensors: dict[str, Any] = {}
    if hasattr(output, "sample"):
        tensors[f"{prefix}.sample"] = output.sample
    if hasattr(output, "last_hidden_state"):
        tensors[f"{prefix}.last_hidden_state"] = output.last_hidden_state
    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        tensors[f"{prefix}.pooler_output"] = output.pooler_output
    if hasattr(output, "text_embeds") and output.text_embeds is not None:
        tensors[f"{prefix}.text_embeds"] = output.text_embeds
    if not tensors and isinstance(output, (tuple, list)) and output:
        tensors[f"{prefix}.output0"] = output[0]
    return tensors


def _compare_outputs(source: dict[str, Any], gguf: dict[str, Any], thresholds: dict[str, float]) -> list[dict[str, Any]]:
    import torch

    records: list[dict[str, Any]] = []
    for name, source_tensor in source.items():
        gguf_tensor = gguf.get(name)
        if gguf_tensor is None:
            records.append({"name": name, "ok": False, "reason": "missing gguf output"})
            continue
        if list(source_tensor.shape) != list(gguf_tensor.shape):
            records.append({"name": name, "ok": False, "reason": "shape mismatch", "source_shape": list(source_tensor.shape), "gguf_shape": list(gguf_tensor.shape)})
            continue
        source_value = source_tensor.detach().cpu().float()
        gguf_value = gguf_tensor.detach().cpu().float()
        diff = (source_value - gguf_value).abs()
        max_abs = float(diff.max().item()) if diff.numel() else 0.0
        denom = source_value.abs().clamp_min(1.0e-6)
        max_rel = float((diff / denom).max().item()) if diff.numel() else 0.0
        finite = bool(torch.isfinite(source_value).all().item() and torch.isfinite(gguf_value).all().item())
        cosine = _cosine_similarity(source_value, gguf_value)
        ok = finite and max_abs <= thresholds["max_abs_error"] and max_rel <= thresholds["max_rel_error"] and cosine >= thresholds["min_cosine_similarity"]
        records.append({"name": name, "ok": ok, "finite": finite, "max_abs_error": max_abs, "max_rel_error": max_rel, "cosine_similarity": cosine})
    return records


def _forward_result(
    component: str,
    comparisons: list[dict[str, Any]],
    source_incompatible: Any,
    gguf_incompatible: Any,
    source_prepared: dict[str, Any],
    gguf_prepared: dict[str, Any],
) -> dict[str, Any]:
    failed = [item for item in comparisons if not item.get("ok")]
    return {
        "ok": not failed and bool(comparisons),
        "status": "passed" if not failed and comparisons else "failed_forward_comparison",
        "status_reason": "reference forward outputs are within tolerance" if not failed and comparisons else "reference forward output drift exceeded tolerance",
        "component": component,
        "builds_model_modules": True,
        "runs_forward_pass": True,
        "comparison_count": len(comparisons),
        "failed_comparison_count": len(failed),
        "comparisons": comparisons,
        "source_missing_keys": list(getattr(source_incompatible, "missing_keys", []) or [])[:16],
        "source_unexpected_keys": list(getattr(source_incompatible, "unexpected_keys", []) or [])[:16],
        "gguf_missing_keys": list(getattr(gguf_incompatible, "missing_keys", []) or [])[:16],
        "gguf_unexpected_keys": list(getattr(gguf_incompatible, "unexpected_keys", []) or [])[:16],
        "source_state_dict_adapter": _state_dict_adapter_summary(source_prepared),
        "gguf_state_dict_adapter": _state_dict_adapter_summary(gguf_prepared),
    }


def _state_dict_adapter_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "adapter": str(report.get("adapter") or ""),
        "source_key_count": int(report.get("source_key_count") or 0),
        "target_key_count": int(report.get("target_key_count") or 0),
        "matched_key_count": int(report.get("matched_key_count") or 0),
    }


def _cosine_similarity(left: Any, right: Any) -> float:
    import torch

    a = left.reshape(-1).float()
    b = right.reshape(-1).float()
    if a.numel() == 0 or b.numel() == 0:
        return 1.0
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.item()) <= 0.0:
        return 1.0 if float(torch.linalg.vector_norm(a - b).item()) == 0.0 else 0.0
    return float(torch.clamp(torch.dot(a, b) / denom, min=-1.0, max=1.0).item())


def _first_config_path(config_probe: dict[str, Any]) -> Path | None:
    for item in config_probe.get("found", []) or []:
        if str(item.get("name") or "").endswith("config.json"):
            return Path(str(item.get("path") or ""))
    return None


def _read_json_config(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _forward_plan(
    *,
    component: str,
    family: str,
    supported: bool,
    config_path: Path | None = None,
    adapter: str = "",
    model_class: str = "",
    reason: str = "",
    requires_trust_remote_code: bool = False,
    trust_remote_code_allowed: bool = False,
    input_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "component": component,
        "family": family,
        "supported": bool(supported),
        "adapter": adapter,
        "model_class": model_class,
        "config_path": str(config_path or ""),
        "config_root": str(config_path.parent if config_path else ""),
        "requires_trust_remote_code": bool(requires_trust_remote_code),
        "trust_remote_code_allowed": bool(trust_remote_code_allowed),
        "input_spec": dict(input_spec or {}),
        "reason": reason,
    }


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return fallback


def _candidate_config_roots(sources: list[Path]) -> list[Path]:
    roots: list[Path] = []
    seen: set[str] = set()
    for source in sources:
        for root in [source.parent, source.parent.parent]:
            key = str(root.resolve()) if root.exists() else str(root)
            if key in seen:
                continue
            seen.add(key)
            roots.append(root)
    return roots


def _normalize_sources(source_paths: str | Path | Iterable[str | Path]) -> list[Path]:
    if isinstance(source_paths, (str, Path)):
        raw_paths = [source_paths]
    else:
        raw_paths = list(source_paths)
    return [Path(path) for path in raw_paths]


__all__ = ["run_image_gguf_forward_quality_gate"]
