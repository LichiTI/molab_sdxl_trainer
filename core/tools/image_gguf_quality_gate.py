"""Lightweight quality gates for image GGUF shadow exports.

These gates validate restored GGUF state_dict payloads structurally. They do not
build model modules, run forward passes, or promote GGUF files to runtime
loadable. Forward quality gates such as VAE reconstruction or CLIP hidden-state
similarity should build on this report later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

try:
    from core.tools.image_gguf_payload_parity import check_image_gguf_payload_parity
    from core.tools.image_gguf_state_dict_loader import load_image_gguf_state_dict_with_report
except ImportError:
    from backend.core.tools.image_gguf_payload_parity import check_image_gguf_payload_parity
    from backend.core.tools.image_gguf_state_dict_loader import load_image_gguf_state_dict_with_report


REQUIRED_KEYS = {
    "vae": {
        "diffusers_vae": [
            "encoder.conv_in.weight",
            "decoder.conv_in.weight",
            "decoder.conv_out.weight",
        ],
        "qwen_image_vae": [
            "conv1.weight",
            "conv2.weight",
            "encoder.downsamples.0.residual.2.weight",
            "decoder.upsamples.0.residual.2.weight",
        ],
    },
    "clip": {
        "clip_text": [
            "text_model.embeddings.token_embedding.weight",
            "text_model.encoder.layers.0.self_attn.q_proj.weight",
            "text_model.final_layer_norm.weight",
        ],
        "jina_clip_text": [
            "model.embeddings.word_embeddings.weight",
            "model.encoder.layers.0.mixer.Wqkv.weight",
            "model.emb_ln.weight",
        ],
    },
}


def run_image_gguf_lightweight_quality_gate(
    source_paths: str | Path | list[str | Path],
    gguf_path: str | Path,
    *,
    sidecar_path: str | Path | None = None,
    max_tensors: int = 0,
    parity_max_tensors: int = 8,
    parity_max_elements_per_tensor: int = 4096,
) -> dict[str, Any]:
    state_report = load_image_gguf_state_dict_with_report(gguf_path, sidecar_path=sidecar_path, max_tensors=max_tensors)
    state_dict: dict[str, torch.Tensor] = state_report["state_dict"]
    parity_report = check_image_gguf_payload_parity(
        source_paths,
        gguf_path,
        sidecar_path=sidecar_path,
        max_tensors=parity_max_tensors,
        max_elements_per_tensor=parity_max_elements_per_tensor,
    )
    component = str(state_report.get("component") or "")
    family = str(state_report.get("family") or "")
    checks = []
    checks.extend(_required_key_checks(component, family, state_dict))
    checks.extend(_tensor_health_checks(state_dict))
    checks.append(_simple_check("shape_contract", "shape contract passed", bool(state_report.get("shape_contract_ok"))))
    checks.append(_simple_check("payload_parity", "sampled payload parity passed", bool(parity_report.get("ok"))))
    issues = [check for check in checks if not check["ok"]]
    return {
        "schema_version": 1,
        "gate": "image_gguf_lightweight_quality_gate_v1",
        "quality_stage": "lightweight_structural_gate",
        "ok": not issues,
        "component": component,
        "family": family,
        "report_only": True,
        "reads_tensor_payloads": True,
        "builds_model_modules": False,
        "runs_forward_pass": False,
        "runtime_loadable_enabled": False,
        "training_path_enabled": False,
        "state_dict_tensor_count": int(state_report.get("state_dict_tensor_count") or 0),
        "gguf_tensor_count": int(state_report.get("gguf_tensor_count") or 0),
        "dtype_counts": dict(state_report.get("dtype_counts") or {}),
        "memory_estimate_bytes": int(state_report.get("memory_estimate_bytes") or 0),
        "payload_parity_ok": bool(parity_report.get("ok")),
        "payload_parity_max_abs_error": float(parity_report.get("max_abs_error") or 0.0),
        "payload_parity_max_rel_error": float(parity_report.get("max_rel_error") or 0.0),
        "check_count": len(checks),
        "failed_check_count": len(issues),
        "failed_checks": [check["name"] for check in issues[:16]],
        "checks": checks,
    }


def _required_key_checks(component: str, family: str, state_dict: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    rules = REQUIRED_KEYS.get(component, {})
    required = rules.get(family) or []
    if not required:
        return [_simple_check("required_keys", f"no lightweight required-key rule for {component}/{family}", False)]
    keys = set(state_dict)
    return [
        _simple_check(f"required_key:{key}", f"required tensor present: {key}", key in keys)
        for key in required
    ]


def _tensor_health_checks(state_dict: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for name, tensor in state_dict.items():
        if not tensor.is_floating_point():
            checks.append(_simple_check(f"tensor_finite:{name}", "non-floating tensor skipped for finite check", True, dtype=str(tensor.dtype)))
            continue
        finite = bool(torch.isfinite(tensor).all().item())
        max_abs = float(tensor.detach().abs().max().item()) if tensor.numel() else 0.0
        zero_count = int((tensor == 0).sum().item()) if tensor.numel() else 0
        checks.append(
            _simple_check(
                f"tensor_finite:{name}",
                "tensor contains only finite values",
                finite,
                dtype=str(tensor.dtype).replace("torch.", ""),
                shape=[int(dim) for dim in tensor.shape],
                max_abs=max_abs,
                zero_count=zero_count,
                numel=int(tensor.numel()),
            )
        )
    return checks


def _simple_check(name: str, message: str, ok: bool, **metadata: Any) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "message": message,
        "metadata": dict(metadata),
    }


__all__ = ["run_image_gguf_lightweight_quality_gate"]
