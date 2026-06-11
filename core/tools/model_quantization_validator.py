"""Validation report for Lulynx model quantization outputs.

The validator proves that toolbox quantization artifacts are readable through
the trainer-owned safetensors loader and that the decoded tensors stay within
format-specific drift thresholds.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch

try:
    from core.lulynx_trainer.lulynx_quantized_safetensors import (
        FORMAT_KEY,
        PLAIN_FORMATS,
        ROWWISE_FORMATS,
        SCHEMA_KEY,
        SCHEMA_VERSION,
        TENSORS_KEY,
        is_lulynx_quantized_metadata,
        normalize_decode_dtype,
        normalize_quant_format,
        parse_quantized_tensor_entries,
        torch_dtype_for_plain_format,
    )
    from core.lulynx_trainer.safetensors_loader import load_safetensors, open_safetensors
except ImportError:
    from backend.core.lulynx_trainer.lulynx_quantized_safetensors import (
        FORMAT_KEY,
        PLAIN_FORMATS,
        ROWWISE_FORMATS,
        SCHEMA_KEY,
        SCHEMA_VERSION,
        TENSORS_KEY,
        is_lulynx_quantized_metadata,
        normalize_decode_dtype,
        normalize_quant_format,
        parse_quantized_tensor_entries,
        torch_dtype_for_plain_format,
    )
    from backend.core.lulynx_trainer.safetensors_loader import load_safetensors, open_safetensors


DRIFT_THRESHOLDS = {
    "fp16": {"abs_base": 0.002, "abs_scale": 0.002, "mean_base": 0.001, "mean_scale": 0.001, "cosine": 0.999},
    "bf16": {"abs_base": 0.02, "abs_scale": 0.01, "mean_base": 0.006, "mean_scale": 0.004, "cosine": 0.999},
    "fp8_e4m3fn": {"abs_base": 0.25, "abs_scale": 0.1, "mean_base": 0.08, "mean_scale": 0.04, "cosine": 0.98},
    "lulynx_int8_rowwise": {"abs_base": 0.03, "abs_scale": 0.015, "mean_base": 0.01, "mean_scale": 0.004, "cosine": 0.999},
    "lulynx_uint4_rowwise": {"abs_base": 0.25, "abs_scale": 0.1, "mean_base": 0.09, "mean_scale": 0.04, "cosine": 0.98},
}


def validate_quantized_model_file(
    source_path: str | Path,
    output_path: str | Path,
    quant_format: str,
    *,
    decode_dtype: str = "fp16",
    max_tensors: int = 0,
) -> dict[str, Any]:
    fmt = normalize_quant_format(quant_format)
    if fmt not in PLAIN_FORMATS | ROWWISE_FORMATS:
        raise ValueError(f"not a Lulynx safetensors quantization format: {quant_format}")
    src = Path(source_path)
    dst = Path(output_path)
    if not src.is_file():
        raise FileNotFoundError(f"source file not found: {src}")
    if not dst.is_file():
        raise FileNotFoundError(f"quantized output file not found: {dst}")
    if dst.suffix.lower() != ".safetensors":
        raise ValueError("quantized output must be a .safetensors file")

    reference = _load_reference_state(src)
    output_metadata = _read_safetensors_metadata(dst)
    decoded = load_safetensors(str(dst), disable_mmap=False)
    checks: list[dict[str, Any]] = []
    checks.extend(_metadata_checks(fmt, output_metadata))
    checks.extend(_loader_contract_checks(reference, decoded, dst))
    entries = parse_quantized_tensor_entries(output_metadata) if is_lulynx_quantized_metadata(output_metadata) else []
    checks.extend(_entry_checks(fmt, decode_dtype, entries, reference, output_metadata))
    drift = _drift_report(reference, decoded, fmt, max_tensors=max_tensors)
    checks.append(_simple_check("drift_thresholds", "decoded tensors are within format drift thresholds", bool(drift["ok"])))
    issues = [check for check in checks if not check["ok"]]
    return {
        "schema_version": 1,
        "validator": "lulynx_model_quantization_validator_v1",
        "ok": not issues,
        "status": "passed" if not issues else "failed",
        "report_only": True,
        "source_path": str(src),
        "output_path": str(dst),
        "quant_format": fmt,
        "output_format": "safetensors",
        "trainer_loader_compatible": _has_same_keys(reference, decoded),
        "source_tensor_count": len(reference),
        "decoded_tensor_count": len(decoded),
        "source_dtype_counts": _dtype_counts(reference),
        "decoded_dtype_counts": _dtype_counts(decoded),
        "metadata_schema": str(output_metadata.get(SCHEMA_KEY) or ""),
        "metadata_format": str(output_metadata.get(FORMAT_KEY) or ""),
        "quantized_entry_count": len(entries),
        "check_count": len(checks),
        "failed_check_count": len(issues),
        "failed_checks": [check["name"] for check in issues[:16]],
        "checks": checks,
        "drift": drift,
    }


def _metadata_checks(fmt: str, metadata: dict[str, str]) -> list[dict[str, Any]]:
    if fmt in PLAIN_FORMATS:
        return [
            _simple_check("plain_metadata_schema_absent", "plain dtype conversion must not keep Lulynx rowwise schema", not metadata.get(SCHEMA_KEY)),
            _simple_check("plain_metadata_entries_absent", "plain dtype conversion must not keep rowwise tensor entries", not metadata.get(TENSORS_KEY)),
        ]
    return [
        _simple_check("rowwise_metadata_schema", "rowwise output declares supported schema version", metadata.get(SCHEMA_KEY) == SCHEMA_VERSION),
        _simple_check("rowwise_metadata_format", "rowwise output declares the requested format", metadata.get(FORMAT_KEY) == fmt),
        _simple_check("rowwise_metadata_entries", "rowwise output declares tensor entries", bool(metadata.get(TENSORS_KEY))),
    ]


def _loader_contract_checks(reference: dict[str, torch.Tensor], decoded: dict[str, torch.Tensor], output_path: Path) -> list[dict[str, Any]]:
    reference_keys = set(reference)
    decoded_keys = set(decoded)
    checks = [
        _simple_check("decoded_key_set", "trainer loader exposes the original tensor keys", decoded_keys == reference_keys, missing=sorted(reference_keys - decoded_keys)[:8], unexpected=sorted(decoded_keys - reference_keys)[:8]),
    ]
    try:
        with open_safetensors(str(output_path), disable_mmap=False) as handle:
            handle_keys = {str(key) for key in handle.keys()}
            checks.append(_simple_check("safe_open_key_set", "open_safetensors exposes the original tensor keys", handle_keys == reference_keys, missing=sorted(reference_keys - handle_keys)[:8], unexpected=sorted(handle_keys - reference_keys)[:8]))
            sample_key = next(iter(sorted(reference_keys)), "")
            if sample_key:
                checks.append(_simple_check("safe_open_sample_shape", "open_safetensors sample shape matches source", list(handle.get_slice(sample_key).get_shape()) == list(reference[sample_key].shape), key=sample_key))
    except Exception as exc:
        checks.append(_simple_check("safe_open_contract", f"open_safetensors failed: {type(exc).__name__}: {exc}", False))
    return checks


def _entry_checks(
    fmt: str,
    decode_dtype: str,
    entries: list[Any],
    reference: dict[str, torch.Tensor],
    metadata: dict[str, str],
) -> list[dict[str, Any]]:
    if fmt in PLAIN_FORMATS:
        return []
    expected_decode_dtype = normalize_decode_dtype(decode_dtype)
    quantizable_count = sum(1 for tensor in reference.values() if _can_rowwise_quantize(tensor))
    entry_keys = {entry.key for entry in entries}
    checks = [
        _simple_check("rowwise_entry_count", "rowwise entries cover every quantizable tensor", len(entries) == quantizable_count, entry_count=len(entries), quantizable_count=quantizable_count),
        _simple_check("rowwise_entry_keys", "rowwise entries refer to source tensor keys", entry_keys.issubset(set(reference)), unexpected=sorted(entry_keys - set(reference))[:8]),
        _simple_check("rowwise_entry_decode_dtype", "rowwise entries use requested decode dtype", all(entry.decode_dtype == expected_decode_dtype for entry in entries), expected=expected_decode_dtype),
    ]
    checks.append(_simple_check("rowwise_metadata_parse", "rowwise metadata parses as a tensor entry list", bool(entries) or quantizable_count == 0, metadata_keys=sorted(metadata)[:8]))
    return checks


def _drift_report(reference: dict[str, torch.Tensor], decoded: dict[str, torch.Tensor], fmt: str, *, max_tensors: int) -> dict[str, Any]:
    limit = int(max_tensors or 0)
    records: list[dict[str, Any]] = []
    failed: list[str] = []
    threshold = DRIFT_THRESHOLDS[fmt]
    for index, key in enumerate(sorted(reference)):
        if limit > 0 and index >= limit:
            break
        source = reference[key]
        value = decoded.get(key)
        if value is None:
            records.append({"name": key, "ok": False, "reason": "missing decoded tensor"})
            failed.append(key)
            continue
        record = _tensor_drift_record(key, source, value, fmt, threshold)
        records.append(record)
        if not record.get("ok"):
            failed.append(key)
    return {
        "ok": not failed,
        "sampled_tensor_count": len(records),
        "failed_tensor_count": len(failed),
        "failed_tensors": failed[:16],
        "threshold": dict(threshold),
        "records": _selected_drift_records(records, failed),
    }


def _selected_drift_records(records: list[dict[str, Any]], failed: list[str]) -> list[dict[str, Any]]:
    selected = records[:32]
    known = {str(record.get("name") or "") for record in selected}
    for record in records:
        name = str(record.get("name") or "")
        if name in known or name not in failed:
            continue
        selected.append(record)
        known.add(name)
        if len(selected) >= 48:
            break
    return selected


def _tensor_drift_record(name: str, source: torch.Tensor, decoded: torch.Tensor, fmt: str, threshold: dict[str, float]) -> dict[str, Any]:
    if list(source.shape) != list(decoded.shape):
        return {"name": name, "ok": False, "reason": "shape mismatch", "source_shape": list(source.shape), "decoded_shape": list(decoded.shape)}
    if not source.is_floating_point():
        same_dtype = source.dtype == decoded.dtype
        same_value = bool(torch.equal(source.cpu(), decoded.cpu()))
        return {"name": name, "ok": same_dtype and same_value, "dtype": _dtype_name(decoded), "source_dtype": _dtype_name(source), "non_floating": True, "same_value": same_value}
    source_value = _finite_float(source)
    decoded_value = _finite_float(decoded)
    diff = (source_value - decoded_value).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0
    mean_abs = float(diff.mean().item()) if diff.numel() else 0.0
    source_max = float(source_value.abs().max().item()) if source_value.numel() else 0.0
    max_allowed = float(threshold["abs_base"] + threshold["abs_scale"] * source_max)
    mean_allowed = float(threshold["mean_base"] + threshold["mean_scale"] * source_max)
    finite = bool(torch.isfinite(decoded_value).all().item() and torch.isfinite(source_value).all().item())
    cosine = _cosine_similarity(source_value, decoded_value)
    dtype_ok = _decoded_dtype_ok(fmt, decoded)
    ok = finite and dtype_ok and max_abs <= max_allowed and mean_abs <= mean_allowed and cosine >= float(threshold["cosine"])
    return {
        "name": name,
        "ok": ok,
        "shape": [int(dim) for dim in source.shape],
        "source_dtype": _dtype_name(source),
        "decoded_dtype": _dtype_name(decoded),
        "decoded_dtype_ok": dtype_ok,
        "finite": finite,
        "max_abs_error": max_abs,
        "max_abs_allowed": max_allowed,
        "mean_abs_error": mean_abs,
        "mean_abs_allowed": mean_allowed,
        "cosine_similarity": cosine,
    }


def _load_reference_state(path: Path) -> dict[str, torch.Tensor]:
    suffix = path.suffix.lower()
    if suffix == ".safetensors":
        return load_safetensors(str(path), disable_mmap=False)
    if suffix in {".pt", ".pth", ".ckpt"}:
        data = torch.load(str(path), map_location="cpu", weights_only=True)
        if isinstance(data, dict) and "state_dict" in data and isinstance(data["state_dict"], dict):
            data = data["state_dict"]
        if not isinstance(data, dict):
            raise ValueError("PyTorch file does not contain a state dict")
        return {str(key): value.detach().cpu().contiguous() for key, value in data.items() if torch.is_tensor(value)}
    raise ValueError("source_path must be a .safetensors, .pt, .pth, or .ckpt file")


def _read_safetensors_metadata(path: Path) -> dict[str, str]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def _decoded_dtype_ok(fmt: str, decoded: torch.Tensor) -> bool:
    if fmt in PLAIN_FORMATS:
        return decoded.dtype == torch_dtype_for_plain_format(fmt)
    return decoded.dtype in {torch.float16, torch.bfloat16, torch.float32}


def _can_rowwise_quantize(tensor: torch.Tensor) -> bool:
    return torch.is_tensor(tensor) and tensor.is_floating_point() and tensor.dim() >= 2 and tensor.numel() > 0


def _finite_float(tensor: torch.Tensor) -> torch.Tensor:
    value = tensor.detach().cpu().float().contiguous()
    if torch.isfinite(value).all():
        return value
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)


def _cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.reshape(-1).float()
    b = right.reshape(-1).float()
    if a.numel() == 0 or b.numel() == 0:
        return 1.0
    denom = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denom.item()) <= 0.0:
        return 1.0 if float(torch.linalg.vector_norm(a - b).item()) == 0.0 else 0.0
    return float(torch.clamp(torch.dot(a, b) / denom, min=-1.0, max=1.0).item())


def _dtype_counts(state: dict[str, torch.Tensor]) -> dict[str, int]:
    return dict(sorted(Counter(_dtype_name(tensor) for tensor in state.values()).items()))


def _dtype_name(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).replace("torch.", "")


def _has_same_keys(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    return set(left) == set(right)


def _simple_check(name: str, message: str, ok: bool, **metadata: Any) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "message": message, "metadata": dict(metadata)}


__all__ = ["DRIFT_THRESHOLDS", "validate_quantized_model_file"]
