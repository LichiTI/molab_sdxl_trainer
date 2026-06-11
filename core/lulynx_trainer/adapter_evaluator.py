"""Lightweight post-training adapter health evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import torch
from core.safe_pickle import safe_torch_load


def evaluate_adapter(path: str | Path) -> dict[str, Any]:
    adapter_path = Path(path)
    warnings: list[str] = []
    if not adapter_path.is_file():
        return {
            "schema_version": 1,
            "adapter_path": str(adapter_path),
            "exists": False,
            "warnings": [f"Adapter file not found: {adapter_path}"],
        }
    state, metadata = _load_state(adapter_path)
    tensor_count = 0
    parameter_count = 0
    nonzero_count = 0
    abs_sum = 0.0
    sq_sum = 0.0
    max_abs = 0.0
    key_families: Counter[str] = Counter()
    rank_estimates: list[int] = []
    shape_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()

    for key, tensor in state.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        data = tensor.detach().float().cpu()
        numel = int(data.numel())
        if numel <= 0:
            continue
        tensor_count += 1
        parameter_count += numel
        nonzero_count += int(torch.count_nonzero(data).item())
        abs_sum += float(data.abs().sum().item())
        sq_sum += float((data * data).sum().item())
        max_abs = max(max_abs, float(data.abs().max().item()))
        key_families[_key_family(key)] += 1
        shape_counts["x".join(str(dim) for dim in data.shape)] += 1
        dtype_counts[str(tensor.dtype).replace("torch.", "")] += 1
        if data.ndim == 2 and min(data.shape) <= 256:
            rank_estimates.append(int(torch.linalg.matrix_rank(data).item()))

    if tensor_count == 0:
        warnings.append("No tensors were found in the adapter file.")
    if parameter_count > 0 and nonzero_count == 0:
        warnings.append("All adapter parameters are zero.")
    nonzero_ratio = nonzero_count / max(parameter_count, 1)
    mean_abs = abs_sum / max(parameter_count, 1)
    rms = (sq_sum / max(parameter_count, 1)) ** 0.5
    if 0 < nonzero_ratio < 0.01:
        warnings.append("Adapter is extremely sparse; check whether training updated the weights.")
    if max_abs > 100:
        warnings.append("Adapter contains very large weights; inspect training stability.")

    return {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "adapter_path": str(adapter_path),
        "exists": True,
        "size_bytes": adapter_path.stat().st_size,
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "nonzero_count": nonzero_count,
        "nonzero_ratio": round(nonzero_ratio, 6),
        "mean_abs": round(mean_abs, 8),
        "rms": round(rms, 8),
        "max_abs": round(max_abs, 8),
        "rank_estimate_mean": round(sum(rank_estimates) / max(len(rank_estimates), 1), 4) if rank_estimates else 0.0,
        "rank_estimate_max": max(rank_estimates) if rank_estimates else 0,
        "key_families": _counter_payload(key_families),
        "shape_counts": _counter_payload(shape_counts, limit=20),
        "dtype_counts": _counter_payload(dtype_counts),
        "metadata": metadata,
        "warnings": warnings,
    }


def _load_state(path: Path) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file
        from safetensors import safe_open
        metadata: dict[str, str] = {}
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = dict(handle.metadata() or {})
        except Exception:
            metadata = {}
        return load_file(str(path), device="cpu"), metadata
    payload = safe_torch_load(str(path), map_location="cpu")
    if isinstance(payload, dict) and all(isinstance(value, torch.Tensor) for value in payload.values()):
        return payload, {}
    if isinstance(payload, dict):
        for key in ("state_dict", "network_state_dict", "adapter_state_dict"):
            value = payload.get(key)
            if isinstance(value, dict):
                return {str(k): v for k, v in value.items() if isinstance(v, torch.Tensor)}, {}
    return {}, {}


def _key_family(key: str) -> str:
    lowered = key.lower()
    if any(token in lowered for token in ("lora_down", "lora_a", "down")):
        return "down"
    if any(token in lowered for token in ("lora_up", "lora_b", "up")):
        return "up"
    if "alpha" in lowered:
        return "alpha"
    if "dora" in lowered:
        return "dora"
    if "lokr" in lowered:
        return "lokr"
    if "hada" in lowered:
        return "hada"
    return "other"


def _counter_payload(counter: Counter[str], limit: int | None = None) -> list[dict[str, Any]]:
    rows = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        rows = rows[:limit]
    return [{"name": key, "count": int(value)} for key, value in rows]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate saved LoRA/adapter tensor health.")
    parser.add_argument("adapter")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    report = evaluate_adapter(args.adapter)
    text = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)
    return 0 if report.get("exists") else 1


if __name__ == "__main__":
    raise SystemExit(main())
