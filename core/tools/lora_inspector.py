"""LoRA inspection helpers used by the web toolbox.

The old toolbox had several overlapping analyzer/XRay implementations.  This
module keeps the reusable backend logic in one small place so UI routes can be
thin and safe to import.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _bf16_to_fp32(raw: bytes, shape: list[int]) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).reshape(shape)


def _safe_round(value: Any, digits: int = 4, default: float = 0.0) -> float:
    try:
        if hasattr(value, "item"):
            value = value.item()
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return round(value, digits)
    except Exception:
        return default


def normalize_block_name(key: str) -> str:
    k = key.lower()
    if "te2" in k or "text_model2" in k:
        return "TE2"
    if any(x in k for x in ["te1", "te_", "text_model", "text_encoder"]):
        return "TE1"
    match = re.search(r"(input_blocks|down_blocks|output_blocks|up_blocks|double_blocks|single_blocks|blocks|layers|down|up)[._]?(\d+)", k)
    if match:
        token, idx = match.group(1), int(match.group(2))
        if token in {"input_blocks", "down_blocks", "down"}:
            return f"IN{idx:02d}"
        if token in {"output_blocks", "up_blocks", "up"}:
            return f"OUT{idx:02d}"
        if token == "double_blocks":
            return f"IN{idx // 2:02d}" if idx < 18 else "M00"
        if token == "single_blocks":
            return f"OUT{min(idx // 4, 8):02d}"
        return f"IN{idx:02d}" if idx < 12 else f"OUT{idx - 12:02d}"
    if "mid" in k or "middle" in k:
        return "M00"
    return "OTHER"


def _is_weight_key(key: str) -> bool:
    k = key.lower()
    if any(x in k for x in [".alpha", ".scale", "metadata"]):
        return False
    return any(x in k for x in ["weight", "lokr", "hada", "diff", "w1", "w2", "lora_up", "lora_down", "matrix"])


def _read_tensor(raw: bytes, dtype: str, shape: list[int]) -> np.ndarray | None:
    dtype = str(dtype or "").upper()
    if dtype == "BF16":
        return _bf16_to_fp32(raw, shape)
    if dtype == "F16":
        return np.frombuffer(raw, dtype=np.float16).astype(np.float32).reshape(shape)
    if dtype == "F32":
        return np.frombuffer(raw, dtype=np.float32).reshape(shape)
    # F8_E4M3 / F8_E5M2 等：numpy 无原生 fp8 语义，按整数伪解码只会产生噪声统计，
    # 显式跳过并由调用方标注 unsupported_dtypes。
    return None


_MODULE_SUFFIX_MARKERS = (
    ".lora_down", ".lora_up", ".lora_A", ".lora_B",
    ".hada_", ".lokr_", ".diff", ".w1", ".w2",
)


def _module_prefix(name: str) -> str:
    """Strip the weight-role suffix so the prefix matches the ``<module>.alpha`` key."""
    for marker in _MODULE_SUFFIX_MARKERS:
        idx = name.find(marker)
        if idx > 0:
            return name[:idx]
    return name.rsplit(".", 1)[0]


def _open_safetensors_header(path: str):
    handle = open(path, "rb")
    try:
        header_size = int.from_bytes(handle.read(8), "little")
        header = json.loads(handle.read(header_size).decode("utf-8"))
        return handle, header, 8 + header_size
    except Exception:
        handle.close()
        raise


def analyze_lora(file_path: str, *, max_samples: int = 1000) -> Dict[str, Any]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(str(path))

    layers: List[Dict[str, Any]] = []
    block_data: Dict[str, Dict[str, Any]] = {}
    unsupported_dtypes: Dict[str, int] = {}
    total_params = 0

    handle, header, offset_base = _open_safetensors_header(str(path))
    try:
        tensor_names = [key for key in header.keys() if key != "__metadata__"]
        alpha_values: Dict[str, float] = {}
        for name in tensor_names:
            if ".alpha" not in name.lower():
                continue
            info = header[name]
            start, end = info["data_offsets"]
            handle.seek(offset_base + start)
            tensor = _read_tensor(handle.read(end - start), info.get("dtype", "F32"), info.get("shape", [1]))
            if tensor is not None and tensor.size:
                alpha_values[name.replace(".alpha", "")] = _safe_round(tensor.flatten()[0], 4)

        for name in tensor_names:
            if not _is_weight_key(name):
                continue
            info = header[name]
            start, end = info["data_offsets"]
            handle.seek(offset_base + start)
            tensor = _read_tensor(handle.read(end - start), info.get("dtype", ""), info.get("shape", []))
            if tensor is None:
                dtype_name = str(info.get("dtype", "") or "unknown").upper()
                unsupported_dtypes[dtype_name] = unsupported_dtypes.get(dtype_name, 0) + 1
                continue

            total_params += int(tensor.size)
            rank = int(tensor.shape[0]) if tensor.ndim >= 2 and any(x in name.lower() for x in ["down", "lora_a", "w1"]) else (min(tensor.shape) if tensor.ndim >= 2 else 0)
            alpha = alpha_values.get(_module_prefix(name), float(rank) if rank > 0 else 1.0)
            scale = alpha / rank if rank > 0 else 1.0
            has_anomaly = bool(np.isnan(tensor).any() or np.isinf(tensor).any())
            if has_anomaly:
                norm_val = rms = mean_val = std_val = 0.0
                sparsity = 1.0
            else:
                tensor64 = tensor.astype(np.float64)
                norm_val = float(np.linalg.norm(tensor))
                rms = float(np.sqrt(np.mean(tensor64 ** 2)))
                mean_val = float(np.mean(tensor))
                std_val = float(np.std(tensor))
                sparsity = float(np.mean(np.abs(tensor) < 1e-7))

            # 有效信息承载量:down 矩阵的 effective rank 占比(奇异值 > 1% max 的数量 / 总秩)
            eff_rank_ratio = None
            if (not has_anomaly and tensor.ndim >= 2 and tensor.size <= 4_000_000
                    and any(x in name.lower() for x in ["down", "lora_a", "w1"])):
                mat = tensor.reshape(tensor.shape[0], -1).astype(np.float32)
                if min(mat.shape) >= 2:
                    try:
                        svals = np.linalg.svd(mat, compute_uv=False)
                        if svals.size and float(svals[0]) > 0.0:
                            eff_rank_ratio = float(np.sum(svals > svals[0] * 0.01) / svals.size)
                    except np.linalg.LinAlgError:
                        eff_rank_ratio = None

            flat = tensor.flatten()
            if flat.size > max_samples:
                sample_idx = np.linspace(0, flat.size - 1, max_samples, dtype=int)
                samples = flat[sample_idx].astype(float).tolist()
            else:
                samples = flat.astype(float).tolist()

            layer = {
                "name": name,
                "shape": list(tensor.shape),
                "params": int(tensor.size),
                "norm": _safe_round(norm_val, 6),
                "rms": _safe_round(rms, 6),
                "rms_scaled": _safe_round(rms * scale, 6),
                "mean": _safe_round(mean_val, 6),
                "std": _safe_round(std_val, 6),
                "rank": int(rank),
                "alpha": _safe_round(alpha, 4),
                "scale": _safe_round(scale, 4),
                "sparsity": _safe_round(sparsity, 4),
                "effective_rank_ratio": _safe_round(eff_rank_ratio, 4) if eff_rank_ratio is not None else None,
                "has_anomaly": has_anomaly,
                "sample_points": samples,
            }
            layers.append(layer)

            block_id = normalize_block_name(name)
            if block_id != "OTHER":
                block = block_data.setdefault(block_id, {"count": 0, "sum_norm": 0.0, "sum_sparsity": 0.0, "sum_rms": 0.0, "sum_eff": 0.0, "eff_count": 0, "dead_count": 0, "overfit_count": 0, "underfit_count": 0})
                block["count"] += 1
                block["sum_norm"] += norm_val
                block["sum_sparsity"] += sparsity
                block["sum_rms"] += rms
                if eff_rank_ratio is not None:
                    block["sum_eff"] += eff_rank_ratio
                    block["eff_count"] += 1
                if rms < 1e-6:
                    block["dead_count"] += 1
                elif rms > 0.15:
                    block["overfit_count"] += 1
                elif rms < 0.001:
                    block["underfit_count"] += 1
    finally:
        handle.close()

    lora_type = "LoRA"
    lower_names = [item["name"].lower() for item in layers]
    if any("lokr" in name for name in lower_names):
        lora_type = "LoKr"
    elif any("hada" in name for name in lower_names):
        lora_type = "LoHa"
    elif any("dora" in name for name in lower_names):
        lora_type = "DoRA"

    position_analysis = []
    recommendations = []
    for block_id in sorted(block_data):
        block = block_data[block_id]
        avg_norm = block["sum_norm"] / max(block["count"], 1)
        avg_sparsity = block["sum_sparsity"] / max(block["count"], 1)
        avg_rms = block["sum_rms"] / max(block["count"], 1)
        info_ratio = (block["sum_eff"] / block["eff_count"] * 100.0) if block["eff_count"] > 0 else None
        status = "good"
        if block["dead_count"] > block["count"] * 0.5 or avg_rms > 0.2:
            status = "critical"
        elif block["overfit_count"] > 0 or block["underfit_count"] > block["count"] * 0.5:
            status = "warning"
        position_analysis.append({
            "key": block_id,
            "component": "UNet" if block_id.startswith(("IN", "OUT", "M")) else "TE",
            "block_type": "input" if block_id.startswith("IN") else ("output" if block_id.startswith("OUT") else "other"),
            "layer_count": block["count"],
            "avg_norm": _safe_round(avg_norm, 4),
            "avg_sparsity": _safe_round(avg_sparsity * 100.0, 2),
            "avg_rms": _safe_round(avg_rms, 6),
            "info_ratio": _safe_round(info_ratio, 2) if info_ratio is not None else None,
            "overfit_count": block["overfit_count"],
            "underfit_count": block["underfit_count"],
            "dead_count": block["dead_count"],
            "status": status,
        })
        if status != "good":
            recommendations.append({
                "type": status,
                "position": block_id,
                "issue": "权重异常" if status == "critical" else "训练强度不均",
                "suggestion": "建议检查学习率、训练步数和对应 block 权重；必要时用剪枝或重训修正。",
            })

    return {
        "file_name": path.name,
        "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
        "total_params": total_params,
        "num_layers": len(layers),
        "lora_type": lora_type,
        "layers": layers,
        "position_analysis": position_analysis,
        "recommendations": recommendations,
        "unsupported_dtypes": unsupported_dtypes,
    }


def block_analyze(file_path: str) -> Dict[str, Any]:
    report = analyze_lora(file_path, max_samples=0)
    blocks = []
    max_rms = max((item.get("avg_rms", 0.0) for item in report["position_analysis"]), default=0.0)
    for item in report["position_analysis"]:
        avg_rms = float(item.get("avg_rms", 0.0) or 0.0)
        blocks.append({
            "id": item["key"],
            "magnitude": avg_rms,
            "normalized_magnitude": (avg_rms / max_rms * 100.0) if max_rms > 0 else 0.0,
            "info_ratio": item.get("info_ratio"),
            "layer_count": item["layer_count"],
            "status": item["status"],
        })
    return {"blocks": blocks}

