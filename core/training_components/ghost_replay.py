"""
Ghost Replay (幽灵重放) - 离线锚点特征蒸馏

在 0 显存占用下实现 Teacher-Student 知识传递。
利用 J-L 随机投影将特征压缩后存储到硬盘。

流程:
1. 录制阶段 (Recorder): 使用 Teacher 模型生成特征指纹
2. 重放阶段 (Replayer): 训练时加载指纹并与 Student 比对
"""

from __future__ import annotations

import gzip
import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("GhostReplay")

_FORMAT_VERSION = 2
_RESERVED_SAMPLE_KEYS = {
    "args",
    "kwargs",
    "forward_fn",
    "seed",
    "timestep_key",
    "metadata",
}


def _feature_tensor(output: Any) -> Optional[torch.Tensor]:
    if isinstance(output, torch.Tensor):
        return output
    sample = getattr(output, "sample", None)
    if isinstance(sample, torch.Tensor):
        return sample
    if isinstance(output, (tuple, list)):
        for item in output:
            tensor = _feature_tensor(item)
            if tensor is not None:
                return tensor
    if isinstance(output, dict):
        for item in output.values():
            tensor = _feature_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _normalize_feature_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor.unsqueeze(1)
    if tensor.dim() == 3:
        return tensor
    if tensor.dim() == 4:
        return tensor.permute(0, 2, 3, 1).reshape(tensor.shape[0], -1, tensor.shape[1])
    if tensor.dim() > 4:
        return tensor.reshape(tensor.shape[0], -1, tensor.shape[-1])
    return tensor.reshape(1, 1, -1)


def _coerce_batch_size(sample: Any) -> int:
    tensors: List[torch.Tensor] = []
    if isinstance(sample, dict):
        for value in sample.values():
            if isinstance(value, torch.Tensor):
                tensors.append(value)
            elif isinstance(value, (list, tuple)):
                tensors.extend(item for item in value if isinstance(item, torch.Tensor))
    elif isinstance(sample, (list, tuple)):
        tensors.extend(item for item in sample if isinstance(item, torch.Tensor))
    elif isinstance(sample, torch.Tensor):
        tensors.append(sample)

    for tensor in tensors:
        if tensor.dim() >= 1 and int(tensor.shape[0]) > 0:
            return int(tensor.shape[0])
    return 1


def _build_timestep_value(sample: Any, timestep: int, device: torch.device) -> Any:
    batch_size = max(_coerce_batch_size(sample), 1)
    return torch.full((batch_size,), int(timestep), device=device, dtype=torch.long)


def _safe_int_list(values: Any) -> List[int]:
    result: List[int] = []
    for value in values if isinstance(values, (list, tuple)) else []:
        try:
            result.append(int(value))
        except Exception:
            continue
    return sorted(set(result))


def _safe_str_list(values: Any) -> List[str]:
    result: List[str] = []
    for value in values if isinstance(values, (list, tuple)) else []:
        item = str(value or "").strip()
        if item:
            result.append(item)
    return sorted(set(result))


def _metadata_arch(metadata: Mapping[str, Any]) -> str:
    return str(
        metadata.get("model_arch")
        or metadata.get("model_family")
        or metadata.get("architecture")
        or ""
    ).strip().lower()


def inspect_ghost_fingerprint(path: str | Path) -> Dict[str, Any]:
    path_obj = Path(path)
    report: Dict[str, Any] = {
        "path": str(path_obj),
        "exists": path_obj.exists(),
        "readable": False,
        "status": "error",
        "errors": [],
        "warnings": [],
        "metadata": {},
        "recorded_layers": [],
        "recorded_layer_count": 0,
        "projection_input_dims": [],
        "timesteps": [],
        "num_samples": 0,
        "proj_dim": 0,
        "format_version": 0,
    }
    if not path_obj.exists():
        report["errors"].append("fingerprint file does not exist")
        return report

    try:
        with gzip.open(path_obj, "rt", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        report["errors"].append(f"failed to read fingerprint: {exc}")
        return report

    report.update(_inspect_ghost_payload(data, path=str(path_obj)))
    report["exists"] = True
    report["readable"] = True
    return report


def _inspect_ghost_payload(data: Mapping[str, Any], *, path: str = "") -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
        warnings.append("metadata section is missing or invalid")

    projection_matrices = data.get("projection_matrices", {})
    if not isinstance(projection_matrices, dict):
        projection_matrices = {}
        errors.append("projection_matrices section is missing or invalid")

    fingerprints = data.get("fingerprints", {})
    if not isinstance(fingerprints, dict):
        fingerprints = {}
        errors.append("fingerprints section is missing or invalid")

    recorded_layers = sorted(str(name) for name in fingerprints.keys() if str(name).strip())
    if not recorded_layers:
        errors.append("fingerprint contains no recorded layers")

    projection_input_dims: List[int] = []
    for key in projection_matrices.keys():
        try:
            projection_input_dims.append(int(key))
        except Exception:
            warnings.append(f"projection matrix key is not an integer dimension: {key}")
    projection_input_dims = sorted(set(projection_input_dims))
    if not projection_input_dims:
        errors.append("fingerprint contains no usable projection matrices")

    discovered_timesteps: List[int] = []
    per_layer_counts: Dict[str, int] = {}
    for layer_name, timestep_map in fingerprints.items():
        if not isinstance(timestep_map, dict):
            warnings.append(f"layer {layer_name} has invalid timestep map")
            continue
        sample_count = 0
        for raw_timestep, cached_list in timestep_map.items():
            try:
                discovered_timesteps.append(int(raw_timestep))
            except Exception:
                warnings.append(f"layer {layer_name} has non-integer timestep key: {raw_timestep}")
            if isinstance(cached_list, list):
                sample_count = max(sample_count, len(cached_list))
            else:
                warnings.append(f"layer {layer_name} timestep {raw_timestep} cache is not a list")
        per_layer_counts[str(layer_name)] = sample_count

    timesteps = sorted(set(discovered_timesteps or _safe_int_list(metadata.get("timesteps", []))))
    if not timesteps:
        warnings.append("fingerprint metadata does not list any timesteps")

    num_samples = max(
        int(metadata.get("num_samples", 0) or 0),
        max(per_layer_counts.values(), default=0),
    )
    if num_samples <= 0:
        warnings.append("fingerprint metadata does not list any samples")

    format_version = int(metadata.get("format_version", 1) or 1)
    if format_version < _FORMAT_VERSION:
        warnings.append(
            f"fingerprint format v{format_version} predates the current validator v{_FORMAT_VERSION}"
        )

    status = "error" if errors else "warning" if warnings else "ok"
    return {
        "path": path,
        "status": status,
        "errors": errors,
        "warnings": warnings,
        "metadata": metadata,
        "recorded_layers": recorded_layers,
        "recorded_layer_count": len(recorded_layers),
        "projection_input_dims": projection_input_dims,
        "timesteps": timesteps,
        "num_samples": num_samples,
        "proj_dim": int(metadata.get("proj_dim", 0) or 0),
        "format_version": format_version,
        "per_layer_sample_count": per_layer_counts,
    }


class GhostRecorder:
    """
    幽灵录制器 - 预先计算并压缩 Teacher 特征指纹

    使用方式:
        recorder = GhostRecorder(proj_dim=128)
        recorder.record(model, prompts, timesteps=[200, 800], forward_fn=...)
        recorder.save("fingerprint.lulynx")
    """

    def __init__(
        self,
        proj_dim: int = 128,
        target_layers: Optional[List[str]] = None,
        device: str = "cuda",
    ):
        self.proj_dim = proj_dim
        self.target_layers = target_layers
        self.device = device
        self._record_lock = threading.Lock()
        self._projection_matrices: Dict[int, np.ndarray] = {}
        self._fingerprints: Dict[str, Dict[int, List[np.ndarray]]] = {}
        self._metadata: Dict[str, Any] = {
            "format_version": _FORMAT_VERSION,
            "proj_dim": proj_dim,
            "target_layers": target_layers,
            "timesteps": [],
            "num_samples": 0,
            "record_device": str(device),
            "captured_layers": [],
            "projection_input_dims": [],
        }

    def _get_projection_matrix(self, input_dim: int) -> np.ndarray:
        if input_dim not in self._projection_matrices:
            H = np.random.randn(input_dim, self.proj_dim).astype(np.float32)
            try:
                Q, _ = np.linalg.qr(H)
            except np.linalg.LinAlgError:
                Q = H / (np.linalg.norm(H, axis=0) + 1e-6)
            R = Q[:, : self.proj_dim].T
            self._projection_matrices[input_dim] = R
        return self._projection_matrices[input_dim]

    def _should_track_layer(self, name: str) -> bool:
        if self.target_layers is not None:
            return any(target in name for target in self.target_layers)
        name_lower = name.lower()
        if "mid_block" in name_lower:
            return True
        if "double" in name_lower and any(f".{i}." in name for i in [4, 10]):
            return True
        return False

    def _invoke_forward(
        self,
        model: nn.Module,
        sample: Any,
        *,
        timestep: int,
        sample_idx: int,
        forward_fn: Optional[Callable[..., Any]] = None,
    ) -> Any:
        callback = forward_fn
        if callback is None and isinstance(sample, dict):
            candidate = sample.get("forward_fn")
            if callable(candidate):
                callback = candidate
        if callable(callback):
            return callback(
                model=model,
                sample=sample,
                timestep=int(timestep),
                sample_idx=int(sample_idx),
                device=str(self.device),
            )

        if isinstance(sample, Mapping):
            args = sample.get("args", ())
            kwargs = dict(sample.get("kwargs", {}) or {})
            for key, value in sample.items():
                if key in _RESERVED_SAMPLE_KEYS:
                    continue
                kwargs.setdefault(key, value)

            timestep_key = str(sample.get("timestep_key", "") or "").strip()
            if timestep_key:
                kwargs.setdefault(
                    timestep_key,
                    _build_timestep_value(sample, int(timestep), torch.device(self.device)),
                )
            elif "timestep" not in kwargs and "timesteps" not in kwargs:
                kwargs["timestep"] = _build_timestep_value(
                    sample,
                    int(timestep),
                    torch.device(self.device),
                )

            if not isinstance(args, (list, tuple)):
                args = (args,)
            return model(*tuple(args), **kwargs)

        if isinstance(sample, (list, tuple)):
            return model(*sample)
        if sample is None:
            return model(timestep=int(timestep))
        return model(sample)

    def record(
        self,
        model: nn.Module,
        sample_inputs: List[Dict[str, Any]],
        timesteps: List[int] = [200, 800],
        seeds: Optional[List[int]] = None,
        *,
        forward_fn: Optional[Callable[..., Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        with self._record_lock:
            return self._record_internal(
                model,
                sample_inputs,
                timesteps,
                seeds,
                forward_fn=forward_fn,
                metadata=metadata,
                strict=strict,
            )

    def _record_internal(
        self,
        model: nn.Module,
        sample_inputs: List[Dict[str, Any]],
        timesteps: List[int] = [200, 800],
        seeds: Optional[List[int]] = None,
        *,
        forward_fn: Optional[Callable[..., Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        strict: bool = True,
    ) -> Dict[str, Any]:
        self._fingerprints.clear()
        self._projection_matrices.clear()

        normalized_timesteps = [int(value) for value in timesteps]
        self._metadata = {
            "format_version": _FORMAT_VERSION,
            "proj_dim": int(self.proj_dim),
            "target_layers": list(self.target_layers or []),
            "timesteps": normalized_timesteps,
            "num_samples": len(sample_inputs),
            "record_device": str(self.device),
            "model_class": model.__class__.__name__,
            "model_module": model.__class__.__module__,
            "captured_layers": [],
            "projection_input_dims": [],
        }
        if isinstance(metadata, dict):
            for key, value in metadata.items():
                if value is not None:
                    self._metadata[key] = value

        hooks = []
        features_buffer: Dict[str, List[torch.Tensor]] = {}
        captured_steps = 0
        missed_steps = 0

        def create_hook(name: str):
            def hook(_module: nn.Module, _inputs: Any, output: Any) -> None:
                tensor = _feature_tensor(output)
                if tensor is None or tensor.dim() < 2:
                    return
                features_buffer.setdefault(name, []).append(
                    _normalize_feature_tensor(tensor.detach()).cpu()
                )

            return hook

        for name, module in model.named_modules():
            if self._should_track_layer(name):
                hooks.append(module.register_forward_hook(create_hook(name)))

        try:
            for sample_idx, sample in enumerate(sample_inputs):
                if isinstance(sample, dict):
                    seed = sample.get("seed", None)
                else:
                    seed = None
                if seed is None and seeds and sample_idx < len(seeds):
                    seed = seeds[sample_idx]
                if seed is None:
                    seed = sample_idx * 42
                try:
                    seed_value = int(seed)
                except Exception:
                    seed_value = sample_idx * 42
                torch.manual_seed(seed_value)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed_value)

                for timestep in normalized_timesteps:
                    features_buffer.clear()
                    with torch.no_grad():
                        self._invoke_forward(
                            model,
                            sample,
                            timestep=timestep,
                            sample_idx=sample_idx,
                            forward_fn=forward_fn,
                        )

                    if not features_buffer:
                        missed_steps += 1
                        message = (
                            f"GhostRecorder captured no features for sample={sample_idx} timestep={timestep}. "
                            "Provide a compatible forward_fn or sample payload."
                        )
                        if strict:
                            raise RuntimeError(message)
                        logger.warning(message)
                        continue

                    captured_steps += 1
                    for name, feat_list in features_buffer.items():
                        layer_store = self._fingerprints.setdefault(name, {})
                        cached_list = layer_store.setdefault(int(timestep), [])
                        for feat in feat_list:
                            input_dim = int(feat.shape[-1])
                            R = self._get_projection_matrix(input_dim)
                            flat = feat.numpy().reshape(-1, input_dim)
                            projected = np.dot(flat, R.T)
                            cached_list.append(projected.astype(np.float16))
        finally:
            for hook in hooks:
                try:
                    hook.remove()
                except Exception:
                    pass

        captured_layers = sorted(self._fingerprints.keys())
        self._metadata["captured_layers"] = captured_layers
        self._metadata["projection_input_dims"] = sorted(self._projection_matrices.keys())

        total_size = sum(
            sum(arr.nbytes for arrs in timestep_map.values() for arr in arrs)
            for timestep_map in self._fingerprints.values()
        )

        stats = {
            "layers": len(self._fingerprints),
            "samples": len(sample_inputs),
            "timesteps": len(normalized_timesteps),
            "captured_steps": captured_steps,
            "missed_steps": missed_steps,
            "size_mb": total_size / (1024 * 1024),
            "metadata": dict(self._metadata),
        }
        if strict and not self._fingerprints:
            raise RuntimeError("GhostRecorder did not capture any fingerprint data")
        return stats

    def save(self, path: str) -> None:
        path_obj = Path(path)
        data = {
            "metadata": self._metadata,
            "projection_matrices": {
                str(k): v.tolist() for k, v in self._projection_matrices.items()
            },
            "fingerprints": {
                layer: {
                    str(t): [arr.tolist() for arr in arrs]
                    for t, arrs in timesteps.items()
                }
                for layer, timesteps in self._fingerprints.items()
            },
        }
        with gzip.open(path_obj, "wt", encoding="utf-8") as f:
            json.dump(data, f)


class GhostReplayer:
    """
    幽灵重放器 - 训练时加载指纹并计算损失

    使用方式:
        replayer = GhostReplayer.load("fingerprint.lulynx")
        loss = replayer.compute_loss(current_features)
    """

    def __init__(self):
        self._projection_matrices: Dict[int, np.ndarray] = {}
        self._fingerprints: Dict[str, Dict[int, List[np.ndarray]]] = {}
        self._metadata: Dict[str, Any] = {}
        self.device = "cuda"
        self.last_result: Dict[str, Any] = {}
        self._load_report: Dict[str, Any] = {}
        self._compatibility_report: Dict[str, Any] = {}

    @classmethod
    def load(cls, path: str) -> "GhostReplayer":
        replayer = cls()
        with gzip.open(path, "rt", encoding="utf-8") as f:
            data = json.load(f)

        replayer._metadata = data.get("metadata", {})
        replayer._projection_matrices = {
            int(k): np.array(v, dtype=np.float32)
            for k, v in dict(data.get("projection_matrices", {})).items()
        }
        replayer._fingerprints = {
            str(layer): {
                int(t): [np.array(arr, dtype=np.float16) for arr in arrs]
                for t, arrs in dict(timesteps).items()
            }
            for layer, timesteps in dict(data.get("fingerprints", {})).items()
        }
        replayer._load_report = _inspect_ghost_payload(data, path=str(path))
        return replayer

    @classmethod
    def inspect_file(cls, path: str | Path) -> Dict[str, Any]:
        return inspect_ghost_fingerprint(path)

    def inspect(self) -> Dict[str, Any]:
        return dict(self._load_report)

    def validate_against_model(
        self,
        model: Optional[nn.Module],
        *,
        model_arch: str = "",
        anchor_layers: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        base = dict(self._load_report or {})
        warnings = list(base.get("warnings", []))
        errors = list(base.get("errors", []))
        recorded_layers = list(base.get("recorded_layers", []))
        matched_layers: List[str] = []
        missing_layers: List[str] = []

        fingerprint_arch = _metadata_arch(self._metadata)
        normalized_arch = str(model_arch or "").strip().lower()
        if normalized_arch and fingerprint_arch and normalized_arch != fingerprint_arch:
            warnings.append(
                f"fingerprint was recorded for model_arch={fingerprint_arch}, current route is {normalized_arch}"
            )

        available_layers: List[str] = []
        if model is not None:
            available_layers = [name for name, _module in model.named_modules() if name]
            layer_set = set(available_layers)
            for layer_name in recorded_layers:
                if layer_name in layer_set:
                    matched_layers.append(layer_name)
                else:
                    missing_layers.append(layer_name)
            if recorded_layers and not matched_layers:
                errors.append("no recorded fingerprint layers were found on the current model")
            elif missing_layers:
                warnings.append(
                    f"{len(missing_layers)} recorded fingerprint layer(s) are missing on the current model"
                )

        normalized_anchors = [
            str(token).strip() for token in (anchor_layers or []) if str(token).strip()
        ]
        if normalized_anchors and recorded_layers:
            anchor_matches = [
                layer_name
                for layer_name in recorded_layers
                if any(token in layer_name for token in normalized_anchors)
            ]
            if not anchor_matches:
                warnings.append("recorded layers do not match the configured anchor layer filters")

        report = {
            **base,
            "status": "error" if errors else "warning" if warnings else "ok",
            "errors": errors,
            "warnings": warnings,
            "fingerprint_model_arch": fingerprint_arch,
            "requested_model_arch": normalized_arch,
            "matched_layers": matched_layers,
            "matched_layer_count": len(matched_layers),
            "missing_layers": missing_layers,
            "available_layer_count": len(available_layers),
            "usable": not errors,
        }
        self._compatibility_report = report
        return dict(report)

    def compute_loss(
        self,
        current_features: Dict[str, torch.Tensor],
        timestep: int,
        sample_idx: int = 0,
        weight: float = 1.0,
    ) -> torch.Tensor:
        total_loss = torch.tensor(0.0, device=self.device)
        matched_layers = 0
        skipped_missing_layer = 0
        skipped_missing_timestep = 0
        skipped_missing_sample = 0
        skipped_missing_projection = 0
        matched_layer_names: List[str] = []

        for name, feat in current_features.items():
            if name not in self._fingerprints:
                skipped_missing_layer += 1
                continue
            if timestep not in self._fingerprints[name]:
                skipped_missing_timestep += 1
                continue

            cached_list = self._fingerprints[name][timestep]
            if sample_idx >= len(cached_list):
                skipped_missing_sample += 1
                continue

            input_dim = int(feat.shape[-1])
            if input_dim not in self._projection_matrices:
                skipped_missing_projection += 1
                continue

            cached = torch.from_numpy(cached_list[sample_idx].astype(np.float32)).to(self.device)
            R = torch.from_numpy(self._projection_matrices[input_dim]).to(self.device)
            flat = feat.reshape(-1, input_dim)
            projected = torch.mm(flat, R.t())
            loss = ((projected - cached) ** 2).mean()
            total_loss = total_loss + loss
            matched_layers += 1
            matched_layer_names.append(str(name))

        if matched_layers > 0:
            total_loss = total_loss / matched_layers

        weighted = weight * total_loss
        self.last_result = {
            "matched": matched_layers > 0,
            "matched_layers": matched_layers,
            "matched_layer_names": matched_layer_names,
            "skipped_missing_layer": skipped_missing_layer,
            "skipped_missing_timestep": skipped_missing_timestep,
            "skipped_missing_sample": skipped_missing_sample,
            "skipped_missing_projection": skipped_missing_projection,
            "timestep": int(timestep),
            "sample_idx": int(sample_idx),
            "weight": float(weight),
            "loss": float(weighted.detach().float().item()) if torch.isfinite(weighted).all() else None,
        }
        return weighted

    @property
    def info(self) -> Dict[str, Any]:
        return {
            "proj_dim": self._metadata.get("proj_dim", 128),
            "timesteps": self._metadata.get("timesteps", []),
            "num_samples": self._metadata.get("num_samples", 0),
            "num_layers": len(self._fingerprints),
            "captured_layers": self._metadata.get("captured_layers", []),
            "format_version": self._metadata.get("format_version", 1),
            "model_arch": _metadata_arch(self._metadata),
            "validation": dict(self._compatibility_report or self._load_report or {}),
        }
