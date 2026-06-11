from __future__ import annotations

from typing import Any, Mapping

from backend.core.contracts.base_model_tensorrt_runtime import (
    BaseModelTensorRtRuntimeMode,
    BaseModelTensorRtRuntimeRequest,
    BaseModelTensorRtRuntimeResult,
)
from core.base_model_tensorrt.newbie_export import NewbieStaticShape
from core.base_model_tensorrt.runtime_gate import build_static_transformer_runtime_gate
from core.base_model_tensorrt.runtime_smoke import run_newbie_static_tensorrt_runtime_smoke


def run_base_model_tensorrt_runtime_request(
    request: BaseModelTensorRtRuntimeRequest | Mapping[str, Any],
) -> BaseModelTensorRtRuntimeResult:
    req = request if isinstance(request, BaseModelTensorRtRuntimeRequest) else BaseModelTensorRtRuntimeRequest.model_validate(dict(request or {}))
    shape = _shape_from_request(req)
    if req.mode == BaseModelTensorRtRuntimeMode.SMOKE.value:
        report = run_newbie_static_tensorrt_runtime_smoke(
            engine_path=req.engine_path,
            layer_indices=req.layer_indices,
            shape=shape,
            device=req.device,
            dtype_name=req.dtype,
            precision=req.precision,
        )
        gate_report = dict(report.get("gate") or {}) if isinstance(report.get("gate"), dict) else report
        result = BaseModelTensorRtRuntimeResult.from_gate_report(gate_report, request=req)
        result.data["runtime_report"] = report
        result.data["mode"] = BaseModelTensorRtRuntimeMode.SMOKE.value
        return result

    report = build_static_transformer_runtime_gate(
        family=req.family,
        engine_path=req.engine_path,
        layer_indices=req.layer_indices,
        shape=shape,
        precision=req.precision,
    )
    result = BaseModelTensorRtRuntimeResult.from_gate_report(report, request=req)
    result.data["mode"] = BaseModelTensorRtRuntimeMode.GATE.value
    return result


def _shape_from_request(request: BaseModelTensorRtRuntimeRequest) -> NewbieStaticShape:
    data = dict(request.shape or {})
    return NewbieStaticShape(
        batch=int(data.get("batch") or 1),
        latent_channels=int(data.get("latent_channels") or 16),
        latent_height=int(data.get("latent_height") or 64),
        latent_width=int(data.get("latent_width") or 64),
        tokens=int(data.get("tokens") or 512),
        hidden_dim=int(data.get("hidden_dim") or 2304),
        pooled_dim=int(data.get("pooled_dim") or 1024),
        patch_size=int(data.get("patch_size") or 2),
    )


__all__ = ["run_base_model_tensorrt_runtime_request"]
