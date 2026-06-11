"""Static transformer TensorRT runtime adapter boundary.

The adapter is intentionally smaller than a generation runtime.  It only knows
how to validate the static transformer IO contract and call a TensorRT engine;
prompt encoding, sampling, scheduler/transport, and VAE stay outside this
module.
"""

from __future__ import annotations

from collections.abc import Mapping as MappingABC, Sequence as SequenceABC
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .newbie_export import NewbieStaticShape, parse_layer_indices
from .static_engine import StaticTensorRtEngine


STATIC_TRANSFORMER_INPUT_NAMES = ("sample", "timestep", "encoder_hidden_states", "text_embeds")
STATIC_TRANSFORMER_OUTPUT_NAME = "sample_out"

EngineFactory = Callable[[str | Path], Any]


@dataclass(frozen=True)
class StaticTransformerRuntimeSpec:
    family: str
    component: str
    engine_path: str
    layer_indices: tuple[int, ...]
    shape: Mapping[str, int]
    precision: str = "fp32"
    input_signature: Mapping[str, Any] | None = None
    output_name: str = STATIC_TRANSFORMER_OUTPUT_NAME
    generation_path_enabled: bool = False
    training_path_enabled: bool = False

    @classmethod
    def from_newbie_shape(
        cls,
        *,
        engine_path: str | Path,
        layer_indices: str | Sequence[int] = (0,),
        shape: NewbieStaticShape | Mapping[str, Any] | None = None,
        precision: str = "fp32",
    ) -> "StaticTransformerRuntimeSpec":
        static_shape = shape if isinstance(shape, NewbieStaticShape) else _newbie_shape_from_mapping(shape)
        return cls(
            family="newbie",
            component="transformer",
            engine_path=str(engine_path),
            layer_indices=parse_layer_indices(layer_indices),
            shape=static_shape.to_dict(),
            precision=_normalize_precision(precision),
            input_signature=static_shape.input_signature(),
        )

    @classmethod
    def from_gate_report(cls, report: Mapping[str, Any]) -> "StaticTransformerRuntimeSpec":
        shape = _newbie_shape_from_mapping(report.get("shape") if isinstance(report, Mapping) else {})
        return cls(
            family=str(report.get("family") or "newbie"),
            component=str(report.get("component") or "transformer"),
            engine_path=str(report.get("engine_path") or ""),
            layer_indices=parse_layer_indices(report.get("layer_indices") or (0,)),
            shape=shape.to_dict(),
            precision=_normalize_precision(str(report.get("precision") or "fp32")),
            input_signature=dict(report.get("input_signature") or shape.input_signature()),
            generation_path_enabled=False,
            training_path_enabled=False,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "component": self.component,
            "engine_path": self.engine_path,
            "layer_indices": list(self.layer_indices),
            "shape": dict(self.shape),
            "precision": self.precision,
            "input_signature": dict(self.input_signature or {}),
            "output_name": self.output_name,
            "generation_path_enabled": False,
            "training_path_enabled": False,
        }


class StaticTransformerTensorRtRuntime:
    """Thin runtime wrapper for static transformer TensorRT engines."""

    def __init__(
        self,
        spec: StaticTransformerRuntimeSpec,
        *,
        engine_factory: EngineFactory | None = None,
    ) -> None:
        self.spec = spec
        self._engine_factory = engine_factory or StaticTensorRtEngine
        self._engine: Any | None = None

    def infer(self, inputs: Mapping[str, Any]) -> Any:
        outputs = self.infer_outputs(inputs)
        if self.spec.output_name not in outputs:
            raise RuntimeError(f"TensorRT engine did not return {self.spec.output_name}, got {list(outputs)}")
        return outputs[self.spec.output_name]

    def infer_outputs(self, inputs: Mapping[str, Any]) -> Mapping[str, Any]:
        issues = validate_static_transformer_inputs(inputs, self.spec)
        if issues:
            raise ValueError("Invalid static transformer TensorRT inputs: " + "; ".join(issues))
        return self._load_engine().infer({name: inputs[name] for name in STATIC_TRANSFORMER_INPUT_NAMES})

    def _load_engine(self) -> Any:
        if self._engine is None:
            self._engine = self._engine_factory(self.spec.engine_path)
        return self._engine


def validate_static_transformer_inputs(
    inputs: Mapping[str, Any],
    spec: StaticTransformerRuntimeSpec,
) -> list[str]:
    issues: list[str] = []
    for name in STATIC_TRANSFORMER_INPUT_NAMES:
        if name not in inputs:
            issues.append(f"missing_input:{name}")
            continue
        expected = _shape_for_input(spec, name)
        actual = _tensor_shape(inputs[name])
        if expected and actual and list(actual) != list(expected):
            issues.append(f"shape_mismatch:{name}:expected={expected}:actual={list(actual)}")
    return issues


def _shape_for_input(spec: StaticTransformerRuntimeSpec, name: str) -> list[int]:
    signature = spec.input_signature if isinstance(spec.input_signature, MappingABC) else {}
    value = signature.get(name) if isinstance(signature, MappingABC) else None
    if isinstance(value, SequenceABC) and not isinstance(value, (str, bytes)):
        return [int(item) for item in value]
    return []


def _tensor_shape(value: Any) -> list[int]:
    shape = getattr(value, "shape", None)
    if shape is None:
        return []
    try:
        return [int(item) for item in tuple(shape)]
    except Exception:
        return []


def _newbie_shape_from_mapping(value: Any) -> NewbieStaticShape:
    data = dict(value or {}) if isinstance(value, MappingABC) else {}
    shape = NewbieStaticShape(
        batch=int(data.get("batch") or 1),
        latent_channels=int(data.get("latent_channels") or 16),
        latent_height=int(data.get("latent_height") or 64),
        latent_width=int(data.get("latent_width") or 64),
        tokens=int(data.get("tokens") or 512),
        hidden_dim=int(data.get("hidden_dim") or 2304),
        pooled_dim=int(data.get("pooled_dim") or 1024),
        patch_size=int(data.get("patch_size") or 2),
    )
    shape.validate()
    return shape


def _normalize_precision(value: str | None) -> str:
    key = str(value or "fp32").strip().lower().replace("-", "_")
    return {"float32": "fp32", "float16": "fp16", "half": "fp16", "bfloat16": "bf16"}.get(key, key)


__all__ = [
    "STATIC_TRANSFORMER_INPUT_NAMES",
    "STATIC_TRANSFORMER_OUTPUT_NAME",
    "StaticTransformerRuntimeSpec",
    "StaticTransformerTensorRtRuntime",
    "validate_static_transformer_inputs",
]
