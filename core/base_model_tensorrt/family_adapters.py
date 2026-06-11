from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_ROOT = PROJECT_ROOT / "models"


@dataclass(frozen=True)
class FamilyComponentSpec:
    key: str
    label: str
    relative_path: str
    required: bool = True
    export_target: bool = False


@dataclass(frozen=True)
class FamilyAdapter:
    family: str
    label: str
    architecture: str
    default_model_root: Path
    export_component: str
    default_dtype: str
    default_static_shape: dict[str, int]
    components: tuple[FamilyComponentSpec, ...]
    notes: tuple[str, ...] = ()


ANIMA_ADAPTER = FamilyAdapter(
    family="anima",
    label="Anima DiT",
    architecture="dit-flow-qwen-image",
    default_model_root=MODELS_ROOT / "anima",
    export_component="transformer",
    default_dtype="bf16",
    default_static_shape={"batch": 1, "latent_channels": 16, "latent_height": 64, "latent_width": 64, "tokens": 512},
    components=(
        FamilyComponentSpec("transformer", "DiT checkpoint", "diffusion_models/anima-base-v1.0.safetensors", export_target=True),
        FamilyComponentSpec("text_encoder", "Qwen3 text encoder", "text_encoders/qwen_3_06b_base.safetensors"),
        FamilyComponentSpec("vae", "Qwen Image VAE", "vae/qwen_image_vae.safetensors"),
    ),
    notes=(
        "Anima uses a DiT/flow path, so the first TensorRT spike should target the transformer only.",
        "Prompt encoding and VAE remain PyTorch-side until transformer parity is established.",
    ),
)


NEWBIE_ADAPTER = FamilyAdapter(
    family="newbie",
    label="Newbie NextDiT",
    architecture="nextdit-transport-gemma-jina",
    default_model_root=MODELS_ROOT / "newbie",
    export_component="transformer",
    default_dtype="float32",
    default_static_shape={"batch": 1, "latent_channels": 16, "latent_height": 64, "latent_width": 64, "tokens": 512},
    components=(
        FamilyComponentSpec("transformer", "NextDiT checkpoint", "diffusion_pytorch_model.safetensors", export_target=True),
        FamilyComponentSpec("transformer_config", "Transformer config", "transformer/config.json", required=False),
        FamilyComponentSpec("text_encoder", "Gemma3 text encoder", "text_encoder/gemma3-4b-it.safetensors"),
        FamilyComponentSpec("clip_model", "Jina CLIP model", "clip_model/jina-clip-v2.safetensors"),
        FamilyComponentSpec("vae", "Newbie VAE", "vae/diffusion_pytorch_model.safetensors"),
    ),
    notes=(
        "Newbie has extra Gemma3 + Jina CLIP conditioning, so it should follow Anima after the common transformer wrapper is proven.",
        "The TensorRT spike should keep transport/scheduler logic outside the engine until transformer parity is stable.",
        "Newbie full 36-layer static transformer parity is proven on FP32 at 4x4/tok4.",
        "Newbie full 36-layer production static transformer parity is proven on FP32 at 64x64/tok512 via split/offline parity.",
        "Newbie FP16 sensitive mixed precision is validated through layers 0-7; sensitive_block_matmul is validated through layers 0-15 but is close to FP32 size.",
        "Newbie BF16 sensitive parity is validated through layers 0-11 and first fails in the tested 0-12 window.",
        "Newbie BF16 tap diagnostics show hidden-state drift before layer 12; local FP32 preservation did not restore parity.",
        "Keep FP32 as the safe full36 default until low-precision hidden-state drift is solved.",
    ),
)


ADAPTERS: dict[str, FamilyAdapter] = {
    ANIMA_ADAPTER.family: ANIMA_ADAPTER,
    NEWBIE_ADAPTER.family: NEWBIE_ADAPTER,
}


def get_family_adapter(family: str) -> FamilyAdapter:
    key = str(family or "anima").strip().lower()
    if key not in ADAPTERS:
        raise ValueError(f"unsupported base-model TensorRT family: {family}")
    return ADAPTERS[key]


def path_record(path: str | Path) -> dict[str, Any]:
    item = Path(path)
    exists = item.exists()
    kind = "directory" if item.is_dir() else "file" if item.is_file() else "missing"
    return {
        "path": str(item),
        "exists": exists,
        "kind": kind,
        "bytes": item.stat().st_size if item.is_file() else 0,
    }


def resolve_family_components(adapter: FamilyAdapter, model_root: str | Path = "") -> dict[str, Any]:
    root = Path(model_root) if str(model_root or "").strip() else adapter.default_model_root
    components: dict[str, Any] = {}
    missing_required: list[str] = []
    export_target_path = ""
    for spec in adapter.components:
        record = path_record(root / spec.relative_path)
        record.update({
            "key": spec.key,
            "label": spec.label,
            "required": spec.required,
            "export_target": spec.export_target,
        })
        components[spec.key] = record
        if spec.required and not record["exists"]:
            missing_required.append(spec.key)
        if spec.export_target:
            export_target_path = str(root / spec.relative_path)
    return {
        "model_root": str(root),
        "components": components,
        "missing_required": missing_required,
        "export_target_path": export_target_path,
    }


def build_export_plan(adapter: FamilyAdapter, resolved: dict[str, Any], *, output_dir: str = "") -> dict[str, Any]:
    out_root = Path(output_dir) if str(output_dir or "").strip() else MODELS_ROOT / adapter.family / "tensorrt_spike"
    onnx_path = out_root / f"{adapter.family}_{adapter.export_component}_static.onnx"
    precision = _default_engine_precision(adapter)
    engine_path = out_root / f"{adapter.family}_{adapter.export_component}_static_{precision}.engine"
    blocking: list[str] = []
    cautions: list[str] = []
    if resolved.get("missing_required"):
        blocking.append("missing_required_components")
    if adapter.family == "newbie":
        cautions.extend([
            "newbie_same_process_production_parity_requires_more_vram",
            "newbie_fp16_mixed_full36_parity_not_acceptable",
        ])
    return {
        "stage": "planning",
        "component": adapter.export_component,
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
        "dtype": adapter.default_dtype,
        "engine_precision": precision,
        "external_data_recommended": adapter.family == "newbie",
        "static_shape": dict(adapter.default_static_shape),
        "blocking": blocking,
        "cautions": cautions,
        "next_actions": _next_actions(adapter),
    }


def _default_engine_precision(adapter: FamilyAdapter) -> str:
    if adapter.family == "newbie":
        return "fp32"
    return "fp16"


def _next_actions(adapter: FamilyAdapter) -> list[str]:
    if adapter.family == "newbie":
        return [
            "keep_fp32_static_transformer_as_safe_path",
            "wire_static_transformer_runtime_behind_lab_gate",
            "keep_fp32_default_until_low_precision_hidden_state_drift_is_solved",
        ]
    return [
        "build_family_forward_wrapper",
        "export_static_onnx_with_synthetic_conditioning",
        "build_tensorrt_engine",
        "compare_torch_vs_tensorrt_outputs",
    ]


def inspect_family_checkpoint(adapter: FamilyAdapter, resolved: dict[str, Any]) -> dict[str, Any]:
    """Best-effort, shape-level checkpoint inspection for planning reports."""
    target = Path(str(resolved.get("export_target_path") or ""))
    if not target.is_file():
        return {"available": False, "reason": "export_target_missing"}
    if adapter.family == "anima":
        try:
            from core.lulynx_trainer.anima_native_dit import inspect_anima_safetensors

            info = inspect_anima_safetensors(target)
            return {
                "available": True,
                "kind": "anima_native_dit_introspection",
                "is_native_dit": bool(info.is_native_dit),
                "tensor_count": info.tensor_count,
                "block_count": info.block_count,
                "hidden_dim": info.hidden_dim,
                "latent_channels_hint": info.latent_channels_hint,
                "has_llm_adapter": info.has_llm_adapter,
                "has_x_embedder": info.has_x_embedder,
                "has_final_layer": info.has_final_layer,
                "limitations": list(info.limitations),
                "notes": list(info.notes),
            }
        except Exception as exc:
            return {"available": False, "reason": "anima_introspection_failed", "error": str(exc)}
    if adapter.family == "newbie":
        try:
            from core.base_model_tensorrt.newbie_export import inspect_newbie_safetensors

            config_path = resolved.get("components", {}).get("transformer_config", {}).get("path", "")
            return inspect_newbie_safetensors(target, config_path=config_path)
        except Exception as exc:
            return {"available": False, "reason": "newbie_introspection_failed", "error": str(exc)}
    return {
        "available": True,
        "kind": f"{adapter.family}_file_record",
        "bytes": target.stat().st_size,
        "notes": ["Deep checkpoint introspection is not enabled for this family yet."],
    }
