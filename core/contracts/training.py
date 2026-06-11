"""Training request contracts."""

from __future__ import annotations

from typing import Any, Dict, List

from pydantic import ConfigDict, Field, field_validator, model_validator

from .base import BaseRequest


_SCHEMA_ALIASES = {
    "sdxl_lora": "sdxl-lora",
    "sd_lora": "sd-lora",
    "anima_lora": "anima-lora",
    "newbie_lora": "newbie-lora",
    "flux_lora": "flux-lora",
    "flux_finetune": "flux-finetune",
    "flux_controlnet": "flux-controlnet",
    "sdxl_finetune": "sdxl-finetune",
    "sdxl_dreambooth": "sdxl-dreambooth",
    "anima_finetune": "anima-finetune",
    "sd_dreambooth": "sd-dreambooth",
    "sdxl_controlnet_lllite": "sdxl-controlnet-lllite",
    "sdxl_controlnet": "sdxl-controlnet",
    "sd_controlnet": "sd-controlnet",
    "sdxl_textual_inversion": "sdxl-textual-inversion",
    "sd_textual_inversion": "sd-textual-inversion",
    "sdxl_ip_adapter": "sdxl-ip-adapter",
    "sd_ip_adapter": "sd-ip-adapter",
    "sdxl_turbo_lora": "sdxl-turbo-lora",
    "lab_distiller": "lab-distiller",
    "anima_few_step_lora": "anima-few-step-lora",
    "newbie_few_step_lora": "newbie-few-step-lora",
}

_MODEL_TYPE_ALIASES = {
    "sd": "sd15",
    "sd1.5": "sd15",
    "sd-1.5": "sd15",
    "stable-diffusion-xl": "sdxl",
}

_ATTENTION_ALIASES = {
    "flash": "flash2",
    "flashattn": "flash2",
    "flashattention": "flash2",
    "flashattention2": "flash2",
    "fa2": "flash2",
    "sage": "sageattn",
    "sageattention": "sageattn",
    "flex": "flexattn",
    "flexattention": "flexattn",
}

_SCHEMA_ROUTE_DEFAULTS = {
    "sdxl-lora": ("sdxl", "lora"),
    "sd-lora": ("sd15", "lora"),
    "anima-lora": ("anima", "lora"),
    "newbie-lora": ("newbie", "lora"),
    "flux-lora": ("flux", "lora"),
    "lumina-lora": ("lumina", "lora"),
    "lumina2-lora": ("lumina", "lora"),
    "qwen-image-lora": ("qwen_image", "lora"),
    "hunyuan-dit-lora": ("hunyuan_dit", "lora"),
    "hunyuan-image-lora": ("hunyuan_dit", "lora"),
    "flux-finetune": ("flux", "full_finetune"),
    "flux-controlnet": ("flux", "controlnet"),
    "sdxl-finetune": ("sdxl", "full_finetune"),
    "anima-finetune": ("anima", "full_finetune"),
    "sdxl-dreambooth": ("sdxl", "dreambooth"),
    "sd-dreambooth": ("sd15", "dreambooth"),
    "sdxl-controlnet": ("sdxl", "controlnet"),
    "sd-controlnet": ("sd15", "controlnet"),
    "sdxl-controlnet-lllite": ("sdxl", "lllite"),
    "sdxl-textual-inversion": ("sdxl", "textual_inversion"),
    "sd-textual-inversion": ("sd15", "textual_inversion"),
    "sdxl-ip-adapter": ("sdxl", "ip-adapter"),
    "sd-ip-adapter": ("sd15", "ip-adapter"),
}

LEGACY_TRAINING_ALIASES = {
    "pretrainedModelPath": "pretrained_model_name_or_path",
    "checkpointPath": "pretrained_model_name_or_path",
    "trainDataDir": "train_data_dir",
    "outputDir": "output_dir",
    "outputName": "output_name",
    "executionProfileId": "execution_profile_id",
    "attentionBackend": "attention_backend",
    "modelTrainType": "schema_id",
    "model_train_type": "schema_id",
    "training_schema": "schema_id",
    "training_type_id": "schema_id",
}

_SCHEMA_ALIASES.update(
    {
        "lumina_lora": "lumina-lora",
        "lumina2_lora": "lumina2-lora",
        "lumina_finetune": "lumina-finetune",
        "qwen_image_lora": "qwen-image-lora",
        "hunyuan_dit_lora": "hunyuan-dit-lora",
        "hunyuan_image_lora": "hunyuan-image-lora",
    }
)


class TrainingRequest(BaseRequest):
    """Canonical request boundary for training launch/preflight payloads."""

    model_config = ConfigDict(extra="allow", use_enum_values=True)

    schema_id: str = ""
    model_type: str = "sdxl"
    training_type: str = "lora"
    pretrained_model_name_or_path: str = ""
    train_data_dir: str = ""
    output_dir: str = ""
    output_name: str = "lora"
    network_module: str = "networks.lora"
    network_dim: int = 32
    network_alpha: float = 16.0
    execution_profile_id: str = ""
    attention_backend: str = "auto"
    allow_attention_fallback: bool = True
    config: Dict[str, Any] = Field(default_factory=dict)
    extra_config_layers: List[Dict[str, Any]] = Field(default_factory=list)
    dry_run: bool = False

    @classmethod
    def from_legacy_payload(
        cls,
        payload: Dict[str, Any],
        *,
        source: str = "unknown",
        compat_mode: bool = True,
    ) -> "TrainingRequest":
        data = dict(payload or {})
        nested = data.get("config") if isinstance(data.get("config"), dict) else {}
        merged = {**nested, **{k: v for k, v in data.items() if k != "config"}}
        if nested and "config" not in merged:
            merged["config"] = dict(nested)
        for old, new in LEGACY_TRAINING_ALIASES.items():
            if old not in merged:
                continue
            current = merged.get(new)
            if new not in merged or current in {None, ""} or (new == "attention_backend" and str(current).lower() == "auto"):
                merged[new] = merged[old]
        return super().from_legacy_payload(merged, source=source, compat_mode=compat_mode)  # type: ignore[return-value]

    @field_validator("schema_id")
    @classmethod
    def _normalize_schema_id(cls, value: str) -> str:
        value = str(value or "").strip().lower().replace("_", "-")
        return _SCHEMA_ALIASES.get(value.replace("-", "_"), value)

    @field_validator("model_type")
    @classmethod
    def _normalize_model_type(cls, value: str) -> str:
        value = str(value or "sdxl").strip().lower()
        return _MODEL_TYPE_ALIASES.get(value, value)

    @field_validator("training_type")
    @classmethod
    def _normalize_training_type(cls, value: str) -> str:
        return str(value or "lora").strip().lower().replace("_", "-")

    @field_validator("attention_backend")
    @classmethod
    def _normalize_attention_backend(cls, value: str) -> str:
        value = str(value or "auto").strip().lower()
        return _ATTENTION_ALIASES.get(value, value or "auto")

    @field_validator("network_dim")
    @classmethod
    def _positive_network_dim(cls, value: int) -> int:
        value = int(value)
        if value < 1:
            raise ValueError("network_dim must be >= 1")
        return value

    @field_validator("network_alpha")
    @classmethod
    def _positive_network_alpha(cls, value: float) -> float:
        value = float(value)
        if value <= 0:
            raise ValueError("network_alpha must be > 0")
        return value

    @model_validator(mode="after")
    def _derive_schema_when_missing(self) -> "TrainingRequest":
        if not self.schema_id:
            if self.training_type in {"lora", "network"}:
                mapping = {
                    "sdxl": "sdxl-lora",
                    "sd15": "sd-lora",
                    "anima": "anima-lora",
                    "newbie": "newbie-lora",
                    "flux": "flux-lora",
                    "lumina": "lumina-lora",
                    "lumina2": "lumina2-lora",
                    "qwen_image": "qwen-image-lora",
                    "qwen-image": "qwen-image-lora",
                    "hunyuan_dit": "hunyuan-dit-lora",
                    "hunyuan-dit": "hunyuan-dit-lora",
                    "hunyuan_image": "hunyuan-image-lora",
                    "hunyuan-image": "hunyuan-image-lora",
                }
                self.schema_id = mapping.get(self.model_type, "")
            elif self.training_type in {"full-finetune", "finetune"}:
                self.schema_id = {"sdxl": "sdxl-finetune", "anima": "anima-finetune", "flux": "flux-finetune"}.get(self.model_type, "")
            elif self.training_type == "dreambooth":
                self.schema_id = {"sdxl": "sdxl-dreambooth", "sd15": "sd-dreambooth"}.get(self.model_type, "")
            elif self.training_type == "controlnet":
                self.schema_id = {"sdxl": "sdxl-controlnet", "sd15": "sd-controlnet", "flux": "flux-controlnet"}.get(self.model_type, "")
            elif self.training_type == "textual-inversion":
                self.schema_id = {"sdxl": "sdxl-textual-inversion", "sd15": "sd-textual-inversion"}.get(self.model_type, "")
            elif self.training_type == "ip-adapter":
                self.schema_id = {"sdxl": "sdxl-ip-adapter", "sd15": "sd-ip-adapter"}.get(self.model_type, "")
        elif self.schema_id in _SCHEMA_ROUTE_DEFAULTS:
            model_type, training_type = _SCHEMA_ROUTE_DEFAULTS[self.schema_id]
            self.model_type = model_type
            self.training_type = training_type
        return self

