"""Minimal native Flux LoRA trainer inside the existing request/runtime boundary."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from .config import LulynxConfig
from .flux_dataset import create_flux_lora_dataloader
from .flux_lora_utils import FluxFlowConfig, build_flux_flow_inputs, compute_flux_loss_weights, prepare_flux_image_ids, sample_flux_sigmas
from .flux_cache import FluxTrainingCache
from .flux_attention_backend import apply_flux_attention_backend
from .flux_compile_runtime import apply_flux_compile_runtime
from .flux_offload import FluxTransformerStreamingOffloader, move_trainable_parameters, normalize_component_offload_strategy
from .flux_optimizer_backend import create_flux_optimizer_backend
from .flux_runtime_profile import build_flux_trainer_runtime_features, refresh_flux_runtime_profile
from .adapter_runtime_profile import build_adapter_runtime_profile
from .lora_activation_recompute_policy import resolve_lora_activation_recompute
from .lora_injector import LoRAInjector
from .model_family import get_model_family
from core.warehouse.training_features.flux_preflight import is_flux_network_module_supported

logger = logging.getLogger(__name__)


def _value(value: Any) -> str:
    return str(getattr(value, "value", value) or "")


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _path_kind(value: str) -> str:
    if not value:
        return ""
    path = Path(value)
    suffix = path.suffix.lower()
    if suffix in {".safetensors", ".ckpt"}:
        return "single_file"
    return "directory"


class FluxLoraTrainer:
    """Preview Flux LoRA trainer with the same callback surface as LulynxTrainer."""

    def __init__(self, config: LulynxConfig):
        self.config = config
        self.device = self._resolve_device()
        self.weight_dtype = self._resolve_dtype()
        self.on_step: Optional[Callable[[int, int, float, float], None]] = None
        self.on_log: Optional[Callable[[str], None]] = None
        self.on_complete: Optional[Callable[[bool], None]] = None
        self.on_runtime_event: Optional[Callable[[Dict[str, Any]], None]] = None
        self.training_loop = SimpleNamespace(
            global_step=0,
            current_epoch=0,
            total_steps=0,
            memory_optimization_state={"enabled": False, "mode": "none", "source": "flux_lora"},
        )
        self.pipeline = None
        self.transformer = None
        self.vae = None
        self._component_offload_strategy = self._resolve_component_offload_strategy()
        self._transformer_offloader: Optional[FluxTransformerStreamingOffloader] = None
        self.lora_injector: Optional[LoRAInjector] = None
        self.optimizer = None
        self.lr_scheduler = None
        self._attention_backend_profile: Dict[str, Any] = {}
        self._compile_runtime_profile: Dict[str, Any] = {}
        self._adapter_runtime_profile: Dict[str, Any] = {}
        self._optimizer_backend_profile: Dict[str, Any] = {}
        self._data_backend_profile: Dict[str, Any] = {}
        self._component_offload_profile: Dict[str, Any] = {}
        self._gradient_checkpointing_profile: Dict[str, Any] = {}
        self._flux_runtime_profile: Dict[str, Any] = {}
        self._cache: Optional[FluxTrainingCache] = None
        self._generator: Optional[torch.Generator] = None
        self._logged_component_moves: set[str] = set()

    def set_callbacks(
        self,
        on_step: Optional[Callable[[int, int, float, float], None]] = None,
        on_log: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[bool], None]] = None,
        on_cull_samples: Optional[Callable[[List[str]], None]] = None,
        on_runtime_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        if on_step is not None:
            self.on_step = on_step
        if on_log is not None:
            self.on_log = on_log
        if on_complete is not None:
            self.on_complete = on_complete
        if on_runtime_event is not None:
            self.on_runtime_event = on_runtime_event

    def train(self) -> bool:
        return self.start()

    def start(self) -> bool:
        success = False
        try:
            self._seed_everything()
            self._load_pipeline()
            self._prepare_trainable_adapters()
            self._cache = FluxTrainingCache.from_config(self.config, log=self._log)
            dataloader = create_flux_lora_dataloader(self.config, pin_memory=self.device.type == "cuda")
            self._data_backend_profile = dict(getattr(dataloader, "lulynx_data_backend_profile", {}) or {})
            if self._data_backend_profile:
                self.training_loop.memory_optimization_state["data_backend"] = dict(self._data_backend_profile)
            self._refresh_flux_runtime_profile()
            total_steps, total_epochs = self._resolve_training_span(len(dataloader))
            self.training_loop.total_steps = total_steps
            self.optimizer = self._create_optimizer()
            self.lr_scheduler = self._create_scheduler(total_steps)
            flux_runtime = self._refresh_flux_runtime_profile()
            self._emit_runtime_event(
                {
                    "type": "flux_lora_start",
                    "total_steps": total_steps,
                    "total_epochs": total_epochs,
                    "trainable_params": len(self.lora_injector.get_trainable_params()) if self.lora_injector else 0,
                    "attention_backend": dict(self._attention_backend_profile),
                    "compile_runtime": dict(self._compile_runtime_profile),
                    "data_backend": dict(self._data_backend_profile),
                    "optimizer_backend": dict(self._optimizer_backend_profile),
                    "cache": dict(self._cache.profile) if self._cache is not None else {},
                    "flux_runtime": flux_runtime,
                }
            )
            self._train_loop(dataloader, total_steps, total_epochs)
            self._save_lora(kind="final")
            success = True
            return True
        finally:
            if self._transformer_offloader is not None:
                self._transformer_offloader.cleanup()
            if self.on_complete is not None:
                self.on_complete(success)

    def _resolve_device(self) -> torch.device:
        requested = str(getattr(self.config, "device", "") or "").strip().lower()
        if requested == "cpu":
            return torch.device("cpu")
        if requested == "mps" and getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _resolve_dtype(self) -> torch.dtype:
        precision = _value(getattr(self.config, "mixed_precision", "bf16")).lower()
        if self.device.type != "cuda":
            return torch.float32
        if precision == "fp16":
            return torch.float16
        if precision == "bf16" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _seed_everything(self) -> None:
        seed = int(getattr(self.config, "seed", 0) or 0)
        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
            self._generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            self._generator = torch.Generator().manual_seed(seed)

    def _device_vram_gb(self) -> float:
        if self.device.type != "cuda" or not torch.cuda.is_available():
            return 0.0
        try:
            return float(torch.cuda.get_device_properties(self.device).total_memory) / float(1024 ** 3)
        except Exception:
            return 0.0

    def _resolve_component_offload_strategy(self) -> str:
        return normalize_component_offload_strategy(
            getattr(self.config, "te_vae_offload_strategy", "phase"),
            cuda_available=self.device.type == "cuda" and torch.cuda.is_available(),
            total_vram_gb=self._device_vram_gb(),
            sequential_cpu_offload=_truthy(getattr(self.config, "enable_sequential_cpu_offload", False)),
            module_offload=_truthy(getattr(self.config, "module_offload_enabled", False)),
        )

    def _build_component_offload_profile(self) -> Dict[str, Any]:
        raw_strategy = str(getattr(self.config, "te_vae_offload_strategy", "phase") or "phase").strip().lower()
        flux_transformer_offload = str(getattr(self.config, "flux_transformer_offload", "auto") or "auto").strip().lower()
        return {
            "resolved": self._component_offload_strategy,
            "te_vae_offload_strategy": raw_strategy,
            "enable_sequential_cpu_offload": _truthy(getattr(self.config, "enable_sequential_cpu_offload", False)),
            "module_offload_enabled": _truthy(getattr(self.config, "module_offload_enabled", False)),
            "flux_transformer_offload": flux_transformer_offload,
            "cuda_available": self.device.type == "cuda" and torch.cuda.is_available(),
            "total_vram_gb": round(self._device_vram_gb(), 2),
        }

    def _refresh_flux_runtime_profile(self) -> Dict[str, Any]:
        return refresh_flux_runtime_profile(self)

    def get_runtime_features(self) -> Dict[str, Any]:
        return build_flux_trainer_runtime_features(self)

    def _run_manifest_extra(self) -> Dict[str, Any]:
        return self.get_runtime_features()

    def _apply_transformer_gradient_checkpointing(
        self,
        checkpoint_fn: Optional[Callable[..., Any]] = None,
        *,
        source: str = "gradient_checkpointing",
    ) -> Dict[str, Any]:
        requested = _truthy(getattr(self.config, "gradient_checkpointing", True)) or checkpoint_fn is not None
        profile: Dict[str, Any] = {
            "requested": requested,
            "enabled": False,
            "source": source,
            "custom_checkpoint_fn": checkpoint_fn is not None,
        }
        if not requested:
            self._gradient_checkpointing_profile = profile
            return profile
        enabler = getattr(self.transformer, "enable_gradient_checkpointing", None)
        if not callable(enabler):
            profile["fallback_reason"] = "transformer_missing_enable_gradient_checkpointing"
            self._gradient_checkpointing_profile = profile
            return profile
        try:
            if checkpoint_fn is not None:
                enabler(checkpoint_fn)
            else:
                enabler()
            profile["enabled"] = True
        except TypeError as exc:
            if checkpoint_fn is None:
                profile["fallback_reason"] = f"enable_gradient_checkpointing TypeError: {exc}"
            else:
                try:
                    enabler()
                    profile["enabled"] = True
                    profile["fallback_reason"] = "custom checkpoint function unsupported; enabled default checkpointing"
                except Exception as fallback_exc:
                    profile["fallback_reason"] = f"custom checkpoint fallback failed: {fallback_exc}"
        except Exception as exc:
            profile["fallback_reason"] = f"enable_gradient_checkpointing failed: {exc}"
        self._gradient_checkpointing_profile = profile
        return profile

    def _clear_device_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    @staticmethod
    def _first_module_device(module: Any) -> torch.device:
        if module is None:
            return torch.device("cpu")
        for tensor in module.parameters(recurse=True):
            return tensor.device
        for tensor in module.buffers(recurse=True):
            return tensor.device
        return torch.device("cpu")

    def _move_component(self, module: Any, device: torch.device | str, label: str) -> None:
        if module is None:
            return
        target = torch.device(device)
        current = self._first_module_device(module)
        if current.type == target.type and (target.index is None or current.index == target.index):
            return
        module.to(target)
        key = f"{label}:{target.type}"
        if key not in self._logged_component_moves:
            self._log(f"Flux component {label} moved to {target.type} ({self._component_offload_strategy} offload).")
            self._logged_component_moves.add(key)

    def _offload_component(self, module: Any, label: str) -> None:
        if self._component_offload_strategy == "resident":
            return
        self._move_component(module, torch.device("cpu"), label)
        self._clear_device_cache()

    def _offload_conditioners(self) -> None:
        if self.pipeline is None:
            return
        self._offload_component(self.vae, "vae")
        self._offload_component(self.pipeline.text_encoder, "clip_l")
        self._offload_component(self.pipeline.text_encoder_2, "t5xxl")

    def _ensure_transformer_for_forward(self) -> None:
        if self._transformer_offloader is not None:
            self._transformer_offloader.prepare_for_forward()
            return
        self._move_component(self.transformer, self.device, "transformer")

    def _offload_transformer_between_steps(self) -> None:
        if self._component_offload_strategy != "aggressive":
            return
        if self._transformer_offloader is not None:
            self._transformer_offloader.offload_all_blocks()
            return
        self._offload_component(self.transformer, "transformer")

    def _wants_transformer_streaming_offload(self) -> bool:
        if self.device.type != "cuda":
            return False
        raw = str(getattr(self.config, "flux_transformer_offload", "auto") or "auto").strip().lower().replace("-", "_")
        if raw in {"off", "false", "0", "disabled", "none"}:
            return False
        if raw in {"on", "true", "1", "yes", "enabled", "aggressive", "streaming", "streaming_offload"}:
            return True
        return self._component_offload_strategy == "aggressive"

    def _load_pipeline(self) -> None:
        try:
            from diffusers import AutoencoderKL, FluxPipeline, FluxTransformer2DModel
            from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
        except ImportError as exc:
            raise RuntimeError(f"Flux LoRA requires diffusers and transformers packages: {exc}") from exc

        model_path = str(getattr(self.config, "pretrained_model_name_or_path", "") or "").strip()
        if not model_path:
            raise RuntimeError("Flux LoRA requires pretrained_model_name_or_path/base_model_path.")
        disable_mmap = _truthy(getattr(self.config, "disable_mmap_load_safetensors", False))
        self._log(f"Loading FluxPipeline: {model_path}")
        if _path_kind(model_path) == "single_file":
            pipe = FluxPipeline.from_single_file(
                model_path,
                torch_dtype=self.weight_dtype,
                local_files_only=True,
                disable_mmap=disable_mmap,
            )
        else:
            pipe = FluxPipeline.from_pretrained(model_path, torch_dtype=self.weight_dtype, local_files_only=True)

        transformer_path = str(getattr(self.config, "flux_transformer_path", "") or "").strip()
        if transformer_path:
            if _path_kind(transformer_path) == "single_file":
                pipe.transformer = FluxTransformer2DModel.from_single_file(
                    transformer_path,
                    torch_dtype=self.weight_dtype,
                    local_files_only=True,
                    disable_mmap=disable_mmap,
                )
            else:
                pipe.transformer = FluxTransformer2DModel.from_pretrained(
                    transformer_path,
                    torch_dtype=self.weight_dtype,
                    local_files_only=True,
                )
        ae_path = str(getattr(self.config, "ae_path", "") or "").strip()
        if ae_path:
            if _path_kind(ae_path) == "single_file":
                pipe.vae = AutoencoderKL.from_single_file(
                    ae_path,
                    torch_dtype=self.weight_dtype,
                    local_files_only=True,
                    disable_mmap=disable_mmap,
                )
            else:
                pipe.vae = AutoencoderKL.from_pretrained(ae_path, torch_dtype=self.weight_dtype, local_files_only=True)
        t5_path = str(getattr(self.config, "t5xxl_path", "") or "").strip()
        if t5_path:
            pipe.text_encoder_2 = T5EncoderModel.from_pretrained(t5_path, torch_dtype=self.weight_dtype, local_files_only=True)
            pipe.tokenizer_2 = T5TokenizerFast.from_pretrained(t5_path, local_files_only=True)
        clip_l_path = str(getattr(self.config, "clip_l_path", "") or "").strip()
        if clip_l_path:
            pipe.text_encoder = CLIPTextModel.from_pretrained(clip_l_path, torch_dtype=self.weight_dtype, local_files_only=True)
            pipe.tokenizer = CLIPTokenizer.from_pretrained(clip_l_path, local_files_only=True)

        self.pipeline = pipe
        self.transformer = pipe.transformer
        self.vae = pipe.vae
        self._attention_backend_profile = apply_flux_attention_backend(
            self.transformer, getattr(self.config, "attention_backend", "auto"),
            log=self._log,
            cuda_available=self.device.type == "cuda" and torch.cuda.is_available(),
        )
        for module in (self.vae, pipe.text_encoder, pipe.text_encoder_2, self.transformer):
            if module is not None:
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
        self._apply_transformer_gradient_checkpointing(source="config")
        self._component_offload_profile = self._build_component_offload_profile()
        if self._component_offload_strategy == "resident":
            pipe.to(self.device)
        elif self._component_offload_strategy == "phase":
            self._ensure_transformer_for_forward()
            self._offload_conditioners()
        else:
            self._offload_conditioners()
        self.training_loop.memory_optimization_state = {
            "enabled": self._component_offload_strategy != "resident",
            "mode": "component_offload" if self._component_offload_strategy != "resident" else "none",
            "source": "flux_lora",
            "component_offload_strategy": self._component_offload_strategy,
            "component_offload": dict(self._component_offload_profile),
            "gradient_checkpointing": dict(self._gradient_checkpointing_profile),
            "total_vram_gb": round(self._device_vram_gb(), 2),
            "attention_backend": self._attention_backend_profile,
        }
        self._refresh_flux_runtime_profile()

    def _prepare_trainable_adapters(self) -> None:
        requested_network_module = str(
            getattr(self.config, "flux_requested_network_module", "")
            or getattr(self.config, "network_module", "")
        ).strip().lower()
        if not is_flux_network_module_supported(requested_network_module):
            raise RuntimeError(
                "Flux LoRA preview currently supports network_module=networks.lora only; "
                f"got {requested_network_module!r}."
            )
        family = get_model_family("flux")
        activation_recompute = resolve_lora_activation_recompute(self.config, auto_default=True)
        injector = LoRAInjector(
            rank=int(getattr(self.config, "network_dim", 16) or 16),
            alpha=float(getattr(self.config, "network_alpha", 8.0) or 8.0),
            dropout=float(getattr(self.config, "network_dropout", 0.0) or 0.0),
            pissa_enabled=_truthy(getattr(self.config, "pissa_enabled", False)),
            pissa_niter=int(getattr(self.config, "pissa_init_iters", 1) or 1),
            dora_enabled=_truthy(getattr(self.config, "use_dora", False)) or _truthy(getattr(self.config, "dora_enabled", False)),
            model_arch="flux",
            activation_recompute=activation_recompute,
            rs_lora_enabled=_truthy(getattr(self.config, "rs_lora_enabled", False)),
        )
        injected = injector.inject(self.transformer, family.unet_target_modules, prefix="transformer")
        if not injected:
            raise RuntimeError("No Flux transformer LoRA targets were matched; check diffusers Flux model structure.")
        weights_path = str(getattr(self.config, "network_weights_path", "") or "").strip()
        if weights_path:
            injector.load_lora(weights_path)
        trainable = injector.get_trainable_params()
        if not trainable:
            raise RuntimeError("Flux LoRA injection produced no trainable parameters.")
        move_trainable_parameters(self.transformer, self.device, self.weight_dtype)
        self.lora_injector = injector
        self._log(f"Injected {len(injected)} Flux LoRA layers.")
        self._adapter_runtime_profile = build_adapter_runtime_profile(self.config, injector, model_arch="flux")
        self.training_loop.memory_optimization_state["adapter_runtime"] = dict(self._adapter_runtime_profile)
        self.training_loop.memory_optimization_state["lora_activation_recompute"] = {
            "enabled": activation_recompute,
            "mode": str(getattr(self.config, "lora_activation_recompute_mode", "auto") or "auto"),
            "auto_default": True,
        }
        self._compile_runtime_profile = apply_flux_compile_runtime(self.config, self.transformer, log=self._log)
        if self._compile_runtime_profile:
            self.training_loop.memory_optimization_state["compile_runtime"] = dict(self._compile_runtime_profile)
        if self._wants_transformer_streaming_offload():
            self._install_transformer_streaming_offload()
        self._refresh_flux_runtime_profile()

    def _install_transformer_streaming_offload(self) -> None:
        offloader = FluxTransformerStreamingOffloader(
            self.transformer,
            device=self.device,
            dtype=self.weight_dtype,
            log=self._log,
        )
        state = offloader.install()
        if state.get("enabled"):
            self._transformer_offloader = offloader
            self._apply_transformer_gradient_checkpointing(offloader.checkpoint, source="transformer_streaming_offload")
            self.training_loop.memory_optimization_state.update(
                {
                    "enabled": True,
                    "mode": "component_streaming_offload",
                    "transformer_offload": offloader.state_dict(),
                    "gradient_checkpointing": dict(self._gradient_checkpointing_profile),
                }
            )
            self._refresh_flux_runtime_profile()
        else:
            offloader.cleanup()

    def _resolve_training_span(self, microbatches_per_epoch: int) -> tuple[int, int]:
        accum = max(int(getattr(self.config, "gradient_accumulation_steps", 1) or 1), 1)
        steps_per_epoch = max(math.ceil(max(microbatches_per_epoch, 1) / accum), 1)
        epochs = max(int(getattr(self.config, "max_train_epochs", 1) or 1), 1)
        requested_steps = max(int(getattr(self.config, "max_train_steps", 0) or 0), 0)
        if requested_steps > 0:
            return requested_steps, max(math.ceil(requested_steps / steps_per_epoch), 1)
        return steps_per_epoch * epochs, epochs

    def _create_optimizer(self):
        params = self.lora_injector.get_trainable_params() if self.lora_injector else []
        lr = float(getattr(self.config, "learning_rate", 1e-4) or 1e-4)
        weight_decay = float(getattr(self.config, "weight_decay", 0.0) or 0.0)
        optimizer, profile = create_flux_optimizer_backend(
            params,
            optimizer_backend=getattr(self.config, "optimizer_backend", "auto"),
            optimizer_type=getattr(self.config, "optimizer_type", "AdamW"),
            lr=lr,
            weight_decay=weight_decay,
            device=self.device,
            log=self._log,
        )
        self._optimizer_backend_profile = dict(profile)
        self.training_loop.memory_optimization_state["optimizer_backend"] = dict(profile)
        self._refresh_flux_runtime_profile()
        return optimizer

    def _create_scheduler(self, total_steps: int):
        scheduler = _value(getattr(self.config, "scheduler", getattr(self.config, "lr_scheduler", "cosine"))).lower()
        if scheduler in {"constant", "constant_with_warmup"}:
            return torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0, total_iters=1)
        if scheduler == "linear":
            return torch.optim.lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.0, total_iters=max(total_steps, 1))
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=max(total_steps, 1))

    def _train_loop(self, dataloader: Any, total_steps: int, total_epochs: int) -> None:
        accum = max(int(getattr(self.config, "gradient_accumulation_steps", 1) or 1), 1)
        self.optimizer.zero_grad(set_to_none=True)
        pending = 0
        last_loss = 0.0
        for epoch in range(total_epochs):
            self.training_loop.current_epoch = epoch + 1
            if hasattr(dataloader.dataset, "set_current_epoch"):
                dataloader.dataset.set_current_epoch(epoch)
            for batch in dataloader:
                if self.training_loop.global_step >= total_steps:
                    return
                if hasattr(dataloader.dataset, "set_global_step"):
                    dataloader.dataset.set_global_step(self.training_loop.global_step)
                loss = self._training_loss(batch) / accum
                loss.backward()
                last_loss = float(loss.detach().float().item() * accum)
                pending += 1
                if pending >= accum:
                    self._optimizer_step(last_loss, epoch + 1)
                    pending = 0
            if pending:
                self._optimizer_step(last_loss, epoch + 1)
                pending = 0
            if hasattr(dataloader.dataset, "increment_variant_epoch"):
                dataloader.dataset.increment_variant_epoch()
            save_every = max(int(getattr(self.config, "save_every_n_epochs", 0) or 0), 0)
            if save_every and (epoch + 1) % save_every == 0 and self.training_loop.global_step < total_steps:
                self._save_lora(kind="epoch", epoch=epoch + 1)

    def _training_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        if self._component_offload_strategy == "aggressive":
            self._offload_transformer_between_steps()
        images = batch["images"].to(device=self.device, dtype=self.weight_dtype)
        captions = batch.get("captions") or [""] * images.shape[0]
        cache = self._cache or FluxTrainingCache.from_config(self.config, log=self._log)
        self._cache = cache

        def _encode_flux_text(items: list[str]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            self._move_component(self.pipeline.text_encoder, self.device, "clip_l")
            self._move_component(self.pipeline.text_encoder_2, self.device, "t5xxl")
            try:
                return self.pipeline.encode_prompt(
                    prompt=list(items),
                    prompt_2=list(items),
                    device=self.device,
                    max_sequence_length=max(int(getattr(self.config, "t5_max_token_length", 512) or 512), 64),
                )
            finally:
                self._offload_component(self.pipeline.text_encoder, "clip_l")
                self._offload_component(self.pipeline.text_encoder_2, "t5xxl")

        with torch.no_grad():
            latents, packed = cache.resolve_latents(
                batch,
                images,
                vae=self.vae,
                device=self.device,
                dtype=self.weight_dtype,
                generator=self._generator,
                ensure_vae=lambda: self._move_component(self.vae, self.device, "vae"),
                release_vae=lambda: self._offload_component(self.vae, "vae"),
            )
            prompt_embeds, pooled_embeds, text_ids = cache.resolve_text(
                list(captions),
                encode=_encode_flux_text,
                device=self.device,
                dtype=self.weight_dtype,
            )
            self._ensure_transformer_for_forward()
        flow_cfg = FluxFlowConfig(
            timestep_sampling=str(getattr(self.config, "timestep_sampling", "") or "shift"),
            sigmoid_scale=float(getattr(self.config, "sigmoid_scale", 1.0) or 1.0),
            discrete_flow_shift=float(getattr(self.config, "discrete_flow_shift", 1.0) or 1.0),
            weighting_scheme=str(getattr(self.config, "weighting_scheme", "none") or "none"),
            logit_mean=float(getattr(self.config, "flow_logit_mean", 0.0) or 0.0),
            logit_std=float(getattr(self.config, "flow_logit_std", 1.0) or 1.0),
        )
        noise = torch.randn(packed.shape, device=self.device, dtype=self.weight_dtype, generator=self._generator)
        sigmas = sample_flux_sigmas(packed.shape[0], device=self.device, dtype=self.weight_dtype, config=flow_cfg, generator=self._generator)
        noisy, target, timesteps = build_flux_flow_inputs(packed, noise, sigmas)
        image_ids = prepare_flux_image_ids(
            packed_height=latents.shape[-2] // 2,
            packed_width=latents.shape[-1] // 2,
            device=self.device,
            dtype=text_ids.dtype,
        )
        guidance = None
        if bool(getattr(getattr(self.transformer, "config", None), "guidance_embeds", False)):
            guidance = torch.full((packed.shape[0],), float(getattr(self.config, "guidance_scale", 1.0) or 1.0), device=self.device, dtype=self.weight_dtype)
        with torch.autocast(device_type="cuda", dtype=self.weight_dtype, enabled=self.device.type == "cuda"):
            prediction = self.transformer(
                hidden_states=noisy,
                timestep=timesteps,
                guidance=guidance,
                pooled_projections=pooled_embeds.to(device=self.device, dtype=self.weight_dtype),
                encoder_hidden_states=prompt_embeds.to(device=self.device, dtype=self.weight_dtype),
                txt_ids=text_ids.to(device=self.device),
                img_ids=image_ids,
                return_dict=False,
            )[0]
        per_item = F.mse_loss(prediction.float(), target.float(), reduction="none").mean(dim=tuple(range(1, prediction.ndim)))
        weights = compute_flux_loss_weights(sigmas, flow_cfg.weighting_scheme, flow_cfg.mode_scale).to(per_item.device)
        caption_weights = batch.get("caption_weights")
        if caption_weights is not None:
            weights = weights * caption_weights.to(device=per_item.device, dtype=weights.dtype)
        return (per_item * weights).mean()

    def _optimizer_step(self, loss_value: float, epoch: int) -> None:
        max_norm = float(getattr(self.config, "max_grad_norm", 0.0) or 0.0)
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(self.lora_injector.get_trainable_params(), max_norm)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.training_loop.global_step += 1
        lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
        self._offload_transformer_between_steps()
        if self._transformer_offloader is not None:
            self.training_loop.memory_optimization_state["transformer_offload"] = self._transformer_offloader.state_dict()
        self._refresh_flux_runtime_profile()
        if self.on_step is not None:
            self.on_step(self.training_loop.global_step, epoch, float(loss_value), lr)
        save_every = max(int(getattr(self.config, "save_every_n_steps", 0) or 0), 0)
        if save_every and self.training_loop.global_step % save_every == 0:
            self._save_lora(kind="step", step=self.training_loop.global_step)

    def _save_lora(self, *, kind: str, step: int = 0, epoch: int = 0) -> Path:
        output_dir = Path(str(getattr(self.config, "save_to", "") or getattr(self.config, "output_dir", "outputs") or "outputs"))
        output_dir.mkdir(parents=True, exist_ok=True)
        name = str(getattr(self.config, "output_name", "lulynx_flux_lora") or "lulynx_flux_lora")
        ext = ".safetensors" if str(getattr(self.config, "save_model_as", "safetensors") or "safetensors").lower() == "safetensors" else ".pt"
        if kind == "step":
            path = output_dir / f"{name}-step{int(step):06d}{ext}"
        elif kind == "epoch":
            path = output_dir / f"{name}-{int(epoch):06d}{ext}"
        else:
            path = output_dir / f"{name}{ext}"
        metadata = None
        if not _truthy(getattr(self.config, "no_metadata", False)):
            metadata = {
                "ss_base_model_version": "flux",
                "ss_model_family": "flux",
                "ss_output_name": name,
                "ss_network_dim": str(getattr(self.config, "network_dim", "")),
                "ss_network_alpha": str(getattr(self.config, "network_alpha", "")),
                "ss_training_step": str(self.training_loop.global_step),
                "ss_training_epoch": str(self.training_loop.current_epoch),
                "ss_lulynx_trainer": "flux_lora_preview",
            }
            comment = str(getattr(self.config, "training_comment", "") or "").strip()
            if comment:
                metadata["ss_training_comment"] = comment
        self.lora_injector.save_lora(str(path), metadata=metadata)
        self._log(f"Saved Flux LoRA: {path}")
        return path

    def _log(self, message: str) -> None:
        logger.info(message)
        if self.on_log is not None:
            self.on_log(message)

    def _emit_runtime_event(self, payload: Dict[str, Any]) -> None:
        if self.on_runtime_event is not None:
            self.on_runtime_event(dict(payload or {}))

__all__ = ["FluxLoraTrainer", "_path_kind"]
