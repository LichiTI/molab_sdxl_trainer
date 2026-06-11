from __future__ import annotations

import gc
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import torch

from backend.core.lulynx_trainer.config_adapter import ConfigAdapter
from backend.core.lulynx_trainer.model_family import get_model_family
from backend.core.lulynx_trainer.model_loader import ModelLoader
from backend.core.lulynx_trainer.training_loop import TrainingLoop
from backend.core.security import validate_path
from backend.core.training_components.ghost_replay import GhostRecorder, inspect_ghost_fingerprint

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "auto": None,
    "fp16": torch.float16,
    "float16": torch.float16,
    "half": torch.float16,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "float32": torch.float32,
    "float": torch.float32,
    "no": torch.float32,
}

_SUPPORTED_ARCHES = {"sdxl", "sd15", "anima", "newbie"}


def _to_str(value: Any) -> str:
    return str(value or "").strip()


def _slugify(value: str, fallback: str = "ghost") -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = text.strip("._-")
    return text or fallback


def _split_patterns(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        items = [str(item or "").strip() for item in value]
    else:
        items = re.split(r"[\r\n,]+", str(value or ""))
    return [item for item in items if item]


def _parse_prompts(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        items = [str(item or "").strip() for item in value]
    else:
        text = str(value or "").replace("\r\n", "\n").replace("\r", "\n")
        items = [line.strip() for line in text.split("\n")]
    prompts = [item for item in items if item]
    return prompts or [""]


def _parse_timesteps(value: Any) -> list[int]:
    if isinstance(value, (list, tuple)):
        raw_items = value
    else:
        raw_items = re.split(r"[\r\n, ]+", str(value or "200,800"))
    steps: list[int] = []
    for item in raw_items:
        text = str(item or "").strip()
        if not text:
            continue
        try:
            steps.append(max(int(float(text)), 0))
        except Exception:
            continue
    deduped = sorted(set(steps))
    return deduped or [200, 800]


def _parse_resolution(value: Any, *, default: tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        try:
            width = max(int(value[0]), 64)
            height = max(int(value[1]), 64)
            return width, height
        except Exception:
            return default
    text = _to_str(value)
    if not text:
        return default
    if text.isdigit():
        size = max(int(text), 64)
        return size, size
    parts = re.split(r"[xX, ]+", text)
    if len(parts) >= 2:
        try:
            width = max(int(parts[0]), 64)
            height = max(int(parts[1]), 64)
            return width, height
        except Exception:
            return default
    return default


def _resolve_device(name: str) -> tuple[str, list[str]]:
    notes: list[str] = []
    requested = _to_str(name).lower()
    if requested in {"", "auto"}:
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        notes.append("CUDA 不可用，已自动回退到 CPU。")
        return "cpu", notes
    if requested.startswith("mps") and not getattr(torch.backends, "mps", None):
        notes.append("MPS 不可用，已自动回退到 CPU。")
        return "cpu", notes
    return requested, notes


def _resolve_dtype(name: Any, device: str) -> tuple[torch.dtype, list[str]]:
    notes: list[str] = []
    normalized = _to_str(name).lower() or "auto"
    dtype = _DTYPE_MAP.get(normalized, None)
    if dtype is None:
        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    if not device.startswith("cuda") and dtype != torch.float32:
        notes.append("CPU 录制默认改用 fp32，以避免半精度路径不稳定。")
        dtype = torch.float32
    return dtype, notes


def _arch_from_training_type(training_type: str) -> str:
    text = _to_str(training_type).lower().replace("_", "-")
    if text.startswith("anima-"):
        return "anima"
    if text.startswith("newbie-"):
        return "newbie"
    if text.startswith("sdxl-"):
        return "sdxl"
    if text.startswith("sd-"):
        return "sd15"
    return ""


def _infer_arch(explicit_arch: str, config: Any) -> str:
    candidates = [
        explicit_arch,
        getattr(config, "model_type", ""),
        getattr(config, "model_arch", ""),
        getattr(config, "schema_id", ""),
        getattr(config, "model_train_type", ""),
        getattr(config, "training_type", ""),
    ]
    for candidate in candidates:
        text = _to_str(candidate).lower()
        if not text:
            continue
        if text in _SUPPORTED_ARCHES:
            return text
        route_arch = _arch_from_training_type(text)
        if route_arch:
            return route_arch
    return "sdxl"


def _default_resolution_for_arch(arch: str) -> tuple[int, int]:
    if arch == "sd15":
        return 512, 512
    return 1024, 1024


def _default_output_filename(config: Any, arch: str, model_path: str) -> str:
    output_name = _to_str(getattr(config, "output_name", ""))
    if output_name:
        stem = _slugify(output_name, fallback=f"{arch}_ghost")
    else:
        stem = _slugify(Path(model_path).stem or arch, fallback=f"{arch}_ghost")
    return f"{stem}_ghost.lulynx"


def _resolve_output_path(raw_value: str, *, config: Any, arch: str, model_path: str) -> Path:
    raw_text = _to_str(raw_value)
    if raw_text:
        target = validate_path(raw_text, allow_files=True, allow_dirs=True)
        treat_as_dir = raw_text.endswith(("\\", "/")) or Path(raw_text).suffix == ""
    else:
        base_dir = _to_str(getattr(config, "output_dir", "")) or "./output/ghost_replay"
        target = validate_path(base_dir, allow_files=False, allow_dirs=True)
        treat_as_dir = True
    if treat_as_dir:
        target = target / _default_output_filename(config, arch, model_path)
    elif target.suffix.lower() != ".lulynx":
        target = target.with_suffix(".lulynx")
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def _set_job_metadata(job: Any, **updates: Any) -> None:
    metadata = dict(getattr(job, "metadata", {}) or {})
    metadata.update(updates)
    job.metadata = metadata


def _resolve_model_paths(
    *,
    config: Any,
    arch: str,
    model_path: str,
    vae_path: str,
    anima_qwen3_path: str,
    anima_t5_tokenizer_path: str,
) -> dict[str, str]:
    resolved_model_path = _to_str(model_path)
    resolved_vae_path = _to_str(vae_path)
    resolved_qwen3_path = _to_str(anima_qwen3_path)
    resolved_t5_tokenizer_path = _to_str(anima_t5_tokenizer_path)

    if arch == "anima":
        resolved_model_path = resolved_model_path or _to_str(getattr(config, "anima_model_path", "")) or _to_str(getattr(config, "pretrained_model_name_or_path", ""))
        resolved_vae_path = resolved_vae_path or _to_str(getattr(config, "vae_path", "")) or _to_str(getattr(config, "vae", ""))
        resolved_qwen3_path = resolved_qwen3_path or _to_str(getattr(config, "anima_qwen3_path", "")) or _to_str(getattr(config, "qwen3", ""))
        resolved_t5_tokenizer_path = resolved_t5_tokenizer_path or _to_str(getattr(config, "anima_t5_tokenizer_path", "")) or _to_str(getattr(config, "t5_tokenizer_path", ""))
    elif arch == "newbie":
        resolved_model_path = resolved_model_path or _to_str(getattr(config, "newbie_diffusers_path", "")) or _to_str(getattr(config, "pretrained_model_name_or_path", ""))
        resolved_vae_path = resolved_vae_path or _to_str(getattr(config, "newbie_vae_path", "")) or _to_str(getattr(config, "vae_path", ""))
    else:
        resolved_model_path = resolved_model_path or _to_str(getattr(config, "pretrained_model_name_or_path", ""))
        resolved_vae_path = resolved_vae_path or _to_str(getattr(config, "vae_path", "")) or _to_str(getattr(config, "vae", ""))

    return {
        "model_path": resolved_model_path,
        "vae_path": resolved_vae_path,
        "anima_qwen3_path": resolved_qwen3_path,
        "anima_t5_tokenizer_path": resolved_t5_tokenizer_path,
    }


def _build_prompt_encoding_shim(model: Any, arch: str, device: str, dtype: torch.dtype) -> Any:
    qwen3_encoder = getattr(model, "anima_qwen3_encoder", None) or getattr(model, "qwen3_encoder", None)
    qwen3_tokenizer = getattr(model, "anima_qwen3_tokenizer", None) or getattr(model, "qwen3_tokenizer", None)
    shim = SimpleNamespace()
    shim.text_encoder_1 = getattr(model, "text_encoder_1", None)
    shim.text_encoder_2 = getattr(model, "text_encoder_2", None)
    shim.tokenizer_1 = getattr(model, "tokenizer_1", None)
    shim.tokenizer_2 = getattr(model, "tokenizer_2", None)
    shim.qwen3_encoder = qwen3_encoder
    shim.qwen3_tokenizer = qwen3_tokenizer
    shim._family = get_model_family(arch)
    shim._model_arch = arch
    shim._runtime_device = torch.device(device)
    shim.device = device
    shim.dtype = dtype
    shim._train_text_encoder_any = False
    shim._text_encoder_cpu_residency = False
    shim._token_padder_1 = None
    shim._token_padder_2 = None
    return shim


def _encode_prompt_sample(
    *,
    model: Any,
    arch: str,
    prompt: str,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
) -> dict[str, Any]:
    shim = _build_prompt_encoding_shim(model, arch, device, dtype)
    prompt_data = dict(TrainingLoop._encode_prompt_native(shim, [prompt]))
    if arch == "anima":
        prompt_data.update(TrainingLoop._encode_qwen3(shim, [prompt]))
    if shim._family.uses_time_ids:
        time_cond = TrainingLoop._get_timestep_embedding(
            shim,
            1,
            [(width, height)],
            [(width, height)],
            [(0, 0, width, height)],
        )
        if time_cond:
            prompt_data["added_cond_kwargs"] = dict(time_cond.get("added_cond_kwargs", {}) or {})
    return prompt_data


def _move_recording_modules(model: Any, device: str, dtype: torch.dtype) -> None:
    modules = [
        getattr(model, "unet", None),
        getattr(model, "text_encoder_1", None),
        getattr(model, "text_encoder_2", None),
        getattr(model, "anima_qwen3_encoder", None),
        getattr(model, "qwen3_encoder", None),
    ]
    for module in modules:
        if module is None:
            continue
        module.requires_grad_(False)
        module.eval()
        module.to(device=device, dtype=dtype)


def _release_model(model: Any) -> None:
    for name in (
        "unet",
        "text_encoder_1",
        "text_encoder_2",
        "vae",
        "anima_qwen3_encoder",
        "qwen3_encoder",
    ):
        module = getattr(model, name, None)
        if module is None:
            continue
        try:
            module.to(device="cpu")
        except Exception:
            pass


def _build_sample_inputs(
    *,
    prompts: Sequence[str],
    model: Any,
    arch: str,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
    seed: int,
) -> list[dict[str, Any]]:
    family = get_model_family(arch)
    latent_h = max(height // 8, 1)
    latent_w = max(width // 8, 1)
    latent_shape = (1, int(family.latent_channels), latent_h, latent_w)
    samples: list[dict[str, Any]] = []
    for index, prompt in enumerate(prompts):
        samples.append(
            {
                "prompt": prompt,
                "prompt_bundle": _encode_prompt_sample(
                    model=model,
                    arch=arch,
                    prompt=prompt,
                    width=width,
                    height=height,
                    device=device,
                    dtype=dtype,
                ),
                "latent_shape": latent_shape,
                "seed": int(seed) + index,
                "model_arch": arch,
            }
        )
    return samples


def _build_forward_fn(*, loaded_model: Any, dtype: torch.dtype):
    def _to_tensor(value: Any, *, device: torch.device) -> Any:
        if isinstance(value, torch.Tensor):
            if value.is_floating_point():
                return value.to(device=device, dtype=dtype)
            return value.to(device=device)
        if isinstance(value, Mapping):
            return {key: _to_tensor(item, device=device) for key, item in value.items()}
        return value

    def _forward(*, model: Any, sample: Mapping[str, Any], timestep: int, sample_idx: int, device: str) -> Any:
        del sample_idx
        local_device = torch.device(device)
        prompt_bundle = dict(sample.get("prompt_bundle", {}) or {})
        latent_shape = tuple(sample.get("latent_shape") or ())
        if not latent_shape:
            raise RuntimeError("Ghost Replay sample is missing latent_shape")

        base_latents = torch.randn(latent_shape, device=local_device, dtype=dtype)
        noise = torch.randn_like(base_latents)
        arch = _to_str(sample.get("model_arch", "sdxl")).lower() or "sdxl"

        if arch in {"sdxl", "sd15"}:
            timestep_tensor = torch.full((latent_shape[0],), int(timestep), device=local_device, dtype=torch.long)
            noisy_latents = loaded_model.noise_scheduler.add_noise(base_latents, noise, timestep_tensor)
        else:
            timestep_value = max(0.0, min(float(timestep), 1000.0))
            t = timestep_value / 1000.0
            view_shape = (latent_shape[0],) + (1,) * (len(latent_shape) - 1)
            view_t = torch.full(view_shape, t, device=local_device, dtype=dtype)
            noisy_latents = (1.0 - view_t) * base_latents + view_t * noise
            timestep_tensor = torch.full((latent_shape[0],), timestep_value, device=local_device, dtype=dtype)

        unet_kwargs = {
            "sample": noisy_latents,
            "timestep": timestep_tensor,
            "encoder_hidden_states": _to_tensor(prompt_bundle["encoder_hidden_states"], device=local_device),
        }

        added_cond = prompt_bundle.get("added_cond_kwargs")
        if added_cond:
            cond = _to_tensor(added_cond, device=local_device)
            pooled = prompt_bundle.get("pooled_prompt_embeds")
            if pooled is not None and isinstance(cond, dict) and "text_embeds" not in cond:
                cond["text_embeds"] = _to_tensor(pooled, device=local_device)
            unet_kwargs["added_cond_kwargs"] = cond
        elif prompt_bundle.get("pooled_prompt_embeds") is not None:
            unet_kwargs["added_cond_kwargs"] = {
                "text_embeds": _to_tensor(prompt_bundle["pooled_prompt_embeds"], device=local_device),
            }

        qwen3_hidden = prompt_bundle.get("qwen3_hidden_states")
        if qwen3_hidden is not None:
            unet_kwargs["qwen3_hidden_states"] = _to_tensor(qwen3_hidden, device=local_device)
            qwen3_mask = prompt_bundle.get("qwen3_attention_mask")
            if qwen3_mask is not None:
                unet_kwargs["qwen3_attention_mask"] = _to_tensor(qwen3_mask, device=local_device)

        return model(**unet_kwargs)

    return _forward


def record_ghost_replay_fingerprint(
    *,
    config_snapshot: Mapping[str, Any] | None,
    model_arch: str = "",
    model_path: str = "",
    vae_path: str = "",
    anima_qwen3_path: str = "",
    anima_t5_tokenizer_path: str = "",
    prompt: Any = "",
    timesteps: Any = "200,800",
    resolution: Any = "",
    proj_dim: int = 128,
    target_layers: Any = "",
    seed: int = 1337,
    output_path: str = "",
    device: str = "",
    dtype: Any = "auto",
) -> dict[str, Any]:
    frontend_config = dict(config_snapshot or {})
    config = ConfigAdapter.from_frontend_dict(frontend_config)
    arch = _infer_arch(model_arch, config)
    if arch not in _SUPPORTED_ARCHES:
        raise ValueError(f"Ghost Replay 预蒸馏当前只支持: {sorted(_SUPPORTED_ARCHES)}")

    path_bundle = _resolve_model_paths(
        config=config,
        arch=arch,
        model_path=model_path,
        vae_path=vae_path,
        anima_qwen3_path=anima_qwen3_path,
        anima_t5_tokenizer_path=anima_t5_tokenizer_path,
    )
    resolved_model_path = _to_str(path_bundle["model_path"])
    resolved_vae_path = _to_str(path_bundle["vae_path"])
    resolved_qwen3_path = _to_str(path_bundle["anima_qwen3_path"])
    resolved_t5_tokenizer_path = _to_str(path_bundle["anima_t5_tokenizer_path"])
    if not resolved_model_path:
        raise ValueError("缺少教师模型路径。")

    validate_path(resolved_model_path, must_exist=True, allow_files=True, allow_dirs=True)
    if resolved_vae_path:
        validate_path(resolved_vae_path, must_exist=True, allow_files=True, allow_dirs=True)
    if resolved_qwen3_path:
        validate_path(resolved_qwen3_path, must_exist=True, allow_files=True, allow_dirs=True)
    if resolved_t5_tokenizer_path:
        validate_path(resolved_t5_tokenizer_path, must_exist=True, allow_files=True, allow_dirs=True)

    resolved_device, device_notes = _resolve_device(device or _to_str(getattr(config, "device", "")))
    resolved_dtype, dtype_notes = _resolve_dtype(dtype or _to_str(getattr(config, "mixed_precision", "")), resolved_device)
    width, height = _parse_resolution(
        resolution or getattr(config, "resolution", ""),
        default=_default_resolution_for_arch(arch),
    )
    prompt_list = _parse_prompts(prompt)
    timestep_list = _parse_timesteps(timesteps)
    target_layer_list = _split_patterns(target_layers or getattr(config, "lulynx_anchor_layers", ""))
    resolved_output = _resolve_output_path(
        output_path,
        config=config,
        arch=arch,
        model_path=resolved_model_path,
    )

    loader = ModelLoader(device=resolved_device, dtype=resolved_dtype)
    loaded_model = None
    try:
        if arch == "anima":
            from backend.core.lulynx_trainer.anima_loader import load_anima_model

            loaded_model, report = load_anima_model(
                model_path=resolved_model_path,
                qwen3_path=resolved_qwen3_path,
                t5_tokenizer_path=resolved_t5_tokenizer_path,
                vae_path=resolved_vae_path,
                device=resolved_device,
                dtype=resolved_dtype,
                disable_mmap=bool(getattr(config, "disable_mmap_load_safetensors", False)),
            )
            loaded_model.anima_load_report = report
        elif arch == "newbie":
            patched = config.model_copy(
                update={
                    "pretrained_model_name_or_path": resolved_model_path,
                    "newbie_diffusers_path": resolved_model_path,
                    "newbie_vae_path": resolved_vae_path or getattr(config, "newbie_vae_path", ""),
                }
            )
            from backend.core.lulynx_trainer.newbie_loader import load_newbie_from_config

            loaded_model = load_newbie_from_config(
                patched,
                device=resolved_device,
                dtype=resolved_dtype,
            )
        else:
            loaded_model = loader.load(
                resolved_model_path,
                model_arch=arch,
                vae_path=resolved_vae_path or None,
            )

        _move_recording_modules(loaded_model, resolved_device, resolved_dtype)
        sample_inputs = _build_sample_inputs(
            prompts=prompt_list,
            model=loaded_model,
            arch=arch,
            width=width,
            height=height,
            device=resolved_device,
            dtype=resolved_dtype,
            seed=int(seed or 1337),
        )
        recorder = GhostRecorder(
            proj_dim=max(int(proj_dim or 128), 1),
            target_layers=target_layer_list or None,
            device=resolved_device,
        )
        metadata = {
            "model_arch": arch,
            "model_path": resolved_model_path,
            "vae_path": resolved_vae_path,
            "qwen3_path": resolved_qwen3_path,
            "t5_tokenizer_path": resolved_t5_tokenizer_path,
            "resolution": [width, height],
            "prompts": prompt_list,
            "timesteps": timestep_list,
            "device": resolved_device,
            "dtype": str(resolved_dtype).replace("torch.", ""),
            "target_layers": target_layer_list,
        }
        stats = recorder.record(
            loaded_model.unet,
            sample_inputs,
            timesteps=timestep_list,
            forward_fn=_build_forward_fn(loaded_model=loaded_model, dtype=resolved_dtype),
            metadata=metadata,
            strict=False,
        )
        if int(stats.get("layers", 0) or 0) <= 0:
            raise RuntimeError("Ghost Replay 录制没有捕获到任何层，请尝试调整目标层或模型类型。")
        recorder.save(str(resolved_output))
        inspect_report = inspect_ghost_fingerprint(resolved_output)
        return {
            "success": True,
            "output_path": str(resolved_output),
            "record_stats": stats,
            "inspect": inspect_report,
            "notes": [*device_notes, *dtype_notes],
            "request_summary": {
                "model_arch": arch,
                "model_path": resolved_model_path,
                "vae_path": resolved_vae_path,
                "resolution": [width, height],
                "prompt_count": len(prompt_list),
                "timesteps": timestep_list,
                "proj_dim": max(int(proj_dim or 128), 1),
                "target_layers": target_layer_list,
                "device": resolved_device,
                "dtype": str(resolved_dtype).replace("torch.", ""),
            },
        }
    finally:
        if loaded_model is not None:
            _release_model(loaded_model)
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def run_ghost_replay_record_job(
    *,
    job: Any,
    config_snapshot: Mapping[str, Any] | None,
    model_arch: str = "",
    model_path: str = "",
    vae_path: str = "",
    anima_qwen3_path: str = "",
    anima_t5_tokenizer_path: str = "",
    prompt: Any = "",
    timesteps: Any = "200,800",
    resolution: Any = "",
    proj_dim: int = 128,
    target_layers: Any = "",
    seed: int = 1337,
    output_path: str = "",
    device: str = "",
    dtype: Any = "auto",
    progress_callback=None,
    cancel_check=None,
) -> dict[str, Any]:
    progress_callback = progress_callback or (lambda *_args, **_kwargs: None)
    cancel_check = cancel_check or (lambda: False)

    _set_job_metadata(job, stage="validating", stage_label="校验输入", progress_message="正在整理预蒸馏参数...")
    progress_callback(0, 4)
    if cancel_check():
        raise RuntimeError("Ghost Replay 预蒸馏已取消。")

    _set_job_metadata(job, stage="loading", stage_label="加载模型", progress_message="正在加载教师模型...")
    progress_callback(1, 4)
    if cancel_check():
        raise RuntimeError("Ghost Replay 预蒸馏已取消。")

    result = record_ghost_replay_fingerprint(
        config_snapshot=config_snapshot,
        model_arch=model_arch,
        model_path=model_path,
        vae_path=vae_path,
        anima_qwen3_path=anima_qwen3_path,
        anima_t5_tokenizer_path=anima_t5_tokenizer_path,
        prompt=prompt,
        timesteps=timesteps,
        resolution=resolution,
        proj_dim=proj_dim,
        target_layers=target_layers,
        seed=seed,
        output_path=output_path,
        device=device,
        dtype=dtype,
    )

    _set_job_metadata(
        job,
        stage="saving",
        stage_label="写入指纹",
        progress_message="正在保存 .lulynx 指纹...",
        output_path=result.get("output_path", ""),
    )
    progress_callback(3, 4)
    if cancel_check():
        raise RuntimeError("Ghost Replay 预蒸馏已取消。")

    _set_job_metadata(
        job,
        stage="completed",
        stage_label="完成",
        progress_message="Ghost Replay 指纹录制完成。",
        output_path=result.get("output_path", ""),
        result=result,
        inspect=result.get("inspect", {}),
        record_stats=result.get("record_stats", {}),
        notes=result.get("notes", []),
    )
    progress_callback(4, 4)
    return result
