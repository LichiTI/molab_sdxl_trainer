"""Experimental SDXL LCM/Turbo LoRA runner.

The default path is a safe contract pass that writes scheduler-aware
safetensors metadata.  With explicit confirmation, the runner can execute a
tiny SDXL diffusers LoRA smoke.  That smoke validates loading, optional teacher
LoRA activation, LoRA injection, gradients, and saving.  The LCM consistency
probe is an early teacher-student objective; it is not yet final quality
LCM/Turbo distillation.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path
from typing import Any

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
_TEACHER_LORA_SCOPES = {"unet_only", "unet_and_text_encoder_experimental"}


def _load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config root must be an object")
    return data


def _normalize_teacher_lora_scope(config: dict[str, Any]) -> str:
    value = str(config.get("teacher_lora_scope") or "unet_only").strip().lower()
    aliases = {
        "full": "unet_and_text_encoder_experimental",
        "all": "unet_and_text_encoder_experimental",
        "unet_text_encoder": "unet_and_text_encoder_experimental",
        "unet_plus_text_encoder": "unet_and_text_encoder_experimental",
    }
    value = aliases.get(value, value)
    if value not in _TEACHER_LORA_SCOPES:
        raise RuntimeError(f"Unsupported teacher_lora_scope: {value}")
    return value


def _normalize_seed(config: dict[str, Any]) -> int:
    raw = config.get("seed", 42)
    if raw is None or str(raw).strip() == "":
        return 42
    try:
        seed = int(raw)
    except (TypeError, ValueError) as exc:
        raise RuntimeError("seed must be an integer") from exc
    if seed < 0:
        raise RuntimeError("seed must be zero or a positive integer")
    return seed


def _make_seeded_generator(torch_module: Any, device: Any, seed: int) -> Any:
    if seed <= 0:
        return None
    torch_module.manual_seed(seed)
    if getattr(torch_module, "cuda", None) is not None and torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(seed)
    for generator_device in (device, None):
        try:
            if generator_device is None:
                return torch_module.Generator().manual_seed(seed)
            return torch_module.Generator(device=generator_device).manual_seed(seed)
        except Exception:
            continue
    return None


def _sample_latent_distribution(latent_dist: Any, generator: Any) -> Any:
    if generator is not None:
        try:
            return latent_dist.sample(generator=generator)
        except (TypeError, RuntimeError):
            pass
    return latent_dist.sample()


def _randn_like(tensor: Any, torch_module: Any, generator: Any) -> Any:
    if generator is not None:
        try:
            return torch_module.randn(
                tensor.shape,
                device=tensor.device,
                dtype=tensor.dtype,
                layout=tensor.layout,
                generator=generator,
            )
        except (TypeError, RuntimeError):
            pass
    return torch_module.randn_like(tensor)


def _randint(torch_module: Any, low: int, high: int, size: tuple[int, ...], device: Any, generator: Any) -> Any:
    if generator is not None:
        try:
            return torch_module.randint(low, high, size, device=device, generator=generator)
        except (TypeError, RuntimeError):
            pass
    return torch_module.randint(low, high, size, device=device)


def _write_safetensors_metadata_stub(path: Path, metadata: dict[str, Any]) -> None:
    header = {"__metadata__": {str(k): str(v) for k, v in metadata.items()}}
    payload = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(len(payload).to_bytes(8, "little"))
        fh.write(payload)


def _find_first_training_sample(train_data_dir: Path) -> tuple[Path, str]:
    if train_data_dir.is_file() and train_data_dir.suffix.lower() in _IMAGE_SUFFIXES:
        image_path = train_data_dir
    else:
        images = sorted(
            path
            for path in train_data_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES
        )
        if not images:
            raise RuntimeError(f"No training images found in {train_data_dir}")
        image_path = images[0]
    caption = image_path.stem.replace("_", " ").replace("-", " ")
    for caption_path in (image_path.with_suffix(".txt"), image_path.with_suffix(".caption")):
        if caption_path.is_file():
            text = caption_path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                caption = text
                break
    return image_path, caption


def _load_image_tensor(image_path: Path, size: int, torch_module: Any, device: Any, dtype: Any):
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    data = torch_module.ByteTensor(torch_module.ByteStorage.from_buffer(image.tobytes()))
    data = data.reshape(size, size, 3).permute(2, 0, 1).float().div(127.5).sub(1.0)
    return data.unsqueeze(0).to(device=device, dtype=dtype)


@contextlib.contextmanager
def _transformers5_clip_text_model_compat():
    try:
        from transformers import CLIPTextModel
    except Exception:
        yield
        return
    try:
        import diffusers.loaders.single_file_utils as single_file_utils
    except Exception:
        single_file_utils = None

    had_text_model = hasattr(CLIPTextModel, "text_model")
    original_text_model = getattr(CLIPTextModel, "text_model", None)
    original_meta_loader = getattr(single_file_utils, "load_model_dict_into_meta", None) if single_file_utils else None
    if not had_text_model:
        CLIPTextModel.text_model = property(lambda self: self)  # type: ignore[attr-defined]
    if callable(original_meta_loader) and single_file_utils is not None:
        def _load_model_dict_into_meta_compat(model: Any, state_dict: dict[str, Any], *args: Any, **kwargs: Any):
            if model.__class__.__name__ == "CLIPTextModel":
                try:
                    model_keys = set(model.state_dict().keys())
                except Exception:
                    model_keys = set()
                if model_keys and not any(key.startswith("text_model.") for key in model_keys):
                    state_dict = {
                        key.removeprefix("text_model."): value
                        for key, value in state_dict.items()
                    }
            return original_meta_loader(model, state_dict, *args, **kwargs)

        single_file_utils.load_model_dict_into_meta = _load_model_dict_into_meta_compat
    try:
        yield
    finally:
        if callable(original_meta_loader) and single_file_utils is not None:
            single_file_utils.load_model_dict_into_meta = original_meta_loader
        if not had_text_model:
            try:
                delattr(CLIPTextModel, "text_model")
            except Exception:
                pass
        else:
            CLIPTextModel.text_model = original_text_model  # type: ignore[assignment]


def _load_sdxl_pipeline(config: dict[str, Any], torch_module: Any, dtype: Any):
    from diffusers import StableDiffusionXLPipeline

    base_model_path = str(config.get("base_model_path") or "")
    if not base_model_path:
        raise ValueError("base_model_path is required")
    path = Path(base_model_path)
    kwargs = {"torch_dtype": dtype, "local_files_only": True}
    if path.is_file() and path.suffix.lower() in {".safetensors", ".ckpt"}:
        if not hasattr(StableDiffusionXLPipeline, "from_single_file"):
            raise RuntimeError("This diffusers version cannot load single-file SDXL checkpoints")
        with _transformers5_clip_text_model_compat():
            pipe = StableDiffusionXLPipeline.from_single_file(str(path), **kwargs)
    else:
        pipe = StableDiffusionXLPipeline.from_pretrained(str(path), **kwargs)
    if config.get("vae_path"):
        from diffusers import AutoencoderKL

        pipe.vae = AutoencoderKL.from_pretrained(str(config["vae_path"]), torch_dtype=dtype, local_files_only=True)
    return pipe


def _adapt_text_encoder_lora_keys_for_target(
    prefix: str,
    target: Any,
    text_state_dict: dict[str, Any],
    text_alphas: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Handle Transformers 5 CLIPTextModel key layout changes.

    Diffusers converts Kohya SDXL TE1 keys to `text_encoder.text_model.*`.
    In the local Transformers 5 path, `CLIPTextModel` exposes modules directly
    under `encoder.*`, while `CLIPTextModelWithProjection` still has
    `text_model.*`.  Keep TE2 untouched and strip only the extra TE1 wrapper
    when the target model truly has no `text_model` module.
    """

    if hasattr(target, "text_model"):
        return text_state_dict, text_alphas, ""

    nested_prefix = f"{prefix}.text_model."
    if not any(str(key).startswith(nested_prefix) for key in text_state_dict):
        return text_state_dict, text_alphas, ""

    def _strip(mapping: dict[str, Any]) -> dict[str, Any]:
        adapted: dict[str, Any] = {}
        for key, value in mapping.items():
            key_text = str(key)
            if key_text.startswith(nested_prefix):
                adapted[f"{prefix}.{key_text[len(nested_prefix):]}"] = value
            else:
                adapted[key_text] = value
        return adapted

    return _strip(text_state_dict), _strip(text_alphas), "stripped_text_model_prefix"


def _text_encoder_lora_load_target(prefix: str, target: Any) -> tuple[Any, str]:
    """Pick the module whose named_modules match converted LoRA keys.

    In the local Transformers 5 SDXL path, TE2 is exposed as
    `CLIPTextModelWithProjection` with an inner `text_model`; PEFT reports
    missing/unexpected keys when loading TE2 LoRA on the outer wrapper.  Loading
    onto the inner text model matches the module names actually used at runtime.
    """

    if prefix == "text_encoder_2":
        inner = getattr(target, "text_model", None)
        if inner is not None:
            return inner, "text_model"
    return target, ""


def _load_optional_teacher_lora(pipe: Any, config: dict[str, Any]) -> dict[str, Any]:
    teacher_lora_path = str(config.get("teacher_lora_path") or "").strip()
    requested_scope = _normalize_teacher_lora_scope(config)
    diagnostics: dict[str, Any] = {
        "requested_scope": requested_scope if teacher_lora_path else "",
        "actual_scope": "",
        "loaded": False,
        "components": {},
        "warnings": [],
    }
    if not teacher_lora_path:
        return diagnostics
    path = Path(teacher_lora_path)
    if not path.exists():
        raise RuntimeError(f"Teacher LoRA not found: {teacher_lora_path}")
    try:
        try:
            state_dict, network_alphas, metadata = pipe.__class__.lora_state_dict(
                str(path),
                unet_config=getattr(pipe.unet, "config", None),
                return_lora_metadata=True,
            )
        except TypeError:
            state_dict, network_alphas = pipe.__class__.lora_state_dict(
                str(path),
                unet_config=getattr(pipe.unet, "config", None),
            )
            metadata = None
        unet_name = getattr(pipe.__class__, "unet_name", "unet")
        unet_prefix = f"{unet_name}."
        network_alphas = network_alphas or {}
        component_counts = {
            "unet": sum(1 for key in state_dict if str(key).startswith(unet_prefix)),
            "text_encoder": sum(1 for key in state_dict if str(key).startswith("text_encoder.")),
            "text_encoder_2": sum(1 for key in state_dict if str(key).startswith("text_encoder_2.")),
        }
        diagnostics["component_counts"] = component_counts
        unet_state_dict = {
            key: value
            for key, value in state_dict.items()
            if str(key).startswith(unet_prefix)
        }
        if not unet_state_dict:
            raise RuntimeError("Teacher LoRA does not contain UNet weights")
        unet_alphas = {
            key: value
            for key, value in network_alphas.items()
            if str(key).startswith(unet_prefix)
        }
        pipe.__class__.load_lora_into_unet(
            unet_state_dict,
            unet_alphas,
            pipe.unet,
            adapter_name="teacher",
            _pipeline=pipe,
            metadata=metadata,
        )
        loaded_scopes = ["unet"]
        diagnostics["loaded"] = True
        diagnostics["components"]["unet"] = {
            "status": "loaded",
            "weights": len(unet_state_dict),
            "alphas": len(unet_alphas),
        }

        if requested_scope == "unet_and_text_encoder_experimental":
            text_encoder_failed = False
            for prefix, attr in (("text_encoder", "text_encoder"), ("text_encoder_2", "text_encoder_2")):
                target = getattr(pipe, attr, None)
                prefix_dot = f"{prefix}."
                text_state_dict = {
                    key: value
                    for key, value in state_dict.items()
                    if str(key).startswith(prefix_dot)
                }
                text_alphas = {
                    key: value
                    for key, value in network_alphas.items()
                    if str(key).startswith(prefix_dot)
                }
                if not text_state_dict:
                    diagnostics["components"][prefix] = {"status": "missing", "weights": 0, "alphas": 0}
                    continue
                if target is None:
                    diagnostics["components"][prefix] = {
                        "status": "missing_target",
                        "weights": len(text_state_dict),
                        "alphas": len(text_alphas),
                    }
                    continue
                if prefix == "text_encoder_2" and text_encoder_failed:
                    diagnostics["components"][prefix] = {
                        "status": "skipped_after_text_encoder_failure",
                        "weights": len(text_state_dict),
                        "alphas": len(text_alphas),
                    }
                    continue
                load_target, load_target_path = _text_encoder_lora_load_target(prefix, target)
                text_state_dict, text_alphas, key_compat = _adapt_text_encoder_lora_keys_for_target(
                    prefix,
                    load_target,
                    text_state_dict,
                    text_alphas,
                )
                try:
                    pipe.__class__.load_lora_into_text_encoder(
                        text_state_dict,
                        text_alphas,
                        load_target,
                        prefix=prefix,
                        adapter_name="teacher",
                        _pipeline=pipe,
                        metadata=metadata,
                    )
                    loaded_scopes.append(prefix)
                    diagnostics["components"][prefix] = {
                        "status": "loaded",
                        "weights": len(text_state_dict),
                        "alphas": len(text_alphas),
                        "key_compat": key_compat,
                        "load_target": load_target_path,
                    }
                except Exception as exc:
                    text_encoder_failed = text_encoder_failed or prefix == "text_encoder"
                    diagnostics["components"][prefix] = {
                        "status": "failed",
                        "weights": len(text_state_dict),
                        "alphas": len(text_alphas),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    diagnostics["warnings"].append(
                        f"{prefix} teacher LoRA failed to load; continuing with loaded components only"
                    )

        diagnostics["actual_scope"] = "_".join(loaded_scopes) if loaded_scopes != ["unet"] else "unet_only"
        return diagnostics
    except TypeError as exc:
        raise RuntimeError(
            "Teacher LoRA smoke requires a diffusers version with named adapter support"
        ) from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to load teacher LoRA: {teacher_lora_path}") from exc


def _prepare_lora(pipe: Any, config: dict[str, Any]) -> Any:
    try:
        from peft import LoraConfig
    except Exception as exc:  # pragma: no cover - depends on runtime extras
        raise RuntimeError("Real Turbo LoRA smoke requires peft to be installed in the selected runtime") from exc

    rank = int(config.get("network_dim") or 16)
    alpha = int(config.get("network_alpha") or rank)
    dropout = float(config.get("network_dropout") or 0.0)
    target_mode = str(config.get("target_modules") or "unet_attention")
    targets = ["to_q", "to_k", "to_v", "to_out.0"]
    if target_mode == "unet_attention_and_mlp":
        targets.extend(["ff.net.0.proj", "ff.net.2"])
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        init_lora_weights="gaussian",
        target_modules=targets,
        lora_dropout=dropout,
    )
    try:
        pipe.unet.add_adapter(lora_config, adapter_name="student")
    except TypeError:
        pipe.unet.add_adapter(lora_config)
    pipe.unet.train()
    named_params = list(pipe.unet.named_parameters())
    has_student_adapter = any("student" in name.lower() for name, _param in named_params)
    for name, param in named_params:
        lowered = name.lower()
        if has_student_adapter:
            param.requires_grad_("student" in lowered)
        else:
            param.requires_grad_((("lora" in lowered) or ("adapter" in lowered)) and "teacher" not in lowered)
    trainable = [param for param in pipe.unet.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable LoRA parameters were created")
    return trainable


def _set_adapters_enabled(unet: Any, enabled: bool) -> bool:
    method_names = ("enable_adapters",) if enabled else ("disable_adapters",)
    for method_name in method_names:
        method = getattr(unet, method_name, None)
        if callable(method):
            try:
                method()
                return True
            except TypeError:
                try:
                    method(enabled)
                    return True
                except Exception:
                    pass
            except Exception:
                pass
    return False


def _set_active_adapter(pipe: Any, adapter_name: str) -> bool:
    for target in (pipe, getattr(pipe, "unet", None)):
        if target is None:
            continue
        method = getattr(target, "set_adapters", None)
        if not callable(method):
            continue
        for value in (adapter_name, [adapter_name]):
            weights = 1.0 if isinstance(value, str) else [1.0]
            try:
                method(value)
                return True
            except TypeError:
                try:
                    method(value, adapter_weights=weights)
                    return True
                except Exception:
                    pass
                try:
                    method(value, weights)
                    return True
                except Exception:
                    pass
            except Exception:
                pass
    return False


def _encode_prompt(pipe: Any, prompt: str, device: Any, torch_module: Any):
    try:
        encoded = pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )
    except TypeError:
        encoded = pipe.encode_prompt(
            prompt,
            device,
            1,
            False,
        )
    prompt_embeds = encoded[0]
    pooled = encoded[2] if len(encoded) > 2 else None
    if pooled is None:
        pooled = torch_module.zeros((1, prompt_embeds.shape[-1]), device=device, dtype=prompt_embeds.dtype)
    return prompt_embeds, pooled


def _predict_x0(noise_scheduler: Any, noisy_latents: Any, timesteps: Any, model_pred: Any):
    import torch

    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=noisy_latents.device, dtype=noisy_latents.dtype)
    alpha_prod_t = alphas_cumprod[timesteps].flatten()
    while len(alpha_prod_t.shape) < len(noisy_latents.shape):
        alpha_prod_t = alpha_prod_t.unsqueeze(-1)
    beta_prod_t = 1 - alpha_prod_t
    return (noisy_latents - beta_prod_t.sqrt() * model_pred) / alpha_prod_t.sqrt().clamp(min=torch.finfo(noisy_latents.dtype).eps)


def _add_noise_at_stride(noise_scheduler: Any, x0: Any, noise: Any, timesteps: Any, stride: int):
    import torch

    target_timesteps = torch.clamp(timesteps - int(stride), min=0)
    return noise_scheduler.add_noise(x0, noise, target_timesteps), target_timesteps


def _assert_finite_tensors(tensors: dict[str, Any], label: str) -> None:
    bad_keys = []
    for key, value in tensors.items():
        detached = value.detach() if hasattr(value, "detach") else value
        isfinite = getattr(detached, "isfinite", None)
        if callable(isfinite) and not bool(isfinite().all()):
            bad_keys.append(str(key))
            if len(bad_keys) >= 8:
                break
    if bad_keys:
        raise RuntimeError(f"Non-finite tensors in {label}: {bad_keys}")


def _assert_finite_trainable(unet: Any, label: str, *, include_grad: bool = False) -> None:
    import torch

    bad_params = []
    bad_grads = []
    for name, param in unet.named_parameters():
        if not getattr(param, "requires_grad", False):
            continue
        if not bool(torch.isfinite(param.detach()).all()):
            bad_params.append(str(name))
        grad = getattr(param, "grad", None)
        if include_grad and grad is not None and not bool(torch.isfinite(grad.detach()).all()):
            bad_grads.append(str(name))
        if len(bad_params) + len(bad_grads) >= 8:
            break
    if bad_params or bad_grads:
        raise RuntimeError(f"Non-finite {label}: params={bad_params}, grads={bad_grads}")


def _tensor_stats(value: Any) -> dict[str, Any]:
    import torch

    detached = value.detach()
    finite = torch.isfinite(detached)
    total = int(detached.numel())
    finite_count = int(finite.sum().item()) if total else 0
    stats: dict[str, Any] = {
        "shape": [int(dim) for dim in detached.shape],
        "dtype": str(detached.dtype).replace("torch.", ""),
        "total": total,
        "finite_ratio": float(finite_count / total) if total else 1.0,
    }
    if finite_count:
        finite_values = detached[finite].float()
        stats.update({
            "mean": float(finite_values.mean().item()),
            "std": float(finite_values.std(unbiased=False).item()),
            "min": float(finite_values.min().item()),
            "max": float(finite_values.max().item()),
            "abs_max": float(finite_values.abs().max().item()),
        })
    return stats


def _loss_diagnostics(losses: list[float]) -> dict[str, Any]:
    finite_losses = [float(loss) for loss in losses if math.isfinite(float(loss))]
    diagnostics: dict[str, Any] = {
        "count": len(losses),
        "finite_count": len(finite_losses),
        "finite_ratio": float(len(finite_losses) / len(losses)) if losses else 1.0,
    }
    if not finite_losses:
        diagnostics["trend"] = "invalid"
        diagnostics["status"] = "failed"
        return diagnostics

    initial = finite_losses[0]
    final = finite_losses[-1]
    delta = final - initial
    relative_delta = delta / abs(initial) if initial else 0.0
    diagnostics.update({
        "initial": initial,
        "final": final,
        "best": min(finite_losses),
        "worst": max(finite_losses),
        "mean": sum(finite_losses) / len(finite_losses),
        "delta": delta,
        "relative_delta": relative_delta,
    })
    if len(finite_losses) < 2:
        trend = "single_step"
    elif final < initial:
        trend = "improved"
    elif math.isclose(final, initial, rel_tol=1e-4, abs_tol=1e-6):
        trend = "flat"
    else:
        trend = "increased"
    diagnostics["trend"] = trend
    diagnostics["status"] = "passed" if len(finite_losses) == len(losses) else "warning"
    return diagnostics


def _summarize_safetensors_file(path: Path) -> dict[str, Any]:
    try:
        import torch
        from safetensors import safe_open
    except Exception as exc:  # pragma: no cover - depends on runtime extras
        return {"summary_error": f"safetensors summary unavailable: {exc}"}

    if not path.is_file():
        return {"summary_error": f"missing output file: {path}"}

    tensor_count = 0
    total_params = 0
    finite_count = 0
    bad_tensor_count = 0
    dtype_counts: dict[str, int] = {}
    bad_keys: list[str] = []
    try:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            keys = list(handle.keys())
            for key in keys:
                tensor = handle.get_tensor(key)
                tensor_count += 1
                numel = int(tensor.numel())
                total_params += numel
                dtype = str(tensor.dtype).replace("torch.", "").upper()
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + 1
                finite = torch.isfinite(tensor)
                current_finite = int(finite.sum().item()) if numel else 0
                finite_count += current_finite
                if current_finite != numel:
                    bad_tensor_count += 1
                    if len(bad_keys) < 16:
                        bad_keys.append(str(key))
    except Exception as exc:
        return {"summary_error": str(exc)}

    return {
        "tensor_count": tensor_count,
        "total_params": total_params,
        "finite_ratio": float(finite_count / total_params) if total_params else 1.0,
        "bad_count": bad_tensor_count,
        "bad_key_preview": bad_keys,
        "dtypes": dtype_counts,
    }


def _save_student_lora_weights(pipe: Any, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _set_active_adapter(pipe, "student")
    state_dict = None
    try:
        from peft import get_peft_model_state_dict

        state_dict = get_peft_model_state_dict(pipe.unet, adapter_name="student")
    except Exception:
        state_dict = None
    if state_dict is not None:
        _assert_finite_tensors(state_dict, "student LoRA state_dict")
        save_lora_weights = getattr(pipe, "save_lora_weights", None)
        if callable(save_lora_weights):
            save_lora_weights(
                str(output_path.parent),
                unet_lora_layers=state_dict,
                safe_serialization=True,
            )
            generated = output_path.parent / "pytorch_lora_weights.safetensors"
            if generated.is_file() and generated.resolve() != output_path.resolve():
                generated.replace(output_path)
            return
    _assert_finite_trainable(pipe.unet, "student LoRA parameters before fallback save")
    try:
        pipe.unet.save_attn_procs(str(output_path.parent), safe_serialization=True, adapter_name="student")
    except TypeError:
        pipe.unet.save_attn_procs(str(output_path.parent), safe_serialization=True)
    generated = output_path.parent / "pytorch_lora_weights.safetensors"
    if generated.is_file() and generated.resolve() != output_path.resolve():
        generated.replace(output_path)


def _run_real_short_smoke(config: dict[str, Any], output_path: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as F
        from diffusers import DDPMScheduler
    except Exception as exc:  # pragma: no cover - depends on runtime extras
        raise RuntimeError("Real Turbo LoRA smoke requires torch and diffusers in the selected runtime") from exc

    if not bool(config.get("confirm_real_run", False)):
        raise RuntimeError("confirm_real_run=true is required for real Turbo LoRA smoke")

    max_train_steps = int(config.get("max_train_steps") or 1)
    batch_size = int(config.get("batch_size") or 1)
    if max_train_steps < 1 or max_train_steps > 4:
        raise RuntimeError("Real Turbo LoRA smoke is capped at 1-4 steps")
    if batch_size != 1:
        raise RuntimeError("Real Turbo LoRA smoke currently requires batch_size=1")

    precision = str(config.get("mixed_precision") or "bf16").lower()
    if precision == "bf16" and torch.cuda.is_available():
        dtype = torch.bfloat16
    elif precision == "fp16" and torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = _normalize_seed(config)
    generator = _make_seeded_generator(torch, device, seed)

    pipe = _load_sdxl_pipeline(config, torch, dtype)
    pipe.to(device)
    pipe.text_encoder.requires_grad_(False)
    if getattr(pipe, "text_encoder_2", None) is not None:
        pipe.text_encoder_2.requires_grad_(False)
    pipe.vae.requires_grad_(False)
    pipe.unet.requires_grad_(False)

    teacher_lora_diagnostics = _load_optional_teacher_lora(pipe, config)
    teacher_lora_loaded = bool(teacher_lora_diagnostics.get("loaded"))
    teacher_lora_load_scope = str(teacher_lora_diagnostics.get("actual_scope") or "")
    teacher_lora_requested_scope = str(teacher_lora_diagnostics.get("requested_scope") or "")
    trainable = _prepare_lora(pipe, config)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(config.get("learning_rate") or 1e-4),
        eps=float(config.get("optimizer_eps") or 1e-6),
    )
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    train_data_dir = Path(str(config.get("train_data_dir") or "")).resolve()
    image_path, caption = _find_first_training_sample(train_data_dir)
    resolution = min(512, int(config.get("resolution") or 512))
    resolution = max(256, resolution)
    if resolution % 8 != 0:
        resolution = int(math.floor(resolution / 8) * 8)

    image_tensor = _load_image_tensor(image_path, resolution, torch, device, dtype)
    prompt_embeds, pooled_prompt_embeds = _encode_prompt(pipe, caption, device, torch)
    time_ids = torch.tensor(
        [[resolution, resolution, 0, 0, resolution, resolution]],
        device=device,
        dtype=prompt_embeds.dtype,
    )

    losses: list[float] = []
    step_records: list[dict[str, Any]] = []
    objective = str(config.get("real_objective") or "lcm_consistency_probe")
    if objective not in {"epsilon_lora_probe", "lcm_consistency_probe"}:
        raise RuntimeError(f"Unsupported real objective: {objective}")
    stride = max(1, int(config.get("lcm_target_stride") or 80))
    for step in range(max_train_steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            latents = _sample_latent_distribution(pipe.vae.encode(image_tensor).latent_dist, generator)
            latents = latents * float(getattr(pipe.vae.config, "scaling_factor", 0.18215))
            noise = _randn_like(latents, torch, generator)
            timesteps = _randint(
                torch,
                0,
                int(noise_scheduler.config.num_train_timesteps),
                (latents.shape[0],),
                device=device,
                generator=generator,
            ).long()
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            teacher_target_x0 = None
            target_timesteps = None
            if objective == "lcm_consistency_probe":
                if teacher_lora_loaded:
                    if not _set_active_adapter(pipe, "teacher"):
                        raise RuntimeError("Failed to activate teacher LoRA adapter")
                elif not _set_adapters_enabled(pipe.unet, False):
                    raise RuntimeError("LCM consistency probe requires a diffusers/PEFT UNet that supports disabling adapters")
                teacher_pred = pipe.unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": time_ids},
                ).sample
                teacher_x0 = _predict_x0(noise_scheduler, noisy_latents, timesteps, teacher_pred)
                teacher_target_x0, target_timesteps = _add_noise_at_stride(
                    noise_scheduler,
                    teacher_x0,
                    noise,
                    timesteps,
                    stride,
                )
                teacher_target_x0 = _predict_x0(
                    noise_scheduler,
                    teacher_target_x0,
                    target_timesteps,
                    noise,
                ).detach()
                if not _set_active_adapter(pipe, "student") and not _set_adapters_enabled(pipe.unet, True):
                    raise RuntimeError("Failed to activate student LoRA adapter after teacher pass")

        if not _set_active_adapter(pipe, "student") and teacher_lora_loaded:
            raise RuntimeError("Failed to activate student LoRA adapter for student pass")
        model_pred = pipe.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=prompt_embeds,
            added_cond_kwargs={"text_embeds": pooled_prompt_embeds, "time_ids": time_ids},
        ).sample
        if objective == "lcm_consistency_probe":
            student_x0 = _predict_x0(noise_scheduler, noisy_latents, timesteps, model_pred)
            loss = F.mse_loss(student_x0.float(), teacher_target_x0.float(), reduction="mean")
            prediction_stats = _tensor_stats(student_x0)
            target_stats = _tensor_stats(teacher_target_x0)
        else:
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            prediction_stats = _tensor_stats(model_pred)
            target_stats = _tensor_stats(noise)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        _assert_finite_trainable(pipe.unet, f"student LoRA gradients at step {step + 1}", include_grad=True)
        optimizer.step()
        _assert_finite_trainable(pipe.unet, f"student LoRA parameters at step {step + 1}")
        losses.append(float(loss.detach().cpu()))
        record = {
            "step": step + 1,
            "loss": losses[-1],
            "objective": objective,
            "timestep": int(timesteps[0].detach().cpu().item()),
            "target_timestep": int(target_timesteps[0].detach().cpu().item()) if target_timesteps is not None else None,
            "prediction": prediction_stats,
            "target": target_stats,
            "latent": _tensor_stats(latents),
            "noisy_latent": _tensor_stats(noisy_latents),
        }
        step_records.append(record)
        print(
            json.dumps(
                {
                    "event": "step",
                    "step": record["step"],
                    "loss": record["loss"],
                    "objective": objective,
                    "timestep": record["timestep"],
                    "target_timestep": record["target_timestep"],
                    "prediction_finite_ratio": prediction_stats["finite_ratio"],
                    "target_finite_ratio": target_stats["finite_ratio"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    _save_student_lora_weights(pipe, output_path)
    output_summary = _summarize_safetensors_file(output_path)
    loss_summary = _loss_diagnostics(losses)
    output_finite_ratio = float(output_summary.get("finite_ratio", 0.0)) if "finite_ratio" in output_summary else 0.0
    smoke_status = (
        "passed"
        if loss_summary.get("status") == "passed"
        and output_summary.get("bad_count", 1) == 0
        and math.isclose(output_finite_ratio, 1.0, rel_tol=0.0, abs_tol=0.0)
        and not teacher_lora_diagnostics.get("warnings")
        else "warning"
    )
    metadata.update({
        "lulynx.seed": seed,
        "lulynx.teacher_lora_requested_scope": teacher_lora_requested_scope,
        "lulynx.teacher_lora_load_scope": teacher_lora_load_scope,
        "lulynx.teacher_lora_text_encoder_status": (
            teacher_lora_diagnostics.get("components", {})
            .get("text_encoder", {})
            .get("status", "")
        ),
        "lulynx.teacher_lora_text_encoder_key_compat": (
            teacher_lora_diagnostics.get("components", {})
            .get("text_encoder", {})
            .get("key_compat", "")
        ),
        "lulynx.teacher_lora_text_encoder_load_target": (
            teacher_lora_diagnostics.get("components", {})
            .get("text_encoder", {})
            .get("load_target", "")
        ),
        "lulynx.teacher_lora_text_encoder_2_status": (
            teacher_lora_diagnostics.get("components", {})
            .get("text_encoder_2", {})
            .get("status", "")
        ),
        "lulynx.teacher_lora_text_encoder_2_key_compat": (
            teacher_lora_diagnostics.get("components", {})
            .get("text_encoder_2", {})
            .get("key_compat", "")
        ),
        "lulynx.teacher_lora_text_encoder_2_load_target": (
            teacher_lora_diagnostics.get("components", {})
            .get("text_encoder_2", {})
            .get("load_target", "")
        ),
        "lulynx.smoke_status": smoke_status,
        "lulynx.loss_initial": loss_summary.get("initial", ""),
        "lulynx.loss_final": loss_summary.get("final", ""),
        "lulynx.loss_trend": loss_summary.get("trend", ""),
        "lulynx.output_finite_ratio": output_summary.get("finite_ratio", ""),
        "lulynx.output_tensor_count": output_summary.get("tensor_count", ""),
        "lulynx.validation_level": "real_short_smoke",
        "lulynx.quality_status": "not_quality_validated",
        "lulynx.quality_note": "short smoke only; not a final quality benchmark",
    })
    diagnostics = {
        "mode": "real_short_smoke",
        "status": smoke_status,
        "validation_level": "real_short_smoke",
        "quality_status": "not_quality_validated",
        "precision": precision,
        "dtype": str(dtype).replace("torch.", ""),
        "device": str(device),
        "seed": seed,
        "seeded_generator": generator is not None,
        "resolution": resolution,
        "batch_size": batch_size,
        "max_train_steps": max_train_steps,
        "teacher_lora_loaded": teacher_lora_loaded,
        "teacher_lora_requested_scope": teacher_lora_requested_scope,
        "teacher_lora_load_scope": teacher_lora_load_scope,
        "teacher_lora": teacher_lora_diagnostics,
        "loss": loss_summary,
        "output": output_summary,
        "steps": step_records,
        "note": "This validates local loading, gradients, finite output, and a tiny consistency objective only.",
    }

    sidecar_path = output_path.with_suffix(output_path.suffix + ".metadata.json")
    sidecar_path.write_text(
        json.dumps(
            {
                **metadata,
                "losses": losses,
                "objective": objective,
                "diagnostics": diagnostics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "status": "success",
        "dry_run": False,
        "objective": objective,
        "output_path": str(output_path),
        "metadata_sidecar": str(sidecar_path),
        "losses": losses,
        "diagnostics": diagnostics,
        "sample": str(image_path),
        "caption": caption,
        "teacher_lora_loaded": teacher_lora_loaded,
        "teacher_lora_requested_scope": teacher_lora_requested_scope,
        "teacher_lora_load_scope": teacher_lora_load_scope,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Lulynx SDXL Turbo/LCM LoRA runner")
    parser.add_argument("--config", required=True, help="Path to launcher-generated runner config JSON")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    output_path = Path(str(config.get("output_path") or "")).resolve()
    if not output_path:
        raise ValueError("output_path is required")

    dry_run = bool(config.get("dry_run", True))
    seed = _normalize_seed(config)
    teacher_lora_path = str(config.get("teacher_lora_path") or "").strip()
    teacher_lora_requested_scope = _normalize_teacher_lora_scope(config) if teacher_lora_path else ""
    metadata = {
        "modelspec.title": output_path.stem,
        "model_type": "lora",
        "lulynx.schema_id": "sdxl-turbo-lora",
        "lulynx.artifact_kind": "acceleration_lora",
        "lulynx.distill_method": config.get("distill_method", "lcm_lora"),
        "lulynx.base_model_path": config.get("base_model_path", ""),
        "lulynx.teacher_lora_path": config.get("teacher_lora_path", ""),
        "lulynx.teacher_lora_requested_scope": teacher_lora_requested_scope,
        "lulynx.teacher_lora_load_scope": "dry_run" if dry_run and teacher_lora_requested_scope else "",
        "lulynx.teacher_scheduler": config.get("teacher_scheduler", ""),
        "lulynx.teacher_steps": config.get("teacher_steps", ""),
        "lulynx.student_scheduler": config.get("student_scheduler", ""),
        "lulynx.student_steps": config.get("student_steps", ""),
        "lulynx.guidance_scale": config.get("guidance_scale", ""),
        "lulynx.timestep_sampling": config.get("timestep_sampling", ""),
        "lulynx.seed": seed,
        "lulynx.network_dim": config.get("network_dim", ""),
        "lulynx.network_alpha": config.get("network_alpha", ""),
        "lulynx.validation_level": "contract_dry_run" if dry_run else "real_short_smoke",
        "lulynx.quality_status": "not_quality_validated",
        "lulynx.quality_note": "contract or short smoke only; not a final quality benchmark",
        "lulynx.recommended_usage": (
            f"{config.get('student_scheduler', 'lcm')} scheduler, "
            f"{config.get('student_steps', 4)} steps, CFG {config.get('guidance_scale', 1.5)}"
        ),
        "lulynx.contract_version": "2",
        "lulynx.dry_run": dry_run,
        "ss_network_module": "networks.lora",
        "ss_base_model_version": "sdxl",
    }
    if config.get("metadata_note"):
        metadata["lulynx.note"] = config.get("metadata_note", "")

    if not dry_run:
        metadata["lulynx.real_objective"] = config.get("real_objective", "lcm_consistency_probe")
        result = _run_real_short_smoke(config, output_path, metadata)
        print(json.dumps(result, ensure_ascii=False))
        return 0

    _write_safetensors_metadata_stub(output_path, metadata)
    print(json.dumps({
        "status": "success",
        "dry_run": dry_run,
        "output_path": str(output_path),
        "metadata_keys": sorted(metadata.keys()),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
