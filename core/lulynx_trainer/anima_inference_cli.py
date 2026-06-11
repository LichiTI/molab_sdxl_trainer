"""Standalone CLI for running inference with trained Anima/Newbie models (Phase 9.6).

Usage::

    python -m core.lulynx_trainer.anima_inference_cli \
        --model_path /path/to/model --prompt "a cat" --output_dir ./out
"""

from __future__ import annotations

import argparse, json, logging
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

try:
    from backend.core.contracts import ArtifactFile, ArtifactManifest, GenerationRequest, GenerationResult, RequestSource, RunContext, RunStatus
    from backend.core.runners import create_generation_registry
except Exception:  # pragma: no cover - keeps direct script execution usable in legacy layouts.
    ArtifactFile = None  # type: ignore[assignment]
    ArtifactManifest = None  # type: ignore[assignment]
    GenerationRequest = None  # type: ignore[assignment]
    GenerationResult = None  # type: ignore[assignment]
    RequestSource = None  # type: ignore[assignment]
    RunContext = None  # type: ignore[assignment]
    RunStatus = None  # type: ignore[assignment]
    create_generation_registry = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_DTYPE_MAP = {
    "fp16": torch.float16, "float16": torch.float16,
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp32": torch.float32, "float32": torch.float32,
}


def _resolve_dtype(name: str) -> torch.dtype:
    d = _DTYPE_MAP.get(name.strip().lower())
    if d is None:
        raise ValueError(f"Unknown dtype '{name}'; choose from {sorted(_DTYPE_MAP)}")
    return d


def _detect_arch(model_path: str) -> str:
    """Auto-detect anima vs newbie from config files and directory layout."""
    p = Path(model_path)
    for sub in (p, p / "unet", p / "transformer"):
        cfg = sub / "config.json" if sub.is_dir() else None
        if cfg and cfg.is_file():
            try:
                arch = json.loads(cfg.read_text("utf-8")).get("model_arch", "")
                if isinstance(arch, str) and arch.strip():
                    return arch.strip().lower()
            except Exception:
                pass
    if p.is_dir() and (p / "transformer").is_dir() and (p / "clip_model").is_dir():
        return "newbie"
    return "anima"


def _load_model(model_path: str, arch: str, device: str, dtype: torch.dtype):
    if arch == "newbie":
        from .newbie_loader import load_newbie_from_config

        p = Path(model_path)
        diffusers = str(p) if p.is_dir() and (p / "unet").is_dir() else ""
        cfg = type("_Cfg", (), dict(
            newbie_diffusers_path=diffusers,
            newbie_transformer_path=str(p / "transformer") if (p / "transformer").is_dir() else "",
            newbie_gemma_model_path=str(p / "text_encoder") if (p / "text_encoder").is_dir() else "",
            newbie_clip_model_path=str(p / "clip_model") if (p / "clip_model").is_dir() else "",
            newbie_vae_path=str(p / "vae") if (p / "vae").is_dir() else "",
            newbie_lora_target="minimal", newbie_gemma3_prompt="", newbie_use_flash_attn2=False,
            newbie_adapter_type="", newbie_target_modules="", newbie_safe_fallback=True,
            newbie_gemma_max_token_length=512, newbie_clip_max_token_length=2048,
            trust_remote_code=True, newbie_run_native_smoke=False,
        ))()
        return load_newbie_from_config(cfg, device=device, dtype=dtype)

    from .anima_loader import load_anima_model
    model, report = load_anima_model(model_path=model_path, device=device, dtype=dtype)
    logger.info("Anima load report: %s", report.summary())
    return model


def _apply_adapter(model, adapter_path: str, arch: str, device: str):
    if not adapter_path:
        return model
    from .lora_injector import LoRAInjector, infer_rank_from_weights
    try:
        from safetensors.torch import load_file as sf_load
        sd = sf_load(adapter_path)
    except Exception:
        sd = torch.load(adapter_path, map_location="cpu", weights_only=True)
    rank = infer_rank_from_weights(sd) or 4
    logger.info("Adapter rank: %d", rank)
    injector = LoRAInjector(rank=rank, model_arch=arch)
    if model.unet is not None:
        injector.inject_unet(model.unet)
        model.unet.to(device)
    if model.text_encoder_1 is not None:
        injector.inject_text_encoder(model.text_encoder_1, name="te1")
    if model.text_encoder_2 is not None:
        injector.inject_text_encoder(model.text_encoder_2, name="te2")
    injector.load_lora(adapter_path)
    logger.info("Adapter loaded from %s", adapter_path)
    return model


def _generate(model, *, prompt, negative_prompt, width, height, steps,
              guidance_scale, seed, sampler_name, device, dtype, batch_size, arch,
              newbie_latent_scale_factor: int = 0,
              smc_cfg: bool = False,
              smc_cfg_lambda: float = 5.0,
              smc_cfg_alpha: float = 0.2,
              tgate_probe: bool = False,
              tgate_start_step: int = 0,
              tgate_min_block: int = 0,
              spectrum_probe: bool = False,
              spectrum_window_size: float = 2.0,
              spectrum_flex_window: float = 0.25,
              spectrum_warmup_steps: int = 6,
              spectrum_stop_caching_step: int = -1,
              smoothcache_probe: bool = False,
              smoothcache_error_threshold: float = 0.08,
              smoothcache_warmup_steps: int = 2,
              cache_seam_backend: str = "none",
              cache_seam_window_size: float = 3.0):
    from .model_family import get_model_family
    from .unified_cache_seam import build_cache_seam, cache_seam_context
    family = get_model_family(arch)
    _accel_backend = str(cache_seam_backend or "none")
    _accel_enabled = _accel_backend.lower() not in {"", "none"}
    images = []
    for idx in range(batch_size):
        img_seed = (seed + idx) if seed is not None else None
        # Fresh cache per image: seeds differ, so caches must not bleed across images.
        _seam = build_cache_seam(
            enabled=_accel_enabled, backend=_accel_backend,
            spectrum_window_size=int(cache_seam_window_size or 3),
        )
        kwargs = dict(
            num_inference_steps=steps, guidance_scale=guidance_scale,
            width=width, height=height, seed=img_seed, device=device,
            dtype=dtype, sampler_name=sampler_name,
            latent_channels=family.latent_channels,
            vae_scaling_factor=family.vae_scaling_factor,
            smc_cfg=smc_cfg,
            smc_cfg_lambda=smc_cfg_lambda,
            smc_cfg_alpha=smc_cfg_alpha,
            tgate_probe=tgate_probe,
            tgate_start_step=tgate_start_step,
            tgate_min_block=tgate_min_block,
            spectrum_probe=spectrum_probe,
            spectrum_window_size=spectrum_window_size,
            spectrum_flex_window=spectrum_flex_window,
            spectrum_warmup_steps=spectrum_warmup_steps,
            spectrum_stop_caching_step=spectrum_stop_caching_step,
            smoothcache_probe=smoothcache_probe,
            smoothcache_error_threshold=smoothcache_error_threshold,
            smoothcache_warmup_steps=smoothcache_warmup_steps,
        )
        with cache_seam_context(_seam):
            if arch == "newbie":
                from .newbie_sampler import sample_newbie
                img = sample_newbie(
                    dit_model=model.unet, vae=model.vae,
                    text_encoder_1=model.text_encoder_1, text_encoder_2=model.text_encoder_2,
                    tokenizer_1=model.tokenizer_1, tokenizer_2=model.tokenizer_2,
                    prompt=prompt, negative_prompt=negative_prompt, **kwargs,
                    latent_scale_factor=newbie_latent_scale_factor,
                )
            else:
                from .anima_sampler import sample_anima
                img = sample_anima(
                    dit_model=model.unet, vae=model.vae,
                    text_encoder=model.text_encoder_1, tokenizer=model.tokenizer_1,
                    prompt=prompt, negative_prompt=negative_prompt, **kwargs,
                )
        if img is not None:
            images.append(img)
        else:
            logger.warning("Sample %d returned None", idx)
    return images


def _maybe_apply_newbie_tensorrt_transformer(model: Any, request: "GenerationRequest", arch: str) -> tuple[Any, dict[str, Any]]:
    if arch != "newbie":
        return model, {"enabled": False, "reason": "not_newbie"}
    extra = request.model_extra or {}
    engine_path = str(extra.get("newbie_tensorrt_engine_path") or extra.get("tensorrt_engine_path") or "").strip()
    enabled = _boolish(extra.get("newbie_tensorrt_enabled") or extra.get("tensorrt_enabled") or bool(engine_path))
    if not enabled:
        return model, {"enabled": False, "reason": "not_requested"}
    if not engine_path:
        raise ValueError("newbie_tensorrt_engine_path is required when newbie_tensorrt_enabled is true")

    from core.base_model_tensorrt.generation_preflight import preflight_newbie_tensorrt_generation_request
    from core.base_model_tensorrt.newbie_export import NewbieStaticShape
    from core.base_model_tensorrt.newbie_generation_adapter import NewbieTensorRtTransformerAdapter
    from core.base_model_tensorrt.runtime_adapter import StaticTransformerRuntimeSpec

    latent_scale = int(extra.get("newbie_tensorrt_vae_scale_factor") or 16)
    tokens = int(extra.get("newbie_tensorrt_tokens") or 512)
    engine_latent_height = int(extra.get("newbie_tensorrt_latent_height") or 64)
    engine_latent_width = int(extra.get("newbie_tensorrt_latent_width") or 64)
    spec = StaticTransformerRuntimeSpec.from_newbie_shape(
        engine_path=engine_path,
        layer_indices=str(extra.get("newbie_tensorrt_layers") or "0-35"),
        shape=NewbieStaticShape(latent_height=engine_latent_height, latent_width=engine_latent_width, tokens=tokens),
        precision=str(extra.get("newbie_tensorrt_precision") or "fp32"),
    )
    preflight = preflight_newbie_tensorrt_generation_request(
        request,
        spec,
        vae_scale_factor=latent_scale,
        positive_tokens=tokens,
        negative_tokens=int(extra.get("newbie_tensorrt_negative_tokens") or 0),
        cfg_strategy=str(extra.get("newbie_tensorrt_cfg_strategy") or "separate_calls"),
    )
    if not preflight.ok:
        raise ValueError("Newbie TensorRT generation preflight failed: " + ", ".join(preflight.blockers))
    old_unet = getattr(model, "unet", None)
    model.unet = NewbieTensorRtTransformerAdapter.from_spec(spec)
    if old_unet is not None:
        del old_unet
        try:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return model, {"enabled": True, "engine_path": engine_path, "latent_scale_factor": latent_scale, "preflight": preflight.to_dict()}


def _boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on", "enabled"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Anima/Newbie inference CLI")
    p.add_argument("--request_json", default="", help="GenerationRequest JSON file")
    p.add_argument("--run_result_json", default="", help="Write dry-run RunResult JSON to this path")
    p.add_argument("--dry_run", action="store_true", help="Validate the request through the generation runner without loading models")
    p.add_argument("--model_path", default="", help="Base model directory or checkpoint")
    p.add_argument("--adapter_path", default="", help="LoRA/LyCORIS adapter weights")
    p.add_argument("--output_dir", default="./inference_output", help="Output directory")
    p.add_argument("--output_name", default="", help="Planned output file name for request-native dry-run")
    p.add_argument("--prompt", default="", help="Positive prompt")
    p.add_argument("--negative_prompt", default="", help="Negative prompt")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--steps", type=int, default=20, help="Denoising steps")
    p.add_argument("--guidance_scale", type=float, default=5.0, help="CFG scale")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--sampler_name", default="euler", help="euler or dpm_solver")
    p.add_argument("--smc_cfg", action="store_true", help="Enable SMC-CFG for cond/uncond CFG combine")
    p.add_argument("--smc_cfg_lambda", type=float, default=5.0, help="SMC-CFG sliding slope")
    p.add_argument("--smc_cfg_alpha", type=float, default=0.2, help="SMC-CFG adaptive gain")
    p.add_argument("--tgate_probe", action="store_true", help="Observe T-GATE cross-attention skip eligibility without skipping")
    p.add_argument("--tgate_start_step", type=int, default=0, help="T-GATE probe warmup step threshold")
    p.add_argument("--tgate_min_block", type=int, default=0, help="T-GATE probe minimum block index")
    p.add_argument("--spectrum_probe", action="store_true", help="Observe Spectrum block-cache eligibility without skipping")
    p.add_argument("--spectrum_window_size", type=float, default=2.0, help="Spectrum probe base cache window size")
    p.add_argument("--spectrum_flex_window", type=float, default=0.25, help="Spectrum probe adaptive window growth")
    p.add_argument("--spectrum_warmup_steps", type=int, default=6, help="Spectrum probe warmup actual-forward steps")
    p.add_argument("--spectrum_stop_caching_step", type=int, default=-1, help="Spectrum probe tail stop step; -1 uses last 3 steps")
    p.add_argument("--dtype", default="bf16", help="fp16/bf16/fp32")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--inference_accel_scheme", default="none", help="Inference accel: none|spectrum|smoothcache (drives the probe+seam pair together)")
    p.add_argument("--arch", default="", help="Override: anima or newbie")
    return p


def build_generation_request_from_args(args: argparse.Namespace | Mapping[str, Any]) -> "GenerationRequest":
    """Normalize CLI arguments into the platform generation request contract."""

    if GenerationRequest is None or RequestSource is None:
        raise RuntimeError("GenerationRequest contract is unavailable in this runtime")

    data = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
    if data.get("request_json"):
        request = load_generation_request_from_file(data["request_json"])
        if data.get("dry_run"):
            request = request.model_copy(update={"dry_run": True})
        return request

    if not data.get("model_path") and not data.get("dry_run"):
        raise ValueError("model_path is required unless --dry_run or --request_json is used")
    seed = data.get("seed")
    request = GenerationRequest.from_legacy_payload(
        {
            "schema_id": "generation.image",
            "model_path": data.get("model_path") or "",
            "adapter_path": data.get("adapter_path") or "",
            "output_dir": data.get("output_dir") or "",
            "prompt": data.get("prompt") or "",
            "negative_prompt": data.get("negative_prompt") or "",
            "width": data.get("width", 1024),
            "height": data.get("height", 1024),
            "steps": data.get("steps", 20),
            "guidance_scale": data.get("guidance_scale", 5.0),
            "seed": -1 if seed is None else seed,
            "sampler": data.get("sampler_name") or data.get("sampler") or "euler",
            "smc_cfg": bool(data.get("smc_cfg", False)),
            "smc_cfg_lambda": data.get("smc_cfg_lambda", 5.0),
            "smc_cfg_alpha": data.get("smc_cfg_alpha", 0.2),
            "tgate_probe": bool(data.get("tgate_probe", False)),
            "tgate_start_step": data.get("tgate_start_step", 0),
            "tgate_min_block": data.get("tgate_min_block", 0),
            "spectrum_probe": bool(data.get("spectrum_probe", False)),
            "spectrum_window_size": data.get("spectrum_window_size", 2.0),
            "spectrum_flex_window": data.get("spectrum_flex_window", 0.25),
            "spectrum_warmup_steps": data.get("spectrum_warmup_steps", 6),
            "spectrum_stop_caching_step": data.get("spectrum_stop_caching_step", -1),
            "dtype": data.get("dtype") or "bf16",
            "device": data.get("device") or "cuda",
            "batch_size": data.get("batch_size", 1),
            "inference_accel_scheme": data.get("inference_accel_scheme", "none"),
            "arch": (data.get("arch") or "auto") or "auto",
            "output_name": data.get("output_name") or "",
            "dry_run": bool(data.get("dry_run", False)),
        },
        source=RequestSource.CLI,
        compat_mode=True,
    )
    return request  # type: ignore[return-value]


def load_generation_request_from_file(path: str | Path) -> "GenerationRequest":
    """Load a GenerationRequest or legacy generation payload from JSON."""

    if GenerationRequest is None or RequestSource is None:
        raise RuntimeError("GenerationRequest contract is unavailable in this runtime")

    payload_path = Path(path)
    data = json.loads(payload_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("GenerationRequest JSON must contain an object")
    if isinstance(data.get("metadata"), dict):
        return GenerationRequest.model_validate(data)  # type: ignore[return-value]
    return GenerationRequest.from_legacy_payload(data, source=RequestSource.CLI, compat_mode=True)  # type: ignore[return-value]


def run_generation_request_dry_run(request: "GenerationRequest"):
    """Run a GenerationRequest through the registry dry-run path."""

    if RunContext is None or create_generation_registry is None:
        raise RuntimeError("Generation runner registry is unavailable in this runtime")

    project_root = Path(__file__).resolve().parents[3]
    backend_root = project_root / "backend"
    context = RunContext(project_root=project_root, backend_root=backend_root, safe_roots=(project_root,))
    dry_request = request if request.dry_run else request.model_copy(update={"dry_run": True})
    return create_generation_registry().run(dry_request, context)


def write_generation_run_result(result: Any, path: str | Path) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(result, "model_dump_json"):
        output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    else:
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def build_generation_success_result(request: "GenerationRequest", image_paths: list[Path], arch: str) -> Any:
    if GenerationResult is None or ArtifactManifest is None or ArtifactFile is None or RunStatus is None:
        return {
            "request_id": request.request_id,
            "status": "succeeded",
            "message": f"Generated {len(image_paths)} image(s).",
            "data": {"schema_id": request.schema_id, "arch": arch, "image_paths": [str(path) for path in image_paths]},
        }
    artifacts = [
        ArtifactManifest(
            artifact_kind="image",
            schema_id=request.schema_id,
            producer="core.lulynx_trainer.anima_inference_cli",
            request_id=request.request_id,
            files=[ArtifactFile(path=str(path), role="image", media_type="image/png")],
            metadata={"arch": arch, "prompt": request.prompt, "seed": request.seed},
        )
        for path in image_paths
    ]
    return GenerationResult(
        request_id=request.request_id,
        status=RunStatus.SUCCEEDED,
        message=f"Generated {len(image_paths)} image(s).",
        images=artifacts,
        metrics={
            "width": request.width,
            "height": request.height,
            "steps": request.steps,
            "batch_size": request.batch_size,
            "image_count": len(image_paths),
        },
        data={"schema_id": request.schema_id, "arch": arch, "image_paths": [str(path) for path in image_paths]},
    )


def main(argv: Optional[list] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    args = build_parser().parse_args(argv)
    generation_request = build_generation_request_from_args(args)
    logger.info("generation request_id=%s schema=%s", generation_request.request_id, generation_request.schema_id)

    if generation_request.dry_run:
        result = run_generation_request_dry_run(generation_request)
        write_generation_run_result(result, args.run_result_json)
        logger.info("dry-run status=%s artifacts=%d", result.status, len(result.artifacts))
        if not result.ok:
            raise SystemExit(1)
        return

    dtype_name = generation_request.dtype if generation_request.dtype != "auto" else "bf16"
    dtype = _resolve_dtype(dtype_name)
    arch = generation_request.arch.strip().lower()
    if not arch or arch == "auto":
        arch = _detect_arch(generation_request.model_path)
    logger.info("arch=%s dtype=%s device=%s", arch, dtype_name, generation_request.device)

    model = _load_model(generation_request.model_path, arch, generation_request.device, dtype)
    if generation_request.adapter_path:
        model = _apply_adapter(model, generation_request.adapter_path, arch, generation_request.device)
    model, tensorrt_report = _maybe_apply_newbie_tensorrt_transformer(model, generation_request, arch)
    if tensorrt_report.get("enabled"):
        logger.info("Newbie TensorRT transformer enabled: %s", tensorrt_report.get("engine_path"))

    logger.info("Generating %d image(s): %dx%d %d steps cfg=%.1f",
                generation_request.batch_size, generation_request.width, generation_request.height,
                generation_request.steps, generation_request.guidance_scale)
    seed = None if generation_request.seed < 0 else generation_request.seed
    # Resolve the high-level accel scheme to its probe+seam pair. When the scheme
    # is set it wins; otherwise fall back to explicit low-level fields (CLI/advanced).
    from .unified_cache_seam import resolve_inference_accel_scheme
    _accel = resolve_inference_accel_scheme(getattr(generation_request, "inference_accel_scheme", "none"))
    if _accel.enabled:
        _spectrum_probe = _accel.spectrum_probe
        _smoothcache_probe = _accel.smoothcache_probe
        _cache_seam_backend = _accel.cache_seam_backend
    else:
        _spectrum_probe = generation_request.spectrum_probe
        _smoothcache_probe = generation_request.smoothcache_probe
        _cache_seam_backend = generation_request.cache_seam_backend
    images = _generate(model, prompt=generation_request.prompt, negative_prompt=generation_request.negative_prompt,
                       width=generation_request.width, height=generation_request.height, steps=generation_request.steps,
                       guidance_scale=generation_request.guidance_scale, seed=seed,
                       sampler_name=generation_request.sampler, device=generation_request.device,
                       dtype=dtype, batch_size=generation_request.batch_size, arch=arch,
                       newbie_latent_scale_factor=int(tensorrt_report.get("latent_scale_factor") or 0),
                       smc_cfg=generation_request.smc_cfg,
                       smc_cfg_lambda=generation_request.smc_cfg_lambda,
                       smc_cfg_alpha=generation_request.smc_cfg_alpha,
                       tgate_probe=generation_request.tgate_probe,
                       tgate_start_step=generation_request.tgate_start_step,
                       tgate_min_block=generation_request.tgate_min_block,
                       spectrum_probe=_spectrum_probe,
                       spectrum_window_size=generation_request.spectrum_window_size,
                       spectrum_flex_window=generation_request.spectrum_flex_window,
                       spectrum_warmup_steps=generation_request.spectrum_warmup_steps,
                       spectrum_stop_caching_step=generation_request.spectrum_stop_caching_step,
                       smoothcache_probe=_smoothcache_probe,
                       smoothcache_error_threshold=generation_request.smoothcache_error_threshold,
                       smoothcache_warmup_steps=generation_request.smoothcache_warmup_steps,
                       cache_seam_backend=_cache_seam_backend,
                       cache_seam_window_size=generation_request.cache_seam_window_size)

    out_dir = Path(generation_request.output_dir or "./inference_output")
    out_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    for i, img in enumerate(images):
        output_path = out_dir / f"output_{i:04d}.png"
        img.save(str(output_path))
        image_paths.append(output_path)
        logger.info("Saved output_%04d.png", i)
    result = build_generation_success_result(generation_request, image_paths, arch)
    if hasattr(result, "data") and isinstance(result.data, dict):
        result.data["newbie_tensorrt"] = tensorrt_report
    elif isinstance(result, dict):
        result.setdefault("data", {})["newbie_tensorrt"] = tensorrt_report
    write_generation_run_result(result, args.run_result_json)
    logger.info("Done — %d image(s) in %s", len(images), out_dir)


if __name__ == "__main__":
    main()
