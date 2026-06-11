"""Flux attention backend wiring for the preview LoRA trainer."""

from __future__ import annotations

from typing import Any, Callable


_ALIASES = {
    "": "auto",
    "auto": "auto",
    "default": "auto",
    "sdpa": "sdpa",
    "torch": "sdpa",
    "native": "sdpa",
    "flash": "flash2",
    "flash2": "flash2",
    "flashattn": "flash2",
    "flashattention": "flash2",
    "flashattention2": "flash2",
    "fa2": "flash2",
    "xformers": "xformers",
    "sage": "sageattn",
    "sageattn": "sageattn",
    "sageattention": "sageattn",
    "flex": "flexattn",
    "flexattn": "flexattn",
    "flexattention": "flexattn",
}
_DIFFUSERS_BACKENDS = {
    "sdpa": "native",
    "flash2": "flash",
    "xformers": "xformers",
    "sageattn": "sage",
    "flexattn": "flex",
}


def normalize_flux_attention_backend(value: Any) -> str:
    raw = str(value or "auto").strip().lower().replace("-", "_").replace(" ", "")
    return _ALIASES.get(raw, "auto")


def _importable(module_name: str) -> bool:
    try:
        __import__(module_name)
        return True
    except Exception:
        return False


def _resolve_flux_backend(requested: str, *, cuda_available: bool) -> tuple[str, str, str]:
    resolved = "sdpa" if requested == "auto" else requested
    fallback_reason = ""
    if resolved == "flash2":
        if not cuda_available:
            fallback_reason = "flash2 requires CUDA; falling back to Flux native SDPA"
            resolved = "sdpa"
        elif not _importable("flash_attn"):
            fallback_reason = "flash_attn is unavailable; falling back to Flux native SDPA"
            resolved = "sdpa"
    elif resolved == "xformers" and not _importable("xformers"):
        fallback_reason = "xformers is unavailable; falling back to Flux native SDPA"
        resolved = "sdpa"
    elif resolved == "sageattn" and not _importable("sageattention"):
        fallback_reason = "sageattention is unavailable; falling back to Flux native SDPA"
        resolved = "sdpa"
    diffusers_backend = _DIFFUSERS_BACKENDS.get(resolved, "native")
    return resolved, diffusers_backend, fallback_reason


def _is_flux_native_processor(processor: Any) -> bool:
    cls = processor.__class__
    if not cls.__module__.startswith("diffusers.models.transformers.transformer_flux"):
        return False
    return hasattr(processor, "_attention_backend")


def apply_flux_attention_backend(
    transformer: Any,
    requested_backend: Any,
    *,
    log: Callable[[str], None] | None = None,
    cuda_available: bool = False,
) -> dict[str, Any]:
    requested = normalize_flux_attention_backend(requested_backend)
    resolved, diffusers_backend, fallback_reason = _resolve_flux_backend(
        requested,
        cuda_available=cuda_available,
    )
    profile: dict[str, Any] = {
        "requested": requested,
        "resolved": resolved,
        "diffusers_backend": diffusers_backend,
        "patched_processors": 0,
        "fallback_reason": fallback_reason,
        "source": "flux_lora_preview",
    }
    processors = getattr(transformer, "attn_processors", None)
    if not isinstance(processors, dict) or not processors:
        profile["profile_only"] = True
        profile["profile_only_reason"] = "Flux transformer does not expose attn_processors"
        return profile
    skipped: list[str] = []
    for name, processor in processors.items():
        if not _is_flux_native_processor(processor):
            skipped.append(str(name))
            continue
        setattr(processor, "_attention_backend", diffusers_backend)
        profile["patched_processors"] += 1
    if skipped:
        profile["skipped_processors"] = skipped[:8]
    if profile["patched_processors"] <= 0:
        profile["profile_only"] = True
        profile["profile_only_reason"] = "No native Flux attention processors were patched"
    elif log is not None:
        log(f"Flux attention backend resolved: {resolved} (diffusers={diffusers_backend}, processors={profile['patched_processors']})")
    return profile


__all__ = ["apply_flux_attention_backend", "normalize_flux_attention_backend"]
