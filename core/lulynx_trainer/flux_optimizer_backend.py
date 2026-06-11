"""Optimizer backend selection for the Flux LoRA preview trainer."""

from __future__ import annotations

from typing import Any, Callable, Iterable

import torch


_SUPPORTED_BACKENDS = {"auto", "torch_adamw", "foreach_adamw", "torch_fused", "bnb_8bit"}


def normalize_flux_optimizer_backend(value: Any) -> str:
    raw = str(value or "auto").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "": "auto",
        "default": "auto",
        "torch": "torch_adamw",
        "adamw": "torch_adamw",
        "foreach": "foreach_adamw",
        "multi_tensor": "foreach_adamw",
        "fused": "torch_fused",
        "torchfused": "torch_fused",
        "bnb": "bnb_8bit",
        "bitsandbytes": "bnb_8bit",
        "bitsandbytes_8bit": "bnb_8bit",
    }
    backend = aliases.get(raw, raw)
    return backend if backend in _SUPPORTED_BACKENDS else "auto"


def create_flux_optimizer_backend(
    params: Iterable[torch.nn.Parameter],
    *,
    optimizer_backend: Any,
    optimizer_type: Any,
    lr: float,
    weight_decay: float,
    device: torch.device,
    log: Callable[[str], None] | None = None,
) -> tuple[torch.optim.Optimizer, dict[str, Any]]:
    """Create an AdamW-compatible optimizer and report the resolved backend."""

    requested = normalize_flux_optimizer_backend(optimizer_backend)
    opt_name = str(getattr(optimizer_type, "value", optimizer_type) or "AdamW").lower()
    if requested == "auto" and "8bit" in opt_name:
        requested = "bnb_8bit"

    def profile(resolved: str, fallback_reason: str = "") -> dict[str, Any]:
        return {
            "requested": requested,
            "resolved": resolved,
            "fallback_reason": fallback_reason,
            "source": "flux_lora_preview",
        }

    param_list = list(params)
    if requested == "bnb_8bit":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(param_list, lr=lr, weight_decay=weight_decay), profile("bnb_8bit")
        except Exception as exc:
            if log is not None:
                log(f"bitsandbytes AdamW8bit unavailable, falling back to torch AdamW: {exc}")
            return torch.optim.AdamW(param_list, lr=lr, weight_decay=weight_decay), profile(
                "torch_adamw",
                f"bitsandbytes unavailable: {exc}",
            )

    if requested == "foreach_adamw":
        return torch.optim.AdamW(param_list, lr=lr, weight_decay=weight_decay, foreach=True), profile("foreach_adamw")

    if requested == "torch_fused":
        if device.type == "cuda":
            try:
                return torch.optim.AdamW(param_list, lr=lr, weight_decay=weight_decay, fused=True), profile("torch_fused")
            except Exception as exc:
                if log is not None:
                    log(f"torch fused AdamW unavailable for Flux, falling back to foreach AdamW: {exc}")
                return torch.optim.AdamW(param_list, lr=lr, weight_decay=weight_decay, foreach=True), profile(
                    "foreach_adamw",
                    f"torch fused unavailable: {exc}",
                )
        return torch.optim.AdamW(param_list, lr=lr, weight_decay=weight_decay, foreach=True), profile(
            "foreach_adamw",
            "torch fused AdamW requires CUDA",
        )

    return torch.optim.AdamW(param_list, lr=lr, weight_decay=weight_decay), profile("torch_adamw")


__all__ = ["create_flux_optimizer_backend", "normalize_flux_optimizer_backend"]
