"""Anima full-parameter finetune helpers.

Phase 1 is intentionally narrow: cache-first, native DiT only, no text-encoder
training.  The shared trainer still owns the training loop, optimizer, and save
cadence; this module only normalizes the Anima-specific full-finetune boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import torch


LogFn = Callable[[str], None]

ANIMA_TE_POLICY_DIT_ONLY = "dit_only"
ANIMA_TE_POLICY_BLOCKED_PHASE1 = "blocked_phase1"


@dataclass(frozen=True)
class AnimaFullFinetuneSetup:
    trainable_params: List[torch.nn.Parameter]
    total_params: int
    train_text_encoder_blocked: bool
    train_text_encoder_requested: bool = False
    text_encoder_policy: str = "dit_only"
    text_conditioning_mode: str = "cache_first"
    checkpoint_prefix: str = "unet."
    mode: str = "dit_only_cache_first"

    def as_dict(self) -> Dict[str, Any]:
        return {
            "trainable_param_tensors": len(self.trainable_params),
            "total_params": int(self.total_params),
            "train_text_encoder_blocked": bool(self.train_text_encoder_blocked),
            "train_text_encoder_requested": bool(self.train_text_encoder_requested),
            "text_encoder_policy": self.text_encoder_policy,
            "text_conditioning_mode": self.text_conditioning_mode,
            "checkpoint_prefix": self.checkpoint_prefix,
            "mode": self.mode,
        }


def _log(log: LogFn | None, message: str) -> None:
    if log is not None:
        log(message)


def is_anima_full_finetune(config: Any) -> bool:
    model_type = getattr(getattr(config, "model_type", ""), "value", getattr(config, "model_type", ""))
    return (
        str(model_type).strip().lower() == "anima"
        and str(getattr(config, "training_type", "") or "").strip().lower() == "full_finetune"
    )


def normalize_anima_text_encoder_policy(value: Any, *, requested: bool = False) -> str:
    """Normalize the Anima full-finetune TE training policy.

    Anima full-finetune may use frozen text encoders for online/cache-builder
    conditioning, but TE weights are not part of the full-finetune optimizer.
    """

    normalized = str(value or "").strip().lower().replace("-", "_")
    if requested or normalized == ANIMA_TE_POLICY_BLOCKED_PHASE1:
        return ANIMA_TE_POLICY_BLOCKED_PHASE1
    return ANIMA_TE_POLICY_DIT_ONLY


def _anima_cache_mode(config: Any) -> str:
    return str(
        getattr(config, "native_cache_mode", "")
        or getattr(config, "anima_cache_mode", "")
        or ""
    ).strip().lower().replace("-", "_")


def _text_conditioning_mode(config: Any) -> str:
    cache_mode = _anima_cache_mode(config)
    if cache_mode in {"online_cache", "online-cache"} or bool(getattr(config, "anima_online_cache", False)):
        return "frozen_te_online_cache"
    if cache_mode in {"raw_online", "raw-online", "online"}:
        return "frozen_te_live"
    return "cache_first"


def _apply_text_encoder_policy_to_config(
    *,
    config: Any,
    requested_text_encoder: bool,
    policy: str,
) -> None:
    if hasattr(config, "anima_full_finetune_train_text_encoder_requested"):
        config.anima_full_finetune_train_text_encoder_requested = requested_text_encoder
    if hasattr(config, "anima_full_finetune_text_encoder_policy"):
        config.anima_full_finetune_text_encoder_policy = policy
    if hasattr(config, "network_train_unet_only"):
        config.network_train_unet_only = True
    if hasattr(config, "network_train_text_encoder_only"):
        config.network_train_text_encoder_only = False
    if hasattr(config, "anima_full_finetune_phase"):
        config.anima_full_finetune_phase = "dit_only_cache_first"


def prepare_anima_dit_only_full_finetune(
    *,
    config: Any,
    model: Any,
    log: LogFn | None = None,
) -> AnimaFullFinetuneSetup:
    """Prepare Phase-1 Anima full finetune on the native cache-first DiT."""

    if model is None or getattr(model, "unet", None) is None:
        raise RuntimeError("Anima full finetune requires a loaded native DiT module.")

    raw_policy = getattr(config, "anima_full_finetune_text_encoder_policy", "")
    requested_text_encoder = bool(
        getattr(config, "anima_full_finetune_train_text_encoder_requested", False)
        or getattr(config, "train_text_encoder", False)
    )
    policy = normalize_anima_text_encoder_policy(
        raw_policy,
        requested=requested_text_encoder,
    )
    _apply_text_encoder_policy_to_config(
        config=config,
        requested_text_encoder=requested_text_encoder,
        policy=policy,
    )

    if not bool(getattr(model, "anima_native_train_ready", False)):
        raise RuntimeError("Anima full finetune requires anima_native_train_ready=True.")
    if not bool(getattr(model, "anima_cached_training_ready", False)):
        raise RuntimeError(
            "Anima full finetune Phase 1 is cache-first only. "
            "Provide paired *_anima and *_anima_te cache files before training."
        )

    if requested_text_encoder:
        _log(
            log,
            "Anima full finetune Phase 1 blocks text-encoder training; "
            "forcing DiT-only full finetune.",
        )
    for attr in ("text_encoder_1", "text_encoder_2", "text_encoder"):
        module = getattr(model, attr, None)
        if module is not None and hasattr(module, "requires_grad_"):
            module.requires_grad_(False)

    model.unet.requires_grad_(True)
    model.unet.train()
    trainable_params = [param for param in model.unet.parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("Anima full finetune found no trainable DiT parameters.")

    total_params = sum(param.numel() for param in trainable_params)
    setattr(model, "anima_full_finetune_ready", True)
    return AnimaFullFinetuneSetup(
        trainable_params=trainable_params,
        total_params=total_params,
        train_text_encoder_blocked=requested_text_encoder,
        train_text_encoder_requested=requested_text_encoder,
        text_encoder_policy=policy,
        text_conditioning_mode=_text_conditioning_mode(config),
    )


def build_anima_full_finetune_state_dict(
    *,
    unet: torch.nn.Module,
    checkpoint_prefix: str = "unet.",
) -> Dict[str, torch.Tensor]:
    """Build the full DiT checkpoint payload used by Anima full finetune."""

    return {f"{checkpoint_prefix}{key}": value for key, value in unet.state_dict().items()}


def collect_trainable_param_name_map(module: torch.nn.Module) -> Dict[int, str]:
    """Map trainable parameter identity to module parameter name."""

    return {id(param): name for name, param in module.named_parameters() if param.requires_grad}


def build_anima_grouped_param_groups(
    *,
    config: Any,
    trainable_params: List[torch.nn.Parameter],
    param_to_name: Dict[int, str],
    log: LogFn | None = None,
) -> List[Dict[str, Any]] | None:
    """Build Anima per-module LR groups from parameter names."""

    grouped_lrs = {
        "self_attn": float(getattr(config, "anima_self_attn_lr", 0) or 0),
        "cross_attn": float(getattr(config, "anima_cross_attn_lr", 0) or 0),
        "mlp": float(getattr(config, "anima_mlp_lr", 0) or 0),
        "mod": float(getattr(config, "anima_mod_lr", 0) or 0),
        "llm_adapter": float(getattr(config, "anima_llm_adapter_lr", 0) or 0),
    }
    if not any(lr > 0 for lr in grouped_lrs.values()):
        return None
    if not trainable_params:
        return None

    base_lr = float(getattr(config, "learning_rate", 1e-4) or 1e-4)
    weight_decay = float(getattr(config, "weight_decay", 0.0) or 0.0)
    groups: Dict[str, Dict[str, Any]] = {}
    unassigned: List[torch.nn.Parameter] = []

    for param in trainable_params:
        name = param_to_name.get(id(param), "")
        matched = False
        for module_key, lr in grouped_lrs.items():
            if lr > 0 and module_key in name:
                groups.setdefault(
                    module_key,
                    {"params": [], "lr": lr, "weight_decay": weight_decay},
                )["params"].append(param)
                matched = True
                break
        if not matched:
            unassigned.append(param)

    if unassigned:
        groups["_base"] = {"params": unassigned, "lr": base_lr, "weight_decay": weight_decay}

    result = list(groups.values())
    _log(
        log,
        "Anima grouped LR: "
        f"{len(result)} param groups - "
        + ", ".join(f"{key}={value['lr']:.1e}" for key, value in groups.items()),
    )
    return result


def load_anima_full_finetune_state(
    *,
    unet: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    log: LogFn | None = None,
    checkpoint_prefix: str = "unet.",
) -> Dict[str, int]:
    """Load a saved Anima full-finetune state into the native DiT module."""

    if isinstance(state_dict, dict) and isinstance(state_dict.get("state_dict"), dict):
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError("Anima full-finetune resume expects a state-dict mapping.")

    prefix_len = len(checkpoint_prefix)
    prefixed = {
        key[prefix_len:]: value
        for key, value in state_dict.items()
        if key.startswith(checkpoint_prefix)
    }
    if prefixed:
        load_payload = prefixed
    else:
        target_keys = set(unet.state_dict().keys())
        load_payload = {key: value for key, value in state_dict.items() if key in target_keys}

    if not load_payload:
        raise RuntimeError(
            "Resume file does not contain Anima full-finetune DiT weights "
            f"with prefix {checkpoint_prefix!r} or matching bare module keys."
        )

    missing, unexpected = unet.load_state_dict(load_payload, strict=False)
    loaded = len(load_payload) - len(unexpected)
    _log(
        log,
        "Anima full-finetune resume: "
        f"loaded={loaded}, missing={len(missing)}, unexpected={len(unexpected)}",
    )
    return {
        "loaded": int(loaded),
        "missing": int(len(missing)),
        "unexpected": int(len(unexpected)),
    }
