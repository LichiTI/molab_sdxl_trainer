"""Resolve advanced optimizer-side strategies without locking in one backend.

Only strategies already wired into optimizer construction are allowed to change
runtime behavior. Reserved strategies stay profile-only until their training
integration is ready.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import re
from typing import Any, Dict, List


_VALID_STRATEGIES = {"auto", "off", "profile_only", "lora_plus", "rs_lora", "galore"}


@dataclass
class AdvancedOptimizerStrategyProfile:
    requested: str
    resolved: str
    active: bool
    fallback_reason: str = ""
    notes: List[str] | None = None
    capabilities: Dict[str, bool] | None = None
    config_effects: List[str] | None = None

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["notes"] = list(self.notes or [])
        data["capabilities"] = dict(self.capabilities or {})
        data["config_effects"] = list(self.config_effects or [])
        return data


def _base_capabilities() -> Dict[str, bool]:
    return {
        "lora_plus_param_groups": True,
        "rs_lora_adapter_scaling": True,
        "galore_optimizer_projection": True,
    }


def _network_arg_bool(config: Any, key: str) -> bool | None:
    raw_args = getattr(config, "network_args", [])
    if isinstance(raw_args, str):
        parts = [part for part in re.split(r"[\s\r\n,]+", raw_args) if part]
    elif isinstance(raw_args, (list, tuple, set, frozenset)):
        parts = [str(part) for part in raw_args]
    else:
        parts = [str(raw_args)] if raw_args is not None else []

    normalized_key = key.strip().lower().replace("-", "_")
    for raw_part in parts:
        part = str(raw_part).strip()
        if not part:
            continue
        part = part.lstrip("-").strip().lower().replace("-", "_")
        if part == normalized_key:
            return True
        if part in {f"no_{normalized_key}", f"disable_{normalized_key}"}:
            return False
        if "=" not in part:
            continue
        name, value = (piece.strip() for piece in part.split("=", 1))
        if name != normalized_key:
            continue
        if value in {"1", "true", "yes", "on", "enabled"}:
            return True
        if value in {"0", "false", "no", "off", "disabled"}:
            return False
    return None


def _network_module_value(config: Any) -> str:
    module = getattr(config, "network_module", "")
    return str(getattr(module, "value", module) or "").strip()


def _rs_lora_support_reason(config: Any) -> str:
    module = _network_module_value(config)
    blocked_flags = {
        "lora_fa_enabled": "LoRA-FA has its own scaling/export path.",
        "vera_enabled": "VeRA exports a derived standard LoRA and is not wired to RS-LoRA scaling yet.",
        "tlora_enabled": "T-LoRA dynamic-rank scaling is not wired to RS-LoRA yet.",
        "hydralora_enabled": "HydraLoRA expert routing is not wired to RS-LoRA scaling yet.",
        "fera_enabled": "FeRA is not wired to RS-LoRA scaling yet.",
        "flexrank_lora_enabled": "FlexRank LoRA is not wired to RS-LoRA scaling yet.",
    }
    if module == "lycoris.locon":
        return "LyCORIS uses its own adapter/network_args route; native rs_lora_enabled is not consumed there."
    if module in {"networks.lora_fa", "networks.vera", "networks.tlora", "networks.flexrank_lora"}:
        return f"{module} is not wired to native RS-LoRA scaling yet."
    for flag, reason in blocked_flags.items():
        if bool(getattr(config, flag, False)):
            return reason
    return ""


def normalize_advanced_optimizer_strategy(raw: Any) -> str:
    value = str(raw or "auto").strip().lower().replace("-", "_").replace("+", "_plus")
    aliases = {
        "": "auto",
        "default": "auto",
        "none": "off",
        "disabled": "off",
        "dry_run": "profile_only",
        "profile": "profile_only",
        "manifest": "profile_only",
        "lora": "lora_plus",
        "loraplus": "lora_plus",
        "lora_plus_plus": "lora_plus",
        "rslora": "rs_lora",
        "rank_stabilized_lora": "rs_lora",
        "rank_stabilized": "rs_lora",
        "gradient_low_rank": "galore",
    }
    value = aliases.get(value.replace(" ", ""), value)
    if value not in _VALID_STRATEGIES:
        return "auto"
    return value


def resolve_advanced_optimizer_strategy(config: Any) -> AdvancedOptimizerStrategyProfile:
    requested = normalize_advanced_optimizer_strategy(
        getattr(config, "advanced_optimizer_strategy", "auto")
    )
    notes: List[str] = []

    legacy_lora_plus = bool(getattr(config, "lora_plus_enabled", False))
    network_args_rs_lora = _network_arg_bool(config, "rs_lora")
    legacy_rs_lora = bool(getattr(config, "rs_lora_enabled", False))
    if network_args_rs_lora is not None:
        legacy_rs_lora = bool(network_args_rs_lora)
        setattr(config, "rs_lora_enabled", legacy_rs_lora)

    if requested == "auto":
        if legacy_rs_lora:
            unsupported_reason = _rs_lora_support_reason(config)
            if unsupported_reason:
                setattr(config, "rs_lora_enabled", False)
                return AdvancedOptimizerStrategyProfile(
                    requested=requested,
                    resolved="profile_only",
                    active=False,
                    fallback_reason=unsupported_reason,
                    notes=["Existing rs_lora request was recorded, but this adapter route does not consume native RS-LoRA scaling yet."],
                    capabilities=_base_capabilities(),
                    config_effects=["set rs_lora_enabled=False for unsupported native route"],
                )
            return AdvancedOptimizerStrategyProfile(
                requested=requested,
                resolved="rs_lora",
                active=True,
                notes=["Resolved from existing rs_lora_enabled/network_args rs_lora=True."],
                capabilities=_base_capabilities(),
                config_effects=["preserved rs_lora_enabled=True"],
            )
        if legacy_lora_plus:
            return AdvancedOptimizerStrategyProfile(
                requested=requested,
                resolved="lora_plus",
                active=True,
                notes=["Resolved from existing lora_plus_enabled=True."],
                capabilities=_base_capabilities(),
                config_effects=["preserved lora_plus_enabled=True"],
            )
        if bool(getattr(config, "svd_grad_proj_enabled", False)):
            return AdvancedOptimizerStrategyProfile(
                requested=requested,
                resolved="galore",
                active=True,
                notes=["Resolved from existing svd_grad_proj_enabled=True."],
                capabilities=_base_capabilities(),
                config_effects=["preserved svd_grad_proj_enabled=True"],
            )
        return AdvancedOptimizerStrategyProfile(
            requested=requested,
            resolved="off",
            active=False,
            notes=["No advanced optimizer-side strategy requested."],
            capabilities=_base_capabilities(),
        )

    if requested == "off":
        if legacy_rs_lora:
            unsupported_reason = _rs_lora_support_reason(config)
            if unsupported_reason:
                setattr(config, "rs_lora_enabled", False)
                return AdvancedOptimizerStrategyProfile(
                    requested=requested,
                    resolved="profile_only",
                    active=False,
                    fallback_reason=unsupported_reason,
                    notes=["Existing rs_lora request was recorded, but this adapter route does not consume native RS-LoRA scaling yet."],
                    capabilities=_base_capabilities(),
                    config_effects=["set rs_lora_enabled=False for unsupported native route"],
                )
            notes.append("Existing rs_lora_enabled=True is preserved; advanced_optimizer_strategy=off only disables the new selector.")
            return AdvancedOptimizerStrategyProfile(
                requested=requested,
                resolved="rs_lora",
                active=True,
                notes=notes,
                capabilities=_base_capabilities(),
                config_effects=["preserved rs_lora_enabled=True"],
            )
        if legacy_lora_plus:
            notes.append("Existing lora_plus_enabled=True is preserved; advanced_optimizer_strategy=off only disables the new selector.")
            return AdvancedOptimizerStrategyProfile(
                requested=requested,
                resolved="lora_plus",
                active=True,
                notes=notes,
                capabilities=_base_capabilities(),
                config_effects=["preserved lora_plus_enabled=True"],
            )
        return AdvancedOptimizerStrategyProfile(
            requested=requested,
            resolved="off",
            active=False,
            capabilities=_base_capabilities(),
        )

    if requested == "profile_only":
        return AdvancedOptimizerStrategyProfile(
            requested=requested,
            resolved="profile_only",
            active=False,
            notes=["Profile-only mode records intent and does not alter optimizer construction."],
            capabilities=_base_capabilities(),
        )

    if requested == "lora_plus":
        setattr(config, "lora_plus_enabled", True)
        ratio = float(getattr(config, "lora_plus_lr_ratio", 16.0) or 16.0)
        return AdvancedOptimizerStrategyProfile(
            requested=requested,
            resolved="lora_plus",
            active=True,
            notes=[f"Uses existing LoRA+ param-group route with B_lr_ratio={ratio:g}."],
            capabilities=_base_capabilities(),
            config_effects=["set lora_plus_enabled=True"],
        )

    if requested == "rs_lora":
        if network_args_rs_lora is False:
            return AdvancedOptimizerStrategyProfile(
                requested=requested,
                resolved="profile_only",
                active=False,
                fallback_reason="Explicit network_args rs_lora=False has higher priority than advanced_optimizer_strategy=rs_lora.",
                notes=["Keeping adapter scaling on alpha/rank for this run."],
                capabilities=_base_capabilities(),
                config_effects=["preserved rs_lora_enabled=False from network_args"],
            )
        unsupported_reason = _rs_lora_support_reason(config)
        if unsupported_reason:
            setattr(config, "rs_lora_enabled", False)
            return AdvancedOptimizerStrategyProfile(
                requested=requested,
                resolved="profile_only",
                active=False,
                fallback_reason=unsupported_reason,
                notes=["Keeping adapter scaling on the existing route for this run."],
                capabilities=_base_capabilities(),
                config_effects=["preserved rs_lora_enabled=False"],
            )
        setattr(config, "rs_lora_enabled", True)
        return AdvancedOptimizerStrategyProfile(
            requested=requested,
            resolved="rs_lora",
            active=True,
            notes=["Uses rank-stabilized LoRA scaling in native LoRA/DoRA adapter injection: alpha/sqrt(rank)."],
            capabilities=_base_capabilities(),
            config_effects=["set rs_lora_enabled=True"],
        )

    if requested == "galore":
        setattr(config, "svd_grad_proj_enabled", True)
        rank = int(getattr(config, "svd_grad_proj_rank", 128) or 128)
        interval = int(getattr(config, "svd_grad_proj_update_interval", 200) or 200)
        return AdvancedOptimizerStrategyProfile(
            requested=requested,
            resolved="galore",
            active=True,
            notes=[f"Uses existing SVD/GaLore-style gradient projection wrapper (rank={rank}, update_interval={interval})."],
            capabilities=_base_capabilities(),
            config_effects=["set svd_grad_proj_enabled=True"],
        )

    return AdvancedOptimizerStrategyProfile(
        requested=requested,
        resolved="off",
        active=False,
        fallback_reason="Unknown advanced optimizer strategy; using off.",
        capabilities=_base_capabilities(),
    )


def apply_lora_plus_runtime_outcome(
    profile: Dict[str, Any] | AdvancedOptimizerStrategyProfile,
    *,
    applied: bool,
    fallback_reason: str = "",
    note: str = "",
    runtime_details: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    if isinstance(profile, AdvancedOptimizerStrategyProfile):
        data = profile.as_dict()
    else:
        data = dict(profile or {})
        data["notes"] = list(data.get("notes") or [])
        data["capabilities"] = dict(data.get("capabilities") or {})
        data["config_effects"] = list(data.get("config_effects") or [])

    if str(data.get("resolved") or "") != "lora_plus":
        return data

    notes = data["notes"]
    if note and note not in notes:
        notes.append(note)
    if runtime_details:
        data["runtime"] = dict(runtime_details)

    if applied:
        data["resolved"] = "lora_plus"
        data["active"] = True
        data["fallback_reason"] = ""
        return data

    data["resolved"] = "profile_only"
    data["active"] = False
    data["fallback_reason"] = (
        fallback_reason
        or "LoRA+ strategy was requested, but runtime param-group splitting did not activate."
    )
    return data

def apply_galore_runtime_outcome(
    profile: Dict[str, Any] | AdvancedOptimizerStrategyProfile,
    *,
    applied: bool,
    fallback_reason: str = "",
    note: str = "",
) -> Dict[str, Any]:
    if isinstance(profile, AdvancedOptimizerStrategyProfile):
        data = profile.as_dict()
    else:
        data = dict(profile or {})
        data["notes"] = list(data.get("notes") or [])
        data["capabilities"] = dict(data.get("capabilities") or {})
        data["config_effects"] = list(data.get("config_effects") or [])

    if str(data.get("resolved") or "") != "galore":
        return data

    notes = data["notes"]
    if note and note not in notes:
        notes.append(note)

    if applied:
        data["active"] = True
        data["fallback_reason"] = ""
        return data

    data["resolved"] = "profile_only"
    data["active"] = False
    data["fallback_reason"] = (
        fallback_reason
        or "GaLore strategy was requested, but the gradient projection wrapper did not activate."
    )
    return data

