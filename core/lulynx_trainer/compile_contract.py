"""Route-aware compile safety contract.

This module keeps the first compile gate deliberately small and conservative.
It does not compile anything by itself; it only resolves the requested compile
mode into a safe mode and mutates the config/runtime plan so downstream code
does not accidentally enter an unsafe static-shape path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


def _boolish(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def _swap_enabled(config: Any) -> bool:
    granularity = str(getattr(config, "swap_granularity", "off") or "off").strip().lower()
    return (
        int(getattr(config, "blocks_to_swap", 0) or 0) > 0
        or int(getattr(config, "swap_count", 0) or 0) > 0
        or float(getattr(config, "swap_ratio", 0.0) or 0.0) > 0.0
        or granularity not in {"", "off", "none"}
    )


def _native_token_bucket_compile_enabled(config: Any, *, route: str) -> bool:
    if route not in {"anima", "newbie"}:
        return False
    if not _boolish(getattr(config, "native_token_bucket_compile", True), default=True):
        return False
    if route == "anima":
        return _boolish(getattr(config, "anima_cached_training", True), default=True)
    if route == "newbie":
        return _boolish(getattr(config, "use_cache", False), default=False)
    return False


def _compile_shape_strategy(config: Any) -> str:
    normalized = str(getattr(config, "compile_shape_strategy", "auto") or "auto").strip().lower().replace("-", "_")
    return normalized if normalized in {"auto", "fixed_pad", "token_flatten", "native"} else "auto"


def _route_name(value: Any) -> str:
    route = str(value or "sdxl").strip().lower().replace("-", "_")
    if route.startswith("sd15") or route.startswith("sd_lora") or route.startswith("sd_1"):
        return "sd15"
    if route.startswith("sdxl"):
        return "sdxl"
    if route.startswith("anima"):
        return "anima"
    if route.startswith("newbie"):
        return "newbie"
    if route.startswith("flux"):
        return "flux"
    return route


def _unet_route_label(route: str) -> str:
    return "SD15" if route == "sd15" else "SDXL"


def _requested_mode(plan: Any) -> str:
    anima_scope = str(getattr(plan, "anima_compile_scope", "") or "").strip().lower()
    torch_scope = str(getattr(plan, "torch_compile_scope", "") or "").strip().lower()
    torch_compile = _boolish(getattr(plan, "torch_compile", False), default=False)

    if anima_scope in {"full", "full_core", "full_cudagraph"}:
        return "full_core"
    if torch_compile and torch_scope in {"full", "full_core"}:
        return "full_core"
    if anima_scope == "per_block" or (torch_compile and torch_scope == "per_block"):
        return "per_block"
    if torch_compile:
        return "full_model"
    return "off"


def _set_resolved_mode(config: Any, plan: Any, mode: str) -> None:
    if mode == "off":
        plan.torch_compile = False
        plan.torch_compile_scope = ""
        plan.anima_compile_scope = ""
        if hasattr(config, "torch_compile"):
            config.torch_compile = False
        if hasattr(config, "torch_compile_scope"):
            config.torch_compile_scope = ""
        if hasattr(config, "anima_compile_scope"):
            config.anima_compile_scope = ""
        return

    if mode == "per_block":
        plan.torch_compile = True
        plan.torch_compile_scope = "per_block"
        if getattr(plan, "anima_compile_scope", ""):
            plan.anima_compile_scope = "per_block"
        if hasattr(config, "torch_compile"):
            config.torch_compile = True
        if hasattr(config, "torch_compile_scope"):
            config.torch_compile_scope = "per_block"
        if hasattr(config, "anima_compile_scope") and str(getattr(config, "anima_compile_scope", "") or ""):
            config.anima_compile_scope = "per_block"
        return

    # full_model/full_core keep the compile request active.  Full-core routes
    # may be handled by route-specific training-loop code instead of
    # torch.compile on the whole model.
    plan.torch_compile = True
    if mode == "full_core":
        plan.torch_compile_scope = "full_core"
    if hasattr(config, "torch_compile"):
        config.torch_compile = True
    if hasattr(config, "torch_compile_scope"):
        config.torch_compile_scope = plan.torch_compile_scope


@dataclass
class CompileContractDecision:
    route: str
    requested: str
    resolved: str
    static_drop_last: bool = False
    cache_first_required: bool = False
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def compile_active(self) -> bool:
        return self.resolved != "off"

    def log_lines(self) -> Iterable[str]:
        yield (
            "[compile-contract] "
            f"route={self.route} requested={self.requested} resolved={self.resolved} "
            f"static_drop_last={'yes' if self.static_drop_last else 'no'} "
            f"cache_first_required={'yes' if self.cache_first_required else 'no'}"
        )
        for reason in self.reasons:
            yield f"[compile-contract] {reason}"
        for warning in self.warnings:
            yield f"[compile-contract][warn] {warning}"


def resolve_compile_contract(config: Any, plan: Any, *, model_arch: str) -> CompileContractDecision:
    """Resolve compile settings into the safest currently supported mode."""

    route = _route_name(model_arch)
    requested = _requested_mode(plan)
    resolved = requested
    reasons: list[str] = []
    warnings: list[str] = []
    strict = _boolish(getattr(config, "compile_contract_strict", True), default=True)

    if requested == "off":
        decision = CompileContractDecision(route=route, requested=requested, resolved="off")
        return decision

    if requested == "full_model":
        if route in {"anima", "newbie"}:
            resolved = "per_block"
            reasons.append("full-model compile downgraded to per_block for DiT routes until full-core targets are validated")
        elif route in {"sdxl", "sd15"}:
            resolved = "per_block"
            reasons.append(
                f"{_unet_route_label(route)} full-model compile downgraded to per_block because full compile is not validated yet"
            )
        elif route == "flux":
            resolved = "per_block"
            reasons.append("Flux full-model compile downgraded to per_block because only transformer block targets are validated")

    if (
        route == "newbie"
        and resolved in {"per_block", "full_core"}
        and _boolish(getattr(config, "newbie_safe_fallback", False), default=False)
    ):
        resolved = "off" if strict else "per_block"
        reasons.append("compile disabled because safe_fallback can change the graph path")

    full_requested = resolved == "full_core"
    if full_requested:
        if _boolish(getattr(config, "gradient_checkpointing", False), default=False):
            resolved = "per_block"
            reasons.append("full_core compile downgraded to per_block because gradient_checkpointing is enabled")
        elif _swap_enabled(config):
            resolved = "per_block"
            reasons.append("full_core compile downgraded to per_block because memory/block swap is enabled")
        elif route in {"sdxl", "sd15"}:
            resolved = "per_block"
            reasons.append(
                f"{_unet_route_label(route)} full_core compile downgraded to per_block; full-core strategy is not validated"
            )
        elif route == "flux":
            resolved = "per_block"
            reasons.append("Flux full_core compile downgraded to per_block until a stable full transformer core target is validated")
        elif route == "anima":
            if _boolish(getattr(config, "compile_anima_full_core_enabled", False), default=False):
                fixed_text = int(getattr(config, "anima_fixed_text_tokens", 0) or 0)
                fixed_visual = int(getattr(config, "anima_fixed_visual_tokens", 0) or 0)
                bucket_compile = _native_token_bucket_compile_enabled(config, route=route)
                token_bucket_shape = _compile_shape_strategy(config) in {"token_flatten", "native"}
                if fixed_text <= 0 or (fixed_visual <= 0 and not (bucket_compile and token_bucket_shape)):
                    resolved = "per_block"
                    reasons.append(
                        "Anima full_core compile downgraded to per_block because "
                        "anima_fixed_text_tokens plus fixed visual tokens or token-bucket static shapes are required"
                    )
                else:
                    if fixed_visual <= 0 and bucket_compile and token_bucket_shape:
                        reasons.append("Anima full_core compile allowed by compile_anima_full_core_enabled=true with token-bucket static shapes")
                    else:
                        reasons.append("Anima full_core compile allowed by compile_anima_full_core_enabled=true")
            else:
                resolved = "per_block"
                reasons.append(
                    "Anima full_core compile downgraded to per_block; "
                    "set compile_anima_full_core_enabled=true to use the experimental route"
                )
        elif route == "newbie":
            resolved = "per_block"
            reasons.append("newbie full_core compile downgraded to per_block until full-core route validation passes")

    if route in {"anima", "newbie"} and resolved in {"per_block", "full_core"}:
        fixed_text = int(getattr(config, f"{route}_fixed_text_tokens", 0) or 0)
        fixed_visual = int(getattr(config, f"{route}_fixed_visual_tokens", 0) or 0)
        bucket_compile = _native_token_bucket_compile_enabled(config, route=route)
        token_bucket_shape = _compile_shape_strategy(config) in {"token_flatten", "native"}
        if route == "newbie":
            # Newbie may not expose both names yet; use a warning instead of
            # blocking per-block compile so existing smoke routes keep working.
            fixed_visual = fixed_visual or int(getattr(config, "newbie_fixed_visual_tokens", 0) or 0)
        if fixed_text <= 0:
            message = f"{route} compile requires fixed token budgets for static shapes"
            if route == "anima" and strict:
                resolved = "off"
                reasons.append(message)
            else:
                warnings.append(f"{message}; compile may recompile or use incomplete static shapes")
        elif fixed_visual <= 0:
            if bucket_compile:
                label = "Anima" if route == "anima" else "Newbie"
                if token_bucket_shape:
                    reasons.append(
                        f"{label} per-block compile uses token-count static shapes from no-pad cached visual token buckets "
                        f"instead of {route}_fixed_visual_tokens"
                    )
                else:
                    reasons.append(
                        f"{label} per-block compile uses no-pad cached visual token buckets "
                        f"instead of {route}_fixed_visual_tokens"
                    )
            elif strict:
                resolved = "off"
                reasons.append(f"{route} compile requires fixed visual tokens or native token buckets")
            else:
                warnings.append(f"{route} compile has no fixed visual tokens; compile may recompile across shapes")

    cache_first_required = (
        _boolish(getattr(config, "compile_require_cache_first", True), default=True)
        and route in {"anima", "newbie"}
        and resolved in {"per_block", "full_core"}
    )
    if cache_first_required:
        if route == "anima" and not _boolish(getattr(config, "anima_cached_training", True), default=True):
            resolved = "off" if strict else "per_block"
            reasons.append("compile disabled because Anima cache-first training is disabled")
        if route == "newbie" and not _boolish(getattr(config, "use_cache", False), default=False):
            resolved = "off" if strict else "per_block"
            reasons.append("compile disabled because Newbie cache-first training is not enabled")

    static_drop_last = (
        _boolish(getattr(config, "compile_static_shape_drop_last", True), default=True)
        and resolved in {"per_block", "full_core"}
    )

    _set_resolved_mode(config, plan, resolved)

    decision = CompileContractDecision(
        route=route,
        requested=requested,
        resolved=resolved,
        static_drop_last=static_drop_last,
        cache_first_required=cache_first_required and resolved != "off",
        reasons=reasons,
        warnings=warnings,
    )
    return decision
