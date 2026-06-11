"""Checkpoint policy resolver for activation-memory strategies.

This keeps legacy flags working while giving the UI a single advanced strategy
selector. Unsupported selective requests still resolve to the closest safe
existing path, while the native Anima route can opt into an experimental live
selective path.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


CHECKPOINT_POLICY_CHOICES = {"auto", "off", "full", "offloaded", "selective"}
_ANIMA_SELECTIVE_ROUTES = {"anima", "anima_native", "native_anima"}
_NEWBIE_SELECTIVE_ROUTES = {"newbie", "newbie_native", "native_newbie"}
_LIVE_SELECTIVE_ROUTES = _ANIMA_SELECTIVE_ROUTES | _NEWBIE_SELECTIVE_ROUTES


def normalize_selective_checkpoint_route(route: Any) -> str:
    return str(route or "").strip().lower().replace("-", "_")


def route_has_live_selective_checkpoint(route: Any) -> bool:
    return normalize_selective_checkpoint_route(route) in _LIVE_SELECTIVE_ROUTES


def normalize_checkpoint_policy(value: Any) -> str:
    policy = str(value or "auto").strip().lower().replace("-", "_")
    aliases = {
        "": "auto",
        "default": "auto",
        "none": "off",
        "disabled": "off",
        "gradient": "full",
        "gradient_checkpointing": "full",
        "checkpoint": "full",
        "cpu": "offloaded",
        "cpu_offload": "offloaded",
        "save_on_cpu": "offloaded",
        "sac": "selective",
        "selective_recompute": "selective",
        "selective_recomputation": "selective",
    }
    policy = aliases.get(policy.replace(" ", ""), policy)
    return policy if policy in CHECKPOINT_POLICY_CHOICES else "auto"


def selective_checkpoint_available() -> bool:
    try:
        import torch.utils.checkpoint as checkpoint_mod
    except Exception:
        checkpoint_mod = getattr(torch.utils, "checkpoint", None)
    return bool(
        checkpoint_mod is not None
        and hasattr(checkpoint_mod, "create_selective_checkpoint_contexts")
        and hasattr(checkpoint_mod, "CheckpointPolicy")
    )


def _checkpoint_module() -> Any:
    try:
        import torch.utils.checkpoint as checkpoint_mod
        return checkpoint_mod
    except Exception:
        return getattr(torch.utils, "checkpoint", None)


def selective_checkpoint_api_profile() -> Dict[str, Any]:
    """Return a non-mutating capability profile for PyTorch SAC.

    This intentionally avoids importing model code or patching forwards.  The
    result is suitable for manifest/debug output and for route-level smoke tests.
    """
    checkpoint_mod = _checkpoint_module()
    create_contexts = getattr(checkpoint_mod, "create_selective_checkpoint_contexts", None) if checkpoint_mod else None
    policy_enum = getattr(checkpoint_mod, "CheckpointPolicy", None) if checkpoint_mod else None
    checkpoint_fn = getattr(checkpoint_mod, "checkpoint", None) if checkpoint_mod else None

    create_signature = ""
    checkpoint_signature = ""
    try:
        create_signature = str(inspect.signature(create_contexts)) if create_contexts is not None else ""
    except Exception:
        create_signature = "uninspectable"
    try:
        checkpoint_signature = str(inspect.signature(checkpoint_fn)) if checkpoint_fn is not None else ""
    except Exception:
        checkpoint_signature = "uninspectable"

    policy_values: List[str] = []
    try:
        policy_values = [str(item.name) for item in list(policy_enum)] if policy_enum is not None else []
    except Exception:
        policy_values = []

    return {
        "torch_version": str(getattr(torch, "__version__", "")),
        "available": bool(create_contexts is not None and policy_enum is not None),
        "has_create_selective_checkpoint_contexts": create_contexts is not None,
        "has_checkpoint_policy": policy_enum is not None,
        "has_checkpoint_context_fn": "context_fn" in checkpoint_signature,
        "create_selective_checkpoint_contexts_signature": create_signature,
        "checkpoint_signature": checkpoint_signature,
        "checkpoint_policy_values": policy_values,
    }


def build_selective_checkpoint_context_fn(profile: str = "balanced") -> Optional[Any]:
    """Build a conservative PyTorch selective-checkpoint context function.

    The policy saves common expensive matrix/convolution ops and prefers
    recomputing cheaper elementwise/normalization work.  It is intentionally a
    small default policy for native-DiT block experiments, not a global model
    rewrite.
    """
    checkpoint_mod = _checkpoint_module()
    if checkpoint_mod is None:
        return None
    create_contexts = getattr(checkpoint_mod, "create_selective_checkpoint_contexts", None)
    policy_enum = getattr(checkpoint_mod, "CheckpointPolicy", None)
    if create_contexts is None or policy_enum is None:
        return None

    expensive_names = {
        "aten.mm.default",
        "aten.addmm.default",
        "aten.bmm.default",
        "aten.matmul.default",
        "aten.linear.default",
        "aten.convolution.default",
        "aten._scaled_dot_product_flash_attention.default",
        "aten._scaled_dot_product_efficient_attention.default",
        "aten.scaled_dot_product_attention.default",
    }

    def policy_fn(ctx: Any, op: Any, *args: Any, **kwargs: Any) -> Any:
        op_name = str(op)
        if op_name in expensive_names or any(name in op_name for name in ("mm", "addmm", "bmm", "convolution")):
            return policy_enum.MUST_SAVE
        return policy_enum.PREFER_RECOMPUTE

    try:
        return lambda: create_contexts(policy_fn)
    except Exception:
        return None


def _module_attr_chain(obj: Any, chain: str) -> Optional[Any]:
    current = obj
    for part in chain.split("."):
        if current is None:
            return None
        current = getattr(current, part, None)
    return current


def _safe_len(value: Any) -> int:
    try:
        return len(value or [])
    except Exception:
        return 0


@dataclass
class SelectiveCheckpointRouteProfile:
    route: str
    api: Dict[str, Any]
    route_supported: bool
    candidate_entrypoint: str = ""
    block_count: int = 0
    existing_checkpoint_mode: str = ""
    wiring_state: str = ""
    forward_wired: bool = False
    fallback_reason: str = ""
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "route": self.route,
            "api": dict(self.api),
            "route_supported": self.route_supported,
            "candidate_entrypoint": self.candidate_entrypoint,
            "block_count": self.block_count,
            "existing_checkpoint_mode": self.existing_checkpoint_mode,
            "wiring_state": self.wiring_state,
            "forward_wired": self.forward_wired,
            "fallback_reason": self.fallback_reason,
            "notes": list(self.notes),
        }


def profile_selective_checkpoint_route(route: str = "", model: Any = None) -> SelectiveCheckpointRouteProfile:
    """Describe the minimal SAC wiring point for a native DiT route.

    The helper is deliberately read-only. It reports whether a route is still a
    candidate-only profile or already has an experimental live selective wiring
    path.
    """
    normalized_route = normalize_selective_checkpoint_route(route)
    api = selective_checkpoint_api_profile()
    notes: List[str] = []
    candidate = ""
    block_count = 0
    existing_mode = ""

    target = getattr(model, "unet", model) if model is not None else None

    if normalized_route in _ANIMA_SELECTIVE_ROUTES:
        candidate = "unet._checkpoint_block via unet._run_blocks"
        if target is not None:
            blocks = _module_attr_chain(target, "net.blocks")
            block_count = _safe_len(blocks)
            existing_mode = str(getattr(target, "anima_block_checkpointing_mode", "") or "")
            if not callable(getattr(target, "_checkpoint_block", None)):
                notes.append("missing _checkpoint_block on Anima target")
            if not callable(getattr(target, "_run_blocks", None)):
                notes.append("missing _run_blocks on Anima target")
        else:
            notes.append("model not supplied; reporting static Anima candidate")
    elif normalized_route in _NEWBIE_SELECTIVE_ROUTES:
        candidate = "unet.forward block loop around _run_dit_block"
        if target is not None:
            blocks = getattr(target, "_block_modules", None)
            block_count = _safe_len(blocks)
            existing_mode = str(getattr(target, "_newbie_block_checkpointing_mode", "") or "")
            if not callable(getattr(target, "_run_dit_block", None)):
                notes.append("missing _run_dit_block on Newbie target")
            if blocks is None:
                notes.append("missing _block_modules on Newbie target")
        else:
            notes.append("model not supplied; reporting static Newbie candidate")
    else:
        candidate = ""
        notes.append("route is not a native DiT route with a known block checkpoint loop")

    route_supported = bool(
        api.get("available")
        and candidate
        and (target is None or not any(note.startswith("missing ") for note in notes))
    )
    live_route = route_has_live_selective_checkpoint(normalized_route)
    forward_wired = bool(route_supported and live_route)
    wiring_state = ""
    fallback_reason = ""
    if not api.get("available"):
        wiring_state = "api_unavailable"
        fallback_reason = "PyTorch selective checkpoint APIs are unavailable"
    elif not candidate:
        wiring_state = "unsupported_route"
        fallback_reason = f"selective checkpoint profile has no known route candidate for route={normalized_route or 'unknown'}"
    elif target is not None and any(note.startswith("missing ") for note in notes):
        wiring_state = "missing_entrypoint"
        fallback_reason = "model target does not expose the expected native DiT block checkpoint entrypoint"
    elif live_route:
        wiring_state = "experimental_live"
        notes.append("route exposes an experimental live selective-checkpoint wiring path")
    else:
        wiring_state = "candidate_only"
        fallback_reason = f"selective checkpoint route={normalized_route or 'unknown'} is profiled, but live selective wiring is not enabled yet"
        notes.append("route is currently profile-only and falls back to full checkpointing when selected")

    return SelectiveCheckpointRouteProfile(
        route=normalized_route or "unknown",
        api=api,
        route_supported=route_supported,
        candidate_entrypoint=candidate,
        block_count=block_count,
        existing_checkpoint_mode=existing_mode,
        wiring_state=wiring_state,
        forward_wired=forward_wired,
        fallback_reason=fallback_reason,
        notes=notes,
    )


@dataclass
class CheckpointPolicyDecision:
    requested_policy: str
    effective_policy: str
    gradient_checkpointing: bool
    cpu_offload_checkpointing: bool
    selective_available: bool
    selective_profile: Dict[str, Any] = field(default_factory=dict)
    fallback_reason: str = ""
    warnings: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "requested_policy": self.requested_policy,
            "effective_policy": self.effective_policy,
            "gradient_checkpointing": self.gradient_checkpointing,
            "cpu_offload_checkpointing": self.cpu_offload_checkpointing,
            "selective_available": self.selective_available,
            "selective_profile": dict(self.selective_profile),
            "fallback_reason": self.fallback_reason,
            "warnings": list(self.warnings),
        }


def resolve_checkpoint_policy(
    config: Any,
    *,
    route: str = "",
    cuda_available: bool = False,
) -> CheckpointPolicyDecision:
    requested = normalize_checkpoint_policy(getattr(config, "checkpoint_policy", "auto"))
    legacy_gradient = bool(getattr(config, "gradient_checkpointing", False))
    legacy_offload = bool(getattr(config, "cpu_offload_checkpointing", False))
    selective_available = selective_checkpoint_available()
    normalized_route = normalize_selective_checkpoint_route(route)
    selective_profile = profile_selective_checkpoint_route(route).as_dict() if requested == "selective" else {}
    warnings: List[str] = []
    fallback_reason = ""

    if requested == "auto":
        if legacy_offload:
            effective = "offloaded"
        elif legacy_gradient:
            effective = "full"
        else:
            effective = "off"
        return CheckpointPolicyDecision(
            requested_policy=requested,
            effective_policy=effective,
            gradient_checkpointing=legacy_gradient,
            cpu_offload_checkpointing=legacy_offload,
            selective_available=selective_available,
        )

    if requested == "off":
        return CheckpointPolicyDecision(
            requested_policy=requested,
            effective_policy="off",
            gradient_checkpointing=False,
            cpu_offload_checkpointing=False,
            selective_available=selective_available,
        )

    if requested == "full":
        return CheckpointPolicyDecision(
            requested_policy=requested,
            effective_policy="full",
            gradient_checkpointing=True,
            cpu_offload_checkpointing=False,
            selective_available=selective_available,
        )

    if requested == "offloaded":
        if not cuda_available:
            fallback_reason = "offloaded checkpointing needs CUDA for pinned async/transfer benefits; using full checkpointing"
            return CheckpointPolicyDecision(
                requested_policy=requested,
                effective_policy="full",
                gradient_checkpointing=True,
                cpu_offload_checkpointing=False,
                selective_available=selective_available,
                fallback_reason=fallback_reason,
            )
        return CheckpointPolicyDecision(
            requested_policy=requested,
            effective_policy="offloaded",
            gradient_checkpointing=False,
            cpu_offload_checkpointing=True,
            selective_available=selective_available,
        )

    if requested == "selective":
        if not selective_available:
            fallback_reason = "PyTorch selective checkpoint APIs are unavailable"
        elif route_has_live_selective_checkpoint(normalized_route):
            return CheckpointPolicyDecision(
                requested_policy=requested,
                effective_policy="selective",
                gradient_checkpointing=False,
                cpu_offload_checkpointing=False,
                selective_available=selective_available,
                selective_profile=selective_profile,
            )
        else:
            fallback_reason = (
                selective_profile.get("fallback_reason")
                or f"selective checkpoint policy is not wired for route={normalized_route or 'unknown'} yet"
            )
        warnings.append(f"checkpoint_policy=selective resolved to full: {fallback_reason}")
        return CheckpointPolicyDecision(
            requested_policy=requested,
            effective_policy="full",
            gradient_checkpointing=True,
            cpu_offload_checkpointing=False,
            selective_available=selective_available,
            selective_profile=selective_profile,
            fallback_reason=fallback_reason,
            warnings=warnings,
        )

    return CheckpointPolicyDecision(
        requested_policy=requested,
        effective_policy="full" if legacy_gradient else "off",
        gradient_checkpointing=legacy_gradient,
        cpu_offload_checkpointing=legacy_offload,
        selective_available=selective_available,
        fallback_reason="unknown policy resolved through legacy flags",
    )

