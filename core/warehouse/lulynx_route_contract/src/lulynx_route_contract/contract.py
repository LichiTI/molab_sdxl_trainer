"""Route contract: classification, capability resolution, and metadata export.

A ``RouteContract`` is an immutable value object that binds a training type
to its structural family, human label, and a set of capability tags derived
from the training configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from lulynx_route_contract.families import resolve_family, resolve_label


# ---------------------------------------------------------------------------
# Capability resolution
# ---------------------------------------------------------------------------

def _is_truthy(value: Any) -> bool:
    """Coerce a loosely-typed config value to bool."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _int_or_zero(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _resolve_capability_tags(
    family: str,
    config: Mapping[str, Any] | None,
) -> tuple[tuple[str, ...], str]:
    """Build a set of capability flag strings for the given route family.

    Every contract carries a baseline of ``shared-contract``,
    ``shared-metadata``, and ``shared-banner`` tags.  Family-specific tags
    are appended based on recognized config keys.

    Returns ``(flags_tuple, human_summary)``.
    """
    cfg = config or {}
    tags: list[str] = ["shared-contract", "shared-metadata", "shared-banner"]

    if family == "newbie":
        tags.extend([
            "newbie-bridge",
            "newbie-cache-phase",
            "newbie-preview-pipeline",
            "newbie-memory-runtime",
            "newbie-state-save",
        ])
        if _is_truthy(cfg.get("use_cache", True)):
            tags.append("persistent-cache")
        else:
            tags.append("transient-cache")
        if _int_or_zero(cfg.get("blocks_to_swap")) > 0:
            tags.append("block-swap")
        summary = (
            "Newbie pipeline: planning, cache, preview, memory, save-state "
            "bound by a shared runtime contract."
        )
        return tuple(tags), summary

    if family == "anima":
        tags.extend([
            "anima-route-normalization",
            "anima-runtime-summary",
            "anima-metadata-contract",
        ])
        if _is_truthy(cfg.get("dora_wd")):
            tags.append("dora")
        if _is_truthy(cfg.get("pissa_init")):
            tags.append("pissa")
        if _is_truthy(cfg.get("network_swap_to_ram")):
            tags.append("vram-swap-to-ram")
        if _int_or_zero(cfg.get("blocks_to_swap")) > 0:
            tags.append("block-swap")
        summary = (
            "Anima route: adapter normalization, runtime policy, and export "
            "metadata connected by a shared contract."
        )
        return tuple(tags), summary

    if family == "sdxl":
        tags.extend([
            "sdxl-route-normalization",
            "sdxl-low-vram-guard",
            "sdxl-text-cache-contract",
            "sdxl-metadata-contract",
        ])
        if _is_truthy(cfg.get("sdxl_low_vram_optimization")):
            tags.append("low-vram-optimization")
        if _is_truthy(cfg.get("sdxl_fixed_block_swap")) or _is_truthy(cfg.get("sdxl_block_swap_enabled")):
            tags.append("block-swap")
        if _is_truthy(cfg.get("cache_text_encoder_outputs")):
            tags.append("text-encoder-cache")
        summary = (
            "SDXL route: low-VRAM policy, text-cache semantics, and export "
            "metadata connected by a shared contract."
        )
        return tuple(tags), summary

    summary = "Generic route: baseline shared contract surface."
    return tuple(tags), summary


# ---------------------------------------------------------------------------
# Route contract value object
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RouteContract:
    """Immutable descriptor for a resolved training route."""

    training_type: str
    family: str
    kind: str
    label: str
    capability_tags: tuple[str, ...]
    capability_summary: str

    def to_metadata_dict(self) -> dict[str, str]:
        """Export contract fields as flat string metadata suitable for
        embedding in safetensors or JSON model files."""
        return {
            "route_training_type": self.training_type,
            "route_kind": self.kind,
            "route_label": self.label,
            "route_family": self.family,
            "route_capabilities": ",".join(self.capability_tags),
            "route_capability_summary": self.capability_summary,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_route(
    training_type: str | None,
    *,
    config: Mapping[str, Any] | None = None,
    kind_override: str | None = None,
    label_override: str | None = None,
) -> RouteContract:
    """Classify a training type string into a full ``RouteContract``.

    Parameters
    ----------
    training_type:
        Raw training type string (e.g. ``"anima-lora"``).  Normalized
        to lowercase and stripped before matching.
    config:
        Optional training configuration mapping used to resolve
        capability flags.
    kind_override:
        If provided, overrides the resolved family as the route kind.
    label_override:
        If provided, overrides the resolved label.
    """
    normalized = str(training_type or "").strip().lower()
    family = resolve_family(normalized)
    kind = str(kind_override or family).strip().lower() or "generic"
    label = str(label_override or resolve_label(normalized)).strip() or "Generic Training"
    tags, summary = _resolve_capability_tags(family, config)
    return RouteContract(
        training_type=normalized or "generic",
        family=family,
        kind=kind,
        label=label,
        capability_tags=tags,
        capability_summary=summary,
    )


def contract_to_metadata(contract: RouteContract) -> dict[str, str]:
    """Shorthand for ``contract.to_metadata_dict()``."""
    return contract.to_metadata_dict()


def extract_metadata_from_mapping(mapping: Mapping[str, Any] | None) -> dict[str, str]:
    """Extract previously-stamped route metadata from a config mapping.

    Returns an empty dict if no embedded metadata is found.
    """
    if not isinstance(mapping, Mapping):
        return {}
    raw = mapping.get("_route_contract_metadata")
    if isinstance(raw, Mapping):
        return {str(k): str(v) for k, v in raw.items()}
    return {}


def label_from_mapping(mapping: Mapping[str, Any] | None, default: str) -> str:
    """Retrieve the route label from an embedded mapping, or fall back."""
    meta = extract_metadata_from_mapping(mapping)
    if meta:
        found = str(meta.get("route_label", "") or "").strip()
        if found:
            return found
    if isinstance(mapping, Mapping):
        direct = str(mapping.get("_route_label", "") or "").strip()
        if direct:
            return direct
    return str(default or "").strip() or "Generic Training"


def kind_from_mapping(mapping: Mapping[str, Any] | None, default: str) -> str:
    """Retrieve the route kind from an embedded mapping, or fall back."""
    meta = extract_metadata_from_mapping(mapping)
    if meta:
        found = str(meta.get("route_kind", "") or "").strip().lower()
        if found:
            return found
    if isinstance(mapping, Mapping):
        direct = str(mapping.get("_route_kind", "") or "").strip().lower()
        if direct:
            return direct
    return str(default or "").strip().lower() or "generic"
