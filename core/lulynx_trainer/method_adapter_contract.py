# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Method-adapter contract for native training families.

This module is intentionally lightweight: it describes what an adapter method
means after config normalization, without importing torch or constructing model
layers.  Trainer/runtime code can use it for logging, preflight, validation
gates, and future UI capability reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple


LORA_METHODS = frozenset({
    "lora",
    "lora_plus",
    "rs_lora",
    "dora",
    "lora_fa",
    "vera",
    "tlora",
    "flexrank",
    "hydralora",
    "fera",
})

LYCORIS_METHODS = frozenset({
    "loha",
    "locon",
    "lokr",
    "ia3",
    "full",
    "diag-oft",
})

SUPPORTED_FAMILIES = frozenset({"sdxl", "anima", "newbie"})


@dataclass(frozen=True)
class AdapterMethodSpec:
    method: str
    family: str
    backend: str
    network_module: str
    lycoris_algo: str = ""
    supported: bool = True
    trainable_adapter: bool = True
    safe_merge: bool = True
    requires_optimizer_grouping: bool = False
    requires_special_save: bool = False
    flags: Tuple[str, ...] = ()
    aliases: Tuple[str, ...] = ()
    warnings: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


def normalize_adapter_method(value: Any) -> str:
    """Normalize user-facing adapter aliases to canonical method names."""

    method = str(value or "").strip().lower().replace("-", "_")
    aliases = {
        "": "lora",
        "standard": "lora",
        "networks.lora": "lora",
        "lycoris_lokr": "lokr",
        "lycoris_loha": "loha",
        "lycoris_locon": "locon",
        "lycoris_ia3": "ia3",
        "lycoris_full": "full",
        "lycoris_diag_oft": "diag-oft",
        "diag_oft": "diag-oft",
        "diag-oft": "diag-oft",
        "oft": "diag-oft",
        "networks.oft": "diag-oft",
        "lora+": "lora_plus",
        "loraplus": "lora_plus",
        "rslora": "rs_lora",
        "rs_lora": "rs_lora",
        "rank_stabilized_lora": "rs_lora",
        "rank_stabilized": "rs_lora",
        "flexrank_lora": "flexrank",
        "flexrank-lora": "flexrank",
        "networks.flexrank_lora": "flexrank",
        "hydra_lora": "hydralora",
        "hydra-lora": "hydralora",
    }
    return aliases.get(method, method)


def resolve_adapter_method(config: Any = None, *, family: str = "", method: str = "") -> AdapterMethodSpec:
    """Resolve adapter method semantics for a family/config pair."""

    data = _config_mapping(config)
    resolved_family = _normalize_family(family or data.get("model_arch") or data.get("model_type") or data.get("training_type"))
    raw_method = method or _method_from_config(data, resolved_family)
    canonical = normalize_adapter_method(raw_method)
    network_module = _network_module(data)
    lycoris_algo = normalize_adapter_method(data.get("lycoris_algo", ""))

    if canonical == "lora" and network_module == "lycoris.locon":
        canonical = lycoris_algo or "loha"
    if canonical == "lora" and lycoris_algo in LYCORIS_METHODS and network_module == "lycoris.locon":
        canonical = lycoris_algo

    warnings = []
    notes = []
    flags = []
    backend = "lora"
    resolved_network_module = "networks.lora"
    resolved_lycoris_algo = ""
    requires_optimizer_grouping = False
    safe_merge = True
    requires_special_save = False

    if canonical in LYCORIS_METHODS:
        backend = "lycoris"
        resolved_network_module = "lycoris.locon"
        resolved_lycoris_algo = canonical
        if canonical in {"ia3", "diag-oft"}:
            safe_merge = False
            warnings.append(f"{canonical} merge/export needs method-aware validation before broad release.")
    elif canonical in LORA_METHODS:
        backend = "lora"
        if canonical == "lora_fa":
            resolved_network_module = "networks.lora_fa"
            flags.append("lora_fa_enabled")
            safe_merge = False
            warnings.append("LoRA-FA freezes down projection; merge/export should be validated per route.")
        elif canonical == "vera":
            resolved_network_module = "networks.vera"
            flags.append("vera_enabled")
            safe_merge = False
            requires_special_save = True
        elif canonical == "tlora":
            resolved_network_module = "networks.tlora"
            flags.append("tlora_enabled")
            safe_merge = False
            warnings.append("T-LoRA rank schedule is train-time behavior; export needs roundtrip proof.")
        elif canonical == "flexrank":
            resolved_network_module = "networks.flexrank_lora"
            flags.append("flexrank_lora_enabled")
            safe_merge = False
            warnings.append("FlexRank merge/export should be validated against its dynamic-rank inference path.")
        elif canonical == "dora":
            flags.extend(("use_dora", "dora_enabled"))
            safe_merge = False
        elif canonical == "lora_plus":
            flags.append("lora_plus_enabled")
            requires_optimizer_grouping = True
            notes.append("LoRA+ changes optimizer param groups, not adapter layer type.")
        elif canonical == "rs_lora":
            flags.append("rs_lora_enabled")
            safe_merge = False
            warnings.append("RS-LoRA scaling needs metadata-aware load/merge validation.")
        elif canonical == "hydralora":
            flags.append("hydralora_enabled")
            safe_merge = False
            requires_special_save = True
        elif canonical == "fera":
            flags.append("fera_enabled")
            safe_merge = False
            requires_special_save = True
    else:
        backend = "unknown"
        resolved_network_module = network_module or "networks.lora"
        warnings.append(f"Unknown adapter method: {canonical}")

    supported = resolved_family in SUPPORTED_FAMILIES and backend != "unknown"
    if resolved_family not in SUPPORTED_FAMILIES:
        supported = False
        warnings.append(f"Unsupported adapter family: {resolved_family or '<empty>'}")

    return AdapterMethodSpec(
        method=canonical,
        family=resolved_family,
        backend=backend,
        network_module=resolved_network_module,
        lycoris_algo=resolved_lycoris_algo,
        supported=supported,
        safe_merge=safe_merge,
        requires_optimizer_grouping=requires_optimizer_grouping,
        requires_special_save=requires_special_save,
        flags=tuple(flags),
        aliases=tuple(_aliases_for(canonical)),
        warnings=tuple(warnings),
        notes=tuple(notes),
        metadata={
            "rank": _int_or_none(data.get("network_dim")),
            "alpha": _int_or_none(data.get("network_alpha")),
            "flexrank_min_rank": _int_or_none(data.get("flexrank_lora_rank_range_min")),
            "configured_network_module": network_module,
        },
    )


def adapter_contract_summary(spec: AdapterMethodSpec) -> str:
    parts = [
        f"family={spec.family or 'unknown'}",
        f"method={spec.method}",
        f"backend={spec.backend}",
        f"network_module={spec.network_module}",
    ]
    if spec.lycoris_algo:
        parts.append(f"lycoris_algo={spec.lycoris_algo}")
    if spec.flags:
        parts.append("flags=" + ",".join(spec.flags))
    parts.append(f"safe_merge={str(spec.safe_merge).lower()}")
    return "adapter contract: " + " | ".join(parts)


def _method_from_config(data: Mapping[str, Any], family: str) -> str:
    if family == "newbie":
        raw = data.get("newbie_adapter_type") or data.get("adapter_type") or data.get("lora_type")
    else:
        raw = data.get("lora_type") or data.get("adapter_type")
    if raw:
        return str(raw)
    network_module = _network_module(data)
    if network_module == "lycoris.locon":
        return str(data.get("lycoris_algo", "") or "loha")
    if network_module == "networks.lora_fa":
        return "lora_fa"
    if network_module == "networks.vera":
        return "vera"
    if network_module == "networks.tlora":
        return "tlora"
    if network_module == "networks.flexrank_lora":
        return "flexrank"
    if network_module == "networks.oft":
        return "diag-oft"
    if _truthy(data.get("dora_enabled")) or _truthy(data.get("use_dora")):
        return "dora"
    if _truthy(data.get("lora_plus_enabled")):
        return "lora_plus"
    if _truthy(data.get("rs_lora_enabled")):
        return "rs_lora"
    if _truthy(data.get("flexrank_lora_enabled")):
        return "flexrank"
    if _truthy(data.get("hydralora_enabled")):
        return "hydralora"
    if _truthy(data.get("fera_enabled")):
        return "fera"
    return "lora"


def _config_mapping(config: Any) -> Dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, Mapping):
        return dict(config)
    if hasattr(config, "model_dump"):
        return dict(config.model_dump())
    if hasattr(config, "dict"):
        return dict(config.dict())
    return {key: value for key, value in vars(config).items() if not key.startswith("_")}


def _network_module(data: Mapping[str, Any]) -> str:
    value = data.get("network_module", "")
    if hasattr(value, "value"):
        value = value.value
    value = str(value or "").strip()
    if value == "lora":
        return "networks.lora"
    if value in {"lycoris", "lycoris.kohya"}:
        return "lycoris.locon"
    return value


def _normalize_family(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("anima"):
        return "anima"
    if text.startswith("newbie"):
        return "newbie"
    if text.startswith("sdxl"):
        return "sdxl"
    if text.startswith("sd15") or text.startswith("sd-1") or text == "sd":
        return "sd15"
    return text


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _aliases_for(method: str) -> Tuple[str, ...]:
    aliases = {
        "lora_plus": ("lora+", "loraplus"),
        "rs_lora": ("rslora", "rank_stabilized_lora", "rank_stabilized"),
        "flexrank": ("flexrank_lora", "flexrank-lora", "networks.flexrank_lora"),
        "hydralora": ("hydra_lora", "hydra-lora"),
        "diag-oft": ("diag_oft", "oft", "networks.oft", "lycoris_diag_oft"),
        "lokr": ("lycoris_lokr",),
        "loha": ("lycoris_loha",),
        "locon": ("lycoris_locon",),
        "ia3": ("lycoris_ia3",),
        "full": ("lycoris_full",),
    }
    return tuple(aliases.get(method, ()))
