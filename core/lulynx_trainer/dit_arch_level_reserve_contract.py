# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Architecture-level reserve contracts for DiT frontier P3 (sub-bucket 3b).

P3 ("layered tech reserve") groups three buckets deliberately *not* landable on
the current single-GPU Anima LoRA path: 3a window attention, 3b architecture-level
ops (QK-Norm / MLA-MLKV / RoPE), and 3c multi-GPU. This module is the 3b
contract/spec reserve: it classifies each arch-level op WITHOUT mutating the model
or the hot path. The facts it pins down:

  * QK-Norm is **already native** in the Anima DiT (per-head RMSNorm q_norm /
    k_norm; see ``anima_native_dit.py`` ~799-847 and ``anima_attention.py``
    ~481-585). The contract therefore *guards against* re-applying it as a LoRA
    patch (which would double-normalize), rather than proposing it as a feature.
  * MLA / MLKV (KV compression) and RoPE are **not** present in Anima today and
    would require a model-family / from-scratch route that owns attention or
    positional structure. They stay as model-family reserve / spec-only.

Everything here is report-only and default-off: ``training_path_enabled`` and
``default_behavior_changed`` are always ``False``, and neither hot-path mutation
nor a LoRA-patch default is ever authorized. Clean-room Lulynx module; references
no external architecture source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ArchLevelReserveItem:
    key: str
    group: str
    goal: str
    dependency: str
    already_native_in_anima: bool = False
    requires_new_model_family: bool = True
    lora_patch_default_allowed: bool = False
    hot_path_mutation_forbidden: bool = True
    source_note: str = ""

    def normalized(self) -> "ArchLevelReserveItem":
        return ArchLevelReserveItem(
            key=_slug(self.key),
            group=_slug(self.group),
            goal=str(self.goal or "unknown").strip() or "unknown",
            dependency=_slug(self.dependency),
            already_native_in_anima=bool(self.already_native_in_anima),
            requires_new_model_family=bool(self.requires_new_model_family),
            lora_patch_default_allowed=bool(self.lora_patch_default_allowed),
            hot_path_mutation_forbidden=bool(self.hot_path_mutation_forbidden),
            source_note=str(self.source_note or "").strip(),
        )


DEFAULT_ARCH_LEVEL_RESERVE_ITEMS: tuple[ArchLevelReserveItem, ...] = (
    ArchLevelReserveItem(
        key="qk_norm",
        group="attention_norm",
        goal="per-head query/key RMS normalization for attention stability",
        dependency="compatible_model_family_route",
        already_native_in_anima=True,
        requires_new_model_family=True,
        source_note="native: anima_native_dit.py _RmsNorm q_norm/k_norm (~799-847); anima_attention.py (~481-585)",
    ),
    ArchLevelReserveItem(
        key="mla_mlkv",
        group="kv_compression",
        goal="multi-head-latent / multi-layer KV compression to shrink attention KV state",
        dependency="compatible_model_family_route",
        already_native_in_anima=False,
        requires_new_model_family=True,
        source_note="not in Anima; architecture-level KV compression, not a drop-in LoRA feature",
    ),
    ArchLevelReserveItem(
        key="rope",
        group="positional",
        goal="rotary positional embedding for a model family that owns positional encoding",
        dependency="model_family_owns_positional_encoding",
        already_native_in_anima=False,
        requires_new_model_family=True,
        source_note="Anima DiT does not apply RoPE in-forward; rope tensors appear only in other-family loaders",
    ),
)


def build_dit_arch_level_reserve_contract(
    items: Sequence[ArchLevelReserveItem | Mapping[str, Any]] | None = None,
    *,
    owned_model_family: bool = False,
) -> dict[str, Any]:
    """Classify arch-level ops as reserve. Never enables training or changes defaults."""
    rows = [_classify(_item(item), owned_model_family) for item in (items or DEFAULT_ARCH_LEVEL_RESERVE_ITEMS)]
    blockers = sorted({reason for row in rows for reason in row["blocked_reasons"]})
    return {
        "schema_version": 1,
        "contract": "dit_arch_level_reserve_contract_v0",
        "item_count": len(rows),
        # Honest reserve scope: nothing here enables training or changes behavior.
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "hot_path_mutation_allowed": False,
        "lora_patch_default_allowed": False,
        "owned_model_family": bool(owned_model_family),
        "groups": sorted({row["group"] for row in rows}),
        "native_present_keys": sorted(r["key"] for r in rows if r["reserve_status"] == "native_present_guarded"),
        "items": rows,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "keep arch-level ops as spec reserve; only a model-family / from-scratch route may adopt "
            "them, never a LoRA-patch default"
        ),
    }


def build_dit_arch_level_reserve_scorecard(contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(contract)
    return {
        "schema_version": 1,
        "scorecard": "dit_arch_level_reserve_contract_v0",
        "ok": bool(payload.get("item_count")),
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "reserve_only": True,
        "contract": payload,
        "blocked_reasons": list(payload.get("blocked_reasons") or []),
        "recommended_next_step": payload.get("recommended_next_step"),
    }


def _classify(item: ArchLevelReserveItem, owned_model_family: bool) -> dict[str, Any]:
    blockers: list[str] = []
    # Re-applying an op the base model already owns (e.g. QK-Norm) as a LoRA patch
    # would double-normalize. Forbidden regardless of model-family ownership.
    if item.already_native_in_anima and not item.lora_patch_default_allowed:
        blockers.append(f"native_present_no_lora_patch:{item.key}")
    if item.requires_new_model_family and not owned_model_family:
        blockers.append("new_model_family_required")

    if item.already_native_in_anima:
        status = "native_present_guarded"
    elif item.requires_new_model_family:
        status = "model_family_adoptable" if owned_model_family else "model_family_reserve"
    else:
        status = "spec_only"
    return {
        "key": item.key,
        "group": item.group,
        "goal": item.goal,
        "dependency": item.dependency,
        "already_native_in_anima": item.already_native_in_anima,
        "requires_new_model_family": item.requires_new_model_family,
        "hot_path_mutation_forbidden": item.hot_path_mutation_forbidden,
        "lora_patch_default_allowed": item.lora_patch_default_allowed,
        "reserve_status": status,
        "source_note": item.source_note,
        "blocked_reasons": blockers,
    }


def _item(item: ArchLevelReserveItem | Mapping[str, Any]) -> ArchLevelReserveItem:
    if isinstance(item, Mapping):
        return ArchLevelReserveItem(**item).normalized()
    return item.normalized()


def _slug(value: str) -> str:
    return str(value or "unknown").strip().lower().replace("-", "_").replace("/", "_").replace(" ", "_") or "unknown"
