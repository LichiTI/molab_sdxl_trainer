"""Report-only runtime canary manifest for AdamW8bit native work."""

from __future__ import annotations

from typing import Any, Mapping

from core.turbocore_paged_adamw8bit_runtime_canary_scorecard import (
    build_paged_adamw8bit_runtime_canary_scorecard,
)


OPTIMIZER_KIND = "adamw8bit"
OPTIMIZER_FAMILY = "adamw_quantized"


def build_adamw8bit_runtime_canary_scorecard(
    *,
    source_runtime_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reuse the quantized AdamW8bit native manifest without enabling dispatch."""

    source = dict(source_runtime_report or build_paged_adamw8bit_runtime_canary_scorecard())
    ready = bool(source.get("runtime_canary_manifest_ready", False))
    blockers = [] if ready else _strings(source.get("blocked_reasons")) or ["adamw8bit_runtime_manifest_missing"]
    return {
        "schema_version": 1,
        "scorecard": "turbocore_adamw8bit_runtime_canary_scorecard_v0",
        "gate": "adamw8bit_runtime_canary_manifest",
        "ok": bool(source.get("ok", False)),
        "promotion_ready": False,
        "runtime_canary_manifest_ready": ready,
        "runtime_canary_ready": False,
        "runtime_canary_hit": False,
        "optimizer_kind": OPTIMIZER_KIND,
        "optimizer_family": OPTIMIZER_FAMILY,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "native_dispatch_allowed": False,
        "canary_shadow_route_only": True,
        "source_scorecard": str(source.get("scorecard") or ""),
        "source_manifest_summary": dict(source.get("manifest_summary") or {}),
        "summary": {
            "runtime_canary_manifest_ready": ready,
            "runtime_canary_ready": False,
            "uses_quantized_adamw8bit_kernel_contract": True,
        },
        "promotion_blockers": _dedupe(blockers + ["training_loop_canary_missing", "product_rollout_review_missing"]),
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "run AdamW8bit TrainingLoop native canary"
            if ready
            else "complete AdamW8bit runtime canary manifest blockers"
        ),
        "notes": [
            "AdamW8bit has the same live uint8/qmap/absmax state roles as PagedAdamW8bit without paged residency.",
            "This manifest is report-only and does not enable native dispatch.",
        ],
    }


def _strings(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_adamw8bit_runtime_canary_scorecard"]
