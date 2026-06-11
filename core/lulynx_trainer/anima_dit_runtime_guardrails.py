"""Shared Anima DiT runtime guardrails.

This module is deliberately lightweight so contract smokes can validate the
Anima full-finetune boundary without importing the full trainer stack.
"""

from __future__ import annotations

from typing import Any, Callable, Dict


LogFn = Callable[[str], None]


def apply_anima_dit_runtime_guardrails(
    *,
    config: Any,
    model: Any,
    device: Any,
    dtype: Any,
    checkpoint_auto_recommended: bool = False,
    log: LogFn | None = None,
) -> Dict[str, Any]:
    """Apply Anima DiT checkpoint/residency knobs and return profiles."""

    result: Dict[str, Any] = {
        "checkpoint_profile": {},
        "residency_profile": {},
        "config_updates": {},
    }
    if model is None or getattr(model, "unet", None) is None:
        return result

    def _log(message: str) -> None:
        if log is not None:
            log(message)

    residency_mode = str(getattr(config, "anima_block_residency", "resident") or "resident")
    checkpoint_requested = bool(getattr(config, "anima_block_checkpointing", False))
    checkpoint_mode = str(getattr(config, "anima_block_checkpointing_mode", "block") or "block")
    checkpoint_source = "anima_block_checkpointing"
    if not checkpoint_requested and bool(checkpoint_auto_recommended):
        checkpoint_requested = True
        checkpoint_source = "auto_for_dit_residency"
        try:
            setattr(config, "anima_block_checkpointing", True)
            result["config_updates"]["anima_block_checkpointing"] = True
        except Exception:
            pass
        _log(
            "Anima block checkpointing auto-enabled for "
            f"residency={residency_mode}; non-resident 1024/4096-token DiT paths need activation recompute."
        )

    if checkpoint_requested:
        try:
            setter = getattr(model.unet, "set_anima_block_checkpointing", None)
            if not callable(setter):
                raise RuntimeError("native Anima DiT does not expose set_anima_block_checkpointing")
            profile = dict(setter(True, checkpoint_mode))
            profile["source"] = checkpoint_source
            result["checkpoint_profile"] = profile
            _log(
                "Anima block checkpointing: "
                f"mode={profile.get('mode', 'block')}, "
                f"checkpointed_blocks={profile.get('checkpointed_blocks', 0)}/{profile.get('block_count', 0)}, "
                f"source={checkpoint_source}"
            )
        except Exception as exc:
            result["checkpoint_profile"] = {
                "enabled": False,
                "mode": checkpoint_mode,
                "error": str(exc),
            }
            _log(f"Anima block checkpointing skipped: {exc}")

    if residency_mode.strip().lower().replace("-", "_") == "resident":
        return result

    try:
        from .anima_block_residency import apply_anima_block_residency

        min_params = max(int(getattr(config, "anima_block_residency_min_params", 0) or 0), 0)
        prefetch_enabled = bool(getattr(config, "anima_block_prefetch", False))
        prefetch_depth = max(int(getattr(config, "anima_block_prefetch_depth", 1) or 0), 0)
        pcie_transfer_format = str(getattr(config, "pcie_transfer_format", "off") or "off").strip().lower()
        if pcie_transfer_format in {"off", "none", "disabled"}:
            pcie_transfer_format = ""
        sparse_swap_enabled = bool(getattr(config, "sparse_swap_enabled", False))
        sparse_swap_budget_mb = max(float(getattr(config, "sparse_swap_budget_mb", 0.0) or 0.0), 0.0)
        sparse_swap_warm_fraction = min(max(float(getattr(config, "sparse_swap_warm_fraction", 0.35) or 0.35), 0.0), 1.0)
        pcie_delta_cache_enabled = bool(getattr(config, "pcie_delta_cache_enabled", False))
        pcie_delta_cache_mode = str(getattr(config, "pcie_delta_cache_mode", "observe") or "observe")
        pcie_delta_cache_budget_mb = max(float(getattr(config, "pcie_delta_cache_budget_mb", 256.0) or 0.0), 0.0)
        report = apply_anima_block_residency(
            model.unet,
            mode=residency_mode,
            min_parameter_count=min_params,
            device=device,
            dtype=dtype,
            prefetch_enabled=prefetch_enabled,
            prefetch_depth=prefetch_depth,
            transfer_format=pcie_transfer_format or None,
            sparse_swap_enabled=sparse_swap_enabled,
            sparse_swap_budget_mb=sparse_swap_budget_mb or None,
            sparse_swap_warm_fraction=sparse_swap_warm_fraction,
            pcie_delta_cache_enabled=pcie_delta_cache_enabled,
            pcie_delta_cache_mode=pcie_delta_cache_mode,
            pcie_delta_cache_budget_mb=pcie_delta_cache_budget_mb,
        )
        profile = report.as_dict()
        result["residency_profile"] = profile
        prefetch = profile.get("prefetch", {})
        prefetch_tail = (
            f", prefetch={'on' if prefetch.get('enabled') else 'off'}"
            f" depth={prefetch.get('depth', prefetch_depth)}"
            f" reason={prefetch.get('reason', '')}"
        )
        cache_profile = profile.get("pcie_delta_cache", {})
        cache_tail = (
            f", delta_cache={'on' if cache_profile.get('enabled') else 'off'}"
            f" candidates={cache_profile.get('candidate_count', 0)}"
            f" high={cache_profile.get('high_value_count', 0)}"
        )
        cache_v0_profile = profile.get("pcie_cache_v0", {})
        cache_v0_tail = (
            f", cache_v0={'on' if cache_v0_profile.get('enabled') else 'off'}"
            f" selected={cache_v0_profile.get('selected_count', 0)}"
            f" cache={float(cache_v0_profile.get('cache_mb', 0.0) or 0.0):.1f}MB"
        )
        auto_threshold_tail = ""
        if report.auto_min_parameter_count:
            auto_threshold_tail = (
                f", auto_min=yes requested_min={report.requested_min_parameter_count}, "
                f"auto_candidates={report.auto_threshold_candidate_count}, "
                f"auto_cold_params={report.auto_threshold_total_parameter_count}"
            )
        _log(
            "Anima block residency: "
            f"mode={report.mode}, strategy={report.strategy}, blocks={report.block_count}, "
            f"planned_linear={report.planned_linear_count}, "
            f"active_linear={report.active_linear_count}/{report.managed_linear_count}, "
            f"lora_wrapped={report.lora_wrapped_linear_count}, "
            f"hot_resident={report.hot_resident_count}, edge_resident={report.edge_resident_count}, "
            f"cold_candidates={report.cold_candidate_count}, "
            f"sparse_swap={'on' if report.sparse_swap_enabled else 'off'}"
            f"(warm={report.sparse_warm_prefetch_count}, cold={report.sparse_cold_on_demand_count}), "
            f"min_params={report.min_parameter_count}{auto_threshold_tail}, "
            f"skipped_small={report.skipped_small_count}, "
            f"planned_cpu_params={report.planned_cpu_parameter_mb:.1f}MB, "
            f"cpu_params={report.cpu_parameter_mb:.1f}MB, "
            f"transfer_format={report.transfer_format}, "
            f"packed_linear={report.transfer_packed_linear_count}, "
            f"transfer_h2d={report.transfer_h2d_mb:.1f}MB"
            f"{prefetch_tail}"
            f"{cache_tail}"
            f"{cache_v0_tail}"
        )
    except Exception as exc:
        result["residency_profile"] = {"mode": residency_mode, "error": str(exc)}
        _log(f"Anima block residency skipped: {exc}")
    return result


__all__ = ["apply_anima_dit_runtime_guardrails"]
