"""Default-off trainer gate for DiT compute reducer research probes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


REQUIRED_DIT_COMPUTE_REDUCERS: tuple[str, ...] = ("tread", "diffcr", "blockskip", "local_window_attention")


@dataclass(frozen=True)
class DiTComputeReducerGateRow:
    reducer_id: str
    present: bool
    probe_ready: bool
    enabled: bool
    shape_stable: bool
    disabled_parity_ok: bool
    loss_gate_ready: bool
    loss_gate_ok: bool
    quality_gate_ready: bool
    quality_gate_ok: bool
    estimated_compute_fraction: float
    training_path_enabled: bool
    default_behavior_changed: bool
    promotion_ready: bool
    blocked_reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "reducer_id": self.reducer_id,
            "present": bool(self.present),
            "probe_ready": bool(self.probe_ready),
            "enabled": bool(self.enabled),
            "shape_stable": bool(self.shape_stable),
            "disabled_parity_ok": bool(self.disabled_parity_ok),
            "loss_gate_ready": bool(self.loss_gate_ready),
            "loss_gate_ok": bool(self.loss_gate_ok),
            "quality_gate_ready": bool(self.quality_gate_ready),
            "quality_gate_ok": bool(self.quality_gate_ok),
            "estimated_compute_fraction": float(self.estimated_compute_fraction),
            "training_path_enabled": bool(self.training_path_enabled),
            "default_behavior_changed": bool(self.default_behavior_changed),
            "promotion_ready": bool(self.promotion_ready),
            "blocked_reasons": list(self.blocked_reasons),
        }


def build_dit_compute_reducer_trainer_gate(
    evidence: Sequence[Mapping[str, Any]],
    *,
    required_reducers: Sequence[str] = REQUIRED_DIT_COMPUTE_REDUCERS,
) -> dict[str, Any]:
    by_reducer = {_reducer_id(item): dict(item) for item in evidence if _reducer_id(item)}
    rows = tuple(_row(reducer, by_reducer.get(reducer)) for reducer in required_reducers)
    blockers = [f"{row.reducer_id}:{reason}" for row in rows for reason in row.blocked_reasons]
    contract_ready = all(row.present and row.probe_ready for row in rows)
    trainer_ready = contract_ready and not blockers
    return {
        "schema_version": 1,
        "scorecard": "dit_compute_reducer_trainer_gate_v0",
        "ok": contract_ready,
        "contract_ready": contract_ready,
        "trainer_ready": trainer_ready,
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "required_reducers": list(required_reducers),
        "present_count": sum(1 for row in rows if row.present),
        "rows": [row.as_dict() for row in rows],
        "blocked_reasons": blockers,
        "recommended_next_step": "collect shape, disabled-parity, loss-parity, and quality-drift evidence before trainer wiring",
    }


def _row(reducer_id: str, payload: Mapping[str, Any] | None) -> DiTComputeReducerGateRow:
    if payload is None:
        return DiTComputeReducerGateRow(
            reducer_id=reducer_id,
            present=False,
            probe_ready=False,
            enabled=False,
            shape_stable=False,
            disabled_parity_ok=False,
            loss_gate_ready=False,
            loss_gate_ok=False,
            quality_gate_ready=False,
            quality_gate_ok=False,
            estimated_compute_fraction=1.0,
            training_path_enabled=False,
            default_behavior_changed=False,
            promotion_ready=False,
            blocked_reasons=("missing_evidence",),
        )
    plan = _plan(payload)
    loss_gate = dict(payload.get("loss_gate") or {})
    quality_gate = dict(payload.get("quality_gate") or {})
    shape_stable = bool(payload.get("shape_stable", payload.get("shape_stability_ok", False)))
    disabled_parity_ok = bool(payload.get("disabled_parity_ok", payload.get("disabled_path_parity_ok", False)))
    loss_ready = bool(loss_gate.get("ready", payload.get("loss_gate_ready", False)))
    loss_ok = bool(loss_gate.get("ok", payload.get("loss_gate_ok", False)))
    quality_ready = bool(quality_gate.get("ready", payload.get("quality_gate_ready", False)))
    quality_ok = bool(quality_gate.get("ok", payload.get("quality_gate_ok", False)))
    training_path_enabled = bool(payload.get("training_path_enabled", False))
    default_behavior_changed = bool(payload.get("default_behavior_changed", False))
    promotion_ready = bool(payload.get("promotion_ready", False))
    enabled = bool(plan.get("enabled", payload.get("enabled", False)))
    probe_ready = bool(payload.get("probe_ready", payload.get("facade_ready", True)))
    blockers: list[str] = []
    if not shape_stable:
        blockers.append("shape_stability_evidence_missing")
    if not disabled_parity_ok:
        blockers.append("disabled_parity_evidence_missing")
    if not loss_ready:
        blockers.append("loss_parity_gate_missing")
    elif not loss_ok:
        blockers.append("loss_parity_gate_failed")
    if not quality_ready:
        blockers.append("quality_gate_missing")
    elif not quality_ok:
        blockers.append("quality_gate_failed")
    if training_path_enabled:
        blockers.append("unsafe_training_path_enabled")
    if default_behavior_changed:
        blockers.append("default_behavior_changed")
    if promotion_ready:
        blockers.append("unexpected_promotion_ready")
    return DiTComputeReducerGateRow(
        reducer_id=reducer_id,
        present=True,
        probe_ready=probe_ready,
        enabled=enabled,
        shape_stable=shape_stable,
        disabled_parity_ok=disabled_parity_ok,
        loss_gate_ready=loss_ready,
        loss_gate_ok=loss_ok,
        quality_gate_ready=quality_ready,
        quality_gate_ok=quality_ok,
        estimated_compute_fraction=_compute_fraction(plan, payload),
        training_path_enabled=training_path_enabled,
        default_behavior_changed=default_behavior_changed,
        promotion_ready=promotion_ready,
        blocked_reasons=tuple(blockers),
    )


def _plan(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = payload.get("plan")
    if isinstance(value, Mapping):
        return dict(value)
    return dict(payload)


def _compute_fraction(plan: Mapping[str, Any], payload: Mapping[str, Any]) -> float:
    for key in ("estimated_attention_fraction", "estimated_block_compute_fraction"):
        if key in plan:
            return _clamp_fraction(plan[key])
        if key in payload:
            return _clamp_fraction(payload[key])
    return 1.0


def _clamp_fraction(value: Any) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 1.0


def _reducer_id(payload: Mapping[str, Any]) -> str:
    explicit = payload.get("reducer_id")
    if explicit:
        return _normalize_reducer_id(str(explicit))
    marker = str(payload.get("scorecard") or payload.get("plan") or payload.get("probe") or "")
    return _normalize_reducer_id(marker)


def _normalize_reducer_id(value: str) -> str:
    lowered = str(value).strip().lower().replace("-", "_")
    if "tread" in lowered:
        return "tread"
    if "diffcr" in lowered:
        return "diffcr"
    if "blockskip" in lowered or "block_skip" in lowered:
        return "blockskip"
    if "local_window" in lowered or "window_attention" in lowered:
        return "local_window_attention"
    return lowered


__all__ = [
    "DiTComputeReducerGateRow",
    "REQUIRED_DIT_COMPUTE_REDUCERS",
    "build_dit_compute_reducer_trainer_gate",
]
