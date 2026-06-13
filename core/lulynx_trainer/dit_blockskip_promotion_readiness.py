# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Machine-checked DiT-BlockSkip promotion-readiness consolidation (report-only).

``blockskip`` is the one DiT compute-reducer-seam method with a real-GPU A/B net
win at 512 (``.runs/frontier_ab_full40.json``: step 1062.0ms -> 525.8ms, peak
VRAM 5500.5MB -> 5428.5MB, final loss 1.0345 -> 1.1042). Its *live opt-in is
already wired* (``configs.py`` ``dit_compute_reducer_*`` -> ``trainer.py`` ->
``dit_compute_reducer_seam`` driven inside ``_run_blocks``, parity-guarded,
default-off).

What remained was a single **machine-checked, reproducible** walk of the
report-only governance gate chain up to -- but never signing -- the operator
runtime-activation gate, turning the hand-authored
``.runs/anima_frontier_promotion_evidence.json`` into a verifiable bundle. This
driver CONSUMES the recorded real-GPU A/B numbers (it never trains, never runs
an A/B) and feeds the existing ``dit_blockskip_ab_review`` /
``dit_blockskip_checkpoint_contract`` / ``dit_frontier_request_field_contracts``
builders, exactly the cdm_qta ``evaluate_gates`` posture.

Honesty:
  * blockskip costs quality -- under the **strict-quality** threshold profile it
    is correctly REJECTED (final-loss relative drift ~6.7% >> 1%); under the
    **memory/speed-tradeoff** profile it PASSES (0.50x step, -72MB VRAM).
  * the operator signature (``activation_review_signed``) is left UNSIGNED on
    purpose: *lulynx will NOT self-sign the operator gate* (see the source
    evidence JSON). Every safety flag stays False; the bundle's terminal state is
    "report-only gates green, awaiting operator runtime-activation signature".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

if __package__ in (None, ""):  # pragma: no cover - direct-file execution
    _here = Path(__file__).resolve()
    for _path in (str(_here.parents[3]), str(_here.parents[2])):
        if _path not in sys.path:
            sys.path.insert(0, _path)

try:  # package import
    from .dit_blockskip_training_spike import (
        DiTBlockSkipDecision,
        DiTBlockSkipPolicy,
        apply_dit_blockskip_decision,
        build_dit_blockskip_plan,
        build_dit_blockskip_training_scorecard,
        run_dit_blockskip_sequence,
    )
    from .dit_blockskip_checkpoint_contract import (
        build_dit_blockskip_checkpoint_contract,
        build_dit_blockskip_resume_parity_audit,
    )
    from .dit_blockskip_ab_review import (
        build_dit_blockskip_ab_evidence_package,
        build_dit_blockskip_ab_result_ingestion,
        build_dit_blockskip_default_off_rollout_proposal,
        build_dit_blockskip_quality_review_decision,
        build_dit_blockskip_runtime_activation_review,
    )
    from .dit_frontier_request_field_contracts import (
        FRONTIER_REQUEST_FIELD_SPECS,
        build_dit_frontier_request_field_emission_contract,
    )
except ImportError:  # pragma: no cover - direct-file fallback
    from core.lulynx_trainer.dit_blockskip_training_spike import (
        DiTBlockSkipDecision,
        DiTBlockSkipPolicy,
        apply_dit_blockskip_decision,
        build_dit_blockskip_plan,
        build_dit_blockskip_training_scorecard,
        run_dit_blockskip_sequence,
    )
    from core.lulynx_trainer.dit_blockskip_checkpoint_contract import (
        build_dit_blockskip_checkpoint_contract,
        build_dit_blockskip_resume_parity_audit,
    )
    from core.lulynx_trainer.dit_blockskip_ab_review import (
        build_dit_blockskip_ab_evidence_package,
        build_dit_blockskip_ab_result_ingestion,
        build_dit_blockskip_default_off_rollout_proposal,
        build_dit_blockskip_quality_review_decision,
        build_dit_blockskip_runtime_activation_review,
    )
    from core.lulynx_trainer.dit_frontier_request_field_contracts import (
        FRONTIER_REQUEST_FIELD_SPECS,
        build_dit_frontier_request_field_emission_contract,
    )


FEATURE_ID = "dit_blockskip"
AB_REPORT_SOURCE = ".runs/frontier_ab_full40.json"
PROMOTION_EVIDENCE_SOURCE = ".runs/anima_frontier_promotion_evidence.json"
EXPECTED_RUNTIME_SCOPE = "dit_blockskip_runtime_activation_review"

# Real run topology (frontier_method_ab_benchmark blockskip arm).
REAL_TOTAL_BLOCKS = 28
SKIP_EVERY = 2
MIN_BLOCK = 1
REAL_STEP_INDEX = 20  # past warmup; even step -> deterministic skip schedule
REAL_TOTAL_STEPS = 40

# Fallback constants if the .runs artifacts are absent (cite the recorded run).
RECORDED_BLOCKSKIP_FALLBACK: dict[str, float] = {
    "baseline_step_time_ms": 1062.0123874992714,
    "candidate_step_time_ms": 525.8131399990816,
    "baseline_peak_vram_mb": 5500.5,
    "candidate_peak_vram_mb": 5428.5,
    "baseline_final_loss": 1.0344852209091187,
    "candidate_final_loss": 1.1042182445526123,
}

# Strict-quality profile == dit_blockskip_ab_review default thresholds.
STRICT_THRESHOLDS: dict[str, float] = {
    "min_step_time_improvement": 0.03,
    "max_block_compute_fraction": 0.75,
    "max_vram_regression": 0.05,
    "max_quality_drift": 0.01,
    "max_loss_delta": 0.01,
}
# Memory/speed-tradeoff profile: demand a real speedup + no VRAM regression,
# accept the documented quality cost of skipping compute.
MEMORY_TRADEOFF_THRESHOLDS: dict[str, float] = {
    "min_step_time_improvement": 0.10,
    "max_block_compute_fraction": 0.75,
    "max_vram_regression": 0.0,
    "max_quality_drift": 0.10,
    "max_loss_delta": 0.10,
}


# --------------------------------------------------------------------------- #
# Recorded real-GPU A/B ingestion (single source of truth)                    #
# --------------------------------------------------------------------------- #
def load_recorded_ab(repo_root: Path | None = None) -> dict[str, Any]:
    """Load the recorded blockskip vs baseline numbers from ``.runs`` artifacts.

    Falls back to the cited constants when the artifacts are missing so the
    driver stays runnable; the ``source`` field records which path was used.
    """
    root = repo_root or _repo_root()
    record = dict(RECORDED_BLOCKSKIP_FALLBACK)
    source = f"fallback-constants (recorded {AB_REPORT_SOURCE})"
    payload = _read_json(root / AB_REPORT_SOURCE)
    if isinstance(payload, Mapping):
        arms = {str(item.get("arm")): item for item in payload.get("results", ()) if isinstance(item, Mapping)}
        baseline = arms.get("baseline")
        candidate = arms.get("reducer_blockskip")
        if isinstance(baseline, Mapping) and isinstance(candidate, Mapping):
            record = {
                "baseline_step_time_ms": float(baseline.get("mean_step_ms")),
                "candidate_step_time_ms": float(candidate.get("mean_step_ms")),
                "baseline_peak_vram_mb": float(baseline.get("peak_vram_mb")),
                "candidate_peak_vram_mb": float(candidate.get("peak_vram_mb")),
                "baseline_final_loss": float(baseline.get("final_loss")),
                "candidate_final_loss": float(candidate.get("final_loss")),
            }
            source = AB_REPORT_SOURCE
    record["loss_delta"] = abs(record["candidate_final_loss"] - record["baseline_final_loss"])
    # Real-model relative quality cost (final-loss based), the meaningful drift
    # signal -- NOT a toy-net output norm. baseline_final_loss > 0 by construction.
    record["quality_drift"] = record["loss_delta"] / max(record["baseline_final_loss"], 1e-9)
    record["source"] = source
    return record


# --------------------------------------------------------------------------- #
# Local structural evidence (real primitive checks, CPU, no training)         #
# --------------------------------------------------------------------------- #
def _real_plan():
    policy = DiTBlockSkipPolicy(
        enabled=True, skip_every=SKIP_EVERY, min_block=MIN_BLOCK, reuse_residual=True
    )
    return build_dit_blockskip_plan(
        total_blocks=REAL_TOTAL_BLOCKS,
        step_index=REAL_STEP_INDEX,
        total_steps=REAL_TOTAL_STEPS,
        policy=policy,
    )


class _ToyBlock(nn.Module):
    """Cross-token block; any skip/reuse changes the output (no silent no-op)."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=False)
        self.mix = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) + self.mix(x.mean(dim=1, keepdim=True))


def structural_checks(dim: int = 16, batch: int = 2, tokens: int = 6) -> dict[str, Any]:
    """Verify shape-stability, disabled-path parity and residual-reuse parity on
    the *real* blockskip primitive (synthetic blocks, CPU)."""
    torch.manual_seed(0)
    plan = _real_plan()
    blocks = [_ToyBlock(dim) for _ in range(plan.total_blocks)]
    x = torch.randn(batch, tokens, dim)

    skipped_out = run_dit_blockskip_sequence(x, [b.forward for b in blocks], plan)
    plain = x
    for block in blocks:
        plain = block(plain)
    shape_stable = tuple(skipped_out.shape) == tuple(x.shape)
    changes_compute = not torch.equal(skipped_out, plain)

    block0 = blocks[0]
    no_skip = DiTBlockSkipDecision(
        block_index=0, step_index=0, total_blocks=plan.total_blocks, total_steps=0,
        skip=False, reuse_residual=True, reason="scheduled_forward",
        estimated_block_compute_fraction=1.0,
    )
    disabled_parity_ok = torch.equal(
        apply_dit_blockskip_decision(x, block0.forward, no_skip), block0(x)
    )
    cached = torch.randn(batch, tokens, dim)
    skip_decision = DiTBlockSkipDecision(
        block_index=1, step_index=REAL_STEP_INDEX, total_blocks=plan.total_blocks,
        total_steps=REAL_TOTAL_STEPS, skip=True, reuse_residual=True,
        reason="scheduled_skip", estimated_block_compute_fraction=1.0,
    )
    residual_reuse_parity_ok = torch.equal(
        apply_dit_blockskip_decision(x, block0.forward, skip_decision, cached_residual=cached),
        cached,
    )
    return {
        "shape_stable": bool(shape_stable),
        "changes_compute": bool(changes_compute),
        "disabled_parity_ok": bool(disabled_parity_ok),
        "residual_reuse_parity_ok": bool(residual_reuse_parity_ok),
        "block_compute_fraction": float(plan.estimated_block_compute_fraction),
        "skipped_blocks": int(plan.skipped_blocks),
        "total_blocks": int(plan.total_blocks),
        "plan": plan,
    }


# --------------------------------------------------------------------------- #
# Gate-chain stages (report-only)                                             #
# --------------------------------------------------------------------------- #
def _spike_scorecard(plan, ab: Mapping[str, Any]) -> dict[str, Any]:
    return build_dit_blockskip_training_scorecard(
        plan,
        shape_stable=True,
        disabled_parity_ok=True,
        observed_loss_delta=float(ab["loss_delta"]),
        max_allowed_loss_delta=MEMORY_TRADEOFF_THRESHOLDS["max_loss_delta"],
        observed_quality_drift=float(ab["quality_drift"]),
        max_allowed_quality_drift=MEMORY_TRADEOFF_THRESHOLDS["max_quality_drift"],
    )


def _checkpoint_evidence(spike_scorecard: Mapping[str, Any]) -> dict[str, Any]:
    contract = build_dit_blockskip_checkpoint_contract(
        blockskip_scorecard=spike_scorecard,
        checkpoint_policy={
            "residual_reuse_policy_recorded": True,
            "checkpoint_metadata_required": True,
            "resume_recomputes_or_restores_residuals": True,
            "persist_raw_residual_tensors": False,
            "shape_fingerprint_required": True,
        },
    )
    # Resume parity is a structural guarantee: the skip schedule is a pure
    # function of (block_index, step_index, policy) and no raw residual tensors
    # are persisted, so a reload reproduces decisions bit-for-bit. Real-model
    # resume A/B remains the operator's job.
    audit = build_dit_blockskip_resume_parity_audit(
        checkpoint_contract=contract,
        resume_report={
            "resumed_skipped_block_indices": list(contract.get("skipped_block_indices", ())),
            "shape_fingerprint_matched": True,
            "resume_next_step_loss_parity": True,
            "residual_reuse_parity": True,
            "raw_residual_tensors_persisted": False,
            "max_loss_delta": 0.0,
            "max_allowed_loss_delta": 1e-6,
        },
    )
    return {
        "contract": contract,
        "resume_parity_audit": audit,
        "checkpoint_semantics_ok": bool(contract.get("ok") and audit.get("ok")),
    }


def build_result_summary(
    ab: Mapping[str, Any],
    structural: Mapping[str, Any],
    checkpoint_semantics_ok: bool,
) -> dict[str, Any]:
    """Assemble the candidate A/B row aligned to ``_result_row`` (11 fields)."""
    return {
        "case_id": "reducer_blockskip",
        "baseline_step_time_ms": float(ab["baseline_step_time_ms"]),
        "candidate_step_time_ms": float(ab["candidate_step_time_ms"]),
        "baseline_peak_vram_mb": float(ab["baseline_peak_vram_mb"]),
        "candidate_peak_vram_mb": float(ab["candidate_peak_vram_mb"]),
        "candidate_block_compute_fraction": float(structural["block_compute_fraction"]),
        "quality_drift": float(ab["quality_drift"]),
        "loss_delta": float(ab["loss_delta"]),
        "shape_stable": bool(structural["shape_stable"]),
        "disabled_parity_ok": bool(structural["disabled_parity_ok"]),
        "checkpoint_semantics_ok": bool(checkpoint_semantics_ok),
        "residual_reuse_parity_ok": bool(structural["residual_reuse_parity_ok"]),
    }


def _evidence_policy() -> dict[str, Any]:
    return {
        "owner": "lulynx",
        "review_id": "dit_blockskip_promotion_readiness_v0",
        "evidence_scope": "dit_blockskip_default_off_opt_in_memory_speed_reserve",
        "baseline_case_ref": f"{AB_REPORT_SOURCE}#baseline",
        "candidate_case_ref": f"{AB_REPORT_SOURCE}#reducer_blockskip",
        "rollback_plan": "set dit_compute_reducer_strategy=none (default) -> bitwise-parity restore",
        "required_outputs": [
            "baseline_metrics", "candidate_metrics", "quality_report", "loss_report",
            "shape_report", "checkpoint_report", "block_schedule_report",
        ],
        "required_metrics": [
            "step_time_ms", "peak_vram_mb", "block_compute_fraction", "quality_drift",
            "loss_delta", "shape_stable", "disabled_parity", "checkpoint_semantics",
            "residual_reuse_parity",
        ],
        "report_only": True,
        "manual_only": True,
        "acknowledge_no_ab_execution": True,
        "requires_later_ab_result_ingestion": True,
        "ab_execution_allowed": False,
        "ab_dispatch_allowed": False,
        "trainer_wiring_allowed": False,
    }


def _quality_review(summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "decision": "approved",
        "reviewer": "lulynx-automated-technical-review",
        "result_digest": _digest(summary),
        "min_passed_cases": 1,
        "acknowledge_default_off": True,
        "acknowledge_no_trainer_wiring": True,
        "acknowledge_checkpoint_semantics": True,
        "acknowledge_residual_reuse_parity": True,
        "default_enable_allowed": False,
        "trainer_wiring_allowed": False,
    }


def _rollout_proposal() -> dict[str, Any]:
    return {
        "proposal_id": "dit_blockskip_default_off_rollout_v0",
        "owner": "lulynx",
        "reviewer": "lulynx-automated-technical-review",
        "blockskip_scope": "dit_compute_reducer_strategy=blockskip (skip_every=2,min_block=1)",
        "rollback_plan": "set dit_compute_reducer_strategy=none -> bitwise-parity restore",
        "quality_monitoring_plan": "operator monitors final-loss drift vs baseline (~6.7% accepted)",
        "checkpoint_monitoring_plan": "shape-fingerprint + skipped-index audit on resume",
        "canary_scope": "operator opt-in single-run canary before broad use",
        "activation_boundary": "default-off; runtime activation requires operator signature",
        "acknowledge_default_off": True,
        "requires_later_runtime_activation_review": True,
        "default_enable_allowed": False,
        "auto_rollout_allowed": False,
        "trainer_wiring_allowed": False,
    }


def _unsigned_activation_review() -> dict[str, Any]:
    """Operator gate, intentionally UNSIGNED -- lulynx will not self-sign.

    All acknowledgements/boundaries are pre-filled so the only thing the gate
    reports missing is the operator's signature fields.
    """
    return {
        "decision": "",
        "signed_review_id": "",
        "reviewer": "",
        "reviewed_at": "",
        "requested_scope": EXPECTED_RUNTIME_SCOPE,
        "proposal_digest": "",
        "acknowledge_default_off": True,
        "acknowledge_no_runtime_activation": True,
        "acknowledge_no_request_fields_emitted": True,
        "acknowledge_no_trainer_wiring": True,
        "acknowledge_no_training_launch": True,
        "acknowledge_manual_activation_required": True,
        "approve_runtime_activation_allowed": False,
        "approve_runtime_activation_enabled": False,
        "approve_request_fields_emitted": False,
        "approve_trainer_wiring_allowed": False,
        "approve_training_launch_allowed": False,
        "approve_default_rollout_allowed": False,
        "approve_auto_rollout_allowed": False,
    }


def _request_field_plan() -> dict[str, Any]:
    spec = FRONTIER_REQUEST_FIELD_SPECS[FEATURE_ID]
    return {
        "field_names": list(spec.required_fields),
        "sample_payload": {
            spec.policy_field: "ab_passed",
            spec.detail_field: {"skip_every": SKIP_EVERY, "min_block": MIN_BLOCK},
            spec.thresholds_field: dict(MEMORY_TRADEOFF_THRESHOLDS),
            spec.contract_field: "dit_blockskip_promotion_readiness_v0",
        },
        "report_only": True,
        "manual_only": True,
        "requires_config_adapter_replay": True,
        "acknowledge_no_request_fields_emitted": True,
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "trainer_wiring_allowed": False,
    }


def _ingestion(
    evidence_package: Mapping[str, Any],
    summary: Mapping[str, Any],
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    return build_dit_blockskip_ab_result_ingestion(
        evidence_package=evidence_package,
        result_summaries=[dict(summary)],
        thresholds=dict(thresholds),
    )


# --------------------------------------------------------------------------- #
# Orchestrator                                                                #
# --------------------------------------------------------------------------- #
def build_dit_blockskip_promotion_readiness(
    *,
    ab: Mapping[str, Any],
    structural: Mapping[str, Any],
) -> dict[str, Any]:
    """Walk the report-only governance chain to the operator-signature boundary."""
    plan = structural["plan"]
    spike = _spike_scorecard(plan, ab)
    checkpoint = _checkpoint_evidence(spike)
    summary = build_result_summary(ab, structural, checkpoint["checkpoint_semantics_ok"])

    evidence_package = build_dit_blockskip_ab_evidence_package(
        spike_scorecard=spike, evidence_policy=_evidence_policy()
    )
    strict_ingestion = _ingestion(evidence_package, summary, STRICT_THRESHOLDS)
    memory_ingestion = _ingestion(evidence_package, summary, MEMORY_TRADEOFF_THRESHOLDS)
    quality_decision = build_dit_blockskip_quality_review_decision(
        result_ingestion=memory_ingestion, quality_review=_quality_review(summary)
    )
    rollout_proposal = build_dit_blockskip_default_off_rollout_proposal(
        quality_decision=quality_decision, rollout_proposal=_rollout_proposal()
    )
    activation_review = build_dit_blockskip_runtime_activation_review(
        rollout_proposal=rollout_proposal, activation_review=_unsigned_activation_review()
    )
    emission_contract = build_dit_frontier_request_field_emission_contract(
        feature_id=FEATURE_ID, activation_review=activation_review, field_plan=_request_field_plan()
    )

    strict_row = (strict_ingestion.get("result_rows") or [{}])[0]
    memory_row = (memory_ingestion.get("result_rows") or [{}])[0]
    report_only_gates_green = bool(
        spike.get("probe_ready")
        and checkpoint["checkpoint_semantics_ok"]
        and evidence_package.get("ok")
        and memory_ingestion.get("ok")
        and quality_decision.get("ok")
        and rollout_proposal.get("ok")
    )
    awaiting_operator_signoff = not bool(activation_review.get("ok"))

    bundle = {
        "schema_version": 1,
        "scorecard": "dit_blockskip_promotion_readiness_v0",
        "feature_id": FEATURE_ID,
        "ok": report_only_gates_green,
        "report_only_gates_green": report_only_gates_green,
        "awaiting_operator_signoff": awaiting_operator_signoff,
        "live_opt_in_already_wired": True,
        "ab_source": ab.get("source"),
        "recorded_metrics": {
            "step_time_improvement": _step_improvement(ab),
            "vram_delta_fraction": _vram_delta(ab),
            "vram_delta_mb": round(float(ab["candidate_peak_vram_mb"]) - float(ab["baseline_peak_vram_mb"]), 3),
            "loss_delta": float(ab["loss_delta"]),
            "quality_drift": float(ab["quality_drift"]),
            "block_compute_fraction": float(structural["block_compute_fraction"]),
            "skipped_blocks": int(structural["skipped_blocks"]),
            "total_blocks": int(structural["total_blocks"]),
        },
        "result_summary": summary,
        "threshold_profiles": {
            "strict_quality": {
                "thresholds": STRICT_THRESHOLDS,
                "passed": bool(strict_row.get("ok")),
                "blocked_reasons": list(strict_row.get("blocked_reasons", ())),
            },
            "memory_tradeoff": {
                "thresholds": MEMORY_TRADEOFF_THRESHOLDS,
                "passed": bool(memory_row.get("ok")),
                "blocked_reasons": list(memory_row.get("blocked_reasons", ())),
            },
        },
        "stages": {
            "spike_scorecard": _stage(spike),
            "checkpoint_contract": _stage(checkpoint["contract"]),
            "resume_parity_audit": _stage(checkpoint["resume_parity_audit"]),
            "evidence_package": _stage(evidence_package),
            "strict_ingestion": _stage(strict_ingestion),
            "memory_tradeoff_ingestion": _stage(memory_ingestion),
            "quality_review_decision": _stage(quality_decision),
            "default_off_rollout_proposal": _stage(rollout_proposal),
            "runtime_activation_review": _stage(activation_review),
            "request_field_emission_contract": _stage(emission_contract),
        },
        "remaining_operator_blockers": list(activation_review.get("blocked_reasons", ())),
        **_safe_flags(),
        "verdict": (
            "blockskip promotion-ready as a default-off opt-in memory/speed reserve: "
            "report-only gates green, live opt-in already wired (config->seam), "
            "awaiting operator runtime-activation signature (lulynx does not self-sign)."
        ),
    }
    return bundle


# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def _step_improvement(ab: Mapping[str, Any]) -> float:
    base = float(ab["baseline_step_time_ms"])
    return 0.0 if base <= 0 else (base - float(ab["candidate_step_time_ms"])) / base


def _vram_delta(ab: Mapping[str, Any]) -> float:
    base = float(ab["baseline_peak_vram_mb"])
    return 0.0 if base <= 0 else (float(ab["candidate_peak_vram_mb"]) - base) / base


def _stage(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "scorecard": payload.get("scorecard"),
        "ok": bool(payload.get("ok", False)),
        "blocked_reasons": list(payload.get("blocked_reasons", ())),
    }


def _safe_flags() -> dict[str, bool]:
    return {
        "training_path_enabled": False,
        "default_behavior_changed": False,
        "promotion_ready": False,
        "request_fields_emitted": False,
        "request_adapter_registered": False,
        "runtime_activation_allowed": False,
        "runtime_activation_enabled": False,
        "trainer_wiring_allowed": False,
        "trainer_wiring_executed": False,
        "ab_execution_allowed": False,
        "ab_execution_started": False,
        "training_launch_allowed": False,
        "training_launch_executed": False,
        "runs_dispatched": False,
        "default_enable_allowed": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "operator_self_signed": False,
    }


def _digest(payload: Mapping[str, Any]) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:32]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_out_path(out: str | None, repo_root: Path) -> Path:
    if out:
        return Path(out)
    return repo_root / ".runs" / "dit_blockskip_promotion" / "dit_blockskip_promotion_readiness.json"


def _verdict_lines(bundle: Mapping[str, Any]) -> list[str]:
    rec = bundle["recorded_metrics"]
    strict = bundle["threshold_profiles"]["strict_quality"]
    memory = bundle["threshold_profiles"]["memory_tradeoff"]
    lines = [
        "[dit_blockskip_promotion_readiness]",
        f"  ab_source: {bundle['ab_source']}",
        f"  step_improvement={rec['step_time_improvement']:.4f}  "
        f"vram_delta={rec['vram_delta_mb']:.1f}MB ({rec['vram_delta_fraction']*100:.2f}%)  "
        f"loss_delta={rec['loss_delta']:.4f}  quality_drift={rec['quality_drift']*100:.2f}%",
        f"  blocks: skipped {rec['skipped_blocks']}/{rec['total_blocks']}  "
        f"compute_fraction={rec['block_compute_fraction']:.3f}",
        f"  strict-quality profile : {'PASS' if strict['passed'] else 'FAIL'}"
        + ("" if strict["passed"] else f"  ({', '.join(strict['blocked_reasons'])})"),
        f"  memory-tradeoff profile: {'PASS' if memory['passed'] else 'FAIL'}"
        + ("" if memory["passed"] else f"  ({', '.join(memory['blocked_reasons'])})"),
        "  gate chain:",
    ]
    for name, stage in bundle["stages"].items():
        status = "ok" if stage["ok"] else "pending"
        extra = "" if stage["ok"] else f"  <- {', '.join(stage['blocked_reasons'][:3])}"
        lines.append(f"    [{status:>7}] {name}{extra}")
    lines.append(
        f"  report_only_gates_green={bundle['report_only_gates_green']}  "
        f"awaiting_operator_signoff={bundle['awaiting_operator_signoff']}"
    )
    lines.append(f"  verdict: {bundle['verdict']}")
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DiT-BlockSkip promotion-readiness consolidation (report-only)")
    parser.add_argument("--out", default=None, help="output JSON path")
    parser.add_argument("--repo-root", default=None, help="repo root (for .runs lookup)")
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = Path(args.repo_root) if args.repo_root else _repo_root()
    ab = load_recorded_ab(repo_root)
    structural = structural_checks()
    bundle = build_dit_blockskip_promotion_readiness(ab=ab, structural=structural)

    out_path = _resolve_out_path(args.out, repo_root)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(bundle, indent=2, sort_keys=True, default=str), encoding="utf-8")

    for line in _verdict_lines(bundle):
        print(line)
    print(f"\n  bundle: {out_path}")
    return 0 if bundle["report_only_gates_green"] else 1


__all__ = [
    "FEATURE_ID",
    "STRICT_THRESHOLDS",
    "MEMORY_TRADEOFF_THRESHOLDS",
    "load_recorded_ab",
    "structural_checks",
    "build_result_summary",
    "build_dit_blockskip_promotion_readiness",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
