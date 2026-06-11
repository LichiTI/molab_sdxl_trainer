"""Checkpoint/resume native-state boundary evidence for V5 manual canary.

This is a report-only live contract probe. It reuses the TrainingLoop
TurboCore checkpoint hooks to prove that native owner state can round-trip,
compatible resume leaves owner state pending, incompatible resume is rejected,
and defaults stay off.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import torch

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.lulynx_trainer.training_loop import TrainingLoop
from core.turbocore_v5_owner_review_evidence_package import load_json


class _Injector:
    def __init__(self, params: list[torch.nn.Parameter]) -> None:
        self.params = params

    def get_trainable_params(self) -> list[torch.nn.Parameter]:
        return self.params


def build_v5_checkpoint_resume_native_state_boundary_evidence(
    *,
    explicit_run_manifest: Mapping[str, Any] | None = None,
    explicit_run_audit: Mapping[str, Any] | None = None,
    run_live_probe: bool = True,
) -> dict[str, Any]:
    manifest = _as_dict(explicit_run_manifest)
    audit = _as_dict(explicit_run_audit)
    live = _run_live_probe() if run_live_probe else _skipped("live_probe_disabled")
    gates = {
        "explicit_manifest_ready_or_not_required": _manifest_ready_or_absent(manifest),
        "checkpoint_metadata_integrated": bool(live.get("checkpoint_metadata_integrated", False)),
        "owner_state_included": bool(live.get("owner_state_included", False)),
        "checkpoint_contract_roundtrip_ok": bool(live.get("checkpoint_contract_roundtrip_ok", False)),
        "compatible_resume_accepted": bool(live.get("restore_loaded", False) and live.get("restore_compatible", False)),
        "owner_state_pending_after_resume": bool(live.get("owner_state_pending", False)),
        "mismatched_resume_rejected": bool(live.get("mismatch_loaded", False) and not live.get("mismatch_compatible", True)),
        "disabled_shadow_checkpoint_default_off": bool(live.get("disabled_checkpoint_default_off", False)),
        "training_path_stays_default_off": bool(live.get("training_path_stays_default_off", False)),
        "default_behavior_unchanged": True,
    }
    ready = all(gates.values())
    blocked = [f"v5_p24_{name}_missing" for name, ok in gates.items() if not ok]
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v5_checkpoint_resume_native_state_boundary_evidence_v0",
        "gate": "v5_checkpoint_resume_native_state_boundary",
        "ok": ready,
        "milestone_completed": ready,
        "checkpoint_resume_native_state_boundary_ready": ready,
        "checkpoint_resume_native_state_boundary": ready,
        "checkpoint_roundtrip_ok": bool(live.get("checkpoint_contract_roundtrip_ok", False)),
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "request_adapter_mapping_allowed": False,
        "manifest_summary": _manifest_summary(manifest),
        "run_audit_summary": _audit_summary(audit),
        "live_probe": live,
        "runtime_evidence_patch": {
            "checkpoint_resume_native_state_boundary": ready,
        },
        "progress_gates": gates,
        "blocked_reasons": _dedupe(blocked + _string_list(live.get("blocked_reasons"))),
        "promotion_blockers": _dedupe(blocked + _string_list(live.get("blocked_reasons"))),
        "recommended_next_step": (
            "replay P22/P23 with checkpoint boundary evidence; default rollout remains off"
            if ready
            else "fix checkpoint/resume native-state boundary blockers"
        ),
        "notes": [
            "This probe does not run long training.",
            "Compatible resume leaves owner-native state pending; it does not auto-enable dispatch.",
            "P24 evidence may satisfy the P22 checkpoint_resume_native_state_boundary field.",
        ],
    }


def _run_live_probe() -> dict[str, Any]:
    loop = _make_loop(shadow_mode="shadow", save_owner_state=True)
    _prime_shadow_owner(loop)
    checkpoint = loop.get_turbocore_update_checkpoint_state()
    restore = loop.load_turbocore_update_checkpoint_state(checkpoint)
    mismatch = loop.load_turbocore_update_checkpoint_state(_mismatched_checkpoint(checkpoint))
    disabled = _make_loop(shadow_mode="off", save_owner_state=False).get_turbocore_update_checkpoint_state()
    contract = _as_dict(checkpoint.get("checkpoint_contract"))
    return {
        "schema_version": 1,
        "probe": "v5_checkpoint_resume_native_state_boundary_live_probe_v0",
        "ok": True,
        "checkpoint_metadata_integrated": bool(checkpoint.get("checkpoint_metadata_integrated", False)),
        "trainer_state_metadata_integrated": bool(checkpoint.get("trainer_state_metadata_integrated", False)),
        "owner_state_included": bool(checkpoint.get("owner_state_included", False)),
        "parameter_tensors": int(checkpoint.get("parameter_tensors", 0) or 0),
        "parameter_numel": int(checkpoint.get("parameter_numel", 0) or 0),
        "checkpoint_contract_roundtrip_ok": bool(contract.get("roundtrip_ok", False)),
        "checkpoint_contract_roundtrip_checked": bool(contract.get("roundtrip_checked", False)),
        "restore_loaded": bool(restore.get("loaded", False)),
        "restore_compatible": bool(restore.get("compatible", False)),
        "owner_state_pending": bool(restore.get("owner_state_pending", False)),
        "mismatch_loaded": bool(mismatch.get("loaded", False)),
        "mismatch_compatible": bool(mismatch.get("compatible", False)),
        "mismatch_owner_state_pending": bool(mismatch.get("owner_state_pending", False)),
        "disabled_checkpoint_default_off": bool(
            disabled.get("enabled") is False and disabled.get("training_path_enabled") is False
        ),
        "training_path_stays_default_off": bool(
            checkpoint.get("training_path_enabled") is False
            and restore.get("training_path_enabled") is False
            and mismatch.get("training_path_enabled") is False
        ),
        "blocked_reasons": [],
    }


def _make_loop(*, shadow_mode: str, save_owner_state: bool) -> TrainingLoop:
    param = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5], dtype=torch.float32))
    optimizer = torch.optim.AdamW([param], lr=1e-3, weight_decay=0.0)
    loop = TrainingLoop(
        unet=torch.nn.Identity(),
        text_encoder_1=torch.nn.Identity(),
        text_encoder_2=None,
        vae=torch.nn.Identity(),
        tokenizer_1=None,
        tokenizer_2=None,
        noise_scheduler=None,
        lora_injector=_Injector([param]),
        optimizer=optimizer,
        lr_scheduler=None,
        device="cpu",
        dtype=torch.float32,
        gradient_accumulation_steps=1,
        max_grad_norm=1000.0,
        layer_monitor_enabled=False,
        vram_smart_sensing_enabled=False,
        turbocore_update_shadow_mode=shadow_mode,
        turbocore_update_shadow_checkpoint_contract=True,
        turbocore_update_shadow_save_owner_state=save_owner_state,
    )
    loop.total_steps = 1
    return loop


def _prime_shadow_owner(loop: TrainingLoop) -> None:
    params = loop._get_trainable_params()
    for param in params:
        param.grad = torch.full_like(param, 0.125)
    loop._turbocore_update_shadow.prepare_before_optimizer(
        params,
        optimizer=loop.optimizer,
        max_grad_norm=loop.max_grad_norm,
        step=0,
    )


def _mismatched_checkpoint(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(checkpoint))
    owner_state = _as_dict(payload.get("owner_state_dict"))
    layout = _as_dict(owner_state.get("layout"))
    layout["total_numel"] = int(layout.get("total_numel", 0) or 0) + 1
    owner_state["layout"] = layout
    payload["owner_state_dict"] = owner_state
    return payload


def _manifest_ready_or_absent(manifest: Mapping[str, Any]) -> bool:
    return not manifest or bool(manifest.get("explicit_run_manifest_ready", False))


def _manifest_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(manifest),
        "explicit_run_manifest_ready": bool(manifest.get("explicit_run_manifest_ready", False)),
        "manual_wider_canary_explicit_run_allowed": bool(
            manifest.get("manual_wider_canary_explicit_run_allowed", False)
        ),
        "default_rollout_allowed": bool(manifest.get("default_rollout_allowed", False)),
        "auto_rollout_allowed": bool(manifest.get("auto_rollout_allowed", False)),
    }


def _audit_summary(audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "present": bool(audit),
        "run_audit_ready": bool(audit.get("run_audit_ready", False)),
        "decision": str(audit.get("decision") or ""),
        "missing_runtime_evidence": _string_list(audit.get("missing_runtime_evidence")),
        "rollback_required": bool(audit.get("rollback_required", False)),
    }


def _skipped(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "probe": "v5_checkpoint_resume_native_state_boundary_live_probe_v0",
        "ok": False,
        "skipped": True,
        "blocked_reasons": [str(reason)],
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, tuple):
        return [str(item) for item in value if str(item)]
    return []


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build V5 checkpoint/resume native-state boundary evidence.")
    parser.add_argument("--explicit-run-manifest", default="", help="Optional P21 manifest JSON.")
    parser.add_argument("--explicit-run-audit", default="", help="Optional P22 audit JSON.")
    parser.add_argument("--skip-live-probe", action="store_true", help="Emit a skipped live probe.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_checkpoint_resume_native_state_boundary_evidence(
        explicit_run_manifest=load_json(args.explicit_run_manifest) if args.explicit_run_manifest else None,
        explicit_run_audit=load_json(args.explicit_run_audit) if args.explicit_run_audit else None,
        run_live_probe=not bool(args.skip_live_probe),
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()


__all__ = ["build_v5_checkpoint_resume_native_state_boundary_evidence"]
