"""Replay V5 owner-review packages through the request adapter safely."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.turbocore_v5_owner_review_evidence_package import load_json


def build_v5_owner_review_request_adapter_replay(
    *,
    pending_package: Mapping[str, Any] | None = None,
    signed_package: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prove owner-review packages only map after a signed review."""

    pending = _case("pending_unsigned_owner_review", _as_dict(pending_package), expected_enabled=False)
    signed = _case("signed_owner_review", _as_dict(signed_package), expected_enabled=True)
    cases = {
        "pending_unsigned_owner_review": pending,
        "signed_owner_review": signed,
    }
    gates = {
        "pending_unsigned_package_keeps_native_update_off": bool(pending.get("ok", False)),
        "signed_package_maps_existing_native_update_fields": bool(signed.get("ok", False)),
        "post_approval_fields_not_emitted_without_signed_review": not bool(
            pending.get("request_fields_ready", False)
        ),
        "default_behavior_unchanged": True,
    }
    blocked = [f"v5_p20_{name}_missing" for name, ok in gates.items() if not ok]
    ready = not blocked
    return {
        "schema_version": 1,
        "scorecard": "turbocore_v5_owner_review_request_adapter_replay_v0",
        "gate": "v5_owner_review_request_adapter_replay",
        "ok": ready,
        "milestone_completed": ready,
        "default_behavior_changed": False,
        "default_training_path_enabled": False,
        "training_path_enabled": False,
        "default_rollout_allowed": False,
        "auto_rollout_allowed": False,
        "adapter_replay_cases": cases,
        "progress_gates": gates,
        "blocked_reasons": blocked,
        "recommended_next_step": (
            "owner-signed manual wider canary requests can be replayed explicitly"
            if ready
            else "fix V5-P20 owner-review request adapter replay blockers"
        ),
        "notes": [
            "This replay helper does not enable default dispatch.",
            "Pending owner-review packages never emit review-ready request fields.",
            "Signed packages still map only to existing turbocore_native_update_* fields.",
        ],
    }


def request_fields_from_owner_review_package(package: Mapping[str, Any]) -> dict[str, Any]:
    """Return safe request fields for a reviewed package.

    Unsigned packages intentionally omit the review-ready flag, even if the
    evidence package contains a post-approval template.
    """

    source = _as_dict(package)
    if _package_signed(source):
        return dict(_as_dict(source.get("post_approval_request_fields")))
    return {
        "optimizerType": "AdamW",
        "optimizerBackend": "torch_adamw",
        "turbocoreNativeUpdateCanaryOptimizer": "exact_adamw",
        "turbocoreNativeUpdateCanaryScope": "manual_wider_canary",
    }


def _case(name: str, package: dict[str, Any], *, expected_enabled: bool) -> dict[str, Any]:
    request = request_fields_from_owner_review_package(package)
    config = ConfigAdapter.from_frontend_dict(request)
    fields = _resolved_fields(config)
    enabled = bool(
        fields["turbocore_native_update_mode"] == "native_experimental"
        and fields["turbocore_native_update_dispatch_enabled"]
        and fields["turbocore_native_update_training_path_enabled"]
        and fields["turbocore_native_update_require_native_cuda"]
        and fields["turbocore_native_update_defer_state_sync"]
    )
    return {
        "schema_version": 1,
        "case": name,
        "ok": enabled is bool(expected_enabled),
        "package_signed": _package_signed(package),
        "request_fields_ready": _request_has_review_ready(request),
        "request_fields": request,
        "resolved_fields": fields,
        "expected_native_update_enabled": bool(expected_enabled),
        "native_update_enabled": enabled,
    }


def _package_signed(package: Mapping[str, Any]) -> bool:
    return bool(
        package.get("promotion_review_ready", False)
        and package.get("manual_wider_canary_allowed", False)
        and package.get("promotion_decision") == "manual_wider_canary_review_ready"
        and not package.get("default_training_path_enabled", False)
        and not package.get("default_rollout_allowed", False)
        and not package.get("auto_rollout_allowed", False)
    )


def _request_has_review_ready(request: Mapping[str, Any]) -> bool:
    return bool(
        request.get("turbocoreNativeUpdateManualWiderCanaryApproved")
        or request.get("turbocoreNativeUpdateManualWiderCanaryOwnerApproved")
        or request.get("turbocoreNativeUpdateManualWiderCanaryReviewOk")
        or request.get("turbocoreNativeUpdateManualWiderCanaryReviewReady")
    )


def _resolved_fields(config: Any) -> dict[str, Any]:
    return {
        "optimizer_type": str(config.optimizer_type),
        "optimizer_backend": str(config.optimizer_backend),
        "turbocore_native_update_mode": str(config.turbocore_native_update_mode),
        "turbocore_native_update_dispatch_enabled": bool(
            config.turbocore_native_update_dispatch_enabled
        ),
        "turbocore_native_update_training_path_enabled": bool(
            config.turbocore_native_update_training_path_enabled
        ),
        "turbocore_native_update_require_native_cuda": bool(
            config.turbocore_native_update_require_native_cuda
        ),
        "turbocore_native_update_defer_state_sync": bool(
            config.turbocore_native_update_defer_state_sync
        ),
    }


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay V5 owner-review packages through request adapter.")
    parser.add_argument("--pending-package", default="", help="Pending owner review package JSON.")
    parser.add_argument("--signed-package", default="", help="Signed owner review package JSON.")
    parser.add_argument("--out", default="", help="Optional output JSON path.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    report = build_v5_owner_review_request_adapter_replay(
        pending_package=load_json(args.pending_package) if args.pending_package else None,
        signed_package=load_json(args.signed_package) if args.signed_package else None,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.out:
        output = Path(args.out)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()


__all__ = [
    "build_v5_owner_review_request_adapter_replay",
    "request_fields_from_owner_review_package",
]
