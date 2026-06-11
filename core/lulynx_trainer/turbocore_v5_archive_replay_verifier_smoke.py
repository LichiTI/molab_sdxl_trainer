"""Smoke checks for V5-P35 archive replay verifier."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
CORE_ROOT = BACKEND_ROOT / "core"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT), str(CORE_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_v5_archive_integrity_ledger import build_v5_archive_integrity_ledger  # noqa: E402
from core.turbocore_v5_archive_replay_verifier import (  # noqa: E402
    build_v5_archive_replay_verification,
    build_v5_archive_replay_verifier,
)
from core.turbocore_v5_owner_archive_signoff import build_v5_owner_archive_signoff  # noqa: E402
from lulynx_trainer.turbocore_v5_archive_integrity_ledger_smoke import _p33_signoff  # noqa: E402
from lulynx_trainer.turbocore_v5_owner_archive_signoff_smoke import (  # noqa: E402
    _archive_review,
    _p32_package,
)


DEFAULT_OFF_FIELDS = (
    "default_training_path_enabled",
    "training_path_enabled",
    "default_rollout_allowed",
    "auto_rollout_allowed",
    "request_adapter_mapping_allowed",
    "request_fields_emitted",
)


def run_smoke() -> dict[str, Any]:
    p32_ready = _p32_package()
    p33_ready = _p33_signoff(p32_ready)
    archived = _ledger(p33_ready, p32_ready)
    ready = build_v5_archive_replay_verifier(
        archived_archive_ledger={**archived, "_source_path": "archive/p34.json"},
        p33_archive_signoff={**p33_ready, "_source_path": "current/p33.json"},
        p32_owner_package={**p32_ready, "_source_path": "current/p32.json"},
    )
    assert ready["ok"] is True, ready
    assert ready["archive_replay_verification_ready"] is True, ready
    assert ready["archived_ledger_matches_replay"] is True, ready
    assert ready["decision"] == "p32_final_archive_replay_verification_ready_default_off", ready
    assert ready["digest_comparisons"][0]["match"] is True, ready
    assert ready["post_replay_verification_request_fields"] == {}, ready
    _assert_default_off(ready)
    alias_ready = build_v5_archive_replay_verification(
        archived_archive_ledger=archived,
        p33_archive_signoff=p33_ready,
        p32_owner_package=p32_ready,
    )
    assert alias_ready["decision"] == ready["decision"], alias_ready

    missing_archived = build_v5_archive_replay_verifier(
        p33_archive_signoff=p33_ready,
        p32_owner_package=p32_ready,
    )
    _assert_blocked(missing_archived, "archived", "missing")

    blocked_archived = build_v5_archive_replay_verifier(
        archived_archive_ledger=build_v5_archive_integrity_ledger(
            p33_archive_signoff=build_v5_owner_archive_signoff(p32_owner_package=p32_ready),
            p32_owner_package=p32_ready,
        ),
        p33_archive_signoff=p33_ready,
        p32_owner_package=p32_ready,
    )
    _assert_blocked(blocked_archived, "archived", "not_ready")

    tampered_ledger = build_v5_archive_replay_verifier(
        archived_archive_ledger={**archived, "ledger_digest": "0" * 64},
        p33_archive_signoff=p33_ready,
        p32_owner_package=p32_ready,
    )
    _assert_blocked(tampered_ledger, "ledger", "mismatch")

    tampered_p32 = copy.deepcopy(p32_ready)
    tampered_p32["p28_collector_bundle"]["ready_run_count"] = 99
    tampered_artifact = build_v5_archive_replay_verifier(
        archived_archive_ledger=archived,
        p33_archive_signoff=_p33_signoff(tampered_p32),
        p32_owner_package=tampered_p32,
    )
    _assert_blocked(tampered_artifact, "artifact", "mismatch")

    missing_artifact = copy.deepcopy(archived)
    missing_artifact["artifact_digests"] = missing_artifact["artifact_digests"][:-1]
    artifact_count = build_v5_archive_replay_verifier(
        archived_archive_ledger=missing_artifact,
        p33_archive_signoff=p33_ready,
        p32_owner_package=p32_ready,
    )
    _assert_blocked(artifact_count, "artifact", "count")

    pending_p33 = build_v5_archive_replay_verifier(
        archived_archive_ledger=archived,
        p33_archive_signoff=build_v5_owner_archive_signoff(p32_owner_package=p32_ready),
        p32_owner_package=p32_ready,
    )
    _assert_blocked(pending_p33, "replay", "not_ready")

    rejected_p33 = build_v5_archive_replay_verifier(
        archived_archive_ledger=archived,
        p33_archive_signoff=build_v5_owner_archive_signoff(
            p32_owner_package=p32_ready,
            owner_archive_review=_archive_review(approve=False),
        ),
        p32_owner_package=p32_ready,
    )
    _assert_blocked(rejected_p33, "replay", "not_ready")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p35_archive_replay_verifier_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_archived": _summary(missing_archived),
        "blocked_archived": _summary(blocked_archived),
        "tampered_ledger": _summary(tampered_ledger),
        "tampered_artifact": _summary(tampered_artifact),
        "artifact_count": _summary(artifact_count),
        "pending_p33": _summary(pending_p33),
        "rejected_p33": _summary(rejected_p33),
    }


def _ledger(p33_ready: dict[str, Any], p32_ready: dict[str, Any]) -> dict[str, Any]:
    return build_v5_archive_integrity_ledger(
        p33_archive_signoff=p33_ready,
        p32_owner_package=p32_ready,
    )


def _assert_default_off(report: dict[str, Any]) -> None:
    for field in DEFAULT_OFF_FIELDS:
        assert report[field] is False, report
    assert report["training_launch_allowed"] is False, report
    assert report["auto_launch_allowed"] is False, report
    assert report["runs_dispatched"] is False, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    assert report["archive_replay_verification_ready"] is False, report
    assert report["archived_ledger_matches_replay"] is False, report
    assert report["decision"] == "p32_final_archive_replay_verification_blocked_default_off", report
    _assert_default_off(report)
    reasons = [reason.lower() for reason in _blocked_reasons(report)]
    assert reasons, report
    for fragment in fragments:
        assert any(fragment.lower() in reason for reason in reasons), report


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok", False)),
        "decision": str(report.get("decision") or ""),
        "archive_replay_verification_ready": bool(report.get("archive_replay_verification_ready", False)),
        "archived_ledger_matches_replay": bool(report.get("archived_ledger_matches_replay", False)),
        "ledger_digest": str(report.get("archived_ledger_summary", {}).get("ledger_digest") or ""),
        "replay_digest": str(report.get("replay_ledger_summary", {}).get("ledger_digest") or ""),
        "comparison_count": len(report.get("digest_comparisons") or []),
        "blocked_reasons": _blocked_reasons(report),
    }


def _blocked_reasons(report: dict[str, Any]) -> list[str]:
    value = report.get("blocked_reasons") or report.get("promotion_blockers") or []
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value if str(item)]
    return []


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
