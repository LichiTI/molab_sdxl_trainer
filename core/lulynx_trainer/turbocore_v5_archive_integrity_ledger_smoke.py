"""Smoke checks for V5-P34 archive integrity ledger."""

from __future__ import annotations

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
from core.turbocore_v5_owner_archive_signoff import build_v5_owner_archive_signoff  # noqa: E402
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
    ready = build_v5_archive_integrity_ledger(
        p33_archive_signoff=p33_ready,
        p32_owner_package=p32_ready,
    )
    assert ready["ok"] is True, ready
    assert ready["ledger_ready"] is True, ready
    assert ready["decision"] == "p32_final_archive_integrity_ledger_ready_default_off", ready
    assert ready["ledger_digest"], ready
    assert len(ready["artifact_digests"]) == 7, ready
    assert all(item["sha256"] for item in ready["artifact_digests"]), ready
    _assert_default_off(ready)

    ready_again = build_v5_archive_integrity_ledger(
        p33_archive_signoff={**p33_ready, "_source_path": "other/p33.json"},
        p32_owner_package={**p32_ready, "_source_path": "other/p32.json"},
    )
    assert ready_again["ledger_digest"] == ready["ledger_digest"], (ready, ready_again)

    missing_p32 = build_v5_archive_integrity_ledger(p33_archive_signoff=p33_ready)
    _assert_blocked(missing_p32, "p32", "missing")

    pending_p33 = build_v5_archive_integrity_ledger(
        p33_archive_signoff=build_v5_owner_archive_signoff(p32_owner_package=p32_ready),
        p32_owner_package=p32_ready,
    )
    _assert_blocked(pending_p33, "p33")

    rejected_p33 = build_v5_archive_integrity_ledger(
        p33_archive_signoff=build_v5_owner_archive_signoff(
            p32_owner_package=p32_ready,
            owner_archive_review=_archive_review(approve=False),
        ),
        p32_owner_package=p32_ready,
    )
    _assert_blocked(rejected_p33, "p33")

    p32_blocked = build_v5_archive_integrity_ledger(
        p33_archive_signoff=p33_ready,
        p32_owner_package={**p32_ready, "default_rollout_allowed": True},
    )
    _assert_blocked(p32_blocked, "p32")

    post_fields = build_v5_archive_integrity_ledger(
        p33_archive_signoff={**p33_ready, "post_archive_request_fields": {"x": 1}},
        p32_owner_package=p32_ready,
    )
    _assert_blocked(post_fields, "post")

    return {
        "schema_version": 1,
        "probe": "turbocore_v5_p34_archive_integrity_ledger_smoke",
        "ok": True,
        "ready": _summary(ready),
        "missing_p32": _summary(missing_p32),
        "pending_p33": _summary(pending_p33),
        "rejected_p33": _summary(rejected_p33),
        "p32_blocked": _summary(p32_blocked),
        "post_fields": _summary(post_fields),
    }


def _p33_signoff(p32_ready: dict[str, Any]) -> dict[str, Any]:
    return build_v5_owner_archive_signoff(
        p32_owner_package=p32_ready,
        owner_archive_review=_archive_review(approve=True),
    )


def _assert_default_off(report: dict[str, Any]) -> None:
    for field in DEFAULT_OFF_FIELDS:
        assert report[field] is False, report


def _assert_blocked(report: dict[str, Any], *fragments: str) -> None:
    assert report["ok"] is False, report
    _assert_default_off(report)
    reasons = [reason.lower() for reason in _blocked_reasons(report)]
    assert reasons, report
    for fragment in fragments:
        assert any(fragment.lower() in reason for reason in reasons), report


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(report.get("ok", False)),
        "decision": str(report.get("decision") or ""),
        "ledger_ready": bool(report.get("ledger_ready", False)),
        "ledger_digest": str(report.get("ledger_digest") or ""),
        "artifact_count": len(report.get("artifact_digests") or []),
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
