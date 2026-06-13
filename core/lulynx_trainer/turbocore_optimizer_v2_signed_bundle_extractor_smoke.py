"""Smoke checks for v2 signed-bundle extraction."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import sys
import tempfile
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

from core.turbocore_optimizer_v2_reviewer_handoff_packet import (  # noqa: E402
    build_optimizer_v2_reviewer_handoff_packet,
)
from core.turbocore_optimizer_v2_signed_bundle_extractor import (  # noqa: E402
    ARTIFACT,
    build_optimizer_v2_signed_bundle_extraction_record,
    main as extractor_cli_main,
)
from core.turbocore_optimizer_v2_signature_bundle_packet import (  # noqa: E402
    build_optimizer_v2_signature_bundle_packet,
)


def run_smoke() -> dict[str, Any]:
    pending = build_optimizer_v2_signed_bundle_extraction_record(
        signed_bundle={},
        write_artifact=True,
        write_extracted_artifacts=False,
    )
    pending_summary = pending["summary"]
    assert pending["ok"] is True, pending
    assert pending["signed_bundle_present"] is False, pending
    assert pending["missing_signed_signature_ids"] == [
        "owner_release_review",
        "product_exposure_review",
        "owner_release_direction",
    ], pending
    assert pending["extraction_ready"] is False, pending
    assert pending["approval_recorded"] is False, pending
    assert pending["approval_artifact_written"] is False, pending
    assert pending["runtime_dispatch_ready"] is False, pending
    assert pending["native_dispatch_allowed"] is False, pending
    assert pending["training_path_enabled"] is False, pending
    assert pending["product_native_ready"] is False, pending
    assert pending_summary["v2_signed_bundle_extraction_entry_count"] == 3, pending
    assert pending_summary["v2_signed_bundle_extraction_present_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_extraction_signed_entry_digest_present_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_extraction_extractable_signed_entry_digest_present_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_extraction_extractable_entry_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_extraction_phase1_extractable_entry_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_extraction_phase1_ready_for_record_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_extraction_phase2_extractable_entry_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_extraction_phase2_ready_for_record_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_extraction_full_ready_for_record_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_extraction_missing_entry_count"] == 3, pending
    assert pending_summary["v2_signed_bundle_extraction_missing_signature_count"] == 3, pending
    assert pending_summary["v2_signed_bundle_extraction_phase1_missing_signature_count"] == 2, pending
    assert pending_summary["v2_signed_bundle_extraction_phase2_missing_signature_count"] == 1, pending
    assert pending_summary["v2_signed_bundle_extraction_artifact_written_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_extraction_ready_for_record_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_extraction_runtime_dispatch_ready_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_extraction_native_dispatch_allowed_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_extraction_training_path_enabled_count"] == 0, pending
    assert pending_summary["v2_signed_bundle_extraction_product_native_ready_count"] == 0, pending
    assert ARTIFACT.exists(), ARTIFACT

    signature_bundle = build_optimizer_v2_signature_bundle_packet(write_artifact=False)
    signature_bundle_digest = _digest_payload(signature_bundle)
    handoff = build_optimizer_v2_reviewer_handoff_packet(
        signature_bundle=signature_bundle,
        write_artifact=False,
        write_signed_bundle_template=False,
    )
    signed_bundle = {
        "schema_version": 1,
        "package": "turbocore_optimizer_v2_signed_bundle_smoke_v0",
        "source_signature_bundle_digest": signature_bundle_digest,
        "signed_entries": {
            signature_id: _sign_template(signature_id, template)
            for signature_id, template in handoff["signed_bundle_template"]["signed_entries"].items()
        },
    }
    stale_signed_bundle = dict(signed_bundle)
    stale_signed_bundle["source_signature_bundle_digest"] = "stale-digest-for-extractor-smoke"
    unsigned_template_signed_bundle = dict(signed_bundle)
    unsigned_template_signed_bundle["unsigned_template"] = True
    template_digest_mismatch_bundle = json.loads(json.dumps(signed_bundle))
    template_digest_mismatch_bundle["signed_entries"]["product_exposure_review"][
        "source_v2_signature_template_digest"
    ] = "mismatched-product-exposure-template-digest"
    unknown_entry_bundle = json.loads(json.dumps(signed_bundle))
    unknown_entry_bundle["signed_entries"]["unexpected_review"] = dict(
        unknown_entry_bundle["signed_entries"]["owner_release_review"]
    )
    premature_direction_bundle = json.loads(json.dumps(signed_bundle))
    signature_entries = {entry["signature_id"]: entry for entry in signature_bundle["signature_entries"]}
    premature_direction_bundle["signed_entries"]["owner_release_direction"] = _sign_template(
        "owner_release_direction",
        signature_entries["owner_release_direction"]["template"],
    )
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir)
        extracted = build_optimizer_v2_signed_bundle_extraction_record(
            signature_bundle=signature_bundle,
            signed_bundle=signed_bundle,
            output_dir=output_dir,
            write_artifact=False,
            write_extracted_artifacts=True,
        )
        stale_output_dir = output_dir / "stale"
        stale_extracted = build_optimizer_v2_signed_bundle_extraction_record(
            signature_bundle=signature_bundle,
            signed_bundle=stale_signed_bundle,
            output_dir=stale_output_dir,
            write_artifact=False,
            write_extracted_artifacts=True,
        )
        unsigned_template_output_dir = output_dir / "unsigned-template"
        unsigned_template_extracted = build_optimizer_v2_signed_bundle_extraction_record(
            signature_bundle=signature_bundle,
            signed_bundle=unsigned_template_signed_bundle,
            output_dir=unsigned_template_output_dir,
            write_artifact=False,
            write_extracted_artifacts=True,
        )
        template_digest_mismatch_output_dir = output_dir / "template-mismatch"
        template_digest_mismatch_extracted = build_optimizer_v2_signed_bundle_extraction_record(
            signature_bundle=signature_bundle,
            signed_bundle=template_digest_mismatch_bundle,
            output_dir=template_digest_mismatch_output_dir,
            write_artifact=False,
            write_extracted_artifacts=True,
        )
        unknown_entry_output_dir = output_dir / "unknown-entry"
        unknown_entry_extracted = build_optimizer_v2_signed_bundle_extraction_record(
            signature_bundle=signature_bundle,
            signed_bundle=unknown_entry_bundle,
            output_dir=unknown_entry_output_dir,
            write_artifact=False,
            write_extracted_artifacts=True,
        )
        premature_direction_output_dir = output_dir / "not-ready-direction"
        premature_direction_extracted = build_optimizer_v2_signed_bundle_extraction_record(
            signature_bundle=signature_bundle,
            signed_bundle=premature_direction_bundle,
            output_dir=premature_direction_output_dir,
            write_artifact=False,
            write_extracted_artifacts=True,
        )
        signature_bundle_path = output_dir / "signature_bundle.json"
        signed_bundle_path = output_dir / "signed_bundle.json"
        stale_signed_bundle_path = output_dir / "signed_bundle.stale.json"
        unsigned_template_signed_bundle_path = output_dir / "signed_bundle.unsigned_template.json"
        template_digest_mismatch_bundle_path = output_dir / "signed_bundle.template_mismatch.json"
        unknown_entry_bundle_path = output_dir / "signed_bundle.unknown_entry.json"
        premature_direction_bundle_path = output_dir / "signed_bundle.not_ready_direction.json"
        _write_json(signature_bundle_path, signature_bundle)
        _write_json(signed_bundle_path, signed_bundle)
        _write_json(stale_signed_bundle_path, stale_signed_bundle)
        _write_json(unsigned_template_signed_bundle_path, unsigned_template_signed_bundle)
        _write_json(template_digest_mismatch_bundle_path, template_digest_mismatch_bundle)
        _write_json(unknown_entry_bundle_path, unknown_entry_bundle)
        _write_json(premature_direction_bundle_path, premature_direction_bundle)
        cli_payload = _run_extractor_cli(
            [
                "--signature-bundle",
                str(signature_bundle_path),
                "--signed-bundle",
                str(signed_bundle_path),
                "--output-dir",
                str(output_dir / "cli"),
                "--write-extracted-artifacts",
                "--no-artifact",
            ]
        )
        stale_cli_payload, stale_cli_exit = _run_extractor_cli_allow_failure(
            [
                "--signature-bundle",
                str(signature_bundle_path),
                "--signed-bundle",
                str(stale_signed_bundle_path),
                "--output-dir",
                str(output_dir / "stale-cli"),
                "--write-extracted-artifacts",
                "--no-artifact",
            ]
        )
        unsigned_template_cli_payload, unsigned_template_cli_exit = _run_extractor_cli_allow_failure(
            [
                "--signature-bundle",
                str(signature_bundle_path),
                "--signed-bundle",
                str(unsigned_template_signed_bundle_path),
                "--output-dir",
                str(output_dir / "unsigned-template-cli"),
                "--write-extracted-artifacts",
                "--no-artifact",
            ]
        )
        template_digest_mismatch_cli_payload, template_digest_mismatch_cli_exit = _run_extractor_cli_allow_failure(
            [
                "--signature-bundle",
                str(signature_bundle_path),
                "--signed-bundle",
                str(template_digest_mismatch_bundle_path),
                "--output-dir",
                str(output_dir / "template-mismatch-cli"),
                "--write-extracted-artifacts",
                "--no-artifact",
            ]
        )
        unknown_entry_cli_payload, unknown_entry_cli_exit = _run_extractor_cli_allow_failure(
            [
                "--signature-bundle",
                str(signature_bundle_path),
                "--signed-bundle",
                str(unknown_entry_bundle_path),
                "--output-dir",
                str(output_dir / "unknown-entry-cli"),
                "--write-extracted-artifacts",
                "--no-artifact",
            ]
        )
        premature_direction_cli_payload, premature_direction_cli_exit = _run_extractor_cli_allow_failure(
            [
                "--signature-bundle",
                str(signature_bundle_path),
                "--signed-bundle",
                str(premature_direction_bundle_path),
                "--output-dir",
                str(output_dir / "not-ready-direction-cli"),
                "--write-extracted-artifacts",
                "--no-artifact",
            ]
        )
        assert (output_dir / "signed_owner_release_review.json").exists(), extracted
        assert (output_dir / "signed_product_exposure_review.json").exists(), extracted
        assert not (output_dir / "signed_owner_release_direction.json").exists(), extracted
        assert not (stale_output_dir / "signed_owner_release_review.json").exists(), stale_extracted
        assert not (unsigned_template_output_dir / "signed_owner_release_review.json").exists(), unsigned_template_extracted
        assert not (template_digest_mismatch_output_dir / "signed_product_exposure_review.json").exists(), template_digest_mismatch_extracted
        assert not (unknown_entry_output_dir / "signed_owner_release_review.json").exists(), unknown_entry_extracted
        assert not (premature_direction_output_dir / "signed_owner_release_direction.json").exists(), premature_direction_extracted
    extracted_summary = extracted["summary"]
    assert extracted["ok"] is True, extracted
    assert extracted["signed_bundle_present"] is True, extracted
    assert extracted["missing_signed_signature_ids"] == ["owner_release_direction"], extracted
    assert extracted["signed_bundle_source_digest_match"] is True, extracted
    assert extracted["signed_bundle_source_digest_stale"] is False, extracted
    assert extracted["extraction_ready"] is False, extracted
    assert extracted["approval_recorded"] is False, extracted
    assert extracted["approval_artifact_written"] is False, extracted
    assert extracted_summary["v2_signed_bundle_extraction_extractable_entry_count"] == 2, extracted
    assert extracted_summary["v2_signed_bundle_extraction_phase1_extractable_entry_count"] == 2, extracted
    assert extracted_summary["v2_signed_bundle_extraction_phase1_ready_for_record_count"] == 1, extracted
    assert extracted_summary["v2_signed_bundle_extraction_phase2_extractable_entry_count"] == 0, extracted
    assert extracted_summary["v2_signed_bundle_extraction_phase2_ready_for_record_count"] == 0, extracted
    assert extracted_summary["v2_signed_bundle_extraction_full_ready_for_record_count"] == 0, extracted
    assert extracted_summary["v2_signed_bundle_extraction_source_digest_match_count"] == 1, extracted
    assert extracted_summary["v2_signed_bundle_extraction_source_digest_stale_count"] == 0, extracted
    assert extracted_summary["v2_signed_bundle_extraction_unsigned_template_count"] == 0, extracted
    assert extracted_summary["v2_signed_bundle_extraction_template_digest_mismatch_count"] == 0, extracted
    assert extracted_summary["v2_signed_bundle_extraction_missing_signature_count"] == 1, extracted
    assert extracted_summary["v2_signed_bundle_extraction_phase1_missing_signature_count"] == 0, extracted
    assert extracted_summary["v2_signed_bundle_extraction_phase2_missing_signature_count"] == 1, extracted
    assert extracted_summary["v2_signed_bundle_extraction_signed_entry_digest_present_count"] == 2, extracted
    assert extracted_summary["v2_signed_bundle_extraction_extractable_signed_entry_digest_present_count"] == 2, extracted
    assert all(
        check["signed_entry_digest"]
        for check in extracted["entry_checks"]
        if check["signature_id"] in ("owner_release_review", "product_exposure_review")
    ), extracted
    assert extracted_summary["v2_signed_bundle_extraction_missing_entry_count"] == 1, extracted
    assert extracted_summary["v2_signed_bundle_extraction_owner_review_extracted_count"] == 1, extracted
    assert extracted_summary["v2_signed_bundle_extraction_product_exposure_extracted_count"] == 1, extracted
    assert extracted_summary["v2_signed_bundle_extraction_owner_direction_extracted_count"] == 0, extracted
    assert extracted_summary["v2_signed_bundle_extraction_artifact_written_count"] == 2, extracted
    assert extracted_summary["v2_signed_bundle_extraction_runtime_dispatch_ready_count"] == 0, extracted
    assert extracted_summary["v2_signed_bundle_extraction_native_dispatch_allowed_count"] == 0, extracted
    assert extracted_summary["v2_signed_bundle_extraction_training_path_enabled_count"] == 0, extracted
    assert extracted_summary["v2_signed_bundle_extraction_product_native_ready_count"] == 0, extracted
    stale_summary = stale_extracted["summary"]
    assert stale_extracted["ok"] is False, stale_extracted
    assert stale_extracted["signed_bundle_source_digest_match"] is False, stale_extracted
    assert stale_extracted["signed_bundle_source_digest_stale"] is True, stale_extracted
    assert stale_summary["v2_signed_bundle_extraction_source_digest_match_count"] == 0, stale_extracted
    assert stale_summary["v2_signed_bundle_extraction_source_digest_stale_count"] == 1, stale_extracted
    assert stale_summary["v2_signed_bundle_extraction_extractable_entry_count"] == 0, stale_extracted
    assert stale_summary["v2_signed_bundle_extraction_phase1_ready_for_record_count"] == 0, stale_extracted
    assert stale_summary["v2_signed_bundle_extraction_phase2_ready_for_record_count"] == 0, stale_extracted
    assert stale_summary["v2_signed_bundle_extraction_full_ready_for_record_count"] == 0, stale_extracted
    assert stale_summary["v2_signed_bundle_extraction_artifact_written_count"] == 0, stale_extracted
    unsigned_template_summary = unsigned_template_extracted["summary"]
    assert unsigned_template_extracted["ok"] is False, unsigned_template_extracted
    assert unsigned_template_extracted["signed_bundle_source_digest_match"] is True, unsigned_template_extracted
    assert unsigned_template_extracted["signed_bundle_unsigned_template_marker"] is True, unsigned_template_extracted
    assert unsigned_template_summary["v2_signed_bundle_extraction_unsigned_template_count"] == 1, unsigned_template_extracted
    assert unsigned_template_summary["v2_signed_bundle_extraction_extractable_entry_count"] == 0, unsigned_template_extracted
    assert unsigned_template_summary["v2_signed_bundle_extraction_phase1_ready_for_record_count"] == 0, unsigned_template_extracted
    assert unsigned_template_summary["v2_signed_bundle_extraction_phase2_ready_for_record_count"] == 0, unsigned_template_extracted
    assert unsigned_template_summary["v2_signed_bundle_extraction_full_ready_for_record_count"] == 0, unsigned_template_extracted
    assert unsigned_template_summary["v2_signed_bundle_extraction_artifact_written_count"] == 0, unsigned_template_extracted
    template_mismatch_summary = template_digest_mismatch_extracted["summary"]
    assert template_digest_mismatch_extracted["ok"] is False, template_digest_mismatch_extracted
    assert template_digest_mismatch_extracted["signed_bundle_source_digest_match"] is True, template_digest_mismatch_extracted
    assert template_digest_mismatch_extracted["extraction_ready"] is False, template_digest_mismatch_extracted
    assert template_mismatch_summary["v2_signed_bundle_extraction_template_digest_mismatch_count"] == 1, template_digest_mismatch_extracted
    assert template_mismatch_summary["v2_signed_bundle_extraction_extractable_entry_count"] == 0, template_digest_mismatch_extracted
    assert template_mismatch_summary["v2_signed_bundle_extraction_phase1_ready_for_record_count"] == 0, template_digest_mismatch_extracted
    assert template_mismatch_summary["v2_signed_bundle_extraction_phase2_ready_for_record_count"] == 0, template_digest_mismatch_extracted
    assert template_mismatch_summary["v2_signed_bundle_extraction_full_ready_for_record_count"] == 0, template_digest_mismatch_extracted
    assert template_mismatch_summary["v2_signed_bundle_extraction_artifact_written_count"] == 0, template_digest_mismatch_extracted
    unknown_entry_summary = unknown_entry_extracted["summary"]
    assert unknown_entry_extracted["ok"] is False, unknown_entry_extracted
    assert unknown_entry_extracted["signed_bundle_source_digest_match"] is True, unknown_entry_extracted
    assert unknown_entry_extracted["extraction_ready"] is False, unknown_entry_extracted
    assert unknown_entry_extracted["unknown_signed_signature_ids"] == ["unexpected_review"], unknown_entry_extracted
    assert unknown_entry_extracted["missing_signed_signature_ids"] == ["owner_release_direction"], unknown_entry_extracted
    assert "unknown_signed_entry:unexpected_review" in unknown_entry_extracted["blocked_reasons"], unknown_entry_extracted
    assert unknown_entry_summary["v2_signed_bundle_extraction_unknown_entry_count"] == 1, unknown_entry_extracted
    assert unknown_entry_summary["v2_signed_bundle_extraction_missing_signature_count"] == 1, unknown_entry_extracted
    assert unknown_entry_summary["v2_signed_bundle_extraction_phase1_missing_signature_count"] == 0, unknown_entry_extracted
    assert unknown_entry_summary["v2_signed_bundle_extraction_phase2_missing_signature_count"] == 1, unknown_entry_extracted
    assert unknown_entry_summary["v2_signed_bundle_extraction_extractable_entry_count"] == 0, unknown_entry_extracted
    assert unknown_entry_summary["v2_signed_bundle_extraction_phase1_ready_for_record_count"] == 0, unknown_entry_extracted
    assert unknown_entry_summary["v2_signed_bundle_extraction_phase2_ready_for_record_count"] == 0, unknown_entry_extracted
    assert unknown_entry_summary["v2_signed_bundle_extraction_full_ready_for_record_count"] == 0, unknown_entry_extracted
    assert unknown_entry_summary["v2_signed_bundle_extraction_artifact_written_count"] == 0, unknown_entry_extracted
    premature_direction_summary = premature_direction_extracted["summary"]
    assert premature_direction_extracted["ok"] is False, premature_direction_extracted
    assert premature_direction_extracted["signed_bundle_source_digest_match"] is True, premature_direction_extracted
    assert premature_direction_extracted["extraction_ready"] is False, premature_direction_extracted
    assert (
        "signed_entry_not_ready_for_signature:owner_release_direction"
        in premature_direction_extracted["blocked_reasons"]
    ), premature_direction_extracted
    assert (
        premature_direction_summary["v2_signed_bundle_extraction_not_ready_entry_count"] == 1
    ), premature_direction_extracted
    assert premature_direction_summary["v2_signed_bundle_extraction_extractable_entry_count"] == 0, premature_direction_extracted
    assert premature_direction_summary["v2_signed_bundle_extraction_phase1_ready_for_record_count"] == 0, premature_direction_extracted
    assert premature_direction_summary["v2_signed_bundle_extraction_phase2_ready_for_record_count"] == 0, premature_direction_extracted
    assert premature_direction_summary["v2_signed_bundle_extraction_full_ready_for_record_count"] == 0, premature_direction_extracted
    assert premature_direction_summary["v2_signed_bundle_extraction_artifact_written_count"] == 0, premature_direction_extracted
    assert cli_payload["ok"] is True, cli_payload
    assert cli_payload["signed_bundle_source_digest_match"] is True, cli_payload
    assert cli_payload["summary"]["v2_signed_bundle_extraction_artifact_written_count"] == 2, cli_payload
    assert cli_payload["summary"]["v2_signed_bundle_extraction_missing_signature_count"] == 1, cli_payload
    assert cli_payload["summary"]["v2_signed_bundle_extraction_phase1_missing_signature_count"] == 0, cli_payload
    assert cli_payload["summary"]["v2_signed_bundle_extraction_phase2_missing_signature_count"] == 1, cli_payload
    assert cli_payload["summary"]["v2_signed_bundle_extraction_signed_entry_digest_present_count"] == 2, cli_payload
    assert cli_payload["summary"]["v2_signed_bundle_extraction_extractable_signed_entry_digest_present_count"] == 2, cli_payload
    assert cli_payload["summary"]["v2_signed_bundle_extraction_approval_recorded_count"] == 0, cli_payload
    assert stale_cli_exit == 1, stale_cli_payload
    assert stale_cli_payload["ok"] is False, stale_cli_payload
    assert stale_cli_payload["summary"]["v2_signed_bundle_extraction_source_digest_stale_count"] == 1, stale_cli_payload
    assert unsigned_template_cli_exit == 1, unsigned_template_cli_payload
    assert unsigned_template_cli_payload["ok"] is False, unsigned_template_cli_payload
    assert unsigned_template_cli_payload["summary"]["v2_signed_bundle_extraction_unsigned_template_count"] == 1, unsigned_template_cli_payload
    assert template_digest_mismatch_cli_exit == 1, template_digest_mismatch_cli_payload
    assert template_digest_mismatch_cli_payload["ok"] is False, template_digest_mismatch_cli_payload
    assert template_digest_mismatch_cli_payload["extraction_ready"] is False, template_digest_mismatch_cli_payload
    assert (
        template_digest_mismatch_cli_payload["summary"]["v2_signed_bundle_extraction_template_digest_mismatch_count"]
        == 1
    ), template_digest_mismatch_cli_payload
    assert unknown_entry_cli_exit == 1, unknown_entry_cli_payload
    assert unknown_entry_cli_payload["ok"] is False, unknown_entry_cli_payload
    assert unknown_entry_cli_payload["extraction_ready"] is False, unknown_entry_cli_payload
    assert unknown_entry_cli_payload["summary"]["v2_signed_bundle_extraction_unknown_entry_count"] == 1, unknown_entry_cli_payload
    assert premature_direction_cli_exit == 1, premature_direction_cli_payload
    assert premature_direction_cli_payload["ok"] is False, premature_direction_cli_payload
    assert premature_direction_cli_payload["extraction_ready"] is False, premature_direction_cli_payload
    assert (
        premature_direction_cli_payload["summary"]["v2_signed_bundle_extraction_not_ready_entry_count"] == 1
    ), premature_direction_cli_payload

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_v2_signed_bundle_extractor_smoke",
        "ok": True,
        "real_artifact_checked": True,
        "synthetic_signed_bundle_extracted_to_temp": True,
        "stale_signed_bundle_rejected": True,
        "unsigned_template_signed_bundle_rejected": True,
        "template_digest_mismatch_rejected": True,
        "unknown_signed_entry_rejected": True,
        "not_ready_signed_entry_rejected": True,
        "approval_artifact_written": False,
        "summary": pending_summary,
        "recommended_next_step": pending["recommended_next_step"],
    }


def _sign_template(signature_id: str, template: dict[str, Any]) -> dict[str, Any]:
    signed = dict(template)
    signed["reviewer"] = f"synthetic_{signature_id}_smoke"
    signed["reviewed_at"] = "2026-06-09"
    for key in list(signed):
        if key.startswith("approve_") or key.startswith("acknowledge_"):
            signed[key] = True
    return signed


def _digest_payload(value: dict[str, Any]) -> str:
    data = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_extractor_cli(args: list[str]) -> dict[str, Any]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exit_code = int(extractor_cli_main(args))
    assert exit_code == 0, stdout.getvalue()
    return json.loads(stdout.getvalue())


def _run_extractor_cli_allow_failure(args: list[str]) -> tuple[dict[str, Any], int]:
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exit_code = int(extractor_cli_main(args))
    return json.loads(stdout.getvalue()), exit_code


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
