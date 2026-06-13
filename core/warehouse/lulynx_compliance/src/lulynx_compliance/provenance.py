"""Provenance metadata fields for model exports.

Builds a dictionary of ``lulynx_*``-prefixed metadata keys that can be
embedded in safetensors headers, JSON sidecars, or export notice files
to record which project, version, and training route produced a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from lulynx_route_contract import classify_route, contract_to_metadata

from lulynx_compliance.signing import sign_payload


def build_provenance_fields(
    *,
    project_name: str,
    project_version: str,
    repo_url: str,
    license_name: str,
    git_commit: str,
    training_type: str | None = None,
    route_kind: str | None = None,
    route_label: str | None = None,
    model_hash: str | None = None,
    signing_secret: str | None = None,
) -> dict[str, str]:
    """Build a flat dict of provenance metadata fields.

    The returned dict uses ``lulynx_`` prefixed keys and is suitable for
    embedding into safetensors metadata or JSON export sidecars.

    Parameters
    ----------
    project_name, project_version, repo_url, license_name, git_commit:
        Core project identity fields.
    training_type, route_kind, route_label:
        Route classification inputs forwarded to the route contract.
    model_hash:
        Optional weight fingerprint (e.g. SHA-256 of the safetensors file).
    signing_secret:
        Optional HMAC secret for signing the provenance payload.
    """
    contract = classify_route(
        training_type or "",
        kind_override=route_kind,
        label_override=route_label,
    )
    fields: dict[str, str] = {
        "lulynx_project_name": project_name,
        "lulynx_project_version": project_version,
        "lulynx_project_repo": repo_url,
        "lulynx_project_license": license_name,
        "lulynx_project_commit": str(git_commit or "").strip() or "(unknown)",
        "lulynx_training_notice": (
            f"Trained/exported with {project_name}. "
            f"Modified or hosted builds should preserve notices and provide "
            f"corresponding source under {license_name}."
        ),
    }
    # Merge route contract metadata
    for key, value in contract_to_metadata(contract).items():
        fields[f"lulynx_{key}"] = value

    if model_hash:
        fields["lulynx_weight_fingerprint"] = str(model_hash)

    # Build a canonical signing payload
    canonical = json.dumps(
        {
            "project": project_name,
            "version": project_version,
            "commit": fields["lulynx_project_commit"],
            "route": contract.kind,
            "model_hash": str(model_hash or ""),
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    fields["lulynx_signature_payload"] = canonical

    if signing_secret:
        sig = sign_payload(canonical, signing_secret)
        fields["lulynx_signature_scheme"] = "hmac-sha256"
        fields["lulynx_signature"] = sig
    else:
        fields["lulynx_signature_scheme"] = "unsigned"

    return fields


def write_export_notice(
    output_path: str | Path,
    *,
    project_name: str,
    project_version: str,
    repo_url: str,
    license_name: str,
    git_commit: str,
    training_type: str | None = None,
    route_kind: str | None = None,
    route_label: str | None = None,
    model_hash: str | None = None,
    signing_secret: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> None:
    """Write a JSON export notice file alongside a trained model.

    The notice file contains all provenance fields plus a copy of
    the source metadata for auditability.
    """
    provenance = build_provenance_fields(
        project_name=project_name,
        project_version=project_version,
        repo_url=repo_url,
        license_name=license_name,
        git_commit=git_commit,
        training_type=training_type,
        route_kind=route_kind,
        route_label=route_label,
        model_hash=model_hash,
        signing_secret=signing_secret,
    )
    notice = {
        "project_name": project_name,
        "project_version": project_version,
        "provenance": provenance,
        "source_metadata": {
            str(k): str(v) for k, v in (extra_metadata or {}).items()
        },
    }
    Path(output_path).write_text(
        json.dumps(notice, indent=2, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
