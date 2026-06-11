"""Artifact and result helpers for Lulynx LAB jobs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.core.contracts import ArtifactFile, ArtifactManifest, RunResult, RunStatus


def guess_lab_artifact_kind(path: str, schema_id: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".safetensors":
        return "model"
    if suffix in {".json", ".jsonl"} and "report" in schema_id:
        return "sample-report"
    if suffix in {".log", ".txt"}:
        return "log"
    return "lab-artifact"


def build_lab_artifacts(
    *,
    schema_id: str,
    run_id: str,
    config_path: str,
    log_path: str,
    metadata: dict[str, Any],
) -> list[ArtifactManifest]:
    artifacts: list[ArtifactManifest] = []
    common_metadata = {
        "job_id": run_id,
        "source": "lulynx-lab",
        **{key: value for key, value in metadata.items() if key not in {"stdout_tail", "result"}},
    }
    for role, path_value in (
        ("output", metadata.get("output_path")),
        ("samples", metadata.get("samples_dir")),
        ("config", config_path),
        ("log", log_path),
    ):
        path_text = str(path_value or "").strip()
        if not path_text:
            continue
        artifacts.append(
            ArtifactManifest(
                artifact_kind=guess_lab_artifact_kind(path_text, schema_id) if role == "output" else role,
                schema_id=schema_id,
                producer="lulynx-lab",
                request_id=run_id,
                files=[ArtifactFile(path=path_text, role=role)],
                metadata={**common_metadata, "role": role},
            )
        )
    return artifacts


def build_lab_run_result(
    *,
    task_id: str,
    schema_id: str,
    message: str,
    artifacts: list[ArtifactManifest],
    metadata: dict[str, Any],
) -> RunResult:
    return RunResult(
        run_id=task_id,
        request_id=task_id,
        status=RunStatus.QUEUED,
        message=message,
        artifacts=artifacts,
        data={
            "task_id": task_id,
            "schema_id": schema_id,
            "job_metadata": {key: value for key, value in metadata.items() if key != "stdout_tail"},
        },
    )
