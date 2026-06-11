"""Contract/status payload helpers for Lulynx LAB routes."""

from __future__ import annotations

from pathlib import Path


LAB_RUNNER_FILES: tuple[tuple[str, str], ...] = (
    ("lab-distiller", "run_distiller.py"),
    ("sdxl-turbo-lora", "run_turbo_lora.py"),
    ("dit-few-step-lora", "run_dit_few_step_lora.py"),
    ("turbo-lora-validator", "validate_turbo_lora_output.py"),
    ("turbo-lora-sample-report", "report_turbo_lora_samples.py"),
)


def lab_runner_path(backend_root: Path, filename: str) -> Path:
    """Return the canonical path for a LAB runner script."""

    return backend_root / "core" / "tools" / "lulynx_lab" / filename


def build_lab_contract_payload(backend_root: Path) -> dict[str, dict[str, object]]:
    """Build the legacy /api/lulynx-lab/contract payload."""

    return {
        key: {"available": path.is_file(), "path": str(path)}
        for key, filename in LAB_RUNNER_FILES
        for path in [lab_runner_path(backend_root, filename)]
    }


__all__ = ["LAB_RUNNER_FILES", "build_lab_contract_payload", "lab_runner_path"]
