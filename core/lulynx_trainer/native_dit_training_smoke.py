"""Shared native DiT training readiness smoke for Anima and Newbie.

This script is the DiT counterpart to the SDXL native U-Net probes.  It keeps
family-specific loaders intact, but verifies the shared contracts that matter
before promoting Anima/Newbie onto the same native-training track:

- keymap coverage is complete
- state mapping plans have no duplicate or mismatched targets
- cache-first trainer paths can complete one adapter step and save an artifact
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.native_unet.keymap_inspector import (
    build_many_state_mapping_plans,
    inspect_many,
)


FAMILIES = ("anima", "newbie")


@dataclass(frozen=True)
class DiTTrainerSmokeResult:
    family: str
    ok: bool
    command: list[str]
    returncode: int
    elapsed_seconds: float
    stdout_tail: list[str]
    stderr_tail: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _keymap_dir() -> Path:
    return Path(__file__).resolve().parent / "native_unet" / "keymaps"


def _manifest_for_family(family: str) -> Path:
    if family == "anima":
        return _keymap_dir() / "anima_dit_keymap_manifest.json"
    if family == "newbie":
        return _keymap_dir() / "newbie_dit_keymap_manifest.json"
    raise ValueError(f"Unsupported DiT family: {family}")


def _trainer_smoke_script(family: str) -> Path:
    if family == "anima":
        return Path(__file__).with_name("anima_trainer_cached_smoke.py")
    if family == "newbie":
        return Path(__file__).with_name("newbie_trainer_cached_smoke.py")
    raise ValueError(f"Unsupported DiT family: {family}")


def _tail(text: str, limit: int = 30) -> list[str]:
    return [line for line in text.splitlines() if line.strip()][-limit:]


def _run_trainer_smoke(family: str) -> DiTTrainerSmokeResult:
    script = _trainer_smoke_script(family)
    cmd = [sys.executable, str(script)]
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(_repo_root()),
        capture_output=True,
        text=True,
    )
    return DiTTrainerSmokeResult(
        family=family,
        ok=proc.returncode == 0,
        command=cmd,
        returncode=int(proc.returncode),
        elapsed_seconds=round(time.perf_counter() - started, 3),
        stdout_tail=_tail(proc.stdout),
        stderr_tail=_tail(proc.stderr),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify shared native DiT training readiness.")
    parser.add_argument("--families", nargs="*", default=list(FAMILIES), choices=list(FAMILIES))
    parser.add_argument("--skip-trainer", action="store_true", help="Only inspect keymaps and state mapping plans.")
    parser.add_argument("--json", default="", help="Optional report path.")
    args = parser.parse_args()

    families = tuple(dict.fromkeys(str(item) for item in args.families))
    manifests = [_manifest_for_family(family) for family in families]
    inspections = inspect_many(manifests)
    state_plans = build_many_state_mapping_plans(manifests)
    trainer_results = [] if args.skip_trainer else [_run_trainer_smoke(family) for family in families]

    report = {
        "ok": (
            all(item.ok for item in inspections)
            and all(item.ok for item in state_plans)
            and all(item.ok for item in trainer_results)
        ),
        "families": list(families),
        "keymaps": [
            {
                "family": inspection.family,
                "component": inspection.component,
                "ok": inspection.ok,
                "matched_keys": inspection.matched_keys,
                "unmatched_keys": inspection.unmatched_keys,
                "duplicate_targets": inspection.duplicate_targets,
                "dtype_counts": inspection.dtype_counts,
                "rank_counts": inspection.rank_counts,
            }
            for inspection in inspections
        ],
        "state_mapping_plans": [
            {
                "family": plan.family,
                "component": plan.component,
                "ok": plan.ok,
                "tensors": plan.tensors,
                "total_parameters": plan.total_parameters,
                "duplicate_targets": plan.duplicate_targets,
                "shape_mismatches": plan.shape_mismatches,
                "missing_targets": plan.missing_targets,
                "unexpected_targets": plan.unexpected_targets,
            }
            for plan in state_plans
        ],
        "trainer_smokes": [item.to_dict() for item in trainer_results],
    }

    if args.json:
        path = Path(args.json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
