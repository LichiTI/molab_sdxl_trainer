from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = Path(__file__).resolve().parents[2]
KEYMAP_DIR = Path(__file__).resolve().parent / "native_unet" / "keymaps"
sys.path.insert(0, str(BACKEND_ROOT))

from core.lulynx_trainer.native_unet.keymap_inspector import build_many_state_mapping_plans, inspect_many
from core.lulynx_trainer.native_unet.state_loader import iter_mapped_safetensors


def _loader_sample(manifest):
    item = next(iter_mapped_safetensors(manifest, limit=1))
    return {
        "source_key": item.source_key,
        "target_key": item.target_key,
        "shape": list(item.tensor.shape),
        "dtype": str(item.tensor.dtype),
    }


def _summary_payload(reports, state_plans):
    return [
        {
            "family": report.family,
            "component": report.component,
            "inspection_ok": report.ok,
            "state_plan_ok": plan.ok,
            "matched_keys": report.matched_keys,
            "unmatched_keys": report.unmatched_keys,
            "duplicate_targets": plan.duplicate_targets,
            "dry_run": plan.dry_run,
            "tensors": plan.tensors,
            "total_parameters": plan.total_parameters,
            "dtype_counts": plan.dtype_counts,
            "rank_counts": plan.rank_counts,
            "shape_mismatches": plan.shape_mismatches,
            "loader_sample": _loader_sample(Path(report.manifest_path)),
        }
        for report, plan in zip(reports, state_plans, strict=True)
    ]


def main() -> int:
    manifests = [
        KEYMAP_DIR / "sdxl_unet_keymap_manifest.json",
        KEYMAP_DIR / "anima_dit_keymap_manifest.json",
        KEYMAP_DIR / "newbie_dit_keymap_manifest.json",
    ]
    reports = inspect_many(manifests)
    state_plans = build_many_state_mapping_plans(manifests)
    if "--full" in sys.argv:
        payload = [
            {
                "inspection": report.to_dict(),
                "state_mapping_plan": plan.to_dict(),
            }
            for report, plan in zip(reports, state_plans, strict=True)
        ]
    else:
        payload = _summary_payload(reports, state_plans)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if all(report.ok for report in reports) and all(plan.ok for plan in state_plans) else 1


if __name__ == "__main__":
    raise SystemExit(main())
