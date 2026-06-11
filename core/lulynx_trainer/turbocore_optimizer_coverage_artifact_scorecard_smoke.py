"""Fast artifact validation for TurboCore optimizer coverage evidence.

The heavy coverage smoke still exists for refreshing the aggregation.  This
smoke is the profiled-suite default: it validates the existing coverage artifact
and cross-checks key counts against the smaller source artifacts.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[3]
BACKEND_ROOT = REPO_ROOT / "backend"
for import_root in (str(REPO_ROOT), str(BACKEND_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

ROADMAP = "devtools/docs/turbocore_optimizer_backend_design.md"
ARTIFACT_DIR = REPO_ROOT / "temp" / "turbocore_optimizer"
COVERAGE_ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_coverage_scorecard.json"
FAMILY_COVERAGE_ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_family_coverage_scorecard.json"
DEFAULT_OFF_ARTIFACT = ARTIFACT_DIR / "turbocore_plugin_selected_default_off_matrix_scorecard.json"
INVENTORY_ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_native_kernel_inventory_scorecard.json"
FAMILY_CONTRACT_ARTIFACT = ARTIFACT_DIR / "turbocore_optimizer_family_kernel_contract_scorecard.json"


def run_smoke() -> dict[str, Any]:
    coverage = _read_json(COVERAGE_ARTIFACT)
    family_coverage = _read_json(FAMILY_COVERAGE_ARTIFACT)
    default_off = _read_json(DEFAULT_OFF_ARTIFACT)
    inventory = _read_json(INVENTORY_ARTIFACT)
    family_contract = _read_json(FAMILY_CONTRACT_ARTIFACT)

    summary = _as_dict(coverage.get("summary"))
    inventory_summary = _as_dict(inventory.get("summary"))
    family_contract_summary = _as_dict(family_contract.get("summary"))
    family_contract_compact = _as_dict(coverage.get("optimizer_family_kernel_contract"))
    route_family_counts = _route_family_counts(coverage)

    assert coverage.get("ok") is True, coverage
    assert family_coverage.get("ok") is True, family_coverage
    assert coverage == family_coverage, {"coverage": COVERAGE_ARTIFACT, "family_coverage": FAMILY_COVERAGE_ARTIFACT}
    assert coverage.get("promotion_ready") is True, coverage
    assert coverage.get("training_path_enabled") is False, coverage
    assert coverage.get("native_dispatch_allowed") is False, coverage
    assert coverage.get("runtime_dispatch_allowed") is False, coverage
    assert coverage.get("product_exposure_allowed") is False, coverage
    assert coverage.get("request_fields_emitted") is False, coverage
    assert coverage.get("schema_exposure_allowed") is False, coverage
    assert coverage.get("ui_exposure_allowed") is False, coverage
    assert summary.get("plugin_optimizer_count") == 124, coverage
    assert summary.get("plugin_selected_native_ready_count") == 0, coverage
    assert int(summary.get("plugin_selected_runtime_dispatch_ready_count", 0) or 0) == 0, coverage
    assert summary.get("optimizer_native_kernel_inventory_source_ready_count") == 124, coverage
    assert summary.get("optimizer_native_kernel_inventory_probe_ready_count") == 124, coverage
    assert summary.get("optimizer_native_kernel_inventory_product_native_ready_count") == 0, coverage
    assert summary.get("optimizer_family_kernel_contract_entrypoint_present_count") == 1, coverage
    assert summary.get("optimizer_family_kernel_contract_ready_count") == 10, coverage
    assert summary.get("optimizer_family_kernel_contract_validation_ok_count") == 10, coverage
    assert summary.get("optimizer_family_kernel_contract_source_ready_count") == 10, coverage
    assert summary.get("optimizer_family_kernel_contract_training_path_enabled_count") == 0, coverage
    assert summary.get("optimizer_family_kernel_contract_native_dispatch_allowed_count") == 0, coverage
    assert summary.get("optimizer_family_kernel_contract_product_native_ready_count") == 0, coverage
    assert len(route_family_counts) == 10, {"route_family_counts": route_family_counts}
    assert sum(route_family_counts.values()) == 124, {"route_family_counts": route_family_counts}

    assert default_off.get("ok") is True, default_off
    assert default_off.get("roadmap") == ROADMAP, default_off
    assert default_off.get("optimizer_count") == 124, default_off
    assert default_off.get("case_count") == 10, default_off
    assert inventory.get("ok") is True, inventory
    assert inventory.get("roadmap") == ROADMAP, inventory
    assert inventory_summary.get("plugin_optimizer_count") == 124, inventory
    assert inventory_summary.get("kernel_source_present_count") == summary.get(
        "optimizer_native_kernel_inventory_source_ready_count"
    ), {"coverage": summary, "inventory": inventory_summary}
    assert inventory_summary.get("rust_probe_present_count") == summary.get(
        "optimizer_native_kernel_inventory_probe_ready_count"
    ), {"coverage": summary, "inventory": inventory_summary}
    assert inventory_summary.get("product_native_ready_count") == 0, inventory

    assert family_contract.get("ok") is True, family_contract
    assert family_contract.get("roadmap") == ROADMAP, family_contract
    assert family_contract_summary.get("entrypoint_present_count") == summary.get(
        "optimizer_family_kernel_contract_entrypoint_present_count"
    ), {"coverage": summary, "family_contract": family_contract_summary}
    assert family_contract_summary.get("optimizer_family_contract_count") == summary.get(
        "optimizer_family_kernel_contract_ready_count"
    ), {"coverage": summary, "family_contract": family_contract_summary}
    assert family_contract_summary.get("validation_ok_count") == summary.get(
        "optimizer_family_kernel_contract_validation_ok_count"
    ), {"coverage": summary, "family_contract": family_contract_summary}
    assert family_contract_summary.get("product_native_ready_count") == 0, family_contract
    assert family_contract_compact.get("ok") is True, family_contract_compact
    assert family_contract_compact.get("entrypoint_present") is True, family_contract_compact
    assert family_contract_compact.get("training_path_enabled") is False, family_contract_compact
    assert family_contract_compact.get("native_dispatch_allowed") is False, family_contract_compact
    assert family_contract_compact.get("product_native_ready") is False, family_contract_compact

    return {
        "schema_version": 1,
        "probe": "turbocore_optimizer_coverage_artifact_scorecard_smoke",
        "ok": True,
        "roadmap": ROADMAP,
        "artifact_mode": "artifact_validated",
        "coverage_artifact": str(COVERAGE_ARTIFACT),
        "summary": {
            "plugin_optimizer_count": int(summary.get("plugin_optimizer_count", 0) or 0),
            "optimizer_native_kernel_inventory_source_ready_count": int(
                summary.get("optimizer_native_kernel_inventory_source_ready_count", 0) or 0
            ),
            "optimizer_native_kernel_inventory_probe_ready_count": int(
                summary.get("optimizer_native_kernel_inventory_probe_ready_count", 0) or 0
            ),
            "optimizer_family_kernel_contract_ready_count": int(
                summary.get("optimizer_family_kernel_contract_ready_count", 0) or 0
            ),
            "family_count": len(route_family_counts),
            "route_family_counts": route_family_counts,
            "product_native_ready_count": int(summary.get("plugin_selected_native_ready_count", 0) or 0),
            "runtime_dispatch_ready_count": int(summary.get("plugin_selected_runtime_dispatch_ready_count", 0) or 0),
            "native_dispatch_allowed_count": int(
                summary.get("optimizer_family_kernel_contract_native_dispatch_allowed_count", 0) or 0
            ),
            "training_path_enabled_count": int(
                summary.get("optimizer_family_kernel_contract_training_path_enabled_count", 0) or 0
            ),
        },
        "recommended_next_step": "run turbocore_optimizer_coverage_scorecard_smoke.py when coverage aggregation must be refreshed",
    }


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise AssertionError({"missing_artifact": str(path)})
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AssertionError({"bad_artifact": str(path), "error": f"{type(exc).__name__}: {exc}"}) from exc
    if not isinstance(payload, dict):
        raise AssertionError({"bad_artifact": str(path), "payload_type": type(payload).__name__})
    return payload


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _route_family_counts(coverage: Mapping[str, Any]) -> dict[str, int]:
    candidates = (
        _as_dict(_as_dict(coverage.get("summary")).get("route_family_counts")),
        _as_dict(_as_dict(coverage.get("plugin_optimizer_summary")).get("plugin_selector_route_family_counts")),
        _as_dict(_as_dict(coverage.get("plugin_optimizer_family_batch")).get("route_family_counts")),
        _as_dict(_as_dict(coverage.get("plugin_selector_scorecard")).get("route_family_counts")),
    )
    for candidate in candidates:
        out: dict[str, int] = {}
        for key, value in candidate.items():
            try:
                out[str(key)] = int(value or 0)
            except (TypeError, ValueError):
                continue
        if out:
            return dict(sorted(out.items()))
    return {}


def main() -> None:
    print(json.dumps(run_smoke(), ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
