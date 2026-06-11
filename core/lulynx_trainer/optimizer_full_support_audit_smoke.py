# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Machine-readable audit for currently exposed optimizer support.

The narrower matrix smokes prove behavior.  This smoke ties their coverage to
the current config enum, local pytorch-optimizer plugin discovery, and WebUI
optimizer catalogs so a new optimizer cannot silently appear as "supported"
without trainer-path resume parity.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    project_root = backend_root.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(backend_root) not in sys.path:
        sys.path.insert(0, str(backend_root))

from core.lulynx_trainer.config import OptimizerType
from core.configs import UnifiedTrainingConfig
from core.lulynx_trainer.config_adapter import ConfigAdapter
from core.lulynx_trainer.optimizer_capabilities import optimizer_capability_report
from core.lulynx_trainer.optimizer_plugin_bridge import list_pytorch_optimizer_capabilities
from core.lulynx_trainer.optimizer_plugin_support import (
    PLUGIN_PENDING_OR_SPECIAL,
    PLUGIN_RESUME_SMOKE_PASSED,
    plugin_support_summary,
)
from core.services.official_ui_config_adapter import build_config_options_payload
from core.services.plugin_settings_adapter import (
    discover_pytorch_optimizer_names,
    resolve_pytorch_optimizer_visible_names,
)


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_PLUGIN_ROOT = _PROJECT_ROOT / "plugin"
_UI_SETTINGS_PATH = _PROJECT_ROOT / "plugin" / "lora-scripts-ui-main" / "ui" / "src" / "features" / "settingsOptions.js"

_KEY_SMOKE_COMMANDS = (
    "backend\\env\\python-flashattention\\python.exe backend\\core\\lulynx_trainer\\optimizer_full_support_audit_smoke.py",
    "backend\\env\\python-flashattention\\python.exe backend\\core\\lulynx_trainer\\optimizer_capabilities_smoke.py",
    "backend\\env\\python-flashattention\\python.exe backend\\core\\lulynx_trainer\\optimizer_state_resume_matrix_smoke.py",
    "backend\\env\\python-flashattention\\python.exe backend\\core\\lulynx_trainer\\optimizer_plugin_resume_matrix_smoke.py",
    "backend\\env\\python-flashattention\\python.exe backend\\core\\lulynx_trainer\\optimizer_step_contracts_smoke.py",
    "backend\\env\\python-flashattention\\python.exe backend\\core\\lulynx_trainer\\optimizer_expanded_matrix_smoke.py",
    "backend\\env\\python-flashattention\\python.exe backend\\core\\lulynx_trainer\\optimizer_scheduler_args_smoke.py",
    "backend\\env\\python-flashattention\\python.exe backend\\core\\lulynx_trainer\\optimizer_step_closure_training_loop_smoke.py",
    "backend\\env\\python-flashattention\\python.exe backend\\core\\lulynx_trainer\\optimizer_adahessian_contract_smoke.py",
    "backend\\env\\python-flashattention\\python.exe backend\\core\\lulynx_trainer\\optimizer_lomo_contract_smoke.py",
)


def _canonical_names(values: Iterable[Any]) -> set[str]:
    return {str(value).strip().lower() for value in values if str(value).strip()}


def _read_js_string_array(export_name: str) -> tuple[str, ...]:
    source = _UI_SETTINGS_PATH.read_text(encoding="utf-8")
    export_marker = f"export const {export_name}"
    start = source.find(export_marker)
    if start < 0:
        raise AssertionError(f"Missing UI export: {export_name}")

    open_bracket = source.find("[", start)
    if open_bracket < 0:
        raise AssertionError(f"Missing UI array opening bracket: {export_name}")

    if "dedupeKeepOrder" in source[start:open_bracket]:
        close_marker = "]);"
    else:
        close_marker = "];"
    end = source.find(close_marker, open_bracket)
    if end < 0:
        raise AssertionError(f"Missing UI array closing marker: {export_name}")

    block = source[open_bracket:end]
    return tuple(match.group(1) for match in re.finditer(r"""['"]([^'"]+)['"]""", block))


def _request_adapter_failures(plugin_names: Iterable[str]) -> list[str]:
    failures: list[str] = []
    for name in plugin_names:
        raw = f"pytorch_optimizer.{name}"
        try:
            frontend = ConfigAdapter.from_frontend_dict({"schema_id": "sdxl-lora", "optimizer_type": raw})
            direct = UnifiedTrainingConfig.from_dict({"optimizer_type": raw})
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue

        expected_arg = f"name={name}"
        if frontend.optimizer != OptimizerType.PYTORCH_OPTIMIZER or expected_arg not in str(frontend.optimizer_args):
            failures.append(f"{name}: frontend adapter produced {frontend.optimizer!r} {frontend.optimizer_args!r}")
        if direct.optimizer != OptimizerType.PYTORCH_OPTIMIZER or expected_arg not in str(direct.optimizer_args):
            failures.append(f"{name}: unified config produced {direct.optimizer!r} {direct.optimizer_args!r}")
    return failures


def build_optimizer_full_support_audit() -> dict[str, Any]:
    capability_report = optimizer_capability_report()
    capability_summary = capability_report["summary"]
    by_type = {str(item["optimizer_type"]): item for item in capability_report["optimizers"]}

    plugin_caps = list_pytorch_optimizer_capabilities()
    plugin_available = _canonical_names(plugin_caps.get("optimizers", []))
    plugin_resume = _canonical_names(PLUGIN_RESUME_SMOKE_PASSED)
    plugin_pending = _canonical_names(PLUGIN_PENDING_OR_SPECIAL.keys())
    plugin_summary = plugin_support_summary(tuple(sorted(plugin_available)))

    available_not_resume_or_pending = sorted(plugin_available - plugin_resume - plugin_pending)
    resume_missing_from_available = sorted(plugin_resume - plugin_available)
    pending_available = sorted(plugin_pending.intersection(plugin_available))

    ui_plugin_names = _canonical_names(_read_js_string_array("PYTORCH_OPTIMIZER_NAMES"))
    ui_target_literals = set(_read_js_string_array("TARGET_LORA_OPTIMIZERS"))
    ui_missing_plugin_optimizers = sorted(plugin_available - ui_plugin_names)
    ui_unsupported_plugin_optimizers = sorted(ui_plugin_names - plugin_available)
    selector_literals_present = {
        "PytorchOptimizer": "PytorchOptimizer" in ui_target_literals,
        "GenericOptimizer": "GenericOptimizer" in ui_target_literals,
    }

    settings_candidate_names = tuple(discover_pytorch_optimizer_names(_PLUGIN_ROOT))
    settings_candidates = _canonical_names(settings_candidate_names)
    settings_expose_all = _canonical_names(
        resolve_pytorch_optimizer_visible_names(
            _PLUGIN_ROOT,
            {"expose_all_optimizers": True, "max_visible_optimizers": 9999},
        )
    )
    settings_missing_plugin_optimizers = sorted(plugin_available - settings_candidates)
    settings_unsupported_plugin_optimizers = sorted(settings_candidates - plugin_available)
    settings_expose_all_missing = sorted(plugin_available - settings_expose_all)

    config_options_payload = build_config_options_payload()
    config_optimizer_options = {
        str(item).strip()
        for item in config_options_payload.get("optimizer_type", [])
        if str(item).strip()
    }
    config_plugin_options = {
        item.split(".", 1)[1].strip().lower()
        for item in config_optimizer_options
        if item.lower().startswith("pytorch_optimizer.")
    }
    config_missing_enum_optimizers = sorted(
        optimizer.value for optimizer in OptimizerType if optimizer.value not in config_optimizer_options
    )
    config_selector_literals_present = {
        "PytorchOptimizer": "PytorchOptimizer" in config_optimizer_options,
        "GenericOptimizer": "GenericOptimizer" in config_optimizer_options,
    }
    config_unsupported_visible_plugin_optimizers = sorted(config_plugin_options - plugin_available)
    config_unverified_visible_plugin_optimizers = sorted(config_plugin_options - plugin_resume)
    request_adapter_failures = _request_adapter_failures(settings_candidate_names)

    all_adapted = (
        capability_summary["missing_capability_mappings"] == []
        and bool(plugin_available)
        and not available_not_resume_or_pending
        and not resume_missing_from_available
        and not pending_available
        and not ui_missing_plugin_optimizers
        and not ui_unsupported_plugin_optimizers
        and all(selector_literals_present.values())
        and not settings_missing_plugin_optimizers
        and not settings_unsupported_plugin_optimizers
        and not settings_expose_all_missing
        and not config_missing_enum_optimizers
        and all(config_selector_literals_present.values())
        and not config_unsupported_visible_plugin_optimizers
        and not config_unverified_visible_plugin_optimizers
        and not request_adapter_failures
    )

    return {
        "schema_version": 1,
        "all_supported_optimizers_adapted": all_adapted,
        "enum": {
            "optimizer_type_count": len(list(OptimizerType)),
            "capability_report_total": capability_summary["total"],
            "missing_capability_mappings": capability_summary["missing_capability_mappings"],
            "selector_types": {
                "PytorchOptimizer": by_type.get("PytorchOptimizer", {}).get("family"),
                "GenericOptimizer": by_type.get("GenericOptimizer", {}).get("family"),
            },
        },
        "pytorch_optimizer_plugin": {
            "available_count": len(plugin_available),
            "resume_passed_count": len(plugin_resume.intersection(plugin_available)),
            "pending_available_count": len(pending_available),
            "available_not_resume_or_pending": available_not_resume_or_pending,
            "resume_missing_from_available": resume_missing_from_available,
            "pending_available": pending_available,
            "support_summary": plugin_summary,
        },
        "webui": {
            "settings_path": str(_UI_SETTINGS_PATH.relative_to(_PROJECT_ROOT)),
            "pytorch_optimizer_names_count": len(ui_plugin_names),
            "missing_plugin_optimizers": ui_missing_plugin_optimizers,
            "unsupported_plugin_optimizers": ui_unsupported_plugin_optimizers,
            "selector_literals_present": selector_literals_present,
        },
        "plugin_settings": {
            "candidate_count": len(settings_candidates),
            "expose_all_count": len(settings_expose_all),
            "missing_plugin_optimizers": settings_missing_plugin_optimizers,
            "unsupported_plugin_optimizers": settings_unsupported_plugin_optimizers,
            "expose_all_missing": settings_expose_all_missing,
        },
        "backend_config_options": {
            "optimizer_type_count": len(config_optimizer_options),
            "visible_pytorch_optimizer_count": len(config_plugin_options),
            "missing_enum_optimizers": config_missing_enum_optimizers,
            "selector_literals_present": config_selector_literals_present,
            "unsupported_visible_plugin_optimizers": config_unsupported_visible_plugin_optimizers,
            "unverified_visible_plugin_optimizers": config_unverified_visible_plugin_optimizers,
        },
        "request_config_adapter": {
            "pytorch_optimizer_alias_count": len(settings_candidate_names),
            "failures": request_adapter_failures,
        },
        "key_smoke_commands": list(_KEY_SMOKE_COMMANDS),
    }


def assert_optimizer_full_support(audit: dict[str, Any]) -> None:
    enum = audit["enum"]
    plugin = audit["pytorch_optimizer_plugin"]
    webui = audit["webui"]
    plugin_settings = audit["plugin_settings"]
    config_options = audit["backend_config_options"]
    request_adapter = audit["request_config_adapter"]

    assert enum["optimizer_type_count"] == enum["capability_report_total"], enum
    assert enum["missing_capability_mappings"] == [], enum
    assert enum["selector_types"]["PytorchOptimizer"] == "plugin_selector", enum
    assert enum["selector_types"]["GenericOptimizer"] == "generic_selector", enum

    assert plugin["available_count"] > 0, plugin
    assert plugin["available_count"] == plugin["resume_passed_count"], plugin
    assert plugin["pending_available_count"] == 0, plugin
    assert plugin["available_not_resume_or_pending"] == [], plugin
    assert plugin["resume_missing_from_available"] == [], plugin

    assert webui["pytorch_optimizer_names_count"] == plugin["available_count"], webui
    assert webui["missing_plugin_optimizers"] == [], webui
    assert webui["unsupported_plugin_optimizers"] == [], webui
    assert all(webui["selector_literals_present"].values()), webui

    assert plugin_settings["candidate_count"] == plugin["available_count"], plugin_settings
    assert plugin_settings["expose_all_count"] == plugin["available_count"], plugin_settings
    assert plugin_settings["missing_plugin_optimizers"] == [], plugin_settings
    assert plugin_settings["unsupported_plugin_optimizers"] == [], plugin_settings
    assert plugin_settings["expose_all_missing"] == [], plugin_settings

    assert config_options["missing_enum_optimizers"] == [], config_options
    assert all(config_options["selector_literals_present"].values()), config_options
    assert config_options["unsupported_visible_plugin_optimizers"] == [], config_options
    assert config_options["unverified_visible_plugin_optimizers"] == [], config_options
    assert request_adapter["pytorch_optimizer_alias_count"] == plugin["available_count"], request_adapter
    assert request_adapter["failures"] == [], request_adapter
    assert audit["all_supported_optimizers_adapted"] is True, audit


def main() -> int:
    audit = build_optimizer_full_support_audit()
    assert_optimizer_full_support(audit)
    print(json.dumps(audit, indent=2, sort_keys=True))
    print("optimizer_full_support_audit_smoke: ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
