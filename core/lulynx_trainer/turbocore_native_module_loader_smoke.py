"""Smoke checks for shared lulynx_native artifact discovery."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.services.native_module_loader import (  # noqa: E402
    discover_lulynx_native_artifact_dirs,
    load_lulynx_native,
    probe_lulynx_native_loader,
)


def run_smoke() -> dict[str, object]:
    dirs = discover_lulynx_native_artifact_dirs()
    loader = probe_lulynx_native_loader()
    native = load_lulynx_native()
    has_local_artifact = bool(dirs)
    importable = native is not None
    return {
        "schema_version": 1,
        "probe": "turbocore_native_module_loader_smoke",
        "ok": bool((not has_local_artifact) or importable),
        "has_local_artifact": has_local_artifact,
        "artifact_dirs": [str(path) for path in dirs],
        "importable": importable,
        "origin": str(getattr(native, "__file__", "") or "") if native is not None else "",
        "loader": loader,
        "training_path_enabled": False,
    }


if __name__ == "__main__":
    result = run_smoke()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not bool(result.get("ok", False)):
        raise SystemExit(1)
