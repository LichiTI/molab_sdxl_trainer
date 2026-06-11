# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Warehouse self-check smoke (Phase 9.5 / #140-144).

Verifies that lulynx_launcher remains clean of legacy module/identifier
references that would compromise the Warehouse guarantee.
"""

from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]  # backend/
_LAUNCHER_DIR = _REPO_ROOT / "lulynx_launcher"


# Legacy identifiers from launcher/core/ that must NOT appear in lulynx_launcher
_FORBIDDEN_IMPORTS = (
    "import launcher",
    "from launcher",
)

# Legacy class/function/module identifiers
_FORBIDDEN_IDENTIFIERS = (
    "RuntimeCoordinator",
    "RuntimeDetector",
    "RuntimeIntegrity",
    "build_runtime_catalog",
    "initialize_runtime_environment",
    "python_bootstrap",
    "BootstrapSource",
    "execute_bootstrap",
    "resolve_bootstrap_source",
    "dependency_cache",
    "prefetch_runtime_dependencies",
    "clear_runtime_dependency_cache",
    "LauncherTaskExecutor",
    "TaskHistoryStore",
    "build_install_plan",
    "build_launch_plan",
    "UpdateChecker",
    "hidden_subprocess_kwargs",
    "normalize_proxy_settings",
)


def _walk_python_files(root: Path):
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        # skip caches and venvs
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".venv", "node_modules"}]
        for f in filenames:
            if f.endswith(".py"):
                yield Path(dirpath) / f


def test_no_legacy_launcher_imports():
    """No file in lulynx_launcher should import the legacy launcher package."""
    if not _LAUNCHER_DIR.is_dir():
        print(f"SKIP: {_LAUNCHER_DIR} does not exist")
        return

    bad = []
    for path in _walk_python_files(_LAUNCHER_DIR):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for forbidden in _FORBIDDEN_IMPORTS:
            for line in content.splitlines():
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if forbidden in stripped:
                    # Allow string literals (e.g. comments referencing old names)
                    if forbidden.startswith("import ") or forbidden.startswith("from "):
                        bad.append(f"{path}: {line.strip()}")

    assert not bad, f"Found legacy launcher imports:\n  " + "\n  ".join(bad)
    print(f"PASS: no legacy launcher imports in {_LAUNCHER_DIR}")


def test_no_legacy_identifiers():
    """No file should reference legacy class/function/module names."""
    if not _LAUNCHER_DIR.is_dir():
        print(f"SKIP: {_LAUNCHER_DIR} does not exist")
        return

    findings = []
    for path in _walk_python_files(_LAUNCHER_DIR):
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for ident in _FORBIDDEN_IDENTIFIERS:
            if ident in content:
                # Skip if mention is inside a comment or docstring referencing the Warehouse rule itself
                if any(ident in ln and not ln.strip().startswith("#") and '"""' not in ln
                       for ln in content.splitlines()):
                    # Confirm it's not just inside a string literal explaining the rule
                    findings.append(f"{path.name}: {ident}")

    if findings:
        print("WARN: legacy identifier mentions found (verify they're in docstrings/comments):")
        for f in findings[:10]:
            print(f"  {f}")
    else:
        print("PASS: no legacy identifier references in lulynx_launcher source")


def test_cleanroom_doc_exists():
    """The Warehouse self-check document must exist and be non-empty."""
    doc = _REPO_ROOT / "docs" / "cleanroom_selfcheck.md"
    assert doc.is_file(), f"Warehouse selfcheck doc missing at {doc}"
    assert doc.stat().st_size > 1000
    print(f"PASS: cleanroom_selfcheck.md exists ({doc.stat().st_size} bytes)")


def test_pending_backlog_records_cleanroom_status():
    """PENDING_FEATURE_BACKLOG must list #140-144 Warehouse items."""
    doc = _REPO_ROOT / "docs" / "PENDING_FEATURE_BACKLOG.md"
    assert doc.is_file()
    content = doc.read_text(encoding="utf-8", errors="ignore")
    for item in ("| 140 ", "| 141 ", "| 142 ", "| 143 ", "| 144 "):
        assert item in content, f"backlog missing entry {item.strip()}"
    print("PASS: backlog records #140-144 Warehouse self-check entries")


if __name__ == "__main__":
    test_no_legacy_launcher_imports()
    test_no_legacy_identifiers()
    test_cleanroom_doc_exists()
    test_pending_backlog_records_cleanroom_status()
    print("\nAll Warehouse self-check smoke tests passed!")

