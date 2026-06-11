# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Tag Editor capability configuration.

Controls which features are available based on environment/configuration.
"""

from __future__ import annotations

import os


def is_tag_editor_advanced_enabled() -> bool:
    """Return whether advanced Tag Editor features are enabled."""
    return str(os.environ.get("LULYNX_TAG_EDITOR_ADVANCED", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


# Capability registry for documentation / introspection
BASIC_CAPABILITIES = {
    "status",
    "dataset",
    "manifest",
    "save_batch",
    "batch_action",
    "move_files",
    "delete_files",
    "history",
    "tokenize",
}

ADVANCED_CAPABILITIES = {
    "analysis",
    "suggestions",
    "interrogate",
    "lint",
    "normalize",
    "sidebar_stats",
}

ALL_CAPABILITIES = BASIC_CAPABILITIES | ADVANCED_CAPABILITIES


def get_available_capabilities() -> set[str]:
    """Return the set of currently available capabilities."""
    if is_tag_editor_advanced_enabled():
        return set(ALL_CAPABILITIES)
    return set(BASIC_CAPABILITIES)
