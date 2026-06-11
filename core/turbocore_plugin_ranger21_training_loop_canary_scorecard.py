"""TrainingLoop canary wrapper for selected plugin Ranger21 native dispatch."""

from __future__ import annotations

from typing import Any

from core.turbocore_plugin_ranger_family_training_loop_canary import (
    build_ranger_family_training_loop_canary_scorecard,
)


def build_plugin_ranger21_training_loop_canary_scorecard() -> dict[str, Any]:
    return build_ranger_family_training_loop_canary_scorecard("ranger21")


__all__ = ["build_plugin_ranger21_training_loop_canary_scorecard"]
