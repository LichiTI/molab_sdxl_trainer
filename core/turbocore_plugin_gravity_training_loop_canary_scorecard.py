"""TrainingLoop canary for selected plugin Gravity native dispatch."""

from __future__ import annotations

from typing import Any

from core.turbocore_plugin_simple_formula_training_loop_canary import (
    build_plugin_simple_formula_training_loop_canary_scorecard,
)


def build_plugin_gravity_training_loop_canary_scorecard() -> dict[str, Any]:
    return build_plugin_simple_formula_training_loop_canary_scorecard("gravity")


__all__ = ["build_plugin_gravity_training_loop_canary_scorecard"]
