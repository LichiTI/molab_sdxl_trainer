"""Default-off execution-channel boundaries for local-window DiT attention."""

from __future__ import annotations

from typing import Any, Mapping

from .dit_frontier_request_adapter_boundaries import (
    build_dit_frontier_execution_job_creation_boundary,
    build_dit_frontier_operator_training_launch_boundary,
    build_dit_frontier_request_submission_boundary,
    build_dit_frontier_run_dispatch_boundary,
    build_dit_frontier_training_launch_boundary,
)


FEATURE_ID = "dit_local_window_attention"


def build_local_window_attention_request_submission_boundary(
    *,
    registration_boundary: Mapping[str, Any],
    submission_plan: Mapping[str, Any],
) -> dict[str, Any]:
    return build_dit_frontier_request_submission_boundary(
        feature_id=FEATURE_ID,
        registration_boundary=registration_boundary,
        submission_plan=submission_plan,
    )


def build_local_window_attention_execution_job_creation_boundary(
    *,
    request_submission: Mapping[str, Any],
    job_plan: Mapping[str, Any],
) -> dict[str, Any]:
    return build_dit_frontier_execution_job_creation_boundary(
        feature_id=FEATURE_ID,
        request_submission=request_submission,
        job_plan=job_plan,
    )


def build_local_window_attention_run_dispatch_boundary(
    *,
    execution_job_boundary: Mapping[str, Any],
    dispatch_plan: Mapping[str, Any],
) -> dict[str, Any]:
    return build_dit_frontier_run_dispatch_boundary(
        feature_id=FEATURE_ID,
        execution_job_boundary=execution_job_boundary,
        dispatch_plan=dispatch_plan,
    )


def build_local_window_attention_training_launch_boundary(
    *,
    run_dispatch_boundary: Mapping[str, Any],
    launch_plan: Mapping[str, Any],
) -> dict[str, Any]:
    return build_dit_frontier_training_launch_boundary(
        feature_id=FEATURE_ID,
        run_dispatch_boundary=run_dispatch_boundary,
        launch_plan=launch_plan,
    )


def build_local_window_attention_operator_training_launch_boundary(
    *,
    training_launch_boundary: Mapping[str, Any],
    operator_review: Mapping[str, Any],
) -> dict[str, Any]:
    return build_dit_frontier_operator_training_launch_boundary(
        feature_id=FEATURE_ID,
        training_launch_boundary=training_launch_boundary,
        operator_review=operator_review,
    )


__all__ = [
    "FEATURE_ID",
    "build_local_window_attention_execution_job_creation_boundary",
    "build_local_window_attention_operator_training_launch_boundary",
    "build_local_window_attention_request_submission_boundary",
    "build_local_window_attention_run_dispatch_boundary",
    "build_local_window_attention_training_launch_boundary",
]
