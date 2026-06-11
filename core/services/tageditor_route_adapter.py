"""Route-facing wrappers for synchronous/async tag-editor compatibility endpoints."""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from backend.core.services.tageditor_analysis_adapter import (
    preview_tageditor_analysis,
    submit_tageditor_analysis_job,
)
from backend.core.services.tageditor_basic_adapter import (
    create_tageditor_history_snapshot,
    get_tageditor_sidebar_stats,
    list_tageditor_history_snapshots,
    restore_tageditor_history_snapshot,
    tokenize_tageditor_caption,
)
from backend.core.services.tageditor_batch_adapter import (
    apply_tageditor_batch_action,
    preview_tageditor_batch_action,
    save_tageditor_batch_captions,
    submit_tageditor_batch_action_job,
)
from backend.core.services.tageditor_dataset_adapter import build_tageditor_dataset_payload
from backend.core.services.tageditor_file_adapter import (
    delete_tageditor_files,
    move_tageditor_files,
)
from backend.core.services.tageditor_interrogate_adapter import (
    interrogate_tageditor_image,
    submit_tageditor_retag_job,
)
from backend.core.services.tageditor_job_route_adapter import submit_tageditor_job_route_payload
from backend.core.services.tageditor_manifest_adapter import (
    build_tageditor_manifest_diff_payload,
    build_tageditor_manifest_payload,
)
from backend.core.services.tageditor_result_adapter import (
    list_tageditor_results,
    load_tageditor_analysis_result,
    load_tageditor_job_result,
)
from backend.core.services.tageditor_suggestion_adapter import (
    preview_tageditor_suggestions,
    refine_tageditor_suggestions_with_llm,
    submit_tageditor_suggestions_refresh_job,
)
from backend.core.services.tageditor_service_locator import (
    dataset_manifest_service,
    invalidate_tag_cache,
    is_cached_result_current,
    tag_analysis_service,
    tag_editor_service,
    tag_job_store,
    tag_suggestion_service,
)


SyncCall = Callable[..., dict[str, Any]]
AsyncCall = Callable[..., Awaitable[dict[str, Any]]]


def run_tageditor_sync(
    params: dict[str, Any],
    *,
    handler: SyncCall,
    **dependencies: Any,
) -> dict[str, Any]:
    """Execute a synchronous tag-editor adapter with injected dependencies."""

    return handler(params, **dependencies)


async def run_tageditor_async(
    params: dict[str, Any],
    *,
    handler: AsyncCall,
    **dependencies: Any,
) -> dict[str, Any]:
    """Execute an async tag-editor adapter with injected dependencies."""

    return await handler(params, **dependencies)


def build_tageditor_dataset_route_payload(params: dict[str, Any], *, tag_editor_service: Any) -> dict[str, Any]:
    return run_tageditor_sync(params, handler=build_tageditor_dataset_payload, tag_editor_service=tag_editor_service)


def build_tageditor_manifest_route_payload(
    params: dict[str, Any],
    *,
    manifest_service: Any,
    job_store: Any,
) -> dict[str, Any]:
    return run_tageditor_sync(
        params,
        handler=build_tageditor_manifest_payload,
        manifest_service=manifest_service,
        job_store=job_store,
    )


def build_tageditor_manifest_diff_route_payload(
    params: dict[str, Any],
    *,
    manifest_service: Any,
    job_store: Any,
) -> dict[str, Any]:
    return run_tageditor_sync(
        params,
        handler=build_tageditor_manifest_diff_payload,
        manifest_service=manifest_service,
        job_store=job_store,
    )


def preview_tageditor_analysis_route_payload(params: dict[str, Any], *, analysis_service: Any) -> dict[str, Any]:
    return run_tageditor_sync(params, handler=preview_tageditor_analysis, analysis_service=analysis_service)


def load_tageditor_analysis_result_route_payload(
    params: dict[str, Any],
    *,
    job_store: Any,
    is_current: Any,
) -> dict[str, Any]:
    return run_tageditor_sync(params, handler=load_tageditor_analysis_result, job_store=job_store, is_current=is_current)


def list_tageditor_results_route_payload(params: dict[str, Any], *, job_store: Any) -> dict[str, Any]:
    return run_tageditor_sync(params, handler=list_tageditor_results, job_store=job_store)


def load_tageditor_job_result_route_payload(params: dict[str, Any], *, job_store: Any) -> dict[str, Any]:
    return run_tageditor_sync(params, handler=load_tageditor_job_result, job_store=job_store)


def preview_tageditor_suggestions_route_payload(
    params: dict[str, Any],
    *,
    suggestion_service: Any,
    job_store: Any,
    is_current: Any,
) -> dict[str, Any]:
    return run_tageditor_sync(
        params,
        handler=preview_tageditor_suggestions,
        suggestion_service=suggestion_service,
        job_store=job_store,
        is_current=is_current,
    )


async def refine_tageditor_suggestions_route_payload(
    params: dict[str, Any],
    *,
    suggestion_service: Any,
    job_store: Any,
) -> dict[str, Any]:
    return await run_tageditor_async(
        params,
        handler=refine_tageditor_suggestions_with_llm,
        suggestion_service=suggestion_service,
        job_store=job_store,
    )


def save_tageditor_batch_route_payload(
    params: dict[str, Any],
    *,
    tag_editor: Any,
    invalidate_cache: Any = None,
) -> dict[str, Any]:
    return run_tageditor_sync(
        params,
        handler=save_tageditor_batch_captions,
        tag_editor=tag_editor,
        invalidate_cache=invalidate_cache,
    )


def apply_tageditor_batch_action_route_payload(
    params: dict[str, Any],
    *,
    tag_editor: Any,
    invalidate_cache: Any = None,
) -> dict[str, Any]:
    return run_tageditor_sync(
        params,
        handler=apply_tageditor_batch_action,
        tag_editor=tag_editor,
        invalidate_cache=invalidate_cache,
    )


def preview_tageditor_batch_action_route_payload(params: dict[str, Any], *, tag_editor: Any) -> dict[str, Any]:
    return run_tageditor_sync(params, handler=preview_tageditor_batch_action, tag_editor=tag_editor)


def get_tageditor_sidebar_stats_route_payload(params: dict[str, Any], *, tag_editor: Any) -> dict[str, Any]:
    return run_tageditor_sync(params, handler=get_tageditor_sidebar_stats, tag_editor=tag_editor)


def tokenize_tageditor_caption_route_payload(params: dict[str, Any], *, tag_editor: Any) -> dict[str, Any]:
    return run_tageditor_sync(params, handler=tokenize_tageditor_caption, tag_editor=tag_editor)


def create_tageditor_history_route_payload(params: dict[str, Any], *, tag_editor: Any) -> dict[str, Any]:
    return run_tageditor_sync(params, handler=create_tageditor_history_snapshot, tag_editor=tag_editor)


def list_tageditor_history_route_payload(params: dict[str, Any], *, tag_editor: Any) -> dict[str, Any]:
    return run_tageditor_sync(params, handler=list_tageditor_history_snapshots, tag_editor=tag_editor)


def restore_tageditor_history_route_payload(params: dict[str, Any], *, tag_editor: Any) -> dict[str, Any]:
    return run_tageditor_sync(params, handler=restore_tageditor_history_snapshot, tag_editor=tag_editor)


def interrogate_tageditor_route_payload(params: dict[str, Any], *, tag_editor: Any) -> dict[str, Any]:
    return run_tageditor_sync(params, handler=interrogate_tageditor_image, tag_editor=tag_editor)


def move_tageditor_files_route_payload(
    params: dict[str, Any],
    *,
    tag_editor: Any,
    invalidate_cache: Any = None,
) -> dict[str, Any]:
    return run_tageditor_sync(
        params,
        handler=move_tageditor_files,
        tag_editor=tag_editor,
        invalidate_cache=invalidate_cache,
    )


def delete_tageditor_files_route_payload(
    params: dict[str, Any],
    *,
    tag_editor: Any,
    invalidate_cache: Any = None,
) -> dict[str, Any]:
    return run_tageditor_sync(
        params,
        handler=delete_tageditor_files,
        tag_editor=tag_editor,
        invalidate_cache=invalidate_cache,
    )


def default_tageditor_dataset_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return build_tageditor_dataset_route_payload(params, tag_editor_service=tag_editor_service())


def default_tageditor_manifest_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return build_tageditor_manifest_route_payload(
        params,
        manifest_service=dataset_manifest_service(),
        job_store=tag_job_store(),
    )


def default_tageditor_manifest_diff_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return build_tageditor_manifest_diff_route_payload(
        params,
        manifest_service=dataset_manifest_service(),
        job_store=tag_job_store(),
    )


def default_tageditor_analysis_preview_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return preview_tageditor_analysis_route_payload(params, analysis_service=tag_analysis_service())


def default_tageditor_analysis_start_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return submit_tageditor_job_route_payload(
        params,
        submitter=submit_tageditor_analysis_job,
        analysis_service=tag_analysis_service(),
        job_store=tag_job_store(),
    )


def default_tageditor_analysis_result_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return load_tageditor_analysis_result_route_payload(
        params,
        job_store=tag_job_store(),
        is_current=is_cached_result_current,
    )


def default_tageditor_results_list_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return list_tageditor_results_route_payload(params, job_store=tag_job_store())


def default_tageditor_job_result_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return load_tageditor_job_result_route_payload(params, job_store=tag_job_store())


def default_tageditor_suggestions_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return preview_tageditor_suggestions_route_payload(
        params,
        suggestion_service=tag_suggestion_service(),
        job_store=tag_job_store(),
        is_current=is_cached_result_current,
    )


async def default_tageditor_suggestions_llm_refine_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return await refine_tageditor_suggestions_route_payload(
        params,
        suggestion_service=tag_suggestion_service(),
        job_store=tag_job_store(),
    )


def default_tageditor_suggestions_refresh_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return submit_tageditor_job_route_payload(
        params,
        submitter=submit_tageditor_suggestions_refresh_job,
        analysis_service=tag_analysis_service(),
        suggestion_service=tag_suggestion_service(),
        job_store=tag_job_store(),
    )


def default_tageditor_save_batch_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return save_tageditor_batch_route_payload(
        params,
        tag_editor=tag_editor_service(),
        invalidate_cache=invalidate_tag_cache,
    )


def default_tageditor_batch_action_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return apply_tageditor_batch_action_route_payload(
        params,
        tag_editor=tag_editor_service(),
        invalidate_cache=invalidate_tag_cache,
    )


def default_tageditor_batch_action_preview_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return preview_tageditor_batch_action_route_payload(params, tag_editor=tag_editor_service())


def default_tageditor_batch_action_start_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return submit_tageditor_job_route_payload(
        params,
        submitter=submit_tageditor_batch_action_job,
        tag_editor=tag_editor_service(),
        job_store=tag_job_store(),
        invalidate_cache=invalidate_tag_cache,
    )


def default_tageditor_sidebar_stats_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return get_tageditor_sidebar_stats_route_payload(params, tag_editor=tag_editor_service())


def default_tageditor_tokenize_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return tokenize_tageditor_caption_route_payload(params, tag_editor=tag_editor_service())


def default_tageditor_history_create_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return create_tageditor_history_route_payload(params, tag_editor=tag_editor_service())


def default_tageditor_history_list_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return list_tageditor_history_route_payload(params, tag_editor=tag_editor_service())


def default_tageditor_history_restore_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return restore_tageditor_history_route_payload(params, tag_editor=tag_editor_service())


def default_tageditor_interrogate_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return interrogate_tageditor_route_payload(params, tag_editor=tag_editor_service())


def default_tageditor_interrogate_batch_start_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return submit_tageditor_job_route_payload(
        params,
        submitter=submit_tageditor_retag_job,
        tag_editor=tag_editor_service(),
        job_store=tag_job_store(),
        invalidate_cache=invalidate_tag_cache,
    )


def default_tageditor_move_files_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return move_tageditor_files_route_payload(
        params,
        tag_editor=tag_editor_service(),
        invalidate_cache=invalidate_tag_cache,
    )


def default_tageditor_delete_files_route_payload(params: dict[str, Any]) -> dict[str, Any]:
    return delete_tageditor_files_route_payload(
        params,
        tag_editor=tag_editor_service(),
        invalidate_cache=invalidate_tag_cache,
    )
