"""Compatibility route helpers for telemetry + job-manager backed endpoints."""

from __future__ import annotations

from typing import Any, Callable


TelemetryReaderFactory = Callable[[], Any]
JobManagerFactory = Callable[[], Any]


def _default_telemetry_reader_factory() -> TelemetryReaderFactory:
    from backend.core.telemetry_store import get_file_telemetry_reader

    return get_file_telemetry_reader


def _default_job_manager_factory() -> JobManagerFactory:
    from backend.core.locator import Locator

    return Locator.get_jobs


def resolve_telemetry_reader(
    *,
    telemetry_reader: Any | None = None,
    telemetry_reader_factory: TelemetryReaderFactory | None = None,
) -> Any:
    """Return a telemetry reader instance for compat routes."""

    if telemetry_reader is not None:
        return telemetry_reader
    factory = telemetry_reader_factory or _default_telemetry_reader_factory()
    return factory()


def resolve_job_manager(
    *,
    job_manager: Any | None = None,
    job_manager_factory: JobManagerFactory | None = None,
) -> Any | None:
    """Return a job manager instance for compat routes when available."""

    if job_manager is not None:
        return job_manager
    factory = job_manager_factory or _default_job_manager_factory()
    return factory()


def run_script_payload(
    params: dict[str, Any],
    *,
    telemetry_reader: Any | None = None,
    telemetry_reader_factory: TelemetryReaderFactory | None = None,
    job_manager: Any | None = None,
    job_manager_factory: JobManagerFactory | None = None,
    submitter: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Submit a registered tool task in the legacy route shape."""

    if submitter is None:
        from backend.core.services.tool_script_adapter import submit_registered_tool_task

        submitter = submit_registered_tool_task
    reader = resolve_telemetry_reader(
        telemetry_reader=telemetry_reader,
        telemetry_reader_factory=telemetry_reader_factory,
    )
    manager = resolve_job_manager(
        job_manager=job_manager,
        job_manager_factory=job_manager_factory,
    )
    return submitter(
        params,
        telemetry_reader=reader,
        job_manager=manager,
    )


def interrogate_payload(
    params: dict[str, Any],
    *,
    telemetry_reader: Any | None = None,
    telemetry_reader_factory: TelemetryReaderFactory | None = None,
    job_manager: Any | None = None,
    job_manager_factory: JobManagerFactory | None = None,
    submitter: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Submit an interrogate task in the legacy route shape."""

    if submitter is None:
        from backend.core.services.interrogate_task_adapter import submit_interrogate_task

        submitter = submit_interrogate_task
    reader = resolve_telemetry_reader(
        telemetry_reader=telemetry_reader,
        telemetry_reader_factory=telemetry_reader_factory,
    )
    manager = resolve_job_manager(
        job_manager=job_manager,
        job_manager_factory=job_manager_factory,
    )
    return submitter(
        params,
        telemetry_reader=reader,
        job_manager=manager,
    )


def interrogate_health_payload(
    params: dict[str, Any],
    *,
    checker: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a preflight report for legacy interrogate tasks."""

    if checker is None:
        from backend.core.services.interrogate_task_adapter import build_interrogate_health_report

        checker = build_interrogate_health_report
    return checker(params)


def llm_channels_payload(
    *,
    list_channels: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """List configured LLM/API tagging channels without exposing secrets."""

    if list_channels is None:
        from backend.core.services.llm_tagger_channels import list_llm_channels

        list_channels = list_llm_channels
    return list_channels()


def save_llm_channel_payload(
    params: dict[str, Any],
    *,
    save_channel: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create or update one LLM/API tagging channel."""

    if save_channel is None:
        from backend.core.services.llm_tagger_channels import save_llm_channel

        save_channel = save_llm_channel
    return save_channel(params)


def delete_llm_channel_payload(
    channel_id: str,
    *,
    delete_channel: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Delete one persisted LLM/API tagging channel."""

    if delete_channel is None:
        from backend.core.services.llm_tagger_channels import delete_llm_channel

        delete_channel = delete_llm_channel
    return delete_channel(channel_id)


def clear_llm_channel_keys_payload(
    channel_id: str,
    *,
    clear_keys: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Clear saved API keys for one persisted LLM/API tagging channel."""

    if clear_keys is None:
        from backend.core.services.llm_tagger_channels import clear_llm_channel_keys

        clear_keys = clear_llm_channel_keys
    return clear_keys(channel_id)


def log_dirs_payload(
    *,
    telemetry_reader: Any | None = None,
    telemetry_reader_factory: TelemetryReaderFactory | None = None,
    list_payload: Callable[[Any], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build payload for ``/api/log_dirs``."""

    if list_payload is None:
        from backend.core.services.telemetry_compat_adapter import list_log_directories

        list_payload = list_log_directories
    reader = resolve_telemetry_reader(
        telemetry_reader=telemetry_reader,
        telemetry_reader_factory=telemetry_reader_factory,
    )
    return list_payload(reader)


def log_detail_payload(
    run_id: str,
    *,
    telemetry_reader: Any | None = None,
    telemetry_reader_factory: TelemetryReaderFactory | None = None,
    detail_payload: Callable[[Any, str], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build payload for ``/api/log_detail``."""

    if detail_payload is None:
        from backend.core.services.telemetry_compat_adapter import get_log_detail

        detail_payload = get_log_detail
    reader = resolve_telemetry_reader(
        telemetry_reader=telemetry_reader,
        telemetry_reader_factory=telemetry_reader_factory,
    )
    return detail_payload(reader, run_id)
