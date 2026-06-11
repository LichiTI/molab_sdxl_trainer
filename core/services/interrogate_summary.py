"""Shared progress and summary helpers for image interrogation tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def new_interrogate_summary(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": "interrogate_summary",
        "method": config.get("method", ""),
        "model": config.get("interrogator_model", ""),
        "conflict_action": config.get("conflict_action", "ignore"),
        "protect_empty_output": bool(config.get("protect_empty_output", True)),
        "additional_tag_count": len(config.get("additional_tags") or []),
        "total_images": 0,
        "processed_count": 0,
        "written_count": 0,
        "created_count": 0,
        "overwritten_count": 0,
        "appended_count": 0,
        "prepended_count": 0,
        "skipped_existing_count": 0,
        "skipped_empty_count": 0,
        "empty_output_count": 0,
        "failed_count": 0,
        "model_output_count": 0,
        "model_tag_count_total": 0,
        "average_model_tag_count": 0.0,
        "llm_attempt_count": 0,
        "llm_fallback_count": 0,
        "llm_channels_used": {},
    }


def set_job_summary(job: Any | None, summary: dict[str, Any]) -> None:
    if not job:
        return
    metadata = dict(getattr(job, "metadata", {}) or {})
    metadata["summary"] = dict(summary)
    metadata.update(
        {
            "total_images": summary.get("total_images", 0),
            "completed_count": summary.get("processed_count", 0),
            "written_count": summary.get("written_count", 0),
            "skipped_existing_count": summary.get("skipped_existing_count", 0),
            "skipped_empty_count": summary.get("skipped_empty_count", 0),
            "empty_output_count": summary.get("empty_output_count", 0),
        }
    )
    job.metadata = metadata


def record_processed(summary: dict[str, Any], job: Any | None) -> None:
    summary["processed_count"] = int(summary.get("processed_count", 0)) + 1
    set_job_summary(job, summary)


def record_generated_tags(summary: dict[str, Any], generated_tags: list[str]) -> None:
    if not generated_tags:
        return
    summary["model_output_count"] = int(summary.get("model_output_count", 0)) + 1
    summary["model_tag_count_total"] = int(summary.get("model_tag_count_total", 0)) + len(generated_tags)


def record_llm_attempt(summary: dict[str, Any], result: Any) -> None:
    attempts = int(getattr(result, "attempt_count", 1) or 1)
    fallbacks = int(getattr(result, "fallback_count", 0) or 0)
    summary["llm_attempt_count"] = int(summary.get("llm_attempt_count", 0)) + attempts
    summary["llm_fallback_count"] = int(summary.get("llm_fallback_count", 0)) + fallbacks
    channels = dict(summary.get("llm_channels_used", {}) or {})
    key = str(getattr(result, "channel_id", "") or getattr(result, "channel_name", "") or "unknown")
    channels[key] = int(channels.get(key, 0)) + 1
    summary["llm_channels_used"] = channels


def record_caption_decision(
    summary: dict[str, Any],
    existing_text: str,
    final_text: str | None,
    conflict_action: str,
) -> None:
    if final_text is None:
        summary["skipped_existing_count"] = int(summary.get("skipped_existing_count", 0)) + 1
        return
    summary["written_count"] = int(summary.get("written_count", 0)) + 1
    if not str(existing_text or "").strip():
        summary["created_count"] = int(summary.get("created_count", 0)) + 1
        return
    action = _normalize_conflict_action(conflict_action)
    if action == "append":
        summary["appended_count"] = int(summary.get("appended_count", 0)) + 1
    elif action == "prepend":
        summary["prepended_count"] = int(summary.get("prepended_count", 0)) + 1
    else:
        summary["overwritten_count"] = int(summary.get("overwritten_count", 0)) + 1


def skip_empty_model_output(
    image_path: Path,
    generated_tags: list[str],
    config: dict[str, Any],
    write_log: Callable[[str], None],
    summary: dict[str, Any],
    job: Any | None,
) -> bool:
    if generated_tags:
        return False
    summary["empty_output_count"] = int(summary.get("empty_output_count", 0)) + 1
    if bool(config.get("protect_empty_output", True)):
        summary["skipped_empty_count"] = int(summary.get("skipped_empty_count", 0)) + 1
        set_job_summary(job, summary)
        write_log(f"[interrogate] empty model output for {image_path.name}; skipped write")
        return True
    write_log(f"[interrogate] empty model output for {image_path.name}; write allowed")
    set_job_summary(job, summary)
    return False


def finalize_summary(summary: dict[str, Any]) -> None:
    output_count = int(summary.get("model_output_count", 0))
    tag_count = int(summary.get("model_tag_count_total", 0))
    summary["average_model_tag_count"] = round(tag_count / output_count, 2) if output_count else 0.0


def format_summary_line(summary: dict[str, Any]) -> str:
    return (
        "[interrogate] summary: "
        f"total={summary.get('total_images', 0)} "
        f"processed={summary.get('processed_count', 0)} "
        f"written={summary.get('written_count', 0)} "
        f"skipped_existing={summary.get('skipped_existing_count', 0)} "
        f"skipped_empty={summary.get('skipped_empty_count', 0)} "
        f"empty_output={summary.get('empty_output_count', 0)} "
        f"avg_tags={summary.get('average_model_tag_count', 0)}"
    )


def _normalize_conflict_action(value: Any) -> str:
    action = str(value or "ignore").strip().lower()
    if action in {"copy", "replace", "overwrite"}:
        return "replace"
    if action in {"ignore", "prepend", "append"}:
        return action
    return "ignore"
