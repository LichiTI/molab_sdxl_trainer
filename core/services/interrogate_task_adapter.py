"""Background interrogation task helpers for legacy compatibility routes."""

from __future__ import annotations

import threading
import uuid
from datetime import datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Callable

from backend.core.services.native_module_loader import native_with_entrypoints
from backend.core.services.interrogate_summary import (
    finalize_summary,
    format_summary_line,
    new_interrogate_summary,
    record_caption_decision,
    record_generated_tags,
    record_llm_attempt,
    record_processed,
    set_job_summary,
    skip_empty_model_output,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
ThreadFactory = Callable[[Callable[[], None]], Any]
TaggerFactory = Callable[[str], Any]


def native_interrogate_image_listing_api() -> Any:
    return native_with_entrypoints("list_image_files")


def submit_interrogate_task(
    params: dict[str, Any],
    *,
    telemetry_reader: Any,
    job_manager: Any | None = None,
    thread_factory: ThreadFactory | None = None,
    wd14_tagger_factory: TaggerFactory | None = None,
    gemini_tagger_factory: Callable[[], Any] | None = None,
    llm_tagger_factory: Callable[[dict[str, Any]], Any] | None = None,
    task_id: str | None = None,
) -> dict[str, str]:
    config = _normalize_interrogate_params(params)
    dataset_dir = Path(config["dir_path"])
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Directory not found: {config['dir_path']}")

    run_id = task_id or str(uuid.uuid4())
    log_file = _prepare_log_file(telemetry_reader, run_id)

    def write_log(message: str) -> None:
        try:
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(message + "\n")
        except Exception:
            pass

    def run_task() -> None:
        job = _register_job(job_manager, run_id, config["method"])
        summary = new_interrogate_summary(config)
        set_job_summary(job, summary)
        write_log(
            f"[interrogate] started method={config['method']} dir={config['dir_path']} "
            f"model={config['interrogator_model']} recursive={config['recursive']}"
        )
        try:
            images = _list_images(dataset_dir, recursive=bool(config["recursive"]))
            total = len(images)
            write_log(f"[interrogate] found {total} images")
            summary["total_images"] = total
            set_job_summary(job, summary)
            if job:
                job.total_items = total
            if config["method"] in ("wd", "wd14"):
                _run_wd14_interrogate(images, config, write_log, job=job, summary=summary, tagger_factory=wd14_tagger_factory)
            elif config["method"] in ("gemini", "llm"):
                _run_gemini_interrogate(
                    images,
                    config,
                    write_log,
                    job=job,
                    summary=summary,
                    tagger_factory=gemini_tagger_factory,
                    llm_tagger_factory=llm_tagger_factory,
                )
            finalize_summary(summary)
            set_job_summary(job, summary)
            write_log(format_summary_line(summary))
            write_log(f"[interrogate] done — {total} images processed")
            _finish_job(job)
        except Exception as exc:
            summary["failed_count"] = int(summary.get("failed_count", 0)) + 1
            set_job_summary(job, summary)
            write_log(f"[interrogate] ERROR: {exc}")
            _fail_job(job, str(exc))

    runner = (thread_factory or _default_thread_factory)(run_task)
    start = getattr(runner, "start", None)
    if callable(start):
        start()
    return {"task_id": run_id}


def build_interrogate_health_report(params: dict[str, Any]) -> dict[str, Any]:
    report: dict[str, Any] = {
        "ok": False,
        "method": str(params.get("method", "wd") or "wd").lower(),
        "model": str(params.get("interrogator_model") or params.get("model") or ""),
        "image_count": 0,
        "errors": [],
        "warnings": [],
        "dependencies": {},
    }
    try:
        config = _normalize_interrogate_params(params)
    except Exception as exc:
        report["errors"].append(str(exc))
        return report

    report["method"] = config["method"]
    report["model"] = config["interrogator_model"]
    dataset_dir = Path(config["dir_path"])
    if not dataset_dir.is_dir():
        report["errors"].append(f"Directory not found: {config['dir_path']}")
    else:
        images = _list_images(dataset_dir, recursive=bool(config["recursive"]))
        report["image_count"] = len(images)
        if not images:
            report["errors"].append("No supported images found in the selected directory.")

    if config["method"] in ("wd", "wd14"):
        _append_wd14_health(report, config)
    elif config["method"] in ("gemini", "llm"):
        _append_llm_health(report, config)

    report["ok"] = not report["errors"]
    return report


def merge_caption_text(
    existing_text: str,
    generated_tags: list[str],
    additional_tags: list[str],
    *,
    conflict_action: str = "ignore",
    escape_tag: bool = True,
) -> str | None:
    if not generated_tags and not additional_tags:
        return existing_text if existing_text else ""
    generated = [_escape_tag_text(tag, enabled=escape_tag) for tag in generated_tags]
    additional = [_escape_tag_text(tag, enabled=escape_tag) for tag in additional_tags]
    merged_new = generated + additional
    existing_tags = [item.strip() for item in existing_text.split(",") if item.strip()]
    conflict_action = _normalize_conflict_action(conflict_action)

    if existing_tags:
        if conflict_action == "ignore":
            return None
        if conflict_action == "prepend":
            merged = merged_new + existing_tags
        elif conflict_action == "append":
            merged = existing_tags + merged_new
        else:
            merged = merged_new
    else:
        merged = merged_new

    deduped: list[str] = []
    seen: set[str] = set()
    for tag in merged:
        key = tag.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(tag)
    return ", ".join(deduped)


def _normalize_conflict_action(value: Any) -> str:
    action = str(value or "ignore").strip().lower()
    if action in {"copy", "replace", "overwrite"}:
        return "replace"
    if action in {"ignore", "prepend", "append"}:
        return action
    return "ignore"


def _normalize_interrogate_params(params: dict[str, Any]) -> dict[str, Any]:
    dir_path = str(params.get("dir", "") or params.get("path", "") or "")
    if not dir_path:
        raise ValueError("Missing dir parameter")
    llm_template = str(params.get("llm_template_preset") or "anime-tags")
    llm_output_mode = str(params.get("llm_output_mode") or _llm_template_output_mode(llm_template) or "")
    return {
        "method": str(params.get("method", "wd") or "wd").lower(),
        "dir_path": dir_path,
        "interrogator_model": str(params.get("interrogator_model") or params.get("model") or "wd-convnext-v3"),
        "threshold": float(params.get("threshold", 0.35) or 0.35),
        "exclude_tags": _parse_tag_list(params.get("exclude_tags", "")),
        "additional_tags": _parse_tag_list(params.get("additional_tags", "")),
        "recursive": bool(params.get("batch_input_recursive", False)),
        "conflict_action": _normalize_conflict_action(params.get("batch_output_action_on_conflict", "ignore")),
        "replace_underscore": bool(params.get("replace_underscore", True)),
        "escape_tag": bool(params.get("escape_tag", True)),
        "protect_empty_output": bool(params.get("protect_empty_output", True)),
        "llm_provider": str(params.get("llm_provider") or params.get("interrogator_model") or "llm-openai"),
        "llm_api_key": str(params.get("llm_api_key") or params.get("api_key") or ""),
        "llm_api_keys": params.get("llm_api_keys", ""),
        "llm_api_base": str(params.get("llm_api_base") or params.get("api_base") or ""),
        "llm_model": str(params.get("llm_model") or params.get("model") or ""),
        "llm_channel_id": str(params.get("llm_channel_id") or ""),
        "llm_fallback_channel_ids": params.get("llm_fallback_channel_ids") or [],
        "llm_fallback_enabled": bool(params.get("llm_fallback_enabled", True)),
        "llm_retries": params.get("llm_retries", 1),
        "llm_min_tags": params.get("llm_min_tags", 1),
        "llm_max_tags": params.get("llm_max_tags", 120),
        "llm_min_caption_chars": params.get("llm_min_caption_chars", 8),
        "llm_max_caption_chars": params.get("llm_max_caption_chars", 1000),
        "llm_template_preset": llm_template,
        "llm_system_prompt": str(params.get("llm_system_prompt") or ""),
        "llm_user_prompt": str(params.get("llm_user_prompt") or params.get("prompt") or ""),
        "llm_output_mode": llm_output_mode,
        "llm_temperature": params.get("llm_temperature", 0.2),
        "llm_max_tokens": params.get("llm_max_tokens", 300),
        "llm_timeout": params.get("llm_timeout", 120),
        "llm_image_max_size": params.get("llm_image_max_size", 1280),
    }


def _llm_template_output_mode(template_id: str) -> str:
    try:
        from backend.core.services.llm_image_tagger import DEFAULT_LLM_TAG_TEMPLATES

        template = DEFAULT_LLM_TAG_TEMPLATES.get(template_id) or {}
        return str(template.get("mode") or "")
    except Exception:
        return ""


def _run_wd14_interrogate(
    images: list[Path],
    config: dict[str, Any],
    write_log: Callable[[str], None],
    *,
    job: Any | None,
    summary: dict[str, Any],
    tagger_factory: TaggerFactory | None,
) -> None:
    if tagger_factory is None:
        from backend.core.wd14_tagger import WD14Tagger

        tagger_factory = lambda model_name: WD14Tagger(model_name=model_name)
    tagger = tagger_factory(str(config["interrogator_model"]))
    try:
        for index, image_path in enumerate(images, start=1):
            caption_path = image_path.with_suffix(".txt")
            existing_text = caption_path.read_text(encoding="utf-8").strip() if caption_path.exists() else ""
            generated_text = tagger.interrogate(
                str(image_path),
                threshold=float(config["threshold"]),
                exclude_tags=list(config["exclude_tags"]),
                replace_underscore=bool(config["replace_underscore"]),
            )
            generated_tags = [item.strip() for item in str(generated_text or "").split(",") if item.strip()]
            if skip_empty_model_output(image_path, generated_tags, config, write_log, summary, job):
                _update_progress(job, index, len(images))
                record_processed(summary, job)
                write_log(f"[interrogate] [{index}/{len(images)}] {image_path.name}")
                continue
            final_text = merge_caption_text(
                existing_text,
                generated_tags,
                list(config["additional_tags"]),
                conflict_action=str(config["conflict_action"]),
                escape_tag=bool(config["escape_tag"]),
            )
            record_caption_decision(summary, existing_text, final_text, str(config["conflict_action"]))
            if final_text is not None:
                caption_path.write_text(final_text, encoding="utf-8")
            _update_progress(job, index, len(images))
            record_generated_tags(summary, generated_tags)
            record_processed(summary, job)
            write_log(f"[interrogate] [{index}/{len(images)}] {image_path.name}")
    finally:
        unload = getattr(tagger, "unload", None)
        if callable(unload):
            unload()


def _run_gemini_interrogate(
    images: list[Path],
    config: dict[str, Any],
    write_log: Callable[[str], None],
    *,
    job: Any | None,
    summary: dict[str, Any],
    tagger_factory: Callable[[], Any] | None,
    llm_tagger_factory: Callable[[dict[str, Any]], Any] | None,
) -> None:
    if tagger_factory is None:
        _run_llm_orchestrated_interrogate(images, config, write_log, job=job, summary=summary, tagger_factory=llm_tagger_factory)
        return

    # Keep the injected factory path simple for focused adapter tests.
    tagger = tagger_factory()
    _run_llm_simple_interrogate(images, config, write_log, job=job, summary=summary, tagger=tagger)


def _run_llm_orchestrated_interrogate(
    images: list[Path],
    config: dict[str, Any],
    write_log: Callable[[str], None],
    *,
    job: Any | None,
    summary: dict[str, Any],
    tagger_factory: Callable[[dict[str, Any]], Any] | None,
) -> None:
    from backend.core.services.llm_tagger_orchestrator import build_llm_orchestrator

    orchestrator = build_llm_orchestrator(config, tagger_factory=tagger_factory)
    plan = orchestrator.describe_plan()
    summary["llm_plan"] = plan
    set_job_summary(job, summary)
    write_log(f"[interrogate] llm plan usable_channels={plan.get('usable_step_count', 0)}")
    for index, image_path in enumerate(images, start=1):
        caption_path = image_path.with_suffix(".txt")
        existing_text = caption_path.read_text(encoding="utf-8").strip() if caption_path.exists() else ""
        result = orchestrator.interrogate(str(image_path), existing_caption=existing_text, image_name=image_path.name)
        record_llm_attempt(summary, result)
        tag_text = str(result.text or "").strip()
        _write_llm_result(image_path, caption_path, existing_text, tag_text, config, write_log, job, summary)
        _update_progress(job, index, len(images))
        record_processed(summary, job)
        write_log(
            f"[interrogate] [{index}/{len(images)}] {image_path.name} "
            f"channel={result.channel_name or result.channel_id} attempts={result.attempt_count}"
        )


def _run_llm_simple_interrogate(
    images: list[Path],
    config: dict[str, Any],
    write_log: Callable[[str], None],
    *,
    job: Any | None,
    summary: dict[str, Any],
    tagger: Any,
) -> None:
    for index, image_path in enumerate(images, start=1):
        caption_path = image_path.with_suffix(".txt")
        existing_text = caption_path.read_text(encoding="utf-8").strip() if caption_path.exists() else ""
        tags = tagger.interrogate(str(image_path), existing_caption=existing_text, image_name=image_path.name)
        tag_text = str(tags or "").strip()
        _write_llm_result(image_path, caption_path, existing_text, tag_text, config, write_log, job, summary)
        _update_progress(job, index, len(images))
        record_processed(summary, job)
        write_log(f"[interrogate] [{index}/{len(images)}] {image_path.name}")


def _write_llm_result(
    image_path: Path,
    caption_path: Path,
    existing_text: str,
    tag_text: str,
    config: dict[str, Any],
    write_log: Callable[[str], None],
    job: Any | None,
    summary: dict[str, Any],
) -> None:
    generated_tags = [item.strip() for item in tag_text.split(",") if item.strip()]
    if skip_empty_model_output(image_path, generated_tags or ([tag_text] if tag_text else []), config, write_log, summary, job):
        return
    final_text = _merge_llm_caption_text(existing_text, tag_text, config)
    record_caption_decision(summary, existing_text, final_text, str(config["conflict_action"]))
    if final_text is not None:
        caption_path.write_text(final_text, encoding="utf-8")
    record_generated_tags(summary, generated_tags or ([tag_text] if tag_text else []))


def _list_images(dataset_dir: Path, *, recursive: bool) -> list[Path]:
    native = native_interrogate_image_listing_api()
    if native is not None:
        try:
            image_paths = native.list_image_files(str(dataset_dir), bool(recursive))
            if isinstance(image_paths, list):
                return sorted(
                    Path(str(path))
                    for path in image_paths
                    if Path(str(path)).suffix.lower() in IMAGE_EXTENSIONS
                )
        except Exception:
            pass
    walker = dataset_dir.rglob("*") if recursive else dataset_dir.glob("*")
    return sorted(path for path in walker if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def _parse_tag_list(value: Any) -> list[str]:
    items = value if isinstance(value, list) else str(value or "").split(",")
    return [str(item).strip() for item in items if str(item).strip()]


def _escape_tag_text(tag: str, *, enabled: bool) -> str:
    if not enabled:
        return tag
    return tag.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _merge_llm_caption_text(existing_text: str, generated_text: str, config: dict[str, Any]) -> str | None:
    mode = str(config.get("llm_output_mode") or "").strip().lower()
    if mode != "caption":
        return merge_caption_text(
            existing_text,
            [item.strip() for item in generated_text.split(",") if item.strip()],
            list(config.get("additional_tags") or []),
            conflict_action=str(config["conflict_action"]),
            escape_tag=bool(config["escape_tag"]),
        )
    if existing_text and config["conflict_action"] == "ignore":
        return None
    if existing_text and config["conflict_action"] == "append":
        return f"{existing_text}, {generated_text}".strip(", ")
    if existing_text and config["conflict_action"] == "prepend":
        return f"{generated_text}, {existing_text}".strip(", ")
    return generated_text


def _append_wd14_health(report: dict[str, Any], config: dict[str, Any]) -> None:
    dependencies = report.setdefault("dependencies", {})
    required_modules = {
        "PIL": "Pillow",
        "numpy": "numpy",
        "huggingface_hub": "huggingface_hub",
        "onnxruntime": "onnxruntime",
    }
    for module_name, label in required_modules.items():
        available = find_spec(module_name) is not None
        dependencies[label] = {"available": available, "required": True}
        if not available:
            report["errors"].append(f"WD14 dependency missing: {label}")

    dependencies["pandas"] = {"available": find_spec("pandas") is not None, "required": False}
    if dependencies["onnxruntime"]["available"]:
        try:
            import onnxruntime as ort

            providers = list(ort.get_available_providers())
            report["onnxruntime_providers"] = providers
            if not providers:
                report["errors"].append("onnxruntime has no available execution providers")
        except Exception as exc:
            report["errors"].append(f"onnxruntime provider check failed: {exc}")

    try:
        from backend.core.wd14_tagger import WD14Tagger

        normalized_model = WD14Tagger.normalize_model_name(config["interrogator_model"])
        report["normalized_model"] = normalized_model
        if normalized_model not in WD14Tagger.MODELS:
            report["errors"].append(f"Unknown WD14 model: {config['interrogator_model']}")
        else:
            repo_id = WD14Tagger.MODELS[normalized_model]
            report["repo_id"] = repo_id
            _append_wd14_cache_health(report, repo_id, WD14Tagger.MODEL_FILENAME, WD14Tagger.LABEL_FILENAME)
    except Exception as exc:
        report["errors"].append(f"WD14 health check failed: {exc}")


def _append_llm_health(report: dict[str, Any], config: dict[str, Any]) -> None:
    try:
        from backend.core.services.llm_image_tagger import build_llm_health_report
        from backend.core.services.llm_tagger_channels import build_llm_channel_health_report

        if config.get("llm_channel_id") or config.get("llm_fallback_channel_ids") or not config.get("llm_api_key"):
            llm_report = build_llm_channel_health_report(config)
            report["llm"] = llm_report
            first_step = (llm_report.get("steps") or [{}])[0]
            report["model"] = first_step.get("model") or report.get("model", "")
            report["provider"] = first_step.get("provider", "")
            report["template_id"] = config.get("llm_template_preset", "")
            report["output_mode"] = config.get("llm_output_mode", "")
        else:
            llm_report = build_llm_health_report(config)
            report["llm"] = llm_report
            report["model"] = llm_report.get("model") or report.get("model", "")
            report["provider"] = llm_report.get("provider", "")
            report["template_id"] = llm_report.get("template_id", "")
            report["output_mode"] = llm_report.get("output_mode", "")
        report["errors"].extend(llm_report.get("errors") or [])
        report["warnings"].extend(llm_report.get("warnings") or [])
    except Exception as exc:
        report["errors"].append(f"LLM health check failed: {exc}")


def _append_wd14_cache_health(report: dict[str, Any], repo_id: str, model_file: str, label_file: str) -> None:
    if find_spec("huggingface_hub") is None:
        return
    try:
        from huggingface_hub.constants import HF_HUB_CACHE

        cache_dir = Path(HF_HUB_CACHE)
        repo_cache = cache_dir / f"models--{repo_id.replace('/', '--')}"
        model_cached = any(repo_cache.glob(f"snapshots/*/{model_file}"))
        label_cached = any(repo_cache.glob(f"snapshots/*/{label_file}"))
        report["model_cache"] = {
            "cache_dir": str(repo_cache),
            "model_file": model_cached,
            "label_file": label_cached,
            "complete": model_cached and label_cached,
        }
        if not (model_cached and label_cached):
            report["warnings"].append("WD14 model files are not fully cached locally; first run may download model files.")
    except Exception as exc:
        report["warnings"].append(f"WD14 cache check skipped: {exc}")


def _prepare_log_file(telemetry_reader: Any, task_id: str) -> Path:
    run_dir = Path(telemetry_reader._runs_dir) / task_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "output.log"


def _default_thread_factory(target: Callable[[], None]) -> threading.Thread:
    return threading.Thread(target=target, daemon=True)


def _register_job(job_manager: Any | None, task_id: str, method: str) -> Any | None:
    if not job_manager:
        return None
    from backend.core.job_manager import Job, JobStatus, JobType

    job = Job(id=task_id, type=JobType.TAGGING, name=f"Interrogate ({method})", status=JobStatus.RUNNING)
    job_manager._jobs[task_id] = job
    return job


def _update_progress(job: Any | None, completed: int, total: int) -> None:
    if not job:
        return
    job.completed_items = completed
    job.update_progress(completed, total)


def _finish_job(job: Any | None) -> None:
    if not job:
        return
    from backend.core.job_manager import JobStatus

    job.status = JobStatus.COMPLETED
    job.progress = 1.0
    job.finished_at = datetime.now()


def _fail_job(job: Any | None, error: str) -> None:
    if not job:
        return
    from backend.core.job_manager import JobStatus

    job.status = JobStatus.FAILED
    job.error = error
    job.finished_at = datetime.now()
