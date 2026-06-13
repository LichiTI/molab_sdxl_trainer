from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.core.services.bubble_closed_loop_history_service import (
    attach_recent_bubble_closed_loop_action_history,
)
from backend.core.turbocore_optimizer_product_training_route_binding_run_local_staging import (
    build_optimizer_product_training_route_binding_run_local_staging,
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_holder_name(value: str) -> str:
    text = str(value or "training").strip().replace(" ", "_")
    return text or "training"


@dataclass
class QueuedTrainingRun:
    run_id: str
    run_dir: str
    holder_id: str
    config_name: str
    model_type: str
    training_type: str
    execution_profile_id: str
    schema_id: str
    requested_attention_backend: str
    resolved_attention_backend: str
    python_executable: str
    profile_env_vars: dict[str, str] = field(default_factory=dict)
    queued_at: str = field(default_factory=now_iso)


@dataclass
class QueueOperationResult:
    status: str
    run_id: str
    config_name: str
    execution_profile_id: str
    requested_attention_backend: str
    resolved_attention_backend: str
    applied_attention_backend: str | None = None
    queue_position: int = 0
    queue_depth: int = 0
    message: str = ""


def prepare_queue_run_artifacts(
    *,
    runs_dir: Path,
    entry_script: Path,
    config: Any,
    resolved: Any,
) -> tuple[QueuedTrainingRun, dict[str, Any], dict[str, Any]]:
    run_id = str(uuid.uuid4())
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    config_dict = config.model_dump()
    config_dict["network_train_unet_only"] = not bool(config_dict.pop("train_text_encoder", True))
    config_dict.pop("network_train_text_encoder_only", None)
    if config_dict.get("training_type") == "dreambooth" and config_dict.get("use_lora"):
        config_dict["network_dim"] = int(config_dict.get("lora_rank") or config_dict.get("network_dim") or 16)
    config_dict["execution_profile_id"] = resolved.execution_profile_id
    config_dict["schema_id"] = resolved.schema_id
    config_dict["attention_backend"] = resolved.resolved_attention
    attach_recent_bubble_closed_loop_action_history(config_dict, runs_dir=runs_dir, limit=3)
    (run_dir / "config.json").write_text(
        json.dumps(config_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    config_name = str(getattr(config, "output_name", "") or config_dict.get("output_name") or "training")
    model_type = str(getattr(config, "model_type", "") or config_dict.get("model_type") or "")
    training_type = str(getattr(config, "training_type", "") or config_dict.get("training_type") or "")
    started_at = now_iso()
    profile_env_vars = {
        str(k): str(v)
        for k, v in dict(getattr(resolved.profile, "env_vars", {}) or {}).items()
    }

    launch_request = {
        "run_id": run_id,
        "engine": "lulynx",
        "config_name": config_name,
        "model_type": model_type,
        "training_type": training_type,
        "execution_profile_id": resolved.execution_profile_id,
        "schema_id": resolved.schema_id,
        "requested_attention_backend": resolved.requested_attention,
        "resolved_attention_backend": resolved.resolved_attention,
        "allow_attention_fallback": resolved.allow_attention_fallback,
        "entry_script": str(entry_script),
        "python_executable": str(resolved.python_executable),
        "started_at": started_at,
    }
    (run_dir / "launch_request.json").write_text(
        json.dumps(launch_request, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    resolved_execution = {
        "execution_profile_id": resolved.execution_profile_id,
        "schema_id": resolved.schema_id,
        "python_executable": str(resolved.python_executable),
        "model_type": resolved.model_type,
        "training_type": resolved.training_type,
        "requested_attention_backend": resolved.requested_attention,
        "resolved_attention_backend": resolved.resolved_attention,
        "applied_attention_backend": None,
        "allow_attention_fallback": resolved.allow_attention_fallback,
        "fallback_reason": resolved.fallback_reason,
        "warnings": resolved.warnings,
        "profile_env_vars": profile_env_vars,
        "resolved_at": started_at,
    }
    (run_dir / "resolved_execution.json").write_text(
        json.dumps(resolved_execution, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    build_optimizer_product_training_route_binding_run_local_staging(
        run_dir=run_dir,
        refresh_config_adapter_artifact=False,
        write_artifact=False,
    )

    run = QueuedTrainingRun(
        run_id=run_id,
        run_dir=str(run_dir),
        holder_id=f"training_{sanitize_holder_name(config_name)}_{run_id[:8]}",
        config_name=config_name,
        model_type=model_type,
        training_type=training_type,
        execution_profile_id=resolved.execution_profile_id,
        schema_id=resolved.schema_id,
        requested_attention_backend=resolved.requested_attention,
        resolved_attention_backend=resolved.resolved_attention,
        python_executable=str(resolved.python_executable),
        profile_env_vars=profile_env_vars,
    )
    return run, config_dict, {"started_at": started_at}


def build_worker_env(
    *,
    backend_root: Path,
    config_dict: dict[str, Any],
    profile_env_vars: dict[str, str],
) -> dict[str, str]:
    env = os.environ.copy()
    env.update({str(k): str(v) for k, v in dict(profile_env_vars or {}).items()})
    project_root = backend_root.parent
    existing_pythonpath = env.get("PYTHONPATH", "")
    pythonpath_parts = [str(project_root), str(backend_root)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(part for part in pythonpath_parts if part)

    gpu_limit = config_dict.get("gpu_limit")
    vram_limit = config_dict.get("vram_limit")
    if isinstance(gpu_limit, (int, float)) and 0 < gpu_limit < 100:
        env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(gpu_limit)
    if isinstance(vram_limit, (int, float)) and vram_limit > 0:
        env["LULYNX_VRAM_LIMIT_GB"] = str(vram_limit)
    cuda_visible_devices = str(config_dict.get("cuda_visible_devices") or "").strip()
    if cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    if bool(config_dict.get("prevent_sleep_during_training")):
        env["LULYNX_PREVENT_SLEEP_DURING_TRAINING"] = "1"
    return env


def write_run_state(
    *,
    runs_dir: Path,
    run_id: str,
    config_name: str,
    model_type: str,
    training_type: str,
    execution_profile_id: str,
    schema_id: str,
    requested_attention_backend: str,
    resolved_attention_backend: str,
    status: str,
    started_at: str,
    updated_at: str,
    pid: int | None,
    total_epochs: int,
    total_steps: int,
    error: Any = None,
    queue_position: int | None = None,
    queue_message: str = "",
    queued_at: str = "",
    stop_reason: str = "",
) -> None:
    run_dir = runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": run_id,
        "pid": pid,
        "status": status,
        "current_step": 0,
        "current_epoch": 0,
        "total_epochs": total_epochs,
        "total_steps": total_steps,
        "last_loss": 0.0,
        "last_lr": 0.0,
        "started_at": started_at,
        "updated_at": updated_at,
        "error": error,
        "execution_profile_id": execution_profile_id or None,
        "requested_attention_backend": requested_attention_backend or None,
        "resolved_attention_backend": resolved_attention_backend or None,
        "applied_attention_backend": None,
        "schema_id": schema_id or None,
        "training_type": training_type or None,
        "model_type": model_type or None,
        "config_name": config_name or None,
        "engine": "lulynx",
    }
    if queue_position is not None:
        payload["queue_position"] = queue_position
    if queue_message:
        payload["queue_message"] = queue_message
    if queued_at:
        payload["queued_at"] = queued_at
    if stop_reason:
        payload["stop_reason"] = stop_reason
    (run_dir / "state.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def list_queued_runs(reader: Any, items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for index, item in enumerate(items, start=1):
        run_id = str(item.get("run_id") or "")
        state = reader.get_state(run_id) or {}
        launch = reader.get_launch_request(run_id) or {}
        runs.append(
            {
                "run_id": run_id,
                "status": "queued",
                "queue_position": int(state.get("queue_position") or index),
                "queued_at": state.get("queued_at") or item.get("queued_at"),
                "queue_message": state.get("queue_message") or "",
                "config_name": state.get("config_name") or launch.get("config_name") or item.get("config_name"),
                "model_type": state.get("model_type") or launch.get("model_type") or item.get("model_type"),
                "training_type": state.get("training_type") or launch.get("training_type") or item.get("training_type"),
                "execution_profile_id": state.get("execution_profile_id") or launch.get("execution_profile_id") or item.get("execution_profile_id"),
                "schema_id": state.get("schema_id") or launch.get("schema_id") or item.get("schema_id"),
            }
        )
    return runs


def refresh_queue_positions(
    *,
    reader: Any,
    runs_dir: Path,
    items: list[dict[str, Any]],
    message_overrides: dict[str, str] | None = None,
) -> dict[str, int]:
    message_overrides = dict(message_overrides or {})
    positions: dict[str, int] = {}
    for index, item in enumerate(items, start=1):
        run_id = str(item.get("run_id") or "")
        if not run_id:
            continue
        positions[run_id] = index
        state = reader.get_state(run_id) or {}
        message = message_overrides.get(run_id)
        if not message:
            ahead = max(index - 1, 0)
            message = f"Waiting for {ahead} queued training run(s) ahead." if ahead > 0 else "Waiting for training slot."
        write_run_state(
            runs_dir=runs_dir,
            run_id=run_id,
            config_name=str(state.get("config_name") or item.get("config_name") or "training"),
            model_type=str(state.get("model_type") or item.get("model_type") or ""),
            training_type=str(state.get("training_type") or item.get("training_type") or ""),
            execution_profile_id=str(state.get("execution_profile_id") or item.get("execution_profile_id") or ""),
            schema_id=str(state.get("schema_id") or item.get("schema_id") or ""),
            requested_attention_backend=str(state.get("requested_attention_backend") or item.get("requested_attention_backend") or ""),
            resolved_attention_backend=str(state.get("resolved_attention_backend") or item.get("resolved_attention_backend") or ""),
            status="queued",
            started_at=str(state.get("started_at") or item.get("queued_at") or now_iso()),
            updated_at=now_iso(),
            pid=None,
            total_epochs=int(state.get("total_epochs") or 0),
            total_steps=int(state.get("total_steps") or 0),
            queue_position=index,
            queue_message=message,
            queued_at=str(state.get("queued_at") or item.get("queued_at") or now_iso()),
        )
    return positions
