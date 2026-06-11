"""Read-only training assistant advice builder."""

from __future__ import annotations

from typing import Any, Callable


WEBUI_EXPERIMENTAL_SCHEMA_IDS = {
    "lab-distiller",
    "sdxl-turbo-lora",
    "anima-few-step-lora",
    "newbie-few-step-lora",
}

WEBUI_EXPERIMENTAL_SCHEMA_LABELS = {
    "lab-distiller": "LAB Distiller",
    "sdxl-turbo-lora": "SDXL Turbo / LCM LoRA",
    "anima-few-step-lora": "Anima Few-step LoRA",
    "newbie-few-step-lora": "Newbie Few-step LoRA",
}


def build_training_assistant_payload(
    request: Any,
    *,
    normalize_schema_id: Callable[[str], str],
    advisor_builder: Callable[[dict[str, Any]], Any],
    schema_lookup: Callable[[str], Any | None] | None = None,
) -> dict[str, Any]:
    schema_id = normalize_schema_id(str(getattr(request, "schema_id", "") or ""))
    config = dict(getattr(request, "config", {}) or {})
    execution_profile_id = str(getattr(request, "execution_profile_id", "") or "")
    attention_backend = str(getattr(request, "attention_backend", "") or "")
    if schema_id:
        config.setdefault("schema_id", schema_id)
    if execution_profile_id:
        config.setdefault("execution_profile_id", execution_profile_id)
    if attention_backend:
        config.setdefault("attention_backend", attention_backend)

    advice: list[dict[str, Any]] = []
    schema = None
    is_webui_experimental_schema = schema_id in WEBUI_EXPERIMENTAL_SCHEMA_IDS
    if schema_id and schema_lookup and not is_webui_experimental_schema:
        schema = schema_lookup(schema_id)

    _append_schema_advice(advice, schema_id=schema_id, schema=schema, is_webui_experimental_schema=is_webui_experimental_schema)
    _append_preflight_advice(advice, getattr(request, "preflight", {}) if isinstance(getattr(request, "preflight", {}), dict) else {})
    _append_form_advice(advice, schema_id=schema_id, config=config, execution_profile_id=execution_profile_id)

    advisor_available = False
    advisor_error = ""
    try:
        advisor_report = advisor_builder(config).to_dict()
        for finding in list(advisor_report.get("findings") or [])[:8]:
            if not isinstance(finding, dict):
                continue
            advice.append(
                assistant_advice(
                    severity=str(finding.get("severity") or "note"),
                    code=str(finding.get("code") or "advisor.finding"),
                    title=str(finding.get("title") or finding.get("code") or "Advisor finding"),
                    message=str(finding.get("message") or finding.get("detail") or "Review advisor details."),
                )
            )
        advisor_available = True
    except Exception as exc:
        advisor_error = str(exc)

    deduped = dedupe_assistant_advice(advice)
    return {
        "schema_id": schema_id,
        "advisor_available": advisor_available,
        "advisor_error": advisor_error,
        "advice": deduped[:16],
        "counts": {
            "error": sum(1 for item in deduped if item.get("severity") == "error"),
            "warning": sum(1 for item in deduped if item.get("severity") == "warning"),
            "note": sum(1 for item in deduped if item.get("severity") == "note"),
        },
    }


def default_training_schema_lookup(schema_id: str) -> Any | None:
    from backend.lulynx_launcher.services.training_registry import LulynxTrainingRegistry

    return LulynxTrainingRegistry.default().get_by_id(schema_id)


def assistant_advice(
    *,
    severity: str,
    code: str,
    title: str,
    message: str,
    action_type: str = "",
    target_page: str = "",
    patch: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "severity": severity,
        "code": code,
        "title": title,
        "message": message,
        "action": {"type": action_type, "target_page": target_page} if action_type or target_page else None,
    }
    if patch:
        item["patch"] = patch
    return item


def dedupe_assistant_advice(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for item in items:
        code = str(item.get("code") or "")
        if code in seen:
            continue
        seen.add(code)
        deduped.append(item)
    return deduped


def _append_schema_advice(
    advice: list[dict[str, Any]],
    *,
    schema_id: str,
    schema: Any | None,
    is_webui_experimental_schema: bool,
) -> None:
    if is_webui_experimental_schema:
        advice.append(
            assistant_advice(
                severity="warning",
                code="assistant.webui_experimental",
                title=f"{WEBUI_EXPERIMENTAL_SCHEMA_LABELS.get(schema_id, schema_id)} is experimental",
                message="This trainer lives in WebUI and launches through the Lulynx LAB backend route. Run a short smoke before long jobs.",
            )
        )
    elif schema is None:
        advice.append(
            assistant_advice(
                severity="error",
                code="assistant.schema_missing",
                title="Training schema is unavailable",
                message="Select a valid training schema before launching.",
                action_type="navigate",
                target_page="training",
            )
        )
    else:
        status = getattr(getattr(schema, "status", None), "value", str(getattr(schema, "status", "")))
        if status == "registered_placeholder":
            advice.append(
                assistant_advice(
                    severity="error",
                    code="assistant.schema_placeholder",
                    title="Training schema is a placeholder",
                    message="This trainer is registered but not yet launchable from the launcher.",
                )
            )
        elif status == "configurable_not_verified":
            advice.append(
                assistant_advice(
                    severity="warning",
                    code="assistant.schema_unverified",
                    title="Training path is experimental",
                    message="Run a short smoke test before a long training session.",
                )
            )


def _append_preflight_advice(advice: list[dict[str, Any]], preflight: dict[str, Any]) -> None:
    for issue in list(preflight.get("issues") or [])[:8]:
        if not isinstance(issue, dict):
            continue
        severity = str(issue.get("severity") or "warning")
        advice.append(
            assistant_advice(
                severity=severity if severity in {"error", "warning", "note"} else "warning",
                code=str(issue.get("code") or "preflight.issue"),
                title=str(issue.get("source") or issue.get("code") or "Preflight issue"),
                message=str(issue.get("message") or issue.get("code") or "Review preflight details."),
                action_type="navigate",
                target_page="runtime" if str(issue.get("code") or "").startswith("profile.") else "training",
            )
        )


def _append_form_advice(
    advice: list[dict[str, Any]],
    *,
    schema_id: str,
    config: dict[str, Any],
    execution_profile_id: str,
) -> None:
    if not execution_profile_id and empty_value(config.get("execution_profile_id")):
        advice.append(
            assistant_advice(
                severity="error",
                code="assistant.profile_missing",
                title="Execution profile missing",
                message="Choose an installed runtime before launching.",
                action_type="navigate",
                target_page="runtime",
            )
        )

    model_keys = ("pretrained_model_name_or_path", "pretrained_model", "base_model_path", "model_path", "anima_model_path")
    if any(key in config for key in model_keys) and all(empty_value(config.get(key)) for key in model_keys):
        advice.append(
            assistant_advice(
                severity="warning",
                code="assistant.model_path_missing",
                title="Model path is empty",
                message="Fill a base model path or manage the model through Resource Center.",
                action_type="navigate",
                target_page="resources",
            )
        )
    if "train_data_dir" in config and empty_value(config.get("train_data_dir")):
        advice.append(
            assistant_advice(
                severity="warning",
                code="assistant.dataset_missing",
                title="Dataset path is empty",
                message="Fill the training dataset path before launch preflight.",
            )
        )
    if schema_id and "controlnet" in schema_id and empty_value(config.get("conditioning_data_dir")):
        advice.append(
            assistant_advice(
                severity="note",
                code="assistant.controlnet_conditioning_hint",
                title="Conditioning folder is not set",
                message="ControlNet can auto-discover sibling *_control files, but separate control images should use conditioning_data_dir.",
            )
        )
    if schema_id and "ip-adapter" in schema_id and empty_value(config.get("ip_image_encoder_path")):
        advice.append(
            assistant_advice(
                severity="warning",
                code="assistant.ip_adapter_encoder_missing",
                title="Image encoder path is empty",
                message="IP-Adapter needs a CLIP vision encoder such as openai/clip-vit-large-patch14.",
                action_type="navigate",
                target_page="resources",
            )
        )
    if schema_id == "yolo":
        if empty_value(config.get("dataset_yaml")) and (empty_value(config.get("train_data_dir")) or empty_value(config.get("class_names"))):
            advice.append(
                assistant_advice(
                    severity="warning",
                    code="assistant.yolo_dataset_incomplete",
                    title="YOLO dataset is incomplete",
                    message="Provide a data.yaml file or both a train directory and class names.",
                )
            )
        if empty_value(config.get("pretrained_model_name_or_path")):
            advice.append(
                assistant_advice(
                    severity="warning",
                    code="assistant.yolo_weights_missing",
                    title="YOLO weights are empty",
                    message="Use yolo11n.pt for a short smoke test or fill a local .pt path.",
                    action_type="navigate",
                    target_page="resources",
                )
            )


def empty_value(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())
