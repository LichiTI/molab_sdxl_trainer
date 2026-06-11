"""Event helpers for the Lulynx-native aesthetic labeling store."""

from __future__ import annotations

import time
import uuid
from typing import Any


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def now_compact() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())


def canonical_status(annotation: dict[str, Any] | None) -> str:
    if not annotation:
        return "unreviewed"
    status = str(annotation.get("status") or "")
    return status if status in {"labeled", "skipped"} else "unreviewed"


def diff_annotations(before: dict[str, Any] | None, after: dict[str, Any] | None) -> dict[str, list[Any]]:
    before_flat = _flatten_annotation(before or {})
    after_flat = _flatten_annotation(after or {})
    changes: dict[str, list[Any]] = {}
    for key in sorted(set(before_flat) | set(after_flat)):
        old = before_flat.get(key)
        new = after_flat.get(key)
        if old != new:
            changes[key] = [old, new]
    return changes


def make_review_event(
    *,
    sample_id: int,
    event_type: str,
    before: dict[str, Any] | None = None,
    after: dict[str, Any] | None = None,
    changes: dict[str, Any] | None = None,
    actor: str = "local",
    request_id: str = "",
) -> dict[str, Any]:
    return {
        "event_id": f"evt_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}",
        "sample_id": int(sample_id),
        "event_type": event_type,
        "created_at": now_iso(),
        "status_before": canonical_status(before),
        "status_after": canonical_status(after),
        "changes": changes if changes is not None else diff_annotations(before, after),
        "actor": actor,
        "request_id": request_id,
    }


def annotation_to_compat(annotation: dict[str, Any] | None) -> dict[str, Any] | None:
    if not annotation:
        return None
    scores = annotation.get("scores") if isinstance(annotation.get("scores"), dict) else {}
    return {
        "status": canonical_status(annotation),
        "aesthetic": scores.get("aesthetic"),
        "composition": scores.get("composition"),
        "color": scores.get("color"),
        "sexual": scores.get("sexual"),
        "in_domain": 1 if bool(annotation.get("in_domain", True)) else 0,
        "content_type": str(annotation.get("content_type") or "anime_illust"),
        "exclude_from_score_train": 1 if bool(annotation.get("exclude_from_score_train", False)) else 0,
        "exclude_from_cls_train": 1 if bool(annotation.get("exclude_from_cls_train", False)) else 0,
        "exclude_reason": annotation.get("exclude_reason"),
        "note": annotation.get("note"),
        "updated_at": annotation.get("updated_at"),
    }


def compat_to_annotation(body: dict[str, Any], *, sample_id: int, status: str) -> dict[str, Any]:
    return {
        "sample_id": int(sample_id),
        "status": status,
        "scores": {
            "aesthetic": body.get("aesthetic"),
            "composition": body.get("composition"),
            "color": body.get("color"),
            "sexual": body.get("sexual"),
        },
        "in_domain": _truthy_int(body.get("in_domain", 1)),
        "content_type": str(body.get("content_type") or "anime_illust"),
        "exclude_from_score_train": _truthy_int(body.get("exclude_from_score_train", 0)),
        "exclude_from_cls_train": _truthy_int(body.get("exclude_from_cls_train", 0)),
        "exclude_reason": body.get("exclude_reason"),
        "note": body.get("note"),
        "updated_at": now_iso(),
    }


def merge_dimension_annotation(
    current: dict[str, Any] | None,
    *,
    sample_id: int,
    dim: str,
    score: Any,
    body: dict[str, Any],
) -> dict[str, Any]:
    merged = _default_annotation(sample_id, status="labeled")
    if current:
        merged.update({key: value for key, value in current.items() if key != "scores"})
        merged["scores"] = dict(current.get("scores") or {})
    merged["sample_id"] = int(sample_id)
    merged["status"] = "labeled"
    merged.setdefault("scores", {})
    if dim in {"aesthetic", "composition", "color", "sexual"}:
        merged["scores"][dim] = score
    _merge_meta(merged, body)
    merged["updated_at"] = now_iso()
    return merged


def skipped_annotation(current: dict[str, Any] | None, *, sample_id: int, body: dict[str, Any]) -> dict[str, Any]:
    merged = _default_annotation(sample_id, status="skipped")
    if current:
        merged.update({key: value for key, value in current.items() if key != "scores"})
        merged["scores"] = dict(current.get("scores") or {})
    merged["sample_id"] = int(sample_id)
    merged["status"] = "skipped"
    _merge_meta(merged, body)
    merged["updated_at"] = now_iso()
    return merged


def _default_annotation(sample_id: int, *, status: str) -> dict[str, Any]:
    return {
        "sample_id": int(sample_id),
        "status": status,
        "scores": {"aesthetic": None, "composition": None, "color": None, "sexual": None},
        "in_domain": True,
        "content_type": "anime_illust",
        "exclude_from_score_train": False,
        "exclude_from_cls_train": False,
        "exclude_reason": None,
        "note": None,
        "updated_at": None,
    }


def _merge_meta(annotation: dict[str, Any], body: dict[str, Any]) -> None:
    if "in_domain" in body:
        annotation["in_domain"] = _truthy_int(body.get("in_domain"))
    if "content_type" in body:
        annotation["content_type"] = str(body.get("content_type") or "anime_illust")
    if "exclude_from_score_train" in body:
        annotation["exclude_from_score_train"] = _truthy_int(body.get("exclude_from_score_train"))
    if "exclude_from_cls_train" in body:
        annotation["exclude_from_cls_train"] = _truthy_int(body.get("exclude_from_cls_train"))
    if "exclude_reason" in body:
        annotation["exclude_reason"] = body.get("exclude_reason")
    if "note" in body:
        annotation["note"] = body.get("note")


def _flatten_annotation(annotation: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in annotation.items():
        if key == "scores" and isinstance(value, dict):
            for score_key, score_value in value.items():
                flat[f"scores.{score_key}"] = score_value
        else:
            flat[key] = value
    return flat


def _truthy_int(value: Any) -> bool:
    try:
        return int(value or 0) != 0
    except Exception:
        return bool(value)

