"""Runtime readiness helpers for DataLoader rebuild actions.

The bubble controller must not hot-swap DataLoaders until the training loop can
prove that iterator drain, worker shutdown, rebuild, DDP rewrap, and rollback
handles are all available.  This module records what is already observable.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping


DATALOADER_REBUILD_READINESS_PROFILE = "dataloader_rebuild_readiness_v0"
DATALOADER_REBUILD_DESCRIPTOR_ATTR = "_lulynx_dataloader_rebuild_descriptor"
DATALOADER_REBUILD_FACTORY_ATTR = "_lulynx_dataloader_rebuild_factory"
DATALOADER_REBUILD_RUNTIME_ATTRS = (
    DATALOADER_REBUILD_DESCRIPTOR_ATTR,
    DATALOADER_REBUILD_FACTORY_ATTR,
)

DATALOADER_REBUILD_REQUIRED_HANDLES = (
    "epoch_boundary_or_safe_step_pause",
    "active_iterator_drain",
    "worker_shutdown_and_join",
    "dataloader_rebuild_factory",
    "ddp_sampler_rewrap_if_needed",
    "rollback_rebuild_factory",
)

DATALOADER_REBUILD_PLAN_PROFILE = "dataloader_rebuild_plan_v0"

_MUTATION_PATH_TO_DESCRIPTOR_FIELD = {
    "cached_dataloader_workers": "num_workers",
    "dataloader_num_workers": "num_workers",
    "cached_dataloader_prefetch_factor": "prefetch_factor",
    "prefetch_factor": "prefetch_factor",
    "cached_dataloader_pin_memory": "pin_memory",
    "pin_memory": "pin_memory",
    "cached_dataloader_persistent_workers": "persistent_workers",
    "persistent_data_loader_workers": "persistent_workers",
}

_CONFIG_ONLY_MUTATION_PATHS = {
    "cached_dataloader_auto_policy",
}


def _safe_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if text in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(round(float(value if value is not None else default)))
    except (TypeError, ValueError, OverflowError):
        return int(default)


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def attach_dataloader_rebuild_descriptor(
    dataloader: Any,
    *,
    route: str,
    batch_size: int | None,
    shuffle: bool,
    drop_last: bool,
    num_workers: int,
    persistent_workers: bool = False,
    pin_memory: bool = True,
    prefetch_factor: int | None = None,
    uses_batch_sampler: bool = False,
    rebuild_factory: Callable[[Mapping[str, Any]], Any] | None = None,
    mutable_descriptor_fields: tuple[str, ...] | None = None,
) -> Any:
    """Attach compact rebuild metadata to a DataLoader-like object."""

    factory_available = callable(rebuild_factory)
    descriptor = {
        "descriptor": "dataloader_rebuild_descriptor_v0",
        "route": str(route or "unknown"),
        "batch_size": _safe_int(batch_size, 0),
        "shuffle": bool(shuffle),
        "drop_last": bool(drop_last),
        "num_workers": max(_safe_int(num_workers), 0),
        "persistent_workers": bool(persistent_workers) and _safe_int(num_workers) > 0,
        "pin_memory": bool(pin_memory),
        "prefetch_factor": None if prefetch_factor is None else max(_safe_int(prefetch_factor, 1), 1),
        "uses_batch_sampler": bool(uses_batch_sampler),
        "rebuild_factory_available": factory_available,
        "rollback_rebuild_factory_available": factory_available,
        "mutable_descriptor_fields": list(mutable_descriptor_fields or ("num_workers", "prefetch_factor", "pin_memory", "persistent_workers")),
    }
    try:
        setattr(dataloader, DATALOADER_REBUILD_DESCRIPTOR_ATTR, descriptor)
    except Exception:
        pass
    if rebuild_factory is not None:
        try:
            setattr(dataloader, DATALOADER_REBUILD_FACTORY_ATTR, rebuild_factory)
        except Exception:
            pass
    return dataloader


def dataloader_rebuild_descriptor(dataloader: Any, *, _depth: int = 0) -> dict[str, Any]:
    """Return attached or inferred rebuild metadata for DataLoader-like objects."""

    if dataloader is None or _depth > 3:
        return {}
    descriptor = _mapping(getattr(dataloader, DATALOADER_REBUILD_DESCRIPTOR_ATTR, None))
    if descriptor:
        return dict(descriptor)
    for child_name in ("_dl", "_dataloader", "dataloader"):
        child = getattr(dataloader, child_name, None)
        if child is not None and child is not dataloader:
            found = dataloader_rebuild_descriptor(child, _depth=_depth + 1)
            if found:
                return found
    inferred = _infer_descriptor(dataloader)
    return inferred


def dataloader_rebuild_factory(dataloader: Any, *, _depth: int = 0) -> Callable[[Mapping[str, Any]], Any] | None:
    """Return the attached rebuild factory, following common wrapper attributes."""

    if dataloader is None or _depth > 3:
        return None
    factory = getattr(dataloader, DATALOADER_REBUILD_FACTORY_ATTR, None)
    if callable(factory):
        return factory
    for child_name in ("_dl", "_dataloader", "dataloader"):
        child = getattr(dataloader, child_name, None)
        if child is not None and child is not dataloader:
            found = dataloader_rebuild_factory(child, _depth=_depth + 1)
            if found is not None:
                return found
    return None


def build_dataloader_rebuild_readiness_profile(
    trainer: Any = None,
    *,
    dataloader: Any = None,
    safe_boundary: str = "unknown",
    current_epoch: int | None = None,
) -> dict[str, Any]:
    """Build a compact proof of which rebuild handles are currently present."""

    descriptor = dataloader_rebuild_descriptor(dataloader)
    factory_available = dataloader_rebuild_factory(dataloader) is not None
    if descriptor:
        descriptor = {
            **descriptor,
            "rebuild_factory_available": factory_available,
            "rollback_rebuild_factory_available": factory_available,
        }
    boundary = str(safe_boundary or "unknown")
    num_workers = _safe_int(descriptor.get("num_workers"), _safe_int(getattr(dataloader, "num_workers", 0), 0))
    persistent_workers = _safe_bool(
        descriptor.get("persistent_workers"),
        _safe_bool(getattr(dataloader, "persistent_workers", False), False),
    )
    worker_shutdown = _worker_shutdown_probe(dataloader, num_workers=num_workers, persistent_workers=persistent_workers)
    epoch_boundary = boundary in {"epoch_start", "epoch_boundary", "epoch_end"}
    ddp_wrapper = getattr(trainer, "_ddp_wrapper", None) if trainer is not None else None
    uses_batch_sampler = _safe_bool(descriptor.get("uses_batch_sampler"), False)
    ddp_rewrap_ready = True if ddp_wrapper is None else hasattr(ddp_wrapper, "_dataloader") and not uses_batch_sampler

    handles = {
        "epoch_boundary_or_safe_step_pause": epoch_boundary,
        "active_iterator_drain": epoch_boundary,
        "worker_shutdown_and_join": epoch_boundary and bool(worker_shutdown.get("available")),
        "dataloader_rebuild_factory": factory_available,
        "ddp_sampler_rewrap_if_needed": ddp_rewrap_ready,
        "rollback_rebuild_factory": factory_available,
    }
    available = [name for name in DATALOADER_REBUILD_REQUIRED_HANDLES if handles.get(name)]
    missing = [name for name in DATALOADER_REBUILD_REQUIRED_HANDLES if not handles.get(name)]
    return {
        "profile": DATALOADER_REBUILD_READINESS_PROFILE,
        "current_run_rebuild_ready": not missing,
        "safe_boundary": boundary,
        "current_epoch": None if current_epoch is None else _safe_int(current_epoch),
        "descriptor": descriptor,
        "worker_shutdown": worker_shutdown,
        "handles": handles,
        "available_runtime_handles": available,
        "missing_runtime_handles": missing,
        "notes": _readiness_notes(
            epoch_boundary=epoch_boundary,
            num_workers=num_workers,
            persistent_workers=persistent_workers,
            worker_shutdown=worker_shutdown,
            missing=missing,
        ),
    }


def build_dataloader_rebuild_plan(
    action_plan: Mapping[str, Any],
    readiness: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an auditable next/rollback descriptor plan without executing it."""

    readiness_map = _mapping(readiness)
    current = dict(_mapping(readiness_map.get("descriptor")))
    mutations = action_plan.get("mutations", [])
    if not isinstance(mutations, list):
        mutations = []
    next_descriptor = dict(current)
    applied: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for item in mutations:
        mutation = dict(item) if isinstance(item, Mapping) else {}
        path = str(mutation.get("path") or "")
        if path in _CONFIG_ONLY_MUTATION_PATHS:
            skipped.append({**mutation, "skip_reason": "config_only_runtime_rebuild_path"})
            continue
        field = _MUTATION_PATH_TO_DESCRIPTOR_FIELD.get(path)
        if not field:
            skipped.append({**mutation, "skip_reason": "unsupported_dataloader_rebuild_path"})
            continue
        mutable_fields = {str(name) for name in current.get("mutable_descriptor_fields", [])}
        if mutable_fields and field not in mutable_fields:
            skipped.append({**mutation, "skip_reason": "unsupported_dataloader_rebuild_path"})
            continue
        value = _normalize_descriptor_value(field, mutation.get("recommended"))
        next_descriptor[field] = value
        applied.append({**mutation, "descriptor_field": field, "descriptor_value": value})
    mutable_fields = {str(name) for name in current.get("mutable_descriptor_fields", [])}
    if _safe_int(current.get("num_workers"), 0) <= 0 and _safe_int(next_descriptor.get("num_workers"), 0) > 0:
        if (not mutable_fields or "prefetch_factor" in mutable_fields) and next_descriptor.get("prefetch_factor") is None:
            next_descriptor["prefetch_factor"] = 2
            applied.append(
                {
                    "op": "set",
                    "path": "cached_dataloader_prefetch_factor",
                    "current": current.get("prefetch_factor"),
                    "recommended": 2,
                    "reason": "derive producer queue depth when enabling DataLoader workers",
                    "descriptor_field": "prefetch_factor",
                    "descriptor_value": 2,
                    "derived": True,
                }
            )
        if (not mutable_fields or "persistent_workers" in mutable_fields) and not _safe_bool(next_descriptor.get("persistent_workers"), False):
            next_descriptor["persistent_workers"] = True
            applied.append(
                {
                    "op": "set",
                    "path": "persistent_data_loader_workers",
                    "current": current.get("persistent_workers"),
                    "recommended": True,
                    "reason": "keep worker pool alive after enabling DataLoader workers",
                    "descriptor_field": "persistent_workers",
                    "descriptor_value": True,
                    "derived": True,
                }
            )
    _normalize_descriptor(next_descriptor)
    rollback_descriptor = dict(current)
    _normalize_descriptor(rollback_descriptor)
    unsupported = [
        str(item.get("path") or "")
        for item in skipped
        if item.get("skip_reason") == "unsupported_dataloader_rebuild_path"
    ]
    factory_available = bool(current.get("rebuild_factory_available"))
    rollback_available = bool(current.get("rollback_rebuild_factory_available"))
    return {
        "profile": DATALOADER_REBUILD_PLAN_PROFILE,
        "plan_available": bool(current) and factory_available,
        "rollback_plan_available": bool(current) and rollback_available,
        "current_run_reversible": bool(current) and factory_available and rollback_available and not unsupported,
        "route": str(current.get("route") or ""),
        "current_descriptor": current,
        "next_descriptor": next_descriptor,
        "rollback_descriptor": rollback_descriptor,
        "applied_descriptor_mutations": applied,
        "skipped_mutations": skipped,
        "unsupported_mutation_paths": unsupported,
    }


def rebuild_dataloader_from_plan(
    dataloader: Any,
    plan: Mapping[str, Any],
    *,
    target: str = "next",
) -> dict[str, Any]:
    """Rebuild a DataLoader from an auditable plan at an epoch boundary."""

    plan_map = _mapping(plan)
    key = "rollback_descriptor" if str(target or "") == "rollback" else "next_descriptor"
    descriptor = dict(_mapping(plan_map.get(key)))
    if not descriptor:
        return {"ok": False, "reason": f"missing_{key}"}

    factory = dataloader_rebuild_factory(dataloader)
    if factory is None:
        return {"ok": False, "reason": "missing_dataloader_rebuild_factory_callable"}

    shutdown = shutdown_dataloader_workers(dataloader)
    try:
        rebuilt = factory(descriptor)
    except Exception as exc:
        return {
            "ok": False,
            "reason": "dataloader_rebuild_factory_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "worker_shutdown": shutdown,
        }
    if rebuilt is None:
        return {"ok": False, "reason": "dataloader_rebuild_factory_returned_none", "worker_shutdown": shutdown}
    return {
        "ok": True,
        "reason": "rebuilt",
        "target": str(target or "next"),
        "dataloader": rebuilt,
        "requested_descriptor": descriptor,
        "rebuilt_descriptor": dataloader_rebuild_descriptor(rebuilt),
        "worker_shutdown": shutdown,
    }


def shutdown_dataloader_workers(dataloader: Any) -> dict[str, Any]:
    """Best-effort shutdown for active persistent DataLoader workers."""

    iterator = getattr(dataloader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        try:
            shutdown()
            try:
                setattr(dataloader, "_iterator", None)
            except Exception:
                pass
            return {"attempted": True, "ok": True, "strategy": "torch_iterator_shutdown_workers"}
        except Exception as exc:
            return {
                "attempted": True,
                "ok": False,
                "strategy": "torch_iterator_shutdown_workers",
                "error": f"{type(exc).__name__}: {exc}",
            }
    return {"attempted": False, "ok": True, "strategy": "not_required_or_no_active_iterator"}


def _infer_descriptor(dataloader: Any) -> dict[str, Any]:
    if dataloader is None:
        return {}
    num_workers = max(_safe_int(getattr(dataloader, "num_workers", 0), 0), 0)
    prefetch = getattr(dataloader, "prefetch_factor", None)
    batch_size = getattr(dataloader, "batch_size", None)
    batch_sampler = getattr(dataloader, "batch_sampler", None)
    return {
        "descriptor": "dataloader_rebuild_descriptor_v0",
        "route": "inferred",
        "batch_size": _safe_int(batch_size, 0),
        "shuffle": False,
        "drop_last": _safe_bool(getattr(dataloader, "drop_last", False), False),
        "num_workers": num_workers,
        "persistent_workers": _safe_bool(getattr(dataloader, "persistent_workers", False), False) and num_workers > 0,
        "pin_memory": _safe_bool(getattr(dataloader, "pin_memory", True), True),
        "prefetch_factor": None if prefetch is None else max(_safe_int(prefetch, 1), 1),
        "uses_batch_sampler": batch_sampler is not None,
        "rebuild_factory_available": False,
        "rollback_rebuild_factory_available": False,
    }


def _worker_shutdown_probe(dataloader: Any, *, num_workers: int, persistent_workers: bool) -> dict[str, Any]:
    required = bool(num_workers > 0 and persistent_workers)
    if not required:
        return {
            "required": False,
            "available": True,
            "strategy": "not_required",
        }
    has_iterator_attr = hasattr(dataloader, "_iterator")
    iterator = getattr(dataloader, "_iterator", None)
    if has_iterator_attr and iterator is None:
        return {
            "required": True,
            "available": True,
            "strategy": "no_active_persistent_iterator",
        }
    if iterator is not None and callable(getattr(iterator, "_shutdown_workers", None)):
        return {
            "required": True,
            "available": True,
            "strategy": "torch_iterator_shutdown_workers",
        }
    return {
        "required": True,
        "available": False,
        "strategy": "missing_worker_shutdown_handle",
    }


def _normalize_descriptor_value(field: str, value: Any) -> Any:
    if field in {"num_workers", "prefetch_factor"}:
        if value is None and field == "prefetch_factor":
            return None
        return max(_safe_int(value, 0 if field == "num_workers" else 1), 0 if field == "num_workers" else 1)
    if field in {"pin_memory", "persistent_workers"}:
        return _safe_bool(value, False)
    return value


def _normalize_descriptor(descriptor: dict[str, Any]) -> None:
    workers = max(_safe_int(descriptor.get("num_workers"), 0), 0)
    descriptor["num_workers"] = workers
    if workers <= 0:
        descriptor["persistent_workers"] = False
        descriptor["prefetch_factor"] = None
    else:
        descriptor["persistent_workers"] = _safe_bool(descriptor.get("persistent_workers"), False)
        prefetch = descriptor.get("prefetch_factor")
        descriptor["prefetch_factor"] = None if prefetch is None else max(_safe_int(prefetch, 1), 1)
    descriptor["pin_memory"] = _safe_bool(descriptor.get("pin_memory"), True)


def _readiness_notes(
    *,
    epoch_boundary: bool,
    num_workers: int,
    persistent_workers: bool,
    worker_shutdown: Mapping[str, Any],
    missing: list[str],
) -> list[str]:
    notes: list[str] = []
    if epoch_boundary:
        notes.append("epoch boundary provides natural iterator drain")
    else:
        notes.append("mid-epoch iterator drain handle is not available")
    if num_workers > 0 and persistent_workers:
        if worker_shutdown.get("available"):
            notes.append(f"persistent worker shutdown handle available: {worker_shutdown.get('strategy')}")
        else:
            notes.append("persistent worker shutdown still needs an explicit join handle")
    if "ddp_sampler_rewrap_if_needed" in missing:
        notes.append("DDP batch_sampler rewrap needs explicit proof before current-run rebuild")
    if "rollback_rebuild_factory" not in missing:
        notes.append("rollback rebuild descriptor is available")
    return notes[:6]


__all__ = [
    "DATALOADER_REBUILD_DESCRIPTOR_ATTR",
    "DATALOADER_REBUILD_FACTORY_ATTR",
    "DATALOADER_REBUILD_PLAN_PROFILE",
    "DATALOADER_REBUILD_READINESS_PROFILE",
    "DATALOADER_REBUILD_REQUIRED_HANDLES",
    "DATALOADER_REBUILD_RUNTIME_ATTRS",
    "attach_dataloader_rebuild_descriptor",
    "build_dataloader_rebuild_plan",
    "build_dataloader_rebuild_readiness_profile",
    "dataloader_rebuild_descriptor",
    "dataloader_rebuild_factory",
    "rebuild_dataloader_from_plan",
    "shutdown_dataloader_workers",
]
