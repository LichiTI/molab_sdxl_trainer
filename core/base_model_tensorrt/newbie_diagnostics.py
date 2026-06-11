from __future__ import annotations

import time
import shutil
from pathlib import Path
from typing import Any, Sequence

from .newbie_engine import (
    build_newbie_tensorrt_engine,
    compare_newbie_tensorrt_parity,
    default_newbie_engine_path,
    default_newbie_onnx_path,
    export_newbie_static_onnx,
)
from .newbie_export import NewbieStaticShape, parse_layer_indices


DEFAULT_DIAGNOSTIC_WINDOWS = ("0-7", "0-15", "0-23", "0-35")


def parse_diagnostic_windows(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None or value == "":
        return DEFAULT_DIAGNOSTIC_WINDOWS
    if isinstance(value, str):
        raw = value.replace("|", ";").replace("\n", ";").split(";")
    else:
        raw = [str(item) for item in value]
    windows: list[str] = []
    seen: set[str] = set()
    for item in raw:
        token = item.strip()
        if not token:
            continue
        layers = parse_layer_indices(token)
        normalized = _window_label(layers)
        if normalized not in seen:
            seen.add(normalized)
            windows.append(normalized)
    return tuple(windows) or DEFAULT_DIAGNOSTIC_WINDOWS


def diagnose_newbie_tensorrt_windows(
    *,
    checkpoint_path: str | Path = "",
    model_root: str | Path = "",
    config_path: str | Path = "",
    output_dir: str | Path = "",
    windows: str | Sequence[str] | None = None,
    shape: NewbieStaticShape | None = None,
    device: str = "cuda",
    dtype_name: str = "float32",
    seed: int = 1337,
    opset: int = 18,
    precision: str = "bf16",
    fp32_layer_policy: str = "sensitive",
    workspace_mb: int = 4096,
    external_data: bool = True,
    reuse_existing: bool = True,
    cleanup_artifacts: bool = False,
    stop_on_failure: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    shape = shape or NewbieStaticShape()
    window_specs = parse_diagnostic_windows(windows)
    root = _diagnostic_root(output_dir)
    results: list[dict[str, Any]] = []
    for window in window_specs:
        result = _run_window(
            checkpoint_path=checkpoint_path,
            model_root=model_root,
            config_path=config_path,
            output_dir=root,
            window=window,
            shape=shape,
            device=device,
            dtype_name=dtype_name,
            seed=seed,
            opset=opset,
            precision=precision,
            fp32_layer_policy=fp32_layer_policy,
            workspace_mb=workspace_mb,
            external_data=external_data,
            reuse_existing=reuse_existing,
            cleanup_artifacts=cleanup_artifacts,
        )
        results.append(result)
        if stop_on_failure and not bool(result.get("success")):
            break
    return {
        "schema_version": 1,
        "kind": "newbie_tensorrt_window_diagnostics",
        "success": all(bool(item.get("success")) for item in results),
        "model_family": "newbie",
        "windows": list(window_specs),
        "precision": _precision_label(precision),
        "fp32_layer_policy": _policy_label(fp32_layer_policy),
        "shape": shape.to_dict(),
        "output_dir": str(root),
        "reuse_existing": bool(reuse_existing),
        "cleanup_artifacts": bool(cleanup_artifacts),
        "external_data": bool(external_data),
        "results": results,
        "summary": _summarize_windows(results),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def _run_window(
    *,
    checkpoint_path: str | Path,
    model_root: str | Path,
    config_path: str | Path,
    output_dir: Path,
    window: str,
    shape: NewbieStaticShape,
    device: str,
    dtype_name: str,
    seed: int,
    opset: int,
    precision: str,
    fp32_layer_policy: str,
    workspace_mb: int,
    external_data: bool,
    reuse_existing: bool,
    cleanup_artifacts: bool,
) -> dict[str, Any]:
    started = time.perf_counter()
    layers = parse_layer_indices(window)
    window_dir = output_dir / _window_dir_name(layers, precision, fp32_layer_policy)
    onnx_path = default_newbie_onnx_path(output_dir=window_dir, shape=shape, layer_indices=layers, opset=opset)
    engine_path = _diagnostic_engine_path(
        output_dir=window_dir,
        shape=shape,
        layer_indices=layers,
        opset=opset,
        precision=precision,
        fp32_layer_policy=fp32_layer_policy,
    )
    item: dict[str, Any] = {
        "window": _window_label(layers),
        "layer_indices": list(layers),
        "layer_count": len(layers),
        "output_dir": str(window_dir),
        "onnx_path": str(onnx_path),
        "engine_path": str(engine_path),
    }
    try:
        export_result = _reuse_step(onnx_path, kind="onnx") if reuse_existing and onnx_path.is_file() else export_newbie_static_onnx(
            checkpoint_path=checkpoint_path,
            model_root=model_root,
            config_path=config_path,
            output_path=onnx_path,
            layer_indices=layers,
            shape=shape,
            device="cpu",
            dtype_name="float32",
            seed=seed,
            opset=opset,
            external_data=external_data,
        )
        item["export"] = _compact_step(export_result)
        if not bool(export_result.get("success")):
            item["success"] = False
            return _finish(item, started)

        build_result = _reuse_step(engine_path, kind="engine") if reuse_existing and engine_path.is_file() else build_newbie_tensorrt_engine(
            onnx_path=onnx_path,
            output_path=engine_path,
            layer_indices=layers,
            shape=shape,
            opset=opset,
            precision=precision,
            workspace_mb=workspace_mb,
            fp32_layer_policy=fp32_layer_policy,
        )
        item["build"] = _compact_step(build_result)
        if not bool(build_result.get("success")):
            item["success"] = False
            return _finish(item, started)

        parity_result = compare_newbie_tensorrt_parity(
            checkpoint_path=checkpoint_path,
            model_root=model_root,
            config_path=config_path,
            engine_path=engine_path,
            layer_indices=layers,
            shape=shape,
            device=device,
            dtype_name=dtype_name,
            seed=seed,
            opset=opset,
            precision=precision,
        )
        item["parity"] = _compact_parity(parity_result)
        item["success"] = bool(parity_result.get("success"))
    except Exception as exc:
        item["success"] = False
        item["error"] = f"{type(exc).__name__}: {exc}"
    if cleanup_artifacts:
        item["cleanup"] = _cleanup_window_artifacts(window_dir=window_dir, root=output_dir)
    return _finish(item, started)


def _compact_step(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": bool(result.get("success")),
        "reused": bool(result.get("reused", False)),
        "bytes": int(result.get("bytes") or result.get("artifact_bytes") or result.get("onnx_artifact_bytes") or 0),
        "precision": result.get("precision", ""),
        "elapsed_seconds": result.get("elapsed_seconds", 0),
        "error": result.get("error", ""),
    }


def _compact_parity(result: dict[str, Any]) -> dict[str, Any]:
    comparison = dict(result.get("comparison") or {})
    return {
        "success": bool(result.get("success")),
        "parity_acceptable": bool(comparison.get("parity_acceptable", False)),
        "mean_abs": comparison.get("mean_abs"),
        "max_abs": comparison.get("max_abs"),
        "mean_rel": comparison.get("mean_rel"),
        "max_rel": comparison.get("max_rel"),
        "all_finite": comparison.get("all_finite"),
        "elapsed_seconds": result.get("elapsed_seconds", 0),
    }


def _reuse_step(path: Path, *, kind: str) -> dict[str, Any]:
    return {
        "success": True,
        "reused": True,
        "kind": kind,
        "path": str(path),
        "bytes": path.stat().st_size if path.is_file() else 0,
        "elapsed_seconds": 0,
    }


def _summarize_windows(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    parity_items = [item for item in results if isinstance(item.get("parity"), dict)]
    unacceptable = [item for item in parity_items if not item["parity"].get("parity_acceptable")]
    best = min(
        parity_items,
        key=lambda item: float(item["parity"].get("mean_abs") if item["parity"].get("mean_abs") is not None else float("inf")),
        default=None,
    )
    return {
        "window_count": len(results),
        "completed_count": len(parity_items),
        "all_parity_acceptable": bool(parity_items) and not unacceptable,
        "first_unacceptable_window": unacceptable[0]["window"] if unacceptable else "",
        "best_window_by_mean_abs": best["window"] if best else "",
        "best_mean_abs": best["parity"].get("mean_abs") if best else None,
    }


def _finish(item: dict[str, Any], started: float) -> dict[str, Any]:
    item["elapsed_seconds"] = round(time.perf_counter() - started, 4)
    return item


def _cleanup_window_artifacts(*, window_dir: Path, root: Path) -> dict[str, Any]:
    try:
        resolved_root = root.resolve()
        resolved_window = window_dir.resolve()
        if resolved_window == resolved_root or resolved_root not in resolved_window.parents:
            return {"ok": False, "removed": False, "reason": "window_dir_outside_output_root", "path": str(window_dir)}
        if not resolved_window.is_dir():
            return {"ok": True, "removed": False, "reason": "missing", "path": str(window_dir)}
        shutil.rmtree(resolved_window)
        return {"ok": True, "removed": True, "path": str(window_dir)}
    except Exception as exc:
        return {"ok": False, "removed": False, "reason": f"{type(exc).__name__}: {exc}", "path": str(window_dir)}


def _diagnostic_root(output_dir: str | Path) -> Path:
    if str(output_dir or "").strip():
        return Path(output_dir)
    default = Path("H:/tmp/lulynx_newbie_tensorrt_windows")
    if default.drive:
        return default
    return Path.cwd() / "tmp" / "lulynx_newbie_tensorrt_windows"


def _diagnostic_engine_path(
    *,
    output_dir: str | Path,
    shape: NewbieStaticShape,
    layer_indices: Sequence[int],
    opset: int,
    precision: str,
    fp32_layer_policy: str,
) -> Path:
    base = default_newbie_engine_path(
        output_dir=output_dir,
        shape=shape,
        layer_indices=layer_indices,
        opset=opset,
        precision=precision,
    )
    policy = _policy_label(fp32_layer_policy)
    if policy == "none":
        return base
    return base.with_name(f"{base.stem}_{policy}.engine")


def _window_dir_name(layers: Sequence[int], precision: str, fp32_layer_policy: str) -> str:
    return f"{_layer_label(layers)}_{_precision_label(precision)}_{_policy_label(fp32_layer_policy)}"


def _layer_label(layers: Sequence[int]) -> str:
    values = tuple(int(item) for item in layers)
    if not values:
        return "l0"
    if len(values) == values[-1] - values[0] + 1:
        return f"l{values[0]}" if len(values) == 1 else f"l{values[0]}-{values[-1]}"
    return "l" + "_".join(str(item) for item in values)


def _window_label(layers: Sequence[int]) -> str:
    values = tuple(int(item) for item in layers)
    if not values:
        return "0"
    if len(values) == values[-1] - values[0] + 1:
        return str(values[0]) if len(values) == 1 else f"{values[0]}-{values[-1]}"
    return "_".join(str(item) for item in values)


def _precision_label(value: str) -> str:
    key = str(value or "fp32").strip().lower().replace("-", "_")
    aliases = {"float32": "fp32", "float16": "fp16", "half": "fp16", "bfloat16": "bf16"}
    return aliases.get(key, key)


def _policy_label(value: str) -> str:
    key = str(value or "none").strip().lower().replace("-", "_")
    if key in {"", "off", "false", "0"}:
        return "none"
    return key
