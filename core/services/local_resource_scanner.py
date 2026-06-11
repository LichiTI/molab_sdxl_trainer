"""Bounded local resource scanning for launcher/WebUI compatibility routes."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

try:
    from core.services.image_gguf_resource_metadata import IMAGE_GGUF_ARCH, read_image_gguf_sidecar
    from core.services.native_module_loader import load_lulynx_native
except ImportError:
    from backend.core.services.image_gguf_resource_metadata import IMAGE_GGUF_ARCH, read_image_gguf_sidecar
    from backend.core.services.native_module_loader import load_lulynx_native

logger = logging.getLogger("lulynx.local_resource_scanner")

RESOURCE_MODEL_SUFFIXES = {".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx"}
RESOURCE_CONFIG_SUFFIXES = {".json", ".yaml", ".yml", ".toml", ".txt", ".model"}
RESOURCE_METADATA_MAX_BYTES = 16 * 1024 * 1024
SAFETENSORS_HEADER_MAX_BYTES = 8 * 1024 * 1024
LOCAL_RESOURCE_CACHE_TTL_SEC = 20.0
_LOCAL_RESOURCE_CACHE: dict[tuple[tuple[str, ...], int, bool, bool], tuple[float, dict[str, Any]]] = {}
ArtifactProvider = Callable[[int], list[dict[str, Any]]]


def native_local_resource_scan_disabled() -> bool:
    return str(os.environ.get("LULYNX_DISABLE_NATIVE_LOCAL_RESOURCE_SCAN", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def load_native_local_resource_scan_api() -> Any:
    return load_lulynx_native()


load_native_local_resource_scan_api.cache_clear = load_lulynx_native.cache_clear  # type: ignore[attr-defined]


def native_local_resource_scan_api() -> Any:
    if native_local_resource_scan_disabled():
        return None
    native = load_native_local_resource_scan_api()
    if not hasattr(native, "scan_local_resource_candidates"):
        return None
    return native


def default_job_artifact_provider() -> ArtifactProvider:
    from resources.web.deps import get_job_manager

    from backend.core.services.job_artifact_resources import list_job_artifact_resources

    def _provider(limit: int) -> list[dict[str, Any]]:
        return list_job_artifact_resources(get_job_manager(), limit=limit)

    return _provider


def scan_local_resources_route_payload(
    *,
    project_root: Path,
    backend_root: Path,
    max_items: int,
    summary_only: bool = False,
    include_hash: bool = False,
    refresh: bool = False,
    artifact_provider: ArtifactProvider | None = None,
) -> dict[str, Any]:
    return scan_local_resources_cached(
        project_root=project_root,
        backend_root=backend_root,
        max_items=max_items,
        summary_only=summary_only,
        include_hash=include_hash,
        refresh=refresh,
        artifact_provider=artifact_provider or default_job_artifact_provider(),
    )


def resource_roots(*, project_root: Path, backend_root: Path) -> list[Path]:
    candidates = [
        project_root / "models",
        project_root / "model",
        project_root / "resources" / "models",
        backend_root / "models",
        backend_root / "resources" / "models",
        project_root / "output",
    ]
    roots: list[Path] = []
    seen: set[str] = set()
    for root in candidates:
        try:
            resolved = root.resolve()
        except Exception:
            continue
        key = str(resolved).lower()
        if key in seen or not resolved.exists() or not resolved.is_dir():
            continue
        seen.add(key)
        roots.append(resolved)
    return roots


def resource_category(path: Path, root: Path, model_type: str = "") -> str:
    text = " ".join(part.lower() for part in path.relative_to(root).parts)
    suffix = path.suffix.lower()
    if "translation" in text or "translator" in text or "nllb" in text or "argos" in text:
        return "translation"
    if model_type == "image-gguf":
        return "models"
    if model_type == "llm" or suffix == ".gguf" or "llm" in text or "ollama" in text or "qwen" in text or "gemma" in text:
        return "llm"
    if "preset" in text or "sample" in text or "training" in text:
        return "training"
    return "models"


def resource_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".safetensors", ".ckpt"}:
        return "checkpoint"
    if suffix == ".gguf":
        return "llm"
    if suffix in {".pt", ".pth"}:
        return "torch"
    if suffix in RESOURCE_CONFIG_SUFFIXES:
        return "config"
    return suffix.lstrip(".") or "file"


def read_json_sidecar(path: Path) -> tuple[dict[str, Any], str]:
    candidates: list[tuple[Path, str]] = [
        (path.with_suffix(path.suffix + ".metadata.json"), "metadata_sidecar"),
        (path.with_suffix(path.suffix + ".json"), "manifest"),
        (path.with_suffix(".json"), "manifest"),
        (path.parent / "manifest.json", "manifest"),
        (path.parent / "model_index.json", "manifest"),
    ]
    for candidate, source in candidates:
        try:
            if candidate.is_file() and candidate.stat().st_size <= 1024 * 1024:
                data = json.loads(candidate.read_text(encoding="utf-8-sig"))
                if isinstance(data, dict):
                    return data, source
        except Exception:
            continue
    return {}, ""


def read_safetensors_header(path: Path) -> tuple[dict[str, Any], list[str], dict[str, Any]]:
    if path.suffix.lower() != ".safetensors":
        return {}, [], {}
    try:
        with path.open("rb") as fh:
            raw_len = fh.read(8)
            if len(raw_len) != 8:
                return {}, [], {}
            header_len = int.from_bytes(raw_len, "little")
            if header_len <= 0 or header_len > SAFETENSORS_HEADER_MAX_BYTES:
                return {}, [], {"header_error": f"header too large: {header_len}"}
            header = json.loads(fh.read(header_len).decode("utf-8"))
    except Exception as exc:
        return {}, [], {"header_error": str(exc)}
    if not isinstance(header, dict):
        return {}, [], {}
    metadata = header.get("__metadata__")
    if not isinstance(metadata, dict):
        metadata = {}
    tensor_keys = [str(key) for key in header.keys() if key != "__metadata__"]
    dtypes = sorted({str(value.get("dtype")) for value in header.values() if isinstance(value, dict) and value.get("dtype")})
    return {str(k): v for k, v in metadata.items()}, tensor_keys[:120], {"tensor_count": len(tensor_keys), "dtypes": dtypes}


def compact_manifest(manifest: dict[str, Any], tensor_info: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id", "name", "model_type", "model_family", "artifact_kind", "base_model", "architecture", "_class_name",
        "modelspec.architecture", "modelspec.title", "modelspec.implementation", "ss_network_module", "ss_base_model_version",
        "ss_sd_model_name", "format", "lulynx.schema_id", "lulynx.artifact_kind", "lulynx.model_family",
        "gguf_arch", "gguf_file_type", "tensor_count", "output_size_bytes", "converted_tensors", "skipped_tensors",
        "dtype_counts", "rank_counts", "lulynx.image_gguf.schema", "lulynx.image_gguf.component",
        "lulynx.image_gguf.family", "lulynx.image_gguf.compatibility", "lulynx.image_gguf.source_count",
        "lulynx.dit_family", "lulynx.distill_method", "lulynx.real_objective", "lulynx.few_step_objective",
        "lulynx.sigma_schedule", "lulynx.contract_status", "lulynx.contract_version", "lulynx.checked_at",
        "lulynx.dry_run", "lulynx.network_dim", "lulynx.network_alpha", "lulynx.adapter_type",
        "lulynx.teacher_lora_requested_scope", "lulynx.teacher_lora_load_scope", "lulynx.teacher_lora_text_encoder_status",
        "lulynx.teacher_lora_text_encoder_key_compat", "lulynx.teacher_lora_text_encoder_load_target",
        "lulynx.teacher_lora_text_encoder_2_status", "lulynx.teacher_lora_text_encoder_2_key_compat",
        "lulynx.teacher_lora_text_encoder_2_load_target", "lulynx.smoke_status", "lulynx.loss_initial",
        "lulynx.loss_final", "lulynx.loss_trend", "lulynx.output_finite_ratio", "lulynx.output_tensor_count",
        "lulynx.output_validation_status", "lulynx.output_validation_level", "lulynx.sidecar_steps_valid",
        "lulynx.sidecar_step_count", "lulynx.sidecar_distribution_status", "lulynx.output_validation_checked_at",
        "lulynx.output_validation_note", "lulynx.experimental_quality_gate_status", "lulynx.experimental_quality_gate_level",
        "lulynx.experimental_quality_gate_checked_at", "lulynx.experimental_quality_gate_note",
        "lulynx.experimental_quality_gate_review_count", "lulynx.sample_eval_status", "lulynx.sample_eval_level",
        "lulynx.sample_eval_sample_count", "lulynx.sample_eval_checked_at", "lulynx.sample_eval_note",
        "lulynx.sample_eval_samples_dir", "lulynx.sample_eval_extension_counts", "lulynx.sample_eval_dimension_count",
        "lulynx.validation_level", "lulynx.quality_status", "lulynx.quality_note", "lulynx.teacher_scheduler",
        "lulynx.teacher_steps", "lulynx.student_scheduler", "lulynx.student_steps", "lulynx.guidance_scale",
        "lulynx.seed", "lulynx.recommended_usage",
    )
    compact = {k: manifest.get(k) for k in keys if k in manifest}
    for key in ("tensor_count", "dtypes", "header_error"):
        if key in tensor_info:
            compact[key] = tensor_info[key]
    return compact


def detect_resource_model_type(path: Path, manifest: dict[str, Any]) -> str:
    model_type = str(manifest.get("model_type") or "").lower()
    architecture = str(manifest.get("modelspec.architecture") or manifest.get("architecture") or "").lower()
    text = " ".join([
        path.name, str(path.parent), model_type, architecture, str(manifest.get("model_family") or ""),
        str(manifest.get("artifact_kind") or ""), str(manifest.get("base_model") or ""),
        str(manifest.get("ss_base_model_version") or ""), str(manifest.get("ss_network_module") or ""),
        str(manifest.get("tensor_key_preview") or ""), str(manifest.get("_class_name") or ""),
        str(manifest.get("lulynx.schema_id") or ""), str(manifest.get("lulynx.artifact_kind") or ""),
        str(manifest.get("lulynx.model_family") or ""), str(manifest.get("lulynx.dit_family") or ""),
        str(manifest.get("lulynx.distill_method") or ""), str(manifest.get("lulynx.few_step_objective") or ""),
        str(manifest.get("lulynx.sigma_schedule") or ""), str(manifest.get("lulynx.student_scheduler") or ""),
        str(manifest.get("lulynx.contract_status") or ""), str(manifest.get("lulynx.experimental_quality_gate_status") or ""),
        str(manifest.get("gguf_arch") or ""), str(manifest.get("lulynx.image_gguf.component") or ""),
        str(manifest.get("lulynx.image_gguf.family") or ""), str(manifest.get("lulynx.image_gguf.compatibility") or ""),
    ]).lower()
    if model_type == "image-gguf" or "lulynx_image" in text or "lulynx.image_gguf" in text:
        return "image-gguf"
    if "acceleration_lora" in text or "lcm_lora" in text or "turbo_lora" in text or "few-step-lora" in text or "few_step_lora" in text:
        return "acceleration-lora"
    if "ip-adapter" in text or "ip_adapter" in text or "image_encoder" in text:
        return "ip-adapter"
    if "controlnet" in text or "control_net" in text:
        return "controlnet"
    if "yolo" in text:
        return "yolo"
    if "lora" in text or "lycoris" in text or "loha" in text or "locon" in text:
        return "lora"
    if "unet" in text:
        return "unet"
    if "sdxl" in text or "stable-diffusion-xl" in text:
        return "sdxl"
    if "sd15" in text or "sd-1.5" in text or "stable-diffusion-v1" in text:
        return "sd15"
    if "qwen" in text or "gemma" in text or "llm" in text or path.suffix.lower() == ".gguf":
        return "llm"
    if "jina_clip" in model_type or "jina-clip" in text:
        return "jina-clip"
    if "clip" in text or "open_clip" in text or "text_encoder" in text:
        return "clip"
    return ""


def resource_text(path: Path, manifest: dict[str, Any], model_type: str = "") -> str:
    return " ".join([
        path.name, str(path.parent), model_type, str(manifest.get("model_type") or ""), str(manifest.get("model_family") or ""),
        str(manifest.get("artifact_kind") or ""), str(manifest.get("base_model") or ""), str(manifest.get("ss_base_model_version") or ""),
        str(manifest.get("ss_sd_model_name") or ""), str(manifest.get("modelspec.architecture") or manifest.get("architecture") or ""),
        str(manifest.get("tensor_key_preview") or ""), str(manifest.get("lulynx.schema_id") or ""),
        str(manifest.get("lulynx.artifact_kind") or ""), str(manifest.get("lulynx.model_family") or ""),
        str(manifest.get("lulynx.dit_family") or ""), str(manifest.get("lulynx.distill_method") or ""),
        str(manifest.get("lulynx.few_step_objective") or ""), str(manifest.get("lulynx.sigma_schedule") or ""),
        str(manifest.get("lulynx.student_scheduler") or ""), str(manifest.get("lulynx.contract_status") or ""),
        str(manifest.get("lulynx.experimental_quality_gate_status") or ""),
        str(manifest.get("gguf_arch") or ""), str(manifest.get("lulynx.image_gguf.component") or ""),
        str(manifest.get("lulynx.image_gguf.family") or ""), str(manifest.get("lulynx.image_gguf.compatibility") or ""),
    ]).lower()


def detect_resource_artifact_kind(path: Path, manifest: dict[str, Any], model_type: str) -> str:
    explicit = str(manifest.get("artifact_kind") or manifest.get("lulynx.artifact_kind") or "").strip().lower()
    if explicit:
        return explicit.replace("-", "_")
    if model_type == "image-gguf":
        return "image_gguf"
    text = resource_text(path, manifest, model_type)
    if model_type == "acceleration-lora" or "lcm_lora" in text or "turbo_lora" in text or "few-step-lora" in text:
        return "acceleration_lora"
    if "lora" in text or model_type == "lora":
        return "lora"
    if model_type in {"sdxl", "sd15", "unet"}:
        return "checkpoint"
    if model_type:
        return model_type.replace("-", "_")
    return resource_kind(path)


def detect_resource_model_family(path: Path, manifest: dict[str, Any], model_type: str) -> str:
    explicit = str(
        manifest.get("model_family") or manifest.get("lulynx.model_family")
        or manifest.get("lulynx.image_gguf.family") or manifest.get("lulynx.dit_family") or ""
    ).strip().lower()
    if explicit:
        return explicit
    text = resource_text(path, manifest, model_type)
    if "newbie" in text:
        return "newbie"
    if "anima" in text:
        return "anima"
    if "sdxl" in text or "stable-diffusion-xl" in text or "xl_base" in text:
        return "sdxl"
    if "sd15" in text or "sd-1.5" in text or "stable-diffusion-v1" in text:
        return "sd15"
    if "flux" in text:
        return "flux"
    if "lumina" in text:
        return "lumina"
    if model_type in {"llm", "clip", "jina-clip", "yolo"}:
        return model_type
    return ""


def sha256_for_small_file(path: Path, size: int) -> str:
    if size > RESOURCE_METADATA_MAX_BYTES:
        return ""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
    except Exception:
        return ""
    return digest.hexdigest()


def resource_detection(
    path: Path,
    size: int,
    *,
    include_metadata: bool = True,
    include_hash: bool = False,
    native_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if include_metadata and native_metadata:
        manifest = dict(native_metadata.get("manifest") or {})
        sidecar_source = str(native_metadata.get("sidecar_source") or "")
        safetensors_metadata = dict(native_metadata.get("safetensors_metadata") or {})
        tensor_keys = [str(key) for key in list(native_metadata.get("tensor_keys") or [])]
        tensor_info = dict(native_metadata.get("tensor_info") or {})
    else:
        manifest, sidecar_source = read_json_sidecar(path) if include_metadata else ({}, "")
        safetensors_metadata, tensor_keys, tensor_info = read_safetensors_header(path) if include_metadata else ({}, [], {})
    image_gguf_manifest, image_gguf_source = read_image_gguf_sidecar(path) if include_metadata else ({}, "")
    if image_gguf_manifest:
        manifest = {**manifest, **image_gguf_manifest}
        sidecar_source = image_gguf_source or sidecar_source
    if safetensors_metadata:
        manifest = {**manifest, **safetensors_metadata}
    if tensor_keys:
        manifest["tensor_key_preview"] = " ".join(tensor_keys[:24])
    model_type = detect_resource_model_type(path, manifest)
    artifact_kind = detect_resource_artifact_kind(path, manifest, model_type)
    model_family = detect_resource_model_family(path, manifest, model_type)
    if artifact_kind:
        manifest.setdefault("artifact_kind", artifact_kind)
    if model_family:
        manifest.setdefault("model_family", model_family)
    detection_source = sidecar_source or ("safetensors_metadata" if safetensors_metadata else "manifest" if manifest else "filename")
    sha256 = sha256_for_small_file(path, size) if include_hash else ""
    hash_status = "complete" if sha256 else "deferred_on_demand" if not include_hash else "unavailable" if size <= RESOURCE_METADATA_MAX_BYTES else "deferred_large_file"
    tags = [
        tag for tag in [
            model_type, artifact_kind, model_family, "image_gguf" if model_type == "image-gguf" else resource_kind(path),
            path.suffix.lower().lstrip("."),
            str(manifest.get("ss_base_model_version") or "").lower(), str(manifest.get("modelspec.architecture") or "").lower(),
        ] if tag
    ]
    return {
        "model_type": model_type,
        "artifact_kind": artifact_kind,
        "model_family": model_family,
        "detection_source": detection_source,
        "sha256": sha256,
        "hash_status": hash_status,
        "manifest": compact_manifest(manifest, tensor_info),
        "tags": sorted(set(tags)),
    }


def scan_local_resources_cached(
    *,
    project_root: Path,
    backend_root: Path,
    max_items: int = 240,
    summary_only: bool = False,
    include_hash: bool = False,
    refresh: bool = False,
    artifact_provider: ArtifactProvider | None = None,
) -> dict[str, Any]:
    roots = resource_roots(project_root=project_root, backend_root=backend_root)
    key = (tuple(str(root) for root in roots), int(max_items), bool(summary_only), bool(include_hash))
    now = time.monotonic()
    cached = _LOCAL_RESOURCE_CACHE.get(key)
    if not refresh and cached and now - cached[0] <= LOCAL_RESOURCE_CACHE_TTL_SEC:
        return {**cached[1], "cached": True, "cache_age_sec": round(now - cached[0], 2)}
    result = scan_local_resources(roots=roots, max_items=max_items, summary_only=summary_only, include_hash=include_hash, artifact_provider=artifact_provider)
    _LOCAL_RESOURCE_CACHE[key] = (now, result)
    return {**result, "cached": False, "cache_age_sec": 0}


def scan_local_resources(
    *,
    roots: list[Path],
    max_items: int = 240,
    summary_only: bool = False,
    include_hash: bool = False,
    artifact_provider: ArtifactProvider | None = None,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    items: list[dict[str, Any]] = []
    skipped = 0
    total_size = 0
    truncated = False
    native_candidates = scan_local_resource_candidates_native(roots, max_items, include_metadata=not summary_only)
    scanner_mode = "native" if native_candidates is not None else "python"
    if native_candidates is not None:
        total_size = int(native_candidates.get("total_size", 0) or 0)
        skipped += int(native_candidates.get("skipped", 0) or 0)
        truncated = bool(native_candidates.get("truncated", False))
        for candidate in native_candidates.get("items", []) or []:
            if len(items) >= max_items:
                truncated = True
                break
            try:
                item = build_local_resource_item_from_candidate(
                    candidate,
                    summary_only=summary_only,
                    include_hash=include_hash,
                )
                if item is not None:
                    items.append(item)
            except Exception:
                skipped += 1
    else:
        for root in roots:
            if len(items) >= max_items:
                truncated = True
                break
            try:
                for path in root.rglob("*"):
                    if len(items) >= max_items:
                        truncated = True
                        break
                    try:
                        if not path.is_file():
                            continue
                        suffix = path.suffix.lower()
                        if suffix not in RESOURCE_MODEL_SUFFIXES and suffix not in RESOURCE_CONFIG_SUFFIXES:
                            continue
                        stat = path.stat()
                        size = int(stat.st_size)
                        total_size += size
                        item = build_local_resource_item(path, root, size, stat.st_mtime, summary_only=summary_only, include_hash=include_hash)
                        items.append(item)
                    except Exception:
                        skipped += 1
            except Exception as exc:
                logger.debug("resource scan skipped root %s: %s", root, exc)
                skipped += 1
    items.sort(key=lambda item: (str(item.get("category")), -int(item.get("size") or 0), str(item.get("name"))))
    if not summary_only and len(items) < max_items and artifact_provider is not None:
        try:
            items.extend(artifact_provider(max_items - len(items)))
        except Exception:
            logger.debug("resource scan skipped job artifacts", exc_info=True)
            skipped += 1
    items.sort(key=lambda item: (str(item.get("category")), str(item.get("source") or "local"), -int(item.get("size") or 0), str(item.get("name"))))
    counts = Counter(str(item.get("category") or "models") for item in items)
    duration_ms = round((time.perf_counter() - started_at) * 1000, 2)
    return {
        "roots": [str(root) for root in roots],
        "items": items,
        "counts": dict(counts),
        "total_size": total_size,
        "skipped": skipped,
        "truncated": truncated,
        "scan_stats": {
            "duration_ms": duration_ms,
            "mode": scanner_mode,
            "summary_only": bool(summary_only),
            "include_hash": bool(include_hash) and not summary_only,
            "metadata": "skipped" if summary_only else "enabled",
            "roots": len(roots),
            "items": len(items),
            "max_items": int(max_items),
        },
    }


def scan_local_resource_candidates_native(roots: list[Path], max_items: int, *, include_metadata: bool = True) -> dict[str, Any] | None:
    native = native_local_resource_scan_api()
    if native is None:
        return None
    try:
        result = native.scan_local_resource_candidates(
            json.dumps([str(root) for root in roots], ensure_ascii=False),
            int(max_items),
            bool(include_metadata),
        )
    except Exception:
        return None
    return result if isinstance(result, dict) else None


def build_local_resource_item(
    path: Path,
    root: Path,
    size: int,
    modified_epoch: float,
    *,
    summary_only: bool,
    include_hash: bool,
    native_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata_enabled = not summary_only
    hash_enabled = bool(include_hash) and not summary_only
    detected = resource_detection(
        path,
        size,
        include_metadata=metadata_enabled,
        include_hash=hash_enabled,
        native_metadata=native_metadata if metadata_enabled else None,
    )
    item = {
        "name": path.name,
        "path": str(path),
        "relative_path": str(path.relative_to(root)),
        "root": str(root),
        "category": resource_category(path, root, str(detected["model_type"] or "")),
        "kind": "image_gguf" if detected["model_type"] == "image-gguf" else resource_kind(path),
        "model_type": detected["model_type"],
        "artifact_kind": detected["artifact_kind"],
        "model_family": detected["model_family"],
        "detection_source": detected["detection_source"],
        "sha256": detected["sha256"],
        "hash_status": detected["hash_status"],
        "manifest": detected["manifest"],
        "tags": detected["tags"],
        "size": size,
        "modified_at": datetime.fromtimestamp(modified_epoch).isoformat(timespec="seconds"),
    }
    if summary_only:
        item["manifest"] = {}
        item["sha256"] = ""
    return item


def build_local_resource_item_from_candidate(
    candidate: dict[str, Any],
    *,
    summary_only: bool,
    include_hash: bool,
) -> dict[str, Any] | None:
    path = Path(str(candidate.get("path", "") or ""))
    root = Path(str(candidate.get("root", "") or ""))
    if not path.is_file() or not root.exists():
        return None
    size = int(candidate.get("size", 0) or 0)
    modified_epoch = float(candidate.get("modified_epoch_sec", 0) or 0)
    native_metadata = candidate.get("native_metadata") if isinstance(candidate.get("native_metadata"), dict) else None
    return build_local_resource_item(
        path,
        root,
        size,
        modified_epoch,
        summary_only=summary_only,
        include_hash=include_hash,
        native_metadata=native_metadata,
    )


# Backwards-compatible private names used by older tests/patch points.
_load_native_local_resource_scan_api = load_native_local_resource_scan_api
_native_local_resource_scan_api = native_local_resource_scan_api
_native_local_resource_scan_disabled = native_local_resource_scan_disabled
