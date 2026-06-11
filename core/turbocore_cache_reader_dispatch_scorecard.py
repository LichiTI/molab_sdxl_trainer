"""Pre-promotion scorecard for native cache reader dispatch."""

from __future__ import annotations

import os
from typing import Any, Dict, Mapping, Sequence

from core.turbocore_cache_reader_dispatch_matrix import build_cache_reader_dispatch_fallback_matrix
from core.turbocore_cache_reader_native_batch_dispatch import run_native_cache_reader_batch_dispatch
from core.turbocore_cache_reader_training_gate import (
    BATCH_CPU_PAYLOAD_BUFFER_BYTES_ENV,
    BATCH_DISPATCH_CONTRACT_ENV,
    BATCH_HANDOFF_SESSION_ENV,
    DISABLE_EXPERIMENTAL_ENV,
    DISPATCH_STRICT_FALLBACK_ENV,
    ENABLE_EXPERIMENTAL_ENV,
    PARITY_BATCHES_ENV,
    PARITY_MAX_BYTES_ENV,
    TEXT_PAYLOAD_BUFFER_BYTES_ENV,
    TEXT_PAYLOAD_PARITY_ENV,
    run_cache_reader_training_experimental_gate,
)


def _set_env(values: dict[str, str | None]) -> dict[str, str | None]:
    old = {key: os.environ.get(key) for key in values}
    for key, value in values.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    return old


def _restore_env(old: dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _closed_flags(value: dict[str, Any]) -> dict[str, bool]:
    return {
        "native_dispatch_eligible": bool(value.get("native_dispatch_eligible", False)),
        "would_allow_native_dispatch": bool(value.get("would_allow_native_dispatch", False)),
        "fallback_to_python_batch": bool(value.get("fallback_to_python_batch", True)),
        "returns_tensor_payloads": bool(value.get("returns_tensor_payloads", False)),
        "cache_reader_path_enabled": bool(value.get("cache_reader_path_enabled", False)),
        "prefetch_queue_training_path_enabled": bool(value.get("prefetch_queue_training_path_enabled", False)),
        "training_path_enabled": bool(value.get("training_path_enabled", False)),
    }


def _compact_probe(report: dict[str, Any]) -> dict[str, Any]:
    batch_contract = dict(report.get("batch_dispatch_contract", {}) or {})
    text_payload = dict(report.get("text_payload_parity", {}) or {})
    native_probe = dict(report.get("native_probe", {}) or {})
    native_batch = dict(native_probe.get("native_latent_batch_summary", {}) or {})
    blockers = [str(item) for item in list(report.get("native_dispatch_blockers", []) or [])]
    return {
        "schema_version": 1,
        "provider": "native_cache_reader_dispatch_scorecard_probe_v1",
        "ok": bool(report.get("ok", False)),
        "dataset_class": str(report.get("dataset_class") or ""),
        "sample_count": int(report.get("sample_count", 0) or 0),
        "batch_size": int(report.get("batch_size", 0) or 0),
        "training_experimental_allowed": bool(report.get("training_experimental_allowed", False)),
        "parity_guard_passed": bool(report.get("parity_guard_passed", False)),
        "batch_parity_guard_passed": bool(report.get("batch_parity_guard_passed", False)),
        "batch_payload_parity_guard_passed": bool(report.get("batch_payload_parity_guard_passed", False)),
        "torch_owned_tensor_handoff_guard_passed": bool(report.get("torch_owned_tensor_handoff_guard_passed", False)),
        "batch_handoff_session_shadow_passed": bool(report.get("batch_handoff_session_shadow_passed", False)),
        "batch_dispatch_contract_ready": bool(report.get("batch_dispatch_contract_ready", False)),
        "text_payload_parity_guard_ran": bool(report.get("text_payload_parity_guard_ran", False)),
        "text_payload_parity_guard_passed": bool(report.get("text_payload_parity_guard_passed", False)),
        "text_payload_fields": [str(item) for item in list(text_payload.get("text_payload_fields", []) or [])],
        "tensor_parity_count": int(report.get("tensor_parity_count", 0) or 0),
        "tensor_parity_matches": int(report.get("tensor_parity_matches", 0) or 0),
        "native_data_payload_bytes_read": int(report.get("native_data_payload_bytes_read", 0) or 0),
        "python_data_payload_bytes_read": int(report.get("python_data_payload_bytes_read", 0) or 0),
        "native_batch_summary_provider": str(native_batch.get("provider") or ""),
        "native_batch_materialization_contract": bool(native_batch.get("native_batch_materialization_contract", False)),
        "materialization_contract_supported": bool(native_batch.get("materialization_contract_supported", False)),
        "dispatch_contract_ready": bool(batch_contract.get("dispatch_contract_ready", False)),
        "batch_handle_count": int(batch_contract.get("batch_handle_count", 0) or 0),
        "native_dispatch_blockers": blockers,
        **_closed_flags(report),
    }


def _run_supported_shadow_probe(
    dataset: Any,
    *,
    batch_size: int,
    parity_batches: int,
    prefetch_factor: int | None,
    max_decode_payload_bytes: int,
    max_batch_cpu_payload_buffer_bytes: int,
    max_text_payload_buffer_bytes: int,
    strict_fallback: bool,
) -> dict[str, Any]:
    env = _set_env(
        {
            ENABLE_EXPERIMENTAL_ENV: "1",
            DISABLE_EXPERIMENTAL_ENV: None,
            PARITY_BATCHES_ENV: str(parity_batches),
            PARITY_MAX_BYTES_ENV: str(max_decode_payload_bytes),
            BATCH_CPU_PAYLOAD_BUFFER_BYTES_ENV: str(max_batch_cpu_payload_buffer_bytes),
            BATCH_HANDOFF_SESSION_ENV: "1",
            BATCH_DISPATCH_CONTRACT_ENV: "1",
            TEXT_PAYLOAD_PARITY_ENV: "1",
            TEXT_PAYLOAD_BUFFER_BYTES_ENV: str(max_text_payload_buffer_bytes),
            DISPATCH_STRICT_FALLBACK_ENV: "1" if strict_fallback else "0",
        }
    )
    try:
        gate_report = run_cache_reader_training_experimental_gate(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            prefetch_factor=prefetch_factor,
            max_parity_batches=parity_batches,
            max_decode_payload_bytes=max_decode_payload_bytes,
            max_batch_cpu_payload_buffer_bytes=max_batch_cpu_payload_buffer_bytes,
            max_text_payload_buffer_bytes=max_text_payload_buffer_bytes,
            enable_batch_handoff_session_shadow=True,
            enable_batch_dispatch_contract_shadow=True,
            enable_text_payload_parity_shadow=True,
        )
    except Exception as exc:
        gate_report = {
            "schema_version": 1,
            "provider": "native_cache_reader_training_gate",
            "ok": False,
            "reason": "scorecard_training_gate_failed",
            "native_error": f"{type(exc).__name__}: {exc}",
            "training_path_enabled": False,
        }
    finally:
        _restore_env(env)
    return _compact_probe(gate_report)


def _required_probe_flags(probe: dict[str, Any]) -> dict[str, bool]:
    return {
        "parity_guard_passed": bool(probe.get("parity_guard_passed", False)),
        "batch_parity_guard_passed": bool(probe.get("batch_parity_guard_passed", False)),
        "batch_payload_parity_guard_passed": bool(probe.get("batch_payload_parity_guard_passed", False)),
        "torch_owned_tensor_handoff_guard_passed": bool(probe.get("torch_owned_tensor_handoff_guard_passed", False)),
        "batch_handoff_session_shadow_passed": bool(probe.get("batch_handoff_session_shadow_passed", False)),
        "batch_dispatch_contract_ready": bool(probe.get("batch_dispatch_contract_ready", False)),
        "text_payload_parity_guard_passed": bool(probe.get("text_payload_parity_guard_passed", False)),
    }


def _failed_probe_checks(probe: dict[str, Any]) -> list[str]:
    return [key for key, passed in _required_probe_flags(probe).items() if not passed]


def _supported_probe_case(case_id: str, dataset: Any, probe: dict[str, Any]) -> dict[str, Any]:
    failed = _failed_probe_checks(probe)
    return {
        "schema_version": 1,
        "case_id": str(case_id),
        "dataset_class": type(dataset).__name__,
        "supported_shadow_probe_passed": not failed,
        "failed_probe_checks": failed,
        "required_probe_flags": _required_probe_flags(probe),
        "probe": probe,
        **_closed_flags({"fallback_to_python_batch": True}),
    }


def _run_training_dispatch_case(
    case_id: str,
    dataset: Any,
    *,
    batch_size: int,
    max_decode_payload_bytes: int,
    max_batch_cpu_payload_buffer_bytes: int,
) -> dict[str, Any]:
    sample_count = len(list(getattr(dataset, "samples", []) or []))
    indices = list(range(min(max(int(batch_size or 1), 1), sample_count)))
    report = run_native_cache_reader_batch_dispatch(
        dataset,
        sample_indices=indices,
        max_decode_payload_bytes=max_decode_payload_bytes,
        max_batch_cpu_payload_buffer_bytes=max_batch_cpu_payload_buffer_bytes,
    )
    return {
        "schema_version": 1,
        "case_id": str(case_id),
        "dataset_class": type(dataset).__name__,
        "representative_training_dispatch_passed": bool(report.get("ok", False))
        and bool(report.get("training_path_enabled", False))
        and bool(report.get("returns_tensor_payloads", False)),
        "dispatch": _compact_training_dispatch(report),
        **_closed_flags(report),
    }


def _compact_training_dispatch(report: Mapping[str, Any]) -> dict[str, Any]:
    parity = dict(report.get("parity", {}) or {})
    native_probe = dict(report.get("native_probe", {}) or {})
    return {
        "ok": bool(report.get("ok", False)),
        "provider": str(report.get("provider") or ""),
        "sample_count": int(report.get("sample_count", 0) or 0),
        "native_runtime": bool(report.get("native_runtime", False)),
        "training_dispatch": bool(report.get("training_dispatch", False)),
        "training_path_enabled": bool(report.get("training_path_enabled", False)),
        "native_dispatch_eligible": bool(report.get("native_dispatch_eligible", False)),
        "would_allow_native_dispatch": bool(report.get("would_allow_native_dispatch", False)),
        "fallback_to_python_batch": bool(report.get("fallback_to_python_batch", True)),
        "returns_tensor_payloads": bool(report.get("returns_tensor_payloads", False)),
        "cache_reader_path_enabled": bool(report.get("cache_reader_path_enabled", False)),
        "parity_ok": bool(parity.get("ok", False)),
        "parity_max_abs_diff": parity.get("max_abs_diff"),
        "native_probe": native_probe,
        "blocked_reasons": [str(item) for item in list(report.get("blocked_reasons", []) or [])],
    }


def _run_supported_case(
    case_id: str,
    dataset: Any,
    *,
    batch_size: int,
    parity_batches: int,
    prefetch_factor: int | None,
    max_decode_payload_bytes: int,
    max_batch_cpu_payload_buffer_bytes: int,
    max_text_payload_buffer_bytes: int,
    strict_fallback: bool,
) -> dict[str, Any]:
    probe = _run_supported_shadow_probe(
        dataset,
        batch_size=max(int(batch_size or 1), 1),
        parity_batches=max(int(parity_batches or 1), 1),
        prefetch_factor=prefetch_factor,
        max_decode_payload_bytes=max(int(max_decode_payload_bytes or 1), 1),
        max_batch_cpu_payload_buffer_bytes=max(int(max_batch_cpu_payload_buffer_bytes or 1), 1),
        max_text_payload_buffer_bytes=max(int(max_text_payload_buffer_bytes or 1), 1),
        strict_fallback=strict_fallback,
    )
    return _supported_probe_case(case_id, dataset, probe)


def _int_from_case(case: Mapping[str, Any], key: str, default: int) -> int:
    try:
        return max(int(case.get(key, default) or default), 1)
    except (TypeError, ValueError):
        return max(int(default or 1), 1)


def build_cache_reader_dispatch_promotion_scorecard(
    dataset: Any,
    *,
    additional_supported_datasets: Sequence[tuple[str, Any]] | None = None,
    additional_supported_cases: Sequence[Mapping[str, Any]] | None = None,
    batch_size: int = 1,
    parity_batches: int = 1,
    prefetch_factor: int | None = 2,
    max_decode_payload_bytes: int = 16 * 1024 * 1024,
    max_batch_cpu_payload_buffer_bytes: int = 4 * 1024 * 1024,
    max_text_payload_buffer_bytes: int = 4 * 1024 * 1024,
    strict_fallback: bool = True,
) -> Dict[str, Any]:
    """Run the no-dispatch matrix and one supported native shadow probe.

    The scorecard is intentionally pre-promotion: a passing scorecard proves the
    debug evidence is coherent and fallback is strict, not that training dispatch
    may be enabled.
    """
    resolved_batch_size = max(int(batch_size or 1), 1)
    resolved_parity_batches = max(int(parity_batches or 1), 1)
    resolved_decode_bytes = max(int(max_decode_payload_bytes or 1), 1)
    resolved_batch_bytes = max(int(max_batch_cpu_payload_buffer_bytes or 1), 1)
    resolved_text_bytes = max(int(max_text_payload_buffer_bytes or 1), 1)
    matrix = build_cache_reader_dispatch_fallback_matrix(
        dataset,
        batch_size=resolved_batch_size,
        prefetch_factor=prefetch_factor,
        strict_fallback=strict_fallback,
    )
    probe = _run_supported_shadow_probe(
        dataset,
        batch_size=resolved_batch_size,
        parity_batches=resolved_parity_batches,
        prefetch_factor=prefetch_factor,
        max_decode_payload_bytes=resolved_decode_bytes,
        max_batch_cpu_payload_buffer_bytes=resolved_batch_bytes,
        max_text_payload_buffer_bytes=resolved_text_bytes,
        strict_fallback=strict_fallback,
    )
    supported_probe_cases = [_supported_probe_case("primary_supported_single_worker", dataset, probe)]
    for case_id, case_dataset in list(additional_supported_datasets or []):
        case_probe = _run_supported_shadow_probe(
            case_dataset,
            batch_size=resolved_batch_size,
            parity_batches=resolved_parity_batches,
            prefetch_factor=prefetch_factor,
            max_decode_payload_bytes=resolved_decode_bytes,
            max_batch_cpu_payload_buffer_bytes=resolved_batch_bytes,
            max_text_payload_buffer_bytes=resolved_text_bytes,
            strict_fallback=strict_fallback,
        )
        supported_probe_cases.append(_supported_probe_case(str(case_id), case_dataset, case_probe))
    for index, case in enumerate(list(additional_supported_cases or []), start=1):
        case_dataset = case.get("dataset")
        case_id = str(case.get("case_id") or f"supported_case_{index}")
        if case_dataset is None:
            supported_probe_cases.append(
                {
                    "schema_version": 1,
                    "case_id": case_id,
                    "dataset_class": "",
                    "supported_shadow_probe_passed": False,
                    "failed_probe_checks": ["supported_case_dataset_missing"],
                    "required_probe_flags": {},
                    "probe": {"ok": False, "native_dispatch_blockers": ["supported_case_dataset_missing"]},
                    **_closed_flags({"fallback_to_python_batch": True}),
                }
            )
            continue
        supported_probe_cases.append(
            _run_supported_case(
                case_id,
                case_dataset,
                batch_size=_int_from_case(case, "batch_size", resolved_batch_size),
                parity_batches=_int_from_case(case, "parity_batches", resolved_parity_batches),
                prefetch_factor=case.get("prefetch_factor", prefetch_factor),
                max_decode_payload_bytes=_int_from_case(case, "max_decode_payload_bytes", resolved_decode_bytes),
                max_batch_cpu_payload_buffer_bytes=_int_from_case(
                    case,
                    "max_batch_cpu_payload_buffer_bytes",
                    resolved_batch_bytes,
                ),
                max_text_payload_buffer_bytes=_int_from_case(
                    case,
                    "max_text_payload_buffer_bytes",
                    resolved_text_bytes,
                ),
                strict_fallback=bool(case.get("strict_fallback", strict_fallback)),
            )
        )
    required_probe_flags = _required_probe_flags(probe)
    failed_probe_checks = _failed_probe_checks(probe)
    supported_probe_matrix_passed = all(bool(case.get("supported_shadow_probe_passed", False)) for case in supported_probe_cases)
    training_dispatch_cases = [
        _run_training_dispatch_case(
            "primary_supported_single_worker",
            dataset,
            batch_size=resolved_batch_size,
            max_decode_payload_bytes=resolved_decode_bytes,
            max_batch_cpu_payload_buffer_bytes=resolved_batch_bytes,
        )
    ]
    for case_id, case_dataset in list(additional_supported_datasets or []):
        training_dispatch_cases.append(
            _run_training_dispatch_case(
                str(case_id),
                case_dataset,
                batch_size=resolved_batch_size,
                max_decode_payload_bytes=resolved_decode_bytes,
                max_batch_cpu_payload_buffer_bytes=resolved_batch_bytes,
            )
        )
    representative_training_matrix_passed = bool(training_dispatch_cases) and all(
        bool(case.get("representative_training_dispatch_passed", False)) for case in training_dispatch_cases
    )
    promotion_blockers = [] if representative_training_matrix_passed else [
        "native_cache_reader_training_dispatch_not_implemented",
        "python_dataloader_batch_remains_authoritative",
        "cache_reader_dispatch_route_not_promoted",
        "representative_training_matrix_not_passed",
    ]
    if failed_probe_checks or not supported_probe_matrix_passed:
        promotion_blockers.append("supported_shadow_probe_failed")
    if not bool(matrix.get("ok", False)):
        promotion_blockers.append("strict_fallback_matrix_failed")
    return {
        "schema_version": 1,
        "provider": "native_cache_reader_dispatch_promotion_scorecard_v1",
        "ok": bool(matrix.get("ok", False)) and supported_probe_matrix_passed,
        "debug_only": True,
        "shadow_run": True,
        "dataset_class": type(dataset).__name__,
        "batch_size": resolved_batch_size,
        "parity_batches": resolved_parity_batches,
        "strict_fallback": bool(strict_fallback),
        "strict_fallback_matrix_passed": bool(matrix.get("strict_fallback_matrix_passed", False)),
        "supported_shadow_probe_passed": supported_probe_matrix_passed,
        "failed_probe_checks": failed_probe_checks,
        "required_probe_flags": required_probe_flags,
        "supported_probe_matrix_passed": supported_probe_matrix_passed,
        "supported_probe_matrix_case_count": len(supported_probe_cases),
        "supported_probe_matrix_case_ids": [str(case.get("case_id") or "") for case in supported_probe_cases],
        "representative_supported_probe_matrix_passed": supported_probe_matrix_passed and len(supported_probe_cases) >= 4,
        "supported_probe_matrix": supported_probe_cases,
        "promotion_ready": bool(matrix.get("ok", False)) and supported_probe_matrix_passed and representative_training_matrix_passed,
        "promotion_blockers": promotion_blockers,
        "representative_fallback_matrix_passed": bool(matrix.get("representative_fallback_matrix_passed", False)),
        "representative_training_matrix_passed": representative_training_matrix_passed,
        "representative_training_matrix": training_dispatch_cases,
        "fallback_matrix": matrix,
        "supported_shadow_probe": probe,
        **_closed_flags({
            "native_dispatch_eligible": representative_training_matrix_passed,
            "would_allow_native_dispatch": representative_training_matrix_passed,
            "fallback_to_python_batch": not representative_training_matrix_passed,
            "returns_tensor_payloads": representative_training_matrix_passed,
            "cache_reader_path_enabled": representative_training_matrix_passed,
            "training_path_enabled": representative_training_matrix_passed,
        }),
    }


__all__ = ["build_cache_reader_dispatch_promotion_scorecard"]
