"""Curated evidence packs for bubble-aware runtime benchmarks."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Iterable, Mapping

from .bubble_natural_load_canary import build_bubble_natural_load_canary_report
from .bubble_natural_data_wait_ab_evidence import NATURAL_DATA_WAIT_AB_EVIDENCE_REPORT
from .bubble_natural_data_wait_evidence import NATURAL_DATA_WAIT_EVIDENCE_REPORT
from .bubble_runtime_closed_loop_evidence import (
    CLOSED_LOOP_EVIDENCE_REPORT,
    build_bubble_closed_loop_evidence_report,
)
from .bubble_runtime_evidence_pack_canary_gate import gate_natural_claims, natural_load_canary_review_items
from .bubble_runtime_followup_plan import build_bubble_runtime_followup_plan
from .bubble_runtime_release_claims import build_release_claim_report


PACK_REPORT = "bubble_runtime_benchmark_evidence_pack_v0"
CUDA_DEBUG_REPEAT_REPORT = "bubble_sdxl_batch2_cuda_debug_repeat_followup_v0"
CLAIMABLE_REPORTS = {
    "gpu_bubble_experiment_report_v0",
    "bubble_advisor_ab_evidence_v0",
    "bubble_runtime_closed_loop_evidence_v0",
    NATURAL_DATA_WAIT_EVIDENCE_REPORT,
    NATURAL_DATA_WAIT_AB_EVIDENCE_REPORT,
}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value if value is not None else default)
    except (TypeError, ValueError, OverflowError):
        return float(default)


def _string_list(value: Any) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    return [str(item) for item in value if item is not None]


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _json_sha1(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _payload_json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, default=str)


def _payload_bytes(payload: Mapping[str, Any]) -> bytes:
    return _payload_json_text(payload).encode("utf-8")


def _payload_sha1(payload: Mapping[str, Any]) -> str:
    return hashlib.sha1(_payload_bytes(payload)).hexdigest()


def _curated_source_payload(kind: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
    if kind != CUDA_DEBUG_REPEAT_REPORT:
        return payload
    curated = dict(payload)
    curated.update(
        {
            "publishable": False,
            "release_claim_allowed": False,
            "diagnostic_only": True,
            "case_specific_only": True,
            "non_publishable_reason": "cuda_debug_repeat_diagnostic_only",
        }
    )
    return curated


def _stable_file_name(path: Path, kind: str, digest: str) -> str:
    stem = path.stem.replace(" ", "_")
    stem = "".join(char if char.isalnum() or char in {"_", "-", "."} else "_" for char in stem)
    stem = stem[:80].strip("._-") or "evidence"
    return f"{kind}_{stem}_{digest[:10]}.json"


def _is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _is_nested_generated_pack_path(path: Path, scan_root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(scan_root.resolve())
    except ValueError:
        return False
    return "evidence_pack" in relative.parts[:-1]


def _iter_json_files(paths: Iterable[Path], *, exclude_dirs: Iterable[Path] = ()) -> list[Path]:
    excluded = [Path(path) for path in exclude_dirs]
    files: list[Path] = []
    for path in paths:
        target = Path(path)
        if target.is_dir():
            files.extend(
                sorted(
                    item
                    for item in target.rglob("*.json")
                    if item.is_file() and not any(_is_under(item, excluded_dir) for excluded_dir in excluded)
                    and not _is_nested_generated_pack_path(item, target)
                )
            )
        elif target.is_file():
            if not any(_is_under(target, excluded_dir) for excluded_dir in excluded):
                files.append(target)
    seen: set[str] = set()
    unique: list[Path] = []
    for path in files:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    return unique


def _report_kind(payload: Mapping[str, Any]) -> str:
    report = str(payload.get("report") or "")
    controller = str(payload.get("controller") or "")
    if report == CUDA_DEBUG_REPEAT_REPORT:
        if "decision" not in payload and "comparisons" not in payload:
            return ""
        return CUDA_DEBUG_REPEAT_REPORT
    if report in CLAIMABLE_REPORTS:
        return report
    if controller == "bubble_aware_runtime_controller_v0":
        return "bubble_aware_runtime_controller_v0"
    if (
        "case_count" in payload
        and "cases" in payload
        and ("match_rate" in payload or "patchable_rate" in payload)
    ):
        return "bubble_runtime_replay_report_v0"
    return ""


def _derived_report_payloads(payload: Mapping[str, Any]) -> list[tuple[str, Mapping[str, Any]]]:
    """Create compact evidence reports from richer source artifacts."""

    derived: list[tuple[str, Mapping[str, Any]]] = []
    extra = _mapping(payload.get("extra"))
    closed_loop = _mapping(_mapping(extra.get("bubble_controller")).get("closed_loop"))
    executor = _mapping(closed_loop.get("executor"))
    has_closed_loop_state = bool(_mapping(extra.get("bubble_closed_loop_state")))
    has_adapter_boundary = bool(_mapping(executor.get("runtime_adapter"))) or bool(executor.get("blocked_reasons"))
    if (payload.get("manifest_version") or extra) and (has_closed_loop_state or has_adapter_boundary):
        report = build_bubble_closed_loop_evidence_report(payload)
        if report.get("action_count") or str(report.get("status") or "") != "no_action":
            derived.append((CLOSED_LOOP_EVIDENCE_REPORT, report))
    return derived


def _case_id(kind: str, payload: Mapping[str, Any]) -> str:
    if kind == "gpu_bubble_experiment_report_v0":
        benchmark = _mapping(payload.get("benchmark"))
        return str(
            benchmark.get("case")
            or benchmark.get("case_id")
            or benchmark.get("name")
            or benchmark.get("family")
            or "gpu_bubble_experiment"
        )
    if kind == "bubble_advisor_ab_evidence_v0":
        matrix_case = _mapping(payload.get("matrix_case"))
        if matrix_case.get("case_id"):
            return str(matrix_case.get("case_id"))
        after = _mapping(payload.get("after"))
        before = _mapping(payload.get("before"))
        return str(after.get("case_id") or before.get("case_id") or "bubble_advisor_ab")
    if kind == "bubble_runtime_closed_loop_evidence_v0":
        return str(payload.get("case_id") or "bubble_closed_loop")
    if kind == NATURAL_DATA_WAIT_EVIDENCE_REPORT:
        return str(payload.get("case_id") or "bubble_natural_data_wait")
    if kind == NATURAL_DATA_WAIT_AB_EVIDENCE_REPORT:
        return str(payload.get("case_id") or "bubble_natural_data_wait_ab")
    if kind == CUDA_DEBUG_REPEAT_REPORT:
        return str(payload.get("case_id") or "sdxl_batch2_cuda_debug_repeat")
    if kind == "bubble_aware_runtime_controller_v0":
        diagnosis = _mapping(payload.get("diagnosis"))
        snapshot = _mapping(payload.get("snapshot"))
        config = _mapping(snapshot.get("config"))
        return str(config.get("family") or diagnosis.get("kind") or "bubble_controller")
    return str(payload.get("case_id") or kind or "unknown")


def _family(kind: str, payload: Mapping[str, Any]) -> str:
    if kind == "gpu_bubble_experiment_report_v0":
        benchmark = _mapping(payload.get("benchmark"))
        return str(benchmark.get("family") or benchmark.get("model_family") or "")
    if kind == "bubble_advisor_ab_evidence_v0":
        matrix_case = _mapping(payload.get("matrix_case"))
        if matrix_case.get("family"):
            return str(matrix_case.get("family"))
        after = _mapping(payload.get("after"))
        before = _mapping(payload.get("before"))
        return str(after.get("family") or before.get("family") or "")
    if kind == "bubble_runtime_closed_loop_evidence_v0":
        return str(payload.get("family") or "")
    if kind == NATURAL_DATA_WAIT_EVIDENCE_REPORT:
        return str(payload.get("family") or "")
    if kind == NATURAL_DATA_WAIT_AB_EVIDENCE_REPORT:
        return str(payload.get("family") or "")
    if kind == CUDA_DEBUG_REPEAT_REPORT:
        return str(payload.get("family") or "sdxl")
    if kind == "bubble_aware_runtime_controller_v0":
        snapshot = _mapping(payload.get("snapshot"))
        config = _mapping(snapshot.get("config"))
        return str(config.get("family") or "")
    return ""


def _status(kind: str, payload: Mapping[str, Any]) -> str:
    if kind == "gpu_bubble_experiment_report_v0":
        return str(_mapping(payload.get("classification")).get("status") or payload.get("status") or "")
    if kind == "bubble_aware_runtime_controller_v0":
        return str(_mapping(payload.get("diagnosis")).get("kind") or payload.get("status") or "")
    return str(payload.get("status") or "")


def _index_item(
    path: Path,
    payload: Mapping[str, Any],
    kind: str,
    *,
    copy_path: str = "",
    sha1: str | None = None,
    bytes_size: int | None = None,
) -> dict[str, Any]:
    digest = sha1 or _json_sha1(path)
    item = {
        "kind": kind,
        "source_path": str(path),
        "copy_path": copy_path,
        "sha1": digest,
        "bytes": int(bytes_size if bytes_size is not None else path.stat().st_size),
        "case_id": _case_id(kind, payload),
        "family": _family(kind, payload),
        "status": _status(kind, payload),
    }
    if kind == CUDA_DEBUG_REPEAT_REPORT:
        item.update(
            {
                "publishable": False,
                "release_claim_allowed": False,
                "diagnostic_only": True,
                "case_specific_only": True,
                "non_publishable_reason": "cuda_debug_repeat_diagnostic_only",
            }
        )
    return item


def _claimable_payload(kind: str, payload: Mapping[str, Any]) -> bool:
    if kind == CUDA_DEBUG_REPEAT_REPORT:
        return False
    return kind in CLAIMABLE_REPORTS or kind == "bubble_aware_runtime_controller_v0"


def _replay_review_items(index: Mapping[str, Any], payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    cases = payload.get("cases")
    if not isinstance(cases, list):
        return items
    for case in cases:
        mapped = _mapping(case)
        if not mapped:
            continue
        if not bool(mapped.get("matched", False)):
            items.append(
                {
                    "severity": "high",
                    "type": "replay_mismatch",
                    "source_path": index.get("copy_path") or index.get("source_path"),
                    "case_id": index.get("case_id"),
                    "source_report_index": mapped.get("source_report_index"),
                    "source_run_index": mapped.get("source_run_index"),
                    "expected_kind": mapped.get("expected_kind"),
                    "actual_kind": mapped.get("actual_kind"),
                }
            )
        elif not bool(mapped.get("patchable", False)) and mapped.get("action_plan_status") not in {"no_action"}:
            items.append(
                {
                    "severity": "medium",
                    "type": "replay_not_patchable",
                    "source_path": index.get("copy_path") or index.get("source_path"),
                    "case_id": index.get("case_id"),
                    "source_report_index": mapped.get("source_report_index"),
                    "source_run_index": mapped.get("source_run_index"),
                    "action_plan_status": mapped.get("action_plan_status"),
                    "blocked_reasons": list(mapped.get("blocked_reasons") or []),
                }
            )
    return items


def _ab_review_items(index: Mapping[str, Any], payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    status = str(payload.get("status") or "")
    decision = _mapping(payload.get("decision"))
    reasons = _string_list(decision.get("reasons"))
    if status == "keep_recommended":
        return []
    severity = "high" if status == "rollback_recommended" else "medium"
    return [
        {
            "severity": severity,
            "type": "ab_evidence_review",
            "source_path": index.get("copy_path") or index.get("source_path"),
            "case_id": index.get("case_id"),
            "status": status,
            "recommended_action": decision.get("recommended_action"),
            "reasons": reasons,
        }
    ]


def _closed_loop_review_items(index: Mapping[str, Any], payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    status = str(payload.get("status") or "")
    decision = _mapping(payload.get("decision"))
    reasons = list(decision.get("reasons") or [])
    if status == "next_request_adapter_blocked":
        runtime_adapter = _mapping(payload.get("runtime_adapter"))
        safety = _mapping(payload.get("safety"))
        return [
            {
                "severity": "low",
                "type": "closed_loop_next_request_boundary",
                "source_path": index.get("copy_path") or index.get("source_path"),
                "case_id": index.get("case_id"),
                "status": status,
                "reasons": reasons,
                "adapter_id": runtime_adapter.get("adapter_id"),
                "next_request_only_adapter": bool(
                    safety.get("next_request_only_adapter", runtime_adapter.get("next_request_only", False))
                ),
                "required_evidence": _string_list(safety.get("required_evidence") or runtime_adapter.get("required_evidence")),
            }
        ]
    if status in {"keep_observed", "duplicate_blocked", "cross_run_cooldown_blocked", "no_action"}:
        return []
    severity = "high" if status in {"needs_review"} else "medium"
    return [
        {
            "severity": severity,
            "type": "closed_loop_evidence_review",
            "source_path": index.get("copy_path") or index.get("source_path"),
            "case_id": index.get("case_id"),
            "status": status,
            "reasons": reasons,
            "action_count": payload.get("action_count"),
            "rolled_back_count": payload.get("rolled_back_count"),
            "rollback_failed_count": payload.get("rollback_failed_count"),
        }
    ]


def _natural_data_wait_review_items(index: Mapping[str, Any], payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    status = str(payload.get("status") or "")
    loss_status = str(_mapping(payload.get("loss_stability")).get("status") or "")
    if status == "natural_dataloader_rebuild_observed" and loss_status not in {"observed", "stable"}:
        return [
            {
                "severity": "medium",
                "type": "natural_data_wait_loss_stability_review",
                "source_path": index.get("copy_path") or index.get("source_path"),
                "case_id": index.get("case_id"),
                "status": status,
                "loss_stability_status": loss_status or "missing",
                "reasons": ["loss_stability_missing"],
            }
        ]
    if status in {"natural_dataloader_rebuild_observed", "no_natural_data_wait"}:
        return []
    decision = _mapping(payload.get("decision"))
    metrics = _mapping(payload.get("metrics"))
    severity = "low" if status == "blocked_benchmark_injection" else "medium"
    return [
        {
            "severity": severity,
            "type": "natural_data_wait_evidence_review",
            "source_path": index.get("copy_path") or index.get("source_path"),
            "case_id": index.get("case_id"),
            "status": status,
            "data_wait_share": metrics.get("data_wait_share"),
            "dominant_bottleneck": metrics.get("dominant_bottleneck"),
            "reasons": _string_list(decision.get("reasons")),
            "benchmark_injection_blockers": _string_list(payload.get("benchmark_injection_blockers")),
        }
    ]


def _natural_data_wait_ab_review_items(index: Mapping[str, Any], payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    status = str(payload.get("status") or "")
    loss_status = str(_mapping(payload.get("loss_stability")).get("status") or "")
    if status == "keep_recommended" and loss_status == "stable":
        return []
    comparison = _mapping(payload.get("comparison"))
    decision = _mapping(payload.get("decision"))
    severity = "low" if status == "blocked_benchmark_injection" else "medium"
    return [
        {
            "severity": severity,
            "type": "natural_data_wait_ab_evidence_review",
            "source_path": index.get("copy_path") or index.get("source_path"),
            "case_id": index.get("case_id"),
            "status": status,
            "steady_samples_per_second_gain_pct": comparison.get("steady_samples_per_second_gain_pct"),
            "data_wait_share_delta": comparison.get("data_wait_share_delta"),
            "loss_stability_status": loss_status or "missing",
            "reasons": _string_list(decision.get("reasons")),
            "benchmark_injection_blockers": _string_list(payload.get("benchmark_injection_blockers")),
        }
    ]


def _cuda_debug_repeat_review_items(index: Mapping[str, Any], payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    status = str(payload.get("status") or "")
    decision = _mapping(payload.get("decision"))
    summary = _mapping(payload.get("summary"))
    pending_statuses = {"pending_manual_gpu_commands", "insufficient_evidence", "pending"}
    failed_statuses = {"execution_failed_needs_review", "execution_failed", "failed"}
    pass_statuses = {"repeat_candidate_review", "pass", "passed"}
    if status in pass_statuses:
        severity = "medium"
        item_type = "cuda_debug_repeat_case_specific_review"
        blocker_kind = ""
    else:
        severity = "high"
        item_type = "cuda_debug_repeat_blocker"
        blocker_kind = "execution_failed" if status in failed_statuses else "pending_or_incomplete"
        if status and status not in pending_statuses and status not in failed_statuses:
            blocker_kind = "repeat_gate_needs_review"
    return [
        {
            "severity": severity,
            "type": item_type,
            "source_path": index.get("copy_path") or index.get("source_path"),
            "case_id": index.get("case_id"),
            "family": index.get("family"),
            "status": status or "missing",
            "blocker_kind": blocker_kind,
            "release_claim_allowed": False,
            "publishable": False,
            "diagnostic_only": True,
            "case_specific_only": True,
            "recommended_action": decision.get("recommended_action"),
            "reasons": _string_list(decision.get("reasons")) or ["cuda_debug_repeat_diagnostic_only"],
            "missing_report_count": summary.get("missing_report_count"),
            "missing_summary_count": summary.get("missing_summary_count"),
            "execution_failure_count": summary.get("execution_failure_count"),
            "comparison_count": summary.get("comparison_count"),
            "repeat_candidate_pass_count": summary.get("repeat_candidate_pass_count"),
        }
    ]


def _claim_gap_review_items(release_claims: Mapping[str, Any]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for gap in release_claims.get("evidence_gaps", []):
        mapped = _mapping(gap)
        if not mapped:
            continue
        gap_id = str(mapped.get("id") or "")
        severity = "high" if gap_id in {"benchmark_case_missing", "benchmark_reports_missing"} else "medium"
        items.append({"severity": severity, "type": "release_evidence_gap", **dict(mapped)})
    return items


def _provenance_review_items(release_claims: Mapping[str, Any]) -> list[dict[str, Any]]:
    provenance = _mapping(release_claims.get("evidence_provenance"))
    buckets = _mapping(provenance.get("buckets"))
    items: list[dict[str, Any]] = []
    for evidence in buckets.get("probe_only", []):
        mapped = _mapping(evidence)
        if not mapped:
            continue
        items.append(
            {
                "severity": "low",
                "type": "release_probe_only_evidence",
                "case_id": mapped.get("case_id"),
                "family": mapped.get("family"),
                "kind": mapped.get("kind"),
                "status": mapped.get("status"),
                "reasons": _string_list(mapped.get("release_probe_only_reasons")),
            }
        )
    return items


def _copy_evidence_file(source: Path, evidence_dir: Path, kind: str, *, sha1: str | None = None) -> str:
    digest = sha1 or _json_sha1(source)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    target = evidence_dir / _stable_file_name(source, kind, digest)
    if not target.exists():
        shutil.copy2(source, target)
    return str(target)


def _write_evidence_payload(source: Path, payload: Mapping[str, Any], evidence_dir: Path, kind: str, *, sha1: str) -> str:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    target = evidence_dir / _stable_file_name(source, kind, sha1)
    if not target.exists():
        target.write_text(_payload_json_text(payload), encoding="utf-8")
    return str(target)


def build_bubble_runtime_evidence_pack(
    paths: Iterable[Path],
    *,
    output_dir: Path | None = None,
    copy_evidence: bool = False,
    min_throughput_gain_pct: float = 3.0,
) -> dict[str, Any]:
    """Build a curated evidence pack and review queue from bubble runtime JSON."""

    output = Path(output_dir) if output_dir is not None else None
    evidence_dir = output / "evidence" if output is not None else None
    indexed: list[dict[str, Any]] = []
    claimable: list[Mapping[str, Any]] = []
    review_queue: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen_evidence_sha1: set[str] = set()

    exclude_dirs = [output] if output is not None else []
    for path in _iter_json_files(paths, exclude_dirs=exclude_dirs):
        try:
            payload = _load_json(path)
        except Exception as exc:
            skipped.append({"path": str(path), "reason": f"{type(exc).__name__}: {exc}"})
            continue
        if not isinstance(payload, Mapping):
            skipped.append({"path": str(path), "reason": "json_root_not_object"})
            continue
        candidates: list[tuple[str, Mapping[str, Any], str, int, bool]] = []
        kind = _report_kind(payload)
        if kind:
            evidence_payload = _curated_source_payload(kind, payload)
            curated = evidence_payload is not payload
            digest = _payload_sha1(evidence_payload) if curated else _json_sha1(path)
            bytes_size = len(_payload_bytes(evidence_payload)) if curated else int(path.stat().st_size)
            candidates.append((kind, evidence_payload, digest, bytes_size, curated))
        for derived_kind, derived_payload in _derived_report_payloads(payload):
            derived_digest = _payload_sha1(derived_payload)
            candidates.append(
                (
                    derived_kind,
                    derived_payload,
                    derived_digest,
                    len(_payload_bytes(derived_payload)),
                    True,
                )
            )
        if not candidates:
            continue
        for kind, evidence_payload, digest, bytes_size, derived in candidates:
            if digest in seen_evidence_sha1:
                continue
            seen_evidence_sha1.add(digest)
            copy_path = ""
            if copy_evidence and evidence_dir:
                copy_path = (
                    _write_evidence_payload(path, evidence_payload, evidence_dir, kind, sha1=digest)
                    if derived
                    else _copy_evidence_file(path, evidence_dir, kind, sha1=digest)
                )
            index = _index_item(
                path,
                evidence_payload,
                kind,
                copy_path=copy_path,
                sha1=digest,
                bytes_size=bytes_size,
            )
            indexed.append(index)
            if _claimable_payload(kind, evidence_payload):
                claimable.append(evidence_payload)
            if kind == "bubble_runtime_replay_report_v0":
                review_queue.extend(_replay_review_items(index, evidence_payload))
            elif kind == "bubble_advisor_ab_evidence_v0":
                review_queue.extend(_ab_review_items(index, evidence_payload))
            elif kind == "bubble_runtime_closed_loop_evidence_v0":
                review_queue.extend(_closed_loop_review_items(index, evidence_payload))
            elif kind == NATURAL_DATA_WAIT_EVIDENCE_REPORT:
                review_queue.extend(_natural_data_wait_review_items(index, evidence_payload))
            elif kind == NATURAL_DATA_WAIT_AB_EVIDENCE_REPORT:
                review_queue.extend(_natural_data_wait_ab_review_items(index, evidence_payload))
            elif kind == CUDA_DEBUG_REPEAT_REPORT:
                review_queue.extend(_cuda_debug_repeat_review_items(index, evidence_payload))

    natural_load_canary = build_bubble_natural_load_canary_report(claimable)
    release_claims = gate_natural_claims(
        build_release_claim_report(claimable, min_throughput_gain_pct=min_throughput_gain_pct),
        natural_load_canary,
    )
    review_queue.extend(_provenance_review_items(release_claims))
    review_queue.extend(natural_load_canary_review_items(natural_load_canary))
    review_queue.extend(_claim_gap_review_items(release_claims))
    followup_plan = build_bubble_runtime_followup_plan(natural_load_canary, release_claims)
    severities = {str(item.get("severity") or "") for item in review_queue}
    status = "needs_review" if "high" in severities else ("review_recommended" if review_queue else "ok")
    pack = {
        "schema_version": 1,
        "report": PACK_REPORT,
        "status": status,
        "evidence_count": len(indexed),
        "claimable_evidence_count": len(claimable),
        "replay_report_count": sum(1 for item in indexed if item.get("kind") == "bubble_runtime_replay_report_v0"),
        "ab_evidence_count": sum(1 for item in indexed if item.get("kind") == "bubble_advisor_ab_evidence_v0"),
        "closed_loop_evidence_count": sum(1 for item in indexed if item.get("kind") == "bubble_runtime_closed_loop_evidence_v0"),
        "natural_data_wait_evidence_count": sum(1 for item in indexed if item.get("kind") == NATURAL_DATA_WAIT_EVIDENCE_REPORT),
        "natural_data_wait_ab_evidence_count": sum(1 for item in indexed if item.get("kind") == NATURAL_DATA_WAIT_AB_EVIDENCE_REPORT),
        "cuda_debug_repeat_evidence_count": sum(1 for item in indexed if item.get("kind") == CUDA_DEBUG_REPEAT_REPORT),
        "gpu_benchmark_count": sum(1 for item in indexed if item.get("kind") == "gpu_bubble_experiment_report_v0"),
        "controller_report_count": sum(1 for item in indexed if item.get("kind") == "bubble_aware_runtime_controller_v0"),
        "min_throughput_gain_pct": _safe_float(min_throughput_gain_pct),
        "evidence_index": indexed,
        "release_claims": release_claims,
        "natural_load_canary": natural_load_canary,
        "followup_plan": followup_plan,
        "review_queue": review_queue,
        "skipped": skipped,
    }
    return pack


def write_bubble_runtime_evidence_pack(pack: Mapping[str, Any], output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pack": str(output_dir / "evidence_pack.json"),
        "index": str(output_dir / "evidence_index.json"),
        "release_claims": str(output_dir / "release_claims.json"),
        "natural_load_canary": str(output_dir / "natural_load_canary.json"),
        "followup_plan": str(output_dir / "followup_plan.json"),
        "review_queue": str(output_dir / "review_queue.json"),
    }
    (output_dir / "evidence_pack.json").write_text(
        json.dumps(dict(pack), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "evidence_index.json").write_text(
        json.dumps(list(pack.get("evidence_index") or []), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "release_claims.json").write_text(
        json.dumps(dict(_mapping(pack.get("release_claims"))), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "natural_load_canary.json").write_text(
        json.dumps(dict(_mapping(pack.get("natural_load_canary"))), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "followup_plan.json").write_text(
        json.dumps(dict(_mapping(pack.get("followup_plan"))), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (output_dir / "review_queue.json").write_text(
        json.dumps(list(pack.get("review_queue") or []), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return paths


__all__ = [
    "PACK_REPORT",
    "build_bubble_runtime_evidence_pack",
    "write_bubble_runtime_evidence_pack",
]
