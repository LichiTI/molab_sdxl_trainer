"""TurboCore candidate scorecard research prototype.

This module keeps experimental candidates behind developer probes.  It does not
activate training paths; it only answers whether a candidate is discoverable,
parity-safe, benchmarked, and worth the next native/Triton implementation step.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.turbocore_candidates import list_turbocore_candidates  # noqa: E402
from core.turbocore_parity import check_lora_delta_parity, check_native_optimizer_parity  # noqa: E402
from core.lulynx_trainer.turbocore_lora_candidate_policy import decide_lora_candidate_for_shape  # noqa: E402
from core.lulynx_trainer.turbocore_lora_fused_benchmark import SHAPE_PRESETS, run_benchmark as run_lora_benchmark  # noqa: E402
from core.lulynx_trainer.turbocore_native_optimizer_benchmark import run_benchmark as run_optimizer_benchmark  # noqa: E402


def _device(value: str) -> torch.device:
    requested = str(value or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _dtype(value: str, device: torch.device) -> torch.dtype:
    normalized = str(value or "float32").strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"} and device.type != "cpu":
        return torch.float16
    return torch.float32


def _safe_call(name: str, fn) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        payload = fn()
        if isinstance(payload, dict):
            payload.setdefault("ok", True)
            payload.setdefault("elapsed_seconds", round(time.perf_counter() - started, 4))
            return payload
        return {"ok": True, "value": payload, "elapsed_seconds": round(time.perf_counter() - started, 4)}
    except Exception as exc:
        return {
            "ok": False,
            "section": name,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": round(time.perf_counter() - started, 4),
        }


def _candidate_gate(candidate: dict[str, Any], parity: dict[str, Any] | None, benchmark: dict[str, Any] | None) -> str:
    if not bool(candidate.get("available", False)):
        return "discovery_only"
    if parity is not None and not bool(parity.get("ok", False)):
        return "blocked_parity"
    if benchmark is not None and not bool(benchmark.get("ok", False)):
        return "blocked_benchmark"
    if bool(candidate.get("native", False)):
        return "native_candidate_needs_repeated_validation"
    if bool(candidate.get("experimental", False)):
        return "experimental_reference_candidate"
    return "reference_baseline"


GATE_EXPLANATIONS: dict[str, str] = {
    "reference_baseline": "Stable PyTorch/reference path used as a comparison baseline.",
    "experimental_probe_skipped": "Optional experimental probe skipped by default so host-specific failures do not pollute readiness.",
    "device_incompatible": "Candidate requires a device capability that is not present for this scorecard run.",
    "discovery_only": "Candidate is registered for discovery, but no callable implementation is available yet.",
    "shape_policy_skipped": "Shape-aware research policy skipped this candidate for the requested preset/rank.",
    "blocked_parity": "Candidate failed correctness/parity and must not be benchmark-promoted.",
    "blocked_benchmark": "Candidate benchmark failed or could not produce usable evidence.",
    "experimental_reference_candidate": "Candidate passed this narrow probe, but remains research-only and is not training-ready.",
    "native_candidate_needs_repeated_validation": "Native candidate exists, but still needs repeated correctness, stability, and benchmark gates.",
}


def _explain_gate(gate: str) -> str:
    return GATE_EXPLANATIONS.get(str(gate or "unknown"), "Unknown gate; inspect row details before using this result.")


def _training_blockers(rows: list[dict[str, Any]], *, iters: int, warmup: int) -> list[str]:
    blockers: list[str] = ["TurboCore candidates are research-only and are not wired to training activation."]
    if int(iters) < 5 or int(warmup) < 1:
        blockers.append("Benchmark evidence is smoke-level only; use more iterations/warmup before treating speedups as evidence.")
    gate_counts: dict[str, int] = {}
    for row in rows:
        gate = str(row.get("gate", "unknown") or "unknown")
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
    if gate_counts.get("blocked_parity", 0):
        blockers.append("At least one candidate is blocked by parity failure.")
    if gate_counts.get("blocked_benchmark", 0):
        blockers.append("At least one candidate is blocked by benchmark failure.")
    if gate_counts.get("shape_policy_skipped", 0):
        blockers.append("Shape policy skipped at least one candidate, so current evidence is shape-limited.")
    if not gate_counts.get("native_candidate_needs_repeated_validation", 0):
        blockers.append("No native candidate has passed repeated validation gates yet.")
    return blockers


def _lora_scorecard_row(
    candidate: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
    preset: str,
    rank: int,
    iters: int,
    warmup: int,
    run_unavailable: bool,
    include_torch_compile: bool,
    shape_policy: str,
) -> dict[str, Any]:
    name = str(candidate.get("name", "") or "")
    available = bool(candidate.get("available", False))
    row: dict[str, Any] = {"candidate": candidate, "feature": "lora_fused"}
    if name == "torch_compile" and not include_torch_compile:
        gate = "experimental_probe_skipped"
        row.update({"gate": gate, "gate_explanation": _explain_gate(gate), "parity": None, "benchmark": None})
        return row
    if name.startswith("triton_") and device.type != "cuda":
        gate = "device_incompatible"
        row.update({"gate": gate, "gate_explanation": _explain_gate(gate), "parity": None, "benchmark": None})
        return row
    if not available and not run_unavailable:
        gate = "discovery_only"
        row.update({"gate": gate, "gate_explanation": _explain_gate(gate), "parity": None, "benchmark": None})
        return row

    shape_filter, skipped_cases, run_allowed = _lora_shape_filter(
        candidate=name,
        preset=preset,
        rank=rank,
        shape_policy=shape_policy,
    )
    if not run_allowed:
        gate = "shape_policy_skipped"
        row.update({
            "gate": gate,
            "gate_explanation": _explain_gate(gate),
            "parity": None,
            "benchmark": None,
            "shape_policy": shape_policy,
            "skipped_cases": skipped_cases,
        })
        return row

    parity = _safe_call(
        f"lora_parity:{name}",
        lambda: check_lora_delta_parity(device=device, dtype=dtype, rank=rank, candidate_name=name).as_dict(),
    )
    benchmark = _safe_call(
        f"lora_benchmark:{name}",
        lambda: run_lora_benchmark(
            preset=preset,
            ranks=[max(int(rank), 1)],
            dtype=dtype,
            device=device,
            iters=max(int(iters), 1),
            warmup=max(int(warmup), 0),
            candidate_name=name,
            shape_filter=shape_filter,
        ),
    )
    gate = _candidate_gate(candidate, parity, benchmark)
    row.update({
        "gate": gate,
        "gate_explanation": _explain_gate(gate),
        "parity": parity,
        "benchmark": benchmark,
        "shape_policy": shape_policy,
        "skipped_cases": skipped_cases,
    })
    return row


def _lora_shape_filter(
    *,
    candidate: str,
    preset: str,
    rank: int,
    shape_policy: str,
) -> tuple[Any, list[dict[str, Any]], bool]:
    shapes = SHAPE_PRESETS.get(preset) or []
    allowed: set[tuple[int, int, int, int]] = set()
    skipped: list[dict[str, Any]] = []
    for batch, tokens, width in shapes:
        decision = decide_lora_candidate_for_shape(
            candidate=candidate,
            preset=preset,
            batch=batch,
            tokens=tokens,
            width=width,
            rank=rank,
            shape_policy=shape_policy,
        )
        if decision.should_run:
            allowed.add((int(batch), int(tokens), int(width), int(rank)))
        else:
            skipped.append(decision.as_dict())

    def shape_filter(batch: int, tokens: int, width: int, row_rank: int) -> bool:
        return (int(batch), int(tokens), int(width), int(row_rank)) in allowed

    return shape_filter, skipped, bool(allowed)


def _optimizer_scorecard_row(
    candidate: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
    preset: str,
    rank: int,
    iters: int,
    warmup: int,
    run_unavailable: bool,
) -> dict[str, Any]:
    name = str(candidate.get("name", "") or "")
    available = bool(candidate.get("available", False))
    row: dict[str, Any] = {"candidate": candidate, "feature": "native_optimizer"}
    if not available and not run_unavailable:
        gate = "discovery_only"
        row.update({"gate": gate, "gate_explanation": _explain_gate(gate), "parity": None, "benchmark": None})
        return row

    optimizer_preset = "tiny" if preset == "tiny" else "sdxl_lora_short"
    parity = _safe_call(
        f"optimizer_parity:{name}",
        lambda: check_native_optimizer_parity(device=device, dtype=dtype, rank=rank, candidate_name=name).as_dict(),
    )
    benchmark = _safe_call(
        f"optimizer_benchmark:{name}",
        lambda: run_optimizer_benchmark(
            preset=optimizer_preset,
            ranks=[max(int(rank), 1)],
            dtype=dtype,
            device=device,
            iters=max(int(iters), 1),
            warmup=max(int(warmup), 0),
            candidate_name=name,
        ),
    )
    gate = _candidate_gate(candidate, parity, benchmark)
    row.update({
        "gate": gate,
        "gate_explanation": _explain_gate(gate),
        "parity": parity,
        "benchmark": benchmark,
    })
    return row


def build_candidate_scorecard(
    *,
    device: torch.device,
    dtype: torch.dtype,
    preset: str = "tiny",
    rank: int = 4,
    iters: int = 1,
    warmup: int = 0,
    run_unavailable: bool = False,
    include_torch_compile: bool = False,
    shape_policy: str = "auto",
) -> dict[str, Any]:
    started = time.perf_counter()
    candidates = list_turbocore_candidates()
    rows: list[dict[str, Any]] = []
    for candidate in candidates.get("lora_fused", []):
        rows.append(_lora_scorecard_row(
            candidate,
            device=device,
            dtype=dtype,
            preset=preset,
            rank=rank,
            iters=iters,
            warmup=warmup,
            run_unavailable=run_unavailable,
            include_torch_compile=include_torch_compile,
            shape_policy=shape_policy,
        ))
    for candidate in candidates.get("native_optimizer", []):
        rows.append(_optimizer_scorecard_row(
            candidate,
            device=device,
            dtype=dtype,
            preset=preset,
            rank=rank,
            iters=iters,
            warmup=warmup,
            run_unavailable=run_unavailable,
        ))

    gate_counts: dict[str, int] = {}
    for row in rows:
        gate = str(row.get("gate", "unknown") or "unknown")
        gate_counts[gate] = gate_counts.get(gate, 0) + 1
    available_rows = [row for row in rows if bool((row.get("candidate") or {}).get("available", False))]
    native_available = [row for row in available_rows if bool((row.get("candidate") or {}).get("native", False))]
    training_blockers = _training_blockers(rows, iters=iters, warmup=warmup)
    return {
        "schema_version": 1,
        "prototype": "turbocore_candidate_scorecard",
        "device": str(device),
        "dtype": str(dtype).replace("torch.", ""),
        "preset": preset,
        "rank": int(rank),
        "iters": int(iters),
        "warmup": int(warmup),
        "run_unavailable": bool(run_unavailable),
        "include_torch_compile": bool(include_torch_compile),
        "shape_policy": shape_policy,
        "summary": {
            "candidate_count": len(rows),
            "available_candidate_count": len(available_rows),
            "native_available_candidate_count": len(native_available),
            "gate_counts": gate_counts,
            "gate_explanations": GATE_EXPLANATIONS,
            "training_activation_blockers": training_blockers,
            "ready_for_training_activation": False,
            "recommended_next_step": "keep candidates behind parity/benchmark gates; treat smoke speedups as directional only",
        },
        "rows": rows,
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TurboCore candidate scorecard research probe")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--preset", default="tiny")
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--iters", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--run-unavailable", action="store_true")
    parser.add_argument("--include-torch-compile", action="store_true")
    parser.add_argument("--shape-policy", default="auto", choices=["auto", "off", "disabled"])
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    device = _device(args.device)
    dtype = _dtype(args.dtype, device)
    payload = build_candidate_scorecard(
        device=device,
        dtype=dtype,
        preset=str(args.preset or "tiny"),
        rank=max(int(args.rank), 1),
        iters=max(int(args.iters), 1),
        warmup=max(int(args.warmup), 0),
        run_unavailable=bool(args.run_unavailable),
        include_torch_compile=bool(args.include_torch_compile),
        shape_policy=str(args.shape_policy or "auto"),
    )
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
