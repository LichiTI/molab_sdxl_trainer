"""Generate optimizer promotion evidence for TurboCore native update gates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.lulynx_trainer.turbocore_cuda_runtime_benchmark_smoke import (  # noqa: E402
    _benchmark_case,
    _inject_native_artifact_dir_from_env,
)
from core.lulynx_trainer.turbocore_optimizer_performance_gate import (  # noqa: E402
    evaluate_optimizer_performance_gate,
)

try:  # noqa: E402
    from core.turbocore_capabilities import probe_native_training_bridge
    from core.turbocore_native_abi import validate_native_optimizer_stateful_capability
except Exception:  # pragma: no cover - direct smoke import fallback
    probe_native_training_bridge = None  # type: ignore[assignment]
    validate_native_optimizer_stateful_capability = None  # type: ignore[assignment]


def _mb_for_numel(numel: int, tensors: int = 1) -> float:
    return round(max(int(numel), 0) * 4 * max(int(tensors), 1) / (1024 * 1024), 4)


def _torch_optimizer_name(provider: str) -> str:
    normalized = str(provider or "").strip().lower()
    return "torch_adamw_fused" if "fused" in normalized else "torch_adamw"


def _stateful_gate() -> dict[str, Any]:
    if probe_native_training_bridge is None or validate_native_optimizer_stateful_capability is None:
        return {
            "schema_version": 1,
            "validator": "turbocore_native_optimizer_stateful_capability",
            "ok": False,
            "reason": "validator_unavailable",
        }
    try:
        return validate_native_optimizer_stateful_capability(probe_native_training_bridge())
    except Exception as exc:  # pragma: no cover - defensive benchmark metadata
        return {
            "schema_version": 1,
            "validator": "turbocore_native_optimizer_stateful_capability",
            "ok": False,
            "reason": "validator_failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _inject_default_native_artifact_dirs() -> None:
    paths = (
        ROOT / "native" / "target" / "release",
        ROOT / "native" / "target" / "release" / "deps",
        ROOT / "native" / "target" / "debug",
        ROOT / "native" / "target" / "debug" / "deps",
    )
    for path in reversed(paths):
        if path.is_dir():
            resolved = str(path.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)


def build_promotion_artifact(*, numel: int, iterations: int, warmup: int, repeats: int) -> dict[str, Any]:
    _inject_native_artifact_dir_from_env()
    _inject_default_native_artifact_dirs()
    spec = importlib.util.find_spec("lulynx_native")
    if spec is None:
        return {
            "schema_version": 1,
            "benchmark": "turbocore_optimizer_promotion_benchmark_v0",
            "ok": False,
            "skipped": True,
            "reason": "lulynx_native_not_importable",
        }
    if not torch.cuda.is_available():
        return {
            "schema_version": 1,
            "benchmark": "turbocore_optimizer_promotion_benchmark_v0",
            "ok": False,
            "skipped": True,
            "reason": "torch_cuda_unavailable",
        }

    import lulynx_native  # type: ignore

    case = _benchmark_case(
        lulynx_native,
        numel=max(int(numel), 1),
        iterations=max(int(iterations), 1),
        warmup=max(int(warmup), 0),
        repeats=max(int(repeats), 1),
    )
    native_ms = float(case["native_avg_ms_stats"]["median"] or 0.0)
    native_event_ms = float(case["native_cuda_event_avg_ms_stats"]["median"] or 0.0)
    native_step_ms = native_event_ms if native_event_ms > 0 else native_ms
    torch_step_ms = float(case["torch_adamw_avg_ms_stats"]["median"] or 0.0)
    parameter_mb = _mb_for_numel(numel, tensors=1)
    state_mb = _mb_for_numel(numel, tensors=2)
    results = [
        {
            "optimizer": _torch_optimizer_name(str(case.get("torch_adamw_provider", ""))),
            "success": True,
            "step_ms": torch_step_ms,
            "state_mb": state_mb,
            "parameter_mb": parameter_mb,
            "native_kernel_present": False,
            "exact_adamw_candidate": False,
        },
        {
            "optimizer": "turbocore_adamw_cuda_runtime_session",
            "success": True,
            "step_ms": native_step_ms,
            "state_mb": state_mb,
            "parameter_mb": parameter_mb,
            "native_kernel_present": True,
            "exact_adamw_candidate": True,
            "parity_max_abs_diff": float(case.get("max_abs_diff", 0.0) or 0.0),
            "parity_max_rel_diff": float(case.get("max_rel_diff", 0.0) or 0.0),
            "timing_source": str(case.get("timing_source", "") or ""),
        },
    ]
    probe_payload = {
        "schema_version": 1,
        "benchmark": "turbocore_optimizer_promotion_benchmark_v0",
        "ok": True,
        "skipped": False,
        "origin": str(getattr(spec, "origin", "") or ""),
        "device": torch.cuda.get_device_name(0),
        "numel": max(int(numel), 1),
        "iters": max(int(iterations), 1),
        "warmup": max(int(warmup), 0),
        "repeats": max(int(repeats), 1),
        "results": results,
        "cases": [case],
        "stateful_abi_gate": _stateful_gate(),
    }
    gate = evaluate_optimizer_performance_gate(probe_payload)
    probe_payload["optimizer_performance_gate"] = gate
    probe_payload["ok"] = bool(gate.get("ok", False))
    probe_payload["promotion_gate_ok"] = bool(gate.get("promotion_gate_ok", False))
    return probe_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numel", type=int, default=2171392)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--out", default="")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    artifact = build_promotion_artifact(
        numel=int(args.numel),
        iterations=int(args.iters),
        warmup=int(args.warmup),
        repeats=int(args.repeats),
    )
    text = json.dumps(artifact, ensure_ascii=False, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    return 0 if bool(artifact.get("ok", False)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
