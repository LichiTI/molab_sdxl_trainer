"""Report-only CUDA graph route scorecard for native training P6 follow-up."""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from core.lulynx_trainer.cudagraph_capture import CUDAGraphCapture, cudagraph_available


DEFAULT_SHAPE = (2, 32)
DEFAULT_WARMUP = 2
DEFAULT_REPLAYS = 4
MAX_REPLAY_DIFF = 1e-5


def build_cuda_graph_route_scorecard(
    *,
    shape: Sequence[int] = DEFAULT_SHAPE,
    warmup: int = DEFAULT_WARMUP,
    replays: int = DEFAULT_REPLAYS,
    run_live_probe: bool = True,
) -> dict[str, Any]:
    """Build a default-off CUDA graph compatibility report.

    The report proves capture/replay mechanics and route guardrails only. It
    does not enable CUDA graph dispatch in the training loop.
    """

    target_shape = tuple(max(int(item), 1) for item in shape)
    capabilities = _capabilities()
    contract = _contract_case(target_shape)
    live_probe = _live_probe(target_shape, warmup=warmup, replays=replays) if run_live_probe else _skipped_probe("live_probe_disabled")
    policy = _policy(capabilities, contract, live_probe)
    validations = _validations(capabilities, contract, live_probe, policy)
    failed = [item for item in validations if not bool(item.get("ok", False))]
    blockers = _dedupe(
        [str(reason) for item in failed for reason in item.get("blocked_reasons", []) or []]
    )
    ready = not blockers
    return {
        "schema_version": 1,
        "scorecard": "turbocore_cuda_graph_route_scorecard_v0",
        "gate": "p6e_cuda_graph_route",
        "ok": True,
        "promotion_ready": ready,
        "training_path_enabled": False,
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "default_behavior_changed": False,
        "experimental_only": True,
        "shape": [int(item) for item in target_shape],
        "warmup": int(warmup),
        "replays": int(replays),
        "capabilities": capabilities,
        "static_contract": contract,
        "live_probe": live_probe,
        "policy": policy,
        "validations": validations,
        "summary": {
            "cuda_graph_available": bool(capabilities.get("cuda_graph_available", False)),
            "static_contract_ready": bool(contract.get("static_contract_ready", False)),
            "live_capture_status": str(live_probe.get("status", "unknown")),
            "live_capture_ready": bool(live_probe.get("capture_replay_ready", False)),
            "max_replay_diff": live_probe.get("max_replay_diff"),
            "allowed_initial_modes": list(policy.get("allowed_initial_modes", []) or []),
            "blocked_modes_until_review": list(policy.get("blocked_modes_until_review", []) or []),
            "training_path_enabled": False,
        },
        "promotion_blockers": blockers,
        "blocked_reasons": blockers,
        "recommended_next_step": (
            "add CUDA graph route observe-mode manifest before any training integration"
            if ready
            else "fix CUDA graph route scorecard blockers"
        ),
        "notes": [
            "This scorecard only validates static-shape CUDA graph capture/replay mechanics.",
            "Training loop dispatch remains disabled and must be reviewed separately.",
            "CUDA graph remains incompatible with offload, safe fallback, torch.compile, or dynamic shape routes.",
        ],
    }


def _capabilities() -> dict[str, Any]:
    cuda = bool(torch.cuda.is_available())
    return {
        "cuda": cuda,
        "cuda_graph_available": bool(cudagraph_available()),
        "device_name": torch.cuda.get_device_name(0) if cuda else "",
        "torch_compile_available": callable(getattr(torch, "compile", None)),
        "helper": "core.lulynx_trainer.cudagraph_capture.CUDAGraphCapture",
    }


def _contract_case(shape: Sequence[int]) -> dict[str, Any]:
    sample = torch.randn(*shape, dtype=torch.float32)
    alt = torch.ones(*shape, dtype=torch.float32)
    bad = torch.ones(shape[0], max(int(shape[-1]) + 1, 1), dtype=torch.float32)
    model = _TinyGraphModel(int(shape[-1]))
    capture = CUDAGraphCapture(model, sample, device="cpu")
    static = capture._make_static(sample)
    capture._static_inputs = static
    capture._copy_to_static(alt)
    copy_ok = torch.allclose(capture._static_inputs, alt)
    shape_blocked = False
    try:
        capture._copy_to_static(bad)
    except RuntimeError:
        shape_blocked = True
    return {
        "schema_version": 1,
        "case": "cuda_graph_static_input_contract",
        "ok": bool(static.shape == sample.shape and static.dtype == sample.dtype and copy_ok),
        "static_contract_ready": bool(static.shape == sample.shape and static.dtype == sample.dtype and copy_ok),
        "shape": [int(item) for item in shape],
        "dtype": "float32",
        "copy_to_static_ok": bool(copy_ok),
        "shape_mismatch_blocked": bool(shape_blocked),
        "requires_static_shape": True,
        "requires_static_dtype": True,
        "blocked_features": [
            "block_offload",
            "module_offload",
            "cpu_offload_checkpointing",
            "safe_fallback",
            "torch_compile_active",
            "dynamic_batch_or_resolution",
        ],
        "blocked_reasons": [] if copy_ok and shape_blocked else ["cuda_graph_static_contract_failed"],
    }


def _live_probe(shape: Sequence[int], *, warmup: int, replays: int) -> dict[str, Any]:
    if not bool(cudagraph_available()):
        return _skipped_probe("cuda_graph_unavailable")
    device = torch.device("cuda")
    model = _TinyGraphModel(int(shape[-1])).to(device)
    model.requires_grad_(False)
    sample = torch.randn(*shape, device=device, dtype=torch.float32)
    capture = CUDAGraphCapture(model, sample, device=device)
    try:
        started = time.perf_counter()
        capture.warmup(num_steps=max(int(warmup), 1))
        capture.capture()
        diffs = []
        for index in range(max(int(replays), 1)):
            inputs = torch.randn(*shape, device=device, dtype=torch.float32) * (1.0 + index * 0.05)
            expected = model(inputs).detach().clone()
            output = capture.replay(inputs)
            torch.cuda.synchronize(device)
            diffs.append(_max_abs_diff(expected, output))
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        max_diff = max(diffs) if diffs else float("inf")
        ready = bool(capture.is_captured and max_diff <= MAX_REPLAY_DIFF)
        return {
            "schema_version": 1,
            "case": "cuda_graph_capture_replay_live_probe",
            "status": "passed" if ready else "failed",
            "ok": ready,
            "capture_replay_ready": ready,
            "graph_captured": bool(capture.is_captured),
            "shape": [int(item) for item in shape],
            "dtype": "float32",
            "warmup": int(warmup),
            "replays": int(replays),
            "max_replay_diff": max_diff,
            "tolerance": MAX_REPLAY_DIFF,
            "elapsed_ms": round(elapsed_ms, 4),
            "blocked_reasons": [] if ready else ["cuda_graph_capture_replay_parity_failed"],
        }
    except Exception as exc:
        return {
            "schema_version": 1,
            "case": "cuda_graph_capture_replay_live_probe",
            "status": "failed",
            "ok": False,
            "capture_replay_ready": False,
            "error": f"{type(exc).__name__}: {exc}",
            "blocked_reasons": [f"cuda_graph_capture_replay_failed:{type(exc).__name__}"],
        }
    finally:
        del capture


def _policy(
    capabilities: Mapping[str, Any],
    contract: Mapping[str, Any],
    live_probe: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "policy_kind": "cuda_graph_route_observe_policy_v0",
        "canary_enabled_by_default": False,
        "explicit_opt_in_required": True,
        "allowed_initial_modes": ["off", "observe"],
        "blocked_modes_until_review": ["canary", "auto"],
        "required_preflight_gates": [
            "cuda_graph_available",
            "static_input_contract",
            "live_capture_replay_probe",
            "offload_disabled",
            "safe_fallback_disabled",
            "torch_compile_disabled",
            "static_batch_resolution",
        ],
        "runtime_incompatibilities": list(contract.get("blocked_features", []) or []),
        "live_capture_ready": bool(live_probe.get("capture_replay_ready", False)),
        "cuda_graph_available": bool(capabilities.get("cuda_graph_available", False)),
        "runtime_dispatch_ready": False,
        "native_dispatch_allowed": False,
        "training_path_enabled": False,
    }


def _validations(
    capabilities: Mapping[str, Any],
    contract: Mapping[str, Any],
    live_probe: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    return [
        _validation(
            "cuda_graph_available",
            bool(capabilities.get("cuda_graph_available", False)),
            "cuda_graph_unavailable",
        ),
        _validation(
            "static_input_contract_ready",
            bool(contract.get("static_contract_ready", False))
            and bool(contract.get("shape_mismatch_blocked", False)),
            "cuda_graph_static_contract_missing",
        ),
        _validation(
            "live_capture_replay_probe_ready",
            bool(live_probe.get("capture_replay_ready", False)),
            "cuda_graph_live_capture_probe_missing",
        ),
        _validation(
            "policy_default_off",
            not bool(policy.get("canary_enabled_by_default", True))
            and bool(policy.get("explicit_opt_in_required", False)),
            "cuda_graph_policy_not_default_off",
        ),
        _validation(
            "runtime_dispatch_disabled",
            not bool(policy.get("runtime_dispatch_ready", True))
            and not bool(policy.get("native_dispatch_allowed", True))
            and not bool(policy.get("training_path_enabled", True)),
            "cuda_graph_policy_enabled_dispatch",
        ),
    ]


def _validation(name: str, ok: bool, blocker: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "validation": name,
        "ok": bool(ok),
        "blocked_reasons": [] if ok else [blocker],
    }


class _TinyGraphModel(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(dim), int(dim))
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(int(dim)))
            self.linear.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.linear(x)) + x * 0.125


def _max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max().cpu().item())


def _skipped_probe(reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "case": "cuda_graph_capture_replay_live_probe",
        "status": "skipped",
        "reason": reason,
        "ok": False,
        "capture_replay_ready": False,
        "blocked_reasons": [],
    }


def _dedupe(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = str(value or "")
        if text and text not in out:
            out.append(text)
    return out


__all__ = ["build_cuda_graph_route_scorecard"]
