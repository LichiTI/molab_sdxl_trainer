# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Probe-level real-GPU A/B benchmark for CDM-QTA quantized LoRA training (P4).

The frontier 4-phase ladder's P4 is "run the CDM-QTA real-GPU A/B, then wire it
into the trainer *only if* it earns its keep". The ``cdm_qta_lora_probe`` module
ships a report-only primitive (a frozen ``Linear`` + LoRA whose LoRA branch is
symmetric straight-through fake-quantized while the fp32 LoRA master stays the
trainable source of truth) and a stack of default-off evidence gates that
*consume* A/B results without ever executing one themselves. This driver is the
separate executor those gates were designed to ingest from: it runs the probe at
realistic DiT-projection dimensions, fp32-LoRA (``baseline``) vs fake-quant LoRA
(``qta8`` / ``qta4``), measures timed step / peak VRAM / NVML energy / loss /
output quality drift, runs an optimizer-state roundtrip + resume parity sub-run,
then feeds everything through the existing gates and prints a conditional-wire
verdict.

What this is NOT: it does not wire CDM-QTA into the live trainer, does not flip
any gate safety flag, and does not stand in for a real Anima/newbie training A/B
(the ``real_anima_newbie_training_ab_missing`` blocker is intentionally left
standing — that is the operator's real-model job). Local theoretical-runnable is
the bar here. Clean-room Lulynx harness, mirroring the structure of
``frontier_method_ab_benchmark.py``.

Run:
  backend/env/python-flashattention/python.exe \\
    backend/core/lulynx_trainer/cdm_qta_lora_ab_benchmark.py --arms baseline qta8 qta4 --steps 50
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[3]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from backend.core.lulynx_trainer.cdm_qta_lora_probe import (
    CDMQTALoraLinearProbe,
    CDMQTALoraProbeConfig,
    build_cdm_qta_lora_quant_train_scorecard,
    estimate_cdm_qta_lora_memory,
)
from backend.core.lulynx_trainer.cdm_qta_lora_ab_review import (
    build_cdm_qta_lora_ab_evidence_package,
    build_cdm_qta_lora_ab_result_ingestion,
)
from backend.core.lulynx_trainer.cdm_qta_lora_optimizer_state_contract import (
    build_cdm_qta_lora_optimizer_state_contract,
    build_cdm_qta_lora_optimizer_state_parity_gate,
)


ARM_SPECS: dict[str, dict[str, Any]] = {
    "baseline": {"enabled": False, "quant_bits": 8},
    "qta8": {"enabled": True, "quant_bits": 8},
    "qta4": {"enabled": True, "quant_bits": 4},
}
DEFAULT_ARMS = ("baseline", "qta8", "qta4")

PARITY_TOL = 1e-6
# Quality-oriented thresholds: CDM-QTA is a quantization-quality probe, not a
# training-speed play, so step/energy regressions are tolerated and the gate
# focuses on output drift, loss parity, and optimizer-state parity.
QUALITY_THRESHOLDS: dict[str, float] = {
    "min_step_time_improvement": -1.0,
    "min_energy_improvement": -1.0,
    "max_vram_regression": 1.0,
    "max_quality_drift": 0.01,
    "max_loss_delta": 0.02,
}


@dataclass
class ArmResult:
    arm: str
    enabled: bool
    quant_bits: int
    device: str
    steps: int
    initial_loss: float = 0.0
    final_loss: float = 0.0
    mean_step_ms: float = 0.0
    peak_vram_mb: float = 0.0
    energy_per_step_j: float | None = None
    mean_power_w: float | None = None
    losses: list = field(default_factory=list)


class _PowerSampler:
    """NVML board-power sampler with a graceful no-op fallback off-GPU."""

    def __init__(self, device: torch.device) -> None:
        self.available = False
        self._nvml = None
        self._handle = None
        self._samples: list[float] = []
        if device.type != "cuda":
            return
        try:
            import pynvml

            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device.index or 0)
            self._nvml = pynvml
            self.available = True
        except Exception:
            self.available = False

    def reset(self) -> None:
        self._samples = []

    def sample(self) -> float | None:
        if not self.available:
            return None
        try:
            watts = float(self._nvml.nvmlDeviceGetPowerUsage(self._handle)) / 1000.0
        except Exception:
            return None
        self._samples.append(watts)
        return watts

    def mean_power_w(self) -> float | None:
        return sum(self._samples) / len(self._samples) if self._samples else None

    def shutdown(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass


def _config(arm: str, dims: Mapping[str, int]) -> CDMQTALoraProbeConfig:
    spec = ARM_SPECS[arm]
    return CDMQTALoraProbeConfig(
        enabled=bool(spec["enabled"]),
        in_features=int(dims["in_features"]),
        out_features=int(dims["out_features"]),
        rank=int(dims["rank"]),
        quant_bits=int(spec["quant_bits"]),
    )


def _build_probe(arm: str, dims: Mapping[str, int], device: torch.device, seed: int) -> CDMQTALoraLinearProbe:
    torch.manual_seed(seed)
    return CDMQTALoraLinearProbe(_config(arm, dims)).to(device)


def _warmup_device(dims: Mapping[str, int], tokens: int, device: torch.device, seed: int, iters: int = 80) -> None:
    """Ramp GPU boost clocks to steady state so every arm is timed/metered fairly.

    Without this the first (``baseline``) arm pays a cold-clock penalty: it both
    times slower and reads lower NVML power than later arms, which silently
    inflates the candidates' apparent energy cost. A throwaway run fixes that.
    """
    if device.type != "cuda":
        return
    probe = _build_probe("qta8", dims, device, seed)
    opt = torch.optim.AdamW([p for p in probe.parameters() if p.requires_grad], lr=1e-3)
    x, y = _fixed_batch(dims, seed + 1, tokens, device)
    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(probe(x), y)
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    del probe, opt, x, y
    torch.cuda.empty_cache()
    gc.collect()


def _fixed_batch(dims: Mapping[str, int], seed: int, tokens: int, device: torch.device):
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    x = torch.randn(tokens, int(dims["in_features"]), generator=gen)
    y = torch.randn(tokens, int(dims["out_features"]), generator=gen)
    return x.to(device), y.to(device)


def run_arm(
    arm: str,
    *,
    dims: Mapping[str, int],
    tokens: int,
    steps: int,
    warmup: int,
    seed: int,
    device: torch.device,
    lr: float,
    sampler: _PowerSampler,
    eval_x: torch.Tensor,
) -> tuple[ArmResult, torch.Tensor]:
    """Train the probe for ``steps`` timed steps and capture the A/B metrics."""
    probe = _build_probe(arm, dims, device, seed)
    opt = torch.optim.AdamW([p for p in probe.parameters() if p.requires_grad], lr=lr)
    x, y = _fixed_batch(dims, seed + 1, tokens, device)
    is_cuda = device.type == "cuda"

    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    sampler.reset()

    losses: list[float] = []
    step_ms: list[float] = []
    for i in range(steps + warmup):
        if is_cuda:
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            start_evt.record()
        else:
            t0 = time.perf_counter()

        opt.zero_grad(set_to_none=True)
        loss = F.mse_loss(probe(x), y)
        loss.backward()
        opt.step()

        if is_cuda:
            end_evt.record()
            torch.cuda.synchronize()
            dt_ms = start_evt.elapsed_time(end_evt)
        else:
            dt_ms = (time.perf_counter() - t0) * 1000.0

        losses.append(float(loss.detach().cpu()))
        if i >= warmup:
            step_ms.append(dt_ms)
            sampler.sample()

    peak_vram_mb = (torch.cuda.max_memory_allocated(device) / (1024 * 1024)) if is_cuda else 0.0
    mean_step_ms = sum(step_ms) / len(step_ms) if step_ms else 0.0
    mean_power = sampler.mean_power_w()
    energy = mean_power * (mean_step_ms / 1000.0) if mean_power is not None else None

    with torch.no_grad():
        eval_out = probe(eval_x).detach().float().cpu()

    result = ArmResult(
        arm=arm,
        enabled=bool(ARM_SPECS[arm]["enabled"]),
        quant_bits=int(ARM_SPECS[arm]["quant_bits"]),
        device=str(device),
        steps=steps,
        initial_loss=losses[0] if losses else 0.0,
        final_loss=losses[-1] if losses else 0.0,
        mean_step_ms=mean_step_ms,
        peak_vram_mb=peak_vram_mb,
        energy_per_step_j=energy,
        mean_power_w=mean_power,
        losses=losses,
    )
    del probe, opt, x, y
    if is_cuda:
        torch.cuda.empty_cache()
    gc.collect()
    return result, eval_out


def optimizer_state_parity_subrun(
    *,
    bits: int,
    dims: Mapping[str, int],
    tokens: int,
    device: torch.device,
    seed: int,
    lr: float,
    steps: int = 5,
) -> dict[str, Any]:
    """State-dict roundtrip + resume parity for a fake-quant LoRA probe.

    Trains a quant probe, snapshots ``state_dict`` + optimizer state, reloads
    into a freshly (differently) initialized probe, then proves the fp32 master
    params are preserved, the quantized forward is rebuilt bit-identically, and
    one resumed step lands on the same loss.
    """
    cfg = CDMQTALoraProbeConfig(
        enabled=True,
        in_features=int(dims["in_features"]),
        out_features=int(dims["out_features"]),
        rank=int(dims["rank"]),
        quant_bits=int(bits),
    )
    torch.manual_seed(seed)
    probe = CDMQTALoraLinearProbe(cfg).to(device)
    opt = torch.optim.AdamW([p for p in probe.parameters() if p.requires_grad], lr=lr)
    x, y = _fixed_batch(dims, seed + 1, tokens, device)

    def _step(p: CDMQTALoraLinearProbe, o: torch.optim.Optimizer) -> float:
        o.zero_grad(set_to_none=True)
        loss = F.mse_loss(p(x), y)
        loss.backward()
        o.step()
        return float(loss.detach().cpu())

    for _ in range(steps):
        _step(probe, opt)

    state = {k: v.detach().clone() for k, v in probe.state_dict().items()}
    opt_state = opt.state_dict()
    master_before = {n: p.detach().clone() for n, p in probe.named_parameters() if p.requires_grad}

    # Reload into a probe with intentionally different init to prove the load wins.
    torch.manual_seed(seed + 9973)
    reloaded = CDMQTALoraLinearProbe(cfg).to(device)
    reloaded.load_state_dict(state)
    opt2 = torch.optim.AdamW([p for p in reloaded.parameters() if p.requires_grad], lr=lr)
    opt2.load_state_dict(opt_state)

    max_param_delta = 0.0
    reloaded_params = dict(reloaded.named_parameters())
    for name, before in master_before.items():
        delta = (reloaded_params[name].detach() - before).abs().max().item()
        max_param_delta = max(max_param_delta, float(delta))

    with torch.no_grad():
        forward_rebuilt = bool(torch.allclose(probe(x), reloaded(x), atol=PARITY_TOL, rtol=0.0))

    loss_a = _step(probe, opt)
    loss_b = _step(reloaded, opt2)
    max_loss_delta = abs(loss_a - loss_b)

    del probe, reloaded, opt, opt2, x, y
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    roundtrip = bool(forward_rebuilt and max_param_delta <= PARITY_TOL)
    return {
        "state_dict_roundtrip_passed": roundtrip,
        "resume_next_step_parity_passed": bool(max_loss_delta <= PARITY_TOL),
        "fp32_master_params_preserved": bool(max_param_delta <= PARITY_TOL),
        "quantized_forward_rebuilt": forward_rebuilt,
        "max_loss_delta": float(max_loss_delta),
        "max_param_delta": float(max_param_delta),
        "max_allowed_loss_delta": PARITY_TOL,
        "max_allowed_param_delta": PARITY_TOL,
    }


def _quality_drift(base_eval: torch.Tensor, cand_eval: torch.Tensor) -> float:
    denom = float(base_eval.norm().item())
    if denom <= 0.0:
        return 0.0
    return float((cand_eval - base_eval).norm().item() / denom)


def _result_summary(
    baseline: ArmResult,
    candidate: ArmResult,
    base_eval: torch.Tensor,
    cand_eval: torch.Tensor,
    optimizer_state_parity: bool,
) -> dict[str, Any]:
    return {
        "case_id": candidate.arm,
        "baseline_step_time_ms": float(baseline.mean_step_ms),
        "candidate_step_time_ms": float(candidate.mean_step_ms),
        "baseline_peak_vram_mb": float(baseline.peak_vram_mb),
        "candidate_peak_vram_mb": float(candidate.peak_vram_mb),
        "baseline_energy_per_step_j": float(baseline.energy_per_step_j or 0.0),
        "candidate_energy_per_step_j": float(candidate.energy_per_step_j or 0.0),
        "quality_drift": _quality_drift(base_eval, cand_eval),
        "loss_delta": abs(float(candidate.final_loss) - float(baseline.final_loss)),
        "optimizer_state_parity": bool(optimizer_state_parity),
    }


def _evidence_policy() -> dict[str, Any]:
    return {
        "owner": "lulynx",
        "review_id": "cdm_qta_p4_real_gpu_ab",
        "evidence_scope": "cdm_qta_lora_runtime_activation_review",
        "baseline_case_ref": "baseline_fp32_lora",
        "candidate_case_ref": "qta_fake_quant_lora",
        "rollback_plan": "remain default-off report-only probe; no trainer wiring, no request fields",
        "required_outputs": [
            "baseline_metrics",
            "candidate_metrics",
            "quality_report",
            "loss_report",
            "optimizer_state_report",
            "energy_report",
        ],
        "required_metrics": [
            "step_time_ms",
            "peak_vram_mb",
            "energy_per_step_j",
            "quality_drift",
            "loss_delta",
            "optimizer_state_parity",
        ],
        "thresholds": {},
        "report_only": True,
        "manual_only": True,
        "acknowledge_no_ab_execution": True,
        "requires_later_ab_result_ingestion": True,
        "ab_execution_allowed": False,
        "ab_dispatch_allowed": False,
        "trainer_wiring_allowed": False,
    }


def evaluate_gates(
    *,
    dims: Mapping[str, int],
    arms_results: Mapping[str, ArmResult],
    eval_outputs: Mapping[str, torch.Tensor],
    parity_reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Pipe the real A/B products through the existing default-off cdm_qta gates."""
    probe_scorecard = build_cdm_qta_lora_quant_train_scorecard(
        CDMQTALoraProbeConfig(
            enabled=True,
            in_features=int(dims["in_features"]),
            out_features=int(dims["out_features"]),
            rank=int(dims["rank"]),
            quant_bits=8,
        )
    )
    contract = build_cdm_qta_lora_optimizer_state_contract(
        probe_scorecard=probe_scorecard,
        optimizer_capability={
            "fp32_master_params": True,
            "optimizer_state_tracks_master_params": True,
            "state_dict_roundtrip_supported": True,
            "resume_parity_required": True,
            "quantized_optimizer_state_required": False,
        },
    )
    parity_gates = {
        arm: build_cdm_qta_lora_optimizer_state_parity_gate(
            optimizer_state_contract=contract, parity_report=report
        )
        for arm, report in parity_reports.items()
    }

    baseline = arms_results["baseline"]
    base_eval = eval_outputs["baseline"]
    summaries = [
        _result_summary(baseline, arms_results[arm], base_eval, eval_outputs[arm], parity_gates[arm]["ok"])
        for arm in arms_results
        if arm != "baseline"
    ]

    package = build_cdm_qta_lora_ab_evidence_package(
        probe_scorecard=probe_scorecard, evidence_policy=_evidence_policy()
    )
    ingestion_default = build_cdm_qta_lora_ab_result_ingestion(
        evidence_package=package, result_summaries=summaries, thresholds=None
    )
    ingestion_quality = build_cdm_qta_lora_ab_result_ingestion(
        evidence_package=package, result_summaries=summaries, thresholds=QUALITY_THRESHOLDS
    )
    return {
        "probe_scorecard": probe_scorecard,
        "optimizer_state_contract": contract,
        "optimizer_state_parity_gates": parity_gates,
        "result_summaries": summaries,
        "evidence_package": package,
        "ingestion_default_thresholds": ingestion_default,
        "ingestion_quality_thresholds": ingestion_quality,
        "memory_estimate": estimate_cdm_qta_lora_memory(
            CDMQTALoraProbeConfig(
                enabled=True,
                in_features=int(dims["in_features"]),
                out_features=int(dims["out_features"]),
                rank=int(dims["rank"]),
                quant_bits=8,
            )
        ),
    }


def _blocker_disposition(gates: Mapping[str, Any], energy_metered: bool) -> dict[str, str]:
    parity_all_ok = bool(gates["optimizer_state_parity_gates"]) and all(
        g["ok"] for g in gates["optimizer_state_parity_gates"].values()
    )
    quality_gate_present = "quality_drift" in (gates["evidence_package"].get("required_metrics") or [])
    return {
        "optimizer_state_parity_missing": "cleared" if parity_all_ok else "stands",
        "quality_drift_gate_missing": "cleared" if quality_gate_present else "stands",
        "energy_metering_missing": "cleared" if energy_metered else "stands",
        "real_anima_newbie_training_ab_missing": "stands (operator real-model job)",
    }


def _resolve_out_path(args: argparse.Namespace) -> Path:
    if args.json:
        return Path(args.json)
    root = Path(args.output_root) if args.output_root else Path(__file__).resolve().parents[3] / ".runs" / "cdm_qta_lora_ab"
    root.mkdir(parents=True, exist_ok=True)
    tag = "-".join(args.arms) if args.arms else "all"
    return root / f"cdm_qta_lora_ab_{tag}_s{args.steps}.json"


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="*", default=None, choices=list(DEFAULT_ARMS))
    parser.add_argument("--in-features", type=int, default=3072)
    parser.add_argument("--out-features", type=int, default=3072)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", default="")
    parser.add_argument("--json", default="")
    args = parser.parse_args(argv)

    want_cuda = args.device == "cuda"
    device = torch.device("cuda" if (want_cuda and torch.cuda.is_available()) else ("cpu" if want_cuda else args.device))

    arms = tuple(args.arms) if args.arms else DEFAULT_ARMS
    if "baseline" not in arms:
        arms = ("baseline",) + arms
    dims = {"in_features": args.in_features, "out_features": args.out_features, "rank": args.rank}

    sampler = _PowerSampler(device)
    _warmup_device(dims, args.tokens, device, args.seed)
    eval_x, _ = _fixed_batch(dims, args.seed + 7, args.tokens, device)

    arms_results: dict[str, ArmResult] = {}
    eval_outputs: dict[str, torch.Tensor] = {}
    parity_reports: dict[str, dict[str, Any]] = {}
    for arm in arms:
        result, eval_out = run_arm(
            arm,
            dims=dims,
            tokens=args.tokens,
            steps=args.steps,
            warmup=args.warmup,
            seed=args.seed,
            device=device,
            lr=args.lr,
            sampler=sampler,
            eval_x=eval_x,
        )
        arms_results[arm] = result
        eval_outputs[arm] = eval_out
        if ARM_SPECS[arm]["enabled"]:
            parity_reports[arm] = optimizer_state_parity_subrun(
                bits=int(ARM_SPECS[arm]["quant_bits"]),
                dims=dims,
                tokens=min(args.tokens, 512),
                device=device,
                seed=args.seed,
                lr=args.lr,
            )
    sampler.shutdown()

    gates = evaluate_gates(
        dims=dims, arms_results=arms_results, eval_outputs=eval_outputs, parity_reports=parity_reports
    )
    disposition = _blocker_disposition(gates, energy_metered=sampler.available)

    payload = {
        "schema_version": 1,
        "benchmark": "cdm_qta_lora_real_gpu_ab_v0",
        "device": str(device),
        "energy_metered": bool(sampler.available),
        "dims": dims,
        "steps": args.steps,
        "warmup": args.warmup,
        "arms": {arm: asdict(res) for arm, res in arms_results.items()},
        "optimizer_state_parity_reports": parity_reports,
        "gates": gates,
        "blocker_disposition": disposition,
    }
    out_path = _resolve_out_path(args)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    _print_verdict(payload, out_path)
    return 0


def _print_verdict(payload: Mapping[str, Any], out_path: Path) -> None:
    arms = payload["arms"]
    base = arms.get("baseline", {})
    print(f"\n=== CDM-QTA LoRA real-GPU A/B ({payload['device']}, energy_metered={payload['energy_metered']}) ===")
    print(f"dims={payload['dims']} steps={payload['steps']}")
    for arm, res in arms.items():
        e = res.get("energy_per_step_j")
        e_str = f"{e:.3f}J" if isinstance(e, (int, float)) else "n/a"
        print(
            f"  {arm:9s} step={res['mean_step_ms']:.3f}ms vram={res['peak_vram_mb']:.1f}MB "
            f"energy/step={e_str} final_loss={res['final_loss']:.5f}"
        )
    g = payload["gates"]
    quality_ok = {row["case_id"]: row["ok"] for row in g["ingestion_quality_thresholds"]["result_rows"]}
    for row in g["result_summaries"]:
        base_step = base.get("mean_step_ms") or 0.0
        cand_step = row["candidate_step_time_ms"]
        step_pct = 0.0 if base_step <= 0 else (cand_step - base_step) / base_step * 100.0
        verdict = "PASS" if quality_ok.get(row["case_id"]) else "FAIL"
        print(
            f"  -> {row['case_id']:5s} quality_drift={row['quality_drift']:.5f} "
            f"loss_delta={row['loss_delta']:.5f} opt_state_parity={row['optimizer_state_parity']} "
            f"step_time {step_pct:+.1f}% vs baseline | quality_gate={verdict}"
        )
    parity_ok = all(x["ok"] for x in g["optimizer_state_parity_gates"].values()) if g["optimizer_state_parity_gates"] else False
    print(f"  optimizer_state_parity_gate: {'PASS' if parity_ok else 'FAIL'}")
    print(f"  ingestion (default speed thresholds): ok={g['ingestion_default_thresholds']['ok']}")
    print(f"  ingestion (quality thresholds):       ok={g['ingestion_quality_thresholds']['ok']}")
    mem = g["memory_estimate"]
    print(f"  training-memory ratio (qta/baseline) = {mem['estimated_training_memory_ratio']:.3f} (>1 => no training VRAM win)")
    print("  blocker disposition:")
    for name, state in payload["blocker_disposition"].items():
        print(f"    - {name}: {state}")
    print(f"  bundle: {out_path}")


if __name__ == "__main__":
    raise SystemExit(main())
