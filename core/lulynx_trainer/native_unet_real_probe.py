"""Real-weight probes for the Warehouse SDXL native UNet.

This script is intentionally separate from ``native_unet_smoke.py`` because it
loads the full mapped SDXL UNet.  It validates the end-to-end native skeleton
with real weights and checks that gradient checkpointing preserves forward and
input-gradient results on the frozen-base training path.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from core.lulynx_trainer.native_unet import build_sdxl_unet_compat_from_manifest


def _manifest_path() -> Path:
    return Path(__file__).resolve().parent / "native_unet" / "keymaps" / "sdxl_unet_keymap_manifest.json"


def _default_model_path(manifest_path: Path) -> Path | None:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = manifest.get("expected_local_model")
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[3] / path
    return path


def _resolve_device(value: str) -> torch.device:
    requested = str(value or "auto").strip().lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _resolve_dtype(value: str, device: torch.device) -> torch.dtype:
    requested = str(value or "auto").strip().lower()
    if requested == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "half": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if requested not in aliases:
        raise ValueError(f"unsupported dtype: {value}")
    return aliases[requested]


def _parse_int_list(value: str, *, default: list[int], minimum: int = 1) -> list[int]:
    text = str(value or "").strip()
    if not text:
        return list(default)
    parsed: list[int] = []
    seen: set[int] = set()
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        item = max(int(chunk), minimum)
        if item not in seen:
            parsed.append(item)
            seen.add(item)
    return parsed or list(default)


def _cuda_snapshot() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {}
    try:
        torch.cuda.synchronize()
    except Exception:
        pass
    return {
        "allocated_mb": round(float(torch.cuda.memory_allocated()) / (1024 * 1024), 1),
        "reserved_mb": round(float(torch.cuda.memory_reserved()) / (1024 * 1024), 1),
        "peak_allocated_mb": round(float(torch.cuda.max_memory_allocated()) / (1024 * 1024), 1),
        "peak_reserved_mb": round(float(torch.cuda.max_memory_reserved()) / (1024 * 1024), 1),
    }


def _make_inputs(
    *,
    latent_size: int,
    encoder_tokens: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    sample = torch.randn(1, 4, latent_size, latent_size, generator=generator, dtype=torch.float32)
    encoder = torch.randn(1, encoder_tokens, 2048, generator=generator, dtype=torch.float32)
    text_embeds = torch.randn(1, 1280, generator=generator, dtype=torch.float32)
    time_ids = torch.tensor([[latent_size * 8, latent_size * 8, 0, 0, latent_size * 8, latent_size * 8]], dtype=torch.float32)
    return {
        "sample": sample.to(device=device, dtype=dtype).requires_grad_(True),
        "encoder_hidden_states": encoder.to(device=device, dtype=dtype).requires_grad_(True),
        "text_embeds": text_embeds.to(device=device, dtype=dtype).requires_grad_(True),
        "time_ids": time_ids.to(device=device, dtype=dtype),
    }


def _run_once(
    model: torch.nn.Module,
    *,
    checkpointing: bool,
    latent_size: int,
    encoder_tokens: int,
    timestep: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
) -> dict[str, Any]:
    if checkpointing:
        model.enable_gradient_checkpointing()
    else:
        model.disable_gradient_checkpointing()
    model.train()
    model.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    inputs = _make_inputs(
        latent_size=latent_size,
        encoder_tokens=encoder_tokens,
        device=device,
        dtype=dtype,
        seed=seed,
    )
    started = time.perf_counter()
    output = model(
        sample=inputs["sample"],
        timestep=torch.tensor([int(timestep)], device=device),
        encoder_hidden_states=inputs["encoder_hidden_states"],
        added_cond_kwargs={
            "text_embeds": inputs["text_embeds"],
            "time_ids": inputs["time_ids"],
        },
    ).sample
    loss = output.float().square().mean()
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return {
        "checkpointing": bool(checkpointing),
        "elapsed_ms": round(elapsed_ms, 2),
        "output": output.detach().float().cpu(),
        "sample_grad": inputs["sample"].grad.detach().float().cpu(),
        "encoder_grad": inputs["encoder_hidden_states"].grad.detach().float().cpu(),
        "text_grad": inputs["text_embeds"].grad.detach().float().cpu(),
        "loss": float(loss.detach().cpu()),
        "cuda": _cuda_snapshot(),
    }


def _load_reference_unet(model_path: Path, *, device: torch.device, dtype: torch.dtype) -> torch.nn.Module:
    from core.lulynx_trainer.single_file_loader import load_sdxl_single_file_components

    components = load_sdxl_single_file_components(model_path, torch_dtype=dtype)
    unet = components["unet"]
    unet.to(device=device, dtype=dtype)
    return unet


def _free_cuda() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    except Exception:
        pass


def _max_abs_delta(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right).abs().max().item())


def _compare_runs(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return {
        "output_max_abs": _max_abs_delta(left["output"], right["output"]),
        "sample_grad_max_abs": _max_abs_delta(left["sample_grad"], right["sample_grad"]),
        "encoder_grad_max_abs": _max_abs_delta(left["encoder_grad"], right["encoder_grad"]),
        "text_grad_max_abs": _max_abs_delta(left["text_grad"], right["text_grad"]),
    }


def _public_run_summary(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "elapsed_ms": run["elapsed_ms"],
        "loss": run["loss"],
        "cuda": run["cuda"],
    }


def _rounded_deltas(deltas: dict[str, float]) -> dict[str, float]:
    return {key: round(float(value), 8) for key, value in deltas.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe real-weight SDXL native UNet forward/backward parity.")
    parser.add_argument("--manifest", default=str(_manifest_path()))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--compare-reference", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--latent-size", type=int, default=8)
    parser.add_argument("--latent-sizes", default="")
    parser.add_argument("--encoder-tokens", type=int, default=4)
    parser.add_argument("--encoder-token-list", default="")
    parser.add_argument("--timesteps", default="")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--rtol", type=float, default=2e-2)
    parser.add_argument("--atol", type=float, default=2e-2)
    parser.add_argument("--reference-rtol", type=float, default=None)
    parser.add_argument("--reference-atol", type=float, default=None)
    parser.add_argument("--json", default="")
    args = parser.parse_args()

    device = _resolve_device(args.device)
    dtype = _resolve_dtype(args.dtype, device)
    manifest = Path(args.manifest)
    model_path = Path(args.model_path) if args.model_path else _default_model_path(manifest)
    if args.compare_reference and model_path is None:
        raise ValueError("--compare-reference requires --model-path or manifest expected_local_model")
    started = time.perf_counter()
    model = build_sdxl_unet_compat_from_manifest(
        manifest,
        str(model_path) if model_path is not None else None,
        device=device,
        dtype=dtype,
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    load_elapsed_ms = (time.perf_counter() - started) * 1000.0

    latent_sizes = _parse_int_list(args.latent_sizes, default=[max(int(args.latent_size), 8)], minimum=8)
    encoder_token_list = _parse_int_list(args.encoder_token_list, default=[max(int(args.encoder_tokens), 1)], minimum=1)
    timestep_list = _parse_int_list(args.timesteps, default=[1], minimum=0)
    native_payloads: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []
    for latent_size in latent_sizes:
        for encoder_tokens in encoder_token_list:
            for timestep in timestep_list:
                case_seed = int(args.seed) + int(latent_size) * 1000 + int(encoder_tokens) * 10 + int(timestep)
                baseline = _run_once(
                    model,
                    checkpointing=False,
                    latent_size=latent_size,
                    encoder_tokens=encoder_tokens,
                    timestep=timestep,
                    device=device,
                    dtype=dtype,
                    seed=case_seed,
                )
                checkpointed = _run_once(
                    model,
                    checkpointing=True,
                    latent_size=latent_size,
                    encoder_tokens=encoder_tokens,
                    timestep=timestep,
                    device=device,
                    dtype=dtype,
                    seed=case_seed,
                )
                deltas = _compare_runs(baseline, checkpointed)
                case_ok = all(value <= float(args.atol) for value in deltas.values())
                case = {
                    "ok": bool(case_ok),
                    "latent_size": int(latent_size),
                    "encoder_tokens": int(encoder_tokens),
                    "timestep": int(timestep),
                    "seed": int(case_seed),
                    "baseline": _public_run_summary(baseline),
                    "checkpointed": _public_run_summary(checkpointed),
                    "deltas": _rounded_deltas(deltas),
                }
                cases.append(case)
                native_payloads.append(
                    {
                        "case": case,
                        "output": baseline["output"],
                        "sample_grad": baseline["sample_grad"],
                        "encoder_grad": baseline["encoder_grad"],
                        "text_grad": baseline["text_grad"],
                    }
                )
    ok = all(bool(case["ok"]) for case in cases)
    first_case = cases[0]
    report = {
        "ok": bool(ok),
        "device": str(device),
        "dtype": str(dtype),
        "manifest": str(manifest),
        "model_path": str(model_path) if model_path is not None else None,
        "case_count": len(cases),
        "latent_sizes": [int(value) for value in latent_sizes],
        "encoder_token_list": [int(value) for value in encoder_token_list],
        "timesteps": [int(value) for value in timestep_list],
        "load_elapsed_ms": round(load_elapsed_ms, 2),
        "cases": cases,
        "tolerance": {"rtol": float(args.rtol), "atol": float(args.atol)},
    }
    if len(cases) == 1:
        report.update(
            {
                "latent_size": first_case["latent_size"],
                "encoder_tokens": first_case["encoder_tokens"],
                "timestep": first_case["timestep"],
                "baseline": first_case["baseline"],
                "checkpointed": first_case["checkpointed"],
                "deltas": first_case["deltas"],
            }
        )
    if args.compare_reference:
        del model
        _free_cuda()
        ref_started = time.perf_counter()
        reference_unet = _load_reference_unet(model_path, device=device, dtype=dtype)  # type: ignore[arg-type]
        for parameter in reference_unet.parameters():
            parameter.requires_grad_(False)
        reference_load_ms = (time.perf_counter() - ref_started) * 1000.0
        reference_atol = float(args.reference_atol) if args.reference_atol is not None else (5e-2 if dtype in {torch.bfloat16, torch.float16} else float(args.atol))
        reference_rtol = float(args.reference_rtol) if args.reference_rtol is not None else float(args.rtol)
        reference_cases: list[dict[str, Any]] = []
        for payload in native_payloads:
            case_meta = payload["case"]
            reference = _run_once(
                reference_unet,
                checkpointing=False,
                latent_size=int(case_meta["latent_size"]),
                encoder_tokens=int(case_meta["encoder_tokens"]),
                timestep=int(case_meta["timestep"]),
                device=device,
                dtype=dtype,
                seed=int(case_meta["seed"]),
            )
            reference_deltas = _compare_runs(payload, reference)
            reference_ok = all(value <= reference_atol for value in reference_deltas.values())
            reference_case = {
                "ok": bool(reference_ok),
                "latent_size": int(case_meta["latent_size"]),
                "encoder_tokens": int(case_meta["encoder_tokens"]),
                "timestep": int(case_meta["timestep"]),
                "elapsed_ms": reference["elapsed_ms"],
                "loss": reference["loss"],
                "cuda": reference["cuda"],
                "native_vs_reference_deltas": _rounded_deltas(reference_deltas),
            }
            reference_cases.append(reference_case)
            case_meta["reference"] = reference_case
            case_meta["ok"] = bool(case_meta["ok"] and reference_ok)
        reference_ok = all(bool(case["ok"]) for case in reference_cases)
        report["reference"] = {
            "load_elapsed_ms": round(reference_load_ms, 2),
            "cases": reference_cases,
            "tolerance": {"rtol": reference_rtol, "atol": reference_atol},
        }
        if len(reference_cases) == 1:
            report["reference"].update(
                {
                    "elapsed_ms": reference_cases[0]["elapsed_ms"],
                    "loss": reference_cases[0]["loss"],
                    "cuda": reference_cases[0]["cuda"],
                }
            )
            report["native_vs_reference_deltas"] = reference_cases[0]["native_vs_reference_deltas"]
        report["ok"] = bool(report["ok"] and reference_ok)
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if bool(report["ok"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

