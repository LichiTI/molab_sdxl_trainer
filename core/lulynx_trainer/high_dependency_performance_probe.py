# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""High-dependency experimental performance probe for roadmap item 5.

This helper is intentionally report-only. It checks optional runtime features and
runs tiny bounded probes for FP8/TransformerEngine, image decode backends, and
``torch.compile`` reduce-overhead viability without changing the training path.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List

import torch
from PIL import Image

if __package__ in (None, ""):
    backend_root = Path(__file__).resolve().parents[2]
    repo_root = Path(__file__).resolve().parents[3]
    for import_root in (repo_root, backend_root):
        if str(import_root) not in sys.path:
            sys.path.insert(0, str(import_root))

from core.lulynx_trainer.compile_probe import evaluate_compile_probe  # noqa: E402
from core.lulynx_trainer.dataset_loader import CaptionDataset  # noqa: E402
from core.lulynx_trainer.fp8_te_profile import build_fp8_te_profile  # noqa: E402


def _package_available(name: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"available": False, "version": "", "error": ""}
    try:
        spec = importlib.util.find_spec(name)
    except ModuleNotFoundError as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    if spec is None:
        return result
    result["available"] = True
    try:
        module = __import__(name)
        result["version"] = str(getattr(module, "__version__", "unknown"))
    except Exception as exc:
        result["available"] = False
        result["error"] = f"{type(exc).__name__}: {exc}"
    return result

def _torch_environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "torch_cuda_version": str(getattr(torch.version, "cuda", None)),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_compile_available": hasattr(torch, "compile"),
    }
    try:
        import torch._inductor.config as inductor_config  # type: ignore

        env["inductor_fx_graph_cache"] = bool(getattr(inductor_config, "fx_graph_cache", False))
        env["inductor_fx_graph_remote_cache"] = bool(getattr(inductor_config, "fx_graph_remote_cache", False))
    except Exception as exc:
        env["inductor_config_error"] = f"{type(exc).__name__}: {exc}"
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
        env.update(
            {
                "cuda_device_index": idx,
                "cuda_device_name": props.name,
                "compute_capability": f"{props.major}.{props.minor}",
                "total_vram_gb": round(total_bytes / (1024 ** 3), 3),
                "free_vram_gb": round(free_bytes / (1024 ** 3), 3),
            }
        )
    return env


def _feature_capabilities() -> Dict[str, Any]:
    capabilities: Dict[str, Any] = {
        "bitsandbytes": _package_available("bitsandbytes"),
        "xformers": _package_available("xformers"),
        "flash_attn": _package_available("flash_attn"),
        "torchvision": _package_available("torchvision"),
        "nvidia_dali": _package_available("nvidia.dali"),
        "kornia": _package_available("kornia"),
        "webdataset": _package_available("webdataset"),
        "transformer_engine": _package_available("transformer_engine"),
    }
    try:
        import torch.nn.attention.flex_attention as flex_attention  # type: ignore

        capabilities["flex_attention"] = {
            "available": True,
            "has_flex_attention": hasattr(flex_attention, "flex_attention"),
            "has_create_block_mask": hasattr(flex_attention, "create_block_mask"),
        }
    except Exception as exc:
        capabilities["flex_attention"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    try:
        import torch.distributed as dist

        capabilities["distributed"] = {
            "available": bool(dist.is_available()),
            "nccl_available": bool(dist.is_nccl_available()) if dist.is_available() else False,
            "gloo_available": bool(dist.is_gloo_available()) if dist.is_available() else False,
        }
    except Exception as exc:
        capabilities["distributed"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return capabilities


def _timed(label: str, fn: Callable[[], Any]) -> Dict[str, Any]:
    started = time.perf_counter()
    try:
        value = fn()
        return {"label": label, "ok": True, "seconds": round(time.perf_counter() - started, 6), "value": value}
    except Exception as exc:
        return {
            "label": label,
            "ok": False,
            "seconds": round(time.perf_counter() - started, 6),
            "error": f"{type(exc).__name__}: {exc}",
        }


def _write_probe_images(root: Path, count: int, size: int) -> List[Path]:
    paths: List[Path] = []
    for idx in range(max(int(count), 1)):
        path = root / f"sample_{idx:03d}.png"
        image = Image.new("RGBA", (size, size), (idx * 31 % 255, idx * 17 % 255, idx * 7 % 255, 128 + idx % 127))
        image.save(path)
        path.with_suffix(".txt").write_text("decode probe", encoding="utf-8")
        paths.append(path)
    return paths


def _decode_backend_probe(*, image_count: int = 4, repeats: int = 2, size: int = 128) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        image_paths = _write_probe_images(root, image_count, size)
        results: Dict[str, Any] = {"image_count": len(image_paths), "repeats": max(int(repeats), 1), "size": size, "backends": {}}
        for backend in ("pil", "pil_lru", "torchvision_cpu"):
            def run_backend(backend_name: str = backend) -> Dict[str, Any]:
                dataset = CaptionDataset(
                    data_dir=str(root),
                    resolution=size,
                    enable_bucket=False,
                    alpha_mask=True,
                    image_decode_backend=backend_name,
                    image_decode_cache_size=8 if backend_name == "pil_lru" else 0,
                )
                pixels = []
                for _ in range(max(int(repeats), 1)):
                    for path in image_paths:
                        image, alpha = dataset._load_image_rgb_alpha(str(path), need_alpha=True)
                        pixels.append(image.getpixel((0, 0)))
                        if alpha is not None:
                            pixels.append(alpha.getpixel((0, 0)))
                return {
                    "resolved_backend": dataset.image_decode_backend,
                    "cache_hits": int(getattr(dataset, "_image_decode_cache_hits", 0)),
                    "cache_misses": int(getattr(dataset, "_image_decode_cache_misses", 0)),
                    "sampled_values": len(pixels),
                }

            results["backends"][backend] = _timed(backend, run_backend)
        dali_package = _package_available("nvidia.dali")
        results["dali"] = {
            "available": bool(dali_package.get("available", False)),
            "package": dali_package,
            "status": "profile_only_not_wired",
        }
        kornia_package = _package_available("kornia")
        results["kornia"] = {
            "available": bool(kornia_package.get("available", False)),
            "package": kornia_package,
            "status": "profile_only_not_wired",
            "training_integration": "not_wired",
        }
        webdataset_package = _package_available("webdataset")
        results["webdataset"] = {
            "available": bool(webdataset_package.get("available", False)),
            "package": webdataset_package,
            "status": "materialized_adapter_only" if webdataset_package.get("available", False) else "profile_only_not_wired",
            "training_integration": "not_wired_to_streaming_pipeline",
        }
        return results


def _fp8_probe() -> Dict[str, Any]:
    profile = build_fp8_te_profile(SimpleNamespace(precision_experiment="fp8_te")).as_dict()
    dtype = getattr(torch, "float8_e4m3fn", None)
    smoke: Dict[str, Any] = {"torch_float8_dtype_available": dtype is not None, "ok": False}
    if dtype is not None:
        def roundtrip() -> Dict[str, Any]:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            tensor = torch.randn(16, 16, device=device, dtype=torch.float32)
            restored = tensor.to(dtype=dtype).to(dtype=torch.float32)
            return {
                "device": str(device),
                "max_abs_error": float((tensor - restored).abs().max().item()),
                "mean_abs_error": float((tensor - restored).abs().mean().item()),
            }

        timed = _timed("torch_float8_roundtrip", roundtrip)
        smoke.update(timed)
        smoke["ok"] = bool(timed.get("ok", False))
    else:
        smoke["reason"] = "torch.float8_e4m3fn is unavailable"
    return {
        "profile": profile,
        "storage_roundtrip_smoke": smoke,
        "training_integration": "not_wired",
        "default_recommendation": "keep_bf16",
    }


def _compile_probe(*, device_arg: str = "auto", iters: int = 8, warmup: int = 2) -> Dict[str, Any]:
    if not hasattr(torch, "compile"):
        return {"ok": False, "available": False, "reason": "torch.compile is unavailable"}
    device = torch.device("cuda" if device_arg == "auto" and torch.cuda.is_available() else ("cpu" if device_arg == "auto" else device_arg))
    dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    torch.manual_seed(1234)
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.SiLU(),
        torch.nn.Linear(256, 128),
    ).to(device=device, dtype=dtype)
    x = torch.randn(8, 128, device=device, dtype=dtype)

    def run_module(module: torch.nn.Module, loops: int) -> float:
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            for _ in range(max(int(loops), 1)):
                y = module(x)
                y = y + 1.0
        if device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - started

    try:
        eager_seconds = run_module(model, max(int(iters), 1))
        compiled = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=False)
        run_module(compiled, max(int(warmup), 0) or 1)
        compiled_seconds = run_module(compiled, max(int(iters), 1))
        decision = evaluate_compile_probe(
            route="high_dependency_probe",
            target="tiny_mlp_reduce_overhead",
            eager_seconds=eager_seconds,
            compiled_seconds=compiled_seconds,
            min_speedup_ratio=0.0,
            max_vram_increase_ratio=1.0,
        )
        return {
            "ok": True,
            "available": True,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "mode": "reduce-overhead",
            "iters": max(int(iters), 1),
            "warmup": max(int(warmup), 0),
            "decision": decision.decision,
            "reason": decision.reason,
            "eager_seconds": round(eager_seconds, 6),
            "compiled_seconds": round(compiled_seconds, 6),
            "speedup_ratio": round(decision.speedup_ratio, 6),
            "training_integration": "not_wired",
        }
    except Exception as exc:
        return {
            "ok": False,
            "available": False,
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "mode": "reduce-overhead",
            "reason": "probe_failed",
            "error": f"{type(exc).__name__}: {exc}",
            "training_integration": "not_wired",
        }


def build_report(args: argparse.Namespace) -> Dict[str, Any]:
    report = {
        "probe": "high_dependency_performance_probe",
        "scope": "roadmap_item_5_1_to_5_4",
        "non_invasive": True,
        "ok": True,
        "environment": _torch_environment(),
        "capabilities": _feature_capabilities(),
        "fp8_transformer_engine": _fp8_probe(),
        "data_decode": _decode_backend_probe(image_count=args.images, repeats=args.decode_repeats, size=args.image_size),
        "torch_compile_reduce_overhead": _compile_probe(device_arg=args.device, iters=args.compile_iters, warmup=args.compile_warmup)
        if not args.skip_compile
        else {"ok": False, "skipped": True, "reason": "--skip-compile"},
        "notes": [
            "This probe does not mutate training defaults or launcher behavior.",
            "FP8/TE, DALI, Kornia, WebDataset streaming, and torch.compile remain experimental until real model quality/throughput A-B validates them.",
            "Data decode probes are report-only; pil_lru and torchvision_cpu are explicit choices, while DALI/Kornia/WebDataset streaming do not replace the training data path here.",
            "Launcher/startup explicit parameters should keep priority over these optional strategies.",
        ],
    }
    hard_sections = ["environment", "capabilities", "fp8_transformer_engine", "data_decode"]
    report["ok"] = all(bool(report.get(name)) for name in hard_sections)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="temp/high_dependency_performance_probe.json")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--images", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--decode-repeats", type=int, default=2)
    parser.add_argument("--compile-iters", type=int, default=8)
    parser.add_argument("--compile-warmup", type=int, default=2)
    parser.add_argument("--skip-compile", action="store_true")
    args = parser.parse_args(argv)

    report = build_report(args)
    root = Path(__file__).resolve().parents[3]
    out = Path(args.out)
    if not out.is_absolute():
        out = root / out
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    out.write_text(text, encoding="utf-8")
    print(text)
    return 0 if report.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())

