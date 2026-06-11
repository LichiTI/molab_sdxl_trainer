"""Route/runtime E2E smoke for Anima full-finetune online_cache.

This smoke keeps the product boundary intact: TrainingRouteService writes the
run config and entry_train.py executes it.  The dataset is copied to a clean
temp folder without prebuilt Anima cache files so the frozen VAE/Qwen3 cache
builder must run before the one-step DiT training path can consume the sample.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    project_root = Path(__file__).resolve().parents[3]
    backend_root = project_root / "backend"
    for item in (project_root, backend_root):
        if str(item) not in sys.path:
            sys.path.insert(0, str(item))

from backend.core.lulynx_trainer.run_manifest import manifest_path_for
from backend.lulynx_launcher.services.training_registry import LulynxTrainingRegistry
from backend.lulynx_launcher.services.training_route_service import TrainingRouteService


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _tail(value: str, limit: int = 5000) -> str:
    return value if len(value) <= limit else value[-limit:]


def _copy_one_raw_pair(source_dir: Path, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    for image_path in sorted(source_dir.iterdir()):
        if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            continue
        caption_path = image_path.with_suffix(".txt")
        if not caption_path.is_file():
            continue
        copied_image = target_dir / image_path.name
        shutil.copy2(image_path, copied_image)
        shutil.copy2(caption_path, target_dir / caption_path.name)
        return copied_image
    raise FileNotFoundError(f"No raw image/caption pair found under {source_dir}")


def _write_worker_logs(run_dir: Path, proc: subprocess.CompletedProcess[str]) -> None:
    _write_text(run_dir / "worker_stdout.log", proc.stdout or "")
    _write_text(run_dir / "worker_stderr.log", proc.stderr or "")


def _write_summary(run_dir: Path | None, output_dir: Path, summary: dict) -> None:
    if run_dir is not None:
        _write_json(run_dir / "online_cache_route_e2e_summary.json", summary)
    _write_json(output_dir / "online_cache_route_e2e_summary.json", summary)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    backend_root = repo_root / "backend"
    model_path = repo_root / "models" / "anima" / "diffusion_models" / "anima-preview2.safetensors"
    qwen3_path = repo_root / "models" / "anima" / "text_encoders" / "qwen_3_06b_base.safetensors"
    vae_path = repo_root / "models" / "anima" / "vae" / "qwen_image_vae.safetensors"
    source_dir = repo_root / "sucai" / "6_lulu"
    for path in (model_path, qwen3_path, vae_path, source_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    run_id = f"anima-full-online-cache-e2e-{int(time.time() * 1000)}"
    output_dir = backend_root / "tmp" / "anima_online_cache_e2e" / run_id
    data_dir = output_dir / "data"
    logging_dir = output_dir / "logs"
    run_dir: Path | None = None
    route_service: TrainingRouteService | None = None
    proc: subprocess.CompletedProcess[str] | None = None
    copied_image = _copy_one_raw_pair(source_dir, data_dir)

    try:
        registry = LulynxTrainingRegistry.default()
        schema = registry.get_by_id("anima-finetune")
        if schema is None:
            raise RuntimeError("anima-finetune schema is not registered")
        route_service = TrainingRouteService(repo_root, backend_root)
        route = route_service.resolve("anima-finetune")
        if not route.is_known:
            raise RuntimeError("anima-finetune route is not known")

        config = route_service.build_config_json(
            schema=schema,
            route=route,
            config_values={
                "pretrained_model": str(model_path),
                "qwen3": str(qwen3_path),
                "vae": str(vae_path),
                "output_name": "lulynx_anima_full_finetune_online_cache_route_e2e",
                "logging_dir": str(logging_dir),
                "device": "cuda",
                "mixed_precision": "bf16",
                "optimizer_type": "AdamW",
                "optimizer_backend": "torch_adamw",
                "learning_rate": 1e-6,
                "weight_decay": 0.0,
                "max_train_epochs": 1,
                "max_train_steps": 1,
                "train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "gradient_checkpointing": False,
                "anima_block_checkpointing": False,
                "native_cache_mode": "online_cache",
                "anima_cached_training": True,
                "anima_native_block_count": 1,
                "anima_qwen3_max_token_length": 16,
                "anima_text_token_limit": 16,
                "anima_cached_text_token_limit": 16,
                "anima_cached_latent_crop_size": 4,
                "native_runtime_profile": "standard",
                "torch_compile": False,
                "compile_runtime": "off",
                "compile_probe_enabled": False,
                "enable_fixed_token_padding": False,
                "cached_dataloader_workers": 0,
                "cached_dataloader_pin_memory": False,
                "pin_memory": False,
                "dataloader_num_workers": 0,
                "save_every_n_epochs": 1,
                "save_model_as": "safetensors",
                "no_metadata": False,
            },
            output_dir=str(output_dir),
            train_data_dir=str(data_dir),
        )
        config["execution_profile_id"] = "flashattention-online-cache-smoke"

        run_dir = route_service.create_run_dir(
            run_id=run_id,
            schema_id="anima-finetune",
            route=route,
            runtime_id="flashattention-online-cache-smoke",
            config_json=config,
            command=[sys.executable, "-u", "core/entry_train.py", "--config", "<run_dir>/config.json"],
            output_dir=str(output_dir),
        )
        _write_json(
            run_dir / "resolved_execution.json",
            {
                "execution_profile_id": "flashattention-online-cache-smoke",
                "schema_id": "anima-finetune",
                "python_executable": sys.executable,
                "model_type": "anima",
                "training_type": "full_finetune",
                "effective_execution_core": "standard",
                "resolved_attention_backend": "torch",
                "warnings": [],
            },
        )

        entry_script = backend_root / route.entry_script
        proc = subprocess.run(
            [sys.executable, "-u", str(entry_script), "--config", str(run_dir / "config.json")],
            cwd=str(backend_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=420,
        )
        _write_worker_logs(run_dir, proc)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            raise RuntimeError(f"entry_train.py failed with exit code {proc.returncode}")

        state = _read_json(run_dir / "state.json")
        manifest_path = manifest_path_for(output_dir)
        manifest = _read_json(manifest_path)
        output_path = output_dir / "lulynx_anima_full_finetune_online_cache_route_e2e.safetensors"
        stem = copied_image.stem
        text_cache = data_dir / f"{stem}_anima_te.npz"
        latent_caches = sorted(data_dir.glob(f"{stem}_*_anima.npz"))

        if state.get("status") != "completed":
            raise RuntimeError(f"state.json did not complete: {state}")
        if not text_cache.is_file() or not latent_caches:
            raise RuntimeError(f"online_cache did not generate cache files for {stem}")
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"final output missing or empty: {output_path}")
        extra = manifest.get("extra", {}) if isinstance(manifest.get("extra"), dict) else {}
        setup = extra.get("anima_full_finetune", {}) if isinstance(extra.get("anima_full_finetune"), dict) else {}
        if setup.get("text_conditioning_mode") != "frozen_te_online_cache":
            raise RuntimeError(f"manifest did not record online-cache text conditioning: {setup}")

        route_service.update_run_status(run_id, "completed", exit_code=0)
        summary = {
            "status": "passed",
            "run_id": run_id,
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "data_dir": str(data_dir),
            "generated_text_cache": str(text_cache),
            "generated_latent_cache": str(latent_caches[0]),
            "state_step": int(state.get("current_step") or 0),
            "loss": state.get("last_loss"),
            "output_path": str(output_path),
            "output_bytes": output_path.stat().st_size,
            "manifest_path": str(manifest_path),
            "text_conditioning_mode": setup.get("text_conditioning_mode"),
        }
        _write_summary(run_dir, output_dir, summary)
        print(
            "Anima full-finetune online_cache route/runtime E2E smoke passed: "
            f"run_id={run_id}, step={state.get('current_step')}, "
            f"loss={state.get('last_loss')}, cache={latent_caches[0].name}, output={output_path}"
        )
        return 0
    except Exception as exc:
        if route_service is not None:
            route_service.update_run_status(run_id, "failed", exit_code=getattr(proc, "returncode", 1) or 1)
        summary = {
            "status": "failed",
            "run_id": run_id,
            "run_dir": str(run_dir) if run_dir is not None else "",
            "output_dir": str(output_dir),
            "data_dir": str(data_dir),
            "error": f"{type(exc).__name__}: {exc}",
            "worker_returncode": getattr(proc, "returncode", None),
            "worker_stdout_tail": _tail(getattr(proc, "stdout", "") or ""),
            "worker_stderr_tail": _tail(getattr(proc, "stderr", "") or ""),
        }
        _write_summary(run_dir, output_dir, summary)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
