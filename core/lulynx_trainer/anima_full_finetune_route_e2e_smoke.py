"""Route/runtime E2E smoke for Anima full-finetune.

This smoke intentionally goes through the same boundary as the product path:
``TrainingRouteService`` writes ``.runs/<run_id>/config.json`` and then
``entry_train.py --config`` runs the trainer worker.  It keeps the real Anima
checkpoint/data tiny at runtime by selecting one DiT block, cropping cached
latents to 4x4, limiting text tokens, and stopping after one step.
"""

from __future__ import annotations

import json
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

from backend.lulynx_launcher.services.training_registry import LulynxTrainingRegistry
from backend.lulynx_launcher.services.training_route_service import TrainingRouteService
from backend.core.lulynx_trainer.run_manifest import manifest_path_for


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def _read_events(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    return events


def _tail(value: str, limit: int = 4000) -> str:
    if len(value) <= limit:
        return value
    return value[-limit:]


def _write_worker_logs(run_dir: Path, proc: subprocess.CompletedProcess[str]) -> None:
    _write_text(run_dir / "worker_stdout.log", proc.stdout or "")
    _write_text(run_dir / "worker_stderr.log", proc.stderr or "")


def _write_summary(run_dir: Path | None, output_dir: Path, summary: dict) -> None:
    if run_dir is not None:
        _write_json(run_dir / "route_e2e_summary.json", summary)
    _write_json(output_dir / "route_e2e_summary.json", summary)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[3]
    backend_root = repo_root / "backend"
    route_service: TrainingRouteService | None = None
    run_dir: Path | None = None
    proc: subprocess.CompletedProcess[str] | None = None
    model_path = repo_root / "models" / "anima" / "diffusion_models" / "anima-preview2.safetensors"
    data_dir = repo_root / "sucai" / "6_lulu"
    latent_path = data_dir / "0_1856x2272_anima.npz"
    text_path = data_dir / "0_anima_te.npz"
    if not model_path.is_file():
        raise FileNotFoundError(f"Anima checkpoint not found: {model_path}")
    if not latent_path.is_file() or not text_path.is_file():
        raise FileNotFoundError(f"Missing cached Anima smoke data under {data_dir}")

    run_id = f"anima-full-e2e-smoke-{int(time.time() * 1000)}"
    output_dir = backend_root / "tmp" / "anima_route_e2e" / run_id
    logging_dir = output_dir / "logs"

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
                "output_name": "lulynx_anima_full_finetune_route_e2e",
                "logging_dir": str(logging_dir),
                "device": "cpu",
                "mixed_precision": "no",
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
                "native_cache_mode": "cache_first",
                "anima_cached_training": True,
                "anima_native_block_count": 1,
                "anima_cached_latent_crop_size": 4,
                "anima_cached_text_token_limit": 16,
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
        config["execution_profile_id"] = "flashattention-smoke"

        run_dir = route_service.create_run_dir(
            run_id=run_id,
            schema_id="anima-finetune",
            route=route,
            runtime_id="flashattention-smoke",
            config_json=config,
            command=[sys.executable, "-u", "core/entry_train.py", "--config", "<run_dir>/config.json"],
            output_dir=str(output_dir),
        )

        resolved_execution = {
            "execution_profile_id": "flashattention-smoke",
            "schema_id": "anima-finetune",
            "python_executable": sys.executable,
            "model_type": "anima",
            "training_type": "full_finetune",
            "requested_execution_core": "standard",
            "effective_execution_core": "standard",
            "requested_attention_backend": "auto",
            "resolved_attention_backend": "torch",
            "allow_attention_fallback": True,
            "fallback_reason": "route e2e smoke uses CPU torch attention",
            "warnings": [],
        }
        _write_json(run_dir / "resolved_execution.json", resolved_execution)

        entry_script = backend_root / route.entry_script
        proc = subprocess.run(
            [sys.executable, "-u", str(entry_script), "--config", str(run_dir / "config.json")],
            cwd=str(backend_root),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=240,
        )
        _write_worker_logs(run_dir, proc)
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            raise RuntimeError(f"entry_train.py failed with exit code {proc.returncode}")

        state = _read_json(run_dir / "state.json")
        events = _read_events(run_dir / "events.jsonl")
        manifest_path = manifest_path_for(output_dir)
        manifest = _read_json(manifest_path)
        output_path = output_dir / "lulynx_anima_full_finetune_route_e2e.safetensors"

        if state.get("status") != "completed":
            raise RuntimeError(f"state.json did not complete: {state}")
        if int(state.get("current_step") or 0) < 1:
            raise RuntimeError(f"state.json did not record a training step: {state}")
        if not any(event.get("event_type") == "run_start" for event in events):
            raise RuntimeError("events.jsonl missing run_start")
        if not any(event.get("event_type") == "step" for event in events):
            raise RuntimeError("events.jsonl missing step event")
        if not any(event.get("event_type") == "run_end" and event.get("data", {}).get("status") == "completed" for event in events):
            raise RuntimeError("events.jsonl missing completed run_end")
        if not output_path.is_file() or output_path.stat().st_size <= 0:
            raise RuntimeError(f"final Anima full-finetune output missing or empty: {output_path}")
        if manifest.get("status") != "completed":
            raise RuntimeError(f"run manifest did not complete: {manifest.get('status')!r}")
        if int(manifest.get("global_step") or 0) < 1:
            raise RuntimeError(f"run manifest did not record a training step: {manifest.get('global_step')!r}")
        manifest_checkpoint = Path(str(manifest.get("checkpoint_path") or ""))
        if not manifest_checkpoint.is_file():
            raise RuntimeError(f"run manifest checkpoint path missing: {manifest_checkpoint}")

        extra = manifest.get("extra", {}) if isinstance(manifest.get("extra"), dict) else {}
        setup = extra.get("anima_full_finetune", {}) if isinstance(extra.get("anima_full_finetune"), dict) else {}
        if setup.get("mode") != "dit_only_cache_first":
            raise RuntimeError(f"manifest missing Anima full-finetune setup: {setup}")

        route_service.update_run_status(run_id, "completed", exit_code=0)
        summary = {
            "status": "passed",
            "run_id": run_id,
            "run_dir": str(run_dir),
            "output_dir": str(output_dir),
            "state_step": int(state.get("current_step") or 0),
            "loss": state.get("last_loss"),
            "output_path": str(output_path),
            "output_bytes": output_path.stat().st_size,
            "manifest_path": str(manifest_path),
            "manifest_mode": setup.get("mode"),
            "worker_returncode": proc.returncode,
        }
        _write_summary(run_dir, output_dir, summary)
        print(
            "Anima full-finetune route/runtime E2E smoke passed: "
            f"run_id={run_id}, step={state.get('current_step')}, "
            f"loss={state.get('last_loss')}, output={output_path}"
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
            "error": f"{type(exc).__name__}: {exc}",
            "worker_returncode": getattr(proc, "returncode", None),
            "worker_stdout_tail": _tail(getattr(proc, "stdout", "") or ""),
            "worker_stderr_tail": _tail(getattr(proc, "stderr", "") or ""),
        }
        _write_summary(run_dir, output_dir, summary)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
