"""
Lulynx Trainer Worker Entry Point - V10.0

This script is designed to run in a standalone process (Work Environment) 
to perform training tasks while keeping the Host process clean and Torch-free.

Usage:
    python entry_train.py --config config.json
"""

import os
import sys
import json
import argparse
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

# Ensure both repository root and backend root are importable before local imports.
# Some runtime modules use ``core.*`` while hook/plugin code may import ``backend.core.*``.
_backend_root = Path(__file__).resolve().parent.parent
_repo_root = _backend_root.parent
for _path in (_repo_root, _backend_root):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from core.training_state_writer import TrainingStateWriter
from core.training_event_writer import TrainingEventWriter
from core.turbocore_capabilities import build_turbocore_capability_report
from core.turbocore_resolver import TurboCoreResolutionError, get_turbocore_resolver
from core.lulynx_trainer.module_offload_contract import (
    build_module_offload_pending_state,
    is_swap_requested,
    resolve_module_offload_config,
)
from core.system_sleep_guard import SleepGuard

# Configure logging to stderr to keep stdout clean for IPC
logging.basicConfig(level=logging.INFO, stream=sys.stderr, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _bool_arg(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on", "enable", "enabled"}:
        return True
    if normalized in {"0", "false", "no", "off", "disable", "disabled"}:
        return False
    raise argparse.ArgumentTypeError(f"expected boolean value, got {value!r}")


def _split_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value or "").split(",") if part.strip()]


def _apply_turbocore_cli_overrides(config_dict: Dict[str, Any], args: argparse.Namespace) -> bool:
    """Apply developer-only TurboCore CLI overrides. Returns True if any were used."""
    used_override = False
    if args.execution_core:
        config_dict["execution_core"] = str(args.execution_core).strip().lower()
        used_override = True
    if args.turbocore_features:
        config_dict["turbocore_features"] = _split_csv(args.turbocore_features)
        used_override = True
    if args.turbocore_disable:
        config_dict["turbocore_disable"] = _split_csv(args.turbocore_disable)
        used_override = True
    if args.turbocore_strict:
        config_dict["turbocore_strict"] = True
        used_override = True
    if args.turbocore_allow_fallback is not None:
        config_dict["turbocore_allow_fallback"] = bool(args.turbocore_allow_fallback)
        used_override = True
    if args.turbocore_profile:
        config_dict["turbocore_profile"] = str(args.turbocore_profile).strip().lower()
        used_override = True
    if args.turbocore_workspace_mb is not None:
        config_dict["turbocore_workspace_mb"] = max(0, int(args.turbocore_workspace_mb))
        used_override = True
    if args.turbocore_prefetch_depth is not None:
        config_dict["turbocore_prefetch_depth"] = max(0, int(args.turbocore_prefetch_depth))
        used_override = True
    if args.turbocore_experimental_fp8:
        config_dict["turbocore_experimental_fp8"] = True
        used_override = True
    return used_override


def _resolved_turbocore_payload(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    resolver = get_turbocore_resolver()
    resolved = resolver.resolve_from_config(
        config_dict,
        model_type=str(config_dict.get("model_type", "unknown") or "unknown"),
        training_type=str(config_dict.get("training_type", "lora") or "lora"),
    )
    payload = asdict(resolved)
    return {
        "requested_execution_core": payload["requested_execution_core"],
        "effective_execution_core": payload["effective_execution_core"],
        "turbocore_features_requested": payload["requested_features"],
        "turbocore_features_active": payload["active_features"],
        "turbocore_features_disabled": payload["disabled_features"],
        "turbocore_disable": payload["disabled_by_request"],
        "turbocore_allow_fallback": payload["allow_fallback"],
        "turbocore_strict": payload["strict"],
        "turbocore_experimental_fp8": payload["experimental_fp8_requested"],
        "turbocore_fallback_reason": payload["fallback_reason"],
        "turbocore_warnings": payload["warnings"],
    }


def _write_json_file(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def select_trainer_key(training_type: Any, model_type: Any) -> str:
    """Return the lightweight trainer dispatch key for a normalized route."""
    tt = str(training_type or "lora").strip().lower().replace("_", "-")
    mt = str(model_type or "").strip().lower()
    if tt == "ip-adapter":
        return "ip_adapter"
    if tt == "controlnet":
        return "controlnet"
    if tt == "lllite":
        return "lllite"
    if tt == "lora" and mt == "flux":
        return "flux_lora"
    return "lulynx"


def collect_trainer_runtime_features(trainer: Any) -> Dict[str, Any]:
    """Collect route-specific runtime evidence without coupling entry_train to one trainer."""
    actual_training_loop = getattr(trainer, "training_loop", None)
    actual_memory_optimization = getattr(actual_training_loop, "memory_optimization_state", None)
    actual_b_tier_state = getattr(actual_training_loop, "_b_tier_last_state", None)
    actual_compile_runtime = getattr(trainer, "_compile_runtime_profile", None)
    actual_low_vram_profile = getattr(trainer, "_sdxl_lora_low_vram_profile", None)
    features: Dict[str, Any] = {}
    getter = getattr(trainer, "get_runtime_features", None)
    if callable(getter):
        try:
            collected = getter()
            if isinstance(collected, dict):
                features.update(collected)
        except Exception as exc:
            features["runtime_features_error"] = f"{type(exc).__name__}: {exc}"
    if actual_memory_optimization is not None:
        features.setdefault("memory_optimization", actual_memory_optimization)
    if actual_b_tier_state:
        features.setdefault("b_tier", actual_b_tier_state)
    if isinstance(actual_compile_runtime, dict) and actual_compile_runtime:
        features.setdefault("compile_runtime", dict(actual_compile_runtime))
    if isinstance(actual_low_vram_profile, dict) and actual_low_vram_profile:
        features.setdefault("sdxl_lora_low_vram_profile", dict(actual_low_vram_profile))
    return features


def main():
    parser = argparse.ArgumentParser(description="Lulynx Trainer Worker")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file")
    parser.add_argument("--execution-core", choices=["standard", "turbo", "auto"], default="", help="Optional execution core override")
    parser.add_argument("--turbocore-features", type=str, default="", help="Developer-only TurboCore feature CSV")
    parser.add_argument("--turbocore-disable", type=str, default="", help="Developer-only TurboCore disable CSV")
    parser.add_argument("--turbocore-strict", action="store_true", help="Fail before launch when requested TurboCore is unavailable")
    parser.add_argument("--turbocore-allow-fallback", type=_bool_arg, default=None, help="Allow TurboCore to fall back to StandardCore")
    parser.add_argument("--turbocore-capability-report", type=str, default="", help="Write TurboCore capability decision JSON")
    parser.add_argument("--turbocore-profile", choices=["basic", "detailed"], default="", help="TurboCore profiling level")
    parser.add_argument("--turbocore-workspace-mb", type=int, default=None, help="TurboCore workspace bound in MiB")
    parser.add_argument("--turbocore-prefetch-depth", type=int, default=None, help="TurboCore data pipeline prefetch depth")
    parser.add_argument("--turbocore-experimental-fp8", action="store_true", help="Request future experimental FP8 TurboCore paths")
    args = parser.parse_args()

    # 1. Load Config
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
    except Exception as e:
        logger.error(f"FAILED: Could not load config: {e}")
        sys.exit(1)

    cli_overrides_used = _apply_turbocore_cli_overrides(config_dict, args)

    run_dir = Path(args.config).parent  # .runs/<run_id>/ or direct CLI config directory
    resolved_exec_path = run_dir / "resolved_execution.json"
    try:
        if cli_overrides_used or not resolved_exec_path.exists():
            turbocore_payload = _resolved_turbocore_payload(config_dict)
            config_dict.update(
                {
                    "requested_execution_core": turbocore_payload["requested_execution_core"],
                    "effective_execution_core": turbocore_payload["effective_execution_core"],
                    "turbocore_fallback_reason": turbocore_payload["turbocore_fallback_reason"],
                }
            )
            existing_payload: Dict[str, Any] = {}
            if resolved_exec_path.exists():
                try:
                    existing_payload = json.loads(resolved_exec_path.read_text(encoding="utf-8"))
                except Exception:
                    existing_payload = {}
            existing_payload.update(turbocore_payload)
            _write_json_file(resolved_exec_path, existing_payload)
        if args.turbocore_capability_report:
            try:
                resolution = json.loads(resolved_exec_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                raise TurboCoreResolutionError(f"Failed to parse resolution config: {exc}") from exc
            report_payload = build_turbocore_capability_report(
                config_dict,
                resolution=resolution,
                source="entry_train.py",
            )
            _write_json_file(Path(args.turbocore_capability_report), report_payload)
    except TurboCoreResolutionError as exc:
        if args.turbocore_capability_report:
            error_payload = {
                "message": str(exc),
                "code": exc.code,
            }
            _write_json_file(
                Path(args.turbocore_capability_report),
                build_turbocore_capability_report(
                    config_dict,
                    error=error_payload,
                    source="entry_train.py",
                ),
            )
        logger.error("FAILED: TurboCore resolution failed: %s", exc)
        sys.exit(1)

    # 2. Delayed Heavy Imports
    try:
        import torch
        from core.lulynx_trainer import LulynxTrainer, LulynxConfig, ConfigAdapter
        from core.lulynx_trainer.ip_adapter_trainer import IPAdapterTrainer
        from core.lulynx_trainer.controlnet_trainer import ControlNetTrainer
        from core.turbocore_optimizer_product_training_route_binding_runtime_applier import (
            apply_optimizer_product_training_route_binding_runtime_patch,
        )
    except ImportError as e:
        logger.error(f"FAILED: Missing dependencies: {e}")
        sys.exit(1)

    # 3. Setup Trainer
    state_writer = None
    event_writer = None
    try:
        # Initialize state writer (sole authority for state.json)

        # Read resolver-layer values from resolved_execution.json (written by routers/training.py)
        resolver_requested = config_dict.get("attention_backend", "auto")
        resolver_resolved = config_dict.get("attention_backend", "auto")
        resolver_fallback = ""
        requested_execution_core = str(
            config_dict.get("requested_execution_core", config_dict.get("execution_core", "standard")) or "standard"
        )
        effective_execution_core = str(
            config_dict.get("effective_execution_core", config_dict.get("execution_core", "standard")) or "standard"
        )
        turbocore_fallback_reason = str(config_dict.get("turbocore_fallback_reason", "") or "")
        if resolved_exec_path.exists():
            try:
                _data = json.loads(resolved_exec_path.read_text(encoding="utf-8"))
                resolver_requested = _data.get("requested_attention_backend", resolver_requested)
                resolver_resolved = _data.get("resolved_attention_backend", resolver_resolved)
                resolver_fallback = _data.get("fallback_reason", "")
                requested_execution_core = str(_data.get("requested_execution_core", requested_execution_core) or requested_execution_core)
                effective_execution_core = str(_data.get("effective_execution_core", effective_execution_core) or effective_execution_core)
                turbocore_fallback_reason = str(
                    _data.get("turbocore_fallback_reason", turbocore_fallback_reason) or ""
                )
            except Exception as exc:
                logger.warning("Failed to read resolved_execution.json: %s", exc)

        state_writer = TrainingStateWriter(run_dir)
        event_writer = TrainingEventWriter(run_dir)

        if torch.cuda.is_available():
            vram_limit_gb = config_dict.get("vram_limit")
            if isinstance(vram_limit_gb, (int, float)) and float(vram_limit_gb) > 0:
                total_memory = float(torch.cuda.get_device_properties(0).total_memory)
                requested_bytes = float(vram_limit_gb) * (1024 ** 3)
                memory_fraction = max(min(requested_bytes / total_memory, 0.99), 0.05)
                torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
                logger.info(
                    "Applied VRAM limit: %.2f GB (fraction %.3f)",
                    float(vram_limit_gb),
                    memory_fraction,
                )

        apply_optimizer_product_training_route_binding_runtime_patch(
            config_dict,
            artifact_dir=run_dir,
            report_path=run_dir / "turbocore_optimizer_route_binding_applier.json",
            refresh_config_adapter_artifact=False,
            write_artifact=True,
        )

        # Convert dict to LulynxConfig
        training_type = config_dict.get("training_type", "lora")
        lulynx_config = ConfigAdapter.from_frontend_dict(config_dict)
        total_epochs = int(getattr(lulynx_config, "epochs", 0) or config_dict.get("max_train_epochs", 0) or 0)
        total_steps = int(getattr(lulynx_config, "max_train_steps", 0) or config_dict.get("max_train_steps", 0) or 0)
        state_writer.start(
            pid=os.getpid(),
            run_id=run_dir.name,
            execution_profile_id=config_dict.get("execution_profile_id", ""),
            requested_execution_core=requested_execution_core,
            effective_execution_core=effective_execution_core,
            requested_attention=resolver_requested,
            resolved_attention=resolver_resolved,
            total_epochs=total_epochs,
            total_steps=total_steps,
        )
        event_writer.start(
            run_id=run_dir.name,
            pid=os.getpid(),
            training_type=str(training_type or ""),
            model_type=str(config_dict.get("model_type", "") or ""),
            total_steps=total_steps,
            total_epochs=total_epochs,
            execution_profile_id=str(config_dict.get("execution_profile_id", "") or ""),
            schema_id=str(config_dict.get("schema_id", "") or ""),
            requested_execution_core=requested_execution_core,
            effective_execution_core=effective_execution_core,
            requested_attention_backend=str(resolver_requested or ""),
            resolved_attention_backend=str(resolver_resolved or ""),
        )

        # 3-layer attention telemetry: resolve applied_attention early
        from core.lulynx_trainer.runtime_optimizations import build_runtime_optimization_plan
        runtime_plan = build_runtime_optimization_plan(lulynx_config)
        applied_attention = runtime_plan.attention_backend

        # Determine fallback_reason: inherit from resolver, or add runtime-plan warnings
        fallback_reason = resolver_fallback
        if runtime_plan.warnings and not fallback_reason:
            fallback_reason = "; ".join(runtime_plan.warnings)

        module_offload_view = resolve_module_offload_config(lulynx_config)
        if module_offload_view.requested:
            memory_optimization = build_module_offload_pending_state(lulynx_config)
        elif is_swap_requested(lulynx_config):
            memory_optimization = {
                "enabled": True,
                "mode": "swap",
                "requested_granularity": getattr(lulynx_config, "swap_granularity", "off"),
                "effective_granularity": getattr(lulynx_config, "swap_granularity", "off"),
                "source": "config",
                "swap_ratio": getattr(lulynx_config, "swap_ratio", 0.0),
                "swap_count": getattr(lulynx_config, "swap_count", 0) or getattr(lulynx_config, "blocks_to_swap", 0),
                "units_total": 0,
                "units_swapped": getattr(lulynx_config, "swap_count", 0) or getattr(lulynx_config, "blocks_to_swap", 0),
                "block_merge_size": getattr(lulynx_config, "block_merge_size", 2),
                "reason": "pending training loop resolution",
                "warnings": [],
            }
        else:
            memory_optimization = {
                "enabled": False,
                "mode": "none",
                "source": "config",
                "reason": "",
                "warnings": [],
            }

        # Write attention telemetry to state.json
        state_writer.update(
            requested_execution_core=requested_execution_core,
            effective_execution_core=effective_execution_core,
            requested_attention=resolver_requested,
            resolved_attention=resolver_resolved,
            applied_attention=applied_attention,
            fallback_reason=fallback_reason,
            turbocore_fallback_reason=turbocore_fallback_reason,
            execution_profile_id=config_dict.get("execution_profile_id", ""),
            memory_optimization=memory_optimization,
        )

        # Update resolved_execution.json with applied_attention
        if resolved_exec_path.exists():
            try:
                resolved_data = json.loads(resolved_exec_path.read_text(encoding="utf-8"))
                resolved_data["applied_attention_backend"] = applied_attention
                if fallback_reason:
                    resolved_data["fallback_reason"] = fallback_reason
                resolved_data["requested_execution_core"] = requested_execution_core
                resolved_data["effective_execution_core"] = effective_execution_core
                if turbocore_fallback_reason:
                    resolved_data["turbocore_fallback_reason"] = turbocore_fallback_reason
                resolved_exec_path.write_text(
                    json.dumps(resolved_data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as exc:
                logger.warning("Failed to update resolved_execution.json: %s", exc)

        model_type = str(config_dict.get("model_type", "") or "").strip().lower()
        trainer_key = select_trainer_key(training_type, model_type)
        if trainer_key == "ip_adapter":
            trainer = IPAdapterTrainer(config=lulynx_config)
            logger.info("Using IPAdapterTrainer")
        elif trainer_key == "controlnet":
            trainer = ControlNetTrainer(config=lulynx_config)
            logger.info("Using ControlNetTrainer")
        elif trainer_key == "lllite":
            from core.lulynx_trainer.lllite_trainer import LLLiteTrainer
            trainer = LLLiteTrainer(config=lulynx_config)
            logger.info("Using LLLiteTrainer")
        elif trainer_key == "flux_lora":
            from core.lulynx_trainer.flux_lora_trainer import FluxLoraTrainer
            trainer = FluxLoraTrainer(config=lulynx_config)
            logger.info("Using FluxLoraTrainer")
        else:
            trainer = LulynxTrainer(config=lulynx_config)
            logger.info("Using LulynxTrainer (%s)", training_type)

        # 5. Set Callbacks for Progress Reporting
        def on_step(step, epoch, loss, lr):
            # Format: PROGRESS_JSON: {"step": 1, "loss": 0.5, ...}
            actual_training_loop = getattr(trainer, "training_loop", None)
            actual_memory_optimization = getattr(actual_training_loop, "memory_optimization_state", None)
            actual_b_tier_state = getattr(actual_training_loop, "_b_tier_last_state", None)
            actual_total_steps = int(getattr(actual_training_loop, "total_steps", 0) or 0)
            actual_compile_runtime = getattr(trainer, "_compile_runtime_profile", None)
            actual_low_vram_profile = getattr(trainer, "_sdxl_lora_low_vram_profile", None)
            runtime_features = collect_trainer_runtime_features(trainer)
            progress = {
                "status": "training",
                "step": step,
                "epoch": epoch,
                "loss": loss,
                "lr": lr,
                "total_steps": actual_total_steps,
                "total_epochs": total_epochs,
                "b_tier": actual_b_tier_state if actual_b_tier_state else None,
                "memory_optimization": actual_memory_optimization,
                "compile_runtime": (
                    dict(actual_compile_runtime)
                    if isinstance(actual_compile_runtime, dict) and actual_compile_runtime
                    else None
                ),
                "sdxl_lora_low_vram_profile": (
                    dict(actual_low_vram_profile)
                    if isinstance(actual_low_vram_profile, dict) and actual_low_vram_profile
                    else None
                ),
                "runtime_features": runtime_features,
            }
            print(f"PROGRESS_JSON: {json.dumps(progress)}", flush=True)
            state_writer.update(
                step=step,
                epoch=epoch,
                loss=loss,
                lr=lr,
                total_steps=actual_total_steps,
                total_epochs=total_epochs,
                memory_optimization=actual_memory_optimization,
                runtime_features=runtime_features,
            )
            if event_writer is not None:
                extra = dict(runtime_features)
                if actual_memory_optimization is not None:
                    extra["memory_optimization"] = actual_memory_optimization
                if actual_b_tier_state:
                    extra["b_tier"] = actual_b_tier_state
                if isinstance(actual_low_vram_profile, dict) and actual_low_vram_profile:
                    extra["sdxl_lora_low_vram_profile"] = dict(actual_low_vram_profile)
                event_writer.step(
                    run_id=run_dir.name,
                    step=int(step),
                    epoch=int(epoch),
                    loss=float(loss),
                    lr=float(lr),
                    total_steps=actual_total_steps,
                    total_epochs=total_epochs,
                    extra=extra,
                )

        def on_log(msg):
            print(f"LOG: {msg}", flush=True)

        def on_runtime_event(payload):
            if event_writer is None:
                return
            event_payload = dict(payload or {})
            event_writer.emit(event_payload)

        trainer.set_callbacks(
            on_step=on_step,
            on_log=on_log,
            on_runtime_event=on_runtime_event,
        )

        # 6. Start Training
        logger.info(f"Starting training: {lulynx_config.output_name}")
        with SleepGuard():
            success = bool(trainer.train())
        if success:
            state_writer.complete()
            if event_writer is not None:
                final_step = int(getattr(getattr(trainer, "training_loop", None), "global_step", 0) or 0)
                final_epoch = int(getattr(getattr(trainer, "training_loop", None), "current_epoch", 0) or 0)
                event_writer.complete(run_id=run_dir.name, final_step=final_step, final_epoch=final_epoch)
            logger.info("Training completed successfully.")
        else:
            state_writer.fail("Training returned unsuccessful status")
            if event_writer is not None:
                final_step = int(getattr(getattr(trainer, "training_loop", None), "global_step", 0) or 0)
                final_epoch = int(getattr(getattr(trainer, "training_loop", None), "current_epoch", 0) or 0)
                event_writer.fail(
                    run_id=run_dir.name,
                    error="Training returned unsuccessful status",
                    final_step=final_step,
                    final_epoch=final_epoch,
                )
            logger.error("FAILED: Training returned unsuccessful status.")
            sys.exit(1)

    except Exception as e:
        if state_writer is not None:
            state_writer.fail(str(e))
        if event_writer is not None:
            event_writer.fail(run_id=run_dir.name if "run_dir" in locals() else "", error=str(e))
        logger.error(f"FAILED: Unexpected error during training: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
