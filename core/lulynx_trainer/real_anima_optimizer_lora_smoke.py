"""Real Anima cached LoRA train smoke for MN-LoRA++ and AutoProdigy.

This uses local real assets:
- models/anima/diffusion_models/anima-preview2.safetensors
- sucai/6_lulu/*_anima.npz and *_anima_te.npz

It intentionally loads a small executable subset of the Anima DiT by default
so the optimizer artifact path can be checked without a full training run.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import numpy as np
import torch

BACKEND_ROOT = Path(__file__).resolve().parents[2]
CORE_ROOT = BACKEND_ROOT / "core"
TRAINER_ROOT = CORE_ROOT / "lulynx_trainer"
MNLORA_ROOT = CORE_ROOT / "training_components" / "mn_lora"
REPO_ROOT = BACKEND_ROOT.parent


def _install_xformers_stub() -> None:
    sys.modules.pop("xformers", None)
    sys.modules.pop("xformers.ops", None)
    xformers_module = ModuleType("xformers")
    ops_module = ModuleType("xformers.ops")
    ops_module.__spec__ = ModuleSpec("xformers.ops", loader=None)
    xformers_module.ops = ops_module
    xformers_module.__spec__ = ModuleSpec("xformers", loader=None)
    sys.modules["xformers"] = xformers_module
    sys.modules["xformers.ops"] = ops_module


def _ensure_namespace(name: str, path: Path) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module
    return module


def _load_module(name: str, path: Path):
    module = sys.modules.get(name)
    if module is not None:
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_install_xformers_stub()
_ensure_namespace("core", CORE_ROOT)
_ensure_namespace("core.lulynx_trainer", TRAINER_ROOT)
_ensure_namespace("core.training_components", CORE_ROOT / "training_components")
_ensure_namespace("core.training_components.mn_lora", MNLORA_ROOT)
_load_module("core.lulynx_trainer.model_family", TRAINER_ROOT / "model_family.py")
_load_module("core.lulynx_trainer.tlora", TRAINER_ROOT / "tlora.py")
_load_module("core.lulynx_trainer.safetensors_loader", TRAINER_ROOT / "safetensors_loader.py")
_load_module("core.lulynx_trainer.anima_flow", TRAINER_ROOT / "anima_flow.py")
_load_module("core.lulynx_trainer.anima_native_dit", TRAINER_ROOT / "anima_native_dit.py")
_load_module("core.lulynx_trainer.anima_targets", TRAINER_ROOT / "anima_targets.py")
_load_module("core.training_components.mn_lora.trace_guided_wd", MNLORA_ROOT / "trace_guided_wd.py")
_load_module("core.training_components.mn_lora.svd_utils", MNLORA_ROOT / "svd_utils.py")
_load_module("core.training_components.mn_lora.gradient_subspace", MNLORA_ROOT / "gradient_subspace.py")
_load_module("core.training_components.mn_lora.mn_lora_plus_plus", MNLORA_ROOT / "mn_lora_plus_plus.py")
lora_mod = _load_module("core.lulynx_trainer.lora_injector", TRAINER_ROOT / "lora_injector.py")
ap_mod = _load_module("core.lulynx_trainer.auto_prodigy_optimizer", TRAINER_ROOT / "auto_prodigy_optimizer.py")
mn_mod = _load_module("core.training_components.mn_lora.mn_optimizer", MNLORA_ROOT / "mn_optimizer.py")

anima_flow = sys.modules["core.lulynx_trainer.anima_flow"]
anima_dit = sys.modules["core.lulynx_trainer.anima_native_dit"]
anima_targets = sys.modules["core.lulynx_trainer.anima_targets"]
LoRAInjector = lora_mod.LoRAInjector
AutoProdigy = ap_mod.AutoProdigy
MNLoRAOptimizer = mn_mod.MNLoRAOptimizer


def _load_cached_latents(path: Path, crop_tokens: int) -> torch.Tensor:
    data = np.load(path)
    latent_keys = sorted(key for key in data.files if key.startswith("latents_"))
    if not latent_keys:
        raise ValueError(f"No latents_* arrays found in {path}")
    latents = torch.from_numpy(data[latent_keys[0]]).float().unsqueeze(0)
    size = max(4, int(crop_tokens) * 2)
    return latents[:, :, :size, :size].contiguous()


def _load_cached_text(path: Path, token_limit: int) -> torch.Tensor:
    data = np.load(path)
    if "prompt_embeds" not in data:
        raise ValueError(f"No prompt_embeds found in {path}")
    prompt_embeds = torch.from_numpy(data["prompt_embeds"]).float().unsqueeze(0)
    return prompt_embeds[:, : max(int(token_limit), 1), :].contiguous()


def _assert_lora_artifact(path: Path, expected_layers: int) -> dict[str, float | int]:
    from safetensors import safe_open

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        keys = list(handle.keys())
        down_keys = [key for key in keys if key.endswith(".lora_down.weight")]
        up_keys = [key for key in keys if key.endswith(".lora_up.weight")]
        if len(down_keys) != expected_layers or len(up_keys) != expected_layers:
            raise AssertionError(
                f"Unexpected LoRA key count for {path}: down={len(down_keys)} up={len(up_keys)} expected={expected_layers}"
            )
        total_abs = 0.0
        up_abs = 0.0
        for key in keys:
            tensor = handle.get_tensor(key)
            if not torch.isfinite(tensor).all():
                raise AssertionError(f"{path}:{key} contains NaN/Inf")
            value = float(tensor.float().abs().sum())
            total_abs += value
            if key.endswith(".lora_up.weight"):
                up_abs += value
        if total_abs <= 0 or up_abs <= 0:
            raise AssertionError(f"{path} has zero LoRA tensors: total_abs={total_abs} up_abs={up_abs}")
        metadata = handle.metadata() or {}
        if metadata.get("model_family") != "anima":
            raise AssertionError(f"Missing model_family=anima metadata in {path}: {metadata}")
        return {"tensor_count": len(keys), "total_abs": total_abs, "up_abs": up_abs}


def _make_optimizer(
    kind: str,
    params: list[torch.nn.Parameter],
    model: torch.nn.Module,
    *,
    adamw_lr: float,
    auto_d0: float,
    auto_growth: float,
    auto_cap: float,
):
    if kind == "mn_lora_plus_plus":
        base = torch.optim.AdamW(params, lr=float(adamw_lr), weight_decay=0.0)
        return MNLoRAOptimizer(
            base,
            enable_tgwd=False,
            enable_gsp=False,
            enable_pilot=False,
            plus_plus_config={
                "enabled": True,
                "lr_up": 1.02,
                "lr_down": 0.9,
                "min_mult": 0.25,
                "max_mult": 2.0,
                "update_rms_cap": 0.01,
            },
            param_names={id(param): name for name, param in model.named_parameters()},
        )
    if kind == "auto_prodigy":
        return AutoProdigy(
            params,
            lr=1.0,
            d0=float(auto_d0),
            growth_rate=float(auto_growth),
            max_update_rms_ratio=float(auto_cap),
        )
    raise ValueError(kind)


def _run_one(
    kind: str,
    *,
    blocks: int,
    steps: int,
    crop_tokens: int,
    text_tokens: int,
    fixed_noise: bool,
    adamw_lr: float,
    auto_d0: float,
    auto_growth: float,
    auto_cap: float,
) -> Path:
    checkpoint = REPO_ROOT / "models" / "anima" / "diffusion_models" / "anima-preview2.safetensors"
    data_dir = REPO_ROOT / "sucai" / "6_lulu"
    latent_path = data_dir / "0_1856x2272_anima.npz"
    text_path = data_dir / "0_anima_te.npz"
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    if not latent_path.exists() or not text_path.exists():
        raise FileNotFoundError(data_dir)

    model, report = anima_dit.load_anima_native_executable_subset(
        checkpoint,
        block_indices=tuple(range(int(blocks))),
        device="cpu",
        dtype=torch.float32,
    )
    if not report.strict_success:
        raise RuntimeError(report.to_dict())
    for param in model.parameters():
        param.requires_grad_(False)

    injector = LoRAInjector(rank=1, alpha=1, model_arch="anima")
    injected = injector._inject_model(
        model,
        anima_targets.get_anima_dit_targets(include_llm_adapter=False),
        prefix="net",
    )
    if not injected:
        raise AssertionError("No LoRA layers injected")

    params = injector.get_trainable_params()
    optimizer = _make_optimizer(
        kind,
        params,
        model,
        adamw_lr=adamw_lr,
        auto_d0=auto_d0,
        auto_growth=auto_growth,
        auto_cap=auto_cap,
    )
    latents = _load_cached_latents(latent_path, crop_tokens)
    context = _load_cached_text(text_path, text_tokens)
    fixed_noise_tensor = torch.randn_like(latents) if fixed_noise else None
    if fixed_noise_tensor is not None:
        fixed_sigmas = anima_flow.sample_anima_sigmas(
            latents.shape[0],
            device=latents.device,
            dtype=latents.dtype,
            config=anima_flow.AnimaFlowConfig(timestep_sampling="sigma"),
        )
    else:
        fixed_sigmas = None

    first_loss = None
    last_loss = None
    grad_hits = 0
    losses: list[float] = []
    for step in range(max(int(steps), 1)):
        noise = fixed_noise_tensor if fixed_noise_tensor is not None else torch.randn_like(latents)
        sigmas = fixed_sigmas if fixed_sigmas is not None else anima_flow.sample_anima_sigmas(
            latents.shape[0],
            device=latents.device,
            dtype=latents.dtype,
            config=anima_flow.AnimaFlowConfig(timestep_sampling="sigma"),
        )
        noisy_latents, target, timesteps = anima_flow.build_anima_flow_inputs(latents, noise, sigmas)
        optimizer.zero_grad(set_to_none=True)
        pred = model(noisy_latents, timesteps, context).sample
        loss = torch.nn.functional.mse_loss(pred.float(), target.float())
        loss.backward()
        if step == 0:
            grad_hits = sum(
                1
                for name, param in model.named_parameters()
                if "lora_" in name
                and param.grad is not None
                and torch.isfinite(param.grad).all()
                and param.grad.abs().sum() > 0
            )
            if grad_hits == 0:
                raise AssertionError("No nonzero LoRA gradients observed")
            first_loss = float(loss.detach())
        last_loss = float(loss.detach())
        losses.append(last_loss)
        optimizer.step()
        if (step + 1) % max(min(10, int(steps)), 1) == 0 or step == 0:
            print(f"    {kind} step {step + 1}/{steps}: loss={last_loss:.6f}", flush=True)

    out_dir = REPO_ROOT / "tmp" / "real_anima_optimizer_lora_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{kind}_blocks{blocks}_steps{steps}.safetensors"
    from safetensors.torch import save_file

    save_file(
        {key: value.detach().cpu() for key, value in injector.get_lora_state_dict().items()},
        str(out_path),
        metadata={
            "model_family": "anima",
            "optimizer_smoke": kind,
            "source_model": checkpoint.name,
            "source_latents": latent_path.name,
            "source_text": text_path.name,
            "blocks": str(blocks),
            "steps": str(steps),
            "first_loss": f"{first_loss:.8f}" if first_loss is not None else "",
            "last_loss": f"{last_loss:.8f}" if last_loss is not None else "",
            "losses": ",".join(f"{value:.8f}" for value in losses),
            "fixed_noise": str(bool(fixed_noise)),
        },
    )
    stats = _assert_lora_artifact(out_path, expected_layers=len(injected))
    first_window = losses[: max(1, min(5, len(losses)))]
    last_window = losses[-max(1, min(5, len(losses))):]
    start_avg = sum(first_window) / len(first_window)
    end_avg = sum(last_window) / len(last_window)
    improvement = (start_avg - end_avg) / max(abs(start_avg), 1e-8)
    best_loss = min(losses)
    print(
        f"  {kind}: blocks={blocks} steps={steps} loss={first_loss:.6f}->{last_loss:.6f} "
        f"best={best_loss:.6f} trend={start_avg:.6f}->{end_avg:.6f} improvement={improvement:.2%} "
        f"layers={len(injected)} grad_hits={grad_hits} tensors={stats['tensor_count']} "
        f"abs={stats['total_abs']:.6f} saved={out_path} -- PASS",
        flush=True,
    )
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--crop-tokens", type=int, default=2)
    parser.add_argument("--text-tokens", type=int, default=16)
    parser.add_argument("--fixed-noise", action="store_true")
    parser.add_argument("--adamw-lr", type=float, default=1e-4)
    parser.add_argument("--auto-d0", type=float, default=1e-3)
    parser.add_argument("--auto-growth", type=float, default=1.1)
    parser.add_argument("--auto-cap", type=float, default=0.02)
    args = parser.parse_args()

    outputs = [
        _run_one(
            "mn_lora_plus_plus",
            blocks=args.blocks,
            steps=args.steps,
            crop_tokens=args.crop_tokens,
            text_tokens=args.text_tokens,
            fixed_noise=bool(args.fixed_noise),
            adamw_lr=args.adamw_lr,
            auto_d0=args.auto_d0,
            auto_growth=args.auto_growth,
            auto_cap=args.auto_cap,
        ),
        _run_one(
            "auto_prodigy",
            blocks=args.blocks,
            steps=args.steps,
            crop_tokens=args.crop_tokens,
            text_tokens=args.text_tokens,
            fixed_noise=bool(args.fixed_noise),
            adamw_lr=args.adamw_lr,
            auto_d0=args.auto_d0,
            auto_growth=args.auto_growth,
            auto_cap=args.auto_cap,
        ),
    ]
    print("Real Anima optimizer LoRA smoke passed:")
    for path in outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
