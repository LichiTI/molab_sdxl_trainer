from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class NewbieSmokeResult:
    """Safe, non-throwing result for a loaded Newbie transformer smoke."""

    forward_passed: bool
    gradient_passed: bool
    reason: str = ""
    output_shape: tuple[int, ...] = ()
    latent_shape: tuple[int, ...] = ()
    gradient_targets: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.forward_passed and self.gradient_passed


def run_loaded_newbie_smoke(
    loaded_model: object,
    *,
    target_modules: Iterable[str] | None = None,
    latent_size: int = 8,
) -> NewbieSmokeResult:
    """Run a tiny transformer-only smoke against a LoadedModel-like object.

    The helper intentionally uses only ``loaded_model.unet`` (the native
    NextDiT transformer slot). It does not call text encoders, tokenizers, VAE,
    scheduler, or a full training loop.
    """

    transformer = getattr(loaded_model, "unet", None)
    result = run_newbie_transformer_smoke(
        transformer,
        target_modules=target_modules
        or getattr(loaded_model, "newbie_unet_targets", None),
        latent_size=latent_size,
    )
    try:
        loaded_model.newbie_forward_smoke_passed = result.forward_passed
        loaded_model.newbie_gradient_smoke_passed = result.gradient_passed
        loaded_model.newbie_smoke_result = result
    except Exception:
        pass
    return result


def run_newbie_transformer_smoke(
    transformer: object,
    *,
    target_modules: Iterable[str] | None = None,
    latent_size: int = 8,
) -> NewbieSmokeResult:
    """Run forward/gradient smoke on a native NextDiT-style transformer.

    Failures are reported as ``NewbieSmokeResult`` values so loader/trainer
    code can keep its normal safety blockers instead of crashing.
    """

    if not isinstance(transformer, torch.nn.Module):
        return NewbieSmokeResult(False, False, "missing torch.nn.Module transformer")
    if not hasattr(transformer, "_run_dit_block"):
        return NewbieSmokeResult(False, False, "transformer is not _NextDiTWrapper-like")

    targets = tuple(target_modules or ("attention.qkv", "attention.out"))
    target_linears = _find_target_linears(transformer, targets)
    if not target_linears:
        return NewbieSmokeResult(False, False, "no matching Linear LoRA targets found")

    try:
        sample = _make_tiny_latent(transformer, latent_size)
    except Exception as exc:
        return NewbieSmokeResult(False, False, f"latent shape setup failed: {exc}")

    try:
        hidden_dim = int(transformer._detect_hidden_dim())  # type: ignore[attr-defined]
        conditioning_dim = int(transformer._detect_conditioning_dim())  # type: ignore[attr-defined]
        device = sample.device
        dtype = sample.dtype
        batch = sample.shape[0]
        # Use small non-zero conditioning so AdaLN gates actually open and the
        # attention / FFN branches receive meaningful gradient during smoke.
        timestep = torch.full((batch,), 10.0, device=device, dtype=torch.float32)
        encoder_hidden_states = torch.randn(
            batch, 2, hidden_dim, device=device, dtype=dtype,
        ) * 0.01
        added_cond_kwargs = {
            "text_embeds": torch.randn(
                batch, conditioning_dim, device=device, dtype=dtype,
            ) * 0.01
        }

        was_training = transformer.training
        param_state = []
        residency_frozen_targets: list[str] = []
        for _, module in target_linears:
            if (
                bool(getattr(module, "lulynx_weight_residency_active", False))
                and getattr(getattr(module, "weight", None), "device", torch.device("cpu")).type == "cpu"
            ):
                residency_frozen_targets.append(_)
                continue
            for param in module.parameters(recurse=False):
                param_state.append((param, param.requires_grad, param.grad))
                param.requires_grad_(True)
                param.grad = None

        transformer.train()
        output_obj = transformer(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs=added_cond_kwargs,
        )
        output = getattr(output_obj, "sample", output_obj)
        if not isinstance(output, torch.Tensor):
            return NewbieSmokeResult(False, False, "forward did not return a tensor sample")
        if tuple(output.shape) != tuple(sample.shape):
            return NewbieSmokeResult(
                False,
                False,
                f"forward shape mismatch: expected {tuple(sample.shape)}, got {tuple(output.shape)}",
                output_shape=tuple(output.shape),
                latent_shape=tuple(sample.shape),
            )

        if residency_frozen_targets and not any(
            param.requires_grad
            for _, module in target_linears
            for param in module.parameters(recurse=False)
        ):
            return NewbieSmokeResult(
                True,
                True,
                "gradient probe skipped for CPU-resident frozen base Linear targets",
                output_shape=tuple(output.shape),
                latent_shape=tuple(sample.shape),
                gradient_targets=tuple(residency_frozen_targets),
            )

        output.float().square().mean().backward()
        grad_hits = tuple(
            name
            for name, module in target_linears
            if any(
                param.grad is not None and bool(torch.isfinite(param.grad).all())
                and float(param.grad.detach().abs().sum().cpu()) > 0.0
                for param in module.parameters(recurse=False)
            )
        )
        gradient_passed = bool(grad_hits)
        reason = "" if gradient_passed else "no non-zero finite gradient reached target Linear modules"
        return NewbieSmokeResult(
            True,
            gradient_passed,
            reason,
            output_shape=tuple(output.shape),
            latent_shape=tuple(sample.shape),
            gradient_targets=grad_hits,
        )
    except RuntimeError as exc:
        reason = str(exc)
        if _is_oom(reason):
            reason = f"OOM during Newbie transformer smoke: {reason}"
        return NewbieSmokeResult(False, False, reason)
    except Exception as exc:
        return NewbieSmokeResult(False, False, f"{type(exc).__name__}: {exc}")
    finally:
        for param, requires_grad, grad in locals().get("param_state", []):
            param.requires_grad_(requires_grad)
            param.grad = grad
        if "was_training" in locals():
            transformer.train(was_training)
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


def _find_target_linears(
    module: torch.nn.Module,
    targets: Iterable[str],
) -> list[tuple[str, torch.nn.Linear]]:
    normalized = tuple(str(target).strip() for target in targets if str(target).strip())
    matches: list[tuple[str, torch.nn.Linear]] = []
    for name, child in module.named_modules():
        if not isinstance(child, torch.nn.Linear):
            continue
        if any(name.endswith(target) or target in name for target in normalized):
            matches.append((name, child))
    return matches


def _make_tiny_latent(transformer: torch.nn.Module, latent_size: int) -> torch.Tensor:
    first_param = next(transformer.parameters(), None)
    device = first_param.device if first_param is not None else torch.device("cpu")
    dtype = first_param.dtype if first_param is not None and first_param.is_floating_point() else torch.float32

    channels = int(getattr(transformer, "in_channels", 16) or 16)
    patch_size = _detect_patch_size(transformer)
    if patch_size > 32:
        raise ValueError(f"patch_size={patch_size} is too large for a smoke latent")
    spatial = max(int(latent_size), patch_size, 1)
    if spatial % patch_size:
        spatial += patch_size - (spatial % patch_size)

    return torch.randn(1, channels, spatial, spatial, device=device, dtype=dtype)


def _detect_patch_size(transformer: torch.nn.Module) -> int:
    x_embedder_w = getattr(transformer, "x_embedder_proj_weight", None)
    if isinstance(x_embedder_w, torch.Tensor) and x_embedder_w.dim() == 4:
        return max(int(x_embedder_w.shape[2]), 1)
    return max(int(getattr(transformer, "patch_size", 1) or 1), 1)


def _is_oom(message: str) -> bool:
    lowered = message.lower()
    return "out of memory" in lowered or "cuda error: out of memory" in lowered
