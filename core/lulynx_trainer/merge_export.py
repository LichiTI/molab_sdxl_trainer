"""
Merged Checkpoint Export — merge adapter (LoRA / LyCORIS) weights back into
the base model and save the result as a full safetensors checkpoint.

This consumes the dormant ``anima_merge_export`` / ``merge_export`` config
field.  The merge is performed on a *copy* of the model so that the in-memory
training state is untouched.
"""

import copy
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LoRA merge
# ---------------------------------------------------------------------------

def merge_lora_into_base(
    base_model: nn.Module,
    lora_injector,
) -> int:
    """Merge every injected LoRALinear back into the base model's Linear.

    For each :class:`LoRALinear` wrapper the function:
    1. Computes ``delta = (lora_up @ lora_down) * scale``.
    2. Adds the delta to the frozen base weight.
    3. Replaces the wrapper with the original (now merged) ``nn.Linear``.

    For DoRA layers the same procedure is applied using
    ``lora_B @ lora_A * scale``.

    Args:
        base_model: The nn.Module whose sub-modules have been wrapped.
        lora_injector: A :class:`LoRAInjector` instance with
            ``injected_layers`` populated.

    Returns:
        Number of layers successfully merged.
    """

    from .lora_injector import LoRALinear
    from .vera_layer import VeRALinear
    from .lora_fa_layer import LoRAFALinear
    from .tlora import TLoRALinear
    from core.lulynx.dora_layer import DoRALinear

    merged = 0

    for full_name, wrapper in list(lora_injector.injected_layers.items()):
        if not isinstance(wrapper, LoRALinear):
            # VeRA / LoRA-FA / T-LoRA — fall back to generic delta weight
            merged += _merge_generic_adapter(base_model, full_name, wrapper)
            continue

        original_linear: nn.Linear = wrapper.original

        with torch.no_grad():
            base_w = original_linear.weight.data

            if wrapper.use_dora:
                # DoRALinear holds base_weight internally along with lora_A, lora_B
                dora: DoRALinear = wrapper.lora  # type: ignore[assignment]
                # DoRA merge: the effective weight is
                #   m * (base_weight + lora_B @ lora_A * scale) / ||...||
                # but for a full merge we simply add the low-rank delta to the base
                # and drop the magnitude normalisation (it becomes baked in at the
                # current training step).
                if hasattr(dora, "lora_A") and hasattr(dora, "lora_B"):
                    delta = (dora.lora_B @ dora.lora_A) * dora.scaling
                    base_w.add_(delta.to(base_w.dtype))
                # If magnitude `m` is available, bake it in:
                if hasattr(dora, "m") and dora.m is not None:
                    # m approximates column-wise norm ratio; multiply column-wise
                    m = dora.m.to(base_w.dtype)
                    base_w.mul_(m.unsqueeze(1) if m.dim() == 1 else m)
            else:
                # Standard LoRA: delta = lora_up @ lora_down * scaling
                adapter = wrapper.lora
                if hasattr(adapter, "lora_up") and hasattr(adapter, "lora_down"):
                    delta = (adapter.lora_up.weight @ adapter.lora_down.weight) * adapter.scaling
                    base_w.add_(delta.to(base_w.dtype))
                elif hasattr(adapter, "lora_A") and hasattr(adapter, "lora_B"):
                    # Alternate DoRA layout
                    delta = (adapter.lora_B @ adapter.lora_A) * adapter.scaling
                    base_w.add_(delta.to(base_w.dtype))

        # Replace the wrapper with the original (now merged) Linear
        _replace_module(base_model, full_name, original_linear)
        merged += 1

    logger.info("merge_lora_into_base: merged %d LoRA layers", merged)
    return merged


def _merge_generic_adapter(
    base_model: nn.Module,
    full_name: str,
    wrapper: nn.Module,
) -> int:
    """Fallback merge for non-standard adapter wrappers (VeRA, LoRA-FA, T-LoRA).

    These wrappers all store a reference to the original layer and add a delta
    during forward.  We cannot always symbolically compute the merged weight
    (VeRA uses shared buffers), so we fall back to loading-then-merging via
    a standard LoRA export path when available.
    """

    from .vera_layer import VeRALinear
    from .lora_fa_layer import LoRAFALinear
    from .tlora import TLoRALinear

    original_linear = getattr(wrapper, "original", None)
    if original_linear is None or not isinstance(original_linear, nn.Linear):
        logger.warning("merge_export: skipping %s — cannot locate original Linear", full_name)
        return 0

    with torch.no_grad():
        base_w = original_linear.weight.data

        if isinstance(wrapper, VeRALinear):
            # VeRA exports to standard lora_down / lora_up weights
            weights = wrapper.export_standard_lora_weights()
            down = weights.get("lora_down.weight")
            up = weights.get("lora_up.weight")
            if down is not None and up is not None:
                scaling = float(getattr(wrapper, "scaling", wrapper.alpha / wrapper.rank))
                delta = (up @ down) * scaling
                base_w.add_(delta.to(base_w.dtype))
            else:
                logger.warning("merge_export: VeRA export missing weights for %s", full_name)
                return 0

        elif isinstance(wrapper, LoRAFALinear):
            down = wrapper.lora_down.weight.data
            up = wrapper.lora_up.weight.data
            scaling = float(getattr(wrapper, "scaling", 1.0))
            delta = (up @ down) * scaling
            base_w.add_(delta.to(base_w.dtype))

        elif isinstance(wrapper, TLoRALinear):
            down = wrapper.lora_down.weight.data
            up = wrapper.lora_up.weight.data
            scaling = float(getattr(wrapper, "scaling", 1.0))
            delta = (up @ down) * scaling
            base_w.add_(delta.to(base_w.dtype))

        else:
            logger.warning("merge_export: unknown wrapper type %s for %s", type(wrapper).__name__, full_name)
            return 0

    _replace_module(base_model, full_name, original_linear)
    return 1


# ---------------------------------------------------------------------------
# LyCORIS merge
# ---------------------------------------------------------------------------

def merge_lycoris_into_base(
    base_model: nn.Module,
    lycoris_injector,
) -> int:
    """Merge LyCORIS adapter deltas back into base model weights.

    LyCORIS layers are *not* wrappers — they are separate modules whose output
    is added via a monkey-patched ``forward``.  For each injected layer:
    1. Compute the delta weight matrix via ``get_delta_weight()``.
    2. Add the delta to the base module's weight.
    3. Restore the base module's original ``forward`` method.

    Args:
        base_model: The nn.Module that was injected.
        lycoris_injector: A :class:`LyCORISInjector` with ``injected_layers``.

    Returns:
        Number of layers merged.
    """

    from .lycoris_layers import LoHaLayer, LoKrLayer, LoConLayer, _NormAdapter
    from .generalized_adapters import GLoRALinearLayer, GLoRAConv2dLayer
    from .glokr_layer import GLoKrLinearLayer

    merged = 0

    for full_name, lycoris_layer in list(lycoris_injector.injected_layers.items()):
        # Recover the base module name (strip prefix like "unet.")
        module_name = full_name.split(".", 1)[-1] if "." in full_name else full_name

        base_module = _get_submodule(base_model, module_name)
        if base_module is None:
            logger.warning("merge_lycoris: cannot find base module %s", module_name)
            continue

        with torch.no_grad():
            if isinstance(lycoris_layer, _NormAdapter):
                # Norm adapter: scale + bias residual
                scale = lycoris_layer.scale.data
                bias = lycoris_layer.bias.data
                s_scaling = lycoris_layer.scaling

                if isinstance(base_module, nn.LayerNorm):
                    # base_module.weight and base_module.bias exist
                    base_module.weight.data.mul_(1.0 + scale.to(base_module.weight.dtype) * s_scaling)
                    if base_module.bias is not None:
                        base_module.bias.data.add_(bias.to(base_module.bias.dtype) * s_scaling)
                elif hasattr(base_module, "weight"):
                    # RMSNorm or similar
                    base_module.weight.data.mul_(1.0 + scale.to(base_module.weight.dtype) * s_scaling)

                # Restore original forward
                if hasattr(base_module, "_original_forward"):
                    base_module.forward = base_module._original_forward  # type: ignore[assignment]
                    delattr(base_module, "_original_forward")

                merged += 1
                continue

            if isinstance(lycoris_layer, (LoHaLayer, LoKrLayer, GLoRALinearLayer, GLoKrLinearLayer)):
                delta = lycoris_layer.get_delta_weight()
                if isinstance(base_module, nn.Linear) and delta.shape == base_module.weight.shape:
                    base_module.weight.data.add_(delta.to(base_module.weight.dtype))
                else:
                    logger.warning(
                        "merge_lycoris: shape mismatch for %s (delta=%s, base=%s)",
                        module_name, tuple(delta.shape), tuple(base_module.weight.shape),
                    )
                    continue

            elif isinstance(lycoris_layer, (LoConLayer, GLoRAConv2dLayer)):
                # Conv2d delta — rebuild as full conv weight
                # LoCon computes: lora_up(lora_down(x)) * scaling
                # Merging requires unfolding Conv2d weights which is non-trivial.
                # We approximate by doing a single-pass merge via F.conv2d on a
                # dummy input and then treating it as a weight update — but the
                # simplest correct approach is to add the unfolded Kronecker delta.
                delta = lycoris_layer.get_delta_weight() if hasattr(lycoris_layer, "get_delta_weight") else None
                if delta is not None and isinstance(base_module, nn.Conv2d) and delta.shape == base_module.weight.shape:
                    base_module.weight.data.add_(delta.to(base_module.weight.dtype))
                else:
                    # Fallback: note that full Conv2d merge is complex; skip with warning
                    logger.warning("merge_lycoris: LoCon Conv2d merge skipped for %s (shape mismatch or no get_delta_weight)", module_name)
                    continue

            else:
                # Generic LoRALayer (LoCon for Linear falls here)
                if hasattr(lycoris_layer, "get_delta_weight"):
                    delta = lycoris_layer.get_delta_weight()
                    if isinstance(base_module, nn.Linear) and delta.shape == base_module.weight.shape:
                        base_module.weight.data.add_(delta.to(base_module.weight.dtype))
                    else:
                        logger.warning("merge_lycoris: shape mismatch for %s", module_name)
                        continue
                elif hasattr(lycoris_layer, "lora_up") and hasattr(lycoris_layer, "lora_down"):
                    delta = lycoris_layer.lora_up.weight.data @ lycoris_layer.lora_down.weight.data * lycoris_layer.scaling
                    if isinstance(base_module, nn.Linear) and delta.shape == base_module.weight.shape:
                        base_module.weight.data.add_(delta.to(base_module.weight.dtype))
                    else:
                        logger.warning("merge_lycoris: shape mismatch for %s", module_name)
                        continue
                else:
                    logger.warning("merge_lycoris: unknown layer type %s for %s", type(lycoris_layer).__name__, module_name)
                    continue

            # Restore original forward (undo the monkey-patch)
            if hasattr(base_module, "_original_forward"):
                base_module.forward = base_module._original_forward  # type: ignore[assignment]
                delattr(base_module, "_original_forward")

            merged += 1

    logger.info("merge_lycoris_into_base: merged %d LyCORIS layers", merged)
    return merged


# ---------------------------------------------------------------------------
# Export merged model
# ---------------------------------------------------------------------------

def export_merged_model(
    model: nn.Module,
    output_path: str,
    save_precision: str = "bf16",
    lora_injector=None,
    lycoris_injector=None,
) -> str:
    """Merge adapter weights into the base model and save the full state dict.

    This creates a *shallow copy* of the model (``copy.deepcopy`` of the
    ``state_dict`` only) so the original training model is not modified.

    Args:
        model: The base nn.Module (e.g. ``LoadedModel.unet``).
        output_path: Destination file path (should end in ``.safetensors``).
        save_precision: Target dtype — ``"bf16"``, ``"fp16"``, or ``"fp32"``.
        lora_injector: Optional :class:`LoRAInjector` with injected layers.
        lycoris_injector: Optional :class:`LyCORISInjector` with injected layers.

    Returns:
        The absolute path the merged checkpoint was written to.
    """

    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    target_dtype = dtype_map.get(save_precision.lower(), torch.bfloat16)

    # Work on a deep copy so the training model stays untouched.
    logger.info("export_merged_model: deep-copying model for merge …")
    merged_model = copy.deepcopy(model)

    # Merge LoRA if present
    lora_count = 0
    if lora_injector is not None and hasattr(lora_injector, "injected_layers") and lora_injector.injected_layers:
        lora_count = merge_lora_into_base(merged_model, lora_injector)

    # Merge LyCORIS if present
    lycoris_count = 0
    if lycoris_injector is not None and hasattr(lycoris_injector, "injected_layers") and lycoris_injector.injected_layers:
        lycoris_count = merge_lycoris_into_base(merged_model, lycoris_injector)

    if lora_count == 0 and lycoris_count == 0:
        logger.warning("export_merged_model: no adapter layers were merged — saving raw base weights")

    # Collect state dict
    state_dict = merged_model.state_dict()
    payload: Dict[str, torch.Tensor] = {}
    for key, tensor in state_dict.items():
        t = tensor.detach().cpu().to(target_dtype)
        payload[key] = t

    # Free the copy immediately
    del merged_model
    torch.cuda.empty_cache()

    # Write
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if out.suffix.lower() == ".safetensors":
        try:
            from safetensors.torch import save_file
            save_file(payload, str(out))
        except ImportError:
            logger.warning("safetensors unavailable — falling back to torch.save")
            torch.save(payload, str(out.with_suffix(".pt")))
    else:
        torch.save(payload, str(out))

    logger.info(
        "export_merged_model: wrote %s (%d tensors, dtype=%s, LoRA=%d, LyCORIS=%d)",
        out, len(payload), str(target_dtype), lora_count, lycoris_count,
    )
    return str(out)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _replace_module(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    """Replace a sub-module inside *root* by its dotted path.

    ``dotted_name`` may look like ``"unet.down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q"``
    or include a prefix like ``"prefix.unet.…"``.  All segments but the last
    resolve to the parent module; the last segment is the attribute name.
    """

    parts = dotted_name.split(".")
    parent = root

    # Walk to the parent of the target
    for part in parts[:-1]:
        if hasattr(parent, part):
            child = getattr(parent, part)
            if isinstance(child, nn.Module):
                parent = child
            else:
                # Might be a nn.ModuleList or nn.ModuleDict
                if isinstance(parent, nn.ModuleList):
                    parent = parent[int(part)]
                elif isinstance(parent, nn.ModuleDict):
                    parent = parent[part]
                else:
                    logger.warning("_replace_module: cannot traverse %s in %s", part, dotted_name)
                    return
        elif isinstance(parent, nn.ModuleList):
            parent = parent[int(part)]
        elif isinstance(parent, nn.ModuleDict):
            parent = parent[part]
        else:
            logger.warning("_replace_module: cannot find segment %s in %s", part, dotted_name)
            return

    attr = parts[-1]
    if isinstance(parent, nn.ModuleList):
        parent[int(attr)] = new_module
    elif isinstance(parent, nn.ModuleDict):
        parent[attr] = new_module
    else:
        setattr(parent, attr, new_module)


def _get_submodule(root: nn.Module, dotted_name: str) -> Optional[nn.Module]:
    """Resolve a dotted module name to the actual ``nn.Module``."""

    parts = dotted_name.split(".")
    mod = root
    for part in parts:
        if hasattr(mod, part):
            mod = getattr(mod, part)
        elif isinstance(mod, nn.ModuleList):
            mod = mod[int(part)]
        elif isinstance(mod, nn.ModuleDict):
            mod = mod[part]
        else:
            return None
    return mod
