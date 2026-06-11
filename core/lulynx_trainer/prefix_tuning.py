"""
Prefix / Postfix Soft-Prompt Tuning — learnable soft prompts prepended or
appended to the hidden-state sequence at a conditioning insertion point.

This implements feature #113.  The soft prompts are small embedding tensors
that are trained alongside the adapter weights and inserted into the model's
forward pass via a hook.

Typical usage
-------------
>>> from .prefix_tuning import install_prefix_tuning, remove_prefix_tuning
>>> hooks = install_prefix_tuning(model, prefix_length=8, postfix_length=4)
>>> # ... training ...
>>> remove_prefix_tuning(model)
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PrefixTuningModule(nn.Module):
    """Learnable prefix soft-prompt.

    Creates an embedding tensor of shape ``[prefix_length, hidden_size]``.
    In ``forward`` the prefix vectors are **prepended** to the input
    hidden-state sequence and the attention mask is extended accordingly.

    Parameters
    ----------
    hidden_size : int
        Dimensionality of the hidden states at the insertion point.
    prefix_length : int
        Number of prefix tokens to learn.
    init : str
        Initialisation strategy — ``"normal"``, ``"uniform"``, or ``"zeros"``.
    """

    def __init__(
        self,
        hidden_size: int,
        prefix_length: int,
        init: str = "normal",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.prefix_length = prefix_length

        self.prefix_embedding = nn.Parameter(
            torch.empty(prefix_length, hidden_size)
        )
        self._init_weights(init)

    def _init_weights(self, init: str) -> None:
        if init == "normal":
            nn.init.normal_(self.prefix_embedding, mean=0.0, std=0.02)
        elif init == "uniform":
            nn.init.uniform_(self.prefix_embedding, a=-0.02, b=0.02)
        elif init == "zeros":
            nn.init.zeros_(self.prefix_embedding)
        else:
            logger.warning("prefix_tuning init '%s' unknown, falling back to normal", init)
            nn.init.normal_(self.prefix_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Prepend the learned prefix to ``hidden_states``.

        Args:
            hidden_states: ``[batch, seq_len, hidden_size]``
            attention_mask: ``[batch, seq_len]`` (1 = valid, 0 = pad) or None.

        Returns:
            ``(hidden_states, attention_mask)`` with the prefix prepended.
        """
        batch = hidden_states.shape[0]
        prefix = self.prefix_embedding.unsqueeze(0).expand(batch, -1, -1)
        # Ensure dtype/device match
        prefix = prefix.to(dtype=hidden_states.dtype, device=hidden_states.device)

        hidden_states = torch.cat([prefix, hidden_states], dim=1)

        if attention_mask is not None:
            # Prefix tokens are always "valid"
            prefix_mask = attention_mask.new_ones(batch, self.prefix_length)
            attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)

        return hidden_states, attention_mask


class PostfixTuningModule(nn.Module):
    """Learnable postfix soft-prompt.

    Identical to :class:`PrefixTuningModule` but the vectors are **appended**
    after the input sequence.
    """

    def __init__(
        self,
        hidden_size: int,
        postfix_length: int,
        init: str = "normal",
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.postfix_length = postfix_length

        self.postfix_embedding = nn.Parameter(
            torch.empty(postfix_length, hidden_size)
        )
        self._init_weights(init)

    def _init_weights(self, init: str) -> None:
        if init == "normal":
            nn.init.normal_(self.postfix_embedding, mean=0.0, std=0.02)
        elif init == "uniform":
            nn.init.uniform_(self.postfix_embedding, a=-0.02, b=0.02)
        elif init == "zeros":
            nn.init.zeros_(self.postfix_embedding)
        else:
            logger.warning("postfix_tuning init '%s' unknown, falling back to normal", init)
            nn.init.normal_(self.postfix_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Append the learned postfix to ``hidden_states``.

        Args:
            hidden_states: ``[batch, seq_len, hidden_size]``
            attention_mask: ``[batch, seq_len]`` (1 = valid, 0 = pad) or None.

        Returns:
            ``(hidden_states, attention_mask)`` with the postfix appended.
        """
        batch = hidden_states.shape[0]
        postfix = self.postfix_embedding.unsqueeze(0).expand(batch, -1, -1)
        postfix = postfix.to(dtype=hidden_states.dtype, device=hidden_states.device)

        hidden_states = torch.cat([hidden_states, postfix], dim=1)

        if attention_mask is not None:
            postfix_mask = attention_mask.new_ones(batch, self.postfix_length)
            attention_mask = torch.cat([attention_mask, postfix_mask], dim=1)

        return hidden_states, attention_mask


# ---------------------------------------------------------------------------
# Installation helpers
# ---------------------------------------------------------------------------

# Registry of installed hooks so remove_prefix_tuning can clean up.
_PREFIX_TUNING_STATE: Dict[int, dict] = {}
"""Maps ``id(model)`` → {"hooks": [...], "modules": nn.ModuleDict, "insertion_module": nn.Module}"""


def install_prefix_tuning(
    model: nn.Module,
    prefix_length: int = 0,
    postfix_length: int = 0,
    target_family: str = "",
    init: str = "normal",
) -> List[torch.utils.hooks.RemovableHook]:
    """Install prefix/postfix soft-prompt hooks into *model*.

    The function identifies the first conditioning insertion point based on
    ``target_family`` and installs a forward hook that prepends/appends the
    soft prompt vectors to the hidden-state sequence.

    For **SDXL** the hook is attached after the text encoder output
    (``model.text_encoder_1``).
    For **Anima** the hook targets the first DiT block's self-attention
    hidden-state stream (``model.unet``).
    For **Newbie** the hook targets the transformer's first block input.
    For other families the hook is attached to ``model.unet`` (or the first
    child module if unet is unavailable).

    Args:
        model: The training model (typically a ``LoadedModel``-like object with
            ``unet``, ``text_encoder_1`` etc.).
        prefix_length: Number of prefix tokens (0 = disabled).
        postfix_length: Number of postfix tokens (0 = disabled).
        target_family: Model family hint (``"sdxl"``, ``"anima"``, ``"newbie"``,
            ``"flux"``, …).  Used to select the insertion point.
        init: Weight initialisation strategy.

    Returns:
        List of ``RemovableHook`` handles (for manual removal if needed).
    """

    if prefix_length <= 0 and postfix_length <= 0:
        logger.info("install_prefix_tuning: nothing to install (prefix=%d, postfix=%d)", prefix_length, postfix_length)
        return []

    # Determine insertion point
    insertion_module, hidden_size = _find_insertion_point(model, target_family)
    if insertion_module is None:
        logger.warning("install_prefix_tuning: cannot find insertion point for family=%s", target_family)
        return []

    if hidden_size <= 0:
        logger.warning("install_prefix_tuning: inferred hidden_size=%d, skipping", hidden_size)
        return []

    logger.info(
        "install_prefix_tuning: insertion point=%s, hidden_size=%d, prefix=%d, postfix=%d",
        _module_name(model, insertion_module), hidden_size, prefix_length, postfix_length,
    )

    # Build the soft-prompt modules
    modules = nn.ModuleDict()

    if prefix_length > 0:
        modules["prefix"] = PrefixTuningModule(hidden_size, prefix_length, init=init)
    if postfix_length > 0:
        modules["postfix"] = PostfixTuningModule(hidden_size, postfix_length, init=init)

    # Move modules to same device/dtype as insertion point
    device = next(insertion_module.parameters(), torch.tensor(0.0)).device
    dtype = next(insertion_module.parameters(), torch.tensor(0.0)).dtype
    modules = modules.to(device=device, dtype=dtype)

    # Register as sub-module so their parameters appear in model.parameters()
    if hasattr(model, "unet") and isinstance(model.unet, nn.Module):
        # Attach to the unet/transformer so the params are part of the right
        # parameter group in the optimizer.
        existing = getattr(model.unet, "_prefix_tuning_modules", None)
        if existing is not None:
            # Remove stale state first
            remove_prefix_tuning(model)
        model.unet._prefix_tuning_modules = modules
    else:
        existing = getattr(model, "_prefix_tuning_modules", None)
        if existing is not None:
            remove_prefix_tuning(model)
        model._prefix_tuning_modules = modules

    # Forward hook
    def _prefix_postfix_hook(module, args, kwargs, output):
        """Torch 2.x forward pre-post hook signature.

        We use a *post-hook* (``register_forward_hook``) so we can modify the
        hidden states that were produced by the insertion module.  The hook
        looks for ``hidden_states`` in the output and prepends/appends the
        soft prompts.
        """
        mods = modules

        # Unpack output
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = ()

        # Determine if attention_mask is present
        attention_mask = None
        # Check kwargs for attention_mask
        if isinstance(kwargs, dict):
            attention_mask = kwargs.get("attention_mask", None)

        # For tuple outputs from transformer blocks that may return
        # (hidden_states, ...) — only the first element is the hidden state.
        if "prefix" in mods:
            hidden_states, attention_mask = mods["prefix"](hidden_states, attention_mask)
        if "postfix" in mods:
            hidden_states, attention_mask = mods["postfix"](hidden_states, attention_mask)

        # Update kwargs if we have an attention_mask
        if attention_mask is not None and isinstance(kwargs, dict):
            kwargs["attention_mask"] = attention_mask

        if rest:
            return (hidden_states,) + rest
        return hidden_states

    # Register with torch.utils.hooks.register_forward_hook
    # We use the modern full-kwargs hook API (PyTorch >= 2.0)
    hook_handle = insertion_module.register_forward_hook(
        _prefix_postfix_hook, with_kwargs=True
    )

    handles = [hook_handle]

    # Save state for removal later
    _PREFIX_TUNING_STATE[id(model)] = {
        "handles": handles,
        "modules": modules,
        "insertion_module": insertion_module,
    }

    return handles


def remove_prefix_tuning(model: nn.Module) -> None:
    """Remove all prefix/postfix hooks and modules installed by
    :func:`install_prefix_tuning`.

    Safe to call multiple times.
    """
    state = _PREFIX_TUNING_STATE.pop(id(model), None)
    if state is not None:
        for h in state.get("handles", []):
            h.remove()
        # Remove sub-module reference
        if hasattr(model, "unet") and hasattr(model.unet, "_prefix_tuning_modules"):
            del model.unet._prefix_tuning_modules
        if hasattr(model, "_prefix_tuning_modules"):
            del model._prefix_tuning_modules
        logger.info("remove_prefix_tuning: hooks removed")

    # Also clean up if the sub-module was attached directly
    if hasattr(model, "unet") and hasattr(model.unet, "_prefix_tuning_modules"):
        del model.unet._prefix_tuning_modules
    if hasattr(model, "_prefix_tuning_modules"):
        del model._prefix_tuning_modules


def get_prefix_tuning_params(model: nn.Module) -> List[nn.Parameter]:
    """Return all trainable parameters from the installed prefix/postfix modules.

    Returns an empty list when no soft prompts are installed.
    """
    container = None
    if hasattr(model, "unet") and hasattr(model.unet, "_prefix_tuning_modules"):
        container = model.unet._prefix_tuning_modules
    elif hasattr(model, "_prefix_tuning_modules"):
        container = model._prefix_tuning_modules

    if container is None:
        return []

    params: List[nn.Parameter] = []
    for mod in container.values():
        params.extend(p for p in mod.parameters() if p.requires_grad)
    return params


def get_prefix_tuning_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return a state dict of the prefix/postfix soft-prompt parameters.

    Keys are of the form ``prefix_tuning.<module_key>.<param_name>`` so that
    prefix and postfix entries are always disjoint even when they share shapes.

    Returns an empty dict when no soft prompts are installed.
    """
    container = None
    if hasattr(model, "unet") and hasattr(model.unet, "_prefix_tuning_modules"):
        container = model.unet._prefix_tuning_modules
    elif hasattr(model, "_prefix_tuning_modules"):
        container = model._prefix_tuning_modules

    if container is None:
        return {}

    state: Dict[str, torch.Tensor] = {}
    for key, mod in container.items():
        for pname, param in mod.named_parameters():
            state[f"prefix_tuning.{key}.{pname}"] = param.data.detach().cpu()
    return state


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_insertion_point(model: nn.Module, target_family: str) -> Tuple[Optional[nn.Module], int]:
    """Return ``(module, hidden_size)`` for the conditioning insertion point."""

    family = target_family.lower().strip() if target_family else ""

    if family in ("anima",):
        # Anima DiT: first transformer block's self-attention.
        # The hidden_size can be read from the first Linear in the block.
        unet = getattr(model, "unet", None)
        if unet is None:
            return None, 0
        # Walk into the first block we can find
        for name, child in unet.named_children():
            if "block" in name.lower() or "transformer" in name.lower():
                hidden_size = _infer_hidden_size(child)
                return child, hidden_size
        # Fallback: the unet itself
        return unet, _infer_hidden_size(unet)

    if family in ("newbie",):
        unet = getattr(model, "unet", None)
        if unet is None:
            return None, 0
        # Newbie transformer — look for transformer blocks
        for name, child in unet.named_children():
            if "transformer" in name.lower() or "block" in name.lower():
                hidden_size = _infer_hidden_size(child)
                return child, hidden_size
        return unet, _infer_hidden_size(unet)

    if family in ("sdxl", "sd15", "flux", "sd3"):
        # For SD-family, prepend after the text encoder output.
        # However, since the text encoder output is typically consumed as a
        # pre-computed conditioning tensor and not through a module forward,
        # we instead hook the first UNet block to inject the prefix into its
        # hidden_states input.
        unet = getattr(model, "unet", None)
        if unet is None:
            return None, 0
        # Try to find first down-block
        for name, child in unet.named_children():
            if "down" in name.lower() or "block" in name.lower():
                hidden_size = _infer_hidden_size(child)
                return child, hidden_size
        return unet, _infer_hidden_size(unet)

    # Generic fallback — hook the unet or the model itself
    unet = getattr(model, "unet", None)
    if unet is not None:
        return unet, _infer_hidden_size(unet)
    return model, _infer_hidden_size(model)


def _infer_hidden_size(module: nn.Module) -> int:
    """Heuristic: find the first ``nn.Linear`` and return its ``in_features``."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
    # Try to read from a config attribute
    cfg = getattr(module, "config", None)
    if cfg is not None:
        for attr in ("hidden_size", "d_model", "dim", "embed_dim"):
            val = getattr(cfg, attr, None)
            if isinstance(val, int) and val > 0:
                return val
    return 0


def _module_name(root: nn.Module, target: nn.Module) -> str:
    """Return dotted name of *target* inside *root*, or "<unknown>"."""
    for name, mod in root.named_modules():
        if mod is target:
            return name
    return "<unknown>"
