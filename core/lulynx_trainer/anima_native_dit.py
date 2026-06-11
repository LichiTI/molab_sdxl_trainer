"""Warehouse Anima native DiT checkpoint introspection.

This module only reads safetensors metadata and tensor shapes.  It does
not construct the transformer, run a forward pass, or claim that native
training is ready.
"""

from __future__ import annotations

import re
import math
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Protocol, Sequence, Tuple


_BLOCK_RE = re.compile(r"^net\.blocks\.(\d+)\.")
_GROUP_NAMES = ("self_attn", "cross_attn", "mlp", "mod", "llm_adapter")


@dataclass(frozen=True)
class AnimaNativeDiTIntrospection:
    """Shape-level description of an Anima ``net.*`` DiT checkpoint."""

    checkpoint_path: str
    format: str
    is_safetensors: bool
    is_native_dit: bool
    tensor_count: int
    block_count: int
    block_indices: Tuple[int, ...] = ()
    hidden_dim: Optional[int] = None
    x_embedder_input_dim: Optional[int] = None
    final_output_dim: Optional[int] = None
    latent_channels_hint: Optional[int] = None
    has_llm_adapter: bool = False
    has_x_embedder: bool = False
    has_final_layer: bool = False
    detected_groups: Dict[str, bool] = field(default_factory=dict)
    sample_keys: Tuple[str, ...] = ()
    limitations: Tuple[str, ...] = ()
    notes: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _NamedParameterLike(Protocol):
    def named_parameters(self) -> Iterable[Tuple[str, Any]]:
        ...


@dataclass(frozen=True)
class AnimaNativeParamGroups:
    """Parameter-name groups for future Anima optimizer construction."""

    self_attn: Tuple[str, ...] = ()
    cross_attn: Tuple[str, ...] = ()
    mlp: Tuple[str, ...] = ()
    mod: Tuple[str, ...] = ()
    llm_adapter: Tuple[str, ...] = ()
    other: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Tuple[str, ...]]:
        return asdict(self)


@dataclass(frozen=True)
class AnimaNativeWeightLoadReport:
    """Report for a strict native Anima safetensors weight-load smoke."""

    checkpoint_path: str
    selected_key_count: int
    module_key_count: int
    loaded_key_count: int
    missing_keys: Tuple[str, ...] = ()
    unexpected_keys: Tuple[str, ...] = ()
    skipped_key_count: int = 0
    device: str = "cpu"
    dtype_name: str = ""

    @property
    def strict_success(self) -> bool:
        return (
            self.selected_key_count == self.module_key_count == self.loaded_key_count
            and not self.missing_keys
            and not self.unexpected_keys
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AnimaNativeDiTStub:
    """Shape-only Anima DiT module-tree stub.

    This is intentionally not a complete training forward.  It exists so
    later wiring can inject a module with Anima-like parameter names and
    optimizer group discovery can be tested without loading a 4GB checkpoint.
    Parameters are allocated on PyTorch's ``meta`` device by default.
    """

    def __new__(
        cls,
        introspection: AnimaNativeDiTIntrospection,
        shapes: Optional[Mapping[str, Sequence[int]]] = None,
        *,
        device: str = "meta",
        dtype: Optional[Any] = None,
    ) -> Any:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("PyTorch is required to build AnimaNativeDiTStub") from exc

        class _WeightOnlyNorm(nn.Module):
            def __init__(self, shape: Sequence[int]):
                super().__init__()
                self.weight = nn.Parameter(
                    torch.empty(tuple(shape), device=device, dtype=dtype)
                )

        class _ProjectionGroup(nn.Module):
            def __init__(self, prefix: str):
                super().__init__()
                for attr in ("q_norm", "k_norm"):
                    norm_shape = _shape_for(shapes, f"{prefix}.{attr}.weight")
                    if norm_shape is not None:
                        setattr(self, attr, _WeightOnlyNorm(norm_shape))
                for attr in ("q_proj", "k_proj", "v_proj", "output_proj"):
                    layer = _linear_from_shape(
                        nn,
                        _shape_for(shapes, f"{prefix}.{attr}.weight"),
                        device=device,
                        dtype=dtype,
                    )
                    if layer is not None:
                        setattr(self, attr, layer)

        class _MlpGroup(nn.Module):
            def __init__(self, prefix: str):
                super().__init__()
                for attr in ("layer1", "layer2"):
                    layer = _linear_from_shape(
                        nn,
                        _shape_for(shapes, f"{prefix}.{attr}.weight"),
                        device=device,
                        dtype=dtype,
                    )
                    if layer is not None:
                        setattr(self, attr, layer)

        class _AnimaBlock(nn.Module):
            def __init__(self, index: int):
                super().__init__()
                prefix = f"net.blocks.{index}"
                if introspection.detected_groups.get("self_attn"):
                    self.self_attn = _ProjectionGroup(f"{prefix}.self_attn")
                if introspection.detected_groups.get("cross_attn"):
                    self.cross_attn = _ProjectionGroup(f"{prefix}.cross_attn")
                if introspection.detected_groups.get("mlp"):
                    self.mlp = _MlpGroup(f"{prefix}.mlp")
                for mod_name in (
                    "adaln_modulation_self_attn",
                    "adaln_modulation_cross_attn",
                    "adaln_modulation_mlp",
                ):
                    seq = _sequential_from_linear_shapes(
                        nn,
                        shapes,
                        f"{prefix}.{mod_name}",
                        device=device,
                        dtype=dtype,
                    )
                    if seq is not None:
                        setattr(self, mod_name, seq)

        class _LlmAdapterAttention(nn.Module):
            def __init__(self, prefix: str):
                super().__init__()
                for attr in ("q_norm", "k_norm"):
                    norm_shape = _shape_for(shapes, f"{prefix}.{attr}.weight")
                    if norm_shape is not None:
                        setattr(self, attr, _WeightOnlyNorm(norm_shape))
                for attr in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    layer = _linear_from_shape(
                        nn,
                        _shape_for(shapes, f"{prefix}.{attr}.weight"),
                        bias_shape=_shape_for(shapes, f"{prefix}.{attr}.bias"),
                        device=device,
                        dtype=dtype,
                    )
                    if layer is not None:
                        setattr(self, attr, layer)

        class _LlmAdapterBlock(nn.Module):
            def __init__(self, index: int):
                super().__init__()
                prefix = f"net.llm_adapter.blocks.{index}"
                self.self_attn = _LlmAdapterAttention(f"{prefix}.self_attn")
                self.cross_attn = _LlmAdapterAttention(f"{prefix}.cross_attn")
                for attr in ("norm_self_attn", "norm_cross_attn", "norm_mlp"):
                    norm_shape = _shape_for(shapes, f"{prefix}.{attr}.weight")
                    if norm_shape is not None:
                        setattr(self, attr, _WeightOnlyNorm(norm_shape))
                mlp0 = _linear_from_shape(
                    nn,
                    _shape_for(shapes, f"{prefix}.mlp.0.weight"),
                    bias_shape=_shape_for(shapes, f"{prefix}.mlp.0.bias"),
                    device=device,
                    dtype=dtype,
                )
                mlp2 = _linear_from_shape(
                    nn,
                    _shape_for(shapes, f"{prefix}.mlp.2.weight"),
                    bias_shape=_shape_for(shapes, f"{prefix}.mlp.2.bias"),
                    device=device,
                    dtype=dtype,
                )
                if mlp0 is not None and mlp2 is not None:
                    self.mlp = nn.Sequential(
                        OrderedDict(
                            [
                                ("0", mlp0),
                                ("1", nn.SiLU()),
                                ("2", mlp2),
                            ]
                        )
                    )

        class _LlmAdapter(nn.Module):
            def __init__(self):
                super().__init__()
                embed_shape = _shape_for(shapes, "net.llm_adapter.embed.weight")
                if embed_shape is not None and len(embed_shape) == 2:
                    self.embed = nn.Embedding(
                        int(embed_shape[0]),
                        int(embed_shape[1]),
                        device=device,
                        dtype=dtype,
                    )
                block_indices = _collect_prefixed_indices(shapes, "net.llm_adapter.blocks.")
                self.blocks = nn.ModuleList([_LlmAdapterBlock(index) for index in block_indices])
                norm_shape = _shape_for(shapes, "net.llm_adapter.norm.weight")
                if norm_shape is not None:
                    self.norm = _WeightOnlyNorm(norm_shape)
                out_proj = _linear_from_shape(
                    nn,
                    _shape_for(shapes, "net.llm_adapter.out_proj.weight"),
                    bias_shape=_shape_for(shapes, "net.llm_adapter.out_proj.bias"),
                    device=device,
                    dtype=dtype,
                )
                if out_proj is not None:
                    self.out_proj = out_proj
                proj = _linear_from_shape(
                    nn,
                    _shape_for(shapes, "net.llm_adapter.proj.weight"),
                    bias_shape=_shape_for(shapes, "net.llm_adapter.proj.bias"),
                    device=device,
                    dtype=dtype,
                )
                if proj is not None:
                    self.proj = proj

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                t1_shape = _shape_for(shapes, "net.t_embedder.1.linear_1.weight")
                t2_shape = _shape_for(shapes, "net.t_embedder.1.linear_2.weight")
                if t1_shape is not None and t2_shape is not None:
                    t_stage = nn.Module()
                    t_stage.linear_1 = _linear_from_shape(
                        nn,
                        t1_shape,
                        device=device,
                        dtype=dtype,
                    )
                    t_stage.linear_2 = _linear_from_shape(
                        nn,
                        t2_shape,
                        device=device,
                        dtype=dtype,
                    )
                    self.t_embedder = nn.Sequential(OrderedDict([("0", nn.Identity()), ("1", t_stage)]))
                t_norm_shape = _shape_for(shapes, "net.t_embedding_norm.weight")
                if t_norm_shape is not None:
                    self.t_embedding_norm = _WeightOnlyNorm(t_norm_shape)
                x_shape = _shape_for(shapes, "net.x_embedder.proj.1.weight")
                if x_shape is not None:
                    self.x_embedder = nn.Module()
                    self.x_embedder.proj = nn.Sequential(
                        OrderedDict(
                            [
                                ("0", nn.Identity()),
                                (
                                    "1",
                                    _linear_from_shape(
                                        nn,
                                        x_shape,
                                        device=device,
                                        dtype=dtype,
                                    ),
                                ),
                            ]
                        )
                    )

                self.blocks = nn.ModuleList(
                    [_AnimaBlock(index) for index in introspection.block_indices]
                )

                final_shape = _shape_for(shapes, "net.final_layer.linear.weight")
                if final_shape is not None:
                    self.final_layer = nn.Module()
                    final_mod = _sequential_from_linear_shapes(
                        nn,
                        shapes,
                        "net.final_layer.adaln_modulation",
                        device=device,
                        dtype=dtype,
                    )
                    if final_mod is not None:
                        self.final_layer.adaln_modulation = final_mod
                    self.final_layer.linear = _linear_from_shape(
                        nn,
                        final_shape,
                        device=device,
                        dtype=dtype,
                    )

                if introspection.has_llm_adapter:
                    self.llm_adapter = _LlmAdapter()

        class _AnimaNativeDiTStubModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.native_introspection = introspection
                self.is_shape_only_stub = True
                self.net = _Net()

            def forward(self, *args: Any, **kwargs: Any) -> Any:
                raise NotImplementedError(
                    "AnimaNativeDiTStub is a shape-only module-tree skeleton. "
                    "It is not the native Anima training forward and must not "
                    "be used to unblock training."
                )

        return _AnimaNativeDiTStubModule()


class AnimaNativeDiTTinyTrainable:
    """Tiny trainable Anima-style DiT used only for native contract smoke tests.

    The module is deliberately small and synthetic.  It validates the contracts
    Anima training needs (patch latent input, timestep/text conditioning,
    flow-style output shape, LoRA-targetable DiT blocks) without claiming that
    real Anima checkpoint loading is ready.
    """

    def __new__(
        cls,
        *,
        latent_channels: int = 16,
        hidden_dim: int = 8,
        patch_size: int = 2,
        block_count: int = 2,
        condition_dim: int = 8,
        device: str = "cpu",
        dtype: Optional[Any] = None,
    ) -> Any:
        try:
            import torch
            from torch import nn
        except ImportError as exc:
            raise ImportError("PyTorch is required to build AnimaNativeDiTTinyTrainable") from exc

        class _TinyAttention(nn.Module):
            def __init__(self):
                super().__init__()
                self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
                self.output_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)

            def forward(self, x: Any, context: Optional[Any] = None) -> Any:
                source = x if context is None else context
                q = self.q_proj(x)
                k = self.k_proj(source)
                v = self.v_proj(source)
                scale = max(q.shape[-1], 1) ** -0.5
                attn = torch.softmax(q @ k.transpose(-1, -2) * scale, dim=-1)
                return self.output_proj(attn @ v)

        class _TinyMlp(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(hidden_dim, hidden_dim * 2, bias=False)
                self.layer2 = nn.Linear(hidden_dim * 2, hidden_dim, bias=False)

            def forward(self, x: Any) -> Any:
                return self.layer2(torch.nn.functional.silu(self.layer1(x)))

        class _TinyBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.self_attn = _TinyAttention()
                self.cross_attn = _TinyAttention()
                self.mlp = _TinyMlp()
                self.adaln_modulation = nn.Sequential(
                    OrderedDict(
                        [
                            ("0", nn.SiLU()),
                            ("1", nn.Linear(hidden_dim, hidden_dim * 6, bias=False)),
                        ]
                    )
                )

            def forward(self, x: Any, context: Any, cond: Any) -> Any:
                gate = self.adaln_modulation(cond).chunk(6, dim=-1)[-1].unsqueeze(1)
                x = x + self.self_attn(x)
                x = x + self.cross_attn(x, context)
                x = x + self.mlp(x) * torch.tanh(gate)
                return x

        class _TinyXEmbedder(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = nn.Sequential(
                    OrderedDict(
                        [
                            ("0", nn.Identity()),
                            (
                                "1",
                                nn.Linear(
                                    latent_channels * patch_size * patch_size,
                                    hidden_dim,
                                    bias=False,
                                ),
                            ),
                        ]
                    )
                )

            def forward(self, sample: Any) -> tuple[Any, int, int]:
                patches = torch.nn.functional.unfold(
                    sample,
                    kernel_size=patch_size,
                    stride=patch_size,
                ).transpose(1, 2)
                patch_h = sample.shape[-2] // patch_size
                patch_w = sample.shape[-1] // patch_size
                return self.proj(patches), patch_h, patch_w

        class _TinyFinalLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(
                    hidden_dim,
                    latent_channels * patch_size * patch_size,
                    bias=False,
                )

        class _TinyNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.x_embedder = _TinyXEmbedder()
                self.blocks = nn.ModuleList([_TinyBlock() for _ in range(block_count)])
                self.final_layer = _TinyFinalLayer()
                self.llm_adapter = nn.Module()
                self.llm_adapter.proj = nn.Linear(condition_dim, hidden_dim, bias=False)

        class _TinyAnimaDiT(nn.Module):
            def __init__(self):
                super().__init__()
                self.is_tiny_anima_trainable_smoke = True
                self.latent_channels = latent_channels
                self.patch_size = patch_size
                self.net = _TinyNet()
                self.timestep_embed = nn.Linear(1, hidden_dim, bias=False)

            def forward(
                self,
                sample: Any,
                timestep: Any,
                encoder_hidden_states: Any,
                added_cond_kwargs: Optional[Mapping[str, Any]] = None,
                **_: Any,
            ) -> Any:
                from types import SimpleNamespace

                if sample.dim() != 4:
                    raise ValueError("Anima tiny DiT expects BCHW latent tensors")
                if sample.shape[1] != latent_channels:
                    raise ValueError(
                        f"Expected {latent_channels} latent channels, got {sample.shape[1]}"
                    )
                if sample.shape[-1] % patch_size or sample.shape[-2] % patch_size:
                    raise ValueError("Latent spatial size must be divisible by patch_size")

                tokens, patch_h, patch_w = self.net.x_embedder(sample)
                timestep = timestep.reshape(sample.shape[0], -1).to(
                    device=sample.device,
                    dtype=sample.dtype,
                )
                if timestep.shape[-1] != 1:
                    timestep = timestep[:, :1]
                cond = self.timestep_embed(timestep)

                text_embeds = None
                if added_cond_kwargs:
                    text_embeds = added_cond_kwargs.get("text_embeds")
                if text_embeds is None:
                    text_embeds = encoder_hidden_states.mean(dim=1)
                cond = cond + self.net.llm_adapter.proj(text_embeds.to(dtype=sample.dtype))

                context = encoder_hidden_states.to(device=sample.device, dtype=sample.dtype)
                if context.shape[-1] != hidden_dim:
                    if context.shape[-1] > hidden_dim:
                        context = context[..., :hidden_dim]
                    else:
                        context = torch.nn.functional.pad(context, (0, hidden_dim - context.shape[-1]))

                x = tokens
                for block in self.net.blocks:
                    x = block(x, context, cond)

                patch_values = self.net.final_layer.linear(x).transpose(1, 2)
                output = torch.nn.functional.fold(
                    patch_values,
                    output_size=(patch_h * patch_size, patch_w * patch_size),
                    kernel_size=patch_size,
                    stride=patch_size,
                )
                return SimpleNamespace(sample=output)

        module = _TinyAnimaDiT()
        kwargs: Dict[str, Any] = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return module.to(**kwargs)


def inspect_anima_safetensors(
    path: str | Path,
    *,
    disable_mmap: bool = False,
) -> AnimaNativeDiTIntrospection:
    """Inspect an Anima single-file checkpoint without loading tensors.

    Args:
        path: Path to a ``.safetensors`` checkpoint.
        disable_mmap: If True, materialise the file fully into RAM (avoids
            pathological random reads on network/HDD storage).

    Returns:
        Shape-level native DiT report.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the path is not a safetensors file.
        ImportError: If ``safetensors`` is unavailable.
    """
    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Anima checkpoint does not exist: {checkpoint}")
    if checkpoint.suffix.lower() != ".safetensors":
        raise ValueError(
            f"Anima native introspection only supports .safetensors files: {checkpoint}"
        )

    try:
        from core.lulynx_trainer.safetensors_loader import open_safetensors
    except ImportError as exc:
        raise ImportError(
            "safetensors is required for Anima native checkpoint introspection"
        ) from exc

    shapes: Dict[str, Tuple[int, ...]] = {}
    with open_safetensors(str(checkpoint), framework="pt", device="cpu", disable_mmap=disable_mmap) as handle:
        for key in handle.keys():
            shapes[key] = tuple(handle.get_slice(key).get_shape())

    return introspect_anima_state_shapes(shapes, checkpoint_path=str(checkpoint))


def introspect_anima_state_shapes(
    shapes: Mapping[str, Sequence[int]],
    checkpoint_path: str = "",
) -> AnimaNativeDiTIntrospection:
    """Build an Anima native DiT report from a key-to-shape mapping."""
    keys = tuple(sorted(shapes))
    block_indices = _collect_block_indices(keys)
    detected_groups = _detect_groups(keys)
    is_native_dit = _looks_like_native_dit(keys, detected_groups)

    hidden_dim = _infer_hidden_dim(shapes)
    x_embedder_input_dim = _shape_dim(shapes, "net.x_embedder.proj.1.weight", 1)
    final_output_dim = _shape_dim(shapes, "net.final_layer.linear.weight", 0)
    latent_channels_hint = _infer_latent_channels(final_output_dim)

    limitations: List[str] = [
        "introspection_only_no_transformer_module",
        "native_forward_contract_not_wired",
        "qwen_t5_llm_conditioning_not_wired",
        "trainer_guard_must_remain_blocked",
    ]
    notes: List[str] = []

    if is_native_dit:
        notes.append(
            "Detected Anima net.* DiT checkpoint; this is not an SDXL UNet contract."
        )
    else:
        limitations.append("native_dit_signature_not_confirmed")
        notes.append("Checkpoint keys do not fully match the expected Anima net.* DiT signature.")

    if x_embedder_input_dim is not None and x_embedder_input_dim != 68:
        notes.append(f"Unexpected x_embedder input dim: {x_embedder_input_dim}.")
    if final_output_dim is not None and final_output_dim != 64:
        notes.append(f"Unexpected final output dim: {final_output_dim}.")

    return AnimaNativeDiTIntrospection(
        checkpoint_path=checkpoint_path,
        format="single_file",
        is_safetensors=True,
        is_native_dit=is_native_dit,
        tensor_count=len(keys),
        block_count=len(block_indices),
        block_indices=block_indices,
        hidden_dim=hidden_dim,
        x_embedder_input_dim=x_embedder_input_dim,
        final_output_dim=final_output_dim,
        latent_channels_hint=latent_channels_hint,
        has_llm_adapter=any(key.startswith("net.llm_adapter.") for key in keys),
        has_x_embedder=any(key.startswith("net.x_embedder.") for key in keys),
        has_final_layer=any(key.startswith("net.final_layer.") for key in keys),
        detected_groups=detected_groups,
        sample_keys=keys[:24],
        limitations=tuple(limitations),
        notes=tuple(notes),
    )


def discover_anima_native_param_groups(
    source: (
        AnimaNativeDiTIntrospection
        | Mapping[str, Sequence[int]]
        | Iterable[str]
        | _NamedParameterLike
    ),
) -> AnimaNativeParamGroups:
    """Discover Anima native parameter groups from shapes, names, or a module.

    The result is name-only and is intended for the next phase of optimizer
    group wiring.  It does not make parameters trainable and does not imply
    that the native forward is ready.
    """
    names = tuple(sorted(_extract_param_names(source)))
    grouped: Dict[str, List[str]] = {name: [] for name in _GROUP_NAMES}
    other: List[str] = []

    for name in names:
        group = _classify_anima_param_name(name)
        if group is None:
            other.append(name)
        else:
            grouped[group].append(name)

    return AnimaNativeParamGroups(
        self_attn=tuple(grouped["self_attn"]),
        cross_attn=tuple(grouped["cross_attn"]),
        mlp=tuple(grouped["mlp"]),
        mod=tuple(grouped["mod"]),
        llm_adapter=tuple(grouped["llm_adapter"]),
        other=tuple(other),
    )


def build_anima_native_dit_stub(
    introspection: AnimaNativeDiTIntrospection,
    shapes: Optional[Mapping[str, Sequence[int]]] = None,
    *,
    device: str = "meta",
    dtype: Optional[Any] = None,
) -> Any:
    """Build a shape-only Anima module-tree stub.

    This helper keeps the intentionally blocked state explicit: the returned
    module mirrors injectable parameter names, but its ``forward`` raises.
    """
    return AnimaNativeDiTStub(
        introspection=introspection,
        shapes=shapes,
        device=device,
        dtype=dtype,
    )


def load_anima_native_weight_subset(
    path: str | Path,
    *,
    prefixes: Sequence[str],
    device: str = "meta",
    dtype: Optional[Any] = None,
    disable_mmap: bool = False,
) -> Tuple[Any, AnimaNativeWeightLoadReport]:
    """Strictly load a selected Anima checkpoint subset into the native skeleton.

    This is an intermediate validation step between metadata introspection and
    a real training forward.  It proves the checkpoint tensors can be mapped
    into our Warehouse module names without loading the full 4GB checkpoint
    into memory.
    """

    checkpoint = Path(path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Anima checkpoint does not exist: {checkpoint}")
    if not prefixes:
        raise ValueError("At least one prefix is required for subset loading")

    try:
        import torch
        from core.lulynx_trainer.safetensors_loader import open_safetensors
    except ImportError as exc:
        raise ImportError("PyTorch and safetensors are required for Anima weight loading") from exc

    normalized_prefixes = tuple(str(prefix) for prefix in prefixes)
    selected_shapes: Dict[str, Tuple[int, ...]] = {}
    selected_tensors: Dict[str, Any] = {}
    total_key_count = 0
    with open_safetensors(str(checkpoint), framework="pt", device="cpu", disable_mmap=disable_mmap) as handle:
        for key in handle.keys():
            total_key_count += 1
            if not key.startswith(normalized_prefixes):
                continue
            selected_shapes[key] = tuple(handle.get_slice(key).get_shape())
            tensor = handle.get_tensor(key)
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            selected_tensors[key] = tensor

    if not selected_tensors:
        raise ValueError(
            f"No Anima tensors matched prefixes {normalized_prefixes!r} in {checkpoint}"
        )

    introspection = introspect_anima_state_shapes(selected_shapes, checkpoint_path=str(checkpoint))
    module = build_anima_native_dit_stub(
        introspection,
        selected_shapes,
        device=device,
        dtype=dtype,
    )
    incompatible = module.load_state_dict(selected_tensors, strict=True, assign=True)
    module_keys = set(module.state_dict().keys())
    selected_keys = set(selected_tensors)
    report = AnimaNativeWeightLoadReport(
        checkpoint_path=str(checkpoint),
        selected_key_count=len(selected_keys),
        module_key_count=len(module_keys),
        loaded_key_count=len(selected_keys & module_keys),
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
        skipped_key_count=total_key_count - len(selected_keys),
        device=str(next(module.parameters()).device) if any(True for _ in module.parameters()) else device,
        dtype_name=str(dtype or ""),
    )
    if not report.strict_success:
        raise RuntimeError(f"Anima native subset load failed strict check: {report.to_dict()}")
    return module, report


class AnimaNativeExecutableSubset:
    """Executable Warehouse subset of the native Anima DiT.

    The subset is intentionally small: it supports x-embedding, a configurable
    list of DiT blocks, and the final projection.  It is used for real-weight
    forward smokes before the full loader/trainer guard is unlocked.
    """

    def __new__(
        cls,
        shapes: Mapping[str, Sequence[int]],
        *,
        block_indices: Sequence[int] = (0,),
        device: str = "cpu",
        dtype: Optional[Any] = None,
    ) -> Any:
        try:
            import torch
            from torch import nn
            import torch.nn.functional as F
        except ImportError as exc:
            raise ImportError("PyTorch is required for AnimaNativeExecutableSubset") from exc

        class _RmsNorm(nn.Module):
            def __init__(self, width: int, eps: float = 1e-6, affine: bool = True):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(width, device=device, dtype=dtype)) if affine else None

            def forward(self, x: Any) -> Any:
                out = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
                if self.weight is not None:
                    out = out * self.weight
                return out

        class _ProjectionAttention(nn.Module):
            def __init__(self, prefix: str):
                super().__init__()
                q_shape = _shape_for(shapes, f"{prefix}.q_proj.weight")
                k_shape = _shape_for(shapes, f"{prefix}.k_proj.weight")
                v_shape = _shape_for(shapes, f"{prefix}.v_proj.weight")
                o_shape = _shape_for(shapes, f"{prefix}.output_proj.weight")
                q_norm_shape = _shape_for(shapes, f"{prefix}.q_norm.weight")
                k_norm_shape = _shape_for(shapes, f"{prefix}.k_norm.weight")
                if q_shape is None or k_shape is None or v_shape is None or o_shape is None:
                    raise ValueError(f"Incomplete attention projection shapes for {prefix}")
                if q_norm_shape is None or k_norm_shape is None:
                    raise ValueError(f"Incomplete attention q/k norm shapes for {prefix}")
                self.hidden_dim = int(q_shape[0])
                self.head_dim = int(q_norm_shape[0])
                if self.hidden_dim % self.head_dim != 0:
                    raise ValueError(f"hidden_dim {self.hidden_dim} is not divisible by head_dim {self.head_dim}")
                self.num_heads = self.hidden_dim // self.head_dim
                self.q_proj = _linear_from_shape(nn, q_shape, device=device, dtype=dtype)
                self.k_proj = _linear_from_shape(nn, k_shape, device=device, dtype=dtype)
                self.v_proj = _linear_from_shape(nn, v_shape, device=device, dtype=dtype)
                self.output_proj = _linear_from_shape(nn, o_shape, device=device, dtype=dtype)
                self.q_norm = _RmsNorm(self.head_dim)
                self.k_norm = _RmsNorm(self.head_dim)

            def _split_heads(self, tensor: Any) -> Any:
                batch, tokens, width = tensor.shape
                return tensor.view(batch, tokens, self.num_heads, self.head_dim).transpose(1, 2)

            def _merge_heads(self, tensor: Any) -> Any:
                batch, _heads, tokens, _head_dim = tensor.shape
                return tensor.transpose(1, 2).reshape(batch, tokens, self.hidden_dim)

            def forward(self, x: Any, context: Optional[Any] = None) -> Any:
                source = x if context is None else context
                q = self.q_norm(self._split_heads(self.q_proj(x)))
                k = self.k_norm(self._split_heads(self.k_proj(source)))
                v = self._split_heads(self.v_proj(source))
                attn = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
                return self.output_proj(self._merge_heads(attn))

        class _Mlp(nn.Module):
            def __init__(self, prefix: str):
                super().__init__()
                self.layer1 = _linear_from_shape(
                    nn,
                    _shape_for(shapes, f"{prefix}.layer1.weight"),
                    device=device,
                    dtype=dtype,
                )
                self.layer2 = _linear_from_shape(
                    nn,
                    _shape_for(shapes, f"{prefix}.layer2.weight"),
                    device=device,
                    dtype=dtype,
                )
                if self.layer1 is None or self.layer2 is None:
                    raise ValueError(f"Incomplete MLP shapes for {prefix}")

            def forward(self, x: Any) -> Any:
                return self.layer2(F.silu(self.layer1(x)))

        class _AdaLn(nn.Module):
            def __init__(self, prefix: str, chunks: int):
                super().__init__()
                layers: List[tuple[str, Any]] = []
                for layer_index in ("1", "2"):
                    layer = _linear_from_shape(
                        nn,
                        _shape_for(shapes, f"{prefix}.{layer_index}.weight"),
                        device=device,
                        dtype=dtype,
                    )
                    if layer is not None:
                        layers.append((layer_index, layer))
                if not layers:
                    raise ValueError(f"Missing AdaLN modulation shapes for {prefix}")
                self.add_module("0", nn.SiLU())
                for name, layer in layers:
                    self.add_module(name, layer)
                self.chunks = chunks

            def forward(self, emb: Any) -> Tuple[Any, ...]:
                out = emb
                for layer in self._modules.values():
                    out = layer(out)
                return out.chunk(self.chunks, dim=-1)

        class _Block(nn.Module):
            def __init__(self, index: int):
                super().__init__()
                prefix = f"net.blocks.{index}"
                self.self_attn = _ProjectionAttention(f"{prefix}.self_attn")
                self.cross_attn = _ProjectionAttention(f"{prefix}.cross_attn")
                self.mlp = _Mlp(f"{prefix}.mlp")
                self.adaln_modulation_self_attn = _AdaLn(f"{prefix}.adaln_modulation_self_attn", 3)
                self.adaln_modulation_cross_attn = _AdaLn(f"{prefix}.adaln_modulation_cross_attn", 3)
                self.adaln_modulation_mlp = _AdaLn(f"{prefix}.adaln_modulation_mlp", 3)

            def _apply_adaln(self, x: Any, shift: Any, scale: Any) -> Any:
                normalized = F.layer_norm(x.float(), (x.shape[-1],)).to(dtype=x.dtype)
                return normalized * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

            def _forward_impl(self, x: Any, emb: Any, context: Any, adaln_lora: Optional[Any] = None) -> Any:
                if emb.dim() == 3:
                    emb = emb[:, 0]
                shift, scale, gate = self._with_adaln_lora(
                    self.adaln_modulation_self_attn(emb),
                    adaln_lora,
                )
                x = x + gate.unsqueeze(1) * self.self_attn(self._apply_adaln(x, shift, scale))
                shift, scale, gate = self._with_adaln_lora(
                    self.adaln_modulation_cross_attn(emb),
                    adaln_lora,
                )
                x = x + gate.unsqueeze(1) * self.cross_attn(self._apply_adaln(x, shift, scale), context)
                shift, scale, gate = self._with_adaln_lora(
                    self.adaln_modulation_mlp(emb),
                    adaln_lora,
                )
                x = x + gate.unsqueeze(1) * self.mlp(self._apply_adaln(x, shift, scale))
                return x

            def forward(self, x: Any, emb: Any, context: Any, adaln_lora: Optional[Any] = None) -> Any:
                return self._forward_impl(x, emb, context, adaln_lora)

            def _with_adaln_lora(self, chunks: Tuple[Any, ...], adaln_lora: Optional[Any]) -> Tuple[Any, Any, Any]:
                if adaln_lora is None:
                    return chunks
                if adaln_lora.dim() == 3:
                    adaln_lora = adaln_lora[:, 0]
                lora_chunks = adaln_lora.chunk(3, dim=-1)
                return tuple(chunks[index] + lora_chunks[index] for index in range(3))

        class _FinalLayer(nn.Module):
            def __init__(self):
                super().__init__()
                self.adaln_modulation = _AdaLn("net.final_layer.adaln_modulation", 2)
                self.linear = _linear_from_shape(
                    nn,
                    _shape_for(shapes, "net.final_layer.linear.weight"),
                    device=device,
                    dtype=dtype,
                )
                if self.linear is None:
                    raise ValueError("Missing final layer linear shape")

            def forward(self, x: Any, emb: Any, adaln_lora: Optional[Any] = None) -> Any:
                if emb.dim() == 3:
                    emb = emb[:, 0]
                shift, scale = self.adaln_modulation(emb)
                if adaln_lora is not None:
                    if adaln_lora.dim() == 3:
                        adaln_lora = adaln_lora[:, 0]
                    shift_lora, scale_lora = adaln_lora[:, : shift.shape[-1]], adaln_lora[:, shift.shape[-1] : 2 * shift.shape[-1]]
                    shift = shift + shift_lora
                    scale = scale + scale_lora
                normalized = F.layer_norm(x.float(), (x.shape[-1],)).to(dtype=x.dtype)
                return self.linear(normalized * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1))

        class _Timesteps(nn.Module):
            def __init__(self, width: int):
                super().__init__()
                self.width = width

            def forward(self, timesteps: Any) -> Any:
                if timesteps.dim() == 1:
                    timesteps = timesteps.unsqueeze(1)
                values = timesteps.flatten().float()
                half = self.width // 2
                exponent = -math.log(10000.0) * torch.arange(
                    half,
                    dtype=torch.float32,
                    device=values.device,
                )
                exponent = exponent / float(half)
                freqs = torch.exp(exponent)
                args = values[:, None] * freqs[None, :]
                emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
                if emb.shape[-1] < self.width:
                    emb = F.pad(emb, (0, self.width - emb.shape[-1]))
                return emb.to(dtype=timesteps.dtype).reshape(timesteps.shape[0], timesteps.shape[1], -1)

        class _TimestepProjection(nn.Module):
            def __init__(self):
                super().__init__()
                linear1 = _linear_from_shape(
                    nn,
                    _shape_for(shapes, "net.t_embedder.1.linear_1.weight"),
                    device=device,
                    dtype=dtype,
                )
                linear2 = _linear_from_shape(
                    nn,
                    _shape_for(shapes, "net.t_embedder.1.linear_2.weight"),
                    device=device,
                    dtype=dtype,
                )
                if linear1 is None or linear2 is None:
                    raise ValueError("Missing timestep projection shapes")
                self.linear_1 = linear1
                self.linear_2 = linear2

            def forward(self, sample: Any) -> Tuple[Any, Any]:
                adaln_lora = self.linear_2(F.silu(self.linear_1(sample)))
                return sample, adaln_lora

        class _TimestepEmbedder(nn.Module):
            def __init__(self):
                super().__init__()
                width = _shape_dim(shapes, "net.t_embedder.1.linear_1.weight", 1)
                if width is None:
                    raise ValueError("Missing timestep feature width")
                self.add_module("0", _Timesteps(width))
                self.add_module("1", _TimestepProjection())

            def forward(self, timesteps: Any) -> Tuple[Any, Any]:
                features = self._modules["0"](timesteps)
                return self._modules["1"](features)

        class _XEmbedder(nn.Module):
            def __init__(self):
                super().__init__()
                layer = _linear_from_shape(
                    nn,
                    _shape_for(shapes, "net.x_embedder.proj.1.weight"),
                    device=device,
                    dtype=dtype,
                )
                if layer is None:
                    raise ValueError("Missing x_embedder shape")
                self.proj = nn.Sequential(OrderedDict([("0", nn.Identity()), ("1", layer)]))

        class _Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.t_embedder = _TimestepEmbedder()
                t_norm_shape = _shape_for(shapes, "net.t_embedding_norm.weight")
                if t_norm_shape is None:
                    raise ValueError("Missing timestep embedding norm shape")
                self.t_embedding_norm = _RmsNorm(int(t_norm_shape[0]))
                self.x_embedder = _XEmbedder()
                self.blocks = nn.ModuleList([_Block(index) for index in block_indices])
                self.final_layer = _FinalLayer()

        class _ExecutableSubsetModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.is_anima_executable_subset = True
                self.anima_block_checkpointing = False
                self.anima_block_checkpointing_mode = "off"
                self.anima_block_checkpointed_blocks = 0
                self.net = _Net()

            def set_anima_block_checkpointing(self, enabled: bool, mode: str = "block") -> Dict[str, Any]:
                normalized = str(mode or "block").strip().lower().replace("-", "_")
                if normalized in {"", "on", "true"}:
                    normalized = "block"
                active = bool(enabled) and normalized in {"block", "selective"}
                self.anima_block_checkpointing = active
                self.anima_block_checkpointing_mode = normalized if active else "off"
                self.anima_block_checkpointed_blocks = len(self.net.blocks) if active else 0
                return {
                    "enabled": active,
                    "mode": self.anima_block_checkpointing_mode,
                    "block_count": len(self.net.blocks),
                    "checkpointed_blocks": self.anima_block_checkpointed_blocks,
                }

            def _checkpoint_block(
                self,
                block: Any,
                x: Any,
                emb: Any,
                context: Any,
                adaln_lora: Optional[Any],
            ) -> Any:
                from torch.utils.checkpoint import checkpoint
                context_fn = None
                if str(getattr(self, "anima_block_checkpointing_mode", "") or "") == "selective":
                    try:
                        from .checkpoint_policy import build_selective_checkpoint_context_fn
                    except ImportError:  # pragma: no cover - direct script usage
                        from checkpoint_policy import build_selective_checkpoint_context_fn
                    context_fn = build_selective_checkpoint_context_fn("balanced")

                if adaln_lora is None:
                    def block_forward(x_arg: Any, emb_arg: Any, context_arg: Any) -> Any:
                        return block(x_arg, emb_arg, context_arg, None)

                    kwargs = {
                        "use_reentrant": False,
                        "preserve_rng_state": False,
                    }
                    if context_fn is not None:
                        kwargs["context_fn"] = context_fn
                    return checkpoint(block_forward, x, emb, context, **kwargs)

                def block_forward(x_arg: Any, emb_arg: Any, context_arg: Any, adaln_arg: Any) -> Any:
                    return block(x_arg, emb_arg, context_arg, adaln_arg)

                kwargs = {
                    "use_reentrant": False,
                    "preserve_rng_state": False,
                }
                if context_fn is not None:
                    kwargs["context_fn"] = context_fn
                return checkpoint(block_forward, x, emb, context, adaln_lora, **kwargs)

            def _run_blocks(
                self,
                x: Any,
                emb: Any,
                context: Any,
                adaln_lora: Optional[Any] = None,
            ) -> Any:
                use_checkpoint = (
                    bool(getattr(self, "anima_block_checkpointing", False))
                    and self.training
                    and torch.is_grad_enabled()
                )
                controller = getattr(self, "_lulynx_dit_prefetch_controller", None)
                try:
                    from .spectrum_probe import has_spectrum_step_context, observe_block_call
                    from .smoothcache import (
                        has_smoothcache_step_context,
                        observe_block_call as observe_smoothcache_block_call,
                    )
                    from .unified_cache_seam import get_active_cache_seam
                    from .dit_compute_reducer_seam import get_active_compute_reducer_seam
                except ImportError:  # pragma: no cover - direct-file smoke fallback.
                    from core.lulynx_trainer.spectrum_probe import has_spectrum_step_context, observe_block_call
                    from core.lulynx_trainer.smoothcache import (
                        has_smoothcache_step_context,
                        observe_block_call as observe_smoothcache_block_call,
                    )
                    from core.lulynx_trainer.unified_cache_seam import get_active_cache_seam
                    from core.lulynx_trainer.dit_compute_reducer_seam import get_active_compute_reducer_seam

                # Opt-in cache execution seam (default off -> bitwise parity).
                # Never cache during a checkpointed training forward.
                seam = None if use_checkpoint else get_active_cache_seam()
                seam_backend = seam.backend if seam is not None else ""
                # Opt-in block compute-reducer seam (TREAD/DiffCR/BlockSkip).
                # Default off -> get_active_compute_reducer_seam() is None ->
                # the plain block(...) path runs and the forward is bitwise
                # identical.  Never reduce during a checkpointed forward.
                reducer = None if use_checkpoint else get_active_compute_reducer_seam()

                for block_index, block in enumerate(self.net.blocks):
                    # The active seam backend observes inside its run_with_* call,
                    # so skip the standalone observe to avoid double-counting.
                    if has_spectrum_step_context() and seam_backend != "spectrum":
                        observe_block_call(block_index=block_index)
                    if has_smoothcache_step_context() and seam_backend != "smoothcache":
                        observe_smoothcache_block_call(block_index=block_index)
                    if controller is not None and hasattr(controller, "before_block"):
                        controller.before_block(block_index, x, emb, context, adaln_lora)
                    if use_checkpoint:
                        x = self._checkpoint_block(block, x, emb, context, adaln_lora)
                    elif seam is not None:
                        x = seam.run_block(block, block_index, x, emb, context, adaln_lora)
                    elif reducer is not None:
                        x = reducer.run_block(block, block_index, x, emb, context, adaln_lora)
                    else:
                        x = block(x, emb, context, adaln_lora)
                return x

            def forward(
                self,
                sample: Any,
                timestep: Any,
                encoder_hidden_states: Any,
                padding_mask: Optional[Any] = None,
            ) -> Any:
                from types import SimpleNamespace

                patches, patch_h, patch_w = patchify_anima_latents(sample, padding_mask)
                x = self.net.x_embedder.proj(patches)
                context = encoder_hidden_states.to(device=x.device, dtype=x.dtype)
                timesteps = timestep.to(device=x.device, dtype=x.dtype)
                emb, adaln_lora = self.net.t_embedder(timesteps)
                emb = self.net.t_embedding_norm(emb)
                x = self._run_blocks(x, emb, context, adaln_lora)
                patch_values = self.net.final_layer(x, emb, adaln_lora)
                output = unpatchify_anima_latents(
                    patch_values,
                    patch_h=patch_h,
                    patch_w=patch_w,
                )
                return SimpleNamespace(sample=output)

        module = _ExecutableSubsetModule()
        kwargs: Dict[str, Any] = {"device": device}
        if dtype is not None:
            kwargs["dtype"] = dtype
        return module.to(**kwargs)


def load_anima_native_executable_subset(
    path: str | Path,
    *,
    block_indices: Sequence[int] = (0,),
    device: str = "cpu",
    dtype: Optional[Any] = None,
    disable_mmap: bool = False,
) -> Tuple[Any, AnimaNativeWeightLoadReport]:
    """Load a real-weight executable Anima subset for forward smokes."""

    prefixes = ["net.x_embedder.", "net.t_embedder.", "net.t_embedding_norm.", "net.final_layer."]
    for index in block_indices:
        prefixes.append(f"net.blocks.{int(index)}.")

    checkpoint = Path(path)
    try:
        from core.lulynx_trainer.safetensors_loader import open_safetensors
    except ImportError as exc:
        raise ImportError("safetensors is required for Anima executable subset loading") from exc

    selected_shapes: Dict[str, Tuple[int, ...]] = {}
    selected_tensors: Dict[str, Any] = {}
    total_key_count = 0
    with open_safetensors(str(checkpoint), framework="pt", device="cpu", disable_mmap=disable_mmap) as handle:
        for key in handle.keys():
            total_key_count += 1
            if not key.startswith(tuple(prefixes)):
                continue
            selected_shapes[key] = tuple(handle.get_slice(key).get_shape())
            tensor = handle.get_tensor(key)
            if dtype is not None and tensor.is_floating_point():
                tensor = tensor.to(dtype=dtype)
            selected_tensors[key] = tensor

    module = AnimaNativeExecutableSubset(
        selected_shapes,
        block_indices=block_indices,
        device=device,
        dtype=dtype,
    )
    incompatible = module.load_state_dict(selected_tensors, strict=True, assign=True)
    module_keys = set(module.state_dict().keys())
    selected_keys = set(selected_tensors)
    report = AnimaNativeWeightLoadReport(
        checkpoint_path=str(checkpoint),
        selected_key_count=len(selected_keys),
        module_key_count=len(module_keys),
        loaded_key_count=len(selected_keys & module_keys),
        missing_keys=tuple(incompatible.missing_keys),
        unexpected_keys=tuple(incompatible.unexpected_keys),
        skipped_key_count=total_key_count - len(selected_keys),
        device=str(next(module.parameters()).device),
        dtype_name=str(dtype or ""),
    )
    if not report.strict_success:
        raise RuntimeError(f"Anima executable subset load failed strict check: {report.to_dict()}")
    return module, report


def patchify_anima_latents(
    sample: Any,
    padding_mask: Optional[Any] = None,
    *,
    patch_size: int = 2,
) -> Tuple[Any, int, int]:
    """Patchify Anima latents with the native padding-mask channel contract.

    Native Anima preview2 uses 16 latent channels plus one mask channel before
    2x2 patch embedding, producing 68 input features per visual token.
    """

    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for Anima patchification") from exc

    if sample.dim() != 4:
        raise ValueError("sample must be a BCHW latent tensor")
    if sample.shape[-2] % patch_size or sample.shape[-1] % patch_size:
        raise ValueError("sample spatial dimensions must be divisible by patch_size")

    if padding_mask is None:
        padding_mask = torch.zeros(
            (sample.shape[0], 1, sample.shape[-2], sample.shape[-1]),
            device=sample.device,
            dtype=sample.dtype,
        )
    else:
        padding_mask = padding_mask.to(device=sample.device, dtype=sample.dtype)
        if padding_mask.dim() != 4 or padding_mask.shape[0] != sample.shape[0] or padding_mask.shape[1] != 1:
            raise ValueError("padding_mask must be shaped [batch, 1, height, width]")
        if padding_mask.shape[-2:] != sample.shape[-2:]:
            padding_mask = torch.nn.functional.interpolate(
                padding_mask,
                size=sample.shape[-2:],
                mode="nearest",
            )

    merged = torch.cat([sample, padding_mask], dim=1)
    patches = torch.nn.functional.unfold(
        merged,
        kernel_size=patch_size,
        stride=patch_size,
    ).transpose(1, 2)
    return patches, sample.shape[-2] // patch_size, sample.shape[-1] // patch_size


def unpatchify_anima_latents(
    patch_values: Any,
    *,
    patch_h: int,
    patch_w: int,
    latent_channels: int = 16,
    patch_size: int = 2,
) -> Any:
    """Fold Anima final-layer patch values back to BCHW latents."""

    try:
        import torch
    except ImportError as exc:
        raise ImportError("PyTorch is required for Anima unpatchification") from exc

    if patch_values.dim() != 3:
        raise ValueError("patch_values must be shaped [batch, tokens, features]")
    expected_tokens = patch_h * patch_w
    expected_features = latent_channels * patch_size * patch_size
    if patch_values.shape[1] != expected_tokens:
        raise ValueError(
            f"Expected {expected_tokens} tokens for {patch_h}x{patch_w}, got {patch_values.shape[1]}"
        )
    if patch_values.shape[2] != expected_features:
        raise ValueError(
            f"Expected {expected_features} patch features, got {patch_values.shape[2]}"
        )
    return torch.nn.functional.fold(
        patch_values.transpose(1, 2),
        output_size=(patch_h * patch_size, patch_w * patch_size),
        kernel_size=patch_size,
        stride=patch_size,
    )


def _collect_block_indices(keys: Iterable[str]) -> Tuple[int, ...]:
    indices = set()
    for key in keys:
        match = _BLOCK_RE.match(key)
        if match:
            indices.add(int(match.group(1)))
    return tuple(sorted(indices))


def _detect_groups(keys: Iterable[str]) -> Dict[str, bool]:
    key_set = set(keys)
    return {
        "self_attn": _has_any_suffix(
            key_set,
            (
                ".self_attn.q_proj.weight",
                ".self_attn.k_proj.weight",
                ".self_attn.v_proj.weight",
                ".self_attn.output_proj.weight",
            ),
        ),
        "cross_attn": _has_any_suffix(
            key_set,
            (
                ".cross_attn.q_proj.weight",
                ".cross_attn.k_proj.weight",
                ".cross_attn.v_proj.weight",
                ".cross_attn.output_proj.weight",
            ),
        ),
        "mlp": _has_any_suffix(
            key_set,
            (
                ".mlp.layer1.weight",
                ".mlp.layer2.weight",
            ),
        ),
        "adaln_modulation": any(".adaln_modulation" in key for key in key_set),
        "llm_adapter": any(key.startswith("net.llm_adapter.") for key in key_set),
        "x_embedder": "net.x_embedder.proj.1.weight" in key_set,
        "final_layer": "net.final_layer.linear.weight" in key_set,
    }


def _looks_like_native_dit(keys: Sequence[str], detected_groups: Mapping[str, bool]) -> bool:
    if not any(key.startswith("net.") for key in keys):
        return False
    required = ("self_attn", "cross_attn", "mlp", "x_embedder", "final_layer")
    return all(detected_groups.get(group, False) for group in required)


def _has_any_suffix(keys: Iterable[str], suffixes: Iterable[str]) -> bool:
    suffix_tuple = tuple(suffixes)
    return any(key.endswith(suffix_tuple) for key in keys)


def _infer_hidden_dim(shapes: Mapping[str, Sequence[int]]) -> Optional[int]:
    for key in (
        "net.blocks.0.self_attn.q_proj.weight",
        "net.blocks.0.cross_attn.q_proj.weight",
        "net.final_layer.linear.weight",
    ):
        value = _shape_dim(shapes, key, 1)
        if value is not None:
            return value
    return None


def _shape_dim(
    shapes: Mapping[str, Sequence[int]],
    key: str,
    dim: int,
) -> Optional[int]:
    shape = shapes.get(key)
    if shape is None or len(shape) <= dim:
        return None
    return int(shape[dim])


def _infer_latent_channels(final_output_dim: Optional[int]) -> Optional[int]:
    # Known preview2 signature: final output is 64 = 16 latent channels * 2x2 patch.
    if final_output_dim == 64:
        return 16
    return None


def _extract_param_names(
    source: (
        AnimaNativeDiTIntrospection
        | Mapping[str, Sequence[int]]
        | Iterable[str]
        | _NamedParameterLike
    ),
) -> Tuple[str, ...]:
    if isinstance(source, AnimaNativeDiTIntrospection):
        return _names_from_introspection(source)
    if hasattr(source, "named_parameters"):
        return tuple(name for name, _param in source.named_parameters())
    if isinstance(source, Mapping):
        return tuple(str(name) for name in source.keys())
    return tuple(str(name) for name in source)


def _names_from_introspection(
    introspection: AnimaNativeDiTIntrospection,
) -> Tuple[str, ...]:
    names: List[str] = []
    for index in introspection.block_indices:
        prefix = f"net.blocks.{index}"
        if introspection.detected_groups.get("self_attn"):
            names.extend(
                f"{prefix}.self_attn.{proj}.weight"
                for proj in ("q_proj", "k_proj", "v_proj", "output_proj")
            )
        if introspection.detected_groups.get("cross_attn"):
            names.extend(
                f"{prefix}.cross_attn.{proj}.weight"
                for proj in ("q_proj", "k_proj", "v_proj", "output_proj")
            )
        if introspection.detected_groups.get("mlp"):
            names.extend(f"{prefix}.mlp.{layer}.weight" for layer in ("layer1", "layer2"))
        if introspection.detected_groups.get("adaln_modulation"):
            for mod_name in (
                "adaln_modulation_self_attn",
                "adaln_modulation_cross_attn",
                "adaln_modulation_mlp",
            ):
                names.extend(f"{prefix}.{mod_name}.{index}.weight" for index in ("1", "2"))
    if introspection.has_llm_adapter:
        names.append("net.llm_adapter.*")
    return tuple(names)


def _classify_anima_param_name(name: str) -> Optional[str]:
    normalized = name
    if normalized.startswith("net.blocks."):
        normalized = normalized[len("net.blocks.") :]
        parts = normalized.split(".", 1)
        normalized = parts[1] if len(parts) == 2 else normalized
    if ".blocks." in normalized:
        normalized = normalized.split(".blocks.", 1)[1]
        parts = normalized.split(".", 1)
        normalized = parts[1] if len(parts) == 2 else normalized

    if normalized.startswith("self_attn.") or ".self_attn." in name:
        return "self_attn"
    if normalized.startswith("cross_attn.") or ".cross_attn." in name:
        return "cross_attn"
    if normalized.startswith("mlp.") or ".mlp." in name:
        return "mlp"
    if (
        normalized.startswith("adaln_modulation.")
        or normalized.startswith("adaln_modulation_")
        or ".adaln_modulation." in name
        or ".adaln_modulation_" in name
        or ".modulation." in name
        or normalized.startswith("mod.")
    ):
        return "mod"
    if name.startswith("net.llm_adapter.") or ".llm_adapter." in name:
        return "llm_adapter"
    return None


def _shape_for(
    shapes: Optional[Mapping[str, Sequence[int]]],
    key: str,
) -> Optional[Tuple[int, ...]]:
    if shapes is None or key not in shapes:
        return None
    return tuple(int(dim) for dim in shapes[key])


def _iter_prefixed_shapes(
    shapes: Optional[Mapping[str, Sequence[int]]],
    prefix: str,
) -> Iterable[Tuple[str, Tuple[int, ...]]]:
    if shapes is None:
        return ()
    return (
        (name, tuple(int(dim) for dim in shape))
        for name, shape in shapes.items()
        if name.startswith(prefix)
    )


def _collect_prefixed_indices(
    shapes: Optional[Mapping[str, Sequence[int]]],
    prefix: str,
) -> Tuple[int, ...]:
    if shapes is None:
        return ()
    indices = set()
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.")
    for name in shapes:
        match = pattern.match(name)
        if match:
            indices.add(int(match.group(1)))
    return tuple(sorted(indices))


def _linear_from_shape(
    nn: Any,
    shape: Optional[Sequence[int]],
    *,
    bias_shape: Optional[Sequence[int]] = None,
    device: str,
    dtype: Optional[Any],
) -> Optional[Any]:
    if shape is None:
        return None
    if len(shape) != 2:
        raise ValueError(f"Expected linear weight shape [out, in], got {tuple(shape)}")
    kwargs: Dict[str, Any] = {"device": device}
    if dtype is not None:
        kwargs["dtype"] = dtype
    linear_cls = nn.Linear
    try:
        from .native_unet.weight_residency import LulynxManagedLinear

        linear_cls = LulynxManagedLinear
    except Exception:
        pass
    layer = linear_cls(int(shape[1]), int(shape[0]), bias=bias_shape is not None, **kwargs)
    if bias_shape is not None and tuple(int(dim) for dim in bias_shape) != (int(shape[0]),):
        raise ValueError(
            f"Expected linear bias shape [{int(shape[0])}], got {tuple(bias_shape)}"
        )
    return layer


def _sequential_from_linear_shapes(
    nn: Any,
    shapes: Optional[Mapping[str, Sequence[int]]],
    prefix: str,
    *,
    device: str,
    dtype: Optional[Any],
) -> Optional[Any]:
    layers: List[tuple[str, Any]] = []
    for index in ("1", "2"):
        layer = _linear_from_shape(
            nn,
            _shape_for(shapes, f"{prefix}.{index}.weight"),
            device=device,
            dtype=dtype,
        )
        if layer is not None:
            layers.append((index, layer))
    if not layers:
        return None
    if layers[0][0] != "0":
        layers.insert(0, ("0", nn.SiLU()))
    return nn.Sequential(OrderedDict(layers))


def _safe_module_attr(name: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z_]+", "_", name).strip("_")
    if not safe:
        return "adapter"
    if safe[0].isdigit():
        return f"adapter_{safe}"
    return safe

