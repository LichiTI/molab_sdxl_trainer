"""
Lumina Image 2.0 DiT adapter.

Architecture notes (for future integration reference, not execution):
  - Lumina Image 2.0 is a flow-matching DiT (diffusion transformer).
  - Text encoder: Gemma 2B (not CLIP / T5 — distinct contract).
  - Latent channels: 16 (same VAE family as Flux).
  - Transformer: single-stream blocks with cross-attention to text embeddings.
  - Timestep conditioning via adaLN (adaptive layer norm).
  - Scheduler: continuous-time flow matching (NOT DDPM / DDIM).
  - Training target: velocity field v = x1 - noise, same as Flux.

This adapter implements the DiTBase ABC for weight introspection, LoRA
target enumeration, and LoRA file parsing. It does NOT load the full
model into GPU memory — that belongs to the loader (lumina_loader.py)
and requires diffusers integration.

Implemented here:
  - LuminaConfig (DiTConfig subclass) with architecture defaults.
  - LuminaAdapter (DiTBase subclass) with all 5 abstract methods.
  - Layer parsing, LoRA target enumeration, LoRA file parsing.

NOT implemented (deferred to shared-core integration):
  - Full model loading via diffusers (lumina_loader.py).
  - LoRA injection at training time (needs injector wiring).
  - Training loop integration (flow-matching loss, timestep sampling).
  - Sampling / inference pipeline.
  - Config/model_family registration (requires edits to shared files).
"""

import logging
from typing import Dict, List, Set, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

from .base import DiTBase, DiTConfig, DiTType, DiTLayerInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lumina-specific DiT config
# ---------------------------------------------------------------------------

# Default LoRA target patterns (component-level names, not fully-qualified).
# These map to nn.Linear layers inside each transformer block.
_LUMINA_DEFAULT_TARGETS: Set[str] = {
    # Self-attention (image tokens attend to each other)
    "attn.to_q",
    "attn.to_k",
    "attn.to_v",
    "attn.to_out.0",
    # Cross-attention (image tokens attend to text embeddings)
    "attn.add_q_proj",
    "attn.add_k_proj",
    "attn.add_v_proj",
    "attn.to_add_out",
    # Feed-forward network
    "ff.net.0.proj",
    "ff.net.2",
}


@dataclass
class LuminaConfig(DiTConfig):
    """Lumina Image 2.0 architecture configuration.

    Defaults target the Lumina-2B variant.  Larger variants (e.g. 7B)
    would override hidden_size, num_attention_heads, and
    num_transformer_blocks.
    """

    dit_type: DiTType = DiTType.LUMINA

    # ── Transformer geometry ──────────────────────────────────────────────
    num_transformer_blocks: int = 24     # Single-stream transformer blocks
    num_single_blocks: int = 0           # Lumina has no separate "single" blocks
    hidden_size: int = 2304              # Lumina-2B hidden dim
    num_attention_heads: int = 24
    head_dim: int = 96                   # hidden_size // num_attention_heads
    mlp_ratio: float = 4.0

    # ── Text encoder contract ─────────────────────────────────────────────
    text_encoder_type: str = "gemma-2b"  # Gemma 2B (NOT CLIP / T5)
    text_encoder_dim: int = 2304         # Gemma 2B hidden size
    max_text_seq_len: int = 256          # Typical Gemma context for Lumina

    # ── Latent / VAE contract ─────────────────────────────────────────────
    latent_channels: int = 16            # Same VAE family as Flux
    vae_scaling_factor: float = 0.3611   # Flux-VAE compatible

    # ── Flow-matching scheduler contract ──────────────────────────────────
    # Lumina uses continuous-time flow matching, same family as Flux.
    # Training target: velocity v = x1 - noise.
    # Timestep range: [0, 1] continuous.
    discrete_flow_shift: float = 3.0     # Sigmoid shift for timestep sampling

    # ── File patterns for diffusers-format weights ────────────────────────
    model_file_patterns: tuple = (
        "diffusion_pytorch_model.safetensors",
        "diffusion_pytorch_model.fp16.safetensors",
    )

    # ── LoRA targets ──────────────────────────────────────────────────────
    lora_targets: Optional[Set[str]] = None

    def __post_init__(self):
        if self.lora_targets is None:
            self.lora_targets = set(_LUMINA_DEFAULT_TARGETS)
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // max(self.num_attention_heads, 1)


# ---------------------------------------------------------------------------
# Layer-type mapping (component name → category)
# ---------------------------------------------------------------------------

_LAYER_TYPE_MAP: Dict[str, str] = {
    "to_q": "attn",
    "to_k": "attn",
    "to_v": "attn",
    "to_out": "attn",
    "add_q_proj": "attn",
    "add_k_proj": "attn",
    "add_v_proj": "attn",
    "to_add_out": "attn",
    "net.0.proj": "ff",
    "net.2": "ff",
    "norm": "norm",
    "adaLN": "norm",
}


# ---------------------------------------------------------------------------
# LuminaAdapter
# ---------------------------------------------------------------------------

class LuminaAdapter(DiTBase):
    """DiT adapter for Lumina Image 2.0 models.

    Responsibilities (per DiTBase contract):
      - ``load(path)``: load model or LoRA safetensors into memory for introspection.
      - ``get_layer_info()``: enumerate all weight tensors with structural metadata.
      - ``get_lora_targets()``: return the set of fully-qualified layer-name patterns
        that are valid LoRA injection points.
      - ``parse_lora(lora_path)``: parse a LoRA safetensors file and return rank,
        layer mapping, and per-block statistics.

    This adapter is intentionally read-only.  It does NOT inject LoRA layers
    or perform forward passes — those are injector/trainer responsibilities.

    Block naming convention (matches diffusers Lumina transformer):
      - All transformer layers live under ``transformer_blocks.{idx}.*``
      - Each block contains self-attention, cross-attention, FFN, and norms.
    """

    # Fully-qualified LoRA patterns with a block-index placeholder.
    # The ``{}`` is replaced by the block index at runtime.
    LORA_PATTERNS: Set[str] = {
        # ── Self-attention ────────────────────────────────────────────────
        "transformer_blocks.{}.attn.to_q",
        "transformer_blocks.{}.attn.to_k",
        "transformer_blocks.{}.attn.to_v",
        "transformer_blocks.{}.attn.to_out.0",
        # ── Cross-attention (image attends to text) ──────────────────────
        "transformer_blocks.{}.attn.add_q_proj",
        "transformer_blocks.{}.attn.add_k_proj",
        "transformer_blocks.{}.attn.add_v_proj",
        "transformer_blocks.{}.attn.to_add_out",
        # ── Feed-forward network ─────────────────────────────────────────
        "transformer_blocks.{}.ff.net.0.proj",
        "transformer_blocks.{}.ff.net.2",
    }

    # Prefix that identifies a transformer block in the state dict keys.
    _BLOCK_PREFIX = "transformer_blocks."

    def __init__(self, config: Optional[LuminaConfig] = None):
        super().__init__(config or LuminaConfig())
        self._layers: List[DiTLayerInfo] = []

    # ── DiTBase abstract properties / methods ─────────────────────────────

    @property
    def dit_type(self) -> DiTType:
        return self.config.dit_type

    def load(self, path: str) -> bool:
        """Load model or LoRA weights from a safetensors file or directory.

        Supports:
          - Single ``.safetensors`` file (LoRA or full model).
          - Directory of ``*.safetensors`` files (diffusers format).

        Returns True on success, False on failure.
        """
        try:
            from safetensors import safe_open

            path_obj = Path(path)

            if not path_obj.exists():
                logger.error("[LuminaAdapter] Path not found: %s", path)
                return False

            if path_obj.suffix == ".safetensors":
                self._load_single_file(path_obj)
            elif path_obj.is_dir():
                self._load_directory(path_obj)
            else:
                logger.error("[LuminaAdapter] Unsupported path type: %s", path)
                return False

            self._parse_layers()
            self._loaded = True
            logger.info(
                "[LuminaAdapter] Loaded %d tensors (%d layers parsed) from %s",
                len(self._state_dict),
                len(self._layers),
                path,
            )
            return True

        except Exception as exc:
            logger.error("[LuminaAdapter] Load failed for %s: %s", path, exc)
            return False

    def get_layer_info(self) -> List[DiTLayerInfo]:
        """Return structural metadata for every parsed weight tensor."""
        return self._layers

    def get_lora_targets(self) -> Set[str]:
        """Return component-level LoRA target names.

        These are the short names (e.g. ``attn.to_q``) that the injector
        uses to match ``nn.Linear`` modules inside each transformer block.
        Fully-qualified patterns (with block index) are available via
        :meth:`get_fully_qualified_targets`.
        """
        return set(self.config.lora_targets)

    def get_fully_qualified_targets(self) -> Set[str]:
        """Return all fully-qualified LoRA target paths (with block indices).

        Example entry:
          ``transformer_blocks.5.attn.to_q``
        """
        targets: Set[str] = set()
        for idx in range(self.config.num_transformer_blocks):
            for pattern in self.LORA_PATTERNS:
                if "{}" in pattern:
                    targets.add(pattern.format(idx))
        return targets

    def parse_lora(self, lora_path: str) -> Dict[str, Any]:
        """Parse a Lumina LoRA safetensors file.

        Returns a dict with keys:
          - ``rank`` (int): detected LoRA rank.
          - ``alpha`` (float): LoRA alpha (1.0 if not stored).
          - ``layers`` (dict): per-layer info (shape, block_type, component).
          - ``block_indices`` (list[int]): sorted list of block indices with LoRA.
          - ``stats`` (dict): parameter counts and block coverage.
        """
        try:
            from safetensors import safe_open

            lora_path_obj = Path(lora_path)
            if not lora_path_obj.exists():
                logger.error("[LuminaAdapter] LoRA file not found: %s", lora_path)
                return {}

            with safe_open(str(lora_path_obj), framework="pt", device="cpu") as f:
                lora_weights = {k: f.get_tensor(k) for k in f.keys()}

            return self._analyze_lora_weights(lora_weights)

        except Exception as exc:
            logger.error("[LuminaAdapter] parse_lora failed: %s", exc)
            return {}

    # ── Internal helpers ──────────────────────────────────────────────────

    def _load_single_file(self, path: Path):
        """Load tensors from a single safetensors file."""
        from safetensors import safe_open

        with safe_open(str(path), framework="pt", device="cpu") as f:
            self._state_dict = {k: f.get_tensor(k) for k in f.keys()}

    def _load_directory(self, directory: Path):
        """Load tensors from a directory of safetensors files (diffusers format)."""
        from safetensors import safe_open

        for pattern in self.config.model_file_patterns:
            candidate = directory / "transformer" / pattern
            if candidate.exists():
                with safe_open(str(candidate), framework="pt", device="cpu") as f:
                    self._state_dict = {k: f.get_tensor(k) for k in f.keys()}
                return

        # Fallback: load all safetensors in directory (top-level)
        safetensor_files = sorted(directory.glob("*.safetensors"))
        if not safetensor_files:
            logger.warning("[LuminaAdapter] No safetensors found in %s", directory)
            return

        for sf in safetensor_files:
            with safe_open(str(sf), framework="pt", device="cpu") as f:
                for k in f.keys():
                    self._state_dict[k] = f.get_tensor(k)

    def _parse_layers(self):
        """Parse all loaded tensors into DiTLayerInfo entries."""
        self._layers = []
        for key, tensor in self._state_dict.items():
            info = self._parse_layer_key(key, tensor)
            if info is not None:
                self._layers.append(info)

    def _parse_layer_key(self, key: str, tensor) -> Optional[DiTLayerInfo]:
        """Parse a single state-dict key into a DiTLayerInfo.

        Returns None for non-weight entries (buffers, metadata, etc.)
        that we don't need for LoRA analysis.
        """
        is_lora = "lora_" in key or "lora." in key

        # Determine block type and index
        block_type = "other"
        block_index = -1

        if self._BLOCK_PREFIX in key:
            block_type = "single"  # Lumina has only single-stream blocks
            remainder = key.split(self._BLOCK_PREFIX, 1)[1]
            parts = remainder.split(".")
            try:
                block_index = int(parts[0])
            except (ValueError, IndexError):
                pass

        # Determine layer type (attn / ff / norm / other)
        layer_type = "other"
        component = ""
        for pattern_key, lt in _LAYER_TYPE_MAP.items():
            if pattern_key in key:
                layer_type = lt
                component = pattern_key
                break

        # Detect LoRA rank from the down-projection matrix
        lora_rank = 0
        if is_lora and ("lora_A" in key or "lora_down" in key):
            lora_rank = tensor.shape[0] if len(tensor.shape) >= 1 else 0

        return DiTLayerInfo(
            name=key,
            block_type=block_type,
            block_index=block_index,
            layer_type=layer_type,
            component=component,
            shape=tuple(tensor.shape),
            dtype=str(tensor.dtype),
            is_lora=is_lora,
            lora_rank=lora_rank,
        )

    def _analyze_lora_weights(self, lora_weights: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a LoRA weight dict and return structured metadata."""
        rank = self.detect_lora_rank(lora_weights)

        layers: Dict[str, Dict[str, Any]] = {}
        block_indices: set = set()
        alpha_value = 1.0
        total_lora_params = 0

        for key, tensor in lora_weights.items():
            # Check for stored alpha
            if key.endswith(".alpha") or key.endswith("alpha"):
                try:
                    alpha_value = float(tensor.item())
                except Exception:
                    pass
                continue

            if "lora_" not in key and "lora." not in key:
                continue

            # Derive the base layer name (strip LoRA suffixes)
            base_name = key
            for suffix in (
                ".lora_A.weight",
                ".lora_B.weight",
                ".lora_down.weight",
                ".lora_up.weight",
                ".lora_A.alpha",
                ".lora_B.alpha",
            ):
                if base_name.endswith(suffix):
                    base_name = base_name[: -len(suffix)]
                    break

            if base_name not in layers:
                layers[base_name] = {
                    "rank": rank,
                    "has_A": False,
                    "has_B": False,
                    "block_index": -1,
                    "block_type": "other",
                    "component": "",
                }

            entry = layers[base_name]

            if "lora_A" in key or "lora_down" in key:
                entry["has_A"] = True
            if "lora_B" in key or "lora_up" in key:
                entry["has_B"] = True

            # Extract block index from the base name
            if self._BLOCK_PREFIX in base_name:
                remainder = base_name.split(self._BLOCK_PREFIX, 1)[1]
                parts = remainder.split(".")
                try:
                    idx = int(parts[0])
                    entry["block_index"] = idx
                    entry["block_type"] = "single"
                    block_indices.add(idx)
                except (ValueError, IndexError):
                    pass

            # Component detection
            for pattern_key in _LAYER_TYPE_MAP:
                if pattern_key in base_name:
                    entry["component"] = pattern_key
                    break

            total_lora_params += tensor.numel()

        return {
            "type": "lumina_lora",
            "rank": rank,
            "alpha": alpha_value,
            "layers": layers,
            "block_indices": sorted(block_indices),
            "stats": {
                "total_lora_params": total_lora_params,
                "num_layers": len(layers),
                "num_blocks": len(block_indices),
                "max_block_index": max(block_indices) if block_indices else -1,
                "coverage_ratio": len(block_indices)
                / max(self.config.num_transformer_blocks, 1),
            },
        }

    def get_block_summary(self) -> Dict[str, Any]:
        """Return per-block layer listing with has_lora flags.

        Returns:
            ``{"blocks": {0: {"layers": [...], "has_lora": bool}, ...}}``
        """
        summary: Dict[str, Any] = {"blocks": {}}

        for layer in self._layers:
            if layer.block_index < 0:
                continue

            if layer.block_index not in summary["blocks"]:
                summary["blocks"][layer.block_index] = {
                    "layers": [],
                    "has_lora": False,
                }

            entry = summary["blocks"][layer.block_index]
            entry["layers"].append(layer.name)
            if layer.is_lora:
                entry["has_lora"] = True

        return summary

    def get_training_config(self) -> Dict[str, Any]:
        """Return recommended hyperparameters for Lumina LoRA training.

        These are starting-point suggestions, not hard requirements.
        Actual values should be tuned per-dataset and VRAM budget.
        """
        return {
            "recommended_rank": 16,
            "recommended_alpha": 16,
            "learning_rate": 1e-4,
            "text_encoder_lr": 1e-5,
            "train_text_encoder": False,  # Gemma 2B is large; off by default
            "optimizer": "AdamW",
            "scheduler": "cosine",
            "mixed_precision": "bf16",
            "gradient_checkpointing": True,
            "timestep_sampling": "sigmoid",
            "discrete_flow_shift": self.config.discrete_flow_shift,
            "notes": (
                "Lumina uses Gemma 2B as text encoder (not CLIP/T5). "
                "Training the text encoder requires significant VRAM. "
                "Flow-matching loss with velocity prediction target. "
                "Recommended to start with rank 8-16 and cosine scheduler."
            ),
        }
