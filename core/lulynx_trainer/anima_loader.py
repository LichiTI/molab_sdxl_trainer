"""
Anima native loader — self-contained module for loading Anima models.

This module is the *native* Anima loading route.  It produces a
``LoadedModel`` (the shared cross-family contract from model_loader.py)
plus an ``AnimaLoadReport`` so callers know exactly what loaded and
what fell back.

Design constraints
------------------
- Does NOT modify any shared core files (model_loader.py, trainer.py, etc.).
- Exposes clear structured limitations via :class:`AnimaLoadReport` when
  something cannot be fully loaded (missing Qwen3 class, missing T5
  tokenizer, etc.).
- Handles both single-file checkpoints (.safetensors / .ckpt) and
  diffusers-format directories.
- Optionally loads a Qwen3 secondary text encoder and a dedicated T5
  tokenizer when the paths are provided.

Integration points (for later wiring)
--------------------------------------
1. ``model_loader.py`` — replace the ``_load_anima`` stub with a call
   to :func:`load_anima_model`.
2. ``trainer.py`` — in the Anima branch of ``prepare()``, call this
   loader directly and pass the report to the training loop or log it.
3. ``model_family.py`` — once the loader is wired, set ``is_stub=False``
   for the "anima" entry.

Usage::

    from .anima_loader import AnimaLoader

    loader = AnimaLoader(device="cuda", dtype=torch.bfloat16)
    model, report = loader.load(
        model_path="/path/to/anima.safetensors",
        qwen3_path="/path/to/qwen3",
        t5_tokenizer_path="/path/to/t5_tokenizer",
        attn_mode="sdpa",
    )
    if report.has_limitation(AnimaLimitation.QWEN3_UNAVAILABLE):
        print("Warning: Qwen3 encoder not loaded, using CLIP fallback")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from .anima_contract import (
    AnimaComponents,
    AnimaLimitation,
    AnimaLoadReport,
    SecondaryEncoderKind,
)
from .anima_native_dit import (
    discover_anima_native_param_groups,
    inspect_anima_safetensors,
)
from .model_loader import LoadedModel

logger = logging.getLogger(__name__)


class AnimaLoader:
    """Native Anima model loader.

    Mirrors the interface of :class:`ModelLoader` (device + dtype) but
    returns an ``(LoadedModel, AnimaLoadReport)`` tuple so the caller
    can inspect what actually happened during loading.

    This class is self-contained — it does not delegate to
    ``ModelLoader._load_anima`` or any other shared loader method.
    """

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        *,
        disable_mmap: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.disable_mmap = disable_mmap

    # ── public API ────────────────────────────────────────────────────

    def load(
        self,
        model_path: str,
        qwen3_path: str = "",
        t5_tokenizer_path: str = "",
        attn_mode: str = "",
        vae_path: str = "",
        llm_adapter_path: str = "",
        dit_adapter_path: str = "",
    ) -> Tuple[LoadedModel, AnimaLoadReport]:
        """Load an Anima model and return (LoadedModel, AnimaLoadReport).

        Args:
            model_path: Primary Anima checkpoint or diffusers directory.
            qwen3_path: Optional path to a Qwen3 text encoder
                (directory or single file).
            t5_tokenizer_path: Optional path to a T5 tokenizer
                (directory or HF tokenizer name).
            attn_mode: Attention backend hint (e.g. "sdpa", "xformers",
                "flash").  The loader records the hint but does not
                apply it — that is the responsibility of
                ``prepare_for_training``.
            vae_path: Optional override VAE path (single file or dir).
            llm_adapter_path: Optional path to an LLM adapter
                (LoRA weights) for the Qwen3 encoder.
            dit_adapter_path: Optional path to pre-trained DiT adapter
                weights.  Stored on the model for deferred loading by
                the trainer after LoRA injection.

        Returns:
            Tuple of (LoadedModel, AnimaLoadReport).

        Raises:
            FileNotFoundError: If model_path does not exist.
            ValueError: If model_path is not a file or directory.
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Anima model path does not exist: {path}")

        report = AnimaLoadReport(
            resolved_model_path=str(path),
            applied_attn_mode=attn_mode or "",
        )

        # ── Step 1: Load the primary model (UNet + VAE + sched + CLIP encoders) ──
        components = self._load_primary(path, report)

        # ── Step 2: Optionally load Qwen3 secondary encoder ──────────
        if qwen3_path:
            self._try_load_qwen3(qwen3_path, components, report)

        # ── Step 2b: Optionally load LLM adapter onto Qwen3 ─────────
        if llm_adapter_path and components.qwen3_encoder is not None:
            self._try_load_llm_adapter(llm_adapter_path, components, report)
        elif llm_adapter_path:
            logger.warning(
                "LLM adapter path specified but Qwen3 encoder not loaded; "
                "adapter will be ignored: %s", llm_adapter_path,
            )

        # ── Step 3: Optionally load T5 tokenizer ─────────────────────
        if t5_tokenizer_path:
            self._try_load_t5_tokenizer(t5_tokenizer_path, components, report)

        # ── Step 4: Optionally swap VAE ──────────────────────────────
        if vae_path:
            self._try_swap_vae(vae_path, components, report)

        # ── Step 5: Build the LoadedModel ────────────────────────────
        model = self._to_loaded_model(components, report)

        # Store DiT adapter path for deferred loading by trainer
        if dit_adapter_path:
            model.anima_dit_adapter_path = dit_adapter_path

        logger.info("Anima load complete: %s", report.summary())
        return model, report

    # ── primary model loading ─────────────────────────────────────────

    def _load_primary(
        self,
        path: Path,
        report: AnimaLoadReport,
    ) -> AnimaComponents:
        """Load the primary Anima model (UNet + VAE + scheduler + CLIP).

        Dispatches between single-file and directory formats.  Anima
        checkpoints are SDXL-compatible at the structural level, so we
        reuse the SDXL single-file loader for .safetensors/.ckpt files
        and the diffusers directory pattern for directories.

        This is *not* the old stub — it loads directly into
        :class:`AnimaComponents` and records limitations honestly.
        """
        if path.is_file() and path.suffix in (".safetensors", ".ckpt", ".pt"):
            return self._load_from_checkpoint(path, report)
        elif path.is_dir():
            return self._load_from_directory(path, report)
        else:
            raise ValueError(
                f"Unsupported Anima model path format: {path}. "
                "Expected a .safetensors/.ckpt/.pt file or a diffusers directory."
            )

    def _load_from_checkpoint(
        self,
        path: Path,
        report: AnimaLoadReport,
    ) -> AnimaComponents:
        """Load from a single-file checkpoint (.safetensors / .ckpt).

        Native Anima preview checkpoints are ``net.*`` DiT safetensors,
        not SDXL UNets.  The current foundation therefore performs
        shape-level introspection and returns a structured blocked report
        instead of attempting an incompatible SDXL conversion.
        """
        if path.suffix.lower() != ".safetensors":
            message = (
                "Anima native single-file loading currently supports safetensors "
                f"introspection only, got: {path.suffix or '(no suffix)'}."
            )
            report.limitations.append(AnimaLimitation.SCAFFOLD_MODE)
            report.notes.append(message)
            return AnimaComponents(report=report)

        try:
            introspection = inspect_anima_safetensors(path, disable_mmap=self.disable_mmap)
        except Exception as exc:
            report.limitations.append(AnimaLimitation.SCAFFOLD_MODE)
            report.notes.append(f"Anima native safetensors introspection failed: {exc}")
            return AnimaComponents(report=report)

        report.limitations.append(AnimaLimitation.SCAFFOLD_MODE)
        report.notes.extend(introspection.notes)
        report.notes.extend(
            [
                "Native Anima DiT checkpoint was identified from safetensors shapes.",
                "Warehouse validation now covers exact weight-key mapping and a real-weight block0 forward smoke.",
                "Full 28-block native forward and Qwen/T5/LLM conditioning are not wired into the loader yet.",
                "anima_native_train_ready remains False so trainer guard continues to block training.",
            ]
        )
        report.native_introspection_ready = True
        report.native_key_map_ready = True
        report.native_block0_forward_smoke_ready = True
        report.native_introspection = introspection
        report.native_introspection_dict = introspection.to_dict()
        report.anima_native_groups = discover_anima_native_param_groups(introspection)
        report.anima_native_groups_dict = report.anima_native_groups.to_dict()
        report.anima_native_train_ready = False

        return AnimaComponents(report=report)

    def _load_from_directory(
        self,
        path: Path,
        report: AnimaLoadReport,
    ) -> AnimaComponents:
        """Load from a diffusers-format directory.

        Follows the standard diffusers subfolder convention:
        unet/, text_encoder/, text_encoder_2/, vae/, tokenizer/,
        tokenizer_2/, scheduler/.

        This does NOT fall back to the SDXL loader — it loads each
        component directly via ``from_pretrained``, which means any
        Anima-specific config.json inside the subfolders is respected.
        """
        from diffusers import (
            AutoencoderKL,
            DDPMScheduler,
            UNet2DConditionModel,
        )
        from transformers import (
            CLIPTextModel,
            CLIPTextModelWithProjection,
            CLIPTokenizer,
        )

        logger.info("Loading Anima from diffusers directory: %s", path)

        unet = UNet2DConditionModel.from_pretrained(
            path, subfolder="unet", torch_dtype=self.dtype,
        )
        text_encoder_1 = CLIPTextModel.from_pretrained(
            path, subfolder="text_encoder", torch_dtype=self.dtype,
        )
        text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
            path, subfolder="text_encoder_2", torch_dtype=self.dtype,
        )
        vae = AutoencoderKL.from_pretrained(
            path, subfolder="vae", torch_dtype=self.dtype,
        )
        tokenizer_1 = CLIPTokenizer.from_pretrained(
            path, subfolder="tokenizer",
        )
        tokenizer_2 = CLIPTokenizer.from_pretrained(
            path, subfolder="tokenizer_2",
        )
        scheduler = DDPMScheduler.from_pretrained(
            path, subfolder="scheduler",
        )

        # Directory loading always produces CLIP encoders.
        report.secondary_encoder_kind = SecondaryEncoderKind.CLIP

        return AnimaComponents(
            unet=unet,
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            vae=vae,
            tokenizer_1=tokenizer_1,
            tokenizer_2=tokenizer_2,
            noise_scheduler=scheduler,
            report=report,
        )

    # ── Qwen3 secondary encoder ──────────────────────────────────────

    def _try_load_qwen3(
        self,
        qwen3_path: str,
        components: AnimaComponents,
        report: AnimaLoadReport,
    ) -> None:
        """Attempt to load a Qwen3 text encoder.

        Supports two path formats:
        - A single ``.safetensors`` file: config is inferred from tensor
          shapes, no ``config.json`` required.
        - A HuggingFace model directory: loaded via
          ``AutoModelForCausalLM.from_pretrained``.

        Qwen3 is a causal-LM — its output shape (``last_hidden_state``,
        no ``text_embeds``) is incompatible with what the training loop
        expects from ``text_encoder_2``.  The encoder is stored in
        ``components.qwen3_encoder`` and flagged as
        ``QWEN3_NOT_CLIP_COMPATIBLE`` so callers know it cannot be used
        as a drop-in replacement for the CLIP secondary slot.

        If the model class is not available or the path is invalid, we
        record ``QWEN3_UNAVAILABLE`` and continue with CLIP only.
        """
        qpath = Path(qwen3_path)
        if not qpath.exists():
            report.limitations.append(AnimaLimitation.QWEN3_UNAVAILABLE)
            report.notes.append(
                f"Qwen3 path does not exist: {qwen3_path}. "
                "Using CLIP-only fallback."
            )
            logger.warning("Qwen3 path does not exist: %s", qwen3_path)
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            is_single_file = qpath.is_file() and qpath.suffix.lower() == ".safetensors"

            if is_single_file:
                qwen3_model = self._load_qwen3_from_single_file(qpath)
            else:
                logger.info("Loading Qwen3 encoder from directory %s", qpath)
                qwen3_model = AutoModelForCausalLM.from_pretrained(
                    str(qpath),
                    torch_dtype=self.dtype,
                    trust_remote_code=True,
                )

            qwen3_model.requires_grad_(False)
            qwen3_model.eval()

            components.qwen3_encoder = qwen3_model
            report.qwen3_loaded = True
            report.qwen3_path = str(qpath)

            report.limitations.append(AnimaLimitation.QWEN3_NOT_CLIP_COMPATIBLE)
            report.notes.append(
                "Qwen3 encoder loaded and stored separately.  It cannot "
                "serve as text_encoder_2 in the current training loop "
                "(missing hidden_states[-2] / text_embeds output)."
            )

            # Try loading tokenizer. Single-file weights often ship next to a
            # shared config/tokenizer directory rather than inside the model dir.
            tok_search = self._resolve_qwen3_tokenizer_dir(qpath) if is_single_file else qpath
            try:
                qwen3_tokenizer = AutoTokenizer.from_pretrained(
                    str(tok_search), trust_remote_code=True,
                )
                components.qwen3_tokenizer = qwen3_tokenizer
            except Exception as tok_exc:
                report.notes.append(
                    f"Qwen3 model loaded but tokenizer failed: {tok_exc}"
                )
                logger.info("Qwen3 tokenizer load failed: %s", tok_exc)

        except ImportError:
            report.limitations.append(AnimaLimitation.QWEN3_UNAVAILABLE)
            report.notes.append(
                "transformers AutoModelForCausalLM not available. "
                "Using CLIP-only fallback."
            )
            logger.warning("Qwen3 loading failed: missing transformers dependency")
        except Exception as exc:
            report.limitations.append(AnimaLimitation.QWEN3_UNAVAILABLE)
            report.notes.append(f"Qwen3 loading failed: {exc}")
            logger.warning("Qwen3 loading failed: %s", exc)

    def _load_qwen3_from_single_file(self, safetensors_path: Path):
        """Load Qwen3 from a single .safetensors file by inferring config."""
        from safetensors import safe_open
        from transformers import AutoModelForCausalLM, AutoConfig

        logger.info("Loading Qwen3 from single file %s", safetensors_path)

        with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
            keys = list(f.keys())
            shapes = {k: tuple(f.get_slice(k).get_shape()) for k in keys}

        embed_shape = shapes.get("model.embed_tokens.weight")
        if embed_shape is None:
            raise ValueError(
                f"Cannot infer Qwen3 config: no model.embed_tokens.weight in {safetensors_path}"
            )

        vocab_size, hidden_size = embed_shape

        layer_ids = set()
        for k in keys:
            if k.startswith("model.layers."):
                layer_ids.add(int(k.split(".")[2]))
        num_hidden_layers = len(layer_ids)

        q_proj_shape = shapes.get("model.layers.0.self_attn.q_proj.weight")
        k_proj_shape = shapes.get("model.layers.0.self_attn.k_proj.weight")
        gate_shape = shapes.get("model.layers.0.mlp.gate_proj.weight")

        if q_proj_shape is None or k_proj_shape is None or gate_shape is None:
            raise ValueError(
                f"Cannot infer Qwen3 config: missing q_proj/k_proj/gate_proj in {safetensors_path}"
            )

        intermediate_size = gate_shape[0]

        q_norm_shape = shapes.get("model.layers.0.self_attn.q_norm.weight")
        head_dim = q_norm_shape[0] if q_norm_shape is not None else 128
        num_attention_heads = q_proj_shape[0] // head_dim
        num_key_value_heads = k_proj_shape[0] // head_dim

        config_dict = {
            "model_type": "qwen3",
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "intermediate_size": intermediate_size,
            "head_dim": head_dim,
            "tie_word_embeddings": True,
        }

        logger.info(
            "Inferred Qwen3 config: hidden=%d, layers=%d, heads=%d, kv_heads=%d, "
            "intermediate=%d, head_dim=%d, vocab=%d",
            hidden_size, num_hidden_layers, num_attention_heads,
            num_key_value_heads, intermediate_size, head_dim, vocab_size,
        )

        config = AutoConfig.for_model(**config_dict)
        model = AutoModelForCausalLM.from_config(config, dtype=self.dtype)

        from safetensors.torch import load_file
        state_dict = load_file(str(safetensors_path))
        incompatible = model.load_state_dict(state_dict, strict=False)

        if incompatible.missing_keys:
            logger.warning(
                "Qwen3 single-file load: %d missing keys (first 5: %s)",
                len(incompatible.missing_keys),
                incompatible.missing_keys[:5],
            )
        if incompatible.unexpected_keys:
            logger.warning(
                "Qwen3 single-file load: %d unexpected keys (first 5: %s)",
                len(incompatible.unexpected_keys),
                incompatible.unexpected_keys[:5],
            )

        return model

    def _resolve_qwen3_tokenizer_dir(self, qwen3_file: Path) -> Path:
        """Find a local Qwen3 tokenizer directory for single-file weights."""

        candidates = [
            qwen3_file.parent,
            qwen3_file.parent / "qwen3_06b",
            qwen3_file.parent / "tokenizer",
        ]
        if len(qwen3_file.parents) > 1:
            candidates.append(qwen3_file.parents[1] / "qwen3_06b")
        repo_root = Path(__file__).resolve().parents[3]
        candidates.extend(
            [
                repo_root / "models" / "anima" / "text_encoders" / "qwen3_06b",
                repo_root / "ref" / "anima_lora-1.6.0.hotfix2" / "library" / "anima" / "configs" / "qwen3_06b",
                repo_root / "ref" / "anima_lora-1.5.5.hotfix" / "library" / "anima" / "configs" / "qwen3_06b",
                repo_root / "ref" / "anima_lora-1.3.2" / "library" / "anima" / "configs" / "qwen3_06b",
                repo_root / "ref" / "DiffPipeForge_ZiYun-main" / "app" / "backend" / "core" / "configs" / "qwen3_06b",
            ]
        )
        for candidate in candidates:
            if candidate.is_dir() and (candidate / "tokenizer.json").is_file():
                return candidate
        return qwen3_file.parent

    # ── LLM adapter (LoRA for Qwen3) ──────────────────────────────────

    def _try_load_llm_adapter(
        self,
        adapter_path: str,
        components: AnimaComponents,
        report: AnimaLoadReport,
    ) -> None:
        """Load an LLM adapter (LoRA weights) and apply it to Qwen3.

        The adapter is expected to be a PEFT-format LoRA checkpoint
        or a single .safetensors file with LoRA weights.  After loading,
        the adapter is merged into the Qwen3 model weights so that
        inference uses the adapted model without PEFT overhead.
        """
        apath = Path(adapter_path)
        if not apath.exists():
            report.notes.append(f"LLM adapter path does not exist: {adapter_path}")
            logger.warning("LLM adapter path does not exist: %s", adapter_path)
            return

        try:
            from peft import PeftModel

            logger.info("Loading LLM adapter from %s", apath)
            qwen3 = components.qwen3_encoder
            adapted = PeftModel.from_pretrained(qwen3, str(apath))
            # Merge adapter weights into base model for inference efficiency
            adapted = adapted.merge_and_unload()
            adapted.requires_grad_(False)
            adapted.eval()
            components.qwen3_encoder = adapted
            report.notes.append(f"LLM adapter loaded and merged: {apath.name}")
            logger.info("LLM adapter loaded and merged successfully")

        except ImportError:
            report.notes.append(
                "peft library not available; cannot load LLM adapter. "
                "Install with: pip install peft"
            )
            logger.warning("peft library not available for LLM adapter loading")
        except Exception as exc:
            report.notes.append(f"LLM adapter loading failed: {exc}")
            logger.warning("LLM adapter loading failed: %s", exc)

    # ── T5 tokenizer ─────────────────────────────────────────────────

    def _try_load_t5_tokenizer(
        self,
        t5_tokenizer_path: str,
        components: AnimaComponents,
        report: AnimaLoadReport,
    ) -> None:
        """Attempt to load a dedicated T5 tokenizer.

        The T5 tokenizer is used when Anima's text pipeline includes a
        T5 encoder branch (similar to how Flux/SD3 use T5-XXL).  If the
        tokenizer class is not available or the path is invalid, we
        record the limitation and the model continues with CLIP
        tokenizers only.

        TODO: When a T5 encoder is integrated into the Anima pipeline,
        this tokenizer will be paired with it.  For now it is stored
        as an optional extra on AnimaComponents.
        """
        try:
            from transformers import T5Tokenizer, T5TokenizerFast

            logger.info("Loading T5 tokenizer from %s", t5_tokenizer_path)
            # Try fast tokenizer first, fall back to slow
            try:
                t5_tok = T5TokenizerFast.from_pretrained(t5_tokenizer_path)
            except Exception:
                t5_tok = T5Tokenizer.from_pretrained(t5_tokenizer_path)

            components.t5_tokenizer = t5_tok
            report.t5_tokenizer_loaded = True
            report.t5_tokenizer_path = t5_tokenizer_path

        except ImportError:
            report.limitations.append(AnimaLimitation.T5_TOKENIZER_UNAVAILABLE)
            report.notes.append(
                "transformers T5Tokenizer not available. "
                "Using CLIP tokenizers only."
            )
            logger.warning("T5 tokenizer loading failed: missing transformers dependency")
        except Exception as exc:
            report.limitations.append(AnimaLimitation.T5_TOKENIZER_UNAVAILABLE)
            report.notes.append(f"T5 tokenizer loading failed: {exc}")
            logger.warning("T5 tokenizer loading failed: %s", exc)

    # ── VAE override ─────────────────────────────────────────────────

    def _try_swap_vae(
        self,
        vae_path: str,
        components: AnimaComponents,
        report: AnimaLoadReport,
    ) -> None:
        """Swap in a custom VAE if a path is provided."""
        vref = Path(vae_path)
        if not vref.exists():
            report.notes.append(f"Custom VAE path does not exist: {vae_path}")
            logger.warning("Custom VAE path does not exist: %s", vae_path)
            return

        try:
            from diffusers import AutoencoderKL, AutoencoderKLQwenImage

            logger.info("Loading custom VAE from %s", vref)
            if vref.is_file() and "qwen" in vref.name.lower():
                components.vae = self._load_qwen_image_vae_from_single_file(AutoencoderKLQwenImage, vref)
            elif vref.is_file():
                components.vae = AutoencoderKL.from_single_file(
                    str(vref), torch_dtype=self.dtype,
                )
            else:
                components.vae = AutoencoderKL.from_pretrained(
                    str(vref), torch_dtype=self.dtype,
                )
        except Exception as exc:
            report.notes.append(f"Custom VAE loading failed: {exc}")
            logger.warning("Custom VAE loading failed: %s", exc)

    def _load_qwen_image_vae_from_single_file(self, vae_cls: Any, vae_path: Path) -> Any:
        """Load local Qwen Image VAE safetensors into diffusers' class."""

        from safetensors.torch import load_file

        vae = vae_cls()
        source = load_file(str(vae_path), device="cpu")
        target = vae.state_dict()
        converted: dict[str, torch.Tensor] = {}
        for source_key, tensor in source.items():
            target_key = self._map_qwen_image_vae_key(source_key)
            if target_key is None or target_key not in target:
                continue
            if tuple(tensor.shape) == tuple(target[target_key].shape):
                converted[target_key] = tensor

        missing_encoder = [
            key for key in target
            if (key.startswith("encoder.") or key.startswith("quant_conv.")) and key not in converted
        ]
        if missing_encoder:
            raise RuntimeError(
                "Qwen Image VAE single-file mapping is incomplete for encoder cache generation: "
                + ", ".join(missing_encoder[:5])
            )
        vae.load_state_dict(converted, strict=False)
        vae.to(dtype=self.dtype)
        vae.requires_grad_(False)
        vae.eval()
        return vae

    @staticmethod
    def _map_qwen_image_residual_tail(tail: str, target_prefix: str) -> str | None:
        mapping = {
            "0.": "norm1.",
            "2.": "conv1.",
            "3.": "norm2.",
            "6.": "conv2.",
        }
        for source, target in mapping.items():
            if tail.startswith(source):
                return target_prefix + target + tail[len(source):]
        return None

    @staticmethod
    def _map_qwen_image_residual(key: str, source_prefix: str, target_prefix: str) -> str | None:
        if not key.startswith(source_prefix):
            return None
        return AnimaLoader._map_qwen_image_residual_tail(key[len(source_prefix):], target_prefix)

    @staticmethod
    def _map_qwen_image_vae_key(key: str) -> str | None:
        if key.startswith("conv1."):
            return "quant_conv." + key[len("conv1."):]
        if key.startswith("conv2."):
            return "post_quant_conv." + key[len("conv2."):]

        for side, flat_name in (("encoder", "downsamples"), ("decoder", "upsamples")):
            if key.startswith(f"{side}.conv1."):
                return f"{side}.conv_in." + key[len(f"{side}.conv1."):]
            if key.startswith(f"{side}.head.0."):
                return f"{side}.norm_out." + key[len(f"{side}.head.0."):]
            if key.startswith(f"{side}.head.2."):
                return f"{side}.conv_out." + key[len(f"{side}.head.2."):]
            if key.startswith(f"{side}.middle.1."):
                return f"{side}.mid_block.attentions.0." + key[len(f"{side}.middle.1."):]
            mapped_mid = AnimaLoader._map_qwen_image_residual(
                key,
                f"{side}.middle.0.residual.",
                f"{side}.mid_block.resnets.0.",
            )
            if mapped_mid:
                return mapped_mid
            mapped_mid = AnimaLoader._map_qwen_image_residual(
                key,
                f"{side}.middle.2.residual.",
                f"{side}.mid_block.resnets.1.",
            )
            if mapped_mid:
                return mapped_mid

            prefix = f"{side}.{flat_name}."
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix):]
            index_text, _, tail = rest.partition(".")
            if not index_text.isdigit() or not tail:
                return None
            index = int(index_text)
            if side == "encoder":
                block_prefix = f"encoder.down_blocks.{index}."
                if tail.startswith("residual."):
                    return AnimaLoader._map_qwen_image_residual_tail(tail[len("residual."):], block_prefix)
                if tail.startswith("shortcut."):
                    return block_prefix + "conv_shortcut." + tail[len("shortcut."):]
                if tail.startswith("resample.") or tail.startswith("time_conv."):
                    return block_prefix + tail
            else:
                group = index // 4 if index < 12 else 3
                slot = index % 4 if index < 12 else index - 12
                block_prefix = f"decoder.up_blocks.{group}."
                if tail.startswith("residual."):
                    return AnimaLoader._map_qwen_image_residual_tail(
                        tail[len("residual."):],
                        f"{block_prefix}resnets.{slot}.",
                    )
                if tail.startswith("shortcut."):
                    return f"{block_prefix}resnets.{slot}.conv_shortcut." + tail[len("shortcut."):]
                if tail.startswith("resample.") or tail.startswith("time_conv."):
                    return f"{block_prefix}upsamplers.0." + tail
        return None

    # ── conversion to LoadedModel ────────────────────────────────────

    def _to_loaded_model(
        self,
        components: AnimaComponents,
        report: AnimaLoadReport,
    ) -> LoadedModel:
        """Convert AnimaComponents into the shared LoadedModel contract.

        The shared training loop currently expects the SDXL-style
        secondary encoder contract: ``hidden_states[-2]`` plus
        ``text_embeds``.  That means only the CLIP-2 path can occupy
        ``text_encoder_2`` today.

        If a Qwen3 encoder was loaded, it is attached as Anima-specific
        metadata on the returned ``LoadedModel`` so future adapters can
        consume it without breaking the current trainer contract.
        """
        # Introspection-only single-file checkpoints intentionally have no
        # runtime components yet.  Keep the LoadedModel shape stable but do not
        # synthesize an SDXL scheduler or claim training readiness.
        if (
            components.unet is None
            and components.text_encoder_1 is None
            and components.text_encoder_2 is None
            and components.vae is None
            and components.noise_scheduler is None
        ):
            model = LoadedModel(
                unet=None,
                text_encoder_1=None,
                text_encoder_2=None,
                vae=None,
                tokenizer_1=None,
                tokenizer_2=None,
                noise_scheduler=None,
                model_arch="anima",
            )
            model.anima_load_report = report
            model.anima_secondary_encoder_kind = report.secondary_encoder_kind.value
            model.anima_native_train_ready = False
            model.anima_native_block0_forward_smoke_ready = report.native_block0_forward_smoke_ready
            if hasattr(report, "native_introspection"):
                model.anima_native_introspection = report.native_introspection
            if hasattr(report, "anima_native_groups"):
                model.anima_native_groups = report.anima_native_groups
            return model

        # Ensure we have a scheduler — fall back to DDPMScheduler if missing
        scheduler = components.noise_scheduler
        if scheduler is None:
            from diffusers import DDPMScheduler

            logger.warning("No scheduler found, falling back to DDPMScheduler")
            scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_start=0.00085,
                beta_end=0.012,
                beta_schedule="scaled_linear",
                clip_sample=False,
                timestep_spacing="leading",
            )

        model = LoadedModel(
            unet=components.unet,
            text_encoder_1=components.text_encoder_1,
            text_encoder_2=components.text_encoder_2,
            vae=components.vae,
            tokenizer_1=components.tokenizer_1,
            tokenizer_2=components.tokenizer_2,
            noise_scheduler=scheduler,
            model_arch="anima",
        )

        model.anima_load_report = report
        model.anima_secondary_encoder_kind = report.secondary_encoder_kind.value
        model.anima_native_train_ready = False
        model.anima_native_block0_forward_smoke_ready = report.native_block0_forward_smoke_ready
        if hasattr(report, "native_introspection"):
            model.anima_native_introspection = report.native_introspection
        if hasattr(report, "anima_native_groups"):
            model.anima_native_groups = report.anima_native_groups
        if components.qwen3_encoder is not None:
            model.anima_qwen3_encoder = components.qwen3_encoder
        if components.qwen3_tokenizer is not None:
            model.anima_qwen3_tokenizer = components.qwen3_tokenizer
        if components.t5_tokenizer is not None:
            model.anima_t5_tokenizer = components.t5_tokenizer

        return model


# ── convenience function ─────────────────────────────────────────────

def load_anima_model(
    model_path: str,
    qwen3_path: str = "",
    t5_tokenizer_path: str = "",
    attn_mode: str = "",
    vae_path: str = "",
    llm_adapter_path: str = "",
    dit_adapter_path: str = "",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    *,
    disable_mmap: bool = False,
) -> Tuple[LoadedModel, AnimaLoadReport]:
    """Convenience wrapper for one-shot Anima loading.

    This is the function that ``model_loader.py`` should call from its
    ``_load_anima`` method when wiring the native route::

        from .anima_loader import load_anima_model

        def _load_anima(self, path: Path) -> LoadedModel:
            model, report = load_anima_model(
                model_path=str(path),
                device=self.device,
                dtype=self.dtype,
            )
            # report is available for logging / diagnostics
            return model
    """
    loader = AnimaLoader(device=device, dtype=dtype, disable_mmap=disable_mmap)
    return loader.load(
        model_path=model_path,
        qwen3_path=qwen3_path,
        t5_tokenizer_path=t5_tokenizer_path,
        attn_mode=attn_mode,
        vae_path=vae_path,
        llm_adapter_path=llm_adapter_path,
        dit_adapter_path=dit_adapter_path,
    )

