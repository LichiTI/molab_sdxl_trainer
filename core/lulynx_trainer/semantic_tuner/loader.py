"""
Semantic Base-Tuner Manager (V3.1)
Handles the lifecycle of the Dual-Stream Architecture (LLM + CLIP).
"""

import torch
import torch.nn as nn
import logging
import os
import ast
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import AutoConfig, AutoModel, AutoTokenizer, CLIPTextModel, CLIPTokenizer

from ...semantic_brain.projector import LulynxUniversalProjector
from ...constants import DEFAULT_DEVICE

logger = logging.getLogger(__name__)

_DEFAULT_TRANSFORMERS_WEIGHT_NAMES = {
    "model.safetensors",
    "pytorch_model.bin",
    "tf_model.h5",
    "model.ckpt.index",
    "flax_model.msgpack",
}


def _local_transformers_config(path: str) -> Any | None:
    """Load local config and point Transformers at a single custom safetensors file."""
    if not path:
        return None
    root = Path(str(path))
    if not root.is_dir():
        return None
    try:
        cfg = AutoConfig.from_pretrained(str(root), trust_remote_code=True)
    except Exception:
        return None
    has_default_weights = any((root / name).is_file() for name in _DEFAULT_TRANSFORMERS_WEIGHT_NAMES)
    has_index = any(root.glob("*.safetensors.index.json"))
    if has_default_weights or has_index:
        return cfg
    safetensors_files = sorted(
        item for item in root.glob("*.safetensors")
        if item.is_file() and not item.name.endswith(".index.safetensors")
    )
    if len(safetensors_files) == 1:
        setattr(cfg, "transformers_weights", safetensors_files[0].name)
        logger.info("Using custom local Transformers weight file: %s", safetensors_files[0].name)
    return cfg


def _read_hidden_dim(config: Any) -> int:
    """Best-effort hidden-size discovery for plain LLM and multimodal configs."""
    candidates = [
        getattr(config, "hidden_size", None),
        getattr(config, "d_model", None),
        getattr(config, "n_embd", None),
    ]
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        candidates.extend([
            getattr(text_config, "hidden_size", None),
            getattr(text_config, "d_model", None),
            getattr(text_config, "n_embd", None),
        ])
    for value in candidates:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return 1024


def _read_clip_embed_dim(config: Any) -> int:
    """Best-effort teacher embedding dim discovery for CLIP-like configs."""
    candidates = [
        getattr(config, "projection_dim", None),
        getattr(config, "hidden_size", None),
        getattr(config, "embed_dim", None),
    ]
    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        candidates.extend([
            getattr(text_config, "embed_dim", None),
            getattr(text_config, "hidden_size", None),
            getattr(text_config, "projection_dim", None),
        ])
    for value in candidates:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return 768


def _missing_local_remote_code_files(path: str) -> list[str]:
    """Return missing relative-import Python files for a local trust_remote_code model."""
    root = Path(str(path))
    if not root.is_dir():
        return []

    queue = [item for item in (root / "modeling_clip.py", root / "configuration_clip.py") if item.is_file()]
    visited: set[Path] = set()
    missing: set[str] = set()

    while queue:
        current = queue.pop()
        if current in visited:
            continue
        visited.add(current)
        try:
            tree = ast.parse(current.read_text(encoding="utf-8"))
        except Exception:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level != 1 or not node.module:
                continue
            candidate = root.joinpath(*node.module.split(".")).with_suffix(".py")
            if candidate.is_file():
                queue.append(candidate)
            else:
                missing.add(str(candidate.relative_to(root)).replace("\\", "/"))

    return sorted(missing)


def _ensure_clip_modeling_compat() -> None:
    """Patch removed Transformers CLIP helpers expected by older Jina remote code."""
    try:
        import transformers.models.clip.modeling_clip as clip_modeling
    except Exception:
        return
    if hasattr(clip_modeling, "clip_loss"):
        return

    def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
        labels = torch.arange(logits.shape[0], device=logits.device)
        return torch.nn.functional.cross_entropy(logits, labels)

    def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
        caption_loss = contrastive_loss(similarity)
        image_loss = contrastive_loss(similarity.t())
        return (caption_loss + image_loss) / 2.0

    clip_modeling.clip_loss = clip_loss
    logger.info("Installed Transformers CLIP clip_loss compatibility shim.")


def _load_jina_text_state_dict(model: nn.Module, weight_path: Path) -> tuple[int, list[str], list[str]]:
    """Load local Jina text-tower weights into a jina-embeddings-v3 text model."""
    from safetensors.torch import load_file

    raw_state = load_file(str(weight_path), device="cpu")
    target_keys = set(model.state_dict().keys())
    mapped: dict[str, torch.Tensor] = {}

    for key, tensor in raw_state.items():
        if key == "spiece_model" or not key.startswith("model."):
            continue
        base_key = f"roberta.{key[len('model.'):]}"
        candidates = [base_key]
        if base_key.endswith(".weight"):
            candidates.append(f"{base_key[:-len('.weight')]}.parametrizations.weight.original")
        for candidate in candidates:
            if candidate in target_keys:
                mapped[candidate] = tensor
                break

    incompatible = model.load_state_dict(mapped, strict=False)
    return len(mapped), list(incompatible.missing_keys), list(incompatible.unexpected_keys)

class SemanticTunerManager:
    def __init__(self, **kwargs):
        self.config = kwargs
        self.device = kwargs.get("device", DEFAULT_DEVICE)
        self.dtype = kwargs.get("dtype", torch.float16)
        
        # Dual-Stream Components
        self.llm_model = None
        self.llm_tokenizer = None
        self.projector = None
        
        # Legacy/Ghost Branch Components
        self.clip_model = None
        self.clip_tokenizer = None
        self.clip_model_kind = ""
        self.clip_load_error = ""
        self.clip_tokenizer_only_dim = 768
        
        self.mode = self.config.get("architecture_mode", "hybrid") # hybrid or pure

    def load_dual_stream_context(self):
        """
        Loads the full context for Semantic Base-Tuner V3.1.
        Includes:
        - Main Branch: LLM + Universal Projector
        - Ghost Branch: CLIP (if mode == 'hybrid')
        """
        logger.info(f"Initializing Semantic Base-Tuner in [{self.mode.upper()}] mode...")
        
        # 1. Load Main Branch (LLM)
        self._load_llm()
        self._load_projector()
        
        # 2. Load Ghost Branch (CLIP) if needed
        if self.mode == "hybrid":
            self._load_clip_ghost()
        else:
            logger.info("Pure Neuro-Link Mode: Skipping CLIP loading.")

    def _load_llm(self):
        llm_path = self.config.get("llm_path")
        if not llm_path:
            raise ValueError("Semantic Base-Tuner requires 'llm_path'")
            
        logger.info(f"Loading LLM Main Brain from: {llm_path}")
        try:
            llm_config = _local_transformers_config(llm_path)
            model_cls = AutoModel
            model_config = llm_config
            key_mapping = None
            if getattr(llm_config, "model_type", "") == "gemma3" and getattr(llm_config, "text_config", None) is not None:
                from transformers import Gemma3TextModel
                model_cls = Gemma3TextModel
                model_config = llm_config.text_config
                custom_weights = getattr(llm_config, "transformers_weights", None)
                if custom_weights:
                    setattr(model_config, "transformers_weights", custom_weights)
                key_mapping = {r"^model\.": ""}
            self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_path, trust_remote_code=True)
            # Load LLM in 4-bit/8-bit if configured, otherwise standard
            # For now standard loading, optimized later
            load_kwargs = {
                "config": model_config,
                "trust_remote_code": True,
                "torch_dtype": self.dtype,
            }
            if key_mapping:
                load_kwargs["key_mapping"] = key_mapping
            self.llm_model = model_cls.from_pretrained(
                llm_path,
                **load_kwargs,
            ).to(self.device)
            
            # Freeze LLM immediately - We NEVER train the LLM backbone in Phase 2
            self.llm_model.eval()
            self.llm_model.requires_grad_(False)
            logger.info("LLM loaded and FROZEN.")
            
        except Exception as e:
            logger.error(f"Failed to load LLM: {str(e)}")
            raise e

    def _load_projector(self):
        proj_path = self.config.get("projector_path")
        
        # Determine dimensions. Gemma3-style multimodal configs keep the text
        # hidden size under config.text_config, while Qwen-style LLMs expose it
        # at the top level.
        llm_dim = _read_hidden_dim(self.llm_model.config) if self.llm_model else 1024
        
        # U-Net typically expects 768 (TE1) or 1280 (TE2). 
        # In V3.1, Sidecar Channel B expects... actually U-Net CrossAttn dim is fixed (2048 for SDXL context).
        # But wait, Sidecar means we add a NEW CrossAttn. 
        # So we can define our OWN dimension for Channel B!
        # Let's align it with LLM dim to avoid compression loss, OR align with U-Net context dim (2048)
        target_dim = 2048 
        
        logger.info(f"Initializing Universal Projector: {llm_dim} -> {target_dim}")
        
        self.projector = LulynxUniversalProjector(
            in_dim=llm_dim,
            out_dim=target_dim,
            hidden_mult=4,
            bake_in_norm=True
        ).to(self.device).to(self.dtype) # Cast to correct dtype
        
        if proj_path and os.path.exists(proj_path):
            logger.info(f"Loading Projector weights from {proj_path}")
            state = torch.load(proj_path, map_location=self.device, weights_only=True)
            self.projector.load_state_dict(state)
        else:
            logger.warning("No Projector weights found. Using RANDOM initialization (Warmup needed).")

    def _load_clip_ghost(self):
        """
        Loads the Legacy CLIP model for the Ghost Branch.
        """
        # Usually we reuse the Trainer's existing TE loading mechanism, 
        # but here we might need a dedicated strictly frozen CLIP if dependencies are complex.
        # Ideally, we should ask TE Manager to give us the CLIP. 
        # For this implementation, we assume we might need to load it if TE Manager delegated everything to us.
        
        # Check if config provides a specific CLIP path, else default SDXL
        clip_path = self.config.get("teacher_path", "openai/clip-vit-large-patch14") # Fallback
        
        logger.info(f"Loading Ghost CLIP from: {clip_path}")
        try:
            self.clip_tokenizer = CLIPTokenizer.from_pretrained(clip_path)
            self.clip_model = CLIPTextModel.from_pretrained(
                clip_path, 
                torch_dtype=self.dtype
            ).to(self.device)
            self.clip_model_kind = "clip_text"
            self.clip_model.eval()
            self.clip_model.requires_grad_(False)
            logger.info("Ghost CLIP loaded and FROZEN.")
            return
        except Exception as e:
            logger.warning(f"Standard CLIP load failed: {e}. Trying AutoModel fallback.")
            self.clip_load_error = str(e)

        try:
            if self._load_jina_text_teacher(clip_path):
                return
        except Exception as e:
            self.clip_load_error = str(e)
            logger.warning(f"Jina text teacher load failed: {e}. Trying generic AutoModel fallback.")

        try:
            clip_config = _local_transformers_config(clip_path)
            _ensure_clip_modeling_compat()
            self.clip_tokenizer = AutoTokenizer.from_pretrained(clip_path, trust_remote_code=True)
            self.clip_model = AutoModel.from_pretrained(
                clip_path,
                config=clip_config,
                trust_remote_code=True,
                torch_dtype="auto",
            ).to(self.device)
            try:
                self.clip_model.to(dtype=self.dtype)
            except (TypeError, RuntimeError):
                logger.debug("Ghost CLIP fallback did not accept dtype cast; continuing with loaded dtype.", exc_info=True)
            self.clip_model_kind = "auto_text"
            self.clip_model.eval()
            self.clip_model.requires_grad_(False)
            logger.info("Ghost CLIP AutoModel fallback loaded and FROZEN.")
        except Exception as e:
            missing_remote_code = _missing_local_remote_code_files(clip_path)
            missing_hint = ""
            if missing_remote_code:
                missing_hint = f" Missing local remote-code files: {', '.join(missing_remote_code)}."
            self.clip_load_error = f"{e}{missing_hint}"
            logger.warning(f"Failed to load Ghost CLIP fallback: {self.clip_load_error}.")
            if self.config.get("allow_tokenizer_only_clip", False):
                try:
                    from transformers import AutoConfig
                    cfg = AutoConfig.from_pretrained(clip_path, trust_remote_code=True)
                    self.clip_tokenizer = self.clip_tokenizer or AutoTokenizer.from_pretrained(clip_path, trust_remote_code=True)
                    self.clip_tokenizer_only_dim = _read_clip_embed_dim(cfg)
                    self.clip_model_kind = "tokenizer_only"
                    logger.warning(
                        "Using tokenizer-only Ghost CLIP fallback (%s dims). This is for smoke/contract validation, not quality distillation.",
                        self.clip_tokenizer_only_dim,
                    )
                    return
                except Exception as fallback_exc:
                    logger.warning(f"Tokenizer-only Ghost CLIP fallback failed: {fallback_exc}.")
            self.clip_model = None
            self.clip_tokenizer = None
            self.clip_model_kind = ""

    def _load_jina_text_teacher(self, clip_path: str) -> bool:
        """Load local Jina CLIP as a real text-only teacher for LAB distillation."""
        clip_config = _local_transformers_config(clip_path)
        if getattr(clip_config, "model_type", "") != "jina_clip":
            return False

        text_config = getattr(clip_config, "text_config", None)
        model_name = str(getattr(text_config, "hf_model_name_or_path", "") or "").strip()
        if not model_name:
            return False

        weight_name = str(getattr(clip_config, "transformers_weights", "") or "").strip()
        weight_path = Path(str(clip_path)) / weight_name if weight_name else None
        if not weight_path or not weight_path.is_file():
            return False

        logger.info("Loading Jina text-only teacher from %s with local weights %s", model_name, weight_path.name)
        model_kwargs = dict(getattr(text_config, "hf_model_config_kwargs", {}) or {})
        model_kwargs["use_flash_attn"] = False
        model_cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_cfg.update(model_kwargs)

        self.clip_tokenizer = AutoTokenizer.from_pretrained(clip_path, trust_remote_code=True)
        self.clip_model = AutoModel.from_config(
            model_cfg,
            trust_remote_code=True,
            add_pooling_layer=False,
        )
        matched, missing, unexpected = _load_jina_text_state_dict(self.clip_model, weight_path)
        lora_missing = sum(1 for key in missing if "lora_" in key)
        base_missing = [key for key in missing if "lora_" not in key]
        if base_missing:
            logger.warning(
                "Jina text teacher loaded with %s mapped tensors, but %s non-LoRA keys are still missing. First missing keys: %s",
                matched,
                len(base_missing),
                base_missing[:8],
            )
        else:
            logger.info(
                "Jina text teacher loaded with %s mapped tensors. Missing LoRA adapter tensors: %s; unexpected: %s",
                matched,
                lora_missing,
                len(unexpected),
            )
        self.clip_model.to(self.device, dtype=self.dtype)
        self.clip_model.eval()
        self.clip_model.requires_grad_(False)
        self.clip_model_kind = "jina_text"
        logger.info("Ghost Jina text teacher loaded and FROZEN.")
        return True

    def get_context(self) -> Dict[str, Any]:
        """Returns the context dictionary for the Trainer/Sidecar"""
        return {
            "llm": self.llm_model,
            "llm_tokenizer": self.llm_tokenizer,
            "projector": self.projector,
            "clip": self.clip_model,
            "clip_tokenizer": self.clip_tokenizer,
            "clip_model_kind": self.clip_model_kind,
            "clip_load_error": self.clip_load_error,
            "clip_tokenizer_only_dim": self.clip_tokenizer_only_dim,
            "mode": self.mode
        }

    def encode_main_branch(self, prompts: list[str]) -> torch.Tensor:
        """
        Encodes prompts via Main Branch (LLM + Projector)
        Returns: [Batch, Seq, Dim] (e.g., [B, N, 2048])
        """
        if not self.llm_model or not self.projector:
            raise RuntimeError("Main Branch not initialized")
            
        inputs = self.llm_tokenizer(
            prompts, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True, 
            max_length=self.config.get("max_token_length", 512) # LLM supports longer context
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.llm_model(**inputs, output_hidden_states=True)
            # Use last hidden state or specific layer? Last is usually fine for general understanding.
            hidden_states = outputs.last_hidden_state 
            
        # Projector is trainable, so we perform this with grad if in training loop
        # But here we are just encoding. 
        projected = self.projector(hidden_states)
        return projected

    def encode_prompt(self, prompts: list[str]) -> torch.Tensor:
        """Backward-compatible alias for older TE manager call sites."""
        return self.encode_main_branch(prompts)

    def encode_ghost_branch(self, prompts: list[str]) -> torch.Tensor:
        """
        Encodes prompts via Ghost Branch (CLIP)
        """
        if self.clip_model_kind == "tokenizer_only":
            if not self.clip_tokenizer:
                return None
            inputs = self.clip_tokenizer(
                prompts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=77,
            ).to(self.device)
            input_ids = inputs.input_ids.to(torch.float32)
            dims = torch.arange(self.clip_tokenizer_only_dim, device=self.device, dtype=torch.float32)
            freqs = (dims + 1.0) / float(max(self.clip_tokenizer_only_dim, 1))
            embeds = torch.sin(input_ids.unsqueeze(-1) * freqs.view(1, 1, -1))
            return embeds.to(dtype=self.dtype)

        if not self.clip_model:
            return None
            
        inputs = self.clip_tokenizer(
            prompts, 
            return_tensors="pt", 
            padding="max_length",
            truncation=True, 
            max_length=77 # Standard CLIP limit
        ).to(self.device)
        
        with torch.no_grad():
            if self.clip_model_kind == "clip_text":
                outputs = self.clip_model(**inputs)
                return outputs.last_hidden_state

            if hasattr(self.clip_model, "get_text_features"):
                try:
                    features = self.clip_model.get_text_features(**inputs)
                except TypeError:
                    features = self.clip_model.get_text_features(input_ids=inputs.input_ids)
                if features.dim() == 2:
                    features = features.unsqueeze(1)
                return features

            if hasattr(self.clip_model, "encode_text"):
                features = self.clip_model.encode_text(
                    prompts,
                    convert_to_numpy=False,
                    convert_to_tensor=True,
                    device=self.device,
                    normalize_embeddings=False,
                )
                if features.dim() == 2:
                    features = features.unsqueeze(1)
                return features.to(self.device)

            outputs = self.clip_model(**inputs, output_hidden_states=True)
            if hasattr(outputs, "last_hidden_state"):
                return outputs.last_hidden_state
            if hasattr(outputs, "text_embeds"):
                return outputs.text_embeds.unsqueeze(1)
            if isinstance(outputs, (tuple, list)) and outputs:
                tensor = outputs[0]
                if isinstance(tensor, torch.Tensor):
                    return tensor.unsqueeze(1) if tensor.dim() == 2 else tensor
            raise RuntimeError("Ghost CLIP fallback did not return text embeddings")

