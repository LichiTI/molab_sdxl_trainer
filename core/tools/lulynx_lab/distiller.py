import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from diffusers import UNet2DConditionModel
from safetensors.torch import load_file, save_file
import logging
import re
from typing import Dict, Optional, List, Tuple
from pathlib import Path

# Core Lulynx Imports
try:
    from core.accelerator import accelerator
    from core.lulynx_trainer.semantic_tuner.unet_sidecar import inject_neural_sidecar, NeuroSidecarNetwork
    from core.lulynx_trainer.semantic_tuner.loader import SemanticTunerManager
except ImportError:
    from backend.core.accelerator import accelerator
    from backend.core.lulynx_trainer.semantic_tuner.unet_sidecar import inject_neural_sidecar, NeuroSidecarNetwork
    from backend.core.lulynx_trainer.semantic_tuner.loader import SemanticTunerManager

logger = logging.getLogger(__name__)
_ALIGN_DIM_WARNED: set[tuple[str, int, int]] = set()

try:
    from diffusers.loaders.lora_conversion_utils import _convert_unet_lora_key, _maybe_map_sgm_blocks_to_diffusers
except Exception:  # pragma: no cover - compatibility fallback for older diffusers
    _convert_unet_lora_key = None
    _maybe_map_sgm_blocks_to_diffusers = None


def _load_unet_topology(unet_path: str, dtype: torch.dtype) -> UNet2DConditionModel:
    """Load SDXL UNet topology from either a diffusers directory or checkpoint.

    The original LAB runner expected a diffusers folder with an ``unet/``
    subfolder. Launcher users often keep SDXL as a single ``.safetensors`` or
    ``.ckpt`` checkpoint, so we support both shapes here.
    """
    path = Path(str(unet_path)).expanduser()
    suffix = path.suffix.lower()
    if path.is_file() and suffix in {".safetensors", ".ckpt"}:
        if not hasattr(UNet2DConditionModel, "from_single_file"):
            raise RuntimeError(
                "This diffusers version cannot load single-file SDXL checkpoints. "
                "Use a diffusers directory with an unet/ subfolder instead."
            )
        return UNet2DConditionModel.from_single_file(str(path), torch_dtype=dtype)
    return UNet2DConditionModel.from_pretrained(
        str(path), subfolder="unet", torch_dtype=dtype
    )


def _align_last_dim(tensor: torch.Tensor, target_dim: int) -> torch.Tensor:
    """Pad or truncate embedding channels for guarded smoke compatibility."""
    current_dim = int(tensor.shape[-1])
    if current_dim == target_dim:
        return tensor
    if current_dim > target_dim:
        warn_key = ("truncate", current_dim, target_dim)
        if warn_key not in _ALIGN_DIM_WARNED:
            _ALIGN_DIM_WARNED.add(warn_key)
            logger.warning(
                "[Distiller] Truncating teacher embedding dim %s -> %s",
                current_dim,
                target_dim,
            )
        return tensor[..., :target_dim]
    warn_key = ("pad", current_dim, target_dim)
    if warn_key not in _ALIGN_DIM_WARNED:
        _ALIGN_DIM_WARNED.add(warn_key)
        logger.warning(
            "[Distiller] Padding teacher embedding dim %s -> %s",
            current_dim,
            target_dim,
        )
    pad = torch.zeros(
        *tensor.shape[:-1],
        target_dim - current_dim,
        device=tensor.device,
        dtype=tensor.dtype,
    )
    return torch.cat([tensor, pad], dim=-1)


def _normalize_lora_base_to_sidecar(base: str, unet_config) -> Optional[Tuple[str, str]]:
    """Map Kohya/diffusers LoRA base keys to sidecar ModuleDict keys.

    Teacher LoRAs may use either diffusers names like
    ``lora_unet_down_blocks_..._attn2_to_k`` or Kohya/SGM names like
    ``lora_unet_input_blocks_..._attn2_to_k``.  The sidecar stores layers as
    sanitized diffusers processor names, for example
    ``down_blocks_1_attentions_0_transformer_blocks_0_attn2_processor``.
    """
    if not base.startswith("lora_unet_"):
        return None
    lane = ""
    if base.endswith("_to_k"):
        lane = "k"
    elif base.endswith("_to_v"):
        lane = "v"
    else:
        return None
    if "_attn2_" not in base:
        return None

    mapped_base = base
    if _maybe_map_sgm_blocks_to_diffusers is not None and any(
        marker in base for marker in ("input_blocks", "middle_block", "output_blocks")
    ):
        pseudo_key = f"{base}.lora_down.weight"
        try:
            mapped = _maybe_map_sgm_blocks_to_diffusers({pseudo_key: None}, unet_config)
            mapped_key = next(iter(mapped.keys()))
            mapped_base = mapped_key.replace(".lora_down.weight", "")
        except Exception:
            logger.debug("Failed to map SGM LoRA key: %s", base, exc_info=True)

    if _convert_unet_lora_key is not None:
        try:
            diffusers_name = _convert_unet_lora_key(mapped_base)
        except Exception:
            logger.debug("Failed to convert LoRA key: %s", mapped_base, exc_info=True)
            diffusers_name = mapped_base.replace("lora_unet_", "").replace("_", ".")
    else:
        diffusers_name = mapped_base.replace("lora_unet_", "").replace("_", ".")

    for suffix in (".to.k", ".to_k"):
        if lane == "k" and diffusers_name.endswith(suffix):
            return diffusers_name[: -len(suffix)].replace(".", "_"), lane
    for suffix in (".to.v", ".to_v"):
        if lane == "v" and diffusers_name.endswith(suffix):
            return diffusers_name[: -len(suffix)].replace(".", "_"), lane
    return None

class LoRADistiller:
    """
    LoRA Distiller Engine (V3.2 - Matrix Regression Mode)
    
    Distills a traditional SDXL LoRA into a Semantic Sidecar Adapter by directly
    regressing the projection matrices, bypassing full U-Net forward passes.
    
    Theory:
    LoRA Effect on Attention:  K' = K + dK = (W_orig + dW) * C_clip
    Sidecar Effect:            K' = W_orig * C_clip + W_sidecar * C_llm
    
    We want the modification to match:
    dW * C_clip ≈ W_sidecar * C_llm
    
    Target: dW (from LoRA) * CLIP_Embeds
    Pred:   W_sidecar * LLM_Embeds
    """
    
    def __init__(
        self,
        unet_path: str,
        lora_path: str,
        llm_path: str = "Qwen/Qwen2.5-0.5B",
        projector_path: str = None,
        teacher_path: str = None,
        allow_tokenizer_only_clip: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
        learning_rate: float = 1e-5,
    ):
        self.device = device
        self.dtype = dtype
        self.learning_rate = float(learning_rate)
        
        logger.info("[Distiller] Initializing...")
        
        # 1. Load U-Net Structure (Topology Only)
        # We need this to initialize the Sidecar with correct dimensions
        logger.info(f"[Distiller] Loading U-Net Topology from {unet_path}...")
        self.unet = _load_unet_topology(unet_path, self.dtype).to(self.device)
        self.unet.requires_grad_(False) # We don't use U-Net weights, just topology
        
        # 2. Inject Sidecar (Student Structure)
        logger.info("[Distiller] Injecting Neuro Sidecar (Student)...")
        # inject_neural_sidecar returns (unet, sidecar_net)
        # It sets up NeuroSidecarNetwork with correct dims based on U-Net
        _, self.sidecar_net = inject_neural_sidecar(self.unet, llm_dim=2048)
        self.sidecar_net.to(self.device, dtype=self.dtype)
        self.sidecar_net.train()
        
        # 3. Load Teacher LoRA Weights (Manual Parsing)
        logger.info(f"[Distiller] Parsing Teacher LoRA from {lora_path}...")
        self.teacher_deltas = self._parse_lora_weights(lora_path)
        logger.info(f"[Distiller] Found {len(self.teacher_deltas)} compatible LoRA layers targets.")
        
        # 4. Setup Semantic Context (CLIP + LLM)
        logger.info("[Distiller] Loading Semantic Context...")
        self.semantic_manager = SemanticTunerManager(
            llm_path=llm_path,
            projector_path=projector_path,
            teacher_path=teacher_path or None,
            allow_tokenizer_only_clip=allow_tokenizer_only_clip,
            device=device,
            dtype=self.dtype,
            load_clip=True # We need CLIP for Teacher Input
        )
        self.semantic_manager.load_dual_stream_context()
        context = self.semantic_manager.get_context()
        self.llm = context.get("llm")
        self.projector = context.get("projector")
        self.clip_text_model = context.get("clip")
        self.clip_tokenizer = context.get("clip_tokenizer")
        self.clip_model_kind = context.get("clip_model_kind") or ""
        self.clip_load_error = context.get("clip_load_error") or ""
        if self.clip_model_kind != "tokenizer_only" and (self.clip_text_model is None or self.clip_tokenizer is None):
            detail = f" Last CLIP load error: {self.clip_load_error}" if self.clip_load_error else ""
            raise RuntimeError(f"LAB Distiller requires a real CLIP teacher_path for real distillation.{detail}")
        if self.clip_model_kind == "tokenizer_only":
            logger.warning("[Distiller] Tokenizer-only teacher fallback is active. Use this for smoke/contract validation only.")
        logger.info("[Distiller] Teacher encoder kind: %s", self.clip_model_kind or "unknown")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(self.sidecar_net.parameters(), lr=self.learning_rate)
        
    def _parse_lora_weights(self, lora_path: str) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Parses safetensors and constructs effective dW matrices (W_up @ W_down * scale).
        Returns mapping: Sidecar_Layer_Name -> {'k': dW_k, 'v': dW_v}
        """
        state_dict = load_file(lora_path)
        
        # Group by layer
        layer_weights = {}
        
        for key, tensor in state_dict.items():
            # Filter for U-Net LoRA
            if "lora_unet" not in key and "lora_up" not in key and "lora_down" not in key:
                continue
                
            # Key format usually: lora_unet_down_blocks_0_attentions_0_..._to_k.lora_up.weight
            parts = key.split(".")
            module_name_raw = parts[0] # e.g. lora_unet_down_blocks_0..._to_k
            
            # Normalize module name to match Sidecar convention
            # Sidecar: down_blocks.0.attentions.0...attn2.processor
            # We need to extract the underlying layer name
            
            # Logic:
            # 1. Remove 'lora_unet_' prefix
            # 2. Replace '_' with '.' but be careful about layer names that have underscores?
            # Diffusers naming: down_blocks.0.attentions.0...
            # LoRA naming: lora_unet_down_blocks_0_attentions_0...
            
            # Robust mapping:
            # We iterate Sidecar modules and try to fuzzy match the LoRA key
            pass 
        
        # IMPLEMENTATION:
        # Since string parsing is brittle, we do a topological search.
        # But for now let's build a map of {normalized_name: {up, down, alpha}}
        
        processed_deltas = {}
        
        # 1. Group keys
        temp_storage = {}
        
        for key, value in state_dict.items():
            if "lora_up.weight" in key:
                base = key.replace(".lora_up.weight", "")
                if base not in temp_storage: temp_storage[base] = {}
                temp_storage[base]["up"] = value.to(self.device, dtype=torch.float32) # Calc in FP32
            elif "lora_down.weight" in key:
                base = key.replace(".lora_down.weight", "")
                if base not in temp_storage: temp_storage[base] = {}
                temp_storage[base]["down"] = value.to(self.device, dtype=torch.float32)
            elif "alpha" in key:
                base = key.replace(".alpha", "")
                if base not in temp_storage: temp_storage[base] = {}
                temp_storage[base]["alpha"] = value.item()
                
        # 2. Normalize LoRA keys into sidecar ModuleDict keys, then compute dW.
        sidecar_modules = self.sidecar_net.neuro_modules # {layer_name: {to_k, to_v}}
        normalized_storage: Dict[str, Dict[str, Dict[str, torch.Tensor]]] = {}
        for base, data in temp_storage.items():
            mapped = _normalize_lora_base_to_sidecar(base, self.unet.config)
            if not mapped:
                continue
            sidecar_name, lane = mapped
            if sidecar_name not in sidecar_modules:
                continue
            normalized_storage.setdefault(sidecar_name, {})[lane] = data

        count = 0
        for sidecar_name, lane_storage in normalized_storage.items():
            layer_deltas = {}
            for lane, data in lane_storage.items():
                if "up" not in data or "down" not in data:
                    continue
                alpha = data.get("alpha", data["down"].shape[0])
                rank = data["down"].shape[0]
                scale = alpha / rank

                # dW = Up @ Down * scale
                dW = (data["up"] @ data["down"]) * scale
                layer_deltas[lane] = dW.to(self.dtype)

            if layer_deltas:
                processed_deltas[sidecar_name] = layer_deltas
                count += 1

        logger.info(
            "[Distiller] Matched %s/%s sidecar layers from %s LoRA bases.",
            count,
            len(sidecar_modules),
            len(temp_storage),
        )
                
        return processed_deltas

    def distill(self, steps: int = 1000, batch_size: int = 4, prompts: List[str] = None):
        """
        Main Distillation Loop (Data-Free / Synthetic)
        """
        logger.info(f"[Distiller] Starting distillation for {steps} steps...")
        
        # Prepare "Calibration Data"
        # Since we use Matrix Regression: dW * C_clip ≈ W_sidecar * C_llm
        # We need representative C_clip and C_llm vectors.
        # We can use a set of diversified prompts.
        
        if not prompts:
            # P1-12: Prompt Coverage (Trigger Words + Styles + Generic)
            prompts = [
                # Generic
                "masterpiece, best quality, ultra detailed",
                "a photo of a cat",
                "cyberpunk city",
                "anime girl",
                "landscape, mountains, nature",
                "1girl, solo, smile",
                # Style / Abstract
                "abstract art, colorful",
                "scifi spaceship",
                "texture, pattern",
                "concept art",
                # Semantic Triggers (Colors, lighting, composition)
                "red hair, blue eyes, sunset lighting",
                "macro photography, depth of field",
                "oil painting, van gogh style",
                "sketch, monochrome, pencil drawing",
                "wide angle shot, fish eye lens",
                "portrait, detailed face, cinematic lighting"
            ]
            
        # Precompute Embeddings (Cache)
        logger.info("[Distiller] Precomputing embeddings for calibration...")
        cache_c = []
        cache_l = []
        
        with torch.no_grad():
            for p in prompts:
                # CLIP Embedding (Teacher Input)
                c_emb = self.semantic_manager.encode_ghost_branch([p])
                if c_emb is None:
                    raise RuntimeError("Ghost CLIP teacher returned no embeddings")
                cache_c.append(c_emb.to(self.device).to(self.dtype))
                
                # LLM Embedding (Student Input)
                l_emb = self.semantic_manager.encode_main_branch([p]) # [1, N, 2048]
                cache_l.append(l_emb)
                
        # Stack
        cache_c = torch.cat(cache_c, dim=0).float() 
        cache_l = torch.cat(cache_l, dim=0).float() 
        
        # Training Loop
        pbar = tqdm(range(steps))
        for step in pbar:
            self.optimizer.zero_grad()
            total_loss = 0
            
            # Random Batch from Cache (or use full)
            C = cache_c.to(self.device)
            L = cache_l.to(self.device).to(self.dtype)
            
            # Iterate layers
            for layer_name, targets in self.teacher_deltas.items():
                if layer_name not in self.sidecar_net.neuro_modules: continue
                
                student_mod = self.sidecar_net.neuro_modules[layer_name]
                
                # Check K
                if "k" in targets:
                    dW_k = targets["k"].to(self.device).to(C.dtype)
                    C_k = _align_last_dim(C, int(dW_k.shape[1]))
                    target_act = C_k @ dW_k.t() # [B, Seq, Out]
                    
                    # Student:
                    # Matrix regression targets the raw LoRA delta projection.
                    # Runtime LaneNorm/gating lives in the fusion processor and
                    # should not be mixed into this direct dW*C objective.
                    pred_act = student_mod["to_k"](L)
                    
                    # Loss
                    t_pool = target_act.mean(dim=1) # [B, Out]
                    p_pool = pred_act.mean(dim=1)   # [B, Out]
                    
                    loss = F.mse_loss(p_pool.float(), t_pool.float())
                    total_loss += loss
                
                # Check V
                if "v" in targets:
                    dW_v = targets["v"].to(self.device).to(C.dtype)
                    C_v = _align_last_dim(C, int(dW_v.shape[1]))
                    target_act = C_v @ dW_v.t()
                    
                    pred_act = student_mod["to_v"](L)
                    
                    loss = F.mse_loss(pred_act.mean(dim=1).float(), target_act.mean(dim=1).float())
                    total_loss += loss
            
            total_loss.backward()
            self.optimizer.step()
            
            pbar.set_description(f"Loss: {total_loss.item():.4f}")
            
        logger.info("[Distiller] Distillation Complete.")
        
    def save(self, path: str):
        save_file(self.sidecar_net.state_dict(), path)
        logger.info(f"[Distiller] Saved Sidecar Adapter to {path}")

