
"""
Semantic Alignment Engine
Part of Lulynx Neuro-Link Architecture

Manages the lifecycle of the Teacher (CLIP) and Student (LLM) models.
Handles model loading, device placement, and forward passes for distillation.
"""

import torch
import torch.nn as nn
import logging
from typing import Tuple, Dict, Any, Optional

from .projector import LulynxUniversalProjector

logger = logging.getLogger(__name__)

class SemanticAlignerEngine(nn.Module):
    def __init__(
        self,
        student_model_path: str,
        teacher_model_path: str, # Usually path to SDXL TE (huggingface id or local path)
        projector_config: Dict[str, Any],
        use_lora: bool = True,
        device: str = "cuda"
    ):
        """
        Initialize the Aligner Engine.
        """
        super().__init__()
        self.device = device
        
        # 1. Load Student (LLM)
        logger.info(f"Loading Student LLM from {student_model_path}...")
        self.student, self.student_config, self.student_tokenizer = self._load_llm(student_model_path, use_lora)
        
        # 2. Load Teacher (CLIP/SDXL TE)
        logger.info(f"Loading Teacher CLIP from {teacher_model_path}...")
        self.teacher, self.teacher_config, self.teacher_tokenizer = self._load_clip(teacher_model_path)
        
        # 3. Setup Projector
        in_dim = self.student_config.hidden_size
        out_dim = self.teacher_config.hidden_size # e.g. 768 or 1280
        
        logger.info(f"Initializing LUP Projector: {in_dim} -> {out_dim}")
        self.projector = LulynxUniversalProjector(
            in_dim=in_dim,
            out_dim=out_dim,
            bake_in_norm=True
        )
        
        # Move to device
        self.to(device)
        
        # Lock Teacher
        self.teacher.eval()
        for param in self.teacher.parameters():
            param.requires_grad = False
            
    def _load_llm(self, path: str, use_lora: bool) -> Tuple[nn.Module, Any, Any]:
        try:
            from transformers import AutoModel, AutoConfig, AutoTokenizer
        except ImportError:
            raise ImportError("transformers is required. Please pip install transformers.")

        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        
        # Load in half precision for memory saving
        model = AutoModel.from_pretrained(
            path, 
            config=config, 
            torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            trust_remote_code=True
        )
        
        if use_lora:
            try:
                from peft import get_peft_model, LoraConfig, TaskType
                logger.info("Injecting LoRA into Student LLM...")
                # Apply LoRA to all linear layers (simplified)
                # For Qwen/Llama, usually target 'q_proj', 'v_proj', etc.
                # Auto-detection is better if possible, or use standard defaults
                peft_config = LoraConfig(
                    task_type=TaskType.FEATURE_EXTRACTION, 
                    inference_mode=False, 
                    r=16, 
                    lora_alpha=32, 
                    lora_dropout=0.1
                )
                model = get_peft_model(model, peft_config)
                model.print_trainable_parameters()
            except ImportError:
                logger.warning("peft not installed. Training full LLM (High VRAM usage warning!)")
        
        return model, config, tokenizer

    def _load_clip(self, path: str) -> Tuple[nn.Module, Any, Any]:
        """Load CLIP Text Model"""
        try:
            from transformers import CLIPTextModel, CLIPTextConfig, AutoConfig, CLIPTokenizer
        except ImportError:
            raise ImportError("transformers is required.")
            
        # Try loading as CLIP explicitly, or AutoModel if generic
        try:
            model = CLIPTextModel.from_pretrained(path)
            config = model.config
            tokenizer = CLIPTokenizer.from_pretrained(path)
        except Exception:
            # Fallback for OpenCLIP / SDXL TE variants
            try:
               model = CLIPTextModel.from_pretrained(path)
               tokenizer = CLIPTokenizer.from_pretrained(path)
               config = model.config
               config = model.config
            except Exception as e:
               # Last resort: Try loading tokenizer from subdirectory if 'tokenizer' exists
               # Simplified for Phase 1
               logger.error(f"Failed to load CLIP: {e}")
               raise ValueError(f"Could not load CLIP model or tokenizer from {path}")
            
        return model, config, tokenizer

    def forward(self, text_inputs_student, text_inputs_teacher):
        """
        Forward pass for both models.
        
        Args:
            text_inputs_student: Tokenized inputs for LLM
            text_inputs_teacher: Tokenized inputs for CLIP
        """
        # Teacher Forward (No Grad)
        with torch.no_grad():
            teacher_out = self.teacher(**text_inputs_teacher)
            # Use pooled output or last hidden state?
            # SDXL uses pooled output for time embedding, and last_hidden_state for cross-attn (2nd TE)
            # For alignment, we probably want to align the SEQUENCE (last_hidden_state)
            teacher_embeds = teacher_out.last_hidden_state # [B, 77, D_clip]
        
        # Student Forward
        student_out = self.student(**text_inputs_student)
        student_embeds = student_out.last_hidden_state # [B, Seq_LLM, D_llm]
        
        # Projection
        # Note: We need to handle length mismatch!
        # Initial strategy: align pooled token or resample?
        # For Phase 1 (Simple Alignment), we might align the mean pooled embedding?
        # OR: We force the student to output 77 tokens via padding/truncation?
        # BETTER: We align the sentence embedding (EOS token or Mean).
        # Wait, U-Net needs sequence [B, 77, D].
        # So we must project student_embeds [B, Seq, D_llm] -> [B, 77, D_clip].
        # The 'Projector' maps D -> D. It doesn't map Time -> Time.
        # We assume for now we use 'compatible' lengths or we just align the semantic Mean/Pool for general knowledge,
        # But U-Net needs spatial meaning.
        # FIX: For now, lets align the MEAN POOLED embedding to check semantic drift,
        # And maybe learn a 'Resampler' later. 
        # Actually, for Mode A (Loader), we don't need to match 77.
        # For Mode B (Compatible), Adapter needs to output 77.
        # Let's project per-token: [B, Seq, D_llm] -> [B, Seq, D_clip]
        # And we calculate loss on the first 77 tokens (if available) or padded?
        
        projected_embeds = self.projector(student_embeds)
        
        return teacher_embeds, projected_embeds

    def save_adapter(self, output_dir: str):
        """Save the projector and LLM Adapter (LoRA)"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save Projector
        torch.save(self.projector.state_dict(), f"{output_dir}/lup_projector.pt")
        
        # Save LoRA
        # If wrapped in PeftModel
        if hasattr(self.student, "save_pretrained"):
            self.student.save_pretrained(f"{output_dir}/llm_adapter")
        
