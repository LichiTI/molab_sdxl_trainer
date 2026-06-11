
"""
Lulynx Semantic Aligner Trainer
Phase 1 of Neuro-Link Architecture

This trainer distills the semantic knowledge of a small LLM (Student)
into the embedding space of SDXL's CLIP Text Encoder (Teacher).
"""

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from typing import Generator, Tuple, Dict, Any, List, Type
import logging
import threading
import os
import time

from ..trainers.base import BaseTrainer, TrainerConfig, TrainerProgress, TrainerResult
from ..semantic_brain.engine import SemanticAlignerEngine
from ..semantic_brain.dataset import get_dataloader

logger = logging.getLogger("SemanticAligner")

class SemanticAlignerConfig(TrainerConfig):
    """Configuration for Semantic Alignment"""
    student_model_path: str
    teacher_model_path: str
    train_data_paths: List[str]
    max_steps: int = 1000
    batch_size: int = 4
    learning_rate: float = 1e-4
    save_every: int = 500
    use_lora: bool = True
    context_length: int = 77 # Target context length (CLIP standard)

class SemanticAlignerTrainer(BaseTrainer):
    id = "semantic-aligner"
    name = "Lulynx Semantic Aligner"
    version = "1.0.0"
    author = "Lulynx Team"
    description = "Distill LLM knowledge into CLIP semantic space (Neuro-Link Phase 1)"

    def get_config_schema(self) -> Type[SemanticAlignerConfig]:
        return SemanticAlignerConfig

    def validate_config(self, config: dict) -> Tuple[bool, str]:
        # Check paths
        required = ["student_model_path", "teacher_model_path", "train_data_paths"]
        for field in required:
            if not config.get(field):
                return False, f"Missing required field: {field}"
        
        if not os.path.exists(config.get("student_model_path", "")):
             return False, "Student model path not found"
        if not os.path.exists(config.get("teacher_model_path", "")):
             return False, "Teacher model path not found"
             
        return True, ""

    def estimate_vram(self, config: dict) -> float:
        # Rough estimate: LLM (1GB) + CLIP (0.5GB) + Opt State (Depends) + Activations
        # 16GB is safe. 8GB might be tight if batch size > 1.
        return 6.0 # Conservative estimate in GB

    def stop(self) -> bool:
        self._stop_event.set()
        return True

    def train(self, config: dict) -> Generator[TrainerProgress, None, TrainerResult]:
        self._stop_event = threading.Event()
        cfg = SemanticAlignerConfig(**config)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. Initialize Engine
        yield TrainerProgress(message="Initializing Semantic Engine...", progress_percent=0.0)
        try:
            engine = SemanticAlignerEngine(
                student_model_path=cfg.student_model_path,
                teacher_model_path=cfg.teacher_model_path,
                projector_config={}, # Defaults
                use_lora=cfg.use_lora,
                device=device
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            return TrainerResult(success=False, message=f"Engine Init Failed: {str(e)}")

        # 2. Optimizer
        # Optimize Projector + Student Adapter (if LoRA)
        params_to_optimize = list(engine.projector.parameters())
        if cfg.use_lora:
            # Only optimize LoRA params in student
            params_to_optimize += [p for n, p in engine.student.named_parameters() if "lora" in n and p.requires_grad]
        else:
            # Full finetune student? Dangerous for VRAM. Assume LoRA for now or user risk.
            if hasattr(engine.student, "parameters"):
                 params_to_optimize += [p for p in engine.student.parameters() if p.requires_grad]

        optimizer = AdamW(params_to_optimize, lr=cfg.learning_rate)
        
        # 3. DataLoader
        dataloader = get_dataloader(
            file_paths=cfg.train_data_paths, 
            weights=[1.0]*len(cfg.train_data_paths), # Uniform for now
            batch_size=cfg.batch_size
        )
        
        # 4. Training Loop
        total_steps = cfg.max_steps
        step = 0
        running_loss = 0.0
        
        engine.train()
        
        start_time = time.time()
        
        # Loop efficiently
        # Since dataloader is infinite, we just iterate 'max_steps' times
        data_iter = iter(dataloader)
        
        while step < total_steps:
            if self._stop_event.is_set():
                return TrainerResult(success=False, message="Stopped by user")
            
            try:
                text_batch = next(data_iter) # List of strings
            except StopIteration:
                break # Should not happen with infinite dataset but safety
            
            # --- Tokenization ---
            # Tokenize for Student (LLM)
            # LLMs handle long context, but for efficiency we might truncate to avoiding OOM
            # Qwen context is 32k, but we don't need that much for alignment typically
            student_inputs = engine.student_tokenizer(
                text_batch, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=512 # Reasonable limit for 24G
            ).to(device)
            
            # Tokenize for Teacher (CLIP) - Max 77
            teacher_inputs = engine.teacher_tokenizer(
                text_batch,
                return_tensors="pt",
                padding="max_length",
                max_length=77,
                truncation=True
            ).to(device)

            # --- Forward ---
            # Mixed Precision
            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                teacher_embeds, projected_embeds = engine(student_inputs, teacher_inputs)
                
                # --- Loss Calculation ---
                # We want projected_embeds [B, Seq, D] to match teacher_embeds [B, 77, D]
                # Problem: Sequence length mismatch.
                # Projector output length = student input length (e.g., 20)
                # Teacher output length = 77
                
                # Strategy: 
                # 1. Padding Logic: We assume the Projector learns to map valid tokens to valid tokens.
                #    But LLM tokenizer != CLIP tokenizer. 1 word might be 2 tokens in LLM and 1 in CLIP.
                #    So direct token-to-token alignment is structurally impossible unless re-tokenized.
                
                # 2. Semantic Distillation (Pooling Strategy):
                #    Align the EOS/Pooling token.
                #    CLIP pooled output represents the "Whole Sentence". 
                #    LLM last token (EOS) represents "Whole Sentence".
                #    Aligning Pooled Output is safest for Phase 1.
                #    BUT U-Net Cross-Attention needs spatial tokens!
                
                # 3. Resampler Strategy (Architecture Update - Implementation Detail):
                #    If we want U-Net compatibility, we need exactly 77 tokens.
                #    The simple Linear Projector cannot change Sequence Length.
                #    We need a "Resampler" (like Perceiver in Flamingo/IP-Adapter) to map N tokens -> 77 tokens.
                #    However, for the MVP "Projector", maybe we just align the mean? 
                
                #    RFC says: "Student+LUP". LUP is MLP.
                #    If LUP is just MLP, it preserves length.
                #    If we use this in "Loader Mode", U-Net Cross-Attention handles ANY length (2048 dim, N tokens).
                #    So for "Loader Mode", we don't need 77 tokens. We just need semantics.
                #    BUT for "Compatible Mode" (Phase 1 target?), we need to behave like CLIP (77 tokens).
                
                #    Compromise for MVP Phase 1 (Compatible Mode):
                #    Pad/Truncate Student output to 77 tokens? 
                #    Or better: Use a simple Resampler (Linear+Interpolate or Attention Pool).
                #    For simplicity in this file: 
                #    Resize student output to 77 using Interpolation?
                #    Or just verify lengths match? (Impossible with different tokenizers).
                
                #    Let's align Mean Pooling + Max Pooling for now to force semantics.
                #    Loss = MSE(Mean(Student), Mean(Teacher)) + MSE(Max(Student), Max(Teacher))
                #    This forces the "Global Semantics" to be identical.
                #    This is valid for "Concept Injection".
                
                #    Let's use Mean Pooling alignment for Phase 1 MVP.
                
                # Mask out padding for mean calculation
                # (Simplified: just plain mean for MVP)
                
                s_mean = projected_embeds.mean(dim=1)
                t_mean = teacher_embeds.mean(dim=1)
                
                loss = F.mse_loss(s_mean, t_mean)
            
            # --- Backward ---
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            step += 1
            
            # Progress update
            if step % 10 == 0:
                avg_loss = running_loss / 10
                running_loss = 0.0
                yield TrainerProgress(
                    step=step, 
                    total_steps=total_steps, 
                    loss=avg_loss, 
                    message=f"Aligning... Loss: {avg_loss:.4f}",
                    progress_percent=step/total_steps
                )
                
            # Checkpoint
            if step % cfg.save_every == 0 or step == total_steps:
                save_path = f"{cfg.output_dir}/checkpoint-{step}"
                engine.save_adapter(save_path)
        
        engine.save_adapter(f"{cfg.output_dir}/final")
        return TrainerResult(success=True, message="Alignment Complete", output_path=f"{cfg.output_dir}/final")

