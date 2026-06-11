"""
Dual Validated TE Manager
双重验证 Text Encoder 管理器

职责：
1. 监控 Text Encoder 的状态 (Method A: Drift, Method B: Grad)
2. 当双重验证确认进入稳态时，将其卸载到 CPU 以节省显存
3. 通知 OrchestraController 触发系统冷却
"""

import torch
import numpy as np
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class DualValidatedTERemoval:
    """
    双重验证 TE 剔除/卸载器
    """
    
    def __init__(
        self,
        # 方法 A：嵌入漂移率
        embedding_drift_window: int = 10,
        embedding_drift_threshold: float = 0.001,
        
        # 方法 B：梯度范数
        grad_norm_window: int = 20,
        grad_norm_threshold: float = 1e-4,
        
        # 双重验证
        consecutive_steps: int = 10,  # 连续 N 步都满足
        
        # 策略
        removal_strategy: str = "to_cpu",  # to_cpu | remove
    ):
        self.embedding_drift_window = embedding_drift_window
        self.embedding_drift_threshold = embedding_drift_threshold
        self.grad_norm_window = grad_norm_window
        self.grad_norm_threshold = grad_norm_threshold
        self.consecutive_steps = consecutive_steps
        self.removal_strategy = removal_strategy
        
        # 状态
        self._step = 0
        self._is_removed = False
        self._consecutive_count = 0
        
        # 方法 A 历史
        self._embedding_history_1: List[torch.Tensor] = []
        self._embedding_history_2: List[torch.Tensor] = []
        
        # 方法 B 历史
        self._grad_norm_history_1: List[float] = []
        self._grad_norm_history_2: List[float] = []
        
        # 备份
        self._optimizer_state_backup: Dict = {}
        
    def validate_and_remove(
        self,
        text_encoder_1: torch.nn.Module,
        text_encoder_2: Optional[torch.nn.Module],
        optimizer: torch.optim.Optimizer,
    ) -> bool:
        """
        验证并可能剔除 TE
        
        Returns:
            True: 刚刚执行了剔除
            False: 未剔除 (或已剔除)
        """
        if self._is_removed:
            return False
            
        self._step += 1
        
        # 0. 检查 Orchestra 准入 (虽然我们不需要 Orchestra 批准才能移除，
        # 但我们需要遵循 "检查频率"。不过为了简单，我们在内部通过 consecutive_steps 控制频率)
        try:
            from core.lulynx_trainer.orchestra_controller import get_orchestra
            orchestra = get_orchestra()
            if not orchestra.should_run_te_check():
                return False
        except ImportError:
            pass

        # 1. 方法 A: Embedding Drift
        is_steady_a = self._method_a_embedding_drift(text_encoder_1, text_encoder_2)
        
        # 2. 方法 B: Gradient Norm
        is_steady_b = self._method_b_grad_norm(text_encoder_1, text_encoder_2)
        
        # 3. 双重验证
        if is_steady_a and is_steady_b:
            self._consecutive_count += 1
            if self._step % 100 == 0:
                logger.debug(f"[TE Manager] Steady count: {self._consecutive_count}/{self.consecutive_steps}")
            
            if self._consecutive_count >= self.consecutive_steps:
                # 执行剔除
                self._execute_removal(text_encoder_1, text_encoder_2, optimizer)
                
                # 通知 Orchestra
                try:
                    from core.lulynx_trainer.orchestra_controller import get_orchestra
                    get_orchestra().notify_te_removed()
                except ImportError:
                    pass
                    
                return True
        else:
            self._consecutive_count = 0
            
        return False

    def _execute_removal(self, te1, te2, optimizer):
        logger.info(f"[TE Manager] 📉 Dual Validation passed. Removing Text Encoders to save VRAM...")
        
        # 备份 optimizer 状态 (如果需要恢复)
        # 简单起见，这里不清空 optimizer state，因为 optimizer state 对 TE 并不大 (如果是 LoRA)
        # 如果是 Full Fine-tune TE，那 state 很大，必须清空。
        # 假设是 LoRA TE training。
        
        if self.removal_strategy == "to_cpu":
            self._move_to_cpu(te1, "TE1")
            if te2:
                self._move_to_cpu(te2, "TE2")
                
            # 释放 CUDA缓存
            torch.cuda.empty_cache()
            self._is_removed = True
            logger.info("[TE Manager] Text Encoders offloaded to CPU. VRAM freed.")

    def _move_to_cpu(self, model: torch.nn.Module, name: str):
        model.to(device="cpu", dtype=torch.float32)
        # 冻结梯度以防意外
        for p in model.parameters():
            p.requires_grad = False
        logger.info(f"[TE Manager] {name} moved to CPU.")

    # ========== Method A: Embedding Drift ==========
    
    def _method_a_embedding_drift(self, te1, te2) -> bool:
        # 获取 Embedding 层权重 snapshot
        emb1 = self._get_embedding_weight(te1)
        emb2 = self._get_embedding_weight(te2) if te2 else None
        
        steady1 = self._check_drift(emb1, self._embedding_history_1)
        steady2 = True
        if te2:
            steady2 = self._check_drift(emb2, self._embedding_history_2)
            
        return steady1 and steady2

    def _get_embedding_weight(self, model) -> Optional[torch.Tensor]:
        # 尝试获取第一个 embedding 层的权重引用
        # 注意：不要 detach 太多次，我们只需要一个小样本或者 mean
        # 为了性能，我们只取 mean 和 std 作为指纹
        for m in model.modules():
            if isinstance(m, torch.nn.Embedding):
                return m.weight.data.clone().detach().cpu() # Clone is expensive if huge
        # Fallback: traverse params
        for p in model.parameters():
            if p.requires_grad:
                return p.data.clone().detach().cpu()
        return None

    def _check_drift(self, current: Optional[torch.Tensor], history: List[torch.Tensor]) -> bool:
        if current is None: return True # 无参数训练 -> 视为稳态
        
        history.append(current)
        if len(history) > self.embedding_drift_window:
            history.pop(0)
            
        if len(history) < 2: return False
        
        # 计算相对于上一帧的漂移
        last = history[-2]
        #以此判断变化幅度
        drift = (current - last).abs().mean().item()
        
        return drift < self.embedding_drift_threshold

    # ========== Method B: Gradient Norm ==========
    
    def _method_b_grad_norm(self, te1, te2) -> bool:
        norm1 = self._compute_grad_norm(te1)
        norm2 = self._compute_grad_norm(te2) if te2 else 0.0
        
        self._grad_norm_history_1.append(norm1)
        if te2: self._grad_norm_history_2.append(norm2)
        
        # 维护窗口
        if len(self._grad_norm_history_1) > self.grad_norm_window:
            self._grad_norm_history_1.pop(0)
        if len(self._grad_norm_history_2) > self.grad_norm_window:
            self._grad_norm_history_2.pop(0)
            
        s1 = np.mean(self._grad_norm_history_1) < self.grad_norm_threshold if self._grad_norm_history_1 else False
        s2 = True
        if te2:
             s2 = np.mean(self._grad_norm_history_2) < self.grad_norm_threshold if self._grad_norm_history_2 else False
             
        return s1 and s2

    def _compute_grad_norm(self, model) -> float:
        total = 0.0
        count = 0
        for p in model.parameters():
            if p.grad is not None:
                total += p.grad.norm().item()
                count += 1
        return total / count if count > 0 else 0.0

from .semantic_tuner.loader import SemanticTunerManager

class SemanticTunerAwareTEManager:
    """
    Manages Text Encoders with support for Lulynx Semantic Base-Tuner (LLM Injection).
    """
    def __init__(self, config: Any, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.config = config
        self.device = device
        self.dtype = dtype
        self.semantic_tuner: Optional[SemanticTunerManager] = None
        self.legacy_removal_manager = DualValidatedTERemoval()
        self._legacy_te_removed = False
        self._cached_encoder_output: Optional[Dict[str, torch.Tensor]] = None

    def _get_config_value(self, key: str, default=None):
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return getattr(self.config, key, default)

    def prepare(self):
        """
        Load Semantic Base-Tuner if enabled.
        """
        enabled = self._get_config_value("semantic_tuner_enabled", False)

        if enabled:
            logger.info("[TE Manager] 🧠 Semantic Base-Tuner Mode Activated. Loading LLM pipeline...")
            llm_path = self._get_config_value("semantic_llm_path", "") or "Qwen/Qwen2.5-0.5B"
            projector_path = self._get_config_value("semantic_projector_path", "") or None
            teacher_path = (
                self._get_config_value("semantic_teacher_path", "")
                or self._get_config_value("teacher_path", "")
                or None
            )
            architecture_mode = self._get_config_value("architecture_mode", "hybrid")
            max_token_length = int(self._get_config_value("max_token_length", 512) or 512)
             
            logger.info("Initializing Semantic Base-Tuner Manager...")
            try:
                from .semantic_tuner.loader import SemanticTunerManager

                tuner_kwargs = {
                    "llm_path": llm_path,
                    "projector_path": projector_path,
                    "architecture_mode": architecture_mode,
                    "max_token_length": max_token_length,
                    "device": self.device,
                    "dtype": self.dtype,
                }
                if teacher_path:
                    tuner_kwargs["teacher_path"] = teacher_path

                self.semantic_tuner = SemanticTunerManager(**tuner_kwargs)
                self.semantic_tuner.load_dual_stream_context()
                logger.info("Semantic Base-Tuner Context Loaded.")
            except Exception as e:
                logger.error(f"Failed to initialize Semantic Tuner: {e}")
                raise
            
    def get_semantic_context(self):
        """Returns the Dual-Stream context (LLM, Projector, CLIP)"""
        if self.semantic_tuner:
            return self.semantic_tuner.get_context()
        return None
            # NeuroLinkManager.load_neuro_link is called in __init__ or we can call it here if needed
            # Assuming NeuroLinkManager.__init__ handles loading for now as per previous implementation

    def encode_prompts(self, prompts: List[str]) -> Dict[str, torch.Tensor]:
        """
        Unified encoding interface for TrainingLoop.
        Returns a diffusers-style dict: {"encoder_hidden_states": ..., "pooled_prompt_embeds": ...}

        After TE removal, returns cached embeddings from the last encode
        (the TE has converged, so embeddings are stable).
        """
        if self._legacy_te_removed and self._cached_encoder_output is not None:
            return self._cached_encoder_output
        if self.semantic_tuner:
            ctx = self.semantic_tuner.get_context() or {}
            mode = str(ctx.get("mode", "hybrid") or "hybrid").lower()

            main_embeds = self.semantic_tuner.encode_main_branch(prompts)
            ghost_embeds = None
            if mode == "hybrid":
                ghost_embeds = self.semantic_tuner.encode_ghost_branch(prompts)

            if ghost_embeds is not None:
                # Sidecar processors consume the dual-stream dict directly.
                return {
                    "encoder_hidden_states": {
                        "main": main_embeds,
                        "ghost": ghost_embeds,
                    },
                }

            return {
                "encoder_hidden_states": main_embeds,
            }
        return {}

    def maybe_remove_text_encoders(
        self,
        text_encoder_1: torch.nn.Module,
        text_encoder_2: Optional[torch.nn.Module],
        optimizer: torch.optim.Optimizer,
    ) -> bool:
        """Run the legacy dual-validated TE offload path once it is safe to do so."""
        if self.semantic_tuner is not None or self._legacy_te_removed:
            return False

        removed = self.legacy_removal_manager.validate_and_remove(
            text_encoder_1=text_encoder_1,
            text_encoder_2=text_encoder_2,
            optimizer=optimizer,
        )
        if removed:
            self._legacy_te_removed = True
        return removed

    def cache_encoder_output(self, output: Dict[str, torch.Tensor]) -> None:
        """Cache the last encoder output for use after TE removal."""
        if not self._legacy_te_removed:
            # Detach and clone to avoid holding the computation graph
            cached = {}
            for k, v in output.items():
                if isinstance(v, torch.Tensor):
                    cached[k] = v.detach().clone()
                elif isinstance(v, dict):
                    cached[k] = {
                        kk: vv.detach().clone() if isinstance(vv, torch.Tensor) else vv
                        for kk, vv in v.items()
                    }
                else:
                    cached[k] = v
            self._cached_encoder_output = cached

    @property
    def legacy_te_removed(self) -> bool:
        return self._legacy_te_removed

