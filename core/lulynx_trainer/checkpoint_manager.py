"""
断点续训管理器

支持训练状态的保存和恢复
"""

import json
import logging
import shutil
from typing import Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime
from core.safe_pickle import safe_torch_load

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    """检查点状态"""
    # 进度
    epoch: int = 0
    global_step: int = 0
    
    # 配置摘要
    config_hash: str = ""
    output_name: str = ""
    
    # 时间
    created_at: str = ""
    training_time_seconds: float = 0.0
    
    # 指标
    last_loss: float = 0.0
    best_loss: float = float("inf")
    
    # 路径
    model_path: str = ""
    optimizer_path: str = ""
    scheduler_path: str = ""
    
    # 元数据
    lora_rank: int = 0
    total_epochs: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointState":
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


class CheckpointManager:
    """
    检查点管理器
    
    功能:
    - 自动保存训练状态
    - 断点续训
    - 检查点轮换 (保留最近 N 个)
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "./checkpoints",
        max_checkpoints: int = 3,
        save_interval_steps: int = 500,
        save_interval_minutes: float = 30.0,
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_checkpoints = max_checkpoints
        self.save_interval_steps = save_interval_steps
        self.save_interval_minutes = save_interval_minutes
        
        # 状态
        self._last_save_step = 0
        self._last_save_time = datetime.now()
        self._current_state: Optional[CheckpointState] = None
    
    def should_save(self, step: int) -> bool:
        """检查是否应该保存"""
        # 按步数
        if step - self._last_save_step >= self.save_interval_steps:
            return True
        
        # 按时间
        elapsed = (datetime.now() - self._last_save_time).total_seconds() / 60
        if elapsed >= self.save_interval_minutes:
            return True
        
        return False
    
    def save(
        self,
        state: CheckpointState,
        model_state_dict: Dict = None,
        optimizer_state_dict: Dict = None,
        scheduler_state_dict: Dict = None,
    ) -> str:
        """
        保存检查点
        
        Returns:
            检查点路径
        """
        import torch
        
        # 创建检查点目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_name = f"checkpoint_step{state.global_step}_{timestamp}"
        ckpt_path = self.checkpoint_dir / ckpt_name
        ckpt_path.mkdir(parents=True, exist_ok=True)
        
        # 保存模型权重
        if model_state_dict:
            model_file = ckpt_path / "model.pt"
            torch.save(model_state_dict, model_file)
            state.model_path = str(model_file)
        
        # 保存优化器状态
        if optimizer_state_dict:
            optimizer_file = ckpt_path / "optimizer.pt"
            torch.save(optimizer_state_dict, optimizer_file)
            state.optimizer_path = str(optimizer_file)
        
        # 保存调度器状态
        if scheduler_state_dict:
            scheduler_file = ckpt_path / "scheduler.pt"
            torch.save(scheduler_state_dict, scheduler_file)
            state.scheduler_path = str(scheduler_file)

        # 保存状态
        state.created_at = datetime.now().isoformat()
        state_file = ckpt_path / "state.json"
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(state.to_dict(), f, indent=2, ensure_ascii=False)
        
        # 更新最新链接
        latest_link = self.checkpoint_dir / "latest"
        try:
            if latest_link.exists():
                latest_link.unlink()
            latest_link.symlink_to(ckpt_name, target_is_directory=True)
        except OSError:
            # Windows symlink requires admin or Dev mode. Fallback to copy or ignore.
            try:
               if latest_link.exists():
                   if latest_link.is_dir():
                       shutil.rmtree(latest_link)
                   else:
                       latest_link.unlink()
               shutil.copytree(ckpt_path, latest_link)
            except Exception as e:
                logger.warning(f"Failed to create 'latest' link/copy: {e}")
        
        # 清理旧检查点
        self._cleanup_old_checkpoints()
        
        # 更新状态
        self._last_save_step = state.global_step
        self._last_save_time = datetime.now()
        self._current_state = state
        
        logger.info(f"[Checkpoint] Saved: {ckpt_path}")
        return str(ckpt_path)
    
    def load_latest(self) -> Optional[CheckpointState]:
        """加载最新的检查点"""
        latest_link = self.checkpoint_dir / "latest"
        
        if not latest_link.exists():
            # 尝试找最新的目录
            checkpoints = sorted(
                self.checkpoint_dir.glob("checkpoint_*"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            if not checkpoints:
                return None
            latest_path = checkpoints[0]
        else:
            latest_path = latest_link.resolve()
        
        return self.load(str(latest_path))
    
    def load(self, checkpoint_path: str) -> Optional[CheckpointState]:
        """加载指定检查点"""
        ckpt_path = Path(checkpoint_path)
        
        if not ckpt_path.exists():
            logger.error(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        state_file = ckpt_path / "state.json"
        if not state_file.exists():
            logger.error(f"State file not found: {state_file}")
            return None
        
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            state = CheckpointState.from_dict(data)
            
            # 更新路径为绝对路径
            if state.model_path:
                state.model_path = str(ckpt_path / Path(state.model_path).name)
            if state.optimizer_path:
                state.optimizer_path = str(ckpt_path / Path(state.optimizer_path).name)
            if state.scheduler_path:
                state.scheduler_path = str(ckpt_path / Path(state.scheduler_path).name)
            
            self._current_state = state
            logger.info(f"[Checkpoint] Loaded: step={state.global_step}, epoch={state.epoch}")
            
            return state
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def restore_model(self, state: CheckpointState, model) -> bool:
        """恢复模型权重"""
        import torch
        
        if not state.model_path or not Path(state.model_path).exists():
            return False
        
        try:
            state_dict = safe_torch_load(state.model_path, map_location="cpu")
            model.load_state_dict(state_dict, strict=False)
            logger.info(f"[Checkpoint] Model restored from {state.model_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore model: {e}")
            return False
    
    def restore_optimizer(self, state: CheckpointState, optimizer) -> bool:
        """恢复优化器状态"""
        import torch
        
        if not state.optimizer_path or not Path(state.optimizer_path).exists():
            return False
        
        try:
            state_dict = safe_torch_load(state.optimizer_path, map_location="cpu")
            optimizer.load_state_dict(state_dict)
            logger.info(f"[Checkpoint] Optimizer restored from {state.optimizer_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore optimizer: {e}")
            return False
    
    def restore_scheduler(self, state: CheckpointState, scheduler) -> bool:
        """恢复调度器状态"""
        import torch
        
        if not state.scheduler_path or not Path(state.scheduler_path).exists():
            return False
        
        try:
            state_dict = safe_torch_load(state.scheduler_path, map_location="cpu")
            scheduler.load_state_dict(state_dict)
            logger.info(f"[Checkpoint] Scheduler restored from {state.scheduler_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore scheduler: {e}")
            return False
    
    def _cleanup_old_checkpoints(self):
        """清理旧检查点"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # 保留最近 N 个
        for old_ckpt in checkpoints[self.max_checkpoints:]:
            try:
                shutil.rmtree(old_ckpt)
                logger.info(f"[Checkpoint] Removed old: {old_ckpt.name}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint: {e}")
    
    def list_checkpoints(self) -> list:
        """列出所有检查点"""
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_*"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        result = []
        for ckpt in checkpoints:
            state_file = ckpt / "state.json"
            if state_file.exists():
                try:
                    with open(state_file, "r") as f:
                        data = json.load(f)
                    result.append({
                        "path": str(ckpt),
                        "name": ckpt.name,
                        "step": data.get("global_step", 0),
                        "epoch": data.get("epoch", 0),
                        "created_at": data.get("created_at", ""),
                    })
                except Exception:
                    pass
        
        return result
    
    def has_checkpoint(self) -> bool:
        """检查是否有可用检查点"""
        return len(list(self.checkpoint_dir.glob("checkpoint_*"))) > 0
