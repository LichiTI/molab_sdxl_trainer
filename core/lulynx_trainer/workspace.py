"""
Workspace 管理器

项目和配置管理
"""

import json
import shutil
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceInfo:
    """工作区信息"""
    name: str
    path: str
    created_at: str
    updated_at: str
    model_type: str = ""
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # 训练相关
    base_model: str = ""
    lora_output: str = ""
    last_checkpoint: str = ""
    total_steps: int = 0
    
    # 统计
    num_images: int = 0
    num_epochs_completed: int = 0


class WorkspaceManager:
    """
    Workspace 管理器
    
    功能:
    - 创建/删除工作区
    - 保存/加载配置
    - 管理检查点
    - 自动备份
    """
    
    WORKSPACE_FILE = ".lulynx_workspace.json"
    CONFIG_FILE = "config.toml"
    CHECKPOINTS_DIR = "checkpoints"
    BACKUPS_DIR = "backups"
    SAMPLES_DIR = "samples"
    
    def __init__(self, base_dir: str = "./workspaces"):
        self.base_dir = Path(base_dir).absolute()
        
        # Path validation
        if self.base_dir.exists() and not self.base_dir.is_dir():
            raise ValueError(f"Base directory path exists but is not a directory: {self.base_dir}")
            
        try:
            self.base_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create base directory: {e}")
            raise
            
        self._workspaces: Dict[str, WorkspaceInfo] = {}
        self._load_all_workspaces()
    
    def _load_all_workspaces(self):
        """加载所有工作区信息"""
        self._workspaces = {}
        
        for workspace_dir in self.base_dir.iterdir():
            if not workspace_dir.is_dir():
                continue
            
            info_path = workspace_dir / self.WORKSPACE_FILE
            if info_path.exists():
                try:
                    with open(info_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    self._workspaces[workspace_dir.name] = WorkspaceInfo(**data)
                except Exception as e:
                    logger.warning(f"Failed to load workspace {workspace_dir.name}: {e}")
    
    def create(
        self,
        name: str,
        model_type: str = "sdxl",
        description: str = "",
    ) -> WorkspaceInfo:
        """创建新工作区"""
        workspace_path = self.base_dir / name
        
        if workspace_path.exists():
            raise ValueError(f"Workspace {name} already exists")
        
        # 创建目录结构
        try:
            workspace_path.mkdir(parents=True)
            (workspace_path / self.CHECKPOINTS_DIR).mkdir(exist_ok=True)
            (workspace_path / self.BACKUPS_DIR).mkdir(exist_ok=True)
            (workspace_path / self.SAMPLES_DIR).mkdir(exist_ok=True)
            (workspace_path / "dataset").mkdir(exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create workspace directories for {name}: {e}")
            # Cleanup if possible
            if workspace_path.exists():
                shutil.rmtree(workspace_path, ignore_errors=True)
            raise
        
        # 创建工作区信息
        now = datetime.now().isoformat()
        info = WorkspaceInfo(
            name=name,
            path=str(workspace_path),
            created_at=now,
            updated_at=now,
            model_type=model_type,
            description=description,
        )
        
        # 保存
        self._save_workspace_info(info)
        self._workspaces[name] = info
        
        logger.info(f"[WorkspaceManager] Created workspace: {name}")
        return info
    
    def delete(self, name: str, force: bool = False):
        """删除工作区"""
        if name not in self._workspaces:
            raise ValueError(f"Workspace {name} not found")
        
        workspace_path = Path(self._workspaces[name].path)
        
        if not force:
            # 检查是否有重要文件
            checkpoints = list((workspace_path / self.CHECKPOINTS_DIR).glob("*"))
            if checkpoints:
                raise ValueError(
                    f"Workspace has {len(checkpoints)} checkpoints. "
                    "Use force=True to delete anyway."
                )
        
        try:
            shutil.rmtree(workspace_path)
        except Exception as e:
            logger.error(f"Failed to remove workspace directory {workspace_path}: {e}")
            if not force:
                raise
        del self._workspaces[name]
        
        logger.info(f"[WorkspaceManager] Deleted workspace: {name}")
    
    def get(self, name: str) -> Optional[WorkspaceInfo]:
        """获取工作区信息"""
        return self._workspaces.get(name)
    
    def list(self) -> List[WorkspaceInfo]:
        """列出所有工作区"""
        return list(self._workspaces.values())
    
    def update(self, name: str, **kwargs):
        """更新工作区信息"""
        if name not in self._workspaces:
            raise ValueError(f"Workspace {name} not found")
        
        info = self._workspaces[name]
        
        for key, value in kwargs.items():
            if hasattr(info, key):
                setattr(info, key, value)
        
        info.updated_at = datetime.now().isoformat()
        
        self._save_workspace_info(info)
    
    def _save_workspace_info(self, info: WorkspaceInfo):
        """保存工作区信息"""
        info_path = Path(info.path) / self.WORKSPACE_FILE
        
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(asdict(info), f, indent=2, ensure_ascii=False)
    
    def save_config(self, name: str, config: Dict[str, Any]):
        """保存训练配置"""
        if name not in self._workspaces:
            raise ValueError(f"Workspace {name} not found")
        
        import toml
        
        config_path = Path(self._workspaces[name].path) / self.CONFIG_FILE
        
        with open(config_path, "w", encoding="utf-8") as f:
            toml.dump(config, f)
        
        logger.info(f"[WorkspaceManager] Saved config to {config_path}")
    
    def load_config(self, name: str) -> Optional[Dict[str, Any]]:
        """加载训练配置"""
        if name not in self._workspaces:
            return None
        
        import toml
        
        config_path = Path(self._workspaces[name].path) / self.CONFIG_FILE
        
        if not config_path.exists():
            return None
        
        with open(config_path, "r", encoding="utf-8") as f:
            return toml.load(f)
    
    def get_latest_checkpoint(self, name: str) -> Optional[str]:
        """获取最新检查点"""
        if name not in self._workspaces:
            return None
        
        checkpoint_dir = Path(self._workspaces[name].path) / self.CHECKPOINTS_DIR

        latest_link = checkpoint_dir / "latest"
        if latest_link.exists():
            try:
                return str(latest_link.resolve())
            except OSError:
                return str(latest_link)

        checkpoints = sorted(
            checkpoint_dir.glob("checkpoint_*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        return str(checkpoints[0]) if checkpoints else None


class AutoBackup:
    """
    自动备份管理器
    
    定期备份检查点和配置
    """
    
    def __init__(
        self,
        workspace_path: str,
        max_backups: int = 5,
        backup_interval_steps: int = 500,
    ):
        self.workspace_path = Path(workspace_path)
        self.max_backups = max_backups
        self.backup_interval_steps = backup_interval_steps
        self.last_backup_step = 0
        
        self.backup_dir = self.workspace_path / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def should_backup(self, current_step: int) -> bool:
        """判断是否应该备份"""
        return current_step - self.last_backup_step >= self.backup_interval_steps
    
    def backup(
        self,
        current_step: int,
        checkpoint_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """执行备份"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"backup_step{current_step}_{timestamp}"
        backup_path = self.backup_dir / backup_name
        backup_path.mkdir(parents=True)
        
        # 备份检查点
        if checkpoint_path and Path(checkpoint_path).exists():
            src = Path(checkpoint_path)
            dst = backup_path / src.name
            if src.is_dir():
                shutil.copytree(src, dst)
            else:
                shutil.copy2(src, dst)
        
        # 备份配置
        if config:
            import toml
            with open(backup_path / "config.toml", "w") as f:
                toml.dump(config, f)
        
        # 清理旧备份
        self._cleanup_old_backups()
        
        self.last_backup_step = current_step
        logger.info(f"[AutoBackup] Created backup: {backup_name}")
    
    def _cleanup_old_backups(self):
        """清理旧备份"""
        backups = sorted(
            self.backup_dir.iterdir(),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        
        for old_backup in backups[self.max_backups:]:
            shutil.rmtree(old_backup)
            logger.debug(f"[AutoBackup] Deleted old backup: {old_backup.name}")
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """列出所有备份"""
        backups = []
        
        for backup_path in sorted(self.backup_dir.iterdir(), reverse=True):
            if backup_path.is_dir():
                backups.append({
                    "name": backup_path.name,
                    "path": str(backup_path),
                    "created_at": datetime.fromtimestamp(
                        backup_path.stat().st_mtime
                    ).isoformat(),
                })
        
        return backups
    
    def restore(self, backup_name: str, target_dir: str):
        """恢复备份"""
        backup_path = self.backup_dir / backup_name
        
        if not backup_path.exists():
            raise ValueError(f"Backup {backup_name} not found")
        
        target = Path(target_dir)
        
        target.mkdir(parents=True, exist_ok=True)

        for file in backup_path.iterdir():
            dst = target / file.name
            if file.is_dir():
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(file, dst)
            else:
                shutil.copy2(file, dst)
        
        logger.info(f"[AutoBackup] Restored backup {backup_name} to {target_dir}")


# ========== 便捷函数 ==========

def get_workspace_manager(base_dir: str = "./workspaces") -> WorkspaceManager:
    """获取工作区管理器"""
    return WorkspaceManager(base_dir)


def create_auto_backup(
    workspace_path: str,
    max_backups: int = 5,
) -> AutoBackup:
    """创建自动备份管理器"""
    return AutoBackup(workspace_path, max_backups=max_backups)
