"""
检查点管理器 (Checkpoint Manager)

提供流程执行的检查点保存和恢复功能，支持：
- 每个步骤后自动保存检查点
- 从任意检查点恢复执行
- 会话历史管理
- 检查点数据的持久化存储
"""

import json
import time
import shutil
import logging
import threading
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from enum import Enum
from PIL import Image


class SessionStatus(Enum):
    """会话状态"""
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class FlowCheckpoint:
    """
    单个检查点
    
    保存了流程执行到某一步骤后的完整状态，
    可用于从该点恢复执行。
    """
    checkpoint_id: str           # 唯一标识
    step_index: int              # 在第几步之后创建的 (-1 表示输入数据)
    step_name: str               # 步骤名称
    timestamp: float             # 创建时间戳
    
    # 数据存储位置
    data_dir: str                # 检查点数据目录
    items_count: int             # 数据项数量
    
    # 流程上下文快照
    flow_definition: List[Dict]  # 流程定义
    execution_config: Dict       # 执行配置
    
    # 统计信息
    success_count: int = 0
    error_count: int = 0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FlowCheckpoint':
        return cls(**data)


@dataclass
class FlowSession:
    """
    流程执行会话
    
    代表一次完整的流程执行，包含多个检查点。
    """
    session_id: str
    task_name: str
    created_at: float
    updated_at: float = 0
    status: str = "running"
    checkpoints: List[FlowCheckpoint] = field(default_factory=list)
    
    # 执行统计
    total_steps: int = 0
    completed_steps: int = 0
    total_items: int = 0
    
    # 错误日志
    errors: List[Dict] = field(default_factory=list)
    
    def __post_init__(self):
        if self.updated_at == 0:
            self.updated_at = self.created_at
    
    def get_latest_checkpoint(self) -> Optional[FlowCheckpoint]:
        """获取最新的检查点"""
        return self.checkpoints[-1] if self.checkpoints else None
    
    def get_checkpoint_at_step(self, step_index: int) -> Optional[FlowCheckpoint]:
        """获取指定步骤的检查点"""
        for cp in reversed(self.checkpoints):
            if cp.step_index == step_index:
                return cp
        return None
    
    def get_resumable_checkpoints(self) -> List[FlowCheckpoint]:
        """获取可用于恢复的检查点列表"""
        return [cp for cp in self.checkpoints if cp.step_index >= -1]
    
    def to_dict(self) -> Dict:
        data = asdict(self)
        data['checkpoints'] = [cp.to_dict() for cp in self.checkpoints]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'FlowSession':
        checkpoints_data = data.pop('checkpoints', [])
        session = cls(**data)
        session.checkpoints = [FlowCheckpoint.from_dict(cp) for cp in checkpoints_data]
        return session


class CheckpointManager:
    """
    检查点管理器
    
    负责检查点的创建、保存、加载和清理。
    """
    
    # 元数据文件名
    SESSIONS_FILE = "sessions.json"
    METADATA_FILE = "metadata.json"
    ITEMS_FILE = "items.json"
    
    def __init__(self, base_dir: Optional[Path] = None):
        """
        初始化检查点管理器
        
        Args:
            base_dir: 检查点存储根目录，默认为 ./checkpoints
        """
        self.logger = logging.getLogger('CheckpointManager')
        
        if base_dir is None:
            base_dir = Path("./checkpoints")
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.sessions: Dict[str, FlowSession] = {}
        self._load_sessions()
    
    def _load_sessions(self):
        """从磁盘加载会话列表"""
        sessions_file = self.base_dir / self.SESSIONS_FILE
        if sessions_file.exists():
            try:
                with open(sessions_file, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError:
                        self.logger.warning(f"Session file {sessions_file} corrupted. Using empty session list.")
                        data = {}
                for session_data in data.get('sessions', []):
                    session = FlowSession.from_dict(session_data)
                    self.sessions[session.session_id] = session
                self.logger.info(f"已加载 {len(self.sessions)} 个会话")
            except Exception as e:
                self.logger.error(f"加载会话失败: {e}")
    
    def _save_sessions(self):
        """保存会话列表到磁盘"""
        sessions_file = self.base_dir / self.SESSIONS_FILE
        try:
            data = {
                'sessions': [s.to_dict() for s in self.sessions.values()],
                'updated_at': time.time()
            }
            with open(sessions_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"保存会话失败: {e}")
    
    def create_session(
        self, 
        task_name: str,
        total_steps: int = 0,
        total_items: int = 0
    ) -> FlowSession:
        """
        创建新的执行会话
        
        Args:
            task_name: 任务名称
            total_steps: 总步骤数
            total_items: 总数据项数
            
        Returns:
            新创建的 FlowSession
        """
        session_id = f"session_{int(time.time() * 1000)}"
        session = FlowSession(
            session_id=session_id,
            task_name=task_name,
            created_at=time.time(),
            total_steps=total_steps,
            total_items=total_items
        )
        
        # 创建会话目录
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        self.sessions[session_id] = session
        self._save_sessions()
        
        self.logger.info(f"创建会话: {session_id} ({task_name})")
        return session
    
    def save_checkpoint(
        self,
        session: FlowSession,
        step_index: int,
        step_name: str,
        data_items: List[Any],
        flow_definition: List[Dict],
        execution_config: Dict,
        success_count: int = 0,
        error_count: int = 0
    ) -> FlowCheckpoint:
        """
        保存检查点
        
        Args:
            session: 所属会话
            step_index: 步骤索引 (-1 表示输入数据)
            step_name: 步骤名称
            data_items: 当前步骤的输出数据
            flow_definition: 流程定义快照
            execution_config: 执行配置快照
            success_count: 成功处理数
            error_count: 错误数
            
        Returns:
            创建的 FlowCheckpoint
        """
        checkpoint_id = f"cp_{session.session_id}_{step_index + 1:03d}"
        checkpoint_dir = self.base_dir / session.session_id / f"step_{step_index + 1:03d}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存数据项
        items_metadata = []
        for i, item in enumerate(data_items):
            item_meta = self._save_item(item, checkpoint_dir, i)
            items_metadata.append(item_meta)
        
        # 保存项目元数据
        items_file = checkpoint_dir / self.ITEMS_FILE
        with open(items_file, 'w', encoding='utf-8') as f:
            json.dump(items_metadata, f, ensure_ascii=False, indent=2)
        
        # 创建检查点对象
        checkpoint = FlowCheckpoint(
            checkpoint_id=checkpoint_id,
            step_index=step_index,
            step_name=step_name,
            timestamp=time.time(),
            data_dir=str(checkpoint_dir),
            items_count=len(data_items),
            flow_definition=flow_definition,
            execution_config=execution_config,
            success_count=success_count,
            error_count=error_count
        )
        
        # 更新会话
        session.checkpoints.append(checkpoint)
        session.completed_steps = step_index + 1
        session.updated_at = time.time()
        self._save_sessions()
        
        self.logger.info(f"保存检查点: {checkpoint_id} ({step_name}, {len(data_items)} 项)")

        # 自动清理旧检查点 (改为从配置读取，默认 0 为不清理)
        keep_last = execution_config.get('checkpoint_keep_last', 0)
        self.cleanup_old_checkpoints(session, keep_last=keep_last)

        return checkpoint

    def cleanup_old_checkpoints(self, session: FlowSession, keep_last: int = 0):
        """
        清理旧检查点，防止磁盘占用过大 (异步执行)

        Args:
            session: 目标会话
            keep_last: 保留最近多少个检查点 (0 表示全部保留)
        """
        if keep_last <= 0 or not session.checkpoints or len(session.checkpoints) <= keep_last:
            return

        # 保留最后 N 个
        to_remove = session.checkpoints[:-keep_last]
        session.checkpoints = session.checkpoints[-keep_last:]

        def _async_cleanup(items_to_remove):
            removed_count = 0
            for cp in items_to_remove:
                try:
                    # 删除磁盘文件
                    cp_dir = Path(cp.data_dir).resolve()
                    if cp_dir.exists():
                        # [SECURITY] Ensure we are deleting a directory inside our base_dir
                        if self.base_dir.resolve() in cp_dir.parents:
                            shutil.rmtree(cp_dir)
                        else:
                            self.logger.warning(f"[Security] Skipped deleting unsafe path: {cp_dir}")
                    removed_count += 1
                except Exception as e:
                    self.logger.warning(f"清理检查点失败 {cp.checkpoint_id}: {e}")

            if removed_count > 0:
                self._save_sessions()
                self.logger.info(f"后台已异步清理 {removed_count} 个旧检查点 (保留最近 {keep_last} 个)")

        # 启动后台线程执行删除，避免阻塞 UI
        cleanup_thread = threading.Thread(target=_async_cleanup, args=(to_remove,), daemon=True)
        cleanup_thread.start()
    
    def _save_item(self, item: Any, checkpoint_dir: Path, index: int) -> Dict:
        """
        保存单个数据项
        
        Returns:
            该项的元数据
        """
        metadata = {
            'index': index,
            'type': type(item).__name__
        }
        
        if isinstance(item, dict):
            # 处理图片
            if 'image' in item and isinstance(item['image'], Image.Image):
                img_filename = f"{index:05d}.png"
                img_path = checkpoint_dir / img_filename
                item['image'].save(img_path, 'PNG')
                metadata['image_file'] = img_filename
            
            # 保存其他元数据
            for key, value in item.items():
                if key == 'image':
                    continue
                # 只保存可序列化的值
                if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    metadata[key] = value
                elif isinstance(value, Path):
                    metadata[key] = str(value)
        
        elif isinstance(item, Image.Image):
            img_filename = f"{index:05d}.png"
            img_path = checkpoint_dir / img_filename
            item.save(img_path, 'PNG')
            metadata['image_file'] = img_filename
            metadata['type'] = 'Image'
        
        elif isinstance(item, (str, Path)):
            metadata['path'] = str(item)
            metadata['type'] = 'path'
        
        return metadata
    
    def load_checkpoint(self, checkpoint: FlowCheckpoint) -> List[Any]:
        """
        从检查点加载数据
        
        Args:
            checkpoint: 要加载的检查点
            
        Returns:
            恢复的数据项列表
        """
        checkpoint_dir = Path(checkpoint.data_dir)
        items_file = checkpoint_dir / self.ITEMS_FILE
        
        if not items_file.exists():
            self.logger.error(f"检查点数据不存在: {items_file}")
            return []
        
        with open(items_file, 'r', encoding='utf-8') as f:
            items_metadata = json.load(f)
        
        items = []
        for meta in items_metadata:
            item = self._load_item(meta, checkpoint_dir)
            if item is not None:
                items.append(item)
        
        self.logger.info(f"从检查点加载 {len(items)} 项数据")
        return items
    
    def _load_item(self, metadata: Dict, checkpoint_dir: Path) -> Any:
        """从元数据加载单个数据项"""
        item_type = metadata.get('type', 'dict')
        
        if item_type == 'path':
            return Path(metadata.get('path', ''))
        
        # 重建字典项
        item = {}
        
        # 加载图片
        if 'image_file' in metadata:
            img_path = checkpoint_dir / metadata['image_file']
            if img_path.exists():
                item['image'] = Image.open(img_path).convert('RGB')
        
        # 恢复其他字段
        skip_keys = {'index', 'type', 'image_file'}
        for key, value in metadata.items():
            if key not in skip_keys:
                item[key] = value
        
        return item if item else None
    
    def update_session_status(
        self, 
        session: FlowSession, 
        status: SessionStatus,
        errors: Optional[List[Dict]] = None
    ):
        """更新会话状态"""
        session.status = status.value
        session.updated_at = time.time()
        if errors:
            session.errors = errors
        self._save_sessions()
    
    def get_session(self, session_id: str) -> Optional[FlowSession]:
        """获取指定会话"""
        return self.sessions.get(session_id)
    
    def list_sessions(
        self, 
        status_filter: Optional[SessionStatus] = None,
        limit: int = 50
    ) -> List[FlowSession]:
        """
        列出会话
        
        Args:
            status_filter: 可选的状态过滤
            limit: 返回数量限制
            
        Returns:
            会话列表，按更新时间倒序
        """
        sessions = list(self.sessions.values())
        
        if status_filter:
            sessions = [s for s in sessions if s.status == status_filter.value]
        
        # 按更新时间倒序
        sessions.sort(key=lambda s: s.updated_at, reverse=True)
        
        return sessions[:limit]
    
    def get_resumable_sessions(self) -> List[FlowSession]:
        """获取可恢复的会话（已暂停或失败且有检查点的会话）"""
        return [
            s for s in self.sessions.values()
            if s.status in (SessionStatus.PAUSED.value, SessionStatus.FAILED.value)
            and s.checkpoints
        ]
    
    def delete_session(self, session_id: str, delete_files: bool = True):
        """
        删除会话
        
        Args:
            session_id: 会话ID
            delete_files: 是否同时删除磁盘文件
        """
        if session_id not in self.sessions:
            return
        
        if delete_files:
            session_dir = self.base_dir / session_id
            if session_dir.exists():
                shutil.rmtree(session_dir)
        
        del self.sessions[session_id]
        self._save_sessions()
        self.logger.info(f"删除会话: {session_id}")
    
    def cleanup_old_sessions(
        self, 
        max_age_days: int = 7,
        keep_completed: bool = True
    ):
        """
        清理旧会话
        
        Args:
            max_age_days: 最大保留天数
            keep_completed: 是否保留已完成的会话
        """
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        
        to_delete = []
        for session_id, session in self.sessions.items():
            if session.updated_at < cutoff_time:
                if keep_completed and session.status == SessionStatus.COMPLETED.value:
                    continue
                to_delete.append(session_id)
        
        for session_id in to_delete:
            self.delete_session(session_id)
        
        if to_delete:
            self.logger.info(f"清理了 {len(to_delete)} 个旧会话")
    
    def get_disk_usage(self) -> Tuple[int, int]:
        """
        获取磁盘使用情况
        
        Returns:
            (总大小bytes, 会话数量)
        """
        total_size = 0
        session_count = len(self.sessions)
        
        for item in self.base_dir.rglob('*'):
            if item.is_file():
                total_size += item.stat().st_size
        
        return total_size, session_count
