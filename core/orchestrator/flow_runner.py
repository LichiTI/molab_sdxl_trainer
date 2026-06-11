import time
import logging
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from .base_node import BaseNode, NodeResult, NodeConfig
from .io_types import DataType, IOSchema, IOField
from .execution_engine import ExecutionMode, ExecutionConfig, MemoryManager, PreloadPool, TempFileManager, stream_images
from .checkpoint_manager import CheckpointManager, FlowCheckpoint, FlowSession, SessionStatus
try:
    from managers import session_logger, TaskHistoryDB, model_manager
except ImportError:
    session_logger = None
    TaskHistoryDB = None
    model_manager = None

@dataclass
class FlowStep:
    node: BaseNode
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

@dataclass
class FlowResult:
    success: bool
    total_items: int = 0
    processed_items: int = 0
    success_count: int = 0
    error_count: int = 0
    skipped_count: int = 0
    duration_seconds: float = 0
    errors: List[Dict] = field(default_factory=list)
    output_data: List[Any] = field(default_factory=list)

class FlowRunner:

    def __init__(self, execution_config: Optional[ExecutionConfig]=None, checkpoint_dir: Optional[Path]=None):
        self.logger = logging.getLogger('FlowRunner')
        self.steps: List[FlowStep] = []
        self.steps: List[FlowStep] = []
        self._lock = __import__('threading').Lock()
        with self._lock:
            self._is_running = False
        self._should_stop = False
        self.execution_config = execution_config or ExecutionConfig()
        self.memory_manager = MemoryManager(self.execution_config)
        self.on_progress: Optional[Callable[[int, int, str], None]] = None
        self.on_step_start: Optional[Callable[[int, str], None]] = None
        self.on_step_complete: Optional[Callable[[int, str, int, int], None]] = None
        
        # 检查点支持
        self.enable_checkpoints = True  # 是否启用检查点
        self.checkpoint_manager: Optional[CheckpointManager] = None
        self.current_session: Optional[FlowSession] = None
        if checkpoint_dir:
            self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # 检查点回调
        self.on_checkpoint_saved: Optional[Callable[[FlowCheckpoint], None]] = None

    def add_step(self, node: BaseNode, config: Optional[Dict]=None, enabled: bool=True):
        step = FlowStep(node=node, config=config or {}, enabled=enabled)
        self.steps.append(step)
        if session_logger:
            session_logger.log_flow_change('添加节点', node.get_name())

    def remove_step(self, index: int):
        if 0 <= index < len(self.steps):
            step = self.steps.pop(index)
            if session_logger:
                session_logger.log_flow_change('移除节点', step.node.get_name())

    def move_step(self, from_index: int, to_index: int):
        if 0 <= from_index < len(self.steps) and 0 <= to_index < len(self.steps):
            step = self.steps.pop(from_index)
            self.steps.insert(to_index, step)

    def clear(self):
        self.steps.clear()

    def get_flow_definition(self) -> List[Dict]:
        return [{'node_type': step.node.__class__.__name__, 'node_name': step.node.get_name(), 'config': step.config, 'enabled': step.enabled} for step in self.steps]

    def validate_flow(self) -> List[str]:
        """
        Perform a static dry-run validation of the flow.
        Checks if the input requirements of each node are satisfied by the outputs of previous nodes.
        """
        errors = []
        
        # Initial Schema (Standard Input)
        current_schema = IOSchema(fields=[
            IOField("image", DataType.IMAGE, required=True),
            IOField("filename", DataType.STRING, required=True),
            IOField("source_path", DataType.STRING, required=True),
            IOField("metadata", DataType.DICT, required=False),
        ])

        for index, step in enumerate(self.steps):
            if not step.enabled:
                continue
                
            node = step.node
            node_name = f"{index+1}. {node.get_name()}"
            input_schema = node.get_input_schema()
            
            # Check Compatibility
            for req_field in input_schema.fields:
                if not req_field.required:
                    continue
                
                # Special case: 'data' accepts everything
                if req_field.name == "data" and req_field.type == DataType.ANY:
                    continue

                # Find matching field in current availability
                available_field = current_schema.get_field(req_field.name)
                
                if not available_field:
                    # Field missing entirely
                    errors.append(f"[{node_name}] 缺少必要输入: '{req_field.name}' (需要 {req_field.type.value})")
                elif available_field.type != DataType.ANY and req_field.type != DataType.ANY and available_field.type != req_field.type:
                    # Type mismatch
                    errors.append(f"[{node_name}] 类型不匹配: '{req_field.name}' 需要 {req_field.type.value}, 但上游提供 {available_field.type.value}")
            
            # Update Schema for next step
            # Generally, nodes modify the flow, so we take their output schema.
            # However, if a node's output schema is "Partial", we might need to merge.
            # For strictness, let's assume the node output DEFINES the new state.
            # If a node passes through data, it should declare it in output schema.
            output_schema = node.get_output_schema()
            
            # Merging logic for "Update" style nodes (simplified: if output has fields, they override/add)
            # But strictly speaking, if we replace the data object, we replace the schema.
            # Given FlowRunner's 'replace' behavior in batch mode, we replace.
            current_schema = output_schema

        return errors


    def run(self, input_data: List[Any], task_name: str='未命名任务') -> FlowResult:
        with self._lock:
            if self._is_running:
                raise RuntimeError('Flow is already running')
            self._is_running = True
            
        self._should_stop = False
        start_time = time.time()
        
        # TaskHistoryDB Support
        task_id = None
        if TaskHistoryDB:
            try:
                history_db = TaskHistoryDB()
                config_snapshot = {step.node.get_name(): step.config for step in self.steps if step.enabled}
                task_id = history_db.create_task(task_name, config_snapshot=config_snapshot, flow_definition={'steps': self.get_flow_definition()})
            except Exception as e:
                self.logger.warning(f'Failed to create task history: {e}')
                
        if session_logger:
            session_logger.log_task_start(task_name, len(input_data))
            
        result = FlowResult(success=True, total_items=len(input_data))
        enabled_steps = [s for s in self.steps if s.enabled]
        
        try:
            self._execute_flow_loop(
                enabled_steps=enabled_steps,
                start_step_index=0,
                input_data=input_data,
                result=result,
                task_id=task_id,
                session=None # No session for basic run
            )
        except Exception as e:
            import traceback
            self.logger.error(f'Flow execution error: {e}')
            traceback.print_exc()
            result.success = False
            result.errors.append({'message': str(e)})
        finally:
            self._finalize_run(result, start_time, task_id, None)
            
        return result

    def run_with_checkpoints(
        self, 
        input_data: List[Any], 
        task_name: str = '未命名任务',
        resume_from: Optional[FlowCheckpoint] = None
    ) -> FlowResult:
        with self._lock:
            if self._is_running:
                raise RuntimeError('Flow is already running')
            self._is_running = True
            
        if not self.steps:
            with self._lock: self._is_running = False
            return FlowResult(success=False, errors=[{'message': '流程为空'}])
        
        # Ensure CheckpointManager
        if self.enable_checkpoints and not self.checkpoint_manager:
            self.checkpoint_manager = CheckpointManager(Path('./checkpoints'))
        
        self._should_stop = False
        start_time = time.time()
        
        enabled_steps = [s for s in self.steps if s.enabled]
        
        # Determine start point
        if resume_from:
            start_step_index = resume_from.step_index + 1
            current_data = self.checkpoint_manager.load_checkpoint(resume_from)
            self.current_session = self._find_session_for_checkpoint(resume_from)
            self.logger.info(f"从检查点恢复: 步骤 {start_step_index} ({resume_from.step_name} 之后)")
            if self.current_session:
                self.checkpoint_manager.update_session_status(self.current_session, SessionStatus.RUNNING)
        else:
            start_step_index = 0
            current_data = input_data
            if self.enable_checkpoints and self.checkpoint_manager:
                self.current_session = self.checkpoint_manager.create_session(
                    task_name, total_steps=len(enabled_steps), total_items=len(input_data)
                )
                self.checkpoint_manager.save_checkpoint(
                    self.current_session, -1, "输入数据", input_data,
                    self.get_flow_definition(), self.execution_config.to_dict(), len(input_data)
                )

        result = FlowResult(success=True, total_items=len(input_data) if not resume_from else len(current_data))
        
        try:
            self._execute_flow_loop(
                enabled_steps=enabled_steps,
                start_step_index=start_step_index,
                input_data=current_data,
                result=result,
                task_id=None, # Checkpoints mode uses session, not TaskHistoryDB directly
                session=self.current_session
            )
        except Exception as e:
            import traceback
            self.logger.error(f'Flow execution error: {e}')
            traceback.print_exc()
            result.success = False
            result.errors.append({'message': str(e)})
            if self.current_session and self.checkpoint_manager:
                self.checkpoint_manager.update_session_status(self.current_session, SessionStatus.FAILED, errors=result.errors)
        finally:
            self._finalize_run(result, start_time, None, self.current_session)
            
        return result

    def _execute_flow_loop(
        self,
        enabled_steps: List[FlowStep],
        start_step_index: int,
        input_data: List[Any],
        result: FlowResult,
        task_id: Optional[str],
        session: Optional[FlowSession]
    ):
        """Shared execution logic for runs"""
        current_data = input_data
        
        for step_idx in range(start_step_index, len(enabled_steps)):
            if self._should_stop:
                self.logger.info('Flow stopped by user')
                if session and self.checkpoint_manager:
                    self.checkpoint_manager.update_session_status(session, SessionStatus.PAUSED)
                break
                
            step = enabled_steps[step_idx]
            node = step.node
            node_name = node.get_name()
            
            self.logger.info(f'Step {step_idx + 1}: {node_name}')
            if self.on_step_start:
                self.on_step_start(step_idx, node_name)
                
            node.configure(step.config)
            if not node._is_loaded:
                node.on_load()
                
            # Progress Callback
            def progress_cb(current, total, msg):
                if self.on_progress:
                    steps_done = step_idx
                    step_progress = current / total if total > 0 else 0
                    global_progress = (steps_done + step_progress) / len(enabled_steps)
                    self.on_progress(int(global_progress * 100), 100, f'[{node_name}] {msg}')
            
            # Execute
            node_results = node.process_batch(current_data, progress_cb)
            
            # Collect results
            next_data = []
            step_success = 0
            step_errors = 0
            
            for i, node_result in enumerate(node_results):
                original_item = current_data[i]
                if node_result.success:
                    step_success += 1
                    if node.get_node_type() == BaseNode.TYPE_FILTER:
                        if node_result.data:
                            next_data.append(original_item)
                    else:
                        next_data.append(node_result.data if node_result.data else original_item)
                else:
                    step_errors += 1
                    result.errors.append({'step': step_idx, 'node': node_name, 'item_index': i, 'error': node_result.error})

            if self.on_step_complete:
                self.on_step_complete(step_idx, node_name, step_success, step_errors)
                
            self.logger.info(f'Step {step_idx + 1} complete: {step_success} success, {step_errors} errors')
            current_data = next_data
            result.processed_items = len(next_data)
            
            # Save Checkpoint
            if session and self.checkpoint_manager and self.enable_checkpoints:
                cp = self.checkpoint_manager.save_checkpoint(
                    session, step_idx, node_name, current_data,
                    self.get_flow_definition(), self.execution_config.to_dict(),
                    step_success, step_errors
                )
                if self.on_checkpoint_saved:
                    self.on_checkpoint_saved(cp)

            if not current_data:
                self.logger.info('No data remaining, stopping flow')
                break
                
        result.output_data = current_data

    def _finalize_run(self, result: FlowResult, start_time: float, task_id: Optional[str], session: Optional[FlowSession]):
        """Shared cleanup logic"""
        with self._lock:
            self._is_running = False
            
        result.duration_seconds = time.time() - start_time
        result.success_count = len(result.output_data)
        result.error_count = len(result.errors)
        # Skipped = total - success - error (approximate for filter nodes)
        result.skipped_count = result.total_items - result.success_count - result.error_count
        
        if task_id and TaskHistoryDB:
            try:
                TaskHistoryDB().complete_task(task_id, success_count=result.success_count, error_count=result.error_count, skipped_count=result.skipped_count, error_log=result.errors[:100])
            except Exception as e:
                self.logger.warning(f'Failed to update task history: {e}')
                
        if session and self.checkpoint_manager:
            if not self._should_stop and result.success:
                 self.checkpoint_manager.update_session_status(session, SessionStatus.COMPLETED, errors=result.errors[:100])
                 
        if session_logger:
            session_logger.log_task_complete(result.success_count, result.error_count, result.skipped_count)
    
    def _find_session_for_checkpoint(self, checkpoint: FlowCheckpoint) -> Optional[FlowSession]:
        """根据检查点找到对应的会话"""
        if not self.checkpoint_manager:
            return None
        
        for session in self.checkpoint_manager.sessions.values():
            for cp in session.checkpoints:
                if cp.checkpoint_id == checkpoint.checkpoint_id:
                    return session
        return None
    
    def get_resumable_sessions(self) -> List[FlowSession]:
        """获取可恢复的会话列表"""
        if not self.checkpoint_manager:
            return []
        return self.checkpoint_manager.get_resumable_sessions()
    
    def resume_session(self, session_id: str, from_step: Optional[int] = None) -> FlowResult:
        """
        恢复指定会话的执行
        
        Args:
            session_id: 会话ID
            from_step: 可选，从指定步骤恢复（默认从最后检查点）
            
        Returns:
            FlowResult: 执行结果
        """
        if not self.checkpoint_manager:
            return FlowResult(success=False, errors=[{'message': '检查点管理器未初始化'}])
        
        session = self.checkpoint_manager.get_session(session_id)
        if not session:
            return FlowResult(success=False, errors=[{'message': f'会话不存在: {session_id}'}])
        
        # 确定恢复点
        if from_step is not None:
            checkpoint = session.get_checkpoint_at_step(from_step)
        else:
            checkpoint = session.get_latest_checkpoint()
        
        if not checkpoint:
            return FlowResult(success=False, errors=[{'message': '没有可用的检查点'}])
        
        self.logger.info(f"恢复会话 {session_id}, 从步骤 {checkpoint.step_index} ({checkpoint.step_name}) 之后")
        
        return self.run_with_checkpoints(
            input_data=[],  # 将从检查点加载
            task_name=session.task_name,
            resume_from=checkpoint
        )
    
    def list_session_checkpoints(self, session_id: str) -> List[Dict]:
        """
        列出指定会话的所有检查点
        
        Returns:
            检查点信息列表
        """
        if not self.checkpoint_manager:
            return []
        
        session = self.checkpoint_manager.get_session(session_id)
        if not session:
            return []
        
        return [
            {
                'checkpoint_id': cp.checkpoint_id,
                'step_index': cp.step_index,
                'step_name': cp.step_name,
                'items_count': cp.items_count,
                'success_count': cp.success_count,
                'error_count': cp.error_count,
                'timestamp': cp.timestamp
            }
            for cp in session.checkpoints
        ]