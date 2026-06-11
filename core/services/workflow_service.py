from typing import List, Dict, Any, Optional, Callable, Tuple
from core.controller import WorkflowController
from core.controller import WorkflowController
from config.settings import ConfigManager
import logging
import traceback

logger = logging.getLogger(__name__)

class WorkflowService:

    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager

    def execute_workflow(self, steps: List[Dict[str, Any]], input_dir: str, output_dir: str, progress_callback: Optional[Callable[[float], None]] = None) -> bool:
        try:
            controller = WorkflowController(progress_callback=progress_callback)
            success = controller.run(input_dir, output_dir, steps)
            return bool(success)
        except Exception as e:
            logger.error(f'[WorkflowService] Error executing workflow: {e}')
            logger.error(traceback.format_exc())
            return False

    def validate_workflow(self, steps: List[Dict[str, Any]], input_dir: str, output_dir: str) -> Tuple[bool, str]:
        if not input_dir or not output_dir:
            return (False, 'Input and output directories are required')
        if not steps or len(steps) == 0:
            return (False, 'At least one processing step is required')
        for step in steps:
            if 'type' not in step:
                return (False, f'Invalid step configuration: missing type in {step}')
        return (True, '')