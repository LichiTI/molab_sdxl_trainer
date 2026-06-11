from pathlib import Path
from typing import Dict, Any, Optional, Callable, Tuple, List
from dataclasses import dataclass
try:
    from core.job_manager import Job, JobType, JobStatus
    from core.locator import Locator
except ImportError:
    # Allow running in isolation for testing
    Job = None
    Locator = None

@dataclass
class TrainingResult:
    success: bool
    output_dir: str
    model_path: str = ''
    total_steps: int = 0
    final_loss: float = 0.0
    error_message: str = ''

class TrainingService:

    @staticmethod
    def validate_config(config) -> Tuple[bool, List[str], List[str]]:
        if config is None:
            return (False, ['配置对象为空'], [])
        if hasattr(config, 'validate'):
            return config.validate()
        errors = []
        warnings = []
        if not getattr(config, 'pretrained_model_name_or_path', ''):
            errors.append('未指定底模路径')
        if not getattr(config, 'train_data_dir', ''):
            errors.append('未指定训练数据目录')
        if not getattr(config, 'output_dir', ''):
            errors.append('未指定输出目录')
        return (len(errors) == 0, errors, warnings)

    @staticmethod
    def submit_training_job(config, lora_scripts_path: str=None, custom_params: List[str]=None, gpu_limit: int=100) -> str:
        is_valid, errors, warnings = TrainingService.validate_config(config)
        if not is_valid:
            raise ValueError(f"配置验证失败: {'; '.join(errors)}")
        total_steps = getattr(config, 'max_train_steps', 0)
        if total_steps == 0:
            total_steps = getattr(config, 'max_train_epochs', 10) * 100
        output_name = getattr(config, 'output_name', 'lora_model')
        job = Job(
            type=JobType.TRAINING, 
            name=f'训练: {output_name}', 
            total_items=total_steps, 
            metadata={
                'config': config, 
                'lora_scripts_path': lora_scripts_path, 
                'custom_params': custom_params or [], 
                'gpu_limit': gpu_limit, 
                'output_dir': getattr(config, 'output_dir', ''), 
                'output_name': output_name
            }
        )
        try:
            job_id = Locator.jobs.submit(job, worker_func=TrainingService._run_training, args=(config, lora_scripts_path, custom_params or [], gpu_limit))
            return job_id
        except Exception as e:
            raise RuntimeError(f"Failed to submit training job: {e}")

    @staticmethod
    def _run_training(config, lora_scripts_path: str, custom_params: List[str], gpu_limit: int, progress_callback: Callable[[int, int], None]=None, cancel_check: Callable[[], bool]=None) -> TrainingResult:
        """Native-only path. Legacy backend has been removed."""
        return TrainingResult(
            success=False,
            output_dir=getattr(config, 'output_dir', ''),
            error_message='Legacy training path has been removed. Use trainer_engine=lulynx with the native entry_train.py worker.',
        )

    @staticmethod
    def estimate_vram(config) -> Dict[str, Any]:
        from core.vram_estimator import VRAMEstimator
        return VRAMEstimator.estimate(model_type=getattr(config, 'model_type', 'sdxl'), network_dim=getattr(config, 'network_dim', 32), network_alpha=getattr(config, 'network_alpha', 16), batch_size=getattr(config, 'train_batch_size', 1), resolution=getattr(config, 'resolution', 1024), gradient_checkpointing=getattr(config, 'gradient_checkpointing', True), mixed_precision=getattr(config, 'mixed_precision', 'bf16'))

    @staticmethod
    def list_presets() -> List[str]:
        return []

    @staticmethod
    def load_preset(name: str):
        return None

def submit_training(config, **kwargs) -> str:
    return TrainingService.submit_training_job(config, **kwargs)