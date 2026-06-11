import gc
import logging
import threading
import time
from typing import Dict, Any, Callable, Optional
from collections import OrderedDict
try:
    import torch
except ImportError:
    torch = None

class ModelManager:
    _instance = None
    _lock = threading.RLock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ModelManager, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, max_cache_size: int = 2):
        if self._initialized:
            return
        self.logger = logging.getLogger('ModelManager')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        self.models: OrderedDict[str, Any] = OrderedDict()
        self.max_cache_size = max_cache_size
        self.device = self._detect_device()
        self._initialized = True
        self.logger.info(f'Initialized ModelManager on device: {self.device}')

    def _detect_device(self) -> str:
        if torch and torch.cuda.is_available():
            return 'cuda'
        return 'cpu'

    def get_vram_info(self) -> Dict[str, float]:
        info = {'total': 0, 'used': 0, 'free': 0, 'percent': 0}
        if self.device == 'cuda' and torch:
            try:
                device_id = torch.cuda.current_device()
                props = torch.cuda.get_device_properties(device_id)
                total = props.total_memory
                allocated = torch.cuda.memory_allocated(device_id)
                cached = torch.cuda.memory_reserved(device_id)
                used = allocated + cached
                info['total'] = total / 1024 ** 3
                info['used'] = used / 1024 ** 3
                info['free'] = (total - used) / 1024 ** 3
                info['percent'] = used / total * 100
            except RuntimeError as e:
                self.logger.warning(f'Failed to get VRAM info: {e}')
        return info

    def load_model(self, key: str, loader_func: Callable[[], Any], force_reload: bool=False, file_path: Optional[str]=None) -> Any:
        with self._lock:
            if key in self.models and (not force_reload):
                self.models.move_to_end(key)
                self.logger.info(f'Model cache hit: {key}')
                return self.models[key]
            
            # [SECURITY] Validate file path if provided
            if file_path:
                try:
                    from core.security import validate_path
                    validate_path(file_path, must_exist=True)
                except Exception as e:
                    self.logger.error(f'Security check failed for model {key}: {e}')
                    raise
            
            self.logger.info(f'Loading model: {key}...')
            while len(self.models) >= self.max_cache_size:
                lru_key, lru_model = self.models.popitem(last=False)
                self._unload_model_instance(lru_key, lru_model)
            if key in self.models:
                self.unload_model(key)
            try:
                start_time = time.time()
                model = loader_func()
                if hasattr(model, 'to') and callable(model.to) and (self.device == 'cuda'):
                    try:
                        model = model.to(self.device)
                    except RuntimeError as e:
                        self.logger.warning(f'Could not move model {key} to {self.device}: {e}')
                self.models[key] = model
                elapsed = time.time() - start_time
                self.logger.info(f'Model {key} loaded successfully in {elapsed:.2f}s')
                vram = self.get_vram_info()
                if vram['total'] > 0:
                    self.logger.info(f"VRAM: {vram['used']:.1f}/{vram['total']:.1f} GB ({vram['percent']:.1f}%)")
                return model
            except Exception as e:
                self.logger.error(f'Failed to load model {key}: {e}')
                raise e

    def unload_model(self, key: str):
        with self._lock:
            if key in self.models:
                model = self.models.pop(key)
                self._unload_model_instance(key, model)

    def _unload_model_instance(self, key: str, model: Any):
        self.logger.info(f'Unloading model: {key}')
        if hasattr(model, 'cpu') and callable(model.cpu):
            try:
                model.cpu()
            except (AttributeError, RuntimeError):
                pass
        del model
        gc.collect()
        if self.device == 'cuda' and torch:
            torch.cuda.empty_cache()
        self.logger.info(f'Model {key} unloaded. VRAM cleared.')

    def unload_all(self):
        with self._lock:
            keys = list(self.models.keys())
            for key in keys:
                self.unload_model(key)
model_manager = ModelManager()