"""
REapp LLM Engine | Shared Inference Service
Centrally manage model lifecycle, VRAM orchestration, and mixed provider routing.
"""

import os
import time
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, List, Any, Union
from dataclasses import dataclass, field

from .accelerator import accelerator
from .llm_client import LLM_PRESETS, LLMClient

logger = logging.getLogger("LLMEngine")

# Constants - use absolute path relative to REapp root
_PROJECT_ROOT = Path(__file__).parent.parent.parent  # h:/REapp
_env_model_dir = os.environ.get("REAPP_LLM_MODEL_DIR")
DEFAULT_MODEL_DIR = Path(_env_model_dir) if _env_model_dir else _PROJECT_ROOT / "models" / "llm"
DEFAULT_MODEL_DIR.mkdir(parents=True, exist_ok=True)

@dataclass
class EngineStatus:
    status: str = "idle"  # idle, loading, busy, error
    loaded_model: Optional[str] = None
    vram_usage_gb: float = 0.0
    error_message: Optional[str] = None
    uptime: float = 0.0

class LLMEngine:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self.llama = None  # The Llama instance from llama-cpp-python
        self.current_model_path: Optional[Path] = None
        self.current_model_name: Optional[str] = None
        self.status = EngineStatus()
        self.start_time = time.time()
        self._lock = asyncio.Lock()
        self._gpu_lock_handle = None
        self._initialized = True
        
        logger.info("[LLMEngine] Service initialized.")

    async def get_status(self) -> EngineStatus:
        """Returns the current status of the engine."""
        self.status.uptime = time.time() - self.start_time
        return self.status

    def list_local_models(self) -> List[Dict[str, Any]]:
        """Scans models/llm for GGUF files."""
        models = []
        if not DEFAULT_MODEL_DIR.exists():
            return []
            
        for file in DEFAULT_MODEL_DIR.glob("*.gguf"):
            models.append({
                "name": file.name,
                "path": str(file.absolute()),
                "size_gb": round(file.stat().st_size / (1024**3), 2)
            })
        return models

    async def load_model(self, model_name: str, n_ctx: int = 2048, n_gpu_layers: int = -1, device: str = "gpu") -> bool:
        """
        Loads a local GGUF model into VRAM or RAM.
        
        Args:
            model_name: Filename of the GGUF model.
            n_ctx: Context window size.
            n_gpu_layers: Number of layers to offload to GPU. -1 for all.
            device: 'gpu' (default) or 'cpu'. If 'cpu', n_gpu_layers is forced to 0.
        """
        async with self._lock:
            try:
                model_path = DEFAULT_MODEL_DIR / model_name
                if not model_path.exists():
                    raise FileNotFoundError(f"Model file not found: {model_name}")

                if self.current_model_path == model_path and self.llama is not None:
                    # Check if device config changed, if so we might need reload.
                    # For simplicity, if already loaded, we assume it's fine unless explicitly unloaded.
                    return True 

                # Unload current first
                await self.unload_model()

                # Handle Device Selection
                use_gpu = (device.lower() == "gpu")
                if not use_gpu:
                    n_gpu_layers = 0
                    logger.info("[LLMEngine] CPU Mode selected. Skipping GPU lock.")
                else:
                    self.status.status = "loading"
                    # Acquire GPU Lock only if using GPU
                    self._gpu_lock_handle = accelerator.lock_gpu("LLMEngine", f"Loading model {model_name}")
                
                # Lazy import Llama
                try:
                    from llama_cpp import Llama
                except ImportError:
                    # Fallback: Inject Isolated Environment (deployment venv) if not found
                    # This handles the case where backend is running outside the deployed venv
                    import sys
                    import json
                    try:
                        logger.info("[LLMEngine] llama_cpp not found. Attempting to inject isolated environment path...")
                        _local_dir = _PROJECT_ROOT / "backend" # Based on new structure
                        _pointer = _local_dir / "active_env.json"
                        if _pointer.exists():
                            with open(_pointer, "r") as f:
                                _active_path = json.load(f).get("active_path", "venv")
                            
                            _site_packages = _local_dir / _active_path / "Lib" / "site-packages"
                            if not _site_packages.exists():
                                 _site_packages = _local_dir / _active_path / "lib" / "site-packages"
                            
                            if _site_packages.exists() and str(_site_packages) not in sys.path:
                                logger.info(f"[LLMEngine] Injecting site-packages: {_site_packages}")
                                sys.path.append(str(_site_packages))
                                from llama_cpp import Llama 
                            else:
                                raise
                        else:
                            raise
                    except Exception as e:
                        logger.error(f"[LLMEngine] Failed to inject isolated env: {e}")
                        raise ImportError("llama-cpp-python not installed or isolated env unreachable")
                except ImportError:
                    self.status.status = "error"
                    self.status.error_message = "llama-cpp-python not installed."
                    logger.error("[LLMEngine] llama-cpp-python missing.")
                    return False

                logger.info(f"[LLMEngine] Loading {model_name} on {device.upper()} (layers={n_gpu_layers})...")
                
                # Start loading
                # Run in threadpool to avoid blocking async loop during heavy load
                from fastapi.concurrency import run_in_threadpool
                
                def _load():
                    return Llama(
                        model_path=str(model_path),
                        n_ctx=n_ctx,
                        n_gpu_layers=n_gpu_layers,
                        verbose=False
                    )
                
                self.llama = await run_in_threadpool(_load)
                
                self.current_model_path = model_path
                self.current_model_name = model_name
                self.status.status = "idle"
                self.status.loaded_model = model_name
                
                # Update status
                if use_gpu:
                    info = accelerator.get_hardware_info()
                    self.status.vram_usage_gb = info.vram_total_gb - info.vram_free_gb
                else:
                    self.status.vram_usage_gb = 0.0 # CPU mode implies 0 VRAM managed by us
                
                logger.info(f"[LLMEngine] Model {model_name} loaded successfully.")
                return True

            except Exception as e:
                logger.error(f"[LLMEngine] Load failed: {e}")
                self.status.status = "error"
                self.status.error_message = str(e)
                await self.unload_model()
                return False

    async def unload_model(self):
        """Unload local model and release GPU lock."""
        if self.llama:
            del self.llama
            self.llama = None
            import gc
            gc.collect()
            
        if self._gpu_lock_handle:
            self._gpu_lock_handle.release()
            self._gpu_lock_handle = None
            
        self.current_model_path = None
        self.current_model_name = None
        self.status.loaded_model = None
        self.status.vram_usage_gb = 0.0
        self.status.status = "idle"
        logger.info("[LLMEngine] Model unloaded.")

    async def chat(self, messages: List[Dict[str, str]], config: Dict[str, Any] = None, stream: bool = False):
        """
        Process a chat request.
        
        Args:
            messages: List of message dicts (role, content).
            config: Inference config (temp, top_p, etc).
            stream: If True, returns an async generator yielding tokens.
        """
        config = config or {}
        provider = config.get("provider", "local")
        
        if provider == "local":
            # Lazy Loading Logic
            if not self.llama:
                model_name = config.get("model_name") or config.get("model")
                device = config.get("device", "gpu")
                
                if model_name:
                    logger.info(f"[LLMEngine] Lazy loading model: {model_name} (Device: {device})")
                    try:
                        # Auto-load the model
                        await self.load_model(model_name, device=device)
                    except Exception as e:
                        err_msg = f"Error: Auto-load failed: {str(e)}"
                        logger.error(f"[LLMEngine] {err_msg}")
                        if stream:
                             async def _err_gen(): yield err_msg
                             return _err_gen()
                        return err_msg

            if not self.llama:
                msg = "Error: Local model not loaded. Please select a model in settings."
                if stream:
                    async def _err_gen(): yield msg
                    return _err_gen()
                return msg
                
            return await self._chat_local(messages, config, stream)
        else:
            from .llm_provider_bridge import chat_online, stream_online

            if stream:
                return stream_online(messages, config)
            try:
                return await chat_online(messages, config)
            except Exception as e:
                logger.error(f"[LLMEngine] Online chat error: {e}")
                return f"Error: {e}"

    async def _chat_local(self, messages: List[Dict[str, str]], config: Dict[str, Any], stream: bool):
        """Internal handler for local Llama inference using create_chat_completion."""
        if not self.llama:
            return "Error: Llama instance is None."
            
        try:
            self.status.status = "busy"
            
            # Run inference in threadpool because Llama.create_chat_completion is blocking
            from fastapi.concurrency import run_in_threadpool
            
            kwargs = {
                "messages": messages,
                "max_tokens": config.get("max_tokens", 2048),
                "temperature": config.get("temperature", 0.7),
                "top_p": config.get("top_p", 0.9),
                "stream": stream
            }

            if stream:
                # For streaming, create_chat_completion returns a generator immediately
                # But we should still probably offload the initial processing?
                # Actually, llama-cpp-python release GIL well? Let's assume yes.
                response_iter = self.llama.create_chat_completion(**kwargs)
                
                async def _async_generator():
                    try:
                        # Iterate over the sync generator
                        for chunk in response_iter:
                            await asyncio.sleep(0) 
                            
                            if "choices" in chunk and len(chunk["choices"]) > 0:
                                delta = chunk["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                    except Exception as e:
                        logger.error(f"[LLMEngine] Streaming error: {e}")
                        yield f"Error: {str(e)}"
                    finally:
                        self.status.status = "idle"
                        
                return _async_generator()
                
            else:
                # Non-streaming
                response = await run_in_threadpool(self.llama.create_chat_completion, **kwargs)
                text = response["choices"][0]["message"]["content"]
                return text

        except Exception as e:
            logger.error(f"[LLMEngine] Local chat error: {e}")
            msg = f"Error: {e}"
            if stream:
                async def _err(): yield msg
                return _err()
            return msg
        finally:
            if not stream:
                self.status.status = "idle"

# Global instance
llm_engine = LLMEngine()
