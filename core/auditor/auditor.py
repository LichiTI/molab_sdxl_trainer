import json
import threading
from pathlib import Path
from datetime import datetime
from queue import Queue, Empty
from typing import Optional, Dict, Any, List, Callable
import logging
from logging.handlers import RotatingFileHandler

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .types import AuditConfig, AuditMode, AuditMetrics
from .watchdog import HardwareWatchdog
from .engine import MetricEngine
from ..constants import PROJECTION_SEED

def stateless_projection(param_index: int, dim: int, k: int) -> Optional['torch.Tensor']:
    """
    V10.0: 零内存投影生成器
    
    使用固定种子动态生成投影向量，不占用长期内存
    每次调用根据 param_index 生成相同的投影矩阵
    """
    if not HAS_TORCH:
        return None
    
    generator = torch.Generator()
    generator.manual_seed(PROJECTION_SEED + param_index)
    try:
        return torch.randn((dim, k), generator=generator, dtype=torch.float32)
    except Exception:
        return None

class LoRAAuditor:
    """
    LoRA 审计器主类
    """
    
    def __init__(
        self,
        output_file: str = "./logs/audit.jsonl",
        config: Optional[AuditConfig] = None,
        on_data: Optional[Callable[[Dict], None]] = None
    ):
        self.config = config or AuditConfig()
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 日志轮转 (Audit Fix)
        # Log Rotation for audit.jsonl to prevent infinite growth
        # 日志轮转 (Audit Fix)
        # Log Rotation for audit.jsonl to prevent infinite growth
        self.logger = logging.getLogger("LoRAAuditor_JSONL")
        self.console_logger = logging.getLogger("LoRAAuditor")
        self.logger.setLevel(logging.INFO)
        # 清除现有 handler 避免重复
        self.logger.handlers = []
        
        handler = RotatingFileHandler(
            self.output_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        # 仅记录 message 本身 (JSON string)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)
        self.logger.propagate = False # 不向上传播

        self.on_data = on_data
        
        # 组件
        self.watchdog = HardwareWatchdog()
        self.engine = MetricEngine(self.config)
        
        # 状态
        self._current_mode = AuditMode.PRO
        self._is_sampling = False
        self._step_count = 0
        
        # 基础指标
        self._ema_loss: Optional[float] = None
        self._prev_ema_loss: Optional[float] = None
        self._ema_alpha = 0.1
        
        # 缓存
        self._cached_topo_metrics: Dict[str, Dict] = {} # layer_name -> metrics
        self._last_report: Optional[Dict] = None
        
        # 异步队列
        self._metrics_queue: Queue = Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = True
        self._cache_lock = threading.Lock()
        
        # 启动工作线程
        self._start_worker()
        
        # 输出文件 (Replaced by logging with rotation)
        # self._file = open(self.output_file, 'a', encoding='utf-8')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
    
    def _start_worker(self):
        """启动后台工作线程"""
        def worker():
            while self._running:
                try:
                    task = self._metrics_queue.get(timeout=1.0)
                    if task is None:
                        break
                    # 处理异步任务
                    task_type, task_data = task
                    if task_type == "topology":
                        self._process_topology_async(task_data)
                except Empty:
                    continue
                except Exception as e:
                    self.console_logger.error(f"[LoRAAuditor] Worker thread error: {e}")
                    continue
        
        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()
    
    def _process_topology_async(self, data):
        try:
            weight = data["weight"]
            mode = data["mode"]
            topo = self.engine.compute_topology(weight, mode)
            dead_rate = self.engine.compute_dead_neuron_rate(weight, mode)
            rms = self.engine.compute_rms(weight)
            
            # 计算缩放后的 RMS (结合 Alpha/Rank)
            # data 中已包含 alpha 和 rank
            alpha = data.get("alpha", 1.0)
            rank = data.get("rank", 1.0)
            rms_scaled = rms * (alpha / rank) if rank > 0 else rms
            
            with self._cache_lock:
                # Auto cleanup cache if it grows too large (prevent memory leak)
                if len(self._cached_topo_metrics) > 1000:
                    self._cached_topo_metrics.clear()

                self._cached_topo_metrics[data["name"]] = {
                    **topo,
                    "dead_neuron_rate": dead_rate,
                    "rms": rms,
                    "rms_scaled": round(rms_scaled, 6)
                }
        except Exception as e:
            self.console_logger.error(f"[LoRAAuditor] Async topology error: {e}")
    
    def set_sampling(self, is_sampling: bool):
        self._is_sampling = is_sampling
    
    def step(
        self,
        step: int,
        total_steps: int,
        epoch: int,
        loss: float,
        lr_unet: float = 0.0,
        lr_te: float = 0.0,
        optimizer=None,
        network=None,
        noise_pred: Optional['torch.Tensor'] = None,
        vram_gb: float = 0.0,
        power_w: float = 0.0,
        throughput: float = 0.0
    ) -> Dict[str, Any]:
        self._step_count = step
        self._current_mode = self.watchdog.check_policy(self._is_sampling)

        self._prev_ema_loss = self._ema_loss
        if self._ema_loss is None:
            self._ema_loss = loss
        else:
            self._ema_loss = self._ema_alpha * loss + (1 - self._ema_alpha) * self._ema_loss
        
        metrics = AuditMetrics(
            mode=self._current_mode.value,
            svd_algorithm=self.config.svd_algorithm.value
        )
        
        if self._current_mode in [AuditMode.STOP, AuditMode.SUSPEND]:
            return self._emit_record(step, total_steps, epoch, loss, lr_unet, lr_te, metrics, vram_gb, power_w, throughput)
        
        interval = self.config.sample_interval_pro if self._current_mode == AuditMode.PRO else self.config.sample_interval_lite
        should_sample = (step % interval == 0)
        
        if noise_pred is not None:
            metrics.noise_pred_std = self.engine.compute_noise_pred_std(noise_pred)
        
        if optimizer is not None:
            metrics.update_ratio = self.engine.compute_update_ratio(optimizer)
        
        if should_sample and self.config.advanced_stats_enabled:
            if network is not None and HAS_TORCH:
                if hasattr(network, "named_modules"):
                    module_iter = network.named_modules()
                elif hasattr(network, "injected_layers") and isinstance(network.injected_layers, dict):
                    module_iter = network.injected_layers.items()
                else:
                    module_iter = []
                for name, module in module_iter:
                    if not (hasattr(module, 'lora_down') and hasattr(module, 'lora_up')):
                        module = getattr(module, "lora", module)
                    if hasattr(module, 'lora_down') and hasattr(module, 'lora_up'):
                        down_weight = module.lora_down.weight
                        if down_weight.dim() >= 2:
                            weight_clone = down_weight.detach().clone().cpu()
                            data = {
                                "name": name, # Add name to data for async processing
                                "weight": weight_clone,
                                "mode": self._current_mode,
                                "alpha": getattr(module, 'alpha', 1.0),
                                "rank": getattr(module, 'rank', 1.0),
                                "step": step
                            }
                            self._metrics_queue.put(("topology", data))
                            
                            # 回调
                            if self.on_data:
                                pass # The instruction did not specify what to pass here.
                            
                            if name in self._cached_topo_metrics:
                                # 为该层附加缓存的指标
                                cached = self._cached_topo_metrics[name]
                                # 这里我们可以选择将指标存入 record，但目前 Record 是全局的
                                # 为了 SmartRank，我们需要在 report 里保留完整的 layers 数据
                                pass

        from .icu_health import compute_icu_score
        metrics.icu_score = compute_icu_score(metrics, self._ema_loss, self._prev_ema_loss)

        return self._emit_record(step, total_steps, epoch, loss, lr_unet, lr_te, metrics, vram_gb, power_w, throughput)
    
    def _emit_record(
        self,
        step: int,
        total_steps: int,
        epoch: int,
        loss: float,
        lr_unet: float,
        lr_te: float,
        metrics: AuditMetrics,
        vram_gb: float,
        power_w: float,
        throughput: float
    ) -> Dict[str, Any]:
        record = {
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "total_steps": total_steps,
            "epoch": epoch,
            "mode": metrics.mode,
            "loss": {
                "step": round(loss, 5),
                "ema": round(self._ema_loss or loss, 5)
            },
            "lr": {
                "unet": lr_unet,
                "te": lr_te
            },
            "hardware": {
                "vram_gb": round(vram_gb, 2),
                "power_w": round(power_w, 1),
                "throughput": round(throughput, 2)
            },
            "metrics": {
                "stable_rank": metrics.stable_rank,
                "svd_entropy": metrics.svd_entropy,
                "spectral_smoothness": metrics.spectral_smoothness,
                "dead_neuron_rate": metrics.dead_neuron_rate,
                "update_ratio": metrics.update_ratio,
                "grad_coherence": metrics.grad_coherence,
                "gsnr": metrics.gsnr,
                "noise_pred_std": metrics.noise_pred_std,
                "attn_entropy": metrics.attn_entropy,
                "act_drift": metrics.act_drift,
                "clip_drift": metrics.clip_drift,
                "forgetting_probe": metrics.forgetting_probe,
                "hessian_trace": metrics.hessian_trace,
                "icu_score": metrics.icu_score
            },
            "config": {
                "svd_algorithm": metrics.svd_algorithm,
                "advanced_stats": self.config.advanced_stats_enabled
            }
        }
        
        # 使用 logger 写入 (支持轮转)
        record_json = json.dumps(record, ensure_ascii=False)
        self.logger.info(record_json)
        
        if self.on_data:
            self.on_data(record)
        
        # 更新最后一次报告 (包含 per-layer 缓存)
        # 更新最后一次报告 (包含 per-layer 缓存)
        with self._cache_lock:
             record["layers"] = self._cached_topo_metrics.copy()
        self._last_report = record
        
        return record

    def get_last_report(self) -> Optional[Dict]:
        """获取最后一次生成的完整报告"""
        return self._last_report
    
    def close(self):
        self._running = False
        if self._worker_thread:
            self._metrics_queue.put(None)
            self._worker_thread.join(timeout=2.0)
        # logger 不需要手动关闭，handler 会处理
        # if self._file:
        #     self._file.close()
        #     self._file = None
