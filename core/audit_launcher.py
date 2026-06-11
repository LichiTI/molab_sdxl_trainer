"""
Lulynx Audit Launcher - V10.0

非侵入式训练监控注入器

工作原理:
1. 启动训练进程
2. 解析训练输出日志提取指标
3. 调用 AuditInterpreter 生成诊断报告

支持的监控方式:
- 日志解析模式: 从训练进程的 stdout 解析 loss/lr/step
- 文件监控模式: 监控 TensorBoard 日志目录
"""

import os
import re
import json
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List
from dataclasses import dataclass, asdict
import logging
import logging.handlers
from queue import Queue

try:
    from .lora_auditor import (
        LoRAAuditor, AuditConfig, AuditInterpreter, 
        DiagnosisReport, create_auditor, SVDAlgorithm
    )
    HAS_AUDITOR = True
except Exception:
    HAS_AUDITOR = False
    LoRAAuditor = None
    AuditConfig = None
    AuditInterpreter = None
    DiagnosisReport = None
    create_auditor = None
    SVDAlgorithm = None


@dataclass
class TrainingMetrics:
    """从日志解析的训练指标"""
    step: int = 0
    total_steps: int = 0
    epoch: int = 0
    loss: float = 0.0
    lr_unet: float = 0.0
    lr_te: float = 0.0
    throughput: float = 0.0  # it/s
    vram_gb: float = 0.0
    timestamp: str = ""


class LogParser:
    """
    训练日志解析器
    
    支持解析的格式:
    - "steps: 100/1000, loss: 0.0123, lr: 1e-4"
    - "epoch 1/10, step 100"
    - tqdm 进度条格式
    """
    
    # 正则表达式模式
    PATTERNS = {
        # 标准格式: "steps: 100/1000"
        "step": re.compile(r"(?:steps?|step)\s*[:\s]\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE),
        # Loss: "loss: 0.0123" or "loss=0.0123"
        "loss": re.compile(r"loss\s*[=:]\s*([\d.e+-]+)", re.IGNORECASE),
        # Learning rate: "lr: 1e-4" or "lr=0.0001"
        "lr": re.compile(r"(?<!te_)lr\s*[=:]\s*([\d.e+-]+)", re.IGNORECASE),
        # TE learning rate: "te_lr: 5e-5"
        "te_lr": re.compile(r"te_lr\s*[=:]\s*([\d.e+-]+)", re.IGNORECASE),
        # Epoch: "epoch 1/10"
        "epoch": re.compile(r"epoch\s*(\d+)\s*/\s*(\d+)", re.IGNORECASE),
        # Throughput: "1.23 it/s" or "1.23it/s"
        "throughput": re.compile(r"([\d.]+)\s*it/s", re.IGNORECASE),
        # VRAM: "VRAM: 8.5GB" or "8.5 GB"
        "vram": re.compile(r"(?:vram|memory)\s*[:\s]\s*([\d.]+)\s*(?:GB|G)", re.IGNORECASE),
    }
    
    def __init__(self):
        self.current_metrics = TrainingMetrics()
        self._history: List[Dict] = []
    
    def parse_line(self, line: str) -> Optional[TrainingMetrics]:
        """
        解析一行日志，返回更新后的指标
        
        如果检测到 step 变化，返回指标副本
        """
        changed = False
        
        # 解析 step
        match = self.PATTERNS["step"].search(line)
        if match:
            new_step = int(match.group(1))
            if new_step != self.current_metrics.step:
                self.current_metrics.step = new_step
                self.current_metrics.total_steps = int(match.group(2))
                changed = True
        
        # 解析 loss
        match = self.PATTERNS["loss"].search(line)
        if match:
            try:
                self.current_metrics.loss = float(match.group(1))
            except ValueError:
                pass
        
        # 解析 lr
        match = self.PATTERNS["lr"].search(line)
        if match:
            try:
                self.current_metrics.lr_unet = float(match.group(1))
            except ValueError:
                pass
        
        # 解析 te_lr
        match = self.PATTERNS["te_lr"].search(line)
        if match:
            try:
                self.current_metrics.lr_te = float(match.group(1))
            except ValueError:
                pass
        
        # 解析 epoch
        match = self.PATTERNS["epoch"].search(line)
        if match:
            self.current_metrics.epoch = int(match.group(1))
        
        # 解析 throughput
        match = self.PATTERNS["throughput"].search(line)
        if match:
            try:
                self.current_metrics.throughput = float(match.group(1))
            except ValueError:
                pass
        
        # 解析 VRAM
        match = self.PATTERNS["vram"].search(line)
        if match:
            try:
                self.current_metrics.vram_gb = float(match.group(1))
            except ValueError:
                pass
        
        # 如果 step 变化了，记录历史
        if changed:
            self.current_metrics.timestamp = datetime.now().isoformat()
            record = asdict(self.current_metrics)
            self._history.append(record)
            return TrainingMetrics(**record)
        
        return None
    
    def get_history(self) -> List[Dict]:
        """获取解析历史"""
        return self._history.copy()


class AuditLauncher:
    """
    审计启动器
    
    用法:
        with AuditLauncher(output_dir="./logs") as launcher:
            # 方式1: 作为日志处理器
            for line in training_process.stdout:
                launcher.process_line(line)
    """
    
    def __init__(
        self,
        output_dir: str = "./logs",
        svd_algorithm: str = "rsvd",
        advanced_stats: bool = True,
        on_metrics: Optional[Callable[[Dict], None]] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.parser = LogParser()
        self.interpreter = AuditInterpreter() if HAS_AUDITOR else None
        self.on_metrics = on_metrics
        
        # 指标记录器 (带轮转)
        self.metrics_file = self.output_dir / "training_metrics.jsonl"
        self._setup_metrics_logger()
        
        # 应用日志
        # logging.basicConfig(level=logging.INFO) # Removed to prevent global config override
        self._logger = logging.getLogger("AuditLauncher")
        
        # 状态
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        self._logger.info(f"[AuditLauncher] 初始化完成")
        self._logger.info(f"  输出目录: {self.output_dir}")
        self._logger.info(f"  指标文件: {self.metrics_file}")

    def _setup_metrics_logger(self):
        """配置指标日志轮转"""
        self._metrics_logger = logging.getLogger("AuditMetrics")
        self._metrics_logger.setLevel(logging.INFO)
        self._metrics_logger.propagate = False
        
        if not self._metrics_logger.handlers:
            # 5MB 轮转，保留 3 个备份
            handler = logging.handlers.RotatingFileHandler(
                self.metrics_file, maxBytes=5*1024*1024, backupCount=3, encoding='utf-8'
            )
            handler.setFormatter(logging.Formatter('%(message)s'))
            self._metrics_logger.addHandler(handler)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
    
    def process_line(self, line: str) -> Optional["TrainingMetrics"]:
        """
        处理一行训练日志
        
        如果解析到新的 step，返回指标并记录
        """
        metrics = self.parser.parse_line(line)
        
        if metrics:
            # 写入 JSONL (使用带轮转的 Logger)
            record = asdict(metrics)
            self._metrics_logger.info(json.dumps(record, ensure_ascii=False))
            
            # 添加到解释器
            if self.interpreter:
                # 转换为审计器格式
                audit_record = {
                    "step": metrics.step,
                    "total_steps": metrics.total_steps,
                    "epoch": metrics.epoch,
                    "loss": {"step": metrics.loss, "ema": metrics.loss},
                    "lr": {"unet": metrics.lr_unet, "te": metrics.lr_te},
                    "hardware": {
                        "vram_gb": metrics.vram_gb,
                        "throughput": metrics.throughput
                    },
                    "metrics": {}  # 日志解析无法获取深度指标
                }
                self.interpreter.add_metrics(audit_record)
            
            # 回调
            if self.on_metrics:
                self.on_metrics(record)
        
        return metrics
    
    def start_file_monitor(self, log_file: str, poll_interval: float = 0.5):
        """
        启动文件监控（类似 tail -f）
        """
        self._running = True
        
        def monitor():
            path = Path(log_file)
            if not path.exists():
                path.touch()
            
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                # 跳到文件末尾
                f.seek(0, 2)
                
                while self._running:
                    line = f.readline()
                    if line:
                        self.process_line(line.strip())
                    else:
                        time.sleep(poll_interval)
        
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
        self._logger.info(f"[AuditLauncher] 文件监控已启动: {log_file}")
    
    def stop_monitor(self):
        """停止文件监控"""
        self._running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
            if self._monitor_thread.is_alive():
                 self._logger.warning("[AuditLauncher] 文件监控线程未能在超时内退出")

    def generate_report(self) -> Optional["DiagnosisReport"]:
        """
        生成诊断报告
        """
        if not self.interpreter:
            self._logger.warning("[AuditLauncher] 警告: 未加载审计模块，无法生成报告")
            return None
        
        report = self.interpreter.generate_report()
        
        # 保存报告
        report_path = self.output_dir / "diagnosis_report.json"
        report.save(str(report_path))
        self._logger.info(f"[AuditLauncher] 诊断报告已保存: {report_path}")
        
        return report
    
    def get_metrics_history(self) -> List[Dict]:
        """获取所有解析到的指标"""
        return self.parser.get_history()
    
    def close(self):
        """关闭启动器"""
        self.stop_monitor()


def create_launcher(
    output_dir: str = "./logs",
    svd_algorithm: str = "rsvd",
    advanced_stats: bool = True,
    on_metrics: Optional[Callable] = None
) -> AuditLauncher:
    """
    创建审计启动器的便捷函数
    """
    return AuditLauncher(
        output_dir=output_dir,
        svd_algorithm=svd_algorithm,
        advanced_stats=advanced_stats,
        on_metrics=on_metrics
    )


# ========== 集成辅助函数 ==========

def wrap_training_output(
    process,
    launcher: AuditLauncher,
    on_line: Optional[Callable[[str], None]] = None
):
    """
    包装训练进程的输出，同时进行日志解析
    
    用法:
        process = subprocess.Popen(...)
        launcher = AuditLauncher()
        
        wrap_training_output(process, launcher, on_line=print)
    """
    for line in iter(process.stdout.readline, ''):
        line = line.strip()
        if line:
            # 解析指标
            launcher.process_line(line)
            # 原始输出
            if on_line:
                on_line(line)
    
    # 训练结束，生成报告
    launcher.generate_report()
