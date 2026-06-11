"""
core.launcher — DEPRECATED stub module.

The subprocess TrainingLauncher has been removed.
This module is kept as a safe import target so that any remaining
``from core.launcher import ...`` does not crash at import time.

Preserved:
  - SVDMonitor utility (no subprocess dependency).
  - TrainingLauncher / MockTrainingMonitor stubs (raise on use).
"""

import json
import time
import subprocess
import sys
import os
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)
try:
    import numpy as np
except ImportError:
    np = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

class SVDMonitor:

    def __init__(self, output_file: str):
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.health_zones = {'healthy': (0.3, 0.7), 'warning': (0.15, 0.85)}

    def analyze_weight(self, weight: 'torch.Tensor', layer_name: str) -> Dict[str, Any]:
        if not HAS_TORCH:
            return {'error': 'torch not available'}
        try:
            if weight.dim() == 4:
                w = weight.view(weight.size(0), -1)
            elif weight.dim() == 2:
                w = weight
            else:
                return {'skipped': True, 'reason': f'unsupported dim: {weight.dim()}'}
            with torch.no_grad():
                try:
                    U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
                except Exception:
                    return {'skipped': True, 'reason': 'svd failed'}
            S_np = S.cpu().numpy()
            S_normalized = S_np / (S_np.max() + 1e-08)
            top_ratio = S_np[:5].sum() / (S_np.sum() + 1e-08)
            effective_rank = (S_np > S_np.max() * 0.01).sum()
            return {'layer': layer_name, 'singular_values': S_np[:20].tolist(), 'top_ratio': float(top_ratio), 'effective_rank': int(effective_rank), 'total_rank': len(S_np), 'health_score': self._compute_health_score(top_ratio, effective_rank, len(S_np))}
        except Exception as e:
            import logging
            logging.error(f"[SVDMonitor] Analysis failed for {layer_name}: {e}")
            return {'error': str(e), 'skipped': True}

    def _compute_health_score(self, top_ratio: float, eff_rank: int, total_rank: int) -> float:
        if 0.3 <= top_ratio <= 0.7:
            ratio_score = 1.0
        elif top_ratio < 0.3:
            ratio_score = top_ratio / 0.3
        else:
            ratio_score = 1.0 - (top_ratio - 0.7) / 0.3
        rank_ratio = eff_rank / max(total_rank, 1)
        if 0.2 <= rank_ratio <= 0.8:
            rank_score = 1.0
        else:
            rank_score = 0.5
        return ratio_score * 0.7 + rank_score * 0.3

    def log_step(self, step: int, loss: float, lr: float, svd_results: Optional[list]=None, throughput: float=0.0, vram_gb: float=0.0, power_w: float=0.0):
        record = {'timestamp': datetime.now().isoformat(), 'step': step, 'loss': loss, 'lr': lr, 'throughput': throughput, 'vram_gb': vram_gb, 'power_w': power_w, 'svd': svd_results or []}
        try:
            with open(self.output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        except Exception as e:
            logger.error(f"[SVDMonitor] Write failed: {e}")

    def close(self):
        pass


class TrainingLauncher:
    """DEPRECATED STUB — subprocess launcher has been removed.

    Instantiation is allowed for import compatibility, but ``start()`` raises
    ``RuntimeError``.  Use the native lulynx engine via core/entry_train.py.
    """

    def __init__(self, scripts_dir: str, config_file: str, output_dir: str, on_step: Optional[Callable]=None, model_type: str = "sdxl"):
        self.scripts_dir = Path(scripts_dir)
        self.config_file = Path(config_file)
        self.output_dir = Path(output_dir)
        self.on_step = on_step
        self.model_type = model_type
        self._process = None
        self._should_stop = False

    def start(self):
        raise RuntimeError(
            "TrainingLauncher (subprocess) has been removed. "
            "Use the native lulynx engine via core/entry_train.py."
        )

    def stop(self):
        self._should_stop = True


class MockTrainingMonitor:
    """DEPRECATED STUB — mock training monitor has been removed.

    Instantiation is allowed, but ``start()`` raises ``RuntimeError``.
    """

    def __init__(self, output_file: str):
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self._running = False

    def start(self, total_steps: int=1000, callback: Optional[Callable]=None):
        raise RuntimeError("MockTrainingMonitor has been removed (legacy runtime deprecated).")

    def stop(self):
        self._running = False
