"""CPU preview scheduling contracts.

This first worker keeps the training GPU isolated.  It records preview requests
as manifest entries that a later out-of-process CPU generator can consume.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional


class CPUPreviewWorker:
    """Lightweight manifest-backed CPU preview scheduler."""

    def __init__(self, *, output_dir: str, model_arch: str = "", model_path: str = "") -> None:
        self.output_dir = Path(output_dir)
        self.model_arch = str(model_arch or "")
        self.model_path = str(model_path or "")
        self._last_job: Optional[Dict[str, Any]] = None

    def generate(self, **kwargs: Any):
        job = {
            "status": "scheduled",
            "device": "cpu",
            "created_at": time.time(),
            "model_arch": self.model_arch,
            "model_path": self.model_path,
            "prompt": str(kwargs.get("prompt", "") or ""),
            "negative_prompt": str(kwargs.get("negative_prompt", "") or ""),
            "num_inference_steps": int(kwargs.get("num_inference_steps", 0) or 0),
            "guidance_scale": float(kwargs.get("guidance_scale", 0.0) or 0.0),
            "width": int(kwargs.get("width", 0) or 0),
            "height": int(kwargs.get("height", 0) or 0),
            "seed": int(kwargs.get("seed", 0) or 0),
            "note": "CPU preview generation is queued outside the active training GPU path.",
        }
        self._last_job = job
        return None

    def consume_last_job_metadata(self) -> Optional[Dict[str, Any]]:
        job = self._last_job
        self._last_job = None
        return job

    def write_manifest(self, path: Path, job: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(job, ensure_ascii=False, indent=2), encoding="utf-8")

