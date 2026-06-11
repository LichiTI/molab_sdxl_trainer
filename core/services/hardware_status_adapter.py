"""Hardware and runtime status payload helpers for compatibility routes."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

RunCommand = Callable[..., subprocess.CompletedProcess[str]]


def query_nvidia_smi_gpus(run_command: RunCommand | None = None) -> list[dict[str, Any]]:
    """Collect whole-device GPU/VRAM metrics from nvidia-smi."""

    runner = run_command or subprocess.run
    try:
        result = runner(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except Exception:
        return []
    if result.returncode != 0:
        return []

    gpus: list[dict[str, Any]] = []
    for line in (result.stdout or "").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 8:
            continue
        try:
            index = int(float(parts[0]))
            total_mb = int(float(parts[2]))
            used_mb = int(float(parts[3]))
            gpu_util = float(parts[4])
            temperature = float(parts[5])
            power_draw = float(parts[6])
            power_limit = float(parts[7])
        except (TypeError, ValueError):
            continue
        utilization_pct = round((used_mb / total_mb) * 100, 1) if total_mb > 0 else 0.0
        gpus.append(
            {
                "index": index,
                "name": parts[1],
                "total_mb": total_mb,
                "used_mb": used_mb,
                "allocated_mb": used_mb,
                "total_gb": round(total_mb / 1024, 2),
                "used_gb": round(used_mb / 1024, 2),
                "utilization_pct": utilization_pct,
                "gpu_utilization_pct": round(gpu_util, 1),
                "temperature_c": round(temperature, 1),
                "power_draw_w": round(power_draw, 1),
                "power_limit_w": round(power_limit, 1),
                "source": "nvidia-smi",
            }
        )
    return gpus


def get_gpu_cards() -> list[dict[str, Any]]:
    smi_gpus = query_nvidia_smi_gpus()
    if smi_gpus:
        return [
            {
                "name": gpu.get("name", f"GPU {gpu.get('index', 0)}"),
                "memory_total": gpu.get("total_mb", 0),
                "memory_used": gpu.get("used_mb", 0),
                "memory_free": max(int(gpu.get("total_mb", 0) or 0) - int(gpu.get("used_mb", 0) or 0), 0),
                "index": gpu.get("index", 0),
            }
            for gpu in smi_gpus
        ]
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        cards = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            total_mb = props.total_mem / (1024**2)
            reserved_mb = torch.cuda.memory_reserved(index) / (1024**2)
            cards.append(
                {
                    "name": props.name,
                    "memory_total": round(total_mb),
                    "memory_used": round(reserved_mb),
                    "memory_free": round(total_mb - reserved_mb),
                    "index": index,
                }
            )
        return cards
    except Exception:
        return []


def _runtime_id_from_value(value: Any, backend_root: Path) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if raw.lower() == "system":
        return "standard"
    try:
        from backend.core.execution_manifest import get_manifest, get_profile_entry, normalize_profile_id

        normalized = normalize_profile_id(raw)
        entry = get_profile_entry(normalized)
        if entry is not None:
            return entry.id
        raw_lower = raw.replace("\\", "/").strip().lower()
        for candidate in get_manifest():
            env_dir = candidate.env_dir_name.strip().lower()
            if raw_lower in {candidate.id.lower(), env_dir, f"env/{env_dir}", f"backend/env/{env_dir}"}:
                return candidate.id
    except Exception:
        return ""
    return ""


def _runtime_id_from_path(path_value: Any, backend_root: Path) -> str:
    raw = str(path_value or "").strip()
    if not raw:
        return ""
    try:
        current = Path(raw).resolve()
    except Exception:
        return ""
    try:
        from backend.core.execution_manifest import get_manifest, resolve_env_dir, resolve_python_executable

        for entry in get_manifest():
            env_dir = resolve_env_dir(entry, backend_root).resolve()
            python_exe = resolve_python_executable(entry, backend_root).resolve()
            if current == env_dir or current == python_exe or env_dir in current.parents:
                return entry.id
    except Exception:
        return ""
    return ""


def get_active_environment(
    backend_root: Path,
    *,
    environ: dict[str, str] | None = None,
    executable: str | None = None,
) -> str:
    """Return the profile id used by the currently running WebUI process."""

    env = os.environ if environ is None else environ
    for key in ("LULYNX_RUNTIME_ID", "LULYNX_ENTRY_RUNTIME_ID", "LULYNX_ENV"):
        runtime_id = _runtime_id_from_value(env.get(key), backend_root)
        if runtime_id:
            return runtime_id
    for key in ("LULYNX_RUNTIME_ENV_DIR", "LULYNX_RUNTIME_PYTHON", "LULYNX_ENTRY_PYTHON"):
        runtime_id = _runtime_id_from_path(env.get(key), backend_root)
        if runtime_id:
            return runtime_id

    runtime_id = _runtime_id_from_path(sys.executable if executable is None else executable, backend_root)
    if runtime_id:
        return runtime_id

    try:
        from backend.core.execution_manifest import get_manifest, is_profile_installed

        for entry in get_manifest():
            if is_profile_installed(entry, backend_root):
                return entry.id
    except Exception:
        pass
    return "standard"


def build_graphic_cards_payload(
    *,
    backend_root: Path,
    environ: dict[str, str] | None = None,
    executable: str | None = None,
) -> dict[str, Any]:
    environment = get_active_environment(backend_root, environ=environ, executable=executable)
    return {"cards": get_gpu_cards(), "runtime": {"environment": environment, "runtime_id": environment}}


def build_gpu_status_payload() -> dict[str, Any]:
    smi_gpus = query_nvidia_smi_gpus()
    if smi_gpus:
        return {"available": True, "gpus": smi_gpus}
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False, "gpus": []}
        gpus = []
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            total_mb = props.total_mem / (1024**2)
            allocated_mb = torch.cuda.memory_allocated(index) / (1024**2)
            utilization_pct = round(allocated_mb / total_mb * 100, 1) if total_mb > 0 else 0
            gpus.append(
                {
                    "index": index,
                    "name": props.name,
                    "total_mb": round(total_mb),
                    "used_mb": round(allocated_mb),
                    "allocated_mb": round(allocated_mb),
                    "utilization_pct": utilization_pct,
                    "source": "torch",
                }
            )
        return {"available": True, "gpus": gpus}
    except Exception:
        return {"available": False, "gpus": []}


def build_system_monitor_payload(psutil_module: Any | None = None) -> dict[str, Any]:
    psutil = psutil_module
    if psutil is None:
        import psutil as psutil_import

        psutil = psutil_import

    gpu_data: dict[str, Any] = {"available": False, "gpus": []}
    smi_gpus = query_nvidia_smi_gpus()
    if smi_gpus:
        gpu_data = {"available": True, "gpus": smi_gpus}
    else:
        try:
            import torch

            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                total = props.total_mem / (1024**3)
                allocated = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_data = {
                    "available": True,
                    "gpus": [
                        {
                            "index": 0,
                            "name": props.name,
                            "total_gb": round(total, 2),
                            "used_gb": round(allocated, 2),
                            "total_mb": round(total * 1024),
                            "used_mb": round(allocated * 1024),
                            "allocated_mb": round(allocated * 1024),
                            "utilization_pct": round(allocated / total * 100, 1) if total > 0 else 0,
                            "source": "torch",
                        }
                    ],
                }
        except Exception:
            pass

    vm = psutil.virtual_memory()
    cpu_data = {"percent": psutil.cpu_percent(interval=0.1), "count": psutil.cpu_count()}
    ram_data = {
        "total_gb": round(vm.total / (1024**3), 2),
        "used_gb": round(vm.used / (1024**3), 2),
        "percent": vm.percent,
    }
    return {"gpu": gpu_data, "cpu": cpu_data, "ram": ram_data}
