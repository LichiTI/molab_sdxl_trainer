"""External GGUF quantizer discovery and execution helpers."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable

GGUF_EXTERNAL_QUANT_FORMATS = {"gguf_q4_k_m", "gguf_q5_k_m"}

EXTERNAL_GGUF_QUANT_TYPES = {
    "gguf_q4_k_m": "Q4_K_M",
    "gguf_q5_k_m": "Q5_K_M",
}

_EXECUTABLE_NAMES = ("llama-quantize.exe", "llama-quantize", "quantize.exe", "quantize")
_TOOL_DIRS = (
    Path("tools") / "llama.cpp",
    Path("tools") / "gguf",
    Path("tools"),
    Path("llama.cpp"),
    Path("bin"),
    Path("Scripts"),
    Path("."),
)


def find_gguf_quantizer_executable(search_roots: Iterable[Path] | None = None) -> Path | None:
    env_path = os.environ.get("LULYNX_GGUF_QUANTIZER", "").strip().strip('"')
    if env_path:
        candidate = Path(env_path)
        if candidate.is_file():
            return candidate

    for root in search_roots or ():
        for tool_dir in _TOOL_DIRS:
            for name in _EXECUTABLE_NAMES:
                candidate = Path(root) / tool_dir / name
                if candidate.is_file():
                    return candidate

    for name in ("llama-quantize", "llama-quantize.exe"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def run_external_gguf_quantizer(
    input_path: Path,
    output_path: Path,
    quant_format: str,
    *,
    search_roots: Iterable[Path] | None = None,
    timeout: int = 3600,
) -> dict[str, Any]:
    fmt = str(quant_format or "").strip().lower().replace("-", "_")
    quant_type = EXTERNAL_GGUF_QUANT_TYPES.get(fmt)
    if not quant_type:
        raise ValueError(f"unsupported external GGUF quantization format: {quant_format}")

    executable = find_gguf_quantizer_executable(search_roots)
    if executable is None:
        raise RuntimeError(
            "External GGUF quantizer engine was not found. "
            "Install Launcher support dependency with llama.cpp quantizer included."
        )

    kwargs: dict[str, Any] = {
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "timeout": timeout,
        "check": False,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    result = subprocess.run([str(executable), str(input_path), str(output_path), quant_type], **kwargs)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "GGUF quantizer failed").strip()
        raise RuntimeError(detail[-2000:])
    if not output_path.is_file():
        raise RuntimeError("External GGUF quantizer completed without creating the output file")
    return {
        "gguf_quantizer": str(executable),
        "gguf_quant_type": quant_type,
        "gguf_provider": "external",
    }


__all__ = [
    "EXTERNAL_GGUF_QUANT_TYPES",
    "GGUF_EXTERNAL_QUANT_FORMATS",
    "find_gguf_quantizer_executable",
    "run_external_gguf_quantizer",
]
