# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""Probe TransformerEngine installation feasibility for current runtime."""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import re
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Sequence


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _trim(text: str, *, limit: int = 2000) -> str:
    value = str(text or "").strip()
    if len(value) <= limit:
        return value
    return value[:limit] + "...<truncated>"


def _run(cmd: Sequence[str], *, timeout: int) -> Dict[str, Any]:
    command_text = " ".join(str(part) for part in cmd)
    try:
        completed = subprocess.run(  # noqa: S603
            list(cmd),
            capture_output=True,
            text=True,
            timeout=max(int(timeout), 10),
            check=False,
        )
        stdout = str(completed.stdout or "")
        stderr = str(completed.stderr or "")
        ok = int(completed.returncode) == 0
        return {
            "command": command_text,
            "ok": ok,
            "return_code": int(completed.returncode),
            "stdout": _trim(stdout),
            "stderr": _trim(stderr),
        }
    except Exception as exc:
        return {
            "command": command_text,
            "ok": False,
            "return_code": -1,
            "stdout": "",
            "stderr": "",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _package_installed(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


def _torch_environment() -> Dict[str, Any]:
    env: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
    }
    try:
        import torch

        env.update(
            {
                "torch": str(torch.__version__),
                "torch_cuda": str(getattr(torch.version, "cuda", None)),
                "cuda_available": bool(torch.cuda.is_available()),
            }
        )
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(idx)
            env["device"] = str(props.name)
            env["compute_capability"] = f"{props.major}.{props.minor}"
    except Exception as exc:
        env["torch_error"] = f"{type(exc).__name__}: {exc}"
    return env


def _extract_first_version(index_stdout: str) -> str:
    text = str(index_stdout or "")
    # Example: "Available versions: 2.15.0, 2.14.0, ..."
    match = re.search(r"Available versions:\s*([^\r\n]+)", text, flags=re.IGNORECASE)
    if match:
        first = match.group(1).split(",", 1)[0].strip()
        if first:
            return first
    # Fallback: "transformer-engine (2.15.0)"
    match = re.search(r"transformer-engine\s*\(([^)]+)\)", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def _summary_for_command(result: Dict[str, Any], *, version: str) -> str:
    command = str(result.get("command", ""))
    ok = bool(result.get("ok", False))
    merged = (str(result.get("stdout", "")) + "\n" + str(result.get("stderr", ""))).lower()
    if "index versions transformer-engine" in command:
        if ok:
            discovered = _extract_first_version(str(result.get("stdout", "")))
            if discovered:
                return f"transformer-engine {discovered} is visible."
            return "pip index query succeeded."
        return "pip index query failed."
    if "download transformer-engine==" in command and "[pytorch]" not in command:
        if ok:
            return "Top-level wheel is downloadable, but this does not prove CUDA/PyTorch extension availability."
        return "Top-level wheel download failed."
    if "[pytorch]" in command and "download" in command:
        if ok:
            return "Extras download succeeded."
        if "no matching distribution found" in merged:
            return f"Extras download failed: no matching distribution for requested version {version}."
        return "Extras download failed."
    if "--dry-run" in command:
        if ok:
            return "Dry-run dependency resolution succeeded."
        if "transformer_engine_cu" in merged and "no matching distribution found" in merged:
            return "Dry-run failed: missing matching transformer_engine_cu* distribution."
        if "transformer_engine_torch" in merged and "no matching distribution found" in merged:
            return "Dry-run failed: missing matching transformer_engine_torch* distribution."
        return "Dry-run failed."
    return "Command completed."


def _blocking_reason(commands: List[Dict[str, Any]]) -> str:
    merged = "\n".join(
        (str(item.get("stdout", "")) + "\n" + str(item.get("stderr", ""))).lower()
        for item in commands
        if isinstance(item, dict)
    )
    if "transformer_engine_cu" in merged and "no matching distribution found" in merged:
        return "missing_transformer_engine_cu_distribution"
    if "transformer_engine_torch" in merged and "no matching distribution found" in merged:
        return "missing_transformer_engine_torch_distribution"
    if "read timed out" in merged or "temporary failure in name resolution" in merged:
        return "network_unstable_or_blocked"
    if "permission denied" in merged:
        return "permission_denied"
    return "unknown"


def _decide(*, te_installed: bool, commands: List[Dict[str, Any]]) -> Dict[str, Any]:
    if te_installed:
        return {
            "resolved": "available_in_current_environment",
            "reason": "transformer_engine importable",
            "training_ab_status": "can_proceed",
        }
    all_ok = bool(commands) and all(bool(item.get("ok", False)) for item in commands)
    if all_ok:
        return {
            "resolved": "installable_not_installed",
            "reason": "pip probe commands succeeded, package is not installed yet",
            "training_ab_status": "can_install_then_run",
        }
    blocked = _blocking_reason(commands)
    is_windows = platform.system().lower().startswith("win")
    if is_windows and blocked in {
        "missing_transformer_engine_cu_distribution",
        "missing_transformer_engine_torch_distribution",
    }:
        return {
            "resolved": "blocked_on_current_windows_environment",
            "reason": blocked,
            "safe_next_environment": "Linux x86_64 or WSL2 CUDA runtime",
            "training_ab_status": "not_run",
        }
    return {
        "resolved": "blocked_or_unverified",
        "reason": blocked,
        "training_ab_status": "not_run",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable for pip commands.")
    parser.add_argument("--te-version", default="2.15.0", help="Target TransformerEngine version for pip probe.")
    parser.add_argument("--skip-pip", action="store_true", help="Skip network/pip command probes.")
    parser.add_argument("--timeout", type=int, default=300, help="Per-command timeout in seconds.")
    parser.add_argument("--out", default="temp/transformer_engine_install_probe_windows.json")
    args = parser.parse_args(argv)

    root = _repo_root()
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path

    commands: List[Dict[str, Any]] = []
    if not args.skip_pip:
        with tempfile.TemporaryDirectory(prefix="te_probe_") as tmp:
            tmpdir = Path(tmp)
            report_path = tmpdir / "pip_report.json"
            version = str(args.te_version or "2.15.0").strip()
            cmd_list: List[List[str]] = [
                [args.python_exe, "-m", "pip", "index", "versions", "transformer-engine"],
                [
                    args.python_exe,
                    "-m",
                    "pip",
                    "download",
                    f"transformer-engine=={version}",
                    "--only-binary=:all:",
                    "--no-deps",
                    "-d",
                    str(tmpdir),
                ],
                [
                    args.python_exe,
                    "-m",
                    "pip",
                    "download",
                    f"transformer-engine[pytorch]=={version}",
                    "--only-binary=:all:",
                    "-d",
                    str(tmpdir),
                ],
                [
                    args.python_exe,
                    "-m",
                    "pip",
                    "install",
                    "--dry-run",
                    "--report",
                    str(report_path),
                    f"transformer-engine[pytorch]=={version}",
                ],
            ]
            for cmd in cmd_list:
                result = _run(cmd, timeout=int(args.timeout))
                result["summary"] = _summary_for_command(result, version=version)
                commands.append(result)
    te_installed = _package_installed("transformer_engine")
    payload = {
        "probe": "transformer_engine_install_probe",
        "date": date.today().isoformat(),
        "environment": _torch_environment(),
        "transformer_engine_importable": te_installed,
        "commands": commands,
        "official_support_note": (
            "TransformerEngine is primarily documented for Linux/NGC environments; Windows-native installs require extra dependency verification."
        ),
        "decision": _decide(te_installed=te_installed, commands=commands),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["decision"]["resolved"] in {"available_in_current_environment", "installable_not_installed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
