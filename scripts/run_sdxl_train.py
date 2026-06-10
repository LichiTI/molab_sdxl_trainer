#!/usr/bin/env python3
"""Run SDXL LoRA training from a standalone molab export repo.

Expected exported repo layout:
    core/entry_train.py
    configs/sdxl_lora_minimal.json
    scripts/run_sdxl_train.py

Usage:
    python scripts/run_sdxl_train.py --config configs/sdxl_lora_minimal.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _make_run_dir(root: Path, name: str) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_") or "sdxl_lora"
    run_dir = root / "work" / "runs" / f"{stamp}-{safe_name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _resolve_path_fields(config: dict[str, Any], repo: Path) -> dict[str, Any]:
    """Keep relative paths relative to repo root and mirror base_model aliases."""
    normalized = dict(config)
    model = normalized.get("pretrained_model_name_or_path") or normalized.get("base_model_path") or normalized.get("pretrained_model")
    if model:
        normalized["pretrained_model_name_or_path"] = str(model)
        normalized["base_model_path"] = str(model)

    defaults = {
        "output_dir": "work/outputs",
        "logging_dir": "work/logs",
    }
    for key, default in defaults.items():
        value = str(normalized.get(key) or default)
        normalized[key] = value
        path = Path(value)
        if not path.is_absolute():
            (repo / path).mkdir(parents=True, exist_ok=True)

    normalized.setdefault("schema_id", "sdxl-lora")
    normalized.setdefault("training_type", "lora")
    normalized.setdefault("model_type", "sdxl")
    normalized.setdefault("execution_core", "standard")
    return normalized


def _preflight(config: dict[str, Any], repo: Path) -> None:
    required_paths = {
        "pretrained_model_name_or_path": "SDXL 底模",
        "train_data_dir": "训练集目录",
    }
    missing: list[str] = []
    for key, label in required_paths.items():
        value = str(config.get(key) or "")
        if not value:
            missing.append(f"{label}: 配置字段 {key} 为空")
            continue
        path = Path(value)
        if not path.is_absolute():
            path = repo / path
        if not path.exists():
            missing.append(f"{label} 不存在: {path}")
    if str(config.get("model_type", "")).lower() != "sdxl":
        missing.append("model_type 必须是 sdxl")
    if str(config.get("training_type", "")).lower() != "lora":
        missing.append("training_type 必须是 lora")
    if missing:
        raise SystemExit("\n".join(["预检失败：", *[f"- {item}" for item in missing]]))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Lulynx SDXL LoRA training in molab")
    parser.add_argument("--config", default="configs/sdxl_lora_minimal.json", help="Config JSON path")
    parser.add_argument("--run-root", default="work/runs", help="Run metadata root, relative to repo")
    parser.add_argument("--no-preflight", action="store_true", help="Skip model/dataset path checks")
    args = parser.parse_args()

    repo = _repo_root()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo / config_path
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path}")

    core_entry = repo / "core" / "entry_train.py"
    if not core_entry.is_file():
        raise SystemExit(
            "core/entry_train.py 不存在。请先用 export_molab_sdxl_repo.py 导出仓库，"
            "不要只复制 molab_sdxl_trainer 这个脚手架目录。"
        )

    config = _resolve_path_fields(_load_json(config_path), repo)
    if not args.no_preflight:
        _preflight(config, repo)

    run_root = Path(args.run_root)
    if not run_root.is_absolute():
        run_root = repo / run_root
    run_dir = _make_run_dir(repo, str(config.get("output_name") or "sdxl_lora"))
    runtime_config = run_dir / "config.json"
    _write_json(runtime_config, config)
    shutil.copy2(config_path, run_dir / "source_config.json")

    env = os.environ.copy()
    pythonpath = [str(repo), str(repo / "core")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cmd = [sys.executable, "-u", str(core_entry), "--config", str(runtime_config)]
    print("Repo:", repo)
    print("Run dir:", run_dir)
    print("Command:", " ".join(cmd))
    print("=" * 80, flush=True)

    proc = subprocess.Popen(
        cmd,
        cwd=str(repo),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
    return int(proc.wait())


if __name__ == "__main__":
    raise SystemExit(main())
