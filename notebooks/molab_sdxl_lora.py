# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo",
#   "torch",
#   "diffusers",
#   "transformers",
#   "accelerate",
#   "safetensors",
#   "peft",
#   "bitsandbytes",
#   "lycoris-lora",
# ]
# ///

"""Molab / marimo helper notebook for Lulynx SDXL LoRA training.

Open in molab, edit the path variables, then run the cells.
This file is also executable as a normal Python script, but marimo is recommended.
"""

import marimo

__generated_with = "0.8.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    import marimo as mo

    repo = Path.cwd()
    mo.md(
        f"""
        # Lulynx SDXL LoRA on molab

        当前目录：`{repo}`

        使用流程：

        1. 安装依赖：`pip install -r requirements-molab.txt`
        2. 准备 `work/models/` 和 `work/datasets/`
        3. 修改下面路径变量
        4. 写入配置并启动训练
        """
    )
    return Path, json, mo, os, repo, subprocess, sys


@app.cell
def __(mo, subprocess, sys):
    try:
        import torch

        gpu_text = "CUDA 可用" if torch.cuda.is_available() else "CUDA 不可用"
        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-"
        torch_version = torch.__version__
    except Exception as exc:  # pragma: no cover - notebook helper
        gpu_text = f"torch 导入失败：{exc}"
        gpu_name = "-"
        torch_version = "-"

    mo.md(f"""
    ## 环境检查

    - Python: `{sys.version.split()[0]}`
    - Torch: `{torch_version}`
    - GPU: `{gpu_text}`
    - GPU 名称: `{gpu_name}`
    """)
    return gpu_name, gpu_text, torch_version


@app.cell
def __(Path, mo, repo):
    # 按需修改这些路径。相对路径均相对于仓库根目录。
    model_path = "work/models/sd_xl_base_1.0.safetensors"
    train_data_dir = "work/datasets/my_lora"
    output_dir = "work/outputs"
    logging_dir = "work/logs"
    output_name = "my_sdxl_lora"

    for rel in ["work/models", "work/datasets", "work/outputs", "work/logs", "work/runs", "configs"]:
        (repo / rel).mkdir(parents=True, exist_ok=True)

    mo.md(f"""
    ## 路径配置

    - 底模：`{model_path}`
    - 数据集：`{train_data_dir}`
    - 输出：`{output_dir}`
    - 日志：`{logging_dir}`
    - 输出名：`{output_name}`
    """)
    return logging_dir, model_path, output_dir, output_name, train_data_dir


@app.cell
def __(json, logging_dir, model_path, output_dir, output_name, repo, train_data_dir):
    config = {
        "schema_id": "sdxl-lora",
        "training_type": "lora",
        "model_type": "sdxl",
        "pretrained_model_name_or_path": model_path,
        "base_model_path": model_path,
        "train_data_dir": train_data_dir,
        "output_dir": output_dir,
        "logging_dir": logging_dir,
        "output_name": output_name,
        "network_module": "networks.lora",
        "network_dim": 16,
        "network_alpha": 8,
        "network_dropout": 0.0,
        "resolution": "1024,1024",
        "enable_bucket": True,
        "min_bucket_reso": 512,
        "max_bucket_reso": 1536,
        "bucket_reso_steps": 64,
        "caption_extension": ".txt",
        "shuffle_caption": True,
        "keep_tokens": 1,
        "max_token_length": 225,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "max_train_epochs": 10,
        "learning_rate": 1e-4,
        "unet_lr": 1e-4,
        "text_encoder_lr": 1e-5,
        "optimizer_type": "AdamW8bit",
        "lr_scheduler": "cosine",
        "mixed_precision": "bf16",
        "save_precision": "bf16",
        "gradient_checkpointing": True,
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "attention_backend": "sdpa",
        "sdpa": True,
        "xformers": False,
        "torch_compile": False,
        "save_model_as": "safetensors",
        "save_every_n_epochs": 1,
        "checkpoint_keep_last": 3,
        "seed": 42,
        "sample_every": False,
        "low_vram_profile": "off",
        "execution_core": "standard",
    }

    config_path = repo / "configs" / "molab_sdxl_lora.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    config_path
    return config, config_path


@app.cell
def __(Path, config, config_path, mo, repo):
    issues = []
    for key in ["pretrained_model_name_or_path", "train_data_dir"]:
        p = Path(config[key])
        if not p.is_absolute():
            p = repo / p
        if not p.exists():
            issues.append(f"`{key}` 不存在：`{p}`")

    if issues:
        mo.md("## 预检\n" + "\n".join([f"- ❌ {x}" for x in issues]) + f"\n\n配置已写入：`{config_path}`")
    else:
        mo.md(f"## 预检\n- ✅ 路径存在\n- ✅ 配置已写入：`{config_path}`")
    return issues,


@app.cell
def __(config_path, mo):
    mo.md(f"""
    ## 启动训练

    确认上面的预检没有报错后，在 molab 终端或新 cell 中执行：

    ```bash
    python scripts/run_sdxl_train.py --config {config_path}
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
