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
#   "huggingface_hub",
# ]
# ///

"""Interactive molab / marimo notebook for standalone SDXL LoRA training.

This notebook is intended to live in the standalone molab repository root layout:

    core/entry_train.py
    configs/
    notebooks/molab_sdxl_lora.py
    scripts/run_sdxl_train.py
    work/

Open it in molab, edit widgets, write config, then start training.
"""

from __future__ import annotations

import marimo

__generated_with = "0.8.0"
app = marimo.App(width="medium")


@app.cell
def __():
    import json
    import os
    import shlex
    import subprocess
    import sys
    import time
    import zipfile
    from pathlib import Path
    from typing import Any

    import marimo as mo

    def is_repo_root(candidate: Path) -> bool:
        return (candidate / "core" / "entry_train.py").is_file() and (candidate / "scripts" / "run_sdxl_train.py").is_file()

    def find_repo_root() -> Path:
        """Find the exported repo root even when molab starts from a single notebook file."""
        cwd = Path.cwd().resolve()

        # 1) Normal case: cwd is repo root, or cwd is repo/notebooks.
        for candidate in [cwd, *cwd.parents]:
            if is_repo_root(candidate):
                return candidate

        # 2) Molab may keep the notebook in /marimo while a cloned repo is a child of /marimo.
        # Keep this search shallow and marker-based to avoid scanning huge model/dataset folders.
        search_roots = [cwd, Path("/marimo"), Path("/workspaces")]
        seen: set[Path] = set()
        for root_candidate in search_roots:
            try:
                root_candidate = root_candidate.resolve()
            except Exception:
                continue
            if root_candidate in seen or not root_candidate.exists():
                continue
            seen.add(root_candidate)
            try:
                for marker in root_candidate.glob("*/core/entry_train.py"):
                    candidate = marker.parent.parent
                    if is_repo_root(candidate):
                        return candidate
            except Exception:
                pass
        return cwd

    repo = find_repo_root()
    repo_ready = is_repo_root(repo)

    github_repo_input = mo.ui.text(
        value="",
        label="如果当前只看到单个 py 文件，请粘贴完整 GitHub 仓库地址，例如 https://github.com/you/repo.git",
        full_width=True,
    )
    clone_dir_input = mo.ui.text(value="/marimo/lulynx-sdxl-trainer", label="clone 到这个目录", full_width=True)
    clone_repo_button = mo.ui.run_button(label="clone / 更新完整仓库")
    clone_message = ""

    def clone_or_update_repo(repo_url: str, target_dir: Path) -> str:
        """Clone or update the full training repository.

        Keep subprocess temporaries inside this helper so marimo does not treat
        names like `proc` as cell-level variables that can collide with other cells.
        """
        if not repo_url:
            return "❌ 请先填写 GitHub 仓库地址。"
        try:
            if (target_dir / ".git").is_dir():
                git_result = subprocess.run(
                    ["git", "-C", str(target_dir), "pull", "--ff-only"],
                    text=True,
                    capture_output=True,
                    timeout=180,
                )
            else:
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                git_result = subprocess.run(
                    ["git", "clone", "--depth", "1", repo_url, str(target_dir)],
                    text=True,
                    capture_output=True,
                    timeout=300,
                )
            if git_result.returncode == 0:
                return "✅ 完整仓库已 clone/update；请重新运行全部 cells。"
            return f"❌ git 失败，退出码 {git_result.returncode}\n\n```text\n{git_result.stdout}\n{git_result.stderr}\n```"
        except Exception as clone_error:
            return f"❌ clone/update 失败：{type(clone_error).__name__}: {clone_error}"

    if clone_repo_button.value:
        clone_message = clone_or_update_repo(
            str(github_repo_input.value or "").strip(),
            Path(str(clone_dir_input.value or "/marimo/lulynx-sdxl-trainer")).expanduser(),
        )
        repo = find_repo_root()
        repo_ready = is_repo_root(repo)
        if repo_ready and clone_message.startswith("✅"):
            clone_message = f"✅ 完整仓库已准备：`{repo}`"

    if repo_ready:
        for rel in ["configs", "work/models", "work/datasets", "work/outputs", "work/logs", "work/runs", "work/archives"]:
            (repo / rel).mkdir(parents=True, exist_ok=True)

    def q(path: Path | str) -> str:
        return shlex.quote(str(path))

    repo_status = "✅ 已找到完整训练仓库" if repo_ready else "⚠️ 当前还没找到完整训练仓库，只找到了 notebook 工作目录"
    bootstrap_panel = [] if repo_ready else [
        mo.callout(
            "如果左侧文件目录只有这个 `.py`，请在下面填 GitHub 仓库地址并点击 clone。clone 完后重新运行全部 cells。",
            kind="warn",
        ),
        mo.hstack([github_repo_input, clone_dir_input]),
        clone_repo_button,
    ]
    if clone_message:
        bootstrap_panel.append(mo.md(clone_message))

    mo.vstack(
        [
            mo.md(
                f"""
                # Lulynx SDXL LoRA — molab 交互训练面板

                {repo_status}

                当前训练仓库根目录：`{repo}`

                这个 notebook 只控制 **SDXL LoRA 训练**，不依赖前端 `plugin/lora-scripts-ui-main`，也不启动 WebUI。
                """
            ),
            *bootstrap_panel,
        ]
    )
    return Any, Path, json, mo, os, q, repo, shlex, subprocess, sys, time, zipfile


@app.cell
def __(Path, mo, repo, subprocess, sys):
    core_entry = repo / "core" / "entry_train.py"
    runner = repo / "scripts" / "run_sdxl_train.py"
    requirements = repo / "requirements-molab.txt"

    try:
        import torch

        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
        cuda_version = str(getattr(torch.version, "cuda", ""))
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "-"
        vram_gb = (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            if cuda_available
            else 0
        )
    except Exception as torch_error:
        torch_version = f"torch 导入失败：{torch_error}"
        cuda_available = False
        cuda_version = "-"
        gpu_name = "-"
        vram_gb = 0

    nvidia_smi_button = mo.ui.run_button(label="刷新 nvidia-smi")
    install_command = f"{sys.executable} -m pip install -r {requirements}" if requirements.is_file() else "requirements-molab.txt 不存在"

    status_lines = [
        f"- core/entry_train.py：{'✅' if core_entry.is_file() else '❌'} `{core_entry}`",
        f"- scripts/run_sdxl_train.py：{'✅' if runner.is_file() else '❌'} `{runner}`",
        f"- requirements-molab.txt：{'✅' if requirements.is_file() else '❌'} `{requirements}`",
        f"- Python：`{sys.version.split()[0]}`",
        f"- Torch：`{torch_version}`",
        f"- CUDA：{'✅ 可用' if cuda_available else '❌ 不可用'}，CUDA 版本：`{cuda_version}`",
        f"- GPU：`{gpu_name}`，显存：`{vram_gb} GB`",
    ]

    mo.vstack(
        [
            mo.md("## 1. 环境与仓库检查\n" + "\n".join(status_lines)),
            mo.callout(
                f"如果依赖未安装，先在 molab cell/terminal 执行：\n\n```bash\n{install_command}\n```",
                kind="info",
            ),
            nvidia_smi_button,
        ]
    )
    return core_entry, cuda_available, gpu_name, install_command, nvidia_smi_button, requirements, runner, torch_version, vram_gb


@app.cell
def __(mo, nvidia_smi_button, subprocess):
    smi_output = ""
    if nvidia_smi_button.value:
        try:
            smi_output = subprocess.check_output(["nvidia-smi"], text=True, stderr=subprocess.STDOUT, timeout=20)
        except Exception as smi_error:
            smi_output = f"nvidia-smi 执行失败：{smi_error}"
    mo.md(f"```text\n{smi_output}\n```") if smi_output else mo.md("刷新后这里会显示 `nvidia-smi` 输出。")
    return smi_output,


@app.cell
def __(mo, repo):
    mo.md("## 2. 模型/数据集路径")

    model_path_input = mo.ui.text(
        value="work/models/sd_xl_base_1.0.safetensors",
        label="SDXL 底模路径 / Hugging Face 已下载文件路径",
        full_width=True,
    )
    vae_path_input = mo.ui.text(
        value="",
        label="可选 VAE 路径，留空使用底模自带 VAE",
        full_width=True,
    )
    train_data_dir_input = mo.ui.text(
        value="work/datasets/my_lora",
        label="训练集目录",
        full_width=True,
    )
    output_name_input = mo.ui.text(value="my_sdxl_lora", label="输出名", full_width=True)
    output_dir_input = mo.ui.text(value="work/outputs", label="LoRA 输出目录", full_width=True)
    logging_dir_input = mo.ui.text(value="work/logs", label="日志目录", full_width=True)
    config_name_input = mo.ui.text(value="molab_sdxl_lora", label="配置文件名，不含 .json", full_width=True)

    mo.vstack(
        [
            mo.hstack([model_path_input, vae_path_input]),
            mo.hstack([train_data_dir_input, output_name_input]),
            mo.hstack([output_dir_input, logging_dir_input]),
            config_name_input,
            mo.callout(
                "相对路径均以仓库根目录为基准。推荐把模型放在 `work/models/`，训练集放在 `work/datasets/`。",
                kind="info",
            ),
        ]
    )
    return config_name_input, logging_dir_input, model_path_input, output_dir_input, output_name_input, train_data_dir_input, vae_path_input


@app.cell
def __(mo):
    mo.md("## 3. 可选：从 Hugging Face 下载模型")

    hf_repo_input = mo.ui.text(
        value="stabilityai/stable-diffusion-xl-base-1.0",
        label="HF repo_id",
        full_width=True,
    )
    hf_file_input = mo.ui.text(
        value="sd_xl_base_1.0.safetensors",
        label="HF filename",
        full_width=True,
    )
    hf_local_dir_input = mo.ui.text(value="work/models", label="下载目录", full_width=True)
    hf_token_env_input = mo.ui.text(value="HF_TOKEN", label="Token 环境变量名，公开模型可忽略", full_width=True)
    hf_download_button = mo.ui.run_button(label="下载/确认 HF 模型")

    mo.vstack(
        [
            mo.hstack([hf_repo_input, hf_file_input]),
            mo.hstack([hf_local_dir_input, hf_token_env_input]),
            hf_download_button,
        ]
    )
    return hf_download_button, hf_file_input, hf_local_dir_input, hf_repo_input, hf_token_env_input


@app.cell
def __(Path, hf_download_button, hf_file_input, hf_local_dir_input, hf_repo_input, hf_token_env_input, mo, os, repo):
    hf_download_result = ""
    if hf_download_button.value:
        try:
            from huggingface_hub import hf_hub_download

            local_dir = Path(hf_local_dir_input.value)
            if not local_dir.is_absolute():
                local_dir = repo / local_dir
            local_dir.mkdir(parents=True, exist_ok=True)
            token_name = str(hf_token_env_input.value or "HF_TOKEN")
            token = os.environ.get(token_name) or None
            downloaded = hf_hub_download(
                repo_id=str(hf_repo_input.value).strip(),
                filename=str(hf_file_input.value).strip(),
                local_dir=str(local_dir),
                token=token,
            )
            hf_download_result = f"✅ 下载/命中缓存：{downloaded}"
        except Exception as hf_error:
            hf_download_result = f"❌ 下载失败：{type(hf_error).__name__}: {hf_error}"

    mo.md(hf_download_result or "未触发下载。")
    return hf_download_result,


@app.cell
def __(mo):
    mo.md("## 4. 训练参数")

    resolution_dropdown = mo.ui.dropdown(
        options=["1024,1024", "896,1152", "832,1216", "768,1344", "768,768"],
        value="1024,1024",
        label="基础分辨率",
    )
    rank_input = mo.ui.number(value=16, start=1, stop=512, step=1, label="LoRA rank / network_dim")
    alpha_input = mo.ui.number(value=8, start=1, stop=512, step=1, label="LoRA alpha")
    dropout_input = mo.ui.number(value=0.0, start=0.0, stop=1.0, step=0.01, label="LoRA dropout")

    epochs_input = mo.ui.number(value=10, start=1, stop=1000, step=1, label="Epochs")
    max_steps_input = mo.ui.number(value=0, start=0, stop=1_000_000, step=100, label="最大 steps，0 表示按 epoch")
    batch_size_input = mo.ui.number(value=1, start=1, stop=64, step=1, label="Batch size")
    grad_accum_input = mo.ui.number(value=1, start=1, stop=128, step=1, label="梯度累积")

    lr_input = mo.ui.number(value=1e-4, start=0.0, stop=1.0, step=1e-5, label="总 learning_rate")
    unet_lr_input = mo.ui.number(value=1e-4, start=0.0, stop=1.0, step=1e-5, label="UNet LR")
    text_encoder_lr_input = mo.ui.number(value=1e-5, start=0.0, stop=1.0, step=1e-6, label="Text Encoder LR")
    optimizer_dropdown = mo.ui.dropdown(
        options=["AdamW8bit", "AdamW", "PagedAdamW8bit", "Prodigy", "Lion", "Lion8bit"],
        value="AdamW8bit",
        label="Optimizer",
    )
    scheduler_dropdown = mo.ui.dropdown(
        options=["cosine", "constant", "constant_with_warmup", "linear", "cosine_with_restarts"],
        value="cosine",
        label="LR scheduler",
    )
    warmup_steps_input = mo.ui.number(value=0, start=0, stop=100000, step=10, label="Warmup steps")
    seed_input = mo.ui.number(value=42, start=0, stop=2**31 - 1, step=1, label="Seed")

    mo.vstack(
        [
            mo.hstack([resolution_dropdown, rank_input, alpha_input, dropout_input]),
            mo.hstack([epochs_input, max_steps_input, batch_size_input, grad_accum_input]),
            mo.hstack([lr_input, unet_lr_input, text_encoder_lr_input]),
            mo.hstack([optimizer_dropdown, scheduler_dropdown, warmup_steps_input, seed_input]),
        ]
    )
    return alpha_input, batch_size_input, dropout_input, epochs_input, grad_accum_input, lr_input, max_steps_input, optimizer_dropdown, rank_input, resolution_dropdown, scheduler_dropdown, seed_input, text_encoder_lr_input, unet_lr_input, warmup_steps_input


@app.cell
def __(mo):
    mo.md("## 5. 数据集、缓存、显存策略")

    enable_bucket_checkbox = mo.ui.checkbox(value=True, label="启用 bucket")
    min_bucket_input = mo.ui.number(value=512, start=64, stop=4096, step=64, label="最小 bucket")
    max_bucket_input = mo.ui.number(value=1536, start=64, stop=4096, step=64, label="最大 bucket")
    bucket_step_input = mo.ui.number(value=64, start=8, stop=512, step=8, label="bucket 步进")
    caption_ext_input = mo.ui.text(value=".txt", label="caption 后缀")
    shuffle_caption_checkbox = mo.ui.checkbox(value=True, label="shuffle caption")
    keep_tokens_input = mo.ui.number(value=1, start=0, stop=256, step=1, label="keep_tokens")
    max_token_length_input = mo.ui.dropdown(options=["75", "150", "225"], value="225", label="max_token_length")
    flip_aug_checkbox = mo.ui.checkbox(value=False, label="flip_aug")
    color_aug_checkbox = mo.ui.checkbox(value=False, label="color_aug")

    mixed_precision_dropdown = mo.ui.dropdown(options=["bf16", "fp16", "no"], value="bf16", label="mixed precision")
    save_precision_dropdown = mo.ui.dropdown(options=["bf16", "fp16", "float"], value="bf16", label="save precision")
    grad_ckpt_checkbox = mo.ui.checkbox(value=True, label="gradient checkpointing")
    cache_latents_checkbox = mo.ui.checkbox(value=True, label="cache latents")
    cache_latents_disk_checkbox = mo.ui.checkbox(value=True, label="cache latents to disk")
    cache_text_encoder_checkbox = mo.ui.checkbox(value=False, label="cache text encoder outputs")
    attention_backend_dropdown = mo.ui.dropdown(options=["sdpa", "auto", "xformers"], value="sdpa", label="attention backend")
    xformers_checkbox = mo.ui.checkbox(value=False, label="xformers 标记")
    torch_compile_checkbox = mo.ui.checkbox(value=False, label="torch_compile，首次跑通前建议关闭")
    low_vram_dropdown = mo.ui.dropdown(options=["off", "standard_16g", "low_12g", "very_low_8g"], value="off", label="low_vram_profile")

    mo.vstack(
        [
            mo.hstack([enable_bucket_checkbox, min_bucket_input, max_bucket_input, bucket_step_input]),
            mo.hstack([caption_ext_input, shuffle_caption_checkbox, keep_tokens_input, max_token_length_input]),
            mo.hstack([flip_aug_checkbox, color_aug_checkbox]),
            mo.hstack([mixed_precision_dropdown, save_precision_dropdown, grad_ckpt_checkbox]),
            mo.hstack([cache_latents_checkbox, cache_latents_disk_checkbox, cache_text_encoder_checkbox]),
            mo.hstack([attention_backend_dropdown, xformers_checkbox, torch_compile_checkbox, low_vram_dropdown]),
        ]
    )
    return attention_backend_dropdown, bucket_step_input, cache_latents_checkbox, cache_latents_disk_checkbox, cache_text_encoder_checkbox, caption_ext_input, color_aug_checkbox, enable_bucket_checkbox, flip_aug_checkbox, grad_ckpt_checkbox, keep_tokens_input, low_vram_dropdown, max_bucket_input, max_token_length_input, min_bucket_input, mixed_precision_dropdown, save_precision_dropdown, shuffle_caption_checkbox, torch_compile_checkbox, xformers_checkbox


@app.cell
def __(mo):
    mo.md("## 6. 保存、采样与高级覆盖")

    save_every_epoch_input = mo.ui.number(value=1, start=1, stop=1000, step=1, label="每 N epoch 保存")
    save_every_steps_input = mo.ui.number(value=0, start=0, stop=1_000_000, step=100, label="每 N steps 保存，0 关闭")
    keep_last_input = mo.ui.number(value=3, start=1, stop=100, step=1, label="最多保留 checkpoint 数")
    save_state_checkbox = mo.ui.checkbox(value=False, label="保存训练 state")
    save_state_end_checkbox = mo.ui.checkbox(value=False, label="结束时保存 state")

    sample_every_checkbox = mo.ui.checkbox(value=False, label="启用采样预览，首次跑通前建议关闭")
    sample_every_epochs_input = mo.ui.number(value=1, start=1, stop=1000, step=1, label="每 N epoch 采样")
    sample_prompt_input = mo.ui.text_area(value="", label="采样 prompt，可留空", full_width=True)
    sample_negative_input = mo.ui.text_area(value="low quality, worst quality, blurry", label="采样 negative", full_width=True)

    advanced_json_input = mo.ui.text_area(
        value="{}",
        label="高级 JSON 覆盖：会 merge 到最终配置里，例如 {\"network_dim\": 32}",
        full_width=True,
    )

    mo.vstack(
        [
            mo.hstack([save_every_epoch_input, save_every_steps_input, keep_last_input]),
            mo.hstack([save_state_checkbox, save_state_end_checkbox]),
            mo.hstack([sample_every_checkbox, sample_every_epochs_input]),
            sample_prompt_input,
            sample_negative_input,
            advanced_json_input,
        ]
    )
    return advanced_json_input, keep_last_input, sample_every_checkbox, sample_every_epochs_input, sample_negative_input, sample_prompt_input, save_every_epoch_input, save_every_steps_input, save_state_checkbox, save_state_end_checkbox


@app.cell
def __(Path, repo):
    def resolve_repo_path(value: str) -> Path:
        p = Path(str(value or "").strip())
        return p if p.is_absolute() else repo / p

    def count_dataset(train_dir: Path, caption_ext: str = ".txt") -> dict[str, object]:
        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        if not train_dir.exists():
            return {"exists": False, "images": 0, "captions": 0, "subsets": []}
        images = [p for p in train_dir.rglob("*") if p.is_file() and p.suffix.lower() in image_exts]
        captions = 0
        for image in images:
            if image.with_suffix(caption_ext).is_file():
                captions += 1
        subsets = sorted({str(p.parent.relative_to(train_dir)) for p in images})[:30]
        return {"exists": True, "images": len(images), "captions": captions, "subsets": subsets}

    def safe_config_name(name: str) -> str:
        raw = str(name or "molab_sdxl_lora").strip().replace(".json", "")
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in raw).strip("_")
        return safe or "molab_sdxl_lora"

    return count_dataset, resolve_repo_path, safe_config_name


@app.cell
def __(Any, advanced_json_input, alpha_input, attention_backend_dropdown, batch_size_input, bucket_step_input, cache_latents_checkbox, cache_latents_disk_checkbox, cache_text_encoder_checkbox, caption_ext_input, color_aug_checkbox, config_name_input, dropout_input, enable_bucket_checkbox, epochs_input, flip_aug_checkbox, grad_accum_input, grad_ckpt_checkbox, json, keep_last_input, keep_tokens_input, logging_dir_input, low_vram_dropdown, lr_input, max_bucket_input, max_steps_input, max_token_length_input, min_bucket_input, mixed_precision_dropdown, model_path_input, optimizer_dropdown, output_dir_input, output_name_input, rank_input, resolution_dropdown, sample_every_checkbox, sample_every_epochs_input, sample_negative_input, sample_prompt_input, save_every_epoch_input, save_every_steps_input, save_precision_dropdown, save_state_checkbox, save_state_end_checkbox, scheduler_dropdown, seed_input, shuffle_caption_checkbox, text_encoder_lr_input, torch_compile_checkbox, train_data_dir_input, unet_lr_input, vae_path_input, warmup_steps_input, xformers_checkbox):
    advanced_error = ""
    try:
        advanced_overrides: dict[str, Any] = json.loads(advanced_json_input.value or "{}")
        if not isinstance(advanced_overrides, dict):
            advanced_overrides = {}
            advanced_error = "高级 JSON 必须是对象。"
    except Exception as advanced_json_error:
        advanced_overrides = {}
        advanced_error = f"高级 JSON 解析失败：{type(advanced_json_error).__name__}: {advanced_json_error}"

    attention_backend = str(attention_backend_dropdown.value)
    config = {
        "schema_id": "sdxl-lora",
        "training_type": "lora",
        "model_type": "sdxl",
        "pretrained_model_name_or_path": str(model_path_input.value).strip(),
        "base_model_path": str(model_path_input.value).strip(),
        "vae_path": str(vae_path_input.value).strip(),
        "train_data_dir": str(train_data_dir_input.value).strip(),
        "output_dir": str(output_dir_input.value).strip(),
        "logging_dir": str(logging_dir_input.value).strip(),
        "output_name": str(output_name_input.value).strip() or "my_sdxl_lora",
        "network_module": "networks.lora",
        "network_dim": int(rank_input.value),
        "network_alpha": int(alpha_input.value),
        "network_dropout": float(dropout_input.value),
        "network_train_unet_only": False,
        "network_train_text_encoder_only": False,
        "resolution": str(resolution_dropdown.value),
        "enable_bucket": bool(enable_bucket_checkbox.value),
        "min_bucket_reso": int(min_bucket_input.value),
        "max_bucket_reso": int(max_bucket_input.value),
        "bucket_reso_steps": int(bucket_step_input.value),
        "bucket_no_upscale": False,
        "caption_extension": str(caption_ext_input.value or ".txt"),
        "shuffle_caption": bool(shuffle_caption_checkbox.value),
        "keep_tokens": int(keep_tokens_input.value),
        "max_token_length": int(max_token_length_input.value),
        "flip_aug": bool(flip_aug_checkbox.value),
        "color_aug": bool(color_aug_checkbox.value),
        "train_batch_size": int(batch_size_input.value),
        "gradient_accumulation_steps": int(grad_accum_input.value),
        "max_train_epochs": int(epochs_input.value),
        "max_train_steps": int(max_steps_input.value),
        "learning_rate": float(lr_input.value),
        "unet_lr": float(unet_lr_input.value),
        "text_encoder_lr": float(text_encoder_lr_input.value),
        "optimizer_type": str(optimizer_dropdown.value),
        "optimizer_args": "",
        "lr_scheduler": str(scheduler_dropdown.value),
        "lr_warmup_steps": int(warmup_steps_input.value),
        "lr_scheduler_num_cycles": 1,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
        "mixed_precision": str(mixed_precision_dropdown.value),
        "save_precision": str(save_precision_dropdown.value),
        "gradient_checkpointing": bool(grad_ckpt_checkbox.value),
        "cache_latents": bool(cache_latents_checkbox.value),
        "cache_latents_to_disk": bool(cache_latents_disk_checkbox.value),
        "cache_text_encoder_outputs": bool(cache_text_encoder_checkbox.value),
        "cache_text_encoder_outputs_to_disk": bool(cache_text_encoder_checkbox.value),
        "attention_backend": attention_backend,
        "sdpa": attention_backend == "sdpa",
        "xformers": bool(xformers_checkbox.value) or attention_backend == "xformers",
        "torch_compile": bool(torch_compile_checkbox.value),
        "save_model_as": "safetensors",
        "save_every_n_epochs": int(save_every_epoch_input.value),
        "save_every_n_steps": int(save_every_steps_input.value),
        "checkpoint_keep_last": int(keep_last_input.value),
        "save_state": bool(save_state_checkbox.value),
        "save_state_on_train_end": bool(save_state_end_checkbox.value),
        "seed": int(seed_input.value),
        "sample_every": bool(sample_every_checkbox.value),
        "sample_every_n_epochs": int(sample_every_epochs_input.value) if sample_every_checkbox.value else 0,
        "sample_prompts": str(sample_prompt_input.value or ""),
        "sample_negative": str(sample_negative_input.value or ""),
        "low_vram_profile": str(low_vram_dropdown.value),
        "sdxl_low_vram_optimization": str(low_vram_dropdown.value) != "off",
        "execution_core": "standard",
    }
    config.update(advanced_overrides)
    return advanced_error, advanced_overrides, config


@app.cell
def __(Path, caption_ext_input, config, config_name_input, core_entry, count_dataset, json, mo, repo, resolve_repo_path, runner, safe_config_name):
    mo.md("## 7. 预检与写入配置")

    write_config_button = mo.ui.run_button(label="写入/更新配置 JSON")
    preflight = []
    model_path = resolve_repo_path(str(config.get("pretrained_model_name_or_path") or ""))
    train_dir = resolve_repo_path(str(config.get("train_data_dir") or ""))
    output_dir = resolve_repo_path(str(config.get("output_dir") or "work/outputs"))
    logging_dir = resolve_repo_path(str(config.get("logging_dir") or "work/logs"))
    dataset_info = count_dataset(train_dir, str(caption_ext_input.value or ".txt"))

    preflight.append((core_entry.is_file(), f"训练核心：`{core_entry}`"))
    preflight.append((runner.is_file(), f"启动脚本：`{runner}`"))
    preflight.append((model_path.is_file(), f"SDXL 底模：`{model_path}`"))
    preflight.append((train_dir.is_dir(), f"训练集目录：`{train_dir}`"))
    preflight.append((int(dataset_info.get("images", 0)) > 0, f"图片数：`{dataset_info.get('images', 0)}`，caption 数：`{dataset_info.get('captions', 0)}`"))

    output_dir.mkdir(parents=True, exist_ok=True)
    logging_dir.mkdir(parents=True, exist_ok=True)

    config_path = repo / "configs" / f"{safe_config_name(str(config_name_input.value))}.json"
    write_message = ""
    if write_config_button.value:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        write_message = f"✅ 已写入：`{config_path}`"

    subset_preview = dataset_info.get("subsets", [])
    subset_text = "\n".join([f"  - `{item}`" for item in subset_preview]) or "  - 暂无"
    status_text = "\n".join([f"- {'✅' if ok else '❌'} {msg}" for ok, msg in preflight])

    mo.vstack(
        [
            mo.md(f"### 预检结果\n{status_text}\n\n### 数据集 subset 预览\n{subset_text}"),
            write_config_button,
            mo.md(write_message or f"配置路径将是：`{config_path}`"),
        ]
    )
    return config_path, dataset_info, logging_dir, model_path, output_dir, preflight, train_dir, write_config_button, write_message


@app.cell
def __(advanced_error, config, config_path, json, mo, q):
    config_preview = json.dumps(config, ensure_ascii=False, indent=2)
    command = f"python scripts/run_sdxl_train.py --config {q(config_path)}"
    warning = f"\n\n> ⚠️ {advanced_error}" if advanced_error else ""
    mo.md(
        f"""
        ## 8. 配置预览与启动命令{warning}

        ```bash
        {command}
        ```

        <details>
        <summary>展开最终 JSON 配置</summary>

        ```json
        {config_preview}
        ```
        </details>
        """
    )
    return command, config_preview


@app.cell
def __(mo):
    mo.md("## 9. 启动训练 / 打包输出")

    start_train_button = mo.ui.run_button(label="启动训练（会阻塞当前 notebook cell，长训练建议用上面的命令在 terminal 跑）")
    zip_outputs_button = mo.ui.run_button(label="打包 outputs 为 zip")
    mo.hstack([start_train_button, zip_outputs_button])
    return start_train_button, zip_outputs_button


@app.cell
def __(config_path, mo, os, repo, start_train_button, subprocess, sys):
    train_output = ""
    train_returncode = None
    if start_train_button.value:
        if not config_path.is_file():
            train_output = f"配置文件不存在，请先点击写入配置：{config_path}"
            train_returncode = 2
        else:
            env = os.environ.copy()
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            cmd = [sys.executable, "scripts/run_sdxl_train.py", "--config", str(config_path)]
            train_proc = subprocess.Popen(
                cmd,
                cwd=str(repo),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            lines: list[str] = []
            assert train_proc.stdout is not None
            for line in train_proc.stdout:
                print(line, end="")
                lines.append(line)
            train_returncode = train_proc.wait()
            train_output = "".join(lines[-300:])

    if train_output:
        mo.md(f"### 训练进程退出码：`{train_returncode}`\n\n```text\n{train_output}\n```")
    else:
        mo.md("点击启动训练后，这里会显示最近日志。")
    return train_output, train_returncode


@app.cell
def __(mo, output_dir, repo, time, zip_outputs_button, zipfile):
    zip_message = ""
    if zip_outputs_button.value:
        try:
            archive_dir = repo / "work" / "archives"
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir / f"outputs-{time.strftime('%Y%m%d-%H%M%S')}.zip"
            files = [p for p in output_dir.rglob("*") if p.is_file()]
            with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for file in files:
                    zf.write(file, file.relative_to(output_dir.parent))
            zip_message = f"✅ 已打包 {len(files)} 个文件：`{archive_path}`"
        except Exception as zip_error:
            zip_message = f"❌ 打包失败：{type(zip_error).__name__}: {zip_error}"

    mo.md(zip_message or "点击打包后，这里会显示 zip 路径。")
    return zip_message,


if __name__ == "__main__":
    app.run()
