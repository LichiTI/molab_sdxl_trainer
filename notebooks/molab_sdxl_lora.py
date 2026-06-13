# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "marimo",
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
    import re
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

    detected_repo = find_repo_root()
    detected_repo_ready = is_repo_root(detected_repo)

    github_repo_input = mo.ui.text(
        value="",
        label="如果当前只看到单个 py 文件，请粘贴完整 GitHub 仓库地址，例如 https://github.com/you/repo.git",
        full_width=True,
    )
    clone_dir_input = mo.ui.text(value="/marimo/lulynx-sdxl-trainer", label="clone 到这个目录", full_width=True)
    clone_repo_button = mo.ui.run_button(label="clone / 更新完整仓库")

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
                return "✅ 完整仓库已 clone/update。"
            return f"❌ git 失败，退出码 {git_result.returncode}\n\n```text\n{git_result.stdout}\n{git_result.stderr}\n```"
        except Exception as clone_error:
            return f"❌ clone/update 失败：{type(clone_error).__name__}: {clone_error}"

    def q(path: Path | str) -> str:
        return shlex.quote(str(path))

    detected_repo_status = "✅ 已找到完整训练仓库" if detected_repo_ready else "⚠️ 当前还没找到完整训练仓库，只找到了 notebook 工作目录"
    bootstrap_panel = [] if detected_repo_ready else [
        mo.callout(
            "如果左侧文件目录只有这个 `.py`，请在下面填 GitHub 仓库地址并点击 clone。clone 完后重新运行全部 cells。",
            kind="warn",
        ),
        mo.hstack([github_repo_input, clone_dir_input]),
        clone_repo_button,
    ]

    mo.vstack(
        [
            mo.md(
                f"""
                # Lulynx SDXL LoRA — molab 交互训练面板

                {detected_repo_status}

                当前检测到的目录：`{detected_repo}`

                这个 notebook 只控制 **SDXL LoRA 训练**，不依赖前端 `plugin/lora-scripts-ui-main`，也不启动 WebUI。
                """
            ),
            *bootstrap_panel,
        ]
    )
    return Any, Path, clone_dir_input, clone_or_update_repo, clone_repo_button, detected_repo, find_repo_root, github_repo_input, is_repo_root, json, mo, os, q, re, shlex, subprocess, sys, time, zipfile


@app.cell
def __(Path, clone_dir_input, clone_or_update_repo, clone_repo_button, detected_repo, find_repo_root, github_repo_input, is_repo_root, mo):
    bootstrap_message = ""
    repo = detected_repo

    if clone_repo_button.value:
        bootstrap_message = clone_or_update_repo(
            str(github_repo_input.value or "").strip(),
            Path(str(clone_dir_input.value or "/marimo/lulynx-sdxl-trainer")).expanduser(),
        )
        repo = find_repo_root()

    repo_is_ready = is_repo_root(repo)
    if repo_is_ready:
        for repo_work_dir in ["configs", "work/models", "work/datasets", "work/outputs", "work/logs", "work/runs", "work/archives"]:
            (repo / repo_work_dir).mkdir(parents=True, exist_ok=True)

    if bootstrap_message:
        mo.md(f"{bootstrap_message}\n\n当前训练仓库根目录：`{repo}`")
    else:
        mo.md(f"当前训练仓库根目录：`{repo}`")

    return bootstrap_message, repo, repo_is_ready


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
        device_capability = torch.cuda.get_device_capability(0) if cuda_available else None
        torch_arch_list = torch.cuda.get_arch_list() if cuda_available else []
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
        device_capability = None
        torch_arch_list = []
        vram_gb = 0

    nvidia_smi_button = mo.ui.run_button(label="刷新 nvidia-smi")
    blackwell_requirements = repo / "requirements-blackwell-cu128.txt"
    sage_requirements = repo / "requirements-sageattention.txt"
    install_command = f"{sys.executable} -m pip install -r {requirements}" if requirements.is_file() else "requirements-molab.txt 不存在"
    blackwell_install_command = (
        f"{sys.executable} -m pip install -r {blackwell_requirements}\n{sys.executable} -m pip install -r {requirements}"
        if blackwell_requirements.is_file() and requirements.is_file()
        else "requirements-blackwell-cu128.txt 或 requirements-molab.txt 不存在"
    )
    sage_install_command = (
        f"{blackwell_install_command}\n{sys.executable} -m pip install -r {sage_requirements}"
        if blackwell_requirements.is_file() and requirements.is_file() and sage_requirements.is_file()
        else "requirements-blackwell-cu128.txt / requirements-molab.txt / requirements-sageattention.txt 不存在"
    )

    sageattention_available = False
    try:
        import sageattention  # noqa: F401
        sageattention_available = True
    except Exception:
        sageattention_available = False

    status_lines = [
        f"- core/entry_train.py：{'✅' if core_entry.is_file() else '❌'} `{core_entry}`",
        f"- scripts/run_sdxl_train.py：{'✅' if runner.is_file() else '❌'} `{runner}`",
        f"- requirements-molab.txt：{'✅' if requirements.is_file() else '❌'} `{requirements}`",
        f"- requirements-sageattention.txt：{'✅' if sage_requirements.is_file() else '❌'} `{sage_requirements}`",
        f"- Python：`{sys.version.split()[0]}`",
        f"- Torch：`{torch_version}`",
        f"- CUDA：{'✅ 可用' if cuda_available else '❌ 不可用'}，CUDA 版本：`{cuda_version}`",
        f"- GPU：`{gpu_name}`，显存：`{vram_gb} GB`",
        f"- Compute capability：`{device_capability}`",
        f"- Torch arch list：`{torch_arch_list}`",
        f"- SageAttention：{'✅ 可导入' if sageattention_available else '⚠️ 未安装或不可导入'}",
    ]

    mo.vstack(
        [
            mo.md("## 1. 环境与仓库检查\n" + "\n".join(status_lines)),
            mo.callout(
                f"如果是 RTX PRO 6000 / Blackwell，优先执行 **SageAttention 快速路线**：\n\n```bash\n{sage_install_command}\n```\n\n如果 SageAttention 安装失败，则执行基础 cu128 路线并使用 SDPA：\n\n```bash\n{blackwell_install_command}\n```\n\n普通 CUDA 环境可执行：\n\n```bash\n{install_command}\n```\n\nBlackwell 正常应看到 CUDA `12.8` 且 arch list 包含 `sm_120` 或 `compute_120`。SageAttention 安装成功后，训练参数里的 attention backend 可选 `sageattn`。",
                kind="info",
            ),
            nvidia_smi_button,
        ]
    )
    return blackwell_install_command, core_entry, cuda_available, gpu_name, install_command, nvidia_smi_button, requirements, runner, sage_install_command, sage_requirements, sageattention_available, torch_version, vram_gb


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
def __(mo):
    mo.md("## 1B. 手动安装 / 更新运行时")

    runtime_install_profile_dropdown = mo.ui.dropdown(
        options=[
            "blackwell_sageattention_recommended",
            "blackwell_cu128_sdpa_stable",
            "sageattention_only",
            "base_requirements_only",
        ],
        value="blackwell_sageattention_recommended",
        label="安装方案",
    )
    runtime_install_show_only_checkbox = mo.ui.checkbox(
        value=True,
        label="只显示命令，不执行；确认无误后取消勾选再点安装",
    )
    runtime_install_button = mo.ui.run_button(label="执行所选安装方案")

    mo.vstack(
        [
            mo.callout(
                "推荐 RTX PRO 6000 / Blackwell 使用 `blackwell_sageattention_recommended`。如果安装 SageAttention 失败，改用 `blackwell_cu128_sdpa_stable`。安装/替换 torch 后建议重启 notebook runtime 再继续训练。",
                kind="warn",
            ),
            mo.hstack([runtime_install_profile_dropdown, runtime_install_show_only_checkbox]),
            runtime_install_button,
        ]
    )
    return runtime_install_button, runtime_install_profile_dropdown, runtime_install_show_only_checkbox


@app.cell
def __(blackwell_install_command, install_command, mo, runtime_install_button, runtime_install_profile_dropdown, runtime_install_show_only_checkbox, sage_install_command, sage_requirements, subprocess, sys):
    runtime_install_profile = str(runtime_install_profile_dropdown.value or "blackwell_sageattention_recommended")
    runtime_install_commands = {
        "blackwell_sageattention_recommended": sage_install_command,
        "blackwell_cu128_sdpa_stable": blackwell_install_command,
        "sageattention_only": f"{sys.executable} -m pip install -r {sage_requirements}",
        "base_requirements_only": install_command,
    }
    runtime_install_command_text = runtime_install_commands.get(runtime_install_profile, sage_install_command)
    runtime_install_message = ""

    if runtime_install_button.value:
        if runtime_install_show_only_checkbox.value:
            runtime_install_message = (
                "当前为只显示命令模式，未执行安装。取消勾选后再次点击按钮即可执行。\n\n"
                f"```bash\n{runtime_install_command_text}\n```"
            )
        else:
            runtime_install_result = subprocess.run(
                runtime_install_command_text,
                shell=True,
                text=True,
                capture_output=True,
                timeout=7200,
            )
            runtime_install_stdout = (runtime_install_result.stdout or "")[-8000:]
            runtime_install_stderr = (runtime_install_result.stderr or "")[-8000:]
            runtime_install_message = (
                f"安装方案：`{runtime_install_profile}`\n\n"
                f"退出码：`{runtime_install_result.returncode}`\n\n"
                "安装/替换 torch 后，请重启 notebook runtime 或重新打开 notebook，再重新运行环境检查。\n\n"
                f"执行命令：\n```bash\n{runtime_install_command_text}\n```\n\n"
                f"stdout/stderr 尾部：\n```text\n{runtime_install_stdout}\n{runtime_install_stderr}\n```"
            )
    else:
        runtime_install_message = (
            "选择安装方案后点击按钮。当前方案对应命令：\n\n"
            f"```bash\n{runtime_install_command_text}\n```"
        )

    mo.md(runtime_install_message)
    return runtime_install_command_text, runtime_install_message, runtime_install_profile


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
    mo.md("## 2B. 可选：上传训练集 ZIP 后解压")

    dataset_zip_path_input = mo.ui.text(
        value="",
        label="训练集 ZIP 路径：上传到 molab 后填写，例如 dataset.zip 或 /marimo/dataset.zip",
        full_width=True,
    )
    dataset_extract_name_input = mo.ui.text(
        value="my_lora",
        label="解压到 work/datasets/<这个名字>",
        full_width=True,
    )
    dataset_repeats_input = mo.ui.number(value=10, start=1, stop=1000, step=1, label="如果 zip 根目录直接是图片，自动整理到几_repeats目录")
    dataset_use_zip_checkbox = mo.ui.checkbox(value=True, label="解压成功后，最终配置自动使用解压后的训练集路径")
    dataset_overwrite_checkbox = mo.ui.checkbox(value=False, label="覆盖已存在的解压目录")
    dataset_unpack_button = mo.ui.run_button(label="解压训练集 ZIP")

    mo.vstack(
        [
            dataset_zip_path_input,
            mo.hstack([dataset_extract_name_input, dataset_repeats_input]),
            mo.hstack([dataset_use_zip_checkbox, dataset_overwrite_checkbox]),
            mo.callout(
                "支持两种 zip：1) 内部已有 `10_xxx/图片+txt`；2) 根目录直接是图片+txt，会自动整理到 `10_default/`。",
                kind="info",
            ),
            dataset_unpack_button,
        ]
    )
    return dataset_extract_name_input, dataset_overwrite_checkbox, dataset_repeats_input, dataset_unpack_button, dataset_use_zip_checkbox, dataset_zip_path_input


@app.cell
def __(Path, dataset_extract_name_input, dataset_overwrite_checkbox, dataset_repeats_input, dataset_unpack_button, dataset_use_zip_checkbox, dataset_zip_path_input, mo, repo, zipfile):
    def safe_dataset_name(raw_name: str) -> str:
        cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(raw_name or "my_lora")).strip("_")
        return cleaned or "my_lora"

    def resolve_upload_path(raw_path: str) -> Path:
        path_text = str(raw_path or "").strip()
        if not path_text:
            return Path("")
        candidate = Path(path_text).expanduser()
        if candidate.is_absolute():
            return candidate
        # Prefer repo-relative path, then /marimo-relative path.
        repo_candidate = repo / candidate
        if repo_candidate.exists():
            return repo_candidate
        marimo_candidate = Path("/marimo") / candidate
        if marimo_candidate.exists():
            return marimo_candidate
        return repo_candidate

    def safe_extract_zip(zip_path: Path, destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        destination_resolved = destination.resolve()
        with zipfile.ZipFile(zip_path, "r") as archive:
            for member in archive.infolist():
                member_name = member.filename.replace("\\\\", "/")
                if not member_name or member_name.endswith("/"):
                    continue
                member_path = Path(member_name)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise RuntimeError(f"ZIP 内含不安全路径：{member.filename}")
                if member_path.parts and member_path.parts[0] == "__MACOSX":
                    continue
                target_path = (destination / member_path).resolve()
                if not str(target_path).startswith(str(destination_resolved)):
                    raise RuntimeError(f"ZIP 路径越界：{member.filename}")
                target_path.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member, "r") as source_handle:
                    target_path.write_bytes(source_handle.read())

    def normalize_dataset_root(extract_dir: Path, repeats: int) -> Path:
        import re
        import shutil

        image_exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        subset_pattern = re.compile(r"^\d+_.+")

        def visible_entries(directory: Path) -> list[Path]:
            return [p for p in directory.iterdir() if not p.name.startswith(".") and p.name != "__MACOSX"]

        def has_direct_images(directory: Path) -> bool:
            return any(p.is_file() and p.suffix.lower() in image_exts for p in directory.iterdir())

        def has_subset_dirs(directory: Path) -> bool:
            return any(p.is_dir() and subset_pattern.match(p.name) for p in directory.iterdir())

        def organize_into_subset(directory: Path) -> None:
            subset_dir = directory / f"{int(repeats)}_default"
            subset_dir.mkdir(parents=True, exist_ok=True)
            sidecar_suffixes = [".txt", ".json", ".caption"]
            root_images = [p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in image_exts]
            for image_path in root_images:
                shutil.move(str(image_path), str(subset_dir / image_path.name))
                for suffix in sidecar_suffixes:
                    sidecar = image_path.with_suffix(suffix)
                    if sidecar.exists():
                        shutil.move(str(sidecar), str(subset_dir / sidecar.name))

        # Case A: extract_dir already contains subset dirs like 10_xxx -> use extract_dir as is.
        if has_subset_dirs(extract_dir):
            return extract_dir

        visible_children = visible_entries(extract_dir)
        dir_children = [p for p in visible_children if p.is_dir()]
        file_children = [p for p in visible_children if p.is_file()]

        # Case B: a single top-level directory inside the zip.
        if len(dir_children) == 1 and not file_children:
            inner = dir_children[0]
            # B1: inner is already a subset dir (e.g. 5_xt) -> extract_dir is the dataset root.
            if subset_pattern.match(inner.name):
                return extract_dir
            # B2: inner already contains subset dirs -> inner is the dataset root.
            if has_subset_dirs(inner):
                return inner
            # B3: inner directly holds images -> organize into <repeats>_default under inner.
            if has_direct_images(inner):
                organize_into_subset(inner)
                return inner
            return inner

        # Case C: zip root directly holds images (no subset dirs) -> organize under extract_dir.
        if has_direct_images(extract_dir):
            organize_into_subset(extract_dir)
            return extract_dir

        return extract_dir

    dataset_zip_message = ""
    zip_train_data_dir = str(dataset_extract_name_input.value or "my_lora")
    effective_train_data_dir = ""

    if dataset_unpack_button.value:
        source_zip_path = resolve_upload_path(str(dataset_zip_path_input.value or ""))
        if not source_zip_path.is_file():
            dataset_zip_message = f"❌ ZIP 不存在：`{source_zip_path}`"
        elif source_zip_path.suffix.lower() != ".zip":
            dataset_zip_message = f"❌ 不是 .zip 文件：`{source_zip_path}`"
        else:
            target_name = safe_dataset_name(str(dataset_extract_name_input.value or "my_lora"))
            extract_dir = repo / "work" / "datasets" / target_name
            if extract_dir.exists() and any(extract_dir.iterdir()) and not dataset_overwrite_checkbox.value:
                dataset_zip_message = f"❌ 目标目录已存在且非空：`{extract_dir}`。如需覆盖，请勾选覆盖。"
            else:
                if extract_dir.exists() and dataset_overwrite_checkbox.value:
                    import shutil
                    shutil.rmtree(extract_dir)
                try:
                    safe_extract_zip(source_zip_path, extract_dir)
                    detected_dataset_root = normalize_dataset_root(extract_dir, int(dataset_repeats_input.value or 10))
                    try:
                        zip_train_data_dir = str(detected_dataset_root.relative_to(repo))
                    except ValueError:
                        zip_train_data_dir = str(detected_dataset_root)
                    effective_train_data_dir = zip_train_data_dir if dataset_use_zip_checkbox.value else ""
                    dataset_zip_message = (
                        f"✅ 已解压：`{source_zip_path}` → `{extract_dir}`\n\n"
                        f"检测到训练集根目录：`{zip_train_data_dir}`\n\n"
                        f"如果需要手动填写，训练集目录填：`{zip_train_data_dir}`"
                    )
                except Exception as dataset_zip_error:
                    dataset_zip_message = f"❌ 解压失败：{type(dataset_zip_error).__name__}: {dataset_zip_error}"

    mo.md(dataset_zip_message or "未触发 ZIP 解压。")
    return dataset_zip_message, effective_train_data_dir, zip_train_data_dir

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
    mo.md("## 3B. 可选：直接 URL / Civitai 下载模型")

    direct_model_url_input = mo.ui.text(
        value="",
        label="模型下载 URL：支持普通直链，也支持 Civitai download URL",
        full_width=True,
    )
    direct_model_filename_input = mo.ui.text(
        value="model.safetensors",
        label="保存文件名，建议 .safetensors；如果 URL 不能推断文件名，请务必填写",
        full_width=True,
    )
    direct_model_local_dir_input = mo.ui.text(value="work/models", label="保存目录", full_width=True)
    direct_model_token_env_input = mo.ui.text(
        value="",
        label="可选 Token 环境变量名：例如 CIVITAI_TOKEN；公开直链留空",
        full_width=True,
    )
    direct_model_auth_mode_dropdown = mo.ui.dropdown(
        options=["none", "bearer_header", "civitai_query"],
        value="none",
        label="Token 使用方式：无 / Authorization Bearer / Civitai ?token=",
    )
    direct_model_download_button = mo.ui.run_button(label="下载 URL / Civitai 模型到 work/models")

    mo.vstack(
        [
            direct_model_url_input,
            mo.hstack([direct_model_filename_input, direct_model_local_dir_input]),
            mo.hstack([direct_model_token_env_input, direct_model_auth_mode_dropdown]),
            mo.callout(
                "Token 不要写进 GitHub。需要鉴权时，先在终端执行 `export CIVITAI_TOKEN=你的token`，然后这里填写环境变量名 `CIVITAI_TOKEN`。",
                kind="info",
            ),
            direct_model_download_button,
        ]
    )
    return direct_model_auth_mode_dropdown, direct_model_download_button, direct_model_filename_input, direct_model_local_dir_input, direct_model_token_env_input, direct_model_url_input


@app.cell
def __(Path, direct_model_auth_mode_dropdown, direct_model_download_button, direct_model_filename_input, direct_model_local_dir_input, direct_model_token_env_input, direct_model_url_input, mo, os, re, repo, subprocess):
    def normalize_civitai_url(model_url: str) -> tuple[str, str]:
        """Convert a Civitai web page URL into the real download API URL.

        Returns (effective_url, note). The note is non-empty when normalization happened.
        """
        url = model_url.strip()
        # Already an API download link.
        if "/api/download/models/" in url:
            return url.replace("civitai.red", "civitai.com"), ""
        # Civitai page URL such as https://civitai.com/models/833294/...?modelVersionId=1022833
        if "civitai" in url and "/models/" in url:
            version_match = re.search(r"[?&]modelVersionId=(\d+)", url)
            if version_match:
                version_id = version_match.group(1)
                return f"https://civitai.com/api/download/models/{version_id}", f"检测到 Civitai 页面 URL，已自动转为下载直链 modelVersionId={version_id}。"
            return url, "⚠️ 这看起来是 Civitai 页面 URL，但没有 modelVersionId。请使用带 ?modelVersionId=的链接，或直接用 https://civitai.com/api/download/models/<版本ID>。"
        return url, ""

    def looks_like_html(target_path: Path) -> bool:
        try:
            with open(target_path, "rb") as handle:
                head = handle.read(512).lstrip().lower()
            return head.startswith(b"<!doctype html") or head.startswith(b"<html") or b"<head" in head[:256]
        except Exception:
            return False

    def download_direct_model(model_url: str, output_name: str, output_dir_text: str, token_env_name: str, auth_mode: str) -> str:
        if not model_url:
            return "❌ 请先填写模型下载 URL。"

        direct_output_dir = Path(output_dir_text or "work/models").expanduser()
        if not direct_output_dir.is_absolute():
            direct_output_dir = repo / direct_output_dir
        direct_output_dir.mkdir(parents=True, exist_ok=True)

        clean_output_name = (output_name or "model.safetensors").strip().split("/")[-1].split("\\\\")[-1]
        if not clean_output_name:
            clean_output_name = "model.safetensors"
        direct_target_path = direct_output_dir / clean_output_name

        effective_url, normalize_note = normalize_civitai_url(model_url)

        direct_token_value = os.environ.get((token_env_name or "").strip(), "") if token_env_name else ""
        curl_command = ["curl", "-L", "--fail", "--retry", "3", "--continue-at", "-", "-o", str(direct_target_path)]

        if auth_mode == "bearer_header" and direct_token_value:
            curl_command.extend(["-H", f"Authorization: Bearer {direct_token_value}"])
        elif auth_mode == "civitai_query" and direct_token_value:
            join_char = "&" if "?" in effective_url else "?"
            effective_url = f"{effective_url}{join_char}token={direct_token_value}"

        curl_command.append(effective_url)
        note_prefix = (normalize_note + "\n\n") if normalize_note else ""

        try:
            direct_download_result = subprocess.run(
                curl_command,
                text=True,
                capture_output=True,
                timeout=7200,
            )
        except Exception as direct_download_error:
            return f"{note_prefix}❌ 下载进程启动失败：{type(direct_download_error).__name__}: {direct_download_error}"

        if direct_download_result.returncode != 0:
            sanitized_stderr = (direct_download_result.stderr or "").replace(direct_token_value, "***") if direct_token_value else (direct_download_result.stderr or "")
            sanitized_stdout = (direct_download_result.stdout or "").replace(direct_token_value, "***") if direct_token_value else (direct_download_result.stdout or "")
            return f"{note_prefix}❌ 下载失败，退出码 {direct_download_result.returncode}\n\n```text\n{sanitized_stdout}\n{sanitized_stderr}\n```"

        if not direct_target_path.exists():
            return f"{note_prefix}❌ 下载后未找到文件：`{direct_target_path}`"

        size_bytes = direct_target_path.stat().st_size
        size_gb = size_bytes / 1024**3
        size_mb = size_bytes / 1024**2

        # Guard: SDXL checkpoints are multi-GB. A tiny file is almost always an HTML page or an error JSON.
        if size_bytes < 50 * 1024 * 1024 or looks_like_html(direct_target_path):
            preview = ""
            try:
                with open(direct_target_path, "rb") as handle:
                    preview = handle.read(300).decode("utf-8", errors="replace")
            except Exception:
                preview = ""
            try:
                direct_target_path.unlink()
            except Exception:
                pass
            return (
                f"{note_prefix}❌ 下载结果只有 `{size_mb:.2f} MB`，看起来不是模型文件（可能是网页 HTML 或错误信息），已自动删除。\n\n"
                f"常见原因：填了 Civitai 网页地址而不是下载直链，或者模型需要登录 token。\n\n"
                f"正确直链格式：`https://civitai.com/api/download/models/<版本ID>`\n\n"
                f"返回内容预览：\n\n```text\n{preview}\n```"
            )

        return f"{note_prefix}✅ 下载完成：`{direct_target_path}`，大小约 `{size_gb:.2f} GB`。上面的底模路径可填写：`{direct_target_path}`"

    direct_model_download_message = ""
    if direct_model_download_button.value:
        direct_model_download_message = download_direct_model(
            str(direct_model_url_input.value or "").strip(),
            str(direct_model_filename_input.value or "model.safetensors").strip(),
            str(direct_model_local_dir_input.value or "work/models").strip(),
            str(direct_model_token_env_input.value or "").strip(),
            str(direct_model_auth_mode_dropdown.value or "none"),
        )

    mo.md(direct_model_download_message or "未触发 URL 下载。")
    return direct_model_download_message,

@app.cell
def __(mo):
    mo.md("## 4. 训练参数")

    resolution_dropdown = mo.ui.dropdown(
        options=["1024,1024", "896,1152", "832,1216", "768,1344", "768,768"],
        value="1024,1024",
        label="基础分辨率",
    )
    adapter_type_dropdown = mo.ui.dropdown(
        options=["lora", "lokr", "loha", "locon", "dora"],
        value="lora",
        label="Adapter 类型（lokr/loha/locon/dora 走 LyCORIS）",
    )
    lokr_factor_input = mo.ui.number(value=8, start=-1, stop=64, step=1, label="LoKr factor（仅 lokr，-1=自动）")
    lokr_full_matrix_checkbox = mo.ui.checkbox(value=False, label="LoKr full matrix（强制全矩阵，不退回分解）")
    rank_input = mo.ui.number(value=16, start=1, stop=4096, step=1, label="rank / network_dim")
    alpha_input = mo.ui.number(value=8, start=0, stop=4096, step=1, label="alpha")
    dropout_input = mo.ui.number(value=0.0, start=0.0, stop=1.0, step=0.01, label="dropout")
    conv_dim_input = mo.ui.number(value=0, start=0, stop=4096, step=1, label="LyCORIS conv_dim（0=同 rank，仅 LyCORIS）")
    conv_alpha_input = mo.ui.number(value=0, start=0, stop=4096, step=1, label="LyCORIS conv_alpha（0=同 alpha）")
    train_norm_checkbox = mo.ui.checkbox(value=False, label="训练 Norm 层（LyCORIS train_norm）")

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
    prodigy_d0_input = mo.ui.text(value="1e-6", label="Prodigy d0（仅 Prodigy，默认 1e-6）", full_width=True)
    prodigy_d_coef_input = mo.ui.text(value="1.0", label="Prodigy d_coef（仅 Prodigy，默认 1.0）", full_width=True)

    mo.vstack(
        [
            mo.hstack([adapter_type_dropdown, lokr_factor_input, lokr_full_matrix_checkbox]),
            mo.hstack([conv_dim_input, conv_alpha_input, train_norm_checkbox]),
            mo.hstack([resolution_dropdown, rank_input, alpha_input, dropout_input]),
            mo.hstack([epochs_input, max_steps_input, batch_size_input, grad_accum_input]),
            mo.hstack([lr_input, unet_lr_input, text_encoder_lr_input]),
            mo.hstack([optimizer_dropdown, scheduler_dropdown, warmup_steps_input, seed_input]),
            mo.hstack([prodigy_d0_input, prodigy_d_coef_input]),
        ]
    )
    return adapter_type_dropdown, alpha_input, batch_size_input, conv_alpha_input, conv_dim_input, dropout_input, epochs_input, grad_accum_input, lokr_factor_input, lokr_full_matrix_checkbox, lr_input, max_steps_input, optimizer_dropdown, prodigy_d0_input, prodigy_d_coef_input, rank_input, resolution_dropdown, scheduler_dropdown, seed_input, text_encoder_lr_input, train_norm_checkbox, unet_lr_input, warmup_steps_input


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
    attention_backend_dropdown = mo.ui.dropdown(options=["sageattn", "sdpa", "auto", "xformers"], value="sageattn", label="attention backend，Blackwell 推荐 sageattn；不可用会回退 SDPA")
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
        if p.is_absolute():
            return p
        # 防止路径重复：如果输入以仓库名开头（例如 lulynx-sdxl-trainer/work/...），去掉该前缀。
        parts = p.parts
        if parts and parts[0] == repo.name:
            p = Path(*parts[1:]) if len(parts) > 1 else Path(".")
        return repo / p

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
def __(Any, adapter_type_dropdown, advanced_json_input, alpha_input, attention_backend_dropdown, batch_size_input, bucket_step_input, cache_latents_checkbox, cache_latents_disk_checkbox, cache_text_encoder_checkbox, caption_ext_input, color_aug_checkbox, config_name_input, conv_alpha_input, conv_dim_input, dropout_input, effective_train_data_dir, enable_bucket_checkbox, epochs_input, flip_aug_checkbox, grad_accum_input, grad_ckpt_checkbox, json, keep_last_input, keep_tokens_input, logging_dir_input, lokr_factor_input, lokr_full_matrix_checkbox, low_vram_dropdown, lr_input, max_bucket_input, max_steps_input, max_token_length_input, min_bucket_input, mixed_precision_dropdown, model_path_input, optimizer_dropdown, output_dir_input, output_name_input, prodigy_d0_input, prodigy_d_coef_input, rank_input, resolution_dropdown, sample_every_checkbox, sample_every_epochs_input, sample_negative_input, sample_prompt_input, save_every_epoch_input, save_every_steps_input, save_precision_dropdown, save_state_checkbox, save_state_end_checkbox, scheduler_dropdown, seed_input, shuffle_caption_checkbox, text_encoder_lr_input, torch_compile_checkbox, train_data_dir_input, train_norm_checkbox, unet_lr_input, vae_path_input, warmup_steps_input, xformers_checkbox):
    advanced_error = ""
    try:
        advanced_overrides: dict[str, Any] = json.loads(advanced_json_input.value or "{}")
        if not isinstance(advanced_overrides, dict):
            advanced_overrides = {}
            advanced_error = "高级 JSON 必须是对象。"
    except Exception as advanced_json_error:
        advanced_overrides = {}
        advanced_error = f"高级 JSON 解析失败：{type(advanced_json_error).__name__}: {advanced_json_error}"

    # marimo 的 number 输入框被清空时 value 会变成 None，int(None)/float(None) 会报 TypeError。
    def _ival(widget, default=0):
        try:
            v = widget.value
            return int(v) if v is not None else int(default)
        except (TypeError, ValueError):
            return int(default)

    def _fval(widget, default=0.0):
        try:
            v = widget.value
            return float(v) if v is not None else float(default)
        except (TypeError, ValueError):
            return float(default)

    def _fval_text(widget, default=0.0):
        try:
            text = str(widget.value).strip()
            return float(text) if text else float(default)
        except (TypeError, ValueError):
            return float(default)

    attention_backend = str(attention_backend_dropdown.value)
    selected_train_data_dir = str(effective_train_data_dir or train_data_dir_input.value or "").strip()

    adapter_type = str(adapter_type_dropdown.value or "lora").strip().lower()
    # lora -> sd-scripts networks.lora; 其余走 LyCORIS。
    _lycoris_algo_map = {"lokr": "lokr", "loha": "loha", "locon": "locon", "dora": "dora"}
    if adapter_type == "lora":
        adapter_network_module = "networks.lora"
        adapter_lycoris_algo = ""
    else:
        adapter_network_module = "lycoris.locon"
        adapter_lycoris_algo = _lycoris_algo_map.get(adapter_type, "lokr")

    is_lycoris_adapter = adapter_type != "lora"
    optimizer_is_prodigy = str(optimizer_dropdown.value or "").strip().lower() == "prodigy"

    config = {
        "schema_id": "sdxl-lora",
        "training_type": "lora",
        "model_type": "sdxl",
        "pretrained_model_name_or_path": str(model_path_input.value).strip(),
        "base_model_path": str(model_path_input.value).strip(),
        "vae_path": str(vae_path_input.value).strip(),
        "train_data_dir": selected_train_data_dir,
        "output_dir": str(output_dir_input.value).strip(),
        "logging_dir": str(logging_dir_input.value).strip(),
        "output_name": str(output_name_input.value).strip() or "my_sdxl_lora",
        "network_module": adapter_network_module,
        "network_dim": _ival(rank_input, 16),
        "network_alpha": _ival(alpha_input, 8),
        "network_dropout": _fval(dropout_input, 0.0),
        "network_train_unet_only": False,
        "network_train_text_encoder_only": False,
        "resolution": str(resolution_dropdown.value),
        "enable_bucket": bool(enable_bucket_checkbox.value),
        "min_bucket_reso": _ival(min_bucket_input, 512),
        "max_bucket_reso": _ival(max_bucket_input, 1536),
        "bucket_reso_steps": _ival(bucket_step_input, 64),
        "bucket_no_upscale": False,
        "caption_extension": str(caption_ext_input.value or ".txt"),
        "shuffle_caption": bool(shuffle_caption_checkbox.value),
        "keep_tokens": _ival(keep_tokens_input, 0),
        "max_token_length": _ival(max_token_length_input, 225),
        "flip_aug": bool(flip_aug_checkbox.value),
        "color_aug": bool(color_aug_checkbox.value),
        "train_batch_size": _ival(batch_size_input, 1),
        "gradient_accumulation_steps": _ival(grad_accum_input, 1),
        "max_train_epochs": _ival(epochs_input, 10),
        "max_train_steps": _ival(max_steps_input, 0),
        "learning_rate": _fval(lr_input, 1e-4),
        "unet_lr": _fval(unet_lr_input, 1e-4),
        "text_encoder_lr": _fval(text_encoder_lr_input, 1e-5),
        "optimizer_type": str(optimizer_dropdown.value),
        "optimizer_args": "",
        "lr_scheduler": str(scheduler_dropdown.value),
        "lr_warmup_steps": _ival(warmup_steps_input, 0),
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
        "use_sage_attn": attention_backend == "sageattn",
        "sdpa": attention_backend == "sdpa",
        "xformers": bool(xformers_checkbox.value) or attention_backend == "xformers",
        "torch_compile": bool(torch_compile_checkbox.value),
        "save_model_as": "safetensors",
        "save_every_n_epochs": _ival(save_every_epoch_input, 1),
        "save_every_n_steps": _ival(save_every_steps_input, 0),
        "checkpoint_keep_last": _ival(keep_last_input, 0),
        "save_state": bool(save_state_checkbox.value),
        "save_state_on_train_end": bool(save_state_end_checkbox.value),
        "seed": _ival(seed_input, 42),
        "sample_every": bool(sample_every_checkbox.value),
        "sample_every_n_epochs": _ival(sample_every_epochs_input, 0) if sample_every_checkbox.value else 0,
        "sample_prompts": str(sample_prompt_input.value or ""),
        "sample_negative": str(sample_negative_input.value or ""),
        "low_vram_profile": str(low_vram_dropdown.value),
        "sdxl_low_vram_optimization": str(low_vram_dropdown.value) != "off",
        "adapter_type": adapter_type,
        "lycoris_algo": adapter_lycoris_algo,
        "lycoris_lokr_factor": _ival(lokr_factor_input, 8) if adapter_type == "lokr" else 0,
        "lokr_full_matrix": bool(lokr_full_matrix_checkbox.value) and adapter_type == "lokr",
        "lycoris_train_norm": bool(train_norm_checkbox.value) and is_lycoris_adapter,
        "lycoris_conv_dim": _ival(conv_dim_input, 0),
        "lycoris_conv_alpha": _fval(conv_alpha_input, 0.0),
        "prodigy_d0": _fval_text(prodigy_d0_input, 1e-6),
        "prodigy_d_coef": _fval_text(prodigy_d_coef_input, 1.0),
        "execution_core": "standard",
    }
    if not adapter_lycoris_algo:
        config.pop("lycoris_algo", None)
        config.pop("lycoris_lokr_factor", None)
    if adapter_type != "lokr":
        config.pop("lokr_full_matrix", None)
    if not is_lycoris_adapter:
        # LoRA 路线不传 LyCORIS 专属字段，避免误导。
        config.pop("lycoris_train_norm", None)
        config.pop("lycoris_conv_dim", None)
        config.pop("lycoris_conv_alpha", None)
    if not optimizer_is_prodigy:
        config.pop("prodigy_d0", None)
        config.pop("prodigy_d_coef", None)
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

    subset_preview = dataset_info.get("subsets", [])
    subset_text = "\n".join([f"  - `{item}`" for item in subset_preview]) or "  - 暂无"
    status_text = "\n".join([f"- {'✅' if ok else '❌'} {msg}" for ok, msg in preflight])

    mo.vstack(
        [
            mo.md(f"### 预检结果\n{status_text}\n\n### 数据集 subset 预览\n{subset_text}"),
            write_config_button,
            mo.md(f"配置路径将是：`{config_path}`"),
        ]
    )
    return config_path, dataset_info, logging_dir, model_path, output_dir, preflight, train_dir, write_config_button


@app.cell
def __(config, config_path, json, mo, write_config_button):
    write_config_message = ""
    if write_config_button.value:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        write_config_message = f"✅ 已写入：`{config_path}`"

    mo.md(write_config_message or "点击上面的按钮后，这里会显示配置写入结果。")
    return write_config_message,


@app.cell
def __(advanced_error, config, config_path, json, mo, q, repo):
    config_preview = json.dumps(config, ensure_ascii=False, indent=2)
    # 展示「从仓库父目录运行」的命令：python <仓库名>/scripts/run_sdxl_train.py --config configs/xxx.json。
    # run_sdxl_train.py 内部会把相对 --config 按脚本所在仓库根解析，与当前 cwd 无关。
    script_rel = f"{repo.name}/scripts/run_sdxl_train.py"
    try:
        config_rel = config_path.relative_to(repo).as_posix()
    except ValueError:
        config_rel = config_path.as_posix()
    command = f"python {q(script_rel)} --config {q(config_rel)}"
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
