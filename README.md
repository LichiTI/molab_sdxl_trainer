# Lulynx SDXL LoRA Molab Trainer

这是从 Lulynx Trainer 后端中拆出的 **SDXL LoRA 训练专用 molab 仓库**。

本仓库不包含前端 `plugin/lora-scripts-ui-main`，也不启动 WebUI / Launcher。molab 上直接使用 marimo notebook 或命令行控制训练。

## 仓库结构

```text
core/                         # 训练核心，来自原项目 backend/core
configs/
  sdxl_lora_minimal.json       # 最小 SDXL LoRA 配置模板
notebooks/
  molab_sdxl_lora.py           # molab / marimo 交互训练面板
scripts/
  run_sdxl_train.py            # 命令行训练启动器
requirements-molab.txt         # molab 依赖
work/
  models/                      # 放 SDXL 底模，不要提交到 GitHub
  datasets/                    # 放训练集，不要提交到 GitHub
  outputs/                     # 训练输出，不要提交到 GitHub
  logs/                        # 日志，不要提交到 GitHub
  runs/                        # 每次训练的运行配置和状态
```

只要仓库根目录下存在：

```text
core/entry_train.py
scripts/run_sdxl_train.py
```

就不再依赖原始大项目工作区。

## molab 快速使用

在 molab 中 clone 你的仓库：

```bash
git clone https://github.com/<你的用户名>/<你的仓库名>.git
cd <你的仓库名>
```

安装依赖。RTX PRO 6000 / Blackwell 推荐优先使用 SageAttention 快速路线：

```bash
python -m pip install -r requirements-blackwell-cu128.txt
python -m pip install -r requirements-molab.txt
python -m pip install -r requirements-sageattention.txt
```

如果 SageAttention 安装失败，可只安装基础 cu128 + SDPA 路线：

```bash
python -m pip install -r requirements-blackwell-cu128.txt
python -m pip install -r requirements-molab.txt
```

普通非 Blackwell CUDA 环境可尝试：

```bash
pip install -r requirements-molab.txt
```

检查 GPU：

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
```

推荐打开交互 notebook：

```text
notebooks/molab_sdxl_lora.py
```

在 notebook 中可以完成：

1. 检查 Python / Torch / CUDA / GPU
2. 通过面板手动安装 / 更新 Blackwell、SageAttention 或基础运行时
3. 可选从 Hugging Face 下载 SDXL 底模
4. 可选从普通直链 / Civitai download URL 下载模型到 `work/models/`
5. 上传训练集 zip 后一键解压并自动识别训练集目录
5. 填写模型路径、数据集路径、输出名
6. 设置 rank、alpha、分辨率、epoch、学习率、optimizer
7. 设置 bucket、cache、混合精度、低显存策略
8. 预检模型和数据集
9. 写出训练配置 JSON
10. 生成启动命令
11. 可选直接启动训练或打包输出

## 命令行训练

如果不用 notebook，也可以直接改：

```text
configs/sdxl_lora_minimal.json
```

重点字段：

```json
{
  "pretrained_model_name_or_path": "work/models/sd_xl_base_1.0.safetensors",
  "base_model_path": "work/models/sd_xl_base_1.0.safetensors",
  "train_data_dir": "work/datasets/my_lora",
  "output_dir": "work/outputs",
  "logging_dir": "work/logs",
  "output_name": "my_sdxl_lora"
}
```

然后运行（在仓库根目录内）：

```bash
python scripts/run_sdxl_train.py --config configs/sdxl_lora_minimal.json
```

如果在仓库的父目录运行，则带上仓库目录名前缀：

```bash
python lulynx-sdxl-trainer/scripts/run_sdxl_train.py --config configs/sdxl_lora_minimal.json
```

（`--config` 的相对路径会按 `run_sdxl_train.py` 所在仓库根解析，与当前 cwd 无关。）

启动脚本会自动创建：

```text
work/runs/<时间戳>-<输出名>/config.json
```

并将该运行配置传给：

```text
core/entry_train.py
```

## 数据集格式

如果 molab 只能单文件上传，推荐把训练集压成 zip 上传。notebook 的 `2B. 上传训练集 ZIP 后解压` 支持：

- zip 内已经是 `10_xxx/图片+txt` 结构：直接解压使用。
- zip 根目录直接是图片和同名 `.txt`：自动整理到 `10_default/`。
- 解压成功后可自动把最终训练配置的 `train_data_dir` 指向解压结果。

推荐使用 sd-scripts 风格目录：

```text
work/datasets/my_lora/
└── 10_character/
    ├── 0001.png
    ├── 0001.txt
    ├── 0002.png
    └── 0002.txt
```

`10_character` 中的 `10` 表示 repeats，后半部分是概念名。同名 `.txt` 是 caption。

## 首次跑通推荐参数

首次建议保守一点：

```json
{
  "network_dim": 16,
  "network_alpha": 8,
  "resolution": "1024,1024",
  "train_batch_size": 1,
  "gradient_accumulation_steps": 1,
  "optimizer_type": "AdamW8bit",
  "mixed_precision": "bf16",
  "gradient_checkpointing": true,
  "cache_latents": true,
  "cache_latents_to_disk": true,
  "attention_backend": "sageattn",
  "use_sage_attn": true,
  "xformers": false,
  "torch_compile": false,
  "sample_every": false
}
```

如果没有安装 SageAttention，训练器会自动回退到 SDPA。跑通后再逐步尝试更高 rank、更大 batch、采样预览或低显存策略。

## 更新 GitHub

如果你已经把这个文件夹作为独立仓库，更新时在该仓库根目录执行：

```bash
git add .
git commit -m "Add standalone core and interactive molab notebook"
git push
```

注意 `.gitignore` 已排除模型、数据集和输出，请不要把 `.safetensors`、训练集图片、输出 LoRA 直接提交到 GitHub。

`work/models/`、`work/datasets/`、`work/outputs/`、`work/logs/`、`work/runs/`、`work/archives/` 会通过 `.gitkeep` 作为空目录保留在仓库中；目录里的实际大文件仍会被忽略。
