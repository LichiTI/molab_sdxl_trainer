# Lulynx SDXL LoRA Molab 精简仓库脚手架

这个目录用于把当前项目中 **SDXL LoRA 训练所需的后端核心** 单独导出成一个可上传到 GitHub、再给 molab 使用的轻量仓库。

> 说明：你把 `plugin/lora-scripts-ui-main` 称为前端；本方案不包含前端、不包含 Launcher/WPF、不包含 WebUI，只保留训练引擎入口和 molab 辅助脚本。

## 为什么不是只复制几个文件？

当前 SDXL LoRA 训练入口是：

```bash
core/entry_train.py
```

它会进入：

```text
core/lulynx_trainer/
```

其中 SDXL LoRA 路线和通用训练循环、数据集读取、LoRA 注入、优化器、缓存、保存、低显存策略等耦合较深。直接手工挑 10～20 个文件很容易漏动态导入或配置兼容层。所以第一版推荐导出：

```text
core/                    # 由原 backend/core 复制而来
scripts/run_sdxl_train.py
configs/sdxl_lora_minimal.json
requirements-molab.txt
notebooks/molab_sdxl_lora.py
```

这已经去掉了：

- 前端 `plugin/lora-scripts-ui-main`
- Launcher/WPF/Tauri/Monitor
- Web API routers
- 安装器、资源包、开发文档

但保留 `core/` 训练引擎，确保 SDXL LoRA 能跑。

## 导出步骤

在当前大项目根目录执行：

```bash
python molab_sdxl_trainer/scripts/export_molab_sdxl_repo.py --out dist/lulynx-sdxl-molab
```

导出后得到：

```text
dist/lulynx-sdxl-molab/
├── core/
├── configs/
│   └── sdxl_lora_minimal.json
├── notebooks/
│   └── molab_sdxl_lora.py
├── scripts/
│   └── run_sdxl_train.py
├── requirements-molab.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

然后进入导出目录，新建 GitHub 仓库并推送：

```bash
cd dist/lulynx-sdxl-molab
git init
git add .
git commit -m "Initial SDXL LoRA molab trainer"
git branch -M main
git remote add origin https://github.com/<你的用户名>/<你的仓库名>.git
git push -u origin main
```

## molab 上使用

在 molab notebook 中：

```bash
git clone https://github.com/<你的用户名>/<你的仓库名>.git
cd <你的仓库名>
pip install -r requirements-molab.txt
```

检查 GPU：

```python
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no cuda")
```

准备目录：

```text
work/
├── models/
│   └── sd_xl_base_1.0.safetensors
├── datasets/
│   └── my_lora/
│       └── 10_character/
│           ├── 0001.png
│           ├── 0001.txt
│           ├── 0002.png
│           └── 0002.txt
├── outputs/
├── logs/
└── runs/
```

修改 `configs/sdxl_lora_minimal.json` 里的：

```json
{
  "pretrained_model_name_or_path": "work/models/sd_xl_base_1.0.safetensors",
  "train_data_dir": "work/datasets/my_lora",
  "output_dir": "work/outputs",
  "logging_dir": "work/logs",
  "output_name": "my_sdxl_lora"
}
```

启动训练：

```bash
python scripts/run_sdxl_train.py --config configs/sdxl_lora_minimal.json
```

## 数据集格式

兼容 sd-scripts 风格 subset：

```text
work/datasets/my_lora/
└── 10_character/
    ├── 0001.png
    ├── 0001.txt
    ├── 0002.png
    └── 0002.txt
```

`10_character` 中 `10` 表示 repeats，`character` 是概念名。每张图片可以有同名 `.txt` caption。

## 推荐先关闭的功能

为了 molab 首次跑通，建议第一版配置里保持：

- 不用 WebUI
- 不启用 `torch_compile`
- 不启用 `xformers/flash-attn/sageattention`
- 不启用复杂插件优化器
- 不启用预览采样
- 先用 `AdamW8bit` 或 `AdamW`

跑通后再逐步打开 cache、低显存策略或采样预览。
