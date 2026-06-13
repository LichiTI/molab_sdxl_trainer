# SPDX-License-Identifier: LicenseRef-PolyFormNoncommercial-1.0.0
"""SDXL 标准 LoRA "裸 key" -> kohya/ComfyUI 兼容 key 规范化。

背景
----
molab native trainer 注入 LoRA 时使用 ``prefix=unet/te1/te2``，
``get_lora_state_dict`` 直接以注入名导出，得到的 key 形如::

    unet_down_blocks_1_attentions_0_..._to_q.lora_down.weight
    te1_encoder_layers_0_self_attn_q_proj.lora_up.weight

这套 "裸 key" 缺少 sd-scripts/kohya 标准的 ``lora_`` 前缀与文本编码器的
``text_model_`` 段，也没有 per-module ``.alpha`` 和 ``ss_network_module`` 元数据。

通用 ComfyUI / A1111 / Forge 的 LoRA 加载器按 ``lora_unet_`` /
``lora_te1_text_model_`` 前缀做正则匹配，匹配不到就把整份 LoRA **静默跳过**，
表现为"加载了但出图毫无变化、不同 epoch 完全一样"。

本模块在保存阶段把裸 key 规范化为标准格式::

    lora_unet_down_blocks_1_attentions_0_..._to_q.lora_down.weight
    lora_te1_text_model_encoder_layers_0_self_attn_q_proj.lora_up.weight
    <module>.alpha               (每模块补齐)
    ss_network_module=networks.lora  (元数据)

说明
----
- 仅处理标准 LoRA 结构（``.lora_down.weight`` / ``.lora_up.weight``）。
- 幂等：已是 ``lora_`` 前缀的 key 原样保留，不会二次加前缀。
- 保留 diffusers 命名（down_blocks/up_blocks/mid_block）。ComfyUI 核心
  LoraLoader 内部有 diffusers<->ldm 映射，可正确加载；若目标是只认 ldm
  命名（input_blocks/output_blocks）的旧工具，需要额外的块名重映射。
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


_RAW_PREFIXES = ("unet_", "te1_", "te2_", "te_")


def _normalize_module_base(base: str) -> str:
    """把裸模块名转成 kohya 标准模块名。已是 ``lora_`` 前缀的原样返回。"""
    if base.startswith("lora_"):
        return base
    if base.startswith("unet_"):
        return "lora_" + base
    if base.startswith("te1_"):
        return "lora_te1_text_model_" + base[len("te1_"):]
    if base.startswith("te2_"):
        return "lora_te2_text_model_" + base[len("te2_"):]
    if base.startswith("te_"):
        return "lora_te_text_model_" + base[len("te_"):]
    return base


def _is_raw_lora_weight_key(key: str) -> bool:
    if not (key.endswith(".lora_down.weight") or key.endswith(".lora_up.weight")):
        return False
    return key.startswith(_RAW_PREFIXES) and not key.startswith("lora_")


def export_sdxl_compatible_lora_keys(
    state_dict: Dict[str, torch.Tensor],
    metadata: Optional[Dict[str, str]],
    *,
    network_alpha: float,
) -> Tuple[Dict[str, torch.Tensor], Optional[Dict[str, str]], int]:
    """规范化 SDXL 标准 LoRA 的 key。

    返回 ``(new_state_dict, new_metadata, converted_module_count)``。
    若 ``state_dict`` 中不存在任何裸 LoRA 权重 key，则原样返回（converted=0）。
    """
    if not any(_is_raw_lora_weight_key(k) for k in state_dict):
        return state_dict, metadata, 0

    new_sd: Dict[str, torch.Tensor] = {}
    module_dtypes: Dict[str, torch.dtype] = {}

    for key, value in state_dict.items():
        if key.endswith(".lora_down.weight") or key.endswith(".lora_up.weight"):
            # "unet_x.lora_down.weight" -> base="unet_x", tail="down.weight"
            mod_base, sep, lora_tail = key.rpartition(".lora_")
            if not sep:
                new_sd[key] = value
                continue
            new_base = _normalize_module_base(mod_base)
            new_sd[f"{new_base}.lora_{lora_tail}"] = value
            if isinstance(value, torch.Tensor):
                module_dtypes[new_base] = value.dtype
        else:
            base, sep, tail = key.rpartition(".")
            if sep:
                new_sd[f"{_normalize_module_base(base)}.{tail}"] = value
            else:
                new_sd[key] = value

    # 补齐 per-module alpha（仅当缺失时）
    alpha_value = float(network_alpha)
    for new_base in module_dtypes:
        alpha_key = f"{new_base}.alpha"
        if alpha_key not in new_sd:
            new_sd[alpha_key] = torch.tensor(alpha_value, dtype=torch.float32)

    # 元数据：尊重 no_metadata（metadata is None 时不强造），否则补 network_module
    if metadata is None:
        new_meta: Optional[Dict[str, str]] = None
    else:
        new_meta = dict(metadata)
        new_meta.setdefault("ss_network_module", "networks.lora")

    return new_sd, new_meta, len(module_dtypes)
