import torch
import numpy as np
from pathlib import Path
from safetensors.torch import safe_open, save_file
try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - optional progress dependency
    def tqdm(iterable, *args, **kwargs):
        return iterable
import gc
import logging

logger = logging.getLogger("LoRASurgeon")

class LoRASurgeon:
    """
    LoRA 手术刀：利用 SVD 和流式处理进行 LoRA 的提取与融合
    Provides SVD-based extraction, merging (surgery), and pruning of LoRA models.
    """
    def __init__(self, device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        
    def _get_svd_lora(self, weight_tensor: torch.Tensor, rank: int):
        """
        核心 SVD 压缩逻辑：将 dense 权重分解为 LoRA A/B
        """
        # 确保是 2D
        original_shape = weight_tensor.shape
        if len(original_shape) == 4:
            # Conv2d: (out, in, k, k) -> (out, in*k*k)
            weight_2d = weight_tensor.reshape(original_shape[0], -1)
        else:
            weight_2d = weight_tensor
            
        # SVD 分解
        try:
            U, S, Vh = torch.linalg.svd(weight_2d.float(), full_matrices=False)
        except (RuntimeError, torch.linalg.LinAlgError):
            # 显存不足回退 CPU
            U, S, Vh = torch.linalg.svd(weight_2d.float().cpu(), full_matrices=False)
            U = U.to(self.device)
            S = S.to(self.device)
            Vh = Vh.to(self.device)
            
        # 截断 Rank
        rank = min(rank, U.shape[1], Vh.shape[0])
        U = U[:, :rank]
        S = S[:rank]
        Vh = Vh[:rank, :]
        
        # 分配奇异值 (sqrt trick)
        # B = U @ sqrt(S), A = sqrt(S) @ Vh
        S_sqrt = torch.diag(torch.sqrt(S))
        lora_up = U @ S_sqrt     # B matrix (out_dim, rank)
        lora_down = S_sqrt @ Vh  # A matrix (rank, in_dim)
        
        # 恢复 Conv2d 形状 (如果是 Conv 层，down 层需要 reshape 回 kernel 形状)
        # LoRA for Conv2d: 
        # Down: (rank, in, k, k)
        # Up: (out, rank, 1, 1)
        if len(original_shape) == 4:
            # A (Down): (rank, in*k*k) -> (rank, in, k, k)
            lora_down = lora_down.reshape(rank, original_shape[1], original_shape[2], original_shape[3])
            # B (Up): (out, rank) -> (out, rank, 1, 1)
            lora_up = lora_up.reshape(original_shape[0], rank, 1, 1)
            
        return lora_down, lora_up

    def extract_lora(self, 
                     tuned_model_path: str, 
                     base_model_path: str, 
                     output_path: str, 
                     rank: int = 128,
                     device: str = None):
        """
        流式 LoRA 提取 (Low RAM Mode)
        不需要同时加载两个大模型，逐层读取计算。
        """
        device = device or self.device
        logger.info(f"[LoRASurgeon] 开始提取 LoRA, Rank={rank}, Device={device}")
        
        lora_state_dict = {}
        
        # 打开两个模型 (Lazy Loading)
        try:
            with safe_open(tuned_model_path, framework="pt", device="cpu") as f_tuned, \
                 safe_open(base_model_path, framework="pt", device="cpu") as f_base:
                
                keys_tuned = set(f_tuned.keys())
                keys_base = set(f_base.keys())
            common_keys = keys_tuned.intersection(keys_base)
            
            # 筛选 Linear 和 Conv 层
            target_keys = [k for k in common_keys if "weight" in k and ("attn" in k or "ff" in k or "proj" in k or "conv" in k)]
            
            for key in tqdm(target_keys, desc="Extracting Layers"):
                # 1. 逐层加载到 GPU (或指定设备)
                w_tuned = f_tuned.get_tensor(key).to(device)
                w_base = f_base.get_tensor(key).to(device)
                
                # 2. 计算差分
                w_diff = w_tuned - w_base
                
                # 释放原始权重
                del w_tuned, w_base
                
                # 3. SVD 分解提取 LoRA
                # key 转换: model.diffusion_model.output_blocks... -> lora_unet_output_blocks...
                lora_key_base = key.replace(".weight", "").replace(".", "_")
                # 简单命名规则适配 (需根据具体架构调整，这里做通用处理)
                if "text_model" in key:
                    lora_prefix = "lora_te_" + lora_key_base
                else:
                    lora_prefix = "lora_unet_" + lora_key_base
                
                try:
                    lora_down, lora_up = self._get_svd_lora(w_diff, rank)
                    
                    lora_state_dict[f"{lora_prefix}.lora_down.weight"] = lora_down.cpu()
                    lora_state_dict[f"{lora_prefix}.lora_up.weight"] = lora_up.cpu()
                    lora_state_dict[f"{lora_prefix}.alpha"] = torch.tensor(float(rank)) # 通常 alpha=rank
                    
                except Exception as e:
                    logger.warning(f"Skipping layer {key} due to error: {e}")
                
                # 清理显存
                del w_diff
                torch.cuda.empty_cache()
                
        except Exception as e:
            logger.error(f"Failed to extract LoRA: {e}")
            raise
        logger.info(f"[LoRASurgeon] 保存 LoRA 到 {output_path}")
        save_file(lora_state_dict, output_path)
        return {"params_extracted": len(lora_state_dict), "rank": rank}

    def merge_loras_svd(self,
                        lora_a_path: str,
                        lora_b_path: str,
                        output_path: str,
                        alpha_a: float = 0.5,
                        alpha_b: float = 0.5,
                        rank: int = 128,
                        layer_weights: dict = None):
        """
        基于 SVD 重投影的 LoRA 融合
        Weights A + Weights B -> Dense -> SVD -> New LoRA
        """
        logger.info(f"[LoRASurgeon] 开始 SVD 融合 LoRA, Target Rank={rank}")
        
        # 加载两个 LoRA (LoRA 通常较小，可以直接加载到内存)
        # 如果 LoRA 很大，也可以改写成流式
        state_a = {}
        with safe_open(lora_a_path, framework="pt", device="cpu") as f:
            for k in f.keys(): state_a[k] = f.get_tensor(k)
            
        state_b = {}
        with safe_open(lora_b_path, framework="pt", device="cpu") as f:
            for k in f.keys(): state_b[k] = f.get_tensor(k)
            
        # 找出所有 module 前缀 (去掉 .lora_up.weight 等后缀)
        modules = set()
        for k in list(state_a.keys()) + list(state_b.keys()):
            if "lora_up" in k:
                modules.add(k.split(".lora_up")[0])
                
        new_state_dict = {}
        
        for module in tqdm(modules, desc="Merging LoRA Layers"):
            # 获取权重
            up_a = state_a.get(f"{module}.lora_up.weight")
            down_a = state_a.get(f"{module}.lora_down.weight")
            
            up_b = state_b.get(f"{module}.lora_up.weight")
            down_b = state_b.get(f"{module}.lora_down.weight")
            
            if up_a is None or down_a is None:
                # 只有 B 有
                if up_b is not None:
                    new_state_dict[f"{module}.lora_up.weight"] = up_b
                    new_state_dict[f"{module}.lora_down.weight"] = down_b
                    new_state_dict[f"{module}.alpha"] = state_b.get(f"{module}.alpha", torch.tensor(rank))
                continue
                
            if up_b is None or down_b is None:
                # 只有 A 有
                new_state_dict[f"{module}.lora_up.weight"] = up_a
                new_state_dict[f"{module}.lora_down.weight"] = down_a
                new_state_dict[f"{module}.alpha"] = state_a.get(f"{module}.alpha", torch.tensor(rank))
                continue

            # 两者都有，进行 SVD 融合
            # 1. 重建 Dense 权重 W = B @ A
            # 注意处理 Conv2d 和 Linear 的维度差异
            def reconstruct(up, down):
                if up.ndim == 4: # Conv2d
                    # Up: (out, rank, 1, 1) -> (out, rank)
                    # Down: (rank, in, k, k) -> (rank, in*k*k)
                    up_mat = up.squeeze()
                    down_mat = down.reshape(down.shape[0], -1)
                    w = up_mat @ down_mat
                    return w, (up.shape[0], down.shape[1], down.shape[2], down.shape[3]) # return target shape
                else: # Linear
                    w = up @ down
                    return w, w.shape
            
            # 决定当前层的 alpha (支持 layer_weights)
            # 简化的 layer matching 逻辑，需要根据 module name 匹配 layer_weights key
            curr_alpha_a = alpha_a
            curr_alpha_b = alpha_b
            if layer_weights:
                for k, v in layer_weights.items():
                    if k in module: # 模糊匹配: "input_blocks.1" in "lora_unet_input_blocks_1_..."
                        curr_alpha_a = 1.0 - v # UI weight usually means Ratio B
                        curr_alpha_b = v
                        break

            w_a, shape_a = reconstruct(up_a.float().to(self.device), down_a.float().to(self.device))
            w_b, shape_b = reconstruct(up_b.float().to(self.device), down_b.float().to(self.device))
            
            # 2. 混合
            w_merged = w_a * curr_alpha_a + w_b * curr_alpha_b
            
            # 手动清理重建的大矩阵
            del w_a, w_b
            
            # 3. 重新 SVD 提取 (确保 shape 恢复)
            # 需要把 reshape 后的 2D 矩阵传进去，形状信息利用 shape_a
            # _get_svd_lora 内部会处理 2D -> 4D 的恢复，但需要传入正确的原始 shape 格式的 tensor
            # 这里的 w_merged 是 2D 的 (out, in*k*k) 或 (out, in)
            
            # Trick: 构造一个 view 传给 _get_svd_lora，让它知道原始 shape
            w_merged_view = w_merged.reshape(shape_a)
            
            new_down, new_up = self._get_svd_lora(w_merged_view, rank)
            
            new_state_dict[f"{module}.lora_up.weight"] = new_up.cpu()
            new_state_dict[f"{module}.lora_down.weight"] = new_down.cpu()
            new_state_dict[f"{module}.alpha"] = torch.tensor(float(rank))
            
            del w_merged, w_merged_view, new_down, new_up
            torch.cuda.empty_cache()

        logger.info(f"[LoRASurgeon] LoRA 融合完成，保存至 {output_path}")
        save_file(new_state_dict, output_path)
        return {"modules_merged": len(modules), "rank": rank}

    def bake_lora(
        self,
        base_model_path: str,
        lora_path: str,
        output_path: str,
        alpha: float = 1.0, # LoRA weight scale
    ) -> dict:
        """
        将 LoRA 固化(Bake)进底模 Checkpoint (Streamed)
        result = base + (lora_up @ lora_down) * alpha
        """
        logger.info(f"[LoRASurgeon] 开始固化 LoRA: {lora_path} -> {base_model_path}")
        
        # 1. 预加载 LoRA (通常较小，全部读入内存)
        lora_state = {}
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                lora_state[k] = f.get_tensor(k)
                
        # 建立 LoRA 键映射加速查找
        # module_name -> { "up": tensor, "down": tensor, "alpha": tensor }
        lora_modules = {}
        for k in lora_state.keys():
            if "lora_up" in k:
                module = k.split(".lora_up")[0]
                lora_modules[module] = {
                    "up": lora_state[f"{module}.lora_up.weight"],
                    "down": lora_state[f"{module}.lora_down.weight"],
                    "alpha": lora_state.get(f"{module}.alpha"),
                }
                
        # 2. 建立 Checkpoint Key -> LoRA Key 的映射规则
        # SD1.5 / SDXL / Flux 都有不同的命名，这里先实现通用 SD/SDXL 映射
        def get_lora_module_name(ckpt_key: str):
            # mapping rules:
            # model.diffusion_model.output_blocks.1.1.proj_out.weight 
            # -> lora_unet_output_blocks_1_1_proj_out
            k = ckpt_key.replace(".weight", "").replace(".", "_")
            if "text_model" in k or "conditioner" in k:
                return "lora_te_" + k # Basic guess, might need refinement for SDXL TE
            else:
                return "lora_unet_" + k

        stats = {"layers_modified": 0, "layers_total": 0}
        new_state_dict = {}
        
        # 3. 流式处理底模
        with safe_open(base_model_path, framework="pt", device="cpu") as f_base:
            for key in tqdm(f_base.keys(), desc="Baking LoRA"):
                stats["layers_total"] += 1
                weight = f_base.get_tensor(key) # Keep on CPU initially
                
                # Check for LoRA match
                lora_module_name = get_lora_module_name(key)
                
                # Try finding in loaded modules
                # Some LoRA trainers might use slightly different naming, fuzzy match if needed?
                # For now, strict match based on standard diffusers-to-lora conversion
                module_data = lora_modules.get(lora_module_name)
                
                if module_data:
                    # Found matching LoRA layer!
                    try:
                        w_base = weight.to(self.device).float()
                        up = module_data["up"].to(self.device).float()
                        down = module_data["down"].to(self.device).float()
                        
                        # Calc scale
                        net_alpha = module_data["alpha"]
                        rank = down.shape[0] # (rank, dim)
                        # Standard LoRA scale formula: (alpha / rank) * multiplier
                        scale = (float(net_alpha) / float(rank)) * alpha if net_alpha is not None else alpha
                        
                        # Reconstruct delta
                        # Conv2d handling
                        if up.ndim == 4:
                            up = up.squeeze()
                            down = down.reshape(down.shape[0], -1)
                            delta = (up @ down).reshape(w_base.shape)
                        else:
                            delta = up @ down
                            
                        # Add to base
                        w_new = w_base + delta * scale
                        
                        new_state_dict[key] = w_new.half().cpu() # Convert back to fp16/save format
                        stats["layers_modified"] += 1
                        
                        del w_base, up, down, delta, w_new
                        
                    except Exception as e:
                        logger.error(f"Error baking layer {key}: {e}")
                        new_state_dict[key] = weight # Keep original on error
                else:
                    new_state_dict[key] = weight
                    
        # 4. Save
        logger.info(f"[LoRASurgeon] Baking complete. Modified {stats['layers_modified']} layers.")
        save_file(new_state_dict, output_path)
        return stats

    def prune_lora(
        self,
        lora_path: str,
        output_path: str,
        keep_blocks: list[str] = None,
        drop_blocks: list[str] = None,
    ) -> dict:
        import json
        import os
        import re

        def normalize_block_name(key: str) -> str:
            import re
            k = key.lower()
            if "te1" in k or "text_model1" in k: return "TE1"
            if "te2" in k or "text_model2" in k: return "TE2"
            if any(x in k for x in ["te_", "text_model", "text_encoder"]): return "TE1"
            m = re.search(r'(block|layer|input_block|down_block|output_block|up_block|double_block|single_block|down|up)[s]?[\._]?(\d+)', k)
            if m:
                t_str, idx = m.group(1), int(m.group(2))
                if "input" in t_str or "down" in t_str: return f"IN{idx:02d}"
                if "output" in t_str or "up" in t_str: return f"OUT{idx:02d}"
                if idx < 10: return f"IN{idx:02d}"
                if idx < 20: return f"OUT{(idx-10):02d}"
                return f"OUT{(idx % 9):02d}"
            if "mid" in k: return "M00"
            return "OTHER"

        def is_weight_key(key: str) -> bool:
            k = key.lower()
            if any(x in k for x in [".alpha", ".scale", "metadata"]): return False
            return any(x in k for x in ["weight", "lokr", "hada", "diff", "w1", "w2", "lora_up", "lora_down", "matrix"])

        try:
            with open(lora_path, "rb") as f_in:
                header_size = int.from_bytes(f_in.read(8), "little")
                header_json = f_in.read(header_size).decode('utf-8')
                header = json.loads(header_json)
                offset_base = 8 + header_size
                
                new_header = {}
                new_data_list = []
                current_offset = 0
                kept_count = 0
                dropped_count = 0
                
                if "__metadata__" in header:
                    new_header["__metadata__"] = header["__metadata__"]

                for key, info in header.items():
                    if key == "__metadata__": continue
                    
                    if is_weight_key(key):
                        block_id = normalize_block_name(key)
                        should_keep = True
                        if keep_blocks is not None:
                            should_keep = block_id in keep_blocks
                        elif drop_blocks is not None:
                            should_keep = block_id not in drop_blocks
                    else:
                        # By default keep metadata and other control tensors
                        should_keep = True
                        
                    if should_keep:
                        start, end = info["data_offsets"]
                        f_in.seek(offset_base + start)
                        raw_bytes = f_in.read(end - start)
                        
                        size = len(raw_bytes)
                        new_header[key] = {
                            "dtype": info["dtype"],
                            "shape": info["shape"],
                            "data_offsets": [current_offset, current_offset + size]
                        }
                        new_data_list.append(raw_bytes)
                        current_offset += size
                        kept_count += 1
                    else:
                        dropped_count += 1
            
            header_bytes = json.dumps(new_header).encode('utf-8')
            padding_len = (8 - (len(header_bytes) % 8)) % 8
            header_bytes += b' ' * padding_len
            header_size_bytes = len(header_bytes).to_bytes(8, "little")
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f_out:
                f_out.write(header_size_bytes)
                f_out.write(header_bytes)
                for chunk in new_data_list:
                    f_out.write(chunk)
                    
            return {
                "kept_tensors": kept_count,
                "dropped_tensors": dropped_count,
                "output_path": output_path
            }
            
        except Exception as e:
            raise RuntimeError(f"Pruning failed: {str(e)}")

    def merge_loras_lbw(
        self,
        lora_a_path: str,
        lora_b_path: str,
        output_path: str,
        alpha_a: float = 1.0,
        alpha_b: float = 1.0,
        rank: int = 128,
        lbw_weights_a: str = "",
        lbw_weights_b: str = "",
        model_type: str = "sdxl",
    ) -> dict:
        """Merge two LoRAs with per-block weight control (LBW).

        ``lbw_weights_a`` / ``lbw_weights_b`` are comma-separated floats,
        one per block (26 for SDXL, 17 for SD1.5).  Each weight scales
        that block's contribution before the SVD re-merge.
        """
        from .lbw_parser import parse_lbw_weights, apply_lbw_to_layer_key

        lbw_a = parse_lbw_weights(lbw_weights_a, model_type) if lbw_weights_a else {}
        lbw_b = parse_lbw_weights(lbw_weights_b, model_type) if lbw_weights_b else {}

        layer_weights = {}
        from safetensors.torch import load_file
        tensors_a = load_file(lora_a_path, device="cpu")
        for key in tensors_a:
            if ".lora_down.weight" in key:
                prefix = key.replace(".lora_down.weight", "")
                w_a = apply_lbw_to_layer_key(key, lbw_a) if lbw_a else 1.0
                w_b = apply_lbw_to_layer_key(key, lbw_b) if lbw_b else 1.0
                layer_weights[prefix] = {"alpha_a": alpha_a * w_a, "alpha_b": alpha_b * w_b}

        return self.merge_loras_svd(
            lora_a_path, lora_b_path, output_path,
            alpha_a=alpha_a, alpha_b=alpha_b, rank=rank,
            layer_weights=layer_weights,
        )
