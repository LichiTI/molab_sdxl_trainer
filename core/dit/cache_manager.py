
import os
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Optional, List, Dict
import logging

logger = logging.getLogger(__name__)

class FluxCacheManager:
    """
    Flux 缓存管理器
    负责将图像和文本预编码为 Latents 和 Embeddings 并保存到磁盘
    """
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def cache_dataset(
        self,
        dataset,
        vae,
        text_encoder,
        tokenizer,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16
    ):
        """
        执行缓存流程
        
        Args:
            dataset: 原始 ImageDataset
            vae: VAE 模型
            text_encoder: 文本编码器
            tokenizer: 分词器
            device: 运算设备
        """
        logger.info(f"[FluxCacheManager] Starting caching to {self.output_dir}")
        logger.info(f"[FluxCacheManager] Dataset size: {len(dataset)}")
        
        # Move models to device
        vae.to(device, dtype=dtype)
        text_encoder.to(device, dtype=dtype)
        vae.eval()
        text_encoder.eval()
        
        cache_index = []
        
        with torch.no_grad():
            for i in tqdm(range(len(dataset)), desc="Caching Latents"):
                sample = dataset[i]
                image_path = sample.get("path", f"sample_{i}") # Assuming dataset returns path or we use index naming
                image_name = Path(image_path).stem
                
                # 1. Encode Image -> Latents
                # Dataset 应该返回预处理好的 tensor [3, H, W]
                pixel_values = sample["image"].unsqueeze(0).to(device, dtype=dtype)
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = (latents * vae.config.scaling_factor).cpu() # Move back to CPU to save RAM
                
                # 2. Encode Text -> Embeddings
                caption = sample["caption"]
                
                # 简化的 Prompt Encoding (Flux 实际需要更复杂的 Masking)
                # 这里假设 text_encoder 是 T5 或兼容接口
                text_inputs = tokenizer(
                    caption, 
                    padding="max_length", 
                    max_length=512, 
                    truncation=True, 
                    return_tensors="pt"
                ).to(device)
                
                # T5 Encoder usually returns last_hidden_state as [0]
                prompt_embeds = text_encoder(text_inputs.input_ids)[0]
                pooled_embeds = prompt_embeds.mean(dim=1) # Simplified pooling for T5
                
                prompt_embeds = prompt_embeds.cpu()
                pooled_embeds = pooled_embeds.cpu()
                
                # 3. Save to Disk (.npz)
                save_path = self.output_dir / f"{image_name}.npz"
                np.savez_compressed(
                    save_path,
                    latents=latents.float().numpy(), # Save as float32 for compatibility or keep bf16 via specialized save? np.savez supports standard types.
                    text_embeddings=prompt_embeds.float().numpy(),
                    pooled_embeddings=pooled_embeds.float().numpy()
                )
                
                cache_index.append({
                    "npz_path": str(save_path),
                    "caption": caption
                })
                
        logger.info(f"[FluxCacheManager] Caching complete. Saved {len(cache_index)} files.")
        
        # Save index? Or just rely on file existence matching
        # For simplicity, we assume 1:1 mapping with original files if we implement CachedDataset intelligently.
        return self.output_dir
