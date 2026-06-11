"""
Textual Inversion (嵌入学习)

学习新的 token embedding 来表示概念
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, Optional, List, Any
from dataclasses import dataclass, field
from pathlib import Path
from core.safe_pickle import safe_torch_load

logger = logging.getLogger(__name__)


@dataclass
class TextualInversionConfig:
    """Textual Inversion 配置"""
    # 模型
    model_path: str = ""
    model_type: str = "sdxl"
    
    # 概念
    placeholder_token: str = "<concept>"
    initializer_token: str = "person"  # 用于初始化的已知 token
    num_vectors: int = 1               # 使用多少个向量表示概念
    
    # 训练数据
    train_data_dir: str = ""
    
    # 训练
    learning_rate: float = 5e-4
    max_train_steps: int = 3000
    
    # 优化
    mixed_precision: str = "fp16"


class ConceptEmbedding(nn.Module):
    """
    可学习的概念嵌入
    
    可以使用多个向量表示一个概念 (Multi-Vector)
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_vectors: int = 1,
        initializer_embedding: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.num_vectors = num_vectors
        self.embed_dim = embed_dim
        
        # 可学习嵌入
        self.embeddings = nn.Parameter(
            torch.randn(num_vectors, embed_dim)
        )
        
        # 初始化
        if initializer_embedding is not None:
            with torch.no_grad():
                if num_vectors == 1:
                    self.embeddings.data = initializer_embedding.unsqueeze(0)
                else:
                    # 多向量：使用加噪的初始化
                    for i in range(num_vectors):
                        noise = torch.randn_like(initializer_embedding) * 0.01
                        self.embeddings.data[i] = initializer_embedding + noise
    
    def forward(self) -> torch.Tensor:
        """返回嵌入"""
        return self.embeddings


class TextualInversionTrainer:
    """
    Textual Inversion 训练器
    
    特点:
    - 只训练新 token 的嵌入
    - 模型权重完全冻结
    - 非常轻量
    """
    
    def __init__(self, config: TextualInversionConfig):
        self.config = config
        self.model = None
        self.text_encoder = None
        self.tokenizer = None
        self.vae = None
        self.scheduler = None
        self.concept_embedding = None
        self.optimizer = None
        
        # 记录原始词表大小
        self._original_vocab_size = 0
        self._placeholder_token_id = None

    @staticmethod
    def _get_embedding_layer(text_encoder):
        """Return the token embedding layer for any CLIPTextModel variant."""
        if hasattr(text_encoder, 'text_model'):
            return text_encoder.text_model.embeddings.token_embedding
        if hasattr(text_encoder, 'embeddings'):
            return text_encoder.embeddings.token_embedding
        return text_encoder.get_input_embeddings()
    
    def load_model(self):
        """加载模型"""
        try:
            from pathlib import Path as _Path
            model_path = _Path(self.config.model_path)
            is_single_file = model_path.suffix.lower() in {".safetensors", ".ckpt", ".pt"}

            if self.config.model_type == "sdxl" and is_single_file:
                # Use ModelLoader's SDXL single-file path to avoid diffusers
                # from_single_file() text_model attribute bug (diffusers 0.38)
                from .model_loader import ModelLoader
                from transformers import CLIPTokenizer
                loader = ModelLoader(device="cpu", dtype=torch.float16)
                components = loader.load(str(model_path), model_arch="sdxl")
                self.model = components.unet
                self.text_encoder = components.text_encoder_1
                self.text_encoder_2 = components.text_encoder_2
                # TI needs a real CLIPTokenizer with add_tokens(); the
                # single-file loader uses OpenClipTokenizerAdapter which
                # doesn't support adding new tokens.
                self.tokenizer = CLIPTokenizer.from_pretrained(
                    "openai/clip-vit-large-patch14"
                )
                self.tokenizer_2 = self.tokenizer
                self.vae = components.vae
                self.scheduler = components.noise_scheduler
            elif self.config.model_type == "sdxl":
                from diffusers import StableDiffusionXLPipeline
                pipe = StableDiffusionXLPipeline.from_pretrained(
                    str(model_path), torch_dtype=torch.float16,
                )
                self.model = pipe.unet
                self.text_encoder = pipe.text_encoder
                self.text_encoder_2 = getattr(pipe, 'text_encoder_2', None)
                self.tokenizer = pipe.tokenizer
                self.tokenizer_2 = getattr(pipe, 'tokenizer_2', None)
                self.vae = pipe.vae
                self.scheduler = pipe.scheduler
            else:
                from diffusers import StableDiffusionPipeline
                if is_single_file:
                    pipe = StableDiffusionPipeline.from_single_file(
                        str(model_path), torch_dtype=torch.float16,
                    )
                else:
                    pipe = StableDiffusionPipeline.from_pretrained(
                        str(model_path), torch_dtype=torch.float16,
                    )
                self.model = pipe.unet
                self.text_encoder = pipe.text_encoder
                self.text_encoder_2 = None
                self.tokenizer = pipe.tokenizer
                self.tokenizer_2 = None
                self.vae = pipe.vae
                self.scheduler = pipe.scheduler
            
            # 冻结所有参数
            self.model.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            self.vae.requires_grad_(False)
            
            # 添加占位符 token
            self._add_placeholder_token()
            
            # 初始化概念嵌入
            self._init_concept_embedding()
            
            logger.info("[TextualInversionTrainer] Model loaded")
            
        except Exception as e:
            logger.error(f"[TextualInversionTrainer] Load failed: {e}")
            raise
    
    def _add_placeholder_token(self):
        """添加占位符 token 到词表"""
        self._original_vocab_size = len(self.tokenizer)
        
        # 添加多个 token (如果 num_vectors > 1)
        placeholder_tokens = [
            self.config.placeholder_token
        ]
        if self.config.num_vectors > 1:
            placeholder_tokens.extend([
                f"{self.config.placeholder_token}_{i}"
                for i in range(1, self.config.num_vectors)
            ])
        
        num_added = self.tokenizer.add_tokens(placeholder_tokens)

        # 扩展 text encoder 嵌入 (works for both standard and from_single_file models)
        embed_layer = self._get_embedding_layer(self.text_encoder)
        old_num = embed_layer.num_embeddings
        new_num = len(self.tokenizer)
        if new_num > old_num:
            import torch.nn as nn
            new_embeds = nn.Embedding(new_num, embed_layer.embedding_dim)
            new_embeds.weight.data[:old_num] = embed_layer.weight.data
            new_embeds.requires_grad_(False)  # Freeze so in-place assignment works
            # Assign back to the correct location
            if hasattr(self.text_encoder, 'text_model'):
                self.text_encoder.text_model.embeddings.token_embedding = new_embeds
            elif hasattr(self.text_encoder, 'embeddings'):
                self.text_encoder.embeddings.token_embedding = new_embeds
            else:
                self.text_encoder.set_input_embeddings(new_embeds)
        
        # SDXL: also resize text_encoder_2 embedding layer
        if getattr(self, 'text_encoder_2', None) is not None:
            embed_layer_2 = self._get_embedding_layer(self.text_encoder_2)
            old_num_2 = embed_layer_2.num_embeddings
            if new_num > old_num_2:
                new_embeds_2 = nn.Embedding(new_num, embed_layer_2.embedding_dim)
                new_embeds_2.weight.data[:old_num_2] = embed_layer_2.weight.data
                new_embeds_2.requires_grad_(False)
                if hasattr(self.text_encoder_2, 'text_model'):
                    self.text_encoder_2.text_model.embeddings.token_embedding = new_embeds_2
                elif hasattr(self.text_encoder_2, 'embeddings'):
                    self.text_encoder_2.embeddings.token_embedding = new_embeds_2
                else:
                    self.text_encoder_2.set_input_embeddings(new_embeds_2)

        # 记录 token ID
        self._placeholder_token_id = self.tokenizer.convert_tokens_to_ids(
            self.config.placeholder_token
        )
        
        logger.info(f"[TextualInversionTrainer] Added {num_added} placeholder tokens")
    
    def _init_concept_embedding(self):
        """初始化概念嵌入"""
        # 获取初始化 token 的嵌入
        init_token_id = self.tokenizer.convert_tokens_to_ids(
            self.config.initializer_token
        )
        
        with torch.no_grad():
            init_embedding = self._get_embedding_layer(self.text_encoder).weight[init_token_id]
        
        # 创建可学习嵌入 (always float32 — fp16 causes AdamW denominator underflow)
        self.concept_embedding = ConceptEmbedding(
            embed_dim=init_embedding.shape[-1],
            num_vectors=self.config.num_vectors,
            initializer_embedding=init_embedding,
        ).to(device=init_embedding.device, dtype=torch.float32)
    
    def prepare_optimizer(self):
        """准备优化器 (只优化概念嵌入)"""
        self.optimizer = torch.optim.AdamW(
            self.concept_embedding.parameters(),
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
        )
        
        logger.info("[TextualInversionTrainer] Optimizer prepared")
    
    def update_text_encoder_embeddings(self):
        """更新 text encoder 中的嵌入"""
        embeddings = self.concept_embedding()

        # 写入 text encoder (use weight, not weight.data, to preserve grad graph)
        embed_layer = self._get_embedding_layer(self.text_encoder)
        for i, emb in enumerate(embeddings):
            token_id = self._placeholder_token_id + i
            embed_layer.weight[token_id] = emb
    
    def _detach_text_encoder_embeddings(self):
        """Detach learned token rows after a step so the next step builds a fresh graph."""
        if self.concept_embedding is None or self._placeholder_token_id is None:
            return

        embed_layer = self._get_embedding_layer(self.text_encoder)
        embeddings = self.concept_embedding().detach().to(
            device=embed_layer.weight.device,
            dtype=embed_layer.weight.dtype,
        )
        with torch.no_grad():
            for i, emb in enumerate(embeddings):
                token_id = self._placeholder_token_id + i
                embed_layer.weight[token_id].copy_(emb)
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """单步训练

        Accepts either:
          - batch["captions"]: raw text, encoded internally after embedding update
            (gradients flow to concept_embedding)
          - batch["text_embeddings"]: pre-encoded (legacy, gradients still flow
            but use stale embeddings from prior step)
        """
        self.optimizer.zero_grad()

        # 更新嵌入
        self.update_text_encoder_embeddings()

        latents = batch["latents"]
        pooled = None

        # Encode text after embedding update so gradients flow to concept_embedding
        if "captions" in batch:
            captions = batch["captions"]
            tokens = self.tokenizer(
                captions,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            )
            enc_out = self.text_encoder(
                tokens.input_ids.to(latents.device),
            )
            text_embeddings = enc_out.last_hidden_state

            # SDXL: concatenate TE1 + TE2 hidden states
            if getattr(self, 'text_encoder_2', None) is not None:
                tokenizer_2 = getattr(self, 'tokenizer_2', self.tokenizer)
                tokens_2 = tokenizer_2(
                    captions,
                    padding="max_length",
                    max_length=tokenizer_2.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                enc_out_2 = self.text_encoder_2(
                    tokens_2.input_ids.to(latents.device),
                )
                hidden_states_2 = enc_out_2.last_hidden_state
                text_embeddings = torch.cat([text_embeddings, hidden_states_2], dim=-1)
                pooled = getattr(enc_out_2, 'text_embeds', None)
                if pooled is None:
                    pooled = hidden_states_2.mean(dim=1)
        else:
            text_embeddings = batch["text_embeddings"]
            pooled = None

        # 采样时间步
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=latents.device,
        ).long()

        # 添加噪声
        noise = torch.randn_like(latents)
        noisy_latents = self.scheduler.add_noise(latents, noise, timesteps)

        # SDXL requires added_cond_kwargs
        unet_kwargs = {
            "sample": noisy_latents,
            "timestep": timesteps,
            "encoder_hidden_states": text_embeddings,
        }
        if pooled is not None:
            b = latents.shape[0]
            unet_kwargs["added_cond_kwargs"] = {
                "text_embeds": pooled,
                "time_ids": torch.tensor(
                    [[64, 64, 0, 0, 64, 64]] * b,
                    device=latents.device, dtype=latents.dtype,
                ),
            }

        # 预测
        noise_pred = self.model(**unet_kwargs).sample

        # 损失
        loss = F.mse_loss(noise_pred, noise)

        # 反向传播
        loss.backward(retain_graph=True)
        self.optimizer.step()
        self._detach_text_encoder_embeddings()

        return loss.item()
    
    def save(self, output_path: str):
        """保存学习到的嵌入"""
        embeddings = self.concept_embedding()
        
        state_dict = {
            "placeholder_token": self.config.placeholder_token,
            "num_vectors": self.config.num_vectors,
            "embeddings": embeddings.cpu(),
        }
        
        torch.save(state_dict, output_path)
        logger.info(f"[TextualInversionTrainer] Saved to {output_path}")
    
    @staticmethod
    def load_embedding(
        path: str,
        tokenizer,
        text_encoder,
    ):
        """加载嵌入到 tokenizer 和 text_encoder"""
        state = safe_torch_load(path, map_location="cpu")
        
        placeholder_token = state["placeholder_token"]
        num_vectors = state["num_vectors"]
        embeddings = state["embeddings"]
        
        # 添加 tokens
        placeholder_tokens = [placeholder_token]
        if num_vectors > 1:
            placeholder_tokens.extend([
                f"{placeholder_token}_{i}"
                for i in range(1, num_vectors)
            ])
        
        tokenizer.add_tokens(placeholder_tokens)
        text_encoder.resize_token_embeddings(len(tokenizer))
        
        # 写入嵌入
        token_id = tokenizer.convert_tokens_to_ids(placeholder_token)
        for i, emb in enumerate(embeddings):
            text_encoder.get_input_embeddings().weight.data[token_id + i] = emb.to(
                text_encoder.get_input_embeddings().weight.device,
                dtype=text_encoder.get_input_embeddings().weight.dtype,
            )
        
        logger.info(f"[TextualInversion] Loaded {placeholder_token} ({num_vectors} vectors)")


# ========== 便捷函数 ==========

def create_textual_inversion_trainer(
    model_path: str,
    placeholder_token: str,
    initializer_token: str = "person",
) -> TextualInversionTrainer:
    """创建 Textual Inversion 训练器"""
    config = TextualInversionConfig(
        model_path=model_path,
        placeholder_token=placeholder_token,
        initializer_token=initializer_token,
    )
    return TextualInversionTrainer(config)


